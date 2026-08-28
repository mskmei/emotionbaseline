# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from keypoints import KEYPOINT_DIM


SYSTEM_PROMPT = "You translate Japanese Sign Language videos into natural Japanese text."
USER_PROMPT = "<video>\nこの日本手話動画を自然な日本語の一文に翻訳してください。/no_think"


def parse_dtype(name: str):
    key = (name or "auto").lower()
    if key == "auto":
        return "auto"
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def default_lora_targets(name: str) -> List[str]:
    key = (name or "all").lower()
    if key == "qv":
        return ["q_proj", "v_proj"]
    if key == "qkvo":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if key == "all":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return [x.strip() for x in name.split(",") if x.strip()]


def apply_chat_template_no_thinking(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def strip_thinking(text: str) -> str:
    text = str(text).strip()
    end = text.rfind("</think>")
    if end >= 0:
        text = text[end + len("</think>") :]
    return text.replace("<think>", "").replace("</think>", "").strip()


class JSLQwenPrefixTranslator(nn.Module):
    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-1.7B",
        keypoint_dim: int = KEYPOINT_DIM,
        num_visual_tokens: int = 64,
        projector_hidden: int = 2048,
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: str = "all",
        gradient_checkpointing: bool = True,
        adapter_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model = base_model
        self.keypoint_dim = int(keypoint_dim)
        self.num_visual_tokens = int(num_visual_tokens)
        self.projector_hidden = int(projector_hidden)
        self.lora_target_modules = lora_target_modules

        tok_source = str(tokenizer_path) if tokenizer_path is not None else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.visual_tokens = [f"<|jsl_video_{i}|>" for i in range(self.num_visual_tokens)]
        self.tokenizer.add_special_tokens({"additional_special_tokens": self.visual_tokens})
        self.visual_token_ids = self.tokenizer.convert_tokens_to_ids(self.visual_tokens)

        model_kwargs: Dict[str, object] = {"trust_remote_code": True, "torch_dtype": parse_dtype(torch_dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["device_map"] = "auto"

        self.lm = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        self.lm.resize_token_embeddings(len(self.tokenizer))
        hidden_size = int(self.lm.config.hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(self.keypoint_dim, self.projector_hidden),
            nn.GELU(),
            nn.Linear(self.projector_hidden, hidden_size),
        )

        if adapter_path is not None:
            from peft import PeftModel

            self.lm = PeftModel.from_pretrained(self.lm, str(adapter_path), is_trainable=False)
        else:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

            if load_in_4bit:
                self.lm = prepare_model_for_kbit_training(self.lm)
            if gradient_checkpointing and hasattr(self.lm, "gradient_checkpointing_enable"):
                self.lm.gradient_checkpointing_enable()
                if hasattr(self.lm.config, "use_cache"):
                    self.lm.config.use_cache = False
            config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=default_lora_targets(lora_target_modules),
            )
            self.lm = get_peft_model(self.lm, config)

    def prompt_ids(self, device: Optional[torch.device] = None) -> torch.Tensor:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
        prompt = apply_chat_template_no_thinking(self.tokenizer, messages, add_generation_prompt=True)
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        if device is not None:
            ids = ids.to(device)
        return ids

    def _visual_embeds(self, keypoints: torch.Tensor) -> torch.Tensor:
        if keypoints.ndim != 3:
            raise RuntimeError(f"Expected keypoints [B,T,D], got {tuple(keypoints.shape)}")
        return self.projector(keypoints)

    def forward(self, keypoints: torch.Tensor, prompt_input_ids: torch.Tensor, target_input_ids: torch.Tensor):
        device = keypoints.device
        embed = self.lm.get_input_embeddings()
        visual_embeds = self._visual_embeds(keypoints)

        prompt_input_ids = prompt_input_ids.to(device)
        target_input_ids = target_input_ids.to(device)

        prompt_mask = prompt_input_ids.ne(self.tokenizer.pad_token_id)
        target_mask = target_input_ids.ne(self.tokenizer.pad_token_id)

        prompt_embeds = embed(prompt_input_ids.clamp_min(0))
        target_embeds = embed(target_input_ids.clamp_min(0))
        inputs_embeds = torch.cat([prompt_embeds, visual_embeds, target_embeds], dim=1)

        visual_mask = torch.ones(
            (keypoints.shape[0], self.num_visual_tokens),
            dtype=torch.bool,
            device=device,
        )
        attention_mask = torch.cat([prompt_mask, visual_mask, target_mask], dim=1).long()

        ignore_prompt = torch.full_like(prompt_input_ids, -100)
        ignore_visual = torch.full(
            (keypoints.shape[0], self.num_visual_tokens),
            -100,
            dtype=torch.long,
            device=device,
        )
        target_labels = target_input_ids.masked_fill(~target_mask, -100)
        labels = torch.cat([ignore_prompt, ignore_visual, target_labels], dim=1)

        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate_texts(
        self,
        keypoints: torch.Tensor,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> List[str]:
        self.eval()
        device = next(self.projector.parameters()).device
        keypoints = keypoints.to(device)
        batch_size = keypoints.shape[0]
        embed = self.lm.get_input_embeddings()
        prompt_ids = self.prompt_ids(device=device).unsqueeze(0).expand(batch_size, -1)
        prompt_embeds = embed(prompt_ids)
        visual_embeds = self._visual_embeds(keypoints)
        inputs_embeds = torch.cat([prompt_embeds, visual_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)
        generated = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        next_attention = attention_mask

        for _ in range(int(max_new_tokens)):
            logits = outputs.logits[:, -1, :]
            if temperature and temperature > 0:
                logits = logits / float(temperature)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = probs.cumsum(dim=-1)
                    remove = cumulative > float(top_p)
                    remove[:, 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                    logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.tokenizer.eos_token_id),
                next_token,
            )
            for i, token_id in enumerate(next_token.detach().cpu().tolist()):
                if not finished[i]:
                    if token_id == self.tokenizer.eos_token_id:
                        finished[i] = True
                    else:
                        generated[i].append(int(token_id))
            if bool(finished.all()):
                break

            next_attention = torch.cat(
                [next_attention, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
                dim=1,
            )
            outputs = self.lm(
                input_ids=next_token.unsqueeze(1),
                attention_mask=next_attention,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )

        texts = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in generated]
        return [strip_thinking(x) for x in texts]

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)
        self.lm.save_pretrained(output_dir / "qwen_lora")
        torch.save(self.projector.state_dict(), output_dir / "projector.pt")
        config = {
            "base_model": self.base_model,
            "keypoint_dim": self.keypoint_dim,
            "num_visual_tokens": self.num_visual_tokens,
            "projector_hidden": self.projector_hidden,
            "lora_target_modules": self.lora_target_modules,
        }
        (output_dir / "jsl_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, torch_dtype: str = "auto", load_in_4bit: bool = False):
        model_dir = Path(model_dir)
        config_path = model_dir / "jsl_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing JSL model config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = cls(
            base_model=config["base_model"],
            keypoint_dim=int(config["keypoint_dim"]),
            num_visual_tokens=int(config["num_visual_tokens"]),
            projector_hidden=int(config["projector_hidden"]),
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            adapter_path=model_dir / "qwen_lora",
            tokenizer_path=model_dir,
        )
        state = torch.load(model_dir / "projector.pt", map_location="cpu")
        model.projector.load_state_dict(state)
        return model
