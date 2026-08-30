import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader, Subset

from dataloader import MELDDataset
from model import DialogueGCNModel


SCRIPT_DIR = Path(__file__).resolve().parent
MMGCN_MELD_LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
ANJS_LABELS = ["A", "N", "J", "S"]
MMGCN_MELD_TO_ANJS = {
    6: 0,  # anger -> A
    0: 1,  # neutral -> N
    4: 2,  # joy -> J
    3: 3,  # sadness -> S
}
MODALITY_ALIASES = {
    "full": "full",
    "avl": "full",
    "text": "text",
    "t": "text",
    "l": "text",
    "video": "video",
    "visual": "video",
    "v": "video",
    "tv": "tv",
    "vt": "tv",
    "vl": "tv",
    "text_video": "tv",
    "video_text": "tv",
    "audio": "audio",
    "a": "audio",
}


class LegacyMELDFocalLoss(nn.Module):
    """Device-safe version of the focal loss used by MMGCN/train.py for MELD."""

    def __init__(self, gamma: float = 2.5, alpha: float = 1.0, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_length = logits.size(1)
        seq_length = logits.size(0)
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros(seq_length, labels_length, device=logits.device).scatter_(1, new_label, 1)
        log_p = F.log_softmax(logits, dim=1)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt) ** self.gamma * log_p
        return fl.mean() if self.size_average else fl.sum()


class FocalNLLLoss(nn.Module):
    def __init__(self, gamma: float = 2.5, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        true_log_probs = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
        pt = true_log_probs.exp().clamp_min(1e-8)
        return (-self.alpha * (1 - pt) ** self.gamma * true_log_probs).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_path(raw: str, must_exist: bool = False) -> Path:
    p = Path(raw).expanduser()
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, SCRIPT_DIR / p, SCRIPT_DIR.parent / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        checked = ", ".join(str(x) for x in candidates)
        raise FileNotFoundError(f"Path not found: {raw}. Checked: {checked}")
    return candidates[0].resolve()


def resolve_output_dir(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def normalize_input_modality(value: str) -> str:
    key = (value or "full").strip().lower()
    if key not in MODALITY_ALIASES:
        valid = ", ".join(sorted(MODALITY_ALIASES))
        raise ValueError(f"Unsupported modality={value!r}; valid values: {valid}")
    return MODALITY_ALIASES[key]


def parse_modalities(raw_values) -> List[str]:
    out: List[str] = []
    for value in raw_values:
        for part in str(value).replace(",", " ").split():
            modality = normalize_input_modality(part)
            if modality not in out:
                out.append(modality)
    if not out:
        raise ValueError("No modality requested")
    return out


def zero_unused_modalities(textf, visuf, acouf, input_modality: str):
    if input_modality == "text":
        return textf, torch.zeros_like(visuf), torch.zeros_like(acouf)
    if input_modality == "video":
        return torch.zeros_like(textf), visuf, torch.zeros_like(acouf)
    if input_modality == "tv":
        return textf, visuf, torch.zeros_like(acouf)
    if input_modality == "audio":
        return torch.zeros_like(textf), torch.zeros_like(visuf), acouf
    return textf, visuf, acouf


def split_indices(size: int, valid_ratio: float, max_train_dialogues: int) -> tuple:
    indices = list(range(size))
    split = int(max(min(valid_ratio, 0.9), 0.0) * size)
    valid_indices = indices[:split]
    train_indices = indices[split:]
    if max_train_dialogues > 0:
        train_indices = train_indices[:max_train_dialogues]
    return train_indices, valid_indices


def build_loaders(args, feature_pkl: Path):
    trainset = MELDDataset(str(feature_pkl), train=True)
    testset = MELDDataset(str(feature_pkl), train=False)
    train_indices, valid_indices = split_indices(len(trainset), args.valid_ratio, args.max_train_dialogues)
    test_indices = list(range(len(testset)))
    if args.max_test_dialogues > 0:
        test_indices = test_indices[: args.max_test_dialogues]

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(trainset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.no_cuda,
        generator=generator,
    )
    valid_loader = None
    if valid_indices:
        valid_loader = DataLoader(
            Subset(trainset, valid_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=trainset.collate_fn,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available() and not args.no_cuda,
        )
    test_loader = DataLoader(
        Subset(testset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=testset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.no_cuda,
    )
    return train_loader, valid_loader, test_loader


def build_model(args, device: torch.device) -> DialogueGCNModel:
    feat2dim = {
        "MELD_text": 600,
        "MELD_audio": 300,
        "denseface": 342,
    }
    d_audio = feat2dim["MELD_audio"]
    d_visual = feat2dim["denseface"]
    d_text = feat2dim["MELD_text"]
    d_m = d_text if args.mm_fusion_mthd != "concat" else d_audio + d_visual + d_text

    model = DialogueGCNModel(
        args.base_model,
        d_m,
        150,
        150,
        100,
        100,
        100,
        100,
        n_speakers=9,
        max_seq_len=200,
        window_past=args.windowp,
        window_future=args.windowf,
        n_classes=len(MMGCN_MELD_LABELS),
        listener_state=args.active_listener,
        context_attention=args.attention,
        dropout=args.dropout,
        nodal_attention=args.nodal_attention,
        no_cuda=args.no_cuda,
        graph_type=args.graph_type,
        use_topic=args.use_topic,
        alpha=args.alpha,
        multiheads=args.multiheads,
        graph_construct=args.graph_construct,
        use_GCN=args.use_gcn,
        use_residue=not args.no_residue,
        D_m_v=d_visual,
        D_m_a=d_audio,
        modals=args.modals,
        att_type=args.mm_fusion_mthd,
        av_using_lstm=args.av_using_lstm,
        Deep_GCN_nlayers=args.deep_gcn_nlayers,
        dataset="MELD",
        use_speaker=args.use_speaker,
        use_modal=args.use_modal,
    )
    return model.to(device)


def build_loss(args) -> nn.Module:
    if args.loss == "legacy_focal":
        return LegacyMELDFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    if args.loss == "focal":
        return FocalNLLLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    if args.loss == "nll":
        return nn.NLLLoss()
    raise ValueError(f"Unsupported loss: {args.loss}")


def sequence_lengths(umask: torch.Tensor) -> List[int]:
    lengths = []
    for row in umask:
        idx = (row == 1).nonzero(as_tuple=False)
        if idx.numel() == 0:
            lengths.append(0)
        else:
            lengths.append(int(idx[-1].item()) + 1)
    return lengths


def run_epoch(
    model: DialogueGCNModel,
    loss_function: nn.Module,
    dataloader: Optional[DataLoader],
    device: torch.device,
    input_modality: str,
    optimizer: Optional[optim.Optimizer] = None,
    max_grad_norm: float = 0.0,
) -> Optional[Dict]:
    if dataloader is None:
        return None

    train = optimizer is not None
    model.train() if train else model.eval()
    losses = []
    loss_weights = []
    all_golds: List[int] = []
    all_preds: List[int] = []
    all_log_probs: List[np.ndarray] = []
    all_ids: List[str] = []

    for data in dataloader:
        if train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-1]]
        textf, visuf, acouf = zero_unused_modalities(textf, visuf, acouf, input_modality)
        lengths = sequence_lengths(umask)

        log_prob, *_ = model(textf, qmask, umask, lengths, acouf, visuf)
        labels_flat = torch.cat([label[j][: lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, labels_flat)

        if train:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(log_prob, dim=1)
            n_items = int(labels_flat.numel())
            losses.append(float(loss.item()) * max(n_items, 1))
            loss_weights.append(max(n_items, 1))
            all_golds.extend(labels_flat.detach().cpu().numpy().tolist())
            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_log_probs.append(log_prob.detach().cpu().numpy())

            vids = data[-1]
            for batch_idx, vid in enumerate(vids):
                for utt_idx in range(lengths[batch_idx]):
                    all_ids.append(f"{vid}:utt{utt_idx}")

    if not all_golds:
        return None

    golds = np.asarray(all_golds, dtype=np.int64)
    preds = np.asarray(all_preds, dtype=np.int64)
    log_probs = np.concatenate(all_log_probs, axis=0)
    label_ids = list(range(len(MMGCN_MELD_LABELS)))
    return {
        "loss": float(np.sum(losses) / max(np.sum(loss_weights), 1)),
        "accuracy": float(metrics.accuracy_score(golds, preds)),
        "macro_f1": float(metrics.f1_score(golds, preds, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds, preds, labels=label_ids, average="weighted", zero_division=0)),
        "golds": golds,
        "preds": preds,
        "log_probs": log_probs,
        "ids": all_ids,
    }


def build_count_table(values: np.ndarray, n_labels: int) -> List[int]:
    return np.bincount(values.astype(np.int64), minlength=n_labels).tolist()


def write_confusion_matrix(cm: np.ndarray, labels: List[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("labels: " + ",".join(labels) + "\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")


def top_confusions(cm: np.ndarray, labels: List[str], limit: int = 20) -> List[Dict]:
    rows = []
    for i, gold in enumerate(labels):
        for j, pred in enumerate(labels):
            if i == j or cm[i, j] == 0:
                continue
            rows.append({"gold": gold, "pred": pred, "count": int(cm[i, j])})
    return sorted(rows, key=lambda x: x["count"], reverse=True)[:limit]


def save_prediction_csv(result: Dict, out_dir: Path, prefix: str) -> None:
    probs = np.exp(result["log_probs"])
    with open(out_dir / f"{prefix}_meld7_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["item_id", "gold_id", "gold_label", "pred_id", "pred_label", "correct"]
            + [f"prob_{name}" for name in MMGCN_MELD_LABELS]
        )
        for item_id, gold, pred, prob in zip(result["ids"], result["golds"], result["preds"], probs):
            writer.writerow(
                [item_id, int(gold), MMGCN_MELD_LABELS[int(gold)], int(pred), MMGCN_MELD_LABELS[int(pred)], int(gold == pred)]
                + [float(x) for x in prob]
            )


def save_meld7_reports(result: Dict, out_dir: Path, prefix: str) -> Dict:
    golds = result["golds"]
    preds = result["preds"]
    probs = np.exp(result["log_probs"])
    label_ids = list(range(len(MMGCN_MELD_LABELS)))
    cm = metrics.confusion_matrix(golds, preds, labels=label_ids)
    report = metrics.classification_report(
        golds,
        preds,
        labels=label_ids,
        target_names=MMGCN_MELD_LABELS,
        digits=4,
        zero_division=0,
    )

    np.save(out_dir / f"{prefix}_meld7_confusion_matrix.npy", cm)
    write_confusion_matrix(cm, MMGCN_MELD_LABELS, out_dir / f"{prefix}_meld7_confusion_matrix.txt")
    (out_dir / f"{prefix}_meld7_classification_report.txt").write_text(report, encoding="utf-8")
    save_prediction_csv(result, out_dir, prefix)

    per_p, per_r, per_f1, per_support = metrics.precision_recall_fscore_support(
        golds,
        preds,
        labels=label_ids,
        zero_division=0,
    )
    per_class = {
        MMGCN_MELD_LABELS[i]: {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
            "support": int(per_support[i]),
            "gold_count": int(cm[i].sum()),
            "pred_count": int(cm[:, i].sum()),
            "mean_pred_prob": float(probs[:, i].mean()) if len(probs) else 0.0,
        }
        for i in label_ids
    }
    summary = {
        "n_samples": int(len(golds)),
        "loss": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_f1": float(result["weighted_f1"]),
        "gold_counts": dict(zip(MMGCN_MELD_LABELS, build_count_table(golds, len(MMGCN_MELD_LABELS)))),
        "pred_counts": dict(zip(MMGCN_MELD_LABELS, build_count_table(preds, len(MMGCN_MELD_LABELS)))),
        "top_confusions": top_confusions(cm, MMGCN_MELD_LABELS),
        "per_class": per_class,
    }
    return summary


def save_anjs_projection_reports(result: Dict, out_dir: Path, prefix: str) -> Dict:
    golds = result["golds"]
    preds = result["preds"]
    probs = np.exp(result["log_probs"])
    keep_gold = np.isin(golds, list(MMGCN_MELD_TO_ANJS))
    if not keep_gold.any():
        return {"n_samples": 0, "note": "no MELD samples in anger/neutral/joy/sadness subset"}

    golds_sub = golds[keep_gold]
    preds_sub = preds[keep_gold]
    probs_sub = probs[keep_gold]
    golds4 = np.asarray([MMGCN_MELD_TO_ANJS[int(x)] for x in golds_sub], dtype=np.int64)
    probs4 = np.stack(
        [
            probs_sub[:, 6],  # A: anger
            probs_sub[:, 0],  # N: neutral
            probs_sub[:, 4],  # J: joy
            probs_sub[:, 3],  # S: sadness
        ],
        axis=1,
    )
    preds4 = probs4.argmax(axis=1).astype(np.int64)

    label_ids = list(range(len(ANJS_LABELS)))
    cm = metrics.confusion_matrix(golds4, preds4, labels=label_ids)
    report = metrics.classification_report(
        golds4,
        preds4,
        labels=label_ids,
        target_names=ANJS_LABELS,
        digits=4,
        zero_division=0,
    )
    np.save(out_dir / f"{prefix}_anjs_projection_confusion_matrix.npy", cm)
    write_confusion_matrix(cm, ANJS_LABELS, out_dir / f"{prefix}_anjs_projection_confusion_matrix.txt")
    (out_dir / f"{prefix}_anjs_projection_classification_report.txt").write_text(report, encoding="utf-8")

    oos_ids = [1, 2, 5]  # surprise, fear, disgust in MMGCN/MELD order.
    oos_by_gold = {}
    for meld_id, anjs_id in MMGCN_MELD_TO_ANJS.items():
        mask = golds_sub == meld_id
        oos_by_gold[ANJS_LABELS[anjs_id]] = {
            MMGCN_MELD_LABELS[oos_id]: int(np.logical_and(mask, preds_sub == oos_id).sum()) for oos_id in oos_ids
        }

    keep_pair = np.isin(preds_sub, list(MMGCN_MELD_TO_ANJS))
    filtered_summary = {"n_samples": int(keep_pair.sum())}
    if keep_pair.any():
        golds4_filter = golds4[keep_pair]
        preds4_filter = np.asarray([MMGCN_MELD_TO_ANJS[int(x)] for x in preds_sub[keep_pair]], dtype=np.int64)
        cm_filter = metrics.confusion_matrix(golds4_filter, preds4_filter, labels=label_ids)
        report_filter = metrics.classification_report(
            golds4_filter,
            preds4_filter,
            labels=label_ids,
            target_names=ANJS_LABELS,
            digits=4,
            zero_division=0,
        )
        np.save(out_dir / f"{prefix}_anjs_trainpy_filter_confusion_matrix.npy", cm_filter)
        write_confusion_matrix(cm_filter, ANJS_LABELS, out_dir / f"{prefix}_anjs_trainpy_filter_confusion_matrix.txt")
        (out_dir / f"{prefix}_anjs_trainpy_filter_classification_report.txt").write_text(report_filter, encoding="utf-8")
        filtered_summary.update(
            {
                "accuracy": float(metrics.accuracy_score(golds4_filter, preds4_filter)),
                "macro_f1": float(metrics.f1_score(golds4_filter, preds4_filter, labels=label_ids, average="macro", zero_division=0)),
                "weighted_f1": float(metrics.f1_score(golds4_filter, preds4_filter, labels=label_ids, average="weighted", zero_division=0)),
                "dropped_oos_predictions": int((~keep_pair).sum()),
            }
        )

    return {
        "n_samples": int(len(golds4)),
        "accuracy": float(metrics.accuracy_score(golds4, preds4)),
        "macro_f1": float(metrics.f1_score(golds4, preds4, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(golds4, preds4, labels=label_ids, average="weighted", zero_division=0)),
        "gold_counts": dict(zip(ANJS_LABELS, build_count_table(golds4, len(ANJS_LABELS)))),
        "pred_counts": dict(zip(ANJS_LABELS, build_count_table(preds4, len(ANJS_LABELS)))),
        "raw_out_of_scope_predictions_by_gold": oos_by_gold,
        "trainpy_filter_style": filtered_summary,
    }


def save_result_bundle(result: Dict, out_dir: Path, prefix: str, extra_summary: Dict) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    meld_summary = save_meld7_reports(result, out_dir, prefix)
    anjs_summary = save_anjs_projection_reports(result, out_dir, prefix)
    summary = {
        **extra_summary,
        "meld7": meld_summary,
        "anjs_projection_on_meld_subset": anjs_summary,
    }
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def train_and_eval_one_modality(args, modality: str, feature_pkl: Path, device: torch.device, root_out_dir: Path) -> None:
    seed_everything(args.seed)
    train_loader, valid_loader, test_loader = build_loaders(args, feature_pkl)
    model = build_model(args, device)
    loss_function = build_loss(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    modality_out_dir = root_out_dir / modality
    modality_out_dir.mkdir(parents=True, exist_ok=True)
    epoch_rows = []
    best_test = None
    best_epoch = 0
    final_test = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_result = run_epoch(
            model,
            loss_function,
            train_loader,
            device,
            modality,
            optimizer=optimizer,
            max_grad_norm=args.max_grad_norm,
        )
        valid_result = run_epoch(model, loss_function, valid_loader, device, modality)

        test_result = None
        if epoch % max(args.eval_every, 1) == 0 or epoch == args.epochs:
            test_result = run_epoch(model, loss_function, test_loader, device, modality)
            final_test = test_result
            if test_result is not None and (best_test is None or test_result["weighted_f1"] > best_test["weighted_f1"]):
                best_test = {
                    key: value.copy() if isinstance(value, np.ndarray) else list(value) if isinstance(value, list) else value
                    for key, value in test_result.items()
                }
                best_epoch = epoch

        row = {
            "epoch": epoch,
            "time_sec": round(time.time() - start, 2),
            "train_loss": train_result["loss"] if train_result else None,
            "train_acc": train_result["accuracy"] if train_result else None,
            "train_macro_f1": train_result["macro_f1"] if train_result else None,
            "train_weighted_f1": train_result["weighted_f1"] if train_result else None,
            "valid_loss": valid_result["loss"] if valid_result else None,
            "valid_acc": valid_result["accuracy"] if valid_result else None,
            "valid_macro_f1": valid_result["macro_f1"] if valid_result else None,
            "valid_weighted_f1": valid_result["weighted_f1"] if valid_result else None,
            "test_loss": test_result["loss"] if test_result else None,
            "test_acc": test_result["accuracy"] if test_result else None,
            "test_macro_f1": test_result["macro_f1"] if test_result else None,
            "test_weighted_f1": test_result["weighted_f1"] if test_result else None,
        }
        epoch_rows.append(row)

        test_msg = ""
        if test_result is not None:
            test_msg = (
                f" test_loss={test_result['loss']:.4f}"
                f" test_acc={test_result['accuracy']:.4f}"
                f" test_wf1={test_result['weighted_f1']:.4f}"
            )
        valid_msg = ""
        if valid_result is not None:
            valid_msg = f" valid_loss={valid_result['loss']:.4f} valid_wf1={valid_result['weighted_f1']:.4f}"
        print(
            f"[MMGCN][{modality}] epoch={epoch}"
            f" train_loss={train_result['loss']:.4f}"
            f" train_acc={train_result['accuracy']:.4f}"
            f" train_wf1={train_result['weighted_f1']:.4f}"
            f"{valid_msg}{test_msg} time={row['time_sec']}s"
        )

    metrics_path = modality_out_dir / f"mmgcn_{modality}_epoch_metrics.csv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
        writer.writeheader()
        writer.writerows(epoch_rows)

    if final_test is None:
        final_test = run_epoch(model, loss_function, test_loader, device, modality)
    if best_test is None:
        best_test = final_test
        best_epoch = args.epochs

    base_summary = {
        "script": "MMGCN/train_eval_mmgcn_meld_detailed.py",
        "feature_pkl": str(feature_pkl),
        "modality": modality,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l2": args.l2,
        "dropout": args.dropout,
        "loss": args.loss,
        "label_order": MMGCN_MELD_LABELS,
        "best_epoch_by_test_weighted_f1": best_epoch,
        "epoch_metrics_csv": str(metrics_path),
    }
    final_summary = save_result_bundle(
        final_test,
        modality_out_dir,
        f"mmgcn_{modality}_test_final",
        {**base_summary, "selection": "final_epoch"},
    )
    best_summary = save_result_bundle(
        best_test,
        modality_out_dir,
        f"mmgcn_{modality}_test_best",
        {**base_summary, "selection": "best_test_weighted_f1"},
    )

    print(
        f"[MMGCN][{modality}] final MELD7 acc={final_summary['meld7']['accuracy']:.4f} "
        f"macro_f1={final_summary['meld7']['macro_f1']:.4f} "
        f"weighted_f1={final_summary['meld7']['weighted_f1']:.4f}"
    )
    print(
        f"[MMGCN][{modality}] best_epoch={best_epoch} MELD7 acc={best_summary['meld7']['accuracy']:.4f} "
        f"macro_f1={best_summary['meld7']['macro_f1']:.4f} "
        f"weighted_f1={best_summary['meld7']['weighted_f1']:.4f}"
    )
    print(f"[MMGCN][{modality}] reports saved to: {modality_out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MMGCN on MELD and save detailed MELD reports.")
    parser.add_argument("--feature_pkl", type=str, default="MELD_features/MELD_features_raw1.pkl")
    parser.add_argument("--out_dir", type=str, default="saved/meld_detailed")
    parser.add_argument("--modalities", nargs="+", default=["text", "video", "tv"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_dialogues", type=int, default=0)
    parser.add_argument("--max_test_dialogues", type=int, default=0)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--l2", type=float, default=0.00003)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--loss", choices=["legacy_focal", "focal", "nll"], default="legacy_focal")
    parser.add_argument("--focal_gamma", type=float, default=2.5)
    parser.add_argument("--focal_alpha", type=float, default=1.0)

    parser.add_argument("--base_model", default="LSTM")
    parser.add_argument("--graph_type", default="MMGCN")
    parser.add_argument("--graph_construct", default="direct")
    parser.add_argument("--mm_fusion_mthd", default="concat_subsequently")
    parser.add_argument("--modals", default="avl")
    parser.add_argument("--deep_gcn_nlayers", type=int, default=4)
    parser.add_argument("--windowp", type=int, default=10)
    parser.add_argument("--windowf", type=int, default=10)
    parser.add_argument("--attention", default="general")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--multiheads", type=int, default=6)
    parser.add_argument("--nodal_attention", action="store_true", default=True)
    parser.add_argument("--active_listener", action="store_true", default=False)
    parser.add_argument("--use_gcn", action="store_true", default=False)
    parser.add_argument("--use_topic", action="store_true", default=False)
    parser.add_argument("--use_speaker", action="store_true", default=True)
    parser.add_argument("--use_modal", action="store_true", default=False)
    parser.add_argument("--av_using_lstm", action="store_true", default=False)
    parser.add_argument("--no_residue", action="store_true", default=False)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modalities = parse_modalities(args.modalities)
    feature_pkl = resolve_path(args.feature_pkl, must_exist=True)
    out_dir = resolve_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if args.graph_type in {"MMGCN", "MMGCN2"} and device.type != "cuda":
        raise RuntimeError(
            "MMGCN/model_mm.py uses CUDA tensors internally. Run this script on the server with a GPU "
            "or choose a non-MMGCN graph_type."
        )

    print(f"[MMGCN] feature_pkl={feature_pkl}")
    print(f"[MMGCN] out_dir={out_dir}")
    print(f"[MMGCN] device={device} modalities={modalities}")
    for modality in modalities:
        train_and_eval_one_modality(args, modality, feature_pkl, device, out_dir)


if __name__ == "__main__":
    main()
