# Non-Oracle Text Pilot Study

Status date: 2026-11-13

This document records a small supporting experiment for the paper. It is not
part of the main five-backbone visual study. We use MMGCN and MAGTKD as two
representative dialogue-emotion backbones to examine whether automatically
generated JSL text is reliable enough for cross-dataset multimodal transfer.

## Setup

The source model is trained on MELD and evaluated on eJSL. MELD provides gold
English transcripts. For eJSL, text is obtained without using the utterance
transcript at inference time:

```text
eJSL video -> JSL-to-Japanese model -> Japanese prediction
           -> Japanese-to-English translation -> RoBERTa text feature
```

The visual feature is extracted with the same TimeSformer pipeline used in the
visual main study. Checkpoints are selected by MELD weighted F1. All results in
this pilot use seed 42.

## JSL Translation Quality

The usable J-Shuwa CC pipeline produced 11,516 cleaned training segments from
15,855 downloaded raw candidates. The best J-Shuwa-only Qwen3-4B checkpoint
remained weak on the 1,920 eJSL utterances:

| Training data | Char BLEU-1 | Char BLEU-2 | Char BLEU-4 | Char F1 | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| J-Shuwa only | - | - | .0026 | .1068 | - |
| J-Shuwa initialization + eJSL train split | .3229 | .1518 | .0530 | .3281 | .2265 |

The second row is measured on the held-out 20% eJSL split. Direct eJSL
adaptation improves lexical overlap substantially, but the remaining BLEU and
ROUGE scores show that the generated sentence is still an imperfect semantic
substitute for the MELD transcript.

## Downstream Evidence

The selected non-oracle text is:

```text
/raid_zoe/home/lr/maokeyu/sign/jsl_old_model_ejsl_split_finetune/ejsl_nonoracle_txt_best
```

Text is highly predictive on MELD but transfers much less reliably to eJSL:

| Backbone | MELD text-only wF1 | eJSL text-only wF1 | Gap |
|---|---:|---:|---:|
| MMGCN | .6910 | .2337 | -.4574 |
| MAGTKD | .5099 | .2899 | -.2200 |

This is the expected reliability mismatch: training observes clean human MELD
transcripts, whereas testing observes generated and translated JSL text.

The same mismatch is visible after fusion:

| Backbone | Visual-only eJSL wF1 | TV, no text dropout | Change |
|---|---:|---:|---:|
| MMGCN | .3726 | .2195 | -.1531 |
| MAGTKD | .2894 | .2811 | -.0083 |

For both representative backbones, directly adding the non-oracle text fails
to improve the visual model. The MMGCN degradation is especially large.

## Text-Dropout Intervention

During MELD TV training, we randomly replace an utterance's text feature with
zero while retaining the visual feature. Evaluation still uses the complete
non-oracle text. This intervention directly tests whether reducing reliance on
the clean MELD transcript improves transfer.

| Backbone | No dropout | Best text dropout | eJSL wF1 | Change |
|---|---:|---:|---:|---:|
| MMGCN | .2195 | 75% | .4005 | +.1810 |
| MAGTKD | .2811 | 50% | .2885 | +.0074 |

MMGCN shows a strong recovery, and MAGTKD shows a smaller improvement. The
result supports the interpretation that unrestricted TV training can
overweight the clean source transcript and that modality regularization can
partly restore visual reliance when target text is noisy.

## Paper Takeaway

This pilot supports two concise observations:

1. Available JSL supervision does not yet produce text with the semantic
   fidelity of MELD's human transcript.
2. A model trained with clean MELD text can rely on that modality more than is
   appropriate for generated eJSL text; text dropout can mitigate this
   behavior.

Accordingly, the main BOBSL transfer study isolates the visual pathway. This
keeps the auxiliary signal and evaluation pathway controlled, while
non-oracle text remains a supporting extension rather than a confounding part
of the main comparison.

The evidence should be described as a two-backbone pilot, not as a universal
claim that every multimodal architecture is harmed by non-oracle text.

## Artifacts

Translation metrics:

```text
/raid_zoe/home/lr/maokeyu/sign/jsl_nonoracle_qwen3_4b_full_clean/ejsl_eval/qwen3_4b_jshuwa_all_clean/best_checkpoint.json
/raid_zoe/home/lr/maokeyu/sign/jsl_old_model_ejsl_split_finetune/eval_checkpoints/best_checkpoint.json
```

MMGCN text and fusion results:

```text
/raid_zoe/home/lr/maokeyu/sign/jsl_four_nonoracle_mmgcn/four_nonoracle_text_rank.csv
/raid_zoe/home/lr/maokeyu/sign/mmgcn_tv_text_dropout_nonoracle/tv_text_dropout_rank.csv
```

MAGTKD text and fusion results:

```text
/raid_zoe/home/lr/maokeyu/sign/non_mmgcn_text_dropout_nonoracle/non_mmgcn_text_dropout_summary.csv
```

Entry points:

```text
run_jsl_four_nonoracle_base_and_mmgcn.sh
run_mmgcn_tv_text_dropout_nonoracle.sh
run_non_mmgcn_text_dropout_nonoracle.sh
```
