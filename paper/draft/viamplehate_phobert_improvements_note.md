# Note: Improvements of Proposed ViAmpleHate_PhoBERT over Baseline AmpleHate_PhoBERT

## Context

This note summarizes the main improvements of:

- `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/`

over:

- `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/`

Both models use `vinai/phobert-base` on the binary ViHSD setting, where `CLEAN` and `OFFENSIVE` are merged into `NON-HATE`, and `HATE` remains the positive class.

## Methodological Improvements

| Component | Baseline AmpleHate_PhoBERT | Proposed ViAmpleHate_PhoBERT |
|---|---|---|
| Target extraction | English NER: `dbmdz/bert-large-cased-finetuned-conll03-english` | Vietnamese NER: `NlpHUST/ner-vietnamese-electra-base` |
| Target coverage | Very low NER coverage: `21/24048` training samples, about `0.09%` | Vietnamese NER plus target cue lexicon; observed target cue coverage around `18.8-20.0%` on checked train/validation samples |
| Target modeling | Mostly falls back to `[CLS]`, making the model close to a standard PhoBERT classifier | Uses explicit Vietnamese target cues and keeps `[CLS]` only as an implicit fallback |
| Attack signal | No separate attack cue modeling | Adds an attack cue bank for offensive predicates, e.g., `khinh`, `ăn_bám` |
| Attention modules | One HeadAttention module | Three HeadAttention modules: explicit target, implicit CLS anchor, and attack cue |
| Batch attention behavior | Original implementation uses `matmul(Q, K.T)`, which can mix samples across a batch | Uses batched attention with `bmm`, so each sample attends only to its own cue tokens |
| Fusion strategy | Fixed scalar injection: `CLS + e * head_attention`, with `e=1.0` | Relation bank fusion via `relation_proj`, followed by an instance-adaptive gate |
| Adaptive gating | Not available | Uses `g = sigmoid(W[h_CLS; r])` to control how much relation information is injected per instance |
| Loss function | Weighted CrossEntropy | Weighted CrossEntropy plus ContrastiveLoss with `alpha=0.1` |
| Input length | `max_len=128` | `max_len=256` |
| Training setup | Batch size 16, 6 epochs | Batch size 16 with gradient accumulation 2, effective batch size 32, up to 8 epochs |
| Evaluation policy | NER disabled for validation/test, causing `[CLS]` fallback during evaluation | Target extraction kept consistent across train/validation/test via `USE_NER_AT_EVAL=True` |

## Quantitative Improvements

| Metric | Baseline | Proposed | Delta |
|---|---:|---:|---:|
| Accuracy | `0.9175` | `0.9205` | `+0.0030` |
| Macro Precision | `0.7762` | `0.7859` | `+0.0097` |
| Macro Recall | `0.7823` | `0.7781` | `-0.0042` |
| Macro F1 | `0.7792` | `0.7819` | `+0.0027` |
| HATE F1 | `0.6045` | `0.6081` | `+0.0036` |

## Interpretation

The proposed model provides a modest but consistent improvement in overall performance, especially in precision. HATE-class precision improves from `0.5972` to `0.6177`, while HATE recall decreases from `0.6119` to `0.5988`. This suggests that the proposed model is more conservative but more precise when predicting hate speech.

The main contribution is not only the metric gain, but the Vietnamese adaptation of AmpleHate. The baseline is a faithful port of the original English AmpleHate pipeline, but it is poorly aligned with Vietnamese because its English NER almost never detects real targets. The proposed model addresses this by introducing Vietnamese NER, target cue mining, attack cue mining, relation-bank attention, corrected batched attention, and adaptive relation injection.

## Draft-ready Summary

Compared with the direct AmpleHate baseline, the proposed ViAmpleHate_PhoBERT adapts target-aware hate speech modeling to Vietnamese by replacing English NER with Vietnamese NER, augmenting target detection with Vietnamese target cue lexicons, and separately modeling attack cues. Architecturally, it extends the single HeadAttention mechanism into a relation bank that captures explicit target, implicit context, and attack relations, then fuses these signals through an instance-adaptive gate rather than a fixed injection scalar. It also corrects the batched attention computation to prevent cross-sample attention contamination. Empirically, the proposed model improves test Accuracy from `0.9175` to `0.9205`, Macro-F1 from `0.7792` to `0.7819`, and HATE-F1 from `0.6045` to `0.6081`, indicating a small but meaningful improvement while providing a more linguistically appropriate Vietnamese hate-speech modeling pipeline.
