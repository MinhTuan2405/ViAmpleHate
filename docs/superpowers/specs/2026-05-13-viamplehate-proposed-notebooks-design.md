# ViAmpleHate Proposed Notebooks — Design Spec

**Date:** 2026-05-13  
**Datasets:** ViHSD, VOZ-HSD  
**Reference:** `docs/improvementAmpleHate.md`  
**Baseline notebooks:**
- `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/vihsd-baseline-amplehate-phobert.ipynb`
- `notebooks/models/baselines/VOZ-HSD - Baseline AmpleHate_PhoBERT/voz-hsd-baseline-amplehate-phobert.ipynb`

---

## Goal

Create two proposed notebooks that apply all 10 improvements documented in `improvementAmpleHate.md` to the baseline AmpleHate PhoBERT notebooks. The proposed notebooks are fully runnable, self-contained, and directly comparable to their baselines.

---

## Output Files

```
notebooks/models/proposed/
  vihsd-proposed-viamplehate-phobert/
    vihsd-proposed-viamplehate-phobert.ipynb
  voz-hsd-proposed-viamplehate-phobert/
    voz-hsd-proposed-viamplehate-phobert.ipynb
```

Folder name = notebook file stem (no `.ipynb`), matching the project convention.

---

## Structure

Both notebooks follow the same 17-section structure as the baselines, with improvements 1–5 active in the main flow (sections 1–17) and improvements 6–10 appended as dedicated optional sections (18–22).

---

## Active Improvements (Sections 1–17)

### §2 Hyperparameters

**Changes:**
- `NER_MODEL = 'NlpHUST/ner-vietnamese-electra-base'` (was English CoNLL-2003 model)
- Add `LAMBDA_CL = 0.1` (contrastive loss weight; tune range [0.05, 0.1, 0.2])
- Add `CONTRASTIVE_MARGIN = 0.5`
- Update `CKPT_NAME` and `PLOT_TITLE` to reflect "proposed" / "ViAmpleHate"

### §2b — New cell: Vietnamese Hate-Target Lexicon

Add `VIET_TARGET_LEXICON` as a Python `set` containing all terms from `improvementAmpleHate.md § Improvement 2`. This set covers:
- Derogatory pronouns/group markers ("thằng", "bọn", etc.)
- Gender, LGBTQ+, regional, ethnic, religious, political, occupation, social class, age, appearance categories
- Implicit reference patterns

### §6 Apply Preprocessing

Store both the word-segmented form (`text_processed`) and the original normalized but **unsegmented** text (`raw_text`) for each split. The segmented form is passed to PhoBERT; the raw form is passed to the NER pipeline to avoid segmentation mismatch.

```python
df['raw_text']       = df['free_text'].apply(normalize_text)   # for NER
df['text_processed'] = df['free_text'].apply(preprocess)       # for PhoBERT
```

### §8 NER and Target Extraction

**NERTagger:**
- Load `NlpHUST/ner-vietnamese-electra-base`
- Filter entity types: `{"PER", "ORG", "LOC", "MISC"}` (VLSP Vietnamese NER types)

**NERProcessor.extract_head_tokens(text):**
- Run NER on `text` (unsegmented)
- Scan `VIET_TARGET_LEXICON` against lowercased text
- Return combined list of entity words + lexicon matches

**NERProcessor.tokenize_and_encode(text_segmented, text_raw):**
- Run `extract_head_tokens(text_raw)` (NER + lexicon on raw text)
- Tokenize `text_segmented` for PhoBERT
- Map entity surface forms to segmented token positions: `ht.replace(' ', '_')` alignment; scan `seg_tokens` for a match
- Fall back to `[0]` (CLS) if no positions found

### §9 AmpleHateDataset

Store `raw_texts` (from `df['raw_text']`) alongside `texts` (from `df['text_processed']`).  
In `__getitem__`, pass both to `tokenize_and_encode(text_segmented, text_raw)`.

### §10 AmpleHatePhoBERT

In `forward`, after computing `final_embedding` and before applying dropout+classifier:

```python
self.last_embedding = final_embedding.detach()  # [batch, hidden] — for ContrastiveLoss
```

`ContrastiveLossCosine` class is already present in baseline — no change needed.

### §11 Loss Setup

Add:
```python
criterion_cl = ContrastiveLossCosine(margin=CONTRASTIVE_MARGIN)
```

### §12 Training Loop

Update `train_epoch` loss computation:
```python
ce_loss = criterion(logits, y)
cl_loss = criterion_cl(model.last_embedding, y)
loss    = ce_loss + LAMBDA_CL * cl_loss
```

---

## Optional Improvement Sections (18–22)

### §18 — Improvement 7: max_length Profiling

Runnable cell that computes token length distribution over the training set and prints how many samples would be truncated at the current `MAX_LEN`. Includes a commented-out config change to `MAX_LEN = 256` (with `BATCH_SIZE = 8`) for use if truncation rate exceeds 5%.

### §19 — Improvement 6: e-Injection Sweep

Runnable sweep loop over `e ∈ [0.5, 0.75, 1.0, 1.25, 1.5]`. For each value: trains a fresh model instance (same config, different `e`), evaluates on the validation set, and prints a F1 summary table. Depends on all Improvements 1–5 being in place (so NER coverage is high enough for `e` to matter).

### §20 — Improvement 8: PhoBERT-large Config

Commented-out hyperparameter block:
```python
# MODEL_NAME = 'vinai/phobert-large'
# HIDDEN_DIM = 1024
# HEAD_DIM   = 1024
# BATCH_SIZE = 8
```
With a note that these replace the base-model defaults; the HeadAttention and classifier resize automatically via `HIDDEN_DIM`.

### §21 — Improvement 9: Within-Example HeadAttention Variant

A standalone `HeadAttentionWithin` class implementing point-wise (within-example) attention:
```python
score  = (Q_h * K_h).sum(-1, keepdim=True) / (head_dim ** 0.5)  # [B, 1]
weight = torch.sigmoid(score)
return weight * V_h
```
A note explains how to swap it into `AmpleHatePhoBERT` and why it's more stable across batch sizes. Not active by default — user swaps `HeadAttention` → `HeadAttentionWithin` to compare.

### §22 — Improvement 10: Multi-Class Setup

Commented-out code block showing how to remove the CLEAN+OFFENSIVE → NON-HATE remapping, set `NUM_CLASSES = 3`, `LABEL_NAMES = ['CLEAN', 'OFFENSIVE', 'HATE']`, and switch metrics to `average='macro'` over 3 classes. A note explains the trade-off.

---

## VOZ-HSD Variant

The VOZ-HSD proposed notebook is identical to the ViHSD proposed notebook except:

| Aspect | ViHSD | VOZ-HSD |
|---|---|---|
| Dataset | `sonlam1102/vihsd`, pre-split train/val/test | `tarudesu/VOZ-HSD`, `split="train"`, rename `texts→free_text`, `labels→label_id`, stratified 100k sample, manual 80/10/10 split |
| Label remapping | `{0,1}→0, {2}→1` | Already binary, no remapping |
| `SAMPLE_SIZE` | N/A | `100_000` |
| `CKPT_NAME` | `best_viamplehate_phobert_vihsd.pt` | `best_viamplehate_phobert_vozhsd.pt` |
| Config JSON key `notebook` | `vihsd-proposed-viamplehate-phobert` | `voz-hsd-proposed-viamplehate-phobert` |
| Config JSON key `dataset` | `ViHSD (sonlam1102/vihsd)` | `VOZ-HSD (100k from tarudesu/VOZ-HSD)` |

---

## What Is NOT Changed

The following baseline components are preserved unchanged:
- `HeadAttention` architecture (W_q, W_k, W_v, batch-level softmax) — keep for reproducibility
- Direct injection formula: `final = CLS + e * sum(head_attentions)`
- CLS fallback when no targets found
- AdamW + differential LR (encoder vs. head)
- Best-threshold grid search on validation set
- AMP (mixed precision) + gradient clipping
- `TEENCODE_MAP` and `preprocess` pipeline
- Class weights for imbalance handling
- All plotting and evaluation code

---

## Key Invariants

1. When `VIET_TARGET_LEXICON` matches and NER finds no entities, the lexicon match provides the head token — the HeadAttention fires on a real target instead of CLS.
2. When NER runs on `raw_text` (unsegmented), entity surface forms align better with segmented token lookup via `ht.replace(' ', '_')`.
3. `self.last_embedding` is detached before the contrastive loss to prevent gradient flow from the auxiliary loss path back through the embedding twice.
4. The e-sweep (§19) should only be run after §12 training completes; it is a separate training run, not a continuation.
