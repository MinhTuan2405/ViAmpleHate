# ViAmpleHate Proposed Notebooks — Design Spec

**Date:** 2026-05-13  
**Status:** Approved  
**Reference:** `docs/improvementAmpleHate.md`

---

## Goal

Create two proposed notebooks by applying Vietnamese-specific improvements from
`improvementAmpleHate.md` to the baseline AmpleHate/PhoBERT notebooks:

1. `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/ViHSD - Proposed ViAmpleHate_PhoBERT.ipynb`
2. `notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/VOZ-HSD - Proposed ViAmpleHate_PhoBERT.ipynb`

---

## Scope Decisions (from brainstorming session)

| Decision | Choice | Reason |
|---|---|---|
| Improvement 9 (within-example HeadAttention) | **Skipped** | Deviates from original; doc recommends keeping original for baseline comparison |
| Improvement 10 (multi-class 3-label) | **Skipped** | Optional/advanced |
| Improvement 7 (max_length=256) | **Config flag, disabled by default** — `MAX_LEN=128` + profiling cell + comment | Conditional on truncation rate |
| Improvement 8 (PhoBERT-large) | **Config flag, disabled by default** — `USE_PHOBERT_LARGE=False` + comment | VRAM/BATCH trade-off |
| Improvement 6 (e-injection sweep) | **E_INJECTION=1.0 + commented-out sweep cell** | Avoids 5x training time; user can activate manually |
| Improvements 1–5 | **Fully implemented** (core improvements) | High impact, tightly coupled |

---

## Approach: Comprehensive (A)

Same 17-section structure as baseline notebooks. Improvements woven into existing cells
with brief inline notes. NER coverage statistics cell runs automatically after dataset build.
Clean top-to-bottom runnable notebook.

---

## Active Improvements (1–5)

### Improvement 1 — Vietnamese NER (Cell 5 + NERTagger)

- `NER_MODEL = 'NlpHUST/ner-vietnamese-electra-base'`
- `NERTagger.extract_named_entities` filters `{"PER", "ORG", "LOC", "MISC"}` (VLSP types)
- Expected coverage improvement: ~0.09% → 20–40% of samples

### Improvement 2 — Vietnamese Hate-Target Lexicon (new constant + NERProcessor)

- `VIET_TARGET_LEXICON` Python `set` added as module-level constant (full lexicon from doc)
- Covers: derogatory pronouns, gender, LGBTQ+, regional, ethnic, religious, political,
  occupation, social class, age, appearance categories, implicit patterns
- `NERProcessor.extract_head_tokens(text_raw, text_segmented)`:
  1. NER on `text_raw` (unsegmented) → entity surface forms
  2. Lexicon scan on lowercased `text_segmented` → matched terms
  3. Returns combined list

### Improvement 3 — Word Segmentation Alignment (NERProcessor.tokenize_and_encode)

- `tokenize_and_encode(text_segmented, text_raw)` takes both forms
- NER runs on `text_raw`; entity surface forms mapped to segmented positions
  via `ht.replace(' ', '_')` to align with underthesea compound-word output (Strategy A)
- Falls back to `[0]` (CLS) if no positions found

### Improvement 4 — Vietnamese Target Types

Addressed as part of Improvement 1 — VLSP type set `{PER, ORG, LOC, MISC}` replaces
CoNLL-2003 set `{ORG, NORP, GPE, LOC, EVENT}`.

### Improvement 5 — ContrastiveLossCosine (model forward + train_epoch)

- `LAMBDA_CL = 0.1` added to hyperparameters
- `AmpleHatePhoBERT.forward` stores `self.last_embedding = final_embedding.detach()`
  before the classifier call
- `criterion_cl = ContrastiveLossCosine(margin=0.5)` added to loss setup
- `train_epoch` computes: `loss = ce_loss + LAMBDA_CL * cl_loss`

---

## Dataset Changes Required

`AmpleHateDataset` stores both:
- `self.raw_texts = df['raw_text'].fillna('').tolist()` — for NER (unsegmented)
- `self.texts     = df['text_processed'].fillna('').tolist()` — for PhoBERT

Preprocessing cell adds:
```python
df['raw_text']       = df['free_text'].apply(normalize_text)  # unsegmented, for NER
df['text_processed'] = df['free_text'].apply(preprocess)      # segmented, for PhoBERT
```

---

## Optional/Disabled Config

| Improvement | Default | How to enable |
|---|---|---|
| Imp 6 (e-sweep) | `E_INJECTION=1.0`, commented sweep cell | Uncomment and run sweep cell |
| Imp 7 (max_length=256) | `MAX_LEN=128`, profiling cell included | Change `MAX_LEN=256` + `BATCH_SIZE=8` |
| Imp 8 (PhoBERT-large) | `USE_PHOBERT_LARGE=False`, comment in §2 | Set `True`: uses `phobert-large`, `HIDDEN_DIM=1024`, `BATCH_SIZE=8` |

---

## VOZ-HSD Adaptation

All model/NER/lexicon/training code is **identical** between notebooks.
Only these cells differ:

| Cell | ViHSD | VOZ-HSD |
|---|---|---|
| Title | ViHSD | VOZ-HSD |
| Data loading | `sonlam1102/vihsd`, 3-split HuggingFace | `tarudesu/VOZ-HSD`, `split="train"`, rename `texts→free_text`, `labels→label_id` |
| Sample size | Full dataset (~24k train) | 100k stratified sample (`SAMPLE_SIZE=100_000`) |
| Label mapping | 3-class → binary (2→1, 0/1→0) | Already binary (0/1), no remapping |
| Train/val/test | Pre-split from HuggingFace | Manual 80/10/10 split |
| `CKPT_NAME` | `best_viamplehate_phobert_vihsd.pt` | `best_viamplehate_phobert_vozhsd.pt` |
| `PLOT_TITLE` | `ViAmpleHate (PhoBERT) — ViHSD Proposed` | `ViAmpleHate (PhoBERT) — VOZ-HSD Proposed` |

---

## Output Files Per Notebook

- `best_viamplehate_phobert_[vihsd|vozhsd].pt`
- `training_curves_viamplehate_[vihsd|vozhsd].png`
- `confusion_matrix_viamplehate_[vihsd|vozhsd].png`
- `outputs/viamplehate_[vihsd|vozhsd]_config.json`

---

## What Does NOT Change

Per improvement doc "What NOT to Change":
- `HeadAttention` architecture (W_q, W_k, W_v, batch-level softmax)
- Direct injection: `final = CLS + e * sum(head_attentions)`
- CLS fallback when no targets found
- AdamW + differential LR (encoder vs. head)
- Best-threshold grid search on validation set
- AMP (mixed precision) + gradient clipping
- `TEENCODE_MAP` and `preprocess` pipeline
- Class weights for imbalance handling
- All plotting and evaluation code

---

## Key Invariants

1. When lexicon matches and NER finds no entities, the lexicon match provides the head
   token — HeadAttention fires on a real target instead of CLS.
2. NER runs on `raw_text` (unsegmented) so entity spans align with PhoBERT token lookup
   via `ht.replace(' ', '_')`.
3. `self.last_embedding` is detached before the contrastive loss to prevent double
   gradient flow through the embedding.
