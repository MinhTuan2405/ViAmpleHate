# AmpleHate-Vi++ Design Spec
**Date:** 2026-05-21  
**Dataset:** ViHSD (sonlam1102/vihsd)  
**Encoder:** vinai/phobert-base  
**Baseline reference:** notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/vihsd-baseline-amplehate-phobert.ipynb

---

## 1. Problem Statement

The baseline AmpleHate notebook ports the original English AmpleHate (Lee et al., EMNLP 2025) to Vietnamese ViHSD using PhoBERT. The core bottleneck is the English NER model (`dbmdz/bert-large-cased-finetuned-conll03-english`) which achieves only **0.09% NER coverage** on Vietnamese text, causing 99.91% of samples to fall back to the CLS anchor. This effectively degrades AmpleHate to a standard PhoBERT classifier for nearly all inputs.

**Baseline results:**
- Test Macro F1: 0.7792
- HATE F1: 0.6045
- Accuracy: 0.9175
- NER coverage: 0.09%

---

## 2. Goals

- Increase NER/cue coverage from 0.09% to ~40–50%
- Improve HATE class F1 (currently 0.6045) as primary metric
- Improve Macro F1 as secondary metric
- Stay faithful to the AmpleHate injection principle; do not replace with a fundamentally different architecture

---

## 3. Architecture: AmpleHate-Vi++

### 3.1 Changes from Baseline

| Component | Baseline | Proposed |
|---|---|---|
| Target extraction | English NER only | Vietnamese NER + Target cue lexicon |
| Attack signal | None | Attack cue lexicon (separate bank) |
| Relation vectors | 1 (HeadAttention) | 3 (r_exp, r_imp, r_atk) |
| Injection | Fixed scalar `e=1.0` | Instance-adaptive gate `g = σ(Linear([h_CLS; r]))` |
| Contrastive loss | Disabled | Enabled, α = 0.1 |
| max_length | 128 | 256 |
| NUM_EPOCHS | 6 | 8 (with early stopping, patience=2) |

### 3.2 P2 — Multi-signal Target Cue Mining

Three sources combined to identify target positions:

**Source 1 — Vietnamese NER**  
Model: `NlpHUST/ner-vietnamese-electra-base`  
Entity types kept: PER, ORG, LOC, GPE, NORP (mapped to Vietnamese equivalents)  
Run on word-segmented text (same as PhoBERT input) to ensure token alignment.

**Source 2 — Target Cue Lexicon**  
Derogatory pronouns and nominal group patterns that reference target groups in Vietnamese social media hate speech. These are pure reference terms, NOT offensive predicates.

```python
TARGET_CUES = [
    # Derogatory pronouns / group references
    'bọn', 'thằng', 'con', 'đứa', 'tụi', 'đám', 'lũ',
    'mấy đứa', 'mấy thằng', 'mấy con', 'loại người', 'loại',
    # Nominal group patterns (prefix → any following noun is target)
    'người', 'dân', 'bên',
]
```

**Source 3 — [CLS] implicit anchor**  
Always present as fallback (index 0), preserving original AmpleHate behavior.

**Coverage estimate:** NER (~5%) + target cues (~40–50%) → combined coverage ~45–55%.

### 3.3 Attack Cue Bank (new, separate from target)

Attack cue tokens signal offensive predicate or evaluation toward the target. These are distinct from target cues (per artifact recommendation: hate lexicon should not be mixed into target identification).

```python
ATTACK_CUES = [
    'ngu', 'đần', 'ngu ngốc', 'khùng', 'điên', 'hèn', 'nhục',
    'ăn bám', 'ký sinh', 'phản quốc', 'vô học', 'man rợ',
    'cút', 'xéo', 'câm', 'im mồm',
    'giết', 'chém', 'đánh',
]
```

Attack cue tokens are identified in the tokenized sequence and used for `r_atk`.

### 3.4 P4 — Relation Bank

Three HeadAttention modules, each sharing the same architecture as baseline:

```
r_exp = HeadAttn(CLS, target_cue_token_embeddings)   # explicit targets
r_imp = HeadAttn(CLS, CLS_embedding)                  # implicit anchor
r_atk = HeadAttn(CLS, attack_cue_token_embeddings)   # attack signal

r_fused = Linear(768*3 → 768)(cat(r_exp, r_imp, r_atk))
```

When no target/attack cues are found, the respective relation defaults to HeadAttn(CLS, CLS).

### 3.5 P5 — Instance-adaptive Gate (replacing fixed `e`)

```
g = σ(W_g · cat(h_CLS, r_fused) + b_g)   # scalar ∈ (0, 1)
z = h_CLS + g * r_fused
logits = Linear(Dropout(z))
```

The gate `g` is a learned scalar per instance, conditioned on both the sentence representation and the fused relation vector. This removes the need to grid-search `e` and adapts injection strength to each input.

**Ablation:** For comparison, keep a fixed-e variant (`z = h_CLS + e * r_fused`, grid-searched `e ∈ {0.5, 0.75, 1.0, 1.25, 1.5}`) in the final evaluation section.

### 3.6 Loss Function

```
L = L_CE + α * L_CL

L_CE  = CrossEntropyLoss(weight=[0.559, 4.704], label_smoothing=0.05)
L_CL  = ContrastiveLossCosine(margin=0.5)  # same as original AmpleHate
α     = 0.1  (tunable)
```

The ContrastiveLoss uses the `z` (post-injection) embedding, pulling same-class embeddings together and pushing different-class embeddings apart.

---

## 4. Dataset & Preprocessing

Unchanged from baseline:
- Dataset: `sonlam1102/vihsd` (Train: 24,048 | Val: 2,672 | Test: 6,680)
- Binary mapping: NON-HATE=0 (CLEAN+OFFENSIVE), HATE=1
- Preprocessing: teencode normalization + underthesea word tokenization

Addition:
- Emoji semantic tagging: map common emojis to semantic tokens before tokenization (🙃→[MOCK], 😡→[ANGER], etc.)

---

## 5. Hyperparameters

| Param | Baseline | Proposed |
|---|---|---|
| MODEL_NAME | vinai/phobert-base | vinai/phobert-base |
| NER_MODEL | dbmdz/bert-large-cased-finetuned-conll03-english | NlpHUST/ner-vietnamese-electra-base |
| MAX_LEN | 128 | 256 |
| BATCH_SIZE | 16 | 16 |
| NUM_EPOCHS | 6 | 8 |
| LR | 2e-5 | 2e-5 |
| HEAD_LR | 5e-5 | 5e-5 |
| WARMUP_RATIO | 0.06 | 0.06 |
| DROPOUT | 0.1 | 0.1 |
| PATIENCE | 2 | 2 |
| WEIGHT_DECAY | 0.01 | 0.01 |
| LABEL_SMOOTHING | 0.05 | 0.05 |
| E_INJECTION | 1.0 (fixed) | adaptive gate (ablation: grid {0.5,0.75,1.0,1.25,1.5}) |
| ALPHA_CL | 0 (disabled) | 0.1 |

---

## 6. Model Class Structure

```
ViAmpleHatePhoBERT
├── bert: AutoModel (vinai/phobert-base)
├── head_attn_exp: HeadAttention(768, 768)   # explicit target
├── head_attn_imp: HeadAttention(768, 768)   # implicit CLS
├── head_attn_atk: HeadAttention(768, 768)   # attack cue
├── relation_proj: Linear(768*3, 768)         # fuse 3 relations
├── gate_proj: Linear(768*2, 1)               # instance-adaptive gate
├── dropout: Dropout(0.1)
└── classifier: Linear(768, 2)

ContrastiveLossCosine (margin=0.5)            # unchanged from baseline
```

---

## 7. Training Loop Changes

1. Loss combines CE + α*CL, using the `z` embedding for CL
2. NER applied only during training (eval uses cue lexicon only, no NER model load on inference)
3. DataLoader: `num_workers=0` for NER pipeline safety (same as baseline)
4. Gradient accumulation = 2 (effective batch = 32)

---

## 8. Evaluation

Same evaluation protocol as baseline:
- Best threshold search on val set (grid 0.05–0.95)
- Report: Accuracy, Macro P/R/F1, HATE F1
- Confusion matrix
- Ablation section: fixed-e vs adaptive gate comparison

---

## 9. Output Notebook

**Destination:** `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb`

The notebook follows the same section structure as baseline (sections 1–17) with additional sections:
- Section 8b: Attack cue bank
- Section 10b: Relation bank + adaptive gate (replacing original section 10)
- Section 17b: Ablation — fixed-e vs adaptive gate

---

## 10. Expected Outcomes

| Metric | Baseline | Target |
|---|---|---|
| Macro F1 | 0.7792 | > 0.80 |
| HATE F1 | 0.6045 | > 0.65 |
| Accuracy | 0.9175 | ≥ 0.92 |
| NER+cue coverage | 0.09% | ~45–55% |
