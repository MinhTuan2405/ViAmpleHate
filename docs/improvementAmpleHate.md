# AmpleHate for Vietnamese: Improvement Notes

> **Reference notebook:** `vihsd-amplehate-phobert-baseline.ipynb`  
> **Dataset:** ViHSD (`sonlam1102/vihsd`, binary: NON-HATE=0, HATE=1)  
> **Status:** The baseline notebook is a faithful, unchanged port of the original English AmpleHate. This document describes everything that needs to change to make it actually work well for Vietnamese.

---

## Quick Overview: What Breaks and Why

AmpleHate's core insight is to amplify **target-context relations** by finding the explicit hate target (via NER) and injecting cross-attention between `[CLS]` and the target token back into `[CLS]`. The whole mechanism depends on finding the right target.

For Vietnamese, the NER step produces near-zero hits, so the HeadAttention is almost always applied to `[CLS]` itself (the CLS fallback). This makes the baseline functionally close to a plain PhoBERT classifier — the AmpleHate mechanism never fires properly.

**Root cause summary:**

| What breaks | Why |
|---|---|
| English NER on Vietnamese text | ~0% entity recall |
| CoNLL-2003 entity types (ORG, NORP, GPE…) | Miss Vietnamese hate targets (gender terms, regional slurs, occupation nouns) |
| NER runs on raw text, PhoBERT gets word-segmented text | Entity token positions don't align after underthesea segmentation |
| Batch-level cross-attention in HeadAttention | Mixes signal across examples in the batch (non-standard, but from original) |
| ContrastiveLoss unused | Potential extra signal left on the table |

---

## Improvement 1: Replace the NER Model with a Vietnamese NER

**File to change:** Cell 17 (`NERTagger`) and Cell 5 (`NER_MODEL` hyperparameter)

### Problem

`dbmdz/bert-large-cased-finetuned-conll03-english` was trained on English CoNLL-2003.
It has essentially zero recall on Vietnamese. For all practical purposes, every ViHSD sample
gets `head_token_idx = [0]` (CLS fallback), and the HeadAttention degenerates into a
learned projection of `[CLS]` against itself — losing the whole point of AmpleHate.

### What to do

Replace `NER_MODEL` with a Vietnamese NER model. Options in order of recommendation:

1. **`NlpHUST/ner-vietnamese-electra-base`** — ELECTRA fine-tuned on VLSP NER, standard Vietnamese NER entity types (PER, ORG, LOC, MISC). Best general-purpose choice.
2. **`vinai/PhoNER_COVID19`** — PhoBERT fine-tuned on COVID-19 Vietnamese NER. Good if the domain matches, but COVID-19-specific entity types (LOCATION, DATE, NAME) may not generalise.
3. **`NlpHUST/vi-word-ner`** — If running word-segmented input (see Improvement 3).

### Code change (Cell 5)

```python
# Change NER_MODEL from English to Vietnamese:
NER_MODEL = 'NlpHUST/ner-vietnamese-electra-base'
```

### Code change (Cell 17 — NERTagger.extract_named_entities)

The VLSP NER model uses different entity group names. Update the filter:

```python
def extract_named_entities(self, text):
    entities = self.ner_pipeline(text)
    # VLSP Vietnamese NER types that correspond to hate targets:
    # PER = person/group (e.g., "người Hà Nội", "phụ nữ")
    # ORG = organization
    # LOC = location (e.g., "miền Nam", "Việt Nam")
    # MISC = miscellaneous groups (religion, ethnicity, nationality)
    target_types = {"PER", "ORG", "LOC", "MISC"}
    return [
        e["word"] for e in entities
        if e["entity_group"] in target_types
    ]
```

### Expected impact

Bumps NER coverage from ~1–5% to an estimated 20–40% of ViHSD samples. The HeadAttention
will fire on real extracted targets instead of always falling back to CLS.

---

## Improvement 2: Add a Vietnamese Hate-Target Lexicon

**File to change:** Cell 17 (`NERProcessor.extract_head_tokens`) and Cell 21 (`AmpleHateDataset`)

### Problem

ViHSD hate speech frequently targets groups expressed as **common nouns and pronouns**, not named entities.
Examples: "thằng" (guy/derogatory), "bọn" (bunch/derogatory), "tụi nó" (those people), "đứa" (kid/derogatory).
No NER model will extract these — they are not entities. But they are the primary target markers in Vietnamese hate speech.

Additionally, regional group terms ("người Bắc", "dân Nam kỳ"), gender terms ("đàn ông", "đàn bà", "LGBT"), and
occupation/religion terms ("bộ đội", "công an", "Phật tử") are critical targets in ViHSD that NER may miss.

### What to do

Build a **Vietnamese hate-target lexicon** and combine it with NER results:

```python
# Add to Cell 5 or a dedicated "Lexicon" cell:
VIET_TARGET_LEXICON = {

    # ── Derogatory pronouns / group markers ──────────────────────
    "thằng", "bọn", "tụi", "đứa", "mấy đứa",
    "lũ", "đám", "cái loại", "hạng", "loại người",
    "chúng mày", "chúng nó", "bọn chúng", "mấy thằng",
    "cái thứ", "đồ", "quân",

    # ── Gender ────────────────────────────────────────────────────
    "đàn ông", "đàn bà", "phụ nữ", "con gái", "con trai",
    "người phụ nữ", "người đàn ông", "giới nữ", "giới nam",
    "đàn bà con gái", "đàn ông con trai",
    # misogyny-specific
    "phụ nữ lái xe", "đàn bà mồm",

    # ── Sexual orientation / gender identity ──────────────────────
    "lgbt", "lgbtq", "đồng tính", "đồng tính luyến ái",
    "gay", "les", "lesbian", "bisexual", "bi",
    "chuyển giới", "transgender", "phi nhị giới",
    "bê đê", "pê đê", "bóng", "bóng lộ",
    "ái nam ái nữ", "lưỡng tính",

    # ── Regional / geographic ─────────────────────────────────────
    "người bắc", "dân bắc", "người miền bắc", "bắc kỳ",
    "người nam", "người miền nam", "dân nam kỳ", "nam kỳ",
    "người miền trung", "dân miền trung", "trung kỳ",
    "người hà nội", "người sài gòn", "người hồ chí minh",
    "dân tỉnh lẻ", "người quê", "dân quê", "nhà quê",
    "dân ngoại tỉnh", "người ngoại tỉnh",
    "dân thành thị", "dân nông thôn",

    # ── Ethnicity / nationality ───────────────────────────────────
    "người kinh", "người thượng", "người dân tộc",
    "dân tộc thiểu số", "người thiểu số",
    "người tàu", "người trung quốc", "người hoa", "hoa kiều",
    "người chăm", "người khmer", "người mường",
    "người tày", "người nùng", "người hmong", "người mông",
    "người việt", "việt nam",
    "người nước ngoài", "tây", "tây ba lô",
    "việt kiều", "người việt hải ngoại", "người mỹ gốc việt",
    "người hàn", "người nhật", "người thái",
    # derogatory
    "chệt", "chệt hoa",

    # ── Religion / creed ─────────────────────────────────────────
    "hồi giáo", "đạo hồi", "muslim", "người hồi giáo",
    "thiên chúa giáo", "công giáo", "đạo thiên chúa",
    "tin lành", "đạo tin lành",
    "phật giáo", "đạo phật", "người theo phật",
    "cao đài", "hòa hảo",
    "người theo đạo", "con chiên", "tín đồ",
    "vô thần", "người vô thần",

    # ── Politics / ideology ───────────────────────────────────────
    "đảng viên", "đảng cộng sản", "cộng sản",
    "chế độ", "nhà nước", "chính quyền",
    "phản động", "việt cộng", "thế lực thù địch",
    "dân chủ", "đối lập", "nhân quyền",
    "thân cộng", "chống cộng",
    "tư bản", "xã hội chủ nghĩa",

    # ── Occupation ────────────────────────────────────────────────
    # law enforcement
    "công an", "cảnh sát", "cảnh sát giao thông",
    "bộ đội", "quân đội", "chiến sĩ",
    # public sector
    "cán bộ", "quan chức", "lãnh đạo", "chính trị gia",
    "đại biểu", "nghị sĩ",
    # media / education
    "nhà báo", "phóng viên", "báo chí",
    "giáo viên", "thầy giáo", "cô giáo", "giảng viên",
    # healthcare
    "bác sĩ", "y tá", "y bác sĩ", "nhân viên y tế",
    # legal
    "luật sư", "thẩm phán",
    # content creators (targeted on YouTube/Facebook)
    "youtuber", "tiktoker", "streamer", "influencer",
    "kol", "idol",

    # ── Social class / economic status ───────────────────────────
    "người nghèo", "dân nghèo", "hộ nghèo",
    "người giàu", "nhà giàu", "trọc phú", "đại gia",
    "tầng lớp trung lưu", "dân lao động",
    "công nhân", "nông dân", "người lao động",
    "ăn mày", "vô gia cư", "người vô gia cư",

    # ── Age ───────────────────────────────────────────────────────
    "người già", "ông già", "bà già", "lão",
    "cụ già", "người cao tuổi",
    "giới trẻ", "thanh niên", "lũ trẻ", "bọn nhóc",
    "thế hệ z", "gen z", "gen y", "millennials",
    "trẻ trâu",

    # ── Appearance / body / disability ───────────────────────────
    "người béo", "người mập", "đồ béo",
    "người gầy", "que củi",
    "người lùn", "người cao",
    "người xấu", "người đẹp",
    "người khuyết tật", "người tàn tật",
    "người điếc", "người mù", "người câm",
    "người tâm thần", "người điên",

    # ── Mental health (increasingly targeted on social media) ─────
    "người trầm cảm", "người lo âu", "bệnh tâm lý",
    "bệnh tâm thần",

    # ── Immigration / social status ───────────────────────────────
    "người nhập cư", "dân nhập cư", "người di cư",
    "người tị nạn",

    # ── Implicit / indirect reference patterns ────────────────────
    # Patterns that precede group references in hate speech
    "tất cả", "toàn bộ", "hết thảy",       # "tất cả bọn..."
    "đặc trưng", "bản chất", "nòi",          # "bản chất của..."
    "giống nòi", "dòng giống",
}

class NERProcessor:
    def extract_head_tokens(self, text):
        tokens_found = []
        # 1. NER-based (entities)
        if self.use_ner:
            tokens_found.extend(self.ner_tagger.extract_named_entities(text))
        # 2. Lexicon-based (common hate target terms)
        text_lower = text.lower()
        for term in VIET_TARGET_LEXICON:
            if term in text_lower:
                tokens_found.append(term)
        return tokens_found
```

Note: the lexicon lookup happens on the preprocessed (word-segmented) text, so terms
need to match the underthesea output format (e.g., "đàn_ông" after segmentation).
Adjust the lexicon to use underscore-joined forms if running on segmented text.

### Expected impact

Lexicon + NER together should cover 40–60% of ViHSD samples with a real target token.

---

## Improvement 3: Fix Word Segmentation Mismatch

**File to change:** Cell 17 (`NERProcessor.tokenize_and_encode`) and Cell 9 (`preprocess`)

### Problem

The preprocessing pipeline applies `underthesea.word_tokenize` (adding underscores to compound words, e.g., "Việt Nam" → "Việt_Nam") before feeding text to PhoBERT.
PhoBERT expects this format and its vocabulary is trained on segmented text.

However, the English NER model (and even most Vietnamese NER models by default) expect **unsegmented** text.
Running NER on unsegmented text means entity spans (e.g., "Việt Nam") won't align with the segmented PhoBERT tokens ("Việt_Nam" → single token `▁Việt_Nam`).

The `tokens.index(ht)` lookup in `tokenize_and_encode` uses `tokenizer.tokenize(text)` where `text` is already segmented, but `ht` comes from NER on unsegmented text. The surface forms don't match, so `ValueError` is silently caught and the entity is dropped.

### What to do

**Strategy A (recommended):** Run NER on unsegmented text, then map entity positions to the segmented tokenized form.

```python
def tokenize_and_encode(self, text_segmented):
    # text_segmented: underthesea output ("Việt_Nam đẹp")
    # text_raw: original before segmentation (stored in dataset)
    head_tokens = self.extract_head_tokens(self.raw_text)  # NER on raw text

    encoding = self.tokenizer(
        text_segmented,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_offsets_mapping=False,
    )
    # Map entity surface forms to segmented token positions
    seg_tokens = self.tokenizer.tokenize(text_segmented)
    head_token_idx = []
    for ht in head_tokens:
        ht_seg = ht.replace(' ', '_')  # align to segmented form
        for i, tok in enumerate(seg_tokens):
            if ht_seg in tok or tok.replace('▁', '') == ht_seg:
                idx = i + 1  # +1 for [CLS]
                if idx < MAX_LEN - 1:
                    head_token_idx.append(idx)
                    break
    if not head_token_idx:
        head_token_idx = [0]
    return encoding["input_ids"], head_token_idx, encoding["attention_mask"]
```

**Strategy B (simpler but approximate):** Use a Vietnamese NER model that accepts PhoBERT tokens directly (PhoBERT-based NER), so entity spans are already in the segmented tokenization space.

### Required dataset change

Store both raw and segmented text in the dataset. Change `AmpleHateDataset.__init__`:

```python
self.raw_texts  = df['free_text'].fillna('').tolist()      # for NER
self.texts      = df['text_processed'].fillna('').tolist() # for PhoBERT
```

Pass `raw_text` to `NERProcessor` separately.

---

## Improvement 4: Redesign Target Types for Vietnamese Implicit Hate

**File to change:** Cell 17 (`NERTagger.extract_named_entities`), the lexicon (Improvement 2)

### Problem

The original AmpleHate entity types (ORG, NORP, GPE, LOC, EVENT) are designed for English implicit hate speech, where targets are typically racial groups (tagged as NORP), countries (GPE), or organizations (ORG).

ViHSD hate speech has a different distribution of hate targets. Based on ViHSD analysis:

| Target type | Examples in ViHSD | CoNLL entity type |
|---|---|---|
| Ethnic/racial groups | "người Bắc", "người Kinh", "dân tộc thiểu số" | LOC or miss |
| Gender/LGBTQ+ | "đàn bà", "LGBT", "đồng tính" | Not an entity |
| Regional groups | "dân Nam kỳ", "người Sài Gòn" | LOC |
| Occupation groups | "công an", "bộ đội", "nhà báo" | Not an entity |
| Derogatory pronouns | "thằng", "bọn", "tụi nó" | Not an entity |
| Political groups | "cộng sản", "đảng viên" | ORG/MISC |

The most common ViHSD hate targets fall outside traditional NER categories entirely.
This means even a perfect Vietnamese NER won't capture the full signal — the lexicon
approach (Improvement 2) is essential to complement NER for ViHSD specifically.

---

## Improvement 5: Enable ContrastiveLossCosine

**File to change:** Cell 27 (`ContrastiveLossCosine`) and Cell 29 (optimizer/loss setup)

### Problem

The baseline uses only CrossEntropyLoss. The original AmpleHate paper includes a
`ContrastiveLossCosine` component that encourages embeddings of same-class samples
to be similar and different-class embeddings to be dissimilar. The class in Cell 27
is present but the loss is never added to the training objective.

### What to do

Add the contrastive loss as an auxiliary training signal.
The combined loss used in the original paper is `CE + λ * CL`.

```python
# Cell 29 changes:
lambda_cl = 0.1   # weight for contrastive loss; tune [0.05, 0.1, 0.2]
criterion_cl = ContrastiveLossCosine(margin=0.5)

# Cell 32 — inside train_epoch, replace loss computation:
with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
    logits = model(ids, heads, mask)
    # Get the final_embedding for contrastive loss (need model change — see below)
    ce_loss = criterion_train(logits, y)
    cl_loss = criterion_cl(model.last_embedding, y)
    loss = ce_loss + lambda_cl * cl_loss
```

This requires a small model change: store `final_embedding` as `self.last_embedding`
before the classifier in `AmpleHatePhoBERT.forward`:

```python
# In AmpleHatePhoBERT.forward, before return:
self.last_embedding = final_embedding.detach()   # [batch, hidden]
return self.classifier(self.dropout(final_embedding))
```

Note: `ContrastiveLossCosine` requires `batch_size >= 2` (the denominator is `batch*(batch-1)`).
This is always satisfied at `BATCH_SIZE=16`.

---

## Improvement 6: Tune the Injection Strength `e`

**File to change:** Cell 5 (`E_INJECTION`) and Cell 34 (training loop or a sweep cell)

### Problem

The original AmpleHate paper tunes `e ∈ {0.5, 0.75, 1.0, 1.25, 1.5}`.
The baseline hardcodes `E_INJECTION = 1.0` (the midpoint). With better NER
coverage (Improvements 1–3), the optimal `e` for ViHSD may differ.

### What to do

After Improvements 1–3 are in place, run a sweep over `e` values using the validation F1:

```python
for e_val in [0.5, 0.75, 1.0, 1.25, 1.5]:
    model = AmpleHatePhoBERT(MODEL_NAME, hidden_dim=HIDDEN_DIM, e=e_val, dropout=DROPOUT).to(DEVICE)
    # ... train and evaluate ...
```

If NER coverage is still low (~10–20%), a smaller `e` (0.5) reduces the impact of
noisy CLS-on-CLS attention. If coverage improves to >30%, larger `e` values may be better.

---

## Improvement 7: Increase max_length

**File to change:** Cell 5 (`MAX_LEN`)

### Current setting

`MAX_LEN = 128` to fit within Kaggle T4 VRAM at `BATCH_SIZE=16`.

### Analysis

Most ViHSD comments are short (median PhoBERT token count is typically ~40–60 tokens),
so 128 is sufficient for the majority of samples. However, longer comments (argumentative posts,
hate manifestos) that exceed 128 tokens will be truncated, potentially losing the hate target.

### What to do

Profile the truncation rate:

```python
tokenized_lengths = [
    len(tokenizer.tokenize(t)) for t in train_df['text_processed']
]
print(pd.Series(tokenized_lengths).describe())
print(f"Truncated at 128: {sum(l > 126 for l in tokenized_lengths)} / {len(tokenized_lengths)}")
```

If >5% of samples are truncated, increase to `MAX_LEN = 256` (requires halving
`BATCH_SIZE` to 8 or enabling gradient checkpointing to keep VRAM usage the same).

---

## Improvement 8: Use PhoBERT-large as the Encoder

**File to change:** Cell 5 (`MODEL_NAME`)

### Current setting

`MODEL_NAME = 'vinai/phobert-base'` — 12-layer, 768-hidden, ~135M params.

### Option

`vinai/phobert-large` — 24-layer, 1024-hidden, ~370M params.

### Trade-off

PhoBERT-large typically yields +1–3 F1 points on Vietnamese NLP benchmarks.
However, it requires `HIDDEN_DIM = 1024` and `HEAD_DIM = 1024` changes, and
roughly doubles VRAM usage (requires `BATCH_SIZE=8` or gradient checkpointing on T4).

```python
# To use PhoBERT-large:
MODEL_NAME = 'vinai/phobert-large'
HIDDEN_DIM = 1024
HEAD_DIM   = 1024
BATCH_SIZE = 8  # halved for VRAM
```

The HeadAttention and classifier will automatically resize because they use `HIDDEN_DIM`.

---

## Improvement 9: Reconsider the HeadAttention Cross-Batch Design

**File to change:** Cell 25 (`HeadAttention.forward`)

### Problem

The original `HeadAttention` computes `scores = Q_h @ K_h.T`, which is a `[B, B]` matrix —
every sample in the batch attends to **every other sample's** target token. This means:

1. Attention is across examples, not across sequence positions within an example.
2. Results are batch-size dependent (batch=1 → score=[[1.0]] → no attention effect).
3. During inference with a single example, the mechanism is a no-op.

This is the original paper's design and may be intentional (batch-level contrastive target attention),
but it is non-standard and worth re-evaluating for Vietnamese.

### What to do (optional, deviates from original)

A more standard within-example attention would be:

```python
def forward(self, cls_embedding, head_token_embedding):
    Q_h = self.W_q(cls_embedding)           # [B, D]
    K_h = self.W_k(head_token_embedding)    # [B, D]
    V_h = self.W_v(head_token_embedding)    # [B, D]  ← V from target, not CLS
    score = (Q_h * K_h).sum(-1, keepdim=True) / (self.head_dim ** 0.5)  # [B, 1]
    weight = torch.sigmoid(score)
    return weight * V_h   # [B, D]
```

This computes attention **within** each example (CLS queries its own target token),
which is more interpretable and stable across batch sizes. But this changes the original
AmpleHate mechanism — keep the original for the baseline comparison.

**Recommendation:** Keep original batch-level attention for the baseline, then experiment
with within-example attention as a variant and compare F1.

---

## Improvement 10: Multi-Label Setup (Optional, Advanced)

### Problem

ViHSD originally has 3 classes: CLEAN=0, OFFENSIVE=1, HATE=2. The baseline collapses
CLEAN and OFFENSIVE into NON-HATE, losing signal. OFFENSIVE comments often share surface
features with HATE but differ in intent — this distinction might help train better boundaries.

### What to do (multi-class, not binary)

```python
# Cell 7 — remove binary mapping, keep 3 classes:
# train_df['label_id'] stays as 0/1/2 (no remapping)
NUM_CLASSES = 3
LABEL_NAMES = ['CLEAN', 'OFFENSIVE', 'HATE']
```

And update all metric computations to use `average='macro'` over 3 classes.
This is a meaningful experiment: does OFFENSIVE vs HATE distinction help the model
learn better HATE-specific features?

---

## Priority Order for Implementation

Based on expected impact vs. effort:

| Priority | Improvement | Expected F1 gain | Effort |
|---|---|---|---|
| 1 | Vietnamese NER (Improvement 1) | +3–6 pts | Low (model swap) |
| 2 | Vietnamese target lexicon (Improvement 2) | +2–4 pts | Medium |
| 3 | Word segmentation alignment (Improvement 3) | +1–3 pts | Medium |
| 4 | Enable ContrastiveLoss (Improvement 5) | +1–2 pts | Low |
| 5 | Tune e injection (Improvement 6) | +0–2 pts | Low (sweep) |
| 6 | max_length increase (Improvement 7) | +0–1 pts | Low |
| 7 | PhoBERT-large (Improvement 8) | +1–3 pts | Low (config) |
| 8 | Redesign target types (Improvement 4) | Support for #1–3 | Medium |
| 9 | Within-example HeadAttention (Improvement 9) | Unknown | Medium |
| 10 | Multi-class setup (Improvement 10) | Unknown | High |

Start with Improvements 1–3 together, since they are tightly coupled (NER → lexicon → alignment).
Then add ContrastiveLoss and e-sweep on top of the fixed target extraction.

---

## What NOT to Change (Keep from Original AmpleHate)

- **HeadAttention architecture** (W_q, W_k, W_v projections, batch-level attention) — keep for reproducibility
- **Direct injection**: `final = CLS + e * sum(head_attentions)` — the core AmpleHate contribution
- **CLS fallback** when no targets found — correct and important
- **AdamW + differential LR** (encoder vs. head) — works well
- **Best-threshold grid search** on validation set — correct for imbalanced data
- **underthesea + PhoBERT tokenizer** pipeline — correct for PhoBERT
- **TEENCODE_MAP normalization** — domain-appropriate for ViHSD social media text
