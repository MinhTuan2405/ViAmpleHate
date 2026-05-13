# ViAmpleHate Proposed Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create two runnable Jupyter notebooks — ViHSD and VOZ-HSD proposed — that apply Improvements 1–5 from `improvementAmpleHate.md` (Vietnamese NER, target lexicon, segmentation alignment, ContrastiveLoss) on top of the existing AmpleHate/PhoBERT baselines, with Improvements 6–8 as disabled-by-default config flags.

**Architecture:** Both notebooks share identical model/NER/lexicon/training code and differ only in dataset loading and checkpoint names. Each notebook is a self-contained `.ipynb` JSON file assembled by reading the corresponding baseline, applying cell-level changes, and writing the result. No baseline notebooks are modified.

**Tech Stack:** PyTorch, PhoBERT (`vinai/phobert-base`), HuggingFace `transformers` (pipeline, AutoModel, AutoTokenizer), `underthesea`, `datasets`, `scikit-learn`, Jupyter nbformat 4.

---

## File Structure

**New files (create):**
- `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/ViHSD - Proposed ViAmpleHate_PhoBERT.ipynb`
- `notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/VOZ-HSD - Proposed ViAmpleHate_PhoBERT.ipynb`

**Reference (read-only, do NOT modify):**
- `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/vihsd-baseline-amplehate-phobert.ipynb`
- `notebooks/models/baselines/VOZ-HSD - Baseline AmpleHate_PhoBERT/voz-hsd-baseline-amplehate-phobert.ipynb`

---

## Task 1: Create output directories

**Files:**
- Create directory: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/`
- Create directory: `notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/`

- [ ] **Step 1: Create both output directories**

```bash
mkdir -p "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT"
mkdir -p "notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT"
```

Expected: Both directories created without error.

- [ ] **Step 2: Verify**

```bash
ls "notebooks/models/proposed/"
```

Expected output contains both folder names.

---

## Task 2: Write ViHSD proposed notebook

**Files:**
- Read: `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/vihsd-baseline-amplehate-phobert.ipynb`
- Create: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/ViHSD - Proposed ViAmpleHate_PhoBERT.ipynb`

The proposed notebook is the baseline with specific cells replaced or added. Below is the complete cell-by-cell specification. Cells not listed are **identical to the baseline** and must be copied verbatim (including cell IDs).

- [ ] **Step 1: Read the baseline notebook**

Read `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/vihsd-baseline-amplehate-phobert.ipynb` to get the full cell list and cell IDs.

- [ ] **Step 2: Replace the title markdown cell**

Find the first markdown cell (cell id `a1b2c3d4-e5f6-7890-abcd-ef1234567890`). Replace its source with:

```markdown
# ViAmpleHate on ViHSD: PhoBERT Proposed

This notebook implements **ViAmpleHate** — an improved version of AmpleHate (Lee et al., EMNLP 2025)
adapted for Vietnamese hate speech on the **ViHSD** dataset using **PhoBERT** as the encoder.

Improvements over the baseline (see `docs/improvementAmpleHate.md`):
1. **Vietnamese NER** (`NlpHUST/ner-vietnamese-electra-base`) replaces English CoNLL-2003 NER
2. **Vietnamese hate-target lexicon** complements NER for non-entity targets
3. **Word segmentation alignment**: NER runs on raw text, mapped to segmented token positions
4. **ContrastiveLossCosine** enabled as auxiliary training signal (CE + λ·CL)

Not applied (see optional cells):
- Improvement 6 (e-sweep): commented cell after training
- Improvement 7 (max_length): truncation profiling cell included; MAX_LEN=128 by default
- Improvement 8 (PhoBERT-large): config flag USE_PHOBERT_LARGE=False
```

- [ ] **Step 3: Replace the hyperparameters cell**

Find cell id `e5f6a7b8-c9d0-1234-efab-234567890004`. Replace its source with:

```python
# Encoder
MODEL_NAME   = 'vinai/phobert-base'
NER_MODEL    = 'NlpHUST/ner-vietnamese-electra-base'  # Improvement 1: Vietnamese NER
MAX_LEN      = 128   # see truncation profiling cell below; increase to 256 if >5% truncated
HIDDEN_DIM   = 768
HEAD_DIM     = HIDDEN_DIM

# Improvement 8: PhoBERT-large option (disabled by default — doubles VRAM, needs BATCH_SIZE=8)
USE_PHOBERT_LARGE = False
if USE_PHOBERT_LARGE:
    MODEL_NAME = 'vinai/phobert-large'
    HIDDEN_DIM = 1024
    HEAD_DIM   = 1024

# AmpleHate injection strength (e in original paper)
E_INJECTION  = 1.0   # Improvement 6: tune [0.5, 0.75, 1.0, 1.25, 1.5] — see commented sweep cell

# Improvement 5: ContrastiveLoss weights
LAMBDA_CL          = 0.1    # auxiliary contrastive loss weight; tune [0.05, 0.1, 0.2]
CONTRASTIVE_MARGIN = 0.5

# Training
BATCH_SIZE   = 16    # reduce to 8 if USE_PHOBERT_LARGE=True
NUM_EPOCHS   = 6
LR           = 2e-5
HEAD_LR      = 5e-5
WARMUP_RATIO = 0.06
DROPOUT      = 0.1
PATIENCE     = 2
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.05

# Labels
NUM_CLASSES  = 2
LABEL_NAMES  = ['NON-HATE', 'HATE']

# Checkpointing
CKPT_NAME    = 'best_viamplehate_phobert_vihsd.pt'
PLOT_TITLE   = 'ViAmpleHate (PhoBERT) — ViHSD Proposed'
```

- [ ] **Step 4: Insert VIET_TARGET_LEXICON cell after the hyperparameters cell**

Insert a new code cell (new id: `lexicon-viet-target-001`) immediately after the hyperparameters cell. Source:

```python
# Improvement 2: Vietnamese hate-target lexicon
# Covers targets that NER misses: derogatory pronouns, gender, LGBTQ+, regional,
# ethnic, religious, political, occupation, social class, age, appearance groups.
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
    "công an", "cảnh sát", "cảnh sát giao thông",
    "bộ đội", "quân đội", "chiến sĩ",
    "cán bộ", "quan chức", "lãnh đạo", "chính trị gia",
    "đại biểu", "nghị sĩ",
    "nhà báo", "phóng viên", "báo chí",
    "giáo viên", "thầy giáo", "cô giáo", "giảng viên",
    "bác sĩ", "y tá", "y bác sĩ", "nhân viên y tế",
    "luật sư", "thẩm phán",
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

    # ── Mental health ─────────────────────────────────────────────
    "người trầm cảm", "người lo âu", "bệnh tâm lý",
    "bệnh tâm thần",

    # ── Immigration / social status ───────────────────────────────
    "người nhập cư", "dân nhập cư", "người di cư",
    "người tị nạn",

    # ── Implicit / indirect reference patterns ────────────────────
    "tất cả", "toàn bộ", "hết thảy",
    "đặc trưng", "bản chất", "nòi",
    "giống nòi", "dòng giống",
}

print(f"VIET_TARGET_LEXICON: {len(VIET_TARGET_LEXICON)} terms loaded")
```

- [ ] **Step 5: Update the "Apply Preprocessing" cell to store raw_text**

Find cell id `ff8f3582`. Replace its source with:

```python
print("Preprocessing texts...")
for df, name in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
    df['raw_text']       = df['free_text'].apply(normalize_text)   # unsegmented — for NER (Improvement 3)
    df['text_processed'] = df['free_text'].apply(preprocess)       # word-segmented — for PhoBERT
    print(f'  {name}: done')

print("\nSample:")
for _, row in train_df.sample(3, random_state=42).iterrows():
    lbl = label_map[row['label_id']]
    print(f'  [{lbl:>8}] {row["free_text"][:50]}')
    print(f'  raw      -> {row["raw_text"][:50]}')
    print(f'  seg      -> {row["text_processed"][:50]}')
```

- [ ] **Step 6: Insert truncation profiling cell after the tokenizer loading cell**

Find cell id `70dd7f22` (tokenizer load). Insert a new cell (id: `truncation-profile-001`) immediately after it. Source:

```python
# Improvement 7: Truncation profiling
# If >5% of training samples are truncated, consider MAX_LEN=256 (and BATCH_SIZE=8 for VRAM).
tokenized_lengths = [
    len(tokenizer.tokenize(t)) for t in train_df['text_processed']
]
series = pd.Series(tokenized_lengths)
print("Token length distribution (training set):")
print(series.describe().round(1))
n_truncated = sum(l > MAX_LEN - 2 for l in tokenized_lengths)
pct = n_truncated / len(tokenized_lengths) * 100
print(f"\nTruncated at MAX_LEN={MAX_LEN}: {n_truncated:,} / {len(tokenized_lengths):,} ({pct:.1f}%)")
if pct > 5:
    print("WARNING: >5% truncated. Consider setting MAX_LEN=256 and BATCH_SIZE=8.")
else:
    print("OK: truncation rate is acceptable.")
```

- [ ] **Step 7: Replace the NER section markdown cell**

Find cell id `25598e8b`. Replace source with:

```markdown
## 8. ViAmpleHate Target Identification: Vietnamese NER + Lexicon

**Improvement 1:** Uses `NlpHUST/ner-vietnamese-electra-base` (ELECTRA fine-tuned on VLSP NER).
Entity types filtered: `PER`, `ORG`, `LOC`, `MISC` — the VLSP types that correspond to hate targets
(persons/groups, organizations, locations, miscellaneous groups).

**Improvement 2:** `VIET_TARGET_LEXICON` supplements NER for targets that are common nouns
(e.g., "thằng", "bọn", gender terms, regional slurs) which NER cannot capture.

**Improvement 3:** NER runs on **unsegmented** raw text to avoid PhoBERT word-segmentation
mismatch. Entity surface forms are then mapped to segmented token positions via
`ht.replace(' ', '_')` alignment with underthesea compound-word output.

Expected NER coverage improvement: ~0.09% (baseline) → 20–40% (Vietnamese NER + lexicon).
```

- [ ] **Step 8: Replace the NERTagger + NERProcessor cell**

Find cell id `9e52dbe7`. Replace source with:

```python
class NERTagger:
    """Vietnamese NER tagger using NlpHUST/ner-vietnamese-electra-base (Improvement 1).
    Filters VLSP entity types: PER, ORG, LOC, MISC.
    """
    def __init__(self, model_name=NER_MODEL):
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=0 if DEVICE.type == 'cuda' else 'cpu'
        )

    def extract_named_entities(self, text):
        entities = self.ner_pipeline(text)
        # VLSP Vietnamese NER types aligned with hate targets:
        # PER=person/group, ORG=organization, LOC=location, MISC=miscellaneous groups
        target_types = {"PER", "ORG", "LOC", "MISC"}
        return [
            e["word"] for e in entities
            if e["entity_group"] in target_types
        ]


class NERProcessor:
    """Tokenizes text and finds head_token_idx using Vietnamese NER + lexicon (Improvements 2, 3).

    tokenize_and_encode(text_segmented, text_raw):
      - Runs NER on text_raw (unsegmented) to avoid segmentation mismatch
      - Scans VIET_TARGET_LEXICON on text_segmented (lowercased)
      - Maps found terms to PhoBERT token positions via '_'-joined alignment
      - Falls back to [0] (CLS) when no targets found (original AmpleHate behavior)
    """
    def __init__(self, tokenizer, ner_tagger=None, use_ner=True):
        self.tokenizer = tokenizer
        self.ner_tagger = ner_tagger
        self.use_ner = use_ner

    def extract_head_tokens(self, text_raw, text_segmented):
        tokens_found = []
        # 1. Vietnamese NER on unsegmented text (Improvement 3)
        if self.use_ner and self.ner_tagger is not None:
            tokens_found.extend(self.ner_tagger.extract_named_entities(text_raw))
        # 2. Lexicon scan on segmented text (Improvement 2)
        text_lower = text_segmented.lower()
        for term in VIET_TARGET_LEXICON:
            if term in text_lower:
                tokens_found.append(term)
        return tokens_found

    def tokenize_and_encode(self, text_segmented, text_raw=""):
        head_tokens = self.extract_head_tokens(text_raw, text_segmented)

        encoding = self.tokenizer(
            text_segmented,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        token_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Map entity surface forms to segmented token positions (Improvement 3)
        seg_tokens = self.tokenizer.tokenize(text_segmented)
        head_token_idx = []
        for ht in head_tokens:
            ht_seg = ht.replace(' ', '_')  # align to underthesea compound-word format
            for i, tok in enumerate(seg_tokens):
                if ht_seg in tok or tok.replace('▁', '') == ht_seg:
                    idx = i + 1  # +1 for [CLS] at position 0
                    if idx < MAX_LEN - 1:
                        head_token_idx.append(idx)
                        break

        if not head_token_idx:
            head_token_idx = [0]  # CLS fallback (original AmpleHate behavior)

        return token_ids, head_token_idx, attention_mask
```

- [ ] **Step 9: Replace the NER loading cell**

Find cell id `f9c9bbdc`. Replace source with:

```python
print("Loading Vietnamese NER model (Improvement 1)...")
print(f"Model: {NER_MODEL}")
ner_tagger = NERTagger()
ner_processor_train = NERProcessor(tokenizer, ner_tagger=ner_tagger, use_ner=True)
ner_processor_eval  = NERProcessor(tokenizer, ner_tagger=None, use_ner=False)

# Verify Vietnamese NER coverage
test_texts = [
    "thằng đó là người Hà Nội",           # regional target — lexicon hit
    "Tao ghét tụi LGBT",                   # LGBTQ+ target — lexicon hit
    "người Bắc toàn nói xàm",             # regional target — lexicon hit
    "cộng sản tham nhũng hết",             # political target — lexicon hit
    "Hôm nay trời đẹp quá",               # no target
]
print("\nVietnamese NER + lexicon check:")
for t in test_texts:
    raw = normalize_text(t)
    seg = preprocess(t)
    entities = ner_tagger.extract_named_entities(raw)
    lex_hits = [term for term in VIET_TARGET_LEXICON if term in seg.lower()]
    combined = list(set(entities + lex_hits))
    print(f"  {t!r}")
    print(f"    NER: {entities if entities else '[]'}")
    print(f"    Lex: {lex_hits[:5]}{'...' if len(lex_hits) > 5 else ''}")
    print(f"    Combined: {combined if combined else '[CLS fallback]'}")
```

- [ ] **Step 10: Replace the AmpleHateDataset cell**

Find cell id `e8a3cbee`. Replace source with:

```python
class AmpleHateDataset(Dataset):
    """AmpleHate dataset with Vietnamese improvements (Improvements 2, 3).
    Stores both raw_texts (for NER) and texts (segmented, for PhoBERT).
    """
    def __init__(self, df, ner_processor):
        self.raw_texts = df['raw_text'].fillna('').tolist()       # unsegmented — for NER
        self.texts     = df['text_processed'].fillna('').tolist() # segmented — for PhoBERT
        self.labels    = df['label_id'].astype(int).tolist()
        self.processor = ner_processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        token_ids, head_token_idx, attention_mask = self.processor.tokenize_and_encode(
            self.texts[idx], self.raw_texts[idx]  # pass both segmented and raw
        )
        return {
            'input_ids':      torch.tensor(token_ids,      dtype=torch.long),
            'head_token_idx': torch.tensor(head_token_idx, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    """Pads head_token_idx to max entity count in the batch."""
    max_heads = max(len(item['head_token_idx']) for item in batch)
    padded_heads = []
    for item in batch:
        h = item['head_token_idx']
        pad = torch.zeros(max_heads - len(h), dtype=torch.long)
        padded_heads.append(torch.cat([h, pad]))
    return {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'head_token_idx': torch.stack(padded_heads),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels':         torch.stack([b['labels']         for b in batch]),
    }
```

- [ ] **Step 11: Replace the dataset build cell**

Find cell id `74139a63`. Replace source with:

```python
print("Building datasets...")
print("  Train: applying Vietnamese NER + lexicon...")
train_ds = AmpleHateDataset(train_df, ner_processor_train)
print("  Val/Test: NER disabled (lexicon only via CLS fallback path)...")
val_ds   = AmpleHateDataset(val_df,   ner_processor_eval)
test_ds  = AmpleHateDataset(test_df,  ner_processor_eval)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, generator=g,
    num_workers=0, pin_memory=PIN_MEMORY  # NER pipeline is not fork-safe
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

print(f'Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}')
```

- [ ] **Step 12: Replace the AmpleHatePhoBERT cell to add self.last_embedding**

Find cell id `596e72ed`. The `HeadAttention` class inside remains **identical to baseline**. Only `AmpleHatePhoBERT.forward` changes — add `self.last_embedding` before the classifier. Replace the entire cell source with:

```python
class HeadAttention(nn.Module):
    """Original AmpleHate HeadAttention (model/model.py:5-27, unchanged)."""
    def __init__(self, hidden_dim, head_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim   = head_dim
        self.softmax    = nn.Softmax(dim=-1)
        self.W_q = nn.Linear(hidden_dim, head_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, head_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, head_dim, bias=False)

    def forward(self, cls_embedding, head_token_embedding):
        Q_h = self.W_q(cls_embedding)          # [batch, head_dim]
        K_h = self.W_k(head_token_embedding)   # [batch, head_dim]
        V_h = self.W_v(cls_embedding)          # [batch, head_dim]
        scores  = torch.matmul(Q_h, K_h.T) / (self.head_dim ** 0.5)
        scores  = scores.float()
        weights = self.softmax(scores)
        return torch.matmul(weights, V_h)      # [batch, head_dim]


class AmpleHatePhoBERT(nn.Module):
    """ViAmpleHate: AmpleHate with PhoBERT encoder and contrastive loss support."""
    def __init__(self, model_name, hidden_dim=HIDDEN_DIM, e=E_INJECTION, dropout=DROPOUT):
        super().__init__()
        self.bert           = AutoModel.from_pretrained(model_name)
        self.hidden_dim     = hidden_dim
        self.e              = e
        self.head_attention = HeadAttention(hidden_dim, HEAD_DIM)
        self.dropout        = nn.Dropout(dropout)
        self.classifier     = nn.Linear(hidden_dim, NUM_CLASSES)
        self.last_embedding = None  # set in forward; used by ContrastiveLoss (Improvement 5)

    def forward(self, input_ids, head_token_idx, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        expanded_idx = head_token_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        head_token_embeddings = torch.gather(outputs.last_hidden_state, 1, expanded_idx)

        outputs_list = [
            self.head_attention(cls_embedding, head_token_embeddings[:, i, :])
            for i in range(head_token_embeddings.shape[1])
        ]
        head_attention_output = sum(outputs_list)

        # Direct injection: cls + e * attention_output (AmpleHate core contribution)
        final_embedding = cls_embedding + head_attention_output * self.e
        self.last_embedding = final_embedding.detach()  # Improvement 5: for ContrastiveLoss
        final_embedding = self.dropout(final_embedding)

        return self.classifier(final_embedding)
```

- [ ] **Step 13: Replace the loss setup cell to add ContrastiveLoss**

Find cell id `411873bb`. Replace source with:

```python
label_counts  = train_df['label_id'].value_counts().sort_index().values
class_weights = torch.tensor(
    len(train_df) / (NUM_CLASSES * label_counts),
    dtype=torch.float32, device=DEVICE
)
print('Class weights:', class_weights.cpu().numpy().round(3))

criterion_train = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
criterion_eval  = nn.CrossEntropyLoss()
criterion_cl    = ContrastiveLossCosine(margin=CONTRASTIVE_MARGIN)  # Improvement 5

no_decay    = ['bias', 'LayerNorm.weight']
bert_decay, bert_no_decay = [], []
for name, param in model.bert.named_parameters():
    if not param.requires_grad:
        continue
    (bert_no_decay if any(nd in name for nd in no_decay) else bert_decay).append(param)

head_params = (
    list(model.head_attention.parameters()) +
    list(model.classifier.parameters())
)

optimizer = optim.AdamW([
    {'params': bert_decay,    'lr': LR,      'weight_decay': WEIGHT_DECAY},
    {'params': bert_no_decay, 'lr': LR,      'weight_decay': 0.0},
    {'params': head_params,   'lr': HEAD_LR, 'weight_decay': WEIGHT_DECAY},
])

total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
print(f'Total steps: {total_steps} | Warmup: {warmup_steps}')
print(f'ContrastiveLoss: margin={CONTRASTIVE_MARGIN}, lambda={LAMBDA_CL}')
```

- [ ] **Step 14: Replace train_epoch cell to include ContrastiveLoss**

Find cell id `0c41dd07`. Replace source with:

```python
def train_epoch(model, loader, optimizer, scheduler, criterion, criterion_cl, scaler):
    model.train()
    total_loss = total_correct = total_n = 0

    for batch in loader:
        ids   = batch['input_ids'].to(DEVICE,      non_blocking=PIN_MEMORY)
        heads = batch['head_token_idx'].to(DEVICE,  non_blocking=PIN_MEMORY)
        mask  = batch['attention_mask'].to(DEVICE,  non_blocking=PIN_MEMORY)
        y     = batch['labels'].to(DEVICE,          non_blocking=PIN_MEMORY)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            logits  = model(ids, heads, mask)
            ce_loss = criterion(logits, y)
            cl_loss = criterion_cl(model.last_embedding, y)  # Improvement 5
            loss    = ce_loss + LAMBDA_CL * cl_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds = logits.argmax(1)
        total_loss    += loss.item() * y.size(0)
        total_correct += (preds == y).sum().item()
        total_n       += y.size(0)

    return total_loss / total_n, total_correct / total_n
```

- [ ] **Step 15: Update the training loop cell to pass criterion_cl**

Find cell id `765df725`. Replace the `train_epoch` call line:

Old:
```python
    tr_loss, tr_acc = train_epoch(
        model, train_loader, optimizer, scheduler, criterion_train, scaler
    )
```

New:
```python
    tr_loss, tr_acc = train_epoch(
        model, train_loader, optimizer, scheduler, criterion_train, criterion_cl, scaler
    )
```

- [ ] **Step 16: Insert e-injection sweep cell (commented out) after the training loop**

Insert a new code cell (id: `e-sweep-commented-001`) immediately after the training loop cell (`765df725`). Source:

```python
# Improvement 6: e-injection sweep (COMMENTED OUT — activates 5x training time)
# Run this after Improvements 1-3 are confirmed active (NER coverage > 20%).
# Higher e is better when NER hits are reliable; lower e reduces noise from CLS-on-CLS fallback.
#
# for e_val in [0.5, 0.75, 1.0, 1.25, 1.5]:
#     print(f'\n--- e = {e_val} ---')
#     m = AmpleHatePhoBERT(MODEL_NAME, hidden_dim=HIDDEN_DIM, e=e_val, dropout=DROPOUT).to(DEVICE)
#     no_d = ['bias', 'LayerNorm.weight']
#     bd, bnd = [], []
#     for n, p in m.bert.named_parameters():
#         if not p.requires_grad: continue
#         (bnd if any(nd in n for nd in no_d) else bd).append(p)
#     hp = list(m.head_attention.parameters()) + list(m.classifier.parameters())
#     opt = optim.AdamW([
#         {'params': bd,  'lr': LR,      'weight_decay': WEIGHT_DECAY},
#         {'params': bnd, 'lr': LR,      'weight_decay': 0.0},
#         {'params': hp,  'lr': HEAD_LR, 'weight_decay': WEIGHT_DECAY},
#     ])
#     ts = len(train_loader) * NUM_EPOCHS
#     ws = int(ts * WARMUP_RATIO)
#     sch = get_linear_schedule_with_warmup(opt, ws, ts)
#     sc  = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
#     cl  = ContrastiveLossCosine(margin=CONTRASTIVE_MARGIN)
#     best_e_f1 = -1.0
#     for ep in range(1, NUM_EPOCHS + 1):
#         train_epoch(m, train_loader, opt, sch, criterion_train, cl, sc)
#         _, _, f1, t = evaluate(m, val_loader, criterion_eval)
#         if f1 > best_e_f1:
#             best_e_f1 = f1
#         print(f'  Epoch {ep}: Val F1={f1:.4f}, Threshold={t:.2f}')
#     print(f'  => Best Val F1 for e={e_val}: {best_e_f1:.4f}')
```

- [ ] **Step 17: Update the output config cell**

Find cell id `f1ddeab6`. Replace source with:

```python
os.makedirs('outputs', exist_ok=True)

config = {
    'notebook'           : 'vihsd-viamplehate-phobert-proposed',
    'dataset'            : 'ViHSD (sonlam1102/vihsd)',
    'method'             : 'ViAmpleHate (Vietnamese NER + Lexicon + ContrastiveLoss)',
    'encoder'            : MODEL_NAME,
    'ner_model'          : NER_MODEL,
    'max_len'            : MAX_LEN,
    'hidden_dim'         : HIDDEN_DIM,
    'e_injection'        : E_INJECTION,
    'lambda_cl'          : LAMBDA_CL,
    'contrastive_margin' : CONTRASTIVE_MARGIN,
    'dropout'            : DROPOUT,
    'num_classes'        : NUM_CLASSES,
    'label_names'        : LABEL_NAMES,
    'lr_encoder'         : float(LR),
    'lr_head'            : float(HEAD_LR),
    'best_epoch'         : int(best_epoch),
    'best_val_f1'        : round(float(best_f1), 4),
    'best_threshold'     : round(float(threshold), 2),
    'test_accuracy'      : round(float(acc), 4),
    'test_macro_f1'      : round(float(macro_f1), 4),
    'test_macro_p'       : round(float(macro_p), 4),
    'test_macro_r'       : round(float(macro_r), 4),
    'test_f1_hate'       : round(float(hate_f1), 4),
}

with open('outputs/viamplehate_vihsd_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(json.dumps(config, indent=2, ensure_ascii=False))
```

- [ ] **Step 18: Replace the limitations/next steps section markdown**

Find cell id `2f177e58`. Replace source with:

```markdown
## 17. What Was Improved and What Remains

This notebook applies the highest-priority improvements from `docs/improvementAmpleHate.md`.

---

### Applied Improvements

| # | Improvement | Status |
|---|---|---|
| 1 | Vietnamese NER (`NlpHUST/ner-vietnamese-electra-base`) | ✅ Active |
| 2 | Vietnamese hate-target lexicon | ✅ Active |
| 3 | Word segmentation alignment (NER on raw text) | ✅ Active |
| 4 | Vietnamese target types (VLSP PER/ORG/LOC/MISC) | ✅ Active (via Imp 1) |
| 5 | ContrastiveLossCosine (CE + λ·CL) | ✅ Active |

---

### Optional / Disabled

| # | Improvement | Status |
|---|---|---|
| 6 | e-injection sweep [0.5, 0.75, 1.0, 1.25, 1.5] | Commented cell after training loop |
| 7 | max_length=256 (if truncation > 5%) | Profiling cell included; `MAX_LEN=128` default |
| 8 | PhoBERT-large (HIDDEN_DIM=1024, BATCH_SIZE=8) | `USE_PHOBERT_LARGE=False` flag in §2 |

---

### Not Applied (out of scope)

| # | Improvement | Reason |
|---|---|---|
| 9 | Within-example HeadAttention | Deviates from original AmpleHate; reserved for variant comparison |
| 10 | Multi-class 3-label setup | Optional/advanced; requires separate ablation |

---

### Summary: What Makes ViAmpleHate Different

| Component | Baseline | Proposed (this notebook) |
|---|---|---|
| NER model | English CoNLL-2003 | Vietnamese VLSP (NlpHUST ELECTRA) |
| Target coverage | ~0.09% | ~20–40% (NER + lexicon) |
| Lexicon | None | 200+ Vietnamese hate-target terms |
| Segmentation | Mismatch (NER on segmented) | Fixed (NER on raw, mapped to segmented positions) |
| Loss | CrossEntropy only | CE + 0.1 × ContrastiveLoss |
```

- [ ] **Step 19: Assemble and write the complete .ipynb file**

Build the notebook JSON by taking all cells from the baseline in order, applying the replacements/insertions from Steps 2–18. The cell order is:

1. Title markdown (replaced — Step 2)
2. pip install cell (unchanged)
3. imports cell (unchanged)
4. Hyperparameters section markdown (unchanged)
5. Hyperparameters code cell (replaced — Step 3)
6. **[NEW]** VIET_TARGET_LEXICON cell (inserted — Step 4)
7. Data loading section markdown (unchanged)
8. Data loading code cell (unchanged)
9. Label mapping markdown (unchanged)
10. Label mapping code cell (unchanged)
11. Preprocessing section markdown (unchanged)
12. Preprocessing code cell (unchanged)
13. Apply preprocessing section markdown (unchanged)
14. Apply preprocessing code cell (replaced — Step 5)
15. Tokenizer section markdown (unchanged)
16. Tokenizer loading code cell (unchanged)
17. **[NEW]** Truncation profiling cell (inserted — Step 6)
18. NER section markdown (replaced — Step 7)
19. NERTagger + NERProcessor class cell (replaced — Step 8)
20. NER loading section markdown (unchanged)
21. NER loading + verification cell (replaced — Step 9)
22. Dataset section markdown (unchanged)
23. AmpleHateDataset + collate_fn cell (replaced — Step 10)
24. Dataset build cell (replaced — Step 11)
25. NER coverage statistics cell (unchanged — baseline `d2571841`)
26. Model section markdown (unchanged)
27. HeadAttention + AmpleHatePhoBERT cell (replaced — Step 12)
28. ContrastiveLossCosine cell (unchanged — `9e3d94a1`)
29. Model instantiation cell (unchanged — `45d2ff95`)
30. Loss/optimizer section markdown (unchanged)
31. Loss + optimizer + scheduler cell (replaced — Step 13)
32. Training section markdown (unchanged)
33. best_threshold cell (unchanged — `9bb148b9`)
34. train_epoch cell (replaced — Step 14)
35. evaluate cell (unchanged — `19219f2f`)
36. Training loop cell (updated — Step 15)
37. **[NEW]** e-sweep commented cell (inserted — Step 16)
38. Training curves section markdown (unchanged)
39. Training curves plot cell (unchanged — filenames updated via PLOT_TITLE)
40. Test evaluation section markdown (unchanged)
41. Checkpoint loading cell (unchanged)
42. get_predictions cell (unchanged)
43. Classification report cell (unchanged)
44. Full test metrics section markdown (unchanged)
45. Metrics cell (unchanged)
46. Confusion matrix cell (unchanged)
47. Inference section markdown (unchanged)
48. predict function + TEST_CASES cell (unchanged)
49. Output config cell (replaced — Step 17)
50. Blank markdown cell (unchanged)
51. Limitations section markdown (replaced — Step 18)

Write the file using the Write tool with proper nbformat 4.5 JSON structure. Cell metadata is `{}`. All code cells have `"outputs": [], "execution_count": null`. Use UUIDs for new cell IDs.

- [ ] **Step 20: Verify the notebook is valid JSON**

```bash
python -c "import json; nb = json.load(open('notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/ViHSD - Proposed ViAmpleHate_PhoBERT.ipynb')); print(f'Valid JSON. nbformat={nb[\"nbformat\"]}. Cells: {len(nb[\"cells\"])}')"
```

Expected: `Valid JSON. nbformat=4. Cells: 51` (approximately)

- [ ] **Step 21: Commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/"
git commit -m "feat: add ViHSD proposed ViAmpleHate PhoBERT notebook

Applies Improvements 1-5 from improvementAmpleHate.md:
Vietnamese NER (NlpHUST/ner-vietnamese-electra-base), hate-target lexicon,
word segmentation alignment, and ContrastiveLossCosine auxiliary loss.
Improvements 6-8 added as disabled-by-default config flags.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Write VOZ-HSD proposed notebook

**Files:**
- Read: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/ViHSD - Proposed ViAmpleHate_PhoBERT.ipynb` (the file just created)
- Read: `notebooks/models/baselines/VOZ-HSD - Baseline AmpleHate_PhoBERT/voz-hsd-baseline-amplehate-phobert.ipynb` (for dataset cell reference)
- Create: `notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/VOZ-HSD - Proposed ViAmpleHate_PhoBERT.ipynb`

Start from the ViHSD proposed notebook and apply the VOZ-HSD adaptations below. Everything not listed is **identical to the ViHSD proposed notebook**.

- [ ] **Step 1: Read both input files**

Read the ViHSD proposed notebook (just created) and the VOZ-HSD baseline notebook.

- [ ] **Step 2: Replace the title markdown cell**

Replace source with:

```markdown
# ViAmpleHate on VOZ-HSD: PhoBERT Proposed

This notebook implements **ViAmpleHate** — an improved version of AmpleHate (Lee et al., EMNLP 2025)
adapted for Vietnamese hate speech on the **VOZ-HSD** dataset using **PhoBERT** as the encoder.

Improvements over the baseline (see `docs/improvementAmpleHate.md`):
1. **Vietnamese NER** (`NlpHUST/ner-vietnamese-electra-base`) replaces English CoNLL-2003 NER
2. **Vietnamese hate-target lexicon** complements NER for non-entity targets
3. **Word segmentation alignment**: NER runs on raw text, mapped to segmented token positions
4. **ContrastiveLossCosine** enabled as auxiliary training signal (CE + λ·CL)

Not applied (see optional cells):
- Improvement 6 (e-sweep): commented cell after training
- Improvement 7 (max_length): truncation profiling cell included; MAX_LEN=128 by default
- Improvement 8 (PhoBERT-large): config flag USE_PHOBERT_LARGE=False
```

- [ ] **Step 3: Update CKPT_NAME and PLOT_TITLE in the hyperparameters cell**

In the hyperparameters cell, change only:
```python
CKPT_NAME    = 'best_viamplehate_phobert_vozhsd.pt'
PLOT_TITLE   = 'ViAmpleHate (PhoBERT) — VOZ-HSD Proposed'
```

Also add `SAMPLE_SIZE` and `RANDOM_STATE` to the hyperparameters cell (after LABEL_NAMES block):
```python
# Dataset sampling (VOZ-HSD only)
SAMPLE_SIZE  = 100_000
RANDOM_STATE = 42
```

- [ ] **Step 4: Replace the data loading section markdown cell**

Find the markdown cell for "## 3. Load ViHSD Dataset". Replace source with:

```markdown
## 3. Load VOZ-HSD Dataset

Loading from HuggingFace Hub. Requires a Kaggle secret `HF_TOKEN`.
VOZ-HSD has one `train` split with columns `texts` and `labels`.
We rename them to `free_text` and `label_id`, then sample 100k rows
with the same class ratio as the full dataset (stratified).
This matches the data loading used in the PhoBERT-CNN VOZ-HSD baseline.
```

- [ ] **Step 5: Replace the data loading code cell**

Replace the `load_dataset` cell (cell id `9172999d` in ViHSD proposed) with the VOZ-HSD loading logic:

```python
from kaggle_secrets import UserSecretsClient
from datasets import load_dataset
import huggingface_hub

secret_value = UserSecretsClient().get_secret("HF_TOKEN")
huggingface_hub.login(token=secret_value, add_to_git_credential=False)

# VOZ-HSD has only the "train" split. Take a stratified 100k sample.
ds      = load_dataset("tarudesu/VOZ-HSD", split="train")
full_df = ds.to_pandas()[["texts", "labels"]]
full_df = full_df.rename(columns={"texts": "free_text", "labels": "label_id"})

sampled_df = (
    full_df
    .groupby("label_id", group_keys=False)
    .apply(lambda g: g.sample(frac=SAMPLE_SIZE / len(full_df), random_state=RANDOM_STATE))
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

print(f"Full dataset : {len(full_df):,} rows")
print(f"Sampled      : {len(sampled_df):,} rows")
print(f"Columns      : {sampled_df.columns.tolist()}")
print()
print("Class ratio in full dataset:")
print(full_df["label_id"].value_counts(normalize=True).sort_index().rename({0: "NON-HATE", 1: "HATE"}).round(4))
print()
print("Class ratio after sampling:")
print(sampled_df["label_id"].value_counts(normalize=True).sort_index().rename({0: "NON-HATE", 1: "HATE"}).round(4))
```

- [ ] **Step 6: Replace the label mapping section markdown**

Replace the label mapping markdown with:

```markdown
## 4. Label Mapping

VOZ-HSD is already binary: NON-HATE=0, HATE=1.
We keep labels unchanged and split into Train/Val/Test = 80/10/10.
```

- [ ] **Step 7: Replace the label mapping code cell**

Replace the label remapping code cell (the one with `train_df['label_id'].map(...)` in ViHSD) with:

```python
# VOZ-HSD labels are already binary: NON-HATE=0, HATE=1.
label_map = {0: 'NON-HATE', 1: 'HATE'}

n       = len(sampled_df)
n_train = int(n * 0.8)
n_val   = int(n * 0.1)

train_df = sampled_df.iloc[:n_train].reset_index(drop=True)
val_df   = sampled_df.iloc[n_train:n_train+n_val].reset_index(drop=True)
test_df  = sampled_df.iloc[n_train+n_val:].reset_index(drop=True)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print()
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = df['label_id'].value_counts().sort_index().rename(label_map)
    print(f"  {name}: {dist.to_dict()}")

train_df.head(3)
```

- [ ] **Step 8: Update the dataset build cell print message**

In the dataset build cell (equivalent of Step 11 from Task 2), update the print message:

```python
print("Building datasets...")
print(f"  Train: applying Vietnamese NER + lexicon on {len(train_df):,} VOZ-HSD samples...")
```

- [ ] **Step 9: Update the output config cell**

Replace the output config cell source with:

```python
os.makedirs('outputs', exist_ok=True)

config = {
    'notebook'           : 'voz-hsd-viamplehate-phobert-proposed',
    'dataset'            : 'VOZ-HSD (100k from tarudesu/VOZ-HSD)',
    'method'             : 'ViAmpleHate (Vietnamese NER + Lexicon + ContrastiveLoss)',
    'encoder'            : MODEL_NAME,
    'ner_model'          : NER_MODEL,
    'sample_size'        : int(SAMPLE_SIZE),
    'max_len'            : MAX_LEN,
    'hidden_dim'         : HIDDEN_DIM,
    'e_injection'        : E_INJECTION,
    'lambda_cl'          : LAMBDA_CL,
    'contrastive_margin' : CONTRASTIVE_MARGIN,
    'dropout'            : DROPOUT,
    'num_classes'        : NUM_CLASSES,
    'label_names'        : LABEL_NAMES,
    'lr_encoder'         : float(LR),
    'lr_head'            : float(HEAD_LR),
    'best_epoch'         : int(best_epoch),
    'best_val_f1'        : round(float(best_f1), 4),
    'best_threshold'     : round(float(threshold), 2),
    'test_accuracy'      : round(float(acc), 4),
    'test_macro_f1'      : round(float(macro_f1), 4),
    'test_macro_p'       : round(float(macro_p), 4),
    'test_macro_r'       : round(float(macro_r), 4),
    'test_f1_hate'       : round(float(hate_f1), 4),
}

with open('outputs/viamplehate_vozhsd_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(json.dumps(config, indent=2, ensure_ascii=False))
```

- [ ] **Step 10: Update the limitations section dataset references**

In the limitations table (equivalent of Step 18 in Task 2), replace "ViHSD" with "VOZ-HSD" in the dataset column. The table content remains the same structure.

- [ ] **Step 11: Assemble and write the complete .ipynb file**

Build the notebook JSON following the same cell order as the ViHSD proposed notebook, with cells from Steps 2–10 replaced. Write using the Write tool.

- [ ] **Step 12: Verify the notebook is valid JSON**

```bash
python -c "import json; nb = json.load(open('notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/VOZ-HSD - Proposed ViAmpleHate_PhoBERT.ipynb')); print(f'Valid JSON. nbformat={nb[\"nbformat\"]}. Cells: {len(nb[\"cells\"])}')"
```

Expected: `Valid JSON. nbformat=4. Cells: 51` (approximately, same as ViHSD)

- [ ] **Step 13: Commit**

```bash
git add "notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/"
git commit -m "feat: add VOZ-HSD proposed ViAmpleHate PhoBERT notebook

Adapts the ViHSD proposed notebook for VOZ-HSD: stratified 100k sample,
no label remapping (already binary), 80/10/10 manual split.
All model/NER/lexicon/training code identical to ViHSD proposed.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Final verification

- [ ] **Step 1: Check both notebooks parse correctly**

```bash
python -c "
import json, pathlib
for p in pathlib.Path('notebooks/models/proposed').rglob('*.ipynb'):
    nb = json.load(open(p))
    cells = nb['cells']
    code_cells = [c for c in cells if c['cell_type'] == 'code']
    print(f'{p.name}: {len(cells)} cells ({len(code_cells)} code)')
"
```

Expected: Both notebooks listed with ~51 cells each.

- [ ] **Step 2: Verify key improvements are present in ViHSD proposed**

```bash
python -c "
import json
nb = json.load(open('notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/ViHSD - Proposed ViAmpleHate_PhoBERT.ipynb'))
src = '\n'.join(''.join(c['source']) for c in nb['cells'])
checks = [
    ('Imp 1: Vietnamese NER model', 'NlpHUST/ner-vietnamese-electra-base'),
    ('Imp 2: Lexicon constant', 'VIET_TARGET_LEXICON'),
    ('Imp 2: Lexicon term', 'thằng'),
    ('Imp 3: raw_text column', 'raw_text'),
    ('Imp 3: NER on raw text', 'extract_head_tokens'),
    ('Imp 3: segmentation align', \"ht.replace(' ', '_')\"),
    ('Imp 5: last_embedding', 'self.last_embedding'),
    ('Imp 5: ContrastiveLoss in train', 'criterion_cl'),
    ('Imp 5: combined loss', 'LAMBDA_CL'),
    ('Imp 6: commented sweep', 'e_val in [0.5'),
    ('Imp 7: profiling cell', 'n_truncated'),
    ('Imp 8: config flag', 'USE_PHOBERT_LARGE'),
]
for name, pattern in checks:
    status = 'OK' if pattern in src else 'MISSING'
    print(f'  [{status}] {name}')
"
```

Expected: All checks show `[OK]`.

- [ ] **Step 3: Verify key differences between ViHSD and VOZ-HSD proposed**

```bash
python -c "
import json
vihsd  = json.load(open('notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/ViHSD - Proposed ViAmpleHate_PhoBERT.ipynb'))
vozhsd = json.load(open('notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/VOZ-HSD - Proposed ViAmpleHate_PhoBERT.ipynb'))

vi_src  = chr(10).join(''.join(c['source']) for c in vihsd['cells'])
voz_src = chr(10).join(''.join(c['source']) for c in vozhsd['cells'])

checks = [
    ('ViHSD: HF dataset',       'sonlam1102/vihsd',               vi_src),
    ('ViHSD: label remap',      'map(lambda x: 1 if x == 2',      vi_src),
    ('ViHSD: ckpt name',        'vihsd.pt',                        vi_src),
    ('VOZ-HSD: HF dataset',     'tarudesu/VOZ-HSD',               voz_src),
    ('VOZ-HSD: SAMPLE_SIZE',    'SAMPLE_SIZE',                    voz_src),
    ('VOZ-HSD: manual split',   'n_train = int(n * 0.8)',          voz_src),
    ('VOZ-HSD: ckpt name',      'vozhsd.pt',                       voz_src),
    ('Both: Vietnamese NER',    'NlpHUST/ner-vietnamese-electra',  vi_src),
    ('Both: Vietnamese NER',    'NlpHUST/ner-vietnamese-electra',  voz_src),
]
for name, pattern, src in checks:
    status = 'OK' if pattern in src else 'MISSING'
    print(f'  [{status}] {name}')
"
```

Expected: All checks show `[OK]`.

- [ ] **Step 4: Final commit**

```bash
git add docs/superpowers/plans/2026-05-13-viamplehate-proposed-notebooks.md
git commit -m "docs: add implementation plan for ViAmpleHate proposed notebooks

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```
