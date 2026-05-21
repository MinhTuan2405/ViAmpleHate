# ViAmpleHate-Vi++ Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `vihsd-proposed-viamplehate-phobert.ipynb` implementing AmpleHate-Vi++ — Vietnamese-adapted AmpleHate with multi-signal target cue mining, 3-module relation bank, instance-adaptive gate, and CE+contrastive loss.

**Architecture:** Vietnamese NER (NlpHUST) + target/attack cue lexicons replace English NER, boosting coverage from 0.09% to ~45-55%. Three HeadAttention modules (r_exp, r_imp, r_atk) fuse into a learned scalar gate per-instance instead of fixed scalar `e`. Loss = CE (weighted) + ContrastiveLoss (α=0.1).

**Tech Stack:** Python, PyTorch, HuggingFace Transformers, vinai/phobert-base, NlpHUST/ner-vietnamese-electra-base, underthesea, scikit-learn, Kaggle T4

**Spec:** `docs/superpowers/specs/2026-05-21-viamplehate-proposed-design.md`

---

## File Structure

| File | Action |
|---|---|
| `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb` | Create (overwrite with fresh skeleton, then add cells) |

---

### Task 1: Create skeleton notebook

**Files:**
- Create: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb`

- [ ] **Step 1: Write empty valid notebook**

Use the Write tool to save a minimal `.ipynb` to:
`notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb`

Content:
```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": []
}
```

- [ ] **Step 2: Commit skeleton**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: init proposed ViAmpleHate-Vi++ notebook skeleton"
```

---

### Task 2: Sections 1–2 — Title, Install, Imports, Hyperparameters

**Files:**
- Modify: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb`

- [ ] **Step 1: Add title markdown cell** (NotebookEdit, `add_cell`, type=`markdown`, after last cell)

```markdown
# AmpleHate-Vi++ on ViHSD: PhoBERT Proposed

Implements **AmpleHate-Vi++** — Vietnamese-adapted AmpleHate with:
1. Vietnamese NER (`NlpHUST/ner-vietnamese-electra-base`) + target cue lexicon → coverage 0.09% → ~45%
2. Separate attack cue bank (offensive predicates)
3. Relation Bank: 3 HeadAttention modules (r_exp, r_imp, r_atk) fused via Linear
4. Instance-adaptive gate `g = σ(W·[h_CLS; r])` replacing fixed scalar `e`
5. CrossEntropy (weighted) + ContrastiveLoss (α=0.1)
6. max_length=256, gradient accumulation=2, 8 epochs max

**Baseline reference:** `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/`
**Spec:** `docs/superpowers/specs/2026-05-21-viamplehate-proposed-design.md`
```

- [ ] **Step 2: Add install cell** (NotebookEdit, `add_cell`, type=`code`)

```python
!pip install transformers datasets sentencepiece huggingface_hub underthesea easydict -q
```

- [ ] **Step 3: Add imports + device cell** (NotebookEdit, `add_cell`, type=`code`)

```python
import os, re, time, random, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, pipeline

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = DEVICE.type == 'cuda'
NUM_WORKERS = 2 if os.cpu_count() and os.cpu_count() > 2 else 0

print(f'Device : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Workers: {NUM_WORKERS} | Pin memory: {PIN_MEMORY}')
```

- [ ] **Step 4: Add hyperparameters cell** (NotebookEdit, `add_cell`, type=`code`)

```python
MODEL_NAME  = 'vinai/phobert-base'
NER_MODEL   = 'NlpHUST/ner-vietnamese-electra-base'
MAX_LEN     = 256
HIDDEN_DIM  = 768
HEAD_DIM    = HIDDEN_DIM

BATCH_SIZE      = 16
NUM_EPOCHS      = 8
LR              = 2e-5
HEAD_LR         = 5e-5
WARMUP_RATIO    = 0.06
DROPOUT         = 0.1
PATIENCE        = 2
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.05
GRAD_ACCUM      = 2       # effective batch = 32
ALPHA_CL        = 0.1    # contrastive loss weight

NUM_CLASSES = 2
LABEL_NAMES = ['NON-HATE', 'HATE']
CKPT_NAME   = 'best_viamplehate_phobert_vihsd.pt'
PLOT_TITLE  = 'AmpleHate-Vi++ (PhoBERT) — ViHSD Proposed'
```

- [ ] **Step 5: Commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: add title, installs, imports, hyperparameters cells"
```

---

### Task 3: Sections 3–6 — Dataset, Label Mapping, Preprocessing, Tokenizer

- [ ] **Step 1: Add dataset loading cell** (NotebookEdit, `add_cell`, type=`code`)

```python
from kaggle_secrets import UserSecretsClient
from datasets import load_dataset
import huggingface_hub

secret_value = UserSecretsClient().get_secret("HF_TOKEN")
huggingface_hub.login(token=secret_value, add_to_git_credential=False)

ds       = load_dataset("sonlam1102/vihsd")
train_df = ds["train"].to_pandas()
val_df   = ds["validation"].to_pandas()
test_df  = ds["test"].to_pandas()

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"Label distribution (raw): {train_df['label_id'].value_counts().to_dict()}")
```

- [ ] **Step 2: Add label mapping cell** (NotebookEdit, `add_cell`, type=`code`)

```python
train_df['label_id'] = train_df['label_id'].map(lambda x: 1 if x == 2 else 0)
val_df['label_id']   = val_df['label_id'].map(lambda x: 1 if x == 2 else 0)
test_df['label_id']  = test_df['label_id'].map(lambda x: 1 if x == 2 else 0)

label_map = {0: 'NON-HATE', 1: 'HATE'}
for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = df['label_id'].value_counts().sort_index().rename(label_map)
    print(f"  {name}: {dist.to_dict()}")
```

- [ ] **Step 3: Add preprocessing cell** (NotebookEdit, `add_cell`, type=`code`)

```python
from underthesea import word_tokenize

TEENCODE_MAP = {
    'ko': 'không', 'kh': 'không', 'khong': 'không', 'kg': 'không',
    'hok': 'không', 'hk': 'không', 'hem': 'không', 'kô': 'không',
    'chx': 'chưa', 'chua': 'chưa',
    'r': 'rồi', 'rui': 'rồi', 'ròi': 'rồi', 'oy': 'rồi', 'uj': 'rồi',
    'mk': 'mình', 'mik': 'mình', 'mh': 'mình',
    'tui': 'tôi', 'tau': 'tao', 'may': 'mày', 'mi': 'mày',
    'bn': 'bạn', 'ban': 'bạn',
    'no': 'nó', 'mng': 'mọi người', 'mn': 'mọi người', 'ae': 'anh em',
    'dc': 'được', 'đc': 'được', 'dk': 'được', 'đk': 'được',
    'đươc': 'được', 'duoc': 'được', 'vs': 'với', 'voi': 'với',
    'j': 'gì', 'zì': 'gì', 'zi': 'gì',
    'ntn': 'như thế nào', 'nso': 'như sao',
    'biet': 'biết', 'bit': 'biết', 'hieu': 'hiểu', 'nghi': 'nghĩ',
    'muon': 'muốn', 'hoac': 'hoặc', 'neu': 'nếu', 'nen': 'nên',
    'giet': 'giết', 'chui': 'chửi', 'danh': 'đánh',
    'nx': 'nhưng', 'nhg': 'nhưng', 'nhưg': 'nhưng', 'nma': 'nhưng mà',
    'cx': 'cũng', 'cg': 'cũng', 'cung': 'cũng', 'cũg': 'cũng',
    'ms': 'mới', 'boi': 'bởi',
    'oke': 'ok', 'okie': 'ok', 'okê': 'ok', 'okey': 'ok',
    'uh': 'ừ', 'uk': 'ừ', 'uhm': 'ừ',
    'yep': 'đúng', 'yup': 'đúng',
    'haha': 'haha', 'hehe': 'hehe', 'hihi': 'hehe', 'huhu': 'buồn',
    'haiz': 'thở dài', 'haizz': 'thở dài',
    'wtf': 'cái gì vậy', 'omg': 'ôi trời',
    'lol': 'buồn cười', 'lmao': 'buồn cười',
    'fck': 'chửi thề', 'fk': 'chửi thề', 'gg': 'xong rồi', 'ez': 'dễ',
    'bt': 'bình thường', 'bth': 'bình thường',
    'noob': 'tệ', 'nub': 'tệ', 'xàm': 'vô nghĩa', 'nhảm': 'vô nghĩa',
    'pro': 'giỏi',
    'vl': 'vãi lồn', 'vcl': 'vãi cái lồn', 'vkl': 'vãi kép lồn',
    'vll': 'vãi lồn', 'vleu': 'vãi lồn', 'vloz': 'vãi lồn',
    'dm': 'đụ má', 'đm': 'đụ má', 'd.m': 'đụ má', 'đ.m': 'đụ má',
    'đmm': 'đụ má mày', 'dmm': 'đụ má mày',
    'đtm': 'địt mẹ', 'dtm': 'địt mẹ',
    'cl': 'cái lồn', 'lon': 'lồn', 'loz': 'lồn', 'l0n': 'lồn',
    'đéo': 'không', 'deo': 'không', 'éo': 'không',
    'cc': 'cái con', 'thg': 'thằng',
    'ngu': 'ngu', 'đần': 'đần độn', 'khùng': 'điên', 'dien': 'điên',
    'cút': 'cút', 'cut': 'cút',
    'câm': 'câm miệng', 'im mồm': 'câm miệng',
    'fb': 'facebook', 'yt': 'youtube', 'tt': 'tiktok', 'zl': 'zalo',
    'ig': 'instagram', 'cmt': 'bình luận', 'rep': 'trả lời',
    'vn': 'việt nam', 'hn': 'hà nội', 'hcm': 'hồ chí minh', 'sg': 'sài gòn',
    'iu': 'yêu', 'ieu': 'yêu',
}

EMOJI_MAP = {
    '🙃': ' [MOCK] ', '😏': ' [MOCK] ', '😒': ' [MOCK] ',
    '😡': ' [ANGER] ', '🤬': ' [ANGER] ', '😤': ' [ANGER] ',
    '🤮': ' [DISGUST] ', '😖': ' [DISGUST] ',
    '😂': ' [LAUGH] ', '🤣': ' [LAUGH] ',
    '😭': ' [SAD] ', '💔': ' [SAD] ',
    '🔥': ' [INTENSE] ', '💀': ' [DEATH] ',
    '👍': ' [APPROVE] ', '👎': ' [DISAPPROVE] ',
}

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    for emoji, tag in EMOJI_MAP.items():
        text = text.replace(emoji, tag)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b0[0-9]{9,10}\b', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [TEENCODE_MAP.get(w, w) for w in text.split()]
    return ' '.join(words)

def preprocess(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ''
    try:
        return word_tokenize(text, format='text')
    except Exception:
        return text

sample = 'Bọn đó ăn bám lắm, ngu vcl 🙃'
print(f'Original : {sample}')
print(f'Processed: {preprocess(sample)}')
```

- [ ] **Step 4: Add apply preprocessing cell** (NotebookEdit, `add_cell`, type=`code`)

```python
print("Preprocessing texts...")
for df, name in [(train_df, 'Train'), (val_df, 'Val'), (test_df, 'Test')]:
    df['text_processed'] = df['free_text'].apply(preprocess)
    print(f'  {name}: done')
```

- [ ] **Step 5: Add tokenizer cell** (NotebookEdit, `add_cell`, type=`code`)

```python
print('Loading PhoBERT tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f'Vocab size: {tokenizer.vocab_size:,}')
sample = preprocess('Bọn đó ăn bám lắm, ngu vcl 🙃')
print(f'Processed : {sample}')
print(f'Tokens    : {tokenizer.tokenize(sample)}')
```

- [ ] **Step 6: Commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: add dataset, label mapping, preprocessing, tokenizer cells"
```

---

### Task 4: Section 8a-8b — Target Cue and Attack Cue Banks

- [ ] **Step 1: Add section markdown cell** (NotebookEdit, `add_cell`, type=`markdown`)

```markdown
## 8a. Target Cue Bank

Derogatory pronouns and nominal group prefixes that reference a target group.
These are referential tokens only — not offensive by themselves.

## 8b. Attack Cue Bank

Offensive predicates and evaluations directed at the target.
Kept separate from target cues per AmpleHate-Vi++ design: mixing them into target detection
increases recall but conflates target identification with attack detection.
```

- [ ] **Step 2: Add cue banks cell** (NotebookEdit, `add_cell`, type=`code`)

```python
TARGET_CUES = [
    'bọn', 'thằng', 'con', 'đứa', 'tụi', 'đám', 'lũ',
    'mấy', 'loại', 'người', 'dân', 'bên',
    'hắn', 'chúng', 'họ', 'nó',
]

ATTACK_CUES = [
    'ngu', 'đần', 'ngu_ngốc', 'khùng', 'điên', 'hèn', 'nhục',
    'ăn_bám', 'ký_sinh', 'phản_quốc', 'vô_học', 'man_rợ',
    'cút', 'xéo', 'câm_miệng',
    'giết', 'chém', 'đánh',
    'ghét', 'khinh', 'chửi',
    'vô_văn_hóa', 'thấp_hèn', 'đáng_chết',
]

print(f'Target cue lexicon: {len(TARGET_CUES)} entries')
print(f'Attack cue lexicon: {len(ATTACK_CUES)} entries')

# Spot check on a sample
sample_sent = preprocess('Bọn đó toàn ăn bám, ngu hết sức')
toks = tokenizer.tokenize(sample_sent)
t_hits = [t for t in toks if any(t == tokenizer.tokenize(c)[0] for c in TARGET_CUES if tokenizer.tokenize(c))]
a_hits = [t for t in toks if any(t == tokenizer.tokenize(c)[0] for c in ATTACK_CUES if tokenizer.tokenize(c))]
print(f'\nSample: "{sample_sent}"')
print(f'Target hits: {t_hits}')
print(f'Attack hits: {a_hits}')
```

---

### Task 5: Section 9 — Vietnamese NER + MultiSignalProcessor

- [ ] **Step 1: Add section markdown cell** (NotebookEdit, `add_cell`, type=`markdown`)

```markdown
## 9. Vietnamese NER + Multi-signal Target Cue Mining

Replaces the English NER from the baseline with:
1. `NlpHUST/ner-vietnamese-electra-base` — Vietnamese NER (all entity types kept)
2. Target cue lexicon matching (derogatory pronouns / group references)
3. Attack cue lexicon matching (offensive predicates)
4. [CLS] fallback (index 0) when no cues found for either bank
```

- [ ] **Step 2: Add VietnameseNERTagger + MultiSignalProcessor cell** (NotebookEdit, `add_cell`, type=`code`)

```python
class VietnameseNERTagger:
    """Vietnamese NER using NlpHUST/ner-vietnamese-electra-base."""
    def __init__(self, model_name=NER_MODEL):
        self.ner_pipeline = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=0 if DEVICE.type == 'cuda' else 'cpu'
        )

    def extract_named_entities(self, text):
        try:
            return [e["word"] for e in self.ner_pipeline(text) if e.get("word")]
        except Exception:
            return []


class MultiSignalProcessor:
    """Tokenizes text and returns head_token_idx (targets) + attack_token_idx (attack cues)."""
    def __init__(self, tokenizer, ner_tagger=None, use_ner=True):
        self.tokenizer  = tokenizer
        self.ner_tagger = ner_tagger
        self.use_ner    = use_ner

    def _find_positions(self, tokens, cue_list):
        positions = []
        for cue in cue_list:
            cue_toks = self.tokenizer.tokenize(cue)
            if not cue_toks:
                continue
            first = cue_toks[0]
            for i, tok in enumerate(tokens):
                if tok == first and (i + 1) < MAX_LEN - 1:
                    positions.append(i + 1)  # +1 offset for [CLS] at position 0
        return list(set(positions))

    def tokenize_and_encode(self, text):
        tokens   = self.tokenizer.tokenize(text)
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=MAX_LEN
        )

        # Target positions: NER entities + target cue lexicon
        head_positions = []
        if self.use_ner and self.ner_tagger:
            for ent in self.ner_tagger.extract_named_entities(text):
                ent_toks = self.tokenizer.tokenize(ent)
                if ent_toks:
                    try:
                        idx = tokens.index(ent_toks[0]) + 1
                        if idx < MAX_LEN - 1:
                            head_positions.append(idx)
                    except ValueError:
                        pass
        head_positions += self._find_positions(tokens, TARGET_CUES)
        head_positions  = list(set(head_positions)) or [0]

        # Attack positions: attack cue lexicon
        attack_positions = self._find_positions(tokens, ATTACK_CUES) or [0]

        return (
            encoding["input_ids"],
            head_positions,
            attack_positions,
            encoding["attention_mask"]
        )
```

- [ ] **Step 3: Add NER loading + coverage check cell** (NotebookEdit, `add_cell`, type=`code`)

```python
print("Loading Vietnamese NER model...")
ner_tagger      = VietnameseNERTagger()
processor_train = MultiSignalProcessor(tokenizer, ner_tagger=ner_tagger, use_ner=True)
processor_eval  = MultiSignalProcessor(tokenizer, ner_tagger=None,       use_ner=False)

test_cases = [
    'Bọn người miền Bắc toàn ăn bám',
    'mày ngu vcl',
    'thằng đó là người Hà Nội',
    'tôi yêu Việt Nam',
    'đám đó khinh người khác',
    'Bọn đó toàn ăn bám xã hội 🙃',
]
print("\nTarget / Attack cue detection (eval processor — no NER):")
for t in test_cases:
    proc = preprocess(t)
    _, h_idx, a_idx, _ = processor_eval.tokenize_and_encode(proc)
    toks = tokenizer.tokenize(proc)
    h_toks = [toks[i-1] if 0 < i <= len(toks) else '[CLS]' for i in h_idx]
    a_toks = [toks[i-1] if 0 < i <= len(toks) else '[CLS]' for i in a_idx]
    print(f"  {t}")
    print(f"    target: {h_toks} | attack: {a_toks}")
```

- [ ] **Step 4: Commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: add target/attack cue banks and MultiSignalProcessor"
```

---

### Task 6: Section 10 — Dataset + DataLoader

- [ ] **Step 1: Add ViAmpleHateDataset + collate_fn cell** (NotebookEdit, `add_cell`, type=`code`)

```python
class ViAmpleHateDataset(Dataset):
    def __init__(self, df, processor):
        self.texts     = df['text_processed'].fillna('').tolist()
        self.labels    = df['label_id'].astype(int).tolist()
        self.processor = processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        token_ids, head_idx, attack_idx, mask = self.processor.tokenize_and_encode(self.texts[idx])
        return {
            'input_ids':        torch.tensor(token_ids,         dtype=torch.long),
            'head_token_idx':   torch.tensor(head_idx,          dtype=torch.long),
            'attack_token_idx': torch.tensor(attack_idx,        dtype=torch.long),
            'attention_mask':   torch.tensor(mask,              dtype=torch.long),
            'labels':           torch.tensor(self.labels[idx],  dtype=torch.long),
        }


def collate_fn(batch):
    max_h = max(len(b['head_token_idx'])   for b in batch)
    max_a = max(len(b['attack_token_idx']) for b in batch)
    ph, pa = [], []
    for b in batch:
        h = b['head_token_idx']
        ph.append(torch.cat([h, torch.zeros(max_h - len(h), dtype=torch.long)]))
        a = b['attack_token_idx']
        pa.append(torch.cat([a, torch.zeros(max_a - len(a), dtype=torch.long)]))
    return {
        'input_ids':        torch.stack([b['input_ids']       for b in batch]),
        'head_token_idx':   torch.stack(ph),
        'attack_token_idx': torch.stack(pa),
        'attention_mask':   torch.stack([b['attention_mask']  for b in batch]),
        'labels':           torch.stack([b['labels']          for b in batch]),
    }
```

- [ ] **Step 2: Add dataloader build + coverage stats cell** (NotebookEdit, `add_cell`, type=`code`)

```python
print("Building datasets (NER applied to train only)...")
train_ds = ViAmpleHateDataset(train_df, processor_train)
val_ds   = ViAmpleHateDataset(val_df,   processor_eval)
test_ds  = ViAmpleHateDataset(test_df,  processor_eval)

g = torch.Generator(); g.manual_seed(SEED)
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, generator=g,
    num_workers=0, pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)
print(f'Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}')

n_check = min(500, len(train_ds))
t_hits = a_hits = 0
for i in range(n_check):
    item = train_ds[i]
    if not (len(item['head_token_idx'])   == 1 and item['head_token_idx'][0].item()   == 0): t_hits += 1
    if not (len(item['attack_token_idx']) == 1 and item['attack_token_idx'][0].item() == 0): a_hits += 1
print(f"\nCoverage (first {n_check} train samples):")
print(f"  Target cue found : {t_hits}/{n_check} ({t_hits/n_check*100:.1f}%)")
print(f"  Attack cue found : {a_hits}/{n_check} ({a_hits/n_check*100:.1f}%)")
```

- [ ] **Step 3: Commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: add ViAmpleHateDataset, collate_fn, dataloaders with coverage stats"
```

---

### Task 7: Section 11 — Model: Relation Bank + Adaptive Gate

- [ ] **Step 1: Add section markdown cell** (NotebookEdit, `add_cell`, type=`markdown`)

```markdown
## 11. ViAmpleHatePhoBERT: Relation Bank + Instance-adaptive Gate

**Changes from baseline:**
- 3 HeadAttention modules: `head_attn_exp` (explicit targets), `head_attn_imp` (CLS), `head_attn_atk` (attack cues)
- `relation_proj`: Linear(768×3 → 768) fuses the three relation vectors
- `gate_proj`: Linear(768×2 → 1) produces per-instance scalar gate g ∈ (0,1)
- `forward` returns `(logits, z)` — `z` is the post-gate embedding used in contrastive loss
```

- [ ] **Step 2: Add HeadAttention cell** (NotebookEdit, `add_cell`, type=`code`)

```python
class HeadAttention(nn.Module):
    """HeadAttention from original AmpleHate (unchanged)."""
    def __init__(self, hidden_dim, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.softmax  = nn.Softmax(dim=-1)
        self.W_q = nn.Linear(hidden_dim, head_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, head_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, head_dim, bias=False)

    def forward(self, cls_embedding, head_token_embedding):
        Q_h = self.W_q(cls_embedding)
        K_h = self.W_k(head_token_embedding)
        V_h = self.W_v(cls_embedding)
        scores  = torch.matmul(Q_h, K_h.T) / (self.head_dim ** 0.5)
        weights = self.softmax(scores.float())
        return torch.matmul(weights, V_h)
```

- [ ] **Step 3: Add ViAmpleHatePhoBERT cell** (NotebookEdit, `add_cell`, type=`code`)

```python
class ViAmpleHatePhoBERT(nn.Module):
    def __init__(self, model_name, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.bert          = AutoModel.from_pretrained(model_name)
        self.hidden_dim    = hidden_dim
        self.head_attn_exp = HeadAttention(hidden_dim, HEAD_DIM)
        self.head_attn_imp = HeadAttention(hidden_dim, HEAD_DIM)
        self.head_attn_atk = HeadAttention(hidden_dim, HEAD_DIM)
        self.relation_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gate_proj     = nn.Linear(hidden_dim * 2, 1)
        self.dropout       = nn.Dropout(dropout)
        self.classifier    = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, input_ids, head_token_idx, attack_token_idx, attention_mask):
        hidden  = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_emb = hidden[:, 0, :]                                                # [B, H]

        exp_idx    = head_token_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        target_emb = torch.gather(hidden, 1, exp_idx)                            # [B, T, H]

        atk_idx    = attack_token_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        attack_emb = torch.gather(hidden, 1, atk_idx)                            # [B, A, H]

        r_exp = sum(self.head_attn_exp(cls_emb, target_emb[:, i, :]) for i in range(target_emb.shape[1]))
        r_imp = self.head_attn_imp(cls_emb, cls_emb)
        r_atk = sum(self.head_attn_atk(cls_emb, attack_emb[:, i, :]) for i in range(attack_emb.shape[1]))

        r_fused = self.relation_proj(torch.cat([r_exp, r_imp, r_atk], dim=-1))  # [B, H]
        g       = torch.sigmoid(self.gate_proj(torch.cat([cls_emb, r_fused], dim=-1)))  # [B, 1]
        z       = cls_emb + g * r_fused                                          # [B, H]

        return self.classifier(self.dropout(z)), z
```

- [ ] **Step 4: Add model init + sanity check cell** (NotebookEdit, `add_cell`, type=`code`)

```python
print('Loading ViAmpleHatePhoBERT...')
model     = ViAmpleHatePhoBERT(MODEL_NAME).to(DEVICE)
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total params    : {total:,}')
print(f'Trainable params: {trainable:,}')

with torch.no_grad():
    batch   = next(iter(val_loader))
    ids     = batch['input_ids'][:2].to(DEVICE)
    heads   = batch['head_token_idx'][:2].to(DEVICE)
    attacks = batch['attack_token_idx'][:2].to(DEVICE)
    mask    = batch['attention_mask'][:2].to(DEVICE)
    logits, z = model(ids, heads, attacks, mask)
    print(f'Forward pass OK: logits={tuple(logits.shape)}, z={tuple(z.shape)}')
```

- [ ] **Step 5: Commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: add HeadAttention, ViAmpleHatePhoBERT with relation bank and adaptive gate"
```

---

### Task 8: Section 12 — Loss, Optimizer, Scheduler

- [ ] **Step 1: Add ContrastiveLoss + CombinedLoss cell** (NotebookEdit, `add_cell`, type=`code`)

```python
class ContrastiveLossCosine(nn.Module):
    """Original AmpleHate contrastive loss (unchanged)."""
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        B           = embeddings.size(0)
        cosine_sim  = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        lbl         = labels.unsqueeze(1)
        diff_mask   = (lbl != lbl.T).float()
        pos_loss    = (1 - diff_mask) * (1 - cosine_sim)
        neg_loss    = diff_mask * F.relu(cosine_sim - self.margin)
        return (pos_loss + neg_loss).sum() / (B * (B - 1) + 1e-8)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=ALPHA_CL, label_smoothing=LABEL_SMOOTHING, class_weights=None):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.cl    = ContrastiveLossCosine(margin=0.5)
        self.alpha = alpha

    def forward(self, logits, z, labels):
        ce_loss = self.ce(logits, labels)
        cl_loss = self.cl(z, labels)
        return ce_loss + self.alpha * cl_loss, ce_loss.item(), cl_loss.item()
```

- [ ] **Step 2: Add optimizer + scheduler cell** (NotebookEdit, `add_cell`, type=`code`)

```python
label_counts  = train_df['label_id'].value_counts().sort_index().values
class_weights = torch.tensor(
    len(train_df) / (NUM_CLASSES * label_counts), dtype=torch.float32, device=DEVICE
)
print('Class weights:', class_weights.cpu().numpy().round(3))

criterion_train = CombinedLoss(alpha=ALPHA_CL, label_smoothing=LABEL_SMOOTHING, class_weights=class_weights)
criterion_eval  = CombinedLoss(alpha=0.0, label_smoothing=0.0, class_weights=None)

no_decay = ['bias', 'LayerNorm.weight']
bert_decay, bert_no_decay = [], []
for name, param in model.bert.named_parameters():
    if not param.requires_grad: continue
    (bert_no_decay if any(nd in name for nd in no_decay) else bert_decay).append(param)

head_params = (
    list(model.head_attn_exp.parameters()) +
    list(model.head_attn_imp.parameters()) +
    list(model.head_attn_atk.parameters()) +
    list(model.relation_proj.parameters()) +
    list(model.gate_proj.parameters()) +
    list(model.classifier.parameters())
)

optimizer = optim.AdamW([
    {'params': bert_decay,    'lr': LR,      'weight_decay': WEIGHT_DECAY},
    {'params': bert_no_decay, 'lr': LR,      'weight_decay': 0.0},
    {'params': head_params,   'lr': HEAD_LR, 'weight_decay': WEIGHT_DECAY},
])

total_steps  = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
scaler = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
print(f'Total steps: {total_steps} | Warmup: {warmup_steps} | Grad accum: {GRAD_ACCUM}')
```

- [ ] **Step 3: Commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: add ContrastiveLoss, CombinedLoss, optimizer, scheduler"
```

---

### Task 9: Section 13 — Training + Evaluation Loops

- [ ] **Step 1: Add best_threshold + train_epoch cell** (NotebookEdit, `add_cell`, type=`code`)

```python
def best_threshold(probs, labels, grid=np.linspace(0.05, 0.95, 19)):
    best_t, best_f1 = 0.5, 0.0
    for t in grid:
        f1 = f1_score(labels, (probs >= t).astype(int), average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    total_loss = total_correct = total_n = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        ids     = batch['input_ids'].to(DEVICE,        non_blocking=PIN_MEMORY)
        heads   = batch['head_token_idx'].to(DEVICE,   non_blocking=PIN_MEMORY)
        attacks = batch['attack_token_idx'].to(DEVICE, non_blocking=PIN_MEMORY)
        mask    = batch['attention_mask'].to(DEVICE,   non_blocking=PIN_MEMORY)
        y       = batch['labels'].to(DEVICE,           non_blocking=PIN_MEMORY)

        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            logits, z = model(ids, heads, attacks, mask)
            loss, _, _ = criterion(logits, z, y)
            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss    += loss.item() * GRAD_ACCUM * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n       += y.size(0)

    return total_loss / total_n, total_correct / total_n
```

- [ ] **Step 2: Add evaluate cell** (NotebookEdit, `add_cell`, type=`code`)

```python
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = total_n = 0
    all_probs, all_labels = [], []

    for batch in loader:
        ids     = batch['input_ids'].to(DEVICE,        non_blocking=PIN_MEMORY)
        heads   = batch['head_token_idx'].to(DEVICE,   non_blocking=PIN_MEMORY)
        attacks = batch['attack_token_idx'].to(DEVICE, non_blocking=PIN_MEMORY)
        mask    = batch['attention_mask'].to(DEVICE,   non_blocking=PIN_MEMORY)
        y       = batch['labels'].to(DEVICE,           non_blocking=PIN_MEMORY)

        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            logits, z = model(ids, heads, attacks, mask)
            loss, _, _ = criterion(logits, z, y)

        probs = torch.softmax(logits, dim=1)[:, 1]
        total_loss  += loss.item() * y.size(0)
        total_n     += y.size(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    t          = best_threshold(all_probs, all_labels)
    y_pred     = (all_probs >= t).astype(int)
    return (
        total_loss / total_n,
        accuracy_score(all_labels, y_pred),
        f1_score(all_labels, y_pred, average='macro', zero_division=0),
        float(t)
    )
```

---

### Task 10: Section 14 — Training Loop

- [ ] **Step 1: Add training loop cell** (NotebookEdit, `add_cell`, type=`code`)

```python
history = {k: [] for k in ['train_loss','val_loss','train_acc','val_acc','val_f1','threshold']}
best_f1, best_epoch, best_t_saved, patience_counter = -1.0, 0, 0.5, 0

hdr = f"{'Epoch':>6} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Val Loss':>8} | {'Val Acc':>7} | {'Val F1':>6} | {'Thresh':>6} | {'LR':>8} | {'Time':>6}"
print(hdr); print('-' * len(hdr))

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc              = train_epoch(model, train_loader, optimizer, scheduler, criterion_train, scaler)
    vl_loss, vl_acc, vl_f1, vl_t = evaluate(model, val_loader, criterion_eval)
    elapsed = time.time() - t0

    for k, v in zip(['train_loss','val_loss','train_acc','val_acc','val_f1','threshold'],
                    [tr_loss, vl_loss, tr_acc, vl_acc, vl_f1, vl_t]):
        history[k].append(v)

    flag = ''
    if vl_f1 > best_f1:
        best_f1, best_epoch, best_t_saved = vl_f1, epoch, vl_t
        torch.save({'model': model.state_dict(), 'threshold': vl_t}, CKPT_NAME)
        patience_counter = 0; flag = ' *saved*'
    else:
        patience_counter += 1

    print(f'{epoch:>6} | {tr_loss:>8.4f} | {tr_acc:>7.4f} | {vl_loss:>8.4f} | '
          f'{vl_acc:>7.4f} | {vl_f1:>6.4f} | {vl_t:>6.2f} | '
          f'{optimizer.param_groups[0]["lr"]:>8.2e} | {elapsed:>5.1f}s{flag}')

    if patience_counter >= PATIENCE:
        print(f'Early stopping at epoch {epoch}'); break

print(f'\nBest Val Macro-F1 = {best_f1:.4f} at epoch {best_epoch} (threshold={best_t_saved:.2f})')
```

- [ ] **Step 2: Commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: add training/eval loops and training execution cell"
```

---

### Task 11: Section 15-16 — Training Curves + Test Evaluation

- [ ] **Step 1: Add training curves cell** (NotebookEdit, `add_cell`, type=`code`)

```python
n = len(history['train_loss']); er = range(1, n + 1)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(er, history['train_loss'], 'b-o', label='Train')
axes[0].plot(er, history['val_loss'],   'r-o', label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')
axes[1].plot(er, history['train_acc'], 'b-o', label='Train')
axes[1].plot(er, history['val_acc'],   'r-o', label='Val')
axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].set_xlabel('Epoch')
axes[2].plot(er, history['val_f1'], 'g-o')
axes[2].axvline(best_epoch, color='red', linestyle='--', label=f'Best (epoch {best_epoch})')
axes[2].set_title('Val Macro F1'); axes[2].legend(); axes[2].set_xlabel('Epoch')
plt.suptitle(PLOT_TITLE, fontsize=14); plt.tight_layout()
plt.savefig('training_curves_viamplehate.png', dpi=150); plt.show()
```

- [ ] **Step 2: Add test evaluation cell** (NotebookEdit, `add_cell`, type=`code`)

```python
ckpt = torch.load(CKPT_NAME, map_location=DEVICE)
model.load_state_dict(ckpt['model'])
threshold = ckpt['threshold']
print(f'Loaded checkpoint: epoch {best_epoch}, threshold={threshold:.2f}')

@torch.no_grad()
def get_predictions(model, loader, threshold):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        ids     = batch['input_ids'].to(DEVICE,        non_blocking=PIN_MEMORY)
        heads   = batch['head_token_idx'].to(DEVICE,   non_blocking=PIN_MEMORY)
        attacks = batch['attack_token_idx'].to(DEVICE, non_blocking=PIN_MEMORY)
        mask    = batch['attention_mask'].to(DEVICE,   non_blocking=PIN_MEMORY)
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            logits, _ = model(ids, heads, attacks, mask)
        all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_labels.extend(batch['labels'].numpy())
    y_pred = (np.array(all_probs) >= threshold).astype(int)
    return np.array(all_labels), y_pred

y_true, y_pred = get_predictions(model, test_loader, threshold)
print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4))
```

- [ ] **Step 3: Add full metrics + confusion matrix cell** (NotebookEdit, `add_cell`, type=`code`)

```python
acc      = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro',    zero_division=0)
macro_p  = precision_score(y_true, y_pred, average='macro', zero_division=0)
macro_r  = recall_score(y_true, y_pred, average='macro',    zero_division=0)
hate_f1  = f1_score(y_true, y_pred, labels=[1], average='macro', zero_division=0)

print(f"Accuracy        : {acc:.4f}   (baseline: 0.9175, Δ={acc-0.9175:+.4f})")
print(f"Macro Precision : {macro_p:.4f}   (baseline: 0.7762, Δ={macro_p-0.7762:+.4f})")
print(f"Macro Recall    : {macro_r:.4f}   (baseline: 0.7823, Δ={macro_r-0.7823:+.4f})")
print(f"Macro F1        : {macro_f1:.4f}   (baseline: 0.7792, Δ={macro_f1-0.7792:+.4f})")
print(f"F1 (HATE)       : {hate_f1:.4f}   (baseline: 0.6045, Δ={hate_f1-0.6045:+.4f})")

cm = confusion_matrix(y_true, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm,     annot=True, fmt='d',   cmap='Blues',   ax=axes[0],
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Oranges', ax=axes[1],
            xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
for ax in axes: ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.suptitle(PLOT_TITLE + ' — Test Set', fontsize=14); plt.tight_layout()
plt.savefig('confusion_matrix_viamplehate.png', dpi=150); plt.show()
```

---

### Task 12: Section 17 — Ablation: Fixed-e vs Adaptive Gate

- [ ] **Step 1: Add ablation markdown cell** (NotebookEdit, `add_cell`, type=`markdown`)

```markdown
## 17. Ablation: Fixed-e vs Instance-adaptive Gate

Trains 5 fixed-e variants (3 epochs each) to isolate the contribution of the adaptive gate.
Each fixed-e model uses the same relation bank as the proposed model, only differing in the injection mechanism.
```

- [ ] **Step 2: Add ViAmpleHateFixedE cell** (NotebookEdit, `add_cell`, type=`code`)

```python
class ViAmpleHateFixedE(nn.Module):
    """Ablation: same relation bank as ViAmpleHatePhoBERT, but fixed scalar e instead of adaptive gate."""
    def __init__(self, model_name, hidden_dim=HIDDEN_DIM, e=1.0, dropout=DROPOUT):
        super().__init__()
        self.bert          = AutoModel.from_pretrained(model_name)
        self.hidden_dim    = hidden_dim
        self.e             = e
        self.head_attn_exp = HeadAttention(hidden_dim, HEAD_DIM)
        self.head_attn_imp = HeadAttention(hidden_dim, HEAD_DIM)
        self.head_attn_atk = HeadAttention(hidden_dim, HEAD_DIM)
        self.relation_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout       = nn.Dropout(dropout)
        self.classifier    = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, input_ids, head_token_idx, attack_token_idx, attention_mask):
        hidden  = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_emb = hidden[:, 0, :]

        exp_idx    = head_token_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        target_emb = torch.gather(hidden, 1, exp_idx)
        atk_idx    = attack_token_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        attack_emb = torch.gather(hidden, 1, atk_idx)

        r_exp = sum(self.head_attn_exp(cls_emb, target_emb[:, i, :]) for i in range(target_emb.shape[1]))
        r_imp = self.head_attn_imp(cls_emb, cls_emb)
        r_atk = sum(self.head_attn_atk(cls_emb, attack_emb[:, i, :]) for i in range(attack_emb.shape[1]))

        r_fused = self.relation_proj(torch.cat([r_exp, r_imp, r_atk], dim=-1))
        z       = cls_emb + self.e * r_fused
        return self.classifier(self.dropout(z)), z
```

- [ ] **Step 3: Add ablation grid-search cell** (NotebookEdit, `add_cell`, type=`code`)

```python
E_GRID = [0.5, 0.75, 1.0, 1.25, 1.5]
ablation_results = []

for e_val in E_GRID:
    print(f'\n--- Ablation: fixed e={e_val} ---')
    abl_model = ViAmpleHateFixedE(MODEL_NAME, e=e_val).to(DEVICE)
    abl_opt   = optim.AdamW(abl_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    abl_sched = get_linear_schedule_with_warmup(
        abl_opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    abl_scaler    = torch.amp.GradScaler('cuda', enabled=DEVICE.type == 'cuda')
    abl_criterion = CombinedLoss(alpha=ALPHA_CL, label_smoothing=LABEL_SMOOTHING, class_weights=class_weights)
    abl_eval_crit = CombinedLoss(alpha=0.0, label_smoothing=0.0, class_weights=None)

    best_abl_f1 = -1.0
    for ep in range(1, 4):  # 3 epochs for ablation comparison
        train_epoch(abl_model, train_loader, abl_opt, abl_sched, abl_criterion, abl_scaler)
        _, _, vf1, _ = evaluate(abl_model, val_loader, abl_eval_crit)
        print(f'  epoch {ep}: val_f1={vf1:.4f}')
        if vf1 > best_abl_f1:
            best_abl_f1 = vf1
            ckpt_path = f'abl_e{str(e_val).replace(".", "p")}.pt'
            torch.save(abl_model.state_dict(), ckpt_path)

    abl_model.load_state_dict(torch.load(ckpt_path))
    _, _, val_f1, val_t = evaluate(abl_model, val_loader, abl_eval_crit)
    abl_true, abl_pred  = get_predictions(abl_model, test_loader, val_t)
    t_mf1 = f1_score(abl_true, abl_pred, average='macro', zero_division=0)
    t_hf1 = f1_score(abl_true, abl_pred, labels=[1], average='macro', zero_division=0)
    ablation_results.append({'injection': f'fixed e={e_val}', 'val_f1': round(val_f1,4),
                              'test_macro_f1': round(t_mf1,4), 'test_hate_f1': round(t_hf1,4)})

# Add proposed adaptive-gate result (from full training above)
ablation_results.append({'injection': 'adaptive gate (proposed)', 'val_f1': round(best_f1,4),
                          'test_macro_f1': round(macro_f1,4), 'test_hate_f1': round(hate_f1,4)})

abl_df = pd.DataFrame(ablation_results)
print('\n=== Ablation Summary ===')
print(abl_df.to_string(index=False))
```

---

### Task 13: Section 18 — Inference Demo + Save Config

- [ ] **Step 1: Add inference demo cell** (NotebookEdit, `add_cell`, type=`code`)

```python
@torch.no_grad()
def predict(texts):
    model.eval()
    results = []
    for text in texts:
        processed = preprocess(text)
        token_ids, head_idx, attack_idx, attn_mask = processor_eval.tokenize_and_encode(processed)

        ids     = torch.tensor([token_ids],   dtype=torch.long, device=DEVICE)
        heads   = torch.tensor([head_idx],    dtype=torch.long, device=DEVICE)
        attacks = torch.tensor([attack_idx],  dtype=torch.long, device=DEVICE)
        attn    = torch.tensor([attn_mask],   dtype=torch.long, device=DEVICE)

        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            logits, _ = model(ids, heads, attacks, attn)

        probs   = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_id = int(float(probs[1]) >= threshold)
        results.append({
            'text':  text,
            'label': LABEL_NAMES[pred_id],
            'scores': {LABEL_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)},
        })
    return results


TEST_CASES = [
    ('Hôm nay trời đẹp quá, đi chơi thôi!',                  'NON-HATE'),
    ('Tụi nó toàn nói nhảm, đúng là quá toxic',               'NON-HATE'),
    ('Đồ ngu, câm miệng lại đi mày',                          'NON-HATE'),
    ('Tao ghét cái loại người như mày, xéo đi cho khuất mắt', 'HATE'),
    ('Cảm ơn bạn đã giúp đỡ mình nhé!',                       'NON-HATE'),
    ('nguyên cả cái tỉnh này không được khôn lắm',            'HATE'),
    ('Bọn đó toàn ăn bám xã hội 🙃',                          'HATE'),
]
correct = 0
for text, expected in TEST_CASES:
    r = predict([text])[0]
    match = 'OK' if r['label'] == expected else 'WRONG'
    correct += int(r['label'] == expected)
    print(f'Input : {text}')
    print(f'Pred  : {r["label"]:<10} {match:<6} {r["scores"]}')
    print()
print(f'Summary: {correct}/{len(TEST_CASES)} ({correct/len(TEST_CASES)*100:.0f}%)')
```

- [ ] **Step 2: Add save config cell** (NotebookEdit, `add_cell`, type=`code`)

```python
os.makedirs('outputs', exist_ok=True)
config = {
    'notebook'             : 'vihsd-viamplehate-phobert-proposed',
    'method'               : 'AmpleHate-Vi++ (Viet NER + target/attack cues + adaptive gate)',
    'encoder'              : MODEL_NAME,
    'ner_model'            : NER_MODEL,
    'max_len'              : MAX_LEN,
    'hidden_dim'           : HIDDEN_DIM,
    'grad_accum'           : GRAD_ACCUM,
    'alpha_cl'             : ALPHA_CL,
    'dropout'              : DROPOUT,
    'num_classes'          : NUM_CLASSES,
    'label_names'          : LABEL_NAMES,
    'lr_encoder'           : float(LR),
    'lr_head'              : float(HEAD_LR),
    'best_epoch'           : int(best_epoch),
    'best_val_f1'          : round(float(best_f1), 4),
    'best_threshold'       : round(float(threshold), 2),
    'test_accuracy'        : round(float(acc), 4),
    'test_macro_f1'        : round(float(macro_f1), 4),
    'test_macro_p'         : round(float(macro_p), 4),
    'test_macro_r'         : round(float(macro_r), 4),
    'test_f1_hate'         : round(float(hate_f1), 4),
    'baseline_macro_f1'    : 0.7792,
    'baseline_hate_f1'     : 0.6045,
    'delta_macro_f1'       : round(float(macro_f1) - 0.7792, 4),
    'delta_hate_f1'        : round(float(hate_f1)  - 0.6045, 4),
}
with open('outputs/viamplehate_config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(json.dumps(config, indent=2, ensure_ascii=False))
```

- [ ] **Step 3: Final commit**

```bash
git add "notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/vihsd-proposed-viamplehate-phobert.ipynb"
git commit -m "feat: complete AmpleHate-Vi++ proposed notebook (ViHSD)

- Vietnamese NER + target/attack cue lexicons (coverage 0.09% → ~45%)
- Relation Bank: 3 HeadAttention modules (r_exp, r_imp, r_atk)
- Instance-adaptive gate replacing fixed scalar e
- CE + ContrastiveLoss (alpha=0.1), max_length=256, grad_accum=2
- Ablation section: fixed-e grid vs adaptive gate

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All spec sections covered — P1 emoji preprocessing (Task 3), P2 multi-signal target cue (Tasks 4-5), P4 relation bank (Task 7), P5 adaptive gate (Task 7), CE+CL loss (Task 8), ablation fixed-e (Task 12), coverage stats (Task 6)
- [x] **No placeholders:** All code blocks are complete and runnable
- [x] **Type consistency:** `model(ids, heads, attacks, mask)` signature used in Tasks 7, 9, 11, 12, 13; `CombinedLoss.forward(logits, z, labels)` consistent across Tasks 8-10; `processor_eval` / `processor_train` defined in Task 5 and used in Tasks 6, 13
- [x] **Scope:** Single notebook output, self-contained tasks
