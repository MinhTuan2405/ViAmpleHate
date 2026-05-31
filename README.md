# ViAmpleHate

**A Proposed AmpleHate- and PhoBERT-based Approach for Vietnamese Hate Speech Detection**

ViAmpleHate is a Vietnamese adaptation of the **AmpleHate** model for hate speech detection on Vietnamese social media, built on top of **PhoBERT**. The project addresses a core limitation: the original AmpleHate relies on English NER and therefore almost never finds a *target* in Vietnamese (only ~0.09% of comments), collapsing the model into a plain sentence classifier. ViAmpleHate replaces English NER with Vietnamese NER plus hand-curated *cue banks*, raising target coverage to ~19–45% and clearly improving F1 on the minority HATE class.

---

## Team

**Students:**

| Student ID | Full name |
|----------|------------------------|
| 23521687 | Tran Nguyen Duc Trung |
| 23521718 | Nguyen Ha Minh Tuan |
| 23520881 | Nguyen Gia Cat Long |
| 23521747 | Tran Phan Thanh Tung |
| 23521741 | Mo Van Tung |

**Supervisor:** M.Sc. Huynh Van Tin

University of Information Technology, VNU-HCM (UIT, VNU-HCM).

---

## Table of Contents

- [Motivation & Contributions](#motivation--contributions)
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Installation & Environment](#installation--environment)
- [How to Run](#how-to-run)
- [Results](#results)
- [Detailed Results (classification reports)](#detailed-results-classification-reports)
- [Limitations](#limitations)
- [Paper](#paper)
- [References](#references)

---

## Motivation & Contributions

Vietnamese hate speech detection is hard because hateful intent rarely sits in an explicit named entity; instead it hides in informal group references, slang, sarcasm, and indirect insinuation. Target-aware models such as AmpleHate show that attending to the relationship between an utterance and the *target it addresses* helps detect even implicit hate. However, AmpleHate's original pipeline relies on English NER and English-oriented target categories, so it transfers poorly to Vietnamese: English NER finds a usable target in only **~0.09%** of training comments.

Key contributions of ViAmpleHate:

1. **Vietnamese target extraction.** Replace English NER with Vietnamese NER plus a hand-curated *target-cue* bank (pronouns, kinship/group nouns, regional and demographic labels, informal address forms) → raising target coverage from ~0.09% to **~18.8–20.0%** (ViHSD) and **~43–45%** (VOZ-HSD).
2. **Separate target and attack channels.** Add a dedicated *attack-cue* bank for hostile predicates, modeling target and attack as distinct signals through a three-channel **relation-bank attention**.
3. **Adaptive fusion.** Replace AmpleHate's fixed-scalar injection with an **instance-adaptive gate**; plus an implementation-level correction to the batch-level attention computation (reported as an *implementation note*).
4. **Training objective.** Combine weighted cross-entropy with a contrastive loss to sharpen HATE/NON-HATE separation under class imbalance.
5. **Empirical study.** Evaluate on ViHSD and VOZ-HSD against 5 baselines, with target-coverage analysis and error analysis.

> Broader message: target-awareness is a genuinely useful inductive bias for hate speech detection, **but only when adapted to the surface forms of the target language**, rather than transferred verbatim from English.

---

## Method Overview

Given a Vietnamese comment `x`, the model predicts a binary label `y ∈ {NON-HATE, HATE}`.

```
Comment
  → normalize (teencode, emoji→tag) → word-segment (required by PhoBERT) → tokenize (max_len=256)
  → PhoBERT-base (768-d) → h_0 = [CLS]
  → Vietnamese NER + target-cue  ⇒  T_x   (empty ⇒ fallback to [CLS])
  → attack-cue                   ⇒  A_x   (empty ⇒ fallback to [CLS])
  → Relation-bank attention (3 channels):
        r_exp = HeadAttn(h_0, H[T_x])     # explicit target
        r_imp = HeadAttn(h_0, h_0)        # implicit context
        r_atk = HeadAttn(h_0, H[A_x])     # attack
        r     = W_r [r_exp ; r_imp ; r_atk] + b_r
  → Instance-adaptive gate:
        g = σ(W_g [h_0 ; r] + b_g)
        z = h_0 + g · r
  → Dropout → Linear → logits
Loss:  L = L_CE (weighted, label-smoothed) + α · L_CL   (α = 0.1)
Inference:  threshold t* chosen on validation by macro-F1, fixed for test
```

**Cue banks** (hand-curated, seeded from the training split, matched by full token-sequence matching):
- **Target cues (16):** referential expressions (not hateful by themselves).
- **Attack cues (24):** hostile predicates (insults, dehumanization, threats).
- The two banks are kept **separate** on purpose: a target without an attack is usually not hate; an attack without a group target is usually mere offensiveness.

Main configuration: `vinai/phobert-base`, NER `NlpHUST/ner-vietnamese-electra-base`, LR 2e-5 (encoder) / 5e-5 (head), dropout 0.1, effective batch 32 (16 × grad-accum 2), up to 8 epochs (best by val macro-F1), seed 42 (single run).

---

## Repository Structure

```
ViAmpleHate/
├── README.md                       # this document
├── app/                            # Streamlit demo app
│   ├── app.py
│   ├── model_runtime.py
│   ├── requirements.txt
│   └── README.md
├── dataset/                        # data download guide + label mapping
│   └── README.md
├── notebooks/
│   └── models/
│       ├── baselines/              # 5 baselines × 2 datasets (.ipynb + output/)
│       │   ├── ViHSD - Baseline TF-IDF LR_SVM
│       │   ├── ViHSD - Baseline BiLSTM_FasttextVi
│       │   ├── ViHSD - Baseline PhoBERT_CNN
│       │   ├── ViHSD - Baseline AmpleHate_PhoBERT
│       │   └── VOZ-HSD - Baseline ...   (similar)
│       └── proposed/               # proposed ViAmpleHate
│           ├── ViHSD - Proposed ViAmpleHate_PhoBERT
│           └── VOZ-HSD - Proposed ViAmpleHate_PhoBERT
├── paper/
│   ├── main/
│   │   ├── latex/v_review/         # LaTeX ACL version (acl_latex.tex, custom.bib)
│   │   └── markdown/v_review/      # parallel markdown version
│   ├── related_work/
│   └── slide/                      # slide content (plain text for Canva)
└── docs/                           # plans/specs/illustrative figures
```

---

## Data

See the detailed download guide at [`dataset/README.md`](dataset/README.md).

- **ViHSD** — [`uitnlp/vihsd`](https://huggingface.co/datasets/sonlam1102/vihsd) (3 labels: CLEAN/OFFENSIVE/HATE).
- **VOZ-HSD** — [`tarudesu/VOZ-HSD`](https://huggingface.co/datasets/tarudesu/VOZ-HSD) (2 labels; labels generated by an AI classifier, ViSoBERT-HSD, not human-annotated).
- **fastText vi** — `cc.vi.300.vec.gz` (used by the BiLSTM baseline).

**Binary relabeling** (relabel only — no examples added or removed):

| Original label | ViHSD → | VOZ-HSD → |
|-----------|------------|-------------|
| CLEAN | NON-HATE | NON-HATE |
| OFFENSIVE | NON-HATE | — |
| HATE | HATE | HATE |

Distribution after relabeling (HATE is ~10% throughout — so **macro-F1 / HATE-F1 matter more than accuracy**):

| Dataset | Split | NON-HATE | HATE | Total |
|---------|-------|---------:|-----:|------:|
| ViHSD | Train | 21,492 | 2,556 | 24,048 |
| ViHSD | Dev | 2,402 | 270 | 2,672 |
| ViHSD | Test | 5,992 | 688 | 6,680 |
| VOZ | Train | 26,993 | 3,007 | 30,000 |
| VOZ | Dev | 4,520 | 480 | 5,000 |
| VOZ | Test | 4,487 | 513 | 5,000 |

---

## Installation & Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
```

- PhoBERT models need to download/cache `vinai/phobert-base`; the proposed model additionally needs `NlpHUST/ner-vietnamese-electra-base`. The first run requires internet access (or a pre-populated HuggingFace cache).
- If PhoBERT reports `upgrade torch to at least v2.6`: run `pip install --upgrade -r app\requirements.txt` and restart.
- To use CUDA on Windows: install PyTorch following the official guide at https://pytorch.org/get-started/locally/ **before** running `pip install -r app\requirements.txt`.

---

## How to Run

### 1. Notebooks (training & evaluation)

Each model is a self-contained notebook under `notebooks/models/`. Open the relevant notebook and run the cells in order; artifacts (model, confusion matrix, training curves) are saved to the `output/` folder next to the notebook.

- Baselines: `notebooks/models/baselines/<DATASET> - Baseline <MODEL>/`
- Proposed: `notebooks/models/proposed/<DATASET> - Proposed ViAmpleHate_PhoBERT/`

### 2. Demo app (Streamlit)

Details at [`app/README.md`](app/README.md).

```powershell
streamlit run app\app.py
```

The app takes a Vietnamese sentence and runs it through all baselines plus the proposed ViAmpleHate, showing the predicted label for each model. On GPUs with small VRAM (≈4GB), the app runs PyTorch models sequentially and offloads to CPU after each inference to avoid CUDA OOM.

---

## Results

All numbers below come from a **single run (seed 42)** on the test set. The comparison is between the **full proposed system** and **each baseline at its own intended configuration** (not an isolated ablation of individual components).

### ViHSD (test)

| Model | Acc. | Macro-F1 | HATE-F1 |
|-------|-----:|---------:|--------:|
| TF-IDF + LR | 0.8910 | 0.7393 | 0.5404 |
| TF-IDF + SVM | 0.9126 | 0.7131 | 0.4739 |
| BiLSTM + fastText | 0.8454 | 0.7072 | 0.5060 |
| PhoBERT-CNN | 0.8945 | 0.7571 | 0.5745 |
| AmpleHate (baseline) | 0.9175 | 0.7792 | 0.6045 |
| **ViAmpleHate** | **0.9205** | **0.7819** | **0.6081** |
| *Δ vs. baseline* | *+.0030* | *+.0027* | *+.0036* |

### VOZ-HSD (test)

| Model | Acc. | Macro-F1 | HATE-F1 |
|-------|-----:|---------:|--------:|
| TF-IDF + LR | 0.9453 | 0.7745 | 0.5783 |
| TF-IDF + SVM | 0.9641 | 0.7831 | 0.5850 |
| BiLSTM + fastText | 0.8650 | 0.6712 | 0.4187 |
| PhoBERT-CNN | 0.9623 | 0.8150 | 0.6500 |
| AmpleHate (baseline) | 0.9643 | 0.8185 | 0.6557 |
| **ViAmpleHate** | 0.9420 | **0.8371** | **0.7065** |
| *Δ vs. baseline* | *−.0223* | *+.0186* | *+.0508* |

**Observations:** The largest and clearest gains are on the minority HATE class on VOZ-HSD (**HATE-F1 +0.0508**). On ViHSD the improvement is small (+0.0036 HATE-F1), so it is reported as **preliminary** (single run, no significance test). Accuracy *drops* on VOZ while macro-F1/HATE-F1 *rise* — illustrating why accuracy is the wrong headline metric under imbalance.

**Target coverage:** English NER ~0.09% → Vietnamese NER + cues: 20.0%/18.8% (ViHSD train/val), 45.2%/43.0% (VOZ train/val). The dataset with higher coverage (VOZ) is also where ViAmpleHate gains the most — a suggestive correlation (correlational, not proven causal).

---

## Detailed Results (classification reports)

> This section preserves the full classification reports recorded from the notebooks, for cross-checking and reproduction.

### ViHSD dataset

```
TF-IDF + SVM (Tuned) — Test
  Accuracy   : 0.9126
  F1 Macro   : 0.7131
  F1 Weighted: 0.9030
  F1   NON-HATE: 0.9523
  F1       HATE: 0.4739

              precision    recall  f1-score   support

    NON-HATE       0.93      0.97      0.95      5992
        HATE       0.62      0.38      0.47       688

    accuracy                           0.91      6680
   macro avg       0.78      0.68      0.71      6680
weighted avg       0.90      0.91      0.90      6680



TF-IDF + LR (Tuned) — Test
  Accuracy   : 0.8910
  F1 Macro   : 0.7393
  F1 Weighted: 0.8972
  F1   NON-HATE: 0.9382
  F1       HATE: 0.5404

              precision    recall  f1-score   support

    NON-HATE       0.96      0.92      0.94      5992
        HATE       0.48      0.62      0.54       688

    accuracy                           0.89      6680
   macro avg       0.72      0.77      0.74      6680
weighted avg       0.91      0.89      0.90      6680

BiLSTM - FastTextVi
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9699    0.8541    0.9083      5992
        HATE     0.3770    0.7689    0.5060       688

    accuracy                         0.8454      6680
   macro avg     0.6735    0.8115    0.7072      6680
weighted avg     0.9088    0.8454    0.8669      6680

PhoBert CNN
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9629    0.9177    0.9398      5992
        HATE     0.4912    0.6919    0.5745       688

    accuracy                         0.8945      6680
   macro avg     0.7271    0.8048    0.7571      6680
weighted avg     0.9143    0.8945    0.9021      6680

PhoBert AmpleHate Origin
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9553    0.9526    0.9540      5992
        HATE     0.5972    0.6119    0.6045       688

    accuracy                         0.9175      6680
   macro avg     0.7762    0.7823    0.7792      6680
weighted avg     0.9184    0.9175    0.9180      6680

PhoBert ViAmpleHate 
              precision    recall  f1-score   support

    NON-HATE     0.9541    0.9574    0.9558      5992
        HATE     0.6177    0.5988    0.6081       688

    accuracy                         0.9205      6680
   macro avg     0.7859    0.7781    0.7819      6680
weighted avg     0.9195    0.9205    0.9200      6680

--------------------------
Summary
Accuracy        : 0.9205   (baseline: 0.9175, Δ=+0.0030)
Macro Precision : 0.7859   (baseline: 0.7762, Δ=+0.0097)
Macro Recall    : 0.7781   (baseline: 0.7823, Δ=-0.0042)
Macro F1        : 0.7819   (baseline: 0.7792, Δ=+0.0027)
F1 (HATE)       : 0.6081   (baseline: 0.6045, Δ=+0.0036)

```

### VOZ-HSD dataset

```
TF-IDF + SVM (Tuned) — Test
  Accuracy   : 0.9641
  F1 Macro   : 0.7831
  F1 Weighted: 0.9609
  F1   NON-HATE: 0.9812
  F1       HATE: 0.5850

              precision    recall  f1-score   support

    NON-HATE       0.97      0.99      0.98      9486
        HATE       0.72      0.49      0.58       514

    accuracy                           0.96     10000
   macro avg       0.85      0.74      0.78     10000
weighted avg       0.96      0.96      0.96     10000

TF-IDF + LR (Tuned) — Test
  Accuracy   : 0.9453
  F1 Macro   : 0.7745
  F1 Weighted: 0.9506
  F1   NON-HATE: 0.9708
  F1       HATE: 0.5783

              precision    recall  f1-score   support

    NON-HATE       0.98      0.96      0.97      9486
        HATE       0.48      0.73      0.58       514

    accuracy                           0.95     10000
   macro avg       0.73      0.84      0.77     10000
weighted avg       0.96      0.95      0.95     10000

BiLSTM - FastTextVi
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9940    0.8626    0.9237     18929
        HATE     0.2721    0.9076    0.4187      1071

    accuracy                         0.8650     20000
   macro avg     0.6330    0.8851    0.6712     20000
weighted avg     0.9553    0.8650    0.8966     20000


PhoBert CNN
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9826    0.9775    0.9801      9486
        HATE     0.6217    0.6809    0.6500       514

    accuracy                         0.9623     10000
   macro avg     0.8021    0.8292    0.8150     10000
weighted avg     0.9641    0.9623    0.9631     10000

PhoBert AmpleHate Origin
Classification Report — Test Set
              precision    recall  f1-score   support

    NON-HATE     0.9816    0.9807    0.9812      9486
        HATE     0.6501    0.6615    0.6557       514

    accuracy                         0.9643     10000
   macro avg     0.8159    0.8211    0.8185     10000
weighted avg     0.9646    0.9643    0.9644     10000

PhoBert ViAmpleHate 
              precision    recall  f1-score   support

    NON-HATE     0.9638    0.9719    0.9678      4487
        HATE     0.7347    0.6803    0.7065       513

    accuracy                         0.9420      5000
   macro avg     0.8492    0.8261    0.8371      5000
weighted avg     0.9403    0.9420    0.9410      5000

--------------------------
Summary
Accuracy        : 0.9420   (baseline: 0.9643, Delta=-0.0223)
Macro Precision : 0.8492   (baseline: 0.8159, Delta=+0.0333)
Macro Recall    : 0.8261   (baseline: 0.8211, Delta=+0.0050)
Macro F1        : 0.8371   (baseline: 0.8185, Delta=+0.0186)
F1 (HATE)       : 0.7065   (baseline: 0.6557, Delta=+0.0508)
```

---

## Limitations

- The cue banks are partly hand-built and cannot cover the full, fast-changing space of Vietnamese slang, spelling variants, and creative profanity → this bounds recall on implicit or novel hate.
- Performance depends on upstream Vietnamese NER / word segmentation / tokenization; misalignment can misplace cues.
- The decision threshold is tuned on validation and may not be optimal under distribution shift.
- **Single run (seed 42):** no multi-seed variance, no significance test, and no per-component ablation are reported → the small ViHSD margin is not established.
- Comparison is against our own re-implementations, not published (SOTA) ViHSD results.
- Only the binary NON-HATE/HATE formulation is addressed; multi-target or graded hatred is not yet handled.

**Ethics:** Hate speech detection is dual-use; system outputs should be treated as *decision support* for human moderators, not automated enforcement. Datasets are used under their public research terms; the paper uses constructed/masked examples rather than reproducing real slurs.

---

## Paper

- **LaTeX (ACL):** [`paper/main/latex/v_review/acl_latex.tex`](paper/main/latex/v_review/acl_latex.tex) — compile with pdfLaTeX (XeLaTeX/LuaLaTeX for full Vietnamese diacritics). Bibliography: `custom.bib`.
- **Parallel markdown:** [`paper/main/markdown/v_review/`](paper/main/markdown/v_review/)
- **Revision notes:** [`paper/main/latex/v_review/REVISION_NOTES.md`](paper/main/latex/v_review/REVISION_NOTES.md)
- **Slides (plain text for Canva):** [`paper/slide/slides_en.txt`](paper/slide/slides_en.txt)

---

## References

- **AmpleHate** — the original target-aware model.
- **PhoBERT** — Nguyen & Nguyen (2020), the foundational Vietnamese encoder (`vinai/phobert-base`).
- **ViHSD** — Vietnamese Hate Speech Detection dataset (`uitnlp/vihsd`).
- **VOZ-HSD** — `tarudesu/VOZ-HSD`.
- **Vietnamese NER** — `NlpHUST/ner-vietnamese-electra-base`.
- **fastText** — Grave et al. (2018), *Learning Word Vectors for 157 Languages*, LREC 2018.

> Full citations with venue/year metadata are in `paper/main/latex/v_review/custom.bib`.

---

*Course project — University of Information Technology, VNU-HCM.*
