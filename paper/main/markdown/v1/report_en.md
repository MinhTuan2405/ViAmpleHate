# ViAmpleHate: A Proposed AmpleHate- and PhoBERT-based Approach for Vietnamese Hate Speech Detection

> **Purpose of this file.** This is a content blueprint for writing an 8–10 page LaTeX paper following the **ACL Conference** template (Overleaf: *Association for Computational Linguistics — ACL Conference*, `acl_latex.tex` + `acl.sty`).
> Each section below maps to a `\section{}` / `\subsection{}` in LaTeX. Equations are written in LaTeX syntax so they can be pasted directly into `equation`/`align`. Tables are in Markdown and should be converted to `table` + `booktabs` (`\toprule`/`\midrule`/`\bottomrule`).
> Positions that need figures are marked with **`📌 [INSERT FIGURE]`** blocks — stating which figure to insert, where, and from which file (or whether it must be drawn new).
>
> **Length note (ACL two columns):** Intro ~1 pg · Related Work ~0.75 pg · Dataset ~1 pg · Methodology ~2 pg (with architecture figure) · Experiments ~1.5 pg · Results & Error Analysis ~2 pg (with tables + confusion matrices) · Conclusion ~0.5 pg. Total ~8.5–9.5 pg + references.

---

## Abstract

*(~150–200 words, write last. Suggested content:)*

Hate speech detection on Vietnamese social media is challenging because hateful intent is often expressed through informal group references, slang, and implicit cues rather than explicit named entities. Target-aware models such as **AmpleHate** improve detection by attending to the relationship between a sentence and its hate target, but the original pipeline relies on English named-entity recognition (NER) and English-centric target categories, which transfer poorly to Vietnamese. We propose **ViAmpleHate**, a Vietnamese adaptation of AmpleHate built on **PhoBERT**. ViAmpleHate replaces English NER with Vietnamese NER, adds Vietnamese **target-cue** and **attack-cue** banks, models three relation channels (explicit target, implicit context, attack) through a **relation-bank attention**, corrects a batch-level attention contamination bug, and injects relation evidence through an **instance-adaptive gate** instead of a fixed scalar. Training combines weighted cross-entropy with a contrastive objective. On **ViHSD** and **VOZ-HSD** (binary NON-HATE/HATE setting), ViAmpleHate improves macro-F1 and the minority HATE-class F1 over a faithful AmpleHate baseline and over TF-IDF, BiLSTM, and PhoBERT-CNN baselines, confirming that target-aware modeling must be linguistically adapted to Vietnamese.

> 📌 **[INSERT FIGURE — optional]** A small *teaser figure* may be added in the right column of page 1 illustrating the idea: a Vietnamese sentence → split into *target cue* / *attack cue* → attention → HATE prediction. (Must be **drawn new**, e.g. in PowerPoint/draw.io.) If space is tight, skip it and keep the architecture figure in Section 4.

---

## 1. Introduction

*(~1 page. The following points should be developed into prose:)*

- **Context & motivation.** Vietnamese social media is growing fast; hate speech harms individuals and communities. Automatic detection is needed for large-scale content moderation.
- **Vietnamese-specific challenges.** (1) Noisy text: teencode, abbreviations, repeated characters, emojis, misspellings. (2) Targets of attack are often expressed through *pronouns, group nouns, informal forms of address*, not named entities. (3) Severe class imbalance: HATE is the minority class.
- **Target-aware approach & the gap.** AmpleHate shows that modeling *the relationship between a sentence and its intended target* helps detect even implicit hate. But the original AmpleHate uses **English NER** ⇒ on Vietnamese it almost never finds valid targets (measured: only **21/24,048 ≈ 0.09%** of train samples have a target from English NER), causing the model to degenerate into a plain PhoBERT classifier relying on `[CLS]`.
- **Contributions.** List as bold bullets:
  1. **ViAmpleHate** — a Vietnamese adaptation of AmpleHate on PhoBERT, replacing English NER with **Vietnamese NER + a target-cue bank** ⇒ raising target coverage from ~0.09% to ~18.8–20.0%.
  2. **Signal-channel separation**: separately modeling *target cues* (who is being discussed) and *attack cues* (is it being attacked) through a three-channel **relation-bank attention** (explicit target / implicit context / attack).
  3. **Fixing the batch attention bug** (from batch-level `matmul(Q, Kᵀ)` that mixes samples to **batched `bmm` + masking** per sample) and replacing **fixed-scalar injection** with an **instance-adaptive gate**.
  4. **Training objective** combining weighted cross-entropy + contrastive loss to improve class separation under imbalance.
  5. Evaluation on **ViHSD** and **VOZ-HSD** (binary), compared against 5 baselines; improvements in macro-F1 and HATE-F1.
- **Closing sentence.** Emphasize: target-awareness is useful, but *it must be adapted to Vietnamese linguistic characteristics* to be effective.

---

## 2. Related Work

### 2.1 AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection

*(~0.4 page)*

- Summarize AmpleHate's idea: detecting implicit hate by **amplifying attention** between the sentence representation (`[CLS]`) and potential **targets** extracted by NER; the target-aware relation vector is **injected** into the sentence representation before classification.
- Mechanism: for a target token, compute HeadAttention $r = \text{softmax}(QK^\top/\sqrt d)\,V$, then $z = h_0 + e\cdot r$ with a fixed coefficient $e$.
- **Limitations when ported to Vietnamese:** depends on English NER and English-style target categories; fixed injection; only one relation channel (target), no *attack* modeling.
- State clearly: this work **inherits AmpleHate's target-aware idea** but redesigns it for Vietnamese.

### 2.2 ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts

*(~0.35 page)*

- Introduce ViTHSD: a **target-grounded** hate speech dataset/approach for Vietnamese social media; labels degree of hatred per target.
- Connection: ViTHSD reinforces the claim that **the target is a core signal** of Vietnamese hate speech ⇒ supports ViAmpleHate's target-aware motivation.
- Difference: ViAmpleHate does not require span-level target labels; instead it uses **cue banks + NER** to locate target/attack positions *without span supervision*.

*(Optional ~0.1 pg: briefly mention Vietnamese baselines: PhoBERT (Nguyen & Nguyen, 2020), PhoBERT-CNN, BiLSTM + fastText, TF-IDF — to lead into the baseline section in §5.)*

---

## 3. Dataset

### 3.1 ViHSD — Vietnamese Hate Speech Detection dataset

*(~0.3 pg)*

- ViHSD: a Vietnamese hate speech dataset of social-media comments with 3 original labels: **CLEAN**, **OFFENSIVE**, **HATE**.
- Pre-split into train/dev/test. After merging to binary (see §3.3), the distribution is as in Table 1.

### 3.2 VOZ-HSD — VOZ Hate Speech Detection dataset

*(~0.3 pg)*

- VOZ-HSD: comments from the VOZ forum (source: `tarudesu/VOZ-HSD` on HuggingFace).
- The data is pre-split into train/dev/test; the class distribution (after relabeling, see §3.3) is in Table 1.

### 3.3 Reshape dataset (Binary reformulation)

*(~0.4 pg)*

- **The only operation on the data is relabeling**, with no change or subsampling of the number of examples. The problem is cast as **binary classification**: merge **CLEAN** and **OFFENSIVE** into **NON-HATE**; keep **HATE** as the positive class.
- Rationale: focus on *hate speech directed at a group/target* rather than general toxicity/profanity; and create a consistent labeling setup across both datasets.
- Common preprocessing: noise normalization (lowercase, remove URLs, collapse repeated characters, normalize teencode, map emojis → coarse pragmatic tags), **word segmentation** before feeding into PhoBERT.

**Table 1 — Dataset statistics (after binary merge).** *(LaTeX: `table` + `booktabs`)*

| Dataset | Split | NON-HATE | HATE | Total | % HATE |
|---|---|---:|---:|---:|---:|
| ViHSD | Train | 21,492 | 2,556 | 24,048 | 10.6% |
| ViHSD | Dev | 2,402 | 270 | 2,672 | 10.1% |
| ViHSD | Test | 5,992 | 688 | 6,680 | 10.3% |
| VOZ-HSD | Train | 26,993 | 3,007 | 30,000 | 10.0% |
| VOZ-HSD | Dev | 4,520 | 480 | 5,000 | 9.6% |
| VOZ-HSD | Test | 4,487 | 513 | 5,000 | 10.3% |

> 📌 **[INSERT FIGURE] Figure 1 — Class distribution.** A bar chart showing the NON-HATE vs HATE imbalance on both datasets (numbers from Table 1). **Draw new** (matplotlib). Place right after Table 1, single column. Caption should emphasize "HATE ≈ 10% ⇒ macro-F1/HATE-F1 matter more than accuracy".

---

## 4. Methodology

*(~2 pages — the core; should include an overall architecture figure)*

Task: binary classification of a Vietnamese comment $x$ into NON-HATE or HATE. ViAmpleHate inherits AmpleHate's target-aware idea and adapts it to Vietnamese through 5 changes: (i) Vietnamese target extraction, (ii) separating target and attack cues, (iii) relation-bank attention, (iv) fixing the batched attention, (v) injecting relation info via an adaptive gate.

> 📌 **[INSERT FIGURE] Figure 2 — Overall ViAmpleHate architecture.** **The most important; draw new** (draw.io/PowerPoint), place at the start of Section 4, **full-width (`figure*`)** or single column. Flow diagram:
> `Comment → Preprocess/normalize → Word segmentation → PhoBERT encoder → [CLS] h₀ + token states H` →
> signal-extraction branches: `Vietnamese NER + Target-cue bank → T_x` and `Attack-cue bank → A_x` →
> `Relation-bank attention` 3 channels: `r_exp (target) / r_imp (CLS) / r_atk (attack)` → `fuse W_r` → `adaptive gate g=σ(W_g[h₀;r])` → `z = h₀ + g·r` → `Linear → {NON-HATE, HATE}`.
> Also annotate the loss `L = L_CE + α·L_CL` at the classifier block.

### 4.1 Text Preprocessing

- Noise normalization: lowercase, remove URLs/non-linguistic symbols, collapse repeated characters, normalize teencode, map emojis → coarse pragmatic tags (mockery, anger, disgust, laughter, intensity).
- **Word segmentation** (required because PhoBERT is trained on word-segmented text), then tokenize with the PhoBERT tokenizer, truncate/pad to a fixed length ($\text{max\_len}=256$).

### 4.2 Multi-Signal Target and Attack Extraction

- **Target signal**: Vietnamese NER (person/org/loc/GPE/group) **union** a **target-cue bank** (pronouns, group nouns, informal forms of address). A cue is *not* itself a hate indicator — it only marks a **possible target position**.
- **Attack signal**: an **attack-cue bank** (offensive predicates, insults, threats, negative evaluations — e.g. `khinh`, `ăn_bám`).
- **Separating target vs attack**: target = *who/what is being discussed*; attack = *is it being evaluated hostilely*. Avoids treating every profane word as a target and every target as an attack.
- Cue matching: normalize → segment → tokenize → **full token-sequence matching** (avoids matching a stray subtoken).
- **Fallback**: if a set is empty ⇒ use the `[CLS]` position (retain an implicit sentence-level representation).

$$
T_x = M_{\text{NER}}(x)\,\cup\,M_{\text{target}}(x), \qquad A_x = M_{\text{attack}}(x)
$$
$$
T_x \leftarrow \{0\}\ \text{if}\ T_x=\varnothing, \qquad A_x \leftarrow \{0\}\ \text{if}\ A_x=\varnothing
$$

### 4.3 PhoBERT Encoder

$$
H = \text{PhoBERT}(x) = [\,h_0, h_1, \dots, h_n\,], \quad h_i \in \mathbb{R}^{d},\ d=768
$$
where $h_0$ is the `[CLS]` representation (global sentence context); the target/attack positions provide localized evidence.

### 4.4 Relation-Bank Attention

Three relation channels:
$$
r_{\text{exp}} = \text{HeadAttn}(h_0, H[T_x]), \quad
r_{\text{imp}} = \text{HeadAttn}(h_0, h_0), \quad
r_{\text{atk}} = \text{HeadAttn}(h_0, H[A_x])
$$
with each HeadAttention over a token matrix $E \in \mathbb{R}^{m\times d}$:
$$
Q = W_q h_0,\quad K = W_k E,\quad V = W_v E,\quad
\alpha = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big),\quad r = \alpha V
$$
Fusion:
$$
r = W_r\,[\,r_{\text{exp}}\,;\,r_{\text{imp}}\,;\,r_{\text{atk}}\,] + b_r
$$

### 4.5 Corrected Batched Attention

The original AmpleHate computes $QK^\top$ at the **batch level** ($Q,K\in\mathbb{R}^{B\times d}\Rightarrow QK^\top\in\mathbb{R}^{B\times B}$), inadvertently **mixing information across samples**. ViAmpleHate uses **batched attention** so each sample attends only to its own cues:
$$
Q\in\mathbb{R}^{B\times 1\times d},\ K\in\mathbb{R}^{B\times m\times d},\quad
\text{scores} = \frac{\text{bmm}(Q,K^\top)}{\sqrt d}\in\mathbb{R}^{B\times1\times m}
$$
Padding positions are **masked** ($\text{scores}_j = -\infty$ if $\text{mask}_j=0$) before softmax.

### 4.6 Instance-Adaptive Relation Gate

Replace AmpleHate's fixed coefficient $e$ with a per-sample adaptive gate:
$$
g = \sigma\big(W_g\,[\,h_0\,;\,r\,] + b_g\big),\qquad z = h_0 + g\cdot r
$$
$z$ passes through dropout + linear → logits for {NON-HATE, HATE}. Strong cues ⇒ large $g$ (more relation contribution); weak/ambiguous cues ⇒ rely more on `[CLS]`.

### 4.7 Training Objective

$$
L = L_{\text{CE}} + \alpha\, L_{\text{CL}}, \qquad \alpha = 0.1
$$
- **Weighted cross-entropy** $L_{\text{CE}} = -\sum_c w_c\,y_c\log \hat y_c$ with class weights $w_c$ (handles imbalance), plus label smoothing.
- **Contrastive loss** on the post-gate representation $z$ (cosine $s_{ij}=\cos(z_i,z_j)$):
$$
L_{\text{CL}} = \frac{1}{N}\sum_{i\neq j}\Big[\mathbb{1}[y_i{=}y_j](1-s_{ij}) + \mathbb{1}[y_i{\neq}y_j]\max(0, s_{ij}-\text{margin})\Big]
$$
- **Threshold selection** for the HATE class on the validation set by macro-F1:
$$
t^{*} = \arg\max_t\ \text{MacroF1}\big(y,\ \mathbb{1}[p_{\text{HATE}}\ge t]\big)
$$

---

## 5. Experiments

*(~1.5 pages)*

### 5.1 Baselines

Compare ViAmpleHate against 5 baselines (same binary setting):
- **TF-IDF + Logistic Regression / SVM** — sparse lexical features.
- **BiLSTM + fastText (vi)** — static embeddings + sequence model.
- **PhoBERT-CNN** — PhoBERT + local feature extraction via CNN.
- **AmpleHate-PhoBERT (baseline)** — a faithful port of the original AmpleHate: English NER, one HeadAttention, fixed injection $z=h_0+e\,r_{\text{base}}$ ($e=1.0$). This is the **direct competitor** since it uses the same PhoBERT encoder.

### 5.2 Baseline vs Proposed — concrete changes

**Table 2 — Architecture/configuration comparison (Baseline AmpleHate-PhoBERT vs ViAmpleHate-PhoBERT).** *(LaTeX `table`, can use `\small`)*

| Component | Baseline AmpleHate-PhoBERT | ViAmpleHate-PhoBERT (proposed) |
|---|---|---|
| Encoder | PhoBERT-base | PhoBERT-base |
| Target extraction | English NER (`dbmdz/bert-large-...-conll03-english`) | Vietnamese NER (`NlpHUST/ner-vietnamese-electra-base`) + target-cue bank |
| Target coverage | ~21/24,048 ≈ 0.09% of train | ~18.8–20.0% (Vietnamese cues) |
| Attack signal | Not modeled | Separate attack-cue bank |
| Attention | 1 HeadAttention | 3 channels: target / implicit / attack |
| Attention computation | Batch-level `matmul(Q,Kᵀ)` (mixes samples) | Per-sample `bmm` + masking |
| Fusion | Fixed injection $h_0+e\,r$ ($e{=}1.0$) | Relation bank + **adaptive gate** $h_0+g\,r$ |
| Loss | Weighted CE | Weighted CE + Contrastive ($\alpha{=}0.1$) |
| max_len | 128 | 256 |
| Batch | 16 | 16 × grad-accum 2 (eff. 32) |
| NER at eval | Off (⇒ `[CLS]` fallback) | On, consistent across train/val/test (`USE_NER_AT_EVAL=True`) |

### 5.3 Implementation Details

**Table 3 — Hyperparameters.** *(LaTeX `table`)*

| Parameter | Value |
|---|---|
| Encoder | `vinai/phobert-base` (768-d) |
| Vietnamese NER | `NlpHUST/ner-vietnamese-electra-base` |
| max_len | 256 |
| LR (encoder / head) | 2e-5 / 5e-5 |
| Dropout | 0.1 |
| Effective batch | 32 (16 × grad-accum 2) |
| Epochs | up to 8 (selected by val macro-F1) |
| α (contrastive) | 0.1 |
| Checkpoint & threshold selection | by macro-F1 on validation |

### 5.4 Evaluation Metrics

- **Macro-F1** and **HATE-class F1** are the primary metrics (reflect the minority class better than accuracy).
- **Accuracy** is reference-only (inflated by imbalance).
- Threshold $t^{*}$ chosen on validation by macro-F1, applied fixed to test.

### 5.5 Experimental Purpose

Each change targets a concrete baseline limitation: Vietnamese NER+cues ↔ low target coverage; attack cues ↔ no hostility modeling; relation-bank ↔ need to separate target/context/attack; batched attention ↔ cross-sample leakage; adaptive gate ↔ fixed injection; contrastive ↔ class separation under imbalance.

---

## 6. Result Analysis / Error Analysis

### 6.1 Experimental Results

**Table 4 — Results on ViHSD (test).** *(LaTeX `table` + `booktabs`; **bold** the best number per column)*

| Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.8910 | 0.7393 | 0.5404 |
| TF-IDF + SVM | 0.9126 | 0.7131 | 0.4739 |
| BiLSTM + fastText | – | 0.7072 | 0.5060 |
| PhoBERT-CNN | – | 0.7571 | 0.5745 |
| AmpleHate-PhoBERT (baseline) | 0.9175 | 0.7792 | 0.6045 |
| **ViAmpleHate-PhoBERT (ours)** | **0.9205** | **0.7819** | **0.6081** |
| *Δ vs baseline* | *+0.0030* | *+0.0027* | *+0.0036* |

**Table 5 — Results on VOZ-HSD (test).**

| Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.9453 | 0.7745 | 0.5783 |
| TF-IDF + SVM | 0.9641 | 0.7831 | 0.5850 |
| BiLSTM + fastText | – | 0.6712 | 0.4187 |
| PhoBERT-CNN | – | 0.8150 | 0.6500 |
| AmpleHate-PhoBERT (baseline) | 0.9643 | 0.8185 | 0.6557 |
| **ViAmpleHate-PhoBERT (ours)** | 0.9420 | **0.8371** | **0.7065** |
| *Δ vs baseline* | *–0.0223* | *+0.0186* | *+0.0508* |

**Interpretation (write as prose):**
- **Transformer > static-embedding > lexical.** PhoBERT-based models outperform BiLSTM/TF-IDF thanks to contextual Vietnamese representations from large-scale pretraining.
- **The AmpleHate baseline is limited** because English NER rarely finds Vietnamese targets ⇒ it usually falls back to `[CLS]`, almost a plain PhoBERT.
- **ViAmpleHate improves the metrics that matter**: macro-F1 and HATE-F1 rise on both datasets; the gain is **more pronounced on VOZ-HSD** (HATE-F1 **+0.0508**). On ViHSD the gain is small but consistent; HATE-precision rises (0.5972 → 0.6177) while HATE-recall drops slightly (0.6119 → 0.5988) ⇒ the model is **more conservative but more precise** in assigning HATE.
- **Accuracy drops on VOZ** while macro/HATE-F1 rise: illustrates why **accuracy should not be the primary metric** on imbalanced data.
- **Dependence on cue coverage**: ViAmpleHate's benefit scales with cue quality/coverage; few cues ⇒ relies more on `[CLS]` ⇒ narrows the gap with the baseline.

> 📌 **[INSERT FIGURE] Figure 3 — Training curves.** Place in §6.1. Available file:
> `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/training_curves_viamplehate.png`
> (optionally beside the baseline `.../baselines/ViHSD - Baseline AmpleHate_PhoBERT/output/training_curves_amplehate.png`). Single column. Caption: loss/F1 over epochs, mark best epoch = 4 (val F1 = 0.7852).

> 📌 **[INSERT FIGURE] Figure 4 — Confusion matrix (ViHSD): Baseline vs ViAmpleHate.** **Side by side (`figure*`, two columns)** in §6.1 or start of §6.2. Available files:
> Baseline: `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/output/confusion_matrix_amplehate.png`
> Proposed: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/confusion_matrix_viamplehate.png`
> Caption: emphasize the reduction in HATE false positives (precision up).

> 📌 **[INSERT FIGURE — optional] Figure 5 — Confusion matrix (VOZ-HSD) proposed.** File:
> `notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/output/confusion_matrix_viamplehate_vozhsd.png`. Use if space permits; illustrates the larger HATE improvement on VOZ.

**Table 6 (optional) — Per-class P/R/F1 on ViHSD (proposed).**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NON-HATE | 0.9541 | 0.9574 | – |
| HATE | 0.6177 | 0.5988 | 0.6081 |

### 6.2 Error Analysis

Group the remaining errors (write as prose, 2–3 sentences each):

1. **Offensive but not hate.** Profane/personally aggressive comments that do not target a protected group ⇒ relying too heavily on attack cues ⇒ **false positives**.
2. **Implicit hate.** Sarcasm, insinuation, stereotypes, indirect comparison — no clear slur/attack token ⇒ hard for cue-based models, must rely on `[CLS]`.
3. **Ambiguous target cue.** Pronouns/group nouns appear in neutral/humorous comments too ⇒ detecting a target without hostile context ⇒ **over-predicting HATE**.
4. **Insufficient cue coverage.** Slang/spelling variants/abbreviations/creative profanity change fast; a fixed cue bank cannot cover everything ⇒ **false negatives**.
5. **Tokenization & span-alignment errors.** NER, word segmentation, and PhoBERT subword tokenization do not always align ⇒ a cue matched at the wrong position ⇒ attention attends to incomplete/irrelevant evidence.
6. **Threshold sensitivity.** The $t^{*}$ optimal on validation may not be optimal when the test/domain distribution shifts ⇒ affects deployment.
7. **Class imbalance.** HATE is the minority ⇒ fewer positive examples in training; weighted loss + threshold reduce but do not eliminate minority-class errors.

**Directions for improvement (leading into §7.2):** expand the cue banks via data-driven mining + manual validation; save **per-instance prediction logs** for systematic FP/FN analysis (by cue coverage, gate value, confidence); add target-span supervision, sarcasm detection, social-context information; apply **probability calibration** for more stable thresholds.

---

## 7. Conclusion and Future Work

### 7.1 Conclusion

- ViAmpleHate **generalizes AmpleHate for Vietnamese** through 5 adaptations: Vietnamese target extraction, separating target/attack cues, relation-bank attention, corrected batched attention, and adaptive gating; trained with CE + contrastive.
- On ViHSD and VOZ-HSD (binary), the model **improves macro-F1 and HATE-F1** over the AmpleHate baseline and the TF-IDF/BiLSTM/PhoBERT-CNN baselines.
- Core message: **target-awareness is useful, but must be adapted to Vietnamese linguistic characteristics** (targets are often informal group references, not named entities).

### 7.2 Future Work

- Expand & automatically mine the target/attack cue banks; manual validation.
- Per-instance error analysis (gate value, cue coverage, confidence); add span supervision, sarcasm detection, user/discourse context.
- Extend from binary to multi-label/multi-target settings (connecting with ViTHSD).
- Probability calibration for stable thresholds in cross-domain deployment.

---

## References

> *(Bibliographic details must be verified and completed before submission — use `\bibliography{}` with `acl_natbib` or import a `.bib` file. Minimal required list:)*

- **AmpleHate** — "AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection" *(fill in authors/venue/year)*.
- **ViTHSD** — "ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts" *(fill in details)*.
- **ViHSD** — Luu, S. T., Nguyen, K. V., Nguyen, N. L.-T. (2021). "A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts." *(IEA/AIE — verify)*.
- **VOZ-HSD** — the `tarudesu/VOZ-HSD` dataset (HuggingFace) *(fill in citation/URL)*.
- **PhoBERT** — Nguyen, D. Q., Nguyen, A. T. (2020). "PhoBERT: Pre-trained language models for Vietnamese." *Findings of EMNLP 2020*.
- **BERT** — Devlin et al. (2019). *NAACL*.
- **fastText** — Bojanowski et al. (2017). *TACL*.
- **Contrastive / SupCon** — Khosla et al. (2020). *NeurIPS* *(if using supervised contrastive)*.
- **Vietnamese NER** — `NlpHUST/ner-vietnamese-electra-base` *(fill in citation/URL)*.
- *(Optional)* VnCoreNLP / RDRSegmenter for word segmentation; hate speech detection surveys.

---

### Appendix: Checklist for converting to LaTeX ACL

- [ ] Download the ACL template (`acl_latex.tex`, `acl.sty`, `acl_natbib.bst`) from Overleaf.
- [ ] Convert each `##`/`###` to `\section`/`\subsection`; Markdown tables to `tabular`+`booktabs`.
- [ ] Equations: paste `$...$`/`$$...$$` into `equation`/`align`.
- [ ] Figures: create Figure 1 (class distribution) & Figure 2 (architecture — **draw new**); Figures 3–5 use existing PNGs in `notebooks/.../output/`.
- [ ] Bold the best number in Tables 4–5; add the Δ row.
- [ ] Write the Abstract (~180 words) last; add `\section*{Limitations}` (fixed cue bank, threshold sensitivity, implicit hate) — ACL requires a Limitations section.
- [ ] Complete references, use `\citep`/`\citet`.
- [ ] Ensure ≤ 8 pages of content (ACL) + references/appendix do not count.
</content>
