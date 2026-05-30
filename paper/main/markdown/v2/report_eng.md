# ViAmpleHate: A Proposed AmpleHate- and PhoBERT-based Approach for Vietnamese Hate Speech Detection

> This document is a prose content draft intended to be turned into an 8–10 page LaTeX paper following the ACL conference template. Equations are written in LaTeX syntax, tables are in Markdown (to be converted to `booktabs`), and positions that need figures are marked with **📌 [INSERT FIGURE]** blocks.

## Abstract

Detecting hate speech on Vietnamese social media is difficult because hateful intent is rarely expressed through clear named entities; instead it hides in informal group references, slang, and indirect insinuation. Target-aware models such as AmpleHate have shown that attending to the relationship between an utterance and the target it addresses helps detect even implicit hate. However, AmpleHate's original pipeline relies on English named-entity recognition (NER) and English-oriented target categories, so it transfers poorly to Vietnamese. We propose **ViAmpleHate**, a Vietnamese adaptation of AmpleHate built on **PhoBERT**. ViAmpleHate replaces English NER with Vietnamese NER, adds two signal banks — a *target-cue* bank and an *attack-cue* bank — models three relation channels (explicit target, implicit context, and attack) through a *relation-bank attention*, fixes a batch-level sample-mixing bug in the attention computation, and replaces fixed-scalar injection with an *instance-adaptive gate*. The model is trained with a combination of weighted cross-entropy and a contrastive loss. On the **ViHSD** and **VOZ-HSD** datasets under a binary NON-HATE/HATE setting, ViAmpleHate improves macro-F1 and the minority HATE-class F1 over a faithful AmpleHate baseline as well as over TF-IDF, BiLSTM, and PhoBERT-CNN baselines. These results confirm that target-aware modeling must be adapted to Vietnamese linguistic characteristics in order to be effective.

## 1. Introduction

The rapid growth of Vietnamese social media has been accompanied by a spread of hateful content that harms both individuals and communities, creating an urgent need for large-scale automatic moderation tools. Detecting hate speech in Vietnamese, however, faces several language-specific difficulties. First, social-media language is noisy: users abbreviate, use teencode, stretch characters, mix in emojis, and misspell words deliberately or accidentally. Second, the target of an attack in Vietnamese is rarely a named entity; instead, people typically aim at a group through pronouns, group nouns, or informal forms of address. Third, the data is severely imbalanced: the HATE class accounts for only about one tenth of the samples.

The target-aware approach, exemplified by AmpleHate, has demonstrated that modeling the relationship between a sentence and the target it addresses helps detect even implicit hate speech. The problem is that the original AmpleHate identifies targets with English NER. When applied to Vietnamese, this NER almost never finds a valid target: on the ViHSD training set, only 21 of 24,048 samples (about 0.09%) contain a target detected by English NER. As a result, the model nearly always falls back to the generic sentence representation at the `[CLS]` token and degenerates into an ordinary PhoBERT classifier, losing all of its target-aware advantage.

In this paper we propose ViAmpleHate to address exactly these weaknesses. Our contributions are fivefold. **First**, we replace English NER with Vietnamese NER and add a *target-cue* bank of pronouns, group nouns, and informal forms of address, which raises target coverage from about 0.09% to roughly 18.8–20.0%. **Second**, we separate two kinds of signal: target cues answer the question "who is being talked about," while attack cues answer "is that target being attacked," and we model them through a three-channel relation-bank attention. **Third**, we fix a batch-level attention bug that mixes information across samples, and we replace fixed-scalar injection with an adaptive gate computed per sample. **Fourth**, we train the model with a combination of weighted cross-entropy and a contrastive loss to improve class separation under imbalance. **Fifth**, we evaluate on ViHSD and VOZ-HSD, compare against five different baselines, and observe improvements in both macro-F1 and HATE-class F1. The recurring message is that target-awareness is genuinely useful, but only when it is designed to fit the way Vietnamese speakers express hate.

## 2. Related Work

### 2.1. AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection

AmpleHate aims to detect implicit hate speech by amplifying the attention between the whole-sentence representation at the `[CLS]` token and potential targets extracted by NER. A target-aware relation vector is computed through a HeadAttention mechanism in which the query comes from `[CLS]` and the keys and values come from the target tokens; this vector is then injected into the sentence representation as $z = h_0 + e\cdot r$ with a fixed coefficient $e$, before classification. This works well for English but exposes three limitations when ported to Vietnamese: it depends on English-style NER and target categories, it injects information with a fixed strength for every sample, and it models only a single relation channel — the target — while ignoring the attack signal. Our work inherits AmpleHate's target-aware idea directly but redesigns the pipeline for Vietnamese.

### 2.2. ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts

ViTHSD approaches the problem as target-grounded hate detection, assigning a degree of hatred to each target addressed in a sentence. This work reinforces the observation that the target is a core signal of Vietnamese hate speech, thereby supporting ViAmpleHate's target-aware motivation. The difference is that ViAmpleHate does not require span-level target labels; instead, we combine NER and cue banks to locate target and attack positions without span supervision. In addition, to situate our contribution, the common Vietnamese baselines we compare against are PhoBERT, PhoBERT-CNN, BiLSTM with fastText embeddings, and a classical TF-IDF model.

## 3. Datasets

### 3.1. ViHSD — Vietnamese Hate Speech Detection dataset

ViHSD is a Vietnamese hate speech detection dataset consisting of social-media comments labeled with three original classes — CLEAN, OFFENSIVE, and HATE — and pre-split into training, validation, and test sets. After conversion to the binary setting (described in Section 3.3), the class distribution of the three splits is shown in Table 1.

### 3.2. VOZ-HSD — VOZ forum Hate Speech Detection dataset

VOZ-HSD consists of comments collected from the VOZ forum, originating from the `tarudesu/VOZ-HSD` set on HuggingFace. The data is also split into training, validation, and test sets; the class distribution after relabeling is presented together in Table 1.

### 3.3. Relabeling the data (binary reformulation)

The only operation we perform on the data is relabeling; we do not alter or subsample the number of examples in any way. Specifically, we cast the problem as binary classification by merging the CLEAN and OFFENSIVE labels into a single NON-HATE class while keeping HATE as the positive class. This choice focuses the task on hate speech directed at a specific group or target, rather than profanity or aggression in general, and at the same time creates a consistent labeling setup across the two datasets. Before being fed to the model, every comment is normalized to reduce noise: lowercasing, removing URLs and non-linguistic symbols, collapsing repeated characters, normalizing teencode, and mapping selected emojis to coarse pragmatic tags such as mockery, anger, disgust, or laughter. The text is then word-segmented — a mandatory step because PhoBERT is trained on word-segmented text — before tokenization.

**Table 1 — Dataset statistics after binary relabeling.**

| Dataset | Split | NON-HATE | HATE | Total | % HATE |
|---|---|---:|---:|---:|---:|
| ViHSD | Train | 21,492 | 2,556 | 24,048 | 10.6% |
| ViHSD | Dev | 2,402 | 270 | 2,672 | 10.1% |
| ViHSD | Test | 5,992 | 688 | 6,680 | 10.3% |
| VOZ-HSD | Train | 26,993 | 3,007 | 30,000 | 10.0% |
| VOZ-HSD | Dev | 4,520 | 480 | 5,000 | 9.6% |
| VOZ-HSD | Test | 4,487 | 513 | 5,000 | 10.3% |

> 📌 **[INSERT FIGURE] Figure 1 — Class distribution.** A bar chart showing the NON-HATE vs HATE imbalance on both datasets (numbers from Table 1). **Create new** with matplotlib, place right after Table 1, single column. The caption should emphasize that because HATE is only about 10%, macro-F1 and HATE-F1 matter more than accuracy.

## 4. Methodology

We frame the problem as binary classification: given a Vietnamese comment $x$, the model predicts NON-HATE or HATE. ViAmpleHate inherits AmpleHate's target-aware idea and adapts it to Vietnamese through five changes: Vietnamese target extraction, separating target cues from attack cues, relation-bank attention, fixing the batch-level attention mechanism, and injecting relation information through an adaptive gate.

> 📌 **[INSERT FIGURE] Figure 2 — Overall architecture of ViAmpleHate.** This is the most important figure; **create new** (draw.io or PowerPoint) and place it at the start of Section 4. The diagram should show the flow: input comment → normalization → word segmentation → PhoBERT encoder producing `[CLS]` $h_0$ and token states $H$; from there, two signal branches split off, one using Vietnamese NER together with the target-cue bank to form the set $T_x$, and one using the attack-cue bank to form the set $A_x$; three relation-bank attention channels produce $r_{exp}$, $r_{imp}$, $r_{atk}$; these vectors are fused through $W_r$, then passed through the adaptive gate $g=\sigma(W_g[h_0;r])$ to form $z = h_0 + g\cdot r$, and finally through a linear layer to predict the two labels. The loss $L = L_{CE} + \alpha L_{CL}$ should be annotated at the classifier block.

### 4.1. Text Preprocessing

Vietnamese social-media text is normalized to reduce noise before encoding: lowercasing, removing URLs and non-linguistic parts, collapsing repeated characters, normalizing teencode variants, and mapping selected emojis to coarse pragmatic tags. After normalization, each comment is word-segmented, then tokenized with the PhoBERT tokenizer and truncated or padded to a fixed length, here 256 tokens. The word-segmentation step is required because PhoBERT is pretrained on word-segmented Vietnamese text.

### 4.2. Multi-Signal Target and Attack Extraction

ViAmpleHate uses a multi-signal extraction strategy to identify linguistically meaningful positions in the input sequence. The target signal comes from Vietnamese NER — detecting persons, organizations, locations, geopolitical entities, and named groups — combined with a target-cue bank containing Vietnamese referential expressions commonly used to introduce a person or a group. These cues are not hate indicators by themselves; they only mark possible target positions. In parallel, an attack-cue bank contains offensive predicates, insults, threats, and negative evaluations. Separating target cues from attack cues is deliberate: a target cue helps answer who is being discussed, while an attack cue helps determine whether any hostile evaluation is being directed at that target. This separation prevents the model from misreading every profane word as a target, or every reference to a target as an attack. Each cue phrase is normalized, word-segmented, tokenized, and matched against the PhoBERT token sequence using full token-sequence matching, in order to avoid matching only a subtoken fragment. If no explicit cue is found, the corresponding set falls back to the `[CLS]` position, allowing the model to retain an implicit sentence representation.

$$
T_x = M_{\text{NER}}(x)\,\cup\,M_{\text{target}}(x), \qquad A_x = M_{\text{attack}}(x)
$$
$$
T_x \leftarrow \{0\}\ \text{if}\ T_x=\varnothing, \qquad A_x \leftarrow \{0\}\ \text{if}\ A_x=\varnothing
$$

### 4.3. PhoBERT Encoder

The normalized, word-segmented input is encoded with PhoBERT, yielding the final hidden states

$$
H = \text{PhoBERT}(x) = [\,h_0, h_1, \dots, h_n\,], \quad h_i \in \mathbb{R}^{d},\ d=768,
$$

where $h_0$ is the `[CLS]` representation capturing the global sentence context, while the extracted target and attack positions provide localized evidence for target-aware reasoning.

### 4.4. Relation-Bank Attention

ViAmpleHate builds a relation bank with three views: an explicit target relation from the target tokens, an implicit context relation from the `[CLS]` anchor, and an attack relation from the attack tokens.

$$
r_{\text{exp}} = \text{HeadAttn}(h_0, H[T_x]), \quad
r_{\text{imp}} = \text{HeadAttn}(h_0, h_0), \quad
r_{\text{atk}} = \text{HeadAttn}(h_0, H[A_x])
$$

Each HeadAttention module, given a relation token matrix $E \in \mathbb{R}^{m\times d}$, is computed as

$$
Q = W_q h_0,\quad K = W_k E,\quad V = W_v E,\quad
\alpha = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big),\quad r = \alpha V.
$$

The three relation vectors are concatenated and projected into a single fused relation representation:

$$
r = W_r\,[\,r_{\text{exp}}\,;\,r_{\text{imp}}\,;\,r_{\text{atk}}\,] + b_r.
$$

This fused relation vector captures complementary information from target mentions, sentence-level context, and attack expressions.

### 4.5. Corrected Batched Attention

The original AmpleHate-style implementation computes attention with a matrix multiplication equivalent to $QK^\top$ over the whole batch. When $Q,K\in\mathbb{R}^{B\times d}$, then $QK^\top\in\mathbb{R}^{B\times B}$, unintentionally mixing information across different samples in the same batch. ViAmpleHate fixes this with batched attention, so that each sample attends only to its own target or attack tokens:

$$
Q\in\mathbb{R}^{B\times 1\times d},\quad K\in\mathbb{R}^{B\times m\times d},\quad
\text{scores} = \frac{\text{bmm}(Q,K^\top)}{\sqrt d}\in\mathbb{R}^{B\times1\times m}.
$$

Padding positions are suppressed with an attention mask before softmax: $\text{scores}_j = -\infty$ if $\text{mask}_j=0$.

### 4.6. Instance-Adaptive Relation Gate

The original AmpleHate mechanism uses a fixed scalar to control how much target-aware attention is injected into the sentence representation. ViAmpleHate replaces this fixed strength with a gate computed per sample:

$$
g = \sigma\big(W_g\,[\,h_0\,;\,r\,] + b_g\big),\qquad z = h_0 + g\cdot r.
$$

The gate lets the model decide how strongly relation information affects each prediction. If a comment has clear target and attack evidence, the model can give more weight to the relation vector; if the cues are weak, ambiguous, or absent, it can rely more on the global sentence representation. The final representation $z$ passes through dropout and a linear layer to produce logits for NON-HATE and HATE.

### 4.7. Training Objective

The model is trained with a combination of weighted cross-entropy and a contrastive loss:

$$
L = L_{\text{CE}} + \alpha\, L_{\text{CL}}, \qquad \alpha = 0.1.
$$

The weighted cross-entropy term, $L_{\text{CE}} = -\sum_c w_c\,y_c\log \hat y_c$, addresses class imbalance by assigning higher weight to the minority HATE class, with label smoothing to reduce overconfident predictions. The contrastive term is applied to the post-gate representation $z$, encouraging same-class examples to have closer representations and different-class examples to be farther apart under cosine similarity:

$$
L_{\text{CL}} = \frac{1}{N}\sum_{i\neq j}\Big[\mathbb{1}[y_i{=}y_j](1-s_{ij}) + \mathbb{1}[y_i{\neq}y_j]\max(0, s_{ij}-\text{margin})\Big].
$$

At evaluation, the decision threshold for the HATE class is selected on the validation set by maximizing macro-F1, and then applied unchanged to the test set:

$$
t^{*} = \arg\max_t\ \text{MacroF1}\big(y,\ \mathbb{1}[p_{\text{HATE}}\ge t]\big).
$$

## 5. Experiments

### 5.1. Baselines

We compare ViAmpleHate against five baselines, all in the binary setting. The two classical baselines are TF-IDF with Logistic Regression and TF-IDF with SVM, using sparse lexical features. Next is BiLSTM with Vietnamese fastText embeddings, combining static embeddings with a sequence model. Stronger still is PhoBERT-CNN, which uses PhoBERT as a contextual encoder and a CNN to extract local features. Finally, the most important baseline is AmpleHate-PhoBERT, a faithful port of the original AmpleHate with English NER, a single HeadAttention module, and fixed injection $z = h_0 + e\,r_{\text{base}}$ with $e=1.0$. This is the direct competitor because it uses the same PhoBERT encoder, so any difference in results faithfully reflects the impact of our proposed changes.

### 5.2. Baseline vs Proposed comparison

Table 2 summarizes the concrete architectural and configuration differences between the AmpleHate-PhoBERT baseline and ViAmpleHate-PhoBERT.

**Table 2 — Architecture and configuration comparison.**

| Component | Baseline AmpleHate-PhoBERT | ViAmpleHate-PhoBERT (proposed) |
|---|---|---|
| Encoder | PhoBERT-base | PhoBERT-base |
| Target extraction | English NER (`dbmdz/bert-large-...-conll03-english`) | Vietnamese NER (`NlpHUST/ner-vietnamese-electra-base`) + target-cue bank |
| Target coverage | ~21/24,048 ≈ 0.09% of train | ~18.8–20.0% via Vietnamese cues |
| Attack signal | Not modeled | Separate attack-cue bank |
| Attention | One HeadAttention | Three channels: target / implicit / attack |
| Attention computation | Batch-level `matmul(Q,Kᵀ)` (mixes samples) | Per-sample `bmm` + masking |
| Fusion | Fixed injection $h_0+e\,r$ ($e{=}1.0$) | Relation bank + adaptive gate $h_0+g\,r$ |
| Loss | Weighted CE | Weighted CE + Contrastive ($\alpha{=}0.1$) |
| Max length | 128 | 256 |
| Batch | 16 | 16 × grad-accum 2 (effective 32) |
| NER at evaluation | Off (leads to `[CLS]` fallback) | On, consistent across train/val/test |

### 5.3. Implementation Details

The encoder is `vinai/phobert-base` with a hidden size of 768, and the Vietnamese NER is `NlpHUST/ner-vietnamese-electra-base`. The maximum sequence length is 256 tokens. Learning rates are 2e-5 for the encoder and 5e-5 for the classification head, with dropout 0.1. The effective batch size is 32, obtained with a batch of 16 and two-step gradient accumulation. The model is trained for up to eight epochs, selecting the best checkpoint by validation macro-F1. The contrastive weight $\alpha$ is set to 0.1, and the decision threshold is also chosen on the validation set by macro-F1.

### 5.4. Evaluation Metrics

The two primary metrics are macro-F1 and HATE-class F1, because they reflect minority-class performance better than accuracy. Accuracy is reported only for reference, since on imbalanced data a model can achieve high accuracy merely by correctly predicting most NON-HATE samples while still missing many hate-speech instances. The optimal threshold is chosen on the validation set and then fixed for prediction on the test set.

### 5.5. Experimental Purpose

The goal is not only to compare final scores but also to verify that each change addresses a concrete limitation of the baseline. Vietnamese NER and cues address low target coverage; attack cues compensate for the absence of hostility modeling; relation-bank attention meets the need to separate target, context, and attack; the corrected batched attention removes cross-sample information leakage; the adaptive gate replaces fixed injection; and the contrastive loss improves class separation under imbalance.

## 6. Result and Error Analysis

### 6.1. Experimental Results

Tables 3 and 4 present the test-set results on ViHSD and VOZ-HSD.

**Table 3 — Results on ViHSD (test set).**

| Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.8910 | 0.7393 | 0.5404 |
| TF-IDF + SVM | 0.9126 | 0.7131 | 0.4739 |
| BiLSTM + fastText | – | 0.7072 | 0.5060 |
| PhoBERT-CNN | – | 0.7571 | 0.5745 |
| AmpleHate-PhoBERT (baseline) | 0.9175 | 0.7792 | 0.6045 |
| **ViAmpleHate-PhoBERT (ours)** | **0.9205** | **0.7819** | **0.6081** |
| *Δ vs baseline* | *+0.0030* | *+0.0027* | *+0.0036* |

**Table 4 — Results on VOZ-HSD (test set).**

| Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.9453 | 0.7745 | 0.5783 |
| TF-IDF + SVM | 0.9641 | 0.7831 | 0.5850 |
| BiLSTM + fastText | – | 0.6712 | 0.4187 |
| PhoBERT-CNN | – | 0.8150 | 0.6500 |
| AmpleHate-PhoBERT (baseline) | 0.9643 | 0.8185 | 0.6557 |
| **ViAmpleHate-PhoBERT (ours)** | 0.9420 | **0.8371** | **0.7065** |
| *Δ vs baseline* | *–0.0223* | *+0.0186* | *+0.0508* |

The results show that transformer-based models generally outperform the lexical and static-embedding baselines. This is reasonable, since hate speech often depends on context, informal phrasing, and the interaction between target mentions and hostile predicates — things that sparse lexical models struggle to capture and static embeddings capture only partially. The AmpleHate baseline improves over a plain PhoBERT classifier thanks to target-aware attention, but its benefit is limited because English NER rarely detects valid Vietnamese targets, causing the model to fall back to the `[CLS]` representation.

ViAmpleHate improves exactly the two most important metrics — macro-F1 and HATE-F1 — on both datasets, and the improvement is far more pronounced on VOZ-HSD, where HATE-F1 rises by 0.0508. On ViHSD the gain is small but consistent: HATE-class precision rises from 0.5972 to 0.6177 while recall drops slightly from 0.6119 to 0.5988, showing that the model becomes more conservative but more precise when assigning the HATE label. Notably, on VOZ-HSD accuracy drops while both macro-F1 and HATE-F1 increase — a direct illustration of why accuracy should not be the primary metric on imbalanced data. Finally, the benefit of ViAmpleHate depends on the quality and coverage of the extracted cues: when cues are detected frequently and accurately, relation-bank attention has more useful evidence to exploit; when cue coverage is low, the model must rely more on the implicit `[CLS]` pathway and the gap with the baseline narrows.

> 📌 **[INSERT FIGURE] Figure 3 — Training curves.** Place in Section 6.1. Available file: `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/training_curves_viamplehate.png` (can be placed beside the baseline curves at `.../baselines/ViHSD - Baseline AmpleHate_PhoBERT/output/training_curves_amplehate.png`). Single column. The caption should note that the best epoch is 4 with a validation macro-F1 of 0.7852.

> 📌 **[INSERT FIGURE] Figure 4 — Confusion matrices on ViHSD: baseline vs ViAmpleHate.** Place side by side (full-width, two columns) in Section 6.1 or at the start of 6.2. Available files: baseline at `notebooks/models/baselines/ViHSD - Baseline AmpleHate_PhoBERT/output/confusion_matrix_amplehate.png` and proposed at `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/confusion_matrix_viamplehate.png`. The caption should highlight the reduction in HATE-class false positives.

> 📌 **[INSERT FIGURE — optional] Figure 5 — Confusion matrix on VOZ-HSD (proposed model).** File: `notebooks/models/proposed/VOZ-HSD - Proposed ViAmpleHate_PhoBERT/output/confusion_matrix_viamplehate_vozhsd.png`. Use if space permits, to illustrate the larger HATE improvement on VOZ.

**Table 5 (optional) — Per-class Precision/Recall/F1 on ViHSD (proposed model).**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| NON-HATE | 0.9541 | 0.9574 | – |
| HATE | 0.6177 | 0.5988 | 0.6081 |

### 6.2. Error Analysis

The remaining errors can be grouped into several categories. The first is confusion between offensive language and hate speech: many Vietnamese comments contain insults or profanity but do not target a protected group, and if the model relies too heavily on attack cues it will misclassify them as HATE, producing false positives. The second is implicit hate speech: some hateful comments contain no slur, named entity, or explicit attack predicate, expressing hostility instead through sarcasm, insinuation, stereotypes, or shared social context, which is hard for cue-based models to detect. The third is ambiguous target reference: Vietnamese pronouns and group nouns can appear in neutral or humorous comments too, so detecting a target cue without hostile context can lead the model to overestimate the likelihood of hate.

The fourth is incomplete cue coverage: Vietnamese online language changes quickly, with countless spelling variants, slang, abbreviations, and creative profanity, so a fixed cue bank cannot cover everything, leading to false negatives when important cues are missing or improperly normalized. The fifth is tokenization and span-alignment error: NER, word segmentation, and PhoBERT subword tokenization do not always align, especially for multiword expressions, so when a cue fails to match the correct position the attention module may attend to incomplete or irrelevant evidence. The sixth is threshold sensitivity: a threshold that is optimal on validation may not remain optimal when the distribution or domain shifts, which is especially important at deployment. The seventh is class imbalance: because HATE is much rarer than NON-HATE, the model sees fewer positive examples during training, and although weighted loss and threshold tuning reduce the problem, minority-class errors remain.

These errors suggest several directions for improvement that lead directly into the conclusion: expanding the target and attack cue banks through data-driven mining combined with manual validation; saving per-instance prediction logs to systematically analyze false positives and false negatives by cue coverage, gate value, and confidence; adding target-span supervision, sarcasm detection, or social-context information; and applying probability calibration so that the decision threshold is more stable across domains.

## 7. Conclusion and Future Work

### 7.1. Conclusion

ViAmpleHate generalizes AmpleHate for Vietnamese hate speech detection through five main adaptations: Vietnamese target extraction, separate modeling of target and attack cues, relation-bank attention, a corrected batched attention mechanism, and adaptive relation injection; the model is trained with a combination of cross-entropy and a contrastive loss. On ViHSD and VOZ-HSD in the binary setting, it improves macro-F1 and HATE-F1 over the AmpleHate baseline as well as the TF-IDF, BiLSTM, and PhoBERT-CNN baselines. The core message is that target-awareness is genuinely useful, but it must be adapted to Vietnamese linguistic patterns, where the target of hate is often expressed through informal group references rather than named entities.

### 7.2. Future Work

In the future we plan to expand and automatically mine the target and attack cue banks, with manual validation, to improve coverage. We also want to analyze errors at the per-instance level based on gate value, cue coverage, and prediction confidence, while adding span supervision, sarcasm detection, and user or discourse context. Another direction is to extend from the binary setting to multi-label or multi-target classification, connecting with the spirit of ViTHSD. Finally, probability calibration could make the decision threshold more stable for cross-domain deployment.

## References

> *(Bibliographic details must be verified and completed before submission; use `\bibliography{}` with `acl_natbib` or import a `.bib` file. Minimal list:)*

- **AmpleHate** — "AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection" *(fill in authors, venue, year)*.
- **ViTHSD** — "ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts" *(fill in details)*.
- **ViHSD** — Luu, S. T., Nguyen, K. V., Nguyen, N. L.-T. (2021). "A Large-scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts" *(verify venue)*.
- **VOZ-HSD** — the `tarudesu/VOZ-HSD` dataset on HuggingFace *(fill in citation/URL)*.
- **PhoBERT** — Nguyen, D. Q., Nguyen, A. T. (2020). "PhoBERT: Pre-trained language models for Vietnamese." *Findings of EMNLP 2020*.
- **BERT** — Devlin et al. (2019). *NAACL*.
- **fastText** — Bojanowski et al. (2017). *TACL*.
- **Supervised Contrastive Learning** — Khosla et al. (2020). *NeurIPS* *(if applicable)*.
- **Vietnamese NER** — `NlpHUST/ner-vietnamese-electra-base` *(fill in citation/URL)*.

---

### Appendix: Checklist for converting to LaTeX ACL

- [ ] Download the ACL template (`acl_latex.tex`, `acl.sty`, `acl_natbib.bst`) from Overleaf.
- [ ] Convert each `##`/`###` heading to `\section`/`\subsection`; Markdown tables to `tabular` + `booktabs`.
- [ ] Move `$...$` and `$$...$$` equations into `equation`/`align` environments.
- [ ] Create Figure 1 (class distribution) and Figure 2 (architecture — **draw new**); Figures 3–5 use the existing PNGs in `notebooks/.../output/`.
- [ ] Bold the best number in Tables 3–4 and add the Δ row.
- [ ] Write the Abstract (~180 words) last; add a `\section*{Limitations}` (fixed cue bank, threshold sensitivity, implicit hate) — ACL requires a Limitations section.
- [ ] Complete the reference list, using `\citep`/`\citet`.
- [ ] Ensure the content stays within 8 pages per ACL rules.
</content>
