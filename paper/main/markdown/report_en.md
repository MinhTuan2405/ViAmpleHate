# ViAmpleHate: A Proposed AmpleHate- and PhoBERT-based Approach for Vietnamese Hate Speech Detection (v_review — revised)

Tran Nguyen Duc Trung, Nguyen Ha Minh Tuan, Nguyen Gia Cat Long, Tran Phan Thanh Tung, Mo Van Tung, Huynh Van Tin
*University of Information Technology, VNU-HCM, Ho Chi Minh City, Vietnam*
`{23521687, 23521718, 23520881, 23521747, 23521741}@gm.uit.edu.vn`

> **Revised copy of v3 (revision rounds 1–2)** per `review_report.md`. Round 2 filled real numbers from the notebooks (hyperparameters, baseline accuracies, cue-bank sizes, target/attack coverage, VOZ per-class) and removed the ablation section per author instruction; the differing baseline/proposed configuration is now presented as intentional design rather than a confound to fix. Remaining open items (single-run only / no significance test; no published-SOTA comparison) are stated in the Limitations. **No numeric results were fabricated.** See `latex/v_review/REVISION_NOTES.md`. Figure-insertion notes use **📌 [INSERT FIGURE]**.

## Abstract

Detecting hate speech on Vietnamese social media is difficult because hateful intent is rarely carried by explicit named entities; instead it hides in informal group references, slang, sarcasm, and indirect insinuation. Target-aware models such as AmpleHate have shown that attending to the relationship between an utterance and the target it addresses helps detect even implicit hate, yet the original AmpleHate pipeline relies on English named-entity recognition (NER) and English-oriented target categories, so it transfers poorly to Vietnamese: on our training data English NER finds a usable target in only about 0.09% of comments, collapsing the model into a plain sentence classifier. We propose **ViAmpleHate**, a Vietnamese adaptation of AmpleHate built on PhoBERT. ViAmpleHate (i) replaces English NER with Vietnamese NER and a hand-curated *target-cue* bank, raising target coverage from ~0.09% to ~19–45% depending on the dataset; (ii) adds a separate *attack-cue* bank and models three relation channels — explicit target, implicit context, and attack — through a *relation-bank attention*; (iii) replaces fixed-scalar injection with an *instance-adaptive gate*; and (iv) trains with weighted cross-entropy combined with a contrastive objective. On the ViHSD and VOZ-HSD datasets in a binary NON-HATE/HATE setting, ViAmpleHate improves macro-F1 and the minority HATE-class F1 over an AmpleHate baseline and over TF-IDF, BiLSTM, and PhoBERT-CNN baselines, with the largest and clearest gains on the minority class (HATE-F1 +0.0508 on VOZ-HSD); improvements on ViHSD are small and we report them as preliminary, pending a controlled, multi-seed evaluation. We further analyse how target-cue coverage relates to these gains, supporting — rather than proving — the view that target-aware modeling should be adapted to Vietnamese rather than transferred verbatim from English.

## 1. Introduction

The rapid growth of Vietnamese-language social media has been accompanied by a corresponding spread of hateful and abusive content. Such content harms targeted individuals and communities, degrades online discourse, and creates legal and reputational risk for platforms. Because the volume of user-generated text vastly exceeds what human moderators can review, automatic hate speech detection has become a practical necessity. Yet most progress in hate speech detection has centred on English and other high-resource languages, and methods that work well there often degrade when applied to Vietnamese.

Vietnamese hate speech detection is hard for three intertwined reasons. First, social-media language is extremely noisy: users routinely abbreviate, write in *teencode*, elongate characters for emphasis, embed emojis with pragmatic meaning, and (deliberately or not) misspell words, including obfuscating profanity to evade filters. Second, and most importantly for this work, the *target* of an attack is rarely a named entity. Whereas English hate speech frequently names a person, organization, or nationality, Vietnamese hostility is typically aimed at a group through pronouns, kinship or group nouns, regional labels, or informal forms of address. Third, the data is severely imbalanced: in both datasets we study, the HATE class accounts for only about one tenth of all comments, so a model can reach high accuracy while detecting almost no hate.

A promising line of work models hate speech in a *target-aware* manner. AmpleHate, in particular, shows that explicitly attending to the relationship between a sentence and its potential target improves detection of *implicit* hate, where hostility is conveyed without overt slurs. AmpleHate extracts candidate targets with NER, computes a target-aware attention vector, and injects it into the sentence representation before classification. This design is attractive for Vietnamese, where the target is so central — but the original pipeline is built around English NER and English target categories. When we run the original AmpleHate pipeline on Vietnamese with an English NER component, it finds a usable target in only 21 of 24,048 training comments (about 0.09%). For the overwhelming majority of inputs the model falls back to the global `[CLS]` representation and behaves like an ordinary PhoBERT classifier, discarding precisely the target-aware signal that motivated it.

In this paper we propose **ViAmpleHate**, a Vietnamese adaptation of AmpleHate that re-engineers each stage of the pipeline around how Vietnamese speakers actually express hate. Our contributions are as follows.

1. **Vietnamese target extraction.** We replace English NER with a Vietnamese NER model and augment it with a hand-curated *target-cue* bank of pronouns, group nouns, and informal forms of address. Together these raise the fraction of comments with at least one detected target from about 0.09% to roughly 18.8–20.0%.
2. **Separate target and attack channels.** We introduce a distinct *attack-cue* bank for hostile predicates and model target cues and attack cues as separate signals: a target cue answers *who* is being discussed, while an attack cue answers *whether* hostility is directed at them. These are combined through a three-channel **relation-bank attention** (explicit target, implicit context, attack).
3. **Adaptive fusion (plus an implementation note).** We replace AmpleHate's fixed-scalar injection with an **instance-adaptive gate** that decides, per comment, how strongly relation evidence should influence the prediction. We additionally describe an implementation-level correction to a batch-level attention computation in our port, which we report as an implementation note rather than a primary contribution.
4. **Training objective.** We train with weighted cross-entropy combined with a contrastive loss that sharpens the separation between HATE and NON-HATE representations under heavy class imbalance.
5. **Empirical study.** We evaluate on ViHSD and VOZ-HSD against five baselines spanning lexical, static-embedding, and contextual models, and analyse where and why the gains arise, including a target-coverage analysis and a qualitative study of remaining errors.

Across both datasets, ViAmpleHate improves the two metrics that matter most under imbalance — macro-F1 and HATE-class F1 — with the clearest gains on VOZ-HSD's minority class; the ViHSD gains are small and we treat them as preliminary pending a controlled, multi-seed evaluation. Our analysis *relates* these gains to cue coverage without yet establishing causation. The broader message is that target-awareness is a genuinely useful inductive bias for hate speech detection, but only when it is adapted to the linguistic surface of the target language instead of being transferred verbatim from English.

## 2. Related Work

### 2.1 Hate speech detection and implicit hate

Automatic hate speech detection has been studied extensively, evolving from lexicon- and feature-based classifiers to deep sequence models and, more recently, pretrained transformers. A persistent difficulty is *implicit* hate: messages that convey hostility without explicit slurs, relying instead on sarcasm, stereotypes, coded references, or in-group knowledge. Implicit hate is both more common than overt slurs in many real settings and substantially harder for keyword- or lexicon-driven systems, which motivates approaches that reason about *who* is targeted and *how*.

### 2.2 AmpleHate

AmpleHate addresses implicit hate by amplifying the attention between a sentence's global representation and its candidate targets. Concretely, it extracts target tokens with NER, computes a HeadAttention vector in which the query derives from the `[CLS]` representation and the keys/values derive from target tokens, and injects the result into the sentence representation with a fixed coefficient before classification. AmpleHate's central insight — that the target is a load-bearing signal for implicit hate — directly motivates our work. Its limitations for Vietnamese are equally direct: it presumes English-style NER and target categories, it injects relation information at a fixed strength for every instance, and it models a single relation (target) while ignoring the hostile predicate, or *attack*, that turns a mention into hate.

### 2.3 Vietnamese hate speech detection

Vietnamese has seen growing interest in hate speech resources and models. ViHSD provides a large-scale, three-way (CLEAN/OFFENSIVE/HATE) dataset of social-media comments and is a standard benchmark. ViTHSD reframes the task around *targets*, annotating the degree of hatred directed at each target in a comment; this reinforces the centrality of the target in Vietnamese hate speech and supports our target-aware design. Unlike ViTHSD, ViAmpleHate does not require span-level target annotations; it instead locates target and attack positions with a combination of NER and cue banks, which keeps the approach applicable to datasets that carry only sentence-level labels.

### 2.4 PhoBERT and Vietnamese pretrained models

PhoBERT is a RoBERTa-style model pretrained on large Vietnamese corpora and is the de facto encoder for Vietnamese text classification. Because PhoBERT is trained on *word-segmented* text, a word-segmentation step is required before tokenization. Building on PhoBERT, hybrid architectures such as PhoBERT-CNN add local feature extractors on top of contextual embeddings. We use PhoBERT-base as the shared encoder for both our baseline and proposed models so that observed differences are attributable to the target-aware modeling rather than the backbone.

### 2.5 Contrastive learning for text classification

Contrastive objectives encourage representations of same-class examples to be close and those of different-class examples to be far apart, and they have been shown to improve robustness and class separation, particularly under imbalance. We adopt a contrastive auxiliary objective on the post-gate representation alongside weighted cross-entropy — a pairwise cosine-margin term inspired by supervised contrastive learning, not the exact SupCon loss — aiming to tighten the boundary between the frequent NON-HATE class and the rare HATE class.

## 3. Datasets

### 3.1 ViHSD

ViHSD is a Vietnamese hate speech detection dataset of social-media comments annotated with three original labels — CLEAN, OFFENSIVE, and HATE — and pre-split into training, validation, and test partitions. After the binary relabeling described in Section 3.3, the class distribution of each split is given in Table 1.

### 3.2 VOZ-HSD

VOZ-HSD, released with the ViHateT5 paper, consists of comments collected from the VOZ forum, originating from the `tarudesu/VOZ-HSD` set on HuggingFace. As with ViHSD, the data is split into training, validation, and test partitions, and its post-relabeling class distribution is reported in Table 1. As a third-party resource, its detailed annotation procedure and licensing are documented by the dataset authors.

### 3.3 Binary relabeling and preprocessing

The only modification we make to the data is *relabeling*; we do not alter or subsample the number of examples. We cast the task as binary classification by merging CLEAN and OFFENSIVE into a single NON-HATE class and keeping HATE as the positive class. This focuses the problem on hate that is directed at a group or target, as opposed to profanity or aggression in general, and yields a consistent label space across the two datasets.

Each comment is then normalized to reduce social-media noise. Normalization lowercases text; removes URLs and non-linguistic symbols; collapses elongated character runs; normalizes common teencode and spelling variants; and maps a set of frequent emojis to coarse pragmatic tags such as mockery, anger, disgust, or laughter. The normalized text is word-segmented — mandatory because PhoBERT is pretrained on word-segmented Vietnamese — and then tokenized with the PhoBERT tokenizer and truncated or padded to a fixed length.

**Table 1 — Dataset statistics after binary relabeling.**

| Dataset | Split | NON-HATE | HATE | Total | % HATE |
|---|---|---:|---:|---:|---:|
| ViHSD | Train | 21,492 | 2,556 | 24,048 | 10.6% |
| ViHSD | Dev | 2,402 | 270 | 2,672 | 10.1% |
| ViHSD | Test | 5,992 | 688 | 6,680 | 10.3% |
| VOZ-HSD | Train | 26,993 | 3,007 | 30,000 | 10.0% |
| VOZ-HSD | Dev | 4,520 | 480 | 5,000 | 9.6% |
| VOZ-HSD | Test | 4,487 | 513 | 5,000 | 10.3% |

> 📌 **[INSERT FIGURE] Figure 1 — Class distribution.** Bar chart of NON-HATE vs HATE on both datasets (numbers from Table 1). **Draw new** (matplotlib). Caption should stress the ~10% HATE rate and the resulting choice of macro-F1/HATE-F1 over accuracy.

### 3.4 Cue bank construction

The target-cue and attack-cue banks are central to ViAmpleHate, so we describe how they are built. Both banks are seeded manually from inspection of the training split and from common Vietnamese referential and offensive expressions, then normalized to match the preprocessing pipeline. The **target-cue bank** contains referential expressions that often introduce a person or group — pronouns, kinship and group nouns, regional and demographic labels, and informal forms of address — but that are *not* by themselves indicators of hate. The **attack-cue bank** contains hostile predicates: insults, dehumanizing terms, threats, and strongly negative evaluations (for example, predicates expressing contempt or parasitism). Keeping the two banks separate is deliberate, since a target mention without a hostile predicate is usually not hate, and a hostile predicate without a clear group target is often mere offensiveness rather than hate. Each cue phrase is normalized, word-segmented, and tokenized with the PhoBERT tokenizer, and is matched against the token stream by full token-sequence matching so that a multi-token Vietnamese phrase is matched as a unit rather than via a stray subtoken. The banks are deliberately small, hand-curated lexicons — 16 target cues and 24 attack cues (Appendix B) — seeded from inspection of the training split and common Vietnamese expressions. Because they are seeded from training data they risk encoding train-specific patterns, and their size bounds recall on novel slang; we report the resulting coverage in §6.2 and revisit these constraints in the Limitations.

## 4. Methodology

### 4.1 Problem formulation

Given a Vietnamese comment `x`, the model predicts a binary label `y ∈ {NON-HATE, HATE}`. ViAmpleHate inherits AmpleHate's target-aware idea and adapts it through five changes: Vietnamese target extraction, separate target/attack cue channels, relation-bank attention, a corrected batched attention computation, and adaptive-gate injection. Figure 2 gives the overall architecture.

> 📌 **[INSERT FIGURE] Figure 2 — Overall architecture of ViAmpleHate.** The most important figure; **draw new** (draw.io/PowerPoint), full-width. Show: comment → normalization → word segmentation → PhoBERT → `[CLS]` `h₀` and token states `H`; branch A (Vietnamese NER + target-cue bank → `T_x`) and branch B (attack-cue bank → `A_x`); three relation-bank attention channels `r_exp / r_imp / r_atk` → fuse via `W_r`; adaptive gate `g = σ(W_g[h₀;r])` → `z = h₀ + g·r` → linear classifier → {NON-HATE, HATE}. Annotate the loss `L = L_CE + α·L_CL`.

### 4.2 Text preprocessing

As in Section 3.3, comments are normalized (lowercasing; URL/symbol removal; elongation collapsing; teencode normalization; emoji-to-tag mapping), word-segmented, tokenized with the PhoBERT tokenizer, and truncated or padded to `max_len = 256`.

### 4.3 Multi-signal target and attack extraction

The target index set combines Vietnamese NER and target-cue matching, while the attack index set comes from attack-cue matching. If either set is empty, the model falls back to the `[CLS]` position so that an implicit, sentence-level representation is always available:

$$
T_x = M_{\text{NER}}(x)\,\cup\,M_{\text{target}}(x), \qquad A_x = M_{\text{attack}}(x)
$$
$$
T_x \leftarrow \{0\}\ \text{if}\ T_x=\varnothing, \qquad A_x \leftarrow \{0\}\ \text{if}\ A_x=\varnothing
$$

### 4.4 PhoBERT encoder

$$
H = \text{PhoBERT}(x) = [\,h_0, h_1, \dots, h_n\,], \quad h_i \in \mathbb{R}^{d},\ d=768,
$$

where `h_0` is the `[CLS]` representation capturing global context, and the target/attack positions provide localized evidence.

### 4.5 Relation-bank attention

We build three relation views — explicit target, implicit context (from the `[CLS]` anchor), and attack:

$$
r_{\text{exp}} = \text{HeadAttn}(h_0, H[T_x]), \quad
r_{\text{imp}} = \text{HeadAttn}(h_0, h_0), \quad
r_{\text{atk}} = \text{HeadAttn}(h_0, H[A_x])
$$

with each HeadAttention over a relation matrix `E ∈ R^{m×d}`:

$$
Q = W_q h_0,\quad K = W_k E,\quad V = W_v E,\quad
\alpha = \text{softmax}\!\Big(\frac{QK^\top}{\sqrt d}\Big),\quad r = \alpha V.
$$

The three vectors are concatenated and projected into a fused relation representation:

$$
r = W_r\,[\,r_{\text{exp}}\,;\,r_{\text{imp}}\,;\,r_{\text{atk}}\,] + b_r.
$$

### 4.6 Batched attention (implementation note)

In our re-implementation, an AmpleHate-style attention computed as `QKᵀ` over the whole batch (`Q,K ∈ R^{B×d}` ⇒ `QKᵀ ∈ R^{B×B}`) can mix information across samples. We therefore use batched attention so that each sample attends only to its own tokens, with `Q ∈ R^{B×1×d}` and `K ∈ R^{B×m×d}`:

$$
\text{scores} = \frac{\text{bmm}(Q,K^\top)}{\sqrt d}\in\mathbb{R}^{B\times1\times m},
$$

and padding positions are masked before softmax (`scores_j = -∞` if `mask_j = 0`). We treat this as an implementation correction to our own port rather than a claim about the original AmpleHate, and report it as an implementation note rather than evaluating it in isolation.

### 4.7 Instance-adaptive relation gate

We replace AmpleHate's fixed scalar with a per-sample gate:

$$
g = \sigma\big(W_g\,[\,h_0\,;\,r\,] + b_g\big),\qquad z = h_0 + g\cdot r.
$$

When a comment carries clear target and attack evidence, the model can raise `g` and rely more on the relation vector; when cues are weak, ambiguous, or absent, it can lower `g` and lean on the global representation. The fused representation `z` passes through dropout and a linear layer to produce logits.

### 4.8 Training objective

We train with weighted cross-entropy combined with a contrastive loss, `L = L_CE + α·L_CL` with `α = 0.1`. Weighted cross-entropy `L_CE = -Σ_c w_c y_c log ŷ_c` handles imbalance by up-weighting the minority HATE class, with label smoothing to curb overconfidence (the class weights `w_c`, the label-smoothing value, and the margin `m` are reported in Appendix A). The contrastive term, applied to the post-gate `z` with cosine similarity `s_ij = cos(z_i, z_j)`, pulls same-class pairs together and pushes different-class pairs apart:

$$
L_{\text{CL}} = \frac{1}{N}\sum_{i\neq j}\Big[\mathbb{1}[y_i{=}y_j](1-s_{ij}) + \mathbb{1}[y_i{\neq}y_j]\max(0, s_{ij}-m)\Big].
$$

This is a pairwise cosine-margin objective inspired by supervised contrastive learning, not the exact SupCon loss.

### 4.9 Inference and threshold selection

At inference we compute the HATE probability and apply a decision threshold chosen on the validation set by maximizing macro-F1, then fixed for the test set:

$$
t^{*} = \arg\max_t\ \text{MacroF1}\big(y,\ \mathbb{1}[p_{\text{HATE}}\ge t]\big).
$$

### 4.10 Computational considerations

The added components are lightweight relative to the PhoBERT encoder. The three HeadAttention modules and the gate operate on the pooled `[CLS]` query and a small number of cue tokens per sample, so the extra cost is dominated by a few linear projections; the corrected batched attention removes the spurious `B×B` interaction and is in fact cheaper than the original at training time. Cue matching is performed once during preprocessing.

## 5. Experiments

### 5.1 Baselines

We compare ViAmpleHate against five baselines, all in the binary setting. **TF-IDF + Logistic Regression** and **TF-IDF + SVM** use sparse lexical features and provide non-neural reference points. **BiLSTM + fastText** combines static Vietnamese embeddings with a recurrent sequence model. **PhoBERT-CNN** stacks convolutional feature extractors on PhoBERT embeddings. Finally, **AmpleHate-PhoBERT** is a port of the original AmpleHate with English NER, a single HeadAttention module, and fixed injection `z = h_0 + e·r_base` with `e = 1.0`. It shares the PhoBERT-base encoder. As Table 2 makes explicit, each model is run at its own intended configuration — ViAmpleHate deliberately uses a longer context window (256) and a larger effective batch, which are part of the proposed system — so the comparison is between the full proposed system and the baseline as configured, rather than an isolation of individual components. Table 2 summarizes the differences.

**Table 2 — Architecture and configuration comparison.**

| Component | Baseline AmpleHate-PhoBERT | ViAmpleHate-PhoBERT (proposed) |
|---|---|---|
| Encoder | PhoBERT-base | PhoBERT-base |
| Target extraction | English NER (`dbmdz/bert-large-...-conll03-english`) | Vietnamese NER (`NlpHUST/ner-vietnamese-electra-base`) + target-cue bank |
| Target coverage | ~21/24,048 ≈ 0.09% of train | ~18.8–20.0% via Vietnamese cues |
| Attack signal | Not modeled | Separate attack-cue bank |
| Attention | One HeadAttention | Three channels: target / implicit / attack |
| Attention computation | Batch-level `matmul` (mixes samples) | Per-sample `bmm` + masking |
| Fusion | Fixed injection `h₀ + e·r` (`e = 1.0`) | Relation bank + adaptive gate `h₀ + g·r` |
| Loss | Weighted CE | Weighted CE + contrastive (`α = 0.1`) |
| Max length | 128 | 256 |
| Batch | 16 | 16 × grad-accum 2 (eff. 32) |
| NER at evaluation | Off (⇒ `[CLS]` fallback) | On, consistent across splits |

### 5.2 Implementation details

The encoder is `vinai/phobert-base` (hidden size 768) and the Vietnamese NER is `NlpHUST/ner-vietnamese-electra-base`. The maximum sequence length is 256. Learning rates are 2e-5 for the encoder and 5e-5 for the classification head, with dropout 0.1. The effective batch size is 32 (batch 16 with two-step gradient accumulation). We train for up to eight epochs and select the best checkpoint by validation macro-F1. The contrastive weight α is 0.1, and the decision threshold is selected on validation by macro-F1. Full hyperparameters are in Appendix A.

### 5.3 Evaluation metrics

Our primary metrics are macro-F1 and HATE-class F1, which reflect minority-class performance far better than accuracy. We report accuracy only for reference: under ~10% positives, a trivial majority predictor already exceeds 0.89 accuracy while detecting no hate. Macro-F1 averages the per-class F1 scores, and HATE-F1 isolates performance on the class of interest.

### 5.4 Experimental design

Beyond comparing final scores, our experiments are designed to test whether each modification addresses a concrete limitation of the baseline. Vietnamese NER and target cues target low coverage; the attack-cue bank addresses the absence of hostility modeling; relation-bank attention separates target, context, and attack; the corrected batched attention removes cross-sample leakage; the adaptive gate replaces fixed injection; and the contrastive loss improves class separation under imbalance. Section 6.2 examines target coverage directly.

## 6. Results and Analysis

### 6.1 Main results

Each model is reported at its own intended configuration (§3.3, Table 2); the proposed-vs-baseline rows therefore compare the full proposed system against the baseline as configured. All numbers are from a single run (seed 42).

**Table 3 — Results on ViHSD (test set). Best per column in bold (accuracy for reference). Single run (seed 42).**

| Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.8910 | 0.7393 | 0.5404 |
| TF-IDF + SVM | 0.9126 | 0.7131 | 0.4739 |
| BiLSTM + fastText | 0.8454 | 0.7072 | 0.5060 |
| PhoBERT-CNN | 0.8945 | 0.7571 | 0.5745 |
| AmpleHate-PhoBERT (baseline) | 0.9175 | 0.7792 | 0.6045 |
| **ViAmpleHate-PhoBERT (ours)** | **0.9205** | **0.7819** | **0.6081** |
| *Δ vs baseline* | *+0.0030* | *+0.0027* | *+0.0036* |

**Table 4 — Results on VOZ-HSD (test set). Best per column in bold among models (accuracy for reference; the baseline has the highest accuracy). Single run (seed 42).**

| Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.9453 | 0.7745 | 0.5783 |
| TF-IDF + SVM | 0.9641 | 0.7831 | 0.5850 |
| BiLSTM + fastText | 0.8650 | 0.6712 | 0.4187 |
| PhoBERT-CNN | 0.9623 | 0.8150 | 0.6500 |
| AmpleHate-PhoBERT (baseline) | 0.9643 | 0.8185 | 0.6557 |
| **ViAmpleHate-PhoBERT (ours)** | 0.9420 | **0.8371** | **0.7065** |
| *Δ vs baseline* | *–0.0223* | *+0.0186* | *+0.0508* |

Transformer-based models clearly outperform the lexical and static-embedding baselines on the metrics that matter, confirming that contextual representations are important for a task in which hate often depends on context, informal phrasing, and the interaction between a mention and a hostile predicate. Within the transformer family, the AmpleHate baseline improves over a plain PhoBERT classifier through target-aware attention, but its benefit is capped because English NER rarely detects valid Vietnamese targets, so most inputs revert to `[CLS]`. We compare only against models we trained under a common protocol; a comparison to published ViHSD results is left to future work.

ViAmpleHate improves macro-F1 and HATE-F1 on VOZ-HSD by a clear margin (HATE-F1 +0.0508). On ViHSD the differences are small (+0.0027 macro-F1, +0.0036 HATE-F1); since all reported numbers come from a single run (seed 42) and we do not run a significance test, we treat the ViHSD margin as preliminary rather than established. Accuracy *drops* on VOZ-HSD while macro-F1 and HATE-F1 rise, illustrating why accuracy is the wrong headline metric under imbalance.

### 6.2 Effect of Vietnamese target extraction

A likely contributor to these gains is target coverage, which we define as the fraction of comments for which `T_x ≠ {0}` (Vietnamese NER or a target cue fires). Under English NER, only ~0.09% of training comments contain a detected target (21/24,048), so the baseline's target-aware pathway is almost never active. With Vietnamese NER and the target-cue bank, target coverage rises to 20.0%/18.8% on the first 500 ViHSD train/validation comments and to 45.2%/43.0% on VOZ-HSD using the same 16-cue bank; attack-cue coverage is lower — 5.2%/4.2% (ViHSD) and 7.6%/6.0% (VOZ). Notably, the dataset with far higher coverage (VOZ-HSD) is also where ViAmpleHate gains most (HATE-F1 +0.0508 vs. +0.0036), a suggestive cross-dataset correlation. We stress, however, that coverage is an *input* statistic: higher coverage gives the relation-bank attention more to attend to but does not by itself prove that firing improves individual decisions.

> 📌 **[INSERT FIGURE] Figure 3 — Training curves.** Use the existing PNG `notebooks/models/proposed/ViHSD - Proposed ViAmpleHate_PhoBERT/output/training_curves_viamplehate.png` (optionally beside the baseline curves). Caption: best epoch 4, validation macro-F1 = 0.7852.

### 6.3 Qualitative analysis

To make the error modes concrete, Table 6 shows illustrative comment types (constructed examples that mirror the categories observed; replace with real anonymized test instances for camera-ready). The pattern is that ViAmpleHate is most helpful when an explicit target co-occurs with an attack predicate, and least helpful for implicit hate that names no target and uses no overt attack term.

**Table 6 — Illustrative cases (constructed; replace with real examples).**

| Comment type | Target cue? | Attack cue? | Gold | Tendency |
|---|---|---|---|---|
| Group label + hostile predicate | yes | yes | HATE | correctly HATE |
| Profanity, no group target | no | yes | NON-HATE | risk of false positive |
| Sarcastic/implicit, no overt cue | no | no | HATE | risk of false negative |
| Group label in neutral/humorous context | yes | no | NON-HATE | usually correct via gate |

> 📌 **[INSERT FIGURE] Figure 5 — Confusion matrices (ViHSD): baseline vs ViAmpleHate.** Side by side (full-width). Use `confusion_matrix_amplehate.png` and `confusion_matrix_viamplehate.png` from the respective `output/` folders; highlight the reduction in HATE false positives.

**Table 7 — Per-class results of ViAmpleHate (test set, single run).**

| Data | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| ViHSD | NON-HATE | 0.9541 | 0.9574 | – |
| ViHSD | HATE | 0.6177 | 0.5988 | 0.6081 |
| VOZ | NON-HATE | 0.9638 | 0.9719 | 0.9678 |
| VOZ | HATE | 0.7347 | 0.6803 | 0.7065 |

### 6.4 Discussion

Three observations stand out. First, on ViHSD the change is a precision/recall reweighting rather than a uniform gain: HATE precision rises (0.5972 → 0.6177) while recall *falls* (0.6119 → 0.5988). Higher precision is desirable when false accusations of hate are costly, but lower recall means more hate is missed; which trade-off is appropriate depends on the deployment. We report results at a single validation-tuned threshold and leave a full precision–recall analysis to future work. On VOZ-HSD, where coverage is higher, HATE recall is also higher (0.6803; Table 7). Second, the larger VOZ-HSD gains alongside an accuracy drop indicate that the model trades majority-class accuracy for minority-class quality, which is the trade-off practitioners typically want when the minority class is the target of interest. Third, the relationship between cue coverage and gains suggests a possible inexpensive path to further improvement — expanding and refining the cue banks — though this remains to be verified (§6.2).

## 7. Error Analysis

We group the remaining errors into seven recurring categories.

**Offensive but not hate.** Many Vietnamese comments contain insults or profanity aimed at an individual rather than a protected group; if the model over-weights attack cues, it labels them HATE, producing false positives.

**Implicit hate.** Some hateful comments carry no slur, named entity, or overt attack predicate, expressing hostility through sarcasm, stereotypes, or shared social context; cue-based extraction finds little to attend to, and the model must rely on `[CLS]`.

**Ambiguous target reference.** Pronouns and group nouns also occur in neutral or humorous comments, so detecting a target cue without hostile context can lead to over-prediction of HATE.

**Incomplete cue coverage.** Vietnamese online language evolves quickly, with abundant spelling variants, slang, abbreviations, and creative or censored profanity; a fixed cue bank cannot cover all of it, causing false negatives.

**Tokenization and span-alignment errors.** NER spans, word segmentation, and PhoBERT subword tokenization do not always align — especially for multi-word expressions — so a cue may match the wrong position and the attention module may attend to incomplete or irrelevant evidence.

**Threshold sensitivity.** A threshold tuned on validation may not remain optimal when the test distribution or domain shifts, which is consequential at deployment.

**Class imbalance.** With few positive examples, the model sees limited hate during training; weighted loss and threshold tuning reduce but do not eliminate minority-class errors.

These patterns point to concrete improvements: expand the cue banks through data-driven mining plus manual validation; log per-instance predictions (with gate value, cue coverage, and confidence) for systematic false-positive/negative analysis; add target-span supervision, sarcasm detection, or social-context features; and apply probability calibration for more stable thresholds.

## 8. Conclusion and Future Work

ViAmpleHate adapts AmpleHate to Vietnamese hate speech detection through Vietnamese target extraction, separate target/attack cue channels, relation-bank attention, and adaptive relation injection (plus an implementation-level batched-attention correction), trained with weighted cross-entropy plus a contrastive loss. On ViHSD and VOZ-HSD in the binary setting it improves macro-F1 and HATE-F1 over the AmpleHate baseline and over TF-IDF, BiLSTM, and PhoBERT-CNN baselines, with the clearest gains on VOZ-HSD's minority class; the ViHSD improvements are small and remain to be confirmed under a controlled, multi-seed comparison. Our analysis relates these gains to target coverage, which the Vietnamese cue banks raise by roughly two orders of magnitude over English NER — a correlational link we have not yet shown to be causal. The broader lesson is that target-awareness is a useful inductive bias only when adapted to the target language's surface forms, where hate targets are often informal group references rather than named entities.

In future work we plan to expand and automatically mine the target and attack cue banks with manual validation; analyse errors at the per-instance level using the gate value and prediction confidence; add span supervision, sarcasm detection, and user/discourse context; extend from the binary setting to multi-label and multi-target classification in the spirit of ViTHSD; and apply probability calibration for stable cross-domain deployment.

## Limitations

The target and attack cue banks are small (16 and 24 entries) and hand-built, and cannot cover the full, fast-changing space of Vietnamese slang, spelling variants, and creative profanity, which bounds recall on implicit or novel hate. Performance depends on the quality of upstream Vietnamese NER, word segmentation, and tokenization, whose misalignment can misplace cues. The decision threshold is tuned on validation and may not be optimal under distribution shift, and reporting at a single threshold does not expose the full precision–recall trade-off. All results come from a single run (seed 42); we do not report multi-seed variance or significance tests, and we do not ablate the individual components, so component-level contributions remain design hypotheses and the small ViHSD margin is not established. We compare against our own re-implementations rather than published ViHSD results. Finally, the model addresses a binary NON-HATE/HATE formulation and does not yet handle multi-target or graded hatred.

## Ethics Statement

Hate speech detection is dual-use: the same models that support content moderation can, if mis-deployed, suppress legitimate speech or disproportionately flag particular communities. Our datasets are existing, publicly described Vietnamese resources used under their intended research terms; we do not collect new user data. Because annotations and cue banks reflect particular annotator and author judgments, the system may carry biases against specific dialects, regions, or groups, and its outputs should be treated as decision support for human moderators rather than as automated enforcement. We avoid reproducing real slurs in the paper, using constructed or masked examples for illustration. We encourage calibration, human-in-the-loop review, and ongoing bias auditing in any deployment.

## References

- **AmpleHate** — Lee, Y., Hahn, J., Ahn, H., Han, Y.-S. (2025). "AmpleHate: Amplifying the Attention for Versatile Implicit Hate Detection." *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (Suzhou, China), pp. 28862–28874. ACL. DOI: 10.18653/v1/2025.emnlp-main.1469.
- **ViTHSD** — Vo, C. N., Huynh, K. B., Luu, S. T., Do, T.-H. (2025). "ViTHSD: Exploiting Hatred by Targets for Hate Speech Detection on Vietnamese Social Media Texts." *Journal of Computational Social Science* 8(2):30. Springer. DOI: 10.1007/s42001-024-00348-6.
- **ViHSD** — Luu, S. T., Nguyen, K. V., Nguyen, N. L.-T. (2021). "A Large-Scale Dataset for Hate Speech Detection on Vietnamese Social Media Texts." *Advances and Trends in Artificial Intelligence. Artificial Intelligence Practices* (LNCS 12798), pp. 415–426. Springer, Cham. DOI: 10.1007/978-3-030-79457-6_35.
- **VOZ-HSD** — Thanh Nguyen, L. (2024). "VOZ-HSD: A Hate Speech Detection Dataset from the VOZ Forum." Hugging Face dataset, `tarudesu/VOZ-HSD`. Released with ViHateT5.
- **ViHateT5** — Thanh Nguyen, L. (2024). "ViHateT5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model." *Findings of the ACL 2024* (Bangkok, Thailand), pp. 5948–5961. ACL. DOI: 10.18653/v1/2024.findings-acl.355.
- **PhoBERT** — Nguyen, D. Q., Nguyen, A. T. (2020). "PhoBERT: Pre-trained language models for Vietnamese." *Findings of EMNLP 2020*.
- **BERT** — Devlin et al. (2019). *NAACL*.
- **fastText** — Bojanowski et al. (2017). *TACL*.
- **Supervised Contrastive Learning** — Khosla et al. (2020). *NeurIPS*.
- **Vietnamese NER** — `NlpHUST/ner-vietnamese-electra-base` *(fill in citation/URL)*.

---

## Appendix A — Hyperparameters

| Parameter | Value |
|---|---|
| Encoder | `vinai/phobert-base` (768-d) |
| Vietnamese NER | `NlpHUST/ner-vietnamese-electra-base` |
| max_len | 256 |
| LR (encoder / head) | 2e-5 / 5e-5 |
| Dropout | 0.1 |
| Effective batch | 32 (16 × grad-accum 2) |
| Epochs | up to 8 (best by val macro-F1) |
| α (contrastive) | 0.1 |
| Class weights `w_c` | inverse frequency `N / (C·n_c)` |
| Contrastive margin `m` | 0.5 |
| Label smoothing | 0.05 |
| Seed / runs | 42 / single run |
| ViHSD best epoch / val macro-F1 / threshold | 4 / 0.7852 / 0.43 |

## Appendix B — Cue banks (full)

The banks are hand-curated lexicons matched by full token-sequence matching after word segmentation. Target cues are referential (not hateful by themselves); attack cues are hostile predicates.

- **Target cues (16):** `bọn, thằng, con, đứa, tụi, đám, lũ, mấy, loại, người, dân, bên, hắn, chúng, họ, nó`.
- **Attack cues (24):** `ngu, đần, ngu_ngốc, khùng, điên, hèn, nhục, ăn_bám, ký_sinh, phản_quốc, vô_học, man_rợ, cút, xéo, câm_miệng, giết, chém, đánh, ghét, khinh, chửi, vô_văn_hóa, thấp_hèn, đáng_chết`.

The same banks are used for both ViHSD and VOZ-HSD. Coverage of these banks differs sharply by dataset (§6.2): target-cue coverage is ~19% on ViHSD vs ~43–45% on VOZ-HSD.

## Appendix C — Reproducibility

All PhoBERT-based models share `vinai/phobert-base`; only the target/relation modeling, attention computation, fusion, and loss differ between the baseline and ViAmpleHate. Best checkpoints are selected by validation macro-F1, and the HATE threshold is selected on validation and fixed for test. Cue matching is performed once during preprocessing using full token-sequence matching against the PhoBERT token stream.
</content>
