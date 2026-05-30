# 4. Methodology

This study addresses Vietnamese hate speech detection as a supervised text classification problem. Given a Vietnamese social media comment, the model predicts whether the comment contains hate speech or belongs to the non-hate class. When a source dataset contains more than two labels, non-hate and offensive-but-not-hate categories are grouped into a single `NON-HATE` class, while explicit hate speech is treated as the positive `HATE` class. This setting focuses the task on detecting hate speech rather than general toxicity or profanity.

The proposed method, ViAmpleHate, adapts the target-aware idea of AmpleHate to Vietnamese. The original AmpleHate framework identifies potential targets with named entity recognition, computes target-aware attention, and injects the resulting signal into the sentence representation before classification. Directly applying this design to Vietnamese is limited because the original pipeline depends on English NER and English-centric target categories. ViAmpleHate modifies the pipeline by using Vietnamese target extraction, separating target cues from attack cues, and replacing fixed attention injection with an adaptive relation gate.

## 4.1 Text Preprocessing

Vietnamese social media text is noisy and contains abbreviations, teencode, spelling variation, emojis, and informal punctuation. We therefore apply a normalization pipeline before model encoding. The preprocessing step lowercases text, removes URLs and other non-linguistic artifacts, collapses repeated characters, normalizes common teencode variants, and maps selected emojis to coarse pragmatic tags such as mockery, anger, disgust, laughter, or intensity.

After normalization, each comment is word-segmented using a Vietnamese word segmentation tool. Word segmentation is required because PhoBERT is pretrained on word-segmented Vietnamese text. The segmented comment is then tokenized with the PhoBERT tokenizer and truncated or padded to a fixed maximum length.

## 4.2 Multi-Signal Target and Attack Extraction

ViAmpleHate uses a multi-signal extraction strategy to identify linguistically meaningful positions in the input sequence. The first signal comes from Vietnamese NER, which extracts named entities that may serve as hate targets, such as persons, organizations, locations, geopolitical entities, and other named groups.

NER alone is not sufficient for Vietnamese hate speech because many targets are expressed through common nouns, pronouns, or group references rather than named entities. The model therefore adds a target cue bank containing Vietnamese referential expressions that often introduce a person or group target. These cues are not considered hate indicators by themselves; they only mark possible target positions.

The model also uses an attack cue bank containing offensive predicates, negative evaluations, threats, and hostile expressions. Target cues and attack cues are deliberately separated. A target cue helps answer who or what is being discussed, while an attack cue helps identify whether hostile evaluation is being directed at that target. This separation prevents the model from treating every offensive word as a target mention and every target reference as an attack.

Each cue phrase is normalized, word-segmented, tokenized, and matched against the PhoBERT token sequence using full token-sequence matching. The extraction module returns two sets of token positions: target positions and attack positions. If no explicit cue is found, the corresponding set falls back to the `[CLS]` position, allowing the model to retain an implicit sentence-level representation.

## 4.3 PhoBERT Encoder

The normalized and segmented input is encoded with PhoBERT. Let the final hidden states be:

```text
H = [h_0, h_1, ..., h_n]
```

where `h_0` is the `[CLS]` representation and `h_i` denotes the contextual embedding of the `i`-th token. The `[CLS]` vector represents the global sentence context, while the extracted target and attack positions provide localized evidence for target-aware hate speech reasoning.

## 4.4 Relation-Bank Attention

ViAmpleHate builds a relation bank with three relation views:

1. An explicit target relation from extracted target tokens.
2. An implicit context relation from the `[CLS]` anchor.
3. An attack relation from extracted attack tokens.

Each relation view is processed by a separate HeadAttention module. For a relation-specific token embedding matrix `E_k` and the `[CLS]` vector `h_cls`, attention is computed as:

```text
Q_k = W_q h_cls
K_k = W_k E_k
V_k = W_v E_k
a_k = softmax(Q_k K_k^T / sqrt(d))
r_k = a_k V_k
```

where `k` denotes the relation type. In implementation, attention is computed with batched matrix multiplication so that each sample attends only to its own extracted tokens. This avoids cross-sample interaction within a mini-batch.

The three relation vectors are concatenated and projected into a single relation representation:

```text
r = W_r [r_target; r_implicit; r_attack]
```

This fused relation vector captures complementary information from target mentions, sentence-level context, and attack expressions.

## 4.5 Instance-Adaptive Relation Gate

The original AmpleHate mechanism uses a fixed scalar to control how much target-aware attention is injected into the sentence representation. ViAmpleHate replaces this fixed injection strength with an instance-adaptive gate:

```text
g = sigmoid(W_g [h_cls; r])
z = h_cls + g * r
```

The gate allows the model to decide how strongly relation information should affect each individual prediction. If a comment contains clear target and attack evidence, the model can assign more weight to the relation vector. If the extracted cues are weak, ambiguous, or absent, the model can rely more on the global sentence representation.

The final representation `z` is passed through dropout and a linear classification layer to produce logits for `NON-HATE` and `HATE`.

## 4.6 Training Objective

The model is trained with a combination of weighted cross-entropy and contrastive loss:

```text
L = L_CE + alpha * L_CL
```

The weighted cross-entropy term addresses class imbalance by assigning higher weight to the minority hate class. Label smoothing is used to reduce overconfident predictions. The contrastive term is applied to the post-gate representation `z`, encouraging examples from the same class to have closer representations and examples from different classes to be farther apart under cosine similarity.

During evaluation, the classification objective is used to compute prediction probabilities. The final threshold for the `HATE` class is selected on the validation set by maximizing macro-F1 and is then applied to the held-out test set.

## 4.7 Method Summary

ViAmpleHate generalizes AmpleHate for Vietnamese hate speech detection through five main adaptations: Vietnamese-aware target extraction, separate modeling of target and attack cues, relation-bank attention, corrected batch-wise attention computation, and adaptive relation injection. These changes make the model better suited to Vietnamese social media language, where hate targets are often expressed through informal group references rather than standard named entities.
