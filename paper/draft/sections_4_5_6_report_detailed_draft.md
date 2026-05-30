# 4. Methodology

We formulate Vietnamese hate speech detection as a binary text classification task. Given an input comment, the model predicts one of two labels: `NON-HATE` or `HATE`. The `NON-HATE` class includes normal text and offensive text that does not express hate toward a specific target group, while the `HATE` class contains comments that attack or discriminate against a person or group based on identity, social group, region, ethnicity, gender, religion, or other target attributes.

The proposed model, ViAmpleHate, is built on the idea of AmpleHate, which improves hate speech detection by focusing on the relationship between a sentence and its potential hate target. However, the original AmpleHate pipeline is designed mainly for English and relies on English named entity recognition. This is not suitable for Vietnamese social media text, where hate targets are often expressed through informal words, pronouns, group nouns, slang, or implicit references rather than standard named entities. Therefore, ViAmpleHate adapts AmpleHate by adding Vietnamese-specific target extraction, attack cue detection, relation-aware attention, and adaptive information fusion.

First, each input comment is normalized to reduce social media noise. The preprocessing step lowercases text, removes URLs and irrelevant symbols, collapses repeated characters, normalizes common Vietnamese teencode, and maps some emojis to coarse semantic tags such as anger, mockery, disgust, or laughter. After normalization, the text is word-segmented because PhoBERT expects Vietnamese input in word-segmented form. The segmented sentence is then tokenized using the PhoBERT tokenizer and encoded by `vinai/phobert-base`.

Second, ViAmpleHate extracts two types of linguistic signals: target cues and attack cues. Target cues indicate possible targets of hate speech, such as people, groups, pronouns, or social categories. These are obtained from Vietnamese NER and a manually defined Vietnamese target cue bank. Attack cues indicate hostile predicates or offensive expressions, such as insults, threats, dehumanizing words, or negative evaluations. The model keeps target cues and attack cues separate because they represent different roles. A target cue answers who is being referred to, while an attack cue answers whether that target is being attacked.

Third, ViAmpleHate uses relation-bank attention to model different kinds of hate-related information. It builds three relation representations: an explicit target relation from detected target tokens, an implicit relation from the global `[CLS]` representation, and an attack relation from detected attack tokens. Each relation is computed using a separate HeadAttention module. This allows the model to attend differently to target mentions, sentence-level context, and attack expressions.

After obtaining the three relation vectors, the model concatenates and projects them into a single fused relation vector. Instead of injecting this vector into the sentence representation with a fixed weight, ViAmpleHate uses an adaptive gate. The gate is computed from both the `[CLS]` representation and the fused relation vector. This allows the model to decide how much relation information should influence each prediction. If the comment contains clear target and attack evidence, the gate can increase the effect of relation information. If the extracted cues are weak or ambiguous, the model can rely more on the original PhoBERT sentence representation.

Finally, the gated representation is passed through a dropout layer and a linear classifier to predict `NON-HATE` or `HATE`. The model is trained with weighted cross-entropy to handle class imbalance. In addition, contrastive loss is used to encourage examples from the same class to have closer representations and examples from different classes to be more separable. This helps the model learn a clearer boundary between hate and non-hate comments.

# 5. Experiments

The experiments evaluate whether the proposed ViAmpleHate_PhoBERT improves the direct AmpleHate_PhoBERT baseline through concrete Vietnamese-specific model changes. Both models use the same PhoBERT encoder, so the comparison focuses on the proposed target extraction, relation modeling, attention correction, adaptive fusion, and loss design.

Given an input comment `x`, PhoBERT produces contextual representations:

```text
H = PhoBERT(x) = [h_0, h_1, ..., h_n],     h_i in R^d
```

where `h_0` is the `[CLS]` vector. The baseline extracts target indices `T` using English NER and computes one target-aware relation vector. If no target is detected, it falls back to `T = {0}`. The baseline representation is:

```text
r_base = HeadAttn(h_0, H[T])
z_base = h_0 + e * r_base
y_hat = softmax(W_c z_base + b_c)
```

where `e` is a fixed injection scalar. This is weak for Vietnamese because English NER rarely detects valid Vietnamese targets, so the model often becomes close to a standard PhoBERT classifier.

ViAmpleHate_PhoBERT first improves target construction. Instead of using only English NER, it constructs target and attack index sets as:

```text
T_x = M_NER(x) union M_target(x)
A_x = M_attack(x)
```

where `M_NER` is Vietnamese NER, `M_target` is Vietnamese target cue matching, and `M_attack` is attack cue matching. Empty sets are replaced by `[CLS]` fallback. In implementation, cue phrases are normalized, word-segmented, tokenized by the PhoBERT tokenizer, and matched by full token-sequence matching.

The proposed model then replaces single attention with relation-bank attention:

```text
r_exp = HeadAttn(h_0, H[T_x])
r_imp = HeadAttn(h_0, h_0)
r_atk = HeadAttn(h_0, H[A_x])
r = W_r [r_exp ; r_imp ; r_atk] + b_r
```

Each HeadAttention module computes:

```text
Q = W_q h_0
K = W_k E
V = W_v E
alpha = softmax(Q K^T / sqrt(d))
r = alpha V
```

This design separately models explicit target information, implicit context, and attack information before fusing them.

The implementation also corrects batch attention. Instead of computing `QK^T` across the whole batch, ViAmpleHate uses batched attention:

```text
scores = bmm(Q, K^T) / sqrt(d)
```

where each sample attends only to its own cue tokens. Padding positions are masked before softmax.

Finally, the proposed model replaces fixed scalar injection with an adaptive gate:

```text
g = sigmoid(W_g [h_0 ; r] + b_g)
z = h_0 + g * r
y_hat = softmax(W_c dropout(z) + b_c)
```

The gate lets the model decide how strongly relation information should affect each individual prediction. Training uses weighted cross-entropy plus contrastive loss:

```text
L = L_CE + alpha * L_CL
```

where `L_CE` handles class imbalance and `L_CL` improves separation between `NON-HATE` and `HATE` representations. Validation macro-F1 is used for checkpoint selection and threshold tuning:

```text
t* = argmax_t MacroF1(y, 1[p_HATE >= t])
```

The main evaluation metrics are macro-F1 and HATE-class F1 because they reflect minority-class hate speech detection more directly than accuracy.

# 6. Result Analysis or Error Analysis

## 6.1 Experimental Results

The experimental results show that contextual transformer-based models perform better than traditional lexical models and static-embedding neural models. This indicates that Vietnamese hate speech detection requires contextual understanding. Many hate speech comments cannot be identified by keywords alone because the meaning often depends on sentence context, target references, sarcasm, or the relationship between a target and an attack expression.

TF-IDF models provide useful baseline performance, especially when hate speech contains explicit keywords or repeated offensive expressions. However, they are limited because they treat text mostly as sparse word features. They cannot effectively model word order, long-distance context, or implicit target-attack relations.

BiLSTM-based models improve over simple lexical features by modeling token sequences. However, they still rely on static embeddings, so the same word receives the same representation regardless of context. This is a limitation for Vietnamese social media text, where informal words, pronouns, and offensive terms can have different meanings depending on how they are used.

PhoBERT-based models perform more strongly because PhoBERT provides contextual Vietnamese representations learned from large-scale pretraining. PhoBERT-CNN benefits from both contextual token embeddings and local pattern extraction. The direct AmpleHate_PhoBERT baseline further introduces target-aware attention, but its effectiveness is limited when target extraction is not adapted to Vietnamese.

ViAmpleHate improves on this target-aware direction by making target extraction and attack modeling language-specific. The model does not rely only on named entities; it also considers Vietnamese referential cues and hostile predicates. The relation-bank design allows the model to separately capture target information, implicit sentence context, and attack information before combining them. As a result, the model is better aligned with the structure of hate speech, where the key signal is often the relation between a target and a hostile expression.

The most meaningful improvements are observed in macro-F1 and HATE-class F1. This suggests that ViAmpleHate is better at handling the minority hate class. Accuracy alone is less informative because a model can achieve high accuracy by correctly predicting many `NON-HATE` samples while still missing hate speech. Therefore, improvements in HATE-F1 provide stronger evidence that the proposed model improves the actual hate speech detection objective.

## 6.2 Error Analysis

Although ViAmpleHate improves target-aware hate speech detection, several types of errors remain.

First, the model can confuse offensive language with hate speech. Many Vietnamese social media comments contain insults, profanity, or aggressive language, but not all of them are hate speech. If a comment attacks an individual without targeting a protected or social group, it may belong to `NON-HATE`. However, because the comment still contains strong attack cues, the model may incorrectly classify it as `HATE`.

Second, implicit hate speech remains difficult. Some hate speech comments do not contain explicit target names, slurs, or attack words. Instead, they rely on sarcasm, stereotypes, coded language, or shared social context. In these cases, cue-based extraction may fail to identify clear target or attack tokens, forcing the model to rely mainly on the general `[CLS]` representation.

Third, target cues can be ambiguous. Vietnamese pronouns and group references such as informal person or group markers may appear in both hateful and non-hateful comments. A cue may identify a possible target, but it does not guarantee that hate is present. If the surrounding context is humorous, neutral, or only mildly offensive, the model may still over-predict the `HATE` class.

Fourth, the attack cue bank cannot cover all forms of Vietnamese online hostility. Social media users often create new spelling variants, abbreviations, slang, and censored profanity. If these variants are not normalized or included in the cue bank, the model may miss important attack signals. This can lead to false negatives, especially for comments that express hate using creative or indirect language.

Fifth, tokenization and span alignment can introduce errors. Vietnamese NER, word segmentation, and PhoBERT subword tokenization may not always produce perfectly aligned spans. When a target or attack phrase is split differently across these steps, the model may fail to match the cue to the correct token position. This weakens the attention mechanism because the model may attend to incomplete or irrelevant tokens.

Sixth, threshold selection affects the balance between precision and recall. A lower threshold may detect more hate speech but also increase false positives. A higher threshold may reduce false positives but miss more hate speech. Since hate speech datasets are imbalanced, threshold tuning is necessary, but the best threshold may vary across domains and data distributions.

Overall, these errors show that Vietnamese hate speech detection requires more than keyword matching. Future improvements should expand the target and attack cue banks, use data-driven cue discovery, save per-instance prediction logs for detailed false-positive and false-negative analysis, and consider additional signals such as sarcasm, discourse context, user context, or probability calibration.
