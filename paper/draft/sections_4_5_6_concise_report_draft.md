# 4. Methodology

We formulate Vietnamese hate speech detection as a binary classification task with two labels: `NON-HATE` and `HATE`. The proposed ViAmpleHate model adapts AmpleHate to Vietnamese by replacing English-oriented target extraction with Vietnamese-aware target and attack modeling.

Input comments are first normalized to reduce social media noise, including teencode, repeated characters, URLs, punctuation, and emojis. The normalized text is then word-segmented and encoded using PhoBERT. To identify hate-related signals, ViAmpleHate combines Vietnamese NER with two cue banks: a target cue bank for possible target mentions and an attack cue bank for offensive predicates or hostile expressions.

The model uses three relation-aware attention modules: one for explicit target cues, one for implicit sentence context, and one for attack cues. These relation vectors are fused and injected into the PhoBERT `[CLS]` representation through an adaptive gate. Unlike the original AmpleHate model, which uses a fixed injection weight, the adaptive gate allows the model to control how much target-attack information should influence each prediction.

The final representation is passed to a classifier for hate speech prediction. Training uses weighted cross-entropy to handle class imbalance and contrastive loss to improve class separation in the learned representation.

# 5. Experiments

The experiments mainly compare ViAmpleHate_PhoBERT with the direct AmpleHate_PhoBERT baseline. Both models use PhoBERT as the encoder, so the comparison focuses on the proposed model improvements rather than a change in the backbone. Given an input `x`, PhoBERT outputs `H = [h_0, h_1, ..., h_n]`, where `h_0` is the `[CLS]` vector.

The baseline uses English-oriented NER to extract target indices `T`, computes one target relation `r_base = HeadAttn(h_0, H[T])`, and injects it with a fixed scalar: `z_base = h_0 + e * r_base`. This is weak for Vietnamese because many comments do not contain targets detectable by English NER, causing the model to fall back to `[CLS]`.

ViAmpleHate improves this design by constructing Vietnamese target and attack sets: `T_x = M_NER(x) union M_target(x)` and `A_x = M_attack(x)`. It then computes three relation vectors: `r_exp = HeadAttn(h_0, H[T_x])`, `r_imp = HeadAttn(h_0, h_0)`, and `r_atk = HeadAttn(h_0, H[A_x])`. These are fused as `r = W_r[r_exp; r_imp; r_atk] + b_r`.

Instead of fixed injection, the proposed model uses an adaptive gate: `g = sigmoid(W_g[h_0; r] + b_g)` and `z = h_0 + g * r`. Training uses `L = L_CE + alpha * L_CL`, where weighted cross-entropy handles class imbalance and contrastive loss improves representation separation. Macro-F1 and HATE-class F1 are used as the main metrics because they better reflect performance on imbalanced hate speech detection.

# 6. Result Analysis or Error Analysis

## 6.1 Experimental Results

The results show that PhoBERT-based models outperform traditional lexical and static-embedding baselines, confirming the importance of contextual Vietnamese representations. The direct AmpleHate_PhoBERT baseline improves target-aware modeling but remains limited because its original target extraction mechanism is not well aligned with Vietnamese.

ViAmpleHate improves the target-aware approach by using Vietnamese NER, target cue mining, attack cue mining, relation-bank attention, and adaptive gating. The main improvement appears in macro-F1 and HATE-class F1, which are more informative than accuracy for imbalanced hate speech detection. This suggests that Vietnamese-specific target and attack modeling helps the model better recognize minority-class hate speech.

## 6.2 Error Analysis

The remaining errors mainly come from four sources. First, offensive but non-hate comments can be misclassified as hate speech because they contain profanity or insults without a clear group-directed target. Second, implicit hate speech is difficult because it may rely on sarcasm, stereotypes, or social context without explicit target or attack cues. Third, target cues such as pronouns or group nouns can be ambiguous and may appear in non-hateful comments. Fourth, the cue banks cannot cover all slang, spelling variants, or newly emerging offensive expressions in Vietnamese social media.

These errors suggest that future work should expand the target and attack cue banks, save per-instance prediction logs for deeper analysis, and explore additional signals such as sarcasm, discourse context, and better probability calibration.
