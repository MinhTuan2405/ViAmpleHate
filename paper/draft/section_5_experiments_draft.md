# 5. Experiments

The experiments are designed to evaluate whether the proposed ViAmpleHate_PhoBERT improves the direct AmpleHate_PhoBERT baseline through concrete Vietnamese-specific architectural changes. Both models use the same encoder, `vinai/phobert-base`, so the comparison focuses on the proposed target extraction, relation modeling, attention correction, adaptive fusion, and training objective.

## 5.1 Baseline Formulation

Given an input comment `x`, PhoBERT produces contextual hidden states:

```text
H = PhoBERT(x) = [h_0, h_1, ..., h_n],     h_i in R^d
```

where `h_0` is the `[CLS]` representation and `d = 768` for PhoBERT-base. The direct AmpleHate_PhoBERT baseline extracts a set of target token indices `T = {t_1, ..., t_m}` using an English NER model. If no target is detected, it uses the fallback `T = {0}`.

For each target token, the baseline computes a HeadAttention vector between `[CLS]` and target embeddings:

```text
q = W_q h_0
k_i = W_k h_{t_i}
v = W_v h_0
alpha_i = softmax(q k_i^T / sqrt(d))
r_base = sum_i alpha_i v
```

The relation vector is then injected into `[CLS]` using a fixed scalar `e`:

```text
z_base = h_0 + e * r_base
y_hat = softmax(W_c z_base + b_c)
```

In the implementation, `e = 1.0`. The limitation is that the NER component is English-oriented and therefore detects very few valid Vietnamese hate targets. Consequently, most inputs use `T = {0}`, reducing the model to a mostly sentence-level PhoBERT classifier with limited target awareness.

## 5.2 Proposed Target and Attack Signal Construction

ViAmpleHate_PhoBERT replaces the English target extraction module with a Vietnamese multi-signal extraction module. For each input `x`, we construct two index sets:

```text
T_x = target indices from Vietnamese NER and target cue matching
A_x = attack indices from attack cue matching
```

The target set `T_x` is obtained by combining Vietnamese NER and a target cue bank:

```text
T_x = M_NER(x) union M_target(x)
```

where `M_NER(x)` returns token positions of Vietnamese named entities, and `M_target(x)` returns token positions matched from Vietnamese referential target cues. These cues include informal group or person references that may not be recognized as named entities.

The attack set `A_x` is obtained from an attack cue bank:

```text
A_x = M_attack(x)
```

where `M_attack(x)` returns token positions of hostile predicates, insults, threats, or negative evaluations. If either set is empty, the model uses `[CLS]` fallback:

```text
T_x = {0} if T_x = empty
A_x = {0} if A_x = empty
```

In implementation, all cue phrases are normalized, word-segmented, tokenized by the PhoBERT tokenizer, and matched by full token-sequence matching. This avoids matching only a subtoken fragment of a Vietnamese phrase.

## 5.3 Relation-Bank Attention

Instead of using a single target-attention vector, ViAmpleHate_PhoBERT constructs three relation vectors:

```text
r_exp = HeadAttn(h_0, H[T_x])
r_imp = HeadAttn(h_0, h_0)
r_atk = HeadAttn(h_0, H[A_x])
```

Here, `r_exp` captures explicit target information, `r_imp` preserves implicit sentence-level context, and `r_atk` captures attack-related information. For a relation input matrix `E in R^{m x d}`, HeadAttention is computed as:

```text
Q = W_q h_0                         in R^d
K = W_k E                           in R^{m x d}
V = W_v E                           in R^{m x d}
alpha = softmax(Q K^T / sqrt(d))    in R^m
r = alpha V                         in R^d
```

The three relation vectors are concatenated and projected into one fused relation vector:

```text
r = W_r [r_exp ; r_imp ; r_atk] + b_r
```

This relation bank is a concrete architectural improvement over the baseline. The baseline only models target attention, while the proposed model explicitly models target, context, and attack signals as separate relation channels.

## 5.4 Corrected Batched Attention

The original AmpleHate-style implementation computes attention using a matrix multiplication pattern equivalent to `Q K^T` across the batch. If `Q in R^{B x d}` and `K in R^{B x d}`, then:

```text
Q K^T in R^{B x B}
```

This can unintentionally mix information between different samples in the same mini-batch. ViAmpleHate_PhoBERT corrects this by using batched attention. For batch size `B` and `m` relation tokens per sample:

```text
Q in R^{B x 1 x d}
K in R^{B x m x d}
scores = bmm(Q, K^T) / sqrt(d)      in R^{B x 1 x m}
```

Thus, each sample attends only to its own target or attack tokens. Padding positions are suppressed with an attention mask before the softmax operation:

```text
scores_j = -infinity if mask_j = 0
```

This makes the implemented attention mechanism consistent with the intended sample-wise relation modeling.

## 5.5 Instance-Adaptive Relation Injection

The baseline injects the relation vector with a constant scalar `e`, so every instance receives the same amount of target-aware information. ViAmpleHate_PhoBERT replaces this with an adaptive gate:

```text
g = sigmoid(W_g [h_0 ; r] + b_g)
z = h_0 + g * r
```

where `g in (0, 1)` is computed separately for each input comment. This design allows the model to adjust relation injection based on the reliability of the extracted target and attack cues. If cue evidence is informative, the gate can increase relation contribution. If cues are absent or ambiguous, the model can rely more on the original PhoBERT `[CLS]` representation.

The final prediction is computed as:

```text
y_hat = softmax(W_c dropout(z) + b_c)
```

## 5.6 Training Objective

The baseline is trained mainly with weighted cross-entropy. ViAmpleHate_PhoBERT keeps weighted cross-entropy and adds contrastive regularization on the final representation `z`.

The weighted cross-entropy loss is:

```text
L_CE = - sum_c w_c y_c log(y_hat_c)
```

where `w_c` is the class weight for class `c`. Class weights are used because hate speech datasets are imbalanced and the `HATE` class is usually the minority class.

The contrastive loss encourages representations from the same class to be close and representations from different classes to be separated. Given a mini-batch of representations `{z_i}` and labels `{y_i}`, cosine similarity is computed as:

```text
s_ij = cos(z_i, z_j)
```

The contrastive term is:

```text
L_CL = sum_{i != j} [ 1[y_i = y_j] * (1 - s_ij)
                    + 1[y_i != y_j] * max(0, s_ij - margin) ] / N
```

The final training objective is:

```text
L = L_CE + alpha * L_CL
```

where `alpha = 0.1` in the implementation. This objective improves not only classification accuracy but also representation separation between `NON-HATE` and `HATE`.

## 5.7 Practical Implementation Settings

The proposed model is implemented with the following concrete changes relative to the baseline:

| Component | Baseline AmpleHate_PhoBERT | Proposed ViAmpleHate_PhoBERT |
|---|---|---|
| Encoder | PhoBERT-base | PhoBERT-base |
| Target extraction | English NER | Vietnamese NER + Vietnamese target cue bank |
| Attack signal | Not modeled | Separate Vietnamese attack cue bank |
| Attention structure | One HeadAttention module | Three HeadAttention modules: target, implicit, attack |
| Attention computation | Batch-level matrix multiplication may mix samples | Batched attention with per-sample masking |
| Fusion | Fixed scalar injection `h_0 + e r` | Adaptive gate `h_0 + g r` |
| Training loss | Weighted cross-entropy | Weighted cross-entropy + contrastive loss |
| Input length | Shorter context window | Longer context window |
| Batch strategy | Standard mini-batch | Gradient accumulation for larger effective batch |
| Evaluation extraction | Target extraction disabled or inconsistent | Same cue extraction policy across train/validation/test |

During training, the best checkpoint is selected using validation macro-F1. The final classification threshold is also selected on the validation set by maximizing macro-F1:

```text
t* = argmax_t MacroF1(y, 1[p_HATE >= t])
```

The selected threshold `t*` is then used for test prediction. Macro-F1 and HATE-F1 are used as the primary metrics because they better reflect minority-class hate speech detection than accuracy.

## 5.8 Experimental Purpose

The purpose of the experiment is not only to compare final scores but also to verify whether each proposed modification addresses a concrete limitation of the baseline. Vietnamese NER and target cues address low target coverage. Attack cues address the absence of explicit hostility modeling. Relation-bank attention addresses the need to model target, context, and attack separately. Corrected batched attention addresses implementation-level cross-sample contamination. Adaptive gating addresses the weakness of fixed scalar injection. Contrastive loss addresses class separation under data imbalance.

Therefore, the proposed model is evaluated as a practical Vietnamese adaptation of AmpleHate rather than as a superficial replacement of the encoder or classifier.
