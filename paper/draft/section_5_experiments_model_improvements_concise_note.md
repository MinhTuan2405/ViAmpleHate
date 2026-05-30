# 5. Experiments

The experiments focus on evaluating whether the proposed ViAmpleHate_PhoBERT improves the direct AmpleHate_PhoBERT baseline. Both models use the same PhoBERT encoder, so the comparison mainly measures the effect of the proposed Vietnamese-specific modules.

Given an input comment `x`, PhoBERT produces contextual representations:

```text
H = PhoBERT(x) = [h_0, h_1, ..., h_n]
```

where `h_0` is the `[CLS]` vector. In the baseline, target indices `T` are extracted using English NER. The model computes a single target relation and injects it into `[CLS]` using a fixed scalar `e`:

```text
r_base = HeadAttn(h_0, H[T])
z_base = h_0 + e * r_base
```

This baseline is limited for Vietnamese because English NER rarely detects valid Vietnamese hate targets. As a result, many inputs fall back to `T = {0}`, making the model close to a standard PhoBERT classifier.

ViAmpleHate_PhoBERT improves this design by constructing Vietnamese target and attack signals:

```text
T_x = M_NER(x) union M_target(x)
A_x = M_attack(x)
```

Here, `M_NER` is Vietnamese NER, `M_target` is Vietnamese target cue matching, and `M_attack` is attack cue matching. The model then computes three relation vectors:

```text
r_exp = HeadAttn(h_0, H[T_x])
r_imp = HeadAttn(h_0, h_0)
r_atk = HeadAttn(h_0, H[A_x])
```

These vectors represent explicit target information, implicit sentence context, and attack information. They are fused into one relation representation:

```text
r = W_r [r_exp ; r_imp ; r_atk] + b_r
```

Instead of using the fixed injection weight from the baseline, ViAmpleHate uses an adaptive gate:

```text
g = sigmoid(W_g [h_0 ; r] + b_g)
z = h_0 + g * r
```

This allows the model to decide how much target-attack information should be used for each comment. If extracted cues are useful, the gate increases their effect; if cues are weak or ambiguous, the model relies more on the original PhoBERT representation.

The attention implementation is also corrected using batched attention:

```text
scores = bmm(Q, K^T) / sqrt(d)
```

This ensures that each sample attends only to its own target or attack tokens, avoiding cross-sample attention inside a batch.

For training, ViAmpleHate uses weighted cross-entropy and contrastive loss:

```text
L = L_CE + alpha * L_CL
```

Weighted cross-entropy handles class imbalance, while contrastive loss encourages better separation between `NON-HATE` and `HATE` representations. The final threshold is selected on the validation set by maximizing macro-F1:

```text
t* = argmax_t MacroF1(y, 1[p_HATE >= t])
```

Overall, the experiment verifies six concrete improvements over the baseline: Vietnamese target extraction, separate attack modeling, relation-bank attention, corrected batched attention, adaptive relation injection, and contrastive training.
