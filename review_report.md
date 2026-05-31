# Peer Review Report — ViAmpleHate (v3)

**Manuscript:** *ViAmpleHate: A Proposed AmpleHate- and PhoBERT-based Approach for Vietnamese Hate Speech Detection*
**File reviewed:** `paper/main/latex/v3/acl_latex.tex`
**Venue assumed:** *ACL (ARR-style) main/findings, short-to-long NLP paper*
**Review mode:** `full` (5 reviewers + Devil's Advocate + Editorial Synthesis)
**Reviewer scales:** ARR-style — Soundness (1–5), Excitement (1–5), Confidence (1–5)

> ⚠️ Reviewers examined the manuscript only; the paper was **not modified**. All findings are reproduced here as a separate document. Figures could not be assessed (all are placeholders / `[DRAW NEW]` / external PNGs), and several numeric claims rest on tables that are still `[TODO]` in the source.

---

## Phase 0 — Field Analysis & Reviewer Configuration

| Attribute | Assessment |
|---|---|
| Primary discipline | NLP — hate speech / toxic-language detection |
| Secondary | Vietnamese (low/medium-resource) language modeling; representation learning |
| Paradigm | Empirical, supervised text classification; architecture adaptation |
| Methodology type | Model proposal + benchmark comparison (two datasets, five baselines) |
| Target tier | Mid-tier NLP venue / ARR; **not yet** top-tier as written |
| Maturity | Working draft: solid framing, but core empirical evidence (ablation, significance) is incomplete and the headline comparison is confounded |

**Panel (dynamically configured):**

1. **EIC** — senior NLP-venue editor; cares about novelty vs. AmpleHate, positioning, and whether claims match evidence.
2. **R1 — Methodology** — empirical ML rigor: experimental controls, significance, ablations, reproducibility.
3. **R2 — Domain** — Vietnamese NLP / hate-speech literature; benchmarks, SOTA comparison, dataset treatment, citations.
4. **R3 — Perspective** — applied content-moderation / deployment & cross-disciplinary impact.
5. **DA — Devil's Advocate** — attacks the central causal claim and looks for the strongest counter-argument.

*(You may adjust any persona and re-run.)*

---

## Phase 1 — Independent Reviews

### Reviewer 1 — Editor-in-Chief (overall, fit, novelty)

**Summary.** The paper adapts AmpleHate's target-aware idea to Vietnamese on a PhoBERT backbone, motivated by a genuinely compelling observation: English NER finds a usable target in only ~0.09% of Vietnamese comments (§1, §6.2), so the original pipeline collapses to a plain classifier. The proposed fixes — Vietnamese NER + cue banks, a three-channel relation bank, an adaptive gate, and a contrastive term — are sensible and clearly described. Writing is clean and the Limitations/Ethics sections are commendably honest.

**Strengths.**
- Clear, well-motivated problem; the 0.09% → ~19% coverage gap is a strong narrative hook (§1, Table 2).
- Correct primary-metric choice (macro-F1 / HATE-F1 over accuracy) and an explicit rationale (§5.3).
- Honest self-disclosure of the missing ablation (§Limitations).

**Concerns (fit/claims).**
- The contribution is an **engineering adaptation** of an existing method. That can be publishable, but only if the empirical case is airtight — and currently it is not (see R1/DA). As written, the headline "improves on both datasets" overstates what the evidence supports, especially on ViHSD where Δmacro-F1 = +0.0027.
- **Novelty framing leans on a baseline-implementation bug** ("corrected batched attention", §4.6). If a contribution is fixing the authors' own port, that is a weak novelty pillar and must be carefully separated from the genuinely novel components.

**Scores:** Soundness **2/5** · Excitement **3/5** · Confidence **4/5**.
**Recommendation:** Major Revision.

---

### Reviewer 2 — Methodology (rigor, controls, reproducibility)

This is where the paper is weakest, and the issues are serious enough to block acceptance.

**[CRITICAL] M-C1 — The headline comparison is confounded.** §5.1 and the abstract claim the baseline and ViAmpleHate "share the same PhoBERT-base encoder … isolates our changes." But Table 2 shows the two models differ in **max length (128 vs 256), batch/effective batch (16 vs 32 via grad-accum), and NER-at-eval (off vs on)** — and the underlying configs also differ in epoch budget. These are exactly the knobs known to move F1 by more than the reported deltas. Therefore the improvement **cannot** be attributed to target-aware modeling; it could be driven by longer context or more optimization. *Fix:* hold max_len, batch, epochs, and seeds identical across baseline and proposed, varying only the target-aware components.

**[CRITICAL] M-C2 — No ablation evidence.** Table 5 (§6.3) is entirely `[TODO]`. The paper's thesis is that *five specific components* help, yet not one is individually validated. Without this, §5.4's "experimental design tests whether each modification addresses a limitation" is unfulfilled. *Fix:* run the five leave-one-out ablations under the controlled setup from M-C1; the paper should not claim component-level benefit until then (the authors acknowledge this in Limitations, which is appreciated but does not resolve it).

**[CRITICAL] M-C3 — No variance, no significance, single run.** All tables report point estimates from what appears to be one run. The ViHSD gains (+0.0027 macro-F1, +0.0036 HATE-F1) are well within typical seed-to-seed variation for PhoBERT fine-tuning. *Fix:* report mean ± std over ≥5 seeds and a paired significance test (e.g., bootstrap or McNemar on predictions); otherwise the ViHSD claim should be dropped to "comparable."

**[MAJOR] M-1 — HATE recall *drops* on ViHSD** (0.6119 → 0.5988, Table 7 / §6.5). For a hate detector, lower recall means *more hate slips through*. The paper frames this favorably ("more conservative but more precise"), but for the stated deployment goal this trade-off needs justification, ideally with a PR curve or F-beta analysis rather than a single threshold.

**[MAJOR] M-2 — Coverage measurement is imprecise.** The central mechanistic number, "~18.8–20.0% of checked train/validation samples" (§6.2, Table 2), is vague: what is "checked," on what split, and is coverage = target cue OR NER hit? This claim does the causal heavy lifting and must be defined and computed on a stated population.

**[MAJOR] M-3 — Underspecified objective.** §4.7 gives `L = L_CE + αL_CL` but leaves the class weights `w_c`, the contrastive margin `m`, and the label-smoothing value unspecified. The loss is also a **custom pairwise hinge, not the cited SupCon** (\citep{supcon}) — either change the citation or justify the divergence.

**[MINOR] M-4 — Missing-cell reporting.** BiLSTM and PhoBERT-CNN have "–" for accuracy (Tables 3–4); fill or explain. Define the bolding rule: on VOZ-HSD the baseline accuracy (0.9643) exceeds ViAmpleHate (0.9420), so "best per column in bold" needs care.

**[MINOR] M-5 — HeadAttention details.** Specify number of heads, whether `r_imp = HeadAttn(h_0,h_0)` is meaningfully different from `h_0`, and the dimensionality of `W_r`.

**Scores:** Soundness **2/5** · Excitement **2/5** · Confidence **5/5**.
**Recommendation:** Major Revision (borderline Reject if M-C1/M-C2/M-C3 are not addressable with available compute).

---

### Reviewer 3 — Domain (Vietnamese NLP / hate-speech literature)

**Strengths.** The target-centric framing is well-aligned with Vietnamese hate speech, and connecting to ViTHSD's target annotation is apt (§2.3). The binary relabeling is clearly stated and the dataset statistics are internally consistent (Table 1).

**[MAJOR] D-1 — No comparison to published SOTA on ViHSD.** ViHSD is an established benchmark with prior published results, yet the paper compares only to its own re-implemented baselines. Without a literature comparison, the reader cannot tell whether 0.7819 macro-F1 is competitive or below the field. *Fix:* add a column / paragraph situating ViAmpleHate against published ViHSD numbers.

**[MAJOR] D-2 — Unverifiable citations.** `custom.bib` carries placeholder/`TODO` entries for AmpleHate, ViTHSD, and VOZ-HSD (authors, venue, year missing). Several substantive claims — e.g., that the baseline is a "faithful port of AmpleHate" and that ViTHSD does X — cannot be verified. This must be fixed before any acceptance; it also risks the hallucinated-citation failure mode.

**[MAJOR] D-3 — Cue-bank provenance and reproducibility.** §3.4 / Appendix B say the banks are "seeded manually from inspection of the training split." This invites **train-set overfitting** and is not reproducible as described: no bank sizes, no construction protocol, no inter-annotator agreement, and Appendix B lists essentially no concrete entries (two example tokens). Because the cue banks carry the method's core signal, this is a first-order reproducibility gap.

**[MINOR] D-4 — VOZ-HSD underdescribed.** One sentence (§3.2). Provide provenance, annotation procedure, label definitions, quality/agreement, and license/ethics of forum data.

**[MINOR] D-5 — Emoji→pragmatic-tag mapping** (§3.3) is asserted but never specified or evaluated; either describe the mapping table or move it to an appendix.

**Scores:** Soundness **2/5** · Excitement **3/5** · Confidence **4/5**.
**Recommendation:** Major Revision.

---

### Reviewer 4 — Perspective (deployment, cross-disciplinary, impact)

**Strengths.** The applied framing is realistic: macro/HATE-F1 emphasis, threshold selection, and a thoughtful Ethics statement (dual-use, human-in-the-loop, bias auditing). The "precision-favoring" behavior could matter for moderation pipelines where false accusations are costly.

**Concerns.**
- **P-1 (MAJOR).** The deployment story is internally inconsistent with the data: §6.5 sells higher precision, but a moderation system that *misses more hate* (lower recall, ViHSD) may be the wrong trade-off for safety-oriented deployments. State the operating point explicitly and, ideally, show the precision-recall trade-off so practitioners can choose.
- **P-2 (MINOR).** Cue banks encode the authors' judgments of what counts as a "target" or "attack." This is a fairness risk (dialect/region bias) the Ethics section gestures at but does not analyze. A short per-group or per-region error breakdown would strengthen both impact and ethics.
- **P-3 (MINOR).** Generalization claims rest on two datasets that may share register (Vietnamese forums/social media); cross-domain robustness is asserted as future work but tempers current "generalizes" language in the conclusion.

**Scores:** Soundness **3/5** · Excitement **3/5** · Confidence **3/5**.
**Recommendation:** Minor-to-Major Revision.

---

### Reviewer 5 — Devil's Advocate

**Strongest counter-argument (steelman against the paper).**
> "ViAmpleHate's reported gains are an artifact of training-setup differences, not target-aware modeling. The proposed model sees twice the context (256 vs 128 tokens), a larger effective batch, NER enabled at evaluation, and a different epoch budget, while the only ablation that could disentangle these factors is blank. On the cleaner benchmark (ViHSD) the macro-F1 gain is +0.0027 and HATE *recall actually decreases* — both consistent with noise plus a threshold shift rather than a better hate detector. The headline 0.09%→~19% 'coverage' jump measures how often a cue *fires*, not whether firing improves decisions; high coverage of noisy cues can just as easily add false positives. In short, the paper has not shown that any of its four modeling contributions causes the improvement."

This counter-argument is currently **difficult to refute with the evidence provided**, which is why the issues below are CRITICAL.

**Issue list.**
- **[CRITICAL] DA-1 — Causal claim unsupported (confound + no ablation + no significance).** Combines M-C1, M-C2, M-C3. The paper's central claim ("these adaptations improve detection") is not established. *Per Checkpoint Rule #4, this blocks an Accept decision.*
- **[CRITICAL] DA-2 — Coverage≠benefit conflation.** §6.2 treats cue coverage as if it were the cause of the gains ("the headline mechanism"), but coverage is an input statistic; no experiment links coverage to decision quality. Cherry-picking risk: the metric chosen to explain success is the one most favorable to the narrative.
- **[MAJOR] DA-3 — Baseline strawman risk.** Disabling NER at evaluation for the baseline ("Off ⇒ `[CLS]` fallback", Table 2) handicaps the competitor at test time in a way the proposed model is not. Is the baseline's reported number its best achievable configuration, or one chosen to look weak?
- **[MAJOR] DA-4 — "Bug fix" as contribution.** The B×B attention claim (§4.6) is asserted about "the original AmpleHate-style implementation" without a code reference or a measured impact. If real, it is a fix to the authors' own port; if it materially changes results, it must be isolated in the ablation.
- **[OBSERVATION] "So what?" test.** Even granting the gains, on ViHSD the practical improvement is marginal; the paper's value, if validated, is mostly the *coverage diagnosis* and the cue-bank recipe — which should arguably be foregrounded over the small score deltas.

**Ignored alternatives.** (a) A plain PhoBERT classifier with the same 256-len / batch / epochs as ViAmpleHate — is that already most of the gain? (b) Simply enabling Vietnamese NER in the baseline without the relation bank/gate. Neither is tested.

---

## Phase 2 — Editorial Synthesis & Decision

### Consensus (raised independently by ≥3 reviewers)
- **Confounded baseline comparison** (R1, R2/M-C1, DA-1): unanimous that the "same encoder isolates our changes" claim is not supported because max_len/batch/epoch/NER-eval differ.
- **No ablation + no significance/variance** (R1, R2/M-C2-3, DA-1): the component-level and ViHSD claims are unsubstantiated.
- **Citations / cue-bank reproducibility** (R3/D-2,D-3, partially R1): unverifiable references and an unreproducible, train-derived cue bank.

### Disagreement / arbitration
- **Severity of the recall drop:** R2 and R4 weight it heavily (safety), R1 less so. *Arbitration:* require an explicit operating-point/PR analysis; do not mandate a particular trade-off, but the favorable framing must be earned with evidence.
- **Excitement:** ranges 2–3. *Arbitration:* the idea and the coverage diagnosis are interesting (3); the thin, confounded evidence drags soundness to 2. The contribution is salvageable.

### Devil's Advocate CRITICAL findings → decision constraint
DA-1 and DA-2 are CRITICAL. **Per IRON RULE / Checkpoint Rule #4, the decision cannot be Accept.** The issues are, however, fixable with controlled re-runs (not fatal design flaws), so the paper is **not** a Reject.

### 📋 EDITORIAL DECISION: **MAJOR REVISION**

**Rationale.** The motivation and method are sound and well-written, and the coverage diagnosis is a real insight. But the empirical core — a clean, controlled comparison; component ablations; and significance over multiple seeds — is currently missing or confounded, and the central causal claim is therefore unproven. None of these are design-fatal; they require disciplined experiments and verified citations. Resolve the CRITICAL items and this becomes a credible contribution.

---

## Revision Roadmap (prioritized — drop-in for `academic-paper` revision mode)

**P0 — Must fix (gate to any acceptance)**
1. **De-confound the main comparison (DA-1/M-C1).** Re-run baseline and ViAmpleHate with **identical** max_len (256), batch/effective batch, epoch budget, and seeds; vary only the target-aware components. Update Tables 3–4.
2. **Fill the ablation (M-C2).** Run the five leave-one-out rows of Table 5 under the controlled setup; report macro-F1 and HATE-F1. Until then, soften all component-level claims.
3. **Add variance + significance (M-C3).** ≥5 seeds; report mean ± std; paired significance test on ViHSD and VOZ-HSD. Re-state the ViHSD result honestly given the tiny deltas.
4. **Verify and complete all citations (D-2).** Replace every `TODO` in `custom.bib` with real authors/venue/year; confirm AmpleHate, ViTHSD, ViHSD, VOZ-HSD details.

**P1 — Strongly recommended**
5. **Define and measure "target coverage" (M-2/DA-2)**, and add an experiment linking coverage to decision quality (e.g., performance on cue-present vs cue-absent subsets) so coverage is shown to *cause* benefit, not merely correlate.
6. **Fair baseline configuration (DA-3).** Report the baseline with NER enabled at eval (and/or with the same context window) so it is not handicapped.
7. **Substantiate the "batched-attention bug" (DA-4).** Cite the exact code/line, and isolate its effect as an ablation row; otherwise demote it from a contribution to an implementation note.
8. **Add SOTA comparison on ViHSD (D-1)** against published numbers.
9. **Document the cue banks (D-3).** Bank sizes, construction protocol, agreement, and a real (not two-token) appendix sample; discuss train-set-derivation bias.
10. **Specify the objective (M-3):** values for `w_c`, margin `m`, label smoothing; fix the SupCon citation vs. the actual pairwise loss.

**P2 — Polish**
11. Replace placeholder figures with real Figure 1 (distribution) and Figure 2 (architecture); embed Figures 3/5 PNGs; assess Figure 4 (gate distribution) — the gate histogram would also help support the adaptive-gate claim.
12. Replace constructed qualitative examples (Table 6) with real anonymized/masked test instances (M-8).
13. Fill BiLSTM/PhoBERT-CNN accuracy cells; clarify bolding convention (incl. VOZ accuracy where baseline > proposed).
14. Expand VOZ-HSD description (provenance, annotation, license) and the emoji-mapping table.
15. Temper "generalizes" language pending cross-domain evidence; add a short fairness/per-group error note to align with the Ethics section.

---

## Scorecard

| Reviewer | Soundness | Excitement | Confidence | Recommendation |
|---|:--:|:--:|:--:|---|
| EIC | 2 | 3 | 4 | Major Revision |
| R1 Methodology | 2 | 2 | 5 | Major Revision (borderline Reject) |
| R2 Domain | 2 | 3 | 4 | Major Revision |
| R3 Perspective | 3 | 3 | 3 | Minor–Major Revision |
| Devil's Advocate | — | — | — | 2 CRITICAL → no Accept |
| **Panel decision** | **~2.2** | **~2.8** | — | **MAJOR REVISION** |

**Top-3 things to fix first:** (1) de-confound the main comparison, (2) run the ablations, (3) add multi-seed variance + significance. Doing these three converts a promising but unproven draft into a defensible empirical paper.

---

*Generated by the academic-paper-reviewer panel (field_analyst → EIC + 3 peer reviewers + Devil's Advocate → editorial synthesis). Reviews are independent; the synthesis cites only issues raised in Phase 1. AI-assisted review — verify all findings against the manuscript before acting.*
</content>
