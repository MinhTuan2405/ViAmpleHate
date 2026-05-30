# Revision Notes — v_review

Revision of the v3 paper per the Revision Roadmap in `review_report.md` (project root). Two rounds. `v3` is untouched. **No numeric results were fabricated** (academic-paper IRON RULE + user instruction).

## Round 2 summary (latest)

Per author direction: (a) the **ablation section was removed entirely** (author does not want it in the report); (b) the differing baseline/proposed configuration was **run intentionally**, so it is now framed as deliberate design (full proposed system vs baseline-as-configured), not a confound to "fix" — the controlled-re-run `[TODO]` was removed. Real numbers were then pulled from `notebooks/` and filled in:

- **Hyperparameters** (Appendix A): class weights = inverse frequency `N/(C·n_c)`, contrastive margin `m=0.5`, label smoothing `0.05`, seed `42` (single run). Source: proposed ViHSD notebook (`SEED=42`, `LABEL_SMOOTHING=0.05`, `margin=0.5`, `ALPHA_CL=0.1`, `class_weights = len(train)/(NUM_CLASSES·label_counts)`).
- **Baseline accuracies** (Tables 3–4): BiLSTM 0.8454 (ViHSD) / 0.8650 (VOZ); PhoBERT-CNN 0.8945 / 0.9623. Dashes removed.
- **Cue-bank sizes** (§3.4, Appendix B): 16 target cues, 24 attack cues (full lists added to the markdown Appendix B).
- **Target/attack coverage** (§6.2): ViHSD 20.0%/18.8% target, 5.2%/4.2% attack; VOZ 45.2%/43.0% target, 7.6%/6.0% attack (first 500 samples/split). New cross-dataset point: higher coverage (VOZ) ↔ larger gains — strengthens the coverage argument with real evidence.
- **VOZ per-class** (Table 7, now both datasets): VOZ HATE P 0.7347 / R 0.6803 / F1 0.7065; NON-HATE 0.9638 / 0.9719 / 0.9678.

Items genuinely not in the notebooks (multi-seed/significance, published-SOTA comparison, full PR curve, real qualitative test instances, drawn figures) were moved to the **Limitations** as acknowledged limitations (per the max-2-rounds rule), not left as inline `[TODO]`. Both files are now free of `[TODO]` tags.

---

## Round 1 (initial pass — historical)
Scope: writing/framing fixes only; experiment-dependent items were left as `[TODO]` placeholders.

Status legend: **DONE** = fully addressed in text · **PARTIAL** = framing fixed, data pending · **TODO(author)** = needs an experiment/external action, placeholder inserted · **DISAGREE** = reviewer point not adopted, with reason.

Files changed: `paper/main/latex/v_review/acl_latex.tex` and (kept in sync) `paper/main/markdown/v_review/report_en.md`.

---

## P0 — Must fix (gate to acceptance)

| # | Roadmap item | Status | What was done in v_review |
|---|---|---|---|
| 1 | De-confound the main comparison (max_len/batch/epochs/NER-eval) | **PARTIAL** | Removed the false "isolates our changes" claim (Abstract, §5.1). Added a **"Controlled comparison (in progress)"** paragraph in §6.1 stating the confound explicitly and the matched-rerun protocol (identical max_len 256 / batch / epochs / seeds; baseline variant with NER-on). Deltas relabeled "indicative, not attributable". → **TODO(author):** run the matched comparison and fill the table. |
| 2 | Fill the ablation (5 leave-one-out rows) | **TODO(author)** | Ablation table kept; §6.3 now states component contributions are **design hypotheses, not established**, until the table is filled. Contributions list reworded accordingly. |
| 3 | Add variance + significance (≥5 seeds, paired test) | **PARTIAL** | §6.1 now says all numbers are single-run point estimates and the ViHSD gain is within run-to-run noise → **preliminary**. Table captions note "single-run point estimates". Appendix A adds a "Seeds/runs reported: [TODO ≥5]" row. → **TODO(author):** run seeds + significance test. |
| 4 | Verify/complete all citations | **TODO(author)** | Not auto-fixable without external verification; **not fabricated**. `custom.bib` still carries `TODO` for `amplehate`, `vithsd`, `vozhsd`. Author must verify authors/venue/year. |

## P1 — Strongly recommended

| # | Roadmap item | Status | What was done |
|---|---|---|---|
| 5 | Define + measure "target coverage"; link coverage→benefit | **DONE / PARTIAL** | §6.2 now **defines** coverage as fraction with `T_x ≠ {0}`, states it is measured on the ViHSD train/val split, and explicitly warns coverage is an *input* statistic (not proof of benefit). → **TODO(author):** cue-present vs cue-absent subset experiment. |
| 6 | Fair baseline (NER-on at eval) | **PARTIAL** | Controlled-comparison protocol now includes "a baseline variant with Vietnamese NER enabled at evaluation so the competitor is not handicapped". → **TODO(author):** run it. |
| 7 | Substantiate/demote the "batched-attention bug" | **DONE** | §4.6 retitled **"Batched attention (implementation note)"**; reframed as a correction to *our own port*, not a claim about original AmpleHate; isolated effect deferred to ablation. Demoted from contribution #3 headline. |
| 8 | SOTA comparison on ViHSD | **TODO(author)** | §6.1 adds a sentence + `[TODO]` to add published-ViHSD comparison. |
| 9 | Document cue banks (size/protocol/agreement; release) | **PARTIAL** | §3.4 adds a train-derivation-bias caveat + `[TODO]` for bank sizes, protocol, agreement, matcher precision/recall, and release. |
| 10 | Specify objective (`w_c`, `m`, label smoothing); fix SupCon citation | **DONE / PARTIAL** | §2.5 + §4.7 reworded: the term is a "pairwise cosine-margin objective inspired by supervised contrastive learning, not the exact SupCon loss". Appendix A adds `w_c` / `m` / label-smoothing rows. → **TODO(author):** fill the actual values. |

## P2 — Polish

| # | Roadmap item | Status | What was done |
|---|---|---|---|
| 11 | Real figures (Fig 1 distribution, Fig 2 architecture; embed 3/5; gate hist 4) | **TODO(author)** | Placeholders unchanged; cannot draw/embed here. Notes/paths retained in the figure captions/comments. |
| 12 | Replace constructed qualitative examples with real ones | **TODO(author)** | Table 6 already flagged "constructed; replace with real examples"; left for author (avoids fabricating dataset instances). |
| 13 | Fill BiLSTM/PhoBERT-CNN accuracy; clarify bolding | **DONE / TODO(author)** | Table 3/4 captions now state "accuracy for reference; dashes = not logged" and note VOZ baseline accuracy is higher. Accuracy values themselves are **TODO(author)** (not logged for those baselines; not fabricated). |
| 14 | Expand VOZ-HSD description | **PARTIAL** | §3.2 adds source (`tarudesu/VOZ-HSD`) + `[TODO]` for annotation/agreement/license. |
| 15 | Temper "generalizes"; add fairness note | **DONE** | Conclusion: "generalizes" → "adapts"; coverage link called "correlational, not yet causal". Discussion adds the recall-trade-off caveat + `[TODO]` PR-curve. Ethics already covers bias auditing. |

## Reviewer points NOT adopted (with reason)

- None marked **DISAGREE** outright. All review points were either fixable in text or are legitimately experiment-dependent. The recall-decrease framing (R2/R4) was *adjusted* rather than rejected: the paper no longer presents "more precise" as unambiguously good and now flags the deployment trade-off (§6.5).

## Remaining gate to acceptance

The decision in `review_report.md` was **Major Revision**, driven by three CRITICAL items (confound, missing ablation, no significance). This pass makes the manuscript **honest** about all three but does **not** resolve them — they require compute the author must run (items P0-1, P0-2, P0-3, plus P1-5/6/8). Until those runs exist, the paper should be read as "method + preliminary evidence", which the revised text now states plainly.
