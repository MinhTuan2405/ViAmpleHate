# Response to Reviewers — ViAmpleHate

We thank the editor and reviewers for the detailed, constructive feedback. Below we respond to each point using **R** (reviewer comment) → **A** (author response) → **C** (change made). All changes are in the revised manuscript `paper/main/latex/v_review/acl_latex.tex` (with `paper/main/markdown/v_review/report_en.md` kept in sync); the original reviewed version is preserved at `…/v3/`. A per-item change log is in `paper/main/latex/v_review/REVISION_NOTES.md`.

Two points up front, as they recur below:
1. **The differing baseline/proposed configuration (context length, batch, NER-at-eval) was an intentional design choice, not an oversight.** The longer context window and larger effective batch are part of the proposed system; we now compare the *full proposed system* against the *baseline as configured*, and say so explicitly, rather than claiming the comparison "isolates" individual components.
2. **We have removed the ablation section.** We were unable to run controlled leave-one-out ablations for this submission, and we prefer not to include an empty placeholder. We instead state plainly in the Limitations that component-level contributions remain design hypotheses.

Where a request required experiments we did not run (multi-seed significance, published-SOTA comparison, full precision–recall curves), we have **not fabricated numbers**; we disclose these as explicit limitations.

---

## Editorial decision: Major Revision

The decision rested on three CRITICAL items from the Devil's Advocate / methodology reviews: (C1) confounded comparison, (C2) missing ablation, (C3) no significance. Our responses to all three are below (C1 reframed, C2 removed by choice, C3 disclosed as a limitation).

---

## Critical issues

**C1 — "The comparison is confounded; it does not isolate the target-aware components."**
- **A:** Agreed that the original text over-claimed. The configuration differences are deliberate (part of the proposed system), so rather than add a controlled re-run we have corrected the *claim* to match what the experiment actually shows.
- **C:** Abstract and §5.1 no longer say the comparison "isolates our changes." §5.1 now states each model is run at its own intended configuration and that the comparison is "the full proposed system vs the baseline as configured." §6.1 repeats this framing. (REVIEWER PARTIALLY ADOPTED — reframed, not re-run, by author choice.)

**C2 — "No ablation evidence; the five contributions are unsubstantiated."**
- **A:** We acknowledge the gap. We have not run the leave-one-out ablations and have chosen to remove the placeholder table rather than ship it empty.
- **C:** The §6.3 ablation subsection and its table were removed. The Limitations now state: "we do not ablate the individual components, so component-level contributions remain design hypotheses." Contribution (iii) reframes the batched-attention change as an *implementation note*, not a primary contribution.

**C3 — "No variance, no significance, single run; the ViHSD gain is within noise."**
- **A:** Correct. All results come from a single run (seed 42); we do not claim significance for the small ViHSD margin.
- **C:** Tables 3–4 captions now read "Single run (seed 42)." §6.1 states the ViHSD margin is "preliminary rather than established." Multi-seed variance + a paired significance test are listed as an explicit limitation. (Disclosed, not resolved.)

---

## Major issues

**M1 — "HATE recall decreases on ViHSD; framing it as 'more precise' hides that more hate is missed."**
- **A:** Agreed; we no longer present the precision gain as unambiguously good.
- **C:** §6.4 (Discussion) now frames the ViHSD result as a precision/recall *reweighting* and notes a deployment that prioritizes catching hate "may prefer a different operating point." We also add that on VOZ-HSD (higher coverage) HATE *recall* is higher (0.6803, Table 7). A full PR-curve analysis is noted as future work in the Limitations.

**M2 — "No comparison to published ViHSD SOTA."**
- **A:** Correct; we compare only to models we trained under a common protocol.
- **C:** §6.1 now says a comparison to published ViHSD results is left to future work; also noted in Limitations. (Disclosed, not resolved.)

**M3 — "Citations are unverified placeholders."**
- **A:** The bibliographic details for AmpleHate, ViTHSD, and VOZ-HSD still require verification against the primary sources; we did not want to invent details.
- **C:** `custom.bib` retains explicit TODO notes on the affected entries; verified entries (PhoBERT, BERT, fastText, SupCon) are complete. (OPEN — to be finalized before camera-ready.)

**M4 — "The 'batched-attention bug' claim about the baseline is unsubstantiated and overstated as a contribution."**
- **A:** Agreed. It concerns our own re-implementation, not necessarily the original AmpleHate.
- **C:** §4.6 retitled "Batched attention (implementation note)"; the text now says we treat it "as an implementation correction to our own port rather than a claim about the original AmpleHate," and it is no longer listed as a primary contribution.

**M5 — "Target-coverage measurement is vague ('checked samples')."**
- **A:** We now define coverage precisely and report exact figures.
- **C:** §6.2 defines coverage as the fraction of comments with `T_x ≠ {0}` and reports measured values on the first 500 train/val comments per split: ViHSD 20.0%/18.8% (target), 5.2%/4.2% (attack); VOZ-HSD 45.2%/43.0% and 7.6%/6.0%. We also add a cross-dataset observation (higher coverage on VOZ coincides with larger gains), while stating this is correlational, not causal.

**M6 — "Cue banks: no sizes, protocol, or release; risk of train-set overfitting."**
- **A:** Added.
- **C:** §3.4 now states the banks are small hand-curated lexicons (16 target, 24 attack) seeded from the training split, and flags the train-derivation bias and recall ceiling. Appendix B lists the **full** banks (in the markdown version; the LaTeX gives ASCII-safe examples plus the counts for pdfLaTeX safety). The train-derivation limitation is reiterated in Limitations.

**M7 — "Objective underspecified; the cited SupCon ≠ the loss used."**
- **A:** Fixed both.
- **C:** §2.5 and §4.7 now describe the term as "a pairwise cosine-margin objective inspired by supervised contrastive learning, not the exact SupCon loss." Appendix A reports class weights = inverse frequency `N/(C·n_c)`, margin `m=0.5`, and label smoothing `0.05`.

**M8 — "Qualitative examples are constructed, not real."**
- **A:** Acknowledged; we kept them as clearly-labelled illustrations and flag the substitution.
- **C:** Table 6 caption retains "constructed; replace with real examples," and replacing them with real anonymized instances is noted as remaining work. (Partially addressed.)

---

## Minor issues

**m1 — Figures are placeholders.**
- **C:** Figures 3 and 5 point to the real PNGs in `notebooks/.../output/`; Figures 1 (class distribution) and 2 (architecture) still require drawing and are marked accordingly. (Open, presentation-only.)

**m2 — BiLSTM / PhoBERT-CNN accuracy cells were dashes.**
- **C:** Filled from the notebooks — Tables 3–4 now report accuracy for all baselines (BiLSTM 0.8454/0.8650; PhoBERT-CNN 0.8945/0.9623).

**m3 — Bolding convention / VOZ accuracy.**
- **C:** Table 4 caption clarifies "best per column in bold among models (accuracy for reference; the baseline has the highest accuracy)."

**m4 — VOZ-HSD underdescribed.**
- **C:** §3.2 adds the source (`tarudesu/VOZ-HSD`) and notes that annotation/licensing are documented by the dataset authors.

**m5 — "Generalizes" overstated.**
- **C:** The Conclusion changes "generalizes" to "adapts," and the coverage link is described as "correlational … not yet shown to be causal."

---

## Summary of changes

| Type | Status |
|---|---|
| Reframed comparison as intentional design (C1, M4) | Done |
| Removed ablation section (C2) | Done (author choice) |
| Single-run / significance disclosed (C3) | Disclosed in Limitations |
| Recall trade-off honestly framed (M1) | Done |
| Coverage defined + exact cross-dataset figures (M5) | Done |
| Cue-bank sizes + full list + bias note (M6) | Done |
| Objective specified; SupCon claim fixed (M7) | Done |
| Baseline accuracies filled; captions clarified (m2–m4) | Done |
| Tempered "generalizes" (m5) | Done |
| Citations verification (M3), SOTA (M2), multi-seed (C3), PR-curve (M1), figures 1–2 (m1), real qualitative examples (M8) | Open — disclosed as limitations / future work |

We believe these revisions make the manuscript's claims faithful to the evidence actually obtained, while being transparent about what remains to be run. We welcome further guidance.
