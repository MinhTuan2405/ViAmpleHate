# Figure 2 — ViAmpleHate architecture (Mermaid)

How to render & export:
- Open https://mermaid.live → paste the code → Export PNG/SVG.
- Or preview directly in VS Code (install the "Markdown Preview Mermaid Support" extension).
- Name the exported file `fig2_architecture.pdf`/`.png` to match `\includegraphics` in the LaTeX.

---

## Version 1 — Horizontal flow (compact, good for a two-column figure)

```mermaid
flowchart LR
    A([Vietnamese comment]) --> B[Preprocessing<br/><small>normalize · word-segment · teencode</small>]

    B --> C[PhoBERT encoder<br/><small>sentence representation</small>]
    B --> D[Target & Attack detection<br/><small>Vietnamese NER + cue banks</small>]

    D --> E[Relation attention<br/><small>target · context · attack</small>]

    C --> F{{Adaptive gate<br/><small>how much relation evidence to use</small>}}
    E --> F

    F --> G[Classifier]
    G --> H([NON-HATE / HATE])

    classDef base fill:#ECEFF1,stroke:#90A4AE,stroke-width:1px,color:#263238;
    classDef novel fill:#FFE0B2,stroke:#FB8C00,stroke-width:2px,color:#5D4037;
    classDef io fill:#E1F5FE,stroke:#039BE5,stroke-width:1px,color:#01579B;

    class B,C,G base;
    class D,E,F novel;
    class A,H io;
```

> Orange = the three core contributions (Vietnamese target/attack detection · 3-view relation attention · adaptive gate).
> Gray = standard components (preprocessing, PhoBERT, classifier). Blue = input/output.

---

## Version 2 — With subgraphs (shows the two parallel branches clearly)

```mermaid
flowchart LR
    A([Vietnamese comment]) --> P[Preprocessing]

    subgraph ENC [Semantic understanding]
        direction TB
        C[PhoBERT encoder]
    end

    subgraph TGT [Target & attack detection]
        direction TB
        D1[Vietnamese NER + target cues] --> R
        D2[Attack cues] --> R
        R[Relation attention<br/><small>target · context · attack</small>]
    end

    P --> C
    P --> D1
    P --> D2

    C --> G{{Adaptive gate}}
    R --> G
    G --> CLF[Classifier] --> O([NON-HATE / HATE])

    classDef base fill:#ECEFF1,stroke:#90A4AE,stroke-width:1px,color:#263238;
    classDef novel fill:#FFE0B2,stroke:#FB8C00,stroke-width:2px,color:#5D4037;
    classDef io fill:#E1F5FE,stroke:#039BE5,stroke-width:1px,color:#01579B;

    class P,C,CLF base;
    class D1,D2,R,G novel;
    class A,O io;
    style ENC fill:#FAFAFA,stroke:#CFD8DC,stroke-dasharray:4 3;
    style TGT fill:#FFF8E1,stroke:#FFB74D,stroke-dasharray:4 3;
```

---

## Insert into LaTeX (after exporting the image)

```latex
\includegraphics[width=0.9\linewidth]{fig2_architecture}
```
and remove `\textbf{[DRAW NEW]}` from the Figure 2 caption.
