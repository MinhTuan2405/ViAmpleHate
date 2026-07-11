# VOZ-HSD - Baseline XLM-RoBERTa + ViCLSR

Completed Kaggle baseline comparing `FacebookAI/xlm-roberta-base` and `huynhtin/ViCLSR` on VOZ-HSD. The notebook includes the archived training logs, validation history, test reports, and final comparison from successful runs (`returncode=0`).

## Experiment setup

| Item | Value |
|---|---|
| Dataset | `tarudesu/VOZ-HSD` |
| Sampling | 100,000 stratified examples, natural class distribution |
| Split | 80,000 train / 10,000 validation / 10,000 test |
| Train labels | 75,646 NON-HATE / 4,354 HATE |
| Validation labels | 9,430 NON-HATE / 570 HATE |
| Test labels | 9,486 NON-HATE / 514 HATE |
| Seed | 42 |
| Maximum sequence length | 128 |
| Checkpoint selection | Best validation macro-F1 |
| Runtime | Kaggle, NVIDIA Tesla T4 |

The sampling and `80/10/10` split follow the VOZ-HSD baseline convention used in this repository. Both models use the same sampled data, split, seed, and maximum sequence length.

## Training configuration

| Model | Checkpoint | Epochs | Batch | Eval batch | Precision |
|---|---|---:|---:|---:|---|
| XLM-RoBERTa | `FacebookAI/xlm-roberta-base` | 5 | 8 | 16 | FP32 |
| ViCLSR | `huynhtin/ViCLSR` | 2 | 4 | 8 | FP16 |

ViCLSR uses an XLM-RoBERTa-Large backbone and therefore runs with a smaller training budget under Kaggle T4 constraints. Its batch size remains greater than one so the supervised contrastive objective can form positive pairs. This is a resource-constrained baseline comparison, not a compute-matched comparison.

## Test results

| Model | Accuracy | Macro-F1 | HATE precision | HATE recall | HATE-F1 |
|---|---:|---:|---:|---:|---:|
| XLM-RoBERTa | **0.9706** | **0.8458** | **0.7245** | **0.6907** | **0.7072** |
| ViCLSR | 0.9486 | 0.4868 | 0.0000 | 0.0000 | 0.0000 |

XLM-RoBERTa reached its best validation result at epoch 5 (`macro-F1=0.8524`, `HATE-F1=0.7210`) and retained meaningful precision and recall for the minority HATE class on the test set.

ViCLSR converged to the majority NON-HATE class in this configuration. It classified all 10,000 test examples as NON-HATE, so its high accuracy reflects the class distribution rather than useful HATE detection. Macro-F1 and HATE-F1 are therefore the primary comparison metrics.

## Repository contents

```text
VOZ-HSD - Baseline XLM-RoBERTa_ViCLSR/
├── voz-hsd-baseline-xlm-roberta-viclsr.ipynb
├── run_two_models.py
├── requirements.txt
├── README.md
└── output/
    ├── README.md
    ├── vozhsd_xlmr_viclsr_results.csv
    ├── training_history.csv
    ├── confusion_matrices.csv
    ├── metrics/
    │   ├── xlm-roberta_metrics.json
    │   └── viclsr_metrics.json
    └── models/
        └── vozhsd_xlmr_viclsr_config.json
```

The notebook is the reproducible Kaggle workflow. The lightweight files under `output/` archive the exact configurations, epoch history, confusion matrices, classification reports, and final metrics used in the comparison.

Large `best_model.pt` checkpoints and tokenizer caches are intentionally excluded from Git. A future rerun should preserve them using Kaggle Save Version or a dedicated model store.

## Reproduce on Kaggle

1. Enable Internet and an NVIDIA T4 accelerator.
2. Add the `HF_TOKEN` Kaggle secret.
3. Open `voz-hsd-baseline-xlm-roberta-viclsr.ipynb`.
4. Run the setup and configuration cells.
5. Run XLM-RoBERTa, then ViCLSR, and archive the generated `/kaggle/working/viamplehate_runs_vozhsd_seed42/` directory.

Use `macro_f1` and `hate_f1` as the main reporting metrics because VOZ-HSD is strongly imbalanced.
