# Archived outputs

Lightweight artifacts from the completed VOZ-HSD Kaggle runs.

| File | Contents |
|---|---|
| `vozhsd_xlmr_viclsr_results.csv` | Final model comparison |
| `training_history.csv` | Per-epoch train loss and validation metrics |
| `confusion_matrices.csv` | Test confusion matrices in numeric form |
| `metrics/xlm-roberta_metrics.json` | XLM-RoBERTa test report and run configuration |
| `metrics/viclsr_metrics.json` | ViCLSR test report and run configuration |
| `models/vozhsd_xlmr_viclsr_config.json` | Shared experiment and model settings |

The metrics, reports, and histories are archived from successful Kaggle processes (`returncode=0`) and match the outputs embedded in the notebook. Large checkpoints and tokenizer caches are intentionally excluded from Git.
