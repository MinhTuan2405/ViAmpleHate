# ViHSD XLM-RoBERTa vs ViCLSR Results

This folder stores lightweight evaluation artifacts from the Kaggle run.

- `vihsd_xlmr_viclsr_results.csv`: compact metric summary.
- `metrics/xlm-roberta_metrics.json`: full classification report for `FacebookAI/xlm-roberta-base`.
- `metrics/viclsr_metrics.json`: full classification report for `huynhtin/ViCLSR`.

Large checkpoints (`best_model.pt`) are intentionally not committed to GitHub.

## Summary

| Dataset | Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|---:|
| ViHSD | XLM-RoBERTa-base | 0.9109 | 0.7484 | 0.5461 |
| ViHSD | ViCLSR | 0.8970 | 0.4729 | 0.0000 |
