# ViHSD - Baseline XLM-RoBERTa + ViCLSR

Kaggle notebook: `vihsd-baseline-xlm-roberta-viclsr.ipynb`.

Notebook fine-tune hai model trên các split chính thức của `sonlam1102/vihsd`:

- `FacebookAI/xlm-roberta-base`
- `huynhtin/ViCLSR`

Nhãn `CLEAN` và `OFFENSIVE` được gộp thành `NON-HATE`; nhãn `HATE` giữ nguyên. Cấu hình mặc định dùng seed 42, 5 epochs, max length 128. Chạy hai smoke test trước, sau đó chạy lần lượt hai cell train đầy đủ.

Kết quả đã chạy nằm trong `output/`. Checkpoint `best_model.pt` không commit vào Git vì dung lượng lớn; notebook lưu checkpoint vào `/kaggle/working/viamplehate_runs_vihsd_seed42/`.
