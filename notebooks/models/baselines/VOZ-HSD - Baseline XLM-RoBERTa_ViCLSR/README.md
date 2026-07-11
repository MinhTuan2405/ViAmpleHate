# VOZ-HSD - Baseline XLM-RoBERTa + ViCLSR

Kaggle notebook: `voz-hsd-baseline-xlm-roberta-viclsr.ipynb`.

Notebook fine-tune hai model trên `tarudesu/VOZ-HSD`:

- `FacebookAI/xlm-roberta-base`
- `huynhtin/ViCLSR`

Cách lấy dữ liệu khớp các notebook baseline VOZ-HSD trong repo: lấy mẫu phân tầng 100.000 dòng, giữ tỉ lệ nhãn tự nhiên, xáo trộn với seed 42 rồi chia train/dev/test theo tỉ lệ 80/10/10. Cấu hình train mặc định là 5 epochs và max length 128.

Trên Kaggle, bật Internet và GPU T4, mở notebook, chạy hai smoke test rồi chạy XLM-RoBERTa và ViCLSR. Kết quả được lưu vào `/kaggle/working/viamplehate_runs_vozhsd_seed42/`.

Checkpoint `best_model.pt` không commit vào Git vì dung lượng lớn. Sau khi train xong, dùng Save Version hoặc tạo Kaggle Dataset để giữ checkpoint.
