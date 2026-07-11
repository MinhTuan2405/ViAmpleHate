# VOZ-HSD - Baseline XLM-RoBERTa + ViCLSR

Kaggle notebook: `voz-hsd-baseline-xlm-roberta-viclsr.ipynb`.

Notebook fine-tune hai model trên `tarudesu/VOZ-HSD`:

- `FacebookAI/xlm-roberta-base`
- `huynhtin/ViCLSR`

Cách lấy dữ liệu khớp các notebook baseline VOZ-HSD trong repo: lấy mẫu phân tầng 100.000 dòng, giữ tỉ lệ nhãn tự nhiên, xáo trộn với seed 42 rồi chia train/dev/test theo tỉ lệ 80/10/10.

Cấu hình mặc định dùng max length 128. XLM-RoBERTa-base chạy 5 epochs với batch size 8; ViCLSR chạy 2 epochs với FP16 và batch size 4 do dùng XLM-RoBERTa-Large và tốn thời gian/VRAM hơn đáng kể. Batch ViCLSR phải lớn hơn 1 để supervised contrastive loss có positive pairs trong batch. Vì training budget khác nhau, kết quả này là so sánh baseline trong giới hạn tài nguyên Kaggle, không phải so sánh compute-matched. Mỗi `metrics.json` lưu toàn bộ cấu hình thực tế trong trường `config`.

Trên Kaggle, bật Internet và GPU T4, mở notebook, chạy hai smoke test rồi chạy XLM-RoBERTa và ViCLSR. Kết quả được lưu vào `/kaggle/working/viamplehate_runs_vozhsd_seed42/`.

Checkpoint `best_model.pt` không commit vào Git vì dung lượng lớn. Sau khi train xong, dùng Save Version hoặc tạo Kaggle Dataset để giữ checkpoint.

## Kết quả VOZ-HSD

Kết quả dưới đây được ghi lại từ hai full run hoàn tất trên Kaggle T4 (`returncode=0`). Cả hai model dùng cùng sample 100.000 dòng, split và seed; training budget khác nhau như đã mô tả ở trên.

| Model | Epochs | Batch | FP16 | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|:---:|---:|---:|---:|
| XLM-RoBERTa-base | 5 | 8 | No | 0.9706 | 0.8458 | 0.7072 |
| ViCLSR | 2 | 4 | Yes | 0.9486 | 0.4868 | 0.0000 |

XLM-RoBERTa đạt kết quả tốt nhất ở epoch 5 theo validation macro-F1 (`0.8524`), với validation HATE-F1 `0.7210`. Trên test set, HATE precision/recall/F1 lần lượt là `0.7245/0.6907/0.7072`.

ViCLSR không học được lớp thiểu số trong cấu hình này: cả hai epoch có validation HATE-F1 bằng `0.0000`, và trên test set model dự đoán toàn bộ mẫu là `NON-HATE`. Accuracy `0.9486` vì `NON-HATE` chiếm 9.486/10.000 mẫu, nên macro-F1 và HATE-F1 phản ánh chất lượng phù hợp hơn accuracy.

Các metrics và classification report đã được lưu lại trong `output/metrics/`; bảng tổng hợp nằm tại `output/vozhsd_xlmr_viclsr_results.csv`. Checkpoint Kaggle không thể khôi phục từ log sau khi session restart và không có trong repository.
