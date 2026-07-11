# VOZ-HSD outputs

Sau khi chạy xong hai model, đưa các file kết quả nhẹ vào đây:

- `metrics/xlm-roberta_metrics.json`
- `metrics/viclsr_metrics.json`
- `vozhsd_xlmr_viclsr_results.csv`

Không commit `best_model.pt` hoặc tokenizer dung lượng lớn vào Git.

Các file metrics hiện tại được khôi phục từ completed Kaggle run logs sau khi session chứa checkpoint bị restart. Chúng giữ nguyên test metrics, classification report và cấu hình lệnh chạy; không có checkpoint đi kèm.
