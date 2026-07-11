# Kaggle setup: XLM-RoBERTa + ViCLSR

Mục tiêu: chạy hai model cho bài toán Vietnamese hate speech detection:

- `FacebookAI/xlm-roberta-base`: baseline theo tài liệu XLM-RoBERTa của Hugging Face.
- `huynhtin/ViCLSR`: checkpoint ViCLSR từ paper `arXiv:2603.21084`, fine-tune thêm classifier nhị phân `NON-HATE/HATE`.

File notebook đã chuẩn bị sẵn:

```text
notebooks/models/experiments/ViHSD - XLM-RoBERTa_ViCLSR/viamplehate-xlm-roberta-viclsr.ipynb
```

Bạn có thể upload notebook này lên Kaggle và chạy từng cell. Các lệnh CLI bên dưới là bản tương đương nếu muốn chạy bằng terminal/cell shell.

Notebook có sẵn 2 smoke-test cell. Chạy smoke test trước để xác nhận dataset, model loading, forward/backward, và output path đều ổn rồi mới train full.

## 1. Tạo Kaggle Notebook

1. Vào Kaggle -> Create Notebook.
2. Bật Accelerator: `GPU T4 x2` hoặc ít nhất `GPU T4`.
3. Bật Internet: `Settings -> Internet -> On`.
4. Upload notebook ở đường dẫn trên, rồi clone repo trực tiếp trong notebook; hoặc upload cả repo.

Clone trực tiếp:

```bash
!git clone -b trung-dev https://github.com/MinhTuan2405/ViAmpleHate.git
%cd ViAmpleHate
```

Nếu bạn đang upload repo thủ công thì `%cd` vào đúng thư mục chứa repo.

## 2. Cài thư viện

```bash
!pip install -q -r "notebooks/models/experiments/ViHSD - XLM-RoBERTa_ViCLSR/requirements.txt"
```

Nếu Kaggle báo version conflict nhẹ, restart session rồi chạy lại từ đầu.

## 3. Chạy trên ViHSD

ViHSD mirror notebook gốc: tải `sonlam1102/vihsd`, dùng các split `train/validation/test`, rồi map nhãn `CLEAN/OFFENSIVE -> NON-HATE`, `HATE -> HATE`.

XLM-RoBERTa:

```bash
!python "notebooks/models/experiments/ViHSD - XLM-RoBERTa_ViCLSR/run_two_models.py" \
  --dataset vihsd \
  --model xlm-roberta \
  --epochs 3 \
  --batch-size 8 \
  --eval-batch-size 16
```

ViCLSR:

```bash
!python "notebooks/models/experiments/ViHSD - XLM-RoBERTa_ViCLSR/run_two_models.py" \
  --dataset vihsd \
  --model viclsr \
  --epochs 3 \
  --batch-size 1 \
  --eval-batch-size 2
```

`huynhtin/ViCLSR` khá nặng vì dùng XLM-RoBERTa-Large và projection head 1024→1024, nên nếu bị CUDA OOM thì giảm `--batch-size 1` hoặc `--max-len 128`.

## 4. Chạy trên VOZ-HSD

VOZ-HSD trên Hugging Face chỉ có một split `train`, giống các notebook gốc trong repo.

Script có 2 policy:

- `--voz-split-policy proposed`: mirror notebook `Proposed ViAmpleHate_PhoBERT` cho VOZ-HSD, sample `40_000`, ép HATE ratio `0.10`, split `75/12.5/12.5`.
- `--voz-split-policy baseline`: mirror các notebook baseline VOZ-HSD, giữ tỉ lệ class tự nhiên, split `80/10/10`.

```bash
!python "notebooks/models/experiments/ViHSD - XLM-RoBERTa_ViCLSR/run_two_models.py" --dataset vozhsd --model xlm-roberta --epochs 3 --batch-size 8 --voz-split-policy proposed
!python "notebooks/models/experiments/ViHSD - XLM-RoBERTa_ViCLSR/run_two_models.py" --dataset vozhsd --model viclsr --epochs 3 --batch-size 1 --eval-batch-size 2 --voz-split-policy proposed
```

Nếu muốn so với baseline notebooks cũ thì đổi thành `--voz-split-policy baseline`. Với baseline BiLSTM gốc còn cần fastText vectors, nhưng XLM-RoBERTa/ViCLSR không dùng file `cc.vi.300.vec.gz`.

## 5. Output

Kết quả được lưu ở:

```text
/kaggle/working/viamplehate_runs/<dataset>/<model>/
├── best_model.pt
├── metrics.json
└── tokenizer/
```

Metrics chính cần báo cáo:

- `accuracy`
- `macro_f1`
- `hate_f1`
- classification report trong log và `metrics.json`

Các artifact nhẹ đã lưu trong:

```text
notebooks/models/experiments/ViHSD - XLM-RoBERTa_ViCLSR/output/
├── README.md
├── vihsd_xlmr_viclsr_results.csv
└── metrics/
    ├── xlm-roberta_metrics.json
    └── viclsr_metrics.json
```

Checkpoint `best_model.pt` không được commit vì mỗi file có thể nặng hàng GB. Nếu cần giữ checkpoint, lưu qua Kaggle Output, Google Drive, Hugging Face Hub, hoặc Git LFS.

## 6. ViHSD Results

Run configuration:

- Dataset: `sonlam1102/vihsd`
- Seed: `42`
- Epochs: `5`
- Max sequence length: `128`
- Metrics: test split

| Dataset | Model | Accuracy | Macro-F1 | HATE-F1 |
|---|---:|---:|---:|---:|
| ViHSD | XLM-RoBERTa-base | 0.9109 | 0.7484 | 0.5461 |
| ViHSD | ViCLSR | 0.8970 | 0.4729 | 0.0000 |

XLM-RoBERTa-base performs better on ViHSD in this direct fine-tuning setup. ViCLSR reaches high accuracy by predicting the majority `NON-HATE` class, but its `HATE-F1` is `0.0000`, showing why macro-F1 and HATE-F1 are more informative than accuracy for this imbalanced dataset.

## 7. Ghi chú đúng với hai link bạn gửi

Tài liệu Hugging Face mô tả XLM-RoBERTa là multilingual masked language model, có thể dùng cho downstream task như text classification qua Transformers. Paper ViCLSR đề xuất supervised contrastive learning cho sentence representations tiếng Việt và công bố checkpoint `huynhtin/ViCLSR`; script này load đúng `XLMRobertaModel` cùng MLP projection head `mlp.dense`, sau đó fine-tune classifier cho dataset hate speech trong repo.
