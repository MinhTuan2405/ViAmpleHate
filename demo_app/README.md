# ViAmpleHate Demo App

Streamlit app chạy cùng một input qua các baseline models trong `notebooks/models/baselines` và ViHSD/VOZ-HSD proposed ViAmpleHate++ PhoBERT trong `notebooks/models/proposed`.

## Chạy local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
streamlit run app\app.py
```

Nếu môi trường đã cài dependencies trước đó, chạy lại lệnh upgrade để cập nhật PyTorch lên bản an toàn:

```powershell
pip install --upgrade -r app\requirements.txt
```

Nếu muốn dùng CUDA trên Windows, cài PyTorch theo lệnh chính thức từ https://pytorch.org/get-started/locally/ trước khi chạy `pip install -r app\requirements.txt`.

## Troubleshooting

- Nếu gặp `ModuleNotFoundError: No module named 'torch'`, `'streamlit'`, `'transformers'`, `'sentencepiece'` hoặc `'underthesea'`, môi trường Python hiện tại chưa cài dependency. Kích hoạt đúng virtual environment rồi chạy lại `pip install -r app\requirements.txt`.
- Nếu PhoBERT báo `upgrade torch to at least v2.6`, môi trường đang dùng `torch < 2.6`. Chạy `pip install --upgrade -r app\requirements.txt`, rồi restart Streamlit.
- Các model PhoBERT cần tải/cache `vinai/phobert-base`; model proposed còn cần `NlpHUST/ner-vietnamese-electra-base`. Lần đầu chạy cần internet hoặc cache HuggingFace đã có sẵn.
- Nếu chỉ các model PhoBERT lỗi, kiểm tra thông báo trong cột `Error` của bảng kết quả để biết lỗi cụ thể.

## Ghi chú

- App load model theo cache của Streamlit, nên lần đầu sẽ chậm hơn các lần sau.
- Với GPU 4GB VRAM, app chạy các model PyTorch tuần tự và offload về CPU sau mỗi lượt inference.
- Nếu chưa có cache HuggingFace, lần đầu dùng PhoBERT/NER sẽ cần internet để tải `vinai/phobert-base` và `NlpHUST/ner-vietnamese-electra-base`.
