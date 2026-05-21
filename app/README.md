# ViAmpleHate Demo App

Streamlit app chạy cùng một input qua 10 baseline models trong `notebooks/models/baselines`.

## Chạy local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
streamlit run app\app.py
```

Nếu muốn dùng CUDA trên Windows, cài PyTorch theo lệnh chính thức từ https://pytorch.org/get-started/locally/ trước khi chạy `pip install -r app\requirements.txt`.

## Ghi chú

- App load model theo cache của Streamlit, nên lần đầu sẽ chậm hơn các lần sau.
- Với GPU 4GB VRAM, app chạy các model PyTorch tuần tự và offload về CPU sau mỗi lượt inference.
- Nếu chưa có cache HuggingFace, lần đầu dùng PhoBERT sẽ cần internet để tải `vinai/phobert-base`.
