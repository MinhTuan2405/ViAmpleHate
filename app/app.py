from __future__ import annotations

import pandas as pd
import streamlit as st
import torch

from model_runtime import MODEL_SPECS, build_predictor


st.set_page_config(
    page_title="ViAmpleHate Demo",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_predictor(spec_id: str):
    return build_predictor(spec_id)


def run_model(spec_id: str, text: str, device_mode: str) -> dict:
    predictor = get_predictor(spec_id)
    return predictor.predict(text, device_mode=device_mode)


st.title("ViAmpleHate Baseline Demo")
st.caption("Nhập một câu văn bản và chạy qua 10 baseline models: ViHSD + VOZ-HSD, TF-IDF/LR/SVM, BiLSTM, PhoBERT-CNN, AmpleHate-PhoBERT.")

with st.sidebar:
    st.header("Cấu hình")
    cuda_available = torch.cuda.is_available()
    default_device = "auto" if cuda_available else "cpu"
    device_mode = st.radio(
        "Thiết bị inference",
        options=["auto", "cpu", "cuda"],
        index=["auto", "cpu", "cuda"].index(default_device),
        help="Với GTX 3050 4GB, app chạy model PyTorch tuần tự và offload về CPU sau mỗi lượt để tránh CUDA OOM.",
    )
    st.write(f"CUDA available: `{cuda_available}`")
    if cuda_available:
        st.write(f"GPU: `{torch.cuda.get_device_name(0)}`")

    selected_ids = st.multiselect(
        "Models",
        options=[spec.id for spec in MODEL_SPECS],
        default=[spec.id for spec in MODEL_SPECS],
        format_func=lambda spec_id: next(spec.display_name for spec in MODEL_SPECS if spec.id == spec_id),
    )

text = st.text_area(
    "Input text",
    value="Tao ghét cái loại người như mày, xéo đi cho khuất mắt",
    height=120,
    placeholder="Nhập một câu tiếng Việt...",
)

run_clicked = st.button("Chạy qua các model đã chọn", type="primary", use_container_width=True)

if run_clicked:
    if not text.strip():
        st.warning("Vui lòng nhập văn bản trước khi chạy model.")
    elif not selected_ids:
        st.warning("Vui lòng chọn ít nhất một model.")
    else:
        rows = []
        progress = st.progress(0)
        status = st.empty()

        for index, spec_id in enumerate(selected_ids, start=1):
            spec = next(spec for spec in MODEL_SPECS if spec.id == spec_id)
            status.info(f"Đang chạy {index}/{len(selected_ids)}: {spec.display_name}")
            try:
                result = run_model(spec_id, text, device_mode)
                rows.append(
                    {
                        "Dataset": result["dataset"],
                        "Model": result["model"],
                        "Final Prediction": result["prediction"],
                        "Probabilities": result["prob_text"],
                        "NON-HATE": result["non_hate_prob"],
                        "HATE": result["hate_prob"],
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "Dataset": spec.dataset,
                        "Model": spec.model,
                        "Final Prediction": "ERROR",
                        "Probabilities": "",
                        "NON-HATE": None,
                        "HATE": None,
                    }
                )
            progress.progress(index / len(selected_ids))

        status.success("Hoàn tất inference.")
        df = pd.DataFrame(rows)

        st.subheader("Kết quả cuối cùng của từng model")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.caption("Lần chạy đầu có thể lâu vì phải load checkpoint và tải/cache PhoBERT từ HuggingFace nếu máy chưa có sẵn.")
else:
    st.info("Bấm nút chạy để predict qua 10 model. Lần đầu load PhoBERT có thể mất vài phút.")
