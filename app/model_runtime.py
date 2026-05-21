from __future__ import annotations

import gc
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from underthesea import word_tokenize
except Exception:  # pragma: no cover - app still runs with basic whitespace tokens.
    word_tokenize = None


ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINES_DIR = ROOT_DIR / "notebooks" / "models" / "baselines"
LABEL_NAMES = ["NON-HATE", "HATE"]


TEENCODE_MAP = {
    "ko": "không",
    "kh": "không",
    "khong": "không",
    "kg": "không",
    "hok": "không",
    "hk": "không",
    "hem": "không",
    "kô": "không",
    "chx": "chưa",
    "chua": "chưa",
    "r": "rồi",
    "rui": "rồi",
    "ròi": "rồi",
    "oy": "rồi",
    "uj": "rồi",
    "mk": "mình",
    "mik": "mình",
    "mh": "mình",
    "tui": "tôi",
    "tau": "tao",
    "may": "mày",
    "mi": "mày",
    "bn": "bạn",
    "ban": "bạn",
    "no": "nó",
    "mng": "mọi người",
    "mn": "mọi người",
    "ae": "anh em",
    "dc": "được",
    "đc": "được",
    "dk": "được",
    "đk": "được",
    "đươc": "được",
    "duoc": "được",
    "vs": "với",
    "voi": "với",
    "j": "gì",
    "zì": "gì",
    "zi": "gì",
    "ntn": "như thế nào",
    "nso": "như sao",
    "biet": "biết",
    "bit": "biết",
    "hieu": "hiểu",
    "nghi": "nghĩ",
    "muon": "muốn",
    "hoac": "hoặc",
    "neu": "nếu",
    "nen": "nên",
    "giet": "giết",
    "chui": "chửi",
    "danh": "đánh",
    "nx": "nhưng",
    "nhg": "nhưng",
    "nhưg": "nhưng",
    "nma": "nhưng mà",
    "cx": "cũng",
    "cg": "cũng",
    "cung": "cũng",
    "cũg": "cũng",
    "ms": "mới",
    "boi": "bởi",
    "oke": "ok",
    "okie": "ok",
    "okê": "ok",
    "okey": "ok",
    "uh": "ừ",
    "uk": "ừ",
    "uhm": "ừ",
    "yep": "đúng",
    "yup": "đúng",
    "haha": "haha",
    "hehe": "hehe",
    "hihi": "hehe",
    "huhu": "buồn",
    "haiz": "thở dài",
    "haizz": "thở dài",
    "wtf": "cái gì vậy",
    "omg": "ôi trời",
    "lol": "buồn cười",
    "lmao": "buồn cười",
    "fck": "chửi thề",
    "fk": "chửi thề",
    "gg": "xong rồi",
    "ez": "dễ",
    "bt": "bình thường",
    "bth": "bình thường",
    "noob": "tệ",
    "nub": "tệ",
    "xàm": "vô nghĩa",
    "nhảm": "vô nghĩa",
    "pro": "giỏi",
    "vl": "vãi lồn",
    "vcl": "vãi cái lồn",
    "vkl": "vãi kép lồn",
    "vll": "vãi lồn",
    "vleu": "vãi lồn",
    "vloz": "vãi lồn",
    "dm": "đụ má",
    "đm": "đụ má",
    "d.m": "đụ má",
    "đ.m": "đụ má",
    "đmm": "đụ má mày",
    "dmm": "đụ má mày",
    "đtm": "địt mẹ",
    "dtm": "địt mẹ",
    "cl": "cái lồn",
    "lon": "lồn",
    "loz": "lồn",
    "l0n": "lồn",
    "đéo": "không",
    "deo": "không",
    "éo": "không",
    "cc": "cái con",
    "thg": "thằng",
    "ngu": "ngu",
    "đần": "đần độn",
    "khùng": "điên",
    "dien": "điên",
    "cút": "cút",
    "cut": "cút",
    "câm": "câm miệng",
    "im mồm": "câm miệng",
    "fb": "facebook",
    "yt": "youtube",
    "tt": "tiktok",
    "zl": "zalo",
    "ig": "instagram",
    "cmt": "bình luận",
    "rep": "trả lời",
    "vn": "việt nam",
    "hn": "hà nội",
    "hcm": "hồ chí minh",
    "sg": "sài gòn",
    "iu": "yêu",
    "ieu": "yêu",
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\b0[0-9]{9,10}\b", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(TEENCODE_MAP.get(word, word) for word in text.split())


def preprocess(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if word_tokenize is None:
        return text
    try:
        return word_tokenize(text, format="text")
    except Exception:
        return text


def tokenize(text: str) -> list[str]:
    processed = preprocess(text)
    tokens = processed.split()
    return tokens if tokens else ["<UNK>"]


class Vocabulary:
    def __init__(self, max_size: int = 15000):
        self.max_size = max_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def encode(self, tokens: list[str], max_len: int | None = None) -> list[int]:
        ml = max_len or 128
        ids = [self.word2idx.get(token, 1) for token in tokens[:ml]]
        ids += [0] * (ml - len(ids))
        return ids


class NotebookUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "Vocabulary":
            return Vocabulary
        return super().find_class(module, name)


def load_notebook_vocab(path: Path) -> Vocabulary:
    with path.open("rb") as file:
        return NotebookUnpickler(file).load()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def softmax_probs(logits: torch.Tensor) -> tuple[float, float]:
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().reshape(-1)
    return float(probs[0]), float(probs[1])


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(device_mode: str) -> torch.device:
    if device_mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def attention_pooling(self, lstm_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        scores = self.attention(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=lstm_out.device)
        mask = mask.unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (lstm_out * weights).sum(dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed_dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (h, _) = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=x.size(1))
        attn_out = self.attention_pooling(lstm_out, lengths)
        last_h = torch.cat([h[-2], h[-1]], dim=-1)
        combined = torch.cat([attn_out, last_h], dim=-1)
        combined = self.layer_norm(combined)
        out = self.dropout(combined)
        out = self.relu(self.bn1(self.fc1(out)))
        out = self.dropout2(out)
        return self.fc2(out)


class PhoBERTCNN(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        num_filters: int,
        kernel_sizes: list[int] | tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.kernel_sizes = tuple(kernel_sizes)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.transpose(1, 2)
        pooled_outputs = []

        for conv, kernel_size in zip(self.convs, self.kernel_sizes):
            conv_out = torch.relu(conv(x))
            valid_mask = F.max_pool1d(
                attention_mask.float().unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
            ).bool()
            conv_out = conv_out.masked_fill(~valid_mask, -1e4)
            pooled_outputs.append(torch.max(conv_out, dim=2).values)

        x = torch.cat(pooled_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class HeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.softmax = nn.Softmax(dim=-1)
        self.W_q = nn.Linear(hidden_dim, head_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, head_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, head_dim, bias=False)

    def forward(self, cls_embedding: torch.Tensor, head_token_embedding: torch.Tensor) -> torch.Tensor:
        q_h = self.W_q(cls_embedding)
        k_h = self.W_k(head_token_embedding)
        v_h = self.W_v(cls_embedding)
        scores = torch.matmul(q_h, k_h.T) / (self.head_dim**0.5)
        weights = self.softmax(scores.float())
        return torch.matmul(weights, v_h)


class AmpleHatePhoBERT(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        num_classes: int,
        e: float,
        dropout: float,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.e = e
        self.head_attention = HeadAttention(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        head_token_idx: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        expanded_idx = head_token_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        head_token_embeddings = torch.gather(outputs.last_hidden_state, 1, expanded_idx)
        outputs_list = [
            self.head_attention(cls_embedding, head_token_embeddings[:, i, :])
            for i in range(head_token_embeddings.shape[1])
        ]
        head_attention_output = sum(outputs_list)
        final_embedding = cls_embedding + head_attention_output * self.e
        final_embedding = self.dropout(final_embedding)
        return self.classifier(final_embedding)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    dataset: str
    model: str
    kind: str
    base_dir: Path
    checkpoint_path: Path | None = None
    config_path: Path | None = None
    vectorizer_path: Path | None = None
    model_path: Path | None = None

    @property
    def display_name(self) -> str:
        return f"{self.dataset} - {self.model}"


def make_specs() -> list[ModelSpec]:
    vihsd_tfidf = BASELINES_DIR / "ViHSD - Baseline TF-IDF LR_SVM"
    voz_tfidf = BASELINES_DIR / "VOZ-HSD - Baseline TF-IDF LR_SVM"
    vihsd_bilstm = BASELINES_DIR / "ViHSD - Baseline BiLSTM_FasttextVi"
    voz_bilstm = BASELINES_DIR / "VOZ-HSD - Baseline BiLSTM_FasttextVi"
    vihsd_cnn = BASELINES_DIR / "ViHSD - Baseline PhoBERT_CNN"
    voz_cnn = BASELINES_DIR / "VOZ-HSD - Baseline PhoBERT_CNN"
    vihsd_ample = BASELINES_DIR / "ViHSD - Baseline AmpleHate_PhoBERT"
    voz_ample = BASELINES_DIR / "VOZ-HSD - Baseline AmpleHate_PhoBERT"

    return [
        ModelSpec(
            id="vihsd_tfidf_lr",
            dataset="ViHSD",
            model="TF-IDF + LR",
            kind="tfidf",
            base_dir=vihsd_tfidf,
            vectorizer_path=vihsd_tfidf / "output" / "tfidf_vectorizer.pkl",
            model_path=vihsd_tfidf / "output" / "lr_model.pkl",
        ),
        ModelSpec(
            id="vihsd_tfidf_svm",
            dataset="ViHSD",
            model="TF-IDF + SVM",
            kind="tfidf",
            base_dir=vihsd_tfidf,
            vectorizer_path=vihsd_tfidf / "output" / "tfidf_vectorizer.pkl",
            model_path=vihsd_tfidf / "output" / "svm_model.pkl",
        ),
        ModelSpec(
            id="vihsd_bilstm",
            dataset="ViHSD",
            model="BiLSTM + FastTextVi",
            kind="bilstm",
            base_dir=vihsd_bilstm,
            checkpoint_path=vihsd_bilstm / "output" / "best_bilstm_vihsd.pt",
            config_path=vihsd_bilstm / "output" / "models" / "config.json",
        ),
        ModelSpec(
            id="vihsd_phobert_cnn",
            dataset="ViHSD",
            model="PhoBERT + CNN",
            kind="phobert_cnn",
            base_dir=vihsd_cnn,
            checkpoint_path=vihsd_cnn / "output" / "best_phobert_cnn_vihsd.pt",
            config_path=vihsd_cnn / "output" / "models" / "config.json",
        ),
        ModelSpec(
            id="vihsd_amplehate",
            dataset="ViHSD",
            model="AmpleHate + PhoBERT",
            kind="amplehate",
            base_dir=vihsd_ample,
            checkpoint_path=vihsd_ample / "output" / "best_amplehate_phobert_vihsd.pt",
            config_path=vihsd_ample / "output" / "models" / "amplehate_config.json",
        ),
        ModelSpec(
            id="voz_tfidf_lr",
            dataset="VOZ-HSD",
            model="TF-IDF + LR",
            kind="tfidf",
            base_dir=voz_tfidf,
            vectorizer_path=voz_tfidf / "output" / "tfidf_vectorizer.pkl",
            model_path=voz_tfidf / "output" / "lr_model.pkl",
        ),
        ModelSpec(
            id="voz_tfidf_svm",
            dataset="VOZ-HSD",
            model="TF-IDF + SVM",
            kind="tfidf",
            base_dir=voz_tfidf,
            vectorizer_path=voz_tfidf / "output" / "tfidf_vectorizer.pkl",
            model_path=voz_tfidf / "output" / "svm_model.pkl",
        ),
        ModelSpec(
            id="voz_bilstm",
            dataset="VOZ-HSD",
            model="BiLSTM + FastTextVi",
            kind="bilstm",
            base_dir=voz_bilstm,
            checkpoint_path=voz_bilstm / "output" / "best_bilstm_vozhsd.pt",
            config_path=voz_bilstm / "output" / "models" / "config.json",
        ),
        ModelSpec(
            id="voz_phobert_cnn",
            dataset="VOZ-HSD",
            model="PhoBERT + CNN",
            kind="phobert_cnn",
            base_dir=voz_cnn,
            checkpoint_path=voz_cnn / "output" / "best_phobert_cnn_vozhsd.pt",
            config_path=voz_cnn / "output" / "models" / "config.json",
        ),
        ModelSpec(
            id="voz_amplehate",
            dataset="VOZ-HSD",
            model="AmpleHate + PhoBERT",
            kind="amplehate",
            base_dir=voz_ample,
            checkpoint_path=voz_ample / "output" / "best_amplehate_phobert_vozhsd.pt",
            config_path=voz_ample / "output" / "models" / "amplehate_vozhsd_config.json",
        ),
    ]


MODEL_SPECS = make_specs()
MODEL_SPEC_BY_ID = {spec.id: spec for spec in MODEL_SPECS}


class BasePredictor:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.config: dict[str, Any] = {}

    def metrics(self) -> dict[str, Any]:
        return {
            "best_val_f1": self.config.get("best_val_f1"),
            "test_macro_f1": self.config.get("test_macro_f1"),
            "test_f1_hate": self.config.get("test_f1_hate"),
            "threshold": self.config.get("best_threshold"),
        }

    def predict(self, text: str, device_mode: str) -> dict[str, Any]:
        raise NotImplementedError


class TfidfPredictor(BasePredictor):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        assert spec.vectorizer_path is not None and spec.model_path is not None
        self.vectorizer = joblib.load(spec.vectorizer_path)
        self.model = joblib.load(spec.model_path)

    def predict(self, text: str, device_mode: str) -> dict[str, Any]:
        start = time.perf_counter()
        processed = preprocess(text)
        x_one = self.vectorizer.transform([processed])
        pred_id = int(self.model.predict(x_one)[0])
        probs = np.zeros(2, dtype=np.float64)

        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(x_one)[0]
            classes = getattr(self.model, "classes_", np.arange(len(raw_probs)))
            for cls, prob in zip(classes, raw_probs):
                cls_id = int(cls)
                if 0 <= cls_id < 2:
                    probs[cls_id] = float(prob)
        else:
            probs[pred_id] = 1.0

        latency_ms = (time.perf_counter() - start) * 1000
        return prediction_payload(self.spec, pred_id, probs, latency_ms, "cpu", processed, self.metrics())


class TorchPredictor(BasePredictor):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self.model: nn.Module

    def _run_with_fallback(self, text: str, device_mode: str) -> dict[str, Any]:
        device = resolve_device(device_mode)
        try:
            return self._predict_on_device(text, device)
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                self.model.to("cpu")
                clear_cuda_cache()
                result = self._predict_on_device(text, torch.device("cpu"))
                result["note"] = "CUDA OOM, fallback CPU"
                return result
            raise
        finally:
            if torch.cuda.is_available():
                self.model.to("cpu")
                clear_cuda_cache()

    def _predict_on_device(self, text: str, device: torch.device) -> dict[str, Any]:
        raise NotImplementedError

    def predict(self, text: str, device_mode: str) -> dict[str, Any]:
        return self._run_with_fallback(text, device_mode)


class BiLSTMPredictor(TorchPredictor):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        assert spec.config_path is not None and spec.checkpoint_path is not None
        self.config = load_json(spec.config_path)
        self.vocab = load_notebook_vocab(spec.base_dir / "output" / "models" / "vocab.pkl")
        self.max_len = int(self.config["MAX_LEN"])
        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocab.word2idx),
            embed_dim=int(self.config["EMBED_DIM"]),
            hidden_dim=int(self.config["HIDDEN_DIM"]),
            num_layers=int(self.config["NUM_LAYERS"]),
            num_classes=int(self.config["NUM_CLASSES"]),
            dropout=float(self.config["DROPOUT"]),
        )
        self.model.load_state_dict(torch_load(spec.checkpoint_path))
        self.model.eval()

    def _predict_on_device(self, text: str, device: torch.device) -> dict[str, Any]:
        start = time.perf_counter()
        tokens = tokenize(text)
        ids = self.vocab.encode(tokens, self.max_len)
        length = max(1, min(len(tokens), self.max_len))
        self.model.to(device)
        self.model.eval()

        with torch.inference_mode():
            x = torch.tensor([ids], dtype=torch.long, device=device)
            lengths = torch.tensor([length], dtype=torch.long, device=device)
            logits = self.model(x, lengths)

        probs = np.array(softmax_probs(logits), dtype=np.float64)
        pred_id = int(probs.argmax())
        latency_ms = (time.perf_counter() - start) * 1000
        return prediction_payload(
            self.spec,
            pred_id,
            probs,
            latency_ms,
            device.type,
            " ".join(tokens),
            self.metrics(),
        )


class PhoBERTCNNPredictor(TorchPredictor):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        assert spec.config_path is not None and spec.checkpoint_path is not None
        self.config = load_json(spec.config_path)
        self.max_len = int(self.config["max_len"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.model = PhoBERTCNN(
            model_name=self.config["model_name"],
            num_classes=int(self.config["num_classes"]),
            num_filters=int(self.config["cnn_filters"]),
            kernel_sizes=self.config["cnn_kernel_sizes"],
            dropout=float(self.config["dropout"]),
        )
        self.model.load_state_dict(torch_load(spec.checkpoint_path))
        self.model.eval()

    def _predict_on_device(self, text: str, device: torch.device) -> dict[str, Any]:
        start = time.perf_counter()
        processed = preprocess(text)
        encoded = self.tokenizer(
            processed,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.model.to(device)
        self.model.eval()

        with torch.inference_mode():
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            logits = self.model(input_ids, attention_mask)

        probs = np.array(softmax_probs(logits), dtype=np.float64)
        pred_id = int(probs.argmax())
        latency_ms = (time.perf_counter() - start) * 1000
        return prediction_payload(self.spec, pred_id, probs, latency_ms, device.type, processed, self.metrics())


class AmpleHatePredictor(TorchPredictor):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        assert spec.config_path is not None and spec.checkpoint_path is not None
        self.config = load_json(spec.config_path)
        self.max_len = int(self.config["max_len"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["encoder"])
        self.model = AmpleHatePhoBERT(
            model_name=self.config["encoder"],
            hidden_dim=int(self.config["hidden_dim"]),
            num_classes=int(self.config["num_classes"]),
            e=float(self.config["e_injection"]),
            dropout=float(self.config["dropout"]),
        )
        checkpoint = torch_load(spec.checkpoint_path)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.threshold = float(
            checkpoint.get("threshold", self.config.get("best_threshold", 0.5))
            if isinstance(checkpoint, dict)
            else self.config.get("best_threshold", 0.5)
        )
        self.config["best_threshold"] = self.threshold
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _predict_on_device(self, text: str, device: torch.device) -> dict[str, Any]:
        start = time.perf_counter()
        processed = preprocess(text)
        encoded = self.tokenizer(
            processed,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.model.to(device)
        self.model.eval()

        with torch.inference_mode():
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            head_token_idx = torch.tensor([[0]], dtype=torch.long, device=device)
            logits = self.model(input_ids, head_token_idx, attention_mask)

        probs = np.array(softmax_probs(logits), dtype=np.float64)
        pred_id = int(probs[1] >= self.threshold)
        latency_ms = (time.perf_counter() - start) * 1000
        return prediction_payload(self.spec, pred_id, probs, latency_ms, device.type, processed, self.metrics())


def prediction_payload(
    spec: ModelSpec,
    pred_id: int,
    probs: np.ndarray,
    latency_ms: float,
    device: str,
    processed: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": spec.dataset,
        "model": spec.model,
        "prediction": LABEL_NAMES[pred_id],
        "non_hate_prob": round(float(probs[0]), 6),
        "hate_prob": round(float(probs[1]), 6),
        "prob_text": f"NON-HATE: {float(probs[0]):.4f} | HATE: {float(probs[1]):.4f}",
        "latency_ms": round(float(latency_ms), 1),
        "device": device,
        "processed": processed,
        "best_val_f1": metrics.get("best_val_f1"),
        "test_macro_f1": metrics.get("test_macro_f1"),
        "test_f1_hate": metrics.get("test_f1_hate"),
        "threshold": metrics.get("threshold"),
        "note": "",
    }


def validate_spec_files(spec: ModelSpec) -> None:
    required_paths = [
        spec.checkpoint_path,
        spec.config_path,
        spec.vectorizer_path,
        spec.model_path,
    ]
    missing = [path for path in required_paths if path is not None and not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(ROOT_DIR)) for path in missing)
        raise FileNotFoundError(f"Missing artifact: {missing_text}")


def build_predictor(spec_id: str) -> BasePredictor:
    spec = MODEL_SPEC_BY_ID[spec_id]
    validate_spec_files(spec)
    if spec.kind == "tfidf":
        return TfidfPredictor(spec)
    if spec.kind == "bilstm":
        return BiLSTMPredictor(spec)
    if spec.kind == "phobert_cnn":
        return PhoBERTCNNPredictor(spec)
    if spec.kind == "amplehate":
        return AmpleHatePredictor(spec)
    raise ValueError(f"Unsupported model kind: {spec.kind}")
