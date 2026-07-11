from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, XLMRobertaModel, get_linear_schedule_with_warmup


MODEL_REGISTRY = {
    "xlm-roberta": "FacebookAI/xlm-roberta-base",
    "viclsr": "huynhtin/ViCLSR",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Cannot find any of these columns: {candidates}. Existing: {list(df.columns)}")


def normalize_labels(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    label_col = find_column(df, ["label_id", "label", "labels"])
    text_col = find_column(df, ["free_text", "texts", "text", "comment", "content"])
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    if dataset_name == "vihsd":
        df["label"] = (df["label"] == 2).astype(int)
    return df.reset_index(drop=True)


def sample_vozhsd(
    df: pd.DataFrame,
    seed: int,
    policy: str,
    sample_size: int,
    target_hate_ratio: float,
) -> tuple[pd.DataFrame, float, tuple[float, float, float]]:
    sample_size = min(sample_size, len(df))
    if policy == "baseline":
        fraction = sample_size / len(df)
        sampled_df = pd.concat(
            [group.sample(frac=fraction, random_state=seed) for _, group in df.groupby("label")],
            ignore_index=True,
        ).sample(frac=1, random_state=seed).reset_index(drop=True)
        return sampled_df, float(sampled_df["label"].mean()), (0.8, 0.1, 0.1)

    if policy == "proposed":
        hate_df = df[df["label"] == 1]
        non_hate_df = df[df["label"] == 0]
        n_hate = min(int(sample_size * target_hate_ratio), len(hate_df))
        n_non_hate = min(sample_size - n_hate, len(non_hate_df))
        sampled_df = pd.concat(
            [
                non_hate_df.sample(n=n_non_hate, random_state=seed),
                hate_df.sample(n=n_hate, random_state=seed),
            ],
            ignore_index=True,
        ).sample(frac=1, random_state=seed).reset_index(drop=True)
        return sampled_df, float(sampled_df["label"].mean()), (0.75, 0.125, 0.125)

    raise ValueError("--voz-split-policy must be proposed or baseline")


def split_by_ratios(df: pd.DataFrame, ratios: tuple[float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_ratio, val_ratio, _ = ratios
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df


def load_hate_speech_dataset(
    dataset_name: str,
    seed: int,
    voz_split_policy: str,
    voz_sample_size: int,
    voz_hate_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if dataset_name == "vihsd":
        ds = load_dataset("sonlam1102/vihsd")
        train_df = normalize_labels(ds["train"].to_pandas(), dataset_name)
        val_df = normalize_labels(ds["validation"].to_pandas(), dataset_name)
        test_df = normalize_labels(ds["test"].to_pandas(), dataset_name)
        return train_df, val_df, test_df

    if dataset_name == "vozhsd":
        ds = load_dataset("tarudesu/VOZ-HSD", split="train")
        df = normalize_labels(ds.to_pandas(), dataset_name)
        sampled_df, hate_ratio, ratios = sample_vozhsd(
            df,
            seed=seed,
            policy=voz_split_policy,
            sample_size=voz_sample_size,
            target_hate_ratio=voz_hate_ratio,
        )
        print(
            f"VOZ-HSD policy={voz_split_policy} | sampled={len(sampled_df):,}/{len(df):,} "
            f"| hate_ratio={hate_ratio:.4f} | split={ratios}"
        )
        return split_by_ratios(sampled_df, ratios)

    raise ValueError("--dataset must be vihsd or vozhsd")


def tiny_stratified_sample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("label"):
        parts.append(group.sample(min(len(group), per_class), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int):
        enc = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class ViCLSRProjection(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dense(features)


def load_viclsr_encoder(model_name: str) -> XLMRobertaModel:
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    encoder = XLMRobertaModel.from_pretrained(model_name, use_safetensors=False)
    encoder.mlp = ViCLSRProjection(encoder.config.hidden_size)
    ckpt_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    encoder.mlp.dense.weight = nn.Parameter(state_dict["mlp.dense.weight"].clone())
    encoder.mlp.dense.bias = nn.Parameter(state_dict["mlp.dense.bias"].clone())
    del state_dict
    print("Loaded ViCLSR projection head: mlp.dense.weight/bias", flush=True)
    return encoder


class EncoderClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1, use_viclsr_head: bool = False):
        super().__init__()
        self.use_viclsr_head = use_viclsr_head
        self.encoder = load_viclsr_encoder(model_name) if use_viclsr_head else AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        if self.use_viclsr_head:
            pooled = F.normalize(self.encoder.mlp(pooled), dim=-1)
        logits = self.classifier(self.dropout(pooled))
        return logits, pooled


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(features.device)
    self_mask = torch.eye(labels.size(0), device=features.device)
    positive_mask = positive_mask * (1.0 - self_mask)

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_tensor(0.0)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob_pos.mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = False) -> dict[str, object]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "hate_f1": f1_score(all_labels, all_preds, pos_label=1),
        "report": classification_report(
            all_labels,
            all_preds,
            target_names=["NON-HATE", "HATE"],
            digits=4,
            zero_division=0,
        ),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = MODEL_REGISTRY[args.model]
    out_dir = Path(args.output_dir) / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} -> {model_name}")

    train_df, val_df, test_df = load_hate_speech_dataset(
        args.dataset,
        args.seed,
        voz_split_policy=args.voz_split_policy,
        voz_sample_size=args.voz_sample_size,
        voz_hate_ratio=args.voz_hate_ratio,
    )
    if args.smoke_test:
        train_df = tiny_stratified_sample(train_df, per_class=8, seed=args.seed)
        val_df = tiny_stratified_sample(val_df, per_class=4, seed=args.seed)
        test_df = tiny_stratified_sample(test_df, per_class=4, seed=args.seed)
        print("SMOKE TEST MODE: using tiny stratified samples.")
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{split_name}: {len(df):,} rows | labels={df['label'].value_counts().sort_index().to_dict()}")

    gc.collect()
    print("Tokenizing dataset...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data = TextDataset(train_df, tokenizer, args.max_len)
    val_data = TextDataset(val_df, tokenizer, args.max_len)
    test_data = TextDataset(test_df, tokenizer, args.max_len)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("Loading model weights...", flush=True)
    model = EncoderClassifier(
        model_name=model_name,
        dropout=args.dropout,
        use_viclsr_head=args.model == "viclsr",
    ).to(device)
    print("Model ready. Starting training...", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    class_counts = train_df["label"].value_counts().sort_index().to_numpy()
    class_weights = (class_counts.sum() / (len(class_counts) * class_counts)).astype("float32")
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    best_macro_f1 = -1.0
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits, features = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = ce_loss(logits, labels)
                if args.model == "viclsr" and args.contrastive_weight > 0:
                    loss = loss + args.contrastive_weight * supervised_contrastive_loss(
                        features, labels, temperature=args.temperature
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)
        mean_loss = running_loss / max(len(train_loader), 1)
        print(
            f"epoch={epoch} train_loss={mean_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_hate_f1={val_metrics['hate_f1']:.4f}"
        )

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            torch.save(model.state_dict(), best_path)
            print(f"saved best checkpoint -> {best_path}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, use_amp=use_amp)
    print("\nTEST")
    print(json.dumps({k: v for k, v in test_metrics.items() if k != "report"}, indent=2))
    print(test_metrics["report"])

    test_metrics["config"] = vars(args)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    print(f"metrics -> {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vihsd", "vozhsd"], default="vihsd")
    parser.add_argument("--model", choices=["xlm-roberta", "viclsr"], required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/viamplehate_runs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voz-split-policy", choices=["proposed", "baseline"], default="baseline")
    parser.add_argument("--voz-sample-size", type=int, default=100_000)
    parser.add_argument("--voz-hate-ratio", type=float, default=0.10)
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny end-to-end check before full training.")
    parser.add_argument("--fp16", action="store_true", help="Use CUDA mixed precision to reduce VRAM and training time.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
