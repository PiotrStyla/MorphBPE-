"""Extrinsic evaluation for MorphBPE at vocab 8k/16k/32k.

Trains small LSTM and Transformer language models on tokenized Sejm corpus
and reports cross-entropy (CE) and perplexity (PPL) for BPE vs MorphBPE.
Matches paper settings: LSTM (1 layer, embed=64, hidden=128) and
Transformer (2 layers, 4 heads, embed=64, ff=256), 3 epochs each.

Usage:
    python scripts/extrinsic_eval_8k_32k.py
"""

import json
import math
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from morphbpe import MorphBPETokenizer

CORPUS_ZIP = "speeches_corpus_all.zip"
CORPUS_URL = (
    "https://huggingface.co/datasets/PiotrSty/sejm-speeches-corpus/"
    "resolve/main/speeches_corpus_all.zip"
)
RESULTS_FILE = "extrinsic_eval_8k_32k_results.txt"
BATCH_SIZE = 128
EPOCHS = 3
LR = 0.001
SEED = 42
N_TRAIN = 8000  # full paper setting
N_TEST = 500
SEQ_LEN = 32     # full paper setting


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_corpus():
    if os.path.exists(CORPUS_ZIP):
        return
    print(f"Downloading {CORPUS_URL}...")
    r = requests.get(CORPUS_URL, stream=True)
    r.raise_for_status()
    with open(CORPUS_ZIP, "wb") as f:
        for chunk in tqdm(r.iter_content(8192)):
            f.write(chunk)


def load_corpus(n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED):
    download_corpus()
    texts = []
    with zipfile.ZipFile(CORPUS_ZIP) as zf:
        jsonl = [f for f in zf.namelist() if f.endswith(".jsonl")][0]
        with zf.open(jsonl) as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("text"):
                    texts.append(d["text"])
    random.seed(seed)
    random.shuffle(texts)
    return texts[:n_train], texts[n_train : n_train + n_test]


class TokenizedDataset(Dataset):
    def __init__(self, token_ids: list[list[int]], seq_len: int = SEQ_LEN):
        self.samples = []
        for ids in token_ids:
            if len(ids) <= seq_len:
                continue
            for i in range(0, len(ids) - seq_len, seq_len):
                x = ids[i : i + seq_len]
                y = ids[i + 1 : i + seq_len + 1]
                self.samples.append((torch.tensor(x, dtype=torch.long),
                                     torch.tensor(y, dtype=torch.long)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out)


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 2,
        max_len: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        emb = self.embedding(x) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        out = self.transformer(emb, mask=mask, is_causal=True)
        return self.fc(out)


def train_model(model: nn.Module, loader: DataLoader, epochs: int, device: torch.device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch}: train loss={avg_loss:.4f}")
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    ce = total_loss / total_tokens
    ppl = math.exp(ce)
    return ce, ppl


def run_experiment(
    train_texts,
    test_texts,
    vocab_size: int,
    alpha: float,
    model_name: str,
    device: torch.device,
    seq_len: int = SEQ_LEN,
):
    label = f"MorphBPE alpha={alpha} vocab={vocab_size}"
    model_dir = Path(f"models/morphbpe-pl-v{vocab_size}-a{alpha:.1f}")

    if model_dir.exists():
        print(f"Loading tokenizer: {model_dir}")
        tok = MorphBPETokenizer.load(str(model_dir))
    else:
        print(f"Training tokenizer: {label}")
        tok = MorphBPETokenizer(
            vocab_size=vocab_size,
            alpha=alpha,
            constraint="suffix",
            use_morfeusz=False,
        )
        tok.train(train_texts)
        tok.save(str(model_dir))

    print(f"Tokenizing {len(train_texts)} train + {len(test_texts)} test texts with {label}...")
    train_ids = [tok.encode(t) for t in tqdm(train_texts, desc="Tokenize train")]
    test_ids = [tok.encode(t) for t in tqdm(test_texts, desc="Tokenize test")]

    actual_vocab = tok.get_vocab_size()
    print(f"Building {model_name} (vocab={actual_vocab})...")
    if model_name == "LSTM":
        model = LSTMLM(actual_vocab, embed_dim=64, hidden_dim=128)
    else:
        model = TransformerLM(
            actual_vocab, embed_dim=64, nhead=4, dim_feedforward=256, num_layers=2
        )

    train_ds = TokenizedDataset(train_ids, seq_len=seq_len)
    test_ds = TokenizedDataset(test_ids, seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training {model_name}...")
    model = train_model(model, train_loader, EPOCHS, device)

    print(f"Evaluating {model_name}...")
    ce, ppl = evaluate(model, test_loader, device)
    print(f"  {model_name} {label}: CE={ce:.4f}, PPL={ppl:.2f}")

    return {
        "vocab_size": vocab_size,
        "alpha": alpha,
        "model": model_name,
        "actual_vocab": actual_vocab,
        "ce": ce,
        "ppl": ppl,
    }


def load_existing_results() -> list[dict]:
    """Parse existing results file to skip already-completed experiments."""
    if not os.path.exists(RESULTS_FILE):
        return []
    results = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                results.append({
                    "vocab_size": int(parts[0]),
                    "alpha": float(parts[1]),
                    "model": parts[2],
                    "actual_vocab": int(parts[3]),
                    "ce": float(parts[4]),
                    "ppl": float(parts[5]),
                })
    return results


def save_results(results: list[dict]):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("Extrinsic Evaluation: MorphBPE vocab 8k-32k\n")
        f.write("=" * 80 + "\n")
        f.write(f"Corpus: {N_TRAIN} train / {N_TEST} test, seq_len={SEQ_LEN}, epochs={EPOCHS}\n\n")
        f.write(
            f"{'Vocab':>8} {'Alpha':>5} {'Model':>12} "
            f"{'ActVocab':>8} {'CE':>10} {'PPL':>10}\n"
        )
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(
                f"{r['vocab_size']:>8} {r['alpha']:>5.1f} {r['model']:>12} "
                f"{r['actual_vocab']:>8} {r['ce']:>10.4f} {r['ppl']:>10.2f}\n"
            )
    print(f"\nResults saved -> {RESULTS_FILE}")


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_texts, test_texts = load_corpus(n_train=N_TRAIN, n_test=N_TEST)
    print(f"Corpus: {len(train_texts)} train, {len(test_texts)} test")

    configs = [
        # (vocab_size, alpha)
        (8000, 0.0),  # BPE baseline
        (8000, 1.0),  # MorphBPE block
        (8000, 2.0),  # MorphBPE over-penalize
        (16000, 0.0),
        (16000, 1.0),
        (16000, 2.0),
        (32000, 0.0),
        (32000, 1.0),
        (32000, 2.0),
    ]
    models = ["LSTM", "Transformer"]

    # Load existing results to skip completed experiments
    results = load_existing_results()
    done = {(r["vocab_size"], r["alpha"], r["model"]) for r in results}
    if done:
        print(f"Skipping {len(done)} already-completed experiments.")

    for vocab_size, alpha in configs:
        for model_name in models:
            if (vocab_size, alpha, model_name) in done:
                continue
            r = run_experiment(
                train_texts, test_texts, vocab_size, alpha, model_name, device
            )
            results.append(r)
            save_results(results)

    print("\n=== SUMMARY ===")
    print(f"{'Vocab':>8} {'Alpha':>5} {'Model':>12} {'CE':>10} {'PPL':>10}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['vocab_size']:>8} {r['alpha']:>5.1f} {r['model']:>12} "
            f"{r['ce']:>10.4f} {r['ppl']:>10.2f}"
        )


if __name__ == "__main__":
    main()
