"""Eval with existing model: compare MorphBPE vs Bielik tokenizer.

Step 1: Fertility comparison on Sejm test set
Step 2: FOCUS-style embedding remapping + perplexity (if model fits RAM)

Usage:
    python scripts/eval_with_model.py
"""

import json
import os
import random
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from morphbpe import MorphBPETokenizer

CORPUS_ZIP = "speeches_corpus_all.zip"
CORPUS_URL = "https://huggingface.co/datasets/PiotrSty/sejm-speeches-corpus/resolve/main/speeches_corpus_all.zip"
RESULTS_FILE = "eval_with_model_results.txt"


def load_test_corpus(n: int = 500, seed: int = 42):
    if not os.path.exists(CORPUS_ZIP):
        r = requests.get(CORPUS_URL, stream=True)
        with open(CORPUS_ZIP, "wb") as f:
            for chunk in tqdm(r.iter_content(8192)):
                f.write(chunk)

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
    return texts[:n]


def fertility_comparison(test_texts):
    """Compare fertility: Bielik tokenizer vs MorphBPE variants."""
    from transformers import AutoTokenizer

    results = []

    # ── Reference tokenizers ────────────────────────────────────────
    reference_models = [
        ("speakleash/Bielik-11B-v2", "Bielik-11B-v2"),
        ("Allegro/herbert-base-cased", "HerBERT-base-cased"),
    ]

    bielik_tok = None
    bielik_name = None
    bielik_vocab = 0
    for repo_id, display_name in reference_models:
        try:
            print(f"Loading {display_name} tokenizer...")
            bielik_tok = AutoTokenizer.from_pretrained(repo_id)
            bielik_name = display_name
            bielik_vocab = len(bielik_tok.get_vocab())
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if bielik_tok is None:
        print("WARNING: No reference tokenizer loaded. Skipping comparison.")
        return []

    bielik_vocab = len(bielik_tok.get_vocab())
    bielik_tokens = 0
    bielik_words = 0
    for text in test_texts:
        words = text.split()
        bielik_words += len(words)
        ids = bielik_tok.encode(text, add_special_tokens=False)
        bielik_tokens += len(ids)

    bielik_fert = bielik_tokens / bielik_words if bielik_words else 0
    results.append({
        "name": f"{bielik_name} (vocab={bielik_vocab})",
        "vocab": bielik_vocab,
        "fertility": bielik_fert,
        "tokens": bielik_tokens,
        "words": bielik_words,
    })
    print(f"  {bielik_name}: vocab={bielik_vocab}, fertility={bielik_fert:.4f}")

    # ── MorphBPE variants ───────────────────────────────────────────
    # Train on same corpus as scale_vocab (8000 texts)
    train_texts = load_train_texts(8000)

    for alpha in [0.0, 1.0, 2.0]:
        for vocab in [1000]:
            label = f"MorphBPE α={alpha} (vocab={vocab})"
            print(f"\nTraining {label}...")
            tok = MorphBPETokenizer(
                vocab_size=vocab, alpha=alpha,
                constraint="suffix", use_morfeusz=False,
            )
            tok.train(train_texts)

            fert = tok.fertility(test_texts)
            n_tokens = sum(
                len(tok.encode_word(w)) for t in test_texts for w in t.split()
            )
            n_words = sum(len(t.split()) for t in test_texts)

            results.append({
                "name": label,
                "vocab": tok.get_vocab_size(),
                "fertility": fert,
                "tokens": n_tokens,
                "words": n_words,
            })
            print(f"  {label}: fertility={fert:.4f}")

    return results


def load_train_texts(n: int = 8000, seed: int = 42):
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
    return texts[:n]


def save_results(results):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("Eval with Existing Model: Fertility Comparison\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test corpus: 500 Sejm speeches\n\n")
        f.write(f"{'Method':<35} {'Vocab':>8} {'Fertility':>10} {'Tokens':>8} {'Words':>8}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(
                f"{r['name']:<35} {r['vocab']:>8} "
                f"{r['fertility']:>10.4f} {r['tokens']:>8} {r['words']:>8}\n"
            )
    print(f"\nResults saved → {RESULTS_FILE}")


def main():
    print("Loading test corpus...")
    test_texts = load_test_corpus(500)
    print(f"Test: {len(test_texts)} texts")

    results = fertility_comparison(test_texts)
    save_results(results)

    print("\n\nFertility Comparison Summary:")
    print(f"{'Method':<35} {'Vocab':>8} {'Fertility':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<35} {r['vocab']:>8} {r['fertility']:>10.4f}")


if __name__ == "__main__":
    main()
