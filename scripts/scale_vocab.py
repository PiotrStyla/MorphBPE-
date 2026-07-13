"""Scale test: train MorphBPE at vocab 8k/16k/32k on full Sejm corpus.

Measures fertility, MorphED, and training time to verify that
alpha=2.0 > alpha=1.0 holds at production vocab sizes.

Usage:
    python scripts/scale_vocab.py
"""

import json
import os
import random
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from morphbpe import MorphBPETokenizer

CORPUS_ZIP = "speeches_corpus_all.zip"
CORPUS_URL = "https://huggingface.co/datasets/PiotrSty/sejm-speeches-corpus/resolve/main/speeches_corpus_all.zip"
RESULTS_FILE = "scale_vocab_results.txt"


def load_corpus(n_train: int = 8000, n_test: int = 500, seed: int = 42):
    if not os.path.exists(CORPUS_ZIP):
        print(f"Downloading {CORPUS_URL}...")
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
    train = texts[:n_train]
    test = texts[n_train : n_train + n_test]
    print(f"Corpus: {len(train)} train, {len(test)} test")
    return train, test


def run_experiment(train, test, vocab_size, alpha):
    print(f"\n{'='*60}")
    print(f"vocab={vocab_size}, alpha={alpha}")
    print(f"{'='*60}")

    tok = MorphBPETokenizer(
        vocab_size=vocab_size,
        alpha=alpha,
        constraint="suffix",
        use_morfeusz=False,  # heuristic for speed on CPU
    )
    t = tok.train(train)
    fert = tok.fertility(test)
    med_mean, med_lo, med_hi = tok.morph_edit_distance_ci(test, n_resamples=500)

    result = {
        "vocab_size": vocab_size,
        "alpha": alpha,
        "actual_vocab": tok.get_vocab_size(),
        "merges": len(tok.merges),
        "fertility": fert,
        "morphed": med_mean,
        "morphed_ci": (med_lo, med_hi),
        "time_s": t,
    }
    print(
        f"  vocab={tok.get_vocab_size()}, fert={fert:.3f}, "
        f"MorphED={med_mean:.3f} [{med_lo:.3f}, {med_hi:.3f}], "
        f"time={t:.1f}s"
    )

    # Save model
    model_dir = f"models/morphbpe-pl-v{vocab_size}-a{alpha}"
    tok.save(model_dir)
    print(f"  Saved → {model_dir}/")

    return result


def main():
    train, test = load_corpus()

    configs = [
        (8000, 0.0),
        (8000, 1.0),
        (8000, 2.0),
        (16000, 0.0),
        (16000, 1.0),
        (16000, 2.0),
        (32000, 0.0),
        (32000, 1.0),
        (32000, 2.0),
    ]

    results = []
    for vocab_size, alpha in configs:
        r = run_experiment(train, test, vocab_size, alpha)
        results.append(r)

        # Save incrementally
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            f.write("Vocab Scaling Results: MorphBPE suffix\n")
            f.write("=" * 80 + "\n")
            f.write(f"Train: 8000 texts, Test: 500 texts (heuristic backend)\n\n")
            f.write(
                f"{'Vocab':>6} {'Alpha':>5} {'ActVocab':>8} "
                f"{'Fertility':>10} {'MorphED':>8} {'CI':>20} {'Time':>8}\n"
            )
            f.write("-" * 80 + "\n")
            for res in results:
                lo, hi = res["morphed_ci"]
                f.write(
                    f"{res['vocab_size']:>6} {res['alpha']:>5.1f} "
                    f"{res['actual_vocab']:>8} {res['fertility']:>10.4f} "
                    f"{res['morphed']:>8.4f} [{lo:.4f}, {hi:.4f}] "
                    f"{res['time_s']:>8.1f}s\n"
                )
        print(f"  Results saved → {RESULTS_FILE}")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
