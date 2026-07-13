"""Benchmark: Dawidek (2026) metrics — SI, ISS, MFL.

Compares BPE (alpha=0) vs MorphBPE alpha=1.0 vs alpha=2.0
on 10 Polish lexemes (including suppletive iść, być).

Usage:
    python scripts/benchmark_dawidek.py
"""

import json
import os
import random
import zipfile

import requests
from tqdm import tqdm

from morphbpe import MorphBPETokenizer
from morphbpe.dawidek_metrics import evaluate_all, print_report

CORPUS_ZIP = "speeches_corpus_all.zip"
CORPUS_URL = "https://huggingface.co/datasets/PiotrSty/sejm-speeches-corpus/resolve/main/speeches_corpus_all.zip"
RESULTS_FILE = "dawidek_benchmark_results.txt"


def load_train_corpus(n: int = 8000, seed: int = 42):
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
    return texts[:n]


def main():
    train = load_train_corpus()

    configs = [
        ("BPE (α=0.0)", 0.0),
        ("MorphBPE (α=1.0)", 1.0),
        ("MorphBPE (α=2.0)", 2.0),
    ]

    all_reports = []

    for label, alpha in configs:
        print(f"\n{'='*60}")
        print(f"Training: {label}, vocab=1000, constraint=suffix")
        print(f"{'='*60}")

        tok = MorphBPETokenizer(
            vocab_size=1000,
            alpha=alpha,
            constraint="suffix",
            use_morfeusz=False,
        )
        tok.train(train)

        results, summary = evaluate_all(tok.encode_word)
        report = print_report(results, summary)
        print(f"\n{report}")

        all_reports.append((label, results, summary, report))

    # Save combined report
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for label, results, summary, report in all_reports:
            f.write(f"\n{'#' * 70}\n")
            f.write(f"# {label}\n")
            f.write(f"{'#' * 70}\n\n")
            f.write(report + "\n")

            # Per-form details for suppletive lexemes
            for r in results:
                if r.lemma in ("iść", "być", "człowiek"):
                    f.write(f"\n  Detail: {r.lemma}\n")
                    for pf in r.per_form:
                        si = "✓" if pf["stem_intact"] else "✗"
                        iss = "✓" if pf["suffix_sep"] else "✗"
                        f.write(
                            f"    {pf['form']:<15} → {' | '.join(pf['tokens']):<30} "
                            f"SI={si} ISS={iss}\n"
                        )
            f.write("\n")

        # Comparison table
        f.write("\n" + "=" * 70 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Method':<25} {'MFL':>6} {'SI':>8} {'ISS':>8}\n")
        f.write("-" * 70 + "\n")
        for label, _, summary, _ in all_reports:
            f.write(
                f"{label:<25} {summary['MFL']:>6.2f} "
                f"{summary['SI']:>7.1%} {summary['ISS']:>7.1%}\n"
            )

    print(f"\nResults saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
