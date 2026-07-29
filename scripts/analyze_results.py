"""Statistical analysis and visualization of extrinsic evaluation results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_FILE = "extrinsic_eval_8k_32k_results.txt"
PLOTS_DIR = "plots"

ALPHA_LABELS = {0.0: "BPE (α=0)", 1.0: "MorphBPE (α=1)", 2.0: "MorphBPE (α=2)"}
VOCABS = [8000, 16000, 32000]
ALPHAS = [0.0, 1.0, 2.0]
MODELS = ["LSTM", "Transformer"]
COLORS = {0.0: "#4e79a7", 1.0: "#59a14f", 2.0: "#e15759"}


def load_results():
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


def get_result(results, vocab, alpha, model, metric):
    for r in results:
        if r["vocab_size"] == vocab and r["alpha"] == alpha and r["model"] == model:
            return r[metric]
    return None


def plot_ce_by_vocab(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for alpha in ALPHAS:
            ces = [get_result(results, v, alpha, model, "ce") for v in VOCABS]
            ax.plot(
                [v // 1000 for v in VOCABS], ces,
                marker="o", linewidth=2, markersize=8,
                color=COLORS[alpha], label=ALPHA_LABELS[alpha],
            )
        ax.set_xlabel("Vocabulary Size (k)", fontsize=12)
        ax.set_ylabel("Cross-Entropy", fontsize=12)
        ax.set_title(f"{model}", fontsize=14, fontweight="bold")
        ax.set_xticks([v // 1000 for v in VOCABS])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Cross-Entropy vs Vocabulary Size", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ce_by_vocab.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_ppl_by_vocab(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for alpha in ALPHAS:
            ppls = [get_result(results, v, alpha, model, "ppl") for v in VOCABS]
            ax.plot(
                [v // 1000 for v in VOCABS], ppls,
                marker="s", linewidth=2, markersize=8,
                color=COLORS[alpha], label=ALPHA_LABELS[alpha],
            )
        ax.set_xlabel("Vocabulary Size (k)", fontsize=12)
        ax.set_ylabel("Perplexity", fontsize=12)
        ax.set_title(f"{model}", fontsize=14, fontweight="bold")
        ax.set_xticks([v // 1000 for v in VOCABS])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Perplexity vs Vocabulary Size", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ppl_by_vocab.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_alpha_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(ALPHAS))
    width = 0.25
    for idx, metric in enumerate(["ce", "ppl"]):
        ax = axes[idx]
        for vi, vocab in enumerate(VOCABS):
            vals = [get_result(results, vocab, a, "Transformer", metric) for a in ALPHAS]
            ax.bar(
                x + vi * width, vals, width,
                label=f"vocab={vocab // 1000}k",
            )
        ax.set_xticks(x + width)
        ax.set_xticklabels([ALPHA_LABELS[a] for a in ALPHAS], fontsize=10)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"Transformer {metric.upper()}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Effect of Alpha on Model Performance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "alpha_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_model_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(VOCABS))
    width = 0.12
    for ai, alpha in enumerate(ALPHAS):
        for mi, model in enumerate(MODELS):
            vals = [get_result(results, v, alpha, model, "ce") for v in VOCABS]
            offset = (ai * len(MODELS) + mi) * width - (len(ALPHAS) * len(MODELS) - 1) * width / 2
            label = f"α={alpha} {model}"
            ax.bar(x + offset, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v // 1000}k" for v in VOCABS], fontsize=12)
    ax.set_xlabel("Vocabulary Size", fontsize=12)
    ax.set_ylabel("Cross-Entropy", fontsize=12)
    ax.set_title("CE: All Configurations", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary(results):
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    print("\n1. Alpha effect (CE reduction vs BPE baseline):")
    for model in MODELS:
        for vocab in VOCABS:
            bpe_ce = get_result(results, vocab, 0.0, model, "ce")
            for alpha in [1.0, 2.0]:
                ce = get_result(results, vocab, alpha, model, "ce")
                if bpe_ce and ce:
                    reduction = (bpe_ce - ce) / bpe_ce * 100
                    print(f"  {model} {vocab // 1000}k: α={alpha} -> CE {bpe_ce:.4f} -> {ce:.4f} ({reduction:+.1f}%)")

    print("\n2. Model comparison (Transformer vs LSTM, CE):")
    for vocab in VOCABS:
        for alpha in ALPHAS:
            lstm_ce = get_result(results, vocab, alpha, "LSTM", "ce")
            tr_ce = get_result(results, vocab, alpha, "Transformer", "ce")
            if lstm_ce and tr_ce:
                reduction = (lstm_ce - tr_ce) / lstm_ce * 100
                print(f"  {vocab // 1000}k α={alpha}: LSTM={lstm_ce:.4f} TR={tr_ce:.4f} ({reduction:+.1f}%)")

    print("\n3. Vocab size effect (CE, Transformer α=2.0):")
    for vocab in VOCABS:
        ce = get_result(results, vocab, 2.0, "Transformer", "ce")
        if ce:
            print(f"  {vocab // 1000}k: CE={ce:.4f}")

    print("\n4. Best configuration:")
    best = min(results, key=lambda r: r["ce"])
    print(f"  {best['model']} vocab={best['vocab_size']} alpha={best['alpha']}: CE={best['ce']:.4f} PPL={best['ppl']:.2f}")

    print("\n5. Worst configuration:")
    worst = max(results, key=lambda r: r["ce"])
    print(f"  {worst['model']} vocab={worst['vocab_size']} alpha={worst['alpha']}: CE={worst['ce']:.4f} PPL={worst['ppl']:.2f}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    results = load_results()
    print(f"Loaded {len(results)} results from {RESULTS_FILE}")

    print_summary(results)
    plot_ce_by_vocab(results)
    plot_ppl_by_vocab(results)
    plot_alpha_comparison(results)
    plot_model_comparison(results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
