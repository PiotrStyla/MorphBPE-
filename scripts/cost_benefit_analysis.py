"""Cost-benefit analysis: MorphBPE vs BPE.

Combines intrinsic (fertility, MorphED) and extrinsic (CE, PPL) results
to show MorphBPE dominates on the Pareto frontier of quality vs cost.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

PLOTS_DIR = "plots"

# Data from scale_vocab_results.txt (intrinsic)
INTRINSIC = {
    # (vocab, alpha): (fertility, morph_ed, ci_lo, ci_hi, train_time_s)
    (8000, 0.0):  (1.1307, 0.5735, 0.5688, 0.5783, 8386.6),
    (8000, 1.0):  (1.1405, 0.5042, 0.4992, 0.5087, 6343.2),
    (8000, 2.0):  (1.1487, 0.4561, 0.4513, 0.4604, 5824.9),
    (16000, 0.0): (1.0506, 0.4418, 0.4377, 0.4461, 10410.2),
    (16000, 1.0): (1.0641, 0.3872, 0.3835, 0.3912, 11540.2),
    (16000, 2.0): (1.0772, 0.3383, 0.3345, 0.3421, 11521.3),
    (32000, 0.0): (1.0202, 0.3893, 0.3854, 0.3933, 21526.3),
    (32000, 1.0): (1.0352, 0.3469, 0.3433, 0.3506, 19021.8),
    (32000, 2.0): (1.0491, 0.3076, 0.3041, 0.3113, 21193.7),
}

# Data from extrinsic_eval_8k_32k_results.txt (extrinsic)
# (vocab, alpha, model): (ce, ppl)
EXTRINSIC = {
    (8000, 0.0, "LSTM"):         (4.1073, 60.78),
    (8000, 0.0, "Transformer"):  (3.9140, 50.10),
    (8000, 1.0, "LSTM"):         (3.9243, 50.62),
    (8000, 1.0, "Transformer"):  (3.7743, 43.57),
    (8000, 2.0, "LSTM"):         (3.7758, 43.63),
    (8000, 2.0, "Transformer"):  (3.6319, 37.79),
    (16000, 0.0, "LSTM"):        (4.1494, 63.40),
    (16000, 0.0, "Transformer"): (3.9473, 51.80),
    (16000, 1.0, "LSTM"):        (3.9870, 53.89),
    (16000, 1.0, "Transformer"): (3.7789, 43.77),
    (16000, 2.0, "LSTM"):        (3.7984, 44.63),
    (16000, 2.0, "Transformer"): (3.6272, 37.61),
    (32000, 0.0, "LSTM"):        (4.2140, 67.62),
    (32000, 0.0, "Transformer"): (3.9605, 52.49),
    (32000, 1.0, "LSTM"):        (4.0106, 55.18),
    (32000, 1.0, "Transformer"): (3.7803, 43.83),
    (32000, 2.0, "LSTM"):        (3.8427, 46.65),
    (32000, 2.0, "Transformer"): (3.6347, 37.89),
}

VOCABS = [8000, 16000, 32000]
ALPHAS = [0.0, 1.0, 2.0]
ALPHA_LABELS = {0.0: "BPE (a=0)", 1.0: "MorphBPE (a=1)", 2.0: "MorphBPE (a=2)"}
COLORS = {0.0: "#4e79a7", 1.0: "#59a14f", 2.0: "#e15759"}
MARKERS = {0.0: "o", 1.0: "s", 2.0: "D"}


def plot_fertility_vs_morphed():
    """Fertility vs MorphED: MorphBPE reduces MorphED at small fertility cost."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for alpha in ALPHAS:
        ferts = [INTRINSIC[(v, alpha)][0] for v in VOCABS]
        meds = [INTRINSIC[(v, alpha)][1] for v in VOCABS]
        ax.scatter(ferts, meds, c=COLORS[alpha], marker=MARKERS[alpha], s=120,
                   label=ALPHA_LABELS[alpha], zorder=5)
        for i, v in enumerate(VOCABS):
            ax.annotate(f"{v//1000}k", (ferts[i], meds[i]),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Fertility (tokens/word)", fontsize=12)
    ax.set_ylabel("MorphED (lower = better morph alignment)", fontsize=12)
    ax.set_title("Fertility vs Morphological Edit Distance", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fertility_vs_morphed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_pareto_ce_vs_vocab():
    """Pareto frontier: CE vs vocab size (cost proxy).
    Shows MorphBPE 8k a=2.0 beats BPE 32k at 4x lower cost."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, model in enumerate(["LSTM", "Transformer"]):
        ax = axes[idx]
        for alpha in ALPHAS:
            vocabs_k = [v // 1000 for v in VOCABS]
            ces = [EXTRINSIC[(v, alpha, model)][0] for v in VOCABS]
            ax.plot(vocabs_k, ces, marker=MARKERS[alpha], linewidth=2,
                    markersize=10, color=COLORS[alpha], label=ALPHA_LABELS[alpha])
        # Highlight Pareto-optimal: 8k a=2.0
        best_ce = EXTRINSIC[(8000, 2.0, model)][0]
        ax.annotate(f"Best: 8k a=2\nCE={best_ce:.3f}",
                    xy=(8, best_ce), xytext=(15, best_ce - 0.15),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="black"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))
        ax.set_xlabel("Vocabulary Size (k) = Cost Proxy", fontsize=12)
        ax.set_ylabel("Cross-Entropy (lower = better)", fontsize=12)
        ax.set_title(f"{model}", fontsize=14, fontweight="bold")
        ax.set_xticks([v // 1000 for v in VOCABS])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Pareto Frontier: CE vs Vocabulary Size (Cost)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pareto_ce_vs_vocab.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_pareto_ce_vs_fertility():
    """CE vs fertility: does MorphBPE's higher fertility pay off?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, model in enumerate(["LSTM", "Transformer"]):
        ax = axes[idx]
        for alpha in ALPHAS:
            ferts = [INTRINSIC[(v, alpha)][0] for v in VOCABS]
            ces = [EXTRINSIC[(v, alpha, model)][0] for v in VOCABS]
            ax.scatter(ferts, ces, c=COLORS[alpha], marker=MARKERS[alpha], s=120,
                       label=ALPHA_LABELS[alpha], zorder=5)
            for i, v in enumerate(VOCABS):
                ax.annotate(f"{v//1000}k", (ferts[i], ces[i]),
                            textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax.set_xlabel("Fertility (tokens/word)", fontsize=12)
        ax.set_ylabel("Cross-Entropy (lower = better)", fontsize=12)
        ax.set_title(f"{model}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("CE vs Fertility: Quality vs Tokenization Cost",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pareto_ce_vs_fertility.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_ce_per_token():
    """CE per token: normalizes for different sequence lengths due to fertility.

    CE_per_token = CE / fertility (approximate, since CE is per position
    and more tokens = more positions but each simpler).
    We use fertility as a proxy for token expansion.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(VOCABS))
    width = 0.25
    for ai, alpha in enumerate(ALPHAS):
        for model in ["Transformer"]:
            ces = [EXTRINSIC[(v, alpha, model)][0] for v in VOCABS]
            ferts = [INTRINSIC[(v, alpha)][0] for v in VOCABS]
            ce_per_tok = [ce / f for ce, f in zip(ces, ferts)]
            ax.bar(x + ai * width, ce_per_tok, width,
                   color=COLORS[alpha], label=ALPHA_LABELS[alpha])
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{v//1000}k" for v in VOCABS], fontsize=12)
    ax.set_xlabel("Vocabulary Size", fontsize=12)
    ax.set_ylabel("CE / Fertility (per-token normalized)", fontsize=12)
    ax.set_title("CE per Token (Transformer)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ce_per_token.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_efficiency_ratio():
    """Efficiency: CE reduction (%) vs vocab size increase (cost).

    Shows that MorphBPE 8k a=2.0 achieves better CE than BPE 32k
    at 4x lower vocab = 4x smaller embedding layer.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bpe_32k_ce = EXTRINSIC[(32000, 0.0, "Transformer")][0]
    bpe_32k_vocab = 32000

    configs = []
    for v in VOCABS:
        for a in ALPHAS:
            ce = EXTRINSIC[(v, a, "Transformer")][0]
            ce_reduction = (bpe_32k_ce - ce) / bpe_32k_ce * 100
            vocab_ratio = bpe_32k_vocab / v
            configs.append((v, a, ce_reduction, vocab_ratio, ce))

    labels = [f"{v//1000}k a={a}" for v, a, _, _, _ in configs]
    ce_reds = [cr for _, _, cr, _, _ in configs]
    vocab_ratios = [vr for _, _, _, vr, _ in configs]

    x = np.arange(len(configs))
    ax.bar(x - 0.15, ce_reds, 0.3, label="CE reduction vs BPE 32k (%)", color="#59a14f")
    ax.bar(x + 0.15, [np.log2(r) for r in vocab_ratios], 0.3,
           label="Vocab reduction (log2 x)", color="#4e79a7")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Efficiency: CE Reduction vs Vocab Savings (Transformer)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "efficiency_ratio.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_cost_benefit():
    print("\n" + "=" * 80)
    print("COST-BENEFIT ANALYSIS: MorphBPE vs BPE")
    print("=" * 80)

    print("\n1. Fertility cost of MorphBPE (tokens/word increase vs BPE):")
    for v in VOCABS:
        bpe_f = INTRINSIC[(v, 0.0)][0]
        for a in [1.0, 2.0]:
            f = INTRINSIC[(v, a)][0]
            pct = (f - bpe_f) / bpe_f * 100
            print(f"  {v//1000}k: a={a} fertility={f:.4f} vs BPE={bpe_f:.4f} ({pct:+.1f}%)")

    print("\n2. MorphED reduction (morphological alignment improvement):")
    for v in VOCABS:
        bpe_m = INTRINSIC[(v, 0.0)][1]
        for a in [1.0, 2.0]:
            m = INTRINSIC[(v, a)][1]
            pct = (bpe_m - m) / bpe_m * 100
            print(f"  {v//1000}k: a={a} MorphED={m:.4f} vs BPE={bpe_m:.4f} ({pct:+.1f}%)")

    print("\n3. CE reduction vs BPE baseline (Transformer):")
    for v in VOCABS:
        bpe_ce = EXTRINSIC[(v, 0.0, "Transformer")][0]
        for a in [1.0, 2.0]:
            ce = EXTRINSIC[(v, a, "Transformer")][0]
            pct = (bpe_ce - ce) / bpe_ce * 100
            print(f"  {v//1000}k: a={a} CE={ce:.4f} vs BPE={bpe_ce:.4f} ({pct:+.1f}%)")

    print("\n4. Pareto dominance (Transformer):")
    bpe_32k_ce = EXTRINSIC[(32000, 0.0, "Transformer")][0]
    print(f"  BPE 32k baseline: CE={bpe_32k_ce:.4f}, vocab=32000")
    for v in VOCABS:
        for a in ALPHAS:
            ce = EXTRINSIC[(v, a, "Transformer")][0]
            dominates = ce < bpe_32k_ce and v < 32000
            if dominates:
                savings = 32000 / v
                print(f"  >> MorphBPE {v//1000}k a={a}: CE={ce:.4f} ({savings:.0f}x smaller vocab, CE -{bpe_32k_ce-ce:.4f})")

    print("\n5. Key insight:")
    best_8k = EXTRINSIC[(8000, 2.0, "Transformer")][0]
    worst_32k = EXTRINSIC[(32000, 0.0, "Transformer")][0]
    print(f"  MorphBPE 8k a=2.0 (CE={best_8k:.4f}) BEATS BPE 32k (CE={worst_32k:.4f})")
    print(f"  with 4x smaller vocabulary = 4x smaller embedding layer")
    print(f"  CE improvement: {(worst_32k - best_8k) / worst_32k * 100:.1f}%")
    print(f"  PPL improvement: {EXTRINSIC[(32000, 0.0, 'Transformer')][1]} -> {EXTRINSIC[(8000, 2.0, 'Transformer')][1]}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print_cost_benefit()
    plot_fertility_vs_morphed()
    plot_pareto_ce_vs_vocab()
    plot_pareto_ce_vs_fertility()
    plot_ce_per_token()
    plot_efficiency_ratio()
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
