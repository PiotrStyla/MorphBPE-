MorphBPE: Can Morphology-Aware Tokenization Beat BPE for Polish?

36 experiments. 6 alpha values. One clear conclusion.

What if your tokenizer actually understood Polish word structure?

Standard BPE (Byte-Pair Encoding) learns merges purely from frequency statistics — it doesn't know what a root, prefix, or suffix is. This often produces tokens that split words in the middle of morphemes, fragmenting meaningful units.

MorphBPE (based on Asgari et al., QCRI, arXiv:2502.00894) changes this by penalizing merges that cross morphological boundaries. Using Morfeusz2 (a Polish morphological analyzer), we detect suffix boundaries and multiply the merge score by (1 - α). The extension to α > 1.0 (active deprioritization) is our own modification — the original work describes a hard block, not a continuous weighting parameter.

One simple change to the scoring function. The results speak for themselves.

═══════════════════════════════════════
THE EXPERIMENT
═══════════════════════════════════════

• 3 vocabulary sizes: 8k, 16k, 32k
• 6 alpha values: 0.0 (BPE), 0.5, 1.0, 1.5, 2.0, 2.5
• 2 architectures: LSTM + Transformer
• 36 experiments total
• Corpus: 8,000 Polish Sejm speeches (train), 500 (test), seq_len=32, 3 epochs
• Single runs, no seeds — this is a pilot study

Alpha controls the penalty:
  α=0.0 → standard BPE (no penalty)
  α=1.0 → full block (crossing merges get zero score)
  α=2.0 → over-penalization (crossing merges actively deprioritized)
  α=2.5 → extreme penalty (where does it break?)

═══════════════════════════════════════
RESULTS: FULL ALPHA SWEEP
═══════════════════════════════════════

▸ Transformer results (8k vocab)

  α     |  CE     |  PPL   | Δ vs BPE
  ------+--------+--------+---------
  0.0   | 3.9140 | 50.10  |    —
  0.5   | 3.8552 | 47.24  |  −1.5%
  1.0   | 3.7743 | 43.57  |  −3.6%
  1.5   | 3.7112 | 40.90  |  −5.2%
  2.0   | 3.6319 | 37.79  |  −7.2%
  2.5   | 3.6210 | 37.37  |  −7.5%  ★ best

▸ Transformer results (16k vocab)

  α     |  CE     |  PPL   | Δ vs BPE
  ------+--------+--------+---------
  0.0   | 3.9473 | 51.80  |    —
  0.5   | 3.8966 | 49.24  |  −1.3%
  1.0   | 3.7789 | 43.77  |  −4.3%
  1.5   | 3.7188 | 41.21  |  −5.8%
  2.0   | 3.6272 | 37.61  |  −8.1%  ★ best
  2.5   | 3.6276 | 37.62  |  −8.1%

▸ Transformer results (32k vocab)

  α     |  CE     |  PPL   | Δ vs BPE
  ------+--------+--------+---------
  0.0   | 3.9605 | 52.49  |    —
  0.5   | 3.9135 | 50.07  |  −1.2%
  1.0   | 3.7803 | 43.83  |  −4.5%
  1.5   | 3.7210 | 41.31  |  −6.0%
  2.0   | 3.6347 | 37.89  |  −8.2%  ★ best
  2.5   | 3.6403 | 38.10  |  −8.1%

▸ LSTM results (all vocab sizes)

  Vocab |  α   |  CE     |  PPL
  ------+------+--------+------
   8k   | 0.0  | 4.1073 | 60.78
   8k   | 0.5  | 4.0433 | 57.01
   8k   | 1.0  | 3.9243 | 50.62
   8k   | 1.5  | 3.8849 | 48.66
   8k   | 2.0  | 3.7758 | 43.63  ★
   8k   | 2.5  | 3.7871 | 44.13
  16k   | 0.0  | 4.1494 | 63.40
  16k   | 0.5  | 4.1132 | 61.14
  16k   | 1.0  | 3.9870 | 53.89
  16k   | 1.5  | 3.9287 | 50.84
  16k   | 2.0  | 3.7984 | 44.63  ★
  16k   | 2.5  | 3.8305 | 46.09
  32k   | 0.0  | 4.2140 | 67.62
  32k   | 0.5  | 4.1380 | 62.68
  32k   | 1.0  | 4.0106 | 55.18
  32k   | 1.5  | 3.9687 | 52.92
  32k   | 2.0  | 3.8427 | 46.65  ★
  32k   | 2.5  | 3.8777 | 48.31

═══════════════════════════════════════
KEY FINDINGS
═══════════════════════════════════════

1️⃣ α=2.0 is the sweet spot

CE improves monotonically from α=0.0 to α=2.0 across all vocab sizes and both architectures. At α=2.5, improvements plateau or reverse:

  • 8k Transformer: α=2.5 still slightly better (CE 3.621 vs 3.632)
  • 16k Transformer: α=2.5 essentially identical to α=2.0 (3.6276 vs 3.6272)
  • 32k Transformer: α=2.5 worse than α=2.0 (3.640 vs 3.635)
  • LSTM: α=2.5 worse than α=2.0 across all vocab sizes

Verdict: α=2.0 is the optimal penalty. Going beyond doesn't help (and starts to hurt at larger vocabularies).

2️⃣ CE reduction grows with alpha — then plateaus

  α=0.5 →  0.9–1.8% reduction
  α=1.0 →  3.6–4.8%
  α=1.5 →  5.2–6.0%
  α=2.0 →  7.2–8.8%
  α=2.5 →  7.2–8.1% (plateau)

3️⃣ A 4× smaller vocab beats standard BPE

8k α=1.5 Transformer (PPL=40.90) beats 32k BPE Transformer (PPL=52.49) — 22% lower perplexity with 4× smaller vocabulary. That means smaller embedding layers, less memory, faster inference.

4️⃣ Best absolute results

  • Best overall: 8k α=2.5 Transformer — CE=3.6210, PPL=37.37
  • Best practical: 8k α=2.0 Transformer — CE=3.6319, PPL=37.79 (consistent across all vocabs)
  • Transformer beats LSTM by 3.8–6.2% CE across all configurations

═══════════════════════════════════════
COST-BENEFIT: HONEST TRADE-OFF
═══════════════════════════════════════

MorphBPE 8k α=2.0 vs BPE 32k:

  Metric                    | BPE 32k | MorphBPE 8k | Winner
  --------------------------+---------+-------------+--------
  Vocab size                | 32,000  |  8,000      | MorphBPE (4× smaller)
  CE Transformer (↓ better) |  3.960  |  3.632      | MorphBPE (−8.3%)
  PPL Transformer (↓ better)| 52.49   | 37.79       | MorphBPE (−28%)
  Fertility (↓ better)      |  1.020  |  1.149      | BPE 32k (+12.6%)
  MorphED (↓ better)        |  0.389  |  0.456      | BPE 32k (+17.2%)

This is NOT Pareto dominance. MorphBPE 8k wins on LM quality and memory, but pays with higher fertility (+12.6%) and worse MorphED (+17.2%). That's expected — with 4× fewer tokens, it segments more aggressively.

The striking result: despite worse intrinsic segmentation metrics, the morphology-aware 8k tokenizer still produces lower cross-entropy than 32k BPE.

Within a fixed vocab size, higher α improves MorphED too (8k: 0.574 → 0.456 = −20.5% from BPE to α=2.0). The trade-off only appears when comparing across different vocab sizes.

═══════════════════════════════════════
IMPORTANT CAVEATS
═══════════════════════════════════════

This is a pilot study, not a benchmark:

• Small corpus (8k sentences) — single runs, no seeds. Differences of thousandths of CE have no measured variance.
• CE/PPL aren't directly comparable across vocab sizes without bits-per-character normalization. The "8k beats 32k" claim needs BPC validation (planned).
• Small data favors smaller vocabs — 32k is under-trained on rare tokens, which may partly explain MorphBPE 8k's edge.
• No claims about production-scale transfer at this corpus size.

═══════════════════════════════════════
WHY THIS MATTERS
═══════════════════════════════════════

For morphologically rich languages like Polish, tokenization isn't just preprocessing — it's a design decision that impacts model quality as much as architecture choice.

MorphBPE shows that injecting linguistic knowledge into the merge scoring function produces measurable, consistent gains across model types and vocabulary sizes. A 4× smaller morphology-aware tokenizer matching or beating standard BPE on downstream quality points to a promising direction: smaller models, less memory, lower inference cost — from a one-line change.

These pilot-scale results now warrant validation with bits-per-character metrics, multiple seeds, and larger corpora.

All code, results, and data: github.com/PiotrStyla/MorphBPE-

---

#NLP #Tokenization #BPE #MorphBPE #Polish #NLProc #MachineLearning #LanguageModels #Morphology #DeepLearning #AI #OpenSource
