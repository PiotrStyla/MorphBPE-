# MorphBPE: Can Morphology-Aware Tokenization Beat BPE for Polish?

## What if your tokenizer actually understood Polish word structure?

Tokenization is the foundation of every LLM — it determines how text is split into units the model processes. Standard BPE (Byte-Pair Encoding) learns merges purely from frequency statistics, completely ignoring linguistic structure. This often produces tokens that split words in the middle of morphemes, fragmenting meaningful units like roots, prefixes, and suffixes.

**MorphBPE** (based on Asgari et al., QCRI, arXiv:2502.00894) changes this by penalizing merges that cross morphological boundaries. Using Morfeusz2 (a Polish morphological analyzer), we detect suffix boundaries and multiply the merge score by `(1 - α)` when a merge would cross one. The extension to α > 1.0 (active deprioritization) is our own modification — the original QCRI work describes a hard block, not a continuous weighting parameter. This simple change produces tokenizers that respect word structure — and the pilot results are encouraging.

## The Experiment

We ran a **pilot evaluation** (small corpus, single runs, no seeds) across:
- **3 vocabulary sizes**: 8k, 16k, 32k
- **5 alpha values**: 0.0 (standard BPE), 0.5, 1.0, 1.5, 2.0
- **2 model architectures**: LSTM and Transformer language models
- **30 experiments total**, trained on 8,000 Polish parliamentary speeches (Sejm corpus), 500 test, sequence length 32, 3 epochs

Alpha controls the penalty strength:
- α=0.0 → standard BPE (no penalty)
- α=1.0 → full block (merges crossing boundaries get zero score)
- α=2.0 → over-penalization (crossing merges get negative score, actively deprioritized)

## Key Findings

### 1. Monotonic improvement with alpha

Cross-entropy decreases consistently as alpha increases. For the Transformer at 8k vocab:

| Alpha | CE | PPL | Reduction vs BPE |
|-------|-----|-----|------------------|
| 0.0 (BPE) | 3.914 | 50.10 | — |
| 0.5 | 3.855 | 47.24 | 1.5% |
| 1.0 | 3.774 | 43.57 | 3.6% |
| 1.5 | 3.711 | 40.90 | 5.2% |
| 2.0 | 3.632 | 37.79 | 7.2% |

The trend holds across all vocabulary sizes and both architectures.

### 2. Over-penalization works

The original MorphBPE paper defines α ∈ [0, 1]. We extended it beyond 1.0 and found that α=2.0 consistently outperforms α=1.0 by an additional ~4% relative CE reduction (3.8–4.0% across vocab sizes). This suggests that fully blocking boundary-crossing merges isn't enough — actively deprioritizing them produces better tokenizations.

### 3. α=1.5 is a sweet spot

8k α=1.5 Transformer achieves PPL=40.90, beating 32k standard BPE Transformer (PPL=52.49) — that's **22% lower perplexity with 4× smaller vocabulary**. This means smaller embedding layers, less memory, and faster inference. *(Caveat: PPL across different vocab sizes isn't strictly comparable without bits-per-character normalization — see caveats below.)*

### 4. Small vocabularies stay competitive under MorphBPE

At α=2.0, the 8k and 16k tokenizers perform nearly identically (CE=3.632 vs 3.627), and even 8k stays within a whisker of 32k (CE=3.635). With standard BPE, larger vocab usually helps — here, morphological awareness lets a small vocab keep pace. *One caveat: on this small pilot corpus, larger vocabularies are under-trained (rare tokens), so part of this effect may be a data-size artifact rather than pure tokenizer quality.*

### 5. Transformer > LSTM consistently

The Transformer outperforms LSTM by 3.8-6.2% CE across all configurations, and the gap widens at larger vocabularies. But the alpha effect is consistent across both architectures — morphology helps regardless of model choice.

## Cost-Benefit: A Real Trade-Off (not "free lunch")

Comparing MorphBPE 8k α=2.0 against a 4× larger BPE 32k tokenizer reveals a genuine **trade-off** — better on some metrics, worse on others:

| Metric | BPE 32k | MorphBPE 8k α=2.0 | Winner |
|--------|---------|---------------------|--------|
| Vocab size | 32,000 | 8,000 | **MorphBPE (4× smaller)** |
| CE Transformer (↓ better) | 3.960 | 3.632 | **MorphBPE (−8.3%)** |
| PPL Transformer (↓ better) | 52.49 | 37.79 | **MorphBPE (−28%)** |
| Fertility (↓ better) | 1.020 | 1.149 | BPE 32k (+12.6% cost) |
| MorphED (↓ better) | 0.389 | 0.456 | BPE 32k (+17.2% worse) |

**This is not Pareto dominance.** MorphBPE 8k wins decisively on language-model quality (CE/PPL) and memory footprint, but its tokens are less morphologically clean (higher MorphED) and it splits words into more pieces (higher fertility) than the 4× larger BPE 32k vocab. That's expected — with 4× fewer tokens it *has* to segment more aggressively. The striking result is that despite worse intrinsic segmentation, the morphology-aware 8k tokenizer still produces **lower LM cross-entropy** than 32k BPE.

Note: within a *fixed* vocab size, higher α genuinely improves MorphED too (e.g. 8k: 0.574 → 0.456 = −20.5% from BPE to α=2.0). The trade-off above only appears when comparing *across* different vocab sizes.

### Important caveats (this is a pilot, not a benchmark)

- **Small corpus**: 8,000 sentences, seq_len=32, 3 epochs, **single runs with no seeds** — differences of a few thousandths of CE have no measured variance.
- **CE/PPL aren't directly comparable across vocab sizes.** Different tokenization granularity affects per-token entropy independently of quality. A rigorous "8k beats 32k" claim needs **bits-per-character** normalization — which we're adding next.
- **Small data favors smaller vocabs**: 32k is under-trained on rare tokens with so little data, which may partly explain the 8k edge.
- Claims about transfer to production-scale models are premature at this corpus size.

## What's Next?

The gains at α=2.0 haven't plateaued — the 1.5→2.0 step still shows substantial improvement. We're now exploring α=2.5 to find where the trend reverses (at very high alpha, over-segmentation should eventually hurt).

All code, results, and visualizations are open source:
- Code & experiments: github.com/PiotrStyla/MorphBPE-
- Full paper (Polish): included in the repo

## Why This Matters

For morphologically rich languages like Polish, tokenization isn't just a preprocessing step — it's a design decision that impacts model quality as much as architecture choice. MorphBPE shows that injecting linguistic knowledge into this step produces measurable, consistent gains across model types and vocabulary sizes.

The fact that a 4× smaller MorphBPE tokenizer matches or beats standard BPE on downstream LM quality — even while accepting slightly higher fertility — points to a promising direction: smaller models, less memory, lower inference cost, from a one-line change to the merge scoring function. These are pilot-scale results that now warrant validation with bits-per-character metrics, multiple seeds, and larger corpora.

---

*This work is the first evaluation of MorphBPE on Polish with Morfeusz2, including intrinsic evaluation (fertility, MorphED), extrinsic evaluation (LSTM and Transformer language models), and scaling to production vocabulary sizes (32k).*
