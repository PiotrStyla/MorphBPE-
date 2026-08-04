# MorphBPE: Can Morphology-Aware Tokenization Beat BPE for Polish?

## What if your tokenizer actually understood Polish word structure?

Tokenization is the foundation of every LLM — it determines how text is split into units the model processes. Standard BPE (Byte-Pair Encoding) learns merges purely from frequency statistics, completely ignoring linguistic structure. This often produces tokens that split words in the middle of morphemes, fragmenting meaningful units like roots, prefixes, and suffixes.

**MorphBPE** changes this by penalizing merges that cross morphological boundaries. Using Morfeusz2 (a Polish morphological analyzer), we detect suffix boundaries and multiply the merge score by `(1 - α)` when a merge would cross one. This simple modification produces tokenizers that respect word structure — and the results are striking.

## The Experiment

We ran a comprehensive evaluation across:
- **3 vocabulary sizes**: 8k, 16k, 32k
- **5 alpha values**: 0.0 (standard BPE), 0.5, 1.0, 1.5, 2.0
- **2 model architectures**: LSTM and Transformer language models
- **30 experiments total**, trained on 8,000 Polish parliamentary speeches (Sejm corpus) with sequence length 32

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

The original MorphBPE paper defines α ∈ [0, 1]. We extended it beyond 1.0 and found that α=2.0 consistently outperforms α=1.0 by an additional 2-3% CE reduction. This suggests that fully blocking boundary-crossing merges isn't enough — actively deprioritizing them produces better tokenizations.

### 3. α=1.5 is a sweet spot

8k α=1.5 Transformer achieves PPL=40.90, beating 32k standard BPE Transformer (PPL=52.49) — that's **22% lower perplexity with 4× smaller vocabulary**. This means smaller embedding layers, less memory, and faster inference, all with better language modeling quality.

### 4. Smaller vocabularies benefit more

At α=2.0, the 8k and 16k tokenizers perform nearly identically (CE=3.632 vs 3.627), while 32k is slightly worse (CE=3.635). Standard BPE shows the opposite pattern — larger vocab is better. MorphBPE inverts this: morphological awareness compensates for smaller vocabulary size.

### 5. Transformer > LSTM consistently

The Transformer outperforms LSTM by 3.8-6.2% CE across all configurations, and the gap widens at larger vocabularies. But the alpha effect is consistent across both architectures — morphology helps regardless of model choice.

## Cost-Benefit: MorphBPE Pareto-Dominates BPE

| Metric | BPE 32k | MorphBPE 8k α=2.0 | Advantage |
|--------|---------|---------------------|-----------|
| Vocab size | 32,000 | 8,000 | **4× smaller** |
| CE (Transformer) | 3.960 | 3.632 | **8.3% lower** |
| PPL (Transformer) | 52.49 | 37.79 | **28% lower** |

MorphBPE 8k α=2.0 Pareto-dominates BPE 32k: it's better on every metric while using 4× fewer tokens. This means 4× smaller embedding layers, 4× less memory for token lookup tables, and better language modeling quality.

## What's Next?

The gains at α=2.0 haven't plateaued — the 1.5→2.0 step still shows substantial improvement. We're now exploring α=2.5 to find where the trend reverses (at very high alpha, over-segmentation should eventually hurt).

All code, results, and visualizations are open source:
- Code & experiments: github.com/PiotrStyla/MorphBPE-
- Full paper (Polish): included in the repo

## Why This Matters

For morphologically rich languages like Polish, tokenization isn't just a preprocessing step — it's a design decision that impacts model quality as much as architecture choice. MorphBPE shows that injecting linguistic knowledge into this step produces measurable, consistent gains across model types and vocabulary sizes.

The fact that a 4× smaller MorphBPE tokenizer beats standard BPE has practical implications: smaller models, faster training, lower inference cost — all from a one-line change to the merge scoring function.

---

*This work is the first evaluation of MorphBPE on Polish with Morfeusz2, including intrinsic evaluation (fertility, MorphED), extrinsic evaluation (LSTM and Transformer language models), and scaling to production vocabulary sizes (32k).*
