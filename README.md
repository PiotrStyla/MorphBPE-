# MorphBPE

**Morphologically-aware BPE tokenizer for Polish.**

MorphBPE extends Byte-Pair Encoding with morphological boundary penalties.
Merges that cross morpheme boundaries receive a frequency penalty controlled by `alpha`,
preserving suffix (or prefix) structure during tokenization.

Based on: Styła (2026), *MorphBPE: morfologicznie informowany BPE dla polszczyzny*.

## Installation

```bash
pip install morphbpe

# With Morfeusz2 support (recommended):
pip install morphbpe[morfeusz]
```

## Quick Start

### Python API

```python
from morphbpe import MorphBPETokenizer

# Train
tok = MorphBPETokenizer(vocab_size=32000, alpha=2.0, constraint="suffix")
tok.train(["list of", "training texts", ...])

# Encode / decode
ids = tok.encode("prezydent podpisał ustawę")
text = tok.decode(ids)

# Subword tokens
tokens = tok.encode_word("ustawodawstwo")
# ['ustaw', 'odaw', 'stwo'] (example)

# Save / load
tok.save("my-tokenizer")
tok = MorphBPETokenizer.load("my-tokenizer")

# Metrics
print(tok.fertility(test_texts))
print(tok.morph_edit_distance(test_texts))
mean, lo, hi = tok.morph_edit_distance_ci(test_texts)
```

### CLI

```bash
# Train
morphbpe train corpus.txt --vocab-size 32000 --alpha 2.0 --constraint suffix -o my-model

# Encode
echo "prezydent podpisał ustawę" | morphbpe encode my-model

# Info
morphbpe info my-model
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 1000 | Target vocabulary size |
| `alpha` | 1.0 | Penalty strength: 0=BPE, 1=block, >1=over-penalization |
| `constraint` | `"suffix"` | Boundary type: `suffix`, `prefix`, or `both` |

**Recommended for Polish:** `alpha=2.0, constraint="suffix"` (see paper, §Penalty Sweep).

## How It Works

During BPE training, each candidate merge is scored by its corpus frequency.
MorphBPE multiplies the frequency of boundary-crossing merges by `(1 - alpha)`:

- `alpha=0.0` → no penalty → standard BPE
- `alpha=0.5` → 50% frequency reduction for crossing merges
- `alpha=1.0` → full block (frequency = 0)
- `alpha=2.0` → negative score → crossing merges actively deprioritized

Boundaries are detected using [Morfeusz2](http://morfeusz.sgjp.pl/) (Polish morphological analyzer)
or a heuristic fallback for environments without Morfeusz2.

## Results

### Intrinsic Evaluation (from paper, v=1000)

| Metric | BPE | MorphBPE α=1.0 | Δ |
|--------|-----|----------------|---|
| Fertility (v=1000) | 1.935 | 1.935 | +0.1% |
| MorphED (v=1000) | 1.838 | 1.712 | **−6.8%** |
| CE LSTM (v=1000) | 3.744 | 3.646 | **−2.6%** |
| CE Transformer (v=1000) | 3.525 | 3.431 | **−2.7%** |

Over-penalization (`alpha=2.0`) reduces CE by an additional 1.6–3.1%.

### Extrinsic Evaluation (vocab 8k–32k, fine-grained alpha sweep)

Corpus: Sejm speeches (8000 train / 500 test), seq_len=32, 3 epochs. See `plots/` for visualizations.

| Vocab | α | Model | CE | PPL |
|-------|---|-------|----|-----|
| 8k | 0.0 (BPE) | LSTM | 4.1073 | 60.78 |
| 8k | 0.0 (BPE) | Transformer | 3.9140 | 50.10 |
| 8k | 0.5 | LSTM | 4.0433 | 57.01 |
| 8k | 0.5 | Transformer | 3.8552 | 47.24 |
| 8k | 1.0 | LSTM | 3.9243 | 50.62 |
| 8k | 1.0 | Transformer | 3.7743 | 43.57 |
| 8k | 1.5 | LSTM | 3.8849 | 48.66 |
| 8k | 1.5 | Transformer | 3.7112 | 40.90 |
| 8k | 2.0 | LSTM | 3.7758 | 43.63 |
| 8k | 2.0 | Transformer | 3.6319 | 37.79 |
| 16k | 0.0 (BPE) | LSTM | 4.1494 | 63.40 |
| 16k | 0.0 (BPE) | Transformer | 3.9473 | 51.80 |
| 16k | 0.5 | LSTM | 4.1132 | 61.14 |
| 16k | 0.5 | Transformer | 3.8966 | 49.24 |
| 16k | 1.0 | LSTM | 3.9870 | 53.89 |
| 16k | 1.0 | Transformer | 3.7789 | 43.77 |
| 16k | 1.5 | LSTM | 3.9287 | 50.84 |
| 16k | 1.5 | Transformer | 3.7188 | 41.21 |
| 16k | 2.0 | LSTM | 3.7984 | 44.63 |
| 16k | 2.0 | Transformer | **3.6272** | **37.61** |
| 32k | 0.0 (BPE) | LSTM | 4.2140 | 67.62 |
| 32k | 0.0 (BPE) | Transformer | 3.9605 | 52.49 |
| 32k | 0.5 | LSTM | 4.1380 | 62.68 |
| 32k | 0.5 | Transformer | 3.9135 | 50.07 |
| 32k | 1.0 | LSTM | 4.0106 | 55.18 |
| 32k | 1.0 | Transformer | 3.7803 | 43.83 |
| 32k | 1.5 | LSTM | 3.9687 | 52.92 |
| 32k | 1.5 | Transformer | 3.7210 | 41.31 |
| 32k | 2.0 | LSTM | 3.8427 | 46.65 |
| 32k | 2.0 | Transformer | 3.6347 | 37.89 |

**Key findings:**

- **Monotonic alpha improvement**: CE decreases monotonically with α for all three vocab sizes (8k, 16k, 32k) across both models
- **α=1.5 sweet spot**: 8k α=1.5 Transformer (CE=3.7112, PPL=40.90) beats 32k BPE Transformer (CE=3.9605, PPL=52.49) — 4x smaller vocab, 22% lower PPL. *Caveat: CE/PPL across different vocab sizes are not strictly comparable without bits-per-character normalization (see Methodological caveats below).*
- **CE reduction vs BPE**: α=0.5 → 1.2–1.8%; α=1.0 → 3.6–4.8%; α=1.5 → 5.2–6.0%; α=2.0 → 7.2–8.8%
- **Transformer > LSTM**: 3.8–6.2% CE reduction across all configs
- **Best config**: 16k α=2.0 Transformer (CE=3.6272, PPL=37.61)
- **Vocab size**: 8k and 16k perform similarly at α=2.0; 32k slightly worse due to sparse coverage on the small pilot corpus

### Cost-Benefit Analysis

Comparing MorphBPE 8k α=2.0 against a 4× larger BPE 32k tokenizer reveals a **trade-off**, not uniform dominance:

| Metric | BPE 32k | MorphBPE 8k α=2.0 | Δ | Winner |
|--------|---------|---------------------|---|--------|
| Vocab size | 32,000 | 8,000 | **4× smaller** | MorphBPE |
| Fertility (↓ better) | 1.020 | 1.149 | **+12.6%** | BPE 32k |
| MorphED (↓ better) | 0.389 | 0.456 | **+17.2%** | BPE 32k |
| CE Transformer (↓ better) | 3.9605 | 3.632 | **−8.3%** | MorphBPE |
| PPL Transformer (↓ better) | 52.49 | 37.79 | **−28.0%** | MorphBPE |

**Interpretation**: MorphBPE 8k α=2.0 wins decisively on language-model quality (CE/PPL) and vocabulary size, but at a cost — its fertility is 12.6% higher and its MorphED is 17.2% higher (worse) than BPE 32k. This is **not Pareto dominance**: the two configs trade off intrinsic segmentation quality (where the larger BPE 32k vocab wins on MorphED/fertility) against downstream LM performance and memory footprint (where MorphBPE 8k wins).

The higher MorphED and fertility for MorphBPE 8k are partly expected: it has 4× fewer tokens, so it must split words into more (and less morphologically-clean) pieces than a 32k vocab. The interesting result is that despite worse intrinsic metrics, the morphologically-informed 8k tokenizer still yields **lower LM cross-entropy** than 32k BPE.

> **Methodological caveats** (this is a small-corpus pilot — single runs, no seeds):
> 1. **CE/PPL are not directly comparable across vocab sizes.** Different tokenization granularity (fertility) affects per-token entropy independently of segmentation quality. A rigorous "8k beats 32k" claim requires bits-per-character, not raw CE/PPL. *(Planned: add bits-per-character.)*
> 2. **Small corpus (8k sentences) favors smaller vocabularies** regardless of morphology — 32k is under-trained with rare tokens on this little data, which may partly explain MorphBPE 8k's edge over BPE 32k.
> 3. **No repeats/seeds** — differences on the order of thousandths of CE (e.g. 8k vs 16k at α=2.0) have no estimated variance.
> 4. Conclusions about transfer to production-scale models are premature with an 8k-sentence corpus and seq_len=32.

See `plots/` for visualizations: `pareto_ce_vs_vocab.png`, `fertility_vs_morphed.png`, `efficiency_ratio.png`.

## License

MIT
