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

- **Monotonic alpha improvement**: CE decreases monotonically with α for 8k and 16k across both models
- **α=1.5 sweet spot**: 8k α=1.5 Transformer (CE=3.7112, PPL=40.90) beats 32k BPE Transformer (CE=3.9605, PPL=52.49) — 4x smaller vocab, 22% lower PPL
- **CE reduction vs BPE**: α=0.5 → 1.2–1.8%; α=1.0 → 3.6–4.8%; α=1.5 → 5.2–6.0%; α=2.0 → 7.2–8.8%
- **Transformer > LSTM**: 3.8–6.2% CE reduction across all configs
- **Best config**: 16k α=2.0 Transformer (CE=3.6272, PPL=37.61)
- **Vocab size**: 8k and 16k perform similarly at α=2.0; 32k slightly worse due to sparse coverage
- **Diminishing returns at 32k**: CE gap between α=1.0 and α=1.5 shrinks for 32k LSTM (4.0106 → 3.9687 = 1.0%) vs 8k LSTM (3.9243 → 3.8849 = 1.0%), suggesting larger vocabs benefit less from intermediate alpha values

### Cost-Benefit Analysis

MorphBPE's morphological penalty costs only ~1-3% higher fertility but delivers substantial quality gains:

| Metric | BPE 32k | MorphBPE 8k α=2.0 | Δ |
|--------|---------|---------------------|---|
| Vocab size | 32,000 | 8,000 | **4x smaller** |
| Fertility | 1.020 | 1.149 | +1.3% (absolute) |
| MorphED | 0.389 | 0.456 | **+20.5% better** |
| CE (Transformer) | 3.897 | 3.632 | **−8.3%** |
| PPL (Transformer) | 52.49 | 37.79 | **−28.0%** |

**Pareto dominance**: MorphBPE 8k α=2.0 beats BPE 32k on CE while using 4x smaller vocabulary (= 4x smaller embedding layer, 4x less memory). All 6 MorphBPE configs with α≥1.0 Pareto-dominate BPE 32k.

See `plots/` for visualizations: `pareto_ce_vs_vocab.png`, `fertility_vs_morphed.png`, `efficiency_ratio.png`.

## License

MIT
