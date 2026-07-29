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

### Extrinsic Evaluation (vocab 8k–32k)

Corpus: Sejm speeches (4000 train / 400 test), seq_len=24, 3 epochs. See `plots/` for visualizations.

| Vocab | α | Model | CE | PPL |
|-------|---|-------|----|-----|
| 8k | 0.0 (BPE) | LSTM | 4.1705 | 64.75 |
| 8k | 0.0 (BPE) | Transformer | 3.8545 | 47.21 |
| 8k | 1.0 | LSTM | 4.0076 | 55.01 |
| 8k | 1.0 | Transformer | 3.6955 | 40.26 |
| 8k | 2.0 | LSTM | 3.8659 | 47.75 |
| 8k | 2.0 | Transformer | **3.5564** | **35.04** |
| 16k | 0.0 (BPE) | LSTM | 4.2060 | 67.09 |
| 16k | 0.0 (BPE) | Transformer | 3.8661 | 47.76 |
| 16k | 1.0 | LSTM | 4.0554 | 57.71 |
| 16k | 1.0 | Transformer | 3.7005 | 40.47 |
| 16k | 2.0 | LSTM | 3.8646 | 47.68 |
| 16k | 2.0 | Transformer | 3.5583 | 35.10 |
| 32k | 0.0 (BPE) | LSTM | 4.2776 | 72.07 |
| 32k | 0.0 (BPE) | Transformer | 3.8966 | 49.23 |
| 32k | 1.0 | LSTM | 4.1401 | 62.81 |
| 32k | 1.0 | Transformer | 3.7305 | 41.70 |
| 32k | 2.0 | LSTM | 3.9881 | 53.95 |
| 32k | 2.0 | Transformer | 3.6014 | 36.65 |

**Key findings:**

- **Alpha ranking**: α=2.0 > α=1.0 > α=0.0 (BPE) consistently across all vocab sizes and models
- **CE reduction**: α=1.0 reduces CE by 3.2–4.3% vs BPE; α=2.0 by 6.8–8.1%
- **Transformer > LSTM**: 7.6–9.9% CE reduction across all configs
- **Best config**: 8k α=2.0 Transformer (CE=3.56, PPL=35.04)
- **Vocab size**: 8k optimal at this corpus size; larger vocabs underperform due to sparse coverage

## License

MIT
