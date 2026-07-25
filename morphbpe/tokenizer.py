"""MorphBPE tokenizer: BPE with morphological boundary penalties."""

import collections
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from morphbpe.morfeusz_backend import MorfeuszBackend


class MorphBPETokenizer:
    """Morphologically-aware BPE tokenizer.

    Parameters
    ----------
    vocab_size : int
        Target vocabulary size (default 1000).
    alpha : float
        Penalty strength for boundary-crossing merges.
        0.0 = standard BPE, 1.0 = full block, >1.0 = over-penalization.
    constraint : str
        Which boundaries to enforce: "suffix", "prefix", or "both".
    use_morfeusz : bool
        Use Morfeusz2 if available; otherwise heuristic fallback.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        alpha: float = 1.0,
        constraint: str = "suffix",
        use_morfeusz: bool = True,
    ):
        if constraint not in ("suffix", "prefix", "both"):
            raise ValueError(f"constraint must be suffix/prefix/both, got {constraint}")

        self.vocab_size = vocab_size
        self.alpha = alpha
        self.constraint = constraint

        self._backend = MorfeuszBackend(use_morfeusz=use_morfeusz)
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []

    # ── Training ────────────────────────────────────────────────────

    def train(self, texts: List[str], verbose: bool = True) -> float:
        """Train tokenizer on a list of texts. Returns training time in seconds."""
        start = time.time()

        word_freq: Dict[str, int] = collections.defaultdict(int)
        boundaries: Dict[str, Tuple[Set[int], Set[int]]] = {}
        it = tqdm(texts, desc="Preprocessing") if verbose else texts
        for text in it:
            for word in text.split():
                word_freq[" ".join(list(word))] += 1
                if word not in boundaries:
                    boundaries[word] = self._backend.get_boundaries(word)

        # Initial character vocab
        chars: Set[str] = set()
        for w in word_freq:
            chars.update(w.split())
        for i, c in enumerate(sorted(chars)):
            self.token2id[c] = i
        self.id2token = {v: k for k, v in self.token2id.items()}

        n_merges = self.vocab_size - len(self.token2id)
        it = tqdm(range(n_merges), desc="Merging") if verbose else range(n_merges)
        for _ in it:
            pairs = self._pair_scores(word_freq, boundaries)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            word_freq = self._apply_merge(word_freq, best)
            token = "".join(best)
            if token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

        elapsed = time.time() - start
        if verbose:
            print(
                f"Trained: vocab={len(self.token2id)}, "
                f"merges={len(self.merges)}, time={elapsed:.1f}s"
            )
        return elapsed

    def train_from_file(self, path: str, verbose: bool = True) -> float:
        """Train from a plain-text file (one document per line)."""
        texts = Path(path).read_text(encoding="utf-8").splitlines()
        texts = [t for t in texts if t.strip()]
        return self.train(texts, verbose=verbose)

    # ── Encode / Decode ─────────────────────────────────────────────

    def encode_word(self, word: str) -> List[str]:
        """Tokenize a single word into subword strings."""
        if not hasattr(self, "_encode_cache"):
            self._encode_cache: Dict[str, List[str]] = {}
        if word in self._encode_cache:
            return self._encode_cache[word]
        wt = " ".join(list(word))
        for a, b in self.merges:
            wt = wt.replace(f"{a} {b}", f"{a}{b}")
        result = wt.split()
        self._encode_cache[word] = result
        return result

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs."""
        ids: List[int] = []
        for word in text.split():
            for tok in self.encode_word(word):
                ids.append(self.token2id.get(tok, 0))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text (no whitespace recovery)."""
        return "".join(self.id2token.get(i, "?") for i in ids)

    @property
    def vocab(self) -> Dict[str, int]:
        """HuggingFace-compatible vocab property."""
        return dict(self.token2id)

    def get_vocab_size(self) -> int:
        return len(self.token2id)

    # ── Save / Load ─────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save tokenizer to a directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "vocab_size": self.vocab_size,
            "alpha": self.alpha,
            "constraint": self.constraint,
            "has_morfeusz": self._backend.has_morfeusz,
        }
        (path / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (path / "vocab.json").write_text(
            json.dumps(self.token2id, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        merges_lines = [f"{a} {b}" for a, b in self.merges]
        (path / "merges.txt").write_text(
            "\n".join(merges_lines), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str) -> "MorphBPETokenizer":
        """Load tokenizer from a directory."""
        path = Path(directory)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        tok = cls(
            vocab_size=config["vocab_size"],
            alpha=config["alpha"],
            constraint=config["constraint"],
            use_morfeusz=config.get("has_morfeusz", True),
        )
        tok.token2id = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        tok.id2token = {int(v): k for k, v in tok.token2id.items()}
        raw = (path / "merges.txt").read_text(encoding="utf-8").strip()
        if raw:
            tok.merges = [tuple(line.split(" ", 1)) for line in raw.splitlines()]
        return tok

    # ── Metrics ─────────────────────────────────────────────────────

    def fertility(self, texts: List[str]) -> float:
        """Average tokens per word."""
        n_tok = n_word = 0
        for text in texts:
            for word in text.split():
                n_tok += len(self.encode_word(word))
                n_word += 1
        return n_tok / n_word if n_word else 0.0

    def morph_edit_distance(self, texts: List[str]) -> float:
        """Mean MorphED across words (≥3 chars)."""
        vals = self._per_word_morphed(texts)
        return float(np.mean(vals)) if vals else 0.0

    def morph_edit_distance_ci(
        self, texts: List[str], n_resamples: int = 1000, seed: int = 42
    ) -> Tuple[float, float, float]:
        """MorphED with bootstrap 95% CI → (mean, lower, upper)."""
        vals = np.array(self._per_word_morphed(texts))
        if len(vals) == 0:
            return 0.0, 0.0, 0.0
        rng = np.random.RandomState(seed)
        means = [
            np.mean(rng.choice(vals, size=len(vals), replace=True))
            for _ in range(n_resamples)
        ]
        return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def _per_word_morphed(self, texts: List[str]) -> List[int]:
        result: List[int] = []
        for text in texts:
            for word in text.split():
                if len(word) < 3:
                    continue
                prefix_b, suffix_b = self._backend.get_boundaries(word)
                all_b = sorted(prefix_b | suffix_b)
                morphemes: List[str] = []
                prev = 0
                for b in all_b:
                    morphemes.append(word[prev:b])
                    prev = b
                morphemes.append(word[prev:])
                tokens = self.encode_word(word)
                result.append(_list_edit_distance(morphemes, tokens))
        return result

    # ── Internal BPE logic ──────────────────────────────────────────

    def _active_boundaries(
        self, prefix_b: Set[int], suffix_b: Set[int]
    ) -> Set[int]:
        if self.constraint == "prefix":
            return prefix_b
        if self.constraint == "suffix":
            return suffix_b
        return prefix_b | suffix_b

    def _pair_scores(
        self,
        word_freq: Dict[str, int],
        boundaries: Dict[str, Tuple[Set[int], Set[int]]],
    ) -> Dict[Tuple[str, str], float]:
        pairs: Dict[Tuple[str, str], float] = collections.defaultdict(float)
        for word, freq in word_freq.items():
            tokens = word.split()
            orig = word.replace(" ", "")
            pfx_b, sfx_b = boundaries.get(orig, (set(), set()))
            active = self._active_boundaries(pfx_b, sfx_b)

            pos = 0
            for i in range(len(tokens) - 1):
                tlen = len(tokens[i])
                crosses = (pos + tlen) in active
                if crosses:
                    pairs[(tokens[i], tokens[i + 1])] += freq * (1.0 - self.alpha)
                else:
                    pairs[(tokens[i], tokens[i + 1])] += freq
                pos += tlen
        return pairs

    @staticmethod
    def _apply_merge(
        word_freq: Dict[str, int], pair: Tuple[str, str]
    ) -> Dict[str, int]:
        bigram = " ".join(pair)
        repl = "".join(pair)
        return {w.replace(bigram, repl): f for w, f in word_freq.items()}


def _list_edit_distance(a: List[str], b: List[str]) -> int:
    """Edit distance between two lists of strings."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
