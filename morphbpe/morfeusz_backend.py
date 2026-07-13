"""Morpheme boundary detection: Morfeusz2 with heuristic fallback."""

from typing import Dict, Set, Tuple

try:
    import morfeusz2
    MORFEUSZ_AVAILABLE = True
except ImportError:
    MORFEUSZ_AVAILABLE = False

# Heuristic suffix/prefix lists for when Morfeusz2 is unavailable
_HEURISTIC_PREFIXES = [
    "nie", "prze", "roz", "pod", "nad", "bez", "naj", "przy",
]
_HEURISTIC_SUFFIXES = [
    "owanie", "ować", "owy", "owa", "owe",
    "ych", "ego", "emu", "ami", "ach",
    "ów", "em", "ę", "ą", "ć",
]


class MorfeuszBackend:
    """Detects morpheme boundaries using Morfeusz2 or heuristic fallback."""

    def __init__(self, use_morfeusz: bool = True):
        self._morfeusz = None
        if use_morfeusz and MORFEUSZ_AVAILABLE:
            self._morfeusz = morfeusz2.Morfeusz()

    @property
    def has_morfeusz(self) -> bool:
        return self._morfeusz is not None

    def get_boundaries(self, word: str) -> Tuple[Set[int], Set[int]]:
        """Return (prefix_boundaries, suffix_boundaries) as character positions."""
        if self._morfeusz is not None:
            return self._morfeusz_boundaries(word)
        return self._heuristic_boundaries(word)

    def get_boundaries_batch(
        self, words: list[str]
    ) -> Dict[str, Tuple[Set[int], Set[int]]]:
        """Compute boundaries for a list of unique words."""
        return {w: self.get_boundaries(w) for w in words}

    # ── Morfeusz2 path ──────────────────────────────────────────────

    def _morfeusz_boundaries(self, word: str) -> Tuple[Set[int], Set[int]]:
        prefix_b: Set[int] = set()
        suffix_b: Set[int] = set()
        analysis = self._morfeusz.analyse(word)
        if not analysis:
            return prefix_b, suffix_b

        interp = analysis[0]
        lemma = interp[2][1] if len(interp[2]) > 1 else interp[2][0]
        if not lemma or len(lemma) >= len(word):
            return prefix_b, suffix_b

        if word.startswith(lemma):
            suffix_b.add(len(lemma))
        else:
            common = 0
            for i in range(min(len(lemma), len(word))):
                if lemma[i] == word[i]:
                    common = i + 1
                else:
                    break
            if 2 < common < len(word):
                prefix_b.add(common)

        return prefix_b, suffix_b

    # ── Heuristic fallback ──────────────────────────────────────────

    @staticmethod
    def _heuristic_boundaries(word: str) -> Tuple[Set[int], Set[int]]:
        prefix_b: Set[int] = set()
        suffix_b: Set[int] = set()
        for pfx in _HEURISTIC_PREFIXES:
            if word.startswith(pfx) and len(word) > len(pfx) + 2:
                prefix_b.add(len(pfx))
        for sfx in _HEURISTIC_SUFFIXES:
            if word.endswith(sfx) and len(word) > len(sfx) + 2:
                suffix_b.add(len(word) - len(sfx))
        return prefix_b, suffix_b
