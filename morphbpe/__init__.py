"""MorphBPE: Morphologically-aware BPE tokenizer for Polish."""

__version__ = "0.1.0"

from morphbpe.tokenizer import MorphBPETokenizer
from morphbpe.dawidek_metrics import evaluate_all, evaluate_lexeme

__all__ = ["MorphBPETokenizer", "evaluate_all", "evaluate_lexeme"]
