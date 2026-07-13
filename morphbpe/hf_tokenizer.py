"""HuggingFace tokenizer wrapper for MorphBPE.

Provides a PreTrainedTokenizer-compatible interface so MorphBPE
can be used with HuggingFace transformers pipelines.

Usage:
    from morphbpe.hf_tokenizer import MorphBPEHFTokenizer

    tok = MorphBPEHFTokenizer.from_pretrained("path/to/model")
    ids = tok("prezydent podpisał ustawę")
"""

import json
from pathlib import Path
from typing import List, Optional, Union

from morphbpe.tokenizer import MorphBPETokenizer


class MorphBPEHFTokenizer:
    """Lightweight HF-compatible wrapper around MorphBPETokenizer.

    Implements the core interface needed by transformers:
    - __call__(text) -> dict with input_ids
    - encode(text) -> List[int]
    - decode(ids) -> str
    - save_pretrained(dir) / from_pretrained(dir)
    - get_vocab() / get_vocab_size()
    """

    def __init__(self, tokenizer: MorphBPETokenizer):
        self._tok = tokenizer

    @classmethod
    def from_pretrained(cls, path: str) -> "MorphBPEHFTokenizer":
        tok = MorphBPETokenizer.load(path)
        return cls(tok)

    def save_pretrained(self, directory: str) -> None:
        self._tok.save(directory)
        config_path = Path(directory) / "tokenizer_config.json"
        config = {
            "tokenizer_class": "MorphBPEHFTokenizer",
            "model_type": "morphbpe",
            "vocab_size": self._tok.vocab_size,
            "alpha": self._tok.alpha,
            "constraint": self._tok.constraint,
        }
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> dict:
        if isinstance(text, str):
            ids = self._tok.encode(text)
        else:
            ids = [self._tok.encode(t) for t in text]

        result = {"input_ids": ids}
        if return_tensors == "pt":
            try:
                import torch

                if isinstance(ids[0], list):
                    max_len = max(len(x) for x in ids)
                    padded = [x + [0] * (max_len - len(x)) for x in ids]
                    result["input_ids"] = torch.tensor(padded)
                    result["attention_mask"] = torch.tensor(
                        [[1] * len(x) + [0] * (max_len - len(x)) for x in ids]
                    )
                else:
                    result["input_ids"] = torch.tensor([ids])
            except ImportError:
                pass
        return result

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids)

    def get_vocab(self) -> dict:
        return self._tok.vocab

    def get_vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def model_max_length(self) -> int:
        return 2048

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._tok.id2token.get(i, "<unk>") for i in ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._tok.token2id.get(t, 0) for t in tokens]
