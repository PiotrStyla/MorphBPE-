"""Tests for HuggingFace tokenizer wrapper."""

import tempfile

import pytest

from morphbpe.tokenizer import MorphBPETokenizer
from morphbpe.hf_tokenizer import MorphBPEHFTokenizer

SAMPLE_TEXTS = [
    "prezydent podpisał ustawę o ochronie środowiska",
    "sejm uchwalił budżet na przyszły rok",
] * 20


@pytest.fixture
def hf_tok():
    tok = MorphBPETokenizer(
        vocab_size=100, alpha=1.0, constraint="suffix", use_morfeusz=False
    )
    tok.train(SAMPLE_TEXTS, verbose=False)
    return MorphBPEHFTokenizer(tok)


def test_call_returns_dict(hf_tok):
    out = hf_tok("sejm uchwalił")
    assert "input_ids" in out
    assert isinstance(out["input_ids"], list)
    assert len(out["input_ids"]) > 0


def test_call_batch(hf_tok):
    out = hf_tok(["sejm", "budżet"])
    assert isinstance(out["input_ids"], list)
    assert len(out["input_ids"]) == 2


def test_encode_decode(hf_tok):
    ids = hf_tok.encode("budżet")
    assert hf_tok.decode(ids) == "budżet"


def test_get_vocab(hf_tok):
    v = hf_tok.get_vocab()
    assert isinstance(v, dict)
    assert len(v) > 0


def test_convert_ids_to_tokens(hf_tok):
    ids = hf_tok.encode("sejm")
    tokens = hf_tok.convert_ids_to_tokens(ids)
    assert len(tokens) == len(ids)


def test_save_load_pretrained(hf_tok):
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_tok.save_pretrained(tmpdir)
        loaded = MorphBPEHFTokenizer.from_pretrained(tmpdir)
        assert loaded.get_vocab_size() == hf_tok.get_vocab_size()
        assert loaded.encode("sejm") == hf_tok.encode("sejm")


def test_tokenizer_config_written(hf_tok):
    import json
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_tok.save_pretrained(tmpdir)
        config = json.loads((Path(tmpdir) / "tokenizer_config.json").read_text())
        assert config["tokenizer_class"] == "MorphBPEHFTokenizer"
        assert config["model_type"] == "morphbpe"
