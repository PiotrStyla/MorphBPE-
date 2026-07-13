"""Tests for MorphBPE tokenizer."""

import tempfile
from pathlib import Path

import pytest

from morphbpe.tokenizer import MorphBPETokenizer, _list_edit_distance


# ── Fixtures ────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "prezydent podpisał ustawę o ochronie środowiska",
    "sejm uchwalił budżet na przyszły rok",
    "komisja rozpatrzyła poprawki do projektu ustawy",
    "posłowie głosowali nad przyjęciem rezolucji",
    "marszałek otworzył posiedzenie sejmu",
] * 20  # repeat for enough data


@pytest.fixture
def small_tok():
    """A small trained tokenizer for testing."""
    tok = MorphBPETokenizer(
        vocab_size=100, alpha=1.0, constraint="suffix", use_morfeusz=False
    )
    tok.train(SAMPLE_TEXTS, verbose=False)
    return tok


# ── Basic tests ─────────────────────────────────────────────────────


def test_train_produces_vocab(small_tok):
    assert small_tok.get_vocab_size() > 0
    assert len(small_tok.merges) > 0


def test_encode_decode_roundtrip(small_tok):
    word = "ustawę"
    tokens = small_tok.encode_word(word)
    rejoined = "".join(tokens)
    assert rejoined == word, f"Roundtrip failed: {tokens} -> {rejoined}"


def test_encode_returns_ids(small_tok):
    ids = small_tok.encode("sejm uchwalił")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_decode_reverses_encode(small_tok):
    text = "budżet"
    ids = small_tok.encode(text)
    decoded = small_tok.decode(ids)
    assert decoded == text


def test_fertility_positive(small_tok):
    f = small_tok.fertility(SAMPLE_TEXTS[:5])
    assert f > 1.0  # at least 1 token per word


def test_morph_edit_distance_nonnegative(small_tok):
    med = small_tok.morph_edit_distance(SAMPLE_TEXTS[:5])
    assert med >= 0.0


# ── Save / Load ─────────────────────────────────────────────────────


def test_save_load_roundtrip(small_tok):
    with tempfile.TemporaryDirectory() as tmpdir:
        small_tok.save(tmpdir)
        loaded = MorphBPETokenizer.load(tmpdir)

        assert loaded.alpha == small_tok.alpha
        assert loaded.constraint == small_tok.constraint
        assert loaded.vocab_size == small_tok.vocab_size
        assert loaded.merges == small_tok.merges
        assert loaded.token2id == small_tok.token2id

        # Encoding consistency
        word = "ustawę"
        assert small_tok.encode_word(word) == loaded.encode_word(word)


# ── Alpha=0 is standard BPE ────────────────────────────────────────


def test_alpha_zero_is_standard_bpe():
    tok_bpe = MorphBPETokenizer(
        vocab_size=80, alpha=0.0, constraint="suffix", use_morfeusz=False
    )
    tok_bpe.train(SAMPLE_TEXTS, verbose=False)

    # Just check it trains without error and produces tokens
    ids = tok_bpe.encode("prezydent")
    assert len(ids) > 0


# ── Constraint validation ──────────────────────────────────────────


def test_invalid_constraint_raises():
    with pytest.raises(ValueError):
        MorphBPETokenizer(constraint="invalid")


# ── Edit distance ──────────────────────────────────────────────────


def test_list_edit_distance_identical():
    assert _list_edit_distance(["a", "b"], ["a", "b"]) == 0


def test_list_edit_distance_insert():
    assert _list_edit_distance(["a"], ["a", "b"]) == 1


def test_list_edit_distance_different():
    assert _list_edit_distance(["a", "b"], ["c", "d"]) == 2


# ── Over-penalization ──────────────────────────────────────────────


def test_over_penalization_trains():
    tok = MorphBPETokenizer(
        vocab_size=80, alpha=2.0, constraint="suffix", use_morfeusz=False
    )
    tok.train(SAMPLE_TEXTS, verbose=False)
    assert tok.get_vocab_size() > 0
    ids = tok.encode("prezydent")
    assert len(ids) > 0
