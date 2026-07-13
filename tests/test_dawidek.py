"""Tests for Dawidek (2026) metrics: SI, ISS, MFL."""

import pytest

from morphbpe import MorphBPETokenizer, evaluate_all, evaluate_lexeme
from morphbpe.dawidek_metrics import _find_stem, DAWIDEK_PARADIGMS


# ── Stem finding ────────────────────────────────────────────────────


def test_find_stem_regular():
    _, stem, suffix = _find_stem("robić", "robiłam")
    assert stem == "robi"
    assert suffix == "łam"


def test_find_stem_suppletive():
    _, stem, suffix = _find_stem("iść", "szedłem")
    # No common prefix ≥ 2 chars → suppletive
    assert stem == "szedłem"
    assert suffix == ""


def test_find_stem_noun():
    _, stem, suffix = _find_stem("ustawa", "ustawą")
    assert stem == "ustaw"
    assert suffix == "ą"


# ── Paradigm data ──────────────────────────────────────────────────


def test_paradigms_have_10_lexemes():
    assert len(DAWIDEK_PARADIGMS) == 10


def test_paradigms_include_suppletive():
    assert "iść" in DAWIDEK_PARADIGMS
    assert "być" in DAWIDEK_PARADIGMS
    assert "człowiek" in DAWIDEK_PARADIGMS


# ── Evaluation with mock tokenizer ─────────────────────────────────

SAMPLE_TEXTS = [
    "prezydent podpisał ustawę o ochronie środowiska",
    "sejm uchwalił budżet na przyszły rok",
] * 50


@pytest.fixture
def tok():
    t = MorphBPETokenizer(
        vocab_size=100, alpha=1.0, constraint="suffix", use_morfeusz=False
    )
    t.train(SAMPLE_TEXTS, verbose=False)
    return t


def test_evaluate_lexeme_returns_result(tok):
    r = evaluate_lexeme("robić", DAWIDEK_PARADIGMS["robić"], tok.encode_word)
    assert r.lemma == "robić"
    assert r.mfl > 0
    assert 0 <= r.si <= 1
    assert 0 <= r.iss <= 1
    assert len(r.per_form) > 0


def test_evaluate_all_returns_summary(tok):
    results, summary = evaluate_all(tok.encode_word)
    assert len(results) == 10
    assert "SI" in summary
    assert "ISS" in summary
    assert "MFL" in summary
    assert summary["MFL"] > 0


def test_si_bounded(tok):
    results, summary = evaluate_all(tok.encode_word)
    assert 0 <= summary["SI"] <= 1


def test_iss_bounded(tok):
    results, summary = evaluate_all(tok.encode_word)
    assert 0 <= summary["ISS"] <= 1
