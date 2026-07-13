"""Dawidek (2026) metrics: SI, ISS, MFL for tokenizer evaluation.

Metrics:
- SI  (Stem Integrity): fraction of forms where the stem is preserved as a single token
- ISS (Inflectional Suffix Separation): fraction of forms where the inflectional
        suffix is a separate token (not merged with stem)
- MFL (Mean Fragmentation per Lexeme): average number of tokens per form,
        averaged across all forms of a lexeme

Reference: Dawidek (2026a-d), SocArXiv preprints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from morphbpe.morfeusz_backend import MorfeuszBackend


@dataclass
class LexemeResult:
    """Per-lexeme evaluation result."""
    lemma: str
    forms: List[str]
    mfl: float = 0.0
    si: float = 0.0
    iss: float = 0.0
    per_form: List[dict] = field(default_factory=list)


# ── Polish test paradigms (from Dawidek 2026b) ─────────────────────

# 10 lexemes: 5 verbs (incl. suppletive iść, być) + 5 nouns
DAWIDEK_PARADIGMS: Dict[str, List[str]] = {
    # Verbs
    "robić": [
        "robię", "robisz", "robi", "robimy", "robicie", "robią",
        "robiłem", "robiłeś", "robił", "robiłam", "robiłaś", "robiła",
        "robiłem", "robiliśmy", "robiliście", "robili",
        "rób", "róbmy", "róbcie",
        "robić",
    ],
    "pisać": [
        "piszę", "piszesz", "pisze", "piszemy", "piszecie", "piszą",
        "pisałem", "pisałeś", "pisał", "pisałam", "pisałaś", "pisała",
        "pisaliśmy", "pisaliście", "pisali",
        "pisz", "piszmy", "piszcie",
        "pisać",
    ],
    "czytać": [
        "czytam", "czytasz", "czyta", "czytamy", "czytacie", "czytają",
        "czytałem", "czytałeś", "czytał", "czytałam", "czytałaś", "czytała",
        "czytaliśmy", "czytaliście", "czytali",
        "czytaj", "czytajmy", "czytajcie",
        "czytać",
    ],
    # Suppletive
    "iść": [
        "idę", "idziesz", "idzie", "idziemy", "idziecie", "idą",
        "szedłem", "szedłeś", "szedł", "szłam", "szłaś", "szła",
        "szliśmy", "szliście", "szli",
        "idź", "idźmy", "idźcie",
        "iść",
    ],
    "być": [
        "jestem", "jesteś", "jest", "jesteśmy", "jesteście", "są",
        "byłem", "byłeś", "był", "byłam", "byłaś", "była",
        "byliśmy", "byliście", "byli",
        "bądź", "bądźmy", "bądźcie",
        "być",
    ],
    # Nouns
    "ustawa": [
        "ustawa", "ustawy", "ustawie", "ustawę", "ustawą", "ustawo",
        "ustawy", "ustaw", "ustawom", "ustawy", "ustawami", "ustawach",
    ],
    "człowiek": [
        "człowiek", "człowieka", "człowiekowi", "człowieka", "człowiekiem", "człowieku",
        "ludzie", "ludzi", "ludziom", "ludzi", "ludźmi", "ludziach",
    ],
    "dziecko": [
        "dziecko", "dziecka", "dziecku", "dziecko", "dzieckiem", "dziecku",
        "dzieci", "dzieci", "dzieciom", "dzieci", "dziećmi", "dzieciach",
    ],
    "prawo": [
        "prawo", "prawa", "prawu", "prawo", "prawem", "prawie",
        "prawa", "praw", "prawom", "prawa", "prawami", "prawach",
    ],
    "kraj": [
        "kraj", "kraju", "krajowi", "kraj", "krajem", "kraju",
        "kraje", "krajów", "krajom", "kraje", "krajami", "krajach",
    ],
}


def _find_stem(lemma: str, form: str) -> Tuple[str, str, str]:
    """Heuristic stem extraction: longest common prefix of lemma and form.

    Returns (prefix, stem, suffix) of the form.
    prefix = characters before stem (empty for most Polish forms)
    stem   = shared initial substring
    suffix = remaining characters after stem
    """
    common = 0
    for i in range(min(len(lemma), len(form))):
        if lemma[i] == form[i]:
            common = i + 1
        else:
            break
    if common < 2:
        # Suppletive — no shared stem
        return "", form, ""
    stem = form[:common]
    suffix = form[common:]
    return "", stem, suffix


def evaluate_lexeme(lemma: str, forms: List[str], encode_word_fn) -> LexemeResult:
    """Evaluate a single lexeme against Dawidek metrics.

    Parameters
    ----------
    lemma : str
        Dictionary form of the lexeme.
    forms : list of str
        All inflected forms to evaluate.
    encode_word_fn : callable
        Function that takes a word and returns list of subword tokens.
    """
    unique_forms = list(dict.fromkeys(forms))  # deduplicate, preserve order
    total_tokens = 0
    stem_intact_count = 0
    suffix_sep_count = 0
    per_form = []

    for form in unique_forms:
        tokens = encode_word_fn(form)
        n_tokens = len(tokens)
        total_tokens += n_tokens

        _, stem, suffix = _find_stem(lemma, form)

        # SI: is the stem a single token (or contained in one token)?
        stem_intact = any(stem in t for t in tokens) if stem else False

        # ISS: is the suffix a separate token?
        if suffix:
            suffix_sep = suffix in tokens
        else:
            suffix_sep = True  # no suffix to separate

        if stem_intact:
            stem_intact_count += 1
        if suffix_sep:
            suffix_sep_count += 1

        per_form.append({
            "form": form,
            "tokens": tokens,
            "n_tokens": n_tokens,
            "stem": stem,
            "suffix": suffix,
            "stem_intact": stem_intact,
            "suffix_sep": suffix_sep,
        })

    n = len(unique_forms)
    return LexemeResult(
        lemma=lemma,
        forms=unique_forms,
        mfl=total_tokens / n if n else 0.0,
        si=stem_intact_count / n if n else 0.0,
        iss=suffix_sep_count / n if n else 0.0,
        per_form=per_form,
    )


def evaluate_all(
    encode_word_fn,
    paradigms: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[LexemeResult], Dict[str, float]]:
    """Evaluate tokenizer on all Dawidek paradigms.

    Returns
    -------
    results : list of LexemeResult
    summary : dict with macro-averaged SI, ISS, MFL
    """
    if paradigms is None:
        paradigms = DAWIDEK_PARADIGMS

    results = []
    for lemma, forms in paradigms.items():
        r = evaluate_lexeme(lemma, forms, encode_word_fn)
        results.append(r)

    n = len(results)
    summary = {
        "SI": sum(r.si for r in results) / n if n else 0.0,
        "ISS": sum(r.iss for r in results) / n if n else 0.0,
        "MFL": sum(r.mfl for r in results) / n if n else 0.0,
        "n_lexemes": n,
        "n_forms": sum(len(r.forms) for r in results),
    }
    return results, summary


def print_report(results: List[LexemeResult], summary: Dict[str, float]) -> str:
    """Format a readable report string."""
    lines = []
    lines.append("Dawidek Metrics Report")
    lines.append("=" * 70)
    lines.append(
        f"{'Lexeme':<15} {'Forms':>5} {'MFL':>6} {'SI':>6} {'ISS':>6}"
    )
    lines.append("-" * 70)
    for r in results:
        lines.append(
            f"{r.lemma:<15} {len(r.forms):>5} "
            f"{r.mfl:>6.2f} {r.si:>6.1%} {r.iss:>6.1%}"
        )
    lines.append("-" * 70)
    lines.append(
        f"{'MACRO AVG':<15} {summary['n_forms']:>5} "
        f"{summary['MFL']:>6.2f} {summary['SI']:>6.1%} {summary['ISS']:>6.1%}"
    )
    lines.append("=" * 70)
    return "\n".join(lines)
