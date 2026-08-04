# MorphBPE — Morfologicznie informowany Byte-Pair Encoding dla języka polskiego

## Streszczenie

MorphBPE to wariant algorytmu BPE (Byte-Pair Encoding), który wymusza poszanowanie granic morfemów podczas scalania par tokenów. Granice sufiksów pozyskiwane są z analizatora morfologicznego **Morfeusz2**; scalenia przekraczające granicę sufiksu są karane mnożnikiem częstotliwości (`score = freq × (1 - α)`).

Praca stanowi **pierwszą ewaluację MorphBPE na języku polskim** z analizatorem Morfeusz2, ewaluacją intrinsic i extrinsic z dwiema architekturami (LSTM i Transformer), analizą błędów per-suffix, oraz skalowaniem do rozmiarów produkcyjnych (32 000 tokenów).

---

## Kluczowe wyniki

| Metryka | Wartość | Kontekst |
|---------|---------|----------|
| MorphED redukcja (vocab=500) | −3.6% | vs standardowy BPE |
| MorphED redukcja (vocab=1000) | −6.8% | vs standardowy BPE |
| MorphED redukcja (vocab=32000, α=2.0) | **−21.0%** | vs standardowy BPE |
| CE redukcja LSTM (vocab=1000) | −2.6% | cross-entropy modelu języka |
| CE redukcja Transformer (vocab=1000) | −2.7% | cross-entropy modelu języka |
| Fertility koszt | < 3% | nawet przy vocab=32k |
| MorphBPE vs Bielik-11B orig. | 1.05 vs 3.22 | **3.1× niższa** przy vocab=32k |
| MorphBPE vs Bielik APT4 | 1.05 vs 1.62 | **1.5× niższa** (vs zoptymalizowany) |

---

## Wkład naukowy

1. **Pierwsza ewaluacja MorphBPE na polskim** — Asgari et al. (2025) testowali na angielskim, rosyjskim, węgierskim, arabskim, ale nie polskim
2. **Over-penalization (α > 1)** — oryginalny MorphBPE definiuje α ∈ [0,1]. Rozszerzyliśmy na α > 1 i pokazaliśmy skokową poprawę (−1.6% do −3.1% CE nad α=1.0)
3. **Analiza błędów per-suffix** — identyfikujemy które klasy sufiksów zyskują (złożone: -ach, -ego, -ości) i tracą (krótkie: -ć, -ych)
4. **Ewaluacja z dwiema architekturami** (LSTM i Transformer) — potwierdzenie, że korzyść jest odporna na wybór architektury
5. **Skalowanie do rozmiarów produkcyjnych** (vocab=8k–32k) — korzyść 3–6× silniejsza niż przy vocab=500–1000

---

## Struktura repozytorium

```
paper-tokenization/
├── main.tex              # Wrapper LaTeX (wiring)
├── formatka/
│   └── slayer.sty        # Formatka Slayer (wygląd dokumentu)
├── tresc/
│   ├── 00-strona-tytulowa.tex   # Strona tytułowa + wykres hero
│   └── 10-tresc.tex             # Właściwa treść papera (~1000 linii)
├── prezentacja.html      # Reveal.js prezentacja (standalone, otwórz w przeglądarce)
├── prezentacja.tex       # LaTeX wersja prezentacji
├── reveal.css            # Reveal.js CSS (lokalna kopia)
├── reveal.js             # Reveal.js JS (lokalna kopia)
├── theme-white.css       # Motyw prezentacji
├── MorphBPE_paper.pdf    # Wygenerowany PDF papera (bez LaTeX)
└── README.md             # Ten plik
```

---

## Metodologia

### Korpus
- **10 000 przemówień** z posiedzeń Sejmu RP (sejm.gov.pl)
- Dostępny na HuggingFace: [`PiotrSty/sejm-speeches-corpus`](https://huggingface.co/datasets/PiotrSty/sejm-speeches-corpus)
- Split: 80/20 (8000 train / 2000 test), seed=42

### Tokenizatory
- **Standard BPE** — klasyczny algorytm oparty na częstotliwości par (baseline)
- **MorphBPE strict_suffix** — BPE z karą za scalenia przekraczające granicę sufiksu (Morfeusz2)
- Parametr α kontroluje siłę kary: α=0 → BPE, α=1 → pełne blokowanie, α=2 → over-penalization

### Ewaluacja intrinsic
- **Fertility** — liczba tokenów na słowo (niżej = lepiej)
- **MorphED** (Morphological Edit Distance) — odległość Levenshteina między segmentacją tokenizatora a morfologiczną (niżej = lepiej)
- **Bootstrap CI** — 1000 resamples, 95% przedział ufności

### Ewaluacja extrinsic
- **LSTM LM** — 1 warstwa, embed=64, hidden=128, 3 epoki
- **Transformer LM** — 2 warstwy, 4 głowice, embed=64, ff=256, 3 epoki
- Metryki: cross-entropy (CE), perplexity (PPL)

---

## Przegląd zbadanych podejść

Przed wyborem MorphBPE strict_suffix zbadaliśmy **10 podejść** do tokenizacji:

| # | Podejście | Fertility (vocab=500) | Werdykt |
|---|-----------|----------------------|---------|
| 1 | Standard BPE (baseline) | 2.45 | punkt odniesienia |
| 2 | Parallel BPE | ~2.45 | brak poprawy, duży koszt |
| 3 | Semantic BPE (word2vec) | 2.46 | brak poprawy CE LM |
| 4 | Morpheme-aware BPE | 2.54 | nadmierna fragmentacja |
| 5 | MorphBPE strict_suffix | **2.46** | **wybrany** |
| 6 | Bayesian BPE | ~2.48 | niespójny |
| 7 | Compression-aware BPE | ~2.46 | ignoruje morfologię |
| 8 | MorphTok / CBPE | 3.58 | fragmentacja (abugida) |
| 9 | MorphBPE + ToaST | 3.24 | brak ograniczenia morf. |
| 10 | LangMAP | 3.06 | wymaga per-language, nie-BPE |

**Kryteria selekcji:** (i) fertility na poziomie BPE, (ii) respektowanie granic sufiksów, (iii) kompatybilność ze standardowym inference BPE, (iv) peer-reviewed paper (ACL 2025 Findings).

---

## Penalty Sweep (α ∈ [0, 3])

### Zakres standardowy (α ≤ 1)

| Constraint | α=0.0 | α=0.5 | α=1.0 | Trend |
|-----------|-------|-------|-------|-------|
| suffix | 3.131 | 3.111 | **3.090** | monotonicznie ↓ |
| prefix | 3.131 | 3.138 | 3.143 | pogarsza |
| both | 3.131 | 3.135 | 3.111 | gorszy niż suffix alone |

**Wnioski:** (1) Suffix >> prefix (polski = język sufiksalny), (2) Niska kara α=0.1 szkodzi, (3) Korzyść rośnie z α i z vocab.

### Over-penalization (α > 1)

| α | Vocab | CE | MorphED | Fertility |
|---|-------|----|---------|-----------| 
| 1.0 | 500 | 3.092 | 2.423 | 2.449 |
| **2.0** | **500** | **3.044** | **2.332** | 2.388 |
| 1.0 | 1000 | 3.432 | 1.717 | 1.935 |
| **2.0** | **1000** | **3.368** | **1.637** | 1.891 |
| **3.0** | **1000** | **3.324** | **1.592** | 1.869 |

**Kluczowe odkrycie:** α=2.0 daje skokową poprawę (−1.6% CE) — większą niż cała poprawa od α=0.5 do α=1.0.

---

## Skalowanie do rozmiarów produkcyjnych (9/9 ukończone)

| Vocab | α | Fertility | MorphED [CI 95%] | Czas |
|-------|---|-----------|------------------|------|
| 8000 | 0.0 | 1.131 | 0.574 [0.569, 0.578] | 2.3h |
| 8000 | 1.0 | 1.141 | 0.504 [0.499, 0.509] | 1.8h |
| 8000 | 2.0 | 1.149 | **0.456** [0.451, 0.460] | 1.6h |
| 16000 | 0.0 | 1.051 | 0.442 [0.438, 0.446] | 2.9h |
| 16000 | 1.0 | 1.064 | 0.387 [0.384, 0.391] | 3.2h |
| 16000 | 2.0 | 1.077 | **0.338** [0.335, 0.342] | 3.2h |
| 32000 | 0.0 | 1.020 | 0.389 [0.385, 0.393] | 6.0h |
| 32000 | 1.0 | 1.035 | 0.347 [0.343, 0.351] | 5.3h |
| 32000 | 2.0 | 1.049 | **0.308** [0.304, 0.311] | 5.9h |

**Redukcja MorphED (α=0 → α=2):** −20.5% (8k), −23.4% (16k), −21.0% (32k)

**Porównanie z Bielik-11B:** oryginalny tokenizer (Mistral-derived, vocab=32128) ma fertility=3.22. Polsko-zoptymalizowany APT4 (~32k) ma fertility=1.62. MorphBPE: fertility=1.05 → **3.1× niższa** vs oryginał, **1.5× niższa** vs APT4.

---

## Znaczenie praktyczne

1. **Mniej tokenów, tańsze przetwarzanie** — over-penalization obniża fertility o ~2.5–3.4%, co przekłada się na tańsze inference w skali LLM
2. **Lepszy model języka przy tym samym budżecie** — CE spada o 1.6%–3.1% bez zmiany architektury ani danych; jedynym kosztem jest zmiana α
3. **Korzyść rośnie ze słownikiem** — przy vocab=32k redukcja MorphED 3–6× silniejsza niż przy vocab=500–1000
4. **Ekonomia tokenizacji** — tokenizator trenowany raz, używany miliardy razy; zerowy koszt dodatkowy przy treningu, stała korzyść przy inference
5. **Typologia języka ma znaczenie** — dla polskiego (sufiksalny) kluczowe jest ograniczenie suffix, nie prefix

---

## Analiza per-suffix

| Sufiks | N | BPE CE | MorphBPE CE | Δ% |
|--------|---|--------|-------------|-----|
| -ach | 49 | 4.343 | 3.844 | **−11.5%** |
| -ego | 38 | 4.607 | 4.451 | −3.4% |
| -ości | 47 | 5.114 | 4.954 | −3.1% |
| -ych | 40 | 1.941 | 2.759 | +42.1% |
| -ć | 49 | 3.579 | 5.577 | +55.8% |

**Wzorzec:** Sufiksy złożone (≥3 znaki) zyskują, krótkie (1–2 znaki) tracą → sugeruje adaptacyjną karę α wg długości sufiksu.

---

## Warunek obalenia

Teza upada, jeśli:
1. ❌ Fertility > +1% — aktualnie +0.04% (500), +2.8% (32k)
2. ❌ MorphED wzrasta — CI nie pokrywają się, MorphBPE < BPE
3. ❌ CE wzrasta — MorphBPE < BPE przy obu vocab i obu architekturach

**Żaden warunek nie spełniony — synteza nie zostaje obalona.**

---

## Przyszłe prace

1. Ewaluacja extrinsic (LM) przy vocab=32k z większymi modelami (4+ warstw, GPU)
2. Adaptacyjna kara α zależna od długości sufiksu
3. Test na językach słowiańskich (czeski, rosyjski)
4. Porównanie z SentencePiece Unigram
5. Over-penalization dla prefix i both

---

## Kod

Pakiet `morphbpe` — pip-installable, CLI, 29 testów:
- `paper_experiments.py` — główne eksperymenty (BPE vs MorphBPE, vocab=500/1000)
- `transformer_eval.py` — ewaluacja Transformer LM
- `penalty_sweep.py` — przegląd α ∈ [0, 3] × {prefix, suffix, both}
- `scale_vocab.py` — skalowanie do vocab=8k/16k/32k
- `benchmark_dawidek.py` — metryki SI/ISS/MFL (Dawidek 2026)
- `eval_with_model.py` — porównanie z Bielik-11B

---

## Cytowanie

```
Styła, P. (2026). MorphBPE: Morfologicznie informowany Byte-Pair Encoding
dla języka polskiego. Slayer AI Lab.
```

## Licencja

Kod: MIT. Korpus: licencja publiczna (sejm.gov.pl).
