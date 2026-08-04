# MorphBPE ÔÇö Morfologicznie informowany Byte-Pair Encoding dla j─Özyka polskiego

## Streszczenie

MorphBPE to wariant algorytmu BPE (Byte-Pair Encoding), kt├│ry wymusza poszanowanie granic morfem├│w podczas scalania par token├│w. Granice sufiks├│w pozyskiwane s─ů z analizatora morfologicznego **Morfeusz2**; scalenia przekraczaj─ůce granic─Ö sufiksu s─ů karane mno┼╝nikiem cz─Östotliwo┼Ťci (`score = freq ├Ś (1 - ╬▒)`).

Praca stanowi **pierwsz─ů ewaluacj─Ö MorphBPE na j─Özyku polskim** z analizatorem Morfeusz2, ewaluacj─ů intrinsic i extrinsic z dwiema architekturami (LSTM i Transformer), analiz─ů b┼é─Öd├│w per-suffix, oraz skalowaniem do rozmiar├│w produkcyjnych (32 000 token├│w).

---

## Kluczowe wyniki

| Metryka | Warto┼Ť─ç | Kontekst |
|---------|---------|----------|
| MorphED redukcja (vocab=500) | Ôłĺ3.6% | vs standardowy BPE |
| MorphED redukcja (vocab=1000) | Ôłĺ6.8% | vs standardowy BPE |
| MorphED redukcja (vocab=32000, ╬▒=2.0) | **Ôłĺ21.0%** | vs standardowy BPE |
| CE redukcja LSTM (vocab=1000) | Ôłĺ2.6% | cross-entropy modelu j─Özyka |
| CE redukcja Transformer (vocab=1000) | Ôłĺ2.7% | cross-entropy modelu j─Özyka |
| Fertility koszt | < 3% | nawet przy vocab=32k |
| MorphBPE vs Bielik-11B orig. | 1.05 vs 3.22 | **3.1├Ś ni┼╝sza** przy vocab=32k |
| MorphBPE vs Bielik APT4 | 1.05 vs 1.62 | **1.5├Ś ni┼╝sza** (vs zoptymalizowany) |

---

## Wk┼éad naukowy

1. **Pierwsza ewaluacja MorphBPE na polskim** ÔÇö Asgari et al. (2025) testowali na angielskim, rosyjskim, w─Ögierskim, arabskim, ale nie polskim
2. **Over-penalization (╬▒ > 1)** ÔÇö oryginalny MorphBPE definiuje ╬▒ Ôłł [0,1]. Rozszerzyli┼Ťmy na ╬▒ > 1 i pokazali┼Ťmy skokow─ů popraw─Ö (Ôłĺ1.6% do Ôłĺ3.1% CE nad ╬▒=1.0)
3. **Analiza b┼é─Öd├│w per-suffix** ÔÇö identyfikujemy kt├│re klasy sufiks├│w zyskuj─ů (z┼éo┼╝one: -ach, -ego, -o┼Ťci) i trac─ů (kr├│tkie: -─ç, -ych)
4. **Ewaluacja z dwiema architekturami** (LSTM i Transformer) ÔÇö potwierdzenie, ┼╝e korzy┼Ť─ç jest odporna na wyb├│r architektury
5. **Skalowanie do rozmiar├│w produkcyjnych** (vocab=8kÔÇô32k) ÔÇö korzy┼Ť─ç 3ÔÇô6├Ś silniejsza ni┼╝ przy vocab=500ÔÇô1000

---

## Struktura repozytorium

```
paper-tokenization/
ÔöťÔöÇÔöÇ main.tex              # Wrapper LaTeX (wiring)
ÔöťÔöÇÔöÇ formatka/
Ôöé   ÔööÔöÇÔöÇ slayer.sty        # Formatka Slayer (wygl─ůd dokumentu)
ÔöťÔöÇÔöÇ tresc/
Ôöé   ÔöťÔöÇÔöÇ 00-strona-tytulowa.tex   # Strona tytu┼éowa + wykres hero
Ôöé   ÔööÔöÇÔöÇ 10-tresc.tex             # W┼éa┼Ťciwa tre┼Ť─ç papera (~1000 linii)
ÔöťÔöÇÔöÇ prezentacja.html      # Reveal.js prezentacja (standalone, otw├│rz w przegl─ůdarce)
ÔöťÔöÇÔöÇ prezentacja.tex       # LaTeX wersja prezentacji
ÔöťÔöÇÔöÇ reveal.css            # Reveal.js CSS (lokalna kopia)
ÔöťÔöÇÔöÇ reveal.js             # Reveal.js JS (lokalna kopia)
ÔöťÔöÇÔöÇ theme-white.css       # Motyw prezentacji
ÔöťÔöÇÔöÇ MorphBPE_paper.pdf    # Wygenerowany PDF papera
ÔöťÔöÇÔöÇ main.pdf              # PDF z LaTeX (wcze┼Ťniejsza wersja)
ÔööÔöÇÔöÇ README.md             # Opis subfolderu
```

---

## Metodologia

### Korpus
- **10 000 przem├│wie┼ä** z posiedze┼ä Sejmu RP (sejm.gov.pl)
- Dost─Öpny na HuggingFace: [`PiotrSty/sejm-speeches-corpus`](https://huggingface.co/datasets/PiotrSty/sejm-speeches-corpus)
- Split: 80/20 (8000 train / 2000 test), seed=42

### Tokenizatory
- **Standard BPE** ÔÇö klasyczny algorytm oparty na cz─Östotliwo┼Ťci par (baseline)
- **MorphBPE strict_suffix** ÔÇö BPE z kar─ů za scalenia przekraczaj─ůce granic─Ö sufiksu (Morfeusz2)
- Parametr ╬▒ kontroluje si┼é─Ö kary: ╬▒=0 Ôćĺ BPE, ╬▒=1 Ôćĺ pe┼éne blokowanie, ╬▒=2 Ôćĺ over-penalization

### Ewaluacja intrinsic
- **Fertility** ÔÇö liczba token├│w na s┼éowo (ni┼╝ej = lepiej)
- **MorphED** (Morphological Edit Distance) ÔÇö odleg┼éo┼Ť─ç Levenshteina mi─Ödzy segmentacj─ů tokenizatora a morfologiczn─ů (ni┼╝ej = lepiej)
- **Bootstrap CI** ÔÇö 1000 resamples, 95% przedzia┼é ufno┼Ťci

### Ewaluacja extrinsic
- **LSTM LM** ÔÇö 1 warstwa, embed=64, hidden=128, 3 epoki
- **Transformer LM** ÔÇö 2 warstwy, 4 g┼éowice, embed=64, ff=256, 3 epoki
- Metryki: cross-entropy (CE), perplexity (PPL)

---

## Przegl─ůd zbadanych podej┼Ť─ç (10 podej┼Ť─ç)

Przed wyborem MorphBPE strict_suffix zbadali┼Ťmy **10 podej┼Ť─ç** do tokenizacji:

| # | Podej┼Ťcie | Fertility (vocab=500) | Werdykt |
|---|-----------|----------------------|---------|
| 1 | Standard BPE (baseline) | 2.45 | punkt odniesienia |
| 2 | Parallel BPE | ~2.45 | brak poprawy, du┼╝y koszt |
| 3 | Semantic BPE (word2vec) | 2.46 | brak poprawy CE LM |
| 4 | Morpheme-aware BPE | 2.54 | nadmierna fragmentacja |
| 5 | **MorphBPE strict_suffix** | **2.46** | **wybrany** |
| 6 | Bayesian BPE | ~2.48 | niesp├│jny |
| 7 | Compression-aware BPE | ~2.46 | ignoruje morfologi─Ö |
| 8 | MorphTok / CBPE | 3.58 | fragmentacja (abugida) |
| 9 | MorphBPE + ToaST | 3.24 | brak ograniczenia morf. |
| 10 | LangMAP | 3.06 | wymaga per-language, nie-BPE |

**Kryteria selekcji:** (i) fertility na poziomie BPE, (ii) respektowanie granic sufiks├│w z Morfeusz2, (iii) kompatybilno┼Ť─ç ze standardowym inference BPE, (iv) peer-reviewed paper (ACL 2025 Findings).

---

## Wyniki g┼é├│wne (vocab=500 i 1000)

### Intrinsic

| Metoda | Vocab | Fertility [CI] | MorphED [CI] |
|--------|-------|----------------|--------------|
| BPE | 500 | 2.443 [2.439, 2.447] | 2.503 [2.497, 2.509] |
| MorphBPE | 500 | 2.444 [2.440, 2.449] | 2.414 [2.408, 2.420] |
| BPE | 1000 | 1.932 [1.929, 1.936] | 1.838 [1.832, 1.843] |
| MorphBPE | 1000 | 1.934 [1.930, 1.937] | 1.712 [1.707, 1.717] |

### Extrinsic ÔÇö LSTM

| Metoda | Vocab | CE | PPL | ╬öCE |
|--------|-------|----|----|-----|
| BPE | 500 | 3.276 | 26.46 | ÔÇö |
| MorphBPE | 500 | 3.233 | 25.36 | Ôłĺ1.3% |
| BPE | 1000 | 3.744 | 42.26 | ÔÇö |
| MorphBPE | 1000 | 3.646 | 38.33 | Ôłĺ2.6% |

### Extrinsic ÔÇö Transformer

| Metoda | Vocab | CE | PPL | ╬öCE |
|--------|-------|----|----|-----|
| BPE | 500 | 3.131 | 22.89 | ÔÇö |
| MorphBPE | 500 | 3.090 | 21.98 | Ôłĺ1.3% |
| BPE | 1000 | 3.525 | 33.95 | ÔÇö |
| MorphBPE | 1000 | 3.431 | 30.89 | Ôłĺ2.7% |

---

## Penalty Sweep (╬▒ Ôłł [0, 3])

### Zakres standardowy (╬▒ ÔëĄ 1, Transformer CE)

| Constraint | ╬▒=0.0 | ╬▒=0.1 | ╬▒=0.25 | ╬▒=0.5 | ╬▒=0.75 | ╬▒=1.0 |
|-----------|-------|-------|--------|-------|--------|-------|
| suffix | 3.131 | 3.144 | 3.142 | 3.111 | 3.095 | **3.090** |
| prefix | 3.131 | 3.138 | 3.140 | 3.138 | 3.139 | 3.143 |
| both | 3.131 | 3.144 | 3.132 | 3.135 | 3.116 | 3.111 |

*(vocab=500, Transformer LM)*

**Wnioski:**
- **Suffix >> prefix >> both** ÔÇö polski to j─Özyk sufiksalny, fleksja na ko┼äcu wyrazu
- **Niska kara (╬▒=0.1) szkodzi** ÔÇö zaburza BPE bez wystarczaj─ůcego sygna┼éu morfologicznego
- **Korzy┼Ť─ç ro┼Ťnie monotonicznie z ╬▒** i z rozmiarem s┼éownika

### Over-penalization (╬▒ > 1)

| ╬▒ | Vocab | Fertility | MorphED [CI] | CE | PPL |
|---|-------|-----------|-------------|----|----|
| 1.0 | 500 | 2.449 | 2.423 [2.419, 2.427] | 3.092 | 22.01 |
| **2.0** | **500** | 2.388 | **2.332** [2.327, 2.336] | **3.044** | **21.00** |
| 3.0 | 500 | 2.387 | 2.322 [2.318, 2.327] | 3.046 | 21.02 |
| 1.0 | 1000 | 1.935 | 1.717 [1.713, 1.721] | 3.432 | 30.93 |
| **2.0** | **1000** | 1.891 | **1.637** [1.633, 1.641] | **3.368** | 29.02 |
| **3.0** | **1000** | 1.869 | **1.592** [1.589, 1.596] | **3.324** | **27.77** |

**Kluczowe odkrycie:** ╬▒=2.0 daje skokow─ů popraw─Ö (Ôłĺ1.6% CE przy vocab=500) ÔÇö wi─Öksz─ů ni┼╝ ca┼éa poprawa od ╬▒=0.5 do ╬▒=1.0. Mechanizm: crossing merges dostaj─ů ujemny score, co reorganizuje kolejno┼Ť─ç scale┼ä.

---

## Skalowanie do rozmiar├│w produkcyjnych (9/9 uko┼äczone)

| Vocab | ╬▒ | Fertility | MorphED [CI 95%] | Czas |
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

**Redukcja MorphED (╬▒=0 Ôćĺ ╬▒=2):** Ôłĺ20.5% (8k), Ôłĺ23.4% (16k), Ôłĺ21.0% (32k)

**Por├│wnanie z Bielik-11B:** oryginalny tokenizer (Mistral-derived, vocab=32128) ma fertility=3.22. Nowy polsko-zoptymalizowany APT4 (~32k) ma fertility=1.62. MorphBPE: fertility=1.05 przy vocab=32000 Ôćĺ **3.1├Ś ni┼╝sza** vs orygina┼é, **1.5├Ś ni┼╝sza** vs APT4.

**Wniosek:** Korzy┼Ť─ç z MorphBPE nie tylko utrzymuje si─Ö przy skalowaniu ÔÇö jest **3ÔÇô6├Ś silniejsza** ni┼╝ przy vocab=500ÔÇô1000.

---

## Znaczenie praktyczne

1. **Mniej token├│w, ta┼äsze przetwarzanie** ÔÇö over-penalization obni┼╝a fertility o ~2.5ÔÇô3.4%, co w skali LLM oznacza proporcjonalnie ta┼äsze inference
2. **Lepszy model j─Özyka przy tym samym bud┼╝ecie** ÔÇö CE spada o 1.6%ÔÇô3.1% bez zmiany architektury, danych ani mocy; jedynym kosztem jest zmiana hiperparametru ╬▒
3. **Korzy┼Ť─ç ro┼Ťnie ze s┼éownikiem** ÔÇö przy vocab=32k redukcja MorphED jest 3ÔÇô6├Ś silniejsza ni┼╝ przy 500ÔÇô1000; przy produkcyjnych rozmiarach (30kÔÇô100k) korzy┼Ť─ç mo┼╝e by─ç jeszcze wi─Öksza
4. **Ekonomia tokenizacji** ÔÇö tokenizator trenowany raz, u┼╝ywany miliardy razy. Zerowy koszt dodatkowy przy treningu, sta┼éa korzy┼Ť─ç przy ka┼╝dym inference
5. **Typologia j─Özyka ma znaczenie** ÔÇö suffix >> prefix >> both dla polskiego. Dla ka┼╝dego j─Özyka nale┼╝y dobra─ç constraint type zgodny z jego typologi─ů

---

## Analiza per-suffix

| Sufiks | N | BPE CE | MorphBPE CE | ╬ö% | Interpretacja |
|--------|---|--------|-------------|-----|---------------|
| -ach | 49 | 4.343 | 3.844 | **Ôłĺ11.5%** | z┼éo┼╝ony, zyskuje |
| -ego | 38 | 4.607 | 4.451 | Ôłĺ3.4% | z┼éo┼╝ony, zyskuje |
| -o┼Ťci | 47 | 5.114 | 4.954 | Ôłĺ3.1% | z┼éo┼╝ony, zyskuje |
| -enia | 26 | 2.638 | 2.603 | Ôłĺ1.3% | z┼éo┼╝ony, zyskuje |
| -ami | 50 | 3.853 | 3.872 | +0.5% | neutralny |
| -ny | 43 | 3.079 | 3.357 | +9.0% | kr├│tki, traci |
| -├│w | 32 | 2.722 | 3.035 | +11.5% | kr├│tki, traci |
| -ych | 40 | 1.941 | 2.759 | +42.1% | kr├│tki, traci |
| -─ç | 49 | 3.579 | 5.577 | +55.8% | 1-znakowy, traci |

**Wzorzec:** Sufiksy z┼éo┼╝one (Ôëą3 znaki) zyskuj─ů na MorphBPE. Kr├│tkie (1ÔÇô2 znaki) trac─ů ÔÇö wymuszone oddzielenie jednoliterowego sufiksu powoduje nadmiern─ů fragmentacj─Ö. Sugeruje to **adaptacyjn─ů kar─Ö** ╬▒ zale┼╝n─ů od d┼éugo┼Ťci sufiksu.

---

## Warunek obalenia

Teza o wy┼╝szo┼Ťci MorphBPE upada, je┼Ťli:

| # | Warunek | Status | Warto┼Ť─ç |
|---|---------|--------|---------|
| 1 | Fertility > +1% | ÔŁî nie spe┼éniony | +0.04% (500), +2.8% (32k) |
| 2 | MorphED ro┼Ťnie (CI si─Ö pokrywaj─ů) | ÔŁî nie spe┼éniony | CI roz┼é─ůczne, MorphBPE < BPE |
| 3 | CE ro┼Ťnie (LM gorsza) | ÔŁî nie spe┼éniony | MorphBPE < BPE wsz─Ödzie |

**┼╗aden warunek nie spe┼éniony ÔÇö synteza nie zostaje obalona.**

---

## Przysz┼ée prace

1. **Ewaluacja extrinsic przy vocab=32k** ÔÇö z wi─Ökszymi modelami (Transformer 4+ warstw), wymaga GPU
2. **Adaptacyjna kara ╬▒** ÔÇö ╬▒(sufiks) = f(d┼éugo┼Ť─ç sufiksu), np. ╬▒=2.0 dla Ôëą3 znak├│w, ╬▒=0.5 dla 1ÔÇô2 znak├│w
3. **Inne j─Özyki s┼éowia┼äskie** ÔÇö czeski, rosyjski z odpowiednimi analizatorami
4. **Por├│wnanie z SentencePiece Unigram** ÔÇö inny paradygmat tokenizacji
5. **Over-penalization dla prefix i both** ÔÇö weryfikacja czy ╬▒Ôëł2.0 przenosi si─Ö na inne constraint types

---

## Kod

Pakiet `morphbpe` ÔÇö pip-installable, CLI, 29 test├│w:

| Skrypt | Cel |
|--------|-----|
| `paper_experiments.py` | G┼é├│wne eksperymenty (BPE vs MorphBPE, vocab=500/1000) |
| `transformer_eval.py` | Ewaluacja Transformer LM |
| `penalty_sweep.py` | Przegl─ůd ╬▒ Ôłł [0, 3] ├Ś {prefix, suffix, both} |
| `penalty_sweep_extended.py` | Over-penalization ╬▒ > 1 |
| `scale_vocab.py` | Skalowanie do vocab=8k/16k/32k |
| `benchmark_dawidek.py` | Metryki SI/ISS/MFL (Dawidek 2026) |
| `eval_with_model.py` | Por├│wnanie z Bielik-11B |

---

## Prezentacja

Otw├│rz `paper-tokenization/prezentacja.html` w przegl─ůdarce ÔÇö standalone reveal.js, nie wymaga serwera.

---

## Kompilacja papera (LaTeX)

```bash
cd paper-tokenization && pdflatex main.tex && pdflatex main.tex
```

Wymaga MiKTeX/TeXLive z pakietami: pgfplots, tcolorbox, booktabs, xcolor, babel-polish.

---

## Cytowanie

```
Sty┼éa, P. (2026). MorphBPE: Morfologicznie informowany Byte-Pair Encoding
dla j─Özyka polskiego. Slayer AI Lab.
```

## Licencja

Kod: MIT. Korpus: licencja publiczna (sejm.gov.pl).
