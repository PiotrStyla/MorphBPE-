---
description: Generuj empiryczny raport LaTeX wg standardu slay-paper z danych badawczych → PDF przez CI
---

## Kontekst standardu

Szablon: https://github.com/slayerlabs/slay-paper  
Dwie bramki jakości:
- **Epistemiczna** — dialektyka: Teza ↔ Antyteza → Synteza
- **Empiryczna** — warunek obalenia: konkretna liczba/test przy każdej syntezie

Twarde reguły: matematyka zawsze w LaTeX (`$\Delta$` nie Δ), formatka ⟂ treść.

---

## Kroki

### 1. Pobierz dane z repozytorium datasetu

Podaj URL repozytorium GitHub z danymi. Cascade pobiera:
- README (struktura, stan zbioru)
- `reports/quality-report.md` (liczby: rekordy, znaki, kategorie, dzielnice, kompletność pól, safety checks)
- `reports/redundancy-report.md` (TF-IDF, pary wysokiego ryzyka)
- `reports/slayer-readiness.md` (werdykt gotowości)
- `reports/dataset-card.md` (metadane, licencja, ograniczenia)

### 2. Utwórz paper jako podfolder w repo z danymi

Paper trafia bezpośrednio do repo z danymi (`PiotrStyla/Datasets`):

```
PiotrStyla/Datasets/
├── datasets/<nazwa>/           ← dane (JSONL, raporty jakości, itp.)
└── papers/<nazwa>/             ← raport LaTeX ← TUTAJ
    ├── main.tex                    ← wrapper (nie edytować)
    ├── formatka/slayer.sty         ← wygląd (nie edytować)
    └── tresc/
        ├── 00-strona-tytulowa.tex  ← tytuł + autor + wykres hero
        └── 10-tresc.tex            ← 7 obowiązkowych sekcji
```

CI w `.github/workflows/latex.yml` (już w repo) **automatycznie wykrywa** każdy nowy folder `papers/*/` i buduje PDF — nie trzeba edytować workflow przy nowych raportach.

Pliki `main.tex` i `formatka/slayer.sty` kopiowane bez zmian.  
Edytowane są tylko `tresc/*.tex` — wypełniane realnymi danymi z kroku 1.

### 3. Wypełnij 00-strona-tytulowa.tex

- Tytuł raportu i autor
- **Wykres hero** (pgfplots `xbar`) — rozkład kategorii lub kluczowa metryka z danych  
- Podpis pod wykresem z realnymi liczbami
- Użyj `symbolic y coords` z kluczami ASCII + `yticklabels` z polskimi etykietami
- `xmajorgrids=true` zamiast `grid=x` (pgfplots nie obsługuje `grid=x`)

### 4. Wypełnij 10-tresc.tex — 7 obowiązkowych sekcji

1. **Streszczenie** — 3–5 zdań: CO, NA JAKICH danych, JAKI wynik liczbowy
2. **Wprowadzenie** — nitka dialektyczna:
   - `\textbf{Teza~(T):}` — pozytywne twierdzenie z liczbami
   - `\textbf{Antyteza~(A):}` — konkretna słabość z liczbami  
   - `\textbf{Synteza~(S):}` — co z tego wynika i co trzeba zrobić
3. **Metoda** — źródło, licencja (`\footnote{\url{...}}`), ekstrakcja, czyszczenie, metryki formalne w LaTeX (`\[ ... \]`)
4. **Wyniki** — min. 2 tabele (`booktabs`) + 1 wykres (pgfplots) z realnymi liczbami
5. **Warunek obalenia** — min. 2–3 mierzalne progi w `enumerate`, każdy z aktualnym wynikiem pogrubionym
6. **Dyskusja** — ograniczenia + priorytety następnego etapu
7. **Podpis** — `\emph{autor}`, licencja danych, `\today`

**Matematyka zawsze w LaTeX:** `$\ge 0{,}50$` nie ≥ 0,50 · `$\binom{n}{2}$` · `$\approx$` nie ≈  
**Tysiące:** `152\,526` · **Ułamki PL:** `0{,}034`

### 5. Commit i push do repo Datasets

```powershell
# PowerShell — używaj ; zamiast &&
# Zakładamy, że PiotrStyla/Datasets jest sklonowany lokalnie
git add papers/<nazwa>; git commit -m "feat: raport <nazwa> v0.1"; git push
```

### 6. Sprawdź CI

Wejdź w `https://github.com/PiotrStyla/Datasets/actions`  
→ zielony run → sekcja **Artifacts** na dole → pobierz `<nazwa>-pdf` (ZIP z `main.pdf`)  
CI **automatycznie buduje każdy nowy folder** `papers/*/` — nie trzeba edytować workflow.

---

## Najczęstsze błędy

| Błąd | Przyczyna | Fix |
|---|---|---|
| `Choice 'x' unknown in choice key '/pgfplots/grid'` | `grid=x` nieznane | zamień na `xmajorgrids=true` |
| `Undefined control sequence \cite` | brak bibliografii | użyj `\footnote{\url{...}}` |
| `&&` nie działa w PowerShell | zły separator | użyj `;` |
| Lint: unicode matematyczny | znak ≤ ≥ w `.tex` | zamień na `$\le$` `$\ge$` |
