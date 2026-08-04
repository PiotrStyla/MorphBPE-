# Slay-Paper — LaTeX raport z danych → PDF przez CI

**5 minut. Zero konfiguracji LaTeX lokalnie. PDF buduje się w GitHub Actions.**

---

## Co dostajesz

- Raport LaTeX → PDF kompilowany automatycznie po każdym `git push`
- Dynamiczny CI: każdy nowy folder `papers/<nazwa>/` budowany automatycznie — zero zmian w workflow
- Dwie bramki jakości: dialektyka (Teza ↔ Antyteza → Synteza) + warunek obalenia
- Lint który sprawdza, czy nie ma unicode w matematyce (zamiast LaTeX)

---

## Quickstart

### Krok 1 — Dodaj CI do swojego repo

Utwórz plik `.github/workflows/latex.yml` w swoim repo:

```yaml
name: Build Papers (PDF)

on:
  push:
    paths:
      - 'papers/**'
      - '.github/workflows/latex.yml'
  pull_request:
  workflow_dispatch:

jobs:
  discover:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.list.outputs.matrix }}
    steps:
      - uses: actions/checkout@v4
      - id: list
        run: |
          matrix=$(find papers -maxdepth 1 -mindepth 1 -type d | sort \
            | jq -R -s -c 'split("\n") | map(select(length > 0))')
          echo "matrix=$matrix" >> $GITHUB_OUTPUT

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint — matematyka w LaTeX, nie unicode
        run: |
          python3 - <<'EOF'
          import sys, glob, re
          BAD = set("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
                    "≤≥≠≈±×÷√∞∑∏∫∂∇⟂∈∉⊂⊆⊇∪∩∀∃⇒⇔≅≡∝∅")
          hits = []
          for f in glob.glob("papers/**/*.tex", recursive=True):
              for i, line in enumerate(open(f, encoding="utf-8"), 1):
                  code = re.sub(r"(?<!\\)%.*$", "", line)
                  for ch in code:
                      if ch in BAD:
                          hits.append((f, i, ch))
          if hits:
              print("Unicode matematyczny — użyj LaTeX ($\\le$, $\\Delta$):")
              for f, i, ch in hits[:50]:
                  print(f"  {f}:{i}  «{ch}» (U+{ord(ch):04X})")
              sys.exit(1)
          print("OK")
          EOF

  build:
    needs: [discover, lint]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        paper: ${{ fromJson(needs.discover.outputs.matrix) }}
    steps:
      - uses: actions/checkout@v4
      - uses: xu-cheng/latex-action@v3
        with:
          working_directory: ${{ matrix.paper }}
          root_file: main.tex
      - id: name
        run: echo "name=$(basename '${{ matrix.paper }}')" >> $GITHUB_OUTPUT
      - uses: actions/upload-artifact@v4
        with:
          name: ${{ steps.name.outputs.name }}-pdf
          path: ${{ matrix.paper }}/main.pdf
          if-no-files-found: error
```

### Krok 2 — Skopiuj strukturę szablonu

Utwórz `papers/<twoja-nazwa>/` z czterema plikami. Pobierz je z:

```
papers/warsaw-civic-budget-2027/        ← przykład gotowego raportu
├── main.tex                            ← skopiuj bez zmian
├── formatka/slayer.sty                 ← skopiuj bez zmian
└── tresc/
    ├── 00-strona-tytulowa.tex          ← podmień tytuł, autora, wykres hero
    └── 10-tresc.tex                    ← wpisz swoje dane (7 sekcji)
```

Gotowe pliki: https://github.com/PiotrStyla/Datasets/tree/main/papers/warsaw-civic-budget-2027

### Krok 3 — Wypełnij dane

Edytuj tylko `tresc/*.tex`. Obowiązkowe sekcje:

1. **Streszczenie** — 3–5 zdań, empiria od razu
2. **Wprowadzenie** — Teza / Antyteza / Synteza
3. **Metoda** — dane, definicje formalne w LaTeX
4. **Wyniki** — tabele + wykresy z liczbami
5. **Warunek obalenia** — konkretna liczba/próg który mógłby obalić tezę *(obowiązkowy)*
6. **Dyskusja** — ograniczenia, co dalej
7. **Podpis** — autor + data

### Krok 4 — Push

```bash
git add papers/<nazwa>/ .github/workflows/latex.yml
git commit -m "feat: raport <nazwa>"
git push
```

→ **Actions → artefakt `<nazwa>-pdf` → pobierz ZIP → `main.pdf`**

---

## Następne raporty

Każdy kolejny raport = nowy folder `papers/<inna-nazwa>/` z tymi samymi 4 plikami.
**CI wykrywa go automatycznie.** Nie trzeba niczego zmieniać w workflow.

---

## Twarde reguły (sprawdza CI)

| ✅ | ❌ |
|---|---|
| `$\Delta$` | Δ |
| `$\le$` | ≤ |
| `$\approx$` | ≈ |
| `$\sum$` | ∑ |

## Znane pułapki

| Błąd CI | Fix |
|---|---|
| `Choice 'x' unknown in /pgfplots/grid` | zamień `grid=x` na `xmajorgrids=true` |
| `Undefined \cite` | użyj `\footnote{\url{...}}` |
| `&&` w PowerShell | użyj `;` |

---

## Przykłady

- **Dataset report:** https://github.com/PiotrStyla/Datasets/tree/main/papers/warsaw-civic-budget-2027
- **Benchmark report:** https://github.com/PiotrStyla/PES-CoM/tree/main/papers/pes-com-v02
- **Oryginalny szablon:** https://github.com/slayerlabs/slay-paper
