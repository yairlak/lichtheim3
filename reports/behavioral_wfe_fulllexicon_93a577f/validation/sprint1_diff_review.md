# Sprint 1 — diff review

**Base commit**: `e876b755d0475ed11e5fbc0419a0bd8860dfd325` · **Working tree**: no tracked file modified; every change is a new file.
**Files**: 52 · **Total size**: 1.6 MB · **Diff sha256**: `421f69e1597ab9be13c31a39aeb1e91b6ca115351dd91dfcda4aa92faa55a353`

## Composition

| Area | Files | Notes |
|---|---|---|
| `scripts/behavioral_analysis/` | 10 .py | The promoted analysis package (inference-free) |
| `tests/test_behavioral_analysis.py` | 1 | 54 tests, groups A–F |
| `docs/` | 2 .md | Method documentation and the analysis matrix |
| `reports/behavioral_wfe_fulllexicon_93a577f/` | 38 | 5 figures × (png, pdf, svg, caption) + 8 plotting tables + manifests, provenance, validation |
| `.gitattributes` | 1 | Whitespace/binary rules for generated data |

## Checks

- `git diff --check`: **0 flagged lines** (after `.gitattributes` exempts generated `.tsv`/`.svg` from whitespace checks — TSVs legitimately end in empty columns and matplotlib SVG path data carries trailing spaces; rewriting either would corrupt exact plotting tables and vector output).
- **No tracked source file modified.** `git status --porcelain --untracked-files=no` is empty; the evaluation code, model, data and frozen protocol are untouched.
- **No absolute user path** in any file to be tracked (asserted by `test_F_no_absolute_user_paths_in_package`).
- **No production predictions staged.** All production outputs live under gitignored `outputs/`; only the publication copies in `reports/` are tracked.
- **No `archives/` staging.** The 21 MB checkpoint bundle stays untracked.
- **Largest tracked file is 213 KB** (a 300-dpi PNG); nothing approaches a size problem for git.

## What is deliberately not here

No morphology, frequency, feature-importance or error-taxonomy computation; no SSP; no causal interpretation. Those are Sprint 2+ per `docs/behavioral_wfe_analysis_matrix.md`. The original gitignored drivers are retained unchanged, with their relationship to the promoted code recorded in `analysis_code_migration.tsv`.
