# Sprint 2 — diff review

**Base commit**: `178f6a55428100a04bcea11966b653f5f444877e` ·
**Files**: 41 · **Total**: 1.05 MB · **Diff sha256**: `095e181985dfe1f575c34f004a84a8b86a8a10570733b546cc13eb3970facdb0`

## Composition

| Area | Change | Notes |
|---|---|---|
| `scripts/behavioral_analysis/morphology.py` | new, 348 lines | morphology estimands, small-cell flags, bootstrap |
| `scripts/behavioral_analysis/plot_morphology.py` | new, 319 lines | two figures + full Sprint-2 report generator |
| `tests/test_behavioral_morphology.py` | new, 391 lines | 29 tests, groups A–G |
| `docs/behavioral_wfe_fulllexicon.md` | modified | morphology usage section |
| `docs/behavioral_wfe_analysis_matrix.md` | modified | A12/A13 → ALREADY_VALIDATED, sprint history |
| `reports/.../README.md` | modified | Sprint-2 index section |
| `reports/.../analysis_matrix.tsv` | modified | A12/A13 status rows |
| `reports/.../morphology/` | new, 34 files | spec, results, figures, tables, provenance, validation |

The 30 deletions are entirely in the two living documents (replaced matrix rows
and the rewritten "where future analyses go" paragraph). No code was deleted.

## Checks

- `git diff --check`: **0 flagged lines**.
- **No file under `outputs/` or `archives/` staged**; no production prediction
  table staged.
- **No absolute local path** in any file to be staged.
- **No Sprint-1 scientific artefact modified** — all 35 non-living Sprint-1
  entries verify byte-identical, including every figure, plotting table,
  caption, manifest and provenance record. Only `README.md` and
  `analysis_matrix.tsv` changed, both by design as living documents.
- **Production manifest unchanged**: 36/36 verify.
- Largest new file 380 KB (a 300-dpi PNG); total well within git limits.

## Not in this sprint

No frequency, feature-importance, error-taxonomy or SSP computation; no causal
length-effect diagnostics; no checkpoint inference; no change to any analysis
set, seed policy or bootstrap parameter.
