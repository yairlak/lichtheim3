# Sprint 3 — diff review

**Base commit**: `469732d173142f9a9062ad536c9043d6d22d7c32` ·
**Files**: 42 · **Total**: 0.63 MB · **Diff sha256**: `ebdeb6279ba9137e637dfcc6ac5c78f89a763fd8b013aab9e3b5984ee9a89b12`

## Composition

| Area | Change | Notes |
|---|---|---|
| `scripts/behavioral_analysis/frequency.py` | new | frequency estimands, standardization anchor, ceiling policy |
| `scripts/behavioral_analysis/plot_frequency.py` | new | audit tables, two figures, sensitivity, full report generator |
| `tests/test_behavioral_frequency.py` | new | 40 tests, groups A–I |
| `docs/behavioral_wfe_fulllexicon.md` | modified | frequency usage section |
| `docs/behavioral_wfe_analysis_matrix.md` | modified | A14 → ALREADY_VALIDATED, sprint history |
| `reports/.../README.md` | modified | Sprint-3 index section |
| `reports/.../analysis_matrix.tsv` | modified | A14 status row |
| `reports/.../frequency/` | new, 35 files | spec, results, audit, figures, tables, provenance, validation |

## Checks

- `git diff --check`: **0 flagged lines**.
- **No file under `outputs/` or `archives/` staged**; no production prediction
  and no canonical item table staged.
- **No absolute local path** in any file to be staged.
- **No Sprint-1 or Sprint-2 scientific artefact modified** — the Sprint-1
  manifest verifies apart from the two living documents, and the morphology
  manifest verifies in full.
- **Production manifest unchanged**: 36/36 verify.
- Largest new file 201 KB (a 300-dpi PNG); total 0.63 MB.

## Not in this sprint

No feature importance, error taxonomy, SSP, morphology extension or causal
length-effect diagnostic; no checkpoint inference; no change to any analysis
set, seed policy or bootstrap parameter; no pseudoword assigned a frequency.
