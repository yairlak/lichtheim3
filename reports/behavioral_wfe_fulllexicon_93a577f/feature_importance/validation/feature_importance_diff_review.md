# Sprint 5 — diff review

**Base commit**: `697917d5141769bf02d08ba853b3c767e18e6370` ·
**Files**: 52 · **Total**: 1.40 MB ·
**Diff sha256**: `9e3dfe428ecfe0aef24025b96f5127507c84635f7ca5ee9bb585846b6cae658c`
(`git diff | shasum -a 256`, tracked unstaged changes only)

> **Closure.** This review describes the working tree immediately before
> Commit A; the hash above is that pre-commit diff and is kept as the record of
> what was reviewed. Commit A is
> `f352de120bda68e3d8497c3ef17f98014d515037`
> (`feat(analysis): add adapted WFE feature-importance analyses`), 48 files.
> This file and the other provenance/validation artefacts follow in Commit B.
> Two prose-only clarifications were applied after this review and before
> Commit A, both in `feature_importance_results.md` §13: the route versus
> lexicality/exposure ordering is now explicitly stated as not robust, with all
> four reasons in one place, and the morphology finding is now explicitly scoped
> to predictive contribution rather than proof of absence from the model's
> representations. No figure, plotting table, permutation result, coefficient,
> split, model formula, random state or analysis set changed; regeneration
> remained byte-identical.

## Composition

| Area | Change | Notes |
|---|---|---|
| `scripts/behavioral_analysis/feature_importance.py` | new | grouped item split, training-fitted design, grouped raw-factor permutation, statuses, seed-resampling interval |
| `scripts/behavioral_analysis/plot_feature_importance.py` | new | three models, 20 tables, up to 3 figures, output manifest |
| `tests/test_behavioral_feature_importance.py` | new | 59 tests, groups A–J |
| `docs/behavioral_wfe_fulllexicon.md` | modified | Sprint-5 usage, identifiability, leakage rule, permutation semantics, sign policy |
| `docs/behavioral_wfe_analysis_matrix.md` | modified | A15 → `ALREADY_VALIDATED`, sprint history, Sprint-4 commits recorded |
| `reports/.../README.md` | modified | Sprint-5 index section, manifest row |
| `reports/.../analysis_matrix.tsv` | modified | A15 and A11 rows (2 lines) |
| `reports/.../feature_importance/` | new, 45 files | spec, results, faithful-vs-adapted, figures, tables, provenance, validation |

## Checks

- `git diff --check`: **0 flagged lines**.
- **Nothing under `outputs/` or `archives/` staged**; no production prediction,
  no canonical table, no checkpoint, no scratchpad script.
- **The faithful A11 outputs are untouched** — their hashes match the values
  recorded in the Sprint-5 preflight, and no file in
  `outputs/.../faithful_replication/` has an mtime later than that preflight.
- **No prior-sprint scientific artefact modified.** Sprint-1, morphology,
  frequency and error-taxonomy manifests verify apart from the two declared
  living documents; the production manifest verifies in full (36/36) with no
  exemption.
- **Canonical table unchanged**; no `data/` path appears in `git status`.
- **Deterministic regeneration run twice**, both byte-identical to the published
  tree.
- **Tests**: 59 new, 241 across the behavioral-analysis suites, **471 passed /
  4 deselected** for `pytest tests/ -m "not slow"`.
- **Validation**: 29/29 checks PASS.

## Corrections made during this sprint

1. **Status collapse.** The first implementation overwrote
   `NEAR_ZERO_VARIANCE` with `NON_ESTIMABLE` whenever held-out R² was NaN,
   losing the reason. `outcome_status` (pre-fit data diagnosis) and
   `model_status` are now both recorded, alongside an explicit
   `negative_test_r2` flag so a negative held-out R² is never hidden behind
   another status. No estimate changed.
2. **Prose-scanning guards.** Three tests and one validation check initially
   flagged the modules' own docstrings — which state the constraints ("Zipf is
   undefined for pseudowords", the path of the faithful analysis, "not tuned")
   — as violations of them. The guards now scan executable code with
   docstrings and comments stripped, and the faithful-directory check compares
   modification times rather than filenames. No scientific code changed.

## Not in this sprint

No SSP; no checkpoint inference; no retraining; no architecture, evaluation-code,
dataset or checkpoint change; no WFE hyperparameter tuning; no row-level leakage
split; no model containing lexicality and exposure together; no pseudoword
frequency; no replacement of the faithful FI; no pooled faithful/adapted claim;
no causal explanation; no architectural recommendation; no overwrite of a prior
output.
