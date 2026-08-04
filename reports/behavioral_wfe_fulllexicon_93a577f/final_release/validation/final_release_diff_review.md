# Final release — diff review

**Base commit**: `2edf9f3b54af202b554054f926d8fa9b457e6c3a` ·
**Files**: 122 · **Total**: 9.26 MB ·
**Diff sha256**: `6cfc333aa77c5f6ffbaf4babcd38b6972871a682ed5e6895fe738b735550da5d` (`git diff | shasum -a 256`, tracked unstaged only)

> **Closure.** This review describes the working tree immediately before
> Commit A; the hash above is that pre-commit diff and is kept as the record of
> what was reviewed. Commit A is
> `4813289598fbec47c48d70b77217cd94e52b3243`
> (`feat(analysis): finalize WFE behavioral publication release`), **118 files**.
> This file and the other provenance/validation artefacts follow in Commit B.
> No scientific figure, table, caption, index or summary changed after the
> review: the committed code regenerates all 96 release artefacts
> byte-identically, all 79 release copies still match their sources, and the
> A09/A10/A11 plotting tables remain byte-identical to their authoritative
> originals.

## Composition

| Area | Change | Notes |
|---|---|---|
| `scripts/behavioral_analysis/final_release.py` | new | selection, copy-with-provenance, legacy table loading, summary tables |
| `scripts/behavioral_analysis/plot_final_release.py` | new | renders A09/A10/A11 from stored tables; assembles the release tree |
| `tests/test_behavioral_final_release.py` | new | 52 tests, groups A–I |
| `docs/behavioral_wfe_fulllexicon.md` | modified | final-release usage section |
| `docs/behavioral_wfe_analysis_matrix.md` | modified | A09/A10/A11 → `ALREADY_VALIDATED_FORMATTED`, A19 → `OPTIONAL_DEFERRED`, new A23 row, final project status |
| `reports/.../README.md` | modified | final-release section, manifest row |
| `reports/.../analysis_matrix.tsv` | modified | 4 rows updated + A23 added (5 changed lines) |
| `reports/.../final_release/` | new, 115 files | spec, audit, summaries, 19 figures × 3 formats, captions, indexes, provenance, validation |

## Checks

- `git diff --check`: **0 flagged lines**.
- **Nothing under `outputs/` or `archives/` staged.** The faithful A09/A10/A11
  sources are read-only: all ten files in
  `outputs/.../faithful_replication/` hash-match the preflight record and the
  directory still holds exactly ten files.
- **No scientific value changed.** A09/A10/A11 plotting tables in the release
  are **byte-identical copies** of their authoritative sources, and the frames
  compare exactly. No model was refitted, so A11's Ridge α = 1.0, its 80/20
  `random_state=42` split, its `n_repeats=100 random_state=42` permutation and
  its historical signed convention are preserved by construction.
- **All 79 release copies are byte-identical to their sources**, each with
  source path, source hash, release hash and an equality verdict recorded.
- **No source figure moved, overwritten or deleted**; every release copy lives
  under `final_release/`.
- All six prior manifests verify (production 36/36 strict; Sprint-1 37/37,
  morphology 33/33, frequency 37/37 with the two declared living documents
  excluded by policy; error taxonomy 57/57 and feature importance 42/42 strict).
- **Canonical table unchanged**; no `data/` path appears in `git status`.
- **Deterministic regeneration run twice**, 96 files each, 0 differing.
- **Tests**: 52 new, 300 across the behavioral-analysis suites, **523 passed /
  4 deselected** for `pytest tests/ -m "not slow"`.
- **Validation**: 29/29 checks PASS.
- **No SSP artefact exists** anywhere in the release tree.

## Corrections made during this sprint

1. **Plotting tables re-serialized.** The first implementation wrote the three
   legacy tables back through pandas, which reformatted floats: values stayed
   exact but the bytes differed from the tables every earlier validation had
   hashed. They are now byte-identical copies.
2. **Caption placement.** Captions were initially moved away from their
   rendered figures, which broke the release-copy step. They now stay beside
   their figures, as for every other figure in the project.
3. **Three guard artefacts.** Two tests and one validation check flagged
   legitimate text: the axis label "Ridge coefficient" (a name for a stored
   A11 column, not an estimator call), the column name `checkpoint_path` (a
   field of the checkpoint summary, not a load), and the release manifest's
   out_root-dependent path strings. The guards now use AST call/import
   analysis and collapse the out_root prefix. **No scientific code changed.**

## Not in this release

No new statistical analysis, estimator, bootstrap, contrast, analysis set or
uncertainty method; no checkpoint inference, retraining, architecture,
evaluation-code, dataset or canonical-table change; no SSP; no causal mechanism
conclusion; no architectural recommendation; no faithful/adapted pooling; no
composite figure; no overwrite of any prior scientific output.
