# Sprint 1 — commit plan (NOT COMMITTED)

**Base commit**: `e876b755d0475ed11e5fbc0419a0bd8860dfd325` (`feat/full-lexicon-ceiling`, = origin)
**Diff sha256**: `421f69e1597ab9be13c31a39aeb1e91b6ca115351dd91dfcda4aa92faa55a353`
**Files**: 52 new, 1.6 MB · **Tracked files modified**: 0 · Nothing staged, committed or pushed.

## Proposed commit message

```
feat(analysis): version WFE behavioral pipeline and finalize core figures

Promotes the validated behavioral analysis out of gitignored outputs/ into a
tracked, inference-free package and publishes the five core figures.

scripts/behavioral_analysis/ (new, never loads a checkpoint):
  * common.py     frozen constants, reserved lexicality colours, repo-relative
                  paths (no absolute user path anywhere)
  * bootstrap.py  OLS length slope and the frozen hierarchical bootstrap
                  (B = 10,000, random seed 20260730, 95% percentile)
  * compute.py    one writer per plotting table, removing the ordering hazard
                  the original drivers had
  * plotting.py   the five figures; deterministic vector output
  * make_figures / build_canonical_table / validate_outputs /
    close_production_manifest entry points

Numerical non-regression against the validated tables is exact: slopes (24
rows), contrasts (8), length cell means (144), serial-position curves (600),
gate summaries (8) and bootstrap intervals (2) all differ by 0.0.

Figures are published to reports/ as PNG (300 dpi), PDF and SVG beside their
exact plotting table and a standalone caption. Regeneration is byte-identical
across runs in all five formats.

Production manifest closed by splitting immutable scientific outputs (36 files)
from append-only operational logs (5 files); both verify 100%. The original
manifest is retained unmodified and marked HISTORICAL/SUPERSEDED, with the
40/41 history explained rather than erased.

No model, checkpoint, dataset, evaluation code or frozen analysis choice was
changed, and no behavioral result was recomputed or reinterpreted.

Tests: 54 new (tests/test_behavioral_analysis.py), 276 in the existing suite.
```

## Files to stage

```
git add .gitattributes \
        scripts/behavioral_analysis/ \
        tests/test_behavioral_analysis.py \
        docs/behavioral_wfe_fulllexicon.md \
        docs/behavioral_wfe_analysis_matrix.md \
        reports/behavioral_wfe_fulllexicon_93a577f/
```

| Path | Files | Purpose |
|---|---|---|
| `.gitattributes` | 1 | whitespace/binary rules for generated data |
| `scripts/behavioral_analysis/` | 10 | the tracked analysis package |
| `tests/test_behavioral_analysis.py` | 1 | 54 tests |
| `docs/behavioral_wfe_fulllexicon.md`, `docs/behavioral_wfe_analysis_matrix.md` | 2 | method docs and sprint matrix |
| `reports/behavioral_wfe_fulllexicon_93a577f/` | 38 | figures, plotting tables, captions, provenance, validation |

## Files NOT to stage

- `archives/` — 21 MB checkpoint bundle, permanently untracked.
- `outputs/` — gitignored; holds the production predictions, the canonical
  table, the original analysis drivers and the manifest-closure artefacts.
  The `reports/` copies are the publication-ready versions.
- Scratchpad regeneration directories used for the determinism check.
- Anything in the sibling `swp-model` checkout (the Dager repository) — separate repo.

## Verification at plan time

| Check | Result |
|---|---|
| `tests/test_behavioral_analysis.py` | **54 passed**, 0 failed |
| `tests/test_behavioral_eval_patch.py` | **51 passed**, 0 failed |
| Full suite (`-m "not slow"`) | **276 passed**, 0 failed, 4 deselected |
| Structural validation | **PASS**, 31 checks, 0 failures |
| `production_scientific_outputs_FINAL.sha256` | **36/36 OK** |
| `production_operational_logs_FINAL.sha256` | **5/5 OK** |
| Double regeneration | tsv 8/8, svg 5/5, pdf 5/5, png 5/5, captions 5/5 byte-identical |
| Numerical non-regression | max abs diff **0.0** across 786 compared values |
| `git diff --check` | 0 flagged lines |
| Model inference during Sprint 1 | **none** |

## Expected provenance after committing

`behavioral_analysis_provenance.json` currently records
`analysis_code_commit: null`, `analysis_code_dirty: false` (tracked-only) and
the 30 untracked analysis files pinned by `analysis_source_sha256`. After this
commit lands, regenerate the provenance so `analysis_code_commit` names the new
commit and `analysis_code_untracked_files` empties.

## Rollback

```
# remove the new untracked files (nothing tracked was modified)
rm -rf scripts/behavioral_analysis reports/behavioral_wfe_fulllexicon_93a577f
rm -f tests/test_behavioral_analysis.py .gitattributes \
      docs/behavioral_wfe_fulllexicon.md docs/behavioral_wfe_analysis_matrix.md

# after committing, to undo the commit but keep the work:
git reset --soft HEAD~1
# to undo commit and files entirely:
git reset --hard e876b755d0475ed11e5fbc0419a0bd8860dfd325
```

Rollback is complete and local: no checkpoint, dataset, production output or
frozen protocol document is affected, and the gitignored originals under
`outputs/` are untouched throughout.
