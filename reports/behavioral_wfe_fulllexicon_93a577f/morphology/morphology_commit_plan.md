# Sprint 2 — commit plan (NOT COMMITTED)

**Base commit**: `178f6a55428100a04bcea11966b653f5f444877e` (`feat/full-lexicon-ceiling`, = origin)
**Diff sha256**: `095e181985dfe1f575c34f004a84a8b86a8a10570733b546cc13eb3970facdb0`
**Files**: 41 (3 new source, 4 modified docs/matrix, 34 new report artefacts) · 1.05 MB
Nothing staged, committed or pushed.

## Proposed commit message

```
feat(analysis): add faithful and clean WFE morphology analyses

Adds the Sprint-2 morphology x phoneme-length analysis, specified and frozen
before any morphology result was inspected
(reports/.../morphology/morphology_analysis_spec.md).

scripts/behavioral_analysis/morphology.py (new):
  * descriptive cells, morphology contrast = mean(simple) - mean(complex),
    morphology x length interaction = simple_slope - complex_slope,
    route contrasts, and across-seed summaries with exact-zero sensitivity
  * frozen small-cell flags (VERY_SMALL_CELL n<10, SMALL_CELL 10<=n<20);
    no cell is ever excluded
  * reuses the Sprint-1 hierarchical bootstrap unchanged
    (B = 10,000, random seed 20260730, 95% percentile)

scripts/behavioral_analysis/plot_morphology.py (new):
  * faithful Figure-2A-style replication (1,200 items, FULL route)
  * clean-set morphology by route (2 lexicality rows x 3 route columns)
  * morphology encoded by line style only - complex solid, simple dashed -
    because red and blue stay reserved for lexicality

Results: no morphology contrast or morphology x length interaction is
distinguishable from zero in either analysis. Every estimable interval spans
zero and no quantity holds its sign across the four seeds. Real words under
FULL and WM are at exact ceiling (zero errors), so their contrasts are
structurally zero and no absence is claimed there. Two clean cells are flagged
SMALL_CELL and retained. No word-error figure was produced
(FIGURE_NOT_CREATED_DUE_TO_CEILING); word-error tables are provided in full.

No Sprint-1 scientific artefact changed: all 35 non-living manifest entries
verify byte-identical. README.md and analysis_matrix.tsv are living documents
and were extended by design.

No checkpoint inference, no model/evaluation/dataset change, no change to any
analysis set, seed policy or bootstrap parameter.

Tests: 29 new (tests/test_behavioral_morphology.py), 305 in the full suite.
```

## Files to stage

```
git add scripts/behavioral_analysis/morphology.py \
        scripts/behavioral_analysis/plot_morphology.py \
        tests/test_behavioral_morphology.py \
        docs/behavioral_wfe_fulllexicon.md \
        docs/behavioral_wfe_analysis_matrix.md \
        reports/behavioral_wfe_fulllexicon_93a577f/README.md \
        reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv \
        reports/behavioral_wfe_fulllexicon_93a577f/morphology/
```

## Files NOT to stage

- `archives/` — permanently untracked checkpoint bundle.
- `outputs/` — gitignored production predictions, canonical table and drivers.
- Scratchpad regeneration directories used for the determinism check.
- Any Sprint-1 figure or plotting table (unchanged and already committed).

## Verification at plan time

| Check | Result |
|---|---|
| `tests/test_behavioral_morphology.py` | **29 passed**, 0 failed |
| `tests/test_behavioral_analysis.py` | **54 passed**, 0 failed |
| Full suite (`-m "not slow"`) | **305 passed**, 0 failed, 4 deselected |
| Deterministic double regeneration | 24/24 files byte-identical |
| Published outputs vs fresh regeneration | 24/24 identical |
| Sprint-1 scientific outputs | 35/35 byte-identical |
| Production manifest | 36/36 verify |
| Morphology output manifest | 31/31 verify |
| PNG resolution | 300 dpi; PDF and SVG present |
| `git diff --check` | 0 flagged lines |
| Model inference | none |

## Output hashes

`morphology/validation/morphology_outputs.sha256` (31 files) and
`morphology/validation/morphology_output_inventory.tsv`.

## Planned two-commit provenance closure

`morphology_provenance.json` currently records
`morphology_analysis_code_commit: null` with the implementation identified by
`morphology_source_sha256` plus the tracked-modification list. A tracked file
cannot contain the SHA of the commit that contains it, so if a commit-level
identifier is wanted, follow the Sprint-1 pattern: commit the analysis and
report first, then regenerate the provenance and close it in a second commit
(`chore(analysis): close WFE Sprint 2 provenance`).

## Rollback

```
# discard the four tracked-document modifications
git checkout -- docs/behavioral_wfe_fulllexicon.md \
                docs/behavioral_wfe_analysis_matrix.md \
                reports/behavioral_wfe_fulllexicon_93a577f/README.md \
                reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv
# remove the new files
rm -f scripts/behavioral_analysis/morphology.py \
      scripts/behavioral_analysis/plot_morphology.py \
      tests/test_behavioral_morphology.py
rm -rf reports/behavioral_wfe_fulllexicon_93a577f/morphology

# after committing, to undo but keep the work:
git reset --soft HEAD~1
# to undo entirely:
git reset --hard 178f6a55428100a04bcea11966b653f5f444877e
```

No checkpoint, dataset, production output or Sprint-1 scientific artefact is
affected by rollback.
