# Sprint 3 — commit plan (NOT COMMITTED)

**Base commit**: `469732d173142f9a9062ad536c9043d6d22d7c32` (`feat/full-lexicon-ceiling`, = origin)
**Diff sha256**: `ebdeb6279ba9137e637dfcc6ac5c78f89a763fd8b013aab9e3b5984ee9a89b12`
**Files**: 42 (2 new source, 1 new test, 4 modified docs/matrix, 35 new report artefacts) · 0.63 MB
Nothing staged, committed or pushed.

## Proposed commit message

```
feat(analysis): add trained-word WFE frequency analyses

Adds the Sprint-3 word-frequency analysis, specified and frozen before any
frequency result was inspected
(reports/.../frequency/frequency_analysis_spec.md).

scripts/behavioral_analysis/frequency.py (new):
  * continuous Zipf slope, length-adjusted model with a centred interaction,
    high/low descriptive contrast, route contrasts in both orientations,
    and confidence/gate slopes
  * one standardization anchor per dataset regime, reused unchanged across
    every seed and route; frequency-class labels recomputed from Zipf rather
    than trusted (0 mismatches, 0 items in the excluded 3.5-4.0 gap)
  * frozen ceiling policy: zero-error cells are ALL_ZERO_OUTCOME, no logistic
    fit is attempted, and no absence of frequency encoding may be claimed
  * reuses the Sprint-1 hierarchical bootstrap unchanged
    (B = 10,000, random seed 20260730, 95% percentile)

scripts/behavioral_analysis/plot_frequency.py (new):
  * distribution and confound audit, primary route figure with the secondary
    high/low panel, and a separate confidence/gate figure
  * neutral colour scale; red and blue stay reserved for lexicality

Results on trained real words (n = 671): the LTM Zipf slope is negative in all
four seeds (mean -0.0130, CI [-0.0288, -0.0027]) - higher-frequency trained
words are reproduced with fewer errors by the ventral route. FULL and WM
produce zero errors on every item in every seed (ALL_ZERO_OUTCOME): their
slopes are structurally zero and the measure cannot identify an effect there,
which is not evidence that those routes lack frequency information. Lexical
confidence rises with Zipf (+0.0914, CI [+0.0866, +0.0962]) and the gate
follows (+0.0450) as a linked monotonic transform, not independent evidence.
Conclusions are unchanged by adding the 7 pronunciation variants and by
restricting to the exact-zero seeds. Untrained real words are reported
separately and remain descriptive only (15 high vs 107 low; frequency
confounded with exposure).

No word-error figure was produced
(FIGURE_NOT_CREATED_DUE_TO_CEILING_OR_SPARSE_ERRORS).

No Sprint-1 or Sprint-2 scientific artefact changed. No checkpoint inference,
no model/evaluation/dataset change, no change to any analysis set, seed policy
or bootstrap parameter, and no pseudoword was assigned a frequency.

Tests: 40 new (tests/test_behavioral_frequency.py), 345 in the full suite.
```

## Files to stage

```
git add scripts/behavioral_analysis/frequency.py \
        scripts/behavioral_analysis/plot_frequency.py \
        tests/test_behavioral_frequency.py \
        docs/behavioral_wfe_fulllexicon.md \
        docs/behavioral_wfe_analysis_matrix.md \
        reports/behavioral_wfe_fulllexicon_93a577f/README.md \
        reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv \
        reports/behavioral_wfe_fulllexicon_93a577f/frequency/
```

## Files NOT to stage

- `archives/` — permanently untracked checkpoint bundle.
- `outputs/` — gitignored production predictions, canonical table, drivers.
- Scratchpad regeneration directories used for the determinism check.
- Any Sprint-1 or Sprint-2 artefact (unchanged and already committed).

## Verification at plan time

| Check | Result |
|---|---|
| `tests/test_behavioral_frequency.py` | **40 passed**, 0 failed |
| Morphology + Sprint-1 tests | **83 passed**, 0 failed |
| Full suite (`-m "not slow"`) | **345 passed**, 0 failed, 4 deselected |
| Deterministic double regeneration | 28/28 byte-identical |
| Published vs fresh regeneration | 28/28 identical |
| Sprint-1 manifest | verifies apart from the two living documents |
| Morphology manifest | 33/33 verify |
| Production manifest | 36/36 verify |
| Frequency output manifest | see `frequency_outputs.sha256` |
| PNG resolution | 300 dpi; PDF and SVG present |
| `git diff --check` | 0 flagged lines |
| Model inference | none |

## Planned two-commit provenance closure

`frequency_provenance.json` records `frequency_analysis_code_commit: null`,
with the implementation identified by `frequency_source_sha256` plus the
tracked-modification and untracked lists. A tracked file cannot contain the SHA
of the commit that contains it, so follow the Sprint-1/2 pattern: commit the
analysis and report first, then regenerate the provenance and close it in a
second commit (`chore(analysis): close WFE frequency provenance`).

## Rollback

```
git checkout -- docs/behavioral_wfe_fulllexicon.md \
                docs/behavioral_wfe_analysis_matrix.md \
                reports/behavioral_wfe_fulllexicon_93a577f/README.md \
                reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv
rm -f scripts/behavioral_analysis/frequency.py \
      scripts/behavioral_analysis/plot_frequency.py \
      tests/test_behavioral_frequency.py
rm -rf reports/behavioral_wfe_fulllexicon_93a577f/frequency

# after committing, to undo but keep the work:
git reset --soft HEAD~1
# to undo entirely:
git reset --hard 469732d173142f9a9062ad536c9043d6d22d7c32
```

No checkpoint, dataset, production output or prior-sprint artefact is affected
by rollback.
