# Sprint 4 — commit plan (nothing staged, committed or pushed)

Base commit `b550580455046a8c420e4a62cefeeb435815804b`. This document describes
the intended closure; **no `git add`, `git commit` or `git push` has been run.**

## Why two commits

A tracked provenance file cannot contain the SHA of the commit that introduces
it. `error_taxonomy_provenance.json` therefore carries
`error_taxonomy_analysis_code_commit: null` in Commit A, and Commit B — which
contains provenance and validation metadata only — fills that field with the
SHA of Commit A. This is the same pattern used to close Sprints 1, 2 and 3
(`f626a69`/`178f6a5`, `96d1626`/`469732d`, `1aa1df8`/`b550580`).

---

## Commit A — `feat(analysis): add WFE error taxonomy and EOS diagnostics`

### Tracked source additions

```
scripts/behavioral_analysis/error_taxonomy.py
scripts/behavioral_analysis/eos_diagnostics.py
scripts/behavioral_analysis/plot_error_taxonomy.py
tests/test_behavioral_error_taxonomy.py
```

### Declared living-document updates

```
docs/behavioral_wfe_fulllexicon.md
docs/behavioral_wfe_analysis_matrix.md
reports/behavioral_wfe_fulllexicon_93a577f/README.md
reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv
```

### Report tree

Everything under `reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/`
**except** the two files listed under Commit B:

```
error_taxonomy/eos_convention.md
error_taxonomy/error_taxonomy_analysis_spec.md
error_taxonomy/error_taxonomy_results.md
error_taxonomy/length_effect_mechanism_handoff.md
error_taxonomy/error_taxonomy_commit_plan.md
error_taxonomy/_control/eos_convention.json
error_taxonomy/_control/error_taxonomy_analysis_spec.json
error_taxonomy/_control/error_taxonomy_preflight.json
error_taxonomy/_control/error_taxonomy_output_manifest.json
error_taxonomy/_control/error_taxonomy_reference_values.json
error_taxonomy/faithful/figures/   (1 figure × png/pdf/svg + caption)
error_taxonomy/faithful/tables/    (4 TSVs)
error_taxonomy/clean/figures/      (2 figures × png/pdf/svg + captions)
error_taxonomy/clean/tables/       (8 TSVs)
error_taxonomy/eos/figures/        (1 figure × png/pdf/svg + caption)
error_taxonomy/eos/tables/         (10 TSVs)
error_taxonomy/strata/tables/      (5 TSVs)
error_taxonomy/examples/           (README.md + 2 TSVs)
```

Suggested command (**not executed**):

```bash
git add scripts/behavioral_analysis/error_taxonomy.py \
        scripts/behavioral_analysis/eos_diagnostics.py \
        scripts/behavioral_analysis/plot_error_taxonomy.py \
        tests/test_behavioral_error_taxonomy.py \
        docs/behavioral_wfe_fulllexicon.md \
        docs/behavioral_wfe_analysis_matrix.md \
        reports/behavioral_wfe_fulllexicon_93a577f/README.md \
        reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv \
        reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/
git reset -- reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/error_taxonomy_provenance.json \
             reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/validation/
```

### Proposed message

```
feat(analysis): add WFE error taxonomy and EOS diagnostics

Sprint 4 of the finalization phase: the Levenshtein error taxonomy (A16, A17)
and the premature-EOS decoder diagnostic (A18), computed from the canonical
seed x item x route table alone. No checkpoint is loaded and no inference runs.

Two analyses, kept apart at every level. Operations are exactly substitution,
deletion and insertion, read from the counts produced by Levenshtein.editops
0.27.3 during the production evaluation; nothing re-aligns a sequence and no
fourth operation is introduced. Premature EOS lives in its own module, tables
and claims: a deletion is not automatically a premature EOS, a premature EOS is
not one deletion, several deletions may follow one early stop, and early stops
may coexist with substitutions or insertions.

The EOS indexing convention was audited from the committed evaluator before any
EOS distribution was read. The readout window holds exactly L tokens at indices
0..L-1, so an EOS at the correct boundary (index L) falls outside it. Only
PREMATURE_EOS is positively observable; ON_TIME_EOS and LATE_EOS are
structurally unobservable; EOS_NOT_OBSERVED means no EOS was observed within the
instrumented evaluation horizon and is ambiguous with respect to eventual
stopping. The frozen class labels are unchanged - only their observability is
clarified.

Results (descriptive, four seeds, none excluded): the LTM route carries a much
larger pseudoword operation burden than FULL or WM, substitutions dominating,
then deletions, then insertions, strongest for Long items and with
non-overlapping seed ranges; trained real words are at or near floor. 87
premature-EOS events occur in total (LTM 82, FULL 3, WM 2), all on pseudowords
and none on trained real words, with a positive LTM length slope in all four
seeds. Premature EOS accounts for only a subset of erroneous behaviour and no
causal claim is made in either direction.

The clean taxonomy figure keeps one common absolute y-scale across routes. The
frozen >10x zoom rule evaluated true (ratio 22.95), so a labelled FULL/WM
companion was added; it does not replace the primary figure.

Specification frozen before any summary was computed. Bootstrap unchanged:
B = 10,000, random seed 20260730, 95% percentile. Deterministic regeneration is
byte-identical across runs (46/46 files).

67 new tests; 412 pass across the suite.
```

---

## Commit B — `chore(analysis): close WFE error-taxonomy provenance`

### Files

```
reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/error_taxonomy_provenance.json
reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/validation/error_taxonomy_validation.json
reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/validation/error_taxonomy_test_log.txt
reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/validation/error_taxonomy_output_inventory.tsv
reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/validation/error_taxonomy_outputs.sha256
reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/validation/error_taxonomy_diff_review.md
```

### Required edits before staging Commit B

1. Set `error_taxonomy_analysis_code_commit` to the SHA of Commit A.
2. Refresh `working_tree_state_at_generation` — after Commit A the four living
   documents and the source files are tracked and clean, so `modified_tracked`
   and `untracked` shrink and `uncommitted_diff_sha256` becomes the hash of an
   empty diff.
3. Regenerate `error_taxonomy_outputs.sha256` and
   `error_taxonomy_output_inventory.tsv` so they cover the final
   `error_taxonomy_provenance.json`.

### Proposed message

```
chore(analysis): close WFE error-taxonomy provenance

Records the Sprint-4 analysis-code commit, the audited EOS convention and its
observability limits, the Levenshtein.editops backend and version, the frozen
zoom-rule threshold with its observed decision, the canonical-table hash, the
bootstrap configuration and the analysis sets. Adds the validation artefacts:
26/26 checks pass, deterministic regeneration is byte-identical, and the
Sprint-1/2/3 and production manifests verify (the two declared living documents
excepted, by the documented living-file policy).
```

---

## Must not be staged

- `archives/`
- `outputs/` — including every production prediction and the canonical item
  table
- any checkpoint
- unrelated scratch files (the provenance and validation generators live in the
  session scratchpad, outside the repository, on purpose)

## Current state

```
$ git status --porcelain
 M docs/behavioral_wfe_analysis_matrix.md
 M docs/behavioral_wfe_fulllexicon.md
 M reports/behavioral_wfe_fulllexicon_93a577f/README.md
 M reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv
?? archives/
?? reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/
?? scripts/behavioral_analysis/eos_diagnostics.py
?? scripts/behavioral_analysis/error_taxonomy.py
?? scripts/behavioral_analysis/plot_error_taxonomy.py
?? tests/test_behavioral_error_taxonomy.py
```

Nothing is staged. `archives/` is untracked and stays that way.
