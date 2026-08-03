# Sprint 5 — commit plan (nothing staged, committed or pushed)

Base commit `697917d5141769bf02d08ba853b3c767e18e6370` (Sprint-4 provenance
closure), branch `feat/full-lexicon-ceiling`, origin equal to HEAD at the start
of this sprint. **No `git add`, `git commit` or `git push` has been run.**

Diff sha256 of the tracked unstaged changes:
`9e3dfe428ecfe0aef24025b96f5127507c84635f7ca5ee9bb585846b6cae658c`
(`git diff | shasum -a 256`).

## Why two commits

A tracked provenance file cannot contain the SHA of the commit that introduces
it. `feature_importance_provenance.json` therefore carries
`adapted_fi_analysis_code_commit: null` in Commit A, and Commit B — provenance
and validation metadata only — fills it with the SHA of Commit A. Same pattern
as Sprints 1–4 (`f626a69`/`178f6a5`, `96d1626`/`469732d`, `1aa1df8`/`b550580`,
`62ae51b`/`697917d`).

---

## Commit A — `feat(analysis): add adapted WFE feature-importance analyses`

### Exact files to stage

Tracked source additions:

```
scripts/behavioral_analysis/feature_importance.py
scripts/behavioral_analysis/plot_feature_importance.py
tests/test_behavioral_feature_importance.py
```

Declared living-document updates:

```
docs/behavioral_wfe_fulllexicon.md
docs/behavioral_wfe_analysis_matrix.md
reports/behavioral_wfe_fulllexicon_93a577f/README.md
reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv
```

Report tree — everything under
`reports/behavioral_wfe_fulllexicon_93a577f/feature_importance/` **except** the
files listed under Commit B:

```
feature_importance/feature_importance_analysis_spec.md
feature_importance/feature_importance_results.md
feature_importance/faithful_vs_adapted.md
feature_importance/feature_importance_commit_plan.md
feature_importance/_control/feature_importance_preflight.json
feature_importance/_control/feature_importance_analysis_spec.json
feature_importance/_control/feature_importance_output_manifest.json
feature_importance/_control/fi_train_items.tsv
feature_importance/_control/fi_test_items.tsv
feature_importance/clean_joint/figures/      (1 figure x png/pdf/svg + caption)
feature_importance/clean_joint/tables/       (8 TSVs)
feature_importance/clean_interactions/figures/ (1 figure x png/pdf/svg + caption)
feature_importance/clean_interactions/tables/  (6 TSVs)
feature_importance/route_specific/figures/   (1 figure x png/pdf/svg + caption)
feature_importance/route_specific/tables/    (6 TSVs)
```

Suggested command (**not executed**):

```bash
git add scripts/behavioral_analysis/feature_importance.py \
        scripts/behavioral_analysis/plot_feature_importance.py \
        tests/test_behavioral_feature_importance.py \
        docs/behavioral_wfe_fulllexicon.md \
        docs/behavioral_wfe_analysis_matrix.md \
        reports/behavioral_wfe_fulllexicon_93a577f/README.md \
        reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv \
        reports/behavioral_wfe_fulllexicon_93a577f/feature_importance/
git reset -- \
  reports/behavioral_wfe_fulllexicon_93a577f/feature_importance/feature_importance_provenance.json \
  reports/behavioral_wfe_fulllexicon_93a577f/feature_importance/validation/
```

### Proposed message

```
feat(analysis): add adapted WFE feature-importance analyses

Sprint 5 of the finalization phase: the adapted feature-importance analysis
(A15) on LICHTHEIM_CLEAN across all three routes, computed from the canonical
seed x item x route table alone. No checkpoint is loaded and no model inference
runs; the only models fitted are the Ridge regressions in the frozen spec.

Three analyses: a clean joint main-effects model (primary), a model adding five
predeclared two-way interaction blocks, and route-specific models. The faithful
Dager analysis (A11) is a separate analysis and is not recomputed, replaced or
pooled; its values are never placed on one axis with these.

Two identifiability constraints are fixed in advance. On the clean set
lexicality and training exposure are perfectly confounded - every Real item is
TRAINED_REAL_EXACT and every Pseudo item is NOVEL_PSEUDOWORD - so the two never
enter one model and the factor is reported as a lexicality/exposure contrast; no
claim separates them. Zipf frequency is undefined for pseudowords, is never
imputed, and is excluded from every all-item clean model.

Two design rules do the methodological work. The 80/20 split is grouped by
item_id, so all three route rows of an item stay together and the identical
split is reused across all four seeds and all three models. Permutation acts on
raw factors rather than model columns: item factors are permuted across held-out
items with one value applied to all three route rows, route labels are permuted
within an item preserving one FULL/WM/LTM row each, and encoding,
standardization and every interaction term are rebuilt afterwards.

Results (descriptive, four seeds, none excluded): held-out R2 is positive in all
four seeds (mean 0.095). Route and lexicality/exposure lead and are not
separated by these data; length is third and morphology last in every seed, with
morphology effectively zero. The interaction model improves held-out R2 and MAE
in all four seeds, route x lexicality being the strongest block. Only the LTM
route is estimable route-specific, where lexicality/exposure > length >
morphology in all four seeds; FULL and WM are ceiling-limited and are labelled,
never given an artificial zero.

Grouped importance is unsigned; coefficients are reported separately with route
as two contrasts. Held-out MAE accompanies R2 throughout because the outcome is
zero-heavy, negative held-out R2 is retained and flagged, and the four-checkpoint
interval is labelled a seed-resampling interval, not a hierarchical bootstrap.

59 new tests; 471 pass across the suite. Deterministic regeneration verified
twice, byte-identical.
```

---

## Commit B — `chore(analysis): close WFE feature-importance provenance`

### Files

```
reports/.../feature_importance/feature_importance_provenance.json
reports/.../feature_importance/validation/feature_importance_validation.json
reports/.../feature_importance/validation/feature_importance_test_log.txt
reports/.../feature_importance/validation/feature_importance_output_inventory.tsv
reports/.../feature_importance/validation/feature_importance_outputs.sha256
reports/.../feature_importance/validation/feature_importance_diff_review.md
```

Commit B must contain **no** analysis source, test code, figure, plotting TSV,
scientific result table or scientific estimate.

### Required edits before staging Commit B

1. Set `adapted_fi_analysis_code_commit` to the SHA of Commit A.
2. Set `adapted_fi_analysis_code_dirty` to `false` and
   `adapted_fi_analysis_code_untracked_files` to `[]` — both become true
   statements once Commit A exists.
3. Refresh `working_tree_state_at_generation` and `output_sha256`, then
   regenerate `feature_importance_outputs.sha256` and
   `feature_importance_output_inventory.tsv` so they cover the final provenance
   file.

### Proposed message

```
chore(analysis): close WFE feature-importance provenance

Records the Sprint-5 analysis-code commit and the honest post-commit source
state. Carries the training, evaluation-code, Sprint-1, morphology, frequency
and error-taxonomy commits, the checkpoint and dataset hashes, the canonical
table hash, the clean-set definition, the identifiability constraints, the three
model formulas, the frozen reference levels, the grouped-split item hashes,
Ridge alpha, both random states, the permutation repeat count, package versions
and every output hash.

Adds the validation artefacts: 29/29 checks pass, regeneration is byte-identical
across two independent runs, the faithful A11 outputs are untouched, and the
production, Sprint-1, morphology, frequency and error-taxonomy manifests verify
(the two declared living documents excepted, by the documented living-file
policy).
```

---

## Must not be staged

- `archives/`
- `outputs/` — every production prediction, the canonical item table, and the
  faithful A11 directory
- any checkpoint
- the scratchpad provenance and validation generators, which live outside the
  repository on purpose
- anything in the sibling `swp-model` repository

## Tests passed

| suite | result |
|---|---|
| `tests/test_behavioral_feature_importance.py` | 59 passed |
| all behavioral-analysis suites | 241 passed |
| `pytest tests/ -m "not slow"` | **471 passed, 4 deselected** |

## Manifests

| manifest | entries | verdict |
|---|---:|---|
| production scientific outputs | 36 | 36/36, strict |
| Sprint-1 outputs | 37 | 37/37, living files excluded |
| morphology outputs | 33 | 33/33, living files excluded |
| frequency outputs | 37 | 37/37, living files excluded |
| error-taxonomy outputs | 57 | 57/57, strict |
| feature-importance outputs | 41 | written by this sprint |

## Rollback procedure

Before any commit: `git checkout -- docs/ reports/behavioral_wfe_fulllexicon_93a577f/README.md
reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv` restores the four
living documents, and `rm -rf reports/behavioral_wfe_fulllexicon_93a577f/feature_importance
scripts/behavioral_analysis/feature_importance.py
scripts/behavioral_analysis/plot_feature_importance.py
tests/test_behavioral_feature_importance.py` removes every Sprint-5 artefact.
Nothing else is touched, because no Sprint-5 command writes outside those paths.

After Commit A but before Commit B: `git reset --hard 697917d5141769bf02d08ba853b3c767e18e6370`
returns the branch to the Sprint-4 closure, since nothing has been pushed.

After both commits: `git reset --hard 697917d…` remains sufficient for the same
reason. If the branch has been pushed by then, revert with two `git revert`
commits in reverse order instead.

## Current state

```
$ git status --porcelain
 M docs/behavioral_wfe_analysis_matrix.md
 M docs/behavioral_wfe_fulllexicon.md
 M reports/behavioral_wfe_fulllexicon_93a577f/README.md
 M reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv
?? archives/
?? reports/behavioral_wfe_fulllexicon_93a577f/feature_importance/
?? scripts/behavioral_analysis/feature_importance.py
?? scripts/behavioral_analysis/plot_feature_importance.py
?? tests/test_behavioral_feature_importance.py
```

Nothing is staged. `archives/` is untracked and stays that way.
