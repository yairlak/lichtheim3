# Final release — commit plan (nothing staged, committed or pushed)

Base commit `2edf9f3b54af202b554054f926d8fa9b457e6c3a` (Sprint-5 provenance
closure), branch `feat/full-lexicon-ceiling`, origin equal to HEAD.
**No `git add`, `git commit` or `git push` has been run.**

Diff sha256 of the tracked unstaged changes:
`6cfc333aa77c5f6ffbaf4babcd38b6972871a682ed5e6895fe738b735550da5d`.

## Why two commits

A tracked provenance file cannot contain the SHA of the commit that introduces
it. `final_release_provenance.json` therefore carries
`final_release_analysis_code_commit: null` in Commit A, and Commit B — metadata
only — fills it with the SHA of Commit A. Same pattern as Sprints 1–5:
`f626a69`/`178f6a5`, `96d1626`/`469732d`, `1aa1df8`/`b550580`,
`62ae51b`/`697917d`, `f352de1`/`2edf9f3`.

---

## Commit A — `feat(analysis): finalize WFE behavioral publication release`

### Files to stage

```
scripts/behavioral_analysis/final_release.py
scripts/behavioral_analysis/plot_final_release.py
tests/test_behavioral_final_release.py
docs/behavioral_wfe_fulllexicon.md
docs/behavioral_wfe_analysis_matrix.md
reports/behavioral_wfe_fulllexicon_93a577f/README.md
reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv
reports/behavioral_wfe_fulllexicon_93a577f/final_release/   (all except the two Commit-B paths)
```

Within `final_release/` that means: `README.md`, `final_release_spec.md`,
`executive_summary.md`, `faithful_vs_adapted_summary.md`, `yair_brief.md`,
`robust_findings_and_limitations.md`, `final_figure_selection.md`,
`final_table_selection.md`, `a09_a10_a11_audit.md`,
`final_release_commit_plan.md`, `_control/` (4 JSONs), `figures/main/` (21
files), `figures/supplementary/` (36), `captions/main/` (7),
`captions/supplementary/` (12), `tables/` (6 TSVs) and `formatted_existing/`
(9 figure files, 3 captions, 3 byte-identical plotting tables).

Suggested command (**not executed**):

```bash
git add scripts/behavioral_analysis/final_release.py \
        scripts/behavioral_analysis/plot_final_release.py \
        tests/test_behavioral_final_release.py \
        docs/behavioral_wfe_fulllexicon.md \
        docs/behavioral_wfe_analysis_matrix.md \
        reports/behavioral_wfe_fulllexicon_93a577f/README.md \
        reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv \
        reports/behavioral_wfe_fulllexicon_93a577f/final_release/
git reset -- \
  reports/behavioral_wfe_fulllexicon_93a577f/final_release/final_release_provenance.json \
  reports/behavioral_wfe_fulllexicon_93a577f/final_release/validation/
```

### Proposed message

```
feat(analysis): finalize WFE behavioral publication release

Editorial, formatting, integration and provenance release. It adds no new
scientific value and changes none: every number already exists in a validated
table produced by Sprints 1-5 or by the analysis phase. No checkpoint inference,
no new estimator, bootstrap, contrast or analysis set, no causal claim.

Closes the three formatting-only matrix rows. A09 (faithful Figure 2A), A10
(faithful Figure 2C) and A11 (faithful feature importance) are rendered through
the tracked package directly from their stored authoritative tables, gaining SVG
and standalone captions. No model is refitted, so A11's Ridge alpha = 1.0, its
80/20 random_state=42 split, its n_repeats=100 random_state=42 permutation and
its historical signed convention are preserved by construction; its plotting
table is copied byte-identically. The faithful outputs under outputs/ are
untouched.

Selects 7 main figures - length by route, slopes and the LTM-WM contrast, serial
position, error taxonomy, premature EOS, frequency, adapted feature importance -
and 12 supplementary, each as PNG (300 dpi), PDF and SVG with a standalone
caption. Every release copy is byte-identical to its source and records source
path, source hash, release hash and an equality verdict. No source figure is
moved, overwritten or deleted, and no composite figure is created.

Adds the executive summary, the faithful-versus-adapted summary, the brief for
Yair, the robust-findings-and-limitations table and the figure and table
indexes. The central result is stated without oversimplification: for trained
real words FULL and WM are at exact ceiling and LTM has only a very weak length
slope, while for novel or untrained forms LTM develops a large length effect and
WM stays much more robust. Faithful and adapted analyses are never pooled and
never placed on a common quantitative axis, and the adapted family is not
described as a correction of the faithful one.

A19 (SSP / sonority) remains OPTIONAL_DEFERRED and unstarted. The causal
mechanism analysis stays a separate project; only a factual, non-causal handoff
is shipped.

52 new tests; 523 pass across the suite. Regeneration verified byte-identical
across two independent runs.
```

---

## Commit B — `chore(analysis): close final WFE release provenance`

### Files

```
reports/.../final_release/final_release_provenance.json
reports/.../final_release/validation/final_release_validation.json
reports/.../final_release/validation/final_release_test_log.txt
reports/.../final_release/validation/final_release_output_inventory.tsv
reports/.../final_release/validation/final_release_outputs.sha256
reports/.../final_release/validation/final_release_diff_review.md
```

Commit B must contain **no** analysis source, test code, figure, plotting table,
scientific result table, coefficient or estimate.

### Required edits before staging Commit B

1. Set `final_release_analysis_code_commit` and
   `sprint_commits.final_release.scientific` to the SHA of Commit A.
2. Set `final_release_analysis_code_dirty` to `false` and
   `final_release_analysis_code_untracked_files` to `[]`.
3. Refresh `working_tree_state_at_generation` and `output_sha256`, then
   regenerate `final_release_outputs.sha256` and
   `final_release_output_inventory.tsv` so they cover the final provenance file.

### Proposed message

```
chore(analysis): close final WFE release provenance

Records the final-release code commit and the honest post-commit source state.
Carries the training and evaluation-code commits, the scientific and
provenance-closure commit for all five prior sprints, the four checkpoint
hashes, the five dataset hashes, the canonical-table hash, all seven prior
manifest hashes, the ten faithful A09/A10/A11 source hashes, the frozen
figure-selection spec hash, the figure and table index hashes and every release
output hash.

Adds the validation artefacts: 29/29 checks pass, regeneration is byte-identical
across two independent runs, all 79 release copies match their sources, the
faithful outputs are untouched, and the production, Sprint-1, morphology,
frequency, error-taxonomy and feature-importance manifests verify.

Records SSP as OPTIONAL_DEFERRED and unstarted, and states the separation of the
causal mechanism project.
```

---

## Must not be staged

- `archives/`
- `outputs/` — production predictions, the canonical item table, and the
  faithful A09/A10/A11 sources
- any checkpoint
- the scratchpad provenance and validation generators
- anything in the sibling `swp-model` repository

## Tests passed

| suite | result |
|---|---|
| `tests/test_behavioral_final_release.py` | 52 passed |
| all behavioral-analysis suites | 300 passed |
| `pytest tests/ -m "not slow"` | **523 passed, 4 deselected** |

## Manifests

| manifest | entries | verdict |
|---|---:|---|
| production scientific outputs | 36 | 36/36, strict |
| Sprint-1 outputs | 37 | 37/37, living files excluded |
| morphology outputs | 33 | 33/33, living files excluded |
| frequency outputs | 37 | 37/37, living files excluded |
| error-taxonomy outputs | 57 | 57/57, strict |
| feature-importance outputs | 42 | 42/42, strict |
| final-release outputs | 111 | written by this release |

## Rollback procedure

Before any commit: `git checkout -- docs/ reports/behavioral_wfe_fulllexicon_93a577f/README.md
reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv` restores the four
living documents, and `rm -rf reports/behavioral_wfe_fulllexicon_93a577f/final_release
scripts/behavioral_analysis/final_release.py
scripts/behavioral_analysis/plot_final_release.py
tests/test_behavioral_final_release.py` removes every release artefact. Nothing
else is touched, because no release command writes outside those paths — in
particular the faithful sources under `outputs/` are read-only.

After Commit A but before Commit B, or after both:
`git reset --hard 2edf9f3b54af202b554054f926d8fa9b457e6c3a` returns the branch
to the Sprint-5 closure, since nothing has been pushed. If the branch has been
pushed by then, revert with two `git revert` commits in reverse order instead.

## Current state

```
$ git status --porcelain
 M docs/behavioral_wfe_analysis_matrix.md
 M docs/behavioral_wfe_fulllexicon.md
 M reports/behavioral_wfe_fulllexicon_93a577f/README.md
 M reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv
?? archives/
?? reports/behavioral_wfe_fulllexicon_93a577f/final_release/
?? scripts/behavioral_analysis/final_release.py
?? scripts/behavioral_analysis/plot_final_release.py
?? tests/test_behavioral_final_release.py
```

Nothing is staged. `archives/` is untracked and stays that way.
