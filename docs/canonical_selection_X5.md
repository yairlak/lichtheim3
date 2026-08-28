# Canonical checkpoint selection and cohort policy

The repository contains analyses produced under two different checkpoint
selection criteria. Both are legitimate and neither is being rewritten. This
document states which cohort each result family uses, so that no analysis is
mislabelled as more (or less) canonical than it is.

## 1. Current canonical model-selection policy

**X = 5 consecutive zero-whole-word-error FULL evaluations** on the full
29,571-word training lexicon. The selected checkpoint is the *first* checkpoint
of the qualifying streak.

Canonical checkpoints:

| Seed | Epoch | SHA-256 |
|---|---|---|
| 19 | 155 | `7d05f9c2ad5a53e705f7d55ccde2581754918938d8ca888da35c0a859666478e` |
| 22 | 140 | `a15846cbf3c7df88ed289512bbb20cbefd2121d0deec1b39f363932a743da595` |

**Raising the criterion did not move any checkpoint.** Observed zero streaks in
cohort `93a577f` were: seed 19 = 6 (epochs 155–180), seed 20 = 2 (130–135),
seed 21 = 0, seed 22 = 13 (140–200). Seeds 19 and 22 therefore select epochs 155
and 140 at every value of X; what changes is **cohort membership**, not the
selected epochs.

Evidence:
- `reports/fulllexicon_cohort_93a577f/selected_checkpoints.tsv`
  (branch `feat/full-lexicon-ceiling`) — per-seed streaks, error counts, hashes
  and selection reasons.
- `reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/stable_zero_audit/`
  — `stable_zero_streaks.tsv`, `stable_zero_verdicts.tsv`,
  `stable_zero_trajectory.tsv`, `stable_zero_cross_check.tsv`.
- `reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/meeting_figures/mf2_stable_zero_bottom_line_caption.md`
  — states the outcome for X = 2, 3 and 5 and that raising X would not have
  changed a single selected checkpoint in this cohort.
- `configs/canonical_93a577f.yaml` (this branch) — `canonical_selection_policy`,
  `canonical_checkpoints`, `historical_noncanonical_checkpoints`.

## 2. Historical four-checkpoint analysis cohort

Cohort `93a577f` was created and archived under the earlier criterion **X = 2**,
under which three seeds qualified (19, 20, 22) and seed 21 was selected by the
fallback rule (earliest checkpoint with the minimum error count, 1 error at
epoch 145).

Analyses produced before the X=5 restriction use all four selected checkpoints:

| Seed | Epoch | Status under X = 5 |
|---|---|---|
| 19 | 155 | canonical |
| 20 | 130 | **not canonical** — zero streak of 2 |
| 21 | 145 | **not canonical** — never reached zero errors |
| 22 | 140 | canonical |

**The established behavioral WFE release uses this historical four-checkpoint
cohort.** Its per-seed claims are stated as "4/4 seeds" and its checkpoint table
lists all four with role `primary`. See
`reports/behavioral_wfe_fulllexicon_93a577f/final_release/tables/checkpoint_summary.tsv`
and `.../final_figure_index.tsv` (branch `feat/full-lexicon-ceiling`).

The correct description of that cohort is a **historical multi-seed robustness /
analysis cohort**: four independently trained checkpoints used to check that an
effect holds across training runs. It is *not* a claim that all four are
currently canonical models.

**These analyses must not be relabelled as if all four checkpoints were
canonical, and they must not be described as having been rerun on seeds 19 and
22 only. They were not rerun.** The archived bundle manifest, which records the
X=2 outcome, is historical and is not rewritten.

## 3. X=5 representational cohort

The ventral-semantic and RSA analyses use only the two X=5 canonical
checkpoints. Both result summaries state this explicitly in their first lines
("Frozen cohort: the **X=5 stable-zero** checkpoints only — seed 19 / epoch 155
and seed 22 / epoch 140"):

- `reports/ventral_semantic_93a577f/RESULTS_SUMMARY.md` (Phase B — semantic
  identification and GloVe-fitted PCA)
- `reports/ventral_semantic_93a577f/RESULTS_SUMMARY_PHASE_C.md` (Phase C —
  dorsal PCA and the exact full-lexicon RSA)

both on branch `feat/ventral-semantic-probe`.

## 4. Warm-start acquisition cohort

Naming/comprehension acquisition (Phase 2) and the warm-start multitask
experiments (Phase 3) principally use **seed 22 / epoch 140** as the single
canonical source checkpoint.

Every retained run records the source checkpoint path, its SHA-256
`a15846cb…`, and the training commit `93a577f`, independently, under
`provenance` in its own `run_summary.json`. See
`reports/naming_comprehension_93a577f/runs/*/run_summary.json` and the
per-artifact rows in `reports/naming_comprehension_93a577f/CURATION_MANIFEST.tsv`
(this branch).

**The frozen naming/comprehension baseline is different and uses the historical
four selected checkpoints.** `cohort_summary.json` records `n_checkpoints: 4`
and `"cohort": "fulllexicon_93a577f_seeds19_22"`, with statistics explicitly
labelled descriptive only ("no inferential statistics with n=4"). The frozen
starting-point figure is likewise a mean over four checkpoints. See
`reports/naming_comprehension_93a577f/frozen_baseline/cohort/cohort_summary.json`,
`.../cohort/cohort_by_seed.tsv`, and the `fig_backup_frozen_starting_point`
entry in `reports/naming_comprehension_93a577f/figures/summary_plots/FIGURE_SUMMARY.md`.

That baseline must be labelled as a historical four-checkpoint aggregate, not as
an X=5 canonical result.

## 5. Phase 4 — joint multitask from scratch

Phase 4 **does not warm-start from any canonical checkpoint**. Both arms (H0 and
J0) initialise randomly at a fixed experimental seed. The driver states this
directly: the historical checkpoints seed19/e155 and seed22/e140 remain the
canonical historical *maturity* references and are explicitly **not** the paired
control for J0 — a fresh H0 at the same seed is.

See `scripts/naming_comprehension/train_joint_scratch.py` (module docstring and
the frozen Phase 4 configuration block).

**Phase 4 is ACTIVE / NOT RELEASE-CANONICAL.** No Phase 4 result is included in
any manifest, figure inventory or release in this repository, and none should be
until validated results exist and are centrally approved.

## 6. Summary table

| Result family | Cohort used | Canonical under X=5? | Correct label |
|---|---|---|---|
| Stable-zero model selection (mf2) | 4 seeds, all evaluations | describes the policy itself | model-selection evidence |
| Behavioral WFE release (F1–F7, S1–S12) | 19/155, 20/130, 21/145, 22/140 | **no** — 2 of 4 are non-canonical | historical multi-seed robustness / analysis cohort |
| Ventral semantic + RSA | 19/155, 22/140 | **yes** | X=5 representational cohort |
| Frozen naming/comprehension baseline | 19/155, 20/130, 21/145, 22/140 | **no** | historical four-checkpoint aggregate, descriptive only |
| Naming/comprehension acquisition (Phase 2) | 22/140 | **yes** (single canonical source) | warm-start acquisition cohort |
| Warm-start multitask (Phase 3A/3B/3C) | 22/140 | **yes** (single canonical source) | warm-start acquisition cohort |
| Joint multitask from scratch (Phase 4) | none — random initialisation | n/a | ACTIVE / NOT RELEASE-CANONICAL |

## 7. What this policy does not do

It does not invalidate any existing analysis. A four-checkpoint robustness check
remains a four-checkpoint robustness check; the X=5 restriction concerns which
models are presented as *the* canonical Lichtheim3 repetition models, not which
checkpoints may be used to test whether an effect replicates across training
runs.

Nothing here was recomputed. No analysis was rerun and no figure regenerated to
produce this document.
