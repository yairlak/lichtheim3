# Naming, comprehension and warm-start multitask — curated evidence

Curated, tracked evidence for the naming/comprehension and warm-start multitask
experiments. Nothing here was recomputed: every file is a byte-identical copy of
an existing output artifact, and the figures are the already-generated ones.
No experiment was rerun and no figure was regenerated to produce this report.

## What this contains

```
frozen_baseline/   cohort aggregates and per-seed frozen-probe summaries
runs/              run_summary.json + trajectory.tsv for the 12 retained runs,
                   plus the three block-index definitions and the one per-item
                   table that the current figures actually read
figures/           the current summary figures (PDF + PNG) and the two
                   FIGURE_SUMMARY documents written by the generating scripts
CURATION_MANIFEST.tsv   one row per artifact, included and excluded
CURATED_FILES.sha256    SHA-256 of every file in this report
```

Retained runs were determined from the `RUNS` maps in
`scripts/naming_comprehension/make_summary_plots.py` and
`scripts/naming_comprehension/make_multitask_summary_plots.py`, not from
directory names.

## Source experiment family

All experiments in this report are **warm-start adaptations of the canonical
full-lexicon repetition cohort `93a577f`** (training commit
`93a577fd9822955fa272ee733fa7e2acf81f1333`), evaluated on the 29,571-word
GloVe-covered lexicon (`data/lexicon_en_glove_covered.tsv`, sha256
`ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66`).

**Frozen baseline (family A)** used the **four historical selected checkpoints**
of that cohort — seeds 19/e155, 20/e130, 21/e145, 22/e140 — as recorded in
`frozen_baseline/cohort/cohort_summary.json` (`n_checkpoints: 4`). Those four
were selected under the historical stable-zero criterion X=2. Under the current
X=5 criterion only seeds 19 and 22 qualify; the selected epochs themselves are
unchanged at every X. The frozen-baseline aggregates in this report are
therefore 4-checkpoint means and are labelled as such by the source files.

**Single-task and warm-start experiments (families B–E)** all start from the
single canonical checkpoint **seed 22 / epoch 140**:

```
seed_22_epoch_0140.pt
sha256  a15846cbf3c7df88ed289512bbb20cbefd2121d0deec1b39f363932a743da595
stored  archives/fulllexicon_93a577f/extracted/
        fulllexicon_final_bundle_93a577f/selected_checkpoints/
```

Every retained `run_summary.json` records that path, that hash, and the training
commit independently; the manifest reproduces them per run.

## Git commits

| Role | Commit |
|---|---|
| Repetition training (cohort `93a577f`) | `93a577fd9822955fa272ee733fa7e2acf81f1333` |
| Frozen probe (family A) | `3d0ce67d910020875bb994b13403049d4d223dda` |
| Cohort aggregation | `9981133009d565378259f2399ed8e97c408c8655` |
| Single-task adaptation runs (families B–D) | per run: `d5f087b`, `bb45299`, `14571e5`, `80906b0`, `c8da363` — see manifest |
| Phase 3A / 3B multitask | `c56355cd4f8298afe901f83aa5f5367f45209b7f` |
| Phase 3C full-lexicon rehearsal | `439316e09b3c79ad5740fbffaecafacfa4a3558f` |
| Single-task figure script | `5a79feee9383a631025e3a6ddfe2b0339119a3cb` |
| Multitask figure script | `8370a00b72adadf8db4cab2bb1794da41f5137c0` |

Several single-task runs recorded `eval_git.dirty = true` at run time, meaning
the working tree carried uncommitted changes when they were produced. The runs
recorded this themselves; it is reproduced per artifact in the manifest and is
not corrected here.

## What is deliberately not in Git

- **Raw outputs.** The full output tree (`outputs/naming_comprehension_93a577f/`,
  ~123 MB, 161 files, 35 run directories) remains gitignored. This report is a
  curated subset, not a replacement. Excluded exploratory and superseded runs
  are not listed individually.
- **Checkpoints.** No `.pt` file is versioned here. The 15 checkpoints produced
  by the retained runs are hash-referenced in the manifest with their SHA-256,
  size and source path, and are intended for external release alongside the
  canonical cohort bundle.
- **Large per-item tables.** The four frozen-baseline `per_item.tsv` files
  (~7.7 MB each, ~30.9 MB total) are excluded: every reported frozen-baseline
  number is an aggregate already present in `cohort_summary.json`,
  `cohort_by_seed.tsv` and the per-seed `summary.json`. They are referenced by
  source path in the manifest.

## Known limitations, as documented by the experiments

These are reproduced from the source `FIGURE_SUMMARY` documents and run
summaries. They are not restated more strongly here.

- **Local three-task coexistence is not monotonically stable.** Repetition LTM
  shows transient dips well below the 95% criterion during training, and
  comprehension itself dipped back under threshold at 740k between two
  successful snapshots.
- **Phase 3B vs Phase 3C confound.** The comparison changes the repetition
  rehearsal population from 3,288 to 29,571 while repetition task-step frequency
  is fixed, so the number of repetition presentations per item is also much
  lower in Phase 3C (approximately 288 passes vs 2,500). The comparison
  therefore changes **both** the breadth/coverage of rehearsal **and** the
  repetition exposure per individual item, and **these two effects are not
  separable from Phase 3B vs Phase 3C alone**. The endpoints also differ
  slightly (780k vs 800k steps).
- **Full-lexicon rehearsal does not perfectly preserve LTM.** 97.1% is 1.8
  absolute points below the canonical 98.9%.
- **No out-of-subset probe exists for Phase 3B.** The probe was introduced with
  the Phase 3C implementation, so only the 3C probe can be shown in the backup
  recovery-dynamics figure.
- **Naming scale result is a cost result.** The full-lexicon curve is still
  rising but decelerating after 3,000 exposures; it is not a proof of
  impossibility, and the cost curve is not extrapolable from two points.
- **Sequential-run acquisition confound.** The sequential run starts a fresh
  AdamW over the same scope because the source checkpoint stores no optimizer
  state. This is a recorded confound for the acquisition slowdown, though not
  for the forgetting.
- **Frozen-baseline statistics are descriptive only.** With n=4 checkpoints the
  cohort statistics are mean, sample SD, min and max; no inferential statistics
  are computed.

Joint multitask development from random initialization is a separate, ongoing
question and no result from it appears in this report.
