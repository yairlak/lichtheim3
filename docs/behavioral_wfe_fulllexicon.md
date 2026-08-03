# Behavioral WFE analysis — full-lexicon cohort 93a577f

How the Word Feature Evaluation (WFE) behavioral results are produced, where
the code and outputs live, and how to regenerate the figures.

## Purpose

Answer, for the full-lexicon Lichtheim3 cohort, whether the length effect sits
in the ventral (LTM) route or the dorsal (WM) route, using the WFE stimulus set
from Dager et al. The analysis protocol was frozen **before any behavioral
result was inspected**; see
`outputs/behavioral_wfe_fulllexicon_93a577f/README_behavioral_analysis_design.md`
and its machine-readable twin `_control/behavioral_analysis_design.json`.

## Dataset regimes

| Set | n | Purpose |
|---|---|---|
| `FAITHFUL_WFE_ALL` | 1200 | Faithful stimulus-level replication with the article's original real/pseudo labels |
| `LICHTHEIM_CLEAN` | 1062 | Primary set: 671 trained real words + 391 novel pseudowords |
| `ALL_WITH_EXPOSURE_STRATA` | 1200 | Six training-exposure categories |
| `TRAINED_REAL_FREQUENCY_PRIMARY` | 671 | Frequency analysis on trained real words |
| `TRAINED_REAL_FREQUENCY_SENSITIVITY` | 678 | Adds the 7 pronunciation-variant words |

"Real words" in the primary figures means WFE words encountered during training
with the same phonological form; "Pseudowords" means WFE pseudowords whose
phonological form is absent from the training lexicon. The distinction matters
because the WFE was built against Dager's 50k lexicon, not our 29,571-word one.

## Checkpoints

Cohort `fulllexicon_cohort_93a577f`, training commit `93a577fd…`: seed 19
epoch 155, seed 20 epoch 130, seed 21 epoch 145, seed 22 epoch 140. All four
seeds are primary; **seed 21 is never excluded**, and seeds 19/20/22 form an
exact-ceiling sensitivity set only.

## Routes and metrics

Routes: `full` (gated), `wm` (dorsal only), `ltm` (ventral only), all decoded
under one convention — deterministic autoregressive, forced-length readout, no
noise, no teacher forcing.

Primary metric: raw Levenshtein edit distance over atomic ARPAbet tokens.
Secondary: word error (1 − exact match). Operation counts (insertions,
deletions, substitutions) come from `Levenshtein.editops`, the backend Dager
used; they feed the error taxonomy only.

## Faithful versus adapted

**Faithful** analyses reproduce Dager's method exactly: Figure 2A line styles
hard-coded (complex solid, simple dashed); Figure 2B via Ridge(α = 1.0), 80/20
split `random_state=42`, `permutation_importance(n_repeats=100,
random_state=42)`, no interactions and no p-values; Figure 2C via zip-mismatch
`Error_Indices` with **no Levenshtein alignment**.

**Adapted** analyses are Lichtheim3 extensions (clean-set restriction,
route contrasts, hierarchical bootstrap) and are always labelled as such. The
two are never mixed in one figure or table.

## Analysis package

`scripts/behavioral_analysis/` is inference-free — it never loads a checkpoint.

```
common.py                     frozen constants, colours, repo-relative paths
io.py                         loading, validation, deterministic TSV writing
bootstrap.py                  OLS slope + hierarchical bootstrap
compute.py                    every plotting table (statistics layer)
plotting.py                   the five figures (presentation layer)
build_canonical_table.py      production outputs -> canonical table
make_figures.py               regenerate all five figures
validate_outputs.py           structural validation
close_production_manifest.py  scientific/operational manifest split
```

### Regenerating the five figures

```bash
python -m scripts.behavioral_analysis.make_figures \
    --out_dir reports/behavioral_wfe_fulllexicon_93a577f/figures

python -m scripts.behavioral_analysis.validate_outputs \
    --figures reports/behavioral_wfe_fulllexicon_93a577f/figures
```

Each figure is emitted as PNG (300 dpi), PDF and SVG, beside the exact TSV that
produced it and a standalone caption. Regeneration is deterministic: the TSVs
are byte-identical across runs.

To rebuild the canonical table from the production predictions:

```bash
python -m scripts.behavioral_analysis.build_canonical_table
```

## Frozen choices that must not be edited

Analysis sets; seed policy; route definitions; metric definitions; the
hierarchical bootstrap (seeds resampled, then items within each analysis-set ×
stratum cell; B = 10,000; random seed 20260730; 95 % percentile interval); the
faithful zip-mismatch serial-position method. Red and blue encode lexicality
only and are never reused for another variable.

## Where future analyses go

Morphology, frequency, adapted feature importance, error taxonomy and premature
EOS will be added as further modules inside `scripts/behavioral_analysis/`,
each with its own compute function in `compute.py` and plot function in
`plotting.py`. The sprint order and status of every planned analysis is
`docs/behavioral_wfe_analysis_matrix.md`.

## Provenance

Every published figure directory carries
`behavioral_analysis_provenance.json` with training commit, evaluation-code
commit, checkpoint and dataset hashes, package versions and the bootstrap
configuration. Production outputs are pinned by
`production_scientific_outputs_FINAL.sha256`; append-only operational logs are
pinned separately (see `manifest_closure_note.md`).

## Limitations

- Four seeds only; the bootstrap draws from 35 distinct seed multisets, so
  intervals are coarse and descriptive.
- Forced-length readout means terminal insertions are unobservable — this
  matches the original Dager implementation.
- FULL and WM are at ceiling on trained real words, so their zero slope there
  is a floor rather than a demonstrated absence of a length effect.
- The clean-set restriction was frozen in advance but removes the harder
  untrained real words from the primary figures; exposure strata are reported
  alongside for that reason.

## Deferred and out of scope

SSP / sonority (Figure 2D) remains deferred. Neural-representation analyses,
route ablations and any causal account of the length effect are a separate
project: they need new experiments, not a re-reading of these outputs.
