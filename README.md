# Lichtheim3

A modern dual-route neurocomputational model of spoken **word repetition**,
trained to exact zero whole-word error on a 29,571-word English lexicon, and
analysed for a behavioural and representational division of labour between a
dorsal working-memory route and a ventral long-term-memory route.

Lichtheim3 is **inspired by** the Lichtheim / Ueno et al. (2011) dual
dorsal–ventral framework. It is **not** a literal implementation or replication
of Lichtheim2: the architecture, the training objective and the evaluation
protocol are all modern redesigns. Claims of correspondence with Ueno et al.
should be checked against that paper and its supplement rather than assumed.

> **Status.** This branch is a *pre-release integration* tree. It gathers the
> established scientific families, their evidence and the reproducibility
> metadata in one place. It is not yet the repository default branch, no LICENSE
> has been agreed, and canonical checkpoints are not yet distributed. See
> [Reproducibility status](#reproducibility-status) and
> [Known blockers](#known-blockers).

---

## Scientific motivation

Repetition of a heard word can proceed by two routes. A **dorsal** route holds
the phonological form in a bounded working-memory state and re-articulates it —
sub-lexical, generalising to novel forms, but capacity- and length-limited. A
**ventral** route maps the form onto a lexical-semantic representation and
regenerates the form from it — robust for known words, but with nothing to look
up for a nonword. Human repetition behaviour, and the dissociable deficits that
follow focal damage, motivate having both.

Lichtheim3 implements both routes explicitly and lets them compete through a
gate, so that the division of labour is something the model can be *measured
for* rather than something built in by fiat.

**Historical context.** A separate earlier effort reimplemented Lichtheim2
faithfully — a tick-by-tick Elman network with copy-back connections — in the
repository `Neuro-Cog-AI/dual_route_single_word_processing`. That work is
independent of this one: the two repositories share no Git ancestry and no
source code. It is mentioned only so the two are not confused.

---

## Architecture

```
phonemes ─► shared phoneme embedding (64)
              │
              ├─► dorsal WM route      GRU encoder→decoder, hidden 128
              │                        (models/wm_route.py)
              │
              └─► ventral LTM route    uniGRU encoder (128) → semantic vector (300)
                                       → GRU decoder (128)
                                       (models/ltm_route.py)
                        │
                  familiarity / confidence gate      (models/gating.py)
                  g = sigmoid(alpha · (c_LTM − threshold))
                        │
                  premotor mixture (128)
                        │
                  shared motor read-out → phoneme logits   (models/motor.py)
```

Canonical historical settings, as recovered from the archived checkpoints and
job scripts (**not** from `config.py`):

| | Value |
|---|---|
| phoneme embedding | 64 |
| WM hidden | 128 |
| LTM encoder | `unigru_last_hidden`, hidden 128, 1 layer |
| LTM decoder hidden | 128 |
| semantic dimension | 300 (GloVe) |
| premotor dimension | 128 |
| gate `alpha` | 2.0 |
| gate `threshold` | 0.7 |
| gate `usage_prior` | 0.5 |
| WM `interference_noise` | 0.0 |
| ventral noise | 0.0 |
| loss weights | rep 1.0, align 1.0, dec 0.5, wm 0.5, gate 0.05 |

> ⚠️ **`config.py` defaults are NOT the canonical training configuration.**
> Of 46 compared fields, 19 differ; four are architecture-incompatible —
> `ltm.enc_hidden` and `ltm.dec_hidden` default to 256 (canonical 128), and
> `ltm.ltm_encoder_mode` defaults to `bigru_masked_mean` (canonical
> `unigru_last_hidden`), with the derived `bidirectional_encoder` flag
> following. `wm.interference_noise` defaults to 0.1 (canonical 0.0), and the
> gate defaults to alpha 4.0 / threshold 0.5 (canonical 2.0 / 0.7).
>
> This is **harmless when loading a canonical checkpoint** — checkpoints store
> their own `cfg_*` dictionaries and the evaluator rebuilds the config from
> them — but running `python train.py`, whose `__main__` calls
> `default_config()`, would silently build a **different model**.
>
> Authoritative record: [`configs/canonical_93a577f.yaml`](configs/canonical_93a577f.yaml)
> · field-by-field diff: [`configs/canonical_93a577f_vs_defaults.tsv`](configs/canonical_93a577f_vs_defaults.tsv)

---

## Canonical repetition model

Cohort `93a577f`, trained on the full **29,571-word GloVe-covered lexicon**
(`data/lexicon_en_glove_covered.tsv`) with **no validation split**, seeds
19–22, **200 epochs** in two stages:

| Stage | Epochs | LR | Notes |
|---|---|---|---|
| 1 | 1–100 | `1e-3` | `--save_every_epochs 0` → `seed_<s>_e100.pt` |
| 2 | 101–200 | `1e-4` | resumes stage 1, `--save_every_epochs 5` |

The learning-rate change is **not a scheduler**: it is a two-job boundary, with
stage 2 resuming the stage-1 checkpoint under a fresh AdamW at the new rate.
Optimiser AdamW, batch 64, weight decay `1e-5`, grad clip 1.0, teacher forcing
1.0, dorsal pseudoword pool 4000, frequency-weighted sampling.

Evaluation: deterministic **autoregressive** decoding, FULL route, exact
whole-word repetition over the full training lexicon, at epochs 105–200 in steps
of 5 (20 evaluations per seed, 80 total).

### Selection policy — X = 5

A seed qualifies on **5 consecutive zero-error FULL evaluations**; the selected
checkpoint is the *first* of the streak.

| Seed | Epoch | Zero streak | Canonical? |
|---|---|---|---|
| **19** | **155** | 6 (155–180) | **✅ canonical** |
| **22** | **140** | 13 (140–200) | **✅ canonical** |
| 20 | 130 | 2 (130–135) | ❌ historical, non-canonical |
| 21 | 145 | 0 (never reached zero) | ❌ historical, non-canonical |

Raising the criterion from the earlier X = 2 **changed no selected epoch** —
seeds 19 and 22 select 155 and 140 at every X. What changed is cohort
*membership*. Several analyses predate the restriction and legitimately use all
four historical selected checkpoints as a multi-seed robustness cohort; they
were not rerun and are not relabelled.

Full policy: [`docs/canonical_selection_X5.md`](docs/canonical_selection_X5.md)
· cohort creation record: [`docs/full_lexicon_ceiling.md`](docs/full_lexicon_ceiling.md)
· hashes and roles: [`manifests/checkpoints.tsv`](manifests/checkpoints.tsv)

---

## Cohort transparency

Which result family used which checkpoints. This matters for reading any figure
in this repository.

| Result family | Checkpoint cohort | Status |
|---|---|---|
| Behavioral WFE release (F1–F7, S1–S12) | 19/e155, 20/e130, 21/e145, 22/e140 | **historical four-checkpoint robustness cohort** — not all canonical |
| Ventral semantic / RSA | 19/e155, 22/e140 | **X = 5 canonical cohort** |
| Frozen naming/comprehension baseline | 19/e155, 20/e130, 21/e145, 22/e140 | historical four-checkpoint aggregate, descriptive only |
| Phase 2/3 acquisition and warm-start | 22/e140 | canonical warm-start source |
| Phase 4 joint multitask from scratch | random initialisation (no source checkpoint) | **ACTIVE / NOT RELEASE-CANONICAL** |

---

## Main scientific findings

### Behavioral route division

From the curated release in
[`reports/behavioral_wfe_fulllexicon_93a577f/final_release/`](reports/behavioral_wfe_fulllexicon_93a577f/final_release/)
(7 main figures F1–F7, 12 supplementary S1–S12, 24 indexed tables, each with a
recorded finding *and* limitation).

- **F1** — for trained real words FULL and WM are at exact ceiling and LTM shows
  only a very weak slope; for novel pseudowords LTM develops a large length
  effect while WM stays far more robust and FULL stays near ceiling.
- **F2** — the LTM−WM pseudoword length-slope difference is positive in all four
  seeds (+0.183 to +0.246 edit operations per phoneme).
- **F4** — substitutions dominate, then deletions, then insertions; the burden is
  concentrated in long LTM pseudowords.
- **F5** — 87 observed premature end-of-sequence events, 82 of them in LTM, all
  on pseudowords, none on trained real words.
- **F6** — the LTM Zipf frequency slope is negative in all four seeds
  (mean −0.0130); higher-frequency trained words have slightly fewer errors.
- **F7** — route and lexicality/exposure lead jointly, length is a stable third,
  morphology negligible. The route-vs-lexicality ordering is **not resolved** by
  these data.
- **F3** — serial-position profile: LTM pseudowords show a strong late-position
  increase; FULL and WM increase much less. This is classified
  **`DESCRIPTIVE_ONLY`** (zip-mismatch positions, no Levenshtein alignment,
  pooled over seeds) and should not be read as an estimated effect.

The ceiling zeros for FULL and WM on trained words are **structural**
(`CEILING_LIMITED`), not evidence that those routes lack length information.

> A "WM rescues LTM errors" result is often discussed, but **no release figure
> exists for it**. The closest tracked evidence is
> `reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/ltm_pseudoword_success/`.
> It is listed as outstanding work, not as a finding.

### Representational dissociation

From [`reports/ventral_semantic_93a577f/`](reports/ventral_semantic_93a577f/)
(X = 5 cohort). The RSA is **exact over all 437,207,235 unique word pairs** —
nothing sampled.

| Seed | Route | vs GloVe (semantic) | vs phonological (raw Levenshtein) | vs phonological (normalised) |
|---|---|---|---|---|
| 19/e155 | **LTM** | **0.398** | 0.021 | 0.041 |
| 19/e155 | **WM** | 0.054 | **0.155** | **0.421** |
| 22/e140 | **LTM** | **0.388** | 0.026 | 0.041 |
| 22/e140 | **WM** | 0.055 | **0.130** | **0.430** |

The two reference geometries are nearly independent (GloVe vs raw phonological
r = 0.0079), so the dissociation is not an artefact of shared structure, and a
partial regression reproduces it.

Two things must be stated with the result, not after it:

1. **Normalisation sensitivity.** Normalising Levenshtein by length raises the
   dorsal–phonological correlation from ~0.13–0.16 to ~0.42–0.43. The
   qualitative conclusion is unchanged and in fact strengthened, but **no single
   number should be quoted as "the" dorsal–phonology correlation**. Effect sizes
   are modest overall (best model R² = 0.16).
2. **The ventral space is not identification-grade.** Semantic target retrieval
   from `s_hat` is 2.65 % / 2.54 % top-1 against a chance level of 0.0034 %
   — far above chance, but for a typical word some *other* GloVe vector is
   nearer than its own (median margin ≈ −0.22). The geometry is organised
   semantically without individual words being recoverable.

Details: [`RESULTS_SUMMARY.md`](reports/ventral_semantic_93a577f/RESULTS_SUMMARY.md)
(Phase B) and [`RESULTS_SUMMARY_PHASE_C.md`](reports/ventral_semantic_93a577f/RESULTS_SUMMARY_PHASE_C.md)
(Phase C).

### Naming and comprehension

Curated in [`reports/naming_comprehension_93a577f/`](reports/naming_comprehension_93a577f/).

**Frozen baseline** (before any task training; mean over the four historical
checkpoints): comprehension top-1 **2.55 %**, naming from **true GloVe exactly
0.00 %**, but naming from the model's **own learned internal semantic code
`s_hat` 98.73 %**. The decoder is already highly functional from its internal
code; raw GloVe is simply not yet an interchangeable interface. This is what
motivated training the two tasks explicitly.

**Trained acquisition.** Naming is learnable but the cost rises steeply with
lexicon size (criterion at 550 exposures for N = 3,288, 1,200 for N = 10,000,
and 9.8 % after 3,000 exposures at N = 29,571 — a cost result, not a proof of
impossibility). Comprehension reaches criterion at 3,850 exposures with no
architecture, scope, objective or hyper-parameter change.

**Catastrophic forgetting.** Single-task training leaves WM numerically
unchanged (0.999763, bit-identical) while LTM repetition falls from 98.9 % to
1.4 % (comprehension) or 0.14 % (naming). The damage is specific to the ventral
route.

**Sequential / block results.** Training Block A then Block B forgets A almost
completely within 50 B-exposures (97.51 % → 0.02 %) *and* slows B acquisition
relative to training B alone. Both blocks are individually learnable, so neither
is a capacity limit. A separate control shows the effect depends on how many
mappings are trained at once, not on which: each ~10k block reaches 97–98 %
alone while the identical words sit at ~10 % inside the full lexicon.

### Warm-start multitask

- **Local three-task coexistence.** With repetition rehearsed on the same
  population as naming and comprehension, all three eventually coexist in the
  same LTM parameters (criterion met at 720k steps, confirmed at 760k and 780k).
  Coexistence is **not monotonically stable** — repetition LTM dips well below
  criterion during training.
- **Local rehearsal does not preserve global repetition.** Phase 3B endpoint:
  comprehension 95.1 / naming 100.0 / repetition-on-subset 100.0 but
  **repetition on the full lexicon 11.8 %**.
- **Full-lexicon rehearsal largely restores it.** Phase 3C: 91.5 / 100.0 / 97.3 /
  **97.1 %** full-lexicon. WM is numerically unchanged in every condition.
- **Preservation–acquisition trade-off.** Global preservation costs
  comprehension acquisition speed, and LTM is *not* perfectly preserved: 97.1 %
  is 1.8 absolute points below the canonical 98.9 %.

> ⚠️ **Documented limitation.** The Phase 3B vs Phase 3C comparison changes the
> repetition rehearsal population from 3,288 to 29,571 while repetition
> task-step frequency is held fixed, so the number of repetition presentations
> *per item* also falls (≈2,500 → ≈288 passes). **Rehearsal breadth and
> effective per-item exposure change together and are not separable from this
> comparison alone.** The endpoints also differ slightly (780k vs 800k steps).

### Joint multitask from scratch (Phase 4)

**ACTIVE / NOT RELEASE-CANONICAL.** This branch contains the Phase 4A1
infrastructure — a paired H0/J0 driver that initialises randomly rather than
warm-starting — but **no release-canonical scientific conclusion**. No Phase 4
result appears in any manifest or figure inventory here.

---

## Installation

Recommended:

```bash
conda env create -f environment.yml
conda activate lichtheim3
```

Alternative:

```bash
pip install -r requirements.txt
```

Both install the same nine packages: `torch`, `numpy`, `matplotlib`,
`Levenshtein`, `pandas`, `scipy`, `scikit-learn`, `rapidfuzz`, `pytest`.
`Levenshtein` must be the rapidfuzz-backed distribution, not the legacy
`python-Levenshtein` wrapper — see the rationale in `requirements.txt`.

**No bit-exact historical reproduction is claimed.** The canonical cohort was
trained on IDRIS Jean-Zay (SLURM, V100, module `pytorch-gpu/py3/2.6.0`); the
later adaptation and analysis work records `torch 2.12.1`. The resolved
CUDA/cuDNN versions and the training-time Python version are **not recoverable**
from any archived artefact. GPU support is deliberately unpinned — install the
`torch` build appropriate to your platform.

---

## Data

See [`data/README.md`](data/README.md) for the full inventory.

| Dataset | Status |
|---|---|
| `data/lexicon_en_glove_covered.tsv` — canonical 29,571-word lexicon | tracked |
| `data/lexicon_en.tsv` — legacy 30,000-word lexicon | tracked, **non-canonical** |
| GloVe 6B 300d | **external download** (~1.04 GB), `bash data/get_glove.sh` |
| `data/raw-nwr_swp/` — NWR/SWP stimuli | tracked, **redistribution status UNRESOLVED** |
| `data/eval_external/` — derived evaluation TSVs | tracked, inherits the unresolved status |

---

## Checkpoint access

**Canonical checkpoints are not distributed through normal Git.** They live in a
frozen archive bundle under `archives/`, which is gitignored, and were produced
on Jean-Zay.

| File | Seed / epoch | SHA-256 |
|---|---|---|
| `seed_19_epoch_0155.pt` | 19 / 155 | `7d05f9c2ad5a53e705f7d55ccde2581754918938d8ca888da35c0a859666478e` |
| `seed_22_epoch_0140.pt` | 22 / 140 | `a15846cbf3c7df88ed289512bbb20cbefd2121d0deec1b39f363932a743da595` |

Historical, non-canonical: `seed_20_epoch_0130.pt`
(`b44548b6916ea89c6f099402b78031063445e572932acee8dd7558a73dfc6cfb`) and
`seed_21_epoch_0145.pt`
(`ab58092e7c2bfac42ab977352e6d5d6416ca605b71a3eacb777300060b30f5cf`).

Checkpoint distribution is being prepared; integrity hashes and provenance are
already recorded in [`manifests/checkpoints.tsv`](manifests/checkpoints.tsv).
**External reproduction requiring checkpoints is therefore not yet fully
self-contained.**

---

## Reproducibility status

| Result family | Tracked evidence | Needs checkpoint | Needs GloVe | Current status |
|---|---|---|---|---|
| Full-lexicon selection | streak audits, `selected_checkpoints.tsv`, mf2 figure | no | no | **REPRODUCIBLE NOW** — `python scripts/reproduce.py stable-zero --out-dir <DIR>` |
| Behavioral analysis (F1–F7, S1–S12) | figures, captions, all pointer tables, provenance JSON, output SHA-256 manifest, and the validated aggregate reference tables under `reports/.../repro_inputs/` | no (release loaded none) | no | **PARTIAL** — figures and numbers inspectable now, and the validated-reference cross-checks run from a clone; tests needing the untracked per-item table skip (see below) |
| RSA / ventral semantic | figures, correlation and partial-regression tables, metadata | to regenerate | yes | **results inspectable now**; recomputation blocked |
| Frozen naming/comprehension | cohort + per-seed summaries, figure | to regenerate | yes | **results inspectable now**; recomputation blocked |
| Single-task acquisition | 9 run summaries + trajectories + 6 figures | to regenerate | yes | **results inspectable now**; recomputation blocked |
| Warm-start multitask | 3 run summaries + trajectories + 4 figures | to regenerate | yes | **results inspectable now**; recomputation blocked |
| Phase 4 | driver + tests only | n/a (random init) | yes | **ACTIVE / NOT RELEASE-CANONICAL** |

**Behavioral fresh-clone status (PARTIAL).** The behavioral analysis code loads
no checkpoint, and the five validated aggregate reference tables it is checked
against are now tracked under
`reports/behavioral_wfe_fulllexicon_93a577f/repro_inputs/` (byte-identical
copies, provenance in `REPRO_INPUTS.tsv`). From a fresh clone the behavioral
safe-test set runs with **167 passed, 113 skipped, 0 failed**.

What remains blocked: the upstream per-item table
`canonical_behavioral_item_table.tsv` (14,400 rows, 3.25 MB) is **not tracked**.
It carries `target`/`prediction` ARPABET strings and the WFE design metadata for
every item, i.e. it reproduces NWR/SWP stimulus content, whose redistribution
status is unresolved. Sixteen tests that need it **skip** with an explicit
reason rather than fail. Regenerating the behavioral analysis end to end from
raw predictions therefore still requires that table, and by extension the
checkpoints; only the validated-reference comparisons are self-contained.

Provenance caveat: seven of the twelve retained naming/comprehension runs were
executed from a dirty working tree. What that does and does not permit is set
out in
[`PROVENANCE_LIMITATIONS.md`](reports/naming_comprehension_93a577f/PROVENANCE_LIMITATIONS.md).
Phase 3B and Phase 3C, which carry the preservation–acquisition result, were
both clean.

---

## Where things live

```
configs/      canonical_93a577f.yaml            recovered historical training recipe
              canonical_93a577f_vs_defaults.tsv 46-field diff against config.py
docs/         canonical_selection_X5.md         which cohort each result family uses
              full_lexicon_ceiling.md           cohort creation record (historical X=2)
manifests/    checkpoints.tsv  datasets.tsv  figures.tsv
reports/      fulllexicon_cohort_93a577f/       selection evidence
              behavioral_wfe_fulllexicon_93a577f/   F1-F7, S1-S12, tables, final_release/
              ventral_semantic_93a577f/         RSA, PCA, semantic identification
              naming_comprehension_93a577f/     curated runs, figures, manifest, limitations
models/       wm_route.py ltm_route.py gating.py motor.py dual_route.py
scripts/      behavioral_analysis/  length_effect_analysis/
              ventral_semantic/  naming_comprehension/
```

Every figure in the publication-oriented set is indexed in
[`manifests/figures.tsv`](manifests/figures.tsv) with its numeric source,
generating script, checkpoint cohort, code commit and documented limitation.

---

## Reproduce a tracked result

Stable-zero model selection is currently the **only** result that regenerates
end to end from the repository alone — no checkpoint, no GloVe, no NWR/SWP data,
no CUDA and no training:

```bash
python scripts/reproduce.py stable-zero --out-dir reproduced/stable_zero
```

This redraws the canonical model-selection figure (mf2) from the tracked
stable-zero audit tables and writes PNG, PDF, SVG and the caption. Verified from
a fresh clone: the PNG is pixel-identical to the tracked canonical figure and the
caption is byte-identical. `--out-dir` is required, so nothing is written unless
you say where.

Nothing is recomputed: the figure is drawn from already-validated audit tables
and its annotations are asserted against those tables as it is drawn.

**No other result family is self-contained yet** — see
[Reproducibility status](#reproducibility-status) for what each one still needs.

## Reproduction commands

Commands below are taken from the archived job scripts and from the scripts'
own CLI definitions. **Where no robust command exists, that is said rather than
guessed.**

Canonical training (from `control/stage{1,2}_*.sbatch` in the archived bundle):

```bash
# stage 1 — epochs 1-100
python scripts/train_checkpoint.py \
  --lexicon_path data/lexicon_en_glove_covered.tsv \
  --max_words 30000 --train_all_words \
  --epochs 100 --seed <SEED> --split_seed 0 \
  --batch_size 64 --lr 1e-3 --num_workers 0 --save_every_epochs 0 \
  --ltm_encoder_mode unigru_last_hidden --hidden_size 128 \
  --teacher_forcing_ratio 1.0 --interference_noise 0.0 --ventral_noise 0.0 \
  --gate_alpha 2.0 --gate_threshold 0.7 \
  --ckpt <CKPT_DIR>/seed_<SEED>_e100.pt --out_dir <RUN_DIR>

# stage 2 — epochs 101-200 (architecture restored from the stage-1 checkpoint)
python scripts/train_checkpoint.py \
  --resume_from <CKPT_DIR>/seed_<SEED>_e100.pt \
  --epochs 200 --lr 1e-4 --num_workers 0 --save_every_epochs 5 \
  --ckpt <CKPT_DIR>/seed_<SEED>.pt --out_dir <RUN_DIR>
```

Ceiling evaluation of a checkpoint:

```bash
python scripts/evaluate_train_lexicon_ceiling.py \
  --ckpt <CHECKPOINT> --decode autoregressive --out_dir <OUT_DIR>
```

Frozen naming/comprehension probe and cohort aggregation:

```bash
python scripts/naming_comprehension/frozen_probe.py --ckpt <CHECKPOINT> --out-dir <DIR>
python scripts/naming_comprehension/aggregate_cohort.py --run-dir <DIR> --seeds 19 20 21 22
```

Tests:

```bash
python -m pytest -m "not slow" tests/
```

**Not yet available as robust commands.** `run_all.py` drives an obsolete
pipeline from June 2026 and must not be used to reproduce anything in this
README. The `scripts/ventral_semantic/` modules take no command-line arguments —
their input paths are module constants — so the RSA and semantic-identification
figures cannot currently be regenerated by an external user without editing the
source. There is no single end-to-end reproduction entry point.

---

## Known blockers

**Blocking a public release**

1. NWR/SWP redistribution status is **unresolved**; no permission is claimed.
2. No repository LICENSE has been agreed.
3. Canonical checkpoints are not yet distributed.
4. This integrated tree is not the repository default branch.

**Important before handoff**

5. `scripts/ventral_semantic/` needs command-line arguments.
6. No one-command evaluation entry point.
7. `data/get_glove.sh` performs no checksum verification, and no trustworthy
   published checksum is recorded in this repository.
8. The test suite has not been executed in this environment.
9. No Git tags exist for `93a577f` or the integration commits.
10. No release figure exists for the WM-rescue claim.

**Polish**

11. Obsolete June-2026 figures under `figures/` are still tracked; they describe
    a superseded model and are flagged `OBSOLETE_SUPERSEDED` in
    `manifests/figures.tsv`.
12. No `CITATION.cff`.
13. Four ventral word-map PNGs are 9–16 MB each.

---

## Reference

Ueno, T., Saito, S., Rogers, T. T., & Lambon Ralph, M. A. (2011). Lichtheim 2:
Synthesizing aphasia and the neural basis of language in a neurocomputational
model of the dual dorsal-ventral language pathways. *Neuron, 72*(2), 385–396.
