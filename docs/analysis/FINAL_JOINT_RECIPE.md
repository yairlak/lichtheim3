# Lichtheim3 — exact recipe of the final joint experiments

Audited from committed code at `ffa3659` (driver blob
`24742adf9f13c205673a3f8ff1f34bdce7a45e58`, unchanged since the u750 runs).
Every value below was read out of the code or the artefacts, not inferred.
Where a value could not be verified locally it is marked **UNVERIFIED** with
the command that verifies it.

## 0. Terminology — relation to Ueno et al.

This is an **Ueno-inspired 1:2:3 task-update / experience allocation**, not a
reproduction of Ueno's training loop.

*What is inspired by Ueno:* the ratio of experience across tasks —
1 repetition : 2 speaking/naming : 3 comprehension presentations per item per
epoch, in randomised order.

*What is ours, and differs:* Ueno used per-item online backpropagation with a
weight update after each item. Lichtheim3 interleaves **task-level minibatch
steps** (batch 64) under **AdamW**, on ~29.6k English words with 300-d GloVe
semantics, a GRU dual-route architecture, and a retrieval-based comprehension
objective. Homophones are handled by canonicalisation (one designated target
per phonological form, all homophones retained as bank competitors) rather
than Ueno's outright removal.

Acceptable wording: "Ueno-inspired 1:2:3 experience allocation", "modernised
Lichtheim3 analogue of the Lichtheim 2 developmental presentation schedule".
**Do not write** "faithful reproduction of Ueno's training algorithm", "exact
Ueno optimizer", or bare "Ueno schedule".

## 1. Populations and data

| | |
|---|---|
| Lexicon file | `data/lexicon_en_glove_covered.tsv` |
| Lexicon SHA256 | `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66` |
| Semantics | GloVe 6B, 300-d, real vectors only (`glove_fallback` must be 0) |
| Repetition population | 29,571 (full lexicon) |
| Naming population | 29,571 (full lexicon, homophones included) |
| Comprehension population | **27,981** canonical targets |
| C population SHA256 | `10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50` |
| Retrieval bank | **29,571** — the FULL lexicon, never shrunk to the C population |
| Dorsal pseudoword pool | 4,000 items |

**Canonical C definition** (`canonical_phonology_indices`): equivalence classes
keyed by the exact phoneme-ID tuple; within each class the entry with the
lowest `LexEntry.rank` (highest frequency) is kept, ties broken to the lowest
bank index. Excluded homophones remain full-bank retrieval competitors and
full members of the R and N populations.

## 2. Architecture

| module | dimension |
|---|---|
| `phon_embed` (shared) | 64 |
| WM encoder / decoder GRU | `wm_hidden` |
| WM `to_premotor` | wm_hidden → 128 |
| LTM encoder GRU (`unigru_last_hidden`) | `enc_hidden` |
| `to_semantic` | enc_hidden → enc_hidden → 300 |
| `sem_to_h0` | 300 → dec_hidden |
| LTM decoder GRU | `dec_hidden` |
| `dec_to_premotor` | dec_hidden → 128 |
| `motor.proj` (shared) | 128 → vocab |
| premotor dim | 128 (fixed in every condition) |

The three widths are **independent** CLI arguments (`--wm-hidden`,
`--enc-hidden`, `--dec-hidden`), each defaulting to 128. There is deliberately
no single hidden-size knob. Conditions used:

- **H256** = wm 128 / enc 256 / dec 256
- **H512** = wm 128 / enc 512 / dec 512

Gate: alpha 2.0, threshold 0.7, usage prior 0.5.
`interference_noise = 0.0`, `ventral_noise = 0.0` (the joint driver's
canonical constants; note these differ from the `WMConfig` dataclass default
of 0.1).

## 3. Losses

**Repetition step** — one backward over
`total_loss(R) + cfg.loss.wm * pool_CE(pool)`:

| term | weight |
|---|---|
| `rep` (gated-route sequence CE) | 1.0 |
| `align` (s_hat ↔ GloVe alignment) | 1.0 |
| `dec` (LTM-route sequence CE) | 0.5 |
| `wm` (WM-route sequence CE) | 0.5 |
| `gate` (gate regulariser) | 0.05 |
| label smoothing | 0.0 |
| dorsal pool CE | 0.5 (= `cfg.loss.wm`) |

**Naming step**: `LAMBDA_N (=1.0) * naming_CE`, teacher forcing ratio 1.0,
decoder input = gold prefix.

**Comprehension step**: `LAMBDA_C (=0.087) * retrieval_CE`, where
`retrieval_CE = CE(cos(normalize(s_hat), bank_normalized) / tau, target)`
over the **full 29,571 bank**, `tau = 0.10`. `c_align_weight = 0.0`
(no alignment term on the C stream).

## 4. Optimizer

**ONE** shared `AdamW` over every model parameter
(`--optimizer-policy shared_adamw`, the default); `task_optims is None`.
On shared weights the moment histories mix tasks. Betas/eps are PyTorch
defaults. Weight decay **1e-5**. Gradient clipping **1.0**, applied per task
step. Batch size **64**.

No per-task optimizer banks (`task_separated_adamw`, `grouped_rn_c_adamw`)
are used in any run reported here.

## 5. The 1:2:3 schedule — exact meaning

**Separate optimizer steps, not a summed backward.** Each optimizer step
trains exactly ONE task: `zero_grad → forward → backward → clip(1.0) → step`,
on the shared AdamW state. Only the streams that step consumes advance.

A **macro-cycle is 6 optimizer steps** containing exactly 1 R, 2 N and 3 C.
The order within a cycle is `macro_cycle(ratio, schedule_seed, cycle_index)`:
a deterministic shuffle drawn from a private `torch.Generator`, a pure
function of (schedule seed, cycle index). It never reads the global RNG and
stores no state, so cycle position is recomputed from `global_step` and
mid-cycle resume is exact. The schedule seed is `seed * 1000003 + 4`,
disjoint from the four sampler seeds (offsets 0–3).

The dorsal pool **rides every R step inside the same backward**, keeping the
pool:R batch ratio at 1:1.

## 6. Samplers

| stream | population | sampler | batches per pass | seed offset |
|---|---|---|---|---|
| repetition | 29,571 | log-frequency weighted `log((N+1)/rank)^1.0`, clipped 1e-6, normalised, **with replacement** (multinomial) | 463 | 0 |
| pool | 4,000 | unweighted permutation | 63 | 1 |
| comprehension | 27,981 | unweighted permutation | **438** | 2 |
| naming | 29,571 | unweighted permutation | 463 | 3 |

All streams are **counter-addressed**: batch *k* is a pure function of
(stream seed, k), via `epoch_seed = seed * 7919 + epoch`. Stream seeds are
`seed * 1000003 + offset`, re-derived on resume and then **verified** against
the checkpoint.

## 7. Progress variable u and exposures

`u` ≡ **repetition exposures**. One macro-cycle = 6 optimizer steps and
advances R by 1 batch, N by 2, C by 3.

```
steps(u) = u × 463 × 6 = 2778 u
R exposures = u
N exposures = 2u                       (exact: |N| = |R| = 29,571)
C exposures = 3u × 463/438 = 3.1712 u  (NOT 3u: |C| = 27,981)
```

**This is a real and easily-missed point**: 1:2:3 is a presentation/update
ratio, not an exposure ratio. At u=1200, C exposures are 3805.5, not 3600.

`exposures()` in the driver is `cursor / batches_per_pass`, so the logged
`r/n/c_exposures` columns are exact.

## 8. Learning-rate history of the reported lineage

| stage | policy | value | clock |
|---|---|---|---|
| u0 → u100 | `two_stage_rep_cursor` stage 1 | 1e-3 | **repetition cursor** < 46,300 |
| u100 → u750 | `two_stage_rep_cursor` stage 2 | 1e-4 | repetition cursor ≥ 46,300 |
| u750 → u850 (pilot) | `task_specific` | control 1e-4 / low 3e-5 (all three tasks) | flat |
| u850 → u1200 (pilot) | `task_specific` | all-3e5 = 3e-5/3e-5/3e-5; C-high = 3e-5/3e-5/**1e-4** | flat |

**The LR clock is the repetition cursor**, i.e. completed R batches, not the
global step and not epochs. `LR_BOUNDARY_STEPS = 46,300 = 100 × 463`. The
argument counts *completed* batches, so `lr_for_step(46_299) = 1e-3` and
`lr_for_step(46_300) = 1e-4`.

Consequence, documented and deliberately not corrected: under 1:2:3 the drop
happens when N has had ~200 and C ~317 exposures at 1e-3.

Under the `task_specific` policy the LR is a constant per task, independent
of step. Setting all three equal makes it a flat global LR; a flat 1e-4 is
**bitwise identical** to continuing under the two-stage policy past its
boundary (verified in `tests/test_lrpilot_u850.py`).

**Adam moments are never reset at any transition.** `load_state_dict`
restores the single shared AdamW and every moment; the per-step LR is then
applied to the param groups. A change of LR policy is refused unless declared
with `--phase-transition`, which records old/new policy and
`moment_initialization: "unchanged"`.

## 9. Branching

Every stage is a **continuation of the same from-scratch lineage**, never a
re-pretrained model. On resume the driver restores: model weights, the shared
AdamW and all moments, all four stream cursors, `global_step`, the schedule
anchor, the ceiling streak (`consecutive_ceiling`, `last_ceiling_step`), and
all four RNG streams (torch/numpy/python/cuda) through an explicit contract.

A branch writes to its **own** run directory; the parent is opened read-only.
Ancestry (`parent_run_id`, `parent_checkpoint`, `parent_global_step`,
`parent_lr_policy`, inherited state) is recorded in the branch's
`provenance.json`. A continuation *in place* appends to `metrics.tsv` and
writes `config_from_step_<N>.json`, leaving the original files untouched.

## 10. Evaluation definitions — strict

| readout | definition |
|---|---|
| **Repetition, canonical** | `evaluate_forms_ar`: greedy AR to the batch maximum, each item truncated to its own gold length + 1, cut at the first EOS. **Forced-length** — it consults the target length. Legitimate for repetition (the form is the input) and is the convention every historical repetition number uses. Route-isolated for full / wm / ltm. |
| **Repetition, genuine free-AR** | Decode to a single GLOBAL cap `FREE_AR_MAX_STEPS = 12`, cut at the first EOS wherever it falls. Gold length **never** consulted, so over-generation and non-termination are errors. Reported for full / wm / ltm alongside the canonical metric; no historical number is restated. |
| **Naming** | Free greedy AR from the raw GloVe vector: BOS → greedy → EOS or global cap `NAMING_MAX_STEPS = 10`. Target length never consulted. |
| **Comprehension** | **Strict** canonical word-ID top-1: `argmax(cos(s_hat, bank)) == target_bank_index`, over the 27,981 canonical targets against the full 29,571 bank. Not top-k, not homophone-class-aware. |

Longest form in the lexicon is 9 phonemes, so cap 12 cannot truncate a
correct repetition answer.

## 11. Ceiling criterion

Stop only after **5 consecutive DISTINCT scheduled full evaluations** with,
simultaneously and exactly 1.0:

- repetition canonical exact
- repetition genuine free-AR exact
- naming genuine free-AR exact
- comprehension strict top-1

Any shortfall resets the streak to zero. Only the in-loop scheduled-milestone
branch counts: dev evaluations and the optional duplicate endpoint evaluation
never move it, and a step already counted cannot be counted twice. The streak
and the last counted step are checkpointed, so a requeue cannot restart them.

*(Raised from 2 to 5 before the u750→u850 continuation. The u0–u500 block ran
under 2, which never fired because no run reached ceiling, so no executed
trajectory is affected.)*

## 12. Dorsal capacity probe — protocol (data UNVERIFIED locally)

`scripts/naming_comprehension/route_capacity_probe.py --route dorsal`,
widths 128/256/512, from scratch, seed 22, 200 exposures.

- **Path**: phonology → `phon_embed` → WM encoder → WM decoder →
  `to_premotor` → shared `motor`. The **entire LTM route is frozen**, so
  lexical retrieval cannot solve repetition.
- **Trainable scope**: `('phon_embed.', 'wm.', 'motor.')`; a fingerprint
  asserts nothing else moved.
- **Sampler**: the canonical log-frequency-weighted-with-replacement
  repetition sampler, verified batch-for-batch identical to the joint
  driver's repetition stream.
- **Readouts**: route-isolated WM AR repetition on the 29,571 lexical items
  (canonical *and* genuine free-AR), plus the committed 400-item WFE
  pseudoword set split short/long at the median length — the pseudoword
  profile is the capacity diagnostic, since a real WM generalises to novel
  forms while a memoriser does not.
- **Schedule**: lr 1e-3 through exposure 100, then 1e-4; wd 1e-5, clip 1.0,
  batch 64 — identical across widths, never tuned per width.

The probe's LR schedule and its unweighted-vs-weighted sampler choices are
**not** the joint recipe; the dorsal probe is a single-task isolation
experiment and its absolute numbers are not comparable to joint numbers.
