# FINAL base-123 — historical baseline audit

Audited at commit `94fe7e5` on `feat/joint-multitask-scratch`, before any
implementation for the high-capacity block.

## 1. Which lineage is the genuine simple shared 1:2:3 baseline

**FINAL-3P**, introduced at commit `73982f3` ("FINAL-3P: from-scratch
interleaved 1:2:3 schedule (--schedule, default summed)"), production script
`scripts/cluster/jeanzay/final3p_run.slurm`.

The name was not trusted; the lineage was confirmed from git order and from
the flags the production script actually passes. FINAL-3P **predates** every
rescue mechanism:

| commit | campaign | mechanism introduced |
|---|---|---|
| `73982f3` | **FINAL-3P** | interleaved 1:2:3, shared optimizer — **the baseline** |
| `45eab2e` | FINAL-4 | task-specific learning rates |
| `b7c3c9f` | FINAL-5P | semantic-LR branch |
| `4e415ae` | FINAL-6P | task-separated AdamW moment banks |
| `5fca979` | FINAL-7P | grouped RN\|C banks |
| `b55df25` | FINAL-8P | L_dec 0.5 → 2.0 |
| `2975f63` | FINAL-9P | ratio 1:2:3 → 2:2:3 |

FINAL-3P's `srun` passes only: `--regime j0 --seed 22 --subset-mode
final_full --schedule interleaved_123 --device cuda --epochs N
--full-eval-at ... --eval-every ... --save-every ... --log-every 200
--endpoint-eval --out-dir ... --run-id ...`.

No `--optimizer-policy`, no `--lr-*`, no `--dec-weight`, no
`--c-align-weight`. **Everything else is driver defaults**, and every default
at HEAD was verified to still hold the FINAL-3P-era value (below). A bitwise
comparison against `94fe7e5` confirms the new width arguments are a no-op when
omitted.

## 2. Resolved recipe

**A. From-scratch ancestry.** Random init from the single experimental seed;
no source checkpoint, no inherited optimizer state. Stream seeds derive as
`seed * STREAM_SEED_STRIDE + offset`; the schedule seed sits at offset 4,
disjoint from the four samplers.

**B. Optimizer.** Exactly ONE AdamW over all model parameters
(`--optimizer-policy` default `shared_adamw`; `task_optims is None`). Weight
decay 1e-5, betas/eps left at PyTorch defaults. Moments are preserved across
the LR change — the LR is re-derived per step, the optimizer is never rebuilt.

**C. Task presentation.** One task per optimizer step
(`zero_grad → backward → clip(1.0) → step`). Six-step macro-cycles holding
exactly 1 R, 2 N, 3 C. Order is a deterministic shuffle of
`(schedule_seed, cycle_index)` drawn from a private `torch.Generator`; it
never reads the global RNG and stores no state, so cycle position is
recomputed from `global_step` and mid-cycle resume is exact.

*1:2:3 is a presentation/update ratio, not an exposure ratio* — see §3.

**D. Samplers** (verified, not assumed):

| stream | population | sampler | batches/pass |
|---|---|---|---|
| repetition | 29,571 | log-frequency weighted, **with replacement**, `freq_temp` 1.0, clipped 1e-6, normalised | 463 |
| naming | 29,571 | unweighted permutation | 463 |
| comprehension | 27,981 | unweighted permutation | 438 |
| dorsal pool | 4,000 | unweighted permutation | 63 |

**E. Repetition loss.** `total_loss` with `CANONICAL_LOSS_WEIGHTS`:
rep 1.0, align 1.0, dec 0.5, wm 0.5, gate 0.05, label_smoothing 0.0.

**F. Pseudoword pool.** ACTIVE. The pool rides **every R step inside the same
backward**: `loss = total_loss(R) + cfg.loss.wm * pool_CE`, i.e. coefficient
0.5, keeping the pool:R batch ratio at 1:1. It is not a separate update.

**G. Naming.** `LAMBDA_N (=1.0) * naming_CE`, teacher forcing 1.0, decoder
input = gold prefix, all parameters trainable.

**H. Comprehension.** `LAMBDA_C * retrieval_CE` only — full-bank cosine
retrieval CE at `tau = 0.10` against all 29,571 entries. `c_align_weight`
defaults to 0.0 (FINAL-2A tested adding alignment and it was worse).

**λ_C = 0.087 DID belong to this baseline** — it is the module constant used
by the FINAL-3P run with no flag override. It is **retained**. Note it is the
weight on the *sole* C term here, so it is a task-weighting scalar; isolated C
probes used unweighted CE, which makes probe CE values **not comparable** to
joint `retrieval_ce`.

**I. LR schedule — the clock.** `LR_STAGE1 = 1e-3 → LR_STAGE2 = 1e-4` at
`LR_BOUNDARY_STEPS = 46,300`, and the clock is **the repetition cursor**
(`cursors["repetition"]`, completed R batches), not the global step, not
epochs. 46,300 = 100 × 463 = **100 R exposures**. The argument counts
completed batches, so `lr_for_step(46_299) = 1e-3` and
`lr_for_step(46_300) = 1e-4`.

**Consequence, documented not corrected:** under 1:2:3 the drop happens after
N has had **200** exposures and C **317.12** at 1e-3. Isolated H512 naming
destabilised under sustained 1e-3 (.840 @100 → .743 @300). See the
preregistration for how that signature must be read.

**J. Other settings.** batch 64; grad clip 1.0 per task step; wd 1e-5;
`interference_noise = 0.0` and `ventral_noise = 0.0` (the joint driver's
canonical constants — *not* the `WMConfig` dataclass default of 0.1);
gate alpha/threshold/usage_prior canonical; no curriculum.

**K. Evaluators.** FULL repetition = `repetition_snapshot` canonical
**forced-length** AR (truncates to gold length + 1, cuts at first EOS),
route-isolated for full/wm/ltm. Naming = free greedy AR, BOS → EOS or global
cap 10, target length never consulted. Comprehension = strict canonical
word-ID top-1 against the full 29,571 bank.

## 3. Exposure semantics (§8) — the one arithmetic correction

`exposures()` = `cursor / batches_per_pass`. Because populations differ
(|C| = 27,981 vs |R| = |N| = 29,571), the 1:2:3 **update** ratio does not give
a 1:2:3 **exposure** ratio:

- R exposures = u
- N exposures = 2u  (exact — same population size)
- C exposures = 3u × 463/438 = **3.1712 u** (not 3u)

At u = 500: R 500, N 1000, **C 1585.6164** (not 1500). 1,389,000 optimizer
steps. This is a reporting correction, not a design change.

## 4. Population differences vs the final protocol

FINAL-3P already used the immutable final populations (`--subset-mode
final_full`): R 29,571 / N 29,571 / C 27,981 canonical / bank 29,571, with the
canonical C hash asserted at runtime. **No population change is needed**, so
the historical-vs-new comparison is not confounded by population.

Note however that the H128 joint failures being contrasted against come from
that historical era; tonight's block contains no H128 arm, so a direct
"capacity alone" claim against H128 rests on the archived runs rather than on
a within-block control. This is stated explicitly and no H128 arm was added.

## 5. Hash note

The brief's lexicon SHA256 is 62 hex characters —
`ae80918165e16b8cb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66` — a
transcription with two characters dropped. The actual file is
`ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66` (64), which
is the file all prior runs used. **No data mismatch**; the brief has a typo.
Canonical C hash `10c2f06e…` and n = 27,981 verified on the real lexicon.
