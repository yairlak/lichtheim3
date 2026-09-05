# FINAL base-123 — preregistration

Frozen **before** any job of the 8-run block is submitted. Independently
reviewed; verdict GO on both the baseline design and the 8-run block.

## Question

Can increased **ventral capacity alone** move the original simple
from-scratch shared-optimizer 1:2:3 joint regime into a successful R/N/C
coexistence regime? This discriminates "H128 needed optimizer/schedule rescue
because multitask learning is intrinsically unstable" from "H128 needed those
rescues largely because the ventral architecture was under-capacity".

## Design

2 ventral capacities × 4 historically predefined seeds (19/20/21/22) = 8 runs,
paired by seed. The audited FINAL-3P recipe (see
`final_baseline123_audit.md`) is held constant; WM is fixed at 128 in every
arm. Encoder and decoder scale **together**, so the factor is a ventral
capacity **bundle** — this block does not separate encoder from decoder.

All eight configs are generated from one definition and frozen before
submission. No seed is inspected before the block is launched.

## Budget and stopping

u = R exposures; u_max = 500 = 1,389,000 optimizer steps (R 500 / N 1000 /
C 1585.62 — C is 3u × 463/438, not 3u). No automatic continuation.

Full evaluations at u = 0, 25, 50, 100, 150, 200, 300, 400, 500.

**Numerical stop** only when TWO CONSECUTIVE scheduled full evaluations have
all four exactly 1.0: canonical (forced-length) repetition, GENUINE free-AR
repetition, free greedy AR naming, strict canonical C top-1. Any shortfall
resets the streak; the streak is checkpointed so a requeue cannot restart it.

**Scientific acceptance is separate.** A 100/100/100 model with a collapsed
ventral contribution to repetition (`full_rep_ltm`) or a degenerate/saturated
gate (`gate_frac_below_0.05`, `gate_frac_above_0.95`) is numerically at
ceiling but is NOT a lesion-ready dual-route model, and must be flagged as
such. No arbitrary numerical threshold is prespecified for this; it is an
inspection requirement, and the relevant columns are logged at every full
evaluation.

## LR-clock risk — documented, deliberately not fixed

The historical clock is the REPETITION cursor: 1e-3 → 1e-4 at 46,300
completed R batches = 100 R exposures. Under 1:2:3 that means N has had ~200
and C ~317 exposures at 1e-3 before the drop, whereas the isolated recipes
dropped at 100 exposures of the task's own clock, and isolated H512 naming
degraded under sustained 1e-3 (.840 @100 → .743 @300).

**Prespecified reading:** naming rising then decaying before u=100, with
rising N training CE, and recovering after the drop, is an **LR-schedule
signature** — intervention (d) below. It must NOT be read as evidence that
high-capacity joint learning is intrinsically incompatible, nor as
interference.

## Failure → next intervention

| observed signature | next intervention | why |
|---|---|---|
| joint N and/or C far below their same-width isolated capability; task-locked oscillation or update-linked destabilisation; not reproducible in isolation at the same width/LR; not specifically LTM-repetition collapse | **(a) grouped RN\|C optimizer banks** | capacity ruled out by the probes; shared Adam state is the remaining coupling channel |
| N and C acquire well; FULL and WM repetition hold; **LTM-only repetition rises then progressively collapses** | **(b) 2:2:3** | the historical preservation-vs-acquisition signature that 2:2:3 demonstrably delayed |
| R at ceiling; N at/near ceiling; C the isolated residual at ~99.x% with a small stable error set, small margins, nearest-neighbour confusions **in the joint model's own audit** | **(c) C hard-negative / margin** | the objective has stopped pushing on resolved-but-close competitors; "more training" already falsified by the isolated plateau |
| width-specific eval collapse with **rising training CE** in the same task, reproducing the isolated instability at that width/LR; or clipping monopolised by one task | **(d) LR clock / stage or loss weight** | the instability is within-task and schedule-caused, so (a) and (b) would treat the wrong mechanism |

**Discriminator:** could this failure occur in the isolated same-width model
at the same LR? If yes → (d). If it requires the other tasks' presence →
(a) or (b), by the LTM-collapse criterion.

C hard-negative/margin refinement is **fully deferred**: adding it now would
confound the capacity question, and joint training may reshape the geometry so
that the isolated 38-error set is not the joint residual.

## What this block can and cannot establish

**Can:** a paired, seed-replicated causal effect of ventral width (bundle,
including its optimization consequences) on joint R/N/C attainment under the
fixed historical recipe; a coarse 256-vs-512 dose–response; and, if H512
succeeds, an existence proof that the simple shared-optimizer 1:2:3 regime
**suffices** at adequate capacity — i.e. the historical rescues are not
necessary at high capacity.

**Cannot:** separate encoder from decoder; separate "capacity" from parameter
count or optimization geometry (no matched-parameter control — say
"width/representational capacity"); establish that the rescues were
unnecessary *at H128* (no H128 arm tonight); disprove task incompatibility for
all recipes; or support any claim of algorithmic fidelity to Ueno.

## Terminology

Use "Ueno-inspired 1:2:3 presentation-frequency regime" or "modernized
Lichtheim3 analogue of the Lichtheim 2 developmental presentation schedule".
Ueno interleaved **per-item online updates** (1/2/3 presentations per word per
epoch, randomized, weights updated after each item); Lichtheim3 interleaves
**task-level minibatch steps** (batch 64) under AdamW in the same 1:2:3
frequency ratio, on ~29.6k English words with 300-d GloVe semantics and a
retrieval-based comprehension objective.

Avoid: "faithful implementation of Ueno's training algorithm", "exact Ueno
optimizer", "exact Lichtheim 2 training regime", or bare "Ueno schedule".
The fidelity is to the experience/presentation principle, not the algorithm.
