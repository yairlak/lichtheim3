# Lichtheim3 final joint results — interpretation

Companion to `FINAL_JOINT_RECIPE.md`. Numbers quoted here are the values
reported from the completed runs; the authoritative versions are produced by
`build_final_report.py` from the archived `metrics.tsv`. **Any number in this
document that disagrees with that output is wrong and the output wins.**

## OBSERVATION — what the data directly show

**O1. Ventral capacity.** In the paired from-scratch joint 1:2:3 block
(4 seeds, everything identical except encoder/decoder width), at u500:
H256 reached C top-1 ≈ .578, H512 ≈ .982. The effect was the same sign in
4/4 seeds and enormous relative to the between-seed spread. R was unaffected
(paired Δ ≈ 0, not even consistent in sign); N improved modestly; LTM-only
repetition improved (~+.17).

**O2. Dorsal capacity.** In the isolated WM-route probe at ~200 exposures:
WM128 lexical ≈ .999865 / pseudoword ≈ .980; WM256 ≈ .999459 / .950;
WM512 ≈ .999527 / .9475. Increasing dorsal width did **not** help, and the
pseudoword readout was mildly worse at larger widths. *Status: recomputed
from `metrics.tsv` by `build_final_report.py`; the figure is drawn
separately because this is a single-task isolation probe and its numbers are
not comparable to joint numbers.*

**O3. Continuation to u750.** C ≈ .9917 (~232 errors mean), R and N near but
not at ceiling, LTM-only repetition ≈ .83.

**O4. Global late-LR pilot (u750 → u850, same source).** Lowering all three
task LRs from 1e-4 to 3e-5 gave: naming exactly 1.0 in **4/4 seeds at all
four milestones**; repetition errors 10.25 → 6.50 (mean); LTM-only
repetition .827 → .939 (+11.1 pp); C statistically indistinguishable
(186.25 vs 188.50 mean errors, per-seed deltas +9/−10/+9/+1, mixed in sign).

**O5. Task-specific LR pilot (u850 → u1200, same source).** Raising **only**
the comprehension LR to 1e-4:

| | all-3e-5 | C-high | paired Δ |
|---|---|---|---|
| C errors | 137.75 | 123.75 | −10, −14, −21, −11 (**4/4 same sign**) |
| N errors | 0 | 0 | — |
| R errors | 3.25 | 4.50 | +1, +2, +1, +1 (4/4 same sign) |
| LTM-only repetition | .934852 | .869518 | ≈ −6.5 pp |

**O6. The u750 C residual was not a fixed core.** u600→u750 within-seed
retention .5868; four-seed sentinel core 27 → 16 → 12; naming residuals
churned completely (intersection 0 within every seed); no mathematically
unavoidable C errors (the bank contains zero duplicate GloVe vectors).

## INTERPRETATION — supported but inferential

**I1. Ventral width was the dominant bottleneck for semantic tasks, and the
bottleneck is route-specific.** The capacity manipulation moved C enormously
and R not at all, while the dorsal manipulation moved nothing. Read together,
this supports a route-specific capacity account: systematic
phonology→phonology mapping is cheap; semantic transformations are expensive.

**I2. The late-stage failure mode was optimization, not capacity or
objective conflict.** At u750 the frontier was still moving, so "more of the
same" was warranted; the global LR drop then converted a fluctuating
near-ceiling regime into a stable one (naming exactly 1.0, R errors down,
LTM up) without costing C. That is the signature of a step size too large for
the final approach.

**I3. There is a genuine C-speed vs LTM-repetition tradeoff, and it is
localised to the ventral route.** C-high buys ~14 fewer C errors and pays
~6.5 pp of LTM-only repetition, with R errors up slightly and WM-only
repetition unchanged at ceiling. Because the *comprehension* LR is the only
thing that changed, and because the damage appears in the **LTM** route while
the **WM** route is untouched, the most parsimonious reading is
**within-ventral-route interference**: larger comprehension updates move the
shared encoder/semantic surface faster than the ventral production pathway
stays co-adapted to it.

**I4. Why the margin/hard-negative intervention was correctly not triggered
at u750.** The preregistered trigger required a *small, stable, near-tie*
residual. At u750 the residual was large (~232), moving (41% turnover per
100u), and only ~40% within 0.01 margin. A margin term would have been fitted
against a set that was about to dissolve, and would have confounded the
capacity result.

## NOT ESTABLISHED — claims we cannot make

- **Not** that H512 is optimal, or that 512 is a threshold. We tested
  {256, 512} jointly and {128, 256, 512} in isolation; no matched-parameter
  control exists, so "capacity" here means **width**, not information content.
- **Not** that encoder and decoder contribute separately. They were scaled
  **together**; this block measures a ventral *bundle*.
- **Not** a mechanism for the C/LTM tradeoff. I3 is the most parsimonious
  reading of a correlational endpoint difference, not a demonstration.
  Establishing it would need, e.g., a gradient-interference measurement or a
  targeted ablation — neither was run.
- **Not** that 3e-5 is optimal. Two values were compared, not a schedule.
- **Not** that the historical H128 joint failures were *caused only* by
  under-capacity: no H128 arm was run in this block, so that contrast rests
  on archived runs under a partly different recipe.
- **Not** any claim of algorithmic fidelity to Ueno (see the recipe doc).
- **Not** that 100/100/100 was achieved: the 5-milestone strict ceiling has
  never fired. R and C remain short.

## Open questions

1. Does the C/LTM tradeoff have an operating point that keeps both — e.g. a
   C LR between 3e-5 and 1e-4, or extra repetition rehearsal to offset it?
2. Is the ~124–138 residual C error set finally stabilising, or still
   turning over? (The u1200 per-item audit answers this and has not been run.)
3. Would extra repetition allocation (2:2:3) restore LTM under C-high without
   costing C — i.e. is the tradeoff a *rehearsal* problem rather than an LR
   problem?
4. Does R reach exact zero under any of these regimes? It is at 3–5 errors.
