# WFE behavioral results — brief for Yair

Cohort `fulllexicon_cohort_93a577f`, four seeds (19, 20, 21, 22), three routes.

## 1. Bottom line

**Yes — the ventral (LTM) route carries the length effect, but only for
unfamiliar phonological forms.**

For **trained real words**, FULL and WM make **no errors at all** and LTM shows
only a very weak length slope. For **novel pseudowords and untrained real
words**, LTM's slope is an order of magnitude larger while WM stays much more
robust and FULL stays near ceiling. The primary contrast LTM − WM is positive in
**all four seeds**.

The precise claim is *the ventral route shows a large length effect for
unfamiliar forms* — **not** "LTM always has a length effect".

## 2. What was evaluated

1,200 WFE items × 3 routes × 4 checkpoints = 14,400 rows, deterministic decoding
with a forced-length readout. Two regimes, never mixed: the **faithful** 1,200
items with the paper's original labels, and the **clean** 1,062 items
(671 trained real + 391 novel pseudo) that the exposure audit certifies. The
distinction matters — 122 of the paper's "real" words were never in our training
lexicon.

## 3. Main figure recommendation

Seven figures, in this order: **F1** length by route → **F2** slopes and LTM−WM
→ **F3** serial position → **F4** error taxonomy → **F5** premature EOS →
**F6** frequency → **F7** feature importance. If you want three for a talk:
**F1, F2, F4**.

## 4. Main numerical results

| quantity | value |
|---|---|
| LTM pseudoword length slope | 0.197 – 0.256 edit ops per phoneme (all 4 seeds) |
| WM pseudoword length slope | 0.007 – 0.037 |
| FULL pseudoword length slope | 0.003 – 0.020 |
| LTM real-word length slope | 0.0007 – 0.0177 |
| FULL / WM real-word slope | **exactly 0** (ceiling, every item, every seed) |
| **LTM − WM, pseudowords** | **+0.246, +0.183, +0.184, +0.205** |
| untrained real words under LTM | mean edit distance 0.549 vs 0.601 novel pseudo, 0.024 trained real |
| LTM long-pseudoword operations / item | subs 0.574, dels 0.245, ins 0.140 |
| premature-EOS events | 87 total — LTM 82, FULL 3, WM 2; **none on trained real words** |
| LTM Zipf slope (trained real) | −0.0130, CI [−0.0288, −0.0027], negative in all 4 seeds |
| adapted FI leaders | route and lexicality/exposure; length third; morphology ≈ 0 |

## 5. What explains part of the effect

- **Unfamiliarity, not lexical status.** Untrained real words behave like
  pseudowords. Exposure is what tracks difficulty.
- **Late-position errors.** The serial-position profile rises sharply toward the
  end of long unfamiliar items.
- **Substitutions, mostly.** Roughly 60 % of LTM long-pseudoword operations are
  substitutions, with deletions and insertions also elevated.
- **Premature stopping — partially.** Early stops are LTM-specific and rise with
  length, but they occur on only ~22 % of erroneous LTM pseudoword items.

## 6. What does not explain it

- **Morphology.** No robust main effect, no length interaction, no route
  interaction; last-ranked in feature importance in every seed.
- **Premature EOS alone.** 82 events against 874 edit operations, and 107 of 189
  deletion-bearing items have no observed early stop. It is a real phenomenon,
  not a complete explanation.
- **Frequency.** Present and consistent in direction, but small and
  sparse-error-limited.

## 7. What remains open

The behavioural data **constrain but do not localize** the mechanism. They do
not tell us whether the length dependence originates in the LTM encoder, the
compressed semantic representation, the LTM decoder, autoregressive
accumulation, or the gate. That needs hidden-state work — a separate project.
Two instrumentation limits also bound any follow-up: the forced-length readout
hides terminal insertions and all on-time/late EOS timing, and on the clean set
lexicality and exposure cannot be told apart.

## 8. Files to inspect

```
Main length figure        reports/.../final_release/figures/main/F1_yair_clean_length_by_route.png
Slopes / LTM−WM contrast  reports/.../final_release/figures/main/F2_yair_clean_length_slopes.png
Serial position           reports/.../final_release/figures/main/F3_yair_clean_serial_position.png
Error taxonomy            reports/.../final_release/figures/main/F4_clean_error_taxonomy_by_route.png
Premature EOS             reports/.../final_release/figures/main/F5_premature_eos_by_route.png
Adapted feature importance reports/.../final_release/figures/main/F7_clean_adapted_factor_importance.png
Executive summary         reports/.../final_release/executive_summary.md
Mechanism handoff         reports/.../error_taxonomy/length_effect_mechanism_handoff.md
```

(`reports/...` = `reports/behavioral_wfe_fulllexicon_93a577f/`. Each figure also
ships as PDF and SVG, with a caption beside it in `captions/main/`.)

## 9. Suggested next discussion questions

1. Is the LTM-versus-WM asymmetry for unfamiliar forms the result you want to
   report, or do you want the exposure story foregrounded instead?
2. Does the untrained-real-word finding deserve its own figure rather than
   living in the exposure strata?
3. Is it worth extending the readout horizon so that on-time and late EOS become
   observable? That would be a new evaluation, separately provenanced.
4. Do you want SSP / sonority (A19) started, or does it stay deferred?
5. Which of the eight open questions in the mechanism handoff should the
   hidden-state project take first?
