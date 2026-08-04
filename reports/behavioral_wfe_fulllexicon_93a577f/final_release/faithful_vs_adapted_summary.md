# Faithful versus adapted — release-level summary

Two parallel analysis families run through this project. They answer **different
questions on different item populations** and are never pooled, averaged,
differenced or plotted on a common quantitative scale.

**The adapted analysis is not a correction of the faithful analysis.** The
faithful family is a valid replication on its own terms; the adapted family
answers a question the faithful design cannot address.

## Structured comparison

| | **FAITHFUL** | **ADAPTED** |
|---|---|---|
| scientific purpose | replication fidelity against Dager et al. | the Lichtheim3-specific scientific question |
| item population | `FAITHFUL_WFE_ALL`, 1,200 items | `LICHTHEIM_CLEAN`, 1,062 items |
| lexicality labels | **original WFE/SWP source labels** | Lichtheim3 exposure-audited labels |
| exposure control | **none** — 122 source-real items were never trained, 9 source-pseudo items collide with the training lexicon | items with ambiguous exposure excluded by the frozen Sprint-1 definition |
| what "Real" means | a source label | `TRAINED_REAL_EXACT` (671): trained, same phonological form |
| what "Pseudo" means | a source label | `NOVEL_PSEUDOWORD` (391): phonological form absent from training |
| route scope | **FULL only** | FULL, WM and LTM |
| frequency handling | not analysed | Zipf on trained real words only; **never assigned to pseudowords**, never imputed |
| split policy | historical 80/20, `random_state = 42` (A11) | 80/20 **grouped by `item_id`**, identical split reused across seeds and models (A15) |
| uncertainty policy | per-seed values; no bootstrap | hierarchical bootstrap (Sprints 1–4); seed-resampling interval over four checkpoints (A15) |
| **interpretation permitted** | how faithfully Lichtheim3 reproduces the published stimulus-level pattern | whether the ventral route carries the length effect, and how training exposure modulates it |
| **interpretation prohibited** | reading source Real/Pseudo as trained/novel; any route claim (there is no route factor) | claiming that lexicality is separated from exposure; pooling with faithful values |

## What each family contains

**FAITHFUL** — A09 (Figure 2A), A10 (Figure 2C), A11 (Figure 2B feature
importance), A13 (morphology × length), A16 (Figure 8A error types). All FULL
route, all 1,200 source-labelled items, all preserving the original frozen
parameters: Ridge α = 1.0, 80/20 `random_state = 42`,
`permutation_importance(n_repeats = 100, random_state = 42)`, no interactions,
no p-values, morphology line styles hard-coded, Figure 2C by zip-mismatch with
**no Levenshtein alignment**.

**ADAPTED** — A04–A08 (length, slopes, serial position, gate), A12 (morphology),
A14 (frequency), A15 (feature importance), A17 (error taxonomy), A18 (premature
EOS). Clean set or exposure strata, all three routes, exposure-audited labels.

## Where the two visibly diverge

Feature importance is the clearest case. **A11 places length first in all four
seeds**; **A15 places lexicality/exposure above length in all four**. Four
design differences fully account for this and none of them makes either analysis
wrong: A11 includes the 122 untrained "real" items that dilute its lexicality
factor; A11 has no route factor while route is one of A15's two leading factors;
A15's split is grouped by item and therefore stricter; and A15 permutes raw
factors, rebuilding every derived column, where A11 uses the historical
column-level procedure.

This is a description of two separate analyses. **It validates neither against
the other and carries no causal reading.**

## The one-line rule

*Faithful answers "did we reproduce the paper?"; adapted answers "what does this
model do?". Report both, label both, mix neither.*
