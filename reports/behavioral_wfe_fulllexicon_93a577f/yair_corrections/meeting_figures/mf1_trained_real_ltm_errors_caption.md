# Which genuinely trained real words does LTM still fail on?

Population: the **671 `TRAINED_REAL_EXACT` items** — WFE real words that were
in the training lexicon with the same phonological form. Route: **LTM only**.
Grey points are the 659 items every seed repeats correctly; red
points are the **12 items that fail in at least one of the four canonical
seeds**, accounting for **14 seed x item error events** in total. Marker
size encodes the number of failing seeds (1 or 2; no item fails in more than 2). Grey dashes are the median Zipf of the correctly repeated items
at that exact length, taken verbatim from the validated table.

**The two signatures are both visible.** The failures sit at the long end —
6 of 12 are 9 phonemes, and no trained real word of 3 or 4
phonemes ever fails — and within each length they sit low in the frequency
distribution: 11 of 12 fall below their same-length median, with a
mean within-length Zipf percentile of 0.188 against a length-matched
permutation null of 0.467 (p = 0.00015, B = 20,000).

*lieutenant* is the single exception, at the 67th percentile of its length
stratum.

**Scope.** FULL and WM make **no** errors at all on this stratum; these are
LTM-only failures. This is descriptive: with 12 items no model is fitted,
and while the frequency-weighted training sampler makes the frequency
association mechanistically unsurprising, nothing here is a causal
demonstration.

Every plotted value and every number in the annotation box is read from
`residual_trained_real/tables/` and is asserted against those tables when the
figure is drawn.
