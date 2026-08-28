# LTM success on novel pseudowords: length dominates, lexical proximity contributes little

Population: the 391 `NOVEL_PSEUDOWORD` items, LTM route only, 4 seeds
(1,564 seed x item observations). Everything plotted is a reaggregation of
already-validated mechanism tables; no model was loaded and no new feature was
created.

**Panel A.** LTM's exact-repetition rate falls steeply and monotonically with
target length. Open markers are the four seeds; the line is their mean; item
counts per length are printed beneath.

**Panel B — the pre-registered length control.** Within each exact length, items
are split at the **within-stratum median** lexical confidence (top-1 cosine
between the item's `s_hat` and the frozen GloVe bank), and the success-rate
difference between the high and low halves is plotted. The split is computed
inside each (seed, length) cell, so it cannot import the length-confidence
relationship.

**Result, stated as found.** The within-length difference is small and not
consistent across seeds: +0.046, +0.025, +0.036 and **-0.011** for seeds 19, 20,
21 and 22, i.e. a mean of about +0.024 with one seed of the opposite sign. Set
against the length effect in panel A - from 0.95 down to 0.47, roughly 48
percentage points - a two-point, sign-unstable difference is not a competing
explanation.

Note also what the control did **not** do here: the stratified and the marginal
(uncontrolled) differences are nearly identical, because in this population
confidence and length are only weakly related (mean length of the high- and
low-confidence halves differs by less than half a phoneme). So confidence is a
weak predictor of success, and it was not a strong one being propped up by
length either. Both statements matter; neither is the one we set out to find.

**Confidence and gate are one variable, not two**
(`gate = sigmoid(2 x (confidence - 0.7))`), and are never counted as two pieces
of evidence.

**What this does not show.** Nothing here is causal: the groups are defined by
the outcome. A small within-stratum difference does not prove that no lexical
influence exists - only that the one validated lexical-proximity measure
available does not predict success to any material degree, with or without the
length control. With four seeds and a two-point effect, this analysis cannot
resolve whether the true effect is zero or merely small.
