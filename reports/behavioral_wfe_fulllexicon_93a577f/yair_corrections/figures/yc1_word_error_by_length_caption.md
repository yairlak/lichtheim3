# Whole-word error rate by exact target length and route

**Primary metric: whole-word error rate** (1 - exact match), not edit distance
and not a slope. Population: `LICHTHEIM_CLEAN`, 671 `TRAINED_REAL_EXACT` (red)
and 391 `NOVEL_PSEUDOWORD` (blue). Open markers are the four individual seeds
(19 circle, 20 square, 21 triangle, 22 diamond); the solid line is the
across-seed mean; the band is the frozen `cell_mean_bootstrap` 95 % interval
(B = 10,000, seed 20260730, seeds resampled then items). Item counts per
lexicality x length bin are printed under each panel. No smoothing is applied.

WFE contains no 6-phoneme items by construction; the gap on the x-axis is real and is not interpolated across.

**What the panels show.** FULL and WM sit at the floor almost everywhere: zero
whole-word errors for real words at every length, and zero for pseudowords up to
length 8, with the first non-zero values appearing only at length 9 (FULL 0.057,
WM 0.072). LTM is the only route with a substantial length effect, and it is
confined to pseudowords: 0.074 at length 3 rising monotonically to 0.534 at
length 9, while LTM real words stay at or near zero (maximum 0.027 at length 9).

Reading the same data as mean edit distance compresses this into a shallow slope
and hides the fact that two of the three routes are at ceiling. That is why word
error rate is the primary axis here.

Mean raw edit distance is carried in
`tables/word_error_by_length_seed.tsv` and
`tables/word_error_by_length_summary.tsv` as
`mean_raw_edit_distance_severity_only`. It describes **how severe** a failure is
once a word is already wrong; it is not a second measure of how often the model
fails.

**Limitation.** Inside `LICHTHEIM_CLEAN`, lexicality and training exposure
coincide exactly (real == trained-exact, pseudo == novel), so this figure cannot
separate the two. The faithful-population companion table
`tables/word_error_by_length_faithful_companion.tsv` shows the different mixture
the faithful labels produce; the two populations are never pooled.
