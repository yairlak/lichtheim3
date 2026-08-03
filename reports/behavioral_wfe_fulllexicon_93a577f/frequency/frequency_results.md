# Sprint 3 — word frequency: results

Specification frozen in `frequency_analysis_spec.md` **before any result below
was computed or inspected**. Sign conventions as frozen: a **negative Zipf
slope means higher-frequency words have fewer errors**; a **positive low−high
contrast means low-frequency words are harder**.

## 1. Scope

Word frequency only, on outputs validated in Sprints 0–2. Frequency is defined
only for real words; no pseudoword is assigned a Zipf value or enters any model
here. No checkpoint inference, no feature importance, error taxonomy, SSP or
morphology recomputation. Sprint-1 and Sprint-2 results are untouched.

## 2. Frequency distribution and confounds

Primary set (`TRAINED_REAL_EXACT`, n = 671): Zipf mean 4.113, SD 0.901, range
2.87–7.01, quartiles 3.29 / 4.23 / 4.71; 292 low and 379 high. Labels were
**recomputed from Zipf rather than trusted**: 0 mismatches against the stored
`frequency_class`, and 0 items in the excluded 3.5–4.0 interval.

Confounds present and reported, not corrected:

- **Zipf × length r = −0.19** — longer trained words are somewhat lower
  frequency. Mean Zipf falls from 4.66 at length 3 to 3.91 at length 9.
- **Zipf × morphology r = 0.05** — negligible (complex 4.064, simple 4.154).
- **Zipf × exposure is substantial**: trained words average 4.113 whereas the
  122 untrained real words average **3.311** with only 15 high-frequency items.
  Exposure and frequency are strongly entangled in the WFE, which is exactly
  why the untrained set is analysed separately.

All 12 frequency × length cells are `OK` (n ≥ 20). No item was matched,
rebalanced or excluded.

## 3. Primary trained-real continuous Zipf analysis

| Route | seed 19 | seed 20 | seed 21 | seed 22 | mean | 95 % CI | status |
|---|---|---|---|---|---|---|---|
| FULL | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0.000, 0.000] | `ALL_ZERO_OUTCOME` |
| WM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | [0.000, 0.000] | `ALL_ZERO_OUTCOME` |
| **LTM** | **−0.0180** | **−0.0091** | **−0.0165** | **−0.0083** | **−0.0130** | **[−0.0288, −0.0027]** | `OK` |

FULL and WM produce **zero errors and zero total edit distance** on all 671
items in every seed. Their slopes are structurally zero by the frozen ceiling
policy, the logistic fit was not attempted, and **no absence of frequency
encoding is claimed for those routes** — the measure simply cannot identify an
effect there.

LTM shows a **negative slope in all four seeds** with a bootstrap interval
excluding zero: higher-frequency trained words are reproduced with fewer errors
by the ventral route. Total LTM evidence is nonetheless thin: 14 erroneous
items and 36 total edit distance across 4 seeds × 671 items.

## 4. High-versus-low description (secondary)

Low−high contrast, raw edit distance (positive ⇒ low-frequency harder), 292 low
vs 379 high:

| Route | 19 | 20 | 21 | 22 | mean |
|---|---|---|---|---|---|
| FULL | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| WM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| LTM | +0.0245 | +0.0171 | +0.0343 | +0.0171 | **+0.0232** |

Word-error counterpart for LTM: +0.0076, +0.0103, +0.0103, +0.0137 (mean
+0.0105). Same direction as the continuous analysis in every seed.

## 5. Frequency × route

Both orientations, as frozen. `raw_route_slope_difference = slope_A − slope_B`;
`frequency_benefit_route_difference = (−slope_A) − (−slope_B)`, positive
meaning route A shows the **larger frequency benefit**.

| Contrast | raw difference (mean, range) | benefit difference (mean, range) |
|---|---|---|
| LTM − WM | −0.0130 [−0.0180, −0.0083] | **+0.0130** [+0.0083, +0.0180] |
| LTM − FULL | −0.0130 [−0.0180, −0.0083] | **+0.0130** [+0.0083, +0.0180] |
| FULL − WM | 0.0000 | 0.0000 (both at ceiling) |

Bootstrap on the raw LTM−WM difference: −0.0129 [−0.0288, −0.0027]. The sign is
identical in all four seeds. The FULL−WM comparison is uninformative because
both sides are at ceiling.

## 6. Frequency × length

Length-adjusted model with both predictors centred, so the Zipf main effect is
the slope **at mean length**:

| Route | Zipf main effect | length coefficient | Zipf × length |
|---|---|---|---|
| FULL | 0.0000 | 0.0000 | 0.0000 |
| WM | 0.0000 | 0.0000 | 0.0000 |
| LTM | −0.0179 | +0.0139 | **−0.0185** |

The LTM interaction is negative in all four seeds (−0.0339, −0.0160, −0.0219,
−0.0022): the frequency benefit is larger for longer trained words. The
magnitude varies about fifteen-fold across seeds, and seed 22's value is close
to zero, so this is consistent in direction but not in size.

## 7. Frequency × lexical confidence

Zipf slope on lexical confidence — the **top-1 cosine similarity between the
encoded form and the semantic bank**, not a word probability and not
phonological similarity:

+0.0931, +0.0906, +0.0899, +0.0919 across seeds; mean **+0.0914**, bootstrap
**[+0.0866, +0.0962]**. Higher-frequency trained words sit closer to their
semantic-bank neighbourhood. This is the tightest estimate in the sprint.

## 8. Frequency × gate

Zipf slope on the gate: +0.0459, +0.0447, +0.0443, +0.0453; mean **+0.0450**,
bootstrap **[+0.0427, +0.0474]**.

The gate is a deterministic monotonic transform of lexical confidence
(`g = sigmoid(2.0·(confidence − 0.7))`), so this is **the same finding
re-expressed, not independent evidence**. The ratio ≈ 0.49 is what the sigmoid
slope predicts in this confidence range.

## 9. Pronunciation-variant sensitivity

Adding the 7 pronunciation-variant items (671 → 678) leaves the primary result
essentially unchanged: LTM slope −0.0126 (vs −0.0130), same sign in all four
seeds, range [−0.0179, −0.0080]. FULL and WM remain `ALL_ZERO_OUTCOME`.

## 10. Untrained-real sensitivity (n = 122, analysed separately)

| Route | 19 | 20 | 21 | 22 | mean | errors |
|---|---|---|---|---|---|---|
| FULL | −0.0304 | −0.0148 | −0.0071 | −0.0019 | −0.0135 | 8 |
| WM | −0.0155 | −0.0226 | 0.0000 | −0.0297 | −0.0170 | 7 |
| LTM | −0.2651 | −0.1751 | −0.2750 | −0.1155 | **−0.2077** | 109 |

Untrained real words are the one condition where FULL and WM leave the floor at
all, and the LTM frequency slope is roughly sixteen times steeper than for
trained words. Two cautions make this **descriptive only**: the set has just
**15 high-frequency items against 107 low-frequency**, and untrained words are
on average a full Zipf unit rarer than trained ones (3.31 vs 4.11). Frequency
and exposure cannot be separated here. These items are never merged with the
trained set in the primary figure.

## 11. Exact-zero sensitivity

Restricting to seeds 19, 20 and 22 leaves every conclusion intact: LTM primary
slope −0.0118 (vs −0.0130 for four seeds); sensitivity-set LTM −0.0115 (vs
−0.0126). Seed 21 is present in every table and figure and is not an outlier.

## 12. Word-error secondary results

`tables/trained_real_frequency_word_error.tsv`,
`tables/frequency_word_error_model_status.tsv`.

FULL and WM: **`ALL_ZERO_OUTCOME`** — zero word errors in every seed, so no
finite logistic MLE exists; the fit was not attempted and no interval was
manufactured. LTM: **`SPARSE_ERROR_LIMITED`** — 14 erroneous items across all
four seeds, too few for a stable logistic fit; descriptive high/low rates and
the bootstrap contrast are reported instead (LTM low−high word error +0.0105,
same sign in all four seeds).

**`FIGURE_NOT_CREATED_DUE_TO_CEILING_OR_SPARSE_ERRORS`.** No word-error figure
was produced: two of three routes are at exact floor and the third rests on 14
errors, so a figure would display sparsity rather than a pattern, and it adds
nothing beyond the raw-edit-distance panels.

## 13. Ceiling and sparse-error limitations

- FULL and WM are at **exact ceiling** on trained real words. Their zero slopes
  are structural. Under the frozen policy this licenses no claim that those
  routes lack frequency information — only that this measure cannot detect one.
- LTM's trained-word result rests on **14 erroneous items**; the direction is
  consistent across seeds but the magnitude is small and the evidence thin.
- The untrained-real comparison has a **15 vs 107** high/low imbalance and
  confounds frequency with exposure.
- Zipf correlates −0.19 with length in the primary set; the length-adjusted
  model is reported for that reason, but no matching was performed.

## 14. Robust findings

| Finding | Category |
|---|---|
| LTM continuous Zipf slope on trained real words is negative — higher frequency, fewer errors (mean −0.0130, CI [−0.0288, −0.0027], same sign in all 4 seeds, stable under both sensitivity analyses) | **ROBUST** |
| Lexical confidence increases with Zipf (mean +0.0914, CI [+0.0866, +0.0962], all 4 seeds) | **ROBUST** |
| Gate increases with Zipf (mean +0.0450, CI [+0.0427, +0.0474]) — linked restatement of the confidence result, not independent | **ROBUST (linked)** |
| LTM shows a larger frequency benefit than WM and than FULL (+0.0130, same sign in all 4 seeds) — with the caveat that WM and FULL are at ceiling | **ROBUST, CEILING-QUALIFIED** |

## 15. Consistent but uncertain findings

| Finding | Category |
|---|---|
| LTM low−high contrast on trained words (+0.0232 edit distance, +0.0105 word error) | `CONSISTENT_BUT_SMALL` |
| LTM Zipf × length interaction negative in all 4 seeds, but varying ~15× in magnitude | `CONSISTENT_BUT_SMALL` |
| Untrained-real LTM slope −0.2077, far steeper than trained | `DESCRIPTIVE_ONLY` (15 vs 107 imbalance; frequency confounded with exposure) |
| Untrained-real FULL and WM negative slopes | `SPARSE_ERROR_LIMITED` (8 and 7 errors) |

## 16. Non-estimable findings

| Finding | Category |
|---|---|
| FULL frequency slope, trained real words | `CEILING_LIMITED` / `ALL_ZERO_OUTCOME` |
| WM frequency slope, trained real words | `CEILING_LIMITED` / `ALL_ZERO_OUTCOME` |
| FULL−WM route contrast | `CEILING_LIMITED` (both sides zero) |
| Word-error logistic model, FULL and WM | `NON_ESTIMABLE` / complete separation |
| Word-error logistic model, LTM | `SPARSE_ERROR_LIMITED` |

## 17. Exploratory observations

Recorded as observations only, not claims, and not to be carried forward
without further work:

- The confidence slope (+0.091) is an order of magnitude tighter across seeds
  than any behavioural slope, which is unsurprising given that confidence is a
  continuous quantity measured on every item whereas edit distance is zero for
  almost all of them.
- The frequency benefit and the length effect appear to compound in LTM (the
  negative Zipf × length interaction), but seed 22 nearly nulls it.
- Untrained real words are simultaneously rarer and unseen, so their steeper
  slope cannot be attributed to frequency alone from these data.

No causal claim is made about why frequency affects the ventral route, and no
zero slope is presented as proof of absence.

## 18. Files generated

```
frequency/
  frequency_analysis_spec.md            frozen before results
  frequency_results.md                  this file
  _control/  frequency_preflight.json, frequency_analysis_spec.json,
             frequency_figure_manifest.json
  tables/    trained_real_frequency_distribution.tsv
             trained_real_frequency_by_length.tsv
             trained_real_frequency_by_morphology.tsv
             frequency_cell_counts.tsv
             frequency_exposure_comparison.tsv
             trained_real_frequency_word_error.tsv
             frequency_word_error_model_status.tsv
  primary/
    figures/ trained_real_frequency_by_route.{png,pdf,svg} + caption
    tables/  trained_real_frequency_slopes.tsv
             trained_real_frequency_route_contrasts.tsv
             trained_real_frequency_bootstrap.tsv
             trained_real_high_low_descriptives.tsv
             trained_real_zipf_length_models.tsv
  gate_confidence/
    figures/ frequency_confidence_gate.{png,pdf,svg} + caption
    tables/  frequency_confidence_slopes.tsv
             frequency_gate_slopes.tsv
             frequency_confidence_gate_bootstrap.tsv
  sensitivity/ pronunciation_variant_sensitivity.tsv
               exact_zero_seed_sensitivity.tsv
               untrained_real_frequency_slopes.tsv
               untrained_real_high_low.tsv
               faithful_all_real_frequency_description.tsv
  validation/  frequency_validation.json, frequency_test_log.txt,
               frequency_output_inventory.tsv, frequency_outputs.sha256,
               frequency_diff_review.md
```
