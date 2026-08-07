# Errors on the 800 WFE source-real items

Population: all 800 items the WFE stimulus set labels *real*, in
`FAITHFUL_WFE_ALL`, across 4 seeds and 3 routes (9,600 seed x route x item
rows). Exposure strata use the neutral grey palette; red and blue are reserved
for lexicality elsewhere and are deliberately not used here.

**Panel A.** Errors are overwhelmingly concentrated in LTM: 126 error events on
70 unique items, against 8 events on 5 items for FULL and 7 events on 5 items
for WM. Both counts are reported because they answer different questions -
events count seed x item failures, unique items count how much of the stimulus
set is ever affected.

**Panel B.** For FULL and WM, **100 %** of source-real errors come from
`UNTRAINED_REAL` - words the stimulus set calls real but which were never in the
Lichtheim3 training lexicon. For LTM, 86.5 % come from `UNTRAINED_REAL`
(109 events / 57 items out of 122 stratum items), 11.1 % from
`TRAINED_REAL_EXACT` (14 events / 12 items out of 671) and 2.4 % from
`TRAINED_REAL_PRON_VARIANT` (3 events / 1 item out of 7).

**Errors do remain in `TRAINED_REAL_EXACT`, but only in LTM**, at a 0.52 %
event rate. FULL and WM make no errors at all on trained-exact real words.

**Panel C.** Failures are only partly item-consistent. In LTM, 53 % of erroneous
items fail in exactly one seed and only 7 items (10 %) fail in all four. For
FULL and WM no item fails in all four seeds.

**Limitations.** This is descriptive. The `TRAINED_REAL_PRON_VARIANT` stratum has
7 items and the FULL/WM error sets have 5 items each - far too small for any
claim beyond enumeration. Association with length and low frequency is reported
in `tables/faithful_real_error_descriptive_bins.tsv` as observed rates and is
**not** adjusted for the exposure confound: inside the faithful real label,
untrained words are both rarer and differently distributed over length, so the
apparent frequency and length effects partly re-express exposure. No confirmatory
model is fitted in this pass.

Exhaustive per-event listing with the literal source columns, including EOS
class derived with the frozen `eos_diagnostics.classify_eos`:
`tables/faithful_real_error_events.tsv`.
