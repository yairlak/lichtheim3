# WFE behavioral analysis — executive summary

Cohort `fulllexicon_cohort_93a577f`, training commit `93a577f`, seeds 19, 20,
21, 22 (epochs 155, 130, 145, 140). Every number below comes from a validated
table produced by Sprints 1–5; this document adds no new analysis.

---

## 1. Research question

Yair's question: **is it the ventral (LTM) pathway rather than the dorsal (WM)
pathway that shows a length effect in word repetition?** The Lichtheim3
dual-route model is compared against the Word Feature Evaluation (WFE) stimulus
set of Dager et al., under a faithful replication and an exposure-audited
adaptation.

## 2. Evaluation design

Four checkpoints, three routes — FULL (gated), WM (dorsal only), LTM (ventral
only) — under one decoding convention: deterministic autoregressive with a
**forced-length readout**, no noise, no teacher forcing. 1,200 WFE items ×
3 routes × 4 seeds = 14,400 evaluated rows. Primary metric: raw Levenshtein
edit distance over ARPAbet tokens; secondary: word error. All four seeds are
primary and **seed 21 is never excluded**; seeds 19/20/22 form an exact-ceiling
sensitivity set reported separately. Uncertainty for Sprints 1–4 is a
hierarchical bootstrap (seeds, then items within stratum; B = 10,000, random
seed 20260730, 95 % percentile).

## 3. Faithful versus clean regimes

`FAITHFUL_WFE_ALL` (1,200) keeps the original source Real/Pseudo labels and
answers **replication fidelity**. `LICHTHEIM_CLEAN` (1,062 = 671
`TRAINED_REAL_EXACT` + 391 `NOVEL_PSEUDOWORD`) answers the **model-specific**
question. The distinction matters because the WFE was built against Dager's 50k
lexicon: 122 source-real words were never trained and 9 source-pseudowords
collide with the training lexicon. **Source labels are not training exposure**,
and the two regimes are never pooled.

## 4. Central behavioral result

**The length effect is route- and familiarity-dependent, not a blanket property
of LTM.**

- **Trained real words**: FULL and WM are at **exact ceiling** — zero errors on
  every item in every seed — and LTM shows only a **very weak** length slope
  (0.0007 to 0.0177 per phoneme).
- **Novel or untrained phonological forms**: LTM develops a **large** length
  effect (slope 0.197–0.256 per phoneme), **WM remains much more robust**
  (0.007–0.037), and FULL stays close to ceiling (0.003–0.020).

The primary contrast, LTM − WM on clean pseudowords, is positive in all four
seeds: **+0.246, +0.183, +0.184, +0.205**. So the answer to the question is
yes — but the correct statement is *the ventral route shows a large length
effect for unfamiliar forms*, **not** "LTM always has a length effect".

Figures F1, F2.

## 5. Familiarity / exposure result

The 122 untrained real words behave **like novel pseudowords rather than like
trained real words**: under LTM their mean edit distance is 0.549 against 0.601
for novel pseudowords and 0.024 for trained real words. Their gate and lexical
confidence pattern the same way (S12).

**Exposure, not lexical status per se, tracks the difficulty.** On the clean set
lexicality and exposure are *perfectly* confounded — every Real item is trained
and every Pseudo item is novel — so the clean analyses **cannot independently
identify** the two, and the factor is reported throughout as the
**lexicality/exposure contrast**.

## 6. Morphology result

**No robust morphology effect anywhere**: no main effect, no morphology × length
interaction, no morphology × route interaction. Every interval spans zero, in
both the faithful and the adapted analyses, and morphology ranks last in the
adapted feature importance in all four seeds with a mean R² drop of 0.0009.
Trained real words remain ceiling-limited under FULL and WM, which restricts
where a morphology effect could have shown at all.

This is a statement about **predictive contribution to these behavioural
outcomes**. It is **not** a claim that morphology is absent from the model's
internal representations. Figures S6, S7.

## 7. Frequency result

Within trained real words, LTM edit distance **decreases slightly as Zipf
frequency rises**: slope negative in all four seeds, mean −0.0130, 95 %
hierarchical bootstrap [−0.0288, −0.0027]. The direction is consistent, but the
result rests on **sparse errors**, and FULL and WM are `ALL_ZERO_OUTCOME` by
ceiling — no absence of frequency encoding may be inferred for them.

Lexical confidence (+0.0914) and the gate (+0.0450) show matching positive
slopes, but the gate is a **deterministic monotonic transform** of confidence
(`g = sigmoid(2.0·(confidence − 0.7))`), so these are **linked outcomes, not
independent evidence**. Figures F6, S10.

## 8. Error taxonomy

Operations are exactly substitution, deletion and insertion, from
`Levenshtein.editops` 0.27.3; **no fourth operation exists**. On clean
pseudowords under LTM, per evaluated item: substitutions 0.574 (Long) /
0.098 (Short), deletions 0.245 / 0.024, insertions 0.140 / 0.018.

**Substitutions dominate; deletions and insertions are also elevated; the burden
is concentrated in long LTM pseudowords.** FULL and WM stay an order of
magnitude lower on the same absolute scale — the frozen >10× zoom rule fired
(ratio 22.95), so a labelled FULL/WM companion accompanies but never replaces
the common-scale figure.

Two measurement limits: **editops tie-breaking moves counts between operation
types without changing total edit distance**, and the **forced-length horizon
underestimates terminal insertions**. Figures F4, S4, S5.

## 9. Premature-EOS result

87 observed premature-EOS events across four seeds and three routes: **LTM 82,
FULL 3, WM 2 — all on pseudowords, none on trained real words**. The LTM rate
rises with length (0.005 at length 3 to 0.189 at length 9) with a positive
linear-probability slope in all four seeds.

But the scale matters: those 82 events sit against **874 edit operations across
365 erroneous LTM pseudoword items**, so an observed early stop is present on
about **22 %** of erroneous items, and **107 of 189 deletion-bearing items carry
no observed premature EOS at all**. **Premature EOS is therefore not a complete
explanation of the error pattern**, and no causal claim is made in either
direction.

`P(deletion | premature EOS) = 1.000` is a **structural consequence** of
trimming at the first EOS, not an empirical discovery. Under the current readout
horizon **`ON_TIME_EOS` and `LATE_EOS` are structurally unobservable**, and
`EOS_NOT_OBSERVED` means only that no EOS was observed within the instrumented
horizon — it is never read as correct stopping. Figure F5.

## 10. Feature importance

In the adapted joint model (Ridge α = 1.0, split grouped by item, factor-level
grouped permutation), **route and lexicality/exposure lead**; their ordering
relative to one another is **not resolved** (means differ by ≈ 0.0064 R², less
than the within-seed permutation SD, they swap in seed 21, and MAE ranks
lexicality/exposure first in every seed). **Length is a stable additional
contributor** — rank 3 and positive coefficient in all four seeds.
**Morphology is negligible** in this behavioural prediction analysis.

Route-specific models are **estimable only for LTM** (lexicality/exposure >
length > morphology in all four seeds); FULL and WM are ceiling and
sparse-error limited and are labelled, never given an artificial zero.

**Faithful (A11) and adapted (A15) feature importance estimate different
quantities** and are never pooled: A11 places length first in all four seeds,
A15 places lexicality/exposure above length in all four — a divergence the
design differences fully account for. Figures F7, S3, S8, S9.

## 11. What is established

- The length effect is **large in LTM for unfamiliar forms and much smaller in
  WM**, consistently across four seeds.
- **FULL is close to ceiling throughout**, so the gate is doing useful work on
  this stimulus set.
- **Training exposure, not the source Real/Pseudo label, tracks difficulty.**
- Errors are **predominantly substitutions**, concentrated late in long
  unfamiliar items.
- **Premature stopping is real, LTM-specific and length-graded, but partial.**
- **Morphology contributes nothing detectable** to these behavioural outcomes.
- **Frequency helps slightly** within trained real words under LTM.
- The whole pipeline is **deterministic and byte-reproducible** from a frozen
  canonical table.

## 12. What remains mechanistically unresolved

The behavioural analyses **constrain** the mechanism but do **not localize it
uniquely** to any of: the LTM encoder; the compressed semantic representation;
the LTM decoder; autoregressive error accumulation; or the gate. Distinguishing
these requires hidden-state work that is a **separate project**, not a
re-reading of these outputs. The factual, non-causal handoff is
`error_taxonomy/length_effect_mechanism_handoff.md`, which lists the exact
measurements and eight open factual questions.

Two instrumentation limits bound any follow-up: the forced-length readout hides
terminal insertions and all boundary/late EOS timing, and clean-set lexicality
cannot be separated from exposure.

## 13. Recommended figures

**Main set:** F1 (length by route), F2 (slopes and LTM−WM), F3 (serial
position), F4 (error taxonomy), F5 (premature EOS), F6 (frequency), F7 (adapted
feature importance). Supplementary: the faithful replications S1–S3, S5, S7, the
zoom S4, morphology S6, feature-importance detail S8–S9, and gate/confidence
S10–S12. See `final_figure_selection.md`.

## 14. Reproducibility status

Every figure ships as PNG (300 dpi), PDF and SVG with a standalone caption and
the exact table that produced it. Regeneration is deterministic and verified
byte-identical. Six manifests verify: production scientific outputs (36),
Sprint 1 (37), morphology (33), frequency (37), error taxonomy (57), feature
importance (42). The canonical table, the checkpoints and the source datasets
are hash-pinned and unchanged. **A19 (SSP / sonority) remains optional,
deferred and unstarted.**
