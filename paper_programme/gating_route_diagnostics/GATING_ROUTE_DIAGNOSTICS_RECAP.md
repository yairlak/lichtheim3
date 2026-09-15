# GATING / ROUTE DIAGNOSTICS — RECAP

**Workstream:** LICHTHEIM3 — GATING / ROUTE DIAGNOSTICS (POST_STAGE / PAPER_PROGRAMME)
**Branch:** `paper-programme/gating-route-diagnostics`
**Pre-execution freeze commit:** `f43ccd0971387ccfce006ddb0d6f985f1ee73441`
**Executed:** 2026-09-15, under `GO_NOW=YES` from CENTRAL STEERING
**Stage:** **EXPERIMENTS 1 AND 2 COMPLETE.** Evidence package assembled. STOP condition respected.

> **Epistemic classes are marked throughout and never blurred:**
> `STRUCTURAL_CODE_FACT` · `EMPIRICAL_RESULT` · `INTERPRETATION` · `NOT_ESTABLISHED`.
> The preregistration (`EXPERIMENT_CONTRACT.md` + AMENDMENTS 1 and 2) was **not** modified after
> execution. Amendments 1 and 2 were both made before any result existed.

---

## QUESTION

**Primary.** Does the deployed gate track **relative route competence** item by item — when only one
route is correct, does the gate preferentially weight that route?

**Secondary.** What does the gate causally contribute at inference, measured against a preregistered
`0.5 / 0.5` symmetric-fusion baseline?

## FROZEN COMMIT

Execution ran at `f43ccd097…`, verified before any evaluation: branch, HEAD and clean worktree all
matched; `SHA256SUMS` 15/15 OK; 12/12 checkpoint artifacts hash-MATCH; 22/22 diagnostic tests passed.

## CHECKPOINTS

Eight states — the four Phase-8 exact FULL/gated witnesses, each paired with its
provenance-established pre-repair source. Full detail in `checkpoint_manifest.tsv`.

| Witness | Seed | u | Selection | Source C → Repaired C |
|---|---|---|---|---|
| W1 `V5_seed20_u3600` | 20 | 3600 | retrospective | 32 → 0 |
| W2 `V5_seed21_u3400` | 21 | 3400 | retrospective | 26 → 0 |
| W3 `V6_seed19_u3825` | 19 | 3825 | prospective first-hit | 34 → 0 |
| W4 `V6_seed20_u3040` | 20 | 3040 | prospective first-hit | 42 → 0 |

Within each pair Arm A modifies only `ltm.to_semantic.2.{weight,bias}`, so the **entire dorsal route
is bit-identical** and any gate movement is attributable to `ŝ` alone.

## PROVENANCE

| role | identifier | hash |
|---|---|---|
| **CANONICAL AUTHORITY NAME** (the baseline the Phase-8 arbitration names as its sole canonical working baseline) | `LICHTHEIM3_SOURCE_OF_TRUTH_CANONICAL_REPAIRED_2026-09-14.zip` | `22f9e6911a504e20767c320b2a98de0a7d9bba438a6c5a2889d96fe7f4c75995` *(as recorded in the arbitration report; that zip is not present locally, only its unpacked staged copy)* |
| **LOCAL ARCHIVE PATH** (the arbitration-output package actually read) | `LICHTHEIM3_SOURCE_OF_TRUTH_CANONICAL_FABLE_ARBITRATED_2026-09-14.zip` | `6d3a0a6615b90b0b45de44dd1aac2b21b7cac594cf6f0ef3fb259cab4a92c628` *(measured)* |

```
EXTERNAL INPUT: MEETING_NOTES_YAIR_EMMANUEL_2026-09-14.md
status:         CONTEMPORARY_RELAYED_MEETING_NOTES  (hypothesis source, not historical authority)
sha256:         c1983db3ee244182e3a3f11c25ed8500c36a01f5dd53d5b454b01d4a0032b70e
in-repository path: NOT PRESENT
```

Model reconstruction reused `frozen_head_probe.build_trainer` and `_isolated_model` verbatim — the
same path that produced the frozen Phase-8 witness metrics.

### Independent reproduction of the frozen Phase-8 record — `EMPIRICAL_RESULT`

This run **exactly reproduced** every frozen isolated-LTM figure, from the checkpoints, through an
independently written evaluator:

| state | isolated-LTM exact (this run) | frozen Phase-8 | errors (this run / frozen) |
|---|---|---|---|
| W1_SRC / W1_REP | 0.885530 / 0.884955 | 0.885530 / 0.884955 | 3385 / 3402 — match |
| W2_SRC / W2_REP | 0.884245 / 0.883974 | 0.884245 / 0.883974 | 3423 / 3431 — match |
| W3_SRC / W3_REP | 0.891820 / 0.892023 | 0.891820 / 0.892023 | 3199 / 3193 — match |
| W4_SRC / W4_REP | 0.876670 / 0.876162 | 0.876670 / 0.876162 | 3647 / 3662 — match |

FULL/gated repetition is exact (1.000000) in all eight states under both conventions, as frozen.
Determinism check: `max|Δgate|` on re-decode = **0.0** in every state. Source checkpoints verified
unmutated after every state.

---

## STRUCTURAL_CODE_FACT

Established by the code audit **before** execution; not newly discovered here. Verified numerically
(`tests/test_gate_route_diagnostics.py`, 22 passing) and unchanged by the results.

1. The gate has **no trainable parameters**. `g = σ(α·(c_LTM − τ))`, α and τ fixed hyperparameters.
2. `g` depends on **ventral lexical confidence only**; the gate is **blind to dorsal-exclusive
   parameters and activations** (perturbing all 10 WM-exclusive tensors moves `wm_logits` by
   1.03e+01 and `g` by exactly 0.000e+00). It is **not** blind to `phon_embed.weight`, which the
   routes share; they also share the `motor.proj` readout.
3. The attainable weighting range is **asymmetric**: ventral `g ∈ [0.0323, 0.6457]`, dorsal
   `1−g ∈ [0.3543, 0.9677]`. Strong dorsal commitment is possible (≈ 29.9 : 1); strong ventral
   commitment is structurally impossible (≤ ≈ 1.82 : 1).
4. `fixed05` algebra is **exact** for the current affine motor readout:
   `motor(g·ltm + (1−g)·wm) = g·ltm_logits + (1−g)·wm_logits` (max|Δ| = 1.6e-07).
5. **Naming and Comprehension do not use the repetition gate** (C is `encode → ŝ → cosine
   retrieval`; N is `ltm.decode_from_s_hat → motor`), so `fixed05` is mathematically inert for C, N
   and both isolated routes. They were correctly **not** re-evaluated under the intervention.
6. Original **H1 is structurally untestable** on this population.
7. The historical repaired `gate_mean` **instrumentation defect** (measured on `tr.model`, the
   source).

---

## EXPERIMENT 1 — EMPIRICAL RESULTS

All 8 states × 29,571 items, complete canonical population, no `--limit`, no smoke.

### 1.1 Route-competence category counts — `EMPIRICAL_RESULT`

| state | BOTH_CORRECT | WM_ONLY_CORRECT | LTM_ONLY_CORRECT | NEITHER_CORRECT |
|---|---:|---:|---:|---:|
| W1_SRC | 26186 | 3385 | **0** | 0 |
| W1_REP | 26169 | 3402 | **0** | 0 |
| W2_SRC | 26148 | 3421 | **0** | 2 |
| W2_REP | 26140 | 3429 | **0** | 2 |
| W3_SRC | 26372 | 3199 | **0** | 0 |
| W3_REP | 26378 | 3193 | **0** | 0 |
| W4_SRC | 25924 | 3646 | **0** | 1 |
| W4_REP | 25909 | 3661 | **0** | 1 |

Internally consistent with the frozen record in every state
(`WM_ONLY + NEITHER` = frozen LTM errors; `NEITHER` = frozen WM errors).

### 1.2 H1 — `STRUCTURALLY_UNTESTABLE`

`n(LTM_ONLY_CORRECT) = 0` in **all eight** states — below even the bound of ≤ 2 anticipated in
AMENDMENT 1, and far below the frozen power floor of 30. H1 is reported as
`STRUCTURALLY_UNTESTABLE`, exactly as preregistered, and is **not** answered by H1′.

### 1.3 Item-level gate distribution — `EMPIRICAL_RESULT`

The quantity that had **never** been recorded (only a mean had been).

| state | mean (item-level) | mean (position-weighted historical) | sd | min | q25 | median | q75 | max | frac > 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1_SRC | 0.519792 | 0.519792 | 0.02953 | 0.3666 | 0.5016 | 0.5219 | 0.5401 | 0.5996 | 0.766 |
| W1_REP | 0.519678 | 0.519678 | 0.02953 | 0.3667 | 0.5015 | 0.5218 | 0.5400 | 0.5991 | 0.765 |
| W2_SRC | 0.518916 | 0.518916 | 0.02979 | 0.3782 | 0.5003 | 0.5205 | 0.5397 | 0.6023 | 0.753 |
| W2_REP | 0.518828 | 0.518828 | 0.02975 | 0.3782 | 0.5002 | 0.5204 | 0.5396 | 0.6010 | 0.752 |
| W3_SRC | 0.521295 | 0.521295 | 0.02977 | 0.3497 | 0.5031 | 0.5233 | 0.5421 | 0.6049 | 0.779 |
| W3_REP | 0.521226 | 0.521226 | 0.02976 | 0.3500 | 0.5029 | 0.5233 | 0.5421 | 0.6039 | 0.779 |
| W4_SRC | 0.517074 | 0.517074 | 0.03025 | 0.3454 | 0.4983 | 0.5189 | 0.5378 | 0.6022 | 0.733 |
| W4_REP | 0.516956 | 0.516956 | 0.03024 | 0.3454 | 0.4982 | 0.5188 | 0.5377 | 0.6022 | 0.732 |

**The two named means coincide to 6 dp in every state.** Reason, recorded rather than assumed: under
the canonical batching the padded decoder width is constant at exactly 10.0 (min = max = 10.0), so
the historical position weighting is uniform here and degenerates to the item-level mean. That is a
property of this population's batching, not a general equivalence.

The **source** means reproduce the frozen Phase-8 values to all recorded digits
(0.5197919607…, 0.5189160108…, 0.5212951898…, 0.5170738101…).

**Distribution shape:** narrow and unimodal, **not** bimodal. `sd ≈ 0.030`. Observed support is
≈ [0.345, 0.605] — about **41 % of the attainable range**, entirely in its upper portion. `g` never
fell below 0.345, so the strong dorsal commitment that the architecture *permits* (down to 0.0323)
**never occurs** on real words. The gate leans ventral on 73–78 % of items.

### 1.4 How large is the `fixed05` intervention? — `EMPIRICAL_RESULT`

AMENDMENT 2 flagged this as an open question and forbade calling it a "~0.02 perturbation". Measured:

| state | mean \|g−0.5\| | median \|g−0.5\| | p95 | max | signed mean |
|---|---:|---:|---:|---:|---:|
| W1_SRC | 0.02940 | 0.02661 | 0.06665 | 0.13336 | 0.01979 |
| W2_SRC | 0.02904 | 0.02596 | 0.06653 | 0.12182 | 0.01892 |
| W3_SRC | 0.03036 | 0.02756 | 0.06794 | 0.15029 | 0.02130 |
| W4_SRC | 0.02842 | 0.02509 | 0.06607 | 0.15465 | 0.01707 |

The caution was **justified in direction**: the item-level mean absolute deviation (≈ 0.029) is
**≈ 1.5×** the signed mean (≈ 0.020), and the largest per-item displacement reaches ≈ 0.15. The
intervention is therefore meaningfully larger per item than the recorded signed mean implied — and
it still produced no behavioural change whatsoever (§ Experiment 2).

### 1.5 H1′ — VENTRAL CONFIDENCE CALIBRATION DIAGNOSTIC — `EMPIRICAL_RESULT`

Hypothesis: `median g(WM_ONLY_CORRECT) < median g(BOTH_CORRECT)`. Positive class = `WM_ONLY_CORRECT`
(ventral fails). Predicted direction: AUROC < 0.5. Powered in all eight states
(`min n` = 3193, floor 30).

| state | n WM_ONLY | n BOTH | median g (WM_ONLY) | median g (BOTH) | median diff | AUROC | Cliff's δ | 95 % CI (AUROC) | direction |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| W1_SRC | 3385 | 26186 | 0.513900 | 0.523122 | −0.009222 | 0.3947 | −0.2106 | [0.3857, 0.4037] | as predicted |
| W1_REP | 3402 | 26169 | 0.513840 | 0.523016 | −0.009177 | 0.3947 | −0.2105 | [0.3858, 0.4035] | as predicted |
| W2_SRC | 3421 | 26148 | 0.512555 | 0.521829 | −0.009275 | 0.3916 | −0.2168 | [0.3826, 0.4008] | as predicted |
| W2_REP | 3429 | 26140 | 0.512593 | 0.521764 | −0.009172 | 0.3928 | −0.2143 | [0.3840, 0.4019] | as predicted |
| W3_SRC | 3199 | 26372 | 0.515073 | 0.524582 | −0.009510 | 0.3938 | −0.2125 | [0.3845, 0.4029] | as predicted |
| W3_REP | 3193 | 26378 | 0.514898 | 0.524571 | −0.009673 | 0.3934 | −0.2133 | [0.3842, 0.4026] | as predicted |
| W4_SRC | 3646 | 25924 | 0.511390 | 0.520162 | −0.008773 | 0.3966 | −0.2068 | [0.3881, 0.4052] | as predicted |
| W4_REP | 3661 | 25909 | 0.511286 | 0.520128 | −0.008841 | 0.3963 | −0.2074 | [0.3878, 0.4050] | as predicted |

Genuine free-AR gives the same values (AUROC 0.3916–0.3966).

**H1′ is CONFIRMED in direction in 8/8 states, with every 95 % CI excluding 0.5.** The effect is
**real but modest**: AUROC ≈ 0.39, Cliff's δ ≈ −0.21, median separation ≈ 0.009 gate units.

**Licensed interpretation (frozen):** *ventral confidence is informative about ventral failure.*
**Excluded (frozen):** *the gate tracks relative route competence*; *the gate arbitrates between
routes*. Both remain excluded structurally, not merely unproven.

Note the identity preregistered in contract §6.5 and confirmed empirically below:
ρ(`g`, `lexical_confidence`) = **1.0000 exactly**, so AUROC(`g`) *is* AUROC(`c_LTM`).

### 1.6 Descriptive covariates — `EMPIRICAL_RESULT` (descriptive; no hypothesis test)

Spearman ρ against `g`:

| state | zipf frequency | length | lexical confidence | margin | density |
|---|---:|---:|---:|---:|---:|
| W1_SRC | +0.4537 | +0.2910 | **+1.0000** | −0.1717 | −0.4991 |
| W2_SRC | +0.4648 | +0.2807 | **+1.0000** | −0.1685 | −0.5112 |
| W3_SRC | +0.4605 | +0.2952 | **+1.0000** | −0.1651 | −0.4983 |
| W4_SRC | +0.4608 | +0.2784 | **+1.0000** | −0.1761 | −0.5185 |

(REP states are within ±0.001 of their sources throughout.)

* ρ(g, frequency) ≈ **+0.46** — more frequent words receive more ventral weight. This is the
  direction the 14 Sep meeting (§9) hypothesised, now measured.
* ρ(g, length) ≈ **+0.29** — longer words also receive more ventral weight.
* ρ(g, confidence) = **+1.0000** — the monotone identity, confirmed.
* ρ(g, density) ≈ **−0.50** — items in denser semantic neighbourhoods receive *less* ventral weight.

## EXPERIMENT 2 — EMPIRICAL RESULTS

Paired item-by-item, `full` (native gate) vs `fixed05` (`g ≡ 0.5`), same items, same order.

### 2.1 Paired contingency — `EMPIRICAL_RESULT`

**Identical in all 8 states and under BOTH decoding conventions:**

| | fixed05 correct | fixed05 wrong |
|---|---:|---:|
| **full correct** | **29 571** | **0** (errors gained) |
| **full wrong** | **0** (errors recovered) | **0** |

* `n_discordant` = **0** in every state, both conventions.
* accuracy(full) = accuracy(fixed05) = **1.000000**; `|Δ accuracy| = 0` exactly.
* Exact McNemar: **undefined — there are no discordant pairs.** This is the strongest possible null,
  not a marginal one. Holm correction across the 16 planned tests is therefore vacuous and was not
  needed to reach the conclusion.
* Breakdown by competence category, frequency quintile and target length: **0 in every cell**
  (figures 4–6, and `fig4/5/6_*.tsv`).

### 2.2 The mandatory caveat — `INTERPRETATION`

FULL repetition starts **exact** on these states, so **`errors_recovered = 0` by construction**.
That half of the table is **not** evidence in favour of the native gate and is not presented as such.

The informative half is `errors_gained = 0`: forcing symmetric fusion — displacing the gate by a mean
of 0.029 and up to 0.15 per item — **broke nothing**, on 29,571 items, in 8 states, under two
decoding conventions.

## SOURCE VS REPAIR

Outcome-E materiality rule, exactly as frozen (§6.6): **material** iff mean per-item `|Δg| > 0.01`
**or** category AUROC shifts by more than 0.05.

| pair | mean \|Δg\| | max \|Δg\| | mean Δg | KS statistic | AUROC src → rep | \|ΔAUROC\| | verdict |
|---|---:|---:|---:|---:|---|---:|---|
| W1 | 3.275e-04 | 2.886e-02 | −1.140e-04 | 0.00257 | 0.3947 → 0.3947 | 0.00003 | **IMMATERIAL** |
| W2 | 2.713e-04 | 1.531e-02 | −8.774e-05 | 0.00227 | 0.3916 → 0.3928 | 0.00127 | **IMMATERIAL** |
| W3 | 2.481e-04 | 1.411e-02 | −6.870e-05 | 0.00196 | 0.3938 → 0.3934 | 0.00040 | **IMMATERIAL** |
| W4 | 3.480e-04 | 3.055e-02 | −1.175e-04 | 0.00247 | 0.3966 → 0.3963 | 0.00026 | **IMMATERIAL** |

Mean `|Δg|` is ~30× below the threshold; `|ΔAUROC|` is ~40–1600× below it.

**The historical instrumentation defect is now closed — `EMPIRICAL_RESULT`.** This is the **first
measurement of the repaired gate**. The repaired means (0.519678, 0.518828, 0.521226, 0.516956)
**differ** from their sources, whereas the frozen record listed them as bit-identical. That confirms
finding A6 directly: the historical column was the source model measured twice. The corrected
conclusion is unchanged in substance — repair moves the gate, but **immaterially**.

Because the dorsal route is bit-identical within each pair, this movement is attributable to `ŝ`
alone.

## NEGATIVE RESULTS

* **The brief's first scientific question cannot be answered in the form posed.** Structurally (the
  gate cannot see the dorsal route's state) and empirically (`n(LTM_ONLY_CORRECT) = 0` in 8/8 —
  even emptier than anticipated).
* **Outcome A's numeric criterion is NOT met.** It required AUROC ≤ 0.35 with CI excluding 0.5.
  Observed AUROC is 0.3916–0.3966, with every CI upper bound ≈ 0.40. The effect is robust in
  direction but **does not reach the preregistered threshold**. Outcome A is **not declared.**
* **Outcome C is UNREACHABLE BY CONSTRUCTION** and was not declared.
* **`fixed05` produced zero behavioural change** — no gains, no recoveries, no discordant pairs.
* **Lexicality was NOT run.** `NOT RUN — PROVENANCE NOT ESTABLISHED`: the frozen secondary
  provenance gate (§10) was not satisfied, as the pseudoword assets are not covered by the Phase-8
  freeze manifest. No pseudoword claim is made.
* The gate **never** enters its strong-dorsal region (min observed 0.345 against an attainable
  0.0323) on this all-real-word population.

## INTERPRETATION

Using the frozen outcome labels only.

### Outcome B — **DECLARED**
> **Confidence-driven weighting is dispensable for intact repetition inference: fixed 0.5 is
> behaviourally equivalent.**

Frozen criterion: McNemar not significant after Holm **and** `|Δ accuracy| < 0.002` on FULL/gated
repetition. Observed: **0 discordant pairs** (no significance possible) and `|Δ accuracy| = 0`
exactly, in 8/8 states across both conventions. The criterion is met at its limiting value.

This is strengthened, not weakened, by §1.4: the intervention was not negligible in size
(mean per-item displacement 0.029, max 0.15) and still changed nothing.

### Outcome D — **DECLARED**
> **The main ventral weakness is upstream of the gate.**

Frozen criterion: both conditions show the same ventral deficit (isolated LTM ≈ 0.876–0.892
unchanged) and `fixed05` does not move it. Observed: isolated LTM is 0.876162–0.892023 in every
state, `fixed05` cannot and does not affect it (`STRUCTURAL_CODE_FACT` 5), and terminal repair moves
it by at most ±17 items. The ~11 % ventral shortfall is untouched by anything at the gate.

### Outcome E — **NOT DECLARED (IMMATERIAL)**
Terminal semantic repair changes gate behaviour **measurably but immaterially** by the frozen
threshold, in 4/4 pairs.

### Outcome A — **NOT DECLARED**
The H1′ effect is real and consistent (8/8, CIs excluding 0.5) but AUROC ≈ 0.39 does not meet the
frozen ≤ 0.35 criterion, and the causal half of the criterion fails outright (`fixed05` loses no
items anywhere).

### Outcome C — **UNREACHABLE BY CONSTRUCTION**, non-operative.

### Observed pattern, named separately
> **The gate is a weakly-calibrated ventral-confidence signal that is behaviourally inert on this
> population.** It carries genuine information about ventral failure (H1′, 8/8) and tracks frequency
> (ρ ≈ +0.46), yet occupies so narrow a band around 0.5 (sd 0.030, 41 % of an asymmetric attainable
> range it never explores downward) that replacing it with a constant changes not one of 29,571
> items in any of 8 states.

This combination — *informative but inert* — is not any single frozen outcome, and is named here
rather than forced into one.

## NOT_ESTABLISHED

This experiment does **not** establish:

* performance of a `fixed05` model **trained from scratch** — this was an inference intervention only;
* usefulness of any other `alpha`;
* usefulness of any other `gate_threshold`;
* usefulness of any other fixed ratio (none was searched);
* usefulness of a genuinely **route-relative adaptive controller** — untested, and impossible for the
  current gate by construction;
* usefulness of **semantic attractor** dynamics;
* **lesion validity** — `SCIENTIFIC_LESION_READINESS=NOT_ESTABLISHED` (Phase 8) stands;
* **aphasia replication**;
* **pseudoword lexicality effects** — the frozen secondary provenance gate was **not** satisfied, so
  the battery was not run;
* standard full-training behaviour after any architecture change.

Also not established: that the gate is inert **under lesion**, at other operating points, or on
non-word inputs. Every result here is for the **intact** model on the frozen all-real-word population.

## LIMITATIONS

* Four witnesses from **three seeds of one historical cohort**. Not an independent population; no
  cross-witness significance test was performed, as frozen.
* Route isolation is **functional, not anatomical**: the routes share `phon_embed.weight` (the only
  shared parameter tensor; 10 WM-exclusive, 16 LTM-exclusive) **and** the `motor.proj` readout.
* Training included `λ_gate·(mean(g) − 0.5)²` with `usage_prior = 0.5`, so the model was optimised
  under mild pressure **toward** the symmetry `fixed05` imposes. **The Outcome B null is therefore
  weaker evidence that confidence-driven weighting is functionally unnecessary than it appears.**
  Note the regularizer constrains only the *mean*: it does not explain the narrow item-level spread,
  which is an unconstrained empirical finding.
* The population is entirely real words, all of which the dorsal route already repeats essentially
  perfectly. The gate may matter where the dorsal route does not suffice — precisely the untested
  regime.
* `fixed05`'s exactness rests on the **affine** motor readout; a nonlinear readout or an attractor
  loop would break it and require re-verification.

## WHAT WAS LEARNED

* **The item-level gate distribution, measured for the first time.** Narrow, unimodal, sd ≈ 0.030,
  confined to ≈ [0.345, 0.605] — 41 % of an asymmetric attainable range whose strong-dorsal half is
  never visited. Not bimodal.
* **Ventral confidence is genuinely calibrated to ventral failure** — H1′ confirmed 8/8, CIs
  excluding 0.5 — but only weakly (AUROC ≈ 0.39, δ ≈ −0.21, median gap ≈ 0.009).
* **That calibration is behaviourally inert here.** Forcing `g ≡ 0.5` changed **zero** of 236,568
  item-evaluations. The gate's information is real and unused.
* **The ventral deficit is upstream of the gate.** Isolated LTM sits at 0.876–0.892 regardless of
  gate, condition, or terminal repair.
* **The historical `gate_mean` defect is closed**, and the repaired gate measured for the first time:
  it does move, and the movement is immaterial.
* **The gate tracks frequency (ρ ≈ +0.46) and length (ρ ≈ +0.29)** — the meeting's §9 hypothesis,
  now quantified.
* The evaluator independently **reproduced every frozen Phase-8 route metric exactly**, which is
  meaningful corroboration of the Phase-8 numbers themselves.

## WHAT WAS NOT LEARNED

* Whether the gate matters when the dorsal route **is** impaired (lesion regime) — out of scope.
* Whether the gate matters for **pseudowords** — provenance gate not satisfied.
* Whether a **route-relative** controller would help — structurally impossible to ask of this gate.
* Whether a fixed-0.5 model **trained from scratch** behaves differently.
* Why the gate occupies such a narrow band; the mean-only regularizer does not account for it.

## RECOMMENDED NEXT DECISION FOR CENTRAL STEERING

No decision is taken here. Options, in the order the evidence supports:

1. **Treat the intact-model gating question as answered and closed.** On this population the gate is
   informative but inert, and the ventral deficit is upstream of it. Further intact-model gate
   diagnostics have low expected yield.
2. **Redirect to the ventral route**, which Outcome D identifies as the locus. The ~11 % isolated-LTM
   shortfall is untouched by gating, by fixed fusion, and by terminal repair — consistent with
   Phase 8's independent 0/4 dedicated-ventral-repair negative.
3. **If gating is pursued further, test it where it could matter** — under lesion, or on pseudowords
   with proper provenance — rather than at other `α`/`τ` on intact real words. Note that any
   genuinely adaptive controller is a **new mechanism**, not a retuning: the current gate cannot
   observe the dorsal route at all.
4. **Do not** read Outcome B as licensing removal of the gate from the architecture: the training
   objective pushed toward symmetry, and the untested regimes are exactly those where a router would
   earn its place.

Each of these requires a **new GO decision**.

## STOP CONDITION

**Respected.** Experiments 1 and 2 are complete and the evidence package is assembled. No
`fixed05` training, no gate tuning, no new gate, no semantic attractor, no H512 retraining, no
lesioning, no V7 mutation, no follow-up experiment was started. Returning to CENTRAL STEERING.

## ARTIFACTS

| File | Status |
|---|---|
| `CODE_AUDIT_GATE.md` | frozen, unchanged by execution |
| `EXPERIMENT_CONTRACT.md` (+ AMENDMENTS 1, 2) | frozen, unchanged by execution |
| `checkpoint_manifest.tsv` | unchanged; all 12 artifacts re-verified after execution |
| `summary_metrics.json` | **produced** (8 states) |
| `item_level_gate_route_metrics.tsv` | **produced** (236,568 rows = 8 × 29,571) |
| `figure_source_data/item_level_<STATE>.tsv` | **produced** (8 shards, 29,571 rows each) |
| `figure_source_data/fig1..6_*.tsv` | **produced** (exact plotted values) |
| `figures/fig1..6_*.png` | **produced**, all reproducible from `figure_source_data/` alone |
| `_smoke_not_results/` | quarantined, stale, unused by any result |
| `SHA256SUMS` | refreshed over the completed package |
