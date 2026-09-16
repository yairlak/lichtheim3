# FINAL RULE FREEZE — GATE × LESION / RECOVERY

**Authority:** CENTRAL STEERING, authoritative final rule decision.
**Status:** FINAL. These rules are frozen before any non-zero canonical science.
**Scope:** analysis rules only. No checkpoint, identity, target, SD, λ, seed, batch
size, NATIVE, FIXED05, gate α/threshold or decoder convention is altered.

> **No canonical non-zero science has yet run.** The largest non-zero-severity
> population evaluated anywhere in this workstream is a 24-item quarantined subset
> under `NOT_SCIENTIFIC_RESULT/`. `scientific_execution/` does not exist.

---

## 1. O-3 — SHARED DIAGNOSTIC VALIDITY

```
O3_VALIDITY_CONVENTION_SCOPE        = SHARED
O3_IMPLEMENTATION_FAILURE_SEMANTICS = ABORT_RUN
```

A route × severity receives **ONE shared diagnostic-validity label**.

For criteria involving decoded isolated-route performance, the frozen requirement
must hold under **BOTH** `CANONICAL` **and** `FREE_AR`. Therefore:

* the decoder convention does **not** select different valid severities;
* the **same** valid-severity set is used for the subsequent `CANONICAL` and
  `FREE_AR` NATIVE-vs-FIXED05 classifications;
* a severity satisfying the competence-separation criterion under **only one**
  decoding convention is **NOT** diagnostically valid for the shared causal
  comparison.

Structural criteria are shared by construction.

### The competence criterion

Across **2 witnesses × 4 lesion seeds = 8 blocks**, the **TARGETED** isolated route
must drop by **≥ 0.10 exact-match** relative to intact in **at least 6 of 8** blocks —
evaluated separately under each convention, and **both must pass**.

### The two failure classes — never to be conflated

| class | meaning | effect |
|---|---|---|
| `SCIENTIFIC_PERTURBATION_VALIDITY_FAILURE` | the ≥ 0.10 targeted-route criterion failed | **this route × severity only** is scientifically invalid; the reason is recorded; **the experiment CONTINUES** |
| `IMPLEMENTATION_FAILURE` | a structural/wiring guarantee was violated | **the scientific run ABORTS**; no classification is emitted |

Implementation failures include at least:

* `IMPLEMENTATION_FAILURE_SHARED_PARAM_MUTATED` — mutation of shared downstream / checkpoint parameters;
* `IMPLEMENTATION_FAILURE_CHECKPOINT_NOT_RESTORED` — failure to restore checkpoint identity;
* `IMPLEMENTATION_FAILURE_UNTARGETED_ROUTE_CHANGED` — untargeted-route change inconsistent with the intervention contract;
* `IMPLEMENTATION_FAILURE_DORSAL_GATE_NULL_VIOLATED` — dorsal perturbation changing `c_LTM` or `g` beyond the frozen exact tolerance;
* `IMPLEMENTATION_FAILURE_LESION_TENSOR_MISMATCH` — NATIVE/FIXED05 receiving non-identical matched lesion tensors.

> **No denominator shrinking. No block exclusion. No scientific classification after
> an implementation failure.** An implementation failure is never silently
> transformed into an invalid severity — the structural check runs **first** and
> raises before any competence count is computed.

### Deterministic equality tolerance — exact, unchanged

`0.0` / **bitwise**, wherever the existing contract specifies bitwise equality.
Inherited, not invented:

* gating `EXPERIMENT_CONTRACT.md` §5 — *"require bitwise equality"*;
* `scripts/gating_diagnostics/run_gate_route_audit.py:168-180` (`assert_determinism`);
* frozen `summary_metrics.json` — `determinism_max_gate_dev == 0.0` in all 8 states.

### The ≥ 0.10 boundary is computed from integer counts

**Retained from the closure pass and NOT reverted.** The drop is one exact rational
from integer counts, never a subtraction of two floating accuracies. The defect this
avoids is concrete: for a block going 900/1000 → 800/1000,

```
(900/1000) - (800/1000) == 0.09999999999999998   →  fails `>= 0.10`
Fraction(900 - 800, 1000) == 1/10                →  passes, correctly
```

On the canonical population an exact tie cannot arise: `0.10 × 29571 = 2957.1`, so
**2957 items fails and 2958 passes**.

### Reason enum

```
VALID
TARGET_ROUTE_DROP_FAILED_CANONICAL
TARGET_ROUTE_DROP_FAILED_FREE_AR
TARGET_ROUTE_DROP_FAILED_BOTH
IMPLEMENTATION_FAILURE_SHARED_PARAM_MUTATED
IMPLEMENTATION_FAILURE_CHECKPOINT_NOT_RESTORED
IMPLEMENTATION_FAILURE_UNTARGETED_ROUTE_CHANGED
IMPLEMENTATION_FAILURE_DORSAL_GATE_NULL_VIOLATED
IMPLEMENTATION_FAILURE_LESION_TENSOR_MISMATCH
```

**Implementation:** `gate_x_lesion/validity.py::decide_shared_validity`.

---

## 2. O-4 — ROBUSTNESS

```
O4_ROBUSTNESS_RULE  = REPLICATED_SIGN_PLUS_MATERIALITY_NO_SIGNIFICANCE_TEST
O4_STATISTICAL_UNIT = NONE
O4_TEST             = NONE
O4_SEED_HANDLING    = REPLICATION_UNIT_4_LESION_SEEDS_WITHIN_EACH_WITNESS
O4_WITNESS_HANDLING = TWO_WITNESSES_MUST_SATISFY_RULE_INDEPENDENTLY
O4_ALPHA            = NOT_APPLICABLE
O4_MULTIPLICITY     = NOT_APPLICABLE
O4_MAGNITUDE_FLOOR  = 0.002_ABSOLUTE_ACCURACY
```

> **THERE IS NO SIGNIFICANCE TEST.** No item-level test, no seed-level test, no
> cross-witness test. No p-value, no α, no multiplicity correction, no bootstrap.
> This is enforced structurally: `gate_x_lesion/robustness.py` may import only
> `{__future__, math, dataclasses, fractions, typing}` and uses no statistical
> identifier — asserted by `test_O4_15_no_significance_test_code_path_exists`.

### Definition

For fixed route `r`, severity `λ`, decoding convention `d`, witness `w`, lesion seed `s`:

```
Delta[w, s] = accuracy_native[w, s] - accuracy_fixed05[w, s]
```

**Sign convention — POSITIVE means NATIVE is better.** This is the opposite of the
legacy `net_change_in_correct` column (`errors_recovered − errors_gained`), which is
the negated numerator. The frozen rule reads `Delta` directly and never the legacy
column.

### The rule

A direction σ ∈ {POSITIVE, NEGATIVE} is **ROBUST** for a fixed route × severity ×
decoding convention **iff all** of the following hold:

1. **BOTH** witnesses independently support the same direction.
2. Within **EACH** witness, at least **3 of 4** lesion seeds have `Delta` with that
   direction. `POSITIVE: Delta > 0`; `NEGATIVE: Delta < 0`; **`Delta == 0` is
   neutral and counts toward neither**.
3. For **EACH** witness separately, `mean_s(Delta[w,s])` over exactly the 4 frozen
   seeds has the same direction **and** `|mean_s(Delta[w,s])| >= 0.002`.
4. **No** lesion seed in either witness shows an OPPOSITE-signed effect with
   `|Delta[w,s]| >= 0.002`. A single materially large reversal prevents ROBUST.
5. **No pooling** of W3 and W4 may rescue a failed witness-level rule.
6. No item-level significance test.
7. No seed-level significance test.
8. No cross-witness significance test.

Result is one of `ROBUST_POSITIVE`, `ROBUST_NEGATIVE`, `NOT_ROBUST`.

### The 0.002 materiality floor

`0.002` is a **prospective MATERIALITY threshold only**. It is **not** empirically
derived from lesion outcomes — no non-zero canonical lesion has run.

### Exact arithmetic at the boundary

The formal rule is **`>= 0.002` ABSOLUTE ACCURACY**. It is **NOT** "59 items":

```
0.002 × 29571      = 59.142    (per seed)
0.002 × 4 × 29571  = 236.568   (sum over a witness's four seeds)
```

Neither is an integer, so any integer-count implementation must take the **CEILING**
at the boundary — not truncate, not round. The derived integer thresholds on the
canonical population are therefore **60** per seed and **237** for the four-seed sum.
These are *derived*, never hard-coded: `witness_direction_from_counts` uses exact
rational arithmetic (`fractions.Fraction`), so it is mathematically equivalent to the
accuracy comparison for any `N`, including when `0.002 × N` happens to be an integer
(then the threshold is that value itself, because the comparison is `>=`).

`Delta` is never rounded before comparison.

**Implementation:** `gate_x_lesion/robustness.py::frozen_robustness` (Delta path) and
`::frozen_robustness_from_counts` (preferred exact integer-count path).

---

## 3. THE O-4 VETO — ROBUST OPPOSITE-SIGNED EFFECT

For a fixed **route family** and **decoding convention**: if one diagnostically valid
severity supports a robust direction and **another diagnostically valid severity
satisfies the FULL ROBUST rule in the opposite direction**, then `A`/`B`/`D` may
**NOT** be declared. Classification must be:

```
OUTCOME_F_HETEROGENEOUS
```

* Only a severity that is **both** diagnostically valid **and** fully robust can
  contribute a direction.
* A severity that merely looks opposite-signed without meeting the full robust rule
  does **not** trigger the veto.
* A diagnostically **invalid** severity **cannot** trigger the veto at all.
* **All diagnostically valid severities remain visible.**
* **No severity may be retrospectively promoted as "the lesion result".**

### Direction → outcome letter

| robust direction | meaning | letter |
|---|---|---|
| `ROBUST_POSITIVE` | NATIVE advantage | `A` |
| `ROBUST_NEGATIVE` | FIXED05 advantage | `B` |
| no valid severity robust | robust null | `D` |
| valid severities robust in BOTH directions | veto | `F_HETEROGENEOUS` |
| no diagnostically valid severity | not a null — never tested | `UNDETERMINED_NO_VALID_SEVERITY` |

---

## 4. DECODING-CONVENTION CLASSIFICATION

`CANONICAL` and `FREE_AR` are classified **independently FIRST**:

```
CANONICAL_OUTCOME = <letter>
FREE_AR_OUTCOME   = <letter>
```

Only **afterward** is the joint outcome derived:

```
both conventions yield the same classification X  ->  JOINT_OUTCOME = CONCORDANT_X
they differ materially                            ->  JOINT_OUTCOME = MIXED_DECODING
```

> A favourable classification in one convention must **never** overwrite a
> conflicting or heterogeneous classification in the other. Both stage letters remain
> visible in the joint result.

**Label form.** Per CENTRAL's instruction to *"use the existing project outcome labels
exactly if they are more specific than A/B/D/F"*, the concordant label is the
`CONCORDANT_` prefix applied to the project's own letter. The project's `F` label is
the more specific `F_HETEROGENEOUS`, so:

| CANONICAL | FREE_AR | JOINT_OUTCOME |
|---|---|---|
| `A` | `A` | `CONCORDANT_A` |
| `B` | `B` | `CONCORDANT_B` |
| `D` | `D` | `CONCORDANT_D` |
| `F_HETEROGENEOUS` | `F_HETEROGENEOUS` | `CONCORDANT_F_HETEROGENEOUS` |
| `A` | `F_HETEROGENEOUS` | `MIXED_DECODING` |
| `A` | `B` | `MIXED_DECODING` |

`CONCORDANT_F_HETEROGENEOUS` is the faithful composition of CENTRAL's `CONCORDANT_F`
with the more specific existing label; it is not a new schema.

**Implementation:** `gate_x_lesion/classify.py::classify_route_family`,
`gate_x_lesion/outcomes.py::classify_joint`.

---

## 5. MACHINE-READABLE CONSTANTS

| constant | value | defined in |
|---|---|---|
| `O3_VALIDITY_CONVENTION_SCOPE` | `SHARED` | `validity.VALIDITY_CONVENTION_SCOPE` |
| `O3_IMPLEMENTATION_FAILURE_SEMANTICS` | `ABORT_RUN` | `validity.IMPLEMENTATION_FAILURE_SEMANTICS` |
| `TARGETED_MIN_DROP` | `0.10` | `validity.TARGETED_MIN_DROP` |
| `MIN_BLOCKS_WITH_DROP` | `6` | `validity.MIN_BLOCKS_WITH_DROP` |
| `N_BLOCKS` | `8` (2 witnesses × 4 seeds) | `validity.N_BLOCKS` |
| `DETERMINISTIC_TOLERANCE` | `0.0` bitwise | `validity.DETERMINISTIC_TOLERANCE` |
| `O4_ROBUSTNESS_RULE` | `REPLICATED_SIGN_PLUS_MATERIALITY_NO_SIGNIFICANCE_TEST` | `robustness.ROBUSTNESS_RULE_ID` |
| `O4_STATISTICAL_UNIT` | `NONE` | `robustness.O4_STATISTICAL_UNIT` |
| `O4_TEST` | `NONE` | `robustness.O4_TEST` |
| `O4_SEED_HANDLING` | `REPLICATION_UNIT_4_LESION_SEEDS_WITHIN_EACH_WITNESS` | `robustness.O4_SEED_HANDLING` |
| `O4_WITNESS_HANDLING` | `TWO_WITNESSES_MUST_SATISFY_RULE_INDEPENDENTLY` | `robustness.O4_WITNESS_HANDLING` |
| `O4_ALPHA` | `NOT_APPLICABLE` | `robustness.O4_ALPHA` |
| `O4_MULTIPLICITY` | `NOT_APPLICABLE` | `robustness.O4_MULTIPLICITY` |
| `MATERIALITY_FLOOR` | `0.002` absolute accuracy | `robustness.MATERIALITY_FLOOR` |
| `MIN_DIRECTIONAL_SEEDS` | `3` of `4` | `robustness.MIN_DIRECTIONAL_SEEDS` |
| `N_WITNESSES` | `2` | `robustness.N_WITNESSES` |
| `JOINT_MIXED_DECODING` | `MIXED_DECODING` | `outcomes.JOINT_MIXED_DECODING` |
| `JOINT_CONCORDANT_PREFIX` | `CONCORDANT_` | `outcomes.JOINT_CONCORDANT_PREFIX` |

Serialisable form: `gxlr_conditions.final.json`.

---

## 6. WHAT IS UNCHANGED

Checkpoints; `RECONSTRUCTED_STATE_SHA256`; lesion targets
`{wm_encoder_state, ltm_encoder_state}`; SD semantics (`std`, ddof=1, all axes pooled,
2048-item sorted sample at seed 7, scalar per state/route); λ ∈ {0.25, 0.50, 1.00};
lesion seeds {0,1,2,3}; `batch_size = 256`; NATIVE; FIXED05; gate `α = 2.0`,
`threshold = 0.7`; canonical forced-length AR and genuine free-AR; the base-noise RNG
identity and nested scaling; `per_item_frozen` timing; the 29,571-item population.

No recovery training. No connectivity lesions. No new scientific criteria.

---

## 7. EXECUTION STATUS

```
FINAL_DESIGN_DECISIONS_COMPLETE = YES
FINAL_RULE_IMPLEMENTATION_COMPLETE = YES
GO_FOR_SCIENTIFIC_EXECUTION = NO
AWAITING_CENTRAL_FINAL_SCIENTIFIC_EXECUTION_GO = YES
```

**No canonical non-zero science has yet run.** The prepared execution command is
non-executable and has never been run; the scientific runner still requires an
explicit CENTRAL-go flag; `scientific_execution/` remains absent.
