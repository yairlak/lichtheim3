# EXPERIMENT_CONTRACT — Gate / route-competence audit and forced-symmetric intervention

**Workstream:** LICHTHEIM3 — GATING / ROUTE DIAGNOSTICS
**Branch:** `paper-programme/gating-route-diagnostics` (base `78f5505`)
**Status:** **PREREGISTRATION — frozen before execution.**
**Date frozen:** 2026-09-15
**Depends on:** `CODE_AUDIT_GATE.md` (same directory). If any test in
`tests/test_gate_route_diagnostics.py` fails, this contract is void and must be re-derived.

> **Freeze rule.** This file is hashed into `SHA256SUMS` before Experiment 1 is executed. Every
> analysis choice below — categories, statistics, thresholds, power rule, outcome mapping — is fixed
> now, while no result exists. Post-hoc changes are permitted only as an explicitly labelled
> AMENDMENT that records what was already known when it was made.

---

## 1. Question

**Primary (Experiment 1).** Does the deployed gate track **relative route competence** item by item —
i.e. when exactly one route is correct, does the gate preferentially weight *that* route?

**Secondary (Experiment 2).** What does the gate causally contribute at inference, measured against
the preregistered `0.5 / 0.5` symmetric-fusion baseline?

Frequency is **not** the entry point. It is a descriptive covariate (§8).

### What the code audit already settles, before any data
`CODE_AUDIT_GATE.md` §5 establishes, with exact numerical verification, that `g` is a function of
`c_LTM` alone and is **bit-invariant** to every dorsal parameter. So the mechanistic answer to
"is the gate an adaptive controller?" is already **no**.

Experiment 1 therefore does not test whether an adaptive mechanism exists. It tests something
narrower and still worth knowing: **whether the ventral confidence signal happens to carry
information about relative route competence anyway.** That is a real empirical question with a real
answer, and it is the question the results must be written up as answering. Any wording implying the
gate "decides" or "selects" is prohibited in the recap.

## 2. No training. No tuning.

Forbidden in this workstream, restated from the brief §9: training a fixed-gate model; implementing
`semantic_attractor`; modifying `alpha` or `tau`; launching H512 training; starting lesioning V2;
searching any fixed ratio other than 0.5; altering route definitions; touching V7 / Jean-Zay.

`0.5 / 0.5` is **the** preregistered intervention. Not one of a grid.

## 3. States under test — four paired SOURCE → POST-REPAIR witnesses

All eight states are enumerated with verified SHA256 in `checkpoint_manifest.tsv`. Every on-disk hash
was checked against the Phase-8 record: **8/8 match.**

| Witness | Seed | u | Selection | Source C | Repaired C | Source LTM err | Repaired LTM err |
|---|---|---|---|---|---|---|---|
| W1 `V5_seed20_u3600` | 20 | 3600 | retrospective | 32 | 0 | 3385 | 3402 |
| W2 `V5_seed21_u3400` | 21 | 3400 | retrospective | 26 | 0 | 3423 | 3431 |
| W3 `V6_seed19_u3825` | 19 | 3825 | prospective first-hit | 34 | 0 | 3199 | 3193 |
| W4 `V6_seed20_u3040` | 20 | 3040 | prospective first-hit | 42 | 0 | 3647 | 3662 |

**Why these, and why paired.** They are the complete set of Phase-8 exact FULL/gated witnesses
(`PHASE08_SOURCE_OF_TRUTH.md`, "EXACT REPAIRED WITNESSES"), and each has a provenance-established
pre-repair source. They were **not** chosen because any diagnostic plot looked favourable — no gate
diagnostic has ever been run on them (see §3.1). The pairing is the design: Arm A modifies **only**
`ltm.to_semantic.2.{weight,bias}`, so within a pair the **entire dorsal route is bit-identical** and
the only thing that can move the gate is `ŝ`. This isolates outcome **E** cleanly.

W1 and W4 share seed 20 at different `u`; they are two distinct states, not a replication. Four
witnesses drawn from three seeds of one historical cohort is **not** an independent population, and
no cross-witness result is to be reported as a seed-level significance test (§9).

### 3.1 The historical gate column must not be reused
`CODE_AUDIT_GATE.md` §7 / finding A6: `full_battery` measures `gate_mean` on `tr.model` (the source)
while threading the repaired `model` into every other evaluator, so all four repaired `gate_mean`
values are bit-identical to their sources and are **not** measurements of the repaired gate. This
contract therefore recomputes the gate independently for **both** members of every pair. No number
from the historical `gate_mean` column enters any result.

## 4. Items

**Primary population — frozen.** The canonical 29,571-entry repetition population, from
`data/lexicon_en_glove_covered.tsv`, verified against each checkpoint's recorded
`lexicon_file_sha256`. Item order is the lexicon order; the order hash is recorded in
`summary_metrics.json`. Identical items, identical order, for all eight states — this is what makes
Experiment 2's item-by-item pairing and the source↔repaired comparison legitimate.

`word`, `rank`, `length`, `zipf_approx` come off the same `LexEntry` that supplied the phonemes,
inside the same loop iteration (`CODE_AUDIT_GATE.md` §10). There is **no join**, therefore no
possible silent misalignment.

**Lexicality is out of scope on this population**: it is entirely real words, so `is_word` is
constant and the `gate vs lexicality` contrast **does not exist here**. It is deferred to §10.

## 5. Conditions

Four routes, decoded under two conventions, for each of the eight states:

| Route | Meaning | Path |
|---|---|---|
| `full` | native confidence gate | unmodified `forward()` |
| `wm` | dorsal only (`g = 0` limit, exact) | `route_logits(route="wm")` |
| `ltm` | ventral only (`g = 1` limit, exact) | `route_logits(route="ltm")` |
| `fixed05` | **the intervention**: `g ≡ 0.5` | `0.5·ltm_logits + 0.5·wm_logits` |

`fixed05` is computed post hoc and is **exactly** the forced-gate result (audit §3); the equality
against a forward-hook implementation is pinned by
`test_claim3_fixed_mix_equals_forced_gate_hook`. No model code is modified.

**Both decoding conventions are reported, neither replaces the other:**
* **canonical forced-length AR** — the convention behind every historical `rep_canonical_*` number,
  so our numbers are comparable to the frozen record;
* **genuine free-AR** — one global cap, no use of the target length, so over-generation and
  non-termination count as errors. The cap is **imported from the historical evaluator**
  (`train_joint_scratch.FREE_AR_MAX_STEPS == 12`), never restated, so the diagnostic cannot drift
  from the frozen free-AR convention. Pinned by
  `test_free_ar_cap_matches_the_historical_evaluator`.

Each route decodes on **its own prefix**. `fixed05` is re-decoded, never reconstructed from stored
per-route predictions (audit §4, caveat 3).

WM interference noise and ventral noise are **off** (`apply_noise=False`, and both are 0.0 in all
four checkpoint configs). Decoding is greedy and deterministic. A determinism check — re-run one
batch, require bitwise equality — runs before the main pass and hard-stops on failure.

## 6. Experiment 1 — gate / route-competence audit

### 6.1 Categories
Defined on the **isolated** routes, under the canonical convention (the free-AR categorisation is
reported in parallel as a robustness check, not as the primary):

`BOTH_CORRECT` · `WM_ONLY_CORRECT` · `LTM_ONLY_CORRECT` · `NEITHER_CORRECT`

### 6.2 Primary diagnostic and its expected direction
Does the gate weight the route that is actually right?

> **H1 (directional, preregistered):** `median g(LTM_ONLY_CORRECT) > median g(WM_ONLY_CORRECT)`.

The gate is one number with one degree of freedom (`ventral weight = g`, `dorsal weight = 1−g`), so
this single contrast *is* the complete "effective route weighting" test. Reporting `1−g` separately
would be the same test with the sign flipped and is not an independent result.

### 6.3 Statistics — fixed now
* **Distributions:** per-category histogram of `g` (50 bins over the attainable range
  [0.032, 0.646]) plus ECDF. Emitted as `figure_source_data/`.
* **Central tendency:** mean, median, SD, IQR, min, max, and **n**, per category, per state.
* **Effect size:** **Cliff's δ** for `LTM_ONLY` vs `WM_ONLY` — non-parametric, robust to the bounded
  non-normal `g`.
* **Discrimination:** **AUROC** of `g` separating `LTM_ONLY` (positive) from `WM_ONLY` (negative).
  Reported *with* the identity **δ = 2·AUROC − 1**: these are the same statistic reparameterised, and
  will be presented as one finding, not two corroborating ones.
* **Uncertainty:** 95 % CI by bootstrap, 10,000 resamples, stratified by category, `seed=20260915`.
* **Covariates (descriptive):** Spearman ρ of `g` against `zipf_approx`, `length`,
  `lexical_confidence`, `lexical_margin`, `lexical_density`.

### 6.4 Power rule — declared before seeing any count
Both routes are explicitly trained to be independently competent at repetition (`λ_wm = 0.5`), so
`BOTH_CORRECT` is expected to dominate and the two discriminative cells may be small.

> **If `min(n_WM_ONLY, n_LTM_ONLY) < 30` for a state: report counts and distributions only, label
> that state `UNDERPOWERED`, and report NO AUROC, NO δ and NO CI for it.**

An effect size on a handful of items is not a weak result, it is not a result. Whatever the counts
turn out to be, they are a **primary reportable**: "how often is exactly one route right?" is itself
a finding about the architecture.

### 6.5 The mediation check that gives the result its meaning
Because `g` is a **strictly monotone** function of `c_LTM` (σ is monotone, α > 0), `g` and `c_LTM`
are rank-identical: any rank statistic on `g` equals the same statistic on `c_LTM`. So AUROC(`g`)
**is** AUROC(`c_LTM`), exactly.

This must be stated wherever the AUROC is reported. A significant effect would mean *ventral
confidence is informative about ventral failure*; it would **not** mean the gate arbitrates between
routes, and it would not be evidence about relative route competence. The distinction is the
scientific content of Experiment 1 and the recap must not blur it.

### 6.6 Source vs repaired — outcome E
For each pair, report Δ of: `gate_mean`, `gate_SD`, the full `g` distribution (two-sample KS statistic),
per-category means, AUROC, and the per-item `g` change. Since the dorsal route is bit-identical within
a pair, **any** movement is attributable to `ŝ`.

Preregistered reporting threshold: a change is called **material** if the per-item mean |Δg| exceeds
**0.01** (≈ 2 % of the observed operating value) or if the category-level AUROC moves by more than
**0.05**. Below both, it is reported as immaterial. This threshold is set now, blind.

## 7. Experiment 2 — forced 0.5 / 0.5 intervention

Same eight states, same items, same order. Condition A = `full` (native gate); condition B = `fixed05`.

**Inference intervention only.** It does **not** establish how a fixed-fusion model would learn from
scratch. That would require a matched from-scratch training run, which is out of scope and needs a
new GO decision (§11). This sentence is to be reproduced verbatim in the recap.

### 7.1 Primary comparison — paired, item by item
2×2 contingency of `full_exact` × `fixed05_exact`, per state, per convention:

| | `fixed05` correct | `fixed05` wrong |
|---|---|---|
| **`full` correct** | unchanged correct | **errors gained** |
| **`full` wrong** | **errors recovered** | unchanged wrong |

* **Test:** exact **McNemar** on the discordant cells (binomial, two-sided). Marginal accuracies are
  reported but are *not* the test — the items are paired.
* **Breakdown:** the same 2×2 within each route-competence category of §6.1, and by frequency
  quintile and length (both computed on the frozen lexicon metadata).
* **Reported metrics:** R canonical and genuine free-AR on the FULL/gated route. **Naming,
  Comprehension and the isolated WM/LTM routes are NOT re-evaluated under `fixed05`**: none of them
  constructs a gate, so the intervention is **mathematically inert** for all four
  (`CODE_AUDIT_GATE.md` §13). They are recorded once as *invariants*, with effect exactly zero.
  Re-running them would spend compute to re-derive an identity and would invite reporting a
  by-construction null as an empirical finding.
* **Multiplicity:** the primary test is the overall McNemar per state per convention (8 × 2 = 16
  planned tests). Holm correction across those 16. Every category/quintile breakdown is explicitly
  **descriptive** and carries no p-value.

### 7.2 The interpretive caveat that must travel with the result
The training objective included `λ_gate · (mean(g) − 0.5)²` with `usage_prior = 0.5`
(`CODE_AUDIT_GATE.md` §8): the model was optimised under mild pressure **toward** the symmetry that
`fixed05` imposes. So:

> A null or near-null result for `fixed05` is **weaker evidence against confidence-driven weighting
> than it appears**, because the network was trained not to depend on asymmetry.

**What must NOT be said.** The intervention must not be described as a "~0.02 perturbation". The only
established fact is that the historical **position-weighted** mean gate is 0.517–0.521, so replacing
it with a constant 0.5 moves that **signed mean** by about 0.02. That says nothing about item-level
`|g − 0.5|`, about the variance or shape of the `g` distribution, or about the per-item effect on the
logits, which is

```
Δlogits_i = (0.5 − g_i) · (ltm_logits_i − wm_logits_i)
```

— a **product**, which can be far from small wherever the two routes disagree, even when `g_i ≈ 0.5`.
The magnitude of the intervention is therefore an **open empirical question**, resolved by the
item-level distribution that Experiment 1 produces, not by the recorded mean.

The training-pressure caveat must be stated in the recap **whatever the outcome**, including if the
outcome is "0.5/0.5 is behaviourally equivalent". It is the single most likely way this experiment
could be over-read.

## 8. Secondary descriptive diagnostics
`g` vs lexical frequency · `g` vs length · `g` vs route correctness · the global `g` distribution
against the attainable range [0.032, 0.646]. **Descriptive. No hypothesis test, no correction, no
gate parameter optimised.**

## 9. What may not be concluded
* No cross-witness significance test. Four witnesses from three seeds of one historical cohort are
  not an independent population (`PHASE08_SOURCE_OF_TRUTH.md`: "the historical cohort, not an
  independent population").
* No lesion interpretation. `SCIENTIFIC_LESION_READINESS=NOT_ESTABLISHED` stands.
* No claim about a from-scratch fixed-gate model (§7).
* No revision of any HARD_FROZEN Phase 1–8 claim. The A6 correction touches one diagnostic-only
  column and no report-safe claim.
* No statement that the gate "chooses" or "arbitrates" (§1).

## 10. Lexicality — provenance-gated, explicitly secondary
The primary population has no pseudowords (§4). A `word` vs `pseudoword` contrast requires
`data/eval_external/{wfe,ssp}_eval.tsv` and/or `data/raw-nwr_swp/`, which are **not** covered by the
Phase-8 freeze manifest.

> **Gate:** run the lexicality battery **only if** those assets' provenance can be established
> against a canonical record. If it can, report it in a clearly separated section marked
> `SECONDARY / NON-FROZEN POPULATION`. If it cannot, report **NOT RUN — PROVENANCE NOT ESTABLISHED**
> and say so in NEGATIVE RESULTS.

Merging a frozen and an unfrozen population into one table is prohibited.

## 11. Stop condition
**Execution stops after Experiments 1 and 2.** Return to CENTRAL STEERING with the evidence package.
A matched fixed-0.5 *training* experiment, any `semantic_attractor` work, any `alpha`/`tau` change and
lesioning V2 each require a **new GO decision**.

## 12. Outcome mapping — fixed before results
From the brief §8. Mapping is declared now so that no result can be retro-fitted.

The **numeric criteria are unchanged** wherever they remain mathematically meaningful. Only the
interpretation **labels** change, to match what the criteria can actually support.

| Outcome | Label (corrected) | Declared if |
|---|---|---|
| **A** | **confidence-driven weighting is informative and causally load-bearing for intact repetition inference** | the H1′ effect is strong in the predicted direction (AUROC ≤ 0.35, CI excluding 0.5) **and** `fixed05` loses items concentrated in the competence cells |
| **B** | **confidence-driven weighting is dispensable for intact repetition inference: fixed 0.5 is behaviourally equivalent** | McNemar not significant after Holm **and** \|Δ accuracy\| < 0.002 on FULL/gated repetition |
| **C** | ~~the gate is an architectural suspect~~ — **UNREACHABLE BY CONSTRUCTION ON THESE STATES** | *(non-operative; see below)* |
| **D** | the problem is upstream of the gate | both conditions show the same ventral deficit (isolated LTM ≈ 0.876–0.892 unchanged) and `fixed05` does not move it |
| **E** | repair changes the mechanism | source→repaired gate change is **material** by the §6.6 threshold |

Neither A nor B is a claim about *adaptive routing*. The gate is not adaptive
(`CODE_AUDIT_GATE.md` §5); what is at stake is whether a **fixed, confidence-driven** weighting is
load-bearing at inference.

**Outcome C is UNREACHABLE BY CONSTRUCTION on these states**, for three independent reasons:
1. FULL/gated repetition is already **exact (0 errors)** in all eight states, so `fixed05` cannot
   improve it — there is nothing above exact;
2. Comprehension, Naming and the isolated WM/LTM routes **do not depend on the repetition gate**
   (`CODE_AUDIT_GATE.md` §13), so the "does not degrade" half of the old criterion is true by
   construction and carries no evidential weight;
3. consequently no combination of measurements on these states can satisfy the old criterion in an
   informative way.

Its original criterion text is retained above **as historical record only** and is **non-operative**.
C must not be declared. If the gate is to be put under suspicion, that requires different states or a
different experiment, and a new GO decision.

Outcomes are **not** mutually exclusive: D can and likely will co-occur with B, since isolated LTM is
~0.88 in every state before anything is intervened on. If the data fit none of these, that is
reported as a sixth, named outcome rather than forced into the nearest box (brief §8).

## 13. Execution

```bash
cd wt-gating-diagnostics
python3 -m pytest tests/test_gate_route_diagnostics.py -q          # contract precondition
python3 scripts/gating_diagnostics/run_gate_route_audit.py \
    --manifest paper_programme/gating_route_diagnostics/checkpoint_manifest.tsv \
    --out-dir  paper_programme/gating_route_diagnostics \
    --state-id ALL
```

Outputs, all reproducible from `figure_source_data/`:
`item_level_gate_route_metrics.tsv` · `summary_metrics.json` · `figure_source_data/` · `figures/` ·
`SHA256SUMS` · results written back into `GATING_ROUTE_DIAGNOSTICS_RECAP.md`.

**Cost note.** Eight states × 29,571 items × 4 routes × 2 conventions, on CPU. The runner takes
`--state-id` and `--limit` and writes one shard per state, so it can be executed incrementally and
resumed. A `--smoke` path (**400 items, one state** — `SMOKE_DEFAULT_LIMIT = 400`) validates the
full pipeline end to end and must be run and inspected before the full pass.

**Two execution-safety guards, both enforced in code and both pinned by tests:**

1. **Smoke output is quarantined.** Every smoke artifact is written under `_smoke_not_results/`
   with a `_SMOKE_TEST_ONLY` marker, and `assert_quarantined` hard-stops if a smoke invocation would
   ever resolve to a full-result path or omit the marker.
   Pinned by `test_smoke_output_is_quarantined`.

2. **A truncated run cannot masquerade as a result.** `--limit` truncates the evaluated population,
   so `assert_full_population` **hard-stops on `--limit` without `--smoke`**. Full scientific
   execution always uses the **complete canonical repetition population** (§4) — the paired
   item-by-item comparisons and the §6.4 power rule are defined over it, and a partial-population
   shard written into `figure_source_data/` would be indistinguishable from a full result after the
   fact. `--limit` is reserved for quarantined smoke/test use.
   Pinned by `test_limit_without_smoke_is_refused` and
   `test_limit_guard_is_wired_into_main_before_any_write`.

Smoke and truncated artifacts are never scientific results.

## 14. Amendments

### AMENDMENT 1 — 2026-09-15 — the primary contrast is structurally empty; H1 is reformulated

**What was known when this amendment was made.** Only the FROZEN Phase-8 battery JSONs — the same
`rep_canonical_wm_errors` / `rep_canonical_ltm_errors` fields already tabulated in
`checkpoint_manifest.tsv`. **No Experiment-1 or Experiment-2 result existed.** The smoke run
(400 items, `W3_SRC`) had been executed to validate the pipeline; its numbers are not used here and
are not a result. This amendment is derivable from the frozen record alone.

**The finding.** The dorsal route is at or within two items of ceiling on the repetition population
in every one of the eight states:

| state | WM errors (canonical) | LTM errors | FULL errors | max possible `n(LTM_ONLY_CORRECT)` |
|---|---|---|---|---|
| W1_SRC / W1_REP | 0 / 0 | 3385 / 3402 | 0 / 0 | **0** |
| W2_SRC / W2_REP | 2 / 2 | 3423 / 3431 | 0 / 0 | **2** |
| W3_SRC / W3_REP | 0 / 0 | 3199 / 3193 | 0 / 0 | **0** |
| W4_SRC / W4_REP | 1 / 1 | 3647 / 3662 | 0 / 0 | **1** |

Since `LTM_ONLY_CORRECT` requires the dorsal route to *fail*, its size is bounded above by the WM
error count. **`n(LTM_ONLY_CORRECT) ≤ 2` in every state**, against a preregistered power floor of 30
(§6.4). `NEITHER_CORRECT` is bounded the same way.

**Consequence.** The §6.2 hypothesis

> ~~H1: median g(LTM_ONLY_CORRECT) > median g(WM_ONLY_CORRECT)~~

**cannot be tested on this population.** It is not underpowered by accident; it is empty by
construction. Under the §6.4 power rule it would return `UNDERPOWERED` for all eight states, which
would be a true but uninformative report. Declaring this now, blind, is preferable to running it and
announcing an empty cell as though it were a discovery.

**This is itself a result, and belongs in the recap.** The route-competence structure of the mature
model is effectively **binary**: every item is dorsally correct, and items divide into those the
ventral route also gets right (~26,200) and those it does not (~3,200–3,700). "When only one route is
correct" almost always means "only the dorsal route is correct". Any dual-route interpretation of
these states must start from that fact.

**Replacement diagnostic — a DIFFERENT question, not a restatement of H1:**

> **H1′ — VENTRAL CONFIDENCE CALIBRATION DIAGNOSTIC** (directional, preregistered as of this
> amendment):
>
> **Question.** Given that the dorsal route is almost always correct, does ventral lexical
> confidence — and therefore `g` — *decrease* on items for which the ventral route is wrong,
> compared with items for which the ventral route is correct?
>
> **Statistic.** `median g(WM_ONLY_CORRECT) < median g(BOTH_CORRECT)`.

**H1′ is not H1 restricted; it is a different question.** H1 asked about *relative route competence*
— a comparison between two routes. H1′ can only ask about *ventral self-knowledge*, because the
dorsal side of the comparison is constant (correct almost everywhere). Confirming H1′ licenses
exactly one interpretation:

> **ventral confidence is informative about ventral failure.**

It does **not** license either of:

> ~~the gate tracks relative route competence~~
> ~~the gate arbitrates between routes~~

Both are excluded structurally, not merely unproven: the gate is blind to dorsal-exclusive parameters
and dorsal activations (`CODE_AUDIT_GATE.md` §5), and `g` is rank-identical to `c_LTM`, so H1′ is a
statement about a ventral confidence signal and nothing else. The recap must use the calibration
wording.

H1′ is well powered on both sides (≈ 3,200 vs ≈ 26,200), so the §6.4 rule is satisfied comfortably.

**Statistics for H1′** are unchanged from §6.3, with `WM_ONLY_CORRECT` as the positive class and the
sign of the expected effect reversed: AUROC **< 0.5** (equivalently Cliff's δ < 0) is the direction
predicted by H1′. Report AUROC of `g` separating `WM_ONLY_CORRECT` from `BOTH_CORRECT`, Cliff's δ,
the bootstrap CI, and the median difference. The §6.5 mediation caveat applies unchanged and with
full force: `g` is rank-identical to `c_LTM`, so a confirmed H1′ means **ventral confidence is
informative about ventral failure**, not that the gate arbitrates.

H1 is retained in the record and in the output schema, and is reported as
**`STRUCTURALLY_UNTESTABLE — n(LTM_ONLY_CORRECT) <= 2`**, never silently dropped and never
represented as having been answered by H1′.

**Consequence for Experiment 2.** FULL is at **0 errors in every state**, so `errors_recovered` is
necessarily 0 and the intervention can only *lose* items. The §7.1 2×2 degenerates to a one-sided
count of errors gained. The McNemar machinery is kept unchanged (it handles this correctly and
reports the exact binomial), but the recap must state that **no recovery was possible**, so a finding
of "0.5/0.5 recovers nothing" is a property of the starting point, not evidence about the gate.
The informative quantity is `errors_gained` and its distribution across competence category,
frequency and length.

**Consequence for outcome mapping (§12).** Outcome **C** ("`fixed05` significantly improves FULL")
is **unreachable on repetition** for these states, since FULL is already exact. Outcome **A**'s first
clause is re-expressed against H1′ (AUROC ≤ 0.35 with CI excluding 0.5, i.e. a strong effect in the
predicted direction). Outcomes B, D and E are unaffected.

> *Superseded in part by AMENDMENT 2:* the suggestion here that C "could only be declared on the C/N
> contract metrics or on the isolated-route batteries" is **withdrawn**. Those quantities do not
> depend on the repetition gate at all, so they cannot support outcome C either. C is
> **UNREACHABLE BY CONSTRUCTION** and non-operative.

*No other section of this contract is changed by this amendment.*

---

### AMENDMENT 2 — 2026-09-15 — independent forensic review corrections

**Trigger.** CENTRAL STEERING review of an independent Fable audit of this package.

**Timing and epistemic position — stated explicitly.**

* These corrections were made **BEFORE any execution of Experiment 1 or Experiment 2.**
* `FULL_EXPERIMENT_EXECUTED = NO` at the time of this amendment and at the time of its commit.
* The **only** data seen when these corrections were made were:
  1. **frozen Phase-8 metrics** (battery JSONs, `PHASE08_SOURCE_OF_TRUTH.md`, checkpoint configs), and
  2. **quarantined smoke output** (400 items, one state, `W3_SRC`, `TEST_ONLY`).
* **No full experimental result existed, and none was consulted.** Nothing in this amendment is
  responsive to an outcome.

**What the independent review CONFIRMED** (unchanged, no correction needed): gate equation and
orientation; gate has no trainable parameters; gate blind to dorsal-exclusive parameters and
activations; `fixed05` algebra; route isolation; the Phase-8 repaired `gate_mean` instrumentation
defect; checkpoint provenance; V7 and pre-existing scientific artifacts untouched.

#### 1. Interpretation labels that CHANGED

| Item | Before | After |
|---|---|---|
| Outcome **A** | "adaptive routing has causal value" | "**confidence-driven weighting is informative and causally load-bearing for intact repetition inference**" |
| Outcome **B** | "learned routing unnecessary at inference" | "**confidence-driven weighting is dispensable for intact repetition inference: fixed 0.5 is behaviourally equivalent**" |
| Outcome **C** | "the gate is an architectural suspect" | **UNREACHABLE BY CONSTRUCTION ON THESE STATES** — non-operative; original criterion retained as historical record only |
| **H1** | `STRUCTURALLY_EMPTY` | **`STRUCTURALLY_UNTESTABLE`**, retained in the record, never represented as answered by H1′ |
| **H1′** | described as "the same scientific question" as H1 | **VENTRAL CONFIDENCE CALIBRATION DIAGNOSTIC** — a *different* question (§below) |
| gate range | "the gate cannot saturate" | **asymmetric attainable range** (§below) |
| `fixed05` magnitude | "a ~0.02 perturbation" | **an open empirical question**; only the *signed position-weighted mean* is known to move by ≈ 0.02 |
| route structure | "the mature states are not dual-route"; "the dorsal route carries everything" | "**the dorsal route is nearly sufficient for exact canonical repetition, whereas the ventral route is not**" |
| LTM-exclusive tensors | 20 | **16** (verified at the checkpoints' own config) |

#### 2. Mathematical criteria that did NOT change

Explicitly preserved, unchanged, and still frozen:

* the §6.4 power rule (`min n ≥ 30` per cell);
* Cliff's δ, AUROC, the `δ = 2·AUROC − 1` identity assertion;
* bootstrap: 10,000 resamples, stratified, `seed = 20260915`;
* exact paired McNemar on discordant pairs, Holm across the 16 planned tests;
* the §6.6 materiality thresholds for outcome E (mean |Δg| > 0.01, or category AUROC shift > 0.05);
* outcome **A**'s numeric clause (AUROC ≤ 0.35 with CI excluding 0.5) and outcome **B**'s
  (|Δ accuracy| < 0.002, McNemar not significant after Holm);
* H1′'s statistic and predicted direction;
* the item population, its order, and the four witness pairs.

Only labels, interpretations and implementation guards changed. **No numeric threshold was moved.**

#### 3. Asymmetric attainable gate range

The blanket claim "the gate cannot saturate" is **withdrawn**. At `α = 2.0`, `gate_threshold = 0.7`,
with cosine confidence in [−1, 1]:

* **ventral weight `g ∈ [0.0323, 0.6457]`**
* **dorsal weight `1 − g ∈ [0.3543, 0.9677]`**

Therefore:

* **strong dorsal commitment IS possible** — up to ≈ **29.9 : 1** dorsal-to-ventral;
* **strong ventral commitment IS structurally impossible** — at most ≈ **1.82 : 1**
  ventral-to-dorsal.

The finding is the **asymmetry of the attainable range**, not an absence of saturation.

#### 4. H1′ reinterpreted as ventral-confidence calibration

H1′ is **not** H1 restricted to a smaller sample. H1 compared *two routes*; H1′ cannot, because the
dorsal side is constant (correct almost everywhere). H1′ asks:

> given that the dorsal route is almost always correct, does ventral lexical confidence — and hence
> `g` — decrease on items for which the ventral route is wrong, compared with items for which it is
> correct?

Statistic unchanged: `median g(WM_ONLY_CORRECT) < median g(BOTH_CORRECT)`.

Licensed interpretation: **ventral confidence is informative about ventral failure.**
Excluded interpretations: *the gate tracks relative route competence*; *the gate arbitrates between
routes*. Both are excluded structurally, not merely unproven.

#### 5. Naming / Comprehension invariance

Traced through the canonical evaluators (`CODE_AUDIT_GATE.md` §13): Comprehension is
`encode → ŝ → cosine retrieval`; Naming is `ltm.decode_from_s_hat → motor`; the isolated routes
bypass the gate by construction. **None constructs a gate.**

Therefore `fixed05` has **mathematically zero** effect on Comprehension, Naming, isolated WM-only and
isolated LTM-only performance. §7.1 no longer evaluates them under the intervention; they are
recorded once as invariants. No redundant evaluation that cannot change is added.

#### 6. Free-AR cap alignment

The diagnostic free-AR evaluator previously used a local `max_steps = 24`, against the historical
`FREE_AR_MAX_STEPS = 12`. A different cap changes what counts as non-termination and would have made
the diagnostic free-AR numbers non-comparable with the frozen record.

Corrected by **importing the historical constant directly** rather than restating it, so the two can
never drift. The historical evaluator is **not modified**. Pinned by
`test_free_ar_cap_matches_the_historical_evaluator`.

#### 7. Smoke-output isolation

Smoke artifacts could previously land on the same shard filenames as a full run. Corrected:

* all smoke output goes under `_smoke_not_results/`, with a `_SMOKE_TEST_ONLY` marker;
* `assert_quarantined` **hard-stops** if a smoke invocation would resolve to a full-result path, or
  if a smoke summary would be written without the marker;
* the smoke item count is now stated consistently as **400** everywhere
  (`SMOKE_DEFAULT_LIMIT = 400`), matching the archived artifact.

Pinned by `test_smoke_output_is_quarantined` and `test_smoke_default_limit_is_documented_value`.

#### 8. Two named gate means

The new package never reports an unqualified `gate_mean`. Experiment 1 reports **two distinct,
explicitly named** quantities:

* **`gate_mean_item_level`** — one `g` per item, averaged over items. The gate is word-level, so this
  is the scientifically meaningful statistic.
* **`gate_mean_position_weighted_historical`** — a reconstruction of the historical convention, which
  flattens the gate over `(B, S, 1)` *including padding positions*, weighting each item by its
  batch's padded decoder width. Reported **only** for reconciliation with the frozen Phase-8 column,
  and explicitly batching-dependent.

Pinned by `test_two_gate_means_are_named_and_distinct`.

#### 9. Shared-parameter caveat recorded precisely

The gate is blind to dorsal-**exclusive** parameters and dorsal activations. It is **not** blind to
`phon_embed.weight`, which the two routes share and which dorsal training can move. The routes also
share the downstream `motor.proj` readout. Functional route isolation therefore means **isolated
premotor contribution into a shared readout**, not anatomically independent subnetworks.

#### 10. Provenance precision

`CANONICAL AUTHORITY NAME` and `LOCAL ARCHIVE PATH` are now distinguished rather than conflated, and
the meeting notes are recorded as an **external input** with their measured hash and an explicit
`NOT PRESENT` for any in-repository path. `checkpoint_manifest.tsv` now carries
`canonical_archived_path` alongside `historical_recorded_path` and the verifying `sha256`. **No
historical JSON was rewritten.**

**AMENDMENT 1 is preserved above in full. Nothing in it is erased; where AMENDMENT 2 supersedes part
of it, the superseding note is inline and marked.**

*No numeric criterion, item population, checkpoint, model file, gate parameter, loss, training code
or historical artifact was changed by this amendment.*
