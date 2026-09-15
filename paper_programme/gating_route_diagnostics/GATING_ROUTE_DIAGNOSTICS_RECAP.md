# GATING / ROUTE DIAGNOSTICS — RECAP

**Workstream:** LICHTHEIM3 — GATING / ROUTE DIAGNOSTICS (POST_STAGE / PAPER_PROGRAMME)
**Branch:** `paper-programme/gating-route-diagnostics` · **Base commit:** `78f5505`
**Date:** 2026-09-15
**Stage:** **AUDIT AND CONTRACT COMPLETE — EXPERIMENTS 1 AND 2 NOT YET EXECUTED.**

> Per the workstream time constraint (§11 of the brief), work before the Mines defense on
> Thursday 17 September is limited to code audit, experiment contract and implementation
> preparation. The RESULTS section below is deliberately empty of experimental results and is
> filled only after the full pass runs. Nothing here requires Louis to interpret results now.
>
> This document incorporates the corrections of **AMENDMENT 2** (independent forensic review),
> applied **before** any execution of Experiment 1 or Experiment 2.

---

## QUESTION

**Primary.** Does the deployed gate track **relative route competence** item by item — when only one
route is correct, does the gate preferentially weight that route?

**Secondary.** What does the gate causally contribute at inference, measured against a preregistered
`0.5 / 0.5` symmetric-fusion baseline?

Frequency is a descriptive covariate, not the entry point.

## CHECKPOINTS

Eight states: the four Phase-8 exact FULL/gated witnesses, each paired with its provenance-established
pre-repair source. Full table with hashes in `checkpoint_manifest.tsv`.

| Witness | Seed | u | Selection | Source C → Repaired C | Source → Repaired isolated-LTM errors |
|---|---|---|---|---|---|
| W1 `V5_seed20_u3600` | 20 | 3600 | retrospective | 32 → 0 | 3385 → 3402 |
| W2 `V5_seed21_u3400` | 21 | 3400 | retrospective | 26 → 0 | 3423 → 3431 |
| W3 `V6_seed19_u3825` | 19 | 3825 | prospective first-hit | 34 → 0 | 3199 → 3193 |
| W4 `V6_seed20_u3040` | 20 | 3040 | prospective first-hit | 42 → 0 | 3647 → 3662 |

Within each pair Arm A modifies only `ltm.to_semantic.2.{weight,bias}`, so **the entire dorsal route
is bit-identical** and any gate movement is attributable to `ŝ` alone. That is what makes the paired
design able to answer interpretation-space outcome E.

## PROVENANCE

**Canonical authority — name vs local path, distinguished, not conflated.**

| role | identifier | hash |
|---|---|---|
| **CANONICAL AUTHORITY NAME** — the baseline the Phase-8 arbitration names as its sole canonical working baseline (`FABLE_PROVENANCE_ARBITRATION_REPORT.md`) | `LICHTHEIM3_SOURCE_OF_TRUTH_CANONICAL_REPAIRED_2026-09-14.zip` | `22f9e6911a504e20767c320b2a98de0a7d9bba438a6c5a2889d96fe7f4c75995` *(as recorded in the arbitration report; the zip itself is not present locally — only its unpacked staged copy)* |
| **LOCAL ARCHIVE PATH** — the arbitration-output package actually read for this workstream | `LICHTHEIM3_SOURCE_OF_TRUTH_CANONICAL_FABLE_ARBITRATED_2026-09-14.zip` (supplied externally) | `6d3a0a6615b90b0b45de44dd1aac2b21b7cac594cf6f0ef3fb259cab4a92c628` *(measured)* |
| unpacked staged copy in the project tree | `stage_source_of_truth/LICHTHEIM3_SOURCE_OF_TRUTH_CANONICAL_REPAIRED_2026-09-14/` | directory, not hashed as a unit |

`CANONICAL_MASTER_REPAIR_STATUS=SUCCESS`; Phases 1–8 HARD_FROZEN. The arbitrated package is the
*provenance-arbitration output* built on the repaired baseline; it did not amend executable evidence,
checkpoint metrics, configs or preregistrations.

**EXTERNAL INPUT — hypothesis source (not historical authority).**

```
EXTERNAL INPUT: MEETING_NOTES_YAIR_EMMANUEL_2026-09-14.md
status:         CONTEMPORARY_RELAYED_MEETING_NOTES
sha256:         c1983db3ee244182e3a3f11c25ed8500c36a01f5dd53d5b454b01d4a0032b70e
in-repository path: NOT PRESENT
```

The notes were supplied externally and do **not** live inside this git workspace; no repository path
is claimed for them. Used to identify questions (§8 `gate=0.5/0.5` baseline, §9 gate distribution,
§15 Q1–Q3), never as historical authority.
* **All 8 checkpoint artifacts verified: on-disk SHA256 matches the Phase-8 record, 8/8.**
* `checkpoint_manifest.tsv` records, per state, the **canonical archived path** (preferred), the
  **historical recorded path** from the frozen battery JSON, and the verifying `sha256`. All four
  repaired heads have immutable archived copies under `archives/`, each hash-identical to the
  `head_sha256` in the frozen record. W1's historical path is a *volatile session scratch* path: it
  still resolves and matches, but the archived copy is authoritative. Three of the four **sources**
  exist only as working copies (`probe_*`), marked `NOT_ARCHIVED_working_copy_only`.
  **No historical JSON was rewritten.**
* Model reconstruction reuses `frozen_head_probe.build_trainer` and `_isolated_model` verbatim —
  the same path `ceiling_source_completion.cmd_battery` used to produce the frozen witness metrics.
* Not used as scientific authority: slide decks, ARTICLE_SPINE / DEFENSE_SPINE, unprovenanced plots,
  exploratory logs. The exploratory lesion work (`lichtheim3-brain-damage@a5c787a`) is cited only for
  a reused *mechanism* and an independently reproduced *equality check*, never for a result.

## CODE COMMIT

Base `78f5505` (`prospective/rn-detector-v6`) — the V6-completion tip, i.e. the exact lineage that
produced the Phase-8 repaired witnesses (P8-IMP-009). No file under `models/` was modified.
New code is additive: `gating_diagnostics/`, `scripts/gating_diagnostics/`,
`tests/test_gate_route_diagnostics.py`, `paper_programme/gating_route_diagnostics/`.

## INTERVENTIONS

| Condition | Definition |
|---|---|
| `full` | native confidence gate, `g = σ(2.0·(c_LTM − 0.7))` |
| `wm` | dorsal only — exact `g = 0` limit |
| `ltm` | ventral only — exact `g = 1` limit |
| `fixed05` | **the preregistered intervention**, `g ≡ 0.5` |

`fixed05` is a **read-only post-hoc recombination** of the logits `forward()` already returns, exact
because `motor` is a single affine map and the blend weights sum to 1. Validated against an
independent forward-hook implementation. **No model code modified. No parameter written. No ratio
other than 0.5 searched.**

Both decoding conventions reported: canonical forced-length AR (comparable to the frozen record) and
genuine free-AR (over-generation counts as error).

## METRICS

Item-level: gate, effective ventral/dorsal weight, lexical confidence/margin/density, per-route exact
match under both conventions, route-competence category, plus `word`/`rank`/`length`/`zipf_approx`
aligned by construction from the frozen lexicon.

Statistics (frozen in `EXPERIMENT_CONTRACT.md` before execution): Cliff's δ, AUROC with the explicit
`δ = 2·AUROC − 1` identity, stratified bootstrap CI (10,000 resamples, seed 20260915), exact paired
McNemar with Holm correction across the 16 planned tests, Spearman for descriptive covariates.

## RESULTS

> **NOT YET EXECUTED.** To be filled from `summary_metrics.json` after the full pass.
> A **400-item** smoke run (one state, `W3_SRC`) validated the pipeline end to end; its output is
> quarantined under `_smoke_not_results/` with a `_SMOKE_TEST_ONLY` marker. Smoke numbers are
> **not** results and are not reported here.

### A. STRUCTURAL FINDINGS — established, not pending

These are **audit findings, not experimental results**. Each is verified numerically
(`tests/test_gate_route_diagnostics.py`, 20 tests passing) or read directly off frozen artifacts.
They do not depend on Experiments 1 or 2 and will not change when those run.

1. **The gate has no learnable parameters.** `g = σ(α·(c_LTM − τ))` with `α`, `τ` fixed
   hyperparameters. "Learned gate" is a misnomer: the only learning reaching `g` is inside the
   ventral encoder producing `ŝ`.
2. **The gate is blind to the dorsal route's state.** Perturbing all ten WM-exclusive parameter
   tensors moves `wm_logits` by `max|Δ| = 1.03e+01` and moves `g` by `max|Δ| = 0.000e+00`, exactly.
   Precisely: blind to dorsal-**exclusive** parameters and dorsal activations. It is **not** blind to
   `phon_embed.weight`, which the routes share and which dorsal training can move; the routes also
   share the `motor.proj` readout. The gate therefore cannot implement adaptive routing on
   inference-time route competence; any apparent tracking is mediated entirely by `c_LTM`.
3. **The attainable gate range is ASYMMETRIC.** At the cohort's `α = 2.0, τ = 0.7`, with cosine
   confidence in [−1, 1]: **ventral weight `g ∈ [0.0323, 0.6457]`**, **dorsal weight
   `1−g ∈ [0.3543, 0.9677]`**. So **strong dorsal commitment is possible** (up to ≈ 29.9 : 1) while
   **strong ventral commitment is structurally impossible** (at most ≈ 1.82 : 1). The finding is the
   asymmetry — not, as earlier stated, that the gate "cannot saturate".
4. **`fixed05` is exactly a post-hoc affine recombination.** `motor` is a single affine map and the
   blend weights sum to 1, so `motor(g·ltm + (1−g)·wm) = g·ltm_logits + (1−g)·wm_logits`
   (`max|Δ| = 1.6e-07`). The intervention needs no model modification, and is cross-checked against
   an independent forward-hook implementation.
5. **Gate behaviour after terminal repair has never been measured.** `full_battery` computes
   `gate_mean` on `tr.model` (the source) while threading the repaired `model` into every other
   evaluator, so all four repaired values are bit-identical to their sources. The frozen quantity is
   additionally a *position-weighted* mean, not an item-level one. Affects one diagnostic-only
   column; **no Phase 1–8 report-safe claim, freeze manifest, or established result is revised.**
   Consequence: outcome **E** is currently untested, not tested-and-negative.
6. **H1 is STRUCTURALLY UNTESTABLE on this population.** The dorsal route is at or within two items
   of ceiling in every state (WM errors 0, 2, 0, 1), so `n(LTM_ONLY_CORRECT) ≤ 2` everywhere against
   a power floor of 30. The supported statement is narrow: **the dorsal route is nearly sufficient
   for exact canonical repetition, whereas the ventral route is not.** It is retained in the record
   as `STRUCTURALLY_UNTESTABLE`, and is **not** answered by H1′, which is a different question.
7. **Naming and Comprehension do not involve the gate.** Comprehension is
   `encode → ŝ → cosine retrieval`; Naming is `ltm.decode_from_s_hat → motor`; the isolated routes
   bypass the gate by construction. `fixed05` therefore has **mathematically zero** effect on C, N,
   isolated WM-only and isolated LTM-only performance. These are recorded as invariants and are not
   re-evaluated under the intervention.
8. **`fixed05` can only lose, never recover, on repetition.** FULL is at 0 errors in every state, so
   `errors_recovered ≡ 0` by construction. Combined with finding 7, outcome **C** is
   **UNREACHABLE BY CONSTRUCTION** and is non-operative.

### B. PENDING EMPIRICAL QUESTIONS — not yet answered

None of these can be stated until the full pass runs. They are listed so that the structural findings
above are not over-read as answering them.

1. **The item-level `g` distribution.** Only a *position-weighted mean* (0.517–0.521) was ever
   recorded. The item-level mean, the variance, the shape (narrow unimodal vs bimodal) and the mean
   `|g − 0.5|` are all **unknown**.
2. **How large the `fixed05` intervention actually is.** The per-item logit change is
   `(0.5 − g_i)·(ltm_logits_i − wm_logits_i)` — a **product**, which can be far from small wherever
   the routes disagree, even when `g_i ≈ 0.5`. It must **not** be described as a "~0.02
   perturbation": what moves by ≈ 0.02 is the recorded signed mean, nothing else.
3. **H1′ — ventral confidence calibration.** Does `g` decrease on items the ventral route gets wrong
   (`median g(WM_ONLY_CORRECT) < median g(BOTH_CORRECT)`)? Well powered (≈ 3,200 vs ≈ 26,200),
   implemented, **not run**.
4. **Source → repaired per-item gate changes.** Within each pair the dorsal route is bit-identical,
   so any movement is attributable to `ŝ` alone. Never measured (finding A5).
5. **Errors gained under `fixed05`**, and their distribution across competence category, frequency
   and length.

## NEGATIVE RESULTS

* **The brief's first scientific question cannot be answered in the form posed**, twice over: once
  structurally (A2 — the gate cannot see the dorsal route's state) and once empirically (A6 — the
  discriminative cell holds ≤ 2 items). Both are reported as findings, not worked around. H1′ is a
  **different** question (ventral-confidence calibration), not a rescue of H1.
* Outcome **C** is **UNREACHABLE BY CONSTRUCTION** and non-operative (A7 + A8).
* `fixed05` cannot move Comprehension, Naming, or either isolated route — exactly zero, by
  construction (A7). Those evaluations are deliberately **not** repeated under the intervention.
* Lexicality (`gate` vs word/pseudoword) is **not computable on the frozen population**, which is
  entirely real words. It is deferred to a provenance-gated secondary battery and will be reported as
  `NOT RUN — PROVENANCE NOT ESTABLISHED` if the pseudoword assets cannot be tied to a canonical record.
* `margin` and `density` are computed and plumbed into the gate at every forward pass and then
  discarded; only `confidence` is read. Recorded as a design fact; no variant using them is built.

## LIMITATIONS

* Four witnesses from three seeds of **one historical cohort**. Not an independent population; no
  cross-witness significance test is performed.
* Gate-level route isolation is **functional, not anatomical**. The routes share
  `phon_embed.weight` — the only shared *parameter* tensor (10 WM-exclusive, **16** LTM-exclusive at
  the checkpoints' own config) — **and** the downstream `motor.proj` readout. Route isolation
  therefore means **isolated premotor contribution into a shared readout**, not anatomically
  independent subnetworks. `phon_embed.weight` is also a channel by which dorsal training can reach
  `ŝ`, and hence the gate, indirectly.
* This is an **inference intervention**. It says nothing about how a fixed-fusion model would learn
  from scratch.
* Training included `λ_gate·(mean(g) − 0.5)²` with `usage_prior = 0.5`, so the model was optimised
  under mild pressure **toward** the symmetry `fixed05` imposes. A null result for `fixed05` is
  therefore weaker evidence against confidence-driven weighting than it appears. This caveat stands
  whatever the outcome.
* The magnitude of the `fixed05` intervention is **not** established. Only the historical
  *position-weighted* mean gate (0.517–0.521) is known; the item-level distribution was never
  recorded (B1, B2).
* `SCIENTIFIC_LESION_READINESS=NOT_ESTABLISHED` (Phase 8) is unchanged by anything here.
* The `motor`-linearity that makes `fixed05` exact is **architecture-specific**. A semantic-attractor
  loop or a nonlinear readout would break it and require re-verification.

## WHAT WAS LEARNED

* The mechanism question is settled at the level of code: **there is no adaptive controller.** The
  gate is a fixed monotone transform of one scalar — ventral lexical confidence — blind to the
  inference-time state of the route it is nominally arbitrating against.
* Its attainable range is **asymmetric**: it can commit strongly to the dorsal route but never
  strongly to the ventral one. If the project wants genuine arbitration, that asymmetry, not the
  value of `α`/`τ`, is the thing to argue about.
* The interesting answerable question is therefore not "does the gate decide well" but **"is ventral
  confidence informative about ventral failure"** — H1′, well powered, worth answering.
* On this population **the dorsal route is nearly sufficient for exact canonical repetition, whereas
  the ventral route is not** (~11 % from ceiling). This is a statement about *sufficiency*, and it
  does **not** license inferring that the ventral route contributes nothing functionally. Terminal
  repair moves the ventral deficit essentially not at all (+17, +8, −6, +15 errors).
* The 14 Sep meeting's `0.5/0.5` baseline changes the *recorded mean gate* by only ≈ 0.02 — but how
  large the intervention is **per item** is an open question (B2), not a settled one.

## WHAT WAS NOT LEARNED

* Whether `g` varies within its attainable band in a way that is calibrated to ventral failure —
  H1′ is implemented and preregistered but **not yet run**.
* The item-level `g` distribution, and therefore the true magnitude of the `fixed05` intervention.
* Whether terminal semantic repair changes gate behaviour — never measured (finding 5).
* Whether the gate does anything useful under lesion, at other `α`/`τ`, or on pseudowords. Out of
  scope by the STOP condition.
* Whether a fixed-0.5 model **trained** from scratch would behave differently. Requires a new GO.
* Whether a semantic-attractor dynamic helps. Deliberately not approached (brief §2: do not jump to C).

## RECOMMENDED NEXT DECISION

**For CENTRAL STEERING. No decision is requested before the defense.**

1. **Execute Experiments 1 and 2 as contracted** (after the defense). The implementation is complete,
   tested and smoke-validated; the full pass needs no supervision and produces the evidence package
   automatically.
2. **Note for the architecture discussion, independent of those results:** structural findings
   A1–A3 already constrain the design space. A gate that cannot observe the dorsal route's state, and
   whose attainable range permits strong dorsal but not strong ventral commitment, is not performing
   route arbitration in any strong sense. If the project wants adaptive routing, that is a **new
   mechanism**, not a retuning of `α`/`τ` — and it should be decided on these structural grounds
   rather than on a marginal `fixed05` effect.
3. **Carry finding 5 forward** into any future lesion or repair interpretation: the historical
   `gate_mean` column does not mean what its name implies, and any downstream document quoting a
   repaired-state gate value should be corrected.
4. **Do not** train a fixed-gate model, implement `semantic_attractor`, modify `α`/`τ`, launch H512
   training, or start lesioning V2 on the basis of this package. Each needs a separate GO.

## ARTIFACTS

| File | Status |
|---|---|
| `CODE_AUDIT_GATE.md` | complete |
| `EXPERIMENT_CONTRACT.md` (+ AMENDMENTS 1 and 2) | frozen |
| `checkpoint_manifest.tsv` | complete, 8/8 hashes verified |
| `item_level_gate_route_metrics.tsv` | pending full pass |
| `summary_metrics.json` | pending full pass |
| `figure_source_data/` | empty pending full pass (smoke quarantined under `_smoke_not_results/`) |
| `figures/` | pending full pass |
| `SHA256SUMS` | covers the frozen stage |
| `gating_diagnostics/`, `scripts/gating_diagnostics/`, `tests/` | complete, 20 tests passing |

Every plot in `figures/` will be reproducible from `figure_source_data/` alone.
