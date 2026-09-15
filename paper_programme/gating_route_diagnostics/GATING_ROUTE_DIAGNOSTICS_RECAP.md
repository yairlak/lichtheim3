# GATING / ROUTE DIAGNOSTICS — RECAP

**Workstream:** LICHTHEIM3 — GATING / ROUTE DIAGNOSTICS (POST_STAGE / PAPER_PROGRAMME)
**Branch:** `paper-programme/gating-route-diagnostics` · **Base commit:** `78f5505`
**Date:** 2026-09-15
**Stage:** **AUDIT AND CONTRACT COMPLETE — EXPERIMENTS 1 AND 2 NOT YET EXECUTED.**

> Per the workstream time constraint (§11 of the brief), work before the Mines defense on
> Thursday 17 September is limited to code audit, experiment contract and implementation
> preparation. The RESULTS section below is deliberately empty and is filled only after the
> full pass runs. Nothing here requires Louis to interpret results now.

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

* Authority: `LICHTHEIM3_SOURCE_OF_TRUTH_CANONICAL_FABLE_ARBITRATED_2026-09-14`
  (`CANONICAL_MASTER_REPAIR_STATUS=SUCCESS`, Phases 1–8 HARD_FROZEN).
* Hypothesis source: `MEETING_NOTES_YAIR_EMMANUEL_2026-09-14.md`, status
  CONTEMPORARY_RELAYED_MEETING_NOTES — used to identify questions (§8 `gate=0.5/0.5` baseline,
  §9 gate distribution, §15 Q1–Q3), never as historical authority.
* **All 8 checkpoint artifacts verified: on-disk SHA256 matches the Phase-8 record, 8/8.**
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
> A 400-item smoke run has validated the pipeline end to end; smoke numbers are **not** results and
> are not reported here.

### Established before execution, from code and from the frozen record

These are audit findings, not experimental results. Each is verified numerically
(`tests/test_gate_route_diagnostics.py`, 15 tests passing) or read directly off frozen artifacts.

1. **The gate has no learnable parameters.** `g = σ(α·(c_LTM − τ))` with `α`, `τ` fixed
   hyperparameters. "Learned gate" is a misnomer: the only learning reaching `g` is inside the
   ventral encoder producing `ŝ`.
2. **The gate is structurally blind to the dorsal route.** Perturbing all ten WM-exclusive parameter
   tensors moves `wm_logits` by `max|Δ| = 1.03e+01` and moves `g` by `max|Δ| = 0.000e+00`, exactly.
   The gate therefore **cannot** implement adaptive routing on relative route competence; any
   apparent tracking must be mediated entirely by `c_LTM`.
3. **The gate cannot saturate.** At the cohort's `α = 2.0, τ = 0.7`, `g` is confined to
   **[0.032, 0.646]**. The ventral route can never receive more than 64.6 % of the blend. The
   meeting's concern that the gate might "saturate too easily" is inverted: under its own
   hyperparameters it cannot saturate at all.
4. **The deployed gate is already near-symmetric.** Frozen `gate_mean` = 0.517–0.521 across all four
   witnesses. The model already runs at approximately 0.5/0.5 on average, which makes the
   preregistered intervention a ~0.02 perturbation rather than a regime change.
5. **Gate behaviour after terminal repair has never been measured.** `full_battery` computes
   `gate_mean` on `tr.model` (the source) while threading the repaired `model` into every other
   evaluator, so all four repaired `gate_mean` values are bit-identical to their sources. Affects one
   diagnostic-only column; **no Phase 1–8 report-safe claim, freeze manifest, or established result
   is revised.** Consequence: outcome **E** is currently untested, not tested-and-negative.
6. **The route-competence structure is binary, and the brief's primary contrast is empty.** The
   dorsal route is at or within two items of ceiling in every state (WM errors 0, 2, 0, 1), so
   `n(LTM_ONLY_CORRECT) ≤ 2` everywhere against a power floor of 30. "When only one route is correct"
   almost always means "only the dorsal route is correct". The primary hypothesis was reformulated,
   blind, as **H1′: `median g(WM_ONLY_CORRECT) < median g(BOTH_CORRECT)`** — the same question
   restricted to the contrast that exists, and well powered (≈ 3,200 vs ≈ 26,200).
   See `EXPERIMENT_CONTRACT.md` AMENDMENT 1.
7. **`fixed05` can only lose, never recover.** FULL is at 0 errors in every state, so
   `errors_recovered ≡ 0` by construction and outcome **C** is unreachable on repetition.

## NEGATIVE RESULTS

* **The brief's first scientific question cannot be answered in the form posed**, twice over: once
  structurally (finding 2 — the gate cannot see the dorsal route) and once empirically (finding 6 —
  the discriminative cell is empty). Both are reported as findings, not worked around.
* Outcome **C** is unreachable on the repetition contract for these states (finding 7).
* Lexicality (`gate` vs word/pseudoword) is **not computable on the frozen population**, which is
  entirely real words. It is deferred to a provenance-gated secondary battery and will be reported as
  `NOT RUN — PROVENANCE NOT ESTABLISHED` if the pseudoword assets cannot be tied to a canonical record.
* `margin` and `density` are computed and plumbed into the gate at every forward pass and then
  discarded; only `confidence` is read. Recorded as a design fact; no variant using them is built.

## LIMITATIONS

* Four witnesses from three seeds of **one historical cohort**. Not an independent population; no
  cross-witness significance test is performed.
* Gate-level route isolation is **functional, not anatomical**: the routes share `phon_embed.weight`
  (verified to be the only shared tensor). "WM-only" means "without the ventral premotor
  contribution", not "untouched by ventral training".
* This is an **inference intervention**. It says nothing about how a fixed-fusion model would learn
  from scratch.
* Training included `λ_gate·(mean(g) − 0.5)²` with `usage_prior = 0.5`, so the model was optimised
  under mild pressure **toward** the symmetry `fixed05` imposes. A null result for `fixed05` is
  therefore weaker evidence against adaptive routing than it appears. This caveat stands whatever the
  outcome.
* `SCIENTIFIC_LESION_READINESS=NOT_ESTABLISHED` (Phase 8) is unchanged by anything here.
* The `motor`-linearity that makes `fixed05` exact is **architecture-specific**. A semantic-attractor
  loop or a nonlinear readout would break it and require re-verification.

## WHAT WAS LEARNED

* The mechanism question is settled at the level of code: **there is no adaptive controller.** The
  gate is a fixed monotone transform of one scalar, ventral lexical confidence, confined to a narrow
  band around 0.5, blind to the route it is nominally arbitrating against.
* The interesting remaining question is therefore not "does the gate decide well" but **"is ventral
  confidence informative about ventral failure"** — which is H1′, is well powered, and is worth
  answering.
* The mature states are **not** dual-route in the behavioural sense on this population: the dorsal
  route carries everything, and the ventral route is ~11 % from ceiling. Terminal repair moves this
  essentially not at all (+17, +8, −6, +15 errors).
* The 14 Sep meeting's `0.5/0.5` baseline is a smaller intervention than it sounds, because the
  deployed gate already sits at ≈ 0.52 — a quantitative fact that was not available at the meeting.

## WHAT WAS NOT LEARNED

* Whether `g` varies **within** its narrow band in a way that tracks ventral failure — H1′ is
  implemented and preregistered but **not yet run**.
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
2. **Note for the architecture discussion, independent of those results:** findings 2–4 already
   constrain the design space. A gate that cannot observe the dorsal route and cannot leave
   [0.032, 0.646] is not performing confidence-based arbitration in any strong sense. If the project
   wants adaptive routing, that is a **new mechanism**, not a retuning of `α`/`τ` — and it should be
   decided on these structural grounds rather than on a marginal `fixed05` effect.
3. **Carry finding 5 forward** into any future lesion or repair interpretation: the historical
   `gate_mean` column does not mean what its name implies, and any downstream document quoting a
   repaired-state gate value should be corrected.
4. **Do not** train a fixed-gate model, implement `semantic_attractor`, modify `α`/`τ`, launch H512
   training, or start lesioning V2 on the basis of this package. Each needs a separate GO.

## ARTIFACTS

| File | Status |
|---|---|
| `CODE_AUDIT_GATE.md` | complete |
| `EXPERIMENT_CONTRACT.md` (+ AMENDMENT 1) | frozen |
| `checkpoint_manifest.tsv` | complete, 8/8 hashes verified |
| `item_level_gate_route_metrics.tsv` | pending full pass |
| `summary_metrics.json` | pending full pass |
| `figure_source_data/` | smoke shard only |
| `figures/` | pending full pass |
| `SHA256SUMS` | covers the frozen stage |
| `gating_diagnostics/`, `scripts/gating_diagnostics/`, `tests/` | complete, 15 tests passing |

Every plot in `figures/` will be reproducible from `figure_source_data/` alone.
