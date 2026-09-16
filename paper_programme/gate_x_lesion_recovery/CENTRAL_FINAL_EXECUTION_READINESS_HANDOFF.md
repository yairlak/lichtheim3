# CENTRAL FINAL EXECUTION READINESS HANDOFF

> **SUPERSEDED IN PART — O-3 and O-4 are now DECIDED.**
> CENTRAL has since issued the authoritative final rule decision. Sections B and C
> below recorded O-3 and O-4 as *requiring* a CENTRAL decision; that decision has
> been made and is frozen in **`FINAL_RULE_FREEZE.md`** and
> **`gxlr_conditions.final.json`**, implemented in `gate_x_lesion/validity.py`,
> `gate_x_lesion/robustness.py` and `gate_x_lesion/classify.py`.
>
> * `O3_VALIDITY_CONVENTION_SCOPE = SHARED`,
>   `O3_IMPLEMENTATION_FAILURE_SEMANTICS = ABORT_RUN`
> * `O4_ROBUSTNESS_RULE = REPLICATED_SIGN_PLUS_MATERIALITY_NO_SIGNIFICANCE_TEST`
>   — none of the candidate rules R1/R2/R3 offered below was adopted verbatim.
> * `FINAL_CONTRACT_HASH = 1c051f1bc8d5265d9768462baca22bcd842f93cbb20a0836c4781d6ae746d701`
>
> This document is retained unmodified below as the record of what was open at the
> time, and of the evidence on which the decision was taken.

**Workstream:** `POST_STAGE / PAPER_PROGRAMME — GATE × LESION / RECOVERY`
**Pass:** final targeted closure pass (no new science; closes pre-execution ambiguities only)
**Branch / worktree:** `paper-programme/gate-x-lesion-recovery` · `wt-gate-x-lesion/`

---

## A. RESOLVED SINCE PREVIOUS HANDOFF

| item | status | evidence | commit / test |
|---|---|---|---|
| **O-1** state identity for RNG | **RESOLVED** | Identity is `RECONSTRUCTED_STATE_SHA256 = sha256("gxlr-state-v1\|" + base_sha256 + "\|" + head_sha256)`. A pure function of **two file digests**; the function takes no other parameter, so `lambda`, fusion condition, decoding convention, AR prefix, `epsilon`, the lesion tensor, batch position and process identity are excluded *structurally*. Computed immediately after reconstruction, **before** `measure_intact_sd` and before any lesion, and hoisted out of the route/λ/seed loops. Value unchanged from `IMPLEMENTATION_COMMIT`. | `gate_x_lesion/identity.py`; `test_O1_*` (6 tests) |
| **FREE-AR** independent trajectories | **RESOLVED — no implementation change needed** | `ar_decode_free` loops over routes outermost and initialises a fresh BOS-only `dec` per route, so each condition extends its own prefix from its own predictions. The hook sits on the **encoder**, whose inputs `(enc_in, enc_mask)` are invariant to the decoder prefix, so divergence cannot redraw the tensor. Forced-divergence test proves A–E simultaneously. | `test_FREEAR_forced_divergence_independent_trajectories` |
| **F-2** ventral slot hazard | **REGISTERED + GUARDED** | `models/ltm_route.py:128-147` reads **different GRU slots by mode**: `unigru_last_hidden` → slot 1 (`h_n`); `bigru_masked_mean` → slot 0, `h_n` **discarded**. Both witnesses audited live as `unigru_last_hidden`, `enc_layers=1`, unidirectional. Guard refuses any other mode, non-GRU module, multi-layer or bidirectional encoder. | `gate_x_lesion/targets.py`; `test_T9b_*`, `test_F2_*` |
| **F-3** reconstructed witnesses | **REGISTERED** | W3/W4 are `base checkpoint + repaired head` via `_isolated_model(tr,"p_last_hinge",head["state"])`, head keys exactly `{2.weight, 2.bias}`. Both component paths + SHA256, the localizer, `state_kind` and `reconstructed_state_sha256` are recorded in the manifest **and** in the runner's per-state summary. | `test_F3_*`, `test_T1*` |
| **F-8** batch size | **REGISTERED + PINNED at 256** | Measured on W3_REP vs the frozen GATING shard: batch **256 → 0/2048** items differ, gate **bitwise identical**; batch 512 → 15/2048 differ, max \|Δg\| = 1.192e-07. No prediction changed at either size. `g`/`c_LTM` are reported continuous measurements, so the batch size is pinned. Scientific mode hard-stops on any other value. | `run_gate_x_lesion.py::assert_scientific_batch_size`; `test_F8_*` |

### Correction to previous wording (O-1)

The previous handoff listed O-1 as *"composite `state_sha256` rule — confirm; **changes every epsilon**"*. That meant: **if CENTRAL changed the composition rule, every `epsilon` would change**, i.e. the rule is consequential and should be confirmed before execution. It did **not** mean the identity varies with `epsilon`.

**It cannot.** `reconstructed_state_sha256` accepts exactly two arguments, both file digests. `epsilon` is derived *from* the identity; the dependency is one-way and is asserted operationally (`test_O1_identity_cannot_be_affected_by_epsilon_or_lesion` runs a real λ=1.0 lesion and shows the identity and the intact parameters unchanged afterwards). The imprecise wording is corrected here and in `gate_x_lesion/identity.py`.

**No hard blocker was found.** The implementation was already correct; only the documentation needed fixing.

### FREE-AR verification detail

Divergence is forced deterministically by stubbing **only the blend output inside the test**, so FIXED05 must select a different token. The real `ar_decode_free` loop, the real encoder and the real lesion hook run unmodified; FIXED05's shipped algebra is untouched. Verified in one test:

* **A** prefixes agree at BOS, differ from the divergence step onward, and stay diverged;
* **B** each branch's `dec[k+1] == cat(dec[k], argmax(its own logits[k]))` — every step, both branches;
* **C** every captured hook firing on both branches holds a tensor bitwise equal to `epsilon(item) × (λ·SD)`;
* **D** `hook.n_eta_builds == 1` across the whole decode — no redraw on divergence or re-encoding;
* **E** genuine free-AR (`ar_decode_free`, imported cap 12), not forced-length teacher forcing.

---

## B. O-3 DIAGNOSTIC VALIDITY

```
O3_STATUS=REQUIRES_CENTRAL_DECISION
```

The accepted rule is encoded and machine-decidable in `gate_x_lesion/validity.py`. **Two sub-fields remain genuinely undetermined and were not invented.**

### Resolved from prior authority (not invented)

**Deterministic evaluator tolerance = `0.0`, bitwise.** Cited:

* gating `EXPERIMENT_CONTRACT.md` §5 — *"A determinism check — re-run one batch, require bitwise equality — runs before the main pass and hard-stops on failure."*
* `scripts/gating_diagnostics/run_gate_route_audit.py:168-180` (`assert_determinism`) compares `gate` and the exact-match columns with `!=` and hard-stops on any difference.
* frozen `summary_metrics.json` — `determinism_max_gate_dev == 0.0` in **all eight** states.

This tolerance governs criterion 2 (route isolation), and criterion 4's `c_LTM` and `g`. It is empirically reachable on this architecture: T8 measures dorsal `max|Δg| = 0.000e+00` and `max|Δc_LTM| = 0.000e+00` at every frozen severity.

**Numerical correction found while encoding criterion 1.** Evaluating "drops by ≥ 0.10" as the difference of two separately-rounded accuracies puts representation error exactly on the decision boundary — `0.90 − 0.10 == 0.09999999999999998`, which fails `>= 0.10` spuriously. The drop is therefore computed from **integer counts via one division**. (On the canonical population an exact tie cannot arise anyway: `0.10 × 29571 = 2957.1` is not an integer.)

**Relation to GATING §9.** Criterion 1 is a replication/diagnostic **count**, not a significance test, so the 8-block aggregation does **not** conflict with the prohibition on cross-witness significance testing. That prohibition binds O-4, not O-3.

**Validity never reads the fusion contrast.** `BlockObservation` carries only isolated-route and gate quantities; no NATIVE/FIXED05 field is representable in it (`test_O3_validity_never_reads_the_fusion_contrast`).

### The minimal unresolved choices

| field | question | allowed values | default taken |
|---|---|---|---|
| `validity_convention_scope` | Is route × severity validity computed **once** from shared route diagnostics, or **separately per decoding convention**? The accepted rule states the grid as 2 checkpoints × 4 seeds and does not mention the convention, yet isolated-route exact-match exists under **both** canonical and free-AR. | `shared` \| `per_convention` | **none** |
| `implementation_failure_block_semantics` | When criterion 3 or 4 is violated (an implementation failure, not a result), what happens to the 8-block denominator? | `abort_run` \| `exclude_block_and_shrink_denominator` \| `exclude_block_keep_denominator` | **none** |

**Fails closed:** `decide_route_severity_validity` requires both explicitly and raises `UnresolvedPolicyError` on anything else. There is no default path.

---

## C. O-4 ROBUSTNESS

```
O4_STATUS=REQUIRES_CENTRAL_DECISION
```

**No authoritative operational definition of "robust" exists.** The predecessor GATING workstream supplies the *primitives* — exact paired McNemar on discordant pairs, Holm correction across the planned family, a materiality floor of `|Δ accuracy| = 0.002` (outcome B), a power floor of `n ≥ 30` per cell, bootstrap 10 000 at `seed = 20260915` — but it has **no severity axis**, so it contains no rule for combining evidence across severities.

> **The binding constraint CENTRAL must weigh.** Gating `EXPERIMENT_CONTRACT.md` §9: *"No cross-witness significance test. Four witnesses from three seeds of one historical cohort are not an independent population."* **Any candidate that pools W3 and W4 into a single significance test would violate frozen authority.** All three candidates below therefore classify per witness or avoid significance testing entirely.

### Candidate rules — NONE IS FROZEN

| | **R1** per-witness significance | **R2** R1 + inherited materiality | **R3** unanimous sign, no test |
|---|---|---|---|
| **Exact rule** | Within **each** witness, ≥ 3 of its 4 seeds significant at α = 0.05 by exact McNemar with the same sign of `net_change_in_correct`; both witnesses must agree on that sign. | As R1, but a block counts only if `\|Δ accuracy\| ≥ 0.002`. | Robust iff all 8 blocks have non-zero `net_change_in_correct` and share one sign. No p-value anywhere. |
| **Conservative?** | Moderately — requires both witnesses to agree, never pools them. | **Most conservative** of the three. | Moderate; 8/8 unanimity is demanding, but magnitude is unconstrained. |
| **Uses only the pre-frozen 8 blocks?** | Yes | Yes | Yes |
| **Introduces an arbitrary effect-size threshold?** | No | **No — 0.002 is inherited**, already frozen in GATING outcome B (`EXPERIMENT_CONTRACT.md:279,488`) | No |
| **Could certify tiny numerical deltas?** | **Yes — its main weakness.** On 29 571 paired items an exact McNemar is significant at discordance splits that are scientifically trivial. | No — excluded by construction. | **Yes** — a net of ±1 item in each of the 8 blocks would qualify. It trades R1's sensitivity for unanimity, not for magnitude. |
| **Conflicts with §9?** | No | No | No (contains no significance test at all) |

### The exact decision CENTRAL must make

> Choose the operational definition of **"robust"** for a route family × severity × decoding convention, given that (a) GATING §9 forbids pooling the two witnesses into one significance test, and (b) on 29 571 paired items statistical significance alone can certify scientifically trivial effects.
>
> Concretely: adopt **R1**, **R2**, **R3**, or supply another rule — and **if a materiality floor is wanted, confirm whether the inherited `0.002` is the right value for a LESIONED regime**, given it was set for the intact regime.

**Recommendation on what to answer, not on the answer:** the decision hinges on one question — *should statistical significance alone be sufficient, or must an effect also clear a magnitude floor?* R1 says significance suffices; R2 says it does not; R3 removes significance testing entirely but adds no magnitude floor. CENTRAL should answer that question first; the rule follows from it.

**Fails closed:** `FROZEN_ROBUSTNESS_RULE = None`; `evaluate_severity` requires an explicit `robustness_rule` with no default and raises `UnfrozenRobustnessError` otherwise.

**Unchanged by this pass:** the heterogeneity veto, independent CANONICAL/FREE_AR classification with JOINT only afterward, visible decoder-convention disagreement, no retrospective promotion, and the requirement to report all diagnostically valid severities. `test_O4_veto_still_binds_under_any_candidate_rule` shows the veto binds whichever candidate is used.

---

## D. SCIENTIFIC EXECUTION SAFETY

| check | status |
|---|---|
| no non-zero canonical science run occurred | **CONFIRMED** — every non-zero lesion artifact is a 24-row quarantined smoke shard under `NOT_SCIENTIFIC_RESULT/`; the largest non-zero-severity population touched anywhere is 24 items |
| execution command remains NOT RUN | **CONFIRMED** — `EXECUTION_COMMAND.prepared.sh`, mode `-rw-r--r--` (never executable) |
| runner still requires explicit CENTRAL-go flag | **CONFIRMED** — non-zero severity on the full population hard-stops without `--i-have-central-go` (verified from CLI) |
| batch size 256 pinned | **CONFIRMED** — scientific mode hard-stops on any other value |
| exact W3/W4 provenance pinned | **CONFIRMED** — both component paths + SHA256, localizer, and `reconstructed_state_sha256` in manifest and runner summary; hashes re-verified against the frozen GATING manifest |
| scientific output namespace absent/unpopulated | **CONFIRMED** — `paper_programme/gate_x_lesion_recovery/scientific_execution/` does not exist |
| DESIGN-frozen artifacts unmodified | **CONFIRMED** — `shasum -c SHA256SUMS.design` → all six **OK** |

Additional layered refusals verified from the CLI this pass: `--limit` without `--smoke`; severity outside `{0.25, 0.50, 1.00}`; smoke writing outside quarantine. All refuse **before any model is loaded**.

---

## E. COMMITS

| role | hash |
|---|---|
| `DESIGN_FREEZE_COMMIT` (original, untouched) | `545ea436fa8f33480d74185a48a61874c04a17eb` |
| `IMPLEMENTATION_COMMIT` (original, untouched) | `975560e6e2f1737f8915f56cf330223fd023be7c` |
| `DESIGN_AMENDMENT_COMMIT` | `2a70f76b13a4d7909c56e8b15a0bf2f1942851ce` |
| `IMPLEMENTATION_AMENDMENT_COMMIT` | `4e65b14d6cfedb7eb73a9ff10d4a0869457c7e53` |

Neither original commit was rewritten, squashed or amended; both remain ancestors of HEAD.

---

## F. TESTS

`python3 -m pytest tests/test_gate_x_lesion.py tests/test_gate_x_lesion_closure.py`
→ **51 passed in 474.98s**. Summary:

* **T1–T12** (original, 24 tests) — all still pass, including the **severity-0 authoritative null** on the full 29 571-item population, both states, both decoding conventions, **0 NATIVE-vs-FIXED05 discordant items**, bitwise-identical to the frozen GATING record.
* **Closure tests** (27 tests) — O-1 (6), FREE-AR forced divergence (1), F-2 (1), F-3 (2), F-8 (2), O-3 boundary (8), O-4 candidates and veto (7).
* **T12f reframed** — it previously asserted the robustness default that has now been removed; it asserts the fail-closed behaviour instead. This is the only original test whose body changed.

No non-zero full-population lesion condition was rerun.

---

## G. FINAL STATUS

```
EXECUTION_READINESS=BLOCKED_PENDING_CENTRAL_DECISION   [status at the time of writing]
```

**Superseded.** Both decisions have since been made by CENTRAL and frozen; see
`FINAL_RULE_FREEZE.md`. The current status is:

```
FINAL_DESIGN_DECISIONS_COMPLETE                = YES
FINAL_RULE_IMPLEMENTATION_COMPLETE             = YES
GO_FOR_SCIENTIFIC_EXECUTION                    = NO
AWAITING_CENTRAL_FINAL_SCIENTIFIC_EXECUTION_GO = YES
```
