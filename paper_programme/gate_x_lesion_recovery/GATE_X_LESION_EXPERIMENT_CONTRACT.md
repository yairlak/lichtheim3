# GATE × LESION / RECOVERY — EXPERIMENT CONTRACT

**Workstream:** `POST_STAGE / PAPER_PROGRAMME — GATE × LESION / RECOVERY` (GXLR)
**Status:** PREREGISTERED. DESIGN FROZEN. **NOT EXECUTED.**
**Authority:** CENTRAL, accepted with micro-amendments 1–3 (§5, §6, §7 below).
**Predecessor:** `paper_programme/gating_route_diagnostics/EXPERIMENT_CONTRACT.md`,
frozen at `993f2e73ecd0b9de2c3318255afbaa159948b09a`.

This document is preregistration. Every free parameter below is fixed **before** any
non-zero severity is executed. No parameter may be revised in the light of a result.

---

## 1. Question

In two POST_REPAIR witness states that have already been shown to be *behaviourally
indistinguishable* under NATIVE and FIXED05 fusion when intact, does a graded,
route-specific lesion **dissociate** the two fusion regimes — and if so, in which
direction, at which severities, and consistently across decoding conventions?

The intact record is the null this experiment perturbs: at severity 0 both witnesses show
**zero** NATIVE/FIXED05 discordant items under both conventions (`LIVE_CODE_AUDIT.md` §6).
The gate is therefore behaviourally silent in the intact repaired regime. The question is
whether it is *functionally* silent, or merely unstressed.

**This contract does not authorise execution.** It fixes what execution would do.

---

## 2. What is explicitly out of scope

Frozen as out of scope by CENTRAL; listed so that no later reading can widen it:

* no training, fine-tuning, or recovery training of any kind;
* no semantic attractor;
* no Lesioning V2;
* no connectivity damage (activation noise only);
* no change to gate architecture, `alpha`, `gate_threshold`, or the `fixed05` definition;
* no severity outside `{0.25, 0.50, 1.00}`;
* no checkpoint substitution;
* no contact with V7 / Phase 9 scientific state;
* no mutation of the frozen GATING results commit `993f2e73…`.

---

## 3. States — exactly two

Both are POST_REPAIR V6 prospective first-hit witnesses. Identities, hashes and the
reconstruction recipe are in `checkpoint_manifest.proposed.tsv` and
`LIVE_CODE_AUDIT.md` §6.

| label | manifest `state_id` | witness | seed | source_u |
|---|---|---|---|---|
| `GXLR_W3_REP` | `W3_REP` | `V6_seed19_u3825` | 19 | 3825 |
| `GXLR_W4_REP` | `W4_REP` | `V6_seed20_u3040` | 20 | 3040 |

**Each state is a reconstruction recipe, not a monolithic checkpoint file:** a base
checkpoint plus a derived final-layer head applied via
`frozen_head_probe._isolated_model(tr, "p_last_hinge", head["state"])`, where
`head["state"]` must carry exactly `{2.weight, 2.bias}`. This is the semantics under
which the frozen GATING witness metrics were produced, and it is preserved verbatim.
Both the base SHA256 and the head SHA256 are verified before use; the run hard-stops on
any mismatch.

**Composite state identity.** For RNG identity and provenance, a state is identified by

```
state_sha256 = sha256("gxlr-state-v1|" + base_artifact_sha256 + "|" + applied_head_sha256)
```

| label | `state_sha256` |
|---|---|
| `GXLR_W3_REP` | `9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1` |
| `GXLR_W4_REP` | `e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6` |

This is CENTRAL's `checkpoint_sha256` slot, defined so that the reconstruction recipe is
fully captured. It is **not** a file hash and no file with this hash exists.

W1/W2 and all `*_SRC` rows are carried in the proposed manifest for traceability with
`IN_SCOPE = 0`. They are never executed by this workstream.

---

## 4. Design grid

| factor | levels | n |
|---|---|---|
| state | `GXLR_W3_REP`, `GXLR_W4_REP` | 2 |
| route | `wm_encoder_state` (dorsal), `ltm_encoder_state` (ventral) | 2 |
| severity `λ` | `0.25`, `0.50`, `1.00` | 3 |
| lesion seed | `0`, `1`, `2`, `3` | 4 |
| fusion | `NATIVE`, `FIXED05` | 2 |
| decoding | `CANONICAL` forced-length AR, `FREE_AR` | 2 |

Plus the **severity-0 intact control**, one cell per state (no seed dependence: `λ=0`
yields an exactly zero perturbation, so all seeds coincide).

Population: **all 29 571 canonical real words**, full and untruncated, in the entry order
whose `item_order_sha256` begins `ab2b193f98eee7fb` — identical to the frozen GATING
population. Truncation is refused outside quarantine.

Severity naming: `LOW = 0.25`, `MODERATE = 0.50`, `HIGH = 1.00`. These are labels for the
same base realisation at three magnitudes (§6); they are **not** independent conditions.

---

## 5. MICRO-AMENDMENT 1 — SD normalisation (RESOLVED)

Recovered from executable historical code at `3af0ef9`; full evidence in
`HISTORICAL_LESION_OPERATOR_AUDIT.md` §2. All six fields are frozen:

| field | frozen value |
|---|---|
| `SD_DEFINITION` | `float(a.std())` — PyTorch sample SD of the flattened, pooled intact activations at the site. **Not RMS.** (`run_premeeting_atlas.py:49` selects the `std` column written by `measure_activation_scales.py:95`.) |
| `SD_AXES` | **All axes pooled.** Batch is pooled with hidden; no `dim=` anywhere. Unitwise statistics are not retained. (`measure_activation_scales.py:66` `.flatten()`, `:86` `torch.cat`.) |
| `SD_DDOF` | **1** — PyTorch default unbiased/sample SD (`correction=1`), verified empirically under torch 2.12.1. |
| `SD_POPULATION` | `deterministic_sample(range(len(entries)), n=2048, seed=7)` — first 2 048 of a seeded permutation, **returned sorted**, i.e. ascending entry index; evaluated `batch_size=256`, one teacher-forced `model(enc_in, enc_mask, dec_in)` per batch. |
| `SD_TENSOR_SHAPE` | **Scalar.** One float per `(state, route)`. Isotropic across hidden units. Not a per-unit vector. (`spec.py:86`, `:177–187`.) |
| `ZERO_SD_BEHAVIOR` | **No special-casing.** A zero scale yields amplitude 0, hence an exactly zero perturbation and no registered hook (`apply.py:169–176`); `draw_noise` independently short-circuits at `amplitude == 0.0` (`masks.py:182–183`). No floor, no clamp, no exception. |

**Device / dtype.** SD is computed on **CPU in float32** (the collection hook casts
`.detach().float()` unconditionally), materialised to a Python float, and stored as a
decimal string. The noise tensor is drawn in the site tensor's own dtype (float32 here).

**Per-state measurement.** The historical atlas measured SD on a *different* checkpoint
(`chigh_15e5_h512_s22_u3000`). That value is **not** inherited. SD is re-measured on each
GXLR state's own **intact** model, by the recipe above, **before any lesion**, written to
`intact_sd.tsv`, and frozen. It is never re-fitted in the light of a lesion outcome.

**Answers to the explicit questions.**

* SD is a **single site/route scalar**, not a per-unit vector or tensor.
* The collected tensor is the GRU **`h_n`** at slot 1 for both routes: `(1, B, 128)` for
  `wm_encoder_state`, `(1, B, 512)` for `ltm_encoder_state`.
* SD is computed across **all dimensions jointly**, after flattening and concatenating
  every batch.
* **Sample (unbiased, ddof=1) SD**, PyTorch default.
* The historical intact population is the 2 048-item sorted deterministic sample at
  seed 7, under a teacher-forced full forward.
* The batch dimension **is** pooled with the hidden dimension; unitwise statistics are
  **not** retained.
* A zero-SD site receives an exactly zero perturbation and no hook; there is no guard.
* SD is computed and stored on **CPU, float32**.

---

## 6. MICRO-AMENDMENT 2 — nested severity noise (FROZEN)

### 6.1 One base realisation

A single deterministic base noise realisation per item:

```
epsilon ~ Uniform(-1, +1)
```

with RNG identity **exactly**

```
(state_sha256, route, lesion_seed, item_id)
```

**`lambda` and the fusion condition are excluded from the RNG identity.** This is
structural, not conventional: the payload is built from a fixed four-key schema and the
generator function accepts no other arguments.

* `state_sha256` — the composite identity of §3.
* `route` — `"wm_encoder_state"` or `"ltm_encoder_state"`.
* `lesion_seed` — `0 | 1 | 2 | 3`.
* `item_id` — the canonical **word string**. Unique across the 29 571-item population
  (verified: zero duplicates) and stable under re-ordering. `item_order_sha256` is
  recorded separately.

Derivation: canonical JSON (`sort_keys`, no whitespace, ASCII) → SHA-256 → SHA-256
counter-mode byte stream → big-endian `uint32` → `x = u/2^32 ∈ [0,1)` →
`epsilon = 2x − 1`. Process-safe, platform-safe, torch-version-independent.
**Python's salted `hash()` is never used.**

`epsilon` has the shape of the site tensor's per-item slice: `(H,)` with `H = 128`
(dorsal) or `H = 512` (ventral), assembled into `(1, B, H)` for a batch. Because
`epsilon` is a pure function of the item, **batch size and batch order have no
scientific footprint.**

### 6.2 Prospective scaling

```
eta(lambda) = lambda * SD * epsilon        for lambda in {0.25, 0.50, 1.00}
```

implemented as `eta = epsilon ⊙ float32(lambda · SD)`, preserving the historical operand
order (`draw_noise`: noise tensor × amplitude).

* **LOW / MODERATE / HIGH use the same direction and differ only in magnitude.**
  `eta(λ)` for the three severities are exact positive scalar multiples of one another.
* `epsilon` is **never redrawn** across severities.
* **NATIVE and FIXED05 consume the same scaled tensor.** Guaranteed structurally: both
  are read from a single `route="full"` forward (`gate_probe.py:155–160`), and `epsilon`
  does not depend on the fusion condition.
* The tensor is **held fixed across autoregressive decoder steps** for that item
  (`per_item_frozen`). This is guaranteed by construction — `epsilon` is a pure function
  of the item — not by a cache heuristic. Caching is a performance detail only.

**Numerical exactness.** `{0.25, 0.50, 1.00}` are exact binary powers of two, so the
nesting `eta(0.50) == 2·eta(0.25)`, `eta(1.00) == 2·eta(0.50)` holds **bitwise,
elementwise**. The frozen deterministic tolerance is **0.0**. SD semantics were not
altered to obtain this (`LIVE_CODE_AUDIT.md` §5).

### 6.3 Application

One forward hook on the target module replaces exactly GRU slot 1 (`h_n`), out of place:
`h ← h + eta`. Handles are removed in `finally`. No parameter is written; the
`state_dict` hash is verified identical before and after every lesion context.

`ltm_encoder_state` is refused unless the state declares
`ltm_encoder_mode == "unigru_last_hidden"`, because under `bigru_masked_mean` slot 1 is
discarded and the lesion would be a silent no-op (`LIVE_CODE_AUDIT.md` §2.3).

---

## 7. MICRO-AMENDMENT 3 — outcome classification (FROZEN)

Classification is performed **three times, in this order**, and never collapsed early.

### 7.1 Stage 1 — CANONICAL forced-length AR, alone

### 7.2 Stage 2 — FREE_AR, alone

Stages 1 and 2 are classified **independently**. Neither may be used to adjudicate,
override, or filter the other.

### 7.3 Stage 3 — JOINT_OUTCOME, only afterward

### 7.4 Diagnostic validity

A severity cell is **diagnostically valid** when the lesion is neither inert nor
saturating for that route family — i.e. it produced a measurable change relative to
intact, and did not collapse the route to degenerate output. The precise validity
predicate is fixed in the machine-readable condition manifest
(`gxlr_conditions.frozen.json`) before execution.

**All diagnostically valid severities must be reported.** None may be suppressed.

### 7.5 Outcome letters

Per route family × convention, over the diagnostically valid severities:

| outcome | meaning |
|---|---|
| `A` | robust NATIVE advantage (NATIVE better than FIXED05) |
| `B` | robust FIXED05 advantage |
| `D` | robust null — no reliable NATIVE/FIXED05 difference |
| `F_HETEROGENEOUS` | valid severities disagree in sign |

### 7.6 The heterogeneity veto — binding

> **`A`, `B` or `D` may NOT be declared for a route family / convention if another
> diagnostically valid severity in that same route family shows a robust
> opposite-signed NATIVE-vs-FIXED05 effect. Such a case is
> `OUTCOME_F_HETEROGENEOUS`.**

The veto is evaluated across severities *within* a route family and convention. It is
checked **before** any letter is emitted, and it strictly dominates: a heterogeneous
family yields `F_HETEROGENEOUS`, never the letter of its "strongest" severity.

### 7.7 No retrospective promotion

**No severity may be promoted retrospectively as "the lesion result".** The reported unit
is the *set* of diagnostically valid severities and their signs. Phrases of the form "at
the appropriate severity…" are contract violations. A severity's diagnostic validity is
determined by the frozen predicate, never by whether its result is congenial.

### 7.8 Convention disagreement

**A CANONICAL / FREE_AR disagreement remains visible and forces a heterogeneous joint
classification.** If Stage 1 and Stage 2 yield different letters for the same route
family, `JOINT_OUTCOME = F_HETEROGENEOUS`, and both stage-level letters are reported
alongside it. The joint outcome never silently adopts one convention.

### 7.9 Severity selection is not an outcome-dependent choice

No severity may be added, dropped, or reweighted on the basis of NATIVE-vs-FIXED05
behaviour. The grid is `{0.25, 0.50, 1.00}`, fixed by CENTRAL, full stop.

---

## 8. Measurements — preregistered outputs only

Per item × state × route × λ × seed × fusion × convention:

* canonical and free-AR predicted phoneme sequences and exact-match;
* NATIVE / FIXED05 accuracy and **exact discordance counts** (paired, item by item);
* **errors prevented** (FIXED05 wrong, NATIVE right) and **errors introduced**
  (NATIVE wrong, FIXED05 right);
* isolated WM competence and isolated LTM competence (route-only decodes);
* `c_LTM` (lexical confidence), `g` (gate value), and their deltas from intact;
* compact discordant token traces (discordant items only, truncated);
* checkpoint and contract provenance: both component SHA256s, the composite
  `state_sha256`, `item_order_sha256`, the frozen SD, contract hash, code commit,
  torch version, wall time.

**No new exploratory metric that would require changing the causal intervention may be
added.** Descriptive columns computed from the same forward are permitted; anything
requiring a different perturbation is a new experiment.

---

## 9. Execution safeguards

* `--limit` truncates the population and is **refused without `--smoke`**.
* Smoke outputs are written only under a path containing `NOT_SCIENTIFIC_RESULT`, carry a
  `SMOKE_TEST_ONLY` filename marker and in-file metadata stating they are not results,
  and **cannot share an output namespace with scientific execution**.
* Scientific output namespace `scientific_execution/` is **reserved and empty** until
  CENTRAL gives final go.
* Source checkpoint and head files are opened read-only; their SHA256 is re-verified
  after every state.
* The in-memory `state_dict` hash is compared before and after every lesion context.
* Determinism: every decode is re-run on a control batch and must be bitwise identical.

---

## 10. Provenance

| artifact | role |
|---|---|
| `HISTORICAL_LESION_OPERATOR_AUDIT.md` | SD semantics recovered from `3af0ef9` |
| `LIVE_CODE_AUDIT.md` | line-by-line reverification of the code to be run |
| `checkpoint_manifest.proposed.tsv` | exact state identities and recipes |
| `GATE_X_LESION_CENTRAL_STEERING_HANDOFF.md` | decision record and open items for CENTRAL |
| `gxlr_conditions.frozen.json` | machine-readable frozen condition grid |

Frozen inheritance: gating commit `f43ccd0971387ccfce006ddb0d6f985f1ee73441`
(experiment), `993f2e73ecd0b9de2c3318255afbaa159948b09a` (results, and the parent of this
branch). Historical lesion lineage: `3af0ef9c00df5aba8200d56f566bb957d084109c`.

---

**GO_FOR_SCIENTIFIC_EXECUTION = NO. This contract is preregistration only.**
