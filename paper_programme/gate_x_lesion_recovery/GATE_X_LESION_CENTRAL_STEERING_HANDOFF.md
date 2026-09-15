# GATE × LESION / RECOVERY — CENTRAL STEERING HANDOFF

**To:** CENTRAL
**From:** repository execution agent
**Stage reached:** MICRO-FREEZE + LIVE CODE REVERIFICATION → IMPLEMENTATION + QUARANTINED
VALIDATION. **Stopped before any canonical scientific lesion execution, as instructed.**

```
GO_FOR_SCIENTIFIC_EXECUTION       = NO
AWAITING_CENTRAL_FINAL_GO_NO_GO   = YES
```

---

## 1. CENTRAL decisions accepted without reopening

| decision | frozen as |
|---|---|
| two POST_REPAIR V6 prospective first-hit witnesses | `GXLR_W3_REP = V6_seed19_u3825`, `GXLR_W4_REP = V6_seed20_u3040` |
| route targets | dorsal `wm_encoder_state`, ventral `ltm_encoder_state` |
| severities | `λ ∈ {0.25, 0.50, 1.00}` |
| lesion seeds | `{0, 1, 2, 3}` |
| population | all 29 571 canonical real words |
| fusion comparison | NATIVE vs FIXED05 |
| decoding | canonical forced-length AR **and** genuine free-AR |
| timing | `per_item_frozen` |
| scientific execution | remains forbidden in this task |

None were revisited.

---

## 2. Micro-amendments applied

### MA-1 — SD normalisation: RESOLVED, not approximated

All six fields were recovered **from executable historical code**, not from notation:

| field | value |
|---|---|
| `SD_DEFINITION` | `float(a.std())` — sample SD of flattened pooled intact activations |
| `SD_AXES` | all axes pooled (batch with hidden); unitwise statistics not retained |
| `SD_DDOF` | 1 (PyTorch default, unbiased) |
| `SD_POPULATION` | `deterministic_sample(range(N), 2048, seed=7)`, sorted; batch 256; teacher-forced full forward |
| `SD_TENSOR_SHAPE` | **scalar** per (state, route) — isotropic, not per-unit |
| `ZERO_SD_BEHAVIOR` | no special-casing; amplitude 0 ⇒ exactly zero perturbation, no hook |

**The finding CENTRAL should register.** The historical calibrator computes and *prints*
RMS, but the atlas consumes the **`std`** column
(`run_premeeting_atlas.py:49`, `scales[target] = float(row["std"])`). A reader of the
calibrator alone would have inherited RMS and silently changed the operator. `std` is
authoritative and is now asserted by name.

**Second finding.** SD is a **single route scalar**, not a per-unit vector
(`lesion/spec.py:86`: `activation_scales: Tuple[Tuple[str, float], ...]`). Normalisation
is isotropic across hidden units. Preserved as found.

**Not inherited:** the historical scales were measured on a *different* checkpoint
(`chigh_15e5_h512_s22_u3000`). SD is re-measured per GXLR state on its own intact model,
by the identical recipe, and frozen before any lesion.

### MA-2 — nested severity noise: FROZEN

One base realisation `epsilon ~ U(−1,+1)` per item, RNG identity exactly
`(state_sha256, route, lesion_seed, item_id)`, with **λ and the fusion condition excluded
structurally** — the payload schema has four keys and the generator accepts no others.
`eta(λ) = epsilon ⊙ float32(λ·SD)`, no redraw.

The nesting `eta(0.50) = 2·eta(0.25)`, `eta(1.00) = 2·eta(0.50)` holds **bitwise**,
because the frozen severities are exact binary powers of two. Frozen tolerance: **0.0**.
SD semantics were not altered to obtain it.

### MA-3 — outcome classification: FROZEN

CANONICAL and FREE_AR classified independently, JOINT only afterward; all
diagnostically valid severities reported; the heterogeneity veto binds over A/B/D; no
retrospective promotion; convention disagreement forces a heterogeneous joint outcome.
Machine-readable in `gxlr_conditions.frozen.json`.

---

## 3. Findings CENTRAL must see before giving go/no-go

### F-1 — the two lineages are disjoint (structural, not cosmetic)

The `lesion/` package **does not exist** on the GATING lineage. `git diff --stat 3af0ef9
993f2e73 -- lesion/ scripts/lesion/` is 3 198 deletions, 0 insertions; the merge base is
`96468c69`. GXLR is therefore a **port**, not a reuse. Every ported semantic is recorded
with its source blob and line range in `HISTORICAL_LESION_OPERATOR_AUDIT.md` §6.

Consequence: the GATING fusion/decoding machinery is reused **verbatim** (it is on this
lineage), while the lesion operator is re-implemented minimally under audit. Connectivity
damage, mask scope/granularity and composite sites were deliberately **not** ported.

### F-2 — a silent-no-op hazard in the ventral route, now guarded

`models/ltm_route.py:128–147` reads **different GRU output slots in its two encoder
modes**:

* `unigru_last_hidden` → `_, h = self.encoder(packed)` → **slot 1** (`h_n`), `pooled = h[-1]`
* `bigru_masked_mean` → `out, _ = self.encoder(emb)` → **slot 0**, and `h_n` is *discarded*

The historical `_GRU_SLOT` hard-codes slot 1 for `ltm_encoder_state`. Under
`bigru_masked_mean` the ventral lesion would perturb a discarded tensor and produce a
**null ventral result that is purely an artefact of wiring** — indistinguishable, after
the fact, from a real scientific null.

Both GXLR checkpoints were read directly and declare
`ltm_encoder_mode = 'unigru_last_hidden'`, `enc_layers = 1`, so slot 1 is correct here.
The implementation nevertheless **asserts the mode at run time and refuses otherwise**,
and T9 independently proves ventral reachability. This is the single most consequential
live-code finding of the reverification.

### F-3 — the W3/W4 states are reconstruction recipes, not checkpoint files

Each POST_REPAIR witness is `base checkpoint + derived final-layer head`, applied by
`_isolated_model(tr, "p_last_hinge", head["state"])` with `head["state"]` keys required to
equal exactly `{2.weight, 2.bias}`. No monolithic checkpoint path was fabricated. Both
component hashes are verified before use.

Because CENTRAL's RNG identity names a `checkpoint_sha256`, a composite identity is
defined so the recipe is fully captured:

```
state_sha256 = sha256("gxlr-state-v1|" + base_sha256 + "|" + head_sha256)

GXLR_W3_REP  9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1
GXLR_W4_REP  e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6
```

**This is not a file hash and no file with this value exists.** If CENTRAL prefers a
different composition rule, it must be settled before execution, because it changes every
`epsilon`.

### F-4 — three defects in the historical draw, all removed by MA-2

| # | defect | effect if inherited |
|---|---|---|
| H-1 | `per_item` identity inferred from a float fingerprint of the encoder input | two distinct batches matching on shape/dtype/sum/first element silently share a draw |
| H-2 | one generator consumed sequentially for whole-batch draws | the realisation depends on batch size and batch order |
| H-3 | a different λ re-enters the same sequential stream at a different offset | severities cannot share a realisation — the nested design is impossible |

MA-2 removes all three by construction. Under GXLR, batching has **no scientific
footprint**.

### F-5 — NATIVE and FIXED05 provably cannot diverge at the intervention

`gate_probe.py:155–160` computes both from **one** `route="full"` forward and differs only
in which blend of the same `wm_logits`/`ltm_logits` is read. Since the perturbed site is
the **encoder**, whose inputs are `(enc_in, enc_mask)` only, the lesion tensor is
invariant to the decoder prefix. The two conditions may therefore follow different
autoregressive trajectories (CENTRAL §8.3 permits this) while provably consuming an
identical `eta`. Pinned by T6.

### F-6 — the intact null is exactly zero, on both states and both conventions

From the frozen GATING shards: NATIVE vs FIXED05 discordant items = **0** for W3_REP and
W4_REP under *both* canonical and free-AR. The gate is behaviourally silent in the intact
repaired regime. This is the authoritative null T2 reproduces, and it is what makes the
experiment well-posed: any non-zero discordance under lesion is attributable to the
perturbation, not to a pre-existing difference.

### F-7 — environment is bit-compatible; no new environment created

python 3.11.15, torch **2.12.1**, identical to the torch version in the frozen GATING
provenance. The existing interpreter was reused as directed; no conda environment was
cloned. Disk free on the data volume: **94 GiB**, far above the 3 GiB stop threshold — the
prior ENOSPC condition is not present. No scientific artifact was deleted.

---

## 4. Open items for CENTRAL — none blocking design, all blocking execution

| # | item | default taken | needs CENTRAL? |
|---|---|---|---|
| O-1 | composite `state_sha256` rule (F-3) | `sha256("gxlr-state-v1\|base\|head")` | **confirm** — changes every `epsilon` |
| O-2 | `item_id` = canonical word string (unique, order-stable) vs entry index | word string | confirm |
| O-3 | diagnostic-validity predicate: not-inert (≥1 changed item) + not-saturated (NATIVE exact ≥ 1 %, modal prediction share ≤ 50 %), ≥3/4 seeds | as stated in `gxlr_conditions.frozen.json` | **confirm before execution** |
| O-4 | robustness rule: ≥3/4 seeds significant at α=0.05 with the same sign **and** pooled exact McNemar significant with that sign | as stated | **confirm before execution** |
| O-5 | SD re-measured per state (historical scales were from another checkpoint) | re-measure by the identical recipe | confirm |
| O-6 | free-AR cap = 12, imported not restated | inherited | no |

O-3 and O-4 are the only choices that could, if changed after seeing data, alter a
reported outcome. They are frozen in a machine-readable file **before** execution
precisely so that they cannot be.

---

## 5. What was built, and what was deliberately not

**Built:** deterministic base-noise generator and nested scaler; per-state intact SD
measurement following the audited recipe; a minimal activation-lesion context manager;
the outcome classifier; the quarantined smoke path; twelve tests (T1–T12).

**Deliberately not built:** connectivity damage; mask machinery; composite sites;
severity levels outside the frozen grid; any recovery-training or attractor code; any
scientific output.

**Not run:** any non-zero-severity lesion over the canonical population. The only
non-zero-severity code execution anywhere in this task is on a deliberately tiny,
non-canonical, quarantined subset written under a path containing
`NOT_SCIENTIFIC_RESULT`, carrying in-file metadata stating it cannot be used as a
scientific result.

---

## 6. Requested decision

1. Confirm or amend **O-1, O-3, O-4** (and, if desired, O-2, O-5).
2. Register findings **F-2** (ventral slot hazard) and **F-3** (recipe, not checkpoint).
3. Issue go / no-go for scientific execution.

The exact execution command is prepared and printed as `EXECUTION_COMMAND`. **It has not
been run.** Nothing in this handoff depends on running it.
