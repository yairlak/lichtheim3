# CODE_AUDIT_GATE — what the Lichtheim3 gate actually is

**Workstream:** LICHTHEIM3 — GATING / ROUTE DIAGNOSTICS (POST_STAGE / PAPER_PROGRAMME)
**Branch:** `paper-programme/gating-route-diagnostics`
**Base commit:** `78f5505` (`prospective/rn-detector-v6`) — the V6-completion tip, i.e. the exact
code lineage that produced the Phase-8 repaired witnesses (P8-IMP-009).
**Date:** 2026-09-15
**Status:** READ-ONLY AUDIT. No model code was modified. No checkpoint was written.
**Authority:** Phases 1–8 as frozen in
`LICHTHEIM3_SOURCE_OF_TRUTH_CANONICAL_FABLE_ARBITRATED_2026-09-14`. Nothing below revises a
HARD_FROZEN claim.

---

## 0. Executive summary

Six findings, each verified numerically (§6), not inferred from documentation:

| # | Finding | Consequence |
|---|---|---|
| **A1** | The gate has **no learnable parameters**. `g` is a fixed sigmoid of one scalar. | "Learned gate" is a misnomer. The only learning that reaches `g` is inside the ventral encoder that produces `s_hat`. |
| **A2** | The gate reads **only** `c_LTM = max_i cos(ŝ, bank_i)`. It is **exactly invariant** to every dorsal parameter. | The gate is **structurally incapable** of tracking *relative* route competence. It can only track *absolute ventral lexical confidence*. |
| **A3** | With the cohort's `α=2.0, τ=0.7`, the attainable range is **asymmetric**: ventral weight `g ∈ [0.0323, 0.6457]`, dorsal weight `1−g ∈ [0.3543, 0.9677]`. | **Strong dorsal commitment is possible (up to ≈ 29.9 : 1 dorsal-to-ventral); strong ventral commitment is structurally impossible (at most ≈ 1.82 : 1 ventral-to-dorsal).** The asymmetry, not an absence of commitment, is the finding. |
| **A4** | The historical **position-weighted signed mean** gate on all four Phase-8 witnesses is **0.517–0.521**. | Setting `g ≡ 0.5` therefore changes that **signed mean** by about **0.02**. The **item-level** and **logit-level** magnitude of the intervention remain **unmeasured** (only the mean was ever recorded), so no claim about how large the intervention is may be made from this row. |
| **A5** | `motor` is a single affine map and the blend weights sum to 1, so **fusion on premotor ≡ fusion on logits, exactly**. | The forced-0.5 intervention needs **no model modification**: it is a read-only recombination of tensors `forward()` already returns. |
| **A6** | The Phase-8 `gate_mean` recorded for *repaired* witnesses is actually the **source** model's gate (instrumentation defect in a diagnostic-only field). It is additionally a *position-weighted* mean, not an item-level one. | **Gate behaviour after Arm-A repair has never been measured.** Interpretation-space outcome E is currently untested, not tested-and-negative. |

---

## 1. Where `g` is computed

`models/gating.py:44-57` — the whole gate:

```python
def gate_value(self, wm, ltm, field):
    B, S, _ = wm.shape
    if field is None or "confidence" not in field:
        return torch.full((B, S, 1), 0.5, device=wm.device)
    conf = field["confidence"].view(B, 1, 1)
    g = torch.sigmoid(self.cfg.alpha * (conf - self.cfg.gate_threshold))
    return g.expand(B, S, 1)

def forward(self, wm, ltm, field=None):
    g = self.gate_value(wm, ltm, field)
    premotor = g * ltm + (1.0 - g) * wm
    return {"premotor": premotor, "gate": g}
```

`class Gate(nn.Module)` declares **no `nn.Parameter` and no submodule**. `build_gate` returns it
unchanged. The gate is a fixed function with two scalar hyperparameters.

Note the signature: `gate_value` **accepts** `wm` and `ltm` but uses them only for `wm.shape`.
The dorsal premotor vector is passed in and discarded.

### The fallback branch is a live 0.5/0.5 gate
When `field is None` (no semantic bank, or bank size ≤ 1) the gate returns a constant `0.5`.
The forced-symmetric condition of Experiment 2 is therefore already an *in-architecture* behaviour,
not a foreign intervention.

## 2. What `g=0` and `g=1` mean, and which route gets which weight

`premotor = g * ltm + (1 - g) * wm`, so:

* **`g → 1` = trust the ventral / LTM / lexical route** (effective ventral weight = `g`)
* **`g → 0` = trust the dorsal / WM / buffer route** (effective dorsal weight = `1 - g`)

The two effective weights are **not free**: they are constrained to sum to exactly 1. There is one
degree of freedom per item, not two. "Effective dorsal weight" and "effective ventral weight" in the
item-level table are therefore `1-g` and `g`; they carry no information beyond `g` itself and are
emitted only for readability.

## 3. What the gate acts on: states, not logits — but the distinction is vacuous here

The gate mixes **premotor state vectors** (`models/dual_route.py:85`), and the mixed state is then
read out by `MotorCortex` (`models/motor.py:17`), which is a **single `nn.Linear`**.

Because the blend weights sum to 1, the bias passes through exactly:

```
motor(g·ltm + (1-g)·wm) = W(g·ltm + (1-g)·wm) + b
                        = g·(W·ltm + b) + (1-g)·(W·wm + b)       [g + (1-g) = 1]
                        = g·ltm_logits + (1-g)·wm_logits
```

**Verified: max|Δ| = 1.6e-07 (float32 round-off).** See §6, CLAIM 1.

This is the single most useful fact in this audit. It means gating-on-states and gating-on-logits are
the same operation, and any fixed-`g` intervention can be applied **post hoc to the logits that
`forward()` already returns**, with no hook, no monkeypatch and no model edit.

⚠️ This equivalence is **specific to the current architecture**. It would break the moment `motor`
gains a nonlinearity, or a semantic-attractor loop makes `ltm` depend on `g`. Any future architecture
change must re-verify it rather than inherit it.

## 4. Is route-only evaluation mathematically equivalent to true route isolation?

**Yes at the gate, with three caveats that must be stated in the paper.**

`models/dual_route.py:157-181` — `route_logits(route="wm")` returns `motor(wm_premotor)`, computed
through a code path that never touches the gate. That is bit-identical to the `g=0` limit of the
gated path; likewise `route="ltm"` vs `g=1`.

**Verified: max|Δ| = 0.000e+00 for both routes.** See §6, CLAIM 2. This independently reproduces the
equality already found in the exploratory lesion work (`lichtheim3-brain-damage@a5c787a`, which
reports `max|diff| = 0.000e+00` for forced `g=0`/`g=1` against the isolated routes).

Caveats:

1. **Isolation is functional, not anatomical.** Two distinct sharing facts, both verified:
   * `phon_embed.weight` is the **only shared *parameter* tensor** reachable under both
     `model.wm` and `model.ltm` (**10 WM-exclusive tensors, 16 LTM-exclusive**, at the
     checkpoints' own config: `wm.hidden=128`, `ltm.enc_hidden=dec_hidden=512`,
     `ltm_encoder_mode="unigru_last_hidden"`). It is shared, and **dorsal training can move it**,
     so it is a channel by which dorsal learning reaches `ŝ` — and hence the gate — indirectly.
   * The routes also share the **downstream motor projection** `motor.proj`
     (`models/dual_route.py:88-90`): `wm_logits`, `ltm_logits` and `logits` are all read out
     through the same `nn.Linear`.

   Therefore **functional route isolation means "isolated premotor contribution into a shared
   readout", not "anatomically independent subnetworks"**. The §5 blindness result is correspondingly
   precise: the gate is blind to dorsal-**exclusive** parameters and dorsal activations. It is not a
   claim that dorsal training can never influence the gate, because `phon_embed.weight` is shared.
2. **Isolation is a gate ablation, not a lesion.** It sets the mixing weight to an endpoint; it does
   not damage a route. Phase 8's `SCIENTIFIC_LESION_READINESS=NOT_ESTABLISHED` is unaffected.
3. **Trajectory divergence under free-AR is real and intended.** Under teacher forcing all routes see
   the gold prefix, so the decomposition in §3 is exact *item-by-item, position-by-position*. Under
   autoregressive decoding each route follows its own prefix, so `full` is **not** a per-position
   blend of the `wm` and `ltm` trajectories — only the *logits at a shared prefix* blend. Both
   canonical evaluators (`_ar_decode_batch`, `free_ar_repetition`) decode each route on its own
   prefix, which is the correct definition of route isolation but means Experiment 2's forced-0.5
   condition must be **re-decoded**, never reconstructed from stored per-route predictions.

## 5. The gate cannot see the dorsal route's state — the decisive structural result

The brief's first scientific question is whether the gate tracks **relative** route competence.
At the level of code the answer is already determined:

`g = σ(α·(c_LTM − τ))` where `c_LTM = max_i cos(ŝ, bank_i)`, and `ŝ = ltm.encode(...)`.
No term in that expression depends on any dorsal quantity.

**Verified:** perturbing **all ten WM-exclusive parameter tensors** by `N(0, 0.5²)` moves `wm_logits`
by `max|Δ| = 1.03e+01` and moves the gate by `max|Δ| = 0.000e+00` exactly. See §6, CLAIM 5.

Stated precisely: the gate is blind to **dorsal-exclusive parameters and dorsal activations**. It is
**not** blind to `phon_embed.weight`, which the two routes share and which dorsal training can move
(§4, caveat 1). The blindness is to the dorsal route's *state and competence at inference*, which is
what matters for the adaptive-routing question.

Therefore any *apparent* item-level relationship between `g` and relative route competence must be
**entirely mediated by `c_LTM`** — i.e. it can only arise if ventral lexical confidence happens to
correlate with the *difference* in route correctness. That is an empirical question worth measuring
(it is exactly Experiment 1), but it is a question about the ventral confidence signal, not about an
adaptive controller. **There is no adaptive controller.**

This also predicts, without running anything, that the gate cannot reweight in response to dorsal
damage — which is what the exploratory lesion diagnostic observed (`gate = 0.5160` unchanged from
intact to 4.23× site SD).

## 6. Numerical verification

Run: `python3 -m pytest tests/test_gate_route_diagnostics.py -v` (see that file; a standalone
transcript is in `figure_source_data/gate_algebra_verification.txt`).

| Claim | Statement | Result |
|---|---|---|
| 1 | `g·ltm_logits + (1−g)·wm_logits == logits` | max\|Δ\| = **1.639e-07** (fp32 round-off) |
| 2a | `route_logits("wm") == wm_logits` | max\|Δ\| = **0.000e+00** |
| 2b | `route_logits("ltm") == ltm_logits` | max\|Δ\| = **0.000e+00** |
| 3 | forced-0.5 mix computable post hoc from one forward | constructed, shape-checked |
| 4 | `g` constant across decoder steps (word-level) | max\|Δ\| = **0.000e+00** |
| 5 | `g` invariant to WM-exclusive parameter perturbation | max\|Δ\| = **0.000e+00** (`wm_logits` moved 1.03e+01) |
| 6 | attainable range of `g` at `α=2.0, τ=0.7` | see table below |

### Attainable gate range (cohort setting `α=2.0`, `τ=0.7`) — **asymmetric**

| `c_LTM` | −1.00 | 0.00 | 0.30 | 0.50 | **0.70** | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| ventral weight `g` | **0.0323** | 0.1978 | 0.3100 | 0.4013 | **0.5000** | 0.5987 | **0.6457** |
| dorsal weight `1−g` | **0.9677** | 0.8022 | 0.6900 | 0.5987 | **0.5000** | 0.4013 | **0.3543** |

Since `c_LTM` is a cosine similarity bounded in [−1, 1]:

* **ventral weight `g ∈ [0.0323, 0.6457]`**
* **dorsal weight `1−g ∈ [0.3543, 0.9677]`**

The two extrema are therefore **not symmetric**:

| extreme | route-weight ratio | attainable? |
|---|---|---|
| maximum dorsal commitment (`c_LTM = −1`) | ≈ **29.9 : 1** dorsal : ventral | **yes** |
| maximum ventral commitment (`c_LTM = +1`) | ≈ **1.82 : 1** ventral : dorsal | **yes, but this is the ceiling** |

So: **strong dorsal commitment is structurally possible; strong ventral commitment is structurally
impossible.** The ventral route can never receive more than 64.6 % of the blend, and the dorsal route
can never be suppressed below 35.4 %.

This refines — and partly inverts — the concern raised in the 14 Sep meeting (§9 of the notes) that
the gate might "saturate too easily towards one route". Under the cohort's own hyperparameters the
gate *can* saturate toward the **dorsal** route but *cannot* saturate toward the **ventral** one. The
correct statement is about **asymmetry of the attainable range**, not about an absence of saturation.

## 7. Empirically measured gate values on the Phase-8 cohort

From the frozen Phase-8 battery JSONs (`gate_mean` over the full 29,571-item repetition population):

| Witness | source `gate_mean` | repaired `gate_mean` | Δ |
|---|---|---|---|
| V5 seed20 u3600 | 0.5197919607162476 | 0.5197919607162476 | **0.0 (bit-identical)** |
| V5 seed21 u3400 | 0.5189160108566284 | 0.5189160108566284 | **0.0 (bit-identical)** |
| V6 seed19 u3825 | 0.5212951898574829 | 0.5212951898574829 | **0.0 (bit-identical)** |
| V6 seed20 u3040 | 0.5170738101005554 | 0.5170738101005554 | **0.0 (bit-identical)** |

Two things follow.

**(a) The deployed gate's *recorded mean* is close to 0.5 — and that is all that is established.**
The frozen values are **position-weighted** means (the historical statistic flattens the gate over
`(B, S, 1)` including padding columns, so each item is weighted by its batch's padded decoder width).
They sit at 0.517–0.521, so **replacing the gate by a constant 0.5 changes that signed mean by about
0.02.**

That statement must not be inflated into "the intervention is a ~0.02 perturbation". A near-0.5 mean
says nothing about:

* the item-level mean `|g − 0.5|` (a symmetric spread around 0.5 has a small signed mean and a large
  mean absolute deviation);
* the variance or shape of the item-level `g` distribution (bimodal vs narrow unimodal);
* the per-item effect on the logits.

The per-item logit change under the intervention is exactly

```
Δlogits_i = (0.5 − g_i) · (ltm_logits_i − wm_logits_i)
```

which is a **product**. It can be large wherever the two routes disagree strongly, even when
`g_i ≈ 0.5`, and it is not bounded by the mean gate offset. Only the item-level distribution — which
was **never recorded**, since only the mean was — can say how large the intervention actually is.
Establishing that distribution is the first thing Experiment 1 must do.

**(b) Finding A6: those repaired numbers are not the repaired model's gate.**

`scripts/naming_comprehension/coexistence_probe.py:125-161` (`full_battery`) threads the repaired
`model` explicitly into every evaluator — `evaluate_comprehension_subset(model, ...)`,
`repetition_snapshot(model, ...)`, `evaluate_naming(model, ...)` — and temporarily swaps `tr.model`
for the free-AR call. But the gate line is:

```python
g = tr.gate_statistics(all_idx)          # <-- tr, not model
out["gate_mean"] = float(g.get("gate_mean", float("nan")))
```

and `train_joint_scratch.py:1613-1635` shows `gate_statistics` evaluating `self.model` — the
trainer's **source** model. Arm A modifies `ltm.to_semantic.2.{weight,bias}`, which changes `ŝ`,
hence `c_LTM`, hence `g`. A genuinely recomputed repaired gate could not be bit-identical to the
source's across four independent witnesses; four-for-four bitwise identity is the signature of the
source model being measured twice.

**Scope of this defect — deliberately narrow:**
* It affects `gate_mean` **only**. Every scientific metric in the Phase-8 batteries (C, N, R
  canonical, R free-AR, isolated LTM) is computed from the correctly-threaded `model`.
* `gate_mean` appears in **no** Phase-8 REPORT-SAFE claim, no freeze-manifest artifact hash, and no
  established result. `PHASE8_FREEZE_MANIFEST.tsv` and `RESULTS_REGISTRY` are untouched by it.
* **No HARD_FROZEN Phase 1–8 claim is revised by this audit.** The correction is that one
  diagnostic-only column was never a measurement of what its name implies.
* Consequence for *this* workstream: the question "does terminal semantic repair change gate
  behaviour?" (interpretation-space outcome **E**) has **never been measured**. Experiment 1 must
  measure both members of each pair independently and must not reuse the historical column.

## 8. Effect of the gate on training

The gate is not inert during learning even though it has no parameters. `losses.py:53-69`:

```python
def gate_regularizer(gate, usage_prior):
    return (gate.mean() - usage_prior) ** 2
```

with `LossConfig.gate = 0.05` and `GatingConfig.usage_prior = 0.5` (confirmed present in all four
checkpoint configs). Gradients flow through `g` into `c_LTM` into the ventral encoder, actively
pulling the **mean** gate toward 0.5. The recorded position-weighted signed mean of ≈ 0.52 is
therefore partly a *trained-for* outcome, not purely an emergent one.

This matters for interpreting Experiment 2: the network was optimised under a mild pressure toward
the very symmetry that the 0.5/0.5 intervention imposes. Finding that 0.5/0.5 changes little would
therefore be **weaker evidence that confidence-driven weighting is functionally necessary than it
first appears**, because the training objective already discouraged strong asymmetry in the mean.
This must be stated in the recap regardless of outcome.

Note that this concerns the **mean** only. The regularizer penalises `(mean(g) − 0.5)²`, which
constrains no item-level quantity: a widely dispersed `g` with mean 0.5 incurs zero penalty. It
therefore says nothing about the item-level or logit-level magnitude of the intervention, which
remains unmeasured (§7a).

Note also `L_wm` (`λ_wm = 0.5`): a dedicated WM-only repetition CE. Both routes are trained to be
independently competent at repetition, which is why `BOTH_CORRECT` is expected to dominate and why
the discriminative categories may be small — `LTM_ONLY_CORRECT` turns out to be bounded above by 2
on these states (`EXPERIMENT_CONTRACT.md` AMENDMENT 1). Sample counts are a primary reportable, not
a footnote.

## 9. Unused signals

`ltm_route.lexical_field` computes `sims`, `confidence`, `margin`, `density`. `DualRouteModel`
forwards `confidence`, `margin` and `density` into the gate. **Only `confidence` is read.**
`margin` (top-1 minus top-2 — an inverse-competition signal) and `density` (soft neighbourhood count)
are computed, carried, and discarded at every forward pass.

This is not a defect, but it is a design fact worth recording: two plausible routing signals are
already available and already plumbed. They are recorded per item in Experiment 1 as *descriptive*
columns. **No gate variant using them is to be built or tuned in this workstream** (§9 STOP).

## 10. Aligning item metadata

`scripts/evaluate_train_lexicon_ceiling.py:198-217` (`evaluate_forms_ar`) already emits, per item:
`word`, `rank`, `target_phonemes`, `length`, `zipf_approx` (frequency), plus per-route
`{route}_exact_match`, `{route}_edit_dist`, `{route}_norm_edit_dist`, `{route}_predicted`.

Alignment is **by construction**, not by a join: the row is built from the same `LexEntry` object
that supplied the phonemes, inside the same loop iteration. `LexEntry` (`data/lexicon.py:75-84`)
carries `word`, `phonemes`, `semantic`, `freq`, `rank` and a `length` property. There is no key-matching
step that could silently misalign, and no fuzzy string match. Frequency and length therefore require
**no new provenance** — they ride on the frozen lexicon
(`data/lexicon_en_glove_covered.tsv`, hash recorded in each checkpoint as `lexicon_file_sha256`).

### Lexicality is the one genuinely missing axis

The canonical 29,571-item repetition population is **entirely real words**. `is_word` is constant, so
`gate vs lexicality` **cannot be computed on the primary population at all** — the contrast does not
exist there. Pseudowords live in separate assets (`data/eval_external/{wfe,ssp}_eval.tsv`,
`data/raw-nwr_swp/`). Those are a different population with their own provenance status, and — unlike
frequency and length — they are **not covered by the Phase-8 freeze manifest**.

Accordingly the experiment contract splits the deliverable in two: a **primary** battery on the frozen
29,571-word population (where frequency, length and route competence are all valid), and a
**secondary, provenance-gated** lexicality battery that is reported **only if** the pseudoword assets
pass an explicit provenance check, and is marked NOT-CANONICAL if they do not. Merging them into one
table would silently mix a frozen population with an unfrozen one.

## 11. What must NOT be concluded from this audit

* Not that the gate is useless. A fixed confidence-based mixer can still be functionally valuable;
  §5 says only that it cannot be *adaptive to the dorsal route's inference-time state*.
* Not that the gate "cannot saturate". It **can** saturate toward the dorsal route (down to
  `g = 0.0323`, ≈ 29.9 : 1) and **cannot** toward the ventral route (up to `g = 0.6457`,
  ≈ 1.82 : 1). The finding is the **asymmetry** (§6, CLAIM 6).
* Not that `fixed05` is a "small" or "~0.02" perturbation. What is established is only that the
  *recorded position-weighted mean* moves by ≈ 0.02. The per-item logit change is
  `(0.5 − g_i)·(ltm_logits_i − wm_logits_i)`, a product that can be large wherever the routes
  disagree (§7a).
* Not that 0.5/0.5 will be behaviourally equivalent. The item-level `g` distribution is unmeasured
  and could still be doing work on exactly the items that matter.
* Not that the two routes are anatomically independent. They share `phon_embed.weight` **and** the
  `motor.proj` readout (§4, caveat 1).
* Not that Phase 8 is wrong. One diagnostic column was mis-sourced; every frozen scientific metric
  stands.
* Not that `α`/`τ` should change. They are frozen inputs here (§9 STOP condition).

## 12. Audit-stage risks carried into the contract

| Risk | Handling |
|---|---|
| Both routes near-ceiling on repetition (`L_wm` is trained) → discriminative cells tiny | Report `n` per cell first; declare the discrimination metric UNDERPOWERED rather than quoting an AUROC on a handful of items |
| `g` has one degree of freedom, so "dorsal weight vs ventral weight" is not two measurements | Report `g` as the single statistic; AUROC on `g` and on `1−g` are trivially complementary |
| Free-AR forced-0.5 cannot be reconstructed from stored per-route predictions | Contract mandates re-decoding (§4 caveat 3) |
| Historical `gate_mean` is source-sourced | Recompute both members of every pair; never reuse the column (§7) |
| Pseudoword assets outside the freeze | Provenance-gated secondary battery (§10) |

## 13. Naming and Comprehension do not involve the gate

Traced through the canonical evaluators. Neither task constructs a gate, so the `fixed05`
intervention is **mathematically inert** for both.

**Comprehension** (`train_tasks.evaluate_comprehension_subset:1091-1126`):
```
encode_all -> s_hat -> comprehension_metrics (cosine retrieval against the bank)
```
No decoder, no premotor, no gate, no `motor`. C is a property of `ŝ` alone.

**Naming** (`train_tasks.evaluate_naming:668-711` → `frozen_probe.semantic_greedy_decode:158-198`):
```
raw GloVe sem -> ltm.decode_from_s_hat -> motor -> argmax
```
The gate module is never called; only the ventral decoder and the shared readout are used.

**Isolated routes** likewise bypass the gate by construction (§4).

Consequences, all of which are *invariances* rather than predictions:

| quantity | effect of forcing `g ≡ 0.5` |
|---|---|
| Comprehension (strict C) | **exactly zero** |
| Naming (exact match) | **exactly zero** |
| isolated WM-only repetition | **exactly zero** |
| isolated LTM-only repetition | **exactly zero** |
| FULL/gated repetition | the only quantity that can move |

Therefore the experiment contract must **not** evaluate C, N or the isolated routes under `fixed05`
as if they could change: doing so would spend compute to re-derive a mathematical identity and would
invite reporting a null as though it were an empirical finding. They are recorded once, as
invariants, and the intervention is assessed on FULL/gated repetition alone.

This also constrains the outcome space: an outcome of the form "`fixed05` improves FULL **without**
degrading C/N or the isolated routes" cannot be evidence about the gate, because the second half of
that conjunction is true by construction.

---

---

## Appendix — audited files

| File | Lines | Role |
|---|---|---|
| `models/gating.py` | 1–61 | the gate; no parameters |
| `models/dual_route.py` | 74–99, 157–181 | fusion, route isolation |
| `models/motor.py` | 14–21 | single `nn.Linear` readout |
| `models/ltm_route.py` | 107–200 | `ŝ`, `lexical_field`, `confidence` |
| `models/wm_route.py` | 52–60 | shared `phon_embed` |
| `config.py` | 164–194 | `GatingConfig`, `LossConfig` |
| `losses.py` | 53–73 | gate regularizer |
| `scripts/evaluate_train_lexicon_ceiling.py` | 137–223 | canonical forced-length AR, item rows |
| `scripts/naming_comprehension/coexistence_probe.py` | 125–161 | `full_battery`; the `tr.model` defect |
| `scripts/naming_comprehension/train_joint_scratch.py` | 1562–1635 | `free_ar_repetition`, `gate_statistics` |
| `scripts/external_eval.py` | 136–176 | per-item gate/field capture (reused) |
| `lichtheim3-brain-damage@a5c787a:lesion/gate_diagnostic.py` | 1–182 | `forced_gate` (EXPLORATORY; ported as cross-check) |
