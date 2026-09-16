# IMPLEMENTATION NOTES — GATE × LESION / RECOVERY

Notes recorded during implementation and quarantined validation, **after** the DESIGN
freeze. This file deliberately does not modify any frozen design artifact: the four
contract artifacts, `LIVE_CODE_AUDIT.md` and `gxlr_conditions.frozen.json` keep the
SHA256 they had at `DESIGN_FREEZE_COMMIT`. Anything here that bears on the contract is
raised to CENTRAL as a finding rather than edited in.

---

## F-8 — `batch_size` is a nuisance parameter for the GATE, and is pinned at 256

**Discovered by:** T2, the severity-0 authoritative null, run at `batch_size=512`.

**What happened.** Every substantive assertion passed on the full 29 571-item
population, on both states: identical `word` order, identical predictions across all
four routes (`full`, `wm`, `ltm`, `fixed05`) under *both* decoding conventions, and
**zero** NATIVE-vs-FIXED05 discordant items. The only mismatch was the gate value, on
a handful of items, at the level of 1–2 units in the last place.

**Measured** (W3_REP, first 2 048 items, against the frozen GATING shard):

| batch size | items whose gate differs | max abs difference |
|---|---|---|
| **256** | **0 / 2048** | **0.0 (bitwise identical)** |
| 512 | 15 / 2048 | 1.192e-07 |

**Cause.** Both encoders pack their input (`pack_padded_sequence(..., enforce_sorted=
False)`, `wm_route.py:87-89`, `ltm_route.py:136-139`). The packed GRU kernel's reduction
order depends on batch composition, so the last bits of `h_n` — and hence of `s_hat`,
`c_LTM` and `g = sigmoid(alpha·(c_LTM − threshold))` — depend on how items were grouped.
This is inherited arithmetic behaviour of the frozen lineage, not something this
workstream introduced.

**Why it does not touch the lesion.** `epsilon` is a pure function of
`(state_sha256, route, lesion_seed, item_id)` and is assembled per item, so it is
**batch-invariant by construction** (T3, T4, T6). The nested scaling is bitwise exact
(T5). Nothing about the intervention moves with batch size.

**Disposition.** `batch_size = 256` — the value the frozen GATING record was produced
with — is the runner default, is what `EXECUTION_COMMAND.prepared.sh` passes explicitly,
and is what T2 asserts against. `g` and `c_LTM` are reported continuous measurements
(contract §8), so pinning the batch size is what makes them exactly comparable with the
frozen intact record.

**Raised to CENTRAL** as finding F-8. It is a reproducibility pin, not a design change:
no severity, route, seed, population or classification rule is affected.

---

## Implementation choices worth recording

**`gate_x_lesion.evaluate.collect_item_level_lesioned` is a deliberate near-duplicate**
of `gating_diagnostics.gate_probe.collect_item_level`. It could not simply call it: the
lesion hook must be bound to each batch's item ids *before* any forward, and the binding
point is inside that function's batch loop. The decoders themselves
(`ar_decode_forced_length`, `ar_decode_free`, `capture_gate_field`, `competence_category`)
are **imported**, never reimplemented, so both decoding conventions and the FIXED05
algebra stay exactly what the GATING workstream validated. T2 pins the equivalence: with
no hook, the wrapper reproduces the frozen shard item by item.

**Severity 0 registers no hook at all** rather than adding an exactly-zero tensor. This
reproduces the historical semantics (`lesion/apply.py:169-176` gates on `amp > 0.0`) and
makes the intact control a genuinely untouched forward, which is what lets T2 assert
bitwise identity against the frozen record.

**`RouteNoiseHook` raises if it fires unbound.** Item identity is stated by the caller,
never inferred from tensor content — the historical fingerprint heuristic
(`lesion/apply.py:43-65`, defect H-1) is not reproduced even as a fallback.

**The outcome classifier is importable without torch.** T12 exercises it on synthetic
records only, so no scientific lesion output can reach a classification test.

---

## What was validated, and on what

| test | population |
|---|---|
| T2 | FULL 29 571 items, both states, **severity 0 only** (intact control) |
| T6–T10 | 24-item non-canonical subset, quarantined |
| T1, T3–T5, T11, T12 | no model / synthetic / hash-only |
| smoke runner | 24-item non-canonical subset, written under `NOT_SCIENTIFIC_RESULT/` |

**No non-zero-severity lesion was run on the canonical population.**
`GO_FOR_SCIENTIFIC_EXECUTION = NO`.
