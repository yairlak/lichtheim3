# Lichtheim3: Evaluation Regimes

Three evaluation regimes are defined for the lichtheim3 dual-route model.
They differ in decoding mode (teacher-forced vs. autoregressive) and whether
WM interference noise is active.

---

## Regime A — Deterministic Teacher-Forced

### Purpose
Ceiling measurement and per-item debugging.
Answers: *"Can the model produce this phoneme sequence given the correct prefix?"*

### Decoding mode
**Teacher-forced**: at each decoder step `t`, the gold (correct) phoneme from
position `t-1` is fed as the decoder input, regardless of what the model predicted.

```
step 1:  input=[BOS],       predict P1
step 2:  input=[gold P1],   predict P2
step 3:  input=[gold P2],   predict P3
...
```

### WM noise
**OFF** (`collect=False`). Fully deterministic: identical predictions on repeated runs.

### LTM s_hat
Batch-padding sensitive (known limitation). Predictions may differ depending on
batch composition for short words in long batches. See `docs/current_pipeline_summary.md §10`.

### Expected outputs
- Train-split ceiling: full_exact_match = 1.0000 on 25,136 training words.
- WFE: full ≈ 0.987, WM ≈ 0.987, LTM ≈ 0.790.
- Error rate is a lower bound on what free-generation would produce.

### CLI flag / code
```python
collect = False   # in evaluate.hooks.route_predictions
```
```bash
python scripts/external_eval.py   # --wm_noise not passed → collect=False for WM
```

### Use in paper figures
**No** for main behavioral comparisons with Dager/SWP (teacher-forcing is not
what participants do in a repetition task). **Yes** for:
- train-ceiling sanity checks;
- per-item debugging;
- confirming that both routes are in principle capable of correct output.

---

## Regime B — Deterministic Autoregressive

### Purpose
**Main behavioral comparison to Dager/SWP.**
Answers: *"What does the model actually output when it runs freely, without gold input?"*

### Decoding mode
**Autoregressive (free generation)**: at each decoder step `t`, the model's own
predicted phoneme from step `t-1` is fed as the next input.

```
step 1:  input=[BOS],         predict P1
step 2:  input=[pred P1],     predict P2   ← uses model's own output
step 3:  input=[pred P2],     predict P3
...
```

Errors propagate: a wrong prediction at position `t` corrupts the context for all
subsequent positions.

### WM noise
**OFF** (`collect=False`). Deterministic across runs: same checkpoint → same
autoregressive output every time.

### LTM s_hat
Batch-padding sensitive (same caveat as Regime A). For clean results, evaluate
each item solo or use a fixed batch-size with consistent padding.

### Expected outputs
- Lower exact-match accuracy than Regime A (errors propagate).
- Stronger length effects: long words accumulate more errors.
- Serial-position curve: primacy/recency effects amplified relative to Regime A.
- Real word / pseudoword dissociation: LTM advantage for real words becomes clearer
  because the LTM decoder starts from a semantically grounded `s_hat` and can
  recover from small early errors; WM loses context after the first wrong phoneme.

### CLI flag / code
```python
# In external_eval.py — autoregressive_decode_batch()
dec_input = batch["enc_in"].new_full((B, 1), vocab.bos_id)
for _ in range(max_steps):
    res = model.route_logits(enc_in, enc_mask, dec_input, route=route)
    next_tok = res["logits"][:, -1, :].argmax(-1, keepdim=True)
    dec_input = torch.cat([dec_input, next_tok], dim=1)
```
```bash
python scripts/external_eval.py \
    --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
    --out_dir outputs/external_eval_30k \
    --decode autoregressive --wfe_only
# outputs: outputs/external_eval_30k/wfe_ar/
```

### Use in paper figures
**Yes** — this is the primary regime for:
- WFE real vs. pseudo accuracy by word length (Dager-comparable figure);
- serial-position curves;
- route dissociation plots;
- any figure that maps to a human behavioral task.

---

## Regime C — Autoregressive + WM Noise (Cognitive / Noisy)

### Purpose
**Length-effect and serial-position analysis with cognitive plausibility.**
Answers: *"What does the model output when the WM buffer is noisy, as it would be
in a capacity-limited biological working-memory system?"*

### Decoding mode
**Autoregressive** (same as Regime B).

### WM noise
**ON** (`collect=True`). Gaussian noise is added to the WM encoder hidden state `h`
after the encoding step:

```python
h = h + torch.randn_like(h) * cfg.interference_noise
```

This simulates the capacity limits and interference effects of phonological working
memory. The noise is applied once per item (per encoder pass), not per decoder step.
Results are **non-deterministic**: different runs give slightly different outputs.
Average over multiple runs (or over items) to get stable estimates.

### LTM s_hat
Unchanged by WM noise: the LTM route is not affected by `collect=True`.
Batch-padding sensitivity still applies.

### Expected outputs
- Lower WM accuracy than Regime B (noise degrades recall).
- **U-shaped serial-position curve** for the WM route: primacy and recency
  advantages, with maximum error rate in the middle of the phoneme sequence.
- Stronger length effect than Regime B: noise compounds over longer sequences.
- LTM route relatively unaffected by WM noise (LTM does not use `h`).
- Gate: noise makes WM less reliable, but gate confidence is determined by LTM
  `s_hat` (not WM state), so gate weights are unchanged.

### CLI flag / code
```python
collect = True   # enables noise in WMRecurrent.forward
```
```bash
python scripts/external_eval.py --wm_noise
```

### Use in paper figures
**Yes**, for:
- length-effect figures comparing WM and LTM routes;
- serial-position curves (primacy/recency);
- **only** when the figure is explicitly labelled "cognitive/noisy evaluation"
  or "WM noise active".
- **Not** for accuracy comparisons to Dager/SWP (noise makes direct comparison
  inappropriate unless matched to the patient data).

---

## Summary Table

| Regime | Decoding | WM noise | Deterministic | Primary use |
|---|---|---|---|---|
| A — Teacher-forced | Teacher-forced | OFF | ✓ | Ceiling / debug |
| B — Autoregressive | Free generation | OFF | ✓ | **Dager comparison (main)** |
| C — Autoregressive + noise | Free generation | ON | ✗ | Length effects / serial-position |

---

## Implementation Status

| Regime | Status |
|---|---|
| A — Deterministic teacher-forced | ✅ Implemented (`external_eval.py --decode teacher_forced`, `evaluate_train_lexicon_ceiling.py`) |
| B — Deterministic autoregressive | ✅ Implemented (`external_eval.py --decode autoregressive`) |
| C — Autoregressive + WM noise | ❌ Not yet implemented (noise alone is implemented; autoregressive loop is not) |

The teacher-forced results in `outputs/external_eval_30k/` are Regime A only.
All current meeting-pack figures are Regime A.
