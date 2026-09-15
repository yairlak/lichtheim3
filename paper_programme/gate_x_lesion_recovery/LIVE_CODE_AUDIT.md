# LIVE CODE AUDIT — GATE × LESION / RECOVERY

**Scope:** line-by-line reverification of the code that the GXLR experiment will actually
execute. Not a reading of design notes; every verdict below is anchored to a blob hash, a
path and a line range in the tree that will be run.

**Base tree.** All *live* paths are read at

```
993f2e73ecd0b9de2c3318255afbaa159948b09a   results(gating): Experiments 1 and 2 complete
```

which is the frozen GATING results commit and the parent of the GXLR branch
`paper-programme/gate-x-lesion-recovery` (worktree `wt-gate-x-lesion/`). Blob ids are
`git rev-parse 993f2e73:<path>`.

**Historical** paths are read at `3af0ef9c00df5aba8200d56f566bb957d084109c` and are audited
in `HISTORICAL_LESION_OPERATOR_AUDIT.md`; this document cites them only where GXLR ports
a semantic.

**Environment.** python 3.11.15, torch 2.12.1 — *bitwise identical* to the torch version
recorded in the frozen GATING provenance (`summary_metrics.json`,
`states.W3_REP.provenance.torch == "2.12.1"`). No new environment was created; the
existing interpreter is reused, as directed.

---

## 0. Blob index

| # | live path @ `993f2e73` | blob |
|---|---|---|
| L1 | `models/gating.py` | `8b9244abeb1ad9150c481ab25b316d0eda3edf8e` |
| L2 | `models/dual_route.py` | `5166df420e0f70994e76c68a3542f6a3abfe9809` |
| L3 | `models/wm_route.py` | `e58b9670e708d419d88367f4cc8f27b92b4f4c3e` |
| L4 | `models/ltm_route.py` | `548cc68735830986ee38801997c83b2781252d22` |
| L5 | `models/motor.py` | `cf8ffef02450451b7e84f5ce554bd82ac41bdd6f` |
| L6 | `gating_diagnostics/gate_probe.py` | `565b80962962c153ef47f081af1d9a9dab02ae0e` |
| L7 | `scripts/gating_diagnostics/run_gate_route_audit.py` | `fa0a259802e8b9b0caa48d7d4d5ac74b79c22ceb` |
| L8 | `scripts/naming_comprehension/train_joint_scratch.py` | `95295d63560ae4c235a6beee8dfb47166f4ed30d` |
| L9 | `scripts/naming_comprehension/frozen_head_probe.py` | `72b5f46d8a007adde37a4915912c1e9ed949a21b` |
| L10 | `evaluate/hooks.py` | `96cf63d921790a853388b7cad5cc7ba46e25f966` |
| L11 | `paper_programme/gating_route_diagnostics/checkpoint_manifest.tsv` | `e5f0bb4fa68ebe1b200a3fcabe3544ebbff5483b` |

Blob L11 is byte-identical at the frozen *experiment* commit `f43ccd09…` and at the frozen
*results* commit `993f2e73…` (both resolve to `e5f0bb4f…`), so the manifest is stable
across the whole frozen GATING lineage.

New GXLR code audited in §7 lives on the GXLR branch and is committed by
`IMPLEMENTATION_COMMIT`.

---

## 1. Lesion machinery

GXLR does **not** import the historical `lesion/` package: it does not exist on this
lineage (`git ls-tree -r 993f2e73 -- lesion/` is empty; the diff from `3af0ef9` is 3 198
deletions). The machinery is re-implemented minimally in `gate_x_lesion/`, porting only
the audited semantics. The historical files are nevertheless audited, because they define
what must be preserved.

### 1.1 `lesion/apply.py` @ `3af0ef9` — hook return semantics

**Lines 99–113 — the exact tensor modified.**

```python
def __call__(self, module, inputs, output):
    key = (self._input_fingerprint(inputs)
           if self.spec.noise_timing == "per_item" else None)
    if isinstance(output, tuple):
        slot = self.slot
        perturbed = list(output)
        ref = perturbed[slot]
        if isinstance(ref, torch.nn.utils.rnn.PackedSequence):
            return output
        perturbed[slot] = ref + self._noise_for(ref, key)
        return tuple(perturbed)
    return output + self._noise_for(output, key)
```

**Verdict — CONFIRMED, with one live-critical caveat.** A `torch` forward hook that
returns a non-`None` value *replaces* the module output. `nn.GRU` returns
`(output, h_n)`. The hook rebuilds the tuple, replacing exactly one slot, and adds noise
out-of-place (`ref + …`), so no autograd-visible or checkpoint-visible tensor is mutated.

The `PackedSequence` guard on line 108 is load-bearing here and is **exercised on this
architecture**: both live encoders are called with a packed input
(`wm_route.py:87–89`, `ltm_route.py:136–139`), so `output[0]` *is* a `PackedSequence`.
GXLR uses slot 1, so the guard is never the operative branch — but had `_GRU_SLOT` been
0 for an encoder, the lesion would have returned the output unchanged and been a **silent
no-op**. GXLR asserts the perturbed slot is a plain tensor and raises otherwise.

**Lines 116–128 — `_GRU_SLOT`.** `wm_encoder_state → 1`, `ltm_encoder_state → 1`
(i.e. `h_n`). Correct for these checkpoints; see §2 for the mode guard.

**Lines 169–176 — amplitude composition.** `amp = resolved_noise_amplitude() · scale ·
activation_scale_for(target)`; the hook is registered only when `amp > 0.0`.
Ported. `scale` is the composite-site multiplier and is always `1.0` for a simple target
(`spec.py:171–175`); GXLR has no composite sites, so it drops out.

**Lines 180–185 — restoration.** Handles removed and parameters restored in `finally`,
including on exception. **Verdict: CONFIRMED.** GXLR's context manager reproduces the
`finally`-removal discipline; it registers no parameter writes at all (activation noise
only), so its restoration surface is strictly smaller. Pinned by **T10**.

### 1.2 `lesion/targets.py` @ `3af0ef9`, lines 98–109

`wm_encoder_state → module_path "wm.encoder"`; `ltm_encoder_state → module_path
"ltm.encoder"`; both `kind="state"`. **Verdict: CONFIRMED against the live module tree**
(`dual_route` owns `self.wm` and `self.ltm`; each route owns `self.encoder`,
`wm_route.py:59`, `ltm_route.py:80/86`). GXLR resolves the module by the same dotted path.

### 1.3 RNG / cache utilities

**Historical** `lesion/masks.py:26–40` derives one `torch.Generator` per
`(lesion_seed, target_id, role)` via `int(sha256(...)[:16],16) % (2**31−1)`, consumed
sequentially. **Verdict: NOT PORTED — defective for this contract** (defects H-1..H-3 in
the historical audit): a sequential stream makes the realisation a function of batch
composition, batch order and call count, and makes it impossible for two severities to
share `epsilon`.

**Historical** `lesion/masks.py:171–192` (`draw_noise`): for `uniform_symmetric`,
`out = (torch.rand(shape) * 2.0 - 1.0) * amplitude`, with an `amplitude == 0.0`
short circuit to `torch.zeros`. **Verdict: SEMANTICS PORTED, GENERATOR REPLACED.** GXLR
keeps `epsilon ~ U(−1,+1)` and the operand order `epsilon · amplitude`; it replaces
`torch.rand(generator=…)` with a SHA-256 counter stream (§7.1), which is process-safe,
platform-safe and torch-version-independent.

**`hash()` is not used anywhere in GXLR's RNG identity** — Python's `str.__hash__` is
salted per process (`PYTHONHASHSEED`) and would be catastrophic here. Pinned by a grep
assertion in the test module and by **T3** (cross-process reconstruction).

### 1.4 Checkpoint restoration / hash utilities

Live: `run_gate_route_audit.py:113–118` (`sha256_file`, 1 MiB streaming) and
`126–165` (`build_state`). **Verdict: CONFIRMED and REUSED.** `build_state` verifies the
source checkpoint hash *before* loading (lines 132–136) and the head hash before applying
it (lines 154–156), and `run_state:211–213` re-verifies the source file after the pass and
hard-stops on any change. GXLR reuses this function unmodified and additionally hashes the
in-memory `state_dict` around every lesion context (**T10**), which is the stronger check:
a hook cannot alter the file, but could in principle alter a parameter.

---

## 2. Route hooks — live architecture trace

The historical target notes were **not** assumed current. The live model was traced and
the live checkpoint configs were read.

### 2.1 Checkpoint-declared architecture (read from the artifacts themselves)

Both GXLR states carry, in `ckpt["config"]`:

```
ltm: {phon_embed_dim: 64, enc_hidden: 512, enc_layers: 1, dec_hidden: 512,
      bidirectional_encoder: False, ltm_encoder_mode: 'unigru_last_hidden',
      ventral_noise: 0.0}
wm:  {hidden: 128, interference_noise: 0.0}
gating: {alpha: 2.0, usage_prior: 0.5, gate_threshold: 0.7}
```

identical for W3 (`seed 19`, `global_step 10625850`) and W4 (`seed 20`,
`global_step 8445120`). The gating block matches the frozen GATING provenance exactly
(`alpha 2.0`, `gate_threshold 0.7`). **Gate architecture, alpha, threshold and the
fixed05 definition are untouched by this workstream.**

### 2.2 `wm_encoder_state` — dorsal

`models/wm_route.py:85–93`:

```python
lengths = enc_mask.sum(1).clamp(min=1).cpu()
emb = self.phon_embed(enc_in)
packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True,
                                           enforce_sorted=False)
_, h = self.encoder(packed)                  # h: (1, B, hidden)
if (self.training or apply_noise) and self.cfg.interference_noise > 0:
    h = h + torch.randn_like(h) * self.cfg.interference_noise
return h
```

* Site tensor: `h_n`, shape **(1, B, 128)**, slot **1**. Matches `_GRU_SLOT`.
* `h` is returned directly and consumed only by `decode_from_state`
  (`wm_route.py:95–106`: `dout, _ = self.decoder(self.phon_embed(dec_in), h)`), i.e. as
  the dorsal decoder's initial hidden state. There is no other consumer.
* `interference_noise == 0.0` on both checkpoints and `apply_noise=False` throughout the
  GATING evaluator (`gate_probe.py:156–159`), so the model's own stochastic noise is
  **inert**. The lesion is the only perturbation. Verdict: CONFIRMED.

**Structural independence.** `h_WM` reaches the fusion only through `wm_premotor`
(`dual_route.py:142`). It never reaches `s_hat`, never reaches `lexical_field`, and
therefore never reaches the gate. **Dorsal perturbation cannot change `c_LTM` or `g`.**
This is the structural null pinned by **T8** — and it is a *wiring* claim, not an
empirical one.

### 2.3 `ltm_encoder_state` — ventral

`models/ltm_route.py:128–147`:

```python
mode = self.cfg.ltm_encoder_mode
if mode == "bigru_masked_mean":
    out, _ = self.encoder(emb)                              # (B, T, 2*H)   <- SLOT 0
    m = enc_mask.unsqueeze(-1).float()
    pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
else:  # unigru_last_hidden
    lengths = enc_mask.sum(1).clamp(min=1).cpu()
    packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True,
                                               enforce_sorted=False)
    _, h = self.encoder(packed)   # h: (num_layers, B, H)                   <- SLOT 1
    pooled = h[-1]                # (B, H)
if (self.training or apply_noise) and self.cfg.ventral_noise > 0:
    pooled = pooled + torch.randn_like(pooled) * self.cfg.ventral_noise
s_hat = self.to_semantic(pooled)
```

**FINDING — the single most dangerous live/historical divergence risk in this
workstream.** The ventral encoder reads **different GRU slots in the two modes**. Under
`bigru_masked_mean` the module output slot 1 (`h_n`) is *computed and discarded*;
perturbing it would have **no effect whatsoever**, and the experiment would silently
report a null ventral result that is an artefact of wiring.

Both GXLR checkpoints declare `ltm_encoder_mode = 'unigru_last_hidden'` with
`enc_layers = 1`, so `h` has shape **(1, B, 512)**, `pooled = h[-1]` is the whole of it,
and slot **1** is correct. **Verdict: CONFIRMED for these checkpoints only.** GXLR asserts
the mode at run time and refuses to lesion `ltm_encoder_state` under any other mode.

**Reachability.** `pooled → to_semantic → s_hat` (line 146), and `s_hat` feeds *both*

* the ventral premotor (`decode_from_s_hat`, `ltm_route.py:150–162`), and
* the lexical field (`dual_route.py:121–124` → `ltm_route.lexical_field:167–190`), whose
  `confidence = top2[:,0]` (line 185) is the gate's only input.

So **ventral perturbation reaches `c_LTM` and `g` by construction**, which is the
reachability claim pinned by **T9**. Conversely, the dorsal route-only decode
(`route="wm"`, `dual_route.py:165–171`) never calls `self.ltm`, so a ventral hook does not
even fire on that path: dorsal route-only output is structurally unchanged. Verdict:
CONFIRMED.

### 2.4 Cross-route structural summary

| perturbed site | `wm_premotor` | `ltm_premotor` | `s_hat` / `c_LTM` / `g` | route-only WM | route-only LTM |
|---|---|---|---|---|---|
| `wm_encoder_state` | changes | **unchanged** | **unchanged** | changes | **unchanged** |
| `ltm_encoder_state` | **unchanged** | changes | changes | **unchanged** | changes |

Every "unchanged" cell is a consequence of the call graph above, not an empirical
expectation. A violation is an implementation/wiring failure (T8, T9), never a result.

---

## 3. Decoding paths

### 3.1 Canonical forced-length AR — `gate_probe.py:163–194`

```python
batch = make_batch([list(f) for f in forms], vocab, device)
max_steps = max(len(f) for f in forms) + 1
for route in routes:
    dec_in = batch["enc_in"].new_full((len(forms), 1), vocab.bos_id)
    for _ in range(max_steps):
        lg = _route_step_logits(model, batch["enc_in"], batch["enc_mask"], dec_in, route)
        dec_in = torch.cat([dec_in, lg[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    for i, form in enumerate(forms):
        raw = dec_in[i, 1: 1 + len(form) + 1].tolist()
        seq = []
        for tok in raw:
            if tok == vocab.eos_id: break
            seq.append(tok)
        preds[route].append(seq)
```

**Verdict: CONFIRMED, REUSED UNMODIFIED.** Greedy to the batch maximum; each item
truncated to its own gold length + 1; cut at first EOS. This is the convention behind
every historical `rep_canonical_*` number. Note `max_steps` is a **batch** property, so
batch composition affects how many steps are run — but the readout window is per-item
(`1 : 1+len(form)+1`), so the emitted sequence is batch-invariant.

**Consequence for the lesion.** The encoder is re-run at every one of `max_steps` steps
(`_route_step_logits` → `route_logits` → `forward` → `wm.encode`/`ltm.encode`). The hook
therefore fires `O(max_steps × routes)` times per batch. `per_item_frozen` requires every
one of those calls to see the *same* tensor — pinned by **T7**.

### 3.2 Genuine free-AR — `gate_probe.py:197–238`

Own encoder construction (lines 216–221: `enc_in[k, :len(f)+1] = f + [eos]`), one global
cap, no use of the target length:

```python
max_steps: int = HISTORICAL_FREE_AR_MAX_STEPS      # imported, == 12
...
for _ in range(max_steps):
    lg = _route_step_logits(...)
    dec = torch.cat([dec, lg[:, -1, :].argmax(-1, keepdim=True)], dim=1)
    if bool((dec == vocab.eos_id).any(dim=1).all()):
        break
...
seq = dec[k, 1:].tolist()
if vocab.eos_id in seq: seq = seq[:seq.index(vocab.eos_id)]
```

**Verdict: CONFIRMED, REUSED UNMODIFIED.** The cap is *imported* from
`train_joint_scratch.FREE_AR_MAX_STEPS` (`gate_probe.py:36–38`), verified `== 12` at
`train_joint_scratch.py:194`. It is not restated, so it cannot drift.

Exact max-step / EOS semantics, stated precisely because the lesion will push items into
these regimes:

* the loop breaks early only when **every** row has emitted EOS at least once;
* a row that never emits EOS within 12 steps yields all 12 tokens — **non-termination is
  scored as an error**, which the forced-length convention structurally cannot see;
* over-generation past the gold length is likewise visible only here.

This is precisely why CENTRAL requires the two conventions to be classified
independently, and why a CANONICAL/FREE_AR disagreement must stay visible
(contract §7).

### 3.3 Route-only WM and LTM — `gate_probe.py:148–160`, `dual_route.py:157–181`

```python
if route in ("wm", "ltm"):
    return model.route_logits(enc_in, enc_mask, dec_in, route=route,
                              collect=False, apply_noise=False)["logits"]
```

`route="wm"` → `motor(self.wm(...)["premotor"])`; `route="ltm"` →
`motor(self.ltm(...)["premotor"])`. **Verdict: CONFIRMED.** Both bypass the gate entirely
and are bit-identical to the `g=0` / `g=1` limits. `apply_noise=False` is passed
explicitly on every path, so the routes' own training-time noise stays off.

---

## 4. Fusion paths

### 4.1 NATIVE — `models/gating.py:44–57`

```python
conf = field["confidence"].view(B, 1, 1)
g = torch.sigmoid(self.cfg.alpha * (conf - self.cfg.gate_threshold))
return g.expand(B, S, 1)
...
premotor = g * ltm + (1.0 - g) * wm
```

**Verdict: CONFIRMED, UNTOUCHED.** Gate orientation: **`g → 1` weights the VENTRAL (LTM)
route**, `1−g` the dorsal. `g` is word-level — one scalar per item, `expand`-broadcast
across decoder steps, so a single BOS-only forward captures the value used at every step
(`capture_gate_field`, `gate_probe.py:114–143`). `alpha = 2.0`, `gate_threshold = 0.7`
come from the checkpoint and are **not modified**.

### 4.2 FIXED05 — `gate_probe.py:71–79`

```python
def fixed_mix_logits(out, g=FIXED_G):        # FIXED_G = 0.5, line 45
    return g * out["ltm_logits"] + (1.0 - g) * out["wm_logits"]
```

**Verdict: CONFIRMED, REUSED VERBATIM — NOT REIMPLEMENTED.** Exactness requires the
motor map to be affine and the blend weights to sum to 1. Both verified live:
`models/motor.py:14–21` is a single `nn.Linear`
(`self.proj = nn.Linear(premotor_dim, vocab_size)`), and `0.5 + 0.5 = 1`. Hence

```
motor(g·ltm + (1−g)·wm) == g·motor(ltm) + (1−g)·motor(wm)
```

identically (the bias term is `g·b + (1−g)·b = b`). This is the already-validated GATING
algebra, independently cross-checked in that workstream by the `forced_gate` hook
(`gate_probe.py:82–109`) and by
`paper_programme/gating_route_diagnostics/figure_source_data/gate_algebra_verification.py`.
GXLR imports it and adds nothing.

### 4.3 Shared motor / readout

`dual_route.py:145–151` emits `logits`, `wm_logits`, `ltm_logits` from **one** `self.motor`
instance applied to three premotor tensors. **Verdict: CONFIRMED** — a single shared
readout, no route-specific head.

### 4.4 Do NATIVE and FIXED05 differ *only* at the intended intervention?

`gate_probe.py:155–160`:

```python
out = model.route_logits(enc_in, enc_mask, dec_in, route="full",
                         collect=False, apply_noise=False)
return out["logits"] if route == "full" else fixed_mix_logits(out, FIXED_G)
```

**Verdict: CONFIRMED.** Both conditions are computed from **the same `out` dict of the
same `route="full"` forward**, differing only in which blend is read. Therefore, at any
given decoder prefix, NATIVE and FIXED05 necessarily consume the *same* encoder states —
hence the *same* lesion tensor — and differ at exactly one point: `g` vs `0.5`.

**The one real hazard, and why it is not a defect.** In `ar_decode_forced_length` the
`for route in routes` loop is **outer**, so NATIVE and FIXED05 are decoded in *separate*
passes with *independent* `dec_in` prefixes. After the first behavioural divergence their
trajectories differ, and each subsequent forward is at a different prefix. CENTRAL §8.3
explicitly allows this. It is safe here because the perturbed site is the **encoder**,
whose inputs are `(enc_in, enc_mask)` only — invariant to the decoder prefix. The lesion
tensor is therefore identical across the two passes by construction, not by caching.
Pinned by **T6**.

---

## 5. Nested-scaling numerics

Frozen deterministic tolerance for **T5: exactly 0.0 (bitwise equality)**, justified
rather than assumed.

`λ ∈ {0.25, 0.50, 1.00}` are exact binary powers of two. Multiplication by a power of two
is exact in IEEE-754 binary32/64 (absent overflow and subnormal underflow), and scaling
commutes with round-to-nearest. With `amp(λ) = float32(λ · SD)` computed in float64 and
rounded once:

```
amp(0.50) == 2·amp(0.25)          bitwise
amp(1.00) == 2·amp(0.50)          bitwise
eta(λ) = epsilon ⊙ amp(λ)   ⇒   eta(0.50) == 2·eta(0.25) elementwise, bitwise
```

Verified empirically under the live interpreter over 1 000 float32 draws with a
representative `SD = 0.30000001192092896`:

```
amp exact doubling:   True True True
eta bitwise nesting:  True True True
```

SD semantics were **not** altered to obtain this; the equality falls out of the frozen
severity grid. Should CENTRAL ever add a non-dyadic severity, the tolerance must be
re-derived — the test asserts bitwise and will fail loudly rather than drift.

---

## 6. W3 / W4 exact identities

Resolved by Git object inspection of the frozen GATING manifest (blob `e5f0bb4f…`,
identical at `f43ccd09` and `993f2e73`), **not** by filename guessing.

**The manifest encodes a reconstruction recipe, not a monolithic checkpoint.** Each
POST_REPAIR witness is `source checkpoint + a derived final-layer head`, applied by
`run_gate_route_audit.build_state:150–162`:

```python
h = torch.load(head_path, map_location="cpu", weights_only=False)
if set(h["state"]) != {"2.weight", "2.bias"}:
    raise RuntimeError("derived head must carry exactly the final-layer parameters")
model = _isolated_model(tr, "p_last_hinge", h["state"])
```

That semantics is preserved exactly; no monolithic path is fabricated.

### GXLR_W3_REP = `V6_seed19_u3825` (manifest `state_id = W3_REP`)

| field | value |
|---|---|
| state / witness | `W3_REP` / `W3` / `V6_seed19_u3825`, arm `POST_REPAIR_WITNESS` |
| seed, source_u | 19, 3825 — `prospective_first_hit` |
| base artifact | `probe_v6_completion_20260912/sources/s19_step_10625850.pt` |
| base sha256 | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` |
| base bytes | 31 174 770 |
| applied head | `archives/prospective_rn_detector_v6_completion_20260912/seed19_u3825/arm_a/head_first_c0.pt` |
| head sha256 | `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` |
| head bytes | 1 849 495 |
| localizer | `_isolated_model(tr, "p_last_hinge", head["state"])`, keys exactly `{2.weight, 2.bias}` |
| phase8 code commit | `78f5505` |
| provenance | `VERIFIED_sha256_matches_phase8_record` |
| composite state id | `sha256("gxlr-state-v1|" + base_sha + "|" + head_sha)` |

### GXLR_W4_REP = `V6_seed20_u3040` (manifest `state_id = W4_REP`)

| field | value |
|---|---|
| state / witness | `W4_REP` / `W4` / `V6_seed20_u3040`, arm `POST_REPAIR_WITNESS` |
| seed, source_u | 20, 3040 — `prospective_first_hit` |
| base artifact | `probe_v6_completion_20260912/sources/s20_step_08445120.pt` |
| base sha256 | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` |
| base bytes | 31 174 386 |
| applied head | `archives/prospective_rn_detector_v6_completion_20260912/seed20_u3040/arm_a/head_first_c0.pt` |
| head sha256 | `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03` |
| head bytes | 1 849 495 |
| localizer | `_isolated_model(tr, "p_last_hinge", head["state"])`, keys exactly `{2.weight, 2.bias}` |
| phase8 code commit | `78f5505` |
| provenance | `VERIFIED_sha256_matches_phase8_record` |
| composite state id | `sha256("gxlr-state-v1|" + base_sha + "|" + head_sha)` |

**Live re-verification (this session).** All four artifacts resolve at paths relative to
the repository parent `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3`, and their computed
SHA256 matches the frozen manifest byte for byte:

```
a5f21de9…aad76c  probe_v6_completion_20260912/sources/s19_step_10625850.pt        31174770
8865ba95…7afcfc  archives/…/seed19_u3825/arm_a/head_first_c0.pt                    1849495
0657f410…dc79f3  probe_v6_completion_20260912/sources/s20_step_08445120.pt        31174386
724ed4c6…b97203  archives/…/seed20_u3040/arm_a/head_first_c0.pt                    1849495
```

**No checkpoint substitution.** `W1`/`W2` (retrospective V5 witnesses) and all four
`*_SRC` (pre-repair) rows are carried in the proposed manifest as `IN_SCOPE=0` for
identity/traceability only and are never executed by this workstream.

### The intact authoritative null (T2 reference)

Read from the frozen GATING item-level shards
(`paper_programme/gating_route_diagnostics/figure_source_data/item_level_W{3,4}_REP.tsv`,
29 571 rows each, `item_order_sha256 = ab2b193f98eee7fb…`, identical across all eight
states):

| state | convention | NATIVE vs FIXED05 discordant predictions | discordant exact-match | full err | wm err | ltm err |
|---|---|---|---|---|---|---|
| W3_REP | canonical | **0** | **0** | 0 | 0 | 3 193 |
| W3_REP | free-AR | **0** | **0** | 0 | 0 | 3 193 |
| W4_REP | canonical | **0** | **0** | 0 | 1 | 3 662 |
| W4_REP | free-AR | **0** | **0** | 0 | 1 | 3 662 |

These reproduce the manifest's frozen columns exactly
(`frozen_rep_canonical_ltm_errors` 3 193 / 3 662; `frozen_rep_canonical_wm_errors` 0 / 1).
**T2 asserts severity 0 reproduces these on both states and both conventions.**

---

## 7. New GXLR code (audited on the GXLR branch)

### 7.1 `gate_x_lesion/noise.py` — base `epsilon`

RNG identity, canonically serialised and digested:

```
key = sha256(b"GXLR-eps-v1|" + json.dumps(
        {"state_sha256":…, "route":…, "lesion_seed":…, "item_id":…},
        sort_keys=True, separators=(",",":"), ensure_ascii=True).encode("utf-8"))
```

**`lambda` and the fusion condition are absent from the payload by construction** — the
payload is built from a fixed 4-key schema and the function that builds it accepts no
other arguments (**T4**). `item_id` is the canonical word string, which is unique across
the 29 571-item population (verified: zero duplicates) and stable under any re-ordering;
`item_order_sha256` is recorded separately for provenance.

Uniform values come from a SHA-256 counter-mode byte stream decoded as big-endian
`uint32`, mapped `x = u/2^32 ∈ [0,1)`, `epsilon = 2x − 1 ∈ [−1,1)` — matching
`torch.rand`'s half-open support and the historical `rand·2−1` exactly in distribution,
while being independent of process, platform and torch version. **`hash()` is never
used.**

### 7.2 `gate_x_lesion/noise.py` — nested scaling

`eta(λ) = epsilon ⊙ float32(λ · SD)`, with `epsilon` produced once per
`(state, route, seed, item)` and cached. No redraw across λ, across NATIVE/FIXED05, or
across decoder steps. Operand order matches the historical `draw_noise` path (§1.3).

### 7.3 `gate_x_lesion/hooks.py` — the lesion context

Registers one forward hook on the target module, replaces exactly the audited slot,
verifies the slot is a plain tensor, asserts `ltm_encoder_mode` for the ventral target,
and removes the handle in `finally`. Item ids are **bound explicitly** for the batch
before the decode; the hook never infers identity from tensor content (defect H-1).

### 7.4 `gate_x_lesion/sd.py` — intact SD

Reproduces the historical recipe exactly: `deterministic_sample(range(N), 2048, 7)`,
`batch_size 256`, teacher-forced `model(enc_in, enc_mask, dec_in)`, hook collects
`out[slot].detach().float().flatten()`, `torch.cat`, `float(a.std())`.

### 7.5 `gate_x_lesion/outcomes.py` — classification

Pure functions over records; no model, no torch. Implements the frozen rules of contract
§7, including the `OUTCOME_F_HETEROGENEOUS` veto and independent CANONICAL / FREE_AR
classification. Unit-tested on synthetic records only (**T12**).

---

## 8. Overall verdict

| item | verdict |
|---|---|
| lesion hook return semantics | CONFIRMED; slot-1 `h_n`, out-of-place, tuple rebuilt |
| `PackedSequence` hazard | CONFIRMED present on this architecture; guarded |
| historical RNG | REJECTED for this contract (H-1..H-3); replaced, semantics preserved |
| `wm_encoder_state` site | CONFIRMED (1, B, 128), reaches only the dorsal premotor |
| `ltm_encoder_state` site | CONFIRMED (1, B, 512) **only under `unigru_last_hidden`**; asserted at run time |
| structural route isolation | CONFIRMED from the call graph |
| canonical forced-length AR | CONFIRMED, reused unmodified |
| free-AR + cap 12 | CONFIRMED, cap imported not restated |
| route-only WM / LTM | CONFIRMED, gate bypassed |
| NATIVE fusion | CONFIRMED, untouched (`alpha 2.0`, `threshold 0.7`) |
| FIXED05 | CONFIRMED, reused verbatim; exactness re-verified (motor is one `nn.Linear`) |
| NATIVE vs FIXED05 differ only at the intervention | CONFIRMED (shared `out`) |
| SD six fields | RESOLVED from executable historical code |
| W3/W4 identities | RESOLVED; recipe semantics preserved; hashes re-verified live |
| nested-scaling tolerance | 0.0 bitwise, justified |

**NO SCIENTIFIC LESION EXECUTION accompanies this audit.**
