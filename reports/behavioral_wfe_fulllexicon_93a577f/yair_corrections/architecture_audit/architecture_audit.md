# Executable architecture and premotor audit

**Evidence: the committed code in `models/` plus the frozen checkpoint's own
configuration and parameter shapes.** No stale Markdown was consulted; the two
sources are cross-checked against each other. The checkpoint
(`archives/fulllexicon_93a577f/extracted/fulllexicon_final_bundle_93a577f/selected_checkpoints/seed_19_epoch_0155.pt`) was opened only to read `cfg_*` dicts and
`model_state_dict` **shapes** — no model was constructed, no forward pass run, no
token generated, nothing modified.

Regenerate with `python scripts/audit_architecture.py`; machine-readable twin:
`architecture_audit.json`.

---

## 1. Route classes

| piece | class | file |
|---|---|---|
| dorsal route | `WMRecurrent` | `models/wm_route.py` |
| ventral route | `LTMLexicon` | `models/ltm_route.py` |
| gate | `Gate` (via `build_gate`) | `models/gating.py` |
| shared readout | `MotorCortex` | `models/motor.py` |
| wiring | `DualRouteModel` | `models/dual_route.py` |

Both routes read the **same** `nn.Embedding` (`phon_embed`,
[42, 64]), constructed once in `DualRouteModel` and passed
into both. It is a shared phonetic feature table, not a lexicon.

## 2. Encoders and decoders, as trained

| module | construction | trained shape |
|---|---|---|
| `wm.encoder` | `nn.GRU(emb, 128, batch_first=True)` | `[384, 64]` |
| `wm.decoder` | `nn.GRU(emb, 128, batch_first=True)` | `[384, 64]` |
| `ltm.encoder` | `nn.GRU(emb, 128, num_layers=1, batch_first=True, bidirectional=False)` | `[384, 64]` |
| `ltm.decoder` | `nn.GRU(emb, 128, batch_first=True)` | `[384, 64]` |

`ltm_encoder_mode = "unigru_last_hidden"`. Reverse-direction GRU parameters
present in the checkpoint: **none** — the
LTM encoder is confirmed **unidirectional**, so the historical
`bigru_masked_mean` path is not the one in these weights.

## 3. The `unigru_last_hidden` path (`ltm_route.py:135-147`)

```
emb     = phon_embed(enc_in)                                     # (B, T, E)
lengths = enc_mask.sum(1).clamp(min=1).cpu()                     # includes EOS
packed  = pack_padded_sequence(emb, lengths, batch_first=True,
                               enforce_sorted=False)
_, h    = self.encoder(packed)                                   # (num_layers, B, H)
pooled  = h[-1]                                                  # (B, 128)
s_hat   = self.to_semantic(pooled)                               # (B, 300)
```

`pack_padded_sequence` means `h[-1]` is the hidden state at each item's **last
real token**, so padding never contributes and there is no batch-composition
artifact. Ventral noise would be added to `pooled` before `to_semantic`, but
`ventral_noise = 0.0`, so in these checkpoints the path is
deterministic.

## 4. `s_hat`: dimensions and use

`to_semantic = Sequential(Linear(128, 128) -> GELU -> Linear(128, 300))`,
i.e. **128 -> 128 -> 300**. The GELU and the dimension lift make it
non-invertible.

Raw `s_hat` (300-d) is used in exactly two places, and they are separate:

1. **Decoder initialisation** — `h0 = tanh(sem_to_h0(s_hat))`,
   `sem_to_h0 = Linear(300, 128)` i.e. 300 -> 128, then
   `unsqueeze(0)` to `(1, B, 128)`. This uses **raw** `s_hat`.
2. **Lexical field** — `q = F.normalize(s_hat, dim=-1)` is a **separate tensor**;
   it does not modify `s_hat`. Only `q` touches the bank.

`s_hat` is also the target of the alignment loss against raw GloVe. It is a
phonology-derived, GloVe-aligned representation used to initialise the LTM
decoder — not a comprehension signal.

## 5. Semantic bank and lexical confidence

The bank is a **frozen, non-persistent buffer** of the training lexicon's GloVe
vectors, L2-normalised at registration (`set_semantic_bank`). It has no
gradient and is never trained.

```
q          = F.normalize(s_hat, dim=-1)
sims       = q @ semantic_bank.t()            # (B, n_words) cosine similarities
top2       = topk(sims, 2)
confidence = top2[:, 0]                       # top-1 cosine  -> the gate input
margin     = top2[:, 0] - top2[:, -1]
density    = sigmoid(20.0 * (sims - (confidence - 0.1))).sum(1)
```

**No bank vector is ever passed to the decoder.** The bank contributes exactly
one scalar per item — `confidence` — and that scalar's only consumer is the gate.
`margin` and `density` are computed and returned but do not enter `gate_value`.

## 6. Gate: formula, direction, and constancy

```
g = sigmoid(alpha * (confidence - gate_threshold))
  with alpha = 2.0, gate_threshold = 0.7
```

**Direction: `g -> 1` means trust LTM (ventral); `g -> 0` means trust WM
(dorsal).** Read off the mixing line itself, `premotor = g * ltm + (1 - g) * wm`.

**The gate is constant across the word.** `gate_value` computes
`conf.view(B, 1, 1)` and returns `g.expand(B, S, 1)` — one scalar per item,
broadcast over all `S` decoder positions. It does not vary with phoneme
position, with WM state, or with anything that happens during decoding. It has
**zero learnable parameters** and **receives no WM input**: `wm` enters
`gate_value` only to supply the shape `B, S`.

When the field is absent the gate falls back to a constant 0.5.

## 7. What FULL actually mixes

The mixed tensor is the **premotor state**, not the logits:

```
premotor_FULL = g * premotor_LTM + (1 - g) * premotor_WM     # (B, S, 128)
logits_FULL   = motor(premotor_FULL)                          # (B, S, 42)
```

Because `motor` is affine and the weights sum to one, at a **fixed prefix** this
is algebraically identical to mixing the logits:

`W(g a + (1-g) b) + c = g(Wa + c) + (1-g)(Wb + c)`

so `logits_FULL = g * logits_LTM + (1 - g) * logits_WM` **at the same prefix**.
The identity does **not** extend to independently generated autoregressive
trajectories, because there each route conditions on its own emitted prefix.

## 8. Premotor projections

| module | construction | trained shape | bias | activation on output |
|---|---|---|---|---|
| `wm.to_premotor` | `nn.Linear(128, 128)` | `[128, 128]` | True | none |
| `ltm.dec_to_premotor` | `nn.Linear(128, 128)` | `[128, 128]` | True | none |
| `motor.proj` (shared) | `nn.Linear(128, 42)` | `[42, 128]` | True | none (raw logits) |

AST verification that each projection is returned **bare**, with no activation
wrapped around it:

- `WMRecurrent.decode_from_state` returns `self.to_premotor(...)` directly:
  **True**
- `LTMLexicon.decode_from_s_hat` returns `self.dec_to_premotor(...)` directly:
  **True**
- `MotorCortex.forward` returns `self.proj(...)` directly:
  **True**

Activation modules constructed anywhere in `wm_route.py`:
**none**. In `ltm_route.py`:
**['GELU']** — and that GELU sits inside
`to_semantic`, on the encoder side, not on either premotor path.

## 9. Shared motor readout

There is exactly **one** `MotorCortex`, owned by `DualRouteModel` and applied
three times per forward pass — to the WM premotor, the LTM premotor and the
gated mixture — producing `wm_logits`, `ltm_logits` and `logits`. The three
route outputs are therefore **not** three separate readouts; they are one
128 -> 42 affine map evaluated at three points of the same space.

## 10. Tick-by-tick autoregressive feedback

Evaluation decoding (`scripts/external_eval.autoregressive_decode_batch`) starts
from `dec_input = [BOS]` and, at each tick, runs the **whole** model over the
current prefix, takes `logits[:, -1, :]`, applies `argmax`, and appends that
token to `dec_input`. Feedback is the emitted token id only — no hidden state is
carried between ticks, and the encoder is re-run each tick on the unchanged
input. Decoding is deterministic (argmax, no sampling, no temperature).

Each route generates its **own** trajectory when evaluated alone, so FULL, WM and
LTM predictions are not position-comparable unless read under a single shared
prefix.

## 11. EOS and forced length

`enc_in = form + [EOS]`, `dec_in = [BOS] + form`, `dec_tgt = form + [EOS]`.
The readout window is `dec_input[i, 1:L+1]` — exactly `L` tokens for a target of
length `L`. Consequences, both structural:

- the prediction can **never** exceed `L` tokens, so terminal insertions past the
  target horizon are unobservable (forced-length readout);
- a boundary EOS would occupy window index `L`, one past the end of the slice, so
  **on-time and late EOS are structurally unobservable and every observed EOS is
  premature**. `EOS_NOT_OBSERVED` conflates correct stopping with never stopping.

---

# Explicit answers

**Is each premotor projection linear?**
**Yes.** `wm.to_premotor` is `nn.Linear(128, 128)` and
`ltm.dec_to_premotor` is `nn.Linear(128, 128)`; both carry a bias
and both are returned bare, with no activation applied to their output (verified
by AST, not by reading a comment). The only non-linearities in either route are
inside the GRU cells upstream, the `tanh` on `h0`, and the GELU inside
`to_semantic` — none of which sits on a premotor path.

**Why is a common premotor space needed in this implementation?**
Two independent reasons, both structural:

1. **The gate performs a convex combination of the two route states.**
   `g * ltm + (1 - g) * wm` is only meaningful if both operands live in the same
   vector space with the same basis. Mixing coordinates that mean different
   things in each route would be arithmetic without semantics.
2. **There is exactly one readout.** A single 128 -> 42 affine map must
   decode phoneme identity from the WM premotor, the LTM premotor **and** every
   intermediate mixture. That forces both routes to encode "which phoneme now" in
   the *same* coordinates — which is the architectural commitment the shared
   motor cortex is there to make.

The premotor projections are the only per-route learnable adapters into that
shared space; they are where each route's recurrent state is rotated into the
common basis.

**Is mixing before or after phoneme logits?**
**Before.** The gate mixes premotor states and the shared motor projection is
applied afterwards. Because that projection is affine and the mixing weights sum
to one, the two orderings are algebraically identical **at a fixed prefix**, so
`logits_FULL = g * logits_LTM + (1 - g) * logits_WM` holds there. It does **not**
hold across independently generated autoregressive trajectories.

**What would break if the premotor projection were removed?**

- *Not shapes, by coincidence.* Here `wm.hidden = 128`,
  `ltm.dec_hidden = 128` and `premotor_dim = 128` all happen to be
  equal, so feeding the two GRU hidden states straight into the gate and the
  motor projection would still run. The failure is scientific, not a crash.
- **The two recurrent state spaces would be forced into alignment.** The
  convex mixture would be taken directly between a WM GRU hidden state and an LTM
  GRU hidden state, and the shared readout would have to decode both. Nothing
  would any longer be free to learn a per-route change of basis, so the two GRUs
  would have to converge on a common coordinate system as a side effect of the
  readout gradient.
- **The routes lose their only independent adapter.** Every per-route degree of
  freedom for "how do I express this state to the shared channel" disappears; the
  routes could only differ in their recurrent dynamics, not in how they address
  the motor cortex.
- **Configuration coupling becomes hard.** Any configuration with
  `wm.hidden != ltm.dec_hidden`, or either differing from `premotor_dim`, becomes
  shape-invalid — the mixture and the shared readout would not typecheck. The
  projections are what currently decouple the two recurrent widths from the
  shared channel width.

**No architecture was modified by this audit.**
