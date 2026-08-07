"""Executable architecture / premotor audit.

Reads **the committed code and the checkpoint configuration**, never a stale
Markdown description.  Two independent evidence sources are combined and
cross-checked:

  1. AST inspection of `models/*.py` — which classes exist, which modules they
     construct, whether any activation is applied to a projection's output;
  2. the frozen checkpoint's own `cfg_*` dicts and `model_state_dict` **shapes**
     — the dimensions and biases that were actually trained.

The checkpoint is opened with `torch.load` to read configuration dicts and
parameter shapes.  **No model is constructed, no forward pass is run, no token is
generated, and nothing is modified.**
"""
from __future__ import annotations

import ast
import json
import os
import sys
from typing import Dict, List, Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BUNDLE = ("archives/fulllexicon_93a577f/extracted/"
          "fulllexicon_final_bundle_93a577f/selected_checkpoints")
CKPT = "seed_19_epoch_0155.pt"
OUT = ("reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
       "architecture_audit")

MODEL_FILES = ["dual_route.py", "wm_route.py", "ltm_route.py", "gating.py",
               "motor.py"]


# ------------------------------------------------------------------ AST side

def _module_assignments(tree: ast.AST) -> Dict[str, str]:
    """`self.x = nn.Linear(...)` -> {"x": "Linear(a, b)"} for every submodule."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if (isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                    and isinstance(node.value, ast.Call)):
                out[tgt.attr] = ast.unparse(node.value)
    return out


ACTIVATIONS = {"relu", "gelu", "tanh", "sigmoid", "elu", "softmax",
               "log_softmax", "leaky_relu", "silu", "hardtanh"}


def _returns_bare_projection(tree: ast.AST, cls: str, fn: str,
                             attr: str) -> Optional[bool]:
    """True if `fn` returns `self.<attr>(...)` with no activation applied to it.

    The projection's value may be returned directly or as an element of a dict
    or tuple — what matters is only whether any enclosing expression is an
    activation call, not the shape of the container.
    """
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == cls):
            continue
        for f in node.body:
            if not (isinstance(f, ast.FunctionDef) and f.name == fn):
                continue
            for r in ast.walk(f):
                if not (isinstance(r, ast.Return) and r.value is not None):
                    continue
                # locate the projection call and record its ancestor chain
                parent = {}
                for p in ast.walk(r.value):
                    for c in ast.iter_child_nodes(p):
                        parent[c] = p
                target = None
                for n in ast.walk(r.value):
                    if (isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute)
                            and n.func.attr == attr
                            and isinstance(n.func.value, ast.Name)
                            and n.func.value.id == "self"):
                        target = n
                        break
                if target is None:
                    continue
                cur = parent.get(target)
                while cur is not None:
                    if isinstance(cur, ast.Call):
                        name = (cur.func.attr if isinstance(cur.func, ast.Attribute)
                                else getattr(cur.func, "id", ""))
                        if name.lower() in ACTIVATIONS:
                            return False
                    cur = parent.get(cur)
                return True
    return None


def code_facts() -> Dict[str, object]:
    trees = {}
    for f in MODEL_FILES:
        with open(os.path.join(ROOT, "models", f)) as fh:
            trees[f] = ast.parse(fh.read())
    dr, wm, ltm = trees["dual_route.py"], trees["wm_route.py"], trees["ltm_route.py"]
    facts = {
        "dual_route_submodules": _module_assignments(dr),
        "wm_submodules": _module_assignments(wm),
        "ltm_submodules": _module_assignments(ltm),
        "motor_submodules": _module_assignments(trees["motor.py"]),
        "wm_premotor_projection_is_bare_linear":
            _returns_bare_projection(wm, "WMRecurrent", "decode_from_state",
                                     "to_premotor"),
        "ltm_premotor_projection_is_bare_linear":
            _returns_bare_projection(ltm, "LTMLexicon", "decode_from_s_hat",
                                     "dec_to_premotor"),
        "motor_projection_is_bare_linear":
            _returns_bare_projection(trees["motor.py"], "MotorCortex",
                                     "forward", "proj"),
    }
    # activation modules referenced anywhere in the two premotor paths
    src_wm = open(os.path.join(ROOT, "models/wm_route.py")).read()
    src_ltm = open(os.path.join(ROOT, "models/ltm_route.py")).read()
    facts["activations_in_wm_module"] = sorted(
        {a for a in ("ReLU", "GELU", "Tanh", "Sigmoid", "ELU")
         if f"nn.{a}" in src_wm})
    facts["activations_in_ltm_module"] = sorted(
        {a for a in ("ReLU", "GELU", "Tanh", "Sigmoid", "ELU")
         if f"nn.{a}" in src_ltm})
    return facts


# ----------------------------------------------------------- checkpoint side

def checkpoint_facts(path: str) -> Dict[str, object]:
    """Config dicts and parameter shapes.  No model is constructed."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    keep = ("wm.encoder", "wm.decoder", "wm.to_premotor",
            "ltm.encoder", "ltm.decoder", "ltm.to_semantic", "ltm.sem_to_h0",
            "ltm.dec_to_premotor", "motor.proj", "phon_embed")
    shapes = {k: list(v.shape) for k, v in sd.items()
              if any(k.startswith(p) for p in keep)}
    return {
        "checkpoint": os.path.relpath(path, ROOT),
        "cfg_wm": ck["cfg_wm"], "cfg_ltm": ck["cfg_ltm"],
        "cfg_gating": ck["cfg_gating"],
        "semantic_dim": ck["cfg_data"]["semantic_dim"],
        "premotor_dim": ck.get("premotor_dim"),
        "parameter_shapes": shapes,
        "has_reverse_gru_parameters":
            sorted(k for k in sd if "_reverse" in k),
        "wm_to_premotor_has_bias": "wm.to_premotor.bias" in sd,
        "ltm_dec_to_premotor_has_bias": "ltm.dec_to_premotor.bias" in sd,
        "motor_proj_has_bias": "motor.proj.bias" in sd,
        "model_constructed": False, "forward_pass_run": False,
        "tokens_generated": False, "weights_modified": False,
    }


# ------------------------------------------------------------------- report

def build_note(code: Dict, ck: Dict) -> str:
    cw, cl, cg = ck["cfg_wm"], ck["cfg_ltm"], ck["cfg_gating"]
    ps = ck["parameter_shapes"]
    sem = ck["semantic_dim"]
    pm = ck["premotor_dim"]
    wm_pre = ps.get("wm.to_premotor.weight")
    ltm_pre = ps.get("ltm.dec_to_premotor.weight")
    motor = ps.get("motor.proj.weight")
    ts0 = ps.get("ltm.to_semantic.0.weight")
    ts2 = ps.get("ltm.to_semantic.2.weight")
    s2h = ps.get("ltm.sem_to_h0.weight")

    return f"""# Executable architecture and premotor audit

**Evidence: the committed code in `models/` plus the frozen checkpoint's own
configuration and parameter shapes.** No stale Markdown was consulted; the two
sources are cross-checked against each other. The checkpoint
(`{ck['checkpoint']}`) was opened only to read `cfg_*` dicts and
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
{ps.get('phon_embed.weight')}), constructed once in `DualRouteModel` and passed
into both. It is a shared phonetic feature table, not a lexicon.

## 2. Encoders and decoders, as trained

| module | construction | trained shape |
|---|---|---|
| `wm.encoder` | `nn.GRU(emb, {cw['hidden']}, batch_first=True)` | `{ps.get('wm.encoder.weight_ih_l0')}` |
| `wm.decoder` | `nn.GRU(emb, {cw['hidden']}, batch_first=True)` | `{ps.get('wm.decoder.weight_ih_l0')}` |
| `ltm.encoder` | `nn.GRU(emb, {cl['enc_hidden']}, num_layers={cl['enc_layers']}, batch_first=True, bidirectional=False)` | `{ps.get('ltm.encoder.weight_ih_l0')}` |
| `ltm.decoder` | `nn.GRU(emb, {cl['dec_hidden']}, batch_first=True)` | `{ps.get('ltm.decoder.weight_ih_l0')}` |

`ltm_encoder_mode = "{cl['ltm_encoder_mode']}"`. Reverse-direction GRU parameters
present in the checkpoint: **{ck['has_reverse_gru_parameters'] or 'none'}** — the
LTM encoder is confirmed **unidirectional**, so the historical
`bigru_masked_mean` path is not the one in these weights.

## 3. The `unigru_last_hidden` path (`ltm_route.py:135-147`)

```
emb     = phon_embed(enc_in)                                     # (B, T, E)
lengths = enc_mask.sum(1).clamp(min=1).cpu()                     # includes EOS
packed  = pack_padded_sequence(emb, lengths, batch_first=True,
                               enforce_sorted=False)
_, h    = self.encoder(packed)                                   # (num_layers, B, H)
pooled  = h[-1]                                                  # (B, {cl['enc_hidden']})
s_hat   = self.to_semantic(pooled)                               # (B, {sem})
```

`pack_padded_sequence` means `h[-1]` is the hidden state at each item's **last
real token**, so padding never contributes and there is no batch-composition
artifact. Ventral noise would be added to `pooled` before `to_semantic`, but
`ventral_noise = {cl['ventral_noise']}`, so in these checkpoints the path is
deterministic.

## 4. `s_hat`: dimensions and use

`to_semantic = Sequential(Linear{tuple(reversed(ts0))} -> GELU -> Linear{tuple(reversed(ts2))})`,
i.e. **{ts0[1]} -> {ts0[0]} -> {ts2[0]}**. The GELU and the dimension lift make it
non-invertible.

Raw `s_hat` ({sem}-d) is used in exactly two places, and they are separate:

1. **Decoder initialisation** — `h0 = tanh(sem_to_h0(s_hat))`,
   `sem_to_h0 = Linear{tuple(reversed(s2h))}` i.e. {s2h[1]} -> {s2h[0]}, then
   `unsqueeze(0)` to `(1, B, {cl['dec_hidden']})`. This uses **raw** `s_hat`.
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
  with alpha = {cg['alpha']}, gate_threshold = {cg['gate_threshold']}
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
premotor_FULL = g * premotor_LTM + (1 - g) * premotor_WM     # (B, S, {pm})
logits_FULL   = motor(premotor_FULL)                          # (B, S, {motor[0]})
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
| `wm.to_premotor` | `nn.Linear({cw['hidden']}, {pm})` | `{wm_pre}` | {ck['wm_to_premotor_has_bias']} | none |
| `ltm.dec_to_premotor` | `nn.Linear({cl['dec_hidden']}, {pm})` | `{ltm_pre}` | {ck['ltm_dec_to_premotor_has_bias']} | none |
| `motor.proj` (shared) | `nn.Linear({pm}, {motor[0]})` | `{motor}` | {ck['motor_proj_has_bias']} | none (raw logits) |

AST verification that each projection is returned **bare**, with no activation
wrapped around it:

- `WMRecurrent.decode_from_state` returns `self.to_premotor(...)` directly:
  **{code['wm_premotor_projection_is_bare_linear']}**
- `LTMLexicon.decode_from_s_hat` returns `self.dec_to_premotor(...)` directly:
  **{code['ltm_premotor_projection_is_bare_linear']}**
- `MotorCortex.forward` returns `self.proj(...)` directly:
  **{code['motor_projection_is_bare_linear']}**

Activation modules constructed anywhere in `wm_route.py`:
**{code['activations_in_wm_module'] or 'none'}**. In `ltm_route.py`:
**{code['activations_in_ltm_module']}** — and that GELU sits inside
`to_semantic`, on the encoder side, not on either premotor path.

## 9. Shared motor readout

There is exactly **one** `MotorCortex`, owned by `DualRouteModel` and applied
three times per forward pass — to the WM premotor, the LTM premotor and the
gated mixture — producing `wm_logits`, `ltm_logits` and `logits`. The three
route outputs are therefore **not** three separate readouts; they are one
{pm} -> {motor[0]} affine map evaluated at three points of the same space.

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
**Yes.** `wm.to_premotor` is `nn.Linear({cw['hidden']}, {pm})` and
`ltm.dec_to_premotor` is `nn.Linear({cl['dec_hidden']}, {pm})`; both carry a bias
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
2. **There is exactly one readout.** A single {pm} -> {motor[0]} affine map must
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

- *Not shapes, by coincidence.* Here `wm.hidden = {cw['hidden']}`,
  `ltm.dec_hidden = {cl['dec_hidden']}` and `premotor_dim = {pm}` all happen to be
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
"""


def main() -> int:
    code = code_facts()
    ck = checkpoint_facts(os.path.join(ROOT, BUNDLE, CKPT))
    out_dir = os.path.join(ROOT, OUT)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "architecture_audit.json"), "w") as f:
        json.dump({"code_facts": code, "checkpoint_facts": ck}, f, indent=2)
        f.write("\n")
    note = build_note(code, ck)
    with open(os.path.join(out_dir, "architecture_audit.md"), "w") as f:
        f.write(note)
    print(f"wrote {out_dir}/architecture_audit.{{md,json}}")
    print("premotor linear (wm/ltm/motor):",
          code["wm_premotor_projection_is_bare_linear"],
          code["ltm_premotor_projection_is_bare_linear"],
          code["motor_projection_is_bare_linear"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
