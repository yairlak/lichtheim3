"""Standalone numerical verification of every claim in CODE_AUDIT_GATE.md §6.

These are properties of the CODE PATH, so a small randomly-initialised model is
the correct instrument: no checkpoint, no lexicon, no GloVe.  The same claims are
pinned as regression tests in tests/test_gate_route_diagnostics.py.

Run:  python3 paper_programme/gating_route_diagnostics/figure_source_data/gate_algebra_verification.py
"""
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import Config
from data.phonemes import build_vocab
from models.dual_route import DualRouteModel

ALPHA, TAU = 2.0, 0.7          # frozen in all four Phase-8 source checkpoints

torch.manual_seed(0)
cfg = Config()
cfg.gating.alpha, cfg.gating.gate_threshold = ALPHA, TAU
vocab = build_vocab()
model = DualRouteModel(cfg, vocab, premotor_dim=128)
model.eval()
model.set_semantic_bank(torch.randn(64, cfg.data.semantic_dim))

B, T, S = 7, 6, 5
enc_in = torch.randint(3, vocab.size, (B, T))
enc_mask = torch.ones(B, T, dtype=torch.bool)
enc_mask[0, 4:] = False                        # ragged, to exercise masking
dec_in = torch.randint(3, vocab.size, (B, S))

print(f"gate hyperparameters: alpha={ALPHA}, tau={TAU}  (cohort 93a577f setting)")
print(f"gate learnable parameters: {len(list(model.gate.parameters()))}\n")

with torch.no_grad():
    out = model(enc_in, enc_mask, dec_in)
    g = out["gate"]

    # CLAIM 1 — fusion on premotor == fusion on logits
    recomposed = g * out["ltm_logits"] + (1.0 - g) * out["wm_logits"]
    d1 = (recomposed - out["logits"]).abs().max().item()

    # CLAIM 2 — the isolated code paths ARE the g=0 / g=1 limits
    wm_only = model.route_logits(enc_in, enc_mask, dec_in, route="wm")["logits"]
    ltm_only = model.route_logits(enc_in, enc_mask, dec_in, route="ltm")["logits"]
    d2w = (wm_only - out["wm_logits"]).abs().max().item()
    d2l = (ltm_only - out["ltm_logits"]).abs().max().item()

    # CLAIM 3 — the forced-0.5 intervention is a pure post-hoc mix
    half = 0.5 * out["ltm_logits"] + 0.5 * out["wm_logits"]

    # CLAIM 4 — gate is word-level (constant across decoder steps)
    d4 = (g - g[:, :1, :]).abs().max().item()

    # CLAIM 5 — gate is blind to the dorsal route.
    # The perturbation must EXCLUDE phon_embed.weight: it is the one tensor the
    # two routes share, so perturbing it would move s_hat and hence the gate for
    # a reason that has nothing to do with dorsal competence.
    shared = {id(p) for p in model.wm.parameters()} & {id(p) for p in model.ltm.parameters()}
    names = {id(p): n for n, p in model.named_parameters()}
    g_ref, c_ref, wm_ref = g.clone(), out["field_confidence"].clone(), out["wm_logits"].clone()
    n_perturbed = 0
    for p in model.wm.parameters():
        if id(p) not in shared:
            p.add_(torch.randn_like(p) * 0.5)
            n_perturbed += 1
    out2 = model(enc_in, enc_mask, dec_in)
    d5 = (out2["gate"] - g_ref).abs().max().item()
    d5c = (out2["field_confidence"] - c_ref).abs().max().item()
    d5w = (out2["wm_logits"] - wm_ref).abs().max().item()

print(f"CLAIM 1  g*ltm_logits+(1-g)*wm_logits  vs  logits      max|d| = {d1:.3e}")
print(f"CLAIM 2a route_logits('wm')   vs  wm_logits            max|d| = {d2w:.3e}")
print(f"CLAIM 2b route_logits('ltm')  vs  ltm_logits           max|d| = {d2l:.3e}")
print(f"CLAIM 3  forced-0.5 mix computable post hoc: shape {tuple(half.shape)}")
print(f"CLAIM 4  gate constant across decoder steps            max|d| = {d4:.3e}")
print(f"CLAIM 5  shared parameter tensors: {[names[i] for i in shared]}")
print(f"         perturbed {n_perturbed} WM-exclusive tensors")
print(f"         gate                                          max|d| = {d5:.3e}")
print(f"         LTM confidence                                max|d| = {d5c:.3e}")
print(f"         wm_logits (sanity: must be large)             max|d| = {d5w:.3e}")

sig = lambda z: 1 / (1 + math.exp(-z))                          # noqa: E731
print(f"\nCLAIM 6  attainable g = sigmoid({ALPHA}*(c-{TAU})) for cosine c in [-1,1]:")
for c in (-1.0, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
    print(f"         c={c:+.2f} -> g={sig(ALPHA*(c-TAU)):.4f}")
print(f"         HARD CEILING on ventral weight: g <= {sig(ALPHA*(1.0-TAU)):.4f}")
