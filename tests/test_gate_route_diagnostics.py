"""Pins every claim asserted in paper_programme/gating_route_diagnostics/CODE_AUDIT_GATE.md.

These are properties of the CODE PATH, so a small randomly-initialised model is
the correct instrument: no checkpoint, no lexicon, no GloVe.  If any of these
fail, the audit is stale and the experiment contract must be re-derived before
any result is reported.
"""
from __future__ import annotations

import math

import pytest
import torch

from config import Config
from data.phonemes import build_vocab
from models.dual_route import DualRouteModel
from gating_diagnostics import (
    ROUTES, CompetenceCategory, competence_category, fixed_mix_logits,
    forced_gate, capture_gate_field, ar_decode_forced_length,
)

COHORT_ALPHA = 2.0        # frozen in all four Phase-8 source checkpoints
COHORT_TAU = 0.7


@pytest.fixture(scope="module")
def fixture():
    torch.manual_seed(0)
    cfg = Config()
    cfg.gating.alpha = COHORT_ALPHA
    cfg.gating.gate_threshold = COHORT_TAU
    vocab = build_vocab()
    model = DualRouteModel(cfg, vocab, premotor_dim=128)
    model.eval()
    model.set_semantic_bank(torch.randn(64, cfg.data.semantic_dim))

    B, T, S = 7, 6, 5
    enc_in = torch.randint(3, vocab.size, (B, T))
    enc_mask = torch.ones(B, T, dtype=torch.bool)
    enc_mask[0, 4:] = False                      # ragged, to exercise masking
    dec_in = torch.randint(3, vocab.size, (B, S))
    return model, vocab, enc_in, enc_mask, dec_in


# ---------------------------------------------------------------- audit §0/§1

def test_gate_has_no_learnable_parameters(fixture):
    """A1: the gate is a fixed function, not a learned controller."""
    model = fixture[0]
    assert list(model.gate.parameters()) == []
    assert list(model.gate.buffers()) == []


# ------------------------------------------------------------------ audit §3

def test_claim1_premotor_fusion_equals_logit_fusion(fixture):
    """CLAIM 1: motor() is affine and weights sum to 1, so the blend commutes."""
    model, _, enc_in, enc_mask, dec_in = fixture
    with torch.no_grad():
        out = model(enc_in, enc_mask, dec_in)
        recomposed = out["gate"] * out["ltm_logits"] + (1 - out["gate"]) * out["wm_logits"]
    assert torch.allclose(recomposed, out["logits"], atol=1e-5), \
        f'max|d| = {(recomposed - out["logits"]).abs().max():.3e}'


# ------------------------------------------------------------------ audit §4

def test_claim2_route_isolation_is_exact(fixture):
    """CLAIM 2: the isolated code paths ARE the g=0 / g=1 limits, bit-exactly."""
    model, _, enc_in, enc_mask, dec_in = fixture
    with torch.no_grad():
        out = model(enc_in, enc_mask, dec_in)
        wm = model.route_logits(enc_in, enc_mask, dec_in, route="wm")["logits"]
        ltm = model.route_logits(enc_in, enc_mask, dec_in, route="ltm")["logits"]
    assert torch.equal(wm, out["wm_logits"])
    assert torch.equal(ltm, out["ltm_logits"])


def test_forced_gate_endpoints_reproduce_isolated_routes(fixture):
    """The hook path and the isolated path agree at both endpoints.

    Independently reproduces the equality reported in
    lichtheim3-brain-damage@a5c787a (max|diff| = 0.000e+00).
    """
    model, _, enc_in, enc_mask, dec_in = fixture
    with torch.no_grad():
        ref = model(enc_in, enc_mask, dec_in)
        with forced_gate(model, 0.0):
            g0 = model(enc_in, enc_mask, dec_in)["logits"]
        with forced_gate(model, 1.0):
            g1 = model(enc_in, enc_mask, dec_in)["logits"]
    assert torch.allclose(g0, ref["wm_logits"], atol=1e-6)
    assert torch.allclose(g1, ref["ltm_logits"], atol=1e-6)


def test_forced_gate_is_removed_on_exit_and_on_exception(fixture):
    """The intervention must never survive its block: no hidden state change."""
    model, _, enc_in, enc_mask, dec_in = fixture
    with torch.no_grad():
        before = model(enc_in, enc_mask, dec_in)["logits"].clone()
        with forced_gate(model, 0.5):
            pass
        assert torch.equal(model(enc_in, enc_mask, dec_in)["logits"], before)
        with pytest.raises(RuntimeError):
            with forced_gate(model, 0.5):
                raise RuntimeError("boom")
        assert torch.equal(model(enc_in, enc_mask, dec_in)["logits"], before)
        assert model.gate._forward_hooks == {} or len(model.gate._forward_hooks) == 0


# ------------------------------------------------------------------ audit §3

def test_claim3_fixed_mix_equals_forced_gate_hook(fixture):
    """The two independent implementations of the 0.5/0.5 intervention agree.

    This is the load-bearing check for Experiment 2: the reported path
    (post-hoc recombination) is validated against a path that actually runs the
    blend inside the gate module.
    """
    model, _, enc_in, enc_mask, dec_in = fixture
    with torch.no_grad():
        out = model(enc_in, enc_mask, dec_in)
        post_hoc = fixed_mix_logits(out, 0.5)
        with forced_gate(model, 0.5):
            through_hook = model(enc_in, enc_mask, dec_in)["logits"]
    assert torch.allclose(post_hoc, through_hook, atol=1e-5), \
        f"max|d| = {(post_hoc - through_hook).abs().max():.3e}"


# ------------------------------------------------------------------ audit §6

def test_claim4_gate_is_word_level(fixture):
    """CLAIM 4: one scalar per item, constant across decoder steps."""
    model, _, enc_in, enc_mask, dec_in = fixture
    with torch.no_grad():
        g = model(enc_in, enc_mask, dec_in)["gate"]
    assert torch.equal(g, g[:, :1, :].expand_as(g))


def test_claim5_gate_is_blind_to_the_dorsal_route(fixture):
    """CLAIM 5 — the decisive structural result.

    Perturbing every WM-EXCLUSIVE parameter must move wm_logits and must leave
    the gate EXACTLY unchanged.  The perturbation deliberately excludes
    phon_embed.weight, the one tensor the two routes share.
    """
    model, _, enc_in, enc_mask, dec_in = fixture
    shared = {id(p) for p in model.wm.parameters()} & {id(p) for p in model.ltm.parameters()}
    assert len(shared) == 1, "expected exactly one shared tensor (phon_embed.weight)"

    with torch.no_grad():
        ref = model(enc_in, enc_mask, dec_in)
        g0, c0, wm0 = ref["gate"].clone(), ref["field_confidence"].clone(), ref["wm_logits"].clone()
        saved = []
        for p in model.wm.parameters():
            if id(p) not in shared:
                saved.append((p, p.detach().clone()))
                p.add_(torch.randn_like(p) * 0.5)
        try:
            out = model(enc_in, enc_mask, dec_in)
            assert torch.equal(out["gate"], g0), "gate moved with a dorsal-only change"
            assert torch.equal(out["field_confidence"], c0)
            assert (out["wm_logits"] - wm0).abs().max() > 1.0, "perturbation was inert"
        finally:
            for p, v in saved:
                p.copy_(v)


def test_only_shared_tensor_is_the_phoneme_embedding(fixture):
    """Route isolation is functional, not anatomical (audit §4, caveat 1)."""
    model = fixture[0]
    names = {id(p): n for n, p in model.named_parameters()}
    shared = {id(p) for p in model.wm.parameters()} & {id(p) for p in model.ltm.parameters()}
    assert [names[i] for i in shared] == ["phon_embed.weight"]


def test_claim6_attainable_gate_range_at_cohort_hyperparameters(fixture):
    """CLAIM 6: g is confined to [0.032, 0.646]; full ventral commitment is unreachable."""
    sig = lambda z: 1 / (1 + math.exp(-z))                       # noqa: E731
    lo = sig(COHORT_ALPHA * (-1.0 - COHORT_TAU))
    hi = sig(COHORT_ALPHA * (1.0 - COHORT_TAU))
    assert lo == pytest.approx(0.0323, abs=1e-4)
    assert hi == pytest.approx(0.6457, abs=1e-4)
    assert sig(COHORT_ALPHA * (COHORT_TAU - COHORT_TAU)) == pytest.approx(0.5)
    assert hi < 0.65, "the ventral route can never take more than ~65% of the blend"


def test_gate_falls_back_to_exactly_one_half_without_a_bank(fixture):
    """The 0.5/0.5 condition is an in-architecture behaviour, not a foreign one."""
    _, vocab, enc_in, enc_mask, dec_in = fixture
    cfg = Config()
    cfg.gating.alpha, cfg.gating.gate_threshold = COHORT_ALPHA, COHORT_TAU
    bare = DualRouteModel(cfg, vocab, premotor_dim=128)     # no semantic bank set
    bare.eval()
    with torch.no_grad():
        out = bare(enc_in, enc_mask, dec_in)
    assert torch.equal(out["gate"], torch.full_like(out["gate"], 0.5))


# ------------------------------------------------------------- probe plumbing

def test_capture_gate_field_matches_the_full_decode_gate(fixture):
    """A BOS-only forward captures the value used at every decode step."""
    model, vocab, enc_in, enc_mask, dec_in = fixture
    with torch.no_grad():
        full = model(enc_in, enc_mask, dec_in)["gate"][:, 0, 0]
    got = capture_gate_field(model, enc_in, enc_mask, vocab.bos_id)
    assert torch.allclose(torch.tensor(got["gate"]), full, atol=1e-6)
    for j, g in enumerate(got["gate"]):
        assert got["effective_ventral_weight"][j] == pytest.approx(g)
        assert got["effective_dorsal_weight"][j] == pytest.approx(1.0 - g)


def test_competence_categories_partition_the_two_by_two(fixture):
    seen = {competence_category(w, l) for w in (0, 1) for l in (0, 1)}
    assert seen == set(CompetenceCategory.ALL)
    assert competence_category(1, 0) == CompetenceCategory.WM_ONLY
    assert competence_category(0, 1) == CompetenceCategory.LTM_ONLY


def test_decoder_emits_all_four_routes_including_the_intervention(fixture):
    model, vocab, *_ = fixture
    forms = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    preds = ar_decode_forced_length(model, vocab, forms, "cpu")
    assert set(preds) == set(ROUTES)
    for r in ROUTES:
        assert len(preds[r]) == len(forms)
        for p, f in zip(preds[r], forms):
            assert len(p) <= len(f) + 1, "forced-length readout window violated"


def test_intervention_route_is_not_silently_the_native_gate(fixture):
    """Guards against a fixed05 column that is accidentally a copy of full."""
    model, _, enc_in, enc_mask, dec_in = fixture
    with torch.no_grad():
        out = model(enc_in, enc_mask, dec_in)
    assert not torch.allclose(fixed_mix_logits(out, 0.5), out["logits"], atol=1e-6), \
        "native gate is already exactly 0.5 on this fixture; the contrast is degenerate"
