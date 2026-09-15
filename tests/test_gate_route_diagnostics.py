"""Pins every claim asserted in paper_programme/gating_route_diagnostics/CODE_AUDIT_GATE.md.

These are properties of the CODE PATH, so a small randomly-initialised model is
the correct instrument: no checkpoint, no lexicon, no GloVe.  If any of these
fail, the audit is stale and the experiment contract must be re-derived before
any result is reported.
"""
from __future__ import annotations

import math
import os

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


# ================================================== AMENDMENT 2 guards ======
# Added by the pre-execution correction pass (EXPERIMENT_CONTRACT.md AMENDMENT 2).

def test_free_ar_cap_matches_the_historical_evaluator():
    """Guard B9: the diagnostic free-AR cap must BE the historical one.

    A different cap changes what counts as non-termination and would make the
    diagnostic free-AR numbers non-comparable with the frozen record.  The
    constant is imported, not restated, so this test also fails if the
    historical evaluator ever moves.
    """
    import inspect

    from scripts.naming_comprehension.train_joint_scratch import FREE_AR_MAX_STEPS
    from gating_diagnostics import HISTORICAL_FREE_AR_MAX_STEPS, ar_decode_free

    assert HISTORICAL_FREE_AR_MAX_STEPS is FREE_AR_MAX_STEPS
    assert HISTORICAL_FREE_AR_MAX_STEPS == 12
    default = inspect.signature(ar_decode_free).parameters["max_steps"].default
    assert default == FREE_AR_MAX_STEPS, \
        f"diagnostic free-AR cap {default} != historical {FREE_AR_MAX_STEPS}"


def test_smoke_output_is_quarantined(tmp_path):
    """Guard B10: a smoke invocation can never land on a full-result path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_runner", "scripts/gating_diagnostics/run_gate_route_audit.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    out = str(tmp_path)
    full_shard, full_sum = runner.resolve_outputs(out, "ALL", smoke=False)
    smoke_shard, smoke_sum = runner.resolve_outputs(out, "W3_SRC", smoke=True)

    # full and smoke never collide
    assert runner.SMOKE_DIR not in full_shard and runner.SMOKE_DIR not in full_sum
    assert runner.SMOKE_DIR in smoke_shard and runner.SMOKE_DIR in smoke_sum
    assert smoke_shard != full_shard and smoke_sum != full_sum
    assert runner.SMOKE_SUFFIX in os.path.basename(smoke_sum)

    # the guard accepts quarantined paths ...
    runner.assert_quarantined(smoke_shard, out, True)
    runner.assert_quarantined(smoke_sum, out, True)
    # ... and hard-stops on anything else
    for bad in (full_shard, full_sum,
                os.path.join(out, "summary_metrics.json"),
                os.path.join(out, "figure_source_data", "item_level_W3_SRC.tsv")):
        with pytest.raises(RuntimeError, match="HARD STOP"):
            runner.assert_quarantined(bad, out, True)
    # a summary inside the quarantine but missing the marker is still refused
    with pytest.raises(RuntimeError, match="HARD STOP"):
        runner.assert_quarantined(
            os.path.join(out, runner.SMOKE_DIR, "summary_metrics.json"), out, True)
    # non-smoke runs are unaffected
    runner.assert_quarantined(full_sum, out, False)


def test_smoke_default_limit_is_documented_value():
    """Guard C15: one consistent smoke item count everywhere (400)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_runner2", "scripts/gating_diagnostics/run_gate_route_audit.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.SMOKE_DEFAULT_LIMIT == 400


def test_two_gate_means_are_named_and_distinct():
    """Guard B11: the item-level and historical-style means are never conflated."""
    from gating_diagnostics.analysis import summarize_state

    rows = []
    for i, (g, w) in enumerate([(0.60, 4), (0.40, 4), (0.60, 8), (0.40, 8)]):
        rows.append({
            "item_index": i, "word": f"w{i}", "batch_dec_width": w,
            "rank": i, "length": 3, "zipf_approx": 1.0, "is_word": 1,
            "gate": g, "effective_ventral_weight": g,
            "effective_dorsal_weight": 1 - g,
            "lexical_confidence": g, "lexical_margin": 0.1, "lexical_density": 1.0,
            "canonical_full_exact": 1, "canonical_wm_exact": 1,
            "canonical_ltm_exact": 1, "canonical_fixed05_exact": 1,
            "competence_category": "BOTH_CORRECT",
        })
    gm = summarize_state(rows, seed=1)["gate_means"]
    assert "gate_mean_item_level" in gm
    assert "gate_mean_position_weighted_historical" in gm
    assert "gate_mean" not in gm, "unqualified 'gate_mean' must not exist in the new package"
    assert gm["gate_mean_item_level"] == pytest.approx(0.5)
    # equal g on each width here, so both agree; the point is that they are
    # computed by different definitions and reported under different names
    assert gm["gate_mean_position_weighted_historical"] == pytest.approx(0.5)

    # now make them genuinely disagree: high g only on the wide batch
    rows[2]["gate"] = rows[2]["lexical_confidence"] = 0.64
    rows[3]["gate"] = rows[3]["lexical_confidence"] = 0.64
    gm2 = summarize_state(rows, seed=1)["gate_means"]
    assert gm2["gate_mean_position_weighted_historical"] > gm2["gate_mean_item_level"]


def test_naming_and_comprehension_never_touch_the_gate():
    """Guard A4: fixed05 is mathematically inert for N and C.

    Comprehension is encode -> s_hat -> cosine retrieval; Naming is
    ltm.decode_from_s_hat -> motor.  Neither path constructs a gate, so the
    intervention cannot move them and no redundant evaluation is warranted.
    """
    import inspect

    from scripts.naming_comprehension.frozen_probe import semantic_greedy_decode
    from scripts.naming_comprehension.train_tasks import (
        evaluate_comprehension_subset, evaluate_naming)

    for fn in (semantic_greedy_decode, evaluate_naming,
               evaluate_comprehension_subset):
        src = inspect.getsource(fn)
        assert "gate" not in src, f"{fn.__name__} unexpectedly references the gate"
        assert 'route="full"' not in src and "route='full'" not in src
