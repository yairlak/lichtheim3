"""CLOSURE PASS tests — O-1, FREE-AR trajectory independence, O-3, O-4, F-2, F-3, F-8.

No non-zero lesion is run on the canonical population anywhere in this file.  The only
model-backed tests use a deliberately tiny non-canonical subset, for wiring validation.

Run:  python3 -m pytest tests/test_gate_x_lesion_closure.py -v
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gate_x_lesion.hooks import lesioned_route, state_dict_sha256
from gate_x_lesion.identity import (STATE_IDENTITY_DOMAIN,
                                    reconstructed_state_sha256,
                                    verify_manifest_state_identity)
from gate_x_lesion.noise import EpsilonCache, FROZEN_LAMBDAS, base_epsilon, eta
from gate_x_lesion.outcomes import (CANDIDATE_ROBUSTNESS_RULES,
                                    FROZEN_ROBUSTNESS_RULE, OUTCOME_A, OUTCOME_B,
                                    OUTCOME_F, SIGN_FIXED05_ADVANTAGE,
                                    SIGN_NATIVE_ADVANTAGE, SeedRecord,
                                    UnfrozenRobustnessError, evaluate_severity)
from gate_x_lesion.sd import measure_intact_sd
from gate_x_lesion.targets import assert_site_compatible
from gate_x_lesion.validity import (DETERMINISTIC_TOLERANCE, MIN_BLOCKS_WITH_DROP,
                                    N_BLOCKS, TARGETED_MIN_DROP, BlockObservation,
                                    UnresolvedPolicyError,
                                    decide_route_severity_validity)
from scripts.gate_x_lesion.run_gate_x_lesion import (SCIENTIFIC_BATCH_SIZE,
                                                     assert_scientific_batch_size)

REPO = "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3"
OUT_BASE = os.path.join(ROOT, "paper_programme", "gate_x_lesion_recovery")
MANIFEST = os.path.join(OUT_BASE, "checkpoint_manifest.proposed.tsv")

SMOKE_N = 8
POLICY = dict(validity_convention_scope="shared",
              implementation_failure_block_semantics="abort_run")


def _rows():
    with open(MANIFEST, newline="") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if r["in_scope"] == "1"]


@pytest.fixture(scope="module")
def states():
    from scripts.gating_diagnostics.run_gate_route_audit import build_state
    out = {}
    for row in _rows():
        legacy = {"state_id": row["state_id"], "arm": row["arm"],
                  "artifact_path": row["base_artifact_path"],
                  "artifact_sha256": row["base_artifact_sha256"],
                  "applies_head_path": row["applies_head_path"],
                  "applies_head_sha256": row["applies_head_sha256"],
                  "source_u": row["source_u"]}
        tr, model, prov, before, ckpt = build_state(legacy, "cpu")
        out[row["state_id"]] = dict(row=row, tr=tr, model=model, ckpt=ckpt)
    return out


# ==========================================================  O-1 state identity

def test_O1_identity_is_pure_function_of_two_digests():
    """Only two file digests can reach the identity — nothing else is a parameter."""
    import inspect
    sig = inspect.signature(reconstructed_state_sha256)
    assert list(sig.parameters) == ["base_artifact_sha256", "applied_head_sha256"]

    b, h = "a" * 64, "b" * 64
    expect = hashlib.sha256(
        (STATE_IDENTITY_DOMAIN + b + "|" + h).encode("ascii")).hexdigest()
    assert reconstructed_state_sha256(b, h) == expect
    # deterministic across calls
    assert reconstructed_state_sha256(b, h) == reconstructed_state_sha256(b, h)
    # both operands are load-bearing
    assert reconstructed_state_sha256("c" * 64, h) != expect
    assert reconstructed_state_sha256(b, "c" * 64) != expect
    # malformed input fails closed rather than silently re-seeding
    for bad in ("", "xyz", "a" * 63, "a" * 65, "g" * 64):
        with pytest.raises(ValueError):
            reconstructed_state_sha256(bad, h)
    # hex CASE is normalised deliberately: two spellings of one digest must map to
    # one identity, not to two different epsilon streams
    assert reconstructed_state_sha256(b.upper(), h) == reconstructed_state_sha256(b, h)


def test_O1_identity_matches_frozen_manifest_and_prior_commit():
    """The named helper reproduces the value IMPLEMENTATION_COMMIT already used."""
    for r in _rows():
        legacy = hashlib.sha256(
            ("gxlr-state-v1|" + r["base_artifact_sha256"] + "|"
             + r["applies_head_sha256"]).encode()).hexdigest()
        named = reconstructed_state_sha256(r["base_artifact_sha256"],
                                           r["applies_head_sha256"])
        assert named == legacy == r["state_sha256"], r["state_id"]
        assert verify_manifest_state_identity(r) == r["state_sha256"]


def test_O1_manifest_tamper_fails_closed():
    r = dict(_rows()[0])
    r["state_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="RECONSTRUCTED_STATE_SHA256 mismatch"):
        verify_manifest_state_identity(r)


def test_O1_identity_invariant_across_every_experimental_factor():
    """Seeds, severities, routes, items, fusion, convention: none is an input."""
    r = _rows()[0]
    ident = reconstructed_state_sha256(r["base_artifact_sha256"],
                                       r["applies_head_sha256"])
    for _seed in (0, 1, 2, 3):
        for _lam in FROZEN_LAMBDAS:
            for _route in ("wm_encoder_state", "ltm_encoder_state"):
                for _fusion in ("NATIVE", "FIXED05"):
                    for _conv in ("CANONICAL", "FREE_AR"):
                        assert reconstructed_state_sha256(
                            r["base_artifact_sha256"],
                            r["applies_head_sha256"]) == ident


def test_O1_identity_cannot_be_affected_by_epsilon_or_lesion(states):
    """epsilon is derived FROM the identity; the dependency is one-way.

    Proven operationally: take the identity, run a real lesion context on a tiny
    non-canonical subset, and show the identity and the intact parameters are
    unchanged afterwards.
    """
    st = states["W3_REP"]
    model, tr, row = st["model"], st["tr"], st["row"]
    ident_before = verify_manifest_state_identity(row)
    params_before = state_dict_sha256(model)

    sd = measure_intact_sd(model, tr.vocab, tr.entries,
                           routes=("wm_encoder_state",),
                           n_items=256)["wm_encoder_state"]["sd"]
    words = [tr.entries[i].word for i in range(SMOKE_N)]

    with lesioned_route(model, route="wm_encoder_state", state_sha256=ident_before,
                        lesion_seed=0, lam=1.0, sd=sd) as hook:
        hook.bind(words)
        from evaluate.hooks import make_batch
        b = make_batch([list(tr.entries[i].phonemes) for i in range(SMOKE_N)],
                       tr.vocab, "cpu")
        model(b["enc_in"], b["enc_mask"], b["dec_in"])
        assert hook._eta.abs().sum() > 0            # a real perturbation happened

    assert verify_manifest_state_identity(row) == ident_before
    assert state_dict_sha256(model) == params_before


def test_O1_identity_stable_across_repeated_reconstruction():
    """Rebuilding the witness from its recipe yields the same parameters and identity."""
    from scripts.gating_diagnostics.run_gate_route_audit import build_state
    r = _rows()[0]
    legacy = {"state_id": r["state_id"], "arm": r["arm"],
              "artifact_path": r["base_artifact_path"],
              "artifact_sha256": r["base_artifact_sha256"],
              "applies_head_path": r["applies_head_path"],
              "applies_head_sha256": r["applies_head_sha256"],
              "source_u": r["source_u"]}
    hashes = []
    for _ in range(2):
        _tr, model, _p, _b, _c = build_state(legacy, "cpu")
        hashes.append(state_dict_sha256(model))
    assert hashes[0] == hashes[1], "reconstruction is not deterministic"
    assert verify_manifest_state_identity(r) == r["state_sha256"]


# ================================================  FREE-AR independent trajectories

def test_FREEAR_forced_divergence_independent_trajectories(states):
    """Force a divergence at step 0 and prove A-E simultaneously.

    A. prefixes differ from step 1 onward;
    B. each branch consumes ITS OWN previous predicted token;
    C. both branches see the same epsilon and the same eta(lambda);
    D. divergence and re-encoding never redraw the lesion tensor;
    E. this is genuine free-AR (`ar_decode_free`), not forced-length teacher forcing.

    Divergence is forced by stubbing ONLY the blend function's output inside this test,
    so that FIXED05 is guaranteed to pick a different token.  The real `ar_decode_free`
    loop, the real encoder and the real lesion hook are exercised unmodified.  No
    shipped code is changed and FIXED05's algebra is untouched outside this test.
    """
    import gating_diagnostics.gate_probe as gp

    st = states["W3_REP"]
    model, tr, row = st["model"], st["tr"], st["row"]
    sd = measure_intact_sd(model, tr.vocab, tr.entries,
                           routes=("ltm_encoder_state",),
                           n_items=256)["ltm_encoder_state"]["sd"]
    forms = [tr.entries[i].phonemes for i in range(SMOKE_N)]
    words = [tr.entries[i].word for i in range(SMOKE_N)]
    ident = row["state_sha256"]
    lam = 0.25

    captured = []                      # (route, dec_in, logits, eta)
    real_step = gp._route_step_logits

    with lesioned_route(model, route="ltm_encoder_state", state_sha256=ident,
                        lesion_seed=0, lam=lam, sd=sd) as hook:
        hook.bind(words)

        def spy(m, enc_in, enc_mask, dec_in, route):
            lg = real_step(m, enc_in, enc_mask, dec_in, route)
            if route == "fixed05":
                lg = lg.flip(-1)       # deterministic guaranteed divergence
            captured.append((route, dec_in.clone(), lg.clone(), hook._eta.clone()))
            return lg

        gp._route_step_logits = spy
        try:
            preds = gp.ar_decode_free(model, tr.vocab, forms, "cpu",
                                      routes=("full", "fixed05"))
        finally:
            gp._route_step_logits = real_step

    nat = [c for c in captured if c[0] == "full"]
    fix = [c for c in captured if c[0] == "fixed05"]
    assert len(nat) > 1 and len(fix) > 1, "free-AR did not take multiple steps"

    # -- B: each branch extends ITS OWN prefix with ITS OWN previous argmax ----
    for branch, name in ((nat, "NATIVE"), (fix, "FIXED05")):
        for k in range(len(branch) - 1):
            _r, dec_k, lg_k, _e = branch[k]
            _r2, dec_next, _lg2, _e2 = branch[k + 1]
            expected = torch.cat(
                [dec_k, lg_k[:, -1, :].argmax(-1, keepdim=True)], dim=1)
            assert torch.equal(dec_next, expected), (
                f"{name} step {k}->{k+1} did not consume its own predicted token")

    # -- A: the two branches genuinely diverge, and stay diverged ---------------
    common = min(len(nat), len(fix))
    diverged_at = None
    for k in range(common):
        if not torch.equal(nat[k][1], fix[k][1]):
            diverged_at = k
            break
    assert diverged_at is not None, "no divergence was produced — test is vacuous"
    assert diverged_at >= 1, "prefixes must agree at BOS (step 0)"
    for k in range(diverged_at, common):
        assert not torch.equal(nat[k][1], fix[k][1]), (
            f"branches re-converged in lockstep at step {k}")

    # -- C: identical epsilon and identical eta(lambda) on BOTH branches --------
    cache = EpsilonCache(ident, "ltm_encoder_state", 0, hook.cache.n_units)
    expected_eta = cache.batch_epsilon(words) * hook.amp
    for r, _dec, _lg, e in captured:
        assert torch.equal(e, expected_eta), f"{r} saw a different eta"
    assert torch.equal(nat[0][3], fix[0][3])

    # -- D: no redraw across divergence, re-encoding or steps -------------------
    assert hook.n_eta_builds == 1, f"lesion tensor rebuilt {hook.n_eta_builds} times"
    assert hook.n_calls == len(captured)
    assert cache.n_computed == SMOKE_N

    # -- E: genuine free-AR, not forced length ---------------------------------
    assert preds["full"] != preds["fixed05"], "forced divergence did not reach output"
    for seq in preds["fixed05"]:
        assert len(seq) <= gp.HISTORICAL_FREE_AR_MAX_STEPS


# ==============================================================  F-2 architecture

def test_F2_runner_asserts_architecture_before_ventral_hook(states):
    """The ventral slot hazard fails closed at the site guard the runner calls."""
    st = states["W3_REP"]
    model = st["model"]
    assert model.ltm.cfg.ltm_encoder_mode == "unigru_last_hidden"
    assert_site_compatible(model, "ltm_encoder_state")

    old = model.ltm.cfg.ltm_encoder_mode
    try:
        object.__setattr__(model.ltm.cfg, "ltm_encoder_mode", "bigru_masked_mean")
        with pytest.raises(RuntimeError, match="SILENT NO-OP"):
            assert_site_compatible(model, "ltm_encoder_state")
        # and the lesion context itself refuses, not merely the standalone guard
        with pytest.raises(RuntimeError, match="SILENT NO-OP"):
            with lesioned_route(model, route="ltm_encoder_state",
                                state_sha256=st["row"]["state_sha256"],
                                lesion_seed=0, lam=0.25, sd=0.17):
                pass
    finally:
        object.__setattr__(model.ltm.cfg, "ltm_encoder_mode", old)
    assert_site_compatible(model, "ltm_encoder_state")


# ===============================================================  F-3 provenance

def test_F3_manifest_records_full_reconstruction_recipe():
    for r in _rows():
        assert r["state_kind"] == "RECONSTRUCTION_RECIPE_base_plus_head"
        for field in ("base_artifact_path", "base_artifact_sha256",
                      "applies_head_path", "applies_head_sha256",
                      "head_localizer", "state_sha256"):
            assert r[field], f"{r['state_id']} missing {field}"
        assert "p_last_hinge" in r["head_localizer"]
        assert os.path.exists(os.path.join(REPO, r["base_artifact_path"]))
        assert os.path.exists(os.path.join(REPO, r["applies_head_path"]))


def test_F3_runner_summary_carries_provenance():
    """The runner's summary dict must record every reconstruction field."""
    import inspect
    from scripts.gate_x_lesion import run_gate_x_lesion as R
    src = inspect.getsource(R.run_state)
    for field in ("reconstructed_state_sha256", "base_artifact_path",
                  "base_artifact_sha256", "applies_head_path",
                  "applies_head_sha256", "head_localizer", "state_kind"):
        assert f'"{field}"' in src, field


# ===============================================================  F-8 batch size

def test_F8_scientific_batch_size_pinned():
    assert SCIENTIFIC_BATCH_SIZE == 256
    assert_scientific_batch_size(256, smoke=False)
    for bad in (128, 255, 257, 512, 1024):
        with pytest.raises(RuntimeError, match="F-8 pins the"):
            assert_scientific_batch_size(bad, smoke=False)
    # quarantined smoke may use any batch size — it is not a scientific result
    for any_bs in (8, 24, 512):
        assert_scientific_batch_size(any_bs, smoke=True) is None


def test_F8_prepared_command_uses_the_pinned_batch_size():
    cmd = open(os.path.join(OUT_BASE, "EXECUTION_COMMAND.prepared.sh")).read()
    assert "--batch-size 256" in cmd


# =========================================================  O-3 diagnostic validity

def _block(state_id, seed, route="wm_encoder_state", lam=0.25, *, drop=0.20,
           untargeted=0.0, shared_ok=True, d_c=0.0, d_g=0.0, conv="CANONICAL"):
    return BlockObservation(
        state_id=state_id, lesion_seed=seed, route=route, lam=lam, convention=conv,
        targeted_exact_intact=0.90, targeted_exact_lesioned=0.90 - drop,
        untargeted_max_abs_delta=untargeted, shared_params_unmutated=shared_ok,
        max_abs_delta_c_ltm=d_c, max_abs_delta_gate=d_g)


def _grid(**kw):
    return [_block(s, k, **kw) for s in ("W3_REP", "W4_REP") for k in range(4)]


def test_O3_grid_and_tolerance_constants():
    assert N_BLOCKS == 8 and MIN_BLOCKS_WITH_DROP == 6
    assert TARGETED_MIN_DROP == 0.10
    assert DETERMINISTIC_TOLERANCE == 0.0       # bitwise, from GATING §5


def test_O3_policy_fields_have_no_default():
    """The two unresolved O-3 fields cannot be silently guessed."""
    with pytest.raises(TypeError):
        decide_route_severity_validity(_grid())                # missing both
    with pytest.raises(UnresolvedPolicyError):
        decide_route_severity_validity(
            _grid(), validity_convention_scope="whatever",
            implementation_failure_block_semantics="abort_run")
    with pytest.raises(UnresolvedPolicyError):
        decide_route_severity_validity(
            _grid(), validity_convention_scope="shared",
            implementation_failure_block_semantics="guess")


def test_O3_boundary_on_block_count():
    # 6/8 -> valid; 5/8 -> invalid
    for n_ok, expect in ((8, True), (6, True), (5, False), (0, False)):
        blocks = [_block("W3_REP", k, drop=0.20 if k < n_ok else 0.01)
                  for k in range(8)]
        v = decide_route_severity_validity(blocks, **POLICY)
        assert v.diagnostically_valid is expect, (n_ok, v.reason)
        assert v.n_blocks_with_drop == n_ok


def _count_block(state_id, seed, n_items, n_intact, n_lesioned):
    return BlockObservation(
        state_id=state_id, lesion_seed=seed, route="wm_encoder_state", lam=0.25,
        convention="CANONICAL", targeted_exact_intact=n_intact / n_items,
        targeted_exact_lesioned=n_lesioned / n_items, untargeted_max_abs_delta=0.0,
        shared_params_unmutated=True, n_items=n_items,
        n_correct_intact=n_intact, n_correct_lesioned=n_lesioned)


def test_O3_boundary_on_drop_magnitude():
    """The >= 0.10 boundary, evaluated the numerically correct way.

    The drop must come from ONE division of an integer count difference. Deriving it
    as the difference of two separately-rounded accuracies puts representation error
    exactly on the decision boundary: 0.90 - 0.10 == 0.09999999999999998, which would
    spuriously fail `>= 0.10`.
    """
    exact = [_count_block(s, k, 1000, 900, 800)          # drop == 0.100 exactly
             for s in ("W3_REP", "W4_REP") for k in range(4)]
    assert decide_route_severity_validity(exact, **POLICY).diagnostically_valid

    just_under = [_count_block(s, k, 1000, 900, 801)     # drop == 0.099
                  for s in ("W3_REP", "W4_REP") for k in range(4)]
    assert not decide_route_severity_validity(just_under, **POLICY).diagnostically_valid

    # the float-subtraction form is why counts are preferred
    naive = BlockObservation(
        state_id="W3_REP", lesion_seed=0, route="wm_encoder_state", lam=0.25,
        convention="CANONICAL", targeted_exact_intact=0.9,
        targeted_exact_lesioned=0.9 - 0.10, untargeted_max_abs_delta=0.0,
        shared_params_unmutated=True)
    assert naive.targeted_drop < TARGETED_MIN_DROP
    assert _count_block("W3_REP", 0, 1000, 900, 800).targeted_drop >= TARGETED_MIN_DROP

    # on the canonical population an exact tie cannot arise at all
    assert abs(TARGETED_MIN_DROP * 29571 - round(TARGETED_MIN_DROP * 29571)) > 1e-9


def test_O3_untargeted_route_must_be_bitwise_unchanged():
    v = decide_route_severity_validity(_grid(untargeted=0.0), **POLICY)
    assert v.diagnostically_valid
    v = decide_route_severity_validity(_grid(untargeted=1e-9), **POLICY)
    assert not v.diagnostically_valid
    assert v.criterion_2_failures


def test_O3_dorsal_gate_violation_is_an_implementation_failure():
    """Criterion 4 binds dorsal only, and a violation aborts rather than returning a null."""
    with pytest.raises(RuntimeError, match="implementation failure"):
        decide_route_severity_validity(_grid(d_g=1e-9), **POLICY)
    with pytest.raises(RuntimeError, match="implementation failure"):
        decide_route_severity_validity(_grid(d_c=1e-9), **POLICY)
    with pytest.raises(RuntimeError, match="implementation failure"):
        decide_route_severity_validity(_grid(shared_ok=False), **POLICY)
    # ventral: criterion 4 is vacuous, a moving gate is expected and allowed
    v = decide_route_severity_validity(
        _grid(route="ltm_encoder_state", d_g=0.5, d_c=0.3), **POLICY)
    assert v.diagnostically_valid


def test_O3_implementation_failure_denominator_semantics_differ():
    """The three denominator policies are genuinely distinct — hence the open question."""
    blocks = _grid()
    blocks[0] = _block("W3_REP", 0, shared_ok=False)           # one failed block
    with pytest.raises(RuntimeError):
        decide_route_severity_validity(
            blocks, validity_convention_scope="shared",
            implementation_failure_block_semantics="abort_run")
    v = decide_route_severity_validity(
        blocks, validity_convention_scope="shared",
        implementation_failure_block_semantics="exclude_block_and_shrink_denominator")
    assert v.n_blocks_considered == 7 and v.n_blocks_with_drop == 7
    assert v.implementation_failures            # recorded, never silently dropped


def test_O3_validity_never_reads_the_fusion_contrast():
    """No NATIVE/FIXED05 quantity is even representable in the validity input."""
    fields = set(BlockObservation.__dataclass_fields__)
    for banned in ("native", "fixed05", "discordant", "errors_gained",
                   "errors_recovered", "net_change_in_correct", "p_exact_mcnemar"):
        assert not any(banned in f for f in fields), banned


# ==============================================================  O-4 robustness

def _rec(state_id, seed, *, net, p, lam=0.25, dacc=0.01):
    return SeedRecord(lam=lam, seed=seed, n_changed_vs_intact=100,
                      native_exact_match=0.6, modal_prediction_share=0.1,
                      net_change_in_correct=net, p_exact_mcnemar=p,
                      state_id=state_id, delta_accuracy=dacc)


def test_O4_no_rule_is_frozen():
    assert FROZEN_ROBUSTNESS_RULE is None
    assert set(CANDIDATE_ROBUSTNESS_RULES) == {
        "R1_per_witness_significance",
        "R2_significance_plus_inherited_materiality",
        "R3_unanimous_sign_no_test"}


def test_O4_classification_requires_an_explicit_rule():
    recs = [_rec("W3_REP", k, net=-40, p=0.001) for k in range(4)]
    with pytest.raises(TypeError):
        evaluate_severity(recs)                       # no default rule exists
    with pytest.raises(UnfrozenRobustnessError):
        evaluate_severity(recs, robustness_rule=None)


def test_O4_candidate_R1_positive_negative_and_noisy():
    R1 = CANDIDATE_ROBUSTNESS_RULES["R1_per_witness_significance"]
    both = lambda net, p: [_rec(s, k, net=net, p=p)
                           for s in ("W3_REP", "W4_REP") for k in range(4)]
    assert R1(both(-40, 0.001)).sign == SIGN_NATIVE_ADVANTAGE
    assert R1(both(+40, 0.001)).sign == SIGN_FIXED05_ADVANTAGE
    assert R1(both(-40, 0.90)).robust is False        # noisy: nothing significant
    # one witness disagrees -> not robust (no cross-witness pooling, GATING §9)
    mixed = ([_rec("W3_REP", k, net=-40, p=0.001) for k in range(4)]
             + [_rec("W4_REP", k, net=+40, p=0.001) for k in range(4)])
    assert R1(mixed).robust is False


def test_O4_candidate_R2_applies_the_inherited_materiality_floor():
    R1 = CANDIDATE_ROBUSTNESS_RULES["R1_per_witness_significance"]
    R2 = CANDIDATE_ROBUSTNESS_RULES["R2_significance_plus_inherited_materiality"]
    tiny = [_rec(s, k, net=-40, p=0.001, dacc=0.0001)
            for s in ("W3_REP", "W4_REP") for k in range(4)]
    assert R1(tiny).robust is True            # R1 certifies a trivial effect
    assert R2(tiny).robust is False           # R2 rejects it at |Δacc| >= 0.002


def test_O4_candidate_R3_needs_unanimous_nonzero_sign():
    R3 = CANDIDATE_ROBUSTNESS_RULES["R3_unanimous_sign_no_test"]
    assert R3([_rec("W3_REP", k, net=-5, p=None) for k in range(8)]).robust is True
    assert R3([_rec("W3_REP", k, net=0, p=None) for k in range(8)]).robust is False
    mixed = [_rec("W3_REP", k, net=(-5 if k else +5), p=None) for k in range(8)]
    assert R3(mixed).robust is False


def test_O4_veto_still_binds_under_any_candidate_rule():
    """Whatever CENTRAL picks, opposite robust signs across severities give F."""
    from gate_x_lesion.outcomes import classify_convention, SeverityVerdict
    R1 = CANDIDATE_ROBUSTNESS_RULES["R1_per_witness_significance"]
    verdicts = []
    for lam, net in ((0.25, -40), (0.50, +40), (1.00, -40)):
        recs = [_rec(s, k, net=net, p=0.001, lam=lam)
                for s in ("W3_REP", "W4_REP") for k in range(4)]
        verdicts.append(evaluate_severity(recs, robustness_rule=R1,
                                          diagnostically_valid=True))
    out = classify_convention(verdicts)
    assert out["outcome"] == OUTCOME_F and out["veto_applied"] is True
    assert out["valid_severities"] == [0.25, 0.50, 1.00]


def test_O4_homogeneous_signs_give_A_or_B():
    from gate_x_lesion.outcomes import classify_convention
    R1 = CANDIDATE_ROBUSTNESS_RULES["R1_per_witness_significance"]
    for net, expect in ((-40, OUTCOME_A), (+40, OUTCOME_B)):
        verdicts = [
            evaluate_severity(
                [_rec(s, k, net=net, p=0.001, lam=lam)
                 for s in ("W3_REP", "W4_REP") for k in range(4)],
                robustness_rule=R1, diagnostically_valid=True)
            for lam in FROZEN_LAMBDAS]
        assert classify_convention(verdicts)["outcome"] == expect
