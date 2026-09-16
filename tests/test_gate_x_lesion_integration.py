"""INTEGRATION REPAIR tests — runner -> producer -> frozen rules -> validator.

I1-I17 plus failure injections A-F, exercising the producer/consumer boundary that
the unit suites could not see: each side was tested in isolation, which is exactly
how the original gap survived.

The quarantined smoke summary these tests consume covers the FULL frozen grid
(2 witnesses x 4 lesion seeds x 2 routes x 3 lambdas x 2 conventions) on a
deliberately tiny 24-item non-canonical population, so the SAME code path the
scientific runner will use is exercised end to end — no rule is relaxed for smoke.

Run:  python3 -m pytest tests/test_gate_x_lesion_integration.py -v
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gate_x_lesion import assemble
from gate_x_lesion.assemble import RawExecutionIncomplete, build_summary
from gate_x_lesion.robustness import ROBUSTNESS_RULE_ID
from gate_x_lesion.validity import ImplementationFailureAbort
from scripts.gate_x_lesion.validate_output_manifest import (OutputIncomplete,
                                                            load_manifest, validate)

BASE = os.path.join(ROOT, "paper_programme", "gate_x_lesion_recovery")
QUARANTINE = os.path.join(BASE, "NOT_SCIENTIFIC_RESULT")
SMOKE_SUMMARY = os.path.join(QUARANTINE, "summary_ALL_SMOKE_TEST_ONLY.json")
SMOKE_MANIFEST = os.path.join(ROOT, "tests", "fixtures",
                              "expected_output_manifest.smoke.json")
SCIENTIFIC_MANIFEST = os.path.join(BASE, "EXPECTED_OUTPUT_MANIFEST.json")

SECTIONS = ("PROVENANCE", "VALIDITY", "RAW_PAIRED_RESULTS", "ROUTE_DIAGNOSTICS",
            "ROBUSTNESS", "CLASSIFICATION", "ITEM_LEVEL_AUDIT", "RUN_COMPLETION")


@pytest.fixture(scope="module")
def smoke():
    if not os.path.isfile(SMOKE_SUMMARY):
        pytest.skip("quarantined smoke summary not present; run --smoke first")
    with open(SMOKE_SUMMARY) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def smoke_manifest():
    return load_manifest(SMOKE_MANIFEST)


# =================================================================  I1 - I8

def test_I1_all_eight_sections_produced(smoke):
    for sec in SECTIONS:
        assert sec in smoke, sec
        assert smoke[sec] not in (None, [], {}), f"{sec} is empty"


def test_I2_provenance_carries_contract_and_state_identities(smoke):
    p = smoke["PROVENANCE"]
    assert len(p["FINAL_CONTRACT_HASH"]) == 64
    assert p["code_commit"] not in (None, "", "UNKNOWN")
    assert p["branch"] == "paper-programme/gate-x-lesion-recovery"
    for sid, want in (("W3_REP",
                       "9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1"),
                      ("W4_REP",
                       "e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6")):
        assert p["reconstructed_state_sha256"][sid] == want
    assert p["base_artifact_sha256"]["W3_REP"].startswith("a5f21de9")
    assert p["applies_head_sha256"]["W4_REP"].startswith("724ed4c6")
    assert p["batch_size"] == 256
    assert p["sd_ddof"] == 1
    assert p["rng_identity_fields"] == ["state_sha256", "route", "lesion_seed", "item_id"]


def test_I3_raw_paired_comes_from_actual_runner_records(smoke):
    raw = smoke["RAW_PAIRED_RESULTS"]
    assert len(raw) == 96
    # the grid is fully covered, exactly once each
    keys = {(r["state_id"], r["lesion_seed"], r["route"], r["lambda"], r["convention"])
            for r in raw}
    assert len(keys) == 96
    for r in raw:
        n = r["n_items"]
        assert n > 0
        # Delta is derived from the integer counts, not re-derived from accuracies
        assert r["accuracy_native"] == r["n_correct_native"] / n
        assert r["accuracy_fixed05"] == r["n_correct_fixed05"] / n
        assert r["delta"] == (r["n_correct_native"] - r["n_correct_fixed05"]) / n
        assert 0 <= r["n_correct_native"] <= n


def test_I4_O3_function_is_actually_called(monkeypatch, smoke):
    """The producer must CALL the frozen O-3 module, not reimplement it."""
    calls = []
    real = assemble.decide_shared_validity

    def spy(canon, free, **kw):
        calls.append((canon[0].route, canon[0].lam, len(canon), len(free)))
        return real(canon, free, **kw)

    monkeypatch.setattr(assemble, "decide_shared_validity", spy)
    assemble.build_validity(*_cells_and_intact(smoke), routes=_routes(smoke),
                            lambdas=_lambdas(smoke), seeds=_seeds(smoke),
                            witnesses=_witnesses(smoke))
    assert len(calls) == 6                      # 2 routes x 3 lambdas
    for _r, _l, nc, nf in calls:
        assert nc == 8 and nf == 8              # full 2x4 grid per convention


def test_I5_O4_function_is_actually_called(monkeypatch, smoke):
    calls = []
    real = assemble.frozen_robustness_from_counts

    def spy(counts, n_items):
        calls.append((sorted(counts), n_items))
        return real(counts, n_items)

    monkeypatch.setattr(assemble, "frozen_robustness_from_counts", spy)
    assemble.build_robustness(_cells_and_intact(smoke)[0], routes=_routes(smoke),
                              lambdas=_lambdas(smoke), seeds=_seeds(smoke),
                              witnesses=_witnesses(smoke))
    assert len(calls) == 12                     # 2 routes x 3 lambdas x 2 conventions
    for wits, _n in calls:
        assert wits == ["W3_REP", "W4_REP"]     # two witnesses, never pooled


def test_I6_I7_convention_and_joint_classifiers_are_called(monkeypatch, smoke):
    import gate_x_lesion.classify as C
    conv_calls, joint_calls = [], []
    real_conv, real_joint = C.classify_convention, C.classify_joint

    def conv_spy(sevs):
        conv_calls.append(len(sevs))
        return real_conv(sevs)

    def joint_spy(a, b):
        joint_calls.append((a, b))
        return real_joint(a, b)

    monkeypatch.setattr(C, "classify_convention", conv_spy)
    monkeypatch.setattr(C, "classify_joint", joint_spy)

    cells, intact = _cells_and_intact(smoke)
    _, v = assemble.build_validity(cells, intact, routes=_routes(smoke),
                                   lambdas=_lambdas(smoke), seeds=_seeds(smoke),
                                   witnesses=_witnesses(smoke))
    _, r = assemble.build_robustness(cells, routes=_routes(smoke),
                                     lambdas=_lambdas(smoke), seeds=_seeds(smoke),
                                     witnesses=_witnesses(smoke))
    assemble.build_classification(v, r, routes=_routes(smoke), lambdas=_lambdas(smoke))
    assert len(conv_calls) == 4                 # 2 route families x 2 conventions
    assert all(n == 3 for n in conv_calls)      # 3 severities each
    assert len(joint_calls) == 2                # one JOINT per route family


def test_I8_no_prohibited_significance_or_pooling_field(smoke):
    banned = ("p_value", "pvalue", "alpha", "bootstrap", "holm", "mcnemar",
              "pooled_witness", "pooled_seed")
    # `rule_id` legitimately contains "..._NO_SIGNIFICANCE_TEST" -- it DECLARES the
    # absence of a test -- so it is excluded from the text scan and asserted exactly.
    scrubbed = [{k: v for k, v in rec.items() if k != "rule_id"}
                for rec in smoke["ROBUSTNESS"]]
    blob = json.dumps(scrubbed).lower()
    for b in banned:
        assert b not in blob, b
    assert "significan" not in blob
    for rec in smoke["ROBUSTNESS"]:
        assert rec["rule_id"] == ROBUSTNESS_RULE_ID
        assert rec["verdict"] in ("ROBUST_POSITIVE", "ROBUST_NEGATIVE", "NOT_ROBUST")
        assert len(rec["per_witness"]) == 2


# ============================================================  I9 - I17

def test_I9_implementation_failure_suppresses_classification(smoke):
    """A structural violation must abort and emit no classification."""
    cells, intact = _cells_and_intact(smoke)
    broken = copy.deepcopy(cells)
    broken[_first_key(broken)]["shared_params_unmutated"] = False
    with pytest.raises(ImplementationFailureAbort):
        assemble.build_validity(broken, intact, routes=_routes(smoke),
                                lambdas=_lambdas(smoke), seeds=_seeds(smoke),
                                witnesses=_witnesses(smoke))
    pkg = _inject(smoke, "shared_params_unmutated", False)
    assert pkg["CLASSIFICATION"] == []
    assert pkg["ROBUSTNESS"] == []
    assert pkg["RUN_COMPLETION"]["implementation_failure_status"] != "NONE"
    assert pkg["RUN_COMPLETION"]["run_status"] == "ABORTED_IMPLEMENTATION_FAILURE"
    # and the validator refuses it
    with pytest.raises(OutputIncomplete):
        validate(dict(pkg, TEST_ONLY=True), load_manifest(SMOKE_MANIFEST),
                 expect_scientific=False)


def test_I10_removing_a_section_fails_validation(smoke, smoke_manifest):
    for sec in SECTIONS:
        s = copy.deepcopy(smoke)
        del s[sec]
        with pytest.raises(OutputIncomplete, match=sec):
            validate(s, smoke_manifest, expect_scientific=False)


def test_I11_wrong_record_count_fails_validation(smoke, smoke_manifest):
    s = copy.deepcopy(smoke)
    s["VALIDITY"] = s["VALIDITY"][:5]
    with pytest.raises(OutputIncomplete, match="VALIDITY"):
        validate(s, smoke_manifest, expect_scientific=False)
    s = copy.deepcopy(smoke)
    s["RAW_PAIRED_RESULTS"] = s["RAW_PAIRED_RESULTS"][:95]
    with pytest.raises(OutputIncomplete, match="RAW_PAIRED_RESULTS"):
        validate(s, smoke_manifest, expect_scientific=False)


def test_I12_wrong_final_contract_hash_fails_validation(smoke, smoke_manifest):
    s = copy.deepcopy(smoke)
    del s["PROVENANCE"]["FINAL_CONTRACT_HASH"]
    with pytest.raises(OutputIncomplete, match="FINAL_CONTRACT_HASH"):
        validate(s, smoke_manifest, expect_scientific=False)
    s = copy.deepcopy(smoke)
    s["PROVENANCE"]["FINAL_CONTRACT_HASH"] = None
    with pytest.raises(OutputIncomplete, match="FINAL_CONTRACT_HASH"):
        validate(s, smoke_manifest, expect_scientific=False)
    s = copy.deepcopy(smoke)
    s["RUN_COMPLETION"]["final_contract_hash_reverified"] = False
    with pytest.raises(OutputIncomplete, match="final_contract_hash_reverified"):
        validate(s, smoke_manifest, expect_scientific=False)


def test_I13_quarantine_cannot_be_accepted_as_scientific(smoke):
    """The scientific validator must refuse a TEST_ONLY summary outright."""
    with pytest.raises(OutputIncomplete, match="TEST_ONLY"):
        validate(smoke)                       # scientific manifest, default strict
    assert smoke["NOT_SCIENTIFIC_RESULT"] is True
    assert smoke["TEST_ONLY"] is True


def test_I14_scientific_namespace_cannot_contain_quarantine_marker(smoke,
                                                                   smoke_manifest):
    paths = [f"scientific_execution/item_level/s{i}.tsv" for i in range(49)]
    paths.append("NOT_SCIENTIFIC_RESULT/item_level/s49.tsv")
    with pytest.raises(OutputIncomplete, match="quarantine"):
        validate(smoke, smoke_manifest, shard_paths=paths, expect_scientific=False)
    # and a well-formed scientific shard list is accepted
    ok = [f"scientific_execution/item_level/s{i}.tsv" for i in range(50)]
    validate(smoke, smoke_manifest, shard_paths=ok, expect_scientific=False)


def test_I15_checkpoint_restoration_reaches_run_completion(smoke):
    rc = smoke["RUN_COMPLETION"]
    assert rc["state_dict_unmutated"] is True
    assert rc["base_checkpoint_unmutated"] is True
    assert rc["applied_head_unmutated"] is True
    for sid in ("W3_REP", "W4_REP"):
        assert rc["state_dict_sha256_before"][sid] == rc["state_dict_sha256_after"][sid]


def test_I16_matched_lesion_tensor_invariant_reaches_structural_validity(smoke):
    """The NATIVE/FIXED05 matched-tensor invariant is carried into O-3's input."""
    for d in smoke["ROUTE_DIAGNOSTICS"]:
        assert d["matched_lesion_tensors_identical"] is True
        assert d["shared_params_unmutated"] is True
        assert d["checkpoint_identity_restored"] is True
    # and a violation is an implementation failure at the O-3 boundary
    pkg = _inject(smoke, "matched_lesion_tensors_identical", False)
    assert pkg["CLASSIFICATION"] == []
    assert "LESION_TENSOR_MISMATCH" in pkg["RUN_COMPLETION"]["implementation_failures"][0]


def test_I17_assembly_is_deterministic(smoke):
    a = _rebuild(smoke)
    b = _rebuild(smoke)
    assert json.dumps(a, sort_keys=True, default=str) == \
        json.dumps(b, sort_keys=True, default=str)


# ===================================================  failure injections A - F

def _first_key(cells, route=None):
    """A deterministic cell key, optionally restricted to a route."""
    keys = sorted(k for k in cells if route is None or k[1] == route)
    assert keys, f"no cell for route {route}"
    return keys[0]


def _inject(smoke, key, value, route=None):
    return _rebuild(smoke, mutate=lambda c: c[_first_key(c, route)].__setitem__(
        key, value))


def test_FAIL_A_perturbation_validity_failure_does_not_abort(smoke):
    """A: competence failure invalidates that route x severity only; run continues."""
    def mutate(cells):
        # make every block's targeted drop zero for ONE route x severity
        for k, c in cells.items():
            if k[1] == "wm_encoder_state" and k[2] == 0.25:
                for conv in ("canonical", "freear"):
                    c[conv]["n_correct_wm_isolated"] = 24
    pkg = _rebuild(smoke, mutate=mutate)
    assert pkg["RUN_COMPLETION"]["implementation_failure_status"] == "NONE"
    assert pkg["CLASSIFICATION"] != []                 # experiment continued
    rec = [v for v in pkg["VALIDITY"]
           if v["route"] == "wm_encoder_state" and v["lambda"] == 0.25][0]
    assert rec["shared_validity_label"] is False
    assert rec["scientific_perturbation_validity_failure"] is True
    assert rec["validity_reason"].startswith("TARGET_ROUTE_DROP_FAILED")
    assert len(pkg["VALIDITY"]) == 6                   # no denominator shrinking


@pytest.mark.parametrize("key,value,marker", [
    ("shared_params_unmutated", False, "SHARED_PARAM_MUTATED"),
    ("checkpoint_identity_restored", False, "CHECKPOINT_NOT_RESTORED"),
    ("matched_lesion_tensors_identical", False, "LESION_TENSOR_MISMATCH"),
])
def test_FAIL_B_C_E_F_structural_violations_abort(smoke, key, value, marker):
    pkg = _inject(smoke, key, value, route="wm_encoder_state")
    rc = pkg["RUN_COMPLETION"]
    assert rc["run_status"] == "ABORTED_IMPLEMENTATION_FAILURE"
    assert rc["implementation_failure_status"] != "NONE"
    assert any(marker in m for m in rc["implementation_failures"])
    assert pkg["CLASSIFICATION"] == [] and pkg["ROBUSTNESS"] == []


def test_FAIL_C_untargeted_route_violation_aborts(smoke):
    """Dorsal perturbation moving the UNTARGETED (ventral) route is a wiring failure."""
    def mutate(cells):
        k = _first_key(cells, "wm_encoder_state")      # dorsal -> untargeted is ltm
        cells[k]["canonical"]["ltm_changed_vs_intact"] = 1
    pkg = _rebuild(smoke, mutate=mutate)
    rc = pkg["RUN_COMPLETION"]
    assert rc["run_status"] == "ABORTED_IMPLEMENTATION_FAILURE"
    assert any("UNTARGETED_ROUTE_CHANGED" in m for m in rc["implementation_failures"])
    assert pkg["CLASSIFICATION"] == []


def test_FAIL_D_dorsal_gate_violation_aborts(smoke):
    """Criterion 4 binds DORSAL only; it is vacuous for ventral by design."""
    for key in ("max_abs_delta_gate", "max_abs_delta_c_LTM"):
        pkg = _inject(smoke, key, 1e-12, route="wm_encoder_state")
        rc = pkg["RUN_COMPLETION"]
        assert rc["run_status"] == "ABORTED_IMPLEMENTATION_FAILURE", key
        assert any("DORSAL_GATE_NULL_VIOLATED" in m
                   for m in rc["implementation_failures"]), key
        assert pkg["CLASSIFICATION"] == []

    # ventral: the same delta is expected and must NOT abort
    pkg = _inject(smoke, "max_abs_delta_gate", 0.5, route="ltm_encoder_state")
    assert pkg["RUN_COMPLETION"]["run_status"] == "COMPLETED"


def test_incomplete_raw_execution_is_refused(smoke):
    """No denominator shrinking: a missing cell refuses assembly outright."""
    states = copy.deepcopy(smoke["_state_summaries"])
    states[0]["cells"] = states[0]["cells"][:-1]
    with pytest.raises(RawExecutionIncomplete, match="incomplete"):
        build_summary(state_summaries=states, audit_records=[], shard_paths=[],
                      cfg=smoke["_cfg"], expected_shards=50, expected_validity=6,
                      expected_paired=96)


# ==========================================================  CLI end-to-end

def test_validator_cli_accepts_smoke_and_refuses_it_as_scientific():
    v = os.path.join(ROOT, "scripts", "gate_x_lesion", "validate_output_manifest.py")
    ok = subprocess.run([sys.executable, v, SMOKE_SUMMARY, SMOKE_MANIFEST,
                         "--test-only"], capture_output=True, text=True, cwd=ROOT)
    assert ok.returncode == 0, ok.stderr
    assert "PASSED" in ok.stdout

    bad = subprocess.run([sys.executable, v, SMOKE_SUMMARY],
                         capture_output=True, text=True, cwd=ROOT)
    assert bad.returncode == 1
    assert "TEST_ONLY" in bad.stderr


def test_scientific_manifest_still_pins_the_scientific_counts():
    m = load_manifest(SCIENTIFIC_MANIFEST)
    assert m["required_shards"]["expected_item_level_count"] == 50
    assert m["VALIDITY"]["expected_records"] == 6
    assert m["RAW_PAIRED_RESULTS"]["expected_records"] == 96
    assert m["PROVENANCE"]["pinned_values"]["population_n"] == 29571
    assert m["PROVENANCE"]["pinned_values"]["batch_size"] == 256


# ------------------------------------------------------------------ helpers

def _routes(s):
    return s["PROVENANCE"]["routes"]


def _lambdas(s):
    return [float(x) for x in s["PROVENANCE"]["lambdas"]]


def _seeds(s):
    return [int(x) for x in s["PROVENANCE"]["lesion_seeds"]]


def _witnesses(s):
    return sorted(s["PROVENANCE"]["reconstructed_state_sha256"])


def _cells_and_intact(s):
    states = s["_state_summaries"]
    return assemble._index_cells(states), assemble._intact(states)


def _rebuild(s, mutate=None):
    states = copy.deepcopy(s["_state_summaries"])
    if mutate is not None:
        cells = assemble._index_cells(states)
        mutate(cells)
    return build_summary(state_summaries=states, audit_records=s["ITEM_LEVEL_AUDIT"],
                         shard_paths=[f"p{i}" for i in range(50)], cfg=s["_cfg"],
                         expected_shards=50, expected_validity=6, expected_paired=96)
