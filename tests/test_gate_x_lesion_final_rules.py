"""FINAL RULE FREEZE tests — CENTRAL's authoritative O-3 and O-4 decisions.

Synthetic data only.  No model, no checkpoint, no lesion, no scientific outcome.

Run:  python3 -m pytest tests/test_gate_x_lesion_final_rules.py -v
"""
from __future__ import annotations

import inspect
import os
import sys
from fractions import Fraction

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gate_x_lesion.classify import (DIRECTION_TO_LETTER, SeverityClassification,
                                    classify_convention, classify_route_family)
from gate_x_lesion.outcomes import (FROZEN_ROBUSTNESS_RULE,
                                    FROZEN_ROBUSTNESS_RULE_ID,
                                    JOINT_MIXED_DECODING, OUTCOME_A, OUTCOME_B,
                                    OUTCOME_D, OUTCOME_F, OUTCOME_UNDETERMINED,
                                    classify_joint, concordant)
from gate_x_lesion.robustness import (DIRECTION_NEGATIVE, DIRECTION_NONE,
                                      DIRECTION_POSITIVE, MATERIALITY_FLOOR,
                                      MIN_DIRECTIONAL_SEEDS, NOT_ROBUST,
                                      N_LESION_SEEDS, ROBUST_NEGATIVE,
                                      ROBUST_POSITIVE, ROBUSTNESS_RULE_ID,
                                      frozen_robustness, frozen_robustness_from_counts,
                                      integer_threshold_for, witness_direction,
                                      witness_direction_from_counts)
from gate_x_lesion.validity import (IMPLEMENTATION_FAILURE_SEMANTICS,
                                    MIN_BLOCKS_WITH_DROP, N_BLOCKS,
                                    TARGETED_MIN_DROP, VALIDITY_CONVENTION_SCOPE,
                                    BlockObservation, ImplementationFailure,
                                    ImplementationFailureAbort, ValidityReason,
                                    decide_shared_validity)

N_ITEMS = 29571
WITNESSES = ("W3_REP", "W4_REP")


# ====================================================================  O-3

def _blk(state, seed, conv, *, n_intact=26614, n_les=None, drop_items=None,
         route="wm_encoder_state", lam=0.25, **structural):
    """A block with the targeted drop expressed in exact integer items."""
    if n_les is None:
        n_les = n_intact - (drop_items if drop_items is not None else 3000)
    return BlockObservation(
        state_id=state, lesion_seed=seed, route=route, lam=lam, convention=conv,
        n_items=N_ITEMS, n_correct_intact=n_intact, n_correct_lesioned=n_les,
        **structural)


def _grid(conv, n_pass, *, route="wm_encoder_state", lam=0.25, **structural):
    """8 blocks for one convention; the first `n_pass` clear the 0.10 drop."""
    # 0.10 * 29571 = 2957.1 -> 2958 items passes, 2957 does not.
    out = []
    i = 0
    for st in WITNESSES:
        for sd in range(N_LESION_SEEDS):
            out.append(_blk(st, sd, conv, drop_items=(2958 if i < n_pass else 2957),
                            route=route, lam=lam, **structural))
            i += 1
    return out


def test_O3_frozen_constants():
    assert VALIDITY_CONVENTION_SCOPE == "SHARED"
    assert IMPLEMENTATION_FAILURE_SEMANTICS == "ABORT_RUN"
    assert N_BLOCKS == 8 and MIN_BLOCKS_WITH_DROP == 6
    assert TARGETED_MIN_DROP == 0.10


def test_O3_1_both_conventions_pass_shared_valid():
    v = decide_shared_validity(_grid("CANONICAL", 6), _grid("FREE_AR", 6))
    assert v.diagnostically_valid is True
    assert v.reason == ValidityReason.VALID
    assert v.canonical.n_blocks_with_drop == 6 and v.free_ar.n_blocks_with_drop == 6


def test_O3_2_canonical_passes_free_ar_fails_shared_invalid():
    v = decide_shared_validity(_grid("CANONICAL", 6), _grid("FREE_AR", 5))
    assert v.diagnostically_valid is False
    assert v.reason == ValidityReason.TARGET_ROUTE_DROP_FAILED_FREE_AR
    assert v.scientific_perturbation_validity_failure is True


def test_O3_3_canonical_fails_free_ar_passes_shared_invalid():
    v = decide_shared_validity(_grid("CANONICAL", 5), _grid("FREE_AR", 6))
    assert v.diagnostically_valid is False
    assert v.reason == ValidityReason.TARGET_ROUTE_DROP_FAILED_CANONICAL


def test_O3_4_both_fail_shared_invalid():
    v = decide_shared_validity(_grid("CANONICAL", 5), _grid("FREE_AR", 5))
    assert v.diagnostically_valid is False
    assert v.reason == ValidityReason.TARGET_ROUTE_DROP_FAILED_BOTH


def test_O3_5_exact_drop_equal_to_threshold_counts_as_pass():
    """A drop of exactly 0.10 passes. Verified with an N where the tie is exact."""
    n = 1000                       # 0.10 * 1000 == 100 exactly
    blocks = {}
    for conv in ("CANONICAL", "FREE_AR"):
        blocks[conv] = [
            BlockObservation(state_id=st, lesion_seed=sd,
                             route="wm_encoder_state", lam=0.25, convention=conv,
                             n_items=n, n_correct_intact=900, n_correct_lesioned=800)
            for st in WITNESSES for sd in range(N_LESION_SEEDS)]
    b = blocks["CANONICAL"][0]
    assert b.targeted_drop == Fraction(1, 10)
    assert b.criterion_1_targeted_drop() is True
    v = decide_shared_validity(blocks["CANONICAL"], blocks["FREE_AR"])
    assert v.diagnostically_valid is True

    # The float-subtraction form is exactly what the count form avoids: computing
    # the drop as (intact_accuracy - lesioned_accuracy) gives 0.09999999999999998
    # for the very same 900/1000 -> 800/1000 block, which fails `>= 0.10`.
    assert (900 / n) - (800 / n) == 0.09999999999999998
    assert ((900 / n) - (800 / n)) < 0.10          # the defect
    assert b.targeted_drop >= Fraction(1, 10)      # the frozen implementation


def test_O3_6_just_below_threshold_fails():
    n = 1000
    blocks = {}
    for conv in ("CANONICAL", "FREE_AR"):
        blocks[conv] = [
            BlockObservation(state_id=st, lesion_seed=sd,
                             route="wm_encoder_state", lam=0.25, convention=conv,
                             n_items=n, n_correct_intact=900, n_correct_lesioned=801)
            for st in WITNESSES for sd in range(N_LESION_SEEDS)]
    assert blocks["CANONICAL"][0].criterion_1_targeted_drop() is False
    v = decide_shared_validity(blocks["CANONICAL"], blocks["FREE_AR"])
    assert v.diagnostically_valid is False
    assert v.reason == ValidityReason.TARGET_ROUTE_DROP_FAILED_BOTH

    # on the canonical population 2957 items fails and 2958 passes
    assert _blk("W3_REP", 0, "CANONICAL", drop_items=2957).criterion_1_targeted_drop() is False
    assert _blk("W3_REP", 0, "CANONICAL", drop_items=2958).criterion_1_targeted_drop() is True


def test_O3_7_competence_failure_does_not_abort_the_experiment():
    """SCIENTIFIC_PERTURBATION_VALIDITY_FAILURE: this severity only, run continues."""
    v = decide_shared_validity(_grid("CANONICAL", 0), _grid("FREE_AR", 0))
    assert v.diagnostically_valid is False
    assert v.structural_ok is True
    assert v.scientific_perturbation_validity_failure is True
    # no exception was raised: the caller keeps going
    v2 = decide_shared_validity(_grid("CANONICAL", 8), _grid("FREE_AR", 8))
    assert v2.diagnostically_valid is True


@pytest.mark.parametrize("kw,marker", [
    (dict(untargeted_max_abs_delta=1e-12), ImplementationFailure.UNTARGETED_ROUTE_CHANGED),
    (dict(shared_params_unmutated=False), ImplementationFailure.SHARED_PARAM_MUTATED),
    (dict(checkpoint_identity_restored=False), ImplementationFailure.CHECKPOINT_NOT_RESTORED),
    (dict(matched_lesion_tensors_identical=False), ImplementationFailure.LESION_TENSOR_MISMATCH),
    (dict(max_abs_delta_c_ltm=1e-12), ImplementationFailure.DORSAL_GATE_NULL_VIOLATED),
    (dict(max_abs_delta_gate=1e-12), ImplementationFailure.DORSAL_GATE_NULL_VIOLATED),
])
def test_O3_8_9_10_11_structural_violations_abort(kw, marker):
    """O3-8/9/10/11: every structural violation aborts, with no denominator shrinking."""
    canon = _grid("CANONICAL", 8)
    free = _grid("FREE_AR", 8)
    canon[0] = _blk("W3_REP", 0, "CANONICAL", drop_items=2958, **kw)
    with pytest.raises(ImplementationFailureAbort) as ei:
        decide_shared_validity(canon, free)
    assert any(marker in f for f in ei.value.failures)
    # aborts even when competence would otherwise pass 8/8 -> never downgraded
    assert "no denominator shrinking" in str(ei.value)


def test_O3_structural_abort_takes_precedence_over_competence_failure():
    canon = _grid("CANONICAL", 0)          # competence would fail
    free = _grid("FREE_AR", 0)
    canon[3] = _blk("W3_REP", 3, "CANONICAL", drop_items=1, shared_params_unmutated=False)
    with pytest.raises(ImplementationFailureAbort):
        decide_shared_validity(canon, free)


def test_O3_dorsal_gate_null_is_vacuous_for_ventral():
    canon = _grid("CANONICAL", 8, route="ltm_encoder_state")
    free = _grid("FREE_AR", 8, route="ltm_encoder_state")
    canon[0] = _blk("W3_REP", 0, "CANONICAL", drop_items=2958,
                    route="ltm_encoder_state", max_abs_delta_c_ltm=0.5,
                    max_abs_delta_gate=0.3)
    v = decide_shared_validity(canon, free)      # must NOT abort
    assert v.diagnostically_valid is True


def test_O3_no_denominator_shrinking_incomplete_grid_refused():
    with pytest.raises(ValueError, match="No denominator shrinking"):
        decide_shared_validity(_grid("CANONICAL", 6)[:7], _grid("FREE_AR", 6))


def test_O3_validity_never_reads_the_fusion_contrast():
    fields = set(BlockObservation.__dataclass_fields__)
    for banned in ("native", "fixed05", "discordant", "delta_accuracy",
                   "errors_gained", "errors_recovered", "p_", "mcnemar"):
        assert not any(banned in f for f in fields), banned


# ====================================================================  O-4

def _w(deltas):
    return list(deltas)


def test_O4_frozen_constants_and_wiring():
    assert ROBUSTNESS_RULE_ID == "REPLICATED_SIGN_PLUS_MATERIALITY_NO_SIGNIFICANCE_TEST"
    assert FROZEN_ROBUSTNESS_RULE_ID == ROBUSTNESS_RULE_ID
    assert FROZEN_ROBUSTNESS_RULE is not None
    assert MATERIALITY_FLOOR == 0.002
    assert N_LESION_SEEDS == 4 and MIN_DIRECTIONAL_SEEDS == 3


def test_O4_1_robust_positive():
    v = frozen_robustness({"W3_REP": _w([0.01, 0.01, 0.01, -0.001]),
                           "W4_REP": _w([0.004, 0.004, 0.004, 0.004])})
    assert v.verdict == ROBUST_POSITIVE and v.direction == DIRECTION_POSITIVE
    assert v.robust is True


def test_O4_2_robust_negative():
    v = frozen_robustness({"W3_REP": _w([-0.01, -0.01, -0.01, 0.001]),
                           "W4_REP": _w([-0.004, -0.004, -0.004, -0.004])})
    assert v.verdict == ROBUST_NEGATIVE and v.direction == DIRECTION_NEGATIVE


def test_O4_3_one_witness_fails():
    v = frozen_robustness({"W3_REP": _w([0.01, 0.01, 0.01, 0.01]),
                           "W4_REP": _w([0.0001, 0.0001, 0.0001, 0.0001])})
    assert v.verdict == NOT_ROBUST


def test_O4_4_two_of_four_same_sign_only():
    v = frozen_robustness({"W3_REP": _w([0.01, 0.01, -0.0001, -0.0001]),
                           "W4_REP": _w([0.01, 0.01, -0.0001, -0.0001])})
    assert v.verdict == NOT_ROBUST


def test_O4_5_mean_just_below_materiality():
    """3/4 positive in each witness but one witness mean = +0.001999..."""
    below = [0.001999, 0.001999, 0.001999, 0.001999]
    v = frozen_robustness({"W3_REP": _w([0.01, 0.01, 0.01, 0.01]),
                           "W4_REP": _w(below)})
    assert v.verdict == NOT_ROBUST
    w4 = [s for s in v.witnesses if s.state_id == "W4_REP"][0]
    assert w4.direction == DIRECTION_NONE


def test_O4_6_mean_exactly_plus_materiality_passes():
    exact = [MATERIALITY_FLOOR] * 4
    s = witness_direction(exact, "W3_REP")
    assert s.direction == DIRECTION_POSITIVE
    v = frozen_robustness({"W3_REP": _w(exact), "W4_REP": _w(exact)})
    assert v.verdict == ROBUST_POSITIVE


def test_O4_7_mean_exactly_minus_materiality_passes():
    exact = [-MATERIALITY_FLOOR] * 4
    s = witness_direction(exact, "W3_REP")
    assert s.direction == DIRECTION_NEGATIVE
    v = frozen_robustness({"W3_REP": _w(exact), "W4_REP": _w(exact)})
    assert v.verdict == ROBUST_NEGATIVE


def test_O4_8_material_reversal_exactly_at_floor_blocks_positive():
    """Otherwise robust positive, but one seed has Delta == -0.002 exactly."""
    v = frozen_robustness({"W3_REP": _w([0.02, 0.02, 0.02, -MATERIALITY_FLOOR]),
                           "W4_REP": _w([0.02, 0.02, 0.02, 0.02])})
    assert v.verdict == NOT_ROBUST
    w3 = [s for s in v.witnesses if s.state_id == "W3_REP"][0]
    assert w3.material_reversal is True


def test_O4_9_material_reversal_exactly_at_floor_blocks_negative():
    v = frozen_robustness({"W3_REP": _w([-0.02, -0.02, -0.02, MATERIALITY_FLOOR]),
                           "W4_REP": _w([-0.02, -0.02, -0.02, -0.02])})
    assert v.verdict == NOT_ROBUST


def test_O4_10_sub_material_reversal_does_not_veto():
    """-0.001999... in a positive pattern does NOT trigger the material reversal veto."""
    v = frozen_robustness({"W3_REP": _w([0.02, 0.02, 0.02, -0.001999]),
                           "W4_REP": _w([0.02, 0.02, 0.02, -0.001999])})
    assert v.verdict == ROBUST_POSITIVE
    for s in v.witnesses:
        assert s.material_reversal is False


def test_O4_11_zero_delta_is_neutral():
    s = witness_direction([0.0, 0.02, 0.02, 0.02], "W3_REP")
    assert s.positive_seed_count == 3 and s.negative_seed_count == 0
    assert s.neutral_seed_count == 1 and s.direction == DIRECTION_POSITIVE
    # a zero seed counts toward neither direction, so 2 positive + 2 zero fails 3/4
    s2 = witness_direction([0.0, 0.0, 0.02, 0.02], "W3_REP")
    assert s2.positive_seed_count == 2 and s2.direction == DIRECTION_NONE


def test_O4_12_opposite_witnesses_not_robust_and_no_pooling_rescue():
    v = frozen_robustness({"W3_REP": _w([0.02, 0.02, 0.02, 0.02]),
                           "W4_REP": _w([-0.02, -0.02, -0.02, -0.02])})
    assert v.verdict == NOT_ROBUST
    # the pooled mean is exactly 0 here, but even a favourable pool cannot rescue:
    assert "no pooling" in v.reason


def test_O4_13_pooled_material_but_one_witness_below_floor():
    """Pooled data would look material; one witness mean < 0.002 -> NOT_ROBUST."""
    w3 = [0.05, 0.05, 0.05, 0.05]          # pooled mean would be ~0.0257
    w4 = [0.0015, 0.0015, 0.0015, 0.0015]  # witness mean below the floor
    pooled_mean = (sum(w3) + sum(w4)) / 8
    assert abs(pooled_mean) >= MATERIALITY_FLOOR       # pooling would pass
    v = frozen_robustness({"W3_REP": w3, "W4_REP": w4})
    assert v.verdict == NOT_ROBUST                      # the frozen rule does not


def test_O4_14_tiny_unanimous_deltas_below_floor():
    tiny = [0.0001] * 4
    v = frozen_robustness({"W3_REP": tiny, "W4_REP": tiny})
    assert v.verdict == NOT_ROBUST


def _imported_modules(module) -> set:
    """Top-level module names imported by `module`, from its AST."""
    import ast
    tree = ast.parse(inspect.getsource(module))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def _identifiers(module) -> set:
    """Every identifier USED in `module` — names, attributes and call targets.

    Scanning identifiers rather than raw source is what makes this check precise:
    the module's own constants contain the string
    "..._NO_SIGNIFICANCE_TEST", which a text scan would flag as a significance
    test when it declares exactly the opposite.
    """
    import ast
    tree = ast.parse(inspect.getsource(module))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            out.add(node.attr.lower())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name.lower())
            for a in node.args.args + node.args.kwonlyargs:
                out.add(a.arg.lower())
    return out


def test_O4_15_no_significance_test_code_path_exists():
    """The frozen rule has no p-value, alpha, bootstrap or multiplicity code path.

    Proven structurally: no statistical library is importable from the module, and
    no statistical identifier is used anywhere in it.
    """
    import gate_x_lesion.robustness as R

    # 1. No statistical library is reachable at all.
    allowed = {"__future__", "math", "dataclasses", "fractions", "typing"}
    assert _imported_modules(R) <= allowed, _imported_modules(R) - allowed

    # 2. No statistical identifier is used.
    idents = _identifiers(R)
    for banned in ("mcnemar", "binomtest", "binom", "ttest", "chi2", "pvalue",
                   "p_value", "bootstrap", "holm", "bonferroni", "alpha",
                   "correction", "n_boot", "stats", "scipy", "statistics"):
        assert banned not in idents, banned

    # 3. The public entry points accept no statistical parameter.
    for fn in (R.frozen_robustness, R.witness_direction,
               R.witness_direction_from_counts, R.frozen_robustness_from_counts):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"alpha", "p", "pvalue", "correction", "n_boot"}), fn

    # 4. The rule takes only Delta values.
    assert list(inspect.signature(R.frozen_robustness).parameters) == ["deltas_by_witness"]

    # 5. The declared constants say so.
    assert R.O4_TEST == "NONE" and R.O4_STATISTICAL_UNIT == "NONE"
    assert R.O4_ALPHA == "NOT_APPLICABLE" and R.O4_MULTIPLICITY == "NOT_APPLICABLE"
    assert R.ROBUSTNESS_RULE_ID.endswith("NO_SIGNIFICANCE_TEST")


def test_O4_requires_exactly_two_witnesses_and_four_seeds():
    with pytest.raises(ValueError):
        frozen_robustness({"W3_REP": [0.01] * 4})
    with pytest.raises(ValueError):
        frozen_robustness({"W3_REP": [0.01] * 3, "W4_REP": [0.01] * 4})


# ---------------------------------------------- exact integer-count equivalence

def test_O4_integer_threshold_uses_ceiling_not_rounding():
    """0.002 x 29571 = 59.142 -> 60 items, NOT 59. The rule is 0.002 accuracy."""
    assert 0.002 * N_ITEMS == pytest.approx(59.142)
    assert integer_threshold_for(N_ITEMS, 1) == 60
    assert integer_threshold_for(N_ITEMS, N_LESION_SEEDS) == 237   # ceil(236.568)
    # when the product IS an integer, the threshold is that value (comparison is >=)
    assert integer_threshold_for(1000, 1) == 2


def test_O4_count_path_matches_accuracy_path_at_the_boundary():
    """59 items is NOT material; 60 is. Equivalent to the >= 0.002 comparison."""
    s59 = witness_direction_from_counts([59] * 4, [0] * 4, N_ITEMS, "W3_REP")
    s60 = witness_direction_from_counts([60] * 4, [0] * 4, N_ITEMS, "W3_REP")
    assert s59.direction == DIRECTION_NONE           # 59/29571 = 0.001995... < 0.002
    assert s60.direction == DIRECTION_POSITIVE       # 60/29571 = 0.002029... >= 0.002
    assert 59 / N_ITEMS < 0.002 <= 60 / N_ITEMS

    v = frozen_robustness_from_counts(
        {"W3_REP": {"native": [60] * 4, "fixed05": [0] * 4},
         "W4_REP": {"native": [60] * 4, "fixed05": [0] * 4}}, N_ITEMS)
    assert v.verdict == ROBUST_POSITIVE


def test_O4_count_path_mean_boundary_sum_237():
    """Witness mean materiality: sum over the 4 seeds must reach ceil(0.002*4*N)=237."""
    below = witness_direction_from_counts([59, 59, 59, 59], [0] * 4, N_ITEMS)  # sum 236
    at = witness_direction_from_counts([60, 59, 59, 59], [0] * 4, N_ITEMS)     # sum 237
    assert sum([59, 59, 59, 59]) == 236 and sum([60, 59, 59, 59]) == 237
    assert below.direction == DIRECTION_NONE
    assert at.direction == DIRECTION_POSITIVE


# ============================================================  O-4 VETO / OUTCOMES

def _sev(lam, valid, direction, *, robust=True,
         reason=ValidityReason.VALID):
    from gate_x_lesion.robustness import RobustnessVerdict
    if direction == DIRECTION_NONE or not robust:
        rv = RobustnessVerdict(NOT_ROBUST, DIRECTION_NONE, (), "synthetic")
    else:
        rv = RobustnessVerdict(
            ROBUST_POSITIVE if direction == DIRECTION_POSITIVE else ROBUST_NEGATIVE,
            direction, (), "synthetic")
    return SeverityClassification(lam=lam, diagnostically_valid=valid,
                                  validity_reason=reason, robustness=rv)


def test_C1_single_robust_direction_remains_eligible():
    out = classify_convention([
        _sev(0.25, True, DIRECTION_POSITIVE),
        _sev(0.50, True, DIRECTION_NONE),
        _sev(1.00, True, DIRECTION_POSITIVE)])
    assert out["outcome"] == OUTCOME_A and out["veto_applied"] is False
    assert out["valid_severities"] == [0.25, 0.50, 1.00]


def test_C2_opposite_robust_signs_force_F():
    out = classify_convention([
        _sev(0.25, True, DIRECTION_POSITIVE),
        _sev(0.50, True, DIRECTION_NEGATIVE),
        _sev(1.00, True, DIRECTION_POSITIVE)])
    assert out["outcome"] == OUTCOME_F and out["veto_applied"] is True
    assert out["outcome"] not in (OUTCOME_A, OUTCOME_B, OUTCOME_D)
    assert out["valid_severities"] == [0.25, 0.50, 1.00]   # all remain visible


def test_C3_non_robust_opposite_does_not_veto():
    out = classify_convention([
        _sev(0.25, True, DIRECTION_POSITIVE),
        _sev(0.50, True, DIRECTION_NEGATIVE, robust=False)])
    assert out["outcome"] == OUTCOME_A and out["veto_applied"] is False


def test_C4_invalid_severity_cannot_veto():
    out = classify_convention([
        _sev(0.25, True, DIRECTION_POSITIVE),
        _sev(0.50, False, DIRECTION_NEGATIVE,
             reason=ValidityReason.TARGET_ROUTE_DROP_FAILED_FREE_AR)])
    assert out["outcome"] == OUTCOME_A and out["veto_applied"] is False
    assert out["valid_severities"] == [0.25]
    assert out["invalid_severities"] == {
        0.50: ValidityReason.TARGET_ROUTE_DROP_FAILED_FREE_AR}


def test_C5_concordant_joint():
    fam = {"CANONICAL": [_sev(0.25, True, DIRECTION_POSITIVE)],
           "FREE_AR": [_sev(0.25, True, DIRECTION_POSITIVE)]}
    out = classify_route_family(fam)
    assert out["CANONICAL_OUTCOME"] == OUTCOME_A
    assert out["FREE_AR_OUTCOME"] == OUTCOME_A
    assert out["JOINT_OUTCOME"] == concordant(OUTCOME_A) == "CONCORDANT_A"


def test_C6_canonical_A_free_ar_F_is_mixed_decoding():
    fam = {"CANONICAL": [_sev(0.25, True, DIRECTION_POSITIVE)],
           "FREE_AR": [_sev(0.25, True, DIRECTION_POSITIVE),
                       _sev(0.50, True, DIRECTION_NEGATIVE)]}
    out = classify_route_family(fam)
    assert out["CANONICAL_OUTCOME"] == OUTCOME_A
    assert out["FREE_AR_OUTCOME"] == OUTCOME_F
    assert out["JOINT_OUTCOME"] == JOINT_MIXED_DECODING
    # the favourable convention did not overwrite the heterogeneous one
    assert out["joint"]["canonical"] == OUTCOME_A
    assert out["joint"]["free_ar"] == OUTCOME_F


def test_C7_canonical_A_free_ar_B_is_mixed_decoding():
    fam = {"CANONICAL": [_sev(0.25, True, DIRECTION_POSITIVE)],
           "FREE_AR": [_sev(0.25, True, DIRECTION_NEGATIVE)]}
    out = classify_route_family(fam)
    assert out["JOINT_OUTCOME"] == JOINT_MIXED_DECODING


def test_C8_implementation_failure_aborts_before_any_joint_outcome():
    """No JOINT_OUTCOME may be emitted after an implementation failure."""
    canon = _grid("CANONICAL", 8)
    free = _grid("FREE_AR", 8)
    canon[0] = _blk("W3_REP", 0, "CANONICAL", drop_items=2958,
                    checkpoint_identity_restored=False)
    with pytest.raises(ImplementationFailureAbort):
        decide_shared_validity(canon, free)
    # the classifier is never reached; nothing partial is emitted


def test_joint_concordant_uses_the_more_specific_project_label():
    """CENTRAL §2: keep existing labels where more specific than A/B/D/F."""
    assert classify_joint(OUTCOME_F, OUTCOME_F)["joint_outcome"] == \
        "CONCORDANT_F_HETEROGENEOUS"
    assert concordant(OUTCOME_D) == "CONCORDANT_D"
    assert concordant(OUTCOME_B) == "CONCORDANT_B"


def test_direction_to_letter_mapping_is_frozen():
    """Delta = native - fixed05, so POSITIVE -> A (NATIVE better)."""
    assert DIRECTION_TO_LETTER[DIRECTION_POSITIVE] == OUTCOME_A
    assert DIRECTION_TO_LETTER[DIRECTION_NEGATIVE] == OUTCOME_B


def test_no_valid_severity_is_not_a_null():
    out = classify_convention([
        _sev(0.25, False, DIRECTION_NONE,
             reason=ValidityReason.TARGET_ROUTE_DROP_FAILED_BOTH)])
    assert out["outcome"] == OUTCOME_UNDETERMINED
    assert out["outcome"] != OUTCOME_D


# ==================================================  EXPECTED OUTPUT MANIFEST

def _good_summary():
    """A synthetic, complete run summary. No science; shaped data only."""
    import json as _json
    m = _json.load(open(os.path.join(
        ROOT, "paper_programme", "gate_x_lesion_recovery",
        "EXPECTED_OUTPUT_MANIFEST.json")))
    prov = {f: "x" for f in m["PROVENANCE"]["fields"]}
    prov.update(m["PROVENANCE"]["pinned_values"])
    comp = {f: "x" for f in m["RUN_COMPLETION"]["fields"]}
    comp.update(m["RUN_COMPLETION"]["required_values"])
    comp["implementation_failures"] = []
    validity = [{f: "x" for f in m["VALIDITY"]["fields"]} | {
        "route": r, "lambda": l, "validity_reason": "VALID"}
        for r in ("wm_encoder_state", "ltm_encoder_state")
        for l in (0.25, 0.5, 1.0)]
    raw = [{f: 0 for f in m["RAW_PAIRED_RESULTS"]["fields"]}
           for _ in range(m["RAW_PAIRED_RESULTS"]["expected_records"])]
    robust = [{"rule_id": m["ROBUSTNESS"]["rule_id"], "verdict": "NOT_ROBUST",
               "direction": "NONE", "reason": "synthetic"}]
    classification = [{f: "x" for f in m["CLASSIFICATION"]["fields"]} | {
        "CANONICAL_OUTCOME": "A", "FREE_AR_OUTCOME": "A",
        "JOINT_OUTCOME": "CONCORDANT_A"}]
    return {
        "PROVENANCE": prov, "VALIDITY": validity, "RAW_PAIRED_RESULTS": raw,
        "ROUTE_DIAGNOSTICS": [{}], "ROBUSTNESS": robust,
        "CLASSIFICATION": classification, "ITEM_LEVEL_AUDIT": [{}],
        "RUN_COMPLETION": comp,
    }


def test_output_manifest_accepts_a_complete_run():
    from scripts.gate_x_lesion.validate_output_manifest import validate
    assert validate(_good_summary())["ok"] is True


def test_output_manifest_fails_closed_on_missing_section():
    from scripts.gate_x_lesion.validate_output_manifest import (OutputIncomplete,
                                                                validate)
    for sec in ("PROVENANCE", "VALIDITY", "ROBUSTNESS", "CLASSIFICATION",
                "RUN_COMPLETION", "ITEM_LEVEL_AUDIT"):
        s = _good_summary()
        del s[sec]
        with pytest.raises(OutputIncomplete, match=sec):
            validate(s)


def test_output_manifest_fails_closed_on_missing_field_and_bad_pin():
    from scripts.gate_x_lesion.validate_output_manifest import (OutputIncomplete,
                                                                validate)
    s = _good_summary()
    del s["PROVENANCE"]["FINAL_CONTRACT_HASH"]
    with pytest.raises(OutputIncomplete, match="FINAL_CONTRACT_HASH"):
        validate(s)

    s = _good_summary()
    s["PROVENANCE"]["batch_size"] = 512
    with pytest.raises(OutputIncomplete, match="batch_size"):
        validate(s)

    s = _good_summary()
    s["RUN_COMPLETION"]["state_dict_unmutated"] = False
    with pytest.raises(OutputIncomplete, match="state_dict_unmutated"):
        validate(s)


def test_output_manifest_fails_closed_on_wrong_record_counts():
    from scripts.gate_x_lesion.validate_output_manifest import (OutputIncomplete,
                                                                validate)
    s = _good_summary()
    s["VALIDITY"] = s["VALIDITY"][:5]
    with pytest.raises(OutputIncomplete, match="VALIDITY"):
        validate(s)
    s = _good_summary()
    s["RAW_PAIRED_RESULTS"] = s["RAW_PAIRED_RESULTS"][:10]
    with pytest.raises(OutputIncomplete, match="RAW_PAIRED_RESULTS"):
        validate(s)


def test_output_manifest_rejects_classification_after_implementation_failure():
    from scripts.gate_x_lesion.validate_output_manifest import (OutputIncomplete,
                                                                validate)
    s = _good_summary()
    s["RUN_COMPLETION"]["implementation_failure_status"] = \
        "IMPLEMENTATION_FAILURE_SHARED_PARAM_MUTATED"
    with pytest.raises(OutputIncomplete, match="no classification may follow"):
        validate(s)


def test_output_manifest_rejects_significance_fields_in_robustness():
    from scripts.gate_x_lesion.validate_output_manifest import (OutputIncomplete,
                                                                validate)
    for banned in ("p_value", "alpha", "bootstrap_ci", "holm",
                   "pooled_witness_mean", "pooled_seed_count"):
        s = _good_summary()
        s["ROBUSTNESS"][0][banned] = 0.01
        with pytest.raises(OutputIncomplete, match="forbidden field"):
            validate(s)


def test_output_manifest_rejects_inconsistent_joint_label():
    from scripts.gate_x_lesion.validate_output_manifest import (OutputIncomplete,
                                                                validate)
    s = _good_summary()
    s["CLASSIFICATION"][0]["FREE_AR_OUTCOME"] = "B"      # differs -> MIXED_DECODING
    with pytest.raises(OutputIncomplete, match="inconsistent"):
        validate(s)

    s = _good_summary()
    s["CLASSIFICATION"][0]["JOINT_OUTCOME"] = "MIXED_DECODING"   # but letters agree
    with pytest.raises(OutputIncomplete, match="inconsistent"):
        validate(s)


def test_output_manifest_rejects_scientific_shard_in_quarantine():
    from scripts.gate_x_lesion.validate_output_manifest import (OutputIncomplete,
                                                                validate)
    paths = [f"scientific_execution/item_level/s{i}.tsv" for i in range(49)]
    paths.append("NOT_SCIENTIFIC_RESULT/item_level/s49.tsv")
    with pytest.raises(OutputIncomplete, match="quarantine"):
        validate(_good_summary(), shard_paths=paths)


# ==========================================================  CONTRACT HASH

def test_final_contract_hash_is_deterministic_and_excludes_results():
    from scripts.gate_x_lesion.compute_final_contract_hash import (
        CONTRACT_INPUTS, build_manifest, final_contract_hash)
    a = build_manifest()
    b = build_manifest()
    assert a == b
    assert final_contract_hash(a) == final_contract_hash(b)
    # canonical ordering
    lines = [l for l in a.splitlines()[1:] if l]
    assert [l.split("  ", 1)[1] for l in lines] == sorted(CONTRACT_INPUTS)
    # excluded by construction
    for banned in ("scientific_execution", "NOT_SCIENTIFIC_RESULT", ".git"):
        assert banned not in a, banned


def test_final_contract_hash_moves_when_a_rule_changes(tmp_path):
    """Editing a frozen rule file must change the contract hash."""
    from scripts.gate_x_lesion.compute_final_contract_hash import (
        build_manifest, final_contract_hash)
    p = os.path.join(ROOT, "gate_x_lesion", "robustness.py")
    before = final_contract_hash(build_manifest())
    original = open(p, "rb").read()
    try:
        open(p, "ab").write(b"\n# temporary probe\n")
        assert final_contract_hash(build_manifest()) != before
    finally:
        open(p, "wb").write(original)
    assert final_contract_hash(build_manifest()) == before
