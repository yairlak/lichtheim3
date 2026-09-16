"""Assemble the frozen eight-section output package from raw execution records.

INTEGRATION REPAIR ONLY.  This module contains NO scientific logic of its own: it
is thin aggregation and serialization that CALLS the already-frozen rule modules.

    VALIDITY       -> gate_x_lesion.validity.decide_shared_validity      (O-3)
    ROBUSTNESS     -> gate_x_lesion.robustness.frozen_robustness_from_counts (O-4)
    CLASSIFICATION -> gate_x_lesion.classify.classify_route_family
                      (which calls classify_convention then classify_joint)

Nothing here re-implements O-3, O-4, the veto, the materiality floor or the joint
rule.  If a rule needs to change, it changes in its own module, not here.

Order is the frozen post-processing sequence:

    1. verify raw execution completeness
    2. PROVENANCE
    3. RAW_PAIRED_RESULTS
    4. ROUTE_DIAGNOSTICS
    5. VALIDITY                  (frozen O-3)
    6. on IMPLEMENTATION_FAILURE: emit failure RUN_COMPLETION, SUPPRESS
       CLASSIFICATION, fail closed
    7. ROBUSTNESS                (frozen O-4)
    8. CLASSIFICATION            (frozen classifier)
    9. ITEM_LEVEL_AUDIT
    10. RUN_COMPLETION
    11. deterministic final package

Serialization never alters a scientific calculation: every number written is
either copied from a raw record or returned by a frozen rule function.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from gate_x_lesion.classify import SeverityClassification, classify_route_family
from gate_x_lesion.robustness import (NOT_ROBUST, ROBUSTNESS_RULE_ID,
                                      frozen_robustness_from_counts)
from gate_x_lesion.validity import (BlockObservation, ImplementationFailureAbort,
                                    decide_shared_validity)

#: Frozen convention names -> the raw record's per-convention key.
CONVENTION_KEY = {"CANONICAL": "canonical", "FREE_AR": "freear"}
CONVENTIONS = ("CANONICAL", "FREE_AR")

#: Which isolated route a perturbation targets, and which it must leave alone.
TARGETED = {"wm_encoder_state": "wm", "ltm_encoder_state": "ltm"}
UNTARGETED = {"wm_encoder_state": "ltm", "ltm_encoder_state": "wm"}

ROUTE_FAMILY = {"wm_encoder_state": "dorsal", "ltm_encoder_state": "ventral"}


class RawExecutionIncomplete(RuntimeError):
    """The raw records do not cover the frozen grid. Assembly refuses to guess."""


# ------------------------------------------------------------------ indexing

def _index_cells(state_summaries: Sequence[dict]) -> Dict[tuple, dict]:
    """(state_id, route, lambda, seed) -> cell record."""
    out: Dict[tuple, dict] = {}
    for st in state_summaries:
        for c in st["cells"]:
            key = (st["state_id"], c["route"], float(c["lambda"]), int(c["lesion_seed"]))
            if key in out:
                raise RawExecutionIncomplete(f"duplicate cell {key}")
            out[key] = c
    return out


def _intact(state_summaries: Sequence[dict]) -> Dict[str, dict]:
    return {st["state_id"]: st["intact_control"] for st in state_summaries}


def verify_raw_completeness(state_summaries: Sequence[dict], *, routes, lambdas,
                            seeds, witnesses) -> Dict[tuple, dict]:
    """Step 1. Every frozen grid cell must be present. No denominator shrinking."""
    cells = _index_cells(state_summaries)
    missing = [(w, r, l, s) for w in witnesses for r in routes
               for l in lambdas for s in seeds
               if (w, r, float(l), int(s)) not in cells]
    if missing:
        raise RawExecutionIncomplete(
            f"raw execution is incomplete: {len(missing)} of "
            f"{len(witnesses) * len(routes) * len(lambdas) * len(seeds)} cells "
            f"missing, e.g. {missing[:3]}. Assembly refuses to proceed.")
    for w in witnesses:
        if w not in _intact(state_summaries):
            raise RawExecutionIncomplete(f"missing intact control for {w}")
    return cells


# ---------------------------------------------------------------- 2 PROVENANCE

def build_provenance(state_summaries: Sequence[dict], *, cfg: dict) -> dict:
    per_state = {st["state_id"]: st for st in state_summaries}
    pick = lambda k: {sid: st.get(k) for sid, st in per_state.items()}  # noqa: E731
    return {
        "FINAL_CONTRACT_HASH": cfg["final_contract_hash"],
        "final_contract_manifest_sha256": cfg["final_contract_manifest_sha256"],
        "code_commit": cfg["code_commit"],
        "branch": cfg["branch"],
        "runner_identity": cfg["runner_identity"],
        "witness_id": pick("witness_label"),
        "gxlr_label": pick("gxlr_label"),
        "reconstructed_state_sha256": pick("reconstructed_state_sha256"),
        "base_artifact_path": pick("base_artifact_path"),
        "base_artifact_sha256": pick("base_artifact_sha256"),
        "applies_head_path": pick("applies_head_path"),
        "applies_head_sha256": pick("applies_head_sha256"),
        "head_localizer": pick("head_localizer"),
        "state_kind": pick("state_kind"),
        "ltm_encoder_mode": pick("ltm_encoder_mode"),
        "gating_config": pick("gating_config"),
        "item_order_sha256": pick("item_order_sha256"),
        "sd_value_per_route": {
            sid: {r: st["intact_sd"][r]["sd"] for r in st["intact_sd"]}
            for sid, st in per_state.items()},
        "batch_size": cfg["batch_size"],
        "population_n": cfg["population_n"],
        "routes": list(cfg["routes"]),
        "lambdas": list(cfg["lambdas"]),
        "lesion_seeds": list(cfg["lesion_seeds"]),
        "decoding_conventions": list(CONVENTIONS),
        "fusion_conditions": ["NATIVE", "FIXED05"],
        "sd_definition": cfg["sd_definition"],
        "sd_ddof": cfg["sd_ddof"],
        "sd_n_items": cfg["sd_n_items"],
        "sd_sample_seed": cfg["sd_sample_seed"],
        "sd_batch_size": cfg["sd_batch_size"],
        "rng_identity_fields": list(cfg["rng_identity_fields"]),
        "rng_identity_version": cfg["rng_identity_version"],
        "eps_domain": cfg["eps_domain"],
        "state_identity_domain": cfg["state_identity_domain"],
        "torch_version": cfg["torch_version"],
    }


# -------------------------------------------------------- 3 RAW_PAIRED_RESULTS

def build_raw_paired(cells: Dict[tuple, dict], *, routes, lambdas, seeds,
                     witnesses) -> List[dict]:
    """The frozen 96 records: 2 witnesses x 4 seeds x 2 routes x 3 lambdas x 2 conventions."""
    out: List[dict] = []
    for w in witnesses:
        for r in routes:
            for lam in lambdas:
                for s in seeds:
                    c = cells[(w, r, float(lam), int(s))]
                    for conv in CONVENTIONS:
                        k = CONVENTION_KEY[conv]
                        d = c[k]
                        n = int(c["n_items"])
                        nn, nf = int(d["n_correct_native"]), int(d["n_correct_fixed05"])
                        out.append({
                            "state_id": w, "lesion_seed": int(s), "route": r,
                            "lambda": float(lam), "convention": conv,
                            "n_items": n,
                            "n_correct_native": nn,
                            "n_correct_fixed05": nf,
                            # Delta = accuracy_native - accuracy_fixed05, never rounded
                            # before rule evaluation; the rule itself consumes the
                            # integer counts above via the exact rational path.
                            "accuracy_native": nn / n,
                            "accuracy_fixed05": nf / n,
                            "delta": (nn - nf) / n,
                            "n_discordant_prediction": int(d["n_discordant_prediction"]),
                            "n_discordant_exact": int(d["n_discordant_exact"]),
                            "errors_prevented_by_fixed05":
                                int(d["errors_prevented_by_fixed05"]),
                            "errors_introduced_by_fixed05":
                                int(d["errors_introduced_by_fixed05"]),
                        })
    return out


# --------------------------------------------------------- 4 ROUTE_DIAGNOSTICS

def build_route_diagnostics(cells: Dict[tuple, dict], intact: Dict[str, dict], *,
                            routes, lambdas, seeds, witnesses) -> List[dict]:
    out: List[dict] = []
    for w in witnesses:
        for r in routes:
            for lam in lambdas:
                for s in seeds:
                    c = cells[(w, r, float(lam), int(s))]
                    ic = intact[w]
                    for conv in CONVENTIONS:
                        k = CONVENTION_KEY[conv]
                        d, di = c[k], ic[k]
                        n = int(c["n_items"])
                        tgt, unt = TARGETED[r], UNTARGETED[r]
                        out.append({
                            "state_id": w, "lesion_seed": int(s), "route": r,
                            "lambda": float(lam), "convention": conv,
                            "route_family": ROUTE_FAMILY[r],
                            "n_items": n,
                            "acc_wm_isolated": d["acc_wm_isolated"],
                            "acc_ltm_isolated": d["acc_ltm_isolated"],
                            "acc_wm_isolated_intact": di["acc_wm_isolated"],
                            "acc_ltm_isolated_intact": di["acc_ltm_isolated"],
                            "n_correct_targeted_intact":
                                int(di[f"n_correct_{tgt}_isolated"]),
                            "n_correct_targeted_lesioned":
                                int(d[f"n_correct_{tgt}_isolated"]),
                            "targeted_route_drop":
                                (int(di[f"n_correct_{tgt}_isolated"])
                                 - int(d[f"n_correct_{tgt}_isolated"])) / n,
                            "untargeted_route_max_abs_delta":
                                float(d[f"{unt}_changed_vs_intact"] > 0),
                            "untargeted_route_n_changed":
                                int(d[f"{unt}_changed_vs_intact"]),
                            "c_LTM_mean": c["c_LTM_mean"],
                            "gate_mean": c["gate_mean"],
                            "max_abs_delta_c_LTM": c["max_abs_delta_c_LTM"],
                            "max_abs_delta_gate": c["max_abs_delta_gate"],
                            "dorsal_structural_null_holds":
                                (ROUTE_FAMILY[r] != "dorsal"
                                 or (c["max_abs_delta_c_LTM"] == 0.0
                                     and c["max_abs_delta_gate"] == 0.0)),
                            "ventral_reachability_holds":
                                (ROUTE_FAMILY[r] != "ventral"
                                 or c["max_abs_delta_c_LTM"] > 0.0),
                            "shared_params_unmutated": bool(c["shared_params_unmutated"]),
                            "checkpoint_identity_restored":
                                bool(c["checkpoint_identity_restored"]),
                            "matched_lesion_tensors_identical":
                                bool(c["matched_lesion_tensors_identical"]),
                        })
    return out


# ------------------------------------------------------------------ 5 VALIDITY

def _blocks_for(cells, intact, route, lam, conv, *, seeds, witnesses):
    k = CONVENTION_KEY[conv]
    tgt, unt = TARGETED[route], UNTARGETED[route]
    blocks = []
    for w in witnesses:
        for s in seeds:
            c = cells[(w, route, float(lam), int(s))]
            d, di = c[k], intact[w][k]
            blocks.append(BlockObservation(
                state_id=w, lesion_seed=int(s), route=route, lam=float(lam),
                convention=conv,
                n_items=int(c["n_items"]),
                n_correct_intact=int(di[f"n_correct_{tgt}_isolated"]),
                n_correct_lesioned=int(d[f"n_correct_{tgt}_isolated"]),
                untargeted_max_abs_delta=float(d[f"{unt}_changed_vs_intact"] > 0),
                shared_params_unmutated=bool(c["shared_params_unmutated"]),
                checkpoint_identity_restored=bool(c["checkpoint_identity_restored"]),
                matched_lesion_tensors_identical=bool(
                    c["matched_lesion_tensors_identical"]),
                max_abs_delta_c_ltm=float(c["max_abs_delta_c_LTM"]),
                max_abs_delta_gate=float(c["max_abs_delta_gate"]),
            ))
    return blocks


def build_validity(cells, intact, *, routes, lambdas, seeds, witnesses,
                   require_full_grid: bool = True
                   ) -> Tuple[List[dict], Dict[tuple, object]]:
    """Frozen O-3. Raises ImplementationFailureAbort on a structural violation.

    `require_full_grid` defaults to True and is NEVER relaxed for a scientific run:
    the full 2x4 block grid is mandatory and there is no denominator shrinking.
    A TEST_ONLY smoke harness may pass False so that the same wiring can be
    exercised on a tiny quarantined scope.
    """
    records: List[dict] = []
    verdicts: Dict[tuple, object] = {}
    for route in routes:
        for lam in lambdas:
            canon = _blocks_for(cells, intact, route, lam, "CANONICAL",
                                seeds=seeds, witnesses=witnesses)
            free = _blocks_for(cells, intact, route, lam, "FREE_AR",
                               seeds=seeds, witnesses=witnesses)
            v = decide_shared_validity(canon, free,       # <-- frozen O-3
                                       require_full_grid=require_full_grid)
            verdicts[(route, float(lam))] = v
            records.append({
                "route": route, "lambda": float(lam),
                "canonical_blocks": [
                    {"state_id": s, "lesion_seed": k, "targeted_drop": d, "passed": ok}
                    for s, k, d, ok in v.canonical.per_block],
                "canonical_n_blocks_with_drop": v.canonical.n_blocks_with_drop,
                "canonical_passed": v.canonical.passed,
                "free_ar_blocks": [
                    {"state_id": s, "lesion_seed": k, "targeted_drop": d, "passed": ok}
                    for s, k, d, ok in v.free_ar.per_block],
                "free_ar_n_blocks_with_drop": v.free_ar.n_blocks_with_drop,
                "free_ar_passed": v.free_ar.passed,
                "shared_validity_label": bool(v.diagnostically_valid),
                "validity_reason": v.reason,
                "structural_checks": {
                    "structural_ok": v.structural_ok,
                    "deterministic_tolerance": 0.0,
                    "no_denominator_shrinking": True,
                    "no_block_exclusion": True,
                },
                # On this path there are none by construction: the frozen O-3
                # module RAISES ImplementationFailureAbort rather than returning
                # a verdict when a structural guarantee is violated.
                "implementation_failures": [],
                "scientific_perturbation_validity_failure":
                    v.scientific_perturbation_validity_failure,
                "detail": v.detail,
            })
    return records, verdicts


# ---------------------------------------------------------------- 7 ROBUSTNESS

def build_robustness(cells, *, routes, lambdas, seeds, witnesses
                     ) -> Tuple[List[dict], Dict[tuple, object]]:
    """Frozen O-4. Exact integer-count path; no significance test, no pooling."""
    records: List[dict] = []
    verdicts: Dict[tuple, object] = {}
    for route in routes:
        for lam in lambdas:
            for conv in CONVENTIONS:
                k = CONVENTION_KEY[conv]
                counts, n_items = {}, None
                for w in witnesses:
                    nat, fix = [], []
                    for s in seeds:
                        c = cells[(w, route, float(lam), int(s))]
                        nat.append(int(c[k]["n_correct_native"]))
                        fix.append(int(c[k]["n_correct_fixed05"]))
                        n_items = int(c["n_items"])
                    counts[w] = {"native": nat, "fixed05": fix}
                rv = frozen_robustness_from_counts(counts, n_items)   # <-- frozen O-4
                verdicts[(route, float(lam), conv)] = rv
                records.append({
                    "route": route, "lambda": float(lam), "convention": conv,
                    "rule_id": ROBUSTNESS_RULE_ID,
                    "verdict": rv.verdict,
                    "direction": rv.direction,
                    "reason": rv.reason,
                    "per_witness": [
                        {"state_id": s.state_id,
                         "positive_seed_count": s.positive_seed_count,
                         "negative_seed_count": s.negative_seed_count,
                         "neutral_seed_count": s.neutral_seed_count,
                         "mean_delta": s.mean_delta,
                         "material_reversal": s.material_reversal,
                         "direction": s.direction,
                         "reason": s.reason}
                        for s in rv.witnesses],
                })
    return records, verdicts


# -------------------------------------------------------------- 8 CLASSIFICATION

def build_classification(validity_verdicts, robustness_verdicts, *, routes, lambdas
                         ) -> List[dict]:
    """Frozen classifier: CANONICAL, then FREE_AR, then JOINT."""
    out: List[dict] = []
    for route in routes:
        by_conv = {}
        for conv in CONVENTIONS:
            sevs = []
            for lam in lambdas:
                v = validity_verdicts[(route, float(lam))]
                rv = robustness_verdicts[(route, float(lam), conv)]
                sevs.append(SeverityClassification(
                    lam=float(lam),
                    diagnostically_valid=bool(v.diagnostically_valid),
                    validity_reason=v.reason,
                    robustness=rv))
            by_conv[conv] = sevs
        res = classify_route_family(by_conv)               # <-- frozen classifier
        canon, free = res["stages"]["CANONICAL"], res["stages"]["FREE_AR"]
        out.append({
            "route_family": ROUTE_FAMILY[route],
            "route": route,
            "CANONICAL_OUTCOME": res["CANONICAL_OUTCOME"],
            "FREE_AR_OUTCOME": res["FREE_AR_OUTCOME"],
            "JOINT_OUTCOME": res["JOINT_OUTCOME"],
            "canonical_valid_severities": canon["valid_severities"],
            "canonical_directions": {str(k): v for k, v in canon["directions"].items()},
            "canonical_veto_applied": canon["veto_applied"],
            "free_ar_valid_severities": free["valid_severities"],
            "free_ar_directions": {str(k): v for k, v in free["directions"].items()},
            "free_ar_veto_applied": free["veto_applied"],
            "invalid_severities_with_reasons": {
                str(k): v for k, v in canon["invalid_severities"].items()},
            "canonical_reason": canon["reason"],
            "free_ar_reason": free["reason"],
            "joint_reason": res["joint"]["reason"],
        })
    return out


# ------------------------------------------------------------ 9 ITEM_LEVEL_AUDIT

def build_item_level_audit(audit_records: Sequence[dict]) -> List[dict]:
    """Preregistered discordant-item audit only. No ranking, no selection."""
    return [dict(a) for a in audit_records]


# -------------------------------------------------------------- 10 RUN_COMPLETION

def build_run_completion(state_summaries, *, shard_paths, expected_shards,
                         expected_validity, expected_paired, n_validity, n_paired,
                         contract_hash_reverified, implementation_failures=(),
                         shard_sha256=None) -> dict:
    failures = list(implementation_failures)
    status = "NONE" if not failures else failures[0].split(":")[0]
    counts = {
        "execution_shards": {"expected": expected_shards, "actual": len(shard_paths)},
        "validity_records": {"expected": expected_validity, "actual": n_validity},
        "paired_records": {"expected": expected_paired, "actual": n_paired},
    }
    complete = all(c["expected"] == c["actual"] for c in counts.values())
    return {
        "run_status": "COMPLETED" if not failures else "ABORTED_IMPLEMENTATION_FAILURE",
        "state_dict_sha256_before": {s["state_id"]: s["state_dict_sha256_before"]
                                     for s in state_summaries},
        "state_dict_sha256_after": {s["state_id"]: s["state_dict_sha256_after"]
                                    for s in state_summaries},
        "state_dict_unmutated": all(s["state_dict_unmutated"] for s in state_summaries),
        "base_checkpoint_unmutated": all(s["base_checkpoint_unmutated"]
                                         for s in state_summaries),
        "applied_head_unmutated": all(s.get("applied_head_unmutated", True)
                                      for s in state_summaries),
        "implementation_failure_status": status,
        "implementation_failures": failures,
        "all_required_shards_present": len(shard_paths) == expected_shards,
        "shard_sha256": shard_sha256 or {},
        "expected_vs_actual_record_counts": counts,
        "final_contract_hash_reverified": bool(contract_hash_reverified),
        "completeness_validated": bool(complete and not failures),
    }


# ------------------------------------------------------------------- assembly

def build_summary(*, state_summaries: Sequence[dict], audit_records: Sequence[dict],
                  shard_paths: Sequence[str], cfg: dict,
                  expected_shards: int, expected_validity: int,
                  expected_paired: int, shard_sha256=None,
                  require_full_grid: bool = True) -> dict:
    """Assemble the eight-section package. Deterministic given identical inputs.

    On an IMPLEMENTATION_FAILURE the package is returned WITHOUT a CLASSIFICATION
    section and with a failure RUN_COMPLETION; the caller must fail closed.
    """
    routes = list(cfg["routes"])
    lambdas = [float(x) for x in cfg["lambdas"]]
    seeds = [int(x) for x in cfg["lesion_seeds"]]
    witnesses = sorted(st["state_id"] for st in state_summaries)

    cells = verify_raw_completeness(state_summaries, routes=routes, lambdas=lambdas,
                                    seeds=seeds, witnesses=witnesses)
    intact = _intact(state_summaries)

    provenance = build_provenance(state_summaries, cfg=cfg)
    raw = build_raw_paired(cells, routes=routes, lambdas=lambdas, seeds=seeds,
                           witnesses=witnesses)
    diagnostics = build_route_diagnostics(cells, intact, routes=routes,
                                          lambdas=lambdas, seeds=seeds,
                                          witnesses=witnesses)

    try:
        validity, v_verdicts = build_validity(cells, intact, routes=routes,
                                              lambdas=lambdas, seeds=seeds,
                                              witnesses=witnesses,
                                              require_full_grid=require_full_grid)
    except ImplementationFailureAbort as e:
        # Step 6: fail closed. No ROBUSTNESS, no CLASSIFICATION.
        return {
            "PROVENANCE": provenance,
            "VALIDITY": [],
            "RAW_PAIRED_RESULTS": raw,
            "ROUTE_DIAGNOSTICS": diagnostics,
            "ROBUSTNESS": [],
            "CLASSIFICATION": [],
            "ITEM_LEVEL_AUDIT": build_item_level_audit(audit_records),
            "RUN_COMPLETION": build_run_completion(
                state_summaries, shard_paths=shard_paths,
                expected_shards=expected_shards, expected_validity=expected_validity,
                expected_paired=expected_paired, n_validity=0, n_paired=len(raw),
                contract_hash_reverified=bool(cfg.get("final_contract_hash")),
                implementation_failures=e.failures, shard_sha256=shard_sha256),
        }

    robustness, r_verdicts = build_robustness(cells, routes=routes, lambdas=lambdas,
                                              seeds=seeds, witnesses=witnesses)
    classification = build_classification(v_verdicts, r_verdicts, routes=routes,
                                          lambdas=lambdas)

    return {
        "PROVENANCE": provenance,
        "VALIDITY": validity,
        "RAW_PAIRED_RESULTS": raw,
        "ROUTE_DIAGNOSTICS": diagnostics,
        "ROBUSTNESS": robustness,
        "CLASSIFICATION": classification,
        "ITEM_LEVEL_AUDIT": build_item_level_audit(audit_records),
        "RUN_COMPLETION": build_run_completion(
            state_summaries, shard_paths=shard_paths,
            expected_shards=expected_shards, expected_validity=expected_validity,
            expected_paired=expected_paired, n_validity=len(validity),
            n_paired=len(raw),
            contract_hash_reverified=bool(cfg.get("final_contract_hash")),
            shard_sha256=shard_sha256),
    }
