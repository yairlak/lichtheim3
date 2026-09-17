"""Frozen output schemas (contract §12, §13, §16b)."""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Dict, List, Sequence

from ventral_directional_dose import ALPHA_KEYS, ALPHAS, CONVENTIONS
from ventral_interface.schema import tsv_bytes, write_tsv  # noqa: F401  (frozen deterministic writer)

SCHEMA_VERSION = "ventral-directional-dose-schema-v1"

IDENTITY_COLUMNS = [
    "witness_id", "state_id", "source_or_repaired", "seed", "source_u",
    "source_checkpoint_sha256", "repaired_head_sha256", "reconstructed_state_identity",
    "item_index", "lexical_identity", "canonical_C_index", "canonical_C_identity",
    "in_C_population", "target_phonology", "phoneme_length", "homophone_group",
    "homophone_group_size", "retrieved_index", "retrieved_lexical_identity", "retrieved_phonology",
    "C_contract_correct", "lexical_identity_correct", "phonology_correct",
]
CONTROL_COLUMNS = ([f"S{k}_{c}_exact_correct" for k in range(4) for c in CONVENTIONS]
                   + [f"prev_S1_rescue_{c}" for c in CONVENTIONS])
GEOMETRY_COLUMNS = ["shat_norm", "retrieved_raw_glove_norm", "cos_us_up", "theta_rad", "norm_r",
                    "dose_case", "live_shat_vs_preflight_shat_max_abs_dev"]
VECTOR_FIELDS = ["s_alpha_norm", "norm_rel_err", "cos_us_ualpha", "cos_ualpha_up",
                 "angle_from_native_rad", "angular_fraction", "angle_err_rad"]
DECODE_FIELDS = ["exact_correct", "predicted_phonology", "pred_length", "eos_emitted", "first_eos_step",
                 "eos_before_target_length", "eos_after_target_length", "terminated_by_cap",
                 "first_divergence_step", "transition_vs_S0", "prev_S1_rescue", "recovers_prev_S1_rescue"]


def alpha_columns() -> List[str]:
    cols = []
    for a in ALPHAS:
        k = ALPHA_KEYS[a]
        cols += [f"{k}_{f}" for f in VECTOR_FIELDS]
        for c in CONVENTIONS:
            cols += [f"{k}_{c}_{f}" for f in DECODE_FIELDS]
    return cols


ITEM_COLUMNS = IDENTITY_COLUMNS + CONTROL_COLUMNS + GEOMETRY_COLUMNS + alpha_columns()

TRANSITIONS = ("WRONG_TO_CORRECT", "CORRECT_TO_WRONG", "CORRECT_TO_CORRECT", "WRONG_TO_WRONG")


def _i(v) -> int:
    return int(v)


STRATA: Dict[str, Callable] = {
    "ALL_REPETITION_ITEMS": lambda r, c: True,
    "C_POPULATION": lambda r, c: _i(r["in_C_population"]) == 1,
    "NONCANONICAL_HOMOPHONE_MEMBERS": lambda r, c: _i(r["in_C_population"]) == 0,
    "NATIVE_LTM_WRONG": lambda r, c: _i(r[f"S0_{c}_exact_correct"]) == 0,
    "C_CORRECT_AND_NATIVE_LTM_WRONG": lambda r, c: _i(r["in_C_population"]) == 1
    and str(r["C_contract_correct"]) == "1" and _i(r[f"S0_{c}_exact_correct"]) == 0,
    "PREV_S1_RESCUES": lambda r, c: _i(r[f"prev_S1_rescue_{c}"]) == 1,
    "NEAR_COLLINEAR_CASES": lambda r, c: r["dose_case"] == "NEAR_COLLINEAR_NLERP",
}


def _cell(members: Sequence[Dict], a: float, c: str) -> Dict:
    k = f"{ALPHA_KEYS[a]}_{c}_"
    n = len(members)
    ex = sum(_i(r[k + "exact_correct"]) for r in members)
    t = dict.fromkeys(TRANSITIONS, 0)
    for r in members:
        t[r[k + "transition_vs_S0"]] += 1
    prev = [r for r in members if _i(r[k + "prev_S1_rescue"]) == 1]
    rec = sum(_i(r[k + "recovers_prev_S1_rescue"]) for r in prev)
    return {"denominator": n, "exact_count": ex, "exact_proportion": (ex / n) if n else None,
            **t, "previous_S1_rescues": len(prev), "previous_S1_rescues_recovered": rec,
            "recovery_fraction": (rec / len(prev)) if prev else None,
            "new_regressions": t["CORRECT_TO_WRONG"]}


def _controls(members: Sequence[Dict], c: str) -> Dict:
    return {"denominator": len(members),
            **{f"S{k}_exact_count": sum(_i(r[f"S{k}_{c}_exact_correct"]) for r in members) for k in range(4)}}


def summarize(rows: Sequence[Dict]) -> Dict:
    out: Dict = {}
    for sid in sorted({r["state_id"] for r in rows}):
        srows = [r for r in rows if r["state_id"] == sid]
        out[sid] = {}
        for c in CONVENTIONS:
            out[sid][c] = {}
            for sname, pred in STRATA.items():
                members = [r for r in srows if pred(r, c)]
                blk = {ALPHA_KEYS[a]: _cell(members, a, c) for a in ALPHAS}
                blk["immutable_controls"] = _controls(members, c)
                out[sid][c][sname] = blk
    return out


def paired(rows: Sequence[Dict]) -> Dict:
    by = {}
    for r in rows:
        by.setdefault(r["state_id"], {})[int(r["item_index"])] = r
    out: Dict = {}
    for w in ("W3", "W4"):
        A, B = by.get(f"{w}_SRC"), by.get(f"{w}_REP")
        if A is None or B is None:
            continue
        items = sorted(A)
        out[w] = {}
        for c in CONVENTIONS:
            out[w][c] = {}
            for a in ALPHAS:
                k = f"{ALPHA_KEYS[a]}_{c}_"

                def stats(S):
                    nf = [i for i in items if _i(S[i][f"S0_{c}_exact_correct"]) == 0]
                    resc = {i for i in nf if _i(S[i][k + "exact_correct"]) == 1}
                    reg = sum(1 for i in items if S[i][k + "transition_vs_S0"] == "CORRECT_TO_WRONG")
                    prev = [i for i in items if _i(S[i][k + "prev_S1_rescue"]) == 1]
                    recf = (sum(_i(S[i][k + "recovers_prev_S1_rescue"]) for i in prev) / len(prev)) if prev else None
                    return {"exact": sum(_i(S[i][k + "exact_correct"]) for i in items),
                            "native_failures": len(nf), "rescues": len(resc),
                            "rescue_fraction": (len(resc) / len(nf)) if nf else None,
                            "regressions": reg, "prev_S1_recovery_fraction": recf}, resc
                sa, ra = stats(A)
                sb, rb = stats(B)
                u = ra | rb
                trans = dict.fromkeys(TRANSITIONS, 0)
                for i in items:
                    x, y = _i(A[i][k + "exact_correct"]), _i(B[i][k + "exact_correct"])
                    trans[("CORRECT" if x else "WRONG") + "_TO_" + ("CORRECT" if y else "WRONG")] += 1
                out[w][c][ALPHA_KEYS[a]] = {
                    "SRC": sa, "REP": sb,
                    "rescue_set_jaccard": (len(ra & rb) / len(u)) if u else None,
                    "prev_S1_recovery_fraction_REP_minus_SRC":
                        (sb["prev_S1_recovery_fraction"] - sa["prev_S1_recovery_fraction"])
                        if sa["prev_S1_recovery_fraction"] is not None and sb["prev_S1_recovery_fraction"] is not None else None,
                    "item_transitions_SRC_to_REP": trans}
    return out


def alpha1_factorization(rows: Sequence[Dict]) -> Dict:
    """Contract §16b: alpha=1 (prototype direction + native norm) vs S1 and S3."""
    out: Dict = {}
    for sid in sorted({r["state_id"] for r in rows}):
        srows = [r for r in rows if r["state_id"] == sid]
        out[sid] = {}
        for c in CONVENTIONS:
            a1 = [_i(r[f"a100_{c}_exact_correct"]) for r in srows]
            s0 = [_i(r[f"S0_{c}_exact_correct"]) for r in srows]
            s1 = [_i(r[f"S1_{c}_exact_correct"]) for r in srows]
            s3 = [_i(r[f"S3_{c}_exact_correct"]) for r in srows]
            r_a1 = {i for i in range(len(srows)) if s0[i] == 0 and a1[i] == 1}
            r_s1 = {i for i in range(len(srows)) if s0[i] == 0 and s1[i] == 1}
            r_s3 = {i for i in range(len(srows)) if s0[i] == 0 and s3[i] == 1}
            out[sid][c] = {
                "n": len(srows),
                "exact": {"S0": sum(s0), "alpha1": sum(a1), "S1": sum(s1), "S3": sum(s3)},
                "alpha1_minus_S1_exact": sum(a1) - sum(s1),
                "alpha1_minus_S3_exact": sum(a1) - sum(s3),
                "alpha1_vs_S1_item_disagreements": sum(1 for x, y in zip(a1, s1) if x != y),
                "alpha1_correct_S1_wrong": sum(1 for x, y in zip(a1, s1) if x == 1 and y == 0),
                "alpha1_wrong_S1_correct": sum(1 for x, y in zip(a1, s1) if x == 0 and y == 1),
                "alpha1_vs_S3_item_disagreements": sum(1 for x, y in zip(a1, s3) if x != y),
                "alpha1_rescues": len(r_a1), "S1_rescues": len(r_s1), "S3_rescues": len(r_s3),
                "alpha1_rescues_also_S1_rescues": len(r_a1 & r_s1),
                "S1_rescues_also_alpha1_rescues_fraction": (len(r_a1 & r_s1) / len(r_s1)) if r_s1 else None,
                "alpha1_regressions": sum(1 for x, y in zip(s0, a1) if x == 1 and y == 0),
                "S1_regressions": sum(1 for x, y in zip(s0, s1) if x == 1 and y == 0),
                "S3_regressions": sum(1 for x, y in zip(s0, s3) if x == 1 and y == 0),
            }
    return out


def schema_descriptor() -> Dict:
    d = {"schema_version": SCHEMA_VERSION,
         "item_level_directional_dose.tsv": ITEM_COLUMNS,
         "summary_metrics_directional_dose.json": {
             "top_level_keys": ["schema_version", "contract_sha256", "provenance", "alphas",
                                "hard_stop_preconditions", "real_state_geometry", "gates",
                                "results", "paired", "alpha1_factorization"],
             "results": "state -> convention -> stratum -> {a025,a050,a075,a100,immutable_controls}",
             "strata": list(STRATA), "transitions": list(TRANSITIONS)}}
    d["descriptor_sha256"] = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return d
