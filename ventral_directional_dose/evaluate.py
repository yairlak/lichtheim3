"""Per-state real-state preflight (§15b) and scientific dose execution (§15, §19).

Preflight never decodes a scientific alpha (asserted).  Execution requires a passed preflight
object for the same state.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ventral_directional_dose import (
    ALPHA_KEYS, ALPHAS, CONVENTIONS, DECODE_BATCH, GEOMETRY_BATCH, TOL_ANGLE_RAD,
    TOL_ENDPOINT_COS, TOL_LIVE_SHAT, TOL_MONOTONE_RAD, TOL_NORM_REL)
from ventral_directional_dose.dose_math import (
    NEAR_COLLINEAR, ORDINARY, DoseHardStop, base_geometry, case_labels, vector_checks)
from ventral_directional_dose.supplier import DoseInjection, sem_to_h0_probe
from ventral_interface.conditions import cosine_top2
from ventral_interface.decode import NA, decode_condition, eos_fields, first_divergence
from ventral_interface.injection import state_dict_sha256, ventral_only
from scripts.naming_comprehension.frozen_probe import encode_all


class GateFailure(RuntimeError):
    def __init__(self, gate: str, report: Dict):
        super().__init__(f"GATE {gate} FAILED: {report}")
        self.gate, self.report = gate, report


def _require(name: str, rep: Dict, gates: Dict) -> None:
    gates[name] = rep
    if not rep.get("pass", False):
        raise GateFailure(name, rep)


def _phon(vocab, ids) -> str:
    return " ".join(vocab.itos[int(p)] for p in ids)


def _retrieval_top1(model, tr) -> List[int]:
    """Frozen historical rule (closed population.retrieval_all_rows)."""
    from ventral_interface.population import retrieval_all_rows
    return [int(x) for x in retrieval_all_rows(model, tr)["metrics"]["top1_idx"]]


@torch.inference_mode()
def preflight_state(model, tr, meta: Dict, controls: List[Dict[str, str]],
                    expected_params_sha256: Optional[str], device: str = "cpu",
                    retrieval_fn=_retrieval_top1) -> Dict:
    """DOSE-A(state), DOSE-I, DOSE-J(retrieval), geometry hard stops, DOSE-B.  Alpha 0 only."""
    model.eval()
    gates: Dict[str, Dict] = {}
    vocab, entries, bank_raw = tr.vocab, tr.entries, tr.bank_raw
    n = len(entries)
    params_before = state_dict_sha256(model)
    _require("DOSE-I_pre", {"params_sha256": params_before, "expected": expected_params_sha256,
                            "pass": expected_params_sha256 is None or params_before == expected_params_sha256},
             gates)
    # DOSE-A (state-level control identity)
    id_ok = (len(controls) == n
             and all(r["state_id"] == meta["state_id"] for r in controls)
             and all(r["reconstructed_state_identity"] == meta["reconstructed_state_identity"] for r in controls)
             and all(r["lexical_identity"] == entries[i].word for i, r in enumerate(controls)))
    _require("DOSE-A_state_controls", {"n_rows": len(controls), "n_entries": n, "pass": bool(id_ok)}, gates)
    # DOSE-J retrieval identity
    retrieved = [int(r["retrieved_index"]) for r in controls]
    top1 = retrieval_fn(model, tr)
    mism = sum(1 for a, b in zip(top1, retrieved) if a != b) + abs(len(top1) - len(retrieved))
    _require("DOSE-J_retrieval_identity", {"n_mismatch": mism, "pass": mism == 0}, gates)
    # geometry over all items with the decode batch shapes
    forms = [e.phonemes for e in entries]
    s_hat = encode_all(model, vocab, forms, device, GEOMETRY_BATCH)
    v = bank_raw[torch.as_tensor(retrieved, dtype=torch.long)]
    geom = base_geometry(s_hat, v)
    counts = {"n_items": n,
              "ZERO_SHAT": int(geom["zero_shat"].sum()),
              "ZERO_PROTOTYPE": int(geom["zero_proto"].sum()),
              "NEAR_ANTIPODAL": int(geom["near_antipodal"].sum()),
              "NEAR_COLLINEAR_NLERP": int(geom["near_collinear"].sum()),
              "ORDINARY": int(geom["ordinary"].sum())}
    desc = {"shat_norm_min": float(geom["ns"].min()), "prototype_norm_min": float(geom["nv"].min()),
            "cos_us_up_min": float(geom["d"].min()), "cos_us_up_max": float(geom["d"].max()),
            "norm_r_min": float(geom["norm_r"].min())}
    hard = {"NO_ZERO_SHAT": counts["ZERO_SHAT"] == 0, "NO_ZERO_PROTOTYPE": counts["ZERO_PROTOTYPE"] == 0,
            "NO_NEAR_ANTIPODAL": counts["NEAR_ANTIPODAL"] == 0}
    geometry = {"counts": counts, "descriptive_bounds": desc, "hard_stop_preconditions": hard,
                "ordinary_plus_collinear_equals_n": counts["ORDINARY"] + counts["NEAR_COLLINEAR_NLERP"] == n}
    for k, ok in hard.items():
        gates[k] = {"pass": ok, "count": counts[k.replace("NO_", "")]}
    failed = [k for k, ok in hard.items() if not ok]
    if failed:
        err = DoseHardStop(failed[0].replace("NO_", ""), counts[failed[0].replace("NO_", "")],
                           f"state {meta['state_id']}; all failed: {failed}")
        err.geometry, err.gates = geometry, gates
        raise err
    # DOSE-B: alpha = 0 through the NEW supplier reproduces immutable S0 item by item
    table = bank_raw[torch.as_tensor(retrieved, dtype=torch.long)]
    mism = {c: {"exact": 0, "phonology": 0} for c in CONVENTIONS}
    probe_bad = 0
    enc_dev = 0.0
    live_dev = 0.0
    with ventral_only(model) as guard:
        for lo in range(0, n, DECODE_BATCH):
            idx = list(range(lo, min(lo + DECODE_BATCH, n)))
            f = [entries[i].phonemes for i in idx]
            for c in CONVENTIONS:
                inj = DoseInjection(0.0, table)
                with sem_to_h0_probe(model, inj) as probe:
                    out = decode_condition(model, vocab, f, idx, inj, c, device)
                probe_bad += probe.n_mismatch + int(probe.n_calls == 0)
                enc_dev = max(enc_dev, out["encoder_step_dev"])
                live_dev = max(live_dev, float((out["live_s_hat"] - s_hat[lo:lo + len(idx)]).abs().max()))
                for k, i in enumerate(idx):
                    pred = out["preds"][k]
                    ok = int(pred == list(entries[i].phonemes))
                    mism[c]["exact"] += int(str(ok) != controls[i][f"S0_{c}_exact_correct"])
                    mism[c]["phonology"] += int(_phon(vocab, pred) != controls[i][f"S0_{c}_predicted_phonology"])
        guard_hits = len(guard.violations)
    total = sum(v for m in mism.values() for v in m.values())
    _require("DOSE-B_alpha0_reproduces_S0", {"mismatches": mism, "pass": total == 0}, gates)
    _require("DOSE-F_preflight_alpha0_path", {"sem_to_h0_mismatch_or_uncalled": probe_bad,
                                              "encoder_step_max_abs_dev": enc_dev,
                                              "pass": probe_bad == 0 and enc_dev == 0.0}, gates)
    _require("DOSE-G_preflight", {"guard_violations": guard_hits, "pass": guard_hits == 0}, gates)
    _require("DOSE-J_live_shat_preflight", {"max_abs_dev_live_vs_geometry_shat": live_dev,
                                            "tolerance": TOL_LIVE_SHAT, "pass": live_dev <= TOL_LIVE_SHAT}, gates)
    params_after = state_dict_sha256(model)
    _require("DOSE-I_post_preflight", {"before": params_before, "after": params_after,
                                       "pass": params_before == params_after}, gates)
    return {"state_id": meta["state_id"], "gates": gates, "geometry": geometry,
            "s_hat": s_hat, "retrieved": retrieved, "cases": case_labels(geom),
            "params_sha256": params_before}


@torch.inference_mode()
def execute_state(model, tr, meta: Dict, controls: List[Dict[str, str]], pre: Dict,
                  device: str = "cpu") -> Dict:
    """Scientific alphas x both conventions on all items.  Requires a passed preflight."""
    if pre.get("state_id") != meta["state_id"]:
        raise RuntimeError("HARD STOP: preflight object does not belong to this state")
    model.eval()
    vocab, entries, bank_raw = tr.vocab, tr.entries, tr.bank_raw
    n = len(entries)
    gates: Dict[str, Dict] = {}
    params_before = state_dict_sha256(model)
    if params_before != pre["params_sha256"]:
        raise GateFailure("DOSE-I", {"pass": False, "reason": "params changed since preflight"})
    retrieved = pre["retrieved"]
    table = bank_raw[torch.as_tensor(retrieved, dtype=torch.long)]
    s_pre = pre["s_hat"]
    res = {a: {c: [None] * n for c in CONVENTIONS} for a in ALPHAS}
    vec = {a: {} for a in ALPHAS}
    live_all = torch.zeros_like(s_pre)
    probe_bad = 0
    enc_dev = 0.0
    conv_vector_mismatch = 0
    with ventral_only(model) as guard:
        for lo in range(0, n, DECODE_BATCH):
            idx = list(range(lo, min(lo + DECODE_BATCH, n)))
            f = [entries[i].phonemes for i in idx]
            for a in ALPHAS:
                supplied = {}
                for c in CONVENTIONS:
                    inj = DoseInjection(a, table)
                    with sem_to_h0_probe(model, inj) as probe:
                        out = decode_condition(model, vocab, f, idx, inj, c, device)
                    probe_bad += probe.n_mismatch + int(probe.n_calls == 0)
                    enc_dev = max(enc_dev, out["encoder_step_dev"])
                    supplied[c] = inj.last_supplied
                    live_all[lo:lo + len(idx)] = out["live_s_hat"]
                    for k, i in enumerate(idx):
                        res[a][c][i] = (out["preds"][k], out["raw"][k])
                if not torch.equal(supplied["freear"], supplied["canonical"]):
                    conv_vector_mismatch += 1
                chk = vector_checks(live_all[lo:lo + len(idx)], table[idx], supplied["freear"], a)
                for key, t in chk.items():
                    vec[a].setdefault(key, []).append(t)
        guard_hits = len(guard.violations)
    for a in ALPHAS:
        vec[a] = {k: torch.cat(v) for k, v in vec[a].items()}
    # ---- gates on the actually supplied vectors
    live_dev = float((live_all - s_pre).abs().max())
    la1, _, _, _ = cosine_top2(live_all, model.ltm.semantic_bank)
    pa1, _, _, _ = cosine_top2(s_pre, model.ltm.semantic_bank)
    n_top1 = int((la1 != pa1).sum())
    _require("DOSE-J_live_shat", {"max_abs_dev": live_dev, "n_top1_changed": n_top1,
                                  "pass": live_dev <= TOL_LIVE_SHAT and n_top1 == 0}, gates)
    geom = base_geometry(live_all, table)
    try:
        from ventral_directional_dose.dose_math import assert_no_hard_stops
        assert_no_hard_stops(geom)
    except DoseHardStop as e:
        raise GateFailure("HARD_STOP_RUNTIME", {"pass": False, "kind": e.kind, "count": e.count})
    max_norm = max(float(vec[a]["norm_rel_err"].max()) for a in ALPHAS)
    _require("DOSE-C_norm_preservation", {"max_norm_rel_err": max_norm, "tolerance": TOL_NORM_REL,
                                          "pass": max_norm <= TOL_NORM_REL}, gates)
    endpoint_def = float((1.0 - vec[1.00]["cos_ualpha_up"]).max())
    _require("DOSE-D_alpha1_direction", {"max_1_minus_cos": endpoint_def, "tolerance": TOL_ENDPOINT_COS,
                                         "pass": endpoint_def <= TOL_ENDPOINT_COS}, gates)
    ordm = geom["ordinary"]
    colm = geom["near_collinear"]
    max_err = 0.0
    for a in ALPHAS:
        e = vec[a]["angle_err_rad"][ordm]
        if e.numel():
            max_err = max(max_err, float(e.max()))
    angles = torch.stack([torch.zeros(n, dtype=torch.float64)] + [vec[a]["angle_from_native_rad"] for a in ALPHAS])
    steps = angles[1:] - angles[:-1]
    mono_viol = int(((steps < -TOL_MONOTONE_RAD).any(dim=0)).sum())
    theta = vec[1.00]["theta_atan2"]
    col_excess = float((angles[:, colm] - theta[colm] - TOL_ANGLE_RAD).max()) if bool(colm.any()) else -1.0
    _require("DOSE-E_angular", {"ordinary_max_angle_err_rad": max_err, "tolerance": TOL_ANGLE_RAD,
                                "monotonicity_violations": mono_viol,
                                "near_collinear_max_excess_rad": col_excess,
                                "pass": max_err <= TOL_ANGLE_RAD and mono_viol == 0 and col_excess <= 0.0},
             gates)
    _require("DOSE-F_shared_path", {"sem_to_h0_mismatch_or_uncalled": probe_bad,
                                    "encoder_step_max_abs_dev": enc_dev,
                                    "freear_vs_canonical_supplied_vector_mismatch_batches": conv_vector_mismatch,
                                    "decoder": "ventral_interface.decode.decode_condition(route='ltm')",
                                    "pass": probe_bad == 0 and enc_dev == 0.0 and conv_vector_mismatch == 0}, gates)
    _require("DOSE-G_no_forbidden_access", {"guard_violations": guard_hits, "pass": guard_hits == 0}, gates)
    params_after = state_dict_sha256(model)
    _require("DOSE-I_parameters", {"before": params_before, "after": params_after,
                                   "pass": params_before == params_after}, gates)
    # ---- rows
    eos = vocab.eos_id
    from ventral_directional_dose.schema import ITEM_COLUMNS
    rows: List[Dict] = []
    cases = case_labels(geom)
    for i in range(n):
        ctrl = controls[i]
        e = entries[i]
        row = {k: ctrl[k] for k in ITEM_COLUMNS if k in ctrl}
        row.update({
            "shat_norm": float(geom["ns"][i]), "retrieved_raw_glove_norm": float(geom["nv"][i]),
            "cos_us_up": float(geom["d"][i]), "theta_rad": float(geom["theta"][i]),
            "norm_r": float(geom["norm_r"][i]), "dose_case": cases[i],
            "live_shat_vs_preflight_shat_max_abs_dev": float((live_all[i] - s_pre[i]).abs().max()),
        })
        for a in ALPHAS:
            ak = ALPHA_KEYS[a]
            for key in ("s_alpha_norm", "norm_rel_err", "cos_us_ualpha", "cos_ualpha_up",
                        "angle_from_native_rad", "angular_fraction", "angle_err_rad"):
                val = float(vec[a][key][i])
                row[f"{ak}_{key}"] = NA if val != val else val
            for c in CONVENTIONS:
                pred, raw = res[a][c][i]
                ok = int(pred == list(e.phonemes))
                s0 = int(ctrl[f"S0_{c}_exact_correct"])
                p = f"{ak}_{c}_"
                row[p + "exact_correct"] = ok
                row[p + "predicted_phonology"] = _phon(vocab, pred)
                for kk, vv in eos_fields(raw, e.phonemes, eos, c).items():
                    row[p + kk] = vv
                fd = first_divergence(raw, e.phonemes, eos)
                row[p + "first_divergence_step"] = NA if fd is None else fd
                row[p + "transition_vs_S0"] = ("CORRECT" if s0 else "WRONG") + "_TO_" + ("CORRECT" if ok else "WRONG")
                prev = int(ctrl[f"prev_S1_rescue_{c}"])
                row[p + "prev_S1_rescue"] = prev
                row[p + "recovers_prev_S1_rescue"] = ok if prev else NA
        rows.append(row)
    return {"rows": rows, "gates": gates,
            "geometry_case_counts": {ORDINARY: cases.count(ORDINARY), NEAR_COLLINEAR: cases.count(NEAR_COLLINEAR)}}
