"""Per-state evaluation for `--execute`, with the frozen gate order (contract §7).

Order, hard-stopping on the first failure:
  0  parameter-state hash recorded
  R  historical C reproduction          A  C-correct => lexical identity
  B  S1 == S2 when rows are equal        D  S3 direction preserved
  C  matched Naming path == historical Naming (per item) and == archived count
  -- S0..S3 x {freear, canonical} decoding (isolated ventral route only) --
  SHAT  live s_hat == retrieval s_hat within tolerance, identical top-1
  H  native S0 reproduces archived LTM repetition error counts
  E  runtime downstream identity: no guard hit, params unchanged, encoder
     bitwise-stable across AR steps, recorded tokens == historical predictions

`synthetic=True` exists ONLY for unit tests on a randomly initialised toy model:
archived-value gates (R count, C count, H) are then reported NOT_APPLICABLE and the
result is marked NON_SCIENTIFIC.  The driver never sets it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from scripts.naming_comprehension.train_tasks import evaluate_naming
from ventral_interface import CONDITIONS, CONVENTIONS
from ventral_interface.conditions import (
    cosine_top2, gate_b, gate_d, radial, raw_retrieved, raw_true)
from ventral_interface.decode import (
    NA, decode_condition, eos_fields, first_divergence, native_freear_diagnostic)
from ventral_interface.injection import SemanticInjection, state_dict_sha256, ventral_only
from ventral_interface.population import (
    gate_a, gate_r, historical_c_contract, population_structure, retrieval_all_rows)

DECODE_BATCH = 256          # = gate_probe.collect_item_level / Trainer.free_ar_repetition
NAMING_BATCH = 512          # = evaluate_naming default, as called by full_battery
NAMING_VALIDITY_MAX_STEPS = 256   # = coexistence_probe.full_battery positional arg
SHAT_MAX_ABS_DEV = 1e-5


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


@torch.inference_mode()
def evaluate_state(model, tr, meta: Dict, archived: Optional[Dict], device: str = "cpu",
                   synthetic: bool = False, item_limit: Optional[int] = None):
    if archived is None and not synthetic:
        raise RuntimeError("archived reference values are mandatory outside synthetic tests")
    model.eval()
    vocab, entries, bank_raw = tr.vocab, tr.entries, tr.bank_raw
    bank_n = model.ltm.semantic_bank
    gates: Dict[str, Dict] = {}
    params_before = state_dict_sha256(model)

    if not torch.equal(bank_n, torch.nn.functional.normalize(bank_raw, dim=-1)):
        raise GateFailure("BANK", {"pass": False, "reason": "semantic_bank != normalize(bank_raw)"})

    # ---------------- population / retrieval
    pop = population_structure(entries)
    comp_idx = pop["comp_idx"]
    hist = historical_c_contract(model, tr, comp_idx)
    retr = retrieval_all_rows(model, tr)
    rep_r = gate_r(hist, retr, comp_idx,
                   hist["c_errors"] if synthetic else archived["c_errors"])
    if synthetic:
        rep_r["archived_comparison"] = "NOT_APPLICABLE_SYNTHETIC"
    _require("R", rep_r, gates)
    _require("A", gate_a(hist, comp_idx, pop["canonical_of"]), gates)

    n = len(entries) if item_limit is None else min(item_limit, len(entries))
    items = list(range(n))
    m = retr["metrics"]
    s_hat_r = retr["s_hat"]
    top1 = [int(x) for x in m["top1_idx"]]
    a1, c1, a2, c2 = cosine_top2(s_hat_r, bank_n)
    if [int(x) for x in a1] != top1:
        raise GateFailure("R", {"pass": False, "reason": "top-2 helper argmax != historical"})

    S1 = raw_retrieved(bank_raw, top1)          # indexed by item row
    S2 = raw_true(bank_raw, range(len(entries)))
    ret_norm = torch.linalg.vector_norm(S1.to(torch.float64), dim=-1).to(S1.dtype)
    rb = gate_b(bank_raw, items, top1[:n])
    _require("B", rb, gates)
    s3_pre, deg_pre = radial(s_hat_r[:n], ret_norm[:n])
    rd = gate_d(s_hat_r[:n], s3_pre, deg_pre, bank_n)
    _require("D", rd, gates)

    # ---------------- Gate C: matched Naming path
    nm = evaluate_naming(model, vocab, entries, bank_raw, items, "cpu",
                         NAMING_VALIDITY_MAX_STEPS, NAMING_BATCH, return_per_item=True)
    hist_pred = {r["bank_index"]: r["pred"] for r in nm["_per_item"]}
    hist_err = sum(1 - r["exact"] for r in nm["_per_item"])
    mismatch = 0
    with ventral_only(model):
        for lo in range(0, n, NAMING_BATCH):
            idx = items[lo:lo + NAMING_BATCH]
            inj = SemanticInjection("S2", fixed=S2)
            out = decode_condition(model, vocab, [entries[i].phonemes for i in idx], idx,
                                   inj, "freear", device, max_steps=NAMING_VALIDITY_MAX_STEPS)
            mismatch += sum(1 for i, p in zip(idx, out["preds"]) if _phon(vocab, p) != hist_pred[i])
    rc = {"historical_naming_errors": hist_err, "n_item_prediction_mismatch": mismatch,
          "max_steps": NAMING_VALIDITY_MAX_STEPS, "batch": NAMING_BATCH}
    if synthetic:
        rc["archived_naming_errors"] = "NOT_APPLICABLE_SYNTHETIC"
        rc["pass"] = mismatch == 0
    else:
        rc["archived_naming_errors"] = archived["naming_errors"]
        rc["pass"] = mismatch == 0 and hist_err == archived["naming_errors"]
    _require("C", rc, gates)

    # ---------------- S0..S3 decoding
    res = {c: {v: {} for v in CONVENTIONS} for c in CONDITIONS}
    live_shat = torch.zeros_like(s_hat_r[:n])
    live_deg = torch.zeros(n, dtype=torch.bool)
    diag: List[Dict] = []
    enc_step_dev = 0.0
    with ventral_only(model) as guard:
        for lo in range(0, n, DECODE_BATCH):
            idx = items[lo:lo + DECODE_BATCH]
            forms = [entries[i].phonemes for i in idx]
            for cond in CONDITIONS:
                inj = {"S0": lambda: SemanticInjection("S0"),
                       "S1": lambda: SemanticInjection("S1", fixed=S1),
                       "S2": lambda: SemanticInjection("S2", fixed=S2),
                       "S3": lambda: SemanticInjection("S3", target_norm=ret_norm)}[cond]
                for conv in CONVENTIONS:
                    out = decode_condition(model, vocab, forms, idx, inj(), conv, device)
                    enc_step_dev = max(enc_step_dev, out["encoder_step_dev"])
                    for k, i in enumerate(idx):
                        res[cond][conv][i] = (out["preds"][k], out["raw"][k])
                    if cond == "S0" and conv == "freear":
                        live_shat[lo:lo + len(idx)] = out["live_s_hat"]
                        diag.extend(native_freear_diagnostic(
                            model, vocab, forms, idx, idx, out, out["live_s_hat"], device))
                    if cond == "S3" and conv == "freear":
                        live_deg[lo:lo + len(idx)] = out["degenerate"]
        if guard.violations:
            raise GateFailure("E", {"pass": False, "violations": guard.violations})

    la1, _, _, _ = cosine_top2(live_shat, bank_n)
    dev = float((live_shat - s_hat_r[:n]).abs().max()) if n else 0.0
    rs = {"max_abs_dev": dev, "tolerance": SHAT_MAX_ABS_DEV,
          "n_top1_changed": int(sum(int(a) != t for a, t in zip(la1, top1[:n])))}
    rs["pass"] = dev <= SHAT_MAX_ABS_DEV and rs["n_top1_changed"] == 0
    _require("SHAT", rs, gates)

    err = {conv: sum(int(res["S0"][conv][i][0] != list(entries[i].phonemes)) for i in items)
           for conv in CONVENTIONS}
    rh = {"S0_freear_errors": err["freear"], "S0_canonical_errors": err["canonical"]}
    if synthetic:
        rh.update({"archived": "NOT_APPLICABLE_SYNTHETIC", "pass": True})
    else:
        rh.update({"archived_rep_freear_ltm_errors": archived["rep_freear_ltm_errors"],
                   "archived_rep_canonical_ltm_errors": archived["rep_canonical_ltm_errors"]})
        rh["pass"] = (err["freear"] == archived["rep_freear_ltm_errors"]
                      and err["canonical"] == archived["rep_canonical_ltm_errors"])
    _require("H", rh, gates)

    params_after = state_dict_sha256(model)
    re_ = {"params_sha256_before": params_before, "params_sha256_after": params_after,
           "encoder_step_max_abs_dev": enc_step_dev, "guard_violations": 0,
           "downstream_path": "gate_probe.ar_decode_{free,forced_length}(routes=('ltm',)) "
                              "-> route_logits(ltm) -> ltm.encode[to_semantic hook] -> "
                              "decode_from_s_hat(sem_to_h0, decoder, dec_to_premotor) -> motor"}
    re_["pass"] = params_before == params_after and enc_step_dev == 0.0
    _require("E", re_, gates)

    # ---------------- rows
    eos = vocab.eos_id
    rows: List[Dict] = []
    for k, i in enumerate(items):
        e = entries[i]
        t = top1[i]
        in_c = pop["in_c"][i]
        c_ok = hist["per_item"][i]["top1"] if in_c else NA
        row = {
            "witness_id": meta["witness_id"], "state_id": meta["state_id"],
            "source_or_repaired": meta["source_or_repaired"], "seed": meta["seed"],
            "source_u": meta["source_u"],
            "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
            "repaired_head_sha256": meta["repaired_head_sha256"] or NA,
            "reconstructed_state_identity": meta["reconstructed_state_identity"],
            "item_index": i, "lexical_identity": e.word,
            "target_phonology": _phon(vocab, e.phonemes), "phoneme_length": len(e.phonemes),
            "lexical_frequency": NA, "lexical_rank": int(e.rank),
            "in_C_population": in_c, "canonical_C_index": pop["canonical_of"][i],
            "canonical_C_identity": entries[pop["canonical_of"][i]].word,
            "homophone_group": pop["canonical_of"][i],
            "homophone_group_size": pop["group_size_of"][i],
            "homophone_group_members": pop["group_members_of"][i],
            "retrieved_index": t, "retrieved_lexical_identity": entries[t].word,
            "retrieved_phonology": _phon(vocab, entries[t].phonemes),
            "C_contract_correct": c_ok,
            "lexical_identity_correct": int(t == i),
            "phonology_correct": int(entries[t].phonemes == e.phonemes),
            "retrieved_is_homophone_not_target": int(t != i and entries[t].phonemes == e.phonemes),
            "cosine_target_shat": float(m["target_cos"][i]),
            "cosine_top1": float(c1[i]), "cosine_top2": float(c2[i]),
            "top2_index": int(a2[i]),
            "retrieval_margin": float(c1[i] - c2[i]),
            "historical_target_margin": float(m["margin"][i]),
            "shat_norm": float(torch.linalg.vector_norm(s_hat_r[i].to(torch.float64))),
            "retrieved_raw_glove_norm": float(torch.linalg.vector_norm(S1[i].to(torch.float64))),
            "true_raw_glove_norm": float(torch.linalg.vector_norm(S2[i].to(torch.float64))),
            "s1_equals_s2_row": int(rb["per_item_equal_row"][k]),
            "s1_s2_max_abs_diff": float(rb["per_item_max_abs_diff"][k]),
            "s3_degenerate": int(live_deg[k]),
            "s3_cos_to_shat": float(rd["per_item_cos_s3_shat"][k]),
            "s3_top1_index": int(rd["per_item_s3_top1_index"][k]),
            "live_shat_vs_retrieval_shat_max_abs_dev":
                float((live_shat[k] - s_hat_r[i]).abs().max()),
        }
        for cond in CONDITIONS:
            for conv in CONVENTIONS:
                pred, raw = res[cond][conv][i]
                fd = first_divergence(raw, e.phonemes, eos)
                p = f"{cond}_{conv}_"
                row[p + "exact_correct"] = int(pred == list(e.phonemes))
                row[p + "predicted_phonology"] = _phon(vocab, pred)
                for kk, vv in eos_fields(raw, e.phonemes, eos, conv).items():
                    row[p + kk] = vv
                target = list(e.phonemes) + [eos]
                row[p + "first_divergence_step"] = NA if fd is None else fd
                row[p + "first_divergence_gold_token"] = NA if fd is None else vocab.itos[target[fd]]
                row[p + "first_divergence_pred_token"] = NA if fd is None else vocab.itos[int(raw[fd])]
        rows.append(row)

    for d in diag:
        e = entries[d["item_index"]]
        d.update({"state_id": meta["state_id"], "lexical_identity": e.word,
                  "target_phonology": _phon(vocab, e.phonemes), "phoneme_length": len(e.phonemes),
                  "S0_freear_predicted_phonology": _phon(vocab, res["S0"]["freear"][d["item_index"]][0])})
    return {"rows": rows, "ar_diagnostic": diag, "gates": gates,
            "population": {k: pop[k] for k in ("n_rows", "n_comp",
                                               "n_noncanonical_homophone_members",
                                               "n_phonology_classes", "n_homophone_classes")},
            "non_scientific": bool(synthetic)}


def summarize_ar_diagnostic(diag_by_state: Dict[str, List[Dict]]) -> Dict:
    out = {}
    for sid, recs in diag_by_state.items():
        steps: Dict[str, int] = {}
        for r in recs:
            steps[str(r["divergence_step"])] = steps.get(str(r["divergence_step"]), 0) + 1
        out[sid] = {
            "n_native_freear_failures": len(recs),
            "divergence_step_counts": dict(sorted(steps.items(), key=lambda kv: int(kv[0]))),
            "divergence_is_eos_position_count": sum(r["divergence_is_eos_position"] for r in recs),
            "DIAGNOSTIC_ONLY_PREFIX_CORRECTION": {
                "note": "not model performance; not an inference mechanism; excluded from all primary metrics",
                "corrected_exact_count": sum(r["diag_prefix_correction_corrected_exact"] for r in recs),
            },
        }
    return out
