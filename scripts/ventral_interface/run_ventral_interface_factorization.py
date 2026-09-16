#!/usr/bin/env python3
"""VENTRAL SEMANTIC INTERFACE — frozen S0-S3 factorization diagnostic driver.

Modes (mutually exclusive, one is required):

  --preflight   provenance + structural + NON_SCIENTIFIC mechanics checks.  Never loops
                over the scientific population; writes only booleans / hashes to
                paper_programme/ventral_semantic_interface/provenance/preflight_report.json
                (deterministic: no timestamps, sorted keys).

  --execute     the preregistered diagnostic.  Refused unless
                --contract-sha256 equals the SHA256 of the frozen contract file, the
                git worktree is clean, a fresh preflight passes, and the output
                directory does not yet exist.

Contract: paper_programme/ventral_semantic_interface/VENTRAL_INTERFACE_EXPERIMENT_CONTRACT.md

Must be run from the worktree root (checkpoint `lexicon_path` is relative).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

PROG = os.path.join(ROOT, "paper_programme", "ventral_semantic_interface")
CONFIG = os.path.join(PROG, "ventral_interface_frozen_config.json")
CONTRACT = os.path.join(PROG, "VENTRAL_INTERFACE_EXPERIMENT_CONTRACT.md")
PREFLIGHT_REPORT = os.path.join(PROG, "provenance", "preflight_report.json")
EXEC_DIR = os.path.join(PROG, "scientific_execution")
SMOKE_ITEMS = list(range(8))       # NON_SCIENTIFIC mechanics only; no outcome recorded
from ventral_interface.decode import LOGIT_PATH_TOL  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_config() -> Dict:
    with open(CONFIG) as f:
        return json.load(f)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def fail(msg: str) -> None:
    raise SystemExit(f"HARD STOP: {msg}")


# ============================================================ provenance checks
def check_provenance(cfg: Dict) -> Dict:
    from gate_x_lesion.identity import reconstructed_state_sha256
    rep: Dict[str, object] = {}
    parent = cfg["repository_parent"]
    if os.path.realpath(os.getcwd()) != os.path.realpath(ROOT):
        fail(f"must run from the worktree root {ROOT}")
    blob = git("rev-parse", f"HEAD:{cfg['state_manifest']['path']}")
    if blob != cfg["state_manifest"]["git_blob"]:
        fail(f"state manifest blob {blob} != frozen {cfg['state_manifest']['git_blob']}")
    rep["state_manifest_blob_ok"] = True
    anc = subprocess.run(["git", "-C", ROOT, "merge-base", "--is-ancestor",
                          cfg["audited_base_commit"], "HEAD"]).returncode == 0
    if not anc:
        fail("HEAD does not descend from the audited base commit")
    rep["head_descends_from_audited_base"] = True

    art = {}
    for s in cfg["states"]:
        sid = s["state_id"]
        src = sha256_file(os.path.join(parent, s["source_checkpoint_path"]))
        arc = sha256_file(os.path.join(parent, s["source_checkpoint_archival_copy"]))
        head = (sha256_file(os.path.join(parent, s["repaired_head_path"]))
                if s["repaired_head_path"] else None)
        ref = sha256_file(os.path.join(parent, s["archived_reference_file"]))
        comp = reconstructed_state_sha256(src, head or "")
        with open(os.path.join(parent, s["archived_reference_file"])) as f:
            refj = json.load(f)
        ok = {
            "source_sha256_ok": src == s["source_checkpoint_sha256"],
            "archival_copy_sha256_ok": arc == s["source_checkpoint_sha256"],
            "head_sha256_ok": head == s["repaired_head_sha256"],
            "composite_identity_ok": comp == s["reconstructed_state_identity"],
            "archived_reference_file_sha256_ok": ref == s["archived_reference_file_sha256"],
            "archived_values_ok": all(refj[k] == v for k, v in s["archived"].items()),
            "archived_reference_names_head": (
                refj.get("head_sha256") == s["repaired_head_sha256"]
                and refj.get("source_sha256") == s["source_checkpoint_sha256"]),
        }
        if not all(ok.values()):
            fail(f"{sid} provenance: {ok}")
        art[sid] = ok
    rep["states"] = art
    d = cfg["data"]
    g = sha256_file(os.path.join(ROOT, d["glove_path_in_worktree"]))
    lx = sha256_file(os.path.join(ROOT, d["lexicon_path"]))
    if g != d["glove_sha256"] or lx != d["lexicon_sha256"]:
        fail("GloVe or lexicon SHA256 mismatch")
    rep["glove_sha256_ok"] = rep["lexicon_sha256_ok"] = True
    return rep


def manifest_rows(cfg: Dict) -> Dict[str, Dict]:
    from scripts.gating_diagnostics.run_gate_route_audit import load_manifest
    rows = {r["state_id"]: r for r in load_manifest(
        os.path.join(ROOT, cfg["state_manifest"]["path"]))}
    for s in cfg["states"]:
        r = rows[s["state_id"]]
        if (r["artifact_path"] != s["source_checkpoint_path"]
                or r["artifact_sha256"] != s["source_checkpoint_sha256"]
                or (r.get("applies_head_sha256") or None) != s["repaired_head_sha256"]):
            fail(f"frozen config disagrees with the GATING manifest for {s['state_id']}")
    return rows


def reconstruct(row: Dict):
    """The canonical reconstruction, unmodified (GATING / GXLR lineage)."""
    from scripts.gating_diagnostics.run_gate_route_audit import build_state
    tr, model, prov, before, ckpt = build_state(row, "cpu")
    model.eval()
    return tr, model


def structural_state_checks(cfg: Dict, tr, model) -> Dict:
    from scripts.naming_comprehension.frozen_head_probe import sha256_tensor
    from scripts.naming_comprehension.train_tasks import subset_definition_hash, subset_records
    from ventral_interface.injection import state_dict_sha256
    from ventral_interface.population import population_structure
    d = cfg["data"]
    pop = population_structure(tr.entries)
    bank_n = model.ltm.semantic_bank
    out = {
        "model_training_flag_false": model.training is False,
        "ltm_encoder_mode_unigru_last_hidden": model.ltm.cfg.ltm_encoder_mode == "unigru_last_hidden",
        "ventral_noise_zero": float(model.ltm.cfg.ventral_noise) == 0.0,
        "semantic_dim_300": model.ltm.semantic_dim == 300,
        "bank_raw_shape_ok": tuple(tr.bank_raw.shape) == (d["n_lexical_rows"], 300),
        "bank_raw_tensor_sha256_ok": sha256_tensor(tr.bank_raw) == d["bank_raw_tensor_sha256"],
        "bank_raw_is_unnormalized": bool(
            torch.linalg.vector_norm(tr.bank_raw, dim=-1).min() > 1.5),
        "semantic_bank_equals_normalize_bank_raw": bool(torch.equal(
            bank_n, torch.nn.functional.normalize(tr.bank_raw, dim=-1))),
        "n_lexical_rows_ok": pop["n_rows"] == d["n_lexical_rows"],
        "n_canonical_c_ok": pop["n_comp"] == d["n_canonical_c_targets"],
        "canonical_c_hash_ok": subset_definition_hash(
            subset_records(tr.entries, pop["comp_idx"], tr.vocab)) == d["canonical_c_population_sha256"],
        "entry_order_ok": hashlib.sha256(",".join(e.word for e in tr.entries).encode()).hexdigest()
        == d["entry_order_sha256"],
        "vocab_ok": (tr.vocab.size, tr.vocab.pad_id, tr.vocab.bos_id, tr.vocab.eos_id)
        == (d["vocab_size"], d["pad_id"], d["bos_id"], d["eos_id"])
        and hashlib.sha256("\n".join(tr.vocab.itos).encode()).hexdigest() == d["vocab_itos_sha256"],
        "max_form_length_ok": max(len(e.phonemes) for e in tr.entries) == d["max_form_length"],
        "glove_fallback_zero": tr.glove_fallback == 0,
        "gate_has_no_parameters": len([k for k in model.state_dict() if k.startswith("gate.")]) == 0,
    }
    if not all(out.values()):
        fail(f"structural checks: {out}")
    out["population_counts"] = {k: pop[k] for k in (
        "n_rows", "n_comp", "n_noncanonical_homophone_members", "n_phonology_classes",
        "n_homophone_classes")}
    out["params_sha256"] = state_dict_sha256(model)
    return out


def pair_diff(src_model, rep_model, head_path: str) -> Dict:
    ss, rs = src_model.state_dict(), rep_model.state_dict()
    diff = sorted(k for k in ss if not torch.equal(ss[k], rs[k]))
    head = torch.load(head_path, map_location="cpu", weights_only=False)
    ok = {
        "same_keys": set(ss) == set(rs),
        "differing_tensors_exactly_to_semantic_2": diff == ["ltm.to_semantic.2.bias",
                                                            "ltm.to_semantic.2.weight"],
        "head_state_keys_exact": set(head["state"]) == {"2.weight", "2.bias"},
        "rep_params_equal_head": all(torch.equal(head["state"][k], rs["ltm.to_semantic." + k])
                                     for k in head["state"]),
        "semantic_bank_identical": bool(torch.equal(src_model.ltm.semantic_bank,
                                                    rep_model.ltm.semantic_bank)),
    }
    if not all(ok.values()):
        fail(f"forbidden parameter change: {ok} diff={diff}")
    return ok


# ======================================================= NON_SCIENTIFIC smoke
@torch.inference_mode()
def smoke_mechanics(tr, model) -> Dict:
    """NON_SCIENTIFIC.  Equivalence/mechanics booleans on 8 fixed items; no
    prediction, correctness or retrieval outcome is recorded."""
    from gating_diagnostics.gate_probe import ar_decode_forced_length, ar_decode_free
    from scripts.naming_comprehension.frozen_probe import (
        comprehension_metrics, encode_all, semantic_greedy_decode)
    from ventral_interface.conditions import gate_b, gate_d, radial, raw_retrieved, raw_true
    from ventral_interface.decode import decode_condition, gold_prefix_logits
    from ventral_interface.injection import (
        ForbiddenRouteAccess, SemanticInjection, state_dict_sha256, ventral_only)
    from evaluate.hooks import make_batch

    idx = SMOKE_ITEMS
    forms = [tr.entries[i].phonemes for i in idx]
    vocab, bank_raw = tr.vocab, tr.bank_raw
    before = state_dict_sha256(model)
    s_hat = encode_all(model, vocab, forms, "cpu", 512)
    top1 = [int(x) for x in comprehension_metrics(s_hat, bank_raw, idx, 512)["top1_idx"]]
    S1 = raw_retrieved(bank_raw, top1)      # table indexed by smoke position
    S2 = raw_true(bank_raw, idx)
    rn = torch.linalg.vector_norm(S1.to(torch.float64), dim=-1).float()
    pos = list(range(len(idx)))
    r: Dict[str, object] = {"label": "NON_SCIENTIFIC_SMOKE", "n_items": len(idx)}

    with ventral_only(model):
        for conv, fn in (("freear", lambda: ar_decode_free(model, vocab, forms, "cpu", routes=("ltm",))),
                         ("canonical", lambda: ar_decode_forced_length(model, vocab, forms, "cpu",
                                                                       routes=("ltm",)))):
            nohook = fn()["ltm"]
            o = decode_condition(model, vocab, forms, pos, SemanticInjection("S0"), conv, "cpu")
            r[f"S0_hook_passthrough_equals_unhooked_{conv}"] = o["preds"] == nohook
            r[f"S0_live_shat_equals_encode_all_{conv}"] = bool(torch.equal(o["live_s_hat"], s_hat))
            for cond, inj in (("S1", SemanticInjection("S1", fixed=S1)),
                              ("S2", SemanticInjection("S2", fixed=S2)),
                              ("S3", SemanticInjection("S3", target_norm=rn))):
                oc = decode_condition(model, vocab, forms, pos, inj, conv, "cpu")
                r[f"{cond}_{conv}_decodes_through_shared_path"] = (
                    len(oc["preds"]) == len(idx) and oc["encoder_step_dev"] == 0.0)
        o2 = decode_condition(model, vocab, forms, pos, SemanticInjection("S2", fixed=S2),
                              "freear", "cpu", max_steps=256)
        ref, _ = semantic_greedy_decode(model, S2, vocab, 256)
        r["S2_injected_cap256_equals_semantic_greedy_decode"] = o2["preds"] == ref
        o0 = decode_condition(model, vocab, forms, pos, SemanticInjection("S0"), "freear", "cpu")
        tf = gold_prefix_logits(model, vocab, forms, pos,
                                SemanticInjection("FIXED", fixed=o0["live_s_hat"]), "cpu")
        dev = 0.0
        for k, f in enumerate(forms):
            for t in range(len(f) + 1):
                if o0["raw"][k][:t] != (list(f) + [vocab.eos_id])[:t]:
                    break
                dev = max(dev, float((o0["step_logits"][t][k] - tf[k, t]).abs().max()))
        r["goldprefix_logits_match_freear_step_logits_within_tol"] = dev <= LOGIT_PATH_TOL

    b = make_batch([list(f) for f in forms], vocab, "cpu")
    for route in ("full", "wm", "fixed05"):
        raised = False
        with ventral_only(model):
            try:
                from gating_diagnostics.gate_probe import _route_step_logits
                _route_step_logits(model, b["enc_in"], b["enc_mask"], b["dec_in"], route)
            except ForbiddenRouteAccess:
                raised = True
        r[f"guard_blocks_route_{route}"] = raised
    s3, deg = radial(s_hat, rn)
    r["gate_B_mechanics_pass"] = gate_b(bank_raw, idx, idx)["pass"]
    r["gate_D_mechanics_pass"] = gate_d(s_hat, s3, deg, model.ltm.semantic_bank)["pass"]
    r["params_unchanged"] = state_dict_sha256(model) == before
    bad = [k for k, v in r.items() if v is False]
    if bad:
        fail(f"smoke mechanics failed: {bad}")
    return r


def runtime_closure(cfg: Dict) -> Dict:
    loaded = sorted({os.path.relpath(os.path.realpath(m.__file__), os.path.realpath(ROOT))
                     for m in list(sys.modules.values())
                     if isinstance(getattr(m, "__file__", None), str)
                     and os.path.isabs(m.__file__) and os.path.isfile(m.__file__)
                     and os.path.realpath(m.__file__).startswith(os.path.realpath(ROOT) + os.sep)})
    frozen = set(cfg["code_closure"])
    missing = [p for p in loaded if p not in frozen]
    if missing:
        fail(f"runtime-loaded worktree modules missing from frozen code_closure: {missing}")
    # Git cleanliness is printed, not stored: the report must be a pure function of the
    # frozen file contents so that a pre-commit and a post-commit preflight are identical.
    dirty = git("status", "--porcelain", "--", *cfg["code_closure"])
    print(f"CLOSURE_GIT_STATUS_CLEAN={dirty == ''}")
    return {"runtime_loaded_modules": loaded,
            "all_runtime_modules_in_frozen_closure": True,
            "closure_sha256": {p: sha256_file(os.path.join(ROOT, p)) for p in cfg["code_closure"]}}


def environment() -> Dict:
    return {"python": platform.python_version(), "torch": torch.__version__,
            "device": "cpu", "default_dtype": str(torch.get_default_dtype()),
            "model_dtype": "torch.float32", "deterministic_algorithms": True}


# ================================================================== modes
def run_preflight(cfg: Dict, write: bool = True):
    torch.use_deterministic_algorithms(True)
    report: Dict[str, object] = {"mode": "PREFLIGHT", "scientific_execution": False,
                                 "environment": environment()}
    report["provenance"] = check_provenance(cfg)
    rows = manifest_rows(cfg)
    states: Dict[str, Dict] = {}
    models = {}
    for s in cfg["states"]:
        tr, model = reconstruct(rows[s["state_id"]])
        chk = structural_state_checks(cfg, tr, model)
        if s["source_or_repaired"] == "SOURCE":
            chk["smoke"] = smoke_mechanics(tr, model) if s["state_id"] == "W3_SRC" else "not_run_on_this_state"
            models[s["witness_id"]] = model
        else:
            chk["pair_diff_vs_source"] = pair_diff(
                models.pop(s["witness_id"]), model,
                os.path.join(cfg["repository_parent"], s["repaired_head_path"]))
        states[s["state_id"]] = chk
        del tr
    report["states"] = states
    from ventral_interface.schema import schema_descriptor
    report["schema"] = schema_descriptor()
    report["closure"] = runtime_closure(cfg)
    report["PREFLIGHT"] = "PASS"
    blob = json.dumps(report, sort_keys=True, indent=1)
    if write:
        os.makedirs(os.path.dirname(PREFLIGHT_REPORT), exist_ok=True)
        with open(PREFLIGHT_REPORT, "w") as f:
            f.write(blob + "\n")
    print(f"PREFLIGHT=PASS report_sha256={hashlib.sha256((blob + chr(10)).encode()).hexdigest()}")
    return report


def run_execute(cfg: Dict, contract_sha: Optional[str]) -> int:   # pragma: no cover
    """The preregistered diagnostic.  NOT run in the freeze pass."""
    from ventral_interface.evaluate import GateFailure, evaluate_state, summarize_ar_diagnostic
    from ventral_interface.schema import (AR_DIAGNOSTIC_COLUMNS, ITEM_LEVEL_COLUMNS,
                                          SCHEMA_VERSION, summarize, write_tsv)
    actual = sha256_file(CONTRACT)
    if not contract_sha or contract_sha != actual:
        fail(f"--contract-sha256 must equal the frozen contract SHA256 {actual}")
    if "CONTRACT_STATUS=FROZEN" not in open(CONTRACT).read():
        fail("contract is not FROZEN")
    if git("status", "--porcelain"):
        fail("worktree not clean")
    if os.path.exists(EXEC_DIR):
        fail(f"{EXEC_DIR} exists; refusing to overwrite")
    pre = run_preflight(cfg, write=False)
    torch.use_deterministic_algorithms(True)
    rows = manifest_rows(cfg)
    all_rows: List[Dict] = []
    diag: Dict[str, List[Dict]] = {}
    gates: Dict[str, Dict] = {}
    os.makedirs(EXEC_DIR)
    for s in cfg["states"]:
        tr, model = reconstruct(rows[s["state_id"]])
        meta = {k: s[k] for k in ("witness_id", "state_id", "source_or_repaired", "seed",
                                  "source_u", "source_checkpoint_sha256",
                                  "repaired_head_sha256", "reconstructed_state_identity")}
        try:
            out = evaluate_state(model, tr, meta, s["archived"], "cpu")
        except GateFailure as g:
            with open(os.path.join(EXEC_DIR, "GATE_FAILURE.json"), "w") as f:
                json.dump({"state_id": s["state_id"], "gate": g.gate, "report": g.report,
                           "interpretation": "STOP"}, f, indent=1, sort_keys=True, default=str)
            print(f"GATE {g.gate} FAILED on {s['state_id']}: STOP")
            return 3
        all_rows.extend(out["rows"])
        diag[s["state_id"]] = out["ar_diagnostic"]
        gates[s["state_id"]] = out["gates"]
        del tr, model
    write_tsv(os.path.join(EXEC_DIR, "item_level_factorization.tsv"), ITEM_LEVEL_COLUMNS, all_rows)
    write_tsv(os.path.join(EXEC_DIR, "ar_diagnostic_native_freear.tsv"), AR_DIAGNOSTIC_COLUMNS,
              [r for sid in diag for r in diag[sid]])
    summary = {"schema_version": SCHEMA_VERSION, "contract_sha256": actual,
               "provenance": {"git_head": git("rev-parse", "HEAD"),
                              "preflight_closure_sha256": pre["closure"]["closure_sha256"],
                              "environment": environment()},
               "gates": gates, "results": summarize(all_rows),
               "ar_diagnostic_native_freear": summarize_ar_diagnostic(diag)}
    with open(os.path.join(EXEC_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True, default=str)
    print("EXECUTE=COMPLETE")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--contract-sha256", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.preflight:
        run_preflight(cfg)
        return 0
    return run_execute(cfg, a.contract_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
