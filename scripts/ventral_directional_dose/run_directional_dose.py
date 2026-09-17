#!/usr/bin/env python3
"""VENTRAL SEMANTIC DIRECTIONAL DOSE — frozen diagnostic driver.

  --preflight --out <new dir outside the worktree>
        Real-state geometry / provenance preflight (contract §15b): lineage hashes, state
        reconstruction, parameter hashes, retrieval identity, geometry hard stops over all
        4 x 29,571 items, DOSE-B alpha=0 reproduction.  NEVER decodes a scientific alpha.

  --execute --contract-sha256 <sha> --freeze-commit <sha>
        Refused unless the contract is FROZEN with that hash, every IMPLEMENTATION_MANIFEST hash
        matches, HEAD == freeze commit, `git status --porcelain` is empty, and the output
        directory does not exist.  Runs a fresh in-process preflight for ALL states first;
        only if it passes are alphas {0.25, 0.50, 0.75, 1.00} decoded, exactly once.

Run from the worktree root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Callable, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

PROG = os.path.join(ROOT, "paper_programme", "ventral_semantic_directional_dose")
CONFIG = os.path.join(PROG, "ventral_directional_dose_frozen_config.json")
MANIFEST = os.path.join(PROG, "IMPLEMENTATION_MANIFEST.json")
EXEC_DIR = os.path.join(PROG, "scientific_execution")


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
    return subprocess.run(["git", "-C", ROOT, *args], check=True, capture_output=True, text=True).stdout.strip()


def fail(msg: str) -> None:
    raise SystemExit(f"HARD STOP: {msg}")


def environment() -> Dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
            "default_dtype": str(torch.get_default_dtype()), "model_semantic_dtype": "torch.float32",
            "deterministic_algorithms": True}


# ------------------------------------------------------------------ DOSE-A / DOSE-H lineage
def verify_sha256sums(base: str, sums_rel: str, expected_entries: int) -> Dict:
    path = os.path.join(base, sums_rel)
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    bad = [l.split("  ", 1)[1] for l in lines if sha256_file(os.path.join(base, l.split("  ", 1)[1])) != l.split()[0]]
    return {"file": path, "entries": len(lines), "expected_entries": expected_entries, "failures": bad,
            "pass": not bad and len(lines) == expected_entries}


def verify_lineage(cfg: Dict) -> Dict:
    from gate_x_lesion.identity import reconstructed_state_sha256
    rep: Dict[str, object] = {}
    if os.path.realpath(os.getcwd()) != os.path.realpath(ROOT):
        fail(f"must run from the worktree root {ROOT}")
    parent = cfg["repository_parent"]
    ic = cfg["immutable_controls"]
    for key in ("closed_contract", "item_level", "summary", "ar_diagnostic"):
        rep[key] = sha256_file(os.path.join(ROOT, ic[key]["path"])) == ic[key]["sha256"]
    closed_wt = cfg["lineage"]["closed_worktree"]
    for key in ("item_level", "summary", "ar_diagnostic", "closed_contract"):
        rep[f"closed_worktree_{key}"] = sha256_file(os.path.join(closed_wt, ic[key]["path"])) == ic[key]["sha256"]
    sums = []
    for s in ic["sha256sums"]:
        for base in (ROOT, closed_wt):
            sums.append(verify_sha256sums(os.path.join(base, s["base"]), s["file"], s["entries"]))
    rep["closed_sha256sums"] = [{"file": x["file"], "entries": x["entries"], "pass": x["pass"]} for x in sums]
    rep["closed_worktree_head"] = subprocess.run(["git", "-C", closed_wt, "rev-parse", "HEAD"], capture_output=True,
                                                 text=True).stdout.strip() == cfg["lineage"]["closed_results_commit"]
    rep["closed_worktree_clean"] = subprocess.run(["git", "-C", closed_wt, "status", "--porcelain"], capture_output=True,
                                                  text=True).stdout.strip() == ""
    rep["state_manifest_blob"] = git("rev-parse", f"HEAD:{cfg['state_manifest']['path']}") == cfg["state_manifest"]["git_blob"]
    rep["head_descends_from_closed_results"] = subprocess.run(
        ["git", "-C", ROOT, "merge-base", "--is-ancestor", cfg["lineage"]["closed_results_commit"], "HEAD"]).returncode == 0
    st = {}
    for s in cfg["states"]:
        src = sha256_file(os.path.join(parent, s["source_checkpoint_path"]))
        arc = sha256_file(os.path.join(parent, s["source_checkpoint_archival_copy"]))
        head = sha256_file(os.path.join(parent, s["repaired_head_path"])) if s["repaired_head_path"] else None
        st[s["state_id"]] = {"source": src == s["source_checkpoint_sha256"],
                             "archival": arc == s["source_checkpoint_sha256"],
                             "head": head == s["repaired_head_sha256"],
                             "composite": reconstructed_state_sha256(src, head or "") == s["reconstructed_state_identity"]}
    rep["states"] = st
    d = cfg["data"]
    rep["glove"] = sha256_file(os.path.join(ROOT, d["glove_path_in_worktree"])) == d["glove_sha256"]
    rep["lexicon"] = sha256_file(os.path.join(ROOT, d["lexicon_path"])) == d["lexicon_sha256"]
    flat = [v for k, v in rep.items() if isinstance(v, bool)]
    flat += [x["pass"] for x in rep["closed_sha256sums"]] + [v for s in st.values() for v in s.values()]
    rep["pass"] = all(flat)
    return rep


def manifest_rows(cfg: Dict) -> Dict[str, Dict]:
    from scripts.gating_diagnostics.run_gate_route_audit import load_manifest
    rows = {r["state_id"]: r for r in load_manifest(os.path.join(ROOT, cfg["state_manifest"]["path"]))}
    for s in cfg["states"]:
        r = rows[s["state_id"]]
        if (r["artifact_sha256"] != s["source_checkpoint_sha256"]
                or (r.get("applies_head_sha256") or None) != s["repaired_head_sha256"]):
            fail(f"frozen config disagrees with GATING manifest for {s['state_id']}")
    return rows


def reconstruct(row: Dict):
    from scripts.gating_diagnostics.run_gate_route_audit import build_state
    tr, model, _, _, _ = build_state(row, "cpu")
    model.eval()
    return tr, model


def meta_of(s: Dict) -> Dict:
    return {k: s[k] for k in ("witness_id", "state_id", "source_or_repaired", "seed", "source_u",
                              "source_checkpoint_sha256", "repaired_head_sha256", "reconstructed_state_identity")}


def runtime_closure(cfg: Dict) -> Dict:
    loaded = sorted({os.path.relpath(os.path.realpath(m.__file__), os.path.realpath(ROOT))
                     for m in list(sys.modules.values())
                     if isinstance(getattr(m, "__file__", None), str) and os.path.isabs(m.__file__)
                     and os.path.isfile(m.__file__)
                     and os.path.realpath(m.__file__).startswith(os.path.realpath(ROOT) + os.sep)})
    missing = [p for p in loaded if p not in set(cfg["code_closure"])]
    return {"runtime_loaded_modules": loaded, "missing_from_frozen_closure": missing,
            "closure_sha256": {p: sha256_file(os.path.join(ROOT, p)) for p in cfg["code_closure"]},
            "pass": not missing}


# ------------------------------------------------------------------ preflight (all states)
def preflight_all(cfg: Dict) -> Dict:
    """Returns {"pass", "reports", "geometry", "pre" (in-memory objects)}; never raises on a
    scientific stop — the stop is recorded and "pass" is False."""
    from ventral_directional_dose.controls import load_immutable_controls
    from ventral_directional_dose.dose_math import DoseHardStop
    from ventral_directional_dose.evaluate import GateFailure, preflight_state
    torch.use_deterministic_algorithms(True)
    out: Dict = {"pass": False, "reports": {}, "geometry": {}, "pre": {}}
    lin = verify_lineage(cfg)
    out["reports"]["DOSE-A_H_lineage"] = lin
    if not lin["pass"]:
        out["failure"] = {"gate": "DOSE-A/DOSE-H", "report": lin}
        return out
    ic = cfg["immutable_controls"]["item_level"]
    controls = load_immutable_controls(os.path.join(ROOT, ic["path"]), ic["sha256"])
    rows = manifest_rows(cfg)
    for s in cfg["states"]:
        sid = s["state_id"]
        tr, model = reconstruct(rows[sid])
        try:
            pre = preflight_state(model, tr, meta_of(s), controls[sid], s["closed_params_state_dict_sha256"])
        except DoseHardStop as e:
            out["geometry"][sid] = getattr(e, "geometry", None)
            out["reports"][sid] = getattr(e, "gates", {})
            out["failure"] = {"gate": f"HARD_STOP_{e.kind}", "state_id": sid, "count": e.count, "message": str(e)}
            return out
        except GateFailure as g:
            out["failure"] = {"gate": g.gate, "state_id": sid, "report": g.report}
            return out
        out["geometry"][sid] = pre["geometry"]
        out["reports"][sid] = pre["gates"]
        out["pre"][sid] = {k: pre[k] for k in ("state_id", "s_hat", "retrieved", "cases", "params_sha256")}
        del tr, model
    clos = runtime_closure(cfg)
    out["reports"]["runtime_closure"] = clos
    if not clos["pass"]:
        out["failure"] = {"gate": "RUNTIME_CLOSURE", "report": clos}
        return out
    out["pass"] = True
    return out


def _json(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n"


def run_preflight(cfg: Dict, out_dir: str) -> int:
    if os.path.exists(out_dir):
        fail(f"preflight output dir {out_dir} exists")
    if os.path.realpath(out_dir).startswith(os.path.realpath(ROOT) + os.sep):
        fail("standalone preflight output must be outside the worktree (keeps the tracked tree clean)")
    res = preflight_all(cfg)
    os.makedirs(out_dir)
    report = {"mode": "PREFLIGHT", "scientific_alpha_decoded": False, "environment": environment(),
              "pass": res["pass"], "failure": res.get("failure"), "reports": res["reports"]}
    open(os.path.join(out_dir, "preflight_report.json"), "w").write(_json(report))
    open(os.path.join(out_dir, "real_state_geometry_preflight.json"), "w").write(_json(res["geometry"]))
    print(f"PREFLIGHT={'PASS' if res['pass'] else 'FAIL'} "
          f"report_sha256={sha256_file(os.path.join(out_dir, 'preflight_report.json'))}")
    return 0 if res["pass"] else 3


# ------------------------------------------------------------------ execute
def check_execute_preconditions(contract_sha: Optional[str], freeze_commit: Optional[str], cfg: Dict, *,
                                git_fn: Callable = git, exec_dir: str = EXEC_DIR,
                                manifest_path: str = MANIFEST) -> Dict:
    contract = os.path.join(ROOT, cfg["contract_path"])
    actual = sha256_file(contract)
    if not contract_sha or contract_sha != actual:
        fail(f"--contract-sha256 must equal the contract SHA256 ({actual})")
    status_lines = [l.strip() for l in open(contract).read().splitlines() if l.strip().startswith("CONTRACT_STATUS=")]
    if status_lines != ["CONTRACT_STATUS=FROZEN"]:
        fail(f"contract status is not FROZEN (status lines: {status_lines})")
    if not os.path.exists(manifest_path):
        fail("IMPLEMENTATION_MANIFEST.json missing")
    man = json.load(open(manifest_path))
    bad = [p for p, h in man["files_sha256"].items() if sha256_file(os.path.join(ROOT, p)) != h]
    if bad:
        fail(f"implementation manifest hash mismatch: {bad}")
    if not freeze_commit or git_fn("rev-parse", "HEAD") != freeze_commit:
        fail("HEAD does not equal the provided implementation-freeze commit")
    if git_fn("status", "--porcelain"):
        fail("worktree is not clean")
    if os.path.exists(exec_dir):
        fail(f"scientific output directory {exec_dir} already exists")
    return {"contract_sha256": actual, "freeze_commit": freeze_commit, "manifest_files": len(man["files_sha256"])}


def run_execute(cfg: Dict, contract_sha: Optional[str], freeze_commit: Optional[str]) -> int:  # pragma: no cover
    from ventral_directional_dose.controls import load_immutable_controls
    from ventral_directional_dose.evaluate import GateFailure, execute_state
    from ventral_directional_dose.schema import (ITEM_COLUMNS, SCHEMA_VERSION, alpha1_factorization,
                                                 paired, summarize, write_tsv)
    pre_ok = check_execute_preconditions(contract_sha, freeze_commit, cfg)
    os.makedirs(EXEC_DIR)
    res = preflight_all(cfg)
    gates_out = {"preconditions": pre_ok, "preflight": res["reports"]}
    open(os.path.join(EXEC_DIR, "real_state_geometry_preflight.json"), "w").write(_json(res["geometry"]))
    if not res["pass"]:
        gates_out["failure"] = res.get("failure")
        open(os.path.join(EXEC_DIR, "validity_gates_directional_dose.json"), "w").write(_json(gates_out))
        open(os.path.join(EXEC_DIR, "GATE_FAILURE.json"), "w").write(_json(res.get("failure")))
        print(f"PREFLIGHT_HARD_STOP {res.get('failure', {}).get('gate')}: no scientific alpha decoded")
        return 3
    print("PREFLIGHT=PASS (in-process); starting scientific alphas", flush=True)
    torch.use_deterministic_algorithms(True)
    ic = cfg["immutable_controls"]["item_level"]
    controls = load_immutable_controls(os.path.join(ROOT, ic["path"]), ic["sha256"])
    rows_by_state = manifest_rows(cfg)
    all_rows: List[Dict] = []
    gates_out["execution"] = {}
    case_counts = {}
    for s in cfg["states"]:
        sid = s["state_id"]
        tr, model = reconstruct(rows_by_state[sid])
        try:
            out = execute_state(model, tr, meta_of(s), controls[sid], res["pre"][sid])
        except GateFailure as g:
            gates_out["failure"] = {"gate": g.gate, "state_id": sid, "report": g.report}
            open(os.path.join(EXEC_DIR, "validity_gates_directional_dose.json"), "w").write(_json(gates_out))
            open(os.path.join(EXEC_DIR, "GATE_FAILURE.json"), "w").write(_json(gates_out["failure"]))
            print(f"GATE {g.gate} FAILED on {sid}: STOP")
            return 3
        gates_out["execution"][sid] = out["gates"]
        case_counts[sid] = out["geometry_case_counts"]
        all_rows.extend(out["rows"])
        print(f"state {sid} done", flush=True)
        del tr, model
    post = verify_lineage(cfg)
    gates_out["DOSE-H_post"] = post
    if not post["pass"]:
        gates_out["failure"] = {"gate": "DOSE-H_post", "report": post}
        open(os.path.join(EXEC_DIR, "validity_gates_directional_dose.json"), "w").write(_json(gates_out))
        open(os.path.join(EXEC_DIR, "GATE_FAILURE.json"), "w").write(_json(gates_out["failure"]))
        return 3
    write_tsv(os.path.join(EXEC_DIR, "item_level_directional_dose.tsv"), ITEM_COLUMNS, all_rows)
    summary = {"schema_version": SCHEMA_VERSION, "contract_sha256": pre_ok["contract_sha256"],
               "provenance": {"git_head": git("rev-parse", "HEAD"), "freeze_commit": freeze_commit,
                              "environment": environment(),
                              "closure_sha256": res["reports"]["runtime_closure"]["closure_sha256"]},
               "alphas": list(cfg["alphas"]),
               "hard_stop_preconditions": {sid: g["hard_stop_preconditions"] for sid, g in res["geometry"].items()},
               "real_state_geometry": {"preflight": res["geometry"], "execution_case_counts": case_counts},
               "gates": {"all_pass": True},
               "results": summarize(all_rows), "paired": paired(all_rows),
               "alpha1_factorization": alpha1_factorization(all_rows)}
    open(os.path.join(EXEC_DIR, "summary_metrics_directional_dose.json"), "w").write(_json(summary))
    gates_out["all_pass"] = True
    open(os.path.join(EXEC_DIR, "validity_gates_directional_dose.json"), "w").write(_json(gates_out))
    print("EXECUTE=COMPLETE")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--contract-sha256", default=None)
    ap.add_argument("--freeze-commit", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.preflight:
        if not a.out:
            fail("--preflight requires --out <new directory outside the worktree>")
        return run_preflight(cfg, a.out)
    return run_execute(cfg, a.contract_sha256, a.freeze_commit)


if __name__ == "__main__":
    raise SystemExit(main())
