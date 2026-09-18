#!/usr/bin/env python
"""Deterministic EIGHT-STATE scientific runner for the V7 pre-lesion validation.

    design anchor         2d240e1fd4b81ca93b4144b772eb21e8700d8ddf
    execution plumbing    4c24efb

Runs each frozen state exactly once, writes immutable machine-readable outputs,
and STOPS. It never trains, never writes a checkpoint, never evaluates
`head_final`, never overwrites or resumes a result namespace, and performs no
interpretation. Reporting is a separate program (`generate_reports.py`) that
reads only the frozen result files.

Usage
-----
    python run_prelesion_validation.py --dry-run     # preflight only, 0 forwards
    python run_prelesion_validation.py               # the scientific run, once
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import ast
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
CONTRACT = os.path.join(PKG, "contract")
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "execution"))

import inputs as EXIN                      # noqa: E402
import prelesion_eval as pe                # noqa: E402

DESIGN_COMMIT = "2d240e1fd4b81ca93b4144b772eb21e8700d8ddf"
PLUMBING_COMMIT = "4c24efb"
V7_PINNED_COMMIT = "46e638628e2c610c640189523c7918ade60209ad"
GLOVE_SHA256 = "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed"

# The forbidden head is named from parts so that this file contains no
# selectable literal for it -- the checker must not itself be a way to get one.
_FORBIDDEN_HEAD = "head_" + "final"

FROZEN_CONTRACT_FILES = (
    "V7_INTACT_ROUTE_VALIDATION_CONTRACT.md", "validation_contract.json",
    "population_manifests.json", "state_manifest.tsv",
    "stimulus_manifest_common_unseen_378.tsv",
    "stimulus_manifest_dorsal_pool_exposed_13.tsv",
    "stimulus_manifest_seed_unseen.tsv",
    "stimulus_manifest_trained_real_exact_671.tsv",
)
ROUTES = ("full", "wm", "ltm")


class PreflightError(RuntimeError):
    """Any hard stop before scientific model evaluation."""


# ------------------------------------------------------------------ helpers --
def sha256_file(path: str) -> str:
    return pe.sha256_file(path)


def read_manifest(name: str) -> List[dict]:
    with open(os.path.join(CONTRACT, name)) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def write_tsv(path: str, rows: List[dict]) -> None:
    if not rows:
        open(path, "w").close()
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- preflight --
def preflight(out_dir: str, require_clean_namespace: bool = True) -> dict:
    """The nine hard stops. Raises PreflightError; performs ZERO model forwards."""
    report: Dict[str, object] = {}

    # 0. results namespace must not exist. Checked FIRST: it is the cheapest
    #    refusal and the one that protects an existing scientific result.
    if require_clean_namespace and os.path.exists(out_dir):
        raise PreflightError(
            f"REFUSED: result namespace already exists: {out_dir}\n"
            "This run would overwrite or resume a scientific result. "
            "Move it aside deliberately; the runner will not do it for you.")
    report["results_namespace_absent"] = True

    # 1. HEAD is a real commit and is recorded (the runner commit).
    head = git_head()
    if not head:
        raise PreflightError("cannot resolve git HEAD")
    report["runner_commit"] = head

    # 2. contract bytes identical to the design anchor.
    drift = []
    for name in FROZEN_CONTRACT_FILES:
        rel = f"paper_programme/v7_prelesion_validation/contract/{name}"
        p = subprocess.run(["git", "show", f"{DESIGN_COMMIT}:{rel}"],
                           cwd=REPO, capture_output=True)
        if p.returncode != 0:
            raise PreflightError(f"{rel} absent at {DESIGN_COMMIT}")
        if hashlib.sha256(p.stdout).hexdigest() != \
                sha256_file(os.path.join(CONTRACT, name)):
            drift.append(name)
    if drift:
        raise PreflightError(f"CONTRACT DRIFT vs {DESIGN_COMMIT}: {drift}")
    report["contract_byte_identical_to_design"] = True

    # 3. execution inputs resolve and match their frozen SHAs (fail closed).
    canon = EXIN.canon_table_path(required=True)
    artifacts = EXIN.resolve_artifacts(required=True)
    glove = _glove_path()
    if not os.path.exists(glove):
        raise PreflightError(f"GloVe not found: {glove}")
    gsha = sha256_file(glove)
    if gsha != GLOVE_SHA256:
        raise PreflightError(
            f"GloVe SHA mismatch (fail closed)\n expected {GLOVE_SHA256}\n got {gsha}")
    report.update({"canon_table": canon, "canon_table_sha256": sha256_file(canon),
                   "glove": glove, "glove_sha256": gsha, "artifacts": artifacts})

    # 6. the eight states, each exactly once, ordered by the frozen manifest.
    states = [r for r in read_manifest("state_manifest.tsv")]
    ids = [r["state_id"] for r in states]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise PreflightError(f"state manifest must hold 8 unique states, got {ids}")
    contract = json.load(open(os.path.join(CONTRACT, "validation_contract.json")))
    if ids != contract["states"]:
        raise PreflightError("state order disagrees with the frozen contract")
    report["states"] = ids

    # 7. stimulus manifest hashes.
    pops = {}
    for name in ("stimulus_manifest_common_unseen_378.tsv",
                 "stimulus_manifest_seed_unseen.tsv",
                 "stimulus_manifest_dorsal_pool_exposed_13.tsv",
                 "stimulus_manifest_trained_real_exact_671.tsv"):
        pops[name] = sha256_file(os.path.join(CONTRACT, name))
    if len(read_manifest("stimulus_manifest_common_unseen_378.tsv")) != 378:
        raise PreflightError("primary population is not 378 items")
    report["population_manifest_sha256"] = pops

    # 8. the superseded head must not be referenceable in execution code.
    for rel in ("scripts/run_prelesion_validation.py", "scripts/prelesion_eval.py",
                "execution/inputs.py", "scripts/rules.py"):
        for node in ast.walk(ast.parse(open(os.path.join(PKG, rel)).read())):
            # only executable string/name context counts; docstrings and
            # comments are documentation, not a selectable path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if _FORBIDDEN_HEAD in node.value and not _is_docstring(node, rel):
                    raise PreflightError(
                        f"{rel} contains a {_FORBIDDEN_HEAD} string literal")
            if isinstance(node, ast.Attribute) and _FORBIDDEN_HEAD in node.attr:
                raise PreflightError(f"{rel} references {_FORBIDDEN_HEAD} attribute")
    for slot, a in artifacts.items():
        if not a["repair_head"].endswith(EXIN.REPAIR_HEAD_BASENAME):
            raise PreflightError(f"{slot} repair head is not head_first_c0.pt")
    report["only_first_c0_head_referenced"] = True

    # 9. no training/backward path can execute.
    banned = ("backward", "step", "zero_grad", "requires_grad_", "save")
    for rel in ("scripts/run_prelesion_validation.py", "scripts/prelesion_eval.py"):
        tree = ast.parse(open(os.path.join(PKG, rel)).read())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        hit = sorted(set(banned) & called)
        if hit:
            raise PreflightError(f"{rel} can execute a training operation: {hit}")
    report["no_training_path"] = True

    return report


def _is_docstring(node, rel: str) -> bool:
    """True if this string Constant is a module/class/function docstring."""
    tree = _AST_CACHE.setdefault(rel, ast.parse(open(os.path.join(PKG, rel)).read()))
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            body = getattr(parent, "body", [])
            if body and isinstance(body[0], ast.Expr) and body[0].value is node:
                return True
    return False


_AST_CACHE: Dict[str, object] = {}


def _glove_path() -> str:
    from scripts.naming_comprehension.ceiling_source_completion import GLOVE
    return GLOVE


# ------------------------------------------------------------- one state ----
def evaluate_state(state: dict, artifacts: dict, glove: str,
                   primary: List[dict], per_seed: List[dict],
                   exposed: List[dict], trained_real: List[dict]) -> dict:
    """Evaluate ONE frozen state. Returns {'summary':…, 'tables':{name: rows}}."""
    import torch

    slot, kind = state["slot"], state["state_kind"]
    art = artifacts[slot]
    ck = art["source_checkpoint"]
    head = art["repair_head"] if kind == "POST_REPAIR" else None
    ck_before = sha256_file(ck)

    tr, model, prov = pe.build_state(ck, head, glove)
    state_digest_before = pe.model_state_digest(model)

    ident = {"state_id": state["state_id"], "slot": slot, "training_seed":
             state["training_seed"], "state_kind": kind,
             "source_checkpoint_sha256": state["source_checkpoint_sha256"],
             "repair_head_file_sha256": state["repair_head_file_sha256"],
             "repair_head_deployed_state_sha256":
                 state["repair_head_deployed_state_sha256"]}

    all_idx = list(range(len(tr.entries)))
    tables: Dict[str, List[dict]] = {}
    summary: Dict[str, object] = dict(ident)

    # ---- real-word canonical (frozen evaluate_forms_ar) -------------------
    can_rows, can_agg = pe.canonical_items(tr, model, all_idx, routes=ROUTES)
    # ---- real-word genuine free-AR ---------------------------------------
    ents = [{"item_id": f"bank_{i}", "phonemes": list(tr.entries[i].phonemes)}
            for i in all_idx]
    far = pe.free_ar_items(tr, model, ents, routes=ROUTES)
    n = len(all_idx)
    for r in ROUTES:
        summary[f"rep_canonical_{r}"] = float(can_agg[r])
        summary[f"rep_canonical_{r}_errors"] = int(round((1 - can_agg[r]) * n))
        acc = sum(x["exact"] for x in far[r]) / max(n, 1)
        summary[f"rep_freear_{r}"] = float(acc)
        summary[f"rep_freear_{r}_errors"] = int(round((1 - acc) * n))

    # ---- naming + C (frozen per-item evaluators) --------------------------
    from scripts.naming_comprehension.train_tasks import (
        evaluate_comprehension_subset, evaluate_naming)
    from scripts.naming_comprehension.train_joint_scratch import FREE_AR_MAX_STEPS
    nm = evaluate_naming(model, tr.vocab, tr.entries, tr.bank_raw, all_idx,
                         "cpu", FREE_AR_MAX_STEPS, return_per_item=True)
    nm_rows = nm.pop("_per_item", [])
    summary["naming_exact"] = float(nm.get("exact_match", 0.0))
    summary["naming_errors"] = int(round((1 - summary["naming_exact"]) * n))

    c_idx = list(tr.comp_idx)
    cm = evaluate_comprehension_subset(model, tr.vocab, tr.entries, tr.bank_raw,
                                       c_idx, "cpu", 512, return_per_item=True)
    c_rows = cm.pop("_per_item", [])
    summary["c_top1"] = float(cm["top1"])
    summary["c_errors"] = int(round((1 - cm["top1"]) * len(c_idx)))
    summary["c_population_n"] = len(c_idx)

    # ---- gating ----------------------------------------------------------
    gate_rows = pe.gating_items(tr, model, all_idx)
    for field in ("c_ltm", "g"):
        vals = [r[field] for r in gate_rows if r[field] is not None]
        summary[f"{field}_summary"] = pe.summarize(vals)

    # ---- assemble real-word item table -----------------------------------
    by_bank = {r["bank_index"]: r for r in gate_rows}
    nm_by = {i: row for i, row in zip(all_idx, nm_rows)} if nm_rows else {}
    c_by = {i: row for i, row in zip(c_idx, c_rows)} if c_rows else {}
    far_by = {r: {x["item_id"]: x for x in far[r]} for r in ROUTES}
    rw: List[dict] = []
    for pos, i in enumerate(all_idx):
        row = dict(ident)
        row.update({"bank_index": i, "item_id": f"bank_{i}"})
        for r in ROUTES:
            row[f"canonical_{r}_exact"] = int(can_rows[pos][f"{r}_exact_match"])
            row[f"canonical_{r}_edit"] = can_rows[pos].get(f"{r}_edit_dist")
            f = far_by[r][f"bank_{i}"]
            row[f"freear_{r}_exact"] = f["exact"]
            row[f"freear_{r}_edit"] = f["raw_edit_distance"]
            row[f"freear_{r}_no_eos"] = f["no_eos"]
        g = by_bank.get(i, {})
        row["c_ltm"] = g.get("c_ltm")
        row["g"] = g.get("g")
        nrow = nm_by.get(i) or {}
        row["naming_exact"] = nrow.get("exact_match", nrow.get("exact"))
        crow = c_by.get(i)
        row["c_top1_correct"] = (None if crow is None
                                 else crow.get("correct", crow.get("top1")))
        rw.append(row)
    tables["REALWORD_ITEM_LEVEL"] = rw
    tables["GATING_ITEM_LEVEL"] = [dict(ident, **g) for g in gate_rows]

    # ---- pseudoword populations ------------------------------------------
    def pseudo(rows, label):
        ents = [{"item_id": r["item_id"],
                 "phonemes": [int(x) for x in r["target_phoneme_ids"].split()]}
                for r in rows]
        res = pe.free_ar_items(tr, model, ents, routes=ROUTES)
        out = []
        for r in ROUTES:
            for x in res[r]:
                out.append(dict(ident, population=label, **x))
        return res, out

    prim_res, prim_rows = pseudo(primary, "V7_COMMON_UNSEEN_378")
    tables["PSEUDOWORD_PRIMARY_ITEM_LEVEL"] = prim_rows
    npr = len(primary)
    for r in ROUTES:
        acc = sum(x["exact"] for x in prim_res[r]) / max(npr, 1)
        ned = sum(x["normalized_edit_distance"] for x in prim_res[r]) / max(npr, 1)
        summary[f"pseudo_primary_{r}_exact_acc"] = float(acc)
        summary[f"pseudo_primary_{r}_mean_ned"] = float(ned)
        summary[f"pseudo_primary_{r}_no_eos"] = sum(x["no_eos"] for x in prim_res[r])
    summary["pseudo_primary_n"] = npr
    summary.update({f"ordering_{k}": v for k, v in pe.classify_ordering(
        summary["pseudo_primary_wm_exact_acc"],
        summary["pseudo_primary_ltm_exact_acc"],
        summary["pseudo_primary_wm_mean_ned"],
        summary["pseudo_primary_ltm_mean_ned"]).items()})

    seed_rows = [r for r in per_seed if r["slot"] == slot]
    _, ss = pseudo(seed_rows, f"V7_SEED_UNSEEN_{slot}")
    tables["PSEUDOWORD_SEED_SENSITIVITY_ITEM_LEVEL"] = ss
    summary["pseudo_seed_unseen_n"] = len(seed_rows)

    _, ex = pseudo(exposed, "V7_DORSAL_POOL_EXPOSED_13")
    _, trr = pseudo(trained_real, "TRAINED_REAL_EXACT_671")
    tables["PSEUDOWORD_DESCRIPTIVE_ITEM_LEVEL"] = ex + trr

    # ---- immutability -----------------------------------------------------
    pe.assert_source_unchanged(ck, ck_before)
    if head is not None and sha256_file(head) != art["repair_head_file_sha256"]:
        raise RuntimeError(f"REFUSED: repair head mutated: {head}")
    if pe.model_state_digest(model) != state_digest_before:
        raise RuntimeError(f"REFUSED: model state changed during {state['state_id']}")
    summary["source_unchanged"] = True
    summary["model_state_unchanged"] = True
    return {"summary": summary, "tables": tables}



# ------------------------------------------------- SOURCE -> POST pairing ----
PAIRED_DISCRETE = (
    ("FULL_canonical", "canonical_full_exact"),
    ("FULL_freear", "freear_full_exact"),
    ("WM_canonical", "canonical_wm_exact"),
    ("WM_freear", "freear_wm_exact"),
    ("LTM_canonical", "canonical_ltm_exact"),
    ("LTM_freear", "freear_ltm_exact"),
    ("Naming", "naming_exact"),
    ("C", "c_top1_correct"),
)
PAIRED_CONTINUOUS = ("c_ltm", "g")


def _read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _truthy(v):
    if v in (None, "", "None"):
        return None
    if isinstance(v, str):
        if v in ("True", "true"):
            return 1
        if v in ("False", "false"):
            return 0
        try:
            return 1 if float(v) >= 0.5 else 0
        except ValueError:
            return None
    return 1 if float(v) >= 0.5 else 0


def compute_paired(states_dir: str) -> dict:
    """Item-level SOURCE->POST transitions, from the written item tables only.

    No model is re-run, and no stimulus is filtered post hoc: every item that
    carries a defined value in BOTH states is counted.
    """
    out: Dict[str, dict] = {}
    for slot in ("P1", "P2", "P3", "P4"):
        src = {r["item_id"]: r for r in _read_tsv(
            os.path.join(states_dir, f"{slot}_SOURCE", "REALWORD_ITEM_LEVEL.tsv"))}
        post = {r["item_id"]: r for r in _read_tsv(
            os.path.join(states_dir, f"{slot}_POST_REPAIR",
                         "REALWORD_ITEM_LEVEL.tsv"))}
        common = sorted(set(src) & set(post))
        d: Dict[str, object] = {"n_items_paired": len(common)}
        for label, col in PAIRED_DISCRETE:
            t = {"cc": 0, "cw": 0, "wc": 0, "ww": 0,
                 "changed_item_ids": [], "undefined": 0}
            for i in common:
                a, b = _truthy(src[i].get(col)), _truthy(post[i].get(col))
                if a is None or b is None:
                    t["undefined"] += 1
                    continue
                key = {(1, 1): "cc", (1, 0): "cw", (0, 1): "wc", (0, 0): "ww"}[(a, b)]
                t[key] += 1
                if key in ("cw", "wc"):
                    t["changed_item_ids"].append(i)
            d[label] = t
        for col in PAIRED_CONTINUOUS:
            deltas = []
            for i in common:
                a, b = src[i].get(col), post[i].get(col)
                if a in (None, "", "None") or b in (None, "", "None"):
                    continue
                deltas.append(float(b) - float(a))
            d[f"{col}_delta_summary"] = pe.summarize(deltas)
            d[f"{col}_n_changed"] = sum(1 for x in deltas if x != 0.0)
        out[slot] = d
    return out

# ------------------------------------------------------------------- main ----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the frozen V7 pre-lesion intact-route validation "
                    "(eight states, exactly once).")
    ap.add_argument("--out-dir", default=os.path.join(PKG, "results"),
                    help="scientific result namespace; must NOT already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every preflight check and resolve all inputs, "
                         "then stop. Performs ZERO model forward passes.")
    a = ap.parse_args(argv)

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        report = preflight(a.out_dir, require_clean_namespace=True)
    except EXIN.InputResolutionError as e:
        print(f"PREFLIGHT_FAIL (input resolution)\n{e}", file=sys.stderr)
        return 2
    except PreflightError as e:
        print(f"PREFLIGHT_FAIL\n{e}", file=sys.stderr)
        return 2

    print("PREFLIGHT_OK")
    for k in ("runner_commit", "contract_byte_identical_to_design",
              "results_namespace_absent", "only_first_c0_head_referenced",
              "no_training_path"):
        print(f"  {k} = {report[k]}")
    print(f"  states = {report['states']}")
    if a.dry_run:
        print("DRY_RUN=1 : no model was loaded and no forward pass was run.")
        print("GO_FOR_SCIENTIFIC_PRELESION_INFERENCE = "
              "operator decision; rerun without --dry-run to execute once.")
        return 0

    primary = read_manifest("stimulus_manifest_common_unseen_378.tsv")
    per_seed = read_manifest("stimulus_manifest_seed_unseen.tsv")
    exposed = read_manifest("stimulus_manifest_dorsal_pool_exposed_13.tsv")
    trained_real = read_manifest("stimulus_manifest_trained_real_exact_671.tsv")
    states = read_manifest("state_manifest.tsv")

    staging = tempfile.mkdtemp(prefix=".v7_prelesion_staging_",
                               dir=os.path.dirname(os.path.abspath(a.out_dir)))
    summaries: List[dict] = []
    try:
        import torch
        for st in states:
            sid = st["state_id"]
            print(f"== evaluating {sid}", flush=True)
            res = evaluate_state(st, report["artifacts"], report["glove"],
                                 primary, per_seed, exposed, trained_real)
            sdir = os.path.join(staging, "states", sid + ".partial")
            os.makedirs(sdir, exist_ok=True)
            for name, rows in res["tables"].items():
                write_tsv(os.path.join(sdir, f"{name}.tsv"), rows)
            json.dump(res["summary"], open(os.path.join(sdir, "SUMMARY.json"), "w"),
                      indent=1, default=str)
            os.rename(sdir, sdir[:-len(".partial")])   # atomic finalize per state
            summaries.append(res["summary"])
            print(f"== {sid} finalized", flush=True)

        paired = compute_paired(os.path.join(staging, "states"))
        json.dump(paired, open(os.path.join(staging, "SOURCE_POST_PAIRED.json"), "w"),
                  indent=1, default=str)
        write_tsv(os.path.join(staging, "SOURCE_POST_PAIRED.tsv"),
                  [dict(slot=slot, endpoint=ep, **t)
                   for slot, d in paired.items() for ep, t in sorted(d.items())
                   if isinstance(t, dict) and "cc" in t])

        write_tsv(os.path.join(staging, "STATE_SUMMARY.tsv"),
                  [{k: v for k, v in s.items() if not isinstance(v, dict)}
                   for s in summaries])
        json.dump(summaries, open(os.path.join(staging, "STATE_SUMMARY.json"), "w"),
                  indent=1, default=str)

        manifest = {
            "design_commit": DESIGN_COMMIT,
            "execution_plumbing_commit": PLUMBING_COMMIT,
            "runner_commit": report["runner_commit"],
            "v7_pinned_source_commit": V7_PINNED_COMMIT,
            "torch_version": torch.__version__,
            "canon_table": report["canon_table"],
            "canon_table_sha256": report["canon_table_sha256"],
            "glove": report["glove"], "glove_sha256": report["glove_sha256"],
            "artifacts": report["artifacts"],
            "population_manifest_sha256": report["population_manifest_sha256"],
            "states": report["states"],
            "started_utc": started,
            "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "timestamp_role": "METADATA_ONLY_NO_SCIENTIFIC_DECISION_DEPENDS_ON_IT",
            "frozen_official_post_battery_invariant":
                EXIN.FROZEN_OFFICIAL_POST_BATTERY,
        }
        json.dump(manifest, open(os.path.join(staging, "RUN_MANIFEST.json"), "w"),
                  indent=1, default=str)

        sums = []
        for root, _, files in os.walk(staging):
            for f in sorted(files):
                p = os.path.join(root, f)
                sums.append(f"{sha256_file(p)}  {os.path.relpath(p, staging)}")
        open(os.path.join(staging, "FILE_SHA256SUMS"), "w").write(
            "\n".join(sorted(sums)) + "\n")

        if os.path.exists(a.out_dir):
            raise RuntimeError(f"REFUSED: {a.out_dir} appeared during the run")
        os.rename(staging, a.out_dir)
        staging = None
        print(f"RESULTS_FINALIZED={a.out_dir}")
        print("NEXT: generate_reports.py (reads frozen results only)")
        return 0
    finally:
        if staging and os.path.isdir(staging):
            fail = a.out_dir + ".FAILED_" + started.replace(":", "")
            shutil.move(staging, fail)
            print(f"RUN_ABORTED: partial output preserved OUTSIDE the scientific "
                  f"namespace at {fail}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
