#!/usr/bin/env python
"""Lesioning V2 scientific runner — GUARDED.

    GO_FOR_SCIENTIFIC_EXECUTION = NO

Every nonzero lesion cell on P1-P4 is REFUSED unless CENTRAL has issued an
explicit authorization artifact naming this exact run-matrix hash. The refusal
is a code path, not a convention; there is no bypass flag.

`--dry-run` performs full provenance validation and emits the execution plan
with ZERO model forwards, and remains available at all times.

This program never trains: it never calls backward(), never constructs an
optimizer, and applies masks only inside the reversible context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
sys.path.insert(0, REPO)

from paper_programme.lesioning_v2.lesion_operator import (  # noqa: E402
    battery, guard, masks, noise, seeds)
from paper_programme.lesioning_v2.lesion_operator.sites import SITES  # noqa: E402

CONTRACT = os.path.join(PKG, "contract")
ITEM_SCHEMA = (
    "state_id", "state_sha256", "site", "severity_k", "severity_s",
    "connectivity_fraction", "n_logical_edges_total", "n_logical_edges_removed",
    "realization", "mask_seed_digest", "item_id", "activation_seed_digest",
    "task", "decoding_convention", "target", "prediction", "correct",
)
SUMMARY_KEYS = ("state_id", "site", "severity_k", "realization", "task",
                "decoding_convention")


class PreflightError(RuntimeError):
    pass


def _sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_matrix() -> dict:
    p = os.path.join(CONTRACT, "LESIONING_V2_RUN_MATRIX.json")
    if not os.path.exists(p):
        raise PreflightError(f"missing run matrix: {p}")
    return json.load(open(p))


def load_sd_constants() -> dict:
    p = os.path.join(CONTRACT, "LESIONING_V2_INTACT_SD_CONSTANTS.json")
    if not os.path.exists(p):
        raise PreflightError(
            "intact SD constants are not frozen yet; run extract_intact_sd.py "
            "on the cluster before any lesion execution")
    return json.load(open(p))


def preflight(out_dir: str, require_sd: bool) -> dict:
    rep = {}
    if os.path.exists(out_dir):
        raise PreflightError(f"REFUSED: results namespace exists: {out_dir}")
    m = load_matrix()
    rep["matrix_sha256"] = m["matrix_sha256"]
    rep["n_cells"] = m["n_cells"]
    rep["n_lesion_cells"] = m["n_lesion_cells"]

    if m.get("source_states_included"):
        raise PreflightError("REFUSED: SOURCE states present in the matrix")
    for c in m["cells"]:
        if not c["state_id"].endswith("_POST_REPAIR"):
            raise PreflightError(f"REFUSED: non-POST_REPAIR state {c['state_id']}")

    # the superseded head must be unnameable in executable context
    forbidden = "head_" + "final"
    import ast
    for rel in ("lesion_operator/masks.py", "lesion_operator/noise.py", "lesion_operator/context.py",
                "lesion_operator/sites.py", "lesion_operator/seeds.py", "lesion_operator/guard.py",
                "scripts/run_lesion_v2.py"):
        tree = ast.parse(open(os.path.join(PKG, rel)).read())
        docs = set()
        for owner in [tree] + [n for n in ast.walk(tree)
                               if isinstance(n, (ast.FunctionDef,
                                                 ast.AsyncFunctionDef,
                                                 ast.ClassDef))]:
            b = getattr(owner, "body", None)
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant):
                docs.add(id(b[0].value))
        for n in ast.walk(tree):
            if (isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and forbidden in n.value and id(n) not in docs):
                raise PreflightError(f"{rel} references the superseded head")
            if isinstance(n, ast.Attribute) and forbidden in n.attr:
                raise PreflightError(f"{rel} references the superseded head")
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for bad in ("backward", "step", "zero_grad", "requires_grad_", "save"):
            if bad in called:
                raise PreflightError(f"{rel} can execute a training operation: {bad}")
    rep["no_training_path"] = True
    rep["superseded_head_unreferenced"] = True

    if require_sd:
        sd = load_sd_constants()
        missing = [f"{c['state_id']}/{c['site']}" for c in m["cells"]
                   if f"{c['state_id']}/{c['site']}" not in sd["constants"]]
        if missing:
            raise PreflightError(f"missing SD constants: {sorted(set(missing))[:5]}")
        rep["sd_procedure_hash"] = sd["procedure_hash"]
    rep["item_schema"] = list(ITEM_SCHEMA)
    rep["primary_endpoints"] = [e.key for e in battery.PRIMARY]
    rep["diagnostic_endpoints"] = [e.key for e in battery.DIAGNOSTIC]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Lesioning V2 scientific runner (guarded).")
    ap.add_argument("--out-dir", default=os.path.join(PKG, "results"))
    ap.add_argument("--dry-run", action="store_true",
                    help="full provenance validation and execution plan; "
                         "ZERO model forwards; always available")
    a = ap.parse_args(argv)

    try:
        rep = preflight(a.out_dir, require_sd=not a.dry_run)
    except PreflightError as e:
        print(f"PREFLIGHT_FAIL\n{e}", file=sys.stderr)
        return 2

    print("PREFLIGHT_OK")
    for k in ("matrix_sha256", "n_cells", "n_lesion_cells",
              "no_training_path", "superseded_head_unreferenced"):
        print(f"  {k} = {rep[k]}")

    if a.dry_run:
        print("DRY_RUN=1 : no model was loaded and no forward pass was run.")
        print(f"GO_FOR_SCIENTIFIC_EXECUTION="
              f"{'YES' if guard.GO_FOR_SCIENTIFIC_EXECUTION else 'NO'}")
        return 0

    # --- the guard. Nonzero lesion cells require CENTRAL authorization. ---
    try:
        guard.check_authorized(rep["matrix_sha256"])
    except guard.ExecutionRefused as e:
        print(f"EXECUTION_REFUSED\n{e}", file=sys.stderr)
        return 3

    print("authorization accepted; scientific execution would proceed here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
