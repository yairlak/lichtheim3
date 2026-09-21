"""READ-ONLY continuity manifest for the 12 existing k=0 intact controls.

Generated on Jean-Zay against the real result namespace produced by the failed
run at execution commit 0b22b904. It NEVER writes into an existing cell
directory, never rewrites a marker or provenance, never deletes anything, and
never recomputes a control forward. The manifest is written to a caller-supplied
path OUTSIDE the scientific result namespace.

It is also the gate for continuation: a repaired run may continue into an
existing namespace only when this manifest validates it.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List

from . import cells

ORIGINAL_EXECUTION_COMMIT = "0b22b90455a30b8d2ee1ca86df0fc96955e4e542"
EXPECTED_K0_CELLS = 12
CELL_FILES = ("items.jsonl", "summary.json", "provenance.json")


class ContinuityError(RuntimeError):
    """The existing namespace is not a valid basis for continuation."""


def _sha_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def matrix_row_sha256(row: Dict) -> str:
    return hashlib.sha256(
        json.dumps({k: row[k] for k in sorted(row)},
                   sort_keys=True).encode("utf-8")).hexdigest()


def generate(result_root: str, matrix: Dict) -> Dict:
    """Inspect an existing result namespace read-only and build the manifest."""
    if not os.path.isdir(result_root):
        raise ContinuityError(f"result root does not exist: {result_root}")

    k0_rows = {cells.cell_identity(r): r for r in matrix["cells"]
               if int(r["severity_k"]) == 0}
    if len(k0_rows) != EXPECTED_K0_CELLS:
        raise ContinuityError(
            f"authoritative matrix has {len(k0_rows)} k=0 rows, "
            f"expected {EXPECTED_K0_CELLS}")
    all_rows = {cells.cell_identity(r): r for r in matrix["cells"]}

    found = cells.read_complete_cells(result_root)
    entries: List[Dict] = []
    seen_ids = set()
    nonzero_complete = []

    for m in found:
        cid = m["cell_identity"]
        if cid in seen_ids:
            raise ContinuityError(f"duplicate scientific cell identity: {cid}")
        seen_ids.add(cid)

        row = all_rows.get(cid)
        if row is None:
            raise ContinuityError(
                f"COMPLETE cell {m['cell_key']} matches no authoritative "
                "matrix row")
        if int(row["severity_k"]) != 0:
            nonzero_complete.append(m["cell_key"])
            continue
        if m["cell_key"] != cells.cell_key(row):
            raise ContinuityError(
                f"marker key {m['cell_key']} disagrees with the authoritative "
                f"row key {cells.cell_key(row)}")
        if not m.get("restoration_verified"):
            raise ContinuityError(f"{m['cell_key']}: restoration not verified")
        if m.get("status") != "COMPLETE":
            raise ContinuityError(f"{m['cell_key']}: status is not COMPLETE")

        # verify the marker's declared hashes against the actual bytes
        digests = {}
        for name in CELL_FILES:
            p = os.path.join(m["_dir"], name)
            if not os.path.exists(p):
                raise ContinuityError(f"{m['cell_key']}: missing {name}")
            got = _sha_file(p)
            want = m["file_sha256"].get(name)
            if want is None:
                raise ContinuityError(
                    f"{m['cell_key']}: marker declares no hash for {name}")
            if got != want:
                raise ContinuityError(
                    f"{m['cell_key']}: {name} differs from its marker hash")
            digests[name] = got

        entries.append({
            "cell_identity": cid,
            "cell_key": m["cell_key"],
            "state_id": row["state_id"],
            "site": row["site"],
            "severity_k": 0,
            "realization": row["realization"],
            "authoritative_matrix_row": {k: row[k] for k in sorted(row)},
            "matrix_row_sha256": matrix_row_sha256(row),
            "original_execution_commit": ORIGINAL_EXECUTION_COMMIT,
            "items_jsonl_sha256": digests["items.jsonl"],
            "summary_json_sha256": digests["summary.json"],
            "provenance_json_sha256": digests["provenance.json"],
            "cell_complete_json_sha256": _sha_file(
                os.path.join(m["_dir"], cells.COMPLETE_MARKER)),
            "restoration_verified": True,
            "status": "COMPLETE",
            "relative_location": os.path.relpath(m["_dir"], result_root),
        })

    if nonzero_complete:
        raise ContinuityError(
            "COMPLETE k>0 cells are present but the failed run finalized none; "
            f"refusing: {sorted(nonzero_complete)[:5]}")
    if len(entries) != EXPECTED_K0_CELLS:
        raise ContinuityError(
            f"found {len(entries)} COMPLETE k=0 cells, expected "
            f"{EXPECTED_K0_CELLS}")
    represented = {e["cell_identity"] for e in entries}
    if represented != set(k0_rows):
        missing = sorted(set(k0_rows) - represented)
        raise ContinuityError(
            f"authoritative k=0 rows not represented exactly once: "
            f"{len(missing)} missing")

    entries.sort(key=lambda e: e["cell_key"])
    body = {
        "manifest_name": "LESIONING_V2_INTACT_CONTROL_CONTINUITY_MANIFEST",
        "original_execution_commit": ORIGINAL_EXECUTION_COMMIT,
        "run_matrix_sha256": matrix["matrix_sha256"],
        "result_root": os.path.abspath(result_root),
        "n_complete_k0_cells": len(entries),
        "n_complete_nonzero_cells": 0,
        "generated_read_only": True,
        "cells": entries,
    }
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    return body


def validate_for_continuation(manifest: Dict, result_root: str,
                              matrix: Dict) -> None:
    """Fail closed unless `result_root` is exactly what the manifest describes.

    Re-reads the namespace; a manifest alone is never trusted. No scientific
    value is consulted.
    """
    if manifest.get("run_matrix_sha256") != matrix["matrix_sha256"]:
        raise ContinuityError("manifest was generated for a different matrix")
    if os.path.abspath(result_root) != manifest.get("result_root"):
        raise ContinuityError(
            "manifest was generated for a different result root")
    fresh = generate(result_root, matrix)
    declared = {c["cell_identity"]: c for c in manifest["cells"]}
    actual = {c["cell_identity"]: c for c in fresh["cells"]}
    if set(declared) != set(actual):
        raise ContinuityError("manifest cell set does not match the namespace")
    for cid, c in declared.items():
        for f in ("items_jsonl_sha256", "summary_json_sha256",
                  "provenance_json_sha256", "cell_complete_json_sha256"):
            if c[f] != actual[cid][f]:
                raise ContinuityError(
                    f"{c['cell_key']}: {f} differs from the manifest "
                    "(stale manifest or altered cell)")
