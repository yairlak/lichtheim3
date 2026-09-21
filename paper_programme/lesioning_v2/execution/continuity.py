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


def inspect_cells(result_root: str, matrix: Dict):
    """Read-only scan of COMPLETE cells under `result_root`.

    Imposes NO count expectation, so it serves both the global parent scan and
    a single shard root. Returns (entries, nonzero_complete_keys).
    """
    if not os.path.isdir(result_root):
        raise ContinuityError(f"result root does not exist: {result_root}")
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

    entries.sort(key=lambda e: e["cell_key"])
    return entries, nonzero_complete


def generate(result_root: str, matrix: Dict) -> Dict:
    """Build the GLOBAL manifest from the results PARENT (read-only).

    The real namespace is a parent holding one shard directory per runner
    invocation, each containing that shard's own COMPLETE control, so this scan
    walks the whole tree and expects 12 in total across all shards.
    """
    k0_rows = {cells.cell_identity(r): r for r in matrix["cells"]
               if int(r["severity_k"]) == 0}
    if len(k0_rows) != EXPECTED_K0_CELLS:
        raise ContinuityError(
            f"authoritative matrix has {len(k0_rows)} k=0 rows, "
            f"expected {EXPECTED_K0_CELLS}")
    entries, nonzero_complete = inspect_cells(result_root, matrix)
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


def _check_manifest_structure(manifest: Dict, matrix: Dict) -> Dict:
    """The GLOBAL manifest must itself be complete, regardless of which root is
    being continued. A shard may not continue on a manifest that is missing
    another shard's control or that declares any COMPLETE k>0 cell."""
    if manifest.get("run_matrix_sha256") != matrix["matrix_sha256"]:
        raise ContinuityError("manifest was generated for a different matrix")
    declared = manifest.get("cells") or []
    if len(declared) != EXPECTED_K0_CELLS:
        raise ContinuityError(
            f"manifest declares {len(declared)} controls, expected "
            f"{EXPECTED_K0_CELLS} (a shard may not continue on a partial "
            "global manifest)")
    if manifest.get("n_complete_nonzero_cells", 0) != 0:
        raise ContinuityError("manifest declares COMPLETE k>0 cells")
    if any(int(c.get("severity_k", -1)) != 0 for c in declared):
        raise ContinuityError("manifest contains a non-k0 cell")
    k0_rows = {cells.cell_identity(r) for r in matrix["cells"]
               if int(r["severity_k"]) == 0}
    if {c["cell_identity"] for c in declared} != k0_rows:
        raise ContinuityError(
            "manifest does not represent every authoritative k=0 row exactly "
            "once")
    return {c["cell_identity"]: c for c in declared}


def _compare(declared: Dict, actual: Dict) -> None:
    for cid, c in declared.items():
        a = actual[cid]
        for f in ("items_jsonl_sha256", "summary_json_sha256",
                  "provenance_json_sha256", "cell_complete_json_sha256"):
            if c[f] != a[f]:
                raise ContinuityError(
                    f"{c['cell_key']}: {f} differs from the manifest "
                    "(stale manifest or altered cell)")


def validate_for_continuation(manifest: Dict, out_root: str,
                              matrix: Dict) -> Dict:
    """Fail closed unless `out_root` is exactly what the manifest describes.

    Two sanctioned shapes, both re-reading the filesystem -- a manifest alone is
    never trusted:

      GLOBAL  out_root IS the results parent: all 12 controls must be present.
      SHARD   out_root is one shard directory UNDER that parent: only that
              shard's own controls must be present. The other 11 live in
              sibling shard roots and are NEVER expected here, moved or copied.

    No scientific value is consulted.
    """
    declared = _check_manifest_structure(manifest, matrix)
    parent = manifest.get("result_root")
    if not parent:
        raise ContinuityError("manifest records no result_root")
    root = os.path.abspath(out_root)

    if root == os.path.abspath(parent):
        entries, nonzero = inspect_cells(root, matrix)
        if nonzero:
            raise ContinuityError(f"COMPLETE k>0 cells present: {nonzero[:5]}")
        actual = {e["cell_identity"]: e for e in entries}
        if set(actual) != set(declared):
            raise ContinuityError("manifest cell set does not match the namespace")
        _compare(declared, actual)
        return {"mode": "GLOBAL", "local_complete_k0": len(actual)}

    if not root.startswith(os.path.abspath(parent) + os.sep):
        raise ContinuityError(
            f"out_root {root} is neither the manifest's result root nor a "
            f"shard directory beneath it ({parent})")

    # ---- SHARD mode -------------------------------------------------------
    rel_prefix = os.path.relpath(root, os.path.abspath(parent))
    expected = {cid: c for cid, c in declared.items()
                if (c["relative_location"].split(os.sep)[0] ==
                    rel_prefix.split(os.sep)[0])}
    entries, nonzero = inspect_cells(root, matrix)
    if nonzero:
        raise ContinuityError(
            f"COMPLETE k>0 cells present under {rel_prefix}: {nonzero[:5]}")
    actual = {e["cell_identity"]: e for e in entries}
    if set(actual) != set(expected):
        unexpected = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        raise ContinuityError(
            f"{rel_prefix}: local COMPLETE controls do not match the manifest "
            f"({len(missing)} missing, {len(unexpected)} unexpected). A "
            "control belonging to another shard must not appear here.")
    _compare(expected, actual)
    return {"mode": "SHARD", "shard_root": rel_prefix,
            "local_complete_k0": len(actual)}
