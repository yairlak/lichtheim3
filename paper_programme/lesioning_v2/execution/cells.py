"""Cell identity, staging and atomic finalization.

Execution-only. The scientific unit is the authoritative matrix row; this module
gives each row a deterministic identity and an exactly-once lifecycle.

A cell becomes COMPLETE only when item rows, summary, provenance, parameter
restoration and output hashes are all written. Finalization is a single atomic
directory rename; a finalized cell is never overwritten and partial output is
never mistaken for complete.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional

COMPLETE_MARKER = "CELL_COMPLETE.json"
STAGING_PREFIX = ".staging_"
FAILED_PREFIX = "FAILED_"

#: Fields of the authoritative matrix row that define the scientific identity.
IDENTITY_FIELDS = ("state_id", "state_sha256", "site", "severity_k",
                   "realization")


class CellError(RuntimeError):
    pass


def cell_identity(row: Dict) -> str:
    """Deterministic identity of one authoritative matrix row."""
    missing = [f for f in IDENTITY_FIELDS if f not in row]
    if missing:
        raise CellError(f"matrix row missing identity fields: {missing}")
    payload = "|".join(str(row[f]) for f in IDENTITY_FIELDS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cell_key(row: Dict) -> str:
    """Human-readable key; must agree 1:1 with cell_identity."""
    return (f"{row['state_id']}/{row['site']}/k{int(row['severity_k']):02d}"
            f"/r{int(row['realization']):02d}")


def cell_dir(root: str, row: Dict) -> str:
    return os.path.join(root, cell_key(row))


def is_complete(root: str, row: Dict) -> bool:
    return os.path.exists(os.path.join(cell_dir(root, row), COMPLETE_MARKER))


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_cell(root: str, row: Dict, item_rows: List[Dict],
               summary: Dict, provenance: Dict,
               restoration_verified: bool) -> str:
    """Stage a cell and finalize it atomically. Refuses to overwrite.

    Nothing in `root` is touched until the final rename, so an interrupted run
    leaves a staging directory that is never mistaken for a finished cell.
    """
    if not restoration_verified:
        raise CellError("refusing to finalize a cell whose parameter "
                        "restoration was not verified")
    dest = cell_dir(root, row)
    if os.path.exists(dest):
        raise CellError(f"REFUSED: cell already exists and is never "
                        f"overwritten: {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    staging = tempfile.mkdtemp(prefix=STAGING_PREFIX,
                               dir=os.path.dirname(dest))
    try:
        files = {}
        items_b = ("\n".join(json.dumps(r, sort_keys=True)
                             for r in item_rows) + "\n").encode("utf-8")
        with open(os.path.join(staging, "items.jsonl"), "wb") as fh:
            fh.write(items_b)
        files["items.jsonl"] = _sha_bytes(items_b)

        for name, obj in (("summary.json", summary),
                          ("provenance.json", provenance)):
            b = json.dumps(obj, indent=1, sort_keys=True).encode("utf-8")
            with open(os.path.join(staging, name), "wb") as fh:
                fh.write(b)
            files[name] = _sha_bytes(b)

        marker = {
            "cell_identity": cell_identity(row),
            "cell_key": cell_key(row),
            "identity_fields": {f: row[f] for f in IDENTITY_FIELDS},
            "n_item_rows": len(item_rows),
            "restoration_verified": True,
            "file_sha256": files,
            "status": "COMPLETE",
        }
        with open(os.path.join(staging, COMPLETE_MARKER), "w") as fh:
            json.dump(marker, fh, indent=1, sort_keys=True)
        os.rename(staging, dest)          # atomic finalization
        staging = None
        return dest
    finally:
        if staging and os.path.isdir(staging):
            failed = os.path.join(os.path.dirname(dest),
                                  FAILED_PREFIX + os.path.basename(dest)
                                  + "_" + os.path.basename(staging))
            shutil.move(staging, failed)


def read_complete_cells(root: str) -> List[Dict]:
    """Every COMPLETE cell marker under `root`. Partial cells are ignored."""
    out = []
    for dirpath, _, files in os.walk(root):
        if COMPLETE_MARKER in files:
            with open(os.path.join(dirpath, COMPLETE_MARKER)) as fh:
                m = json.load(fh)
            m["_dir"] = dirpath
            out.append(m)
    return sorted(out, key=lambda m: m["cell_key"])


def verify_cell_integrity(cell_marker: Dict) -> None:
    """Re-hash a finalized cell's files against its own marker."""
    d = cell_marker["_dir"]
    for name, want in cell_marker["file_sha256"].items():
        p = os.path.join(d, name)
        if not os.path.exists(p):
            raise CellError(f"{cell_marker['cell_key']}: missing {name}")
        got = _sha_bytes(open(p, "rb").read())
        if got != want:
            raise CellError(f"{cell_marker['cell_key']}: {name} altered")


def retry_allowed(root: str, row: Dict) -> bool:
    """Operational retry rule. Performance-blind by construction.

    A retry is permitted ONLY for a cell that was never finalized COMPLETE.
    No scientific value is consulted anywhere in this decision.
    """
    return not is_complete(root, row)
