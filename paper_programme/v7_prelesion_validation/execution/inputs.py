"""Machine-portable EXECUTION INPUT resolution for the V7 pre-lesion validation.

This module is deliberately OUTSIDE the frozen scientific contract. It resolves
*where* the inputs live on a given machine; it never decides *what* they are.
Every scientific identity (populations, SHAs, head convention, decode rules)
comes from `contract/`, which is frozen at design commit 2d240e1f.

Two inputs must be supplied per machine:

    L3_CANON_TABLE   path to canonical_behavioral_item_table.tsv
    L3_V7_RUN_ROOT   root holding the four V7 run directories

Both may instead be given by an execution-input manifest:

    L3_EXECUTION_INPUTS=/path/to/execution_inputs.json
    {"canon_table": "...", "v7_run_root": "...",
     "artifacts": {"P1": {"source_checkpoint": "...", "repair_head": "..."}, ...}}

`artifacts` is an optional per-slot override for layouts that differ from the
frozen V7 one; when absent, paths are derived deterministically from the slurm
layout (see `derive_artifact_paths`).

EVERYTHING FAILS CLOSED: a resolved file must exist AND match the frozen SHA256
recorded in the contract, or resolution raises. No related table is ever
silently substituted, and path logistics never change a population.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACT_DIR = os.path.normpath(os.path.join(HERE, "..", "contract"))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))

# Frozen identity of the canonical historical WFE table (contract section 2).
CANON_TABLE_SHA256 = \
    "8988aff6fac55ca36cb43ce758f5684f30ae10a6303bdbd7b0b9f462433d5a67"

# Frozen V7 run layout, read out of scripts/cluster/jeanzay/fresh_ceiling_v7.slurm:
#   RUN_ID   = fresh_ceiling_v7_{slot}_s{seed}        (fresh_ceiling_v7.py:119)
#   RUN_DIR  = $RUNS/$RUN_ID                          (slurm:132)
#   CKPT     = $RUN_DIR/checkpoints/step_%08d.pt      (slurm:133, 288)
#   ARM_DIR  = $RUN_DIR/post/seed{seed}_u{u}_A        (slurm:134, 303)
#   HEAD     = $ARM_DIR/head_first_c0.pt              (slurm:304)
RUN_ID_FMT = "fresh_ceiling_v7_{slot}_s{seed}"
REPAIR_HEAD_BASENAME = "head_first_c0.pt"

# Repo-relative fallbacks tried when L3_CANON_TABLE is unset. Each candidate is
# still SHA-verified; a candidate that exists but mismatches raises.
CANON_TABLE_FALLBACKS = (
    os.path.join(REPO_ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
                 "behavioral_analysis", "tables",
                 "canonical_behavioral_item_table.tsv"),
    os.path.join(REPO_ROOT, "..", "lichtheim3", "outputs",
                 "behavioral_wfe_fulllexicon_93a577f", "behavioral_analysis",
                 "tables", "canonical_behavioral_item_table.tsv"),
)


class InputResolutionError(RuntimeError):
    """Raised when an execution input is missing or fails its frozen SHA."""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(path: str, expected_sha: str, what: str) -> str:
    if not os.path.exists(path):
        raise InputResolutionError(f"{what}: file does not exist: {path}")
    got = sha256_file(path)
    if got != expected_sha:
        raise InputResolutionError(
            f"{what}: SHA256 mismatch (fail closed)\n"
            f"  path     {path}\n  expected {expected_sha}\n  got      {got}")
    return os.path.abspath(path)


def _manifest_rows() -> List[dict]:
    with open(os.path.join(CONTRACT_DIR, "state_manifest.tsv")) as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def execution_inputs() -> dict:
    p = os.environ.get("L3_EXECUTION_INPUTS")
    if not p:
        return {}
    if not os.path.exists(p):
        raise InputResolutionError(f"L3_EXECUTION_INPUTS does not exist: {p}")
    return json.load(open(p))


# --------------------------------------------------------- canonical table --
def canon_table_path(required: bool = True) -> Optional[str]:
    """Resolve and SHA-verify the canonical WFE table, or return None.

    Order: L3_EXECUTION_INPUTS["canon_table"], L3_CANON_TABLE, repo fallbacks.
    An explicitly supplied path that is missing or mismatched always raises --
    it is never downgraded to "not configured".
    """
    explicit = execution_inputs().get("canon_table") or \
        os.environ.get("L3_CANON_TABLE")
    if explicit:
        return _require(explicit, CANON_TABLE_SHA256, "canonical WFE table")
    for cand in CANON_TABLE_FALLBACKS:
        if os.path.exists(cand):
            return _require(cand, CANON_TABLE_SHA256, "canonical WFE table")
    if required:
        raise InputResolutionError(
            "canonical WFE table not configured. Set L3_CANON_TABLE to the file "
            f"with sha256 {CANON_TABLE_SHA256}, or provide L3_EXECUTION_INPUTS.")
    return None


# ------------------------------------------------------------- V7 artifacts --
def v7_run_root(required: bool = True) -> Optional[str]:
    root = execution_inputs().get("v7_run_root") or os.environ.get("L3_V7_RUN_ROOT")
    if root:
        if not os.path.isdir(root):
            raise InputResolutionError(f"L3_V7_RUN_ROOT is not a directory: {root}")
        return os.path.abspath(root)
    if required:
        raise InputResolutionError(
            "V7 run root not configured. Set L3_V7_RUN_ROOT to the directory "
            "holding fresh_ceiling_v7_p{1..4}_s{31..34}/.")
    return None


def derive_artifact_paths(run_root: str, slot: str, seed: int,
                          step: int, source_u: int) -> Dict[str, str]:
    """Deterministically derive the frozen V7 file locations under `run_root`.

    Layout comes from the frozen slurm driver, not from this machine.
    """
    run_id = RUN_ID_FMT.format(slot=slot.lower(), seed=int(seed))
    run_dir = os.path.join(run_root, run_id)
    return {
        "run_id": run_id,
        "source_checkpoint": os.path.join(run_dir, "checkpoints",
                                          f"step_{int(step):08d}.pt"),
        "repair_head": os.path.join(run_dir, "post",
                                    f"seed{int(seed)}_u{int(source_u)}_A",
                                    REPAIR_HEAD_BASENAME),
    }


def resolve_artifacts(required: bool = True) -> Optional[Dict[str, dict]]:
    """Resolve + SHA-verify SOURCE checkpoint and head_first_c0 for P1-P4.

    Returns {slot: {source_checkpoint, repair_head, and their frozen SHAs}}.
    Fails closed on any missing file or SHA mismatch.
    """
    root = v7_run_root(required=required)
    if root is None:
        return None
    overrides = execution_inputs().get("artifacts") or {}
    out: Dict[str, dict] = {}
    for r in _manifest_rows():
        if r["state_kind"] != "POST_REPAIR":
            continue          # SOURCE shares the same checkpoint identity
        slot = r["slot"]
        derived = derive_artifact_paths(root, slot, int(r["training_seed"]),
                                        int(r["source_global_step"]),
                                        int(r["source_u"]))
        ov = overrides.get(slot, {})
        ckpt = ov.get("source_checkpoint", derived["source_checkpoint"])
        head = ov.get("repair_head", derived["repair_head"])
        out[slot] = {
            "run_id": derived["run_id"],
            "source_checkpoint": _require(
                ckpt, r["source_checkpoint_sha256"], f"{slot} SOURCE checkpoint"),
            "source_checkpoint_sha256": r["source_checkpoint_sha256"],
            "repair_head": _require(
                head, r["repair_head_file_sha256"], f"{slot} head_first_c0.pt"),
            "repair_head_file_sha256": r["repair_head_file_sha256"],
            "repair_head_deployed_state_sha256":
                r["repair_head_deployed_state_sha256"],
        }
    return out


def artifacts_available() -> bool:
    """True only if every P1-P4 artifact resolves AND matches its frozen SHA."""
    try:
        return resolve_artifacts(required=True) is not None
    except InputResolutionError:
        return False


def canon_table_available() -> bool:
    try:
        return canon_table_path(required=True) is not None
    except InputResolutionError:
        return False


# ------------------------------------------ external aggregate invariants ----
# The frozen official V7 battery outcomes (contract section 1). These are the
# external expected invariant for the POST_REPAIR global battery during the
# scientific run. They are NOT used by the smoke tests.
FROZEN_OFFICIAL_POST_BATTERY = {
    "P1": {"Rcan": 0, "Rfree": 0, "N": 0, "C": 0},
    "P2": {"Rcan": 0, "Rfree": 0, "N": 0, "C": 0},
    "P3": {"Rcan": 0, "Rfree": 0, "N": 0, "C": 0},
    "P4": {"Rcan": 1, "Rfree": 1, "N": 0, "C": 0},
}

# Predeclared deterministic smoke sets, fixed here BEFORE any execution and
# chosen without reference to any scientific outcome: the lowest bank indices
# and the lexicographically first primary item ids.
SMOKE_REAL_BANK_INDICES = list(range(64))
SMOKE_PSEUDOWORD_N = 8
