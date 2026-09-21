"""Full executable preflight (CENTRAL section 11, checks A-L).

Every check fails closed. None attempts repair. No scientific value is consulted
anywhere; nothing here depends on a lesion outcome.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from typing import Dict, List

from paper_programme.lesioning_v2.lesion_operator import (
    battery, masks, sd_procedure, seeds)
from paper_programme.lesioning_v2.lesion_operator.sites import SITES, SITE_ORDER
from . import authorization

PKG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
CONTRACT = os.path.join(PKG, "contract")

TRANSFER_REQUIRED = {
    "LESIONING_V2_STATE_VERIFICATION.json":
        "d216fe562396be19d918a1f7c433839e846ca94602c6aae1e21754c931f96349",
    "LESIONING_V2_INTACT_SD_CONSTANTS.json":
        "3c6dcd3595d7432d592bcecbc691d33ec117f816932771ec1fccc0b0001b0602",
}
REQUIRED_TORCH = "2.6.0"
EXPECTED_REALIZATIONS = {"P1_POST_REPAIR": 12, "P2_POST_REPAIR": 12,
                         "P3_POST_REPAIR": 12, "P4_POST_REPAIR": 4}

#: Files whose bytes define scientific behaviour. Their hashes are the evaluator
#: identity; they are the only files that may not drift after the freeze.
EVALUATOR_IDENTITY_FILES = (
    "lesion_operator/masks.py", "lesion_operator/noise.py",
    "lesion_operator/context.py", "lesion_operator/sites.py",
    "lesion_operator/seeds.py", "lesion_operator/sd_procedure.py",
    "lesion_operator/battery.py",
    "execution/injection.py", "execution/evaluators.py",
    # k>0 scientific output depends on this driver's bytes: it selects the
    # batch boundaries at which each frozen evaluator is invoked and the ids
    # the perturbation is derived from.
    "execution/lesioned_eval.py",
)


class PreflightError(RuntimeError):
    pass


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def evaluator_identity() -> Dict[str, str]:
    out = {}
    for rel in EVALUATOR_IDENTITY_FILES:
        p = os.path.join(PKG, rel)
        if not os.path.exists(p):
            raise PreflightError(f"evaluator identity file missing: {rel}")
        out[rel] = sha256_file(p)
    out["_combined"] = hashlib.sha256(
        json.dumps({k: v for k, v in sorted(out.items())},
                   sort_keys=True).encode()).hexdigest()
    return out


def load_matrix() -> Dict:
    p = os.path.join(CONTRACT, "LESIONING_V2_RUN_MATRIX.json")
    if not os.path.exists(p):
        raise PreflightError(f"authoritative run matrix missing: {p}")
    return json.load(open(p))


def _recompute_matrix_sha(m: Dict) -> str:
    body = {k: v for k, v in m.items() if k != "matrix_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def run(out_root: str, *, require_authorization: bool,
        transfer_dir: str = None, allow_missing_transfer: bool = False,
        continuity_manifest: str = None) -> Dict:
    """Execute checks A-L. `require_authorization` is False only for --dry-run.

    `continuity_manifest` enables the ONLY sanctioned way to run into an
    existing namespace: the manifest must validate that namespace as exactly
    the 12 COMPLETE k=0 controls and nothing else. Absent it, check L refuses
    any existing namespace, as before.
    """
    rep: Dict[str, object] = {}

    # ---- A. execution commit (verified against the authorization, below) ---
    head = _git("rev-parse", "HEAD")
    if not head:
        raise PreflightError("cannot resolve git HEAD")
    rep["head_commit"] = head

    # ---- B. clean tree -----------------------------------------------------
    dirty = [l for l in _git("status", "--porcelain").splitlines() if l.strip()]
    rep["tree_clean"] = not dirty
    if require_authorization and dirty:
        raise PreflightError(
            "REFUSED: working tree is not clean before scientific execution:\n"
            + "\n".join(dirty[:20]))

    # ---- C. matrix identity ------------------------------------------------
    m = load_matrix()
    recomputed = _recompute_matrix_sha(m)
    if recomputed != m.get("matrix_sha256"):
        raise PreflightError("run matrix self-hash does not recompute")
    if recomputed == authorization.SUPERSEDED_RUN_MATRIX_SHA256:
        raise PreflightError("REFUSED: superseded pre-cluster run matrix")
    if recomputed != authorization.CLUSTER_RUN_MATRIX_SHA256:
        raise PreflightError(
            f"run matrix {recomputed} is not the authoritative "
            f"{authorization.CLUSTER_RUN_MATRIX_SHA256}")
    rep["matrix_sha256"] = recomputed
    rep["n_cells"] = m["n_cells"]

    # ---- D. transfer-required artifacts ------------------------------------
    transfer_status = {}
    for name, want in TRANSFER_REQUIRED.items():
        candidates = [os.path.join(CONTRACT, name)]
        if transfer_dir:
            candidates.insert(0, os.path.join(transfer_dir, name))
        found = next((p for p in candidates if os.path.exists(p)), None)
        if found is None:
            transfer_status[name] = "ABSENT"
            if not allow_missing_transfer:
                raise PreflightError(
                    f"transfer-required artifact absent: {name}\n"
                    f"  expected sha256 {want}\n"
                    "  stage it from the execution-control directory "
                    "(--transfer-dir) or place it in contract/")
            continue
        got = sha256_file(found)
        if got != want:
            raise PreflightError(
                f"{name}: SHA mismatch (fail closed)\n  expected {want}\n"
                f"  got      {got}")
        transfer_status[name] = found
    rep["transfer_artifacts"] = transfer_status

    # ---- E. package integrity ----------------------------------------------
    sums = os.path.join(PKG, "SHA256SUMS")
    if not os.path.exists(sums):
        raise PreflightError("package SHA256SUMS missing")
    bad = []
    for line in open(sums):
        line = line.strip()
        if not line:
            continue
        want, rel = line.split(None, 1)
        p = os.path.join(PKG, rel.strip())
        if not os.path.exists(p):
            bad.append(f"missing {rel.strip()}")
        elif sha256_file(p) != want:
            bad.append(f"altered {rel.strip()}")
    if bad:
        raise PreflightError("package integrity failure:\n" + "\n".join(bad[:20]))
    rep["package_integrity"] = "OK"

    # ---- F. state identities ------------------------------------------------
    sv = transfer_status.get("LESIONING_V2_STATE_VERIFICATION.json")
    if sv and sv != "ABSENT":
        states = json.load(open(sv)).get("states", {})
        for cell in m["cells"]:
            sid = cell["state_id"]
            if not sid.endswith("_POST_REPAIR"):
                raise PreflightError(f"non-POST_REPAIR state in matrix: {sid}")
            if states and sid not in states:
                raise PreflightError(f"{sid} absent from state verification")
        rep["state_identities"] = "VERIFIED"
    else:
        rep["state_identities"] = "PENDING_TRANSFER"
    if m.get("source_states_included"):
        raise PreflightError("REFUSED: SOURCE states present in the matrix")

    # ---- G. scientific configuration ---------------------------------------
    if masks.CONNECTIVITY_P_MAX != 0.30:
        raise PreflightError("p_max changed")
    if masks.N_SEVERITY_LEVELS != 15:
        raise PreflightError("severity level count changed")
    expect_sites = {"L1": "wm.encoder.weight_ih_l0",
                    "L2": "ltm.encoder.weight_ih_l0",
                    "L3": "ltm.sem_to_h0.weight"}
    for name, param in expect_sites.items():
        if SITES[name].connectivity_param != param:
            raise PreflightError(f"{name} connectivity tensor changed")
    counts: Dict[str, set] = {}
    for c in m["cells"]:
        counts.setdefault(c["state_id"], set()).add(c["realization"])
    for sid, want in EXPECTED_REALIZATIONS.items():
        if len(counts.get(sid, set())) != want:
            raise PreflightError(f"{sid} realization count != {want}")
    if sd_procedure.procedure_hash() != \
            "122ae40c600e639f875bdcb18e588bd32242981af0fa42c0e975a296bc851942":
        raise PreflightError("SD procedure hash changed")
    if seeds.NAMESPACE != "L3_LESION_V2_V1":
        raise PreflightError("seed namespace changed")
    rep["scientific_configuration"] = "UNCHANGED"

    # ---- H. evaluator identity ----------------------------------------------
    rep["evaluator_identity"] = evaluator_identity()["_combined"]
    rep["primary_endpoints"] = [e.key for e in battery.PRIMARY]

    # ---- I. environment ------------------------------------------------------
    try:
        import torch
        tv = torch.__version__
    except Exception:
        tv = None
    rep["torch_version"] = tv
    if require_authorization and tv != REQUIRED_TORCH:
        raise PreflightError(
            f"REFUSED: torch {tv} is not the canonical {REQUIRED_TORCH}")

    # ---- J. data / model / population identities ----------------------------
    sd_path = transfer_status.get("LESIONING_V2_INTACT_SD_CONSTANTS.json")
    if sd_path and sd_path != "ABSENT":
        sd = json.load(open(sd_path))
        if sd.get("procedure_hash") != sd_procedure.procedure_hash():
            raise PreflightError("SD constants were produced by a different "
                                 "procedure than the frozen one")
        need = {f"{c['state_id']}/{c['site']}" for c in m["cells"]}
        have = set(sd.get("constants", {}))
        if not need <= have:
            raise PreflightError(
                f"SD constants missing for {sorted(need - have)[:5]}")
        rep["sd_constants"] = "VERIFIED"
    else:
        rep["sd_constants"] = "PENDING_TRANSFER"

    # ---- K. no training path -------------------------------------------------
    banned_calls = ("backward", "step", "zero_grad", "requires_grad_", "save")
    for rel in EVALUATOR_IDENTITY_FILES + ("execution/cells.py",
                                           "execution/aggregate.py",
                                           "execution/continuity.py",
                                           "execution/preflight.py",
                                           "scripts/run_lesion_v2.py"):
        p = os.path.join(PKG, rel)
        if not os.path.exists(p):
            continue
        tree = ast.parse(open(p).read())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        hit = sorted(set(banned_calls) & called)
        if hit:
            raise PreflightError(f"{rel} can execute a training operation: {hit}")
        ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if any("optim" in x.lower() for x in ids):
            raise PreflightError(f"{rel} names an optimizer")
    rep["no_training_path"] = True

    # ---- L. output namespace -------------------------------------------------
    if continuity_manifest:
        from . import continuity
        if not os.path.exists(continuity_manifest):
            raise PreflightError(
                f"continuity manifest not found: {continuity_manifest}")
        if not os.path.isdir(out_root):
            raise PreflightError(
                "continuation requested but the result namespace does not "
                f"exist: {out_root}")
        try:
            man = json.load(open(continuity_manifest))
            cont = continuity.validate_for_continuation(man, out_root, m)
        except continuity.ContinuityError as e:
            raise PreflightError(f"continuation refused: {e}")
        rep["mode"] = "CONTINUATION"
        rep["continuation"] = cont
        rep["continuity_manifest_sha256"] = man.get("manifest_sha256")
        rep["existing_complete_k0_cells"] = man.get("n_complete_k0_cells")
        rep["output_namespace_absent"] = False
    else:
        if os.path.exists(out_root):
            raise PreflightError(
                f"REFUSED: scientific result namespace already exists: "
                f"{out_root}\n  (a continuation requires --continuity-manifest)")
        rep["mode"] = "FRESH"
        rep["output_namespace_absent"] = True

    # ---- A (completed): authorization binds commit AND matrix ----------------
    if require_authorization:
        auth = authorization.authorize(head, recomputed)
        rep["authorized"] = True
        rep["authorization_execution_commit"] = auth["execution_commit"]
    else:
        rep["authorized"] = False
    return rep
