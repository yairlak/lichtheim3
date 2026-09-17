"""C-ALIGN CAUSAL PILOT — shared, read-only helpers.

Launch-config loading, hashing, the prospective task/batch digest derivation and
the pre-training gate implementations.  Nothing in this module trains, and no
function here calls `optimizer.step()`.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CONFIG_DIR = "paper_programme/c_align_causal_pilot/launch_configs"
CONTRACT_PATH = "paper_programme/c_align_causal_pilot/C_ALIGN_CAUSAL_PILOT_CONTRACT.md"
ARMS = ("W3_OFF", "W3_ON", "W4_OFF", "W4_ON")

# Frozen scientific constants (contract §4, §10, §11, §17.1).
BUDGET_STEPS = 138_900
SAVE_EVERY = 13_890
N_CHECKPOINTS = 10
CYCLE_STEPS = 6
C_ALIGN_ON = 0.1
C_ALIGN_OFF = 0.0

# Baseline readouts that V7 requires, from the closed ventral-interface results.
BASELINE = {
    "W3": {"S0_freear_errors": 3199, "S0_canonical_errors": 3199,
           "S1c_freear_errors": 42, "S1c_canonical_errors": 42,
           "c_top1_errors": 34, "wm_canonical_errors": 0,
           "full_rep_canonical_errors": 0, "naming_strict_errors": 0},
    "W4": {"S0_freear_errors": 3647, "S0_canonical_errors": 3647,
           "S1c_freear_errors": 51, "S1c_canonical_errors": 51,
           "c_top1_errors": 42, "wm_canonical_errors": 1,
           "full_rep_canonical_errors": 0, "naming_strict_errors": 0},
}

# Recipe fields V4 verifies after the checkpoint is loaded.
RECIPE = {
    "regime": "j0", "schedule": "interleaved_123", "subset_mode": "final_full",
    "lr_repetition": 3e-5, "lr_naming": 3e-5, "lr_comprehension": 1e-4,
    "lambda_C": 0.087, "lambda_N": 1.0, "tau": 0.1, "dec_weight": 0.5,
    "batch_size": 64, "wm_hidden": 128, "enc_hidden": 512, "dec_hidden": 512,
    "optimizer_policy": "shared_adamw",
    "loss_weights": {"rep": 1.0, "align": 1.0, "dec": 0.5, "wm": 0.5, "gate": 0.05},
    "comprehension_population_n": 27981,
    "comprehension_population_sha256":
        "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_config(arm: str, root: str = ROOT) -> Dict[str, object]:
    path = os.path.join(root, CONFIG_DIR, f"{arm}.json")
    cfg = json.load(open(path))
    cfg["_path"] = path
    cfg["_sha256"] = sha256_file(path)
    return cfg


def git(root: str, *args: str) -> str:
    return subprocess.run(["git", "-C", root, *args], capture_output=True,
                          text=True, check=True).stdout.strip()


# --------------------------------------------------- prospective V6 sequence
def expected_digest_sequence(tr, start_step: int, n_steps: int,
                             cursors: Optional[Dict[str, int]] = None
                             ) -> Tuple[List[str], str, List[str]]:
    """Derive the task/batch digest sequence the run WILL produce, without
    training anything.

    The task at a step is a pure function of (schedule seed, anchor, step), and
    each stream's batch is a pure function of its own cursor, so the whole
    sequence is determined by the restored checkpoint state.  Local cursor
    copies are advanced here; the trainer's own cursors are never touched.

    Returns (per-cycle digests, cumulative digest, ordered task labels).
    """
    from scripts.naming_comprehension.train_joint_scratch import TASK_STREAMS
    cur = dict(cursors if cursors is not None else tr.cursors)
    cycle_digests: List[str] = []
    tasks: List[str] = []
    cumulative = ""
    pending: List[str] = []
    for step in range(int(start_step), int(start_step) + int(n_steps)):
        task = tr.task_for_step(step)
        tasks.append(task)
        parts = [f"{step}\t{task}"]
        for stream in TASK_STREAMS[task]:
            idx = tr.streams[stream].indices(cur[stream])
            parts.append(f"{stream}:" + ",".join(str(int(i)) for i in idx))
        pending.append("\t".join(parts))
        for stream in TASK_STREAMS[task]:
            cur[stream] += 1
        rel = step - int(tr.schedule_anchor_step)
        if (rel + 1) % CYCLE_STEPS == 0:
            digest = sha256_text("\n".join(pending) + "\n")
            cumulative = sha256_text(cumulative + digest)
            cycle_digests.append(digest)
            pending = []
    if pending:
        raise RuntimeError("step range does not end on a macro-cycle boundary")
    return cycle_digests, cumulative, tasks


def read_digest_file(path: str) -> Tuple[List[str], str]:
    """(per-cycle digests, final cumulative digest) from a training run's log."""
    rows = [l.rstrip("\n").split("\t") for l in open(path).read().splitlines()]
    if not rows or rows[0][0] != "cycle_index":
        raise RuntimeError(f"{path}: not a batch-digest file")
    body = rows[1:]
    return [r[4] for r in body], (body[-1][5] if body else "")


# ------------------------------------------------------------------- gates --
def check_state_identity(ck_path: str, expect: Dict[str, str],
                         device: str = "cpu") -> Dict[str, object]:
    """V3: source file, parameter and optimizer-state hashes of a start state.

    Hashes are produced by the SAME frozen helpers that recorded the design
    package values: `gate_x_lesion.hooks.state_dict_sha256` for parameters and
    the gradient diagnostic's `optimizer_state_sha256` for the shared AdamW
    state.  Building the trainer performs no update.
    """
    import torch
    from gate_x_lesion.hooks import state_dict_sha256
    from scripts.c_align_design.c_align_gradient_diagnostic import (
        optimizer_state_sha256)
    from scripts.naming_comprehension.frozen_head_probe import build_trainer

    got_file = sha256_file(ck_path)
    glove = os.environ.get(
        "L3_GLOVE",
        os.path.join(os.path.dirname(ROOT), "lichtheim3/data/glove.6B.300d.txt"))
    tr, ckd = build_trainer(ck_path, device, glove)
    osd = ckd["optimizer_state_dict"]
    out = {
        "checkpoint_sha256": got_file,
        "params_state_dict_sha256": state_dict_sha256(tr.model),
        "optimizer_state_sha256": optimizer_state_sha256(tr.optim),
        "global_step": int(ckd["global_step"]),
        "seed": int(ckd["seed"]),
        "c_align_weight": float(ckd.get("c_align_weight", 0.0)),
        "optimizer_policy": ckd.get("optimizer_policy", "shared_adamw"),
        "n_optimizer_tensors": len(osd["state"]),
        "has_single_shared_optimizer": ("optimizer_state_dict" in ckd
                                        and "optimizer_states" not in ckd),
        "rng_states_present": sorted((ckd.get("rng_states") or {}).keys()),
        "cursors": {k: int(v) for k, v in ckd["cursors"].items()},
        "schedule_anchor_step": int(ckd.get("schedule_anchor_step", -1)),
        "schedule": ckd.get("schedule"),
        "adam_steps_present": all(
            "step" in st for st in osd["state"].values()),
    }
    out["pass"] = all([
        got_file == expect["source_sha256"],
        out["params_state_dict_sha256"] == expect["params_sha256"],
        out["optimizer_state_sha256"] == expect["optimizer_state_sha256"],
        out["global_step"] == int(expect["start_step"]),
        out["seed"] == int(expect["seed"]),
        out["c_align_weight"] == 0.0,
        out["has_single_shared_optimizer"],
        out["optimizer_policy"] == "shared_adamw",
        out["adam_steps_present"],
    ])
    return out, tr, ckd


def determinism_probe(device: str) -> Dict[str, object]:
    """V5: can strict deterministic execution actually be enabled HERE?"""
    import torch
    rep: Dict[str, object] = {
        "requested_device": device,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device.startswith("cuda") and not torch.cuda.is_available():
        rep["pass"] = False
        rep["reason"] = ("the configured training device is CUDA but no CUDA "
                         "device is visible in this environment")
        return rep
    try:
        torch.use_deterministic_algorithms(True)
        if device.startswith("cuda"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                rep["pass"] = False
                rep["reason"] = "CUBLAS_WORKSPACE_CONFIG is not set for CUDA determinism"
                return rep
        rep["deterministic_algorithms_enabled"] = bool(
            torch.are_deterministic_algorithms_enabled())
        rep["pass"] = bool(rep["deterministic_algorithms_enabled"])
    except Exception as exc:                                  # pragma: no cover
        rep["pass"] = False
        rep["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        torch.use_deterministic_algorithms(False)
    return rep


def closed_artifacts_report(root: str) -> Dict[str, object]:
    """V10: the closed packages and the SOURCE checkpoints, by hash."""
    base = os.path.dirname(root)
    out: Dict[str, object] = {"sources": {}, "packages": {}}
    for sid, rel, want in (
        ("W3_SRC", "probe_v6_completion_20260912/sources/s19_step_10625850.pt",
         "a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c"),
        ("W4_SRC", "probe_v6_completion_20260912/sources/s20_step_08445120.pt",
         "0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3"),
    ):
        got = sha256_file(os.path.join(base, rel))
        out["sources"][sid] = {"sha256": got, "expected": want, "pass": got == want}
    for name, wt, rel in (
        ("ventral_interface",
         "wt-ventral-interface",
         "paper_programme/ventral_semantic_interface/scientific_execution/SHA256SUMS"),
        ("directional_dose",
         "wt-ventral-directional-dose",
         "paper_programme/ventral_semantic_directional_dose/scientific_execution/SHA256SUMS"),
    ):
        pkg_root = os.path.join(base, wt, os.path.dirname(os.path.dirname(rel)))
        sums = os.path.join(base, wt, rel)
        ok = fail = 0
        for line in open(sums).read().splitlines():
            want, _, target = line.partition("  ")
            p = os.path.join(pkg_root, target)
            if os.path.exists(p) and sha256_file(p) == want:
                ok += 1
            else:
                fail += 1
        out["packages"][name] = {"verified": ok, "failed": fail, "pass": fail == 0}
    out["pass"] = (all(v["pass"] for v in out["sources"].values())
                   and all(v["pass"] for v in out["packages"].values()))
    return out
