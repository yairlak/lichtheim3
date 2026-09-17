#!/usr/bin/env python3
"""C-ALIGN CAUSAL PILOT — arm launcher (refuses far more often than it runs).

The launcher owns no science.  It reads ONE immutable launch config, refuses to
start unless every precondition holds, and then execs the historical training
driver with exactly the frozen fields.  No scientific field can be supplied or
overridden on this command line.

Refusals (CENTRAL §13 tests 15-20):
  * dirty tracked worktree;
  * contract sha256 != the config's pinned value;
  * implementation digest != the config's pinned value;
  * target run namespace already exists and is not empty;
  * SOURCE checkpoint / parameter / optimizer-state hash mismatch;
  * strict determinism unavailable on the configured device;
  * ON/OFF declaration inconsistent with the configured weight.

Without --execute it prints the launch plan and exits 0, having trained nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.c_align_pilot.common import (  # noqa: E402
    BUDGET_STEPS, CONTRACT_PATH, N_CHECKPOINTS, SAVE_EVERY, check_state_identity,
    determinism_probe, git, load_config, sha256_file, sha256_text)

# Files whose bytes define the frozen implementation.
IMPLEMENTATION_FILES = (
    "scripts/naming_comprehension/train_joint_scratch.py",
    "scripts/c_align_pilot/common.py",
    "scripts/c_align_pilot/evaluate_checkpoint.py",
    "scripts/c_align_pilot/launch_pilot_arm.py",
    "scripts/c_align_pilot/preflight.py",
    "tests/test_c_align_pilot.py",
)


def implementation_digest(root: str = ROOT) -> str:
    """One digest over the frozen implementation files (launcher excluded from
    its own digest is NOT done: the launcher is frozen too, and the digest is
    recorded in the manifest at freeze time)."""
    parts = []
    for rel in IMPLEMENTATION_FILES:
        parts.append(f"{rel}:{sha256_file(os.path.join(root, rel))}")
    return sha256_text("\n".join(parts) + "\n")


class Refusal(RuntimeError):
    pass


def preconditions(cfg: Dict[str, object], root: str = ROOT,
                  require_clean: bool = True) -> Dict[str, object]:
    rep: Dict[str, object] = {"arm": cfg["arm"], "refusals": []}

    def refuse(msg: str) -> None:
        rep["refusals"].append(msg)

    status = git(root, "status", "--porcelain")
    rep["worktree_clean"] = not status
    rep["head"] = git(root, "rev-parse", "HEAD")
    if require_clean and status:
        refuse(f"dirty tracked worktree: {status.splitlines()[:3]}")

    got = sha256_file(os.path.join(root, CONTRACT_PATH))
    rep["contract_sha256"] = got
    if got != cfg["pilot_contract_sha256"]:
        refuse(f"contract sha256 {got} != pinned {cfg['pilot_contract_sha256']}")

    digest = implementation_digest(root)
    rep["implementation_digest"] = digest
    pinned = cfg.get("implementation_digest")
    if pinned and digest != pinned:
        refuse(f"implementation digest {digest} != pinned {pinned}")

    runs = os.environ.get(str(cfg["out_dir_env"]))
    rep["runs_root"] = runs
    if not runs:
        refuse(f"{cfg['out_dir_env']} is not set: no target run namespace")
    else:
        run_dir = os.path.join(runs, str(cfg["run_id"]))
        rep["run_dir"] = run_dir
        if os.path.isdir(run_dir) and os.listdir(run_dir):
            refuse(f"target run namespace is not empty: {run_dir}")

    src = os.path.join(os.path.dirname(root), str(cfg["source_checkpoint"]))
    rep["source_checkpoint"] = src
    if not os.path.exists(src):
        refuse(f"SOURCE checkpoint not found: {src}")
    else:
        ident, _tr, _ckd = check_state_identity(src, cfg)
        rep["state_identity"] = ident
        if not ident["pass"]:
            refuse("SOURCE state identity mismatch (file/params/optimizer/step/seed)")

    det = determinism_probe(str(cfg["device"]))
    rep["determinism"] = det
    if not det.get("pass"):
        refuse(f"strict determinism unavailable: {det.get('reason', 'probe failed')}")

    on = bool(cfg["declare_c_align_transition"])
    w = float(cfg["c_align_weight"])
    rep["factor_ok"] = (on and w == 0.1) or ((not on) and w == 0.0)
    if not rep["factor_ok"]:
        refuse(f"declaration {on} inconsistent with c_align_weight {w}")

    if int(cfg["end_step"]) - int(cfg["start_step"]) != BUDGET_STEPS:
        refuse("budget != 138,900 steps")
    if int(cfg["save_every"]) != SAVE_EVERY or len(
            cfg["checkpoint_steps"]) != N_CHECKPOINTS:
        refuse("checkpoint cadence does not match the frozen plan")

    rep["pass"] = not rep["refusals"]
    return rep


def build_argv(cfg: Dict[str, object], root: str = ROOT) -> List[str]:
    runs = os.environ[str(cfg["out_dir_env"])]
    glove = os.environ.get("L3_GLOVE", "data/glove.6B.300d.txt")
    argv = [
        sys.executable, "-u",
        os.path.join(root, "scripts/naming_comprehension/train_joint_scratch.py"),
        "--regime", str(cfg["regime"]), "--seed", str(cfg["seed"]),
        "--subset-mode", str(cfg["subset_mode"]),
        "--schedule", str(cfg["schedule"]),
        "--wm-hidden", str(cfg["wm_hidden"]),
        "--enc-hidden", str(cfg["enc_hidden"]),
        "--dec-hidden", str(cfg["dec_hidden"]),
        "--device", str(cfg["device"]),
        "--max-steps", str(cfg["max_steps"]),
        "--lr-repetition", repr(cfg["lr_repetition"]),
        "--lr-naming", repr(cfg["lr_naming"]),
        "--lr-comprehension", repr(cfg["lr_comprehension"]),
        "--glove-path", glove,
        "--save-every", str(cfg["save_every"]),
        "--eval-every", "0", "--probe-every", "0", "--log-every", "500",
        "--torch-deterministic",
        "--c-align-weight", repr(cfg["c_align_weight"]),
        "--batch-digest-path", os.path.join(runs, str(cfg["batch_digest_file"])),
        "--pilot-contract-sha256", str(cfg["pilot_contract_sha256"]),
        "--out-dir", runs, "--run-id", str(cfg["run_id"]),
        "--resume", os.path.join(os.path.dirname(root),
                                 str(cfg["source_checkpoint"])),
    ]
    if cfg["declare_c_align_transition"]:
        argv.append("--declare-c-align-transition")
    return argv


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=(
        "W3_OFF", "W3_ON", "W4_OFF", "W4_ON"))
    ap.add_argument("--execute", action="store_true",
                    help="actually start training; without it nothing runs")
    ap.add_argument("--report", default=None, help="write the gate report here")
    a = ap.parse_args(argv)

    cfg = load_config(a.arm)
    rep = preconditions(cfg)
    rep["launch_argv"] = build_argv(cfg) if rep["pass"] else None
    if a.report:
        os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
        json.dump(rep, open(a.report, "w"), indent=1, sort_keys=True)
    print(json.dumps({k: rep[k] for k in ("arm", "pass", "refusals")}, indent=1))
    if not rep["pass"]:
        print("REFUSED: not launching", file=sys.stderr)
        return 2
    if not a.execute:
        print("DRY RUN: preconditions pass; --execute not given, nothing trained")
        return 0
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = str(cfg["cublas_workspace_config"])
    return subprocess.call(rep["launch_argv"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
