#!/usr/bin/env python3
"""C-ALIGN CAUSAL PILOT — pre-training preflight, gates V1-V10 (read-only).

Runs every authorized pre-training check and writes one report.  It performs NO
optimizer update: gates that concern the future run verify the CONFIGURED or
EXPECTED requirement now, and are marked so; post-run validation verifies the
ACTUAL outcome.

    python3 scripts/c_align_pilot/preflight.py --out <dir> [--skip-baseline]

Exit code 0 only if every gate passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.c_align_pilot.common import (  # noqa: E402
    ARMS, BASELINE, BUDGET_STEPS, CONTRACT_PATH, N_CHECKPOINTS, RECIPE,
    SAVE_EVERY, check_state_identity, closed_artifacts_report,
    determinism_probe, expected_digest_sequence, git, load_config, sha256_file)
from scripts.c_align_pilot.launch_pilot_arm import (  # noqa: E402
    implementation_digest, preconditions)

PAIR_SOURCE = {"W3": "W3_SRC", "W4": "W4_SRC"}


def v1_driver_amendment() -> Dict[str, object]:
    """The declaration permits exactly 0.0 -> 0.1 and relaxes nothing else."""
    import inspect
    from scripts.naming_comprehension import train_joint_scratch as T
    src = inspect.getsource(T.JointScratchTrainer.load_state_dict)
    rep = {
        "C_ALIGN_PILOT_FROM": T.C_ALIGN_PILOT_FROM,
        "C_ALIGN_PILOT_TO": T.C_ALIGN_PILOT_TO,
        "declaration_flag": "--declare-c-align-transition",
        "guard_mentions_declaration": "declare_c_align_transition" in src,
        "other_guards_present": all(k in src for k in (
            "checkpoint regime", "checkpoint seed", "subset hash",
            "stream seeds", "PHASE TRANSITION")),
        "no_blanket_bypass": "allow_phase_transition" not in src.split(
            "c_align_transition")[0].split("ck_calign")[-1],
    }
    rep["pass"] = bool(
        rep["C_ALIGN_PILOT_FROM"] == 0.0 and rep["C_ALIGN_PILOT_TO"] == 0.1
        and rep["guard_mentions_declaration"] and rep["other_guards_present"])
    return rep


def v2_toy_equivalence() -> Dict[str, object]:
    """Toy-state equivalence is proven by the frozen test module, not here."""
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         os.path.join(ROOT, "tests/test_c_align_pilot.py")],
        capture_output=True, text=True, cwd=ROOT)
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    return {"pytest_returncode": r.returncode, "summary": tail[0],
            "pass": r.returncode == 0}


def v3_v4_state_and_recipe(device: str) -> Dict[str, object]:
    out: Dict[str, object] = {"arms": {}}
    for arm in ARMS:
        cfg = load_config(arm)
        src = os.path.join(os.path.dirname(ROOT), str(cfg["source_checkpoint"]))
        ident, tr, ckd = check_state_identity(src, cfg, device="cpu")
        recipe = {
            "regime": ckd.get("regime"), "schedule": ckd.get("schedule"),
            "subset_mode": ckd.get("subset_mode"),
            "lr_policy": ckd.get("lr_policy"),
            "dec_weight": float(ckd.get("dec_weight", 0.5)),
            "widths": ckd.get("widths"),
            "optimizer_policy": ckd.get("optimizer_policy"),
            "comprehension_population_n": ckd.get("comprehension_population_n"),
            "comprehension_population_sha256": ckd.get(
                "comprehension_population_sha256"),
            "c_align_weight_checkpoint": float(ckd.get("c_align_weight", 0.0)),
            "c_align_weight_configured": float(cfg["c_align_weight"]),
            "declare_c_align_transition": bool(cfg["declare_c_align_transition"]),
            "stop_at_ceiling_configured": bool(cfg["stop_at_ceiling"]),
            "reanchor_configured": bool(cfg["reanchor_schedule"]),
        }
        lp = recipe["lr_policy"] or {}
        recipe["pass"] = all([
            recipe["regime"] == RECIPE["regime"],
            recipe["schedule"] == RECIPE["schedule"],
            recipe["subset_mode"] == RECIPE["subset_mode"],
            abs(float(lp.get("repetition", 0)) - RECIPE["lr_repetition"]) < 1e-12,
            abs(float(lp.get("naming", 0)) - RECIPE["lr_naming"]) < 1e-12,
            abs(float(lp.get("comprehension", 0)) - RECIPE["lr_comprehension"]) < 1e-12,
            recipe["dec_weight"] == RECIPE["dec_weight"],
            recipe["widths"] == {"wm_hidden": RECIPE["wm_hidden"],
                                 "ltm_enc_hidden": RECIPE["enc_hidden"],
                                 "ltm_dec_hidden": RECIPE["dec_hidden"]},
            recipe["optimizer_policy"] == RECIPE["optimizer_policy"],
            recipe["comprehension_population_n"] == RECIPE["comprehension_population_n"],
            recipe["comprehension_population_sha256"] ==
            RECIPE["comprehension_population_sha256"],
            recipe["c_align_weight_checkpoint"] == 0.0,
            not recipe["stop_at_ceiling_configured"],
            not recipe["reanchor_configured"],
            (recipe["c_align_weight_configured"] == 0.1)
            == recipe["declare_c_align_transition"],
        ])
        out["arms"][arm] = {"V3_state_identity": ident, "V4_recipe": recipe,
                            "config_sha256": cfg["_sha256"]}
        del tr, ckd
    out["pass"] = all(a["V3_state_identity"]["pass"] and a["V4_recipe"]["pass"]
                      for a in out["arms"].values())
    return out


def v6_expected_sequence() -> Dict[str, object]:
    """Derive, from restored stream state, the task/batch digest sequence both
    arms of each pair must produce.  Nothing is trained and no SOURCE artifact
    is modified: cursors are advanced on local copies only."""
    from scripts.naming_comprehension.frozen_head_probe import build_trainer
    glove = os.environ.get(
        "L3_GLOVE",
        os.path.join(os.path.dirname(ROOT), "lichtheim3/data/glove.6B.300d.txt"))
    out: Dict[str, object] = {"pairs": {}}
    for pair, _sid in PAIR_SOURCE.items():
        cfg = load_config(f"{pair}_OFF")
        src = os.path.join(os.path.dirname(ROOT), str(cfg["source_checkpoint"]))
        before = sha256_file(src)
        tr, ckd = build_trainer(src, "cpu", glove)
        digests, cumulative, tasks = expected_digest_sequence(
            tr, int(cfg["start_step"]), BUDGET_STEPS)
        counts = {t: tasks.count(t) for t in set(tasks)}
        out["pairs"][pair] = {
            "start_step": int(cfg["start_step"]),
            "n_steps": BUDGET_STEPS,
            "n_macro_cycles": len(digests),
            "task_counts": counts,
            "first_cycle_digest": digests[0],
            "last_cycle_digest": digests[-1],
            "cumulative_digest": cumulative,
            "source_sha256_unchanged": sha256_file(src) == before,
            "cursors_unchanged": {k: int(v) for k, v in tr.cursors.items()} ==
                                 {k: int(v) for k, v in ckd["cursors"].items()},
            "pass": (len(digests) == BUDGET_STEPS // 6
                     and counts.get("repetition") == 23150
                     and counts.get("naming") == 46300
                     and counts.get("comprehension") == 69450
                     and sha256_file(src) == before),
        }
        del tr, ckd
    out["note"] = ("expected sequence only; the ACTUAL logged sequence is "
                   "verified post-run and ON vs OFF within each pair")
    out["pass"] = all(p["pass"] for p in out["pairs"].values())
    return out


def v7_baseline(device: str, out_dir: str) -> Dict[str, object]:
    """Frozen evaluator on every arm's start state, before update 1."""
    from scripts.c_align_pilot.evaluate_checkpoint import evaluate
    rep: Dict[str, object] = {"arms": {}}
    cache: Dict[str, Dict[str, object]] = {}
    for arm in ARMS:
        cfg = load_config(arm)
        pair = str(cfg["pair"])
        src = os.path.join(os.path.dirname(ROOT), str(cfg["source_checkpoint"]))
        if pair not in cache:
            rec = evaluate(src, f"BASELINE_{pair}", device)
            json.dump(rec, open(os.path.join(out_dir, f"baseline_{pair}.json"), "w"),
                      indent=1, sort_keys=True)
            cache[pair] = rec
        rec = cache[pair]
        want = BASELINE[pair]
        got = {k: rec[k] for k in want}
        rep["arms"][arm] = {
            "pair": pair, "expected": want, "observed": got,
            "identical_within_pair": True,   # both arms load the same file
            "S0_matches_global_ltm_freear": rec.get("S0_matches_global_ltm_freear"),
            "pass": got == want and bool(rec.get("S0_matches_global_ltm_freear")),
        }
    rep["pass"] = all(a["pass"] for a in rep["arms"].values())
    return rep


def v8_step_plan() -> Dict[str, object]:
    out: Dict[str, object] = {"arms": {}}
    for arm in ARMS:
        cfg = load_config(arm)
        steps = list(cfg["checkpoint_steps"])
        ok = (int(cfg["end_step"]) - int(cfg["start_step"]) == BUDGET_STEPS
              and int(cfg["save_every"]) == SAVE_EVERY
              and len(steps) == N_CHECKPOINTS
              and steps[-1] == int(cfg["end_step"])
              and all(s % SAVE_EVERY == 0 for s in steps))
        out["arms"][arm] = {"configured_steps": BUDGET_STEPS,
                            "checkpoint_steps": steps, "pass": bool(ok)}
    out["note"] = "configuration only; actual counts verified post-run"
    out["pass"] = all(a["pass"] for a in out["arms"].values())
    return out


def v9_evaluator_freeze(out_dir: str) -> Dict[str, object]:
    from scripts.c_align_pilot.evaluate_checkpoint import EVALUATOR_VERSION
    files = {rel: sha256_file(os.path.join(ROOT, rel)) for rel in (
        "scripts/c_align_pilot/evaluate_checkpoint.py",
        "scripts/c_align_pilot/common.py",
        "ventral_interface/injection.py", "ventral_interface/decode.py",
        "ventral_interface/conditions.py",
        "scripts/naming_comprehension/frozen_probe.py",
        "scripts/naming_comprehension/train_tasks.py")}
    checkpoints_exist = any(
        os.path.isdir(os.path.join(os.environ.get("L3_PILOT_RUNS", "/nonexistent"),
                                   str(load_config(a)["run_id"])))
        for a in ARMS)
    return {"evaluator_version": EVALUATOR_VERSION, "file_sha256": files,
            "pilot_checkpoints_exist_yet": checkpoints_exist,
            "pass": not checkpoints_exist}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu",
                    help="device for the READ-ONLY baseline evaluation")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip V7 (expensive); it is then reported NOT_RUN")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    rep: Dict[str, object] = {
        "preflight_version": "c_align_pilot_preflight_v1",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "head": git(ROOT, "rev-parse", "HEAD"),
        "worktree_clean": not git(ROOT, "status", "--porcelain"),
        "contract_sha256": sha256_file(os.path.join(ROOT, CONTRACT_PATH)),
        "implementation_digest": implementation_digest(),
        "optimizer_step_calls": 0,
        "gates": {},
    }
    g = rep["gates"]
    g["V1_driver_amendment"] = v1_driver_amendment()
    g["V2_toy_equivalence"] = v2_toy_equivalence()
    g["V3_V4_state_and_recipe"] = v3_v4_state_and_recipe(a.device)
    g["V5_determinism"] = determinism_probe(str(load_config("W3_OFF")["device"]))
    g["V6_expected_sequence"] = v6_expected_sequence()
    g["V7_baseline"] = ({"pass": False, "status": "NOT_RUN"} if a.skip_baseline
                        else v7_baseline(a.device, a.out))
    g["V8_step_plan"] = v8_step_plan()
    g["V9_evaluator_freeze"] = v9_evaluator_freeze(a.out)
    g["V10_closed_artifacts"] = closed_artifacts_report(ROOT)
    g["launcher_refusals"] = {arm: preconditions(load_config(arm))
                              for arm in ARMS}

    rep["all_gates_pass"] = all(
        bool(v.get("pass")) for k, v in g.items() if k != "launcher_refusals")
    rep["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = os.path.join(a.out, "preflight_report.json")
    json.dump(rep, open(path, "w"), indent=1, sort_keys=True, default=str)
    for k, v in g.items():
        if k == "launcher_refusals":
            continue
        print(f"{k:28s} {'PASS' if v.get('pass') else 'FAIL'}")
    print(f"ALL_GATES_PASS={rep['all_gates_pass']} -> {path}")
    return 0 if rep["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
