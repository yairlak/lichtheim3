#!/usr/bin/env python3
"""Read-only bitwise comparison of two training checkpoints.

Written for the FINAL-7L replay check: a genuinely from-scratch run reaches
R100 under exactly the historical recipe, and this reports whether that new
checkpoint is bitwise identical to the historical one.

The comparison is VALIDATION ONLY and is deliberately NON-FATAL: a difference
is reported, never repaired, and the historical checkpoint is never
substituted for the new one.  Both files are opened read-only.

Compares: every model tensor, the optimizer state (shared or per bank),
Python/NumPy/Torch/CUDA RNG states, global step, and the task cursors.

Usage:
    python scripts/naming_comprehension/compare_checkpoints.py \
        --a <new>/step_00277800.pt --b <historical>/step_00277800.pt \
        --json-out reports/.../r100_replay.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch


def tensors_equal(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return bool(a == b)


def compare_state(a: dict, b: dict, label: str) -> dict:
    """Elementwise comparison of two {key: tensor-or-scalar} mappings."""
    ka, kb = set(a), set(b)
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    differing = sorted(k for k in (ka & kb) if not tensors_equal(a[k], b[k]))
    return {"label": label, "n_common": len(ka & kb),
            "n_differing": len(differing),
            "differing": differing[:32],
            "only_in_a": only_a[:16], "only_in_b": only_b[:16],
            "identical": not differing and not only_a and not only_b}


def flat_optimizer(ck: dict) -> Dict[str, dict]:
    """{bank name: {(param index, key): value}} for any optimizer policy."""
    out: Dict[str, dict] = {}
    if "optimizer_state_dict" in ck:
        banks = {"shared": ck["optimizer_state_dict"]}
    else:
        banks = dict(ck.get("optimizer_states", {}))
    for name, sd in banks.items():
        flat = {}
        for pid, entry in (sd.get("state") or {}).items():
            for k, v in entry.items():
                flat[f"{pid}.{k}"] = v
        out[name] = flat
    return out


def compare_checkpoints(path_a: str, path_b: str) -> dict:
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)

    sections: List[dict] = [
        compare_state(a["model_state_dict"], b["model_state_dict"], "model"),
    ]

    oa, ob = flat_optimizer(a), flat_optimizer(b)
    if set(oa) != set(ob):
        sections.append({"label": "optimizer_banks", "identical": False,
                         "n_differing": -1, "differing": [],
                         "note": f"bank sets differ: {sorted(oa)} vs {sorted(ob)}"})
    else:
        for name in sorted(oa):
            sections.append(compare_state(oa[name], ob[name],
                                          f"optimizer[{name}]"))

    rng_a = a.get("rng_states") or {}
    rng_b = b.get("rng_states") or {}
    rng_diff = []
    for key in sorted(set(rng_a) | set(rng_b)):
        va, vb = rng_a.get(key), rng_b.get(key)
        if va is None and vb is None:
            continue
        same = (tensors_equal(va, vb) if not isinstance(va, (tuple, list))
                else repr(va) == repr(vb))
        if not same:
            rng_diff.append(key)
    sections.append({"label": "rng_states", "n_differing": len(rng_diff),
                     "differing": rng_diff, "identical": not rng_diff})

    scalars = {}
    for key in ("global_step", "cursors", "rep_epoch", "lr",
                "optimizer_policy", "subset_mode", "schedule"):
        va, vb = a.get(key), b.get(key)
        scalars[key] = {"a": va, "b": vb, "equal": va == vb}
    sections.append({
        "label": "run_state",
        "n_differing": sum(0 if v["equal"] else 1 for v in scalars.values()),
        "differing": [k for k, v in scalars.items() if not v["equal"]],
        "identical": all(v["equal"] for v in scalars.values()),
    })

    return {
        "comparison_only": True,
        "non_fatal": True,
        "note": ("Validation only. A difference is reported, never repaired; "
                 "the b-side checkpoint is never substituted for the a-side "
                 "trajectory."),
        "a": {"path": os.path.abspath(path_a),
              "global_step": a.get("global_step"),
              "git_commit": (a.get("git") or {}).get("commit")},
        "b": {"path": os.path.abspath(path_b),
              "global_step": b.get("global_step"),
              "git_commit": (b.get("git") or {}).get("commit")},
        "scalars": scalars,
        "sections": sections,
        "identical": all(s["identical"] for s in sections),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", required=True, help="the run's OWN checkpoint")
    p.add_argument("--b", required=True, help="reference, comparison only")
    p.add_argument("--json-out", default=None)
    args = p.parse_args(argv)

    rep = compare_checkpoints(args.a, args.b)
    print(f"[compare] a = {rep['a']['path']}  (step {rep['a']['global_step']})")
    print(f"[compare] b = {rep['b']['path']}  (step {rep['b']['global_step']})")
    for s in rep["sections"]:
        status = "IDENTICAL" if s["identical"] else f"{s['n_differing']} DIFFER"
        print(f"  {s['label']:22s} {status}")
        if not s["identical"] and s.get("differing"):
            print(f"      e.g. {s['differing'][:6]}")
        if s.get("note"):
            print(f"      {s['note']}")
    print(f"[compare] BITWISE IDENTICAL: {rep['identical']}")
    if not rep["identical"]:
        print("[compare] NOTE: reported only. The reference is never "
              "substituted; the from-scratch trajectory is preserved.")
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        json.dump(rep, open(args.json_out, "w"), indent=2, default=str)
        print(f"[compare] wrote {args.json_out}")
    return 0                      # never fatal: this is a validation report


if __name__ == "__main__":
    raise SystemExit(main())
