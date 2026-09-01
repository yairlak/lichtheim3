#!/usr/bin/env python3
"""Matched Mac / Jean Zay throughput benchmark for the Phase-4 joint driver.

NOT a scientific run: no checkpoint is written, no run directory is created,
and the model state is discarded.  It builds the exact scientific trainer
(`JointScratchTrainer`), runs `--warmup` untimed optimizer steps, then times
`--timed` steps of pure training (no evaluation inside the timed window), and
reports steps/s, examples/s and projected wall-clock per 100k updates.

The SAME command must be used on every platform being compared, changing only
--device.  Use --allow-glove-fallback for throughput work (semantic values do
not affect compute cost); scientific runs always use real GloVe.

Examples:
    # Mac CPU reference
    python scripts/cluster/benchmark_throughput.py --device cpu

    # Jean Zay GPU (inside a SLURM job)
    python scripts/cluster/benchmark_throughput.py --device cuda

Decision rule (see scripts/cluster/jeanzay/README.md): compare
`projected_hours_per_100k_steps` across platforms; prefer Jean Zay for long
runs only if its projection is at most half the Mac's, otherwise the Mac
remains the reference platform.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (  # noqa: E402
    FINAL_FULL_MODE, JointScratchTrainer,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default="cpu")
    p.add_argument("--regime", default="j0",
                   choices=("h0", "c_only", "n_only", "j0"))
    p.add_argument("--seed", type=int, default=999,
                   help="throwaway benchmark seed (not a scientific seed)")
    p.add_argument("--subset-mode", default=FINAL_FULL_MODE,
                   choices=("nested", "representative", FINAL_FULL_MODE))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--timed", type=int, default=200)
    p.add_argument("--allow-glove-fallback", action="store_true", default=True,
                   help="default ON: throughput does not depend on GloVe values")
    p.add_argument("--real-glove", action="store_true",
                   help="parse the real GloVe file instead (slower startup)")
    p.add_argument("--torch-deterministic", action="store_true",
                   help="benchmark WITH strict determinism enabled, to "
                        "measure its cost explicitly")
    p.add_argument("--json-out", default=None,
                   help="optional path for the JSON report")
    return p.parse_args()


def sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def main() -> int:
    args = parse_args()
    t_start = time.time()

    if args.torch_deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    glove = ("data/glove.6B.300d.txt" if args.real_glove
             else "scripts/cluster/_no_such_glove.txt")
    t0 = time.time()
    trainer = JointScratchTrainer(
        regime=args.regime, seed=args.seed, device=args.device,
        max_words=30000, lexicon_path="data/lexicon_en_glove_covered.tsv",
        dorsal_pool_size=4000, batch_size=args.batch_size,
        subset_mode=args.subset_mode, subset_per_band=822, subset_size=64,
        lr_boundary_steps=46_300,
        allow_glove_fallback=not args.real_glove,
        require_subset_hash=False, glove_path=glove)
    t_init = time.time() - t0

    # examples consumed per optimizer step = batch size of every ACTIVE stream
    active = [k for k, v in trainer.stream_active.items() if v]
    examples_per_step = args.batch_size * len(active)

    for _ in range(args.warmup):
        trainer.train_step()
    sync(args.device)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t1 = time.time()
    for _ in range(args.timed):
        trainer.train_step()
    sync(args.device)
    dt = time.time() - t1

    steps_per_s = args.timed / dt
    report = {
        "benchmark": "joint_scratch_throughput.v1",
        "scientific_result": False,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_device": (torch.cuda.get_device_name(0)
                        if args.device.startswith("cuda")
                        and torch.cuda.is_available() else None),
        "regime": args.regime,
        "subset_mode": args.subset_mode,
        "batch_size": args.batch_size,
        "active_streams": active,
        "examples_per_step": examples_per_step,
        "torch_deterministic": bool(args.torch_deterministic),
        "warmup_steps": args.warmup,
        "timed_steps": args.timed,
        "timed_seconds": round(dt, 3),
        "seconds_per_step": round(dt / args.timed, 5),
        "steps_per_second": round(steps_per_s, 3),
        "examples_per_second": round(steps_per_s * examples_per_step, 1),
        "projected_hours_per_100k_steps": round(100_000 / steps_per_s / 3600, 3),
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if args.device.startswith("cuda") and torch.cuda.is_available()
            else None),
        "trainer_init_seconds": round(t_init, 2),
        "end_to_end_seconds": round(time.time() - t_start, 2),
        "populations": {
            "repetition": trainer.streams["repetition"].n,
            "pool": trainer.streams["pool"].n,
            "comprehension": trainer.streams["comprehension"].n,
            "naming": trainer.streams["naming"].n,
        },
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
