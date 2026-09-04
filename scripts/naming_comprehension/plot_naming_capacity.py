"""CAP-1 figures: ventral-width naming curves vs exposures, with historical
warm-start baselines overlaid where raw snapshot data exists (phase2h full
29,571 and phase2g representative 10k; both hidden 128, warm start, LR 1e-4
-- labelled as such, never presented as the same recipe)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) if v not in ("", None) else None
                 for k, v in row.items()}
                for row in csv.DictReader(f, delimiter="\t")]


def historical_curve(summary_path: str) -> Tuple[List[float], List[float]]:
    s = json.load(open(summary_path))
    pts = [(sn["epoch"], sn["naming"]["exact_match"]) for sn in s["snapshots"]]
    return [p[0] for p in pts], [p[1] for p in pts]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep-dir", default="outputs/naming_capacity")
    ap.add_argument("--widths", default="128,256,512")
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--historical-full", default=None,
                    help="run_summary.json of phase2h (full 29,571 warm start)")
    ap.add_argument("--historical-10k", default=None,
                    help="run_summary.json of phase2g (representative 10k)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    widths = [int(w) for w in args.widths.split(",")]
    runs: Dict[int, List[dict]] = {}
    for w in widths:
        p = os.path.join(args.sweep_dir, f"cap1_w{w}_seed{args.seed}",
                         "metrics.tsv")
        runs[w] = read_metrics(p)

    colors = {128: "tab:blue", 256: "tab:orange", 512: "tab:red"}
    panels = [("exact_match", "whole-word exact match (free greedy AR)",
               "exact_vs_exposures.png"),
              ("full_ce", "naming CE (full population, teacher-forced)",
               "ce_vs_exposures.png"),
              ("tf_token_acc", "teacher-forced token accuracy",
               "tf_token_acc_vs_exposures.png"),
              ("greedy_token_acc", "greedy positional token accuracy",
               "greedy_token_acc_vs_exposures.png")]
    for key, title, fname in panels:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for w in widths:
            xs = [r["exposures"] for r in runs[w]]
            ys = [r[key] for r in runs[w]]
            ax.plot(xs, ys, "o-", color=colors.get(w), label=f"H={w} (scratch)")
        if key == "exact_match":
            if args.historical_full and os.path.exists(args.historical_full):
                xs, ys = historical_curve(args.historical_full)
                ax.plot(xs, ys, "s--", color="grey", alpha=0.8,
                        label="hist 29,571 H=128 (warm, lr 1e-4)")
            if args.historical_10k and os.path.exists(args.historical_10k):
                xs, ys = historical_curve(args.historical_10k)
                ax.plot(xs, ys, "^--", color="black", alpha=0.6,
                        label="hist 10k H=128 (warm, lr 1e-4)")
        ax.set_xlabel("exposures per item")
        ax.set_ylabel(key)
        ax.set_title(f"CAP-1 full-lexicon naming: {title}")
        if key == "full_ce":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = os.path.join(args.out_dir, fname)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] {out}")

    # matched-exposure table
    cadence = [100, 300, 500, 750, 1000, 1500]
    lines = ["H | params(naming path) | " +
             " | ".join(f"exact@{e}" for e in cadence) + " | best"]
    for w in widths:
        cfg = json.load(open(os.path.join(
            args.sweep_dir, f"cap1_w{w}_seed{args.seed}", "config.json")))
        by_exp = {int(round(r["exposures"])): r for r in runs[w]}
        cells = [f"{by_exp[e]['exact_match']:.4f}" if e in by_exp else "-"
                 for e in cadence]
        best = max(r["exact_match"] for r in runs[w])
        lines.append(f"{w} | {cfg['params']['naming_path']:,} | " +
                     " | ".join(cells) + f" | {best:.4f}")
    table = "\n".join(lines)
    out = os.path.join(args.out_dir, "matched_exposure_table.txt")
    open(out, "w").write(table + "\n")
    print(f"[table] {out}\n{table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
