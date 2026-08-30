#!/usr/bin/env python3
"""Reproduce a tracked Lichtheim3 result from the repository alone.

This is deliberately narrow. It exposes only results that have been verified to
regenerate from tracked artifacts with no checkpoint, no GloVe, no NWR/SWP data,
no CUDA and no training. Today that is one result:

    stable-zero   the canonical model-selection figure (mf2), showing the
                  full-lexicon train-error trajectory for the four cohort seeds
                  and the X = 2 / 3 / 5 stable-zero criterion outcomes.

Usage (from the repository root):

    python scripts/reproduce.py stable-zero --out-dir reproduced/stable_zero

--out-dir is required: nothing is written unless you say where. Pointing it at
the tracked canonical directory
(reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/meeting_figures)
will overwrite those files, so choose a fresh directory unless that is what you
intend.

Nothing scientific is recomputed. The figure is drawn from already-validated
audit tables and its annotations are asserted against those tables as it is
drawn; this command reproduces the artifact, it does not re-derive the result.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Tracked inputs each target needs, relative to the repository root.
TARGETS = {
    "stable-zero": {
        "description": "canonical model-selection figure (mf2)",
        "inputs": [
            "reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
            "stable_zero_audit/stable_zero_trajectory.tsv",
            "reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
            "stable_zero_audit/stable_zero_streaks.tsv",
            "reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
            "stable_zero_audit/stable_zero_verdicts.tsv",
        ],
    },
}


def missing_inputs(target: str) -> list:
    """Tracked inputs for `target` that are not present, repo-relative."""
    return [rel for rel in TARGETS[target]["inputs"]
            if not os.path.exists(os.path.join(ROOT, rel))]


def run_stable_zero(out_dir: str) -> dict:
    from scripts.make_meeting_figures import figure_stable_zero
    return figure_stable_zero(out_dir=out_dir)


RUNNERS = {"stable-zero": run_stable_zero}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reproduce.py",
        description="Reproduce a tracked Lichtheim3 result from the repository "
                    "alone (no checkpoint, no GloVe, no training).")
    sub = p.add_subparsers(dest="target", required=True)
    for name, meta in TARGETS.items():
        sp = sub.add_parser(name, help=meta["description"])
        sp.add_argument("--out-dir", required=True,
                        help="directory to write the artifacts into; created "
                             "if it does not exist")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    target = args.target

    absent = missing_inputs(target)
    if absent:
        print(f"ERROR: cannot reproduce '{target}': "
              f"{len(absent)} required tracked input(s) missing:",
              file=sys.stderr)
        for rel in absent:
            print(f"  {rel}", file=sys.stderr)
        print("Run this from the repository root of a complete checkout.",
              file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = RUNNERS[target](out_dir)

    shown = os.path.relpath(out_dir, os.getcwd())
    if shown.startswith(".."):
        shown = out_dir
    print(f"reproduced '{target}' -> {shown}")
    for kind in sorted(written):
        print(f"  {kind:8s} {os.path.basename(written[kind])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
