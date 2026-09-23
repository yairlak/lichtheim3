#!/usr/bin/env python
"""Drive the complete READ-ONLY post-analysis package.

    python run_post_analysis.py --results-parent <results> --out-dir <external>

Writes ONLY under --out-dir, which must be outside the results namespace.
Loads no model, runs no forward pass, applies no lesion.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from paper_programme.lesioning_v2.post_analysis import (  # noqa: E402
    aggregate_multishard as agg, curves, figures, io_utils, report, validate)
from paper_programme.lesioning_v2.post_analysis.io_utils import AnalysisError  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lesioning V2 post-analysis (read-only).")
    ap.add_argument("--results-parent", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)

    try:
        out = io_utils.guard_out_dir(a.out_dir, a.results_parent)
        res = agg.build(a.results_parent)
    except AnalysisError as e:
        print(f"ANALYSIS_FAIL\n{e}", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    rows = res["rows"]
    art = {}

    p = os.path.join(out, "CELL_ENDPOINT_EXACT_MATCH.tsv")
    art["CELL_ENDPOINT_EXACT_MATCH.tsv"] = io_utils.write_tsv(
        p, rows, agg.CANONICAL_COLUMNS)

    curve = curves.curve_summary(rows)
    art["CURVE_SUMMARY.tsv"] = io_utils.write_tsv(
        os.path.join(out, "CURVE_SUMMARY.tsv"), curve, curves.CURVE_COLUMNS)

    base = curves.intact_baselines(rows)
    art["INTACT_BASELINES.tsv"] = io_utils.write_tsv(
        os.path.join(out, "INTACT_BASELINES.tsv"), base, curves.BASELINE_COLUMNS)

    k15 = curves.max_severity_table(rows)
    art["MAX_SEVERITY_K15.tsv"] = io_utils.write_tsv(
        os.path.join(out, "MAX_SEVERITY_K15.tsv"), k15, curves.K15_COLUMNS)

    consistency = curves.k0_cross_site_consistency(base)

    if not a.no_figures:
        art.update(figures.primary_figures(curve, out))
        art.update(figures.diagnostic_figures(curve, out))

    val = validate.build(a.results_parent, res, rows, consistency, art)
    art["ANALYSIS_VALIDATION.json"] = validate.write(
        os.path.join(out, "ANALYSIS_VALIDATION.json"), val)

    art["LESIONING_V2_RESULTS_DESCRIPTIVE.md"] = report.write(
        os.path.join(out, "LESIONING_V2_RESULTS_DESCRIPTIVE.md"),
        report.descriptive_report(curve, base, consistency, k15, val))
    art["CENTRAL_FINAL_HANDOFF_LESIONING_V2_RESULTS.md"] = report.write(
        os.path.join(out, "CENTRAL_FINAL_HANDOFF_LESIONING_V2_RESULTS.md"),
        report.central_handoff(curve, k15, val))

    sums = os.path.join(out, "ANALYSIS_OUTPUT_SHA256SUMS")
    with open(sums, "w") as fh:
        for k in sorted(art):
            fh.write(f"{art[k]}  {k}\n")

    print(f"cells={res['census']['n_cells']} rows={len(rows)} "
          f"(expected {agg.EXPECTED_ROWS})")
    print(f"artifacts written to {out}:")
    for k in sorted(art):
        print(f"  {art[k]}  {k}")
    print("REAL_RESULTS_NAMESPACE_TOUCHED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
