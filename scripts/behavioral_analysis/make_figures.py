"""Regenerate the five validated behavioral figures.  No model inference.

    python -m scripts.behavioral_analysis.make_figures \
        --out_dir reports/behavioral_wfe_fulllexicon_93a577f/figures

Reads the validated canonical table, recomputes every plotting table with the
frozen estimators, writes each table beside its figure, and renders PNG/PDF/SVG
plus a standalone caption.  Deterministic: rerunning produces byte-identical
TSVs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from scripts.behavioral_analysis import compute, plotting            # noqa: E402
from scripts.behavioral_analysis.common import (CANONICAL_TABLE,     # noqa: E402
                                                REPORT_ROOT, repo_relative)
from scripts.behavioral_analysis.io import (clean_subset,            # noqa: E402
                                            load_canonical, sha256_file,
                                            write_table)

FIGURES = ["yair_clean_length_by_route", "yair_clean_length_slopes",
           "yair_clean_serial_position", "gate_by_clean_lexicality",
           "gate_by_exposure_status"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default=CANONICAL_TABLE,
                    help="validated canonical seed x item x route table")
    ap.add_argument("--out_dir", default=os.path.join(REPORT_ROOT, "figures"),
                    help="destination for figures, plotting tables, captions")
    ap.add_argument("--manifest", default=None,
                    help="optional JSON manifest of what was written")
    args = ap.parse_args(argv)

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    canon = load_canonical(args.canonical)
    clean = clean_subset(canon)
    written = {}

    print("[1/5] clean length curves")
    t = compute.clean_length_table(clean)
    p = write_table(t, os.path.join(out, "yair_clean_length_by_route.tsv"),
                    sort_by=["route", "source_lexicality", "phoneme_length",
                             "seed"])
    written["yair_clean_length_by_route"] = {
        "table": p, **plotting.plot_clean_length(t, out)}

    print("[2/5] clean length slopes and LTM-WM contrast")
    slopes, contrasts, boot = compute.clean_slope_tables(clean)
    p1 = write_table(slopes,
                     os.path.join(out, "clean_length_slopes_by_seed.tsv"),
                     sort_by=["source_lexicality", "route", "seed"])
    p2 = write_table(contrasts,
                     os.path.join(out, "clean_route_length_contrasts.tsv"),
                     sort_by=["source_lexicality", "seed"])
    p3 = write_table(boot,
                     os.path.join(out, "clean_bootstrap_results.tsv"),
                     sort_by=["stratum", "quantity"])
    written["yair_clean_length_slopes"] = {
        "table": p1, "contrasts": p2, "bootstrap": p3,
        **plotting.plot_clean_slopes(slopes, contrasts, boot, out)}

    print("[3/5] clean serial position (faithful zip-mismatch)")
    raw, curves = compute.serial_position_tables(clean)
    p1 = write_table(raw, os.path.join(out, "yair_clean_serial_position.tsv"),
                     sort_by=["route", "source_lexicality", "phoneme_length",
                              "position_index_1based"])
    p2 = write_table(curves, os.path.join(
        out, "yair_clean_serial_position_interpolated.tsv"),
        sort_by=["route", "source_lexicality", "relative_position"])
    written["yair_clean_serial_position"] = {
        "table": p2, "positions": p1,
        **plotting.plot_clean_serial_position(curves, out)}

    print("[4/5] gate by clean lexicality")
    gate_lex, gate_exp = compute.gate_tables(canon)
    p = write_table(gate_lex, os.path.join(out, "gate_by_clean_lexicality.tsv"),
                    sort_by=["source_lexicality", "seed"])
    written["gate_by_clean_lexicality"] = {
        "table": p, **plotting.plot_gate_clean(gate_lex, out)}

    print("[5/5] gate by exposure status")
    p = write_table(gate_exp, os.path.join(out, "gate_by_exposure_status.tsv"),
                    sort_by=["exposure_status", "seed"])
    written["gate_by_exposure_status"] = {
        "table": p, **plotting.plot_gate_exposure(gate_exp, out)}

    manifest = {
        "canonical_table": repo_relative(args.canonical),
        "canonical_table_sha256": sha256_file(args.canonical),
        "out_dir": repo_relative(out),
        "figures": {k: {kk: repo_relative(vv) for kk, vv in v.items()}
                    for k, v in written.items()},
        "model_inference_performed": False,
    }
    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump(manifest, f, indent=2)
    print(f"\n{len(written)} figures written to {repo_relative(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
