"""Sprint 2 — morphology figures and the full Sprint-2 report generator.

    python -m scripts.behavioral_analysis.plot_morphology \
        --out_root reports/behavioral_wfe_fulllexicon_93a577f/morphology

Regenerates every Sprint-2 table and figure from the canonical table alone.
No checkpoint, no torch, no model inference.

Visual encoding (frozen, inherited from Sprint 1):
    Real words = red, Pseudowords = blue, and those colours encode nothing else.
    Morphologically complex = solid, simple = dashed; morphology is carried by
    line style alone.  Individual seed traces are light and thin, the
    across-seed mean is prominent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from scripts.behavioral_analysis import morphology as mo          # noqa: E402
from scripts.behavioral_analysis.common import (CANONICAL_TABLE,   # noqa: E402
                                                LENGTHS, LEXICALITY_COLOR,
                                                LEXICALITY_LABEL, REPORT_ROOT,
                                                ROUTE_LABEL, ROUTES, SEEDS,
                                                SEED_MARKER, repo_relative)
from scripts.behavioral_analysis.io import (load_canonical,        # noqa: E402
                                            sha256_file, write_table)
from scripts.behavioral_analysis.plotting import save_figure       # noqa: E402

MORPH_STYLE = {"complex": "-", "simple": "--"}      # frozen encoding
CLEAN_CAPTION_DEF = (
    "Real words were restricted to WFE words encountered during training with "
    "the same phonological form. Pseudowords were restricted to WFE "
    "pseudowords whose phonological form was absent from the training lexicon.")


def _morph_legend(ax, loc="upper left"):
    return ax.legend(handles=[
        Line2D([], [], color="0.25", ls=MORPH_STYLE["complex"], lw=2.4,
               label="Complex (solid)"),
        Line2D([], [], color="0.25", ls=MORPH_STYLE["simple"], lw=2.4,
               label="Simple (dashed)")],
        loc=loc, frameon=False, fontsize=8.5)


def _lex_legend(ax, loc="upper left"):
    return ax.legend(handles=[
        Line2D([], [], color=LEXICALITY_COLOR[k], lw=2.6,
               label=LEXICALITY_LABEL[k]) for k in ("real", "pseudo")],
        loc=loc, frameon=False, fontsize=9)


# ------------------------------------------------------- faithful figure

def plot_faithful(tab: pd.DataFrame, out_dir: str) -> dict:
    """Figure-2A-style: one panel, FULL route, all 1,200 original items."""
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for lex in ("real", "pseudo"):
        for mor in ("complex", "simple"):
            sub = tab[(tab["source_lexicality"] == lex)
                      & (tab["morphology"] == mor)]
            if sub.empty:
                continue
            for s in SEEDS:                      # secondary: light and thin
                ss = sub[sub["seed"] == s].sort_values("phoneme_length")
                ax.plot(ss["phoneme_length"], ss["mean_raw_edit_distance"],
                        ls=MORPH_STYLE[mor], color=LEXICALITY_COLOR[lex],
                        lw=0.7, alpha=0.35, marker=SEED_MARKER[s], ms=3)
            agg = (sub.drop_duplicates("phoneme_length")
                   .sort_values("phoneme_length"))
            ax.plot(agg["phoneme_length"], agg["mean_across_seeds"],
                    ls=MORPH_STYLE[mor], color=LEXICALITY_COLOR[lex],
                    lw=2.8, marker="o", ms=7)
    ax.set_xticks(LENGTHS)
    ax.set_xlabel("Phoneme length")
    ax.set_ylabel("Mean raw Levenshtein edit distance")
    ax.grid(alpha=0.25)
    leg = _lex_legend(ax, "upper left")
    ax.add_artist(leg)
    _morph_legend(ax, "upper center")
    ax.set_title("Faithful WFE replication — length × lexicality × morphology\n"
                 "(FULL route, all 1,200 original items)", fontsize=11)
    caption = (
        "**Faithful WFE morphology replication.** Script-faithful "
        "stimulus-level replication of the original Figure-2A design: all "
        "1,200 WFE items, FULL route, mean raw Levenshtein edit distance "
        "against phoneme length. The source real/pseudo labels are preserved "
        "exactly as published; **this is not a trained-versus-novel analysis "
        "for Lichtheim3** — 122 of the 800 source real words were never in the "
        "Lichtheim3 training lexicon and 9 source pseudowords collide with it. "
        "Complex = solid, simple = dashed. Thin traces with small markers are "
        "individual seeds; thick lines with large markers are the mean across "
        "the four selected Lichtheim3 seeds (19, 20, 21, 22), which is also "
        "the source of the displayed uncertainty. Length 6 is absent from the "
        "WFE by construction.")
    return save_figure(fig, out_dir, "faithful_length_lexicality_morphology",
                       caption)


# ---------------------------------------------------------- clean figure

def plot_clean(tab: pd.DataFrame, out_dir: str) -> dict:
    """Two rows (lexicality) x three columns (route); no 8-curve overlap."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    ymax = float(tab["mean_raw_edit_distance"].max()) * 1.1
    for r_i, lex in enumerate(("real", "pseudo")):
        for c_i, route in enumerate(ROUTES):
            ax = axes[r_i, c_i]
            for mor in ("complex", "simple"):
                sub = tab[(tab["source_lexicality"] == lex)
                          & (tab["route"] == route)
                          & (tab["morphology"] == mor)]
                if sub.empty:
                    continue
                for s in SEEDS:
                    ss = sub[sub["seed"] == s].sort_values("phoneme_length")
                    ax.plot(ss["phoneme_length"], ss["mean_raw_edit_distance"],
                            ls=MORPH_STYLE[mor], color=LEXICALITY_COLOR[lex],
                            lw=0.7, alpha=0.35, marker=SEED_MARKER[s], ms=3)
                agg = (sub.drop_duplicates("phoneme_length")
                       .sort_values("phoneme_length"))
                ax.plot(agg["phoneme_length"], agg["mean_across_seeds"],
                        ls=MORPH_STYLE[mor], color=LEXICALITY_COLOR[lex],
                        lw=2.6, marker="o", ms=6)
            ax.set_ylim(0, ymax)
            ax.grid(alpha=0.25)
            ax.set_xticks(LENGTHS)
            if r_i == 0:
                ax.set_title(ROUTE_LABEL[route], fontsize=12)
            if c_i == 0:
                ax.set_ylabel(f"{LEXICALITY_LABEL[lex]}\n"
                              "mean raw edit distance", fontsize=10)
            if r_i == 1:
                ax.set_xlabel("Phoneme length")
    _morph_legend(axes[0, 0], "upper left")
    axes[0, 2].legend(handles=[
        Line2D([], [], color="0.35", lw=0.7, alpha=0.5, marker="o", ms=3,
               label="individual seed"),
        Line2D([], [], color="0.35", lw=2.6, marker="o", ms=6,
               label="mean over seeds")],
        loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Clean WFE set — length × morphology by route", y=0.98,
                 fontsize=13)
    fig.tight_layout()
    caption = (
        "**Clean-set morphology by route.** Mean raw Levenshtein edit distance "
        "against phoneme length, split by route (columns) and lexicality "
        "(rows) so that no panel carries eight overlapping curves. Complex = "
        "solid, simple = dashed; red is used only for Real words and blue only "
        "for Pseudowords. Thin traces are individual seeds (19, 20, 21, 22, "
        "all shown); thick lines are the mean across seeds. Panels share a "
        "y-axis.\n\n" + CLEAN_CAPTION_DEF)
    return save_figure(fig, out_dir, "clean_length_morphology_by_route",
                       caption)


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default=CANONICAL_TABLE)
    ap.add_argument("--out_root",
                    default=os.path.join(REPORT_ROOT, "morphology"))
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    root = args.out_root
    fa_fig = os.path.join(root, "faithful_replication", "figures")
    fa_tab = os.path.join(root, "faithful_replication", "tables")
    cl_fig = os.path.join(root, "clean_adapted", "figures")
    cl_tab = os.path.join(root, "clean_adapted", "tables")
    shared = os.path.join(root, "tables")
    for p in (fa_fig, fa_tab, cl_fig, cl_tab, shared):
        os.makedirs(p, exist_ok=True)

    canon = load_canonical(args.canonical)
    written = {}

    # ---- cell counts (audit) -----------------------------------------
    fa_counts = mo.cell_counts(canon, "FAITHFUL_WFE_ALL")
    cl_counts = mo.cell_counts(canon, "LICHTHEIM_CLEAN")
    write_table(fa_counts, os.path.join(
        shared, "faithful_morphology_cell_counts.tsv"),
        sort_by=["source_lexicality", "morphology", "phoneme_length"])
    write_table(cl_counts, os.path.join(
        shared, "clean_morphology_cell_counts.tsv"),
        sort_by=["route", "source_lexicality", "morphology", "phoneme_length"])
    bal = _balance_summary(fa_counts, cl_counts)
    write_table(bal, os.path.join(shared, "morphology_cell_balance_summary.tsv"),
                sort_by=["dataset_regime", "source_lexicality", "morphology"])

    # ---- faithful -----------------------------------------------------
    print("[faithful] tables")
    fa_plot = mo.plot_table(canon, "FAITHFUL_WFE_ALL")
    write_table(fa_plot, os.path.join(
        fa_tab, "faithful_length_lexicality_morphology_plot.tsv"),
        sort_by=["source_lexicality", "morphology", "phoneme_length", "seed"])
    fa_con = mo.seed_contrasts(canon, "FAITHFUL_WFE_ALL")
    write_table(fa_con, os.path.join(fa_tab,
                                     "faithful_morphology_seed_contrasts.tsv"),
                sort_by=["source_lexicality", "seed"])
    fa_int = mo.seed_length_interactions(canon, "FAITHFUL_WFE_ALL")
    write_table(fa_int, os.path.join(
        fa_tab, "faithful_morphology_length_interactions.tsv"),
        sort_by=["source_lexicality", "seed"])
    fa_boot = mo.bootstrap_morphology(canon, "FAITHFUL_WFE_ALL")
    write_table(fa_boot, os.path.join(fa_tab,
                                      "faithful_morphology_bootstrap.tsv"),
                sort_by=["source_lexicality", "quantity"])
    print("[faithful] figure")
    written["faithful_length_lexicality_morphology"] = plot_faithful(
        fa_plot, fa_fig)

    # ---- clean adapted -------------------------------------------------
    print("[clean] tables")
    cl_plot = mo.plot_table(canon, "LICHTHEIM_CLEAN")
    write_table(cl_plot, os.path.join(
        cl_tab, "clean_length_morphology_by_route_plot.tsv"),
        sort_by=["route", "source_lexicality", "morphology",
                 "phoneme_length", "seed"])
    cl_con = mo.seed_contrasts(canon, "LICHTHEIM_CLEAN")
    write_table(cl_con, os.path.join(cl_tab,
                                     "clean_morphology_seed_contrasts.tsv"),
                sort_by=["route", "source_lexicality", "seed"])
    cl_int = mo.seed_length_interactions(canon, "LICHTHEIM_CLEAN")
    write_table(cl_int, os.path.join(
        cl_tab, "clean_morphology_length_interactions.tsv"),
        sort_by=["route", "source_lexicality", "seed"])
    cl_route = mo.route_contrasts(cl_con, cl_int)
    write_table(cl_route, os.path.join(
        cl_tab, "clean_morphology_route_contrasts.tsv"),
        sort_by=["route_contrast", "source_lexicality", "seed"])
    cl_boot = mo.bootstrap_morphology(canon, "LICHTHEIM_CLEAN")
    write_table(cl_boot, os.path.join(cl_tab,
                                      "clean_morphology_bootstrap.tsv"),
                sort_by=["route", "source_lexicality", "quantity"])
    sens = pd.concat([
        mo.summarise_across_seeds(cl_con,
                                  "morphology_contrast_raw_edit_distance",
                                  ["dataset_regime", "route",
                                   "source_lexicality"]),
        mo.summarise_across_seeds(cl_int, "morphology_length_interaction",
                                  ["dataset_regime", "route",
                                   "source_lexicality"])], ignore_index=True)
    write_table(sens, os.path.join(
        cl_tab, "clean_morphology_exact_zero_sensitivity.tsv"),
        sort_by=["quantity", "route", "source_lexicality"])
    print("[clean] figure")
    written["clean_length_morphology_by_route"] = plot_clean(cl_plot, cl_fig)

    # ---- word error (tables only; see Phase 6 policy) -------------------
    print("[word error] tables")
    fa_we = mo.descriptive_cells(canon, "FAITHFUL_WFE_ALL")
    write_table(fa_we, os.path.join(shared,
                                    "faithful_morphology_word_error.tsv"),
                sort_by=["source_lexicality", "morphology", "phoneme_length",
                         "seed"])
    cl_we = mo.descriptive_cells(canon, "LICHTHEIM_CLEAN")
    write_table(cl_we, os.path.join(shared, "clean_morphology_word_error.tsv"),
                sort_by=["route", "source_lexicality", "morphology",
                         "phoneme_length", "seed"])
    we_boot = pd.concat([
        mo.bootstrap_morphology(canon, "FAITHFUL_WFE_ALL", metric="word_error"),
        mo.bootstrap_morphology(canon, "LICHTHEIM_CLEAN", metric="word_error")],
        ignore_index=True)
    write_table(we_boot, os.path.join(shared,
                                      "morphology_word_error_bootstrap.tsv"),
                sort_by=["dataset_regime", "route", "source_lexicality",
                         "quantity"])

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({"canonical_table": repo_relative(args.canonical),
                       "canonical_table_sha256": sha256_file(args.canonical),
                       "out_root": repo_relative(root),
                       "figures": {k: {kk: repo_relative(vv)
                                       for kk, vv in v.items()}
                                   for k, v in written.items()},
                       "model_inference_performed": False}, f, indent=2)
    print(f"\nSprint-2 morphology outputs written to {repo_relative(root)}")
    return 0


def _balance_summary(fa: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, df in (("FAITHFUL_WFE_ALL", fa), ("LICHTHEIM_CLEAN", cl)):
        d = df[df["route"] == "full"] if "route" in df.columns else df
        for lex in ("real", "pseudo"):
            for mor in ("complex", "simple"):
                sub = d[(d["source_lexicality"] == lex)
                        & (d["morphology"] == mor)]
                rows.append({
                    "dataset_regime": regime, "source_lexicality": lex,
                    "morphology": mor,
                    "n_items_total": int(sub["n_items"].sum()),
                    "n_length_cells": int(len(sub)),
                    "min_cell_n": int(sub["n_items"].min()),
                    "max_cell_n": int(sub["n_items"].max()),
                    "n_small_cells": int((sub["cell_flag"] == "SMALL_CELL").sum()),
                    "n_very_small_cells":
                        int((sub["cell_flag"] == "VERY_SMALL_CELL").sum()),
                    "balanced_across_lengths":
                        bool(sub["n_items"].nunique() == 1)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sys.exit(main())
