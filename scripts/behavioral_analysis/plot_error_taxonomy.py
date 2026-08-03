"""Sprint 4 — error-taxonomy and premature-EOS figures, tables and examples.

    python -m scripts.behavioral_analysis.plot_error_taxonomy \
        --out_root reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy

Regenerates every Sprint-4 output from the canonical table alone: no torch, no
checkpoint, no model inference, no re-alignment of any sequence.

Presentation (frozen in error_taxonomy_analysis_spec.md):
  * red = Real words, blue = Pseudowords, reserved — **operation type is never
    encoded by red or blue**, it is encoded by hatch and by x-position;
  * all four seed values are drawn on every panel, the mean is prominent;
  * the clean taxonomy figure uses ONE common absolute y-scale across routes, so
    the route magnitude difference cannot be hidden by per-panel rescaling; the
    conditional `_full_wm_zoom` companion never replaces it.

The Levenshtein taxonomy and the premature-EOS diagnostic are produced by
separate modules into separate directories and are never merged into a single
"error type" axis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from scripts.behavioral_analysis import error_taxonomy as et            # noqa: E402
from scripts.behavioral_analysis import eos_diagnostics as eos          # noqa: E402
from scripts.behavioral_analysis.common import (CANONICAL_TABLE,        # noqa: E402
                                                CLEAN_CAPTION, LENGTHS,
                                                LEXICALITY_COLOR,
                                                LEXICALITY_LABEL, REPORT_ROOT,
                                                ROUTE_LABEL, ROUTES,
                                                SEED_MARKER, SEEDS,
                                                repo_relative)
from scripts.behavioral_analysis.io import (load_canonical,             # noqa: E402
                                            sha256_file, write_table)
from scripts.behavioral_analysis.plotting import save_figure            # noqa: E402

# Operation type is carried by hatch, never by colour.
OPERATION_HATCH = {"substitutions": "", "deletions": "///", "insertions": "..."}
LONG_HATCH_EDGE = "0.15"
ZOOM_RATIO_TRIGGER = 10.0

MEASUREMENT_LIMITS = (
    "Two measurement limits apply to every panel. `Levenshtein.editops` "
    "tie-breaking can move counts between substitution, deletion and insertion "
    "**without changing the total edit distance**, so the split is backend-"
    "dependent while the total is not. The forced-length readout bounds each "
    "prediction by the gold target length, so **terminal insertions beyond the "
    "target horizon are unobservable** and insertion counts are a lower bound.")

EOS_LIMITS = (
    "Premature EOS is a decoder diagnostic, not a fourth edit operation. Under "
    "the audited readout convention the window holds exactly L tokens at "
    "indices 0…L−1, so an EOS at the correct boundary (index L) lies outside "
    "it: **every observable EOS is premature, and on-time and late EOS are "
    "structurally unobservable**. `EOS_NOT_OBSERVED` is therefore ambiguous — "
    "it conflates correct stopping with never stopping — and is never read as "
    "evidence of correct stopping.")

NOT_CAUSAL = (
    "These are descriptive co-occurrences. No claim is made that premature EOS "
    "causes the route length effect, or that either measure explains the other.")


def _seed_legend(ax, loc="best", color="0.35"):
    return ax.legend(handles=[
        Line2D([], [], color=color, marker=SEED_MARKER[s], lw=0, ms=6,
               label=f"seed {s}") for s in SEEDS],
        loc=loc, frameon=False, fontsize=7.5, ncol=2)


def _operation_legend(ax, loc="upper right"):
    return ax.legend(handles=[
        Patch(facecolor="white", edgecolor="0.2", hatch=OPERATION_HATCH[op],
              label=et.OPERATION_LABEL[op]) for op in et.OPERATIONS],
        loc=loc, frameon=False, fontsize=8)


# --------------------------------------------------- Phase 4: faithful 8A

def plot_faithful_8a(cond: pd.DataFrame, out_dir: str) -> dict:
    order = (cond[cond["seed"] == SEEDS[0]].sort_values("condition_order"))
    labels = order["condition"].tolist()
    lex_of = dict(zip(order["condition"], order["source_lexicality"]))
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(11.0, 0.95 * n + 3.5), 5.0))
    width = 0.26
    for j, op in enumerate(et.OPERATIONS):
        col = f"mean_{op}_per_item"
        xs = np.arange(n) + (j - 1) * width
        means = [float(cond[cond["condition"] == c][col].mean()) for c in labels]
        cols = [LEXICALITY_COLOR[lex_of[c]] for c in labels]
        ax.bar(xs, means, width=width * 0.92, color=cols, alpha=0.55,
               edgecolor="0.15", linewidth=0.7, hatch=OPERATION_HATCH[op],
               zorder=2)
        for s in SEEDS:
            ys = [float(cond[(cond["condition"] == c) & (cond["seed"] == s)]
                        [col].iloc[0]) for c in labels]
            ax.plot(xs, ys, marker=SEED_MARKER[s], ms=3.2, lw=0, color="0.15",
                    alpha=0.8, zorder=3)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean operations per item")
    ax.grid(alpha=0.25, axis="y")
    leg = _operation_legend(ax, loc="upper left")
    ax.add_artist(leg)
    lex_leg = ax.legend(handles=[
        Patch(facecolor=LEXICALITY_COLOR[k], alpha=0.55, edgecolor="0.15",
              label=f"{LEXICALITY_LABEL[k]} (WFE source label)")
        for k in ("real", "pseudo")], loc="upper center", frameon=False,
        fontsize=8)
    ax.add_artist(lex_leg)
    _seed_legend(ax, loc="upper right", color="0.15")
    fig.suptitle("Error types by WFE condition — faithful Figure-8A "
                 "replication (FULL route, all 1,200 items)", y=1.0,
                 fontsize=12.5)
    fig.tight_layout()
    caption = (
        "**Error types by WFE condition (faithful Figure-8A replication).** "
        "Mean substitutions, deletions and insertions per evaluated item for "
        "the FULL route on the complete 1,200-item WFE set, with the original "
        "twelve conditions in their source order. Bars are the mean over the "
        "four seeds; small black markers are the individual seeds (19, 20, 21, "
        "22). Operation type is encoded by **hatch**: plain = substitutions, "
        "diagonal = deletions, dotted = insertions. Bar colour encodes the "
        "**WFE source label** (red = Real, blue = Pseudoword); that source "
        "label is a property of the stimulus set and **is not training "
        "exposure** — 122 source-Real items were never trained and 9 "
        "source-Pseudo items collide with training forms, which is why the "
        "clean-set figure is analysed separately. This is a faithful stimulus-"
        "and-metric replication adapted to four Lichtheim3 checkpoints, not a "
        "reproduction of the SWP model; Dager Figures 8B and 8C concern "
        "ablated SWP models and are out of scope.\n\n" + MEASUREMENT_LIMITS)
    return save_figure(fig, out_dir, "faithful_figure8a_error_types", caption)


# ----------------------------------------------- Phase 5: clean taxonomy

def _clean_panel(ax, cells: pd.DataFrame, route: str, ymax: float):
    groups = [(lex, grp) for lex in ("real", "pseudo")
              for grp in ("Short", "Long")]
    width = 0.19
    for j, op in enumerate(et.OPERATIONS):
        col = f"mean_{op}_per_item"
        for k, (lex, grp) in enumerate(groups):
            x = j + (k - 1.5) * width
            sub = cells[(cells["route"] == route)
                        & (cells["source_lexicality"] == lex)
                        & (cells["broad_length"] == grp)]
            m = float(sub[col].mean())
            ax.bar(x, m, width=width * 0.9, color=LEXICALITY_COLOR[lex],
                   alpha=0.35 if grp == "Short" else 0.75, edgecolor="0.15",
                   linewidth=0.7, hatch=OPERATION_HATCH[op], zorder=2)
            for s in SEEDS:
                v = sub[sub["seed"] == s][col]
                if len(v):
                    ax.plot(x, float(v.iloc[0]), marker=SEED_MARKER[s], ms=3.4,
                            lw=0, color="0.1", alpha=0.85, zorder=3)
    ax.set_xticks(range(len(et.OPERATIONS)))
    ax.set_xticklabels([et.OPERATION_LABEL[o] for o in et.OPERATIONS],
                       fontsize=9)
    ax.set_title(ROUTE_LABEL[route], fontsize=12)
    ax.set_ylim(0, ymax)
    ax.grid(alpha=0.25, axis="y")


def plot_clean_taxonomy(cells: pd.DataFrame, out_dir: str, routes=ROUTES,
                        stem: str = "clean_error_taxonomy_by_route",
                        zoom: bool = False) -> dict:
    cols = [f"mean_{op}_per_item" for op in et.OPERATIONS]
    sub = cells[cells["route"].isin(routes)]
    ymax = float(sub[cols].to_numpy(float).max()) * 1.15 or 1.0
    fig, axes = plt.subplots(1, len(routes),
                             figsize=(4.5 * len(routes) + 0.5, 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, route in zip(axes, routes):
        _clean_panel(ax, cells, route, ymax)
    axes[0].set_ylabel("Mean operations per evaluated item")
    leg = _operation_legend(axes[0], loc="upper left")
    axes[0].add_artist(leg)
    axes[-1].legend(handles=[
        Patch(facecolor=LEXICALITY_COLOR[lex], alpha=0.35 if grp == "Short"
              else 0.75, edgecolor="0.15",
              label=f"{LEXICALITY_LABEL[lex]} — {grp}")
        for lex in ("real", "pseudo") for grp in ("Short", "Long")]
        + [Line2D([], [], color="0.1", marker="o", lw=0, ms=4,
                  label="individual seeds")],
        loc="upper left", frameon=False, fontsize=7.5)
    title = ("Levenshtein error taxonomy by route — clean WFE set"
             + (" (FULL and WM only — zoom)" if zoom else ""))
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    caption = (
        f"**{title}.** Mean substitutions, deletions and insertions per "
        "evaluated item (never conditioned on being erroneous), split by "
        "lexicality and by broad phoneme length (Short = 3, 4, 5; Long = 7, 8, "
        "9; length 6 is absent from the WFE by construction). Operation type is "
        "encoded by **hatch** and x-position; colour encodes lexicality only "
        "(red = trained Real words, n = 671; blue = novel Pseudowords, n = "
        "391), and lighter versus darker shading distinguishes Short from Long. "
        "Small black markers are the four individual seeds. "
        + ("**This is the zoom companion, restricted to the FULL and WM routes "
           "because the LTM route's error magnitude compresses them on a common "
           "scale. It does not replace the absolute-scale figure, which remains "
           "the primary presentation.**"
           if zoom else
           "All panels share one absolute y-scale, so the magnitude difference "
           "between routes is visible rather than hidden by per-panel "
           "rescaling.")
        + "\n\n" + MEASUREMENT_LIMITS + "\n\n" + CLEAN_CAPTION)
    return save_figure(fig, out_dir, stem, caption)


def zoom_rule(cells: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the frozen >10x zoom rule; recorded whether it fires or not."""
    cols = [f"mean_{op}_per_item" for op in et.OPERATIONS]
    mean_over_seeds = (cells.groupby(["route", "source_lexicality",
                                      "broad_length"], as_index=False)[cols]
                       .mean())
    ltm = float(mean_over_seeds[mean_over_seeds["route"] == "ltm"][cols]
                .to_numpy(float).max())
    other = float(mean_over_seeds[mean_over_seeds["route"].isin(["full", "wm"])]
                  [cols].to_numpy(float).max())
    ratio = float(ltm / other) if other > 0 else (np.inf if ltm > 0 else np.nan)
    fires = bool(np.isfinite(ratio) and ratio > ZOOM_RATIO_TRIGGER) or \
        bool(np.isinf(ratio))
    return pd.DataFrame([{
        "rule": "frozen zoom rule (error_taxonomy_analysis_spec.md)",
        "trigger": f"max mean LTM operation count > {ZOOM_RATIO_TRIGGER:g} x "
                   "max mean FULL/WM operation count",
        "max_mean_ltm_operation_per_item": ltm,
        "max_mean_full_wm_operation_per_item": other,
        "ratio": ratio, "rule_fires": fires,
        "zoom_figure_produced": fires,
        "zoom_replaces_primary": False,
        "frozen_before_results": True}])


# --------------------------------------------------- Phase 6: premature EOS

def plot_premature_eos(by_len: pd.DataFrame, by_seed: pd.DataFrame,
                       slopes: pd.DataFrame, out_dir: str) -> dict:
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    ymax = float(by_len["premature_eos_rate"].max())
    ymax = ymax * 1.15 if ymax > 0 else 1.0
    for i, route in enumerate(ROUTES):
        ax = fig.add_subplot(gs[0, i])
        for lex in ("real", "pseudo"):
            sub = by_len[(by_len["route"] == route)
                         & (by_len["source_lexicality"] == lex)]
            for s in SEEDS:
                ss = sub[sub["seed"] == s].sort_values("phoneme_length")
                ax.plot(ss["phoneme_length"], ss["premature_eos_rate"],
                        marker=SEED_MARKER[s], ms=3.2, lw=0.7, ls=":",
                        alpha=0.45, color=LEXICALITY_COLOR[lex])
            m = (sub.groupby("phoneme_length", as_index=False)
                 ["premature_eos_rate"].mean().sort_values("phoneme_length"))
            ax.plot(m["phoneme_length"], m["premature_eos_rate"], marker="o",
                    ms=6.5, lw=2.6, color=LEXICALITY_COLOR[lex])
        ax.set_title(ROUTE_LABEL[route], fontsize=12)
        ax.set_xlabel("Phoneme length")
        ax.set_xticks(LENGTHS)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_ylabel("Premature-EOS rate")
            ax.legend(handles=[
                Line2D([], [], color=LEXICALITY_COLOR[k], lw=2.6,
                       label=LEXICALITY_LABEL[k]) for k in ("real", "pseudo")],
                loc="upper left", frameon=False, fontsize=8)

    panels = [("premature_eos_rate", "Premature-EOS rate", by_seed),
              ("mean_eos_shortfall_per_item",
               "Mean EOS shortfall per item (primary)", by_seed),
              ("length_slope", "Length slope of premature EOS\n"
                               "(linear probability, descriptive)", slopes)]
    for j, (col, title, src) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        for i, route in enumerate(ROUTES):
            for k, lex in enumerate(("real", "pseudo")):
                x = i + (k - 0.5) * 0.26
                sub = src[(src["route"] == route)
                          & (src["source_lexicality"] == lex)]
                for _, r in sub.iterrows():
                    ax.plot(x, r[col], marker=SEED_MARKER[int(r["seed"])],
                            ms=7, color=LEXICALITY_COLOR[lex], alpha=0.85,
                            zorder=3)
                ax.hlines(float(sub[col].mean()), x - 0.1, x + 0.1,
                          color="black", lw=2.0, zorder=4)
        ax.set_xticks(range(len(ROUTES)))
        ax.set_xticklabels([ROUTE_LABEL[r] for r in ROUTES])
        ax.set_xlim(-0.5, len(ROUTES) - 0.5)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        if j == 0:
            _seed_legend(ax, loc="upper left")
        if j == 2:
            ax.legend(handles=[Line2D([], [], color="black", lw=2.0,
                                      label="mean over seeds")],
                      loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("Premature end-of-sequence by route — clean WFE set", y=0.98,
                 fontsize=13)
    caption = (
        "**Premature end-of-sequence by route (clean WFE set).** Top row: the "
        "premature-EOS rate against phoneme length for each route, thin dotted "
        "lines the four individual seeds and thick lines the mean over seeds. "
        "Bottom row, left to right: the premature-EOS rate per route and "
        "lexicality; the **primary** mean EOS shortfall averaged over all "
        "evaluated items (zero for items with no observed premature EOS); and "
        "the descriptive linear-probability slope of premature EOS on phoneme "
        "length. One marker per seed throughout, with the black bar the mean "
        "over seeds. Colour encodes lexicality only (red = trained Real words, "
        "blue = novel Pseudowords). A logistic model is deliberately not forced "
        "where events are sparse or completely separated; the accompanying "
        "tables carry the per-cell model status.\n\n" + EOS_LIMITS + "\n\n"
        + NOT_CAUSAL + "\n\n" + CLEAN_CAPTION)
    return save_figure(fig, out_dir, "premature_eos_by_route", caption)


def eos_class_counts(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for route in ROUTES:
            sub = items[(items["seed"] == seed) & (items["route"] == route)]
            for cls, observable in ((eos.PREMATURE, True), (eos.ON_TIME, False),
                                    (eos.LATE, False), (eos.NOT_OBSERVED, True),
                                    (eos.UNAVAILABLE, True)):
                rows.append({
                    "seed": seed, "route": route, "eos_class": cls,
                    "n_items": int(len(sub)),
                    "n": int((sub["eos_class"] == cls).sum()),
                    "structurally_observable": observable,
                    "note": ("outside the readout window; cannot occur in this "
                             "instrumentation" if not observable else
                             "ambiguous: correct stop or no stop"
                             if cls == eos.NOT_OBSERVED else "")})
    return pd.DataFrame(rows)


# ------------------------------------------------- Phase 8: seed-22 examples

def seed22_examples(canon: pd.DataFrame, items: pd.DataFrame,
                    out_dir: str) -> dict:
    """Deterministic illustrations, declared in the spec before analysis."""
    os.makedirs(out_dir, exist_ok=True)
    cols = ["seed", "route", "item_id", "target", "prediction", "target_length",
            "predicted_length", "raw_edit_distance", "substitutions",
            "deletions", "insertions", "eos_position", "expected_eos_position",
            "eos_class", "eos_shortfall", "morphology", "size"]
    d = items[(items["seed"] == 22) & (items["source_lexicality"] == "pseudo")]

    err = d[d["word_error"] == 1].copy()
    err["_shortfall"] = err["eos_shortfall"].fillna(-1.0)
    err = err.sort_values(["route", "raw_edit_distance", "_shortfall",
                           "item_id"],
                          ascending=[True, False, False, True])
    top = err.groupby("route", group_keys=False).head(20)[cols]
    p1 = write_table(top, os.path.join(
        out_dir, "seed22_illustrative_pseudoword_errors.tsv"),
        sort_by=["route", "raw_edit_distance", "item_id"])

    pre = d[d["premature_eos"] == 1].sort_values(
        ["route", "eos_shortfall", "raw_edit_distance", "item_id"],
        ascending=[True, False, False, True])
    p2 = write_table(pre.groupby("route", group_keys=False).head(20)[cols],
                     os.path.join(out_dir,
                                  "seed22_illustrative_premature_eos.tsv"),
                     sort_by=["route", "eos_shortfall", "item_id"])

    readme = os.path.join(out_dir, "README.md")
    with open(readme, "w") as f:
        f.write(
            "# Seed-22 illustrative items\n\n"
            "**These are deterministic illustrations, not a representative "
            "sample, and they carry no inference.** Seed 22 was declared in "
            "`../error_taxonomy_analysis_spec.md` before any item was "
            "inspected; nothing here was chosen after seeing the outcome, and "
            "no claim in `../error_taxonomy_results.md` rests on these rows.\n\n"
            "Scope: clean-set novel pseudowords only (`NOVEL_PSEUDOWORD`), up "
            "to 20 rows per route.\n\n"
            "## `seed22_illustrative_pseudoword_errors.tsv`\n\n"
            "Erroneous items ordered by raw edit distance descending, then "
            "`eos_shortfall` descending with missing values last, then "
            "`item_id` ascending.\n\n"
            "## `seed22_illustrative_premature_eos.tsv`\n\n"
            "Items with an observed premature EOS, ordered by `eos_shortfall` "
            "descending, then raw edit distance descending, then `item_id` "
            "ascending.\n\n"
            "## Reading the columns\n\n"
            "`eos_position` is a **0-based index into the item's readout "
            "window** and equals the number of phonemes emitted before EOS; "
            "`expected_eos_position` is the target length L; `eos_shortfall = "
            "expected − observed`. The window holds only indices 0…L−1, so "
            "every observed EOS is premature and an on-time EOS is not "
            "representable. See `../eos_convention.md`.\n\n"
            "A premature EOS is **not** the same event as a deletion: the two "
            "columns are independent measurements and rows may show either, "
            "both or neither.\n")
    return {"errors": p1, "premature_eos": p2, "readme": readme}


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default=CANONICAL_TABLE)
    ap.add_argument("--out_root",
                    default=os.path.join(REPORT_ROOT, "error_taxonomy"))
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    root = args.out_root
    dirs = {k: os.path.join(root, *p) for k, p in {
        "faith_f": ("faithful", "figures"), "faith_t": ("faithful", "tables"),
        "clean_f": ("clean", "figures"), "clean_t": ("clean", "tables"),
        "eos_f": ("eos", "figures"), "eos_t": ("eos", "tables"),
        "strata": ("strata", "tables"), "examples": ("examples",),
    }.items()}
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)

    canon = load_canonical(args.canonical)
    written, tables = {}, []

    def _t(df, key, name, sort_by=None):
        path = write_table(df, os.path.join(dirs[key], name), sort_by=sort_by)
        tables.append(repo_relative(path))
        return path

    # -------------------------------------------------- Phase 4: faithful
    print("[faithful] Figure-8A replication")
    cond = et.faithful_condition_table(canon)
    _t(cond, "faith_t", "faithful_condition_error_types.tsv",
       sort_by=["condition_order", "seed"])
    _t(et.summarise(cond, "mean_edit_distance_per_item",
                    ["condition", "condition_order", "source_lexicality",
                     "morphology", "size"]),
       "faith_t", "faithful_condition_summary.tsv", sort_by=["condition_order"])
    comp = cond[["seed", "condition", "condition_order", "source_lexicality",
                 "n_items", "n_erroneous_items", "total_edit_operations",
                 "composition_status"]
                + [f"proportion_{op}" for op in et.OPERATIONS]]
    _t(comp, "faith_t", "faithful_condition_composition.tsv",
       sort_by=["condition_order", "seed"])
    _t(cond[["seed", "condition", "condition_order", "n_items",
             "n_erroneous_items", "cell_flag"]],
       "faith_t", "faithful_condition_cell_counts.tsv",
       sort_by=["condition_order", "seed"])
    written["faithful_figure8a_error_types"] = plot_faithful_8a(
        cond, dirs["faith_f"])

    # ----------------------------------------------------- Phase 5: clean
    print("[clean] route taxonomy")
    cells = et.clean_cells(canon)
    _t(cells, "clean_t", "clean_error_taxonomy_cells.tsv",
       sort_by=["route", "source_lexicality", "broad_length", "seed"])
    summ = pd.concat([et.summarise(cells, f"mean_{op}_per_item",
                                   ["route", "source_lexicality",
                                    "broad_length"])
                      for op in et.OPERATIONS]
                     + [et.summarise(cells, "mean_edit_distance_per_item",
                                     ["route", "source_lexicality",
                                      "broad_length"])], ignore_index=True)
    _t(summ, "clean_t", "clean_error_taxonomy_summary.tsv",
       sort_by=["quantity", "route", "source_lexicality", "broad_length"])
    _t(et.clean_by_exact_length(canon), "clean_t",
       "clean_error_taxonomy_by_exact_length.tsv",
       sort_by=["route", "source_lexicality", "phoneme_length", "seed"])
    _t(cells[["seed", "route", "source_lexicality", "broad_length", "n_items",
              "n_erroneous_items", "cell_flag", "total_edit_operations",
              "composition_status"]
             + [f"proportion_{op}" for op in et.OPERATIONS]],
       "clean_t", "clean_error_taxonomy_composition.tsv",
       sort_by=["route", "source_lexicality", "broad_length", "seed"])
    rc = et.route_contrasts(cells)
    _t(rc, "clean_t", "clean_error_taxonomy_route_contrasts.tsv",
       sort_by=["source_lexicality", "route_contrast", "seed"])
    _t(pd.concat([et.summarise(rc, f"{op}_difference",
                               ["source_lexicality", "route_contrast"])
                  for op in et.OPERATIONS]
                 + [et.summarise(rc, "total_edit_operations_difference",
                                 ["source_lexicality", "route_contrast"])],
                 ignore_index=True),
       "clean_t", "clean_error_taxonomy_route_contrasts_summary.tsv",
       sort_by=["quantity", "source_lexicality", "route_contrast"])
    print("[clean] hierarchical bootstrap (B = 10,000)")
    _t(et.bootstrap_operations(canon), "clean_t",
       "clean_error_taxonomy_bootstrap.tsv",
       sort_by=["quantity", "source_lexicality", "route"])
    zr = zoom_rule(cells)
    _t(zr, "clean_t", "clean_error_taxonomy_zoom_rule.tsv")
    written["clean_error_taxonomy_by_route"] = plot_clean_taxonomy(
        cells, dirs["clean_f"])
    if bool(zr["rule_fires"].iloc[0]):
        print("[clean] zoom rule fires -> FULL/WM companion figure")
        written["clean_error_taxonomy_full_wm_zoom"] = plot_clean_taxonomy(
            cells, dirs["clean_f"], routes=["full", "wm"],
            stem="clean_error_taxonomy_full_wm_zoom", zoom=True)

    # ------------------------------------------------------- Phase 6: EOS
    print("[eos] premature-EOS diagnostics")
    items = eos.item_level(canon, "LICHTHEIM_CLEAN")
    all_items = eos.item_level(canon, "ALL_WITH_EXPOSURE_STRATA")
    e_seed = eos.by_seed(items)
    _t(e_seed, "eos_t", "premature_eos_by_seed_route.tsv",
       sort_by=["route", "source_lexicality", "seed"])
    _t(pd.concat([eos.summarise(e_seed, c, ["route", "source_lexicality"])
                  for c in ("premature_eos_rate",
                            "mean_eos_shortfall_per_item",
                            "conditional_mean_eos_shortfall")],
                 ignore_index=True),
       "eos_t", "premature_eos_by_seed_route_summary.tsv",
       sort_by=["quantity", "route", "source_lexicality"])
    _t(eos.by_length(items), "eos_t", "premature_eos_by_exact_length.tsv",
       sort_by=["route", "source_lexicality", "phoneme_length", "seed"])
    _t(eos.by_broad_length(items), "eos_t",
       "premature_eos_by_broad_length.tsv",
       sort_by=["route", "source_lexicality", "broad_length", "seed"])
    _t(eos.by_exposure(all_items), "eos_t",
       "premature_eos_by_exposure_status.tsv",
       sort_by=["route", "lichtheim_exposure_status", "seed"])
    e_slopes = eos.length_slopes(items)
    _t(e_slopes, "eos_t", "premature_eos_length_slopes.tsv",
       sort_by=["route", "source_lexicality", "seed"])
    _t(eos.summarise(e_slopes, "length_slope", ["route", "source_lexicality"]),
       "eos_t", "premature_eos_length_slopes_summary.tsv",
       sort_by=["route", "source_lexicality"])
    _t(eos.deletion_overlap(items), "eos_t",
       "premature_eos_deletion_overlap.tsv",
       sort_by=["route", "source_lexicality"])
    _t(eos.deletion_overlap(items,
                            ["route", "source_lexicality", "broad_length"]),
       "eos_t", "premature_eos_deletion_overlap_by_length.tsv",
       sort_by=["route", "source_lexicality", "broad_length"])
    _t(eos_class_counts(items), "eos_t", "premature_eos_class_counts.tsv",
       sort_by=["route", "eos_class", "seed"])
    written["premature_eos_by_route"] = plot_premature_eos(
        eos.by_length(items), e_seed, e_slopes, dirs["eos_f"])

    # -------------------------------------------------- Phase 7: strata
    print("[strata] exposure and morphology descriptives")
    expo = et.by_exposure_status(canon)
    _t(expo, "strata", "exposure_error_taxonomy.tsv",
       sort_by=["route", "lichtheim_exposure_status", "seed"])
    _t(pd.concat([et.summarise(expo, f"mean_{op}_per_item",
                               ["route", "lichtheim_exposure_status"])
                  for op in et.OPERATIONS], ignore_index=True),
       "strata", "exposure_error_taxonomy_summary.tsv",
       sort_by=["quantity", "route", "lichtheim_exposure_status"])
    mor = et.clean_by_morphology(canon)
    _t(mor, "strata", "morphology_error_taxonomy.tsv",
       sort_by=["route", "source_lexicality", "morphology", "seed"])
    _t(pd.concat([et.summarise(mor, f"mean_{op}_per_item",
                               ["route", "source_lexicality", "morphology"])
                  for op in et.OPERATIONS], ignore_index=True),
       "strata", "morphology_error_taxonomy_summary.tsv",
       sort_by=["quantity", "route", "source_lexicality", "morphology"])
    _t(eos.by_morphology(items), "strata", "morphology_premature_eos.tsv",
       sort_by=["route", "source_lexicality", "morphology", "seed"])

    # ------------------------------------------------ Phase 8: examples
    print("[examples] seed-22 deterministic illustrations")
    ex = seed22_examples(canon, items, dirs["examples"])

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({
                "canonical_table": repo_relative(args.canonical),
                "canonical_table_sha256": sha256_file(args.canonical),
                "out_root": repo_relative(root),
                "editops_backend": et.EDITOPS_BACKEND,
                "editops_version": et.EDITOPS_VERSION,
                "zoom_rule_fired": bool(zr["rule_fires"].iloc[0]),
                "figures": {k: {kk: repo_relative(vv) for kk, vv in v.items()}
                            for k, v in written.items()},
                "tables": sorted(tables),
                "examples": {k: repo_relative(v) for k, v in ex.items()},
                "model_inference_performed": False,
                "eos_and_levenshtein_kept_separate": True}, f, indent=2)
    print(f"\nSprint-4 error-taxonomy outputs written to {repo_relative(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
