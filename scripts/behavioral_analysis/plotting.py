"""Presentation layer: the five publication figures.

Conventions enforced here and tested in tests/test_behavioral_analysis.py:
  * red = Real words, blue = Pseudowords, and those two colours encode nothing
    else — exposure categories use a neutral grey palette;
  * colour meaning lives in a legend, never in a title;
  * every figure is written as PNG (300 dpi), PDF and SVG next to the exact
    TSV that produced it and a standalone caption file.
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# Fixed salt makes SVG element ids reproducible, so two runs of the same code
# on the same data produce byte-identical vector output.  Without it matplotlib
# randomises ids per process and the files differ despite identical content.
matplotlib.rcParams["svg.hashsalt"] = "lichtheim3-behavioral-wfe"

from .common import (CLEAN_CAPTION, EXPOSURE_ACCENT, EXPOSURE_COLOR,
                     EXPOSURE_ORDER, GATE_CAPTION, LENGTHS, LEXICALITY_COLOR,
                     LEXICALITY_LABEL, ROUTE_LABEL, ROUTES, SEED_MARKER, SEEDS,
                     SERIAL_POSITION_METHOD, SMALL_STRATUM_MAX_N)

DPI = 300
FORMATS = ("png", "pdf", "svg")


def save_figure(fig, out_dir: str, stem: str, caption: str) -> Dict[str, str]:
    """Write PNG/PDF/SVG plus a standalone caption file."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    # Suppressing the embedded timestamp makes vector output byte-reproducible.
    # The key differs by backend: SVG writes "Date", PDF writes "CreationDate".
    no_timestamp = {"svg": {"Date": None}, "pdf": {"CreationDate": None}}
    for ext in FORMATS:
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    metadata=no_timestamp.get(ext))
        written[ext] = path
    plt.close(fig)
    cap = os.path.join(out_dir, f"{stem}_caption.md")
    with open(cap, "w") as f:
        f.write(caption.strip() + "\n")
    written["caption"] = cap
    return written


def _lexicality_legend(ax, loc="upper left"):
    handles = [Line2D([], [], color=LEXICALITY_COLOR[k], lw=2.6,
                      label=LEXICALITY_LABEL[k]) for k in ("real", "pseudo")]
    return ax.legend(handles=handles, loc=loc, frameon=False, fontsize=9,
                     title=None)


def _seed_legend(ax, loc="upper right"):
    handles = [Line2D([], [], color="0.35", marker=SEED_MARKER[s], lw=0,
                      ms=6, label=f"seed {s}") for s in SEEDS]
    return ax.legend(handles=handles, loc=loc, frameon=False, fontsize=8,
                     ncol=2)


# ------------------------------------------------------------------ figure 1

def plot_clean_length(table: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    ymax = float(table["ci_high"].max()) * 1.08
    for ax, route in zip(axes, ROUTES):
        for lex in ("real", "pseudo"):
            sub = table[(table["route"] == route)
                        & (table["source_lexicality"] == lex)]
            if sub.empty:
                continue
            agg = sub.drop_duplicates("phoneme_length").sort_values("phoneme_length")
            ax.fill_between(agg["phoneme_length"], agg["ci_low"], agg["ci_high"],
                            color=LEXICALITY_COLOR[lex], alpha=0.15, linewidth=0)
            for s in SEEDS:
                ss = sub[sub["seed"] == s].sort_values("phoneme_length")
                ax.plot(ss["phoneme_length"], ss["mean_raw_edit_distance"],
                        marker=SEED_MARKER[s], ms=3.5, lw=0.7, alpha=0.45,
                        ls=":", color=LEXICALITY_COLOR[lex])
            ax.plot(agg["phoneme_length"], agg["mean_across_seeds"],
                    marker="o", ms=7.5, lw=2.8, color=LEXICALITY_COLOR[lex])
        ax.set_title(ROUTE_LABEL[route], fontsize=13)
        ax.set_xlabel("Phoneme length")
        ax.set_xticks(LENGTHS)
        ax.set_ylim(0, ymax)                       # shared across panels
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean raw Levenshtein edit distance")
    leg1 = _lexicality_legend(axes[0], loc="upper left")
    axes[0].add_artist(leg1)
    style = [
        Line2D([], [], color="0.35", lw=0.8, ls=":", marker="o", ms=3.5,
               label="individual seed"),
        Line2D([], [], color="0.35", lw=2.8, marker="o", ms=7.5,
               label="mean over seeds"),
        Line2D([], [], color="0.6", lw=8, alpha=0.3,
               label="95% hierarchical bootstrap"),
    ]
    axes[1].legend(handles=style, loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Length effect by route — clean WFE set", y=1.02, fontsize=13)
    caption = (
        "**Length effect by route (clean WFE set).** Mean raw Levenshtein edit "
        "distance against phoneme length for the FULL, WM-only and LTM-only "
        "routes. Thin dotted lines with small markers are individual seeds "
        "(19, 20, 21, 22); thick lines with large markers are the mean across "
        "seeds; shaded bands are 95% hierarchical bootstrap intervals "
        "(seeds resampled, then items within stratum; B = 10,000, "
        "random seed 20260730). The three panels share a y-axis. Length 6 is "
        "absent from the WFE by construction.\n\n" + CLEAN_CAPTION)
    return save_figure(fig, out_dir, "yair_clean_length_by_route", caption)


# ------------------------------------------------------------------ figure 2

def plot_clean_slopes(slopes: pd.DataFrame, contrasts: pd.DataFrame,
                      boot: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    xpos = {}
    for i, r in enumerate(ROUTES):
        xpos[("real", r)] = i
        xpos[("pseudo", r)] = i + 4
    for lex in ("real", "pseudo"):
        for s in SEEDS:
            xs = [xpos[(lex, r)] for r in ROUTES]
            ys = [float(slopes[(slopes["seed"] == s) & (slopes["route"] == r)
                               & (slopes["source_lexicality"] == lex)]
                        ["length_slope"].iloc[0]) for r in ROUTES]
            ax.plot(xs, ys, marker=SEED_MARKER[s], ms=6, lw=1.0, alpha=0.75,
                    color=LEXICALITY_COLOR[lex])
        xs = [xpos[(lex, r)] for r in ROUTES]
        ms_ = [float(slopes[(slopes["route"] == r)
                            & (slopes["source_lexicality"] == lex)]
                     ["length_slope"].mean()) for r in ROUTES]
        ax.plot(xs, ms_, marker="_", ms=26, lw=0, color="black", zorder=5)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([ROUTE_LABEL[r] for r in ROUTES] * 2)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("Length slope (edit distance per phoneme)")
    ax.set_title("Seed-level slopes by route", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    leg = _lexicality_legend(ax, loc="upper left")
    ax.add_artist(leg)
    _seed_legend(ax, loc="upper center")

    ax = axes[1]
    for i, lex in enumerate(("real", "pseudo")):
        vals = contrasts[contrasts["source_lexicality"] == lex]
        for _, row in vals.iterrows():
            ax.plot(i, row["ltm_minus_wm"], marker=SEED_MARKER[int(row["seed"])],
                    ms=9, color=LEXICALITY_COLOR[lex], alpha=0.85)
        b = boot[(boot["stratum"] == lex) & (boot["quantity"] == "ltm_minus_wm")]
        if len(b):
            ax.vlines(i, b["ci_low"].iloc[0], b["ci_high"].iloc[0],
                      color=LEXICALITY_COLOR[lex], lw=3, alpha=0.35)
        ax.hlines(vals["ltm_minus_wm"].mean(), i - 0.16, i + 0.16,
                  color="black", lw=2.2, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([LEXICALITY_LABEL["real"], LEXICALITY_LABEL["pseudo"]])
    ax.set_xlim(-0.5, 1.5)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("LTM slope − WM slope")
    ax.set_title("Primary contrast: LTM − WM", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(handles=[
        Line2D([], [], color="black", lw=2.2, label="mean over seeds"),
        Line2D([], [], color="0.6", lw=3, alpha=0.5,
               label="95% hierarchical bootstrap")],
        loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("Length-effect slopes by route — clean WFE set", y=1.02,
                 fontsize=13)
    fig.tight_layout()
    caption = (
        "**Length-effect slopes by route (clean WFE set).** Left: per-seed OLS "
        "slopes of raw edit distance on continuous phoneme length, for each "
        "route; black dashes mark the mean over seeds. Right: the primary "
        "contrast, LTM slope minus WM slope, with all four seed values shown "
        "individually (seed 21 included), the mean in black and a 95% "
        "hierarchical bootstrap interval (B = 10,000, random seed 20260730).\n\n"
        + CLEAN_CAPTION)
    return save_figure(fig, out_dir, "yair_clean_length_slopes", caption)


# ------------------------------------------------------------------ figure 3

def plot_clean_serial_position(curves: pd.DataFrame,
                               out_dir: str) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    ymax = float(curves["interpolated_error_rate"].max()) * 1.1
    for ax, route in zip(axes, ROUTES):
        for lex in ("real", "pseudo"):
            sub = curves[(curves["route"] == route)
                         & (curves["source_lexicality"] == lex)] \
                .sort_values("relative_position")
            ax.plot(sub["relative_position"], sub["interpolated_error_rate"],
                    color=LEXICALITY_COLOR[lex], lw=2.5)
        ax.set_title(ROUTE_LABEL[route], fontsize=13)
        ax.set_xlabel("Relative position in the word")
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Positional errors per item")
    _lexicality_legend(axes[0], loc="upper left")
    fig.suptitle("Serial-position error profile — clean WFE set", y=1.02,
                 fontsize=13)
    caption = (
        "**Serial-position error profile (clean WFE set).** Relative-position "
        "profile of phoneme errors for each route, pooled over the four seeds. "
        "The LTM pseudoword curve shows a strong late-position increase, while "
        "FULL and WM show much smaller increases. Method: " +
        SERIAL_POSITION_METHOD + ".\n\n" + CLEAN_CAPTION)
    return save_figure(fig, out_dir, "yair_clean_serial_position", caption)


# ------------------------------------------------------------------ figure 4

def plot_gate_clean(table: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = (("mean_lexical_confidence", "Lexical confidence"),
              ("mean_gate", "Gate g"))
    for ax, (col, name) in zip(axes, panels):
        for i, lex in enumerate(("real", "pseudo")):
            vals = table[table["source_lexicality"] == lex][col]
            ax.scatter([i] * len(vals), vals, s=70,
                       color=LEXICALITY_COLOR[lex], alpha=0.85, zorder=3)
            ax.hlines(vals.mean(), i - 0.16, i + 0.16, color="black", lw=2.2,
                      zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([LEXICALITY_LABEL["real"], LEXICALITY_LABEL["pseudo"]])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
    axes[1].axhline(0.5, ls=":", color="grey")
    axes[1].annotate("Equal WM/LTM weighting", xy=(1.45, 0.5),
                     xytext=(0, 3), textcoords="offset points",
                     ha="right", va="bottom", fontsize=8, color="grey")
    axes[0].legend(handles=[
        Line2D([], [], color="0.35", marker="o", lw=0, ms=7,
               label="one point per seed"),
        Line2D([], [], color="black", lw=2.2, label="mean over seeds")],
        loc="best", frameon=False, fontsize=8)
    fig.suptitle("Gate and lexical confidence — clean WFE set", y=1.02,
                 fontsize=13)
    fig.tight_layout()
    caption = (
        "**Gate and lexical confidence (clean WFE set).** One point per seed. "
        "The dotted line at g = 0.5 marks equal WM/LTM weighting. " +
        GATE_CAPTION + "\n\n" + CLEAN_CAPTION)
    return save_figure(fig, out_dir, "gate_by_clean_lexicality", caption)


# ------------------------------------------------------------------ figure 5

def plot_gate_exposure(table: pd.DataFrame, out_dir: str) -> Dict[str, str]:
    order = [s for s in EXPOSURE_ORDER
             if s in set(table["exposure_status"])]
    n_by = {s: int(table[table["exposure_status"] == s]["n_items"].iloc[0])
            for s in order}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True)
    panels = (("mean_lexical_confidence", "Lexical confidence"),
              ("mean_gate", "Gate g"))
    for ax, (col, name) in zip(axes, panels):
        for i, st in enumerate(order):
            vals = table[table["exposure_status"] == st][col]
            small = n_by[st] <= SMALL_STRATUM_MAX_N
            ax.scatter([i] * len(vals), vals, s=62,
                       color=EXPOSURE_ACCENT if small else EXPOSURE_COLOR,
                       marker="x" if small else "o", alpha=0.9, zorder=3)
            ax.hlines(vals.mean(), i - 0.18, i + 0.18, color="black", lw=2.0,
                      zorder=4)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25, axis="y")
    axes[1].axhline(0.5, ls=":", color="grey")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(
        [f"{s.replace('_', chr(10))}\nn = {n_by[s]}"
         + ("\n(descriptive only)" if n_by[s] <= SMALL_STRATUM_MAX_N else "")
         for s in order], fontsize=7.5)
    axes[0].legend(handles=[
        Line2D([], [], color=EXPOSURE_COLOR, marker="o", lw=0, ms=7,
               label="one point per seed"),
        Line2D([], [], color=EXPOSURE_ACCENT, marker="x", lw=0, ms=7,
               label=f"n ≤ {SMALL_STRATUM_MAX_N}: descriptive only"),
        Line2D([], [], color="black", lw=2.0, label="mean over seeds")],
        loc="best", frameon=False, fontsize=8)
    fig.suptitle("Gate and lexical confidence by training-exposure status",
                 y=0.98, fontsize=13)
    fig.tight_layout()
    caption = (
        "**Gate and lexical confidence by training-exposure status.** One "
        "point per seed for each of the six exposure categories, with the "
        "sample size printed under each category. Categories with n ≤ "
        f"{SMALL_STRATUM_MAX_N} are marked with crosses and are descriptive "
        "only; no claim rests on them. Neutral greys distinguish exposure "
        "categories: red and blue are reserved for lexicality elsewhere and "
        "are deliberately not reused here. " + GATE_CAPTION)
    return save_figure(fig, out_dir, "gate_by_exposure_status", caption)
