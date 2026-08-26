"""Summary figures for the Phase 3 interleaved multitask experiments.

Read-only: builds the figures from the run artifacts that already exist under
outputs/naming_comprehension_93a577f/.  It launches no training, performs no
new analysis, and never modifies a scientific output.

Run:
    python scripts/naming_comprehension/make_multitask_summary_plots.py
    python scripts/naming_comprehension/make_multitask_summary_plots.py --out-dir <dir>

Source note: the Phase 3B run summary already carries the COMPLETE 0 -> 780k
trajectory, because the continuation inherited Phase 3A M2's snapshots through
step 400,000.  Those shared snapshots were verified bit-identical to Phase 3A's
own, and step 400,000 appears exactly once, so Figure 1 uses Phase 3B alone as
one continuous trajectory rather than stitching two files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(ROOT, "outputs", "naming_comprehension_93a577f")
DEFAULT_FIG_DIR = os.path.join(OUT_ROOT, "multitask_summary_plots")

RUNS = {
    "phase3a": "phase3a_multitask_123_subset3288_seed22",
    "phase3b": "phase3b_multitask_123_continue400k_to800k_subset3288_seed22",
    "phase3c": "phase3c_multitask_123_replexfull_subset3288_seed22",
}

# Canonical historical full-lexicon repetition (seed22/e140).
CANONICAL = {"full": 1.000000, "wm": 0.999763, "ltm": 0.989449}

C_BLUE, C_ORANGE, C_RED, C_GREY = "#1b6ca8", "#e08b1e", "#c0392b", "#8a8a8a"
CRIT = 95.0


def rpath(key: str, *parts: str) -> str:
    p = os.path.join(OUT_ROOT, RUNS[key], *parts)
    if not os.path.exists(p):
        raise FileNotFoundError(f"required source missing: {p}")
    return p


def load(key: str) -> dict:
    with open(rpath(key, "run_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def style(ax, xlabel: str, ylabel: str, title: str, title_size: float = 12.5) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=title_size, pad=10)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def k_ticks(ax) -> None:
    """Render step counts as 0, 100k, 200k ... -- easier to read on a slide."""
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}k"))


def save(fig, fig_dir: str, name: str) -> List[str]:
    os.makedirs(fig_dir, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def series(summary: dict) -> Dict[str, List[float]]:
    """Steps and the three subset3288 curves, in percent, unsmoothed."""
    s = summary["snapshots"]
    return {
        "step": [x["step"] for x in s],
        "ltm": [100.0 * x["repetition"]["ltm"] for x in s],
        "naming": [100.0 * x["naming"]["exact_match"] for x in s],
        "comp": [100.0 * x["comprehension"]["top1"] for x in s],
        "probe": [100.0 * x["repetition_probe"]["ltm"]
                  if "repetition_probe" in x else None for x in s],
    }


def bars(ax, groups: Sequence[str], conditions: Sequence[Tuple[str, str]],
         values: Dict[str, Sequence[float]], width: float = 0.36,
         fmt: str = "{:.1f}") -> None:
    x = range(len(groups))
    n = len(conditions)
    off = [(i - (n - 1) / 2) * width for i in range(n)]
    for (label, colour), o in zip(conditions, off):
        b = ax.bar([i + o for i in x], values[label], width, color=colour,
                   label=label)
        for r in b:
            ax.annotate(fmt.format(r.get_height()),
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(list(x))


# =========================================================  figure 1  ======

def figure1(fig_dir: str) -> List[str]:
    d = series(load("phase3b"))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.axhline(CRIT, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(20_000, CRIT + 2.0, "95% criterion", ha="left", fontsize=9.5)

    ax.plot(d["step"], d["ltm"], color=C_RED, linewidth=2.0,
            label="repetition (LTM route)", zorder=4)
    ax.plot(d["step"], d["naming"], color=C_BLUE, linewidth=2.0,
            label="naming", zorder=3)
    ax.plot(d["step"], d["comp"], color=C_ORANGE, linewidth=2.0,
            label="comprehension", zorder=3)

    by = dict(zip(d["step"], d["comp"]))
    ax.plot([400_000], [by[400_000]], marker="o", color=C_ORANGE, markersize=6,
            zorder=5)
    ax.annotate(f"{by[400_000]:.1f}% at 400k", (400_000, by[400_000]),
                xytext=(400_000 - 20_000, by[400_000] - 13), fontsize=10,
                color=C_ORANGE, ha="right")
    ax.annotate("first simultaneous\ncrossing 720k",
                xy=(720_000, CRIT), xytext=(560_000, 62), fontsize=10,
                color="black", ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    ax.text(0.985, 0.055, "780k: R 100% / N 100% / C 95.1%\n"
                          "confirmed at 760k + 780k",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc"))

    style(ax, "Total optimizer steps", "Performance (%)",
          "Interleaved training enables local three-task coexistence")
    ax.set_ylim(-3, 108)
    ax.set_xlim(-15_000, 800_000)
    ax.legend(fontsize=10, loc="center right", frameon=False)
    k_ticks(ax)
    return save(fig, fig_dir, "fig1_multitask_local_coexistence")


# =========================================================  figure 2  ======

def figure2(fig_dir: str) -> List[str]:
    b, c = load("phase3b"), load("phase3c")
    sb, sc = b["snapshots"][-1], c["snapshots"][-1]
    fb = b["endpoint_full_lexicon_repetition"]["primary_readout"]["exact_match"]
    fc = c["endpoint_full_lexicon_repetition"]["primary_readout"]["exact_match"]

    groups = ["comprehension\n(subset3288)", "naming\n(subset3288)",
              "repetition LTM\n(subset3288)", "repetition LTM\n(full lexicon)"]
    lab_b = "Local rehearsal  R$_{3288}$  @780k"
    lab_c = "Full-lexicon rehearsal  R$_{29571}$  @800k"
    vals = {
        lab_b: [100 * sb["comprehension"]["top1"], 100 * sb["naming"]["exact_match"],
                100 * sb["repetition"]["ltm"], 100 * fb["ltm"]],
        lab_c: [100 * sc["comprehension"]["top1"], 100 * sc["naming"]["exact_match"],
                100 * sc["repetition"]["ltm"], 100 * fc["ltm"]],
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.axhline(CRIT, color="black", linestyle="--", linewidth=1.0, zorder=1)
    bars(ax, groups, [(lab_b, C_BLUE), (lab_c, C_ORANGE)], vals)
    ax.set_xticklabels(groups, fontsize=10)
    style(ax, "", "Performance (%)",
          "Rehearsal population determines the preservation-acquisition trade-off",
          title_size=12.0)
    ax.set_title(ax.get_title(), fontsize=12.0, pad=34)
    ax.set_ylim(0, 112)
    ax.text(-0.44, CRIT + 2, "95%", fontsize=9, color="black", ha="left")
    ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.11))
    ax.text(0.5, -0.19, "naming and comprehension stay on subset3288 in both runs; "
                        "M2 = 1:2:3 unchanged. Endpoints differ slightly (780k vs 800k).",
            transform=ax.transAxes, ha="center", fontsize=9, color=C_GREY)
    return save(fig, fig_dir, "fig2_local_vs_global_rehearsal")


# =========================================================  figure 3  ======

def figure3(fig_dir: str) -> List[str]:
    b, c = load("phase3b"), load("phase3c")
    fb = b["endpoint_full_lexicon_repetition"]["primary_readout"]["exact_match"]
    fc = c["endpoint_full_lexicon_repetition"]["primary_readout"]["exact_match"]

    groups = ["FULL", "WM (dorsal)", "LTM (ventral)"]
    conds = [("Canonical seed22/e140", C_GREY),
             ("Local rehearsal @780k", C_BLUE),
             ("Full-lexicon rehearsal @800k", C_ORANGE)]
    vals = {
        "Canonical seed22/e140": [100 * CANONICAL[k] for k in ("full", "wm", "ltm")],
        "Local rehearsal @780k": [100 * fb[k] for k in ("full", "wm", "ltm")],
        "Full-lexicon rehearsal @800k": [100 * fc[k] for k in ("full", "wm", "ltm")],
    }
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    bars(ax, groups, conds, vals, width=0.26, fmt="{:.1f}")
    ax.set_xticklabels(groups, fontsize=10.5)
    style(ax, "", "Repetition exact-match (%), full 29,571-word lexicon",
          "Full-lexicon rehearsal preserves historical repetition")
    ax.set_title(ax.get_title(), fontsize=12.5, pad=34)
    ax.set_ylim(0, 116)
    ax.legend(fontsize=9.5, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.11))
    drop = 100 * (CANONICAL["ltm"] - fc["ltm"])
    ax.text(0.5, -0.13, f"LTM is preserved but not identical: "
                        f"{drop:.1f} points below canonical.",
            transform=ax.transAxes, ha="center", fontsize=9, color=C_GREY)
    return save(fig, fig_dir, "fig3_global_repetition_preservation")


# ====================================================  backup figure  ======

def figure_backup(fig_dir: str) -> List[str]:
    b, c = series(load("phase3b")), series(load("phase3c"))
    xmax = 200_000

    def cut(d, key):
        xs = [s for s in d["step"] if s <= xmax]
        return xs, [v for s, v in zip(d["step"], d[key]) if s <= xmax]

    xb, yb = cut(b, "ltm")
    xc, yc = cut(c, "ltm")
    xp, yp = cut(c, "probe")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axhline(CRIT, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(3_000, CRIT + 1.4, "95%", ha="left", fontsize=9.5)
    ax.plot(xb, yb, color=C_BLUE, linewidth=2.0, marker="o", markersize=3.5,
            label="local rehearsal (subset3288 LTM)")
    ax.plot(xc, yc, color=C_ORANGE, linewidth=2.0, marker="o", markersize=3.5,
            label="full-lexicon rehearsal (subset3288 LTM)")
    ax.plot(xp, yp, color=C_ORANGE, linewidth=1.5, linestyle=(0, (5, 2)),
            label="full-lexicon rehearsal (out-of-subset probe)")

    for xs, ys, colour in ((xb, yb, C_BLUE), (xc, yc, C_ORANGE)):
        i = min(range(len(ys)), key=lambda k: ys[k])
        ax.annotate(f"{ys[i]:.1f}% at {xs[i] // 1000}k", (xs[i], ys[i]),
                    xytext=(xs[i] + 12_000, ys[i] - 6), fontsize=10, color=colour)

    style(ax, "Total optimizer steps", "Repetition exact-match (%)",
          "Repetition is initially destabilized before recovering")
    ax.set_ylim(50, 105)
    ax.set_xlim(-5_000, xmax + 5_000)
    ax.legend(fontsize=9.5, loc="lower right", frameon=False)
    k_ticks(ax)
    ax.text(0.02, 0.04, "the probe exists only for Phase 3C", fontsize=9,
            color=C_GREY, transform=ax.transAxes)
    return save(fig, fig_dir, "fig_backup_ltm_recovery_dynamics")


# ==============================================================  main  =====

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=DEFAULT_FIG_DIR)
    args = ap.parse_args(argv)
    for k in RUNS:
        rpath(k, "run_summary.json")
    produced: List[str] = []
    for fn in (figure1, figure2, figure3, figure_backup):
        produced += fn(args.out_dir)
    for p in produced:
        print(p)
    print(f"[multitask_plots] {len(produced)} files -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
