"""Summary figures for the naming / comprehension diagnostics.

Read-only: builds the synthesis plots from the run artifacts that already
exist under outputs/naming_comprehension_93a577f/.  It launches no training,
writes nothing outside the (gitignored) figure directory, and never modifies
an existing run output.

Run:
    python scripts/naming_comprehension/make_summary_plots.py
    python scripts/naming_comprehension/make_summary_plots.py --out-dir <dir>

Every source path is resolved and asserted at start-up, so a missing or
renamed run fails loudly instead of silently producing a wrong figure.

Note on the x axis: in all of these runs one epoch is exactly one complete
pass over the training population, so "epoch" and "exposures per item" are
the same quantity.  The axes are labelled with exposures per item because
that is the quantity comparable across population sizes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(ROOT, "outputs", "naming_comprehension_93a577f")
DEFAULT_FIG_DIR = os.path.join(OUT_ROOT, "summary_plots")

# ---- source runs, resolved once and asserted before anything is drawn ----
RUNS = {
    "naming_rep3288":   "phase2f_n0_representative3288_seed22",
    "naming_rep10000":  "phase2g_n0_representative10000_seed22",
    "naming_full29571": "phase2h_n0_fulllexicon_3000_seed22",
    "naming_blockB":    "phase2i_n0_repblockB_10000_seed22",
    "naming_blockC":    "phase2i_n0_repblockC_9571_seed22",
    "comprehension":    "phase2d3_c3_subset3288_stress_seed22",
    "seq_A_then_B":     "phase2j_n0_sequential_A_then_B_seed22",
    "rep_after_comp":   "phase2b_c3_flat_seed22",
    "rep_after_naming": "phase2b_n0_naming_seed22",
    "cohort":           "cohort",
}

C_BLUE, C_ORANGE, C_RED, C_GREY = "#1b6ca8", "#e08b1e", "#c0392b", "#8a8a8a"
CRIT = 95.0


def rpath(key: str, *parts: str) -> str:
    p = os.path.join(OUT_ROOT, RUNS[key], *parts)
    if not os.path.exists(p):
        raise FileNotFoundError(f"required source missing: {p}")
    return p


def load_summary(key: str) -> dict:
    with open(rpath(key, "run_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def naming_curve(summary: dict) -> Tuple[List[int], List[float]]:
    """Exposures per item and exact-match (%) at each scheduled evaluation."""
    xs = [s["epoch"] for s in summary["snapshots"]]
    ys = [100.0 * s["naming"]["exact_match"] for s in summary["snapshots"]]
    return xs, ys


def first_crossing(xs: Sequence[int], ys: Sequence[float],
                   thr: float = CRIT) -> Optional[int]:
    for x, y in zip(xs, ys):
        if y >= thr:
            return x
    return None


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12.5, pad=10)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, fig_dir: str, name: str) -> List[str]:
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# =========================================================  figure 1  ======

def figure1(fig_dir: str) -> List[str]:
    """Naming acquisition cost as a function of simultaneous population size."""
    series = [
        ("N = 3,288 (representative)", "naming_rep3288", C_BLUE),
        ("N = 10,000 (representative)", "naming_rep10000", C_ORANGE),
        ("N = 29,571 (full lexicon)", "naming_full29571", C_RED),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axhline(CRIT, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(2960, CRIT + 1.5, "95% criterion", ha="right", fontsize=9.5,
            color="black")

    for label, key, colour in series:
        xs, ys = naming_curve(load_summary(key))
        ax.plot(xs, ys, color=colour, linewidth=2.0, label=label, zorder=3)
        hit = first_crossing(xs, ys)
        if hit is not None:
            ax.plot([hit], [CRIT], marker="o", color=colour, markersize=7,
                    zorder=4)
            ax.annotate(f"{hit}", xy=(hit, CRIT), xytext=(hit + 40, CRIT - 11),
                        fontsize=10, color=colour, fontweight="bold")
        else:  # criterion not reached within budget: report the endpoint plainly
            ax.plot([xs[-1]], [ys[-1]], marker="X", color=colour, markersize=9,
                    zorder=4)
            ax.annotate(f"{ys[-1]:.1f}% after {xs[-1]:,} exposures",
                        xy=(xs[-1], ys[-1]), xytext=(xs[-1] - 120, ys[-1] + 11),
                        fontsize=10, color=colour, ha="right")

    style(ax, "exposures per item", "naming exact-match (%)",
          "Naming acquisition slows sharply with lexical scale")
    ax.set_ylim(-3, 108)
    ax.set_xlim(-40, 3080)
    ax.legend(fontsize=10, loc="center right", frameon=False)
    return save(fig, fig_dir, "fig1_naming_scaling_curve")


# =========================================================  figure 2  ======

def block_indices(key: str) -> List[int]:
    with open(rpath(key, "subset_definition.json"), encoding="utf-8") as f:
        return json.load(f)["bank_indices_in_order"]


def exact_within_full(idx: Sequence[int]) -> float:
    """Exact-match (%) of the given items inside the full-lexicon endpoint."""
    with open(rpath("naming_full29571", "final_per_item.tsv"), encoding="utf-8") as f:
        per = {int(r["bank_index"]): int(r["exact"])
               for r in csv.DictReader(f, delimiter="\t")}
    vals = [per[i] for i in idx]
    return 100.0 * sum(vals) / len(vals)


def figure2(fig_dir: str) -> List[str]:
    """Same words, trained in a ~10k block vs inside the full lexicon."""
    blocks = [
        ("Block A\n[0:10k)", "naming_rep10000"),
        ("Block B\n[10k:20k)", "naming_blockB"),
        ("Block C\n[20k:29.6k)", "naming_blockC"),
    ]
    alone, inside = [], []
    for _, key in blocks:
        alone.append(100.0 * load_summary(key)["snapshots"][-1]["naming"]["exact_match"])
        inside.append(exact_within_full(block_indices(key)))

    x = range(len(blocks))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    b1 = ax.bar([i - w / 2 for i in x], alone, w, color=C_BLUE,
                label="trained alone (~10k population)")
    b2 = ax.bar([i + w / 2 for i in x], inside, w, color=C_RED,
                label="same words inside full lexicon (29,571)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2,
                                                  b.get_height()),
                        ha="center", va="bottom", fontsize=10)

    style(ax, "", "naming exact-match (%)",
          "The same mappings are learned in ~10k blocks "
          "but not jointly at full scale")
    ax.set_xticks(list(x))
    ax.set_xticklabels([b[0] for b in blocks], fontsize=10.5)
    ax.set_ylim(0, 108)
    ax.set_title(ax.get_title(), fontsize=12.0, pad=34)
    ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.10))
    return save(fig, fig_dir, "fig2_same_words_alone_vs_full")


# =========================================================  figure 3  ======

def figure3(fig_dir: str) -> List[str]:
    """Comprehension retrieval learning curve (C3, subset 3288)."""
    s = load_summary("comprehension")
    xs = [x["epoch"] for x in s["snapshots"]]
    ys = [100.0 * x["comprehension"]["top1"] for x in s["snapshots"]]
    first = s["outcome"]["first_epoch_criterion_met"]
    stop = s["outcome"]["stopped_at_epoch"]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axhline(CRIT, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(60, CRIT + 1.6, "95% criterion", fontsize=9.5)
    ax.plot(xs, ys, color=C_BLUE, linewidth=2.0, zorder=3)

    by_epoch = dict(zip(xs, ys))
    for ep in (1000, 2000):
        if ep in by_epoch:
            ax.plot([ep], [by_epoch[ep]], marker="o", color=C_GREY, markersize=6)
            ax.annotate(f"{by_epoch[ep]:.1f}% at {ep}", (ep, by_epoch[ep]),
                        xytext=(ep + 60, by_epoch[ep] - 9), fontsize=10,
                        color="black")
    # The criterion requires two consecutive evaluations >= 95%. Top-1 crosses
    # 95% at 3700, dips below at 3750, then holds at 3800 and 3850 -- so the
    # only unambiguous landmark is the CONFIRMATION epoch, annotated here.
    ax.plot([stop], [by_epoch[stop]], marker="o", color=C_BLUE,
            markersize=8, zorder=4)
    ax.annotate(f"criterion confirmed: {stop:,}", (stop, by_epoch[stop]),
                xytext=(stop - 260, 62), fontsize=10, color=C_BLUE, ha="right",
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0))

    style(ax, "exposures per item", "comprehension top-1 retrieval (%)",
          "Comprehension is learnable, but retrieval converges slowly")
    ax.set_ylim(0, 108)
    ax.set_xlim(-60, stop + 120)
    ax.text(0.99, 0.03,
            "retrieval against the full 29,571-word GloVe bank",
            transform=ax.transAxes, ha="right", fontsize=9, color=C_GREY)
    return save(fig, fig_dir, "fig3_comprehension_learning_curve")


# =========================================================  figure 4  ======

def rep_from_snapshots(key: str, which: str) -> Dict[str, float]:
    """FULL/WM/LTM repetition from the first or last snapshot carrying it."""
    snaps = [s for s in load_summary(key)["snapshots"] if "repetition" in s]
    s = snaps[0] if which == "before" else snaps[-1]
    return s["repetition"]["primary_readout"]["exact_match"]


def figure4(fig_dir: str) -> List[str]:
    """Repetition cost of single-task training (matched full-scale runs)."""
    before = rep_from_snapshots("rep_after_comp", "before")
    after_c = rep_from_snapshots("rep_after_comp", "after")
    after_n = rep_from_snapshots("rep_after_naming", "after")
    conds = [("Before\n(frozen model)", before),
             ("After comprehension\nsingle-task", after_c),
             ("After naming\nsingle-task", after_n)]
    routes = [("FULL", C_BLUE), ("WM (dorsal)", C_ORANGE), ("LTM (ventral)", C_RED)]

    x = range(len(conds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for j, (rname, colour) in enumerate(routes):
        key = rname.split()[0].lower()
        vals = [100.0 * c[1][key] for c in conds]
        bars = ax.bar([i + (j - 1) * w for i in x], vals, w, color=colour,
                      label=rname)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9.5)

    style(ax, "", "repetition exact-match (%)",
          "Single-task training preserves WM but collapses LTM repetition")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c[0] for c in conds], fontsize=10.5)
    ax.set_ylim(0, 116)
    ax.legend(fontsize=10, frameon=False, ncol=3, loc="upper center")
    return save(fig, fig_dir, "fig4_repetition_cost_single_task")


# =========================================================  figure 5  ======

def figure5(fig_dir: str) -> List[str]:
    """Sequential A -> B: retention of A and acquisition of B (Phase 2J).

    Two distinct phenomena on one axis: Block A is evaluated throughout but
    never receives a gradient, and Block B's acquisition is compared against
    the same block learned from the canonical checkpoint (Phase 2I control).
    """
    seq = load_summary("seq_A_then_B")
    ctl = load_summary("naming_blockB")

    xs = [s["epoch"] for s in seq["snapshots"]]
    a_ret = [100.0 * s["retention"]["exact_match"] for s in seq["snapshots"]]
    b_after = [100.0 * s["naming"]["exact_match"] for s in seq["snapshots"]]
    cxs, c_alone = naming_curve(ctl)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.axhline(CRIT, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(1490, CRIT + 1.8, "95% criterion", ha="right", fontsize=9.5)

    ax.plot(cxs, c_alone, color=C_BLUE, linewidth=1.8, linestyle=(0, (5, 2)),
            label="B alone (control, from canonical)", zorder=3)
    ax.plot(xs, b_after, color=C_BLUE, linewidth=2.2, label="B after A", zorder=4)
    ax.plot(xs, a_ret, color=C_RED, linewidth=2.2,
            label="A retention during B training", zorder=5)

    # --- A: start point and the immediate collapse ---
    ax.plot([xs[0]], [a_ret[0]], marker="o", color=C_RED, markersize=7, zorder=6)
    ax.annotate(f"A: {a_ret[0]:.1f}%", (xs[0], a_ret[0]), xytext=(60, a_ret[0] + 3),
                fontsize=10.5, color=C_RED, fontweight="bold")
    ax.annotate(f"A: {a_ret[1]:.2f}% after {xs[1]} B exposures",
                xy=(xs[1], a_ret[1]), xytext=(300, 26), fontsize=10.5,
                color=C_RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))

    # --- B endpoints: after A vs alone ---
    ax.plot([xs[-1]], [b_after[-1]], marker="o", color=C_BLUE, markersize=7,
            zorder=6)
    ax.annotate(f"B after A: {b_after[-1]:.1f}%", (xs[-1], b_after[-1]),
                xytext=(xs[-1] - 30, b_after[-1] + 5), fontsize=10.5,
                color=C_BLUE, ha="right", fontweight="bold")
    ax.annotate(f"B alone: {c_alone[-1]:.1f}%", (cxs[-1], c_alone[-1]),
                xytext=(cxs[-1] - 40, c_alone[-1] - 13), fontsize=10.5,
                color=C_BLUE, ha="right")

    style(ax, "B exposures per item", "naming exact-match (%)",
          "Sequential naming causes catastrophic forgetting "
          "and slows new learning")
    ax.set_title(ax.get_title(), fontsize=12.0, pad=10)
    ax.set_ylim(-3, 108)
    ax.set_xlim(-30, 1540)
    ax.legend(fontsize=10, loc="upper left", frameon=False,
              bbox_to_anchor=(0.015, 0.90))
    return save(fig, fig_dir, "fig5_sequential_forgetting")


# ====================================================  backup figure  ======

def figure_backup(fig_dir: str) -> List[str]:
    """BACKUP: what the frozen model already does before any task training."""
    with open(os.path.join(OUT_ROOT, RUNS["cohort"], "cohort_summary.json"),
              encoding="utf-8") as f:
        cohort = json.load(f)
    seeds = cohort["by_seed"]

    def mean(k: str) -> float:
        return 100.0 * sum(s[k] for s in seeds) / len(seeds)

    labels = ["comprehension\ntop-1",
              "naming from\ntrue GloVe",
              "naming from\ns_hat (control)"]
    vals = [mean("comp_top1"), mean("naming_glove_exact"), mean("naming_shat_exact")]
    colours = [C_BLUE, C_RED, C_GREY]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    bars = ax.bar(range(3), vals, 0.55, color=colours)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10.5)
    style(ax, "", "performance (%)",
          "Frozen model: useful internal structure, neither task solved")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 112)
    ax.text(0.02, 0.94, f"mean of {len(seeds)} canonical seeds",
            transform=ax.transAxes, ha="left", fontsize=9, color=C_GREY)
    return save(fig, fig_dir, "fig_backup_frozen_starting_point")


# ==============================================================  main  =====

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=DEFAULT_FIG_DIR)
    args = ap.parse_args(argv)

    for key in RUNS:
        if key == "cohort":
            continue
        rpath(key, "run_summary.json")          # fail fast on a missing run

    produced: List[str] = []
    for fn in (figure1, figure2, figure3, figure4, figure5, figure_backup):
        produced += fn(args.out_dir)
    for p in produced:
        print(p)
    print(f"[summary_plots] {len(produced)} files -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
