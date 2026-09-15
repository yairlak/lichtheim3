"""Figures for the GATING / ROUTE DIAGNOSTICS evidence package.

Reproducibility contract: every figure is built SOLELY from files in
`figure_source_data/`, and each figure writes the exact values it plots back to
`figure_source_data/figNN_*.tsv`.  Deleting `figures/` and re-running this
script must reproduce them byte-for-byte in content.

Only the figures implied by the frozen diagnostics are generated (contract
PHASE 6).  Nothing here searches a new hypothesis, fits anything, or tunes a
parameter.

Palette: the dataviz reference instance, slots 1-3, used UNCHANGED.  That
subset is the documented all-pairs-validated subset (worst pair CVD dE 9.2
light, normal-vision 24.0 light).  No new palette is invented.  Static PNG for
a print-bound evidence package, so a single light look is a deliberate
commitment; the per-figure TSVs are the table view that the relief rule
requires.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- dataviz reference palette, slots 1-3 (light), unchanged -----------------
C_BOTH = "#2a78d6"      # slot 1 blue    - BOTH_CORRECT / primary
C_WMONLY = "#eb6834"    # slot 2 orange  - WM_ONLY_CORRECT
C_THIRD = "#1baf7a"     # slot 3 aqua    - third series where needed
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"

GATE_LO, GATE_HI = 0.0323, 0.6457      # attainable range, alpha=2.0 tau=0.7
MATERIALITY_DG = 0.01                   # contract 6.6

STATE_ORDER = ["W1_SRC", "W1_REP", "W2_SRC", "W2_REP",
               "W3_SRC", "W3_REP", "W4_SRC", "W4_REP"]
WITNESS_LABEL = {"W1": "W1 seed20 u3600", "W2": "W2 seed21 u3400",
                 "W3": "W3 seed19 u3825", "W4": "W4 seed20 u3040"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
})


def load_shards(src: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for p in sorted(glob.glob(os.path.join(src, "item_level_*.tsv"))):
        sid = os.path.basename(p)[len("item_level_"):-len(".tsv")]
        if "SMOKE" in sid:
            raise RuntimeError(f"HARD STOP: smoke shard in figure source data: {p}")
        with open(p, newline="") as f:
            out[sid] = list(csv.DictReader(f, delimiter="\t"))
    return out


def write_source(src: str, name: str, rows: List[dict]) -> str:
    p = os.path.join(src, name)
    if not rows:
        open(p, "w").close()
        return p
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return p


def _ecdf(v: np.ndarray):
    x = np.sort(v)
    return x, np.arange(1, x.size + 1) / x.size


def _grid(n_rows, n_cols, w, h, title):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w, h))
    fig.suptitle(title, fontsize=10, color=INK, y=0.985, ha="left", x=0.055)
    return fig, np.atleast_1d(axes).ravel()


# ------------------------------------------------------------------- fig 1
def fig1(data, src, figs):
    rows = []
    fig, axes = _grid(2, 4, 13, 6.2,
                      "Fig 1  ECDF of gate g by route-competence category, per state")
    for ax, sid in zip(axes, STATE_ORDER):
        d = data.get(sid, [])
        ax.grid(True, alpha=0.6, linewidth=0.6)
        for cat, col in (("BOTH_CORRECT", C_BOTH), ("WM_ONLY_CORRECT", C_WMONLY)):
            v = np.array([float(r["gate"]) for r in d
                          if r["competence_category"] == cat], float)
            if v.size == 0:
                continue
            x, y = _ecdf(v)
            ax.plot(x, y, color=col, linewidth=2, solid_capstyle="round")
            for q in (0.05, 0.25, 0.5, 0.75, 0.95):
                rows.append({"state_id": sid, "category": cat, "n": v.size,
                             "quantile": q, "gate": float(np.quantile(v, q))})
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_title(sid, fontsize=8.5, loc="left")
        # Scaled to the OBSERVED support, not the attainable range: this figure's
        # job is the between-category separation.  Fig 2 carries the attainable
        # range, so the compression is not lost from the package.
        allv = np.array([float(r["gate"]) for r in d], float)
        ax.set_xlim(float(allv.min()) - 0.005, float(allv.max()) + 0.005)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("cumulative fraction")
    axes[4].set_ylabel("cumulative fraction")
    for a in axes[4:]:
        a.set_xlabel("gate g  (ventral weight)")
    h = [plt.Line2D([], [], color=C_BOTH, lw=2),
         plt.Line2D([], [], color=C_WMONLY, lw=2),
         plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 3)))]
    fig.legend(h, ["BOTH_CORRECT (ventral succeeds)",
                   "WM_ONLY_CORRECT (ventral fails)", "g = 0.5"],
               loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(os.path.join(figs, "fig1_gate_ecdf_by_competence.png"), dpi=200)
    plt.close(fig)
    write_source(src, "fig1_gate_ecdf_by_competence.tsv", rows)


# ------------------------------------------------------------------- fig 2
def fig2(data, src, figs):
    rows = []
    fig, axes = _grid(2, 4, 13, 6.2,
                      "Fig 2  Item-level gate distribution, against g = 0.5 and the "
                      "attainable range [0.0323, 0.6457]")
    bins = np.linspace(GATE_LO, GATE_HI, 61)
    for ax, sid in zip(axes, STATE_ORDER):
        d = data.get(sid, [])
        v = np.array([float(r["gate"]) for r in d], float)
        ax.grid(True, alpha=0.6, linewidth=0.6, axis="y")
        cnt, edges = np.histogram(v, bins=bins)
        ax.bar(edges[:-1], cnt, width=np.diff(edges), align="edge",
               color=C_BOTH, edgecolor=SURFACE, linewidth=0.4)
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.axvline(float(v.mean()), color=C_WMONLY, linewidth=1.6)
        ax.set_title(f"{sid}   mean {v.mean():.4f}", fontsize=8.5, loc="left")
        ax.set_xlim(GATE_LO, GATE_HI)
        for c, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
            rows.append({"state_id": sid, "bin_lo": float(e0), "bin_hi": float(e1),
                         "count": int(c)})
    axes[0].set_ylabel("items")
    axes[4].set_ylabel("items")
    for a in axes[4:]:
        a.set_xlabel("gate g  (ventral weight)")
    h = [plt.Line2D([], [], color=C_WMONLY, lw=1.6),
         plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (3, 3)))]
    fig.legend(h, ["item-level mean g", "g = 0.5 (fixed05)"], loc="lower center",
               ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(os.path.join(figs, "fig2_gate_distribution_vs_half.png"), dpi=200)
    plt.close(fig)
    write_source(src, "fig2_gate_distribution_vs_half.tsv", rows)


# ------------------------------------------------------------------- fig 3
def fig3(data, src, figs):
    rows = []
    fig, axes = _grid(1, 4, 13, 3.4,
                      "Fig 3  Source -> repaired per-item gate change  "
                      "(dorsal route bit-identical within each pair)")
    for ax, wid in zip(axes, ["W1", "W2", "W3", "W4"]):
        s, r = data.get(f"{wid}_SRC", []), data.get(f"{wid}_REP", [])
        ax.grid(True, alpha=0.6, linewidth=0.6, axis="y")
        if not s or not r:
            ax.set_title(f"{WITNESS_LABEL[wid]} — missing", fontsize=8.5, loc="left")
            continue
        gs = {x["item_index"]: float(x["gate"]) for x in s}
        dg = np.array([float(x["gate"]) - gs[x["item_index"]]
                       for x in r if x["item_index"] in gs], float)
        lim = max(float(np.abs(dg).max()), MATERIALITY_DG) * 1.15
        cnt, edges = np.histogram(dg, bins=np.linspace(-lim, lim, 61))
        ax.bar(edges[:-1], cnt, width=np.diff(edges), align="edge",
               color=C_BOTH, edgecolor=SURFACE, linewidth=0.4)
        ax.axvline(0, color=MUTED, linewidth=1)
        for t in (-MATERIALITY_DG, MATERIALITY_DG):
            ax.axvline(t, color=C_WMONLY, linewidth=1.2, linestyle=(0, (3, 3)))
        ax.set_title(f"{WITNESS_LABEL[wid]}\nmean |dg| = {np.abs(dg).mean():.3e}",
                     fontsize=8.5, loc="left")
        ax.set_xlabel("dg = g(repaired) - g(source)")
        for c, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
            rows.append({"witness": wid, "bin_lo": float(e0), "bin_hi": float(e1),
                         "count": int(c)})
    axes[0].set_ylabel("items")
    h = [plt.Line2D([], [], color=C_WMONLY, lw=1.2, ls=(0, (3, 3)))]
    fig.legend(h, [f"materiality threshold +/- {MATERIALITY_DG} (contract 6.6)"],
               loc="lower center", ncol=1, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    fig.savefig(os.path.join(figs, "fig3_source_vs_repaired_delta_gate.png"), dpi=200)
    plt.close(fig)
    write_source(src, "fig3_source_vs_repaired_delta_gate.tsv", rows)


# ------------------------------------------- figs 4-6: errors gained under fixed05
def _gained(d, conv, key=None):
    """Items correct under the native gate and wrong under fixed05."""
    out = []
    for r in d:
        if r.get(f"{conv}_full_exact") == "1" and r.get(f"{conv}_fixed05_exact") == "0":
            out.append(r)
    return out


def _zero_matrix_figure(title, col_labels, cell_counts, cell_ns, xlabel,
                        out_png, note_extra=""):
    """One panel: states x strata, each cell annotated with errors gained.

    All-zero is a single fact, and eight empty bar panels communicate it badly.
    This renders the same stratification compactly so the null is legible at a
    glance, while the per-figure TSV keeps every number.  Non-zero cells are
    shaded, so this layout does not hide a future non-null result.
    """
    n_rows, n_cols = len(STATE_ORDER), len(col_labels)
    m = np.array([[cell_counts[(s_, c)] for c in col_labels] for s_ in STATE_ORDER],
                 float)
    total = int(m.sum())
    fig, ax = plt.subplots(figsize=(max(6.5, 1.05 * n_cols + 3.2), 4.4))
    fig.suptitle(title, fontsize=10, color=INK, y=0.975, ha="left", x=0.02)
    ax.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "z", [SURFACE, C_WMONLY]), vmin=0, vmax=max(m.max(), 1), aspect="auto")
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(STATE_ORDER, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8.5)
    for i in range(n_rows):
        for j in range(n_cols):
            v = int(m[i, j])
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                    color=INK if v == 0 else "#ffffff")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    msg = (f"{total} errors gained in total across all {n_rows} states — "
           "fixed05 produced NO discordant pairs" if total == 0
           else f"{total} errors gained in total")
    ax.text(0.0, -0.215, msg + note_extra, transform=ax.transAxes,
            fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _gained_flag(r, conv):
    return (r.get(f"{conv}_full_exact") == "1"
            and r.get(f"{conv}_fixed05_exact") == "0")


def figs456(data, src, figs):
    conv = "canonical"
    note = ("\nCanonical forced-length AR; genuine free-AR is identical "
            "(both conventions: 0 discordant pairs).")

    # ---- fig 4: by competence category ----
    cats = ["BOTH_CORRECT", "WM_ONLY_CORRECT", "LTM_ONLY_CORRECT", "NEITHER_CORRECT"]
    short = ["BOTH", "WM_ONLY", "LTM_ONLY", "NEITHER"]
    counts, ns, rows4 = {}, {}, []
    for sid in STATE_ORDER:
        d = data.get(sid, [])
        for c, sc in zip(cats, short):
            sub = [r for r in d if r["competence_category"] == c]
            v = sum(1 for r in sub if _gained_flag(r, conv))
            counts[(sid, sc)] = v
            rows4.append({"state_id": sid, "competence_category": c,
                          "n_items": len(sub), "errors_gained": v})
    _zero_matrix_figure(
        "Fig 4  fixed05 errors gained, by route-competence category",
        short, counts, ns, "route-competence category",
        os.path.join(figs, "fig4_fixed05_errors_gained_by_competence.png"), note)
    write_source(src, "fig4_fixed05_errors_gained_by_competence.tsv", rows4)

    # ---- fig 5: by frequency quintile (per state) ----
    qlab = [f"Q{q+1}" for q in range(5)]
    counts, rows5 = {}, []
    for sid in STATE_ORDER:
        d = data.get(sid, [])
        z = np.array([float(r["zipf_approx"]) for r in d], float)
        edges = np.percentile(z, [20, 40, 60, 80])
        qi = np.digitize(z, edges)
        for q in range(5):
            sub = [r for r, k in zip(d, qi) if k == q]
            v = sum(1 for r in sub if _gained_flag(r, conv))
            counts[(sid, qlab[q])] = v
            rows5.append({"state_id": sid, "quintile": qlab[q],
                          "n_items": len(sub), "errors_gained": v})
    _zero_matrix_figure(
        "Fig 5  fixed05 errors gained, by lexical-frequency quintile (Q1 lowest)",
        qlab, counts, {}, "frequency quintile",
        os.path.join(figs, "fig5_fixed05_errors_gained_by_frequency.png"), note)
    write_source(src, "fig5_fixed05_errors_gained_by_frequency.tsv", rows5)

    # ---- fig 6: by target length ----
    lengths = sorted({int(r["length"]) for d in data.values() for r in d})
    llab = [str(L) for L in lengths]
    counts, rows6 = {}, []
    for sid in STATE_ORDER:
        d = data.get(sid, [])
        for L in lengths:
            sub = [r for r in d if int(r["length"]) == L]
            v = sum(1 for r in sub if _gained_flag(r, conv))
            counts[(sid, str(L))] = v
            rows6.append({"state_id": sid, "length": L, "n_items": len(sub),
                          "errors_gained": v})
    _zero_matrix_figure(
        "Fig 6  fixed05 errors gained, by target phoneme length",
        llab, counts, {}, "target length (phonemes)",
        os.path.join(figs, "fig6_fixed05_errors_gained_by_length.png"), note)
    write_source(src, "fig6_fixed05_errors_gained_by_length.tsv", rows6)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "paper_programme", "gating_route_diagnostics"))
    a = ap.parse_args(argv)
    src = os.path.join(a.out_dir, "figure_source_data")
    figs = os.path.join(a.out_dir, "figures")
    os.makedirs(figs, exist_ok=True)

    data = load_shards(src)
    missing = [s for s in STATE_ORDER if s not in data]
    if missing:
        print(f"HARD STOP: missing state shards: {missing}", file=sys.stderr)
        return 2
    print(f"loaded {len(data)} state shards "
          f"({ {k: len(v) for k, v in sorted(data.items())} })")

    fig1(data, src, figs); print("  fig1 ok")
    fig2(data, src, figs); print("  fig2 ok")
    fig3(data, src, figs); print("  fig3 ok")
    figs456(data, src, figs); print("  fig4-6 ok")
    print(f"figures -> {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
