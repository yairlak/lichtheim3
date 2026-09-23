"""Descriptive figures. Every figure shows ALL severities k=0..15.

Conventions, applied uniformly and stated in each caption:
  * k>0 : marker at the mean over realizations, band = mean +/- 1 SAMPLE SD
          (ddof=1). Where only one realization exists no band is drawn.
  * k=0 : the exact preserved intact control, drawn as a horizontal reference
          line with NO uncertainty band -- there is one observation, and an
          invented interval would misrepresent it.
  * P1/P2/P3 keep distinct colours and markers; P4 is drawn dashed and
          labelled SENSITIVITY. Models are never pooled.
No severity is hidden, selected or highlighted.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

from paper_programme.lesioning_v2.lesion_operator import battery
from paper_programme.lesioning_v2.post_analysis import io_utils

STATE_STYLE = {
    "P1_POST_REPAIR": dict(color="#1f77b4", marker="o", ls="-",  label="P1 (primary)"),
    "P2_POST_REPAIR": dict(color="#2ca02c", marker="s", ls="-",  label="P2 (replication)"),
    "P3_POST_REPAIR": dict(color="#9467bd", marker="^", ls="-",  label="P3 (replication)"),
    "P4_POST_REPAIR": dict(color="#d62728", marker="D", ls="--", label="P4 (SENSITIVITY)"),
}
STATE_ORDER = ("P1_POST_REPAIR", "P2_POST_REPAIR", "P3_POST_REPAIR",
               "P4_POST_REPAIR")
SITES = ("L1", "L2", "L3")
BAND = "mean +/- 1 sample SD (ddof=1); no band where n_realizations < 2"


def _rows_for(curve: Sequence[Dict], site: str, endpoint: str, state: str):
    sel = [r for r in curve
           if r["site"] == site and r["endpoint"] == endpoint
           and r["state_id"] == state]
    return sorted(sel, key=lambda r: int(r["severity_k"]))


def _panel(ax, curve, site, ep):
    for state in STATE_ORDER:
        rows = _rows_for(curve, site, ep.key, state)
        if not rows:
            continue
        st = STATE_STYLE[state]
        k0 = [r for r in rows if int(r["severity_k"]) == 0]
        nz = [r for r in rows if int(r["severity_k"]) > 0]
        xs = [int(r["severity_k"]) for r in nz]
        ys = [io_utils.fnum(r["mean_exact_match"]) for r in nz]
        sd = [io_utils.fnum(r["sd_exact_match"]) for r in nz]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls=st["ls"],
                ms=3.5, lw=1.3, label=st["label"])
        lo = [y - (s or 0.0) for y, s in zip(ys, sd)]
        hi = [y + (s or 0.0) for y, s in zip(ys, sd)]
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15, lw=0)
        if k0:                      # exact baseline, no band
            b = io_utils.fnum(k0[0]["mean_exact_match"])
            ax.axhline(b, color=st["color"], ls=":", lw=1.0, alpha=0.8)
            ax.plot([0], [b], color=st["color"], marker=st["marker"], ms=5,
                    mfc="white", mew=1.2)
    ax.set_xlim(-0.6, 15.6)
    ax.set_xticks(range(0, 16))
    ax.set_xticklabels([str(k) for k in range(16)], fontsize=6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{ep.key}  [{ep.tier}]", fontsize=8)
    ax.set_xlabel("severity k (0 = intact control)", fontsize=7)
    ax.set_ylabel("exact match", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, lw=0.5)


def _figure(curve, site, eps, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(eps)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.1), squeeze=False)
    for ax, ep in zip(axes[0], eps):
        _panel(ax, curve, site, ep)
    axes[0][0].legend(fontsize=6, loc="lower left", framealpha=0.9)
    fig.suptitle(f"{title}\nband: {BAND}; k=0 is the exact preserved control "
                 f"(no band); models are not pooled", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return io_utils.sha256_file(path)


def primary_figures(curve: Sequence[Dict], out_dir: str) -> Dict[str, str]:
    out = {}
    for site in SITES:
        p = os.path.join(out_dir, "figures", f"PRIMARY_{site}.png")
        out[os.path.join("figures", f"PRIMARY_{site}.png")] = _figure(
            curve, site, list(battery.PRIMARY),
            f"Lesioning V2 - site {site} - PRIMARY endpoints (all k=0..15)", p)
    return out


def diagnostic_figures(curve: Sequence[Dict], out_dir: str) -> Dict[str, str]:
    """Route-isolated readouts. Explanatory only: they can illuminate a primary
    effect but never replace or rescue one."""
    out = {}
    for site in SITES:
        p = os.path.join(out_dir, "figures_diagnostic", f"DIAGNOSTIC_{site}.png")
        out[os.path.join("figures_diagnostic", f"DIAGNOSTIC_{site}.png")] = _figure(
            curve, site, list(battery.DIAGNOSTIC),
            f"Lesioning V2 - site {site} - DIAGNOSTIC route-isolated "
            f"(explanatory only)", p)
    return out
