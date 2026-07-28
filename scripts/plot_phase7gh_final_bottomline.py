#!/usr/bin/env python3
"""Phase 7G–7H final configuration — gate sensitivity, noise robustness, and fresh-seed confirmation.

Panel A: 2×3 gate sensitivity matrix (τ ∈ {0.5, 0.7} × α ∈ {2, 4, 8}).
Panel B: noise sensitivity across 5 training-noise conditions (BASELINE / WM05 / WM10 / V05 / V10).
Panel C: Phase 7H confirmatory trajectories, seeds 15–18 (log1p scale), with PASS result box.

Inputs (all inside --bundle-dir):
  phase7g2_gate_alpha/_control/phase7g2_matrix_summary.tsv
  phase7g2_gate_alpha/_control/phase7g2_matrix_branches.tsv
  phase7g3_noise/_control/phase7g3_noise_summary.tsv
  phase7g3_noise/_control/phase7g3_noise_branches.tsv
  phase7h_selected_config_confirmatory/_control/phase7h_confirmatory_branches.tsv
  phase7h_selected_config_confirmatory/_control/phase7h_confirmatory_summary.tsv

Outputs (all inside --out-dir):
  phase7gh_final_bottomline.{png,pdf,svg}
  phase7gh_panel_a_data.tsv
  phase7gh_panel_b_data.tsv
  phase7gh_panel_c_data.tsv

Usage:
  python scripts/plot_phase7gh_final_bottomline.py \\
      --bundle-dir outputs/plot_bundle_after_yair_20260727 \\
      --out-dir outputs/plot_bundle_after_yair_20260727/figures
"""
from __future__ import annotations
import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── palette ───────────────────────────────────────────────────────────────────
_C_SEL   = "#009E73"    # green — selected cell / PASS / BASELINE
_C_DARK  = "#005a32"    # dark green — stable zero bars
_C_LIGHT = "#b7e4c7"    # light green — ever-zero bars
_SEED_COLS = {15: "#009E73", 16: "#E69F00", 17: "#0072B2", 18: "#D55E00"}

# ── noise condition ordering and labels ───────────────────────────────────────
_NOISE_ORDER = ["BASELINE", "WM05", "WM10", "V05", "V10"]
_NOISE_LABEL = {
    "BASELINE": "Baseline\n(no noise)",
    "WM05":     "WM noise\n0.05",
    "WM10":     "WM noise\n0.10",
    "V05":      "Ventral\nnoise 0.05",
    "V10":      "Ventral\nnoise 0.10",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_trajectory(s: str) -> tuple[list[int], list[int]]:
    epochs, errors = [], []
    for tok in str(s).strip().split(","):
        e, n = tok.split(":")
        epochs.append(int(e))
        errors.append(int(n))
    return epochs, errors

# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7G–7H gate/noise/confirmatory figure")
    p.add_argument("--bundle-dir", required=True, dest="bundle_dir")
    p.add_argument("--out-dir",    required=True, dest="out_dir")
    p.add_argument("--dpi",        type=int, default=300)
    p.add_argument("--show",       action="store_true")
    return p.parse_args()

# ── data loading ─────────────────────────────────────────────────────────────

def _load(bundle: str, rel: str) -> pd.DataFrame:
    path = os.path.join(bundle, rel)
    print(f"  [load] {path}")
    if not os.path.exists(path):
        print(f"ERROR: not found: {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, sep="\t")


def load_data(bundle: str) -> dict[str, pd.DataFrame]:
    return {
        "7g2_summ":   _load(bundle, "phase7g2_gate_alpha/_control/phase7g2_matrix_summary.tsv"),
        "7g2_branch": _load(bundle, "phase7g2_gate_alpha/_control/phase7g2_matrix_branches.tsv"),
        "7g3_summ":   _load(bundle, "phase7g3_noise/_control/phase7g3_noise_summary.tsv"),
        "7g3_branch": _load(bundle, "phase7g3_noise/_control/phase7g3_noise_branches.tsv"),
        "7h_branch":  _load(bundle,
            "phase7h_selected_config_confirmatory/_control/phase7h_confirmatory_branches.tsv"),
        "7h_summ":    _load(bundle,
            "phase7h_selected_config_confirmatory/_control/phase7h_confirmatory_summary.tsv"),
    }

# ── validation ───────────────────────────────────────────────────────────────

def validate(d: dict[str, pd.DataFrame]) -> None:
    errs = []
    g2s, g2b = d["7g2_summ"], d["7g2_branch"]
    g3s, g3b = d["7g3_summ"], d["7g3_branch"]
    h_b, h_s = d["7h_branch"], d["7h_summ"]

    if len(g2s) != 6:
        errs.append(f"7G2 summary: expected 6 rows (2τ×3α), got {len(g2s)}")
    if len(g2b) != 12:
        errs.append(f"7G2 branches: expected 12 rows, got {len(g2b)}")
    sel = g2s[(g2s["gate_threshold"] == 0.7) & (g2s["gate_alpha"] == 2.0)]
    if len(sel) == 0:
        errs.append("7G2: missing (τ=0.7, α=2.0) row")
    elif int(sel["stable_zero"].iloc[0]) < 2:
        errs.append(f"7G2: (τ=0.7, α=2.0) stable_zero expected ≥2, got {sel['stable_zero'].iloc[0]}")

    if len(g3s) != 5:
        errs.append(f"7G3 summary: expected 5 rows, got {len(g3s)}")
    if len(g3b) != 10:
        errs.append(f"7G3 branches: expected 10 rows, got {len(g3b)}")

    if len(h_b) != 4:
        errs.append(f"7H branches: expected 4 rows, got {len(h_b)}")
    if set(h_b["seed"].unique()) != {15, 16, 17, 18}:
        errs.append("7H: expected seeds 15–18")

    if errs:
        print("[validate] FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)

    sel_7g2 = g2s[(g2s["gate_threshold"] == 0.7) & (g2s["gate_alpha"] == 2.0)].iloc[0]
    n_le1   = int((h_b["best_errors"] <= 1).sum()) if "best_errors" in h_b.columns else "?"
    n_zero  = int(h_b["ever_zero"].sum())
    n_stab  = int(h_b["stable_zero"].sum()) if "stable_zero" in h_b.columns else "?"
    print("[validate] OK")
    print(f"  7G2 selected (τ=0.7, α=2.0): "
          f"ever_zero={sel_7g2['ever_zero']}, stable_zero={sel_7g2['stable_zero']}")
    print(f"  7H: {n_le1}/4 ≤1  /  {n_zero}/4 zero  /  {n_stab}/4 stable")

# ── Panel A — gate sensitivity matrix ────────────────────────────────────────

def _draw_panel_a(ax: plt.Axes, df_summ: pd.DataFrame) -> None:
    ax.set_title("A.  Gate sensitivity — τ × α sweep (2 seeds per cell)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("gate_alpha (α)  —  higher α → sharper route switch", fontsize=8.5)
    ax.set_ylabel("gate_threshold (τ)  —  higher τ → more dorsal/WM",   fontsize=8.5)

    alphas     = sorted(df_summ["gate_alpha"].unique())
    thresholds = sorted(df_summ["gate_threshold"].unique(), reverse=True)
    n_a, n_t   = len(alphas), len(thresholds)
    n_seeds    = 2

    ever_grid   = np.full((n_t, n_a), np.nan)
    stable_grid = np.full((n_t, n_a), np.nan)
    for _, row in df_summ.iterrows():
        ti = thresholds.index(row["gate_threshold"])
        ai = alphas.index(row["gate_alpha"])
        ever_grid[ti, ai]   = int(row["ever_zero"])
        stable_grid[ti, ai] = int(row["stable_zero"])

    def _cell_color(ez, sz):
        if np.isnan(ez):
            return "#dddddd"
        if sz == n_seeds:
            return _C_LIGHT       # both stable
        if ez == n_seeds:
            return "#fff3cd"      # both ever-zero, not stable
        if ez > 0:
            return "#ffe5cc"      # partial
        return "#f9c0b0"          # none

    for ti, tau in enumerate(thresholds):
        for ai, alpha in enumerate(alphas):
            ez = ever_grid[ti, ai]
            sz = stable_grid[ti, ai]
            fc = _cell_color(ez, sz)
            ax.add_patch(mpatches.FancyBboxPatch(
                (ai - 0.45, ti - 0.42), 0.90, 0.84,
                boxstyle="round,pad=0.04", linewidth=1.2,
                edgecolor="#888888", facecolor=fc, zorder=2
            ))

            is_sel = (tau == 0.7 and alpha == 2.0)
            if is_sel:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (ai - 0.46, ti - 0.43), 0.92, 0.86,
                    boxstyle="round,pad=0.04", linewidth=2.8,
                    edgecolor=_C_SEL, facecolor="none", zorder=4
                ))

            txt = "\n".join(filter(None, [
                f"ever-0: {int(ez)}/{n_seeds}" if not np.isnan(ez) else "—",
                f"stable: {int(sz)}/{n_seeds}" if not np.isnan(sz) else "—",
                "★ selected" if is_sel else "",
            ]))
            ax.text(ai, ti, txt, ha="center", va="center",
                    fontsize=8, fontweight=("bold" if is_sel else "normal"),
                    color=("#004040" if is_sel else "#333333"), zorder=5)

    ax.set_xlim(-0.6, n_a - 0.4)
    ax.set_ylim(-0.6, n_t - 0.4)
    ax.set_xticks(range(n_a))
    ax.set_xticklabels([str(a) for a in alphas], fontsize=9)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([str(t) for t in thresholds], fontsize=9)
    ax.tick_params(length=0)
    ax.set_facecolor("#f8f8f8")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # gate equation — placed just above the axes, inside the figure header gap
    gate_txt = ("Gate equation:  g = σ(α · (cᴸᵀᴹ − τ))\n"
                "g → 1: ventral/LTM route   |   g → 0: dorsal/WM route")
    ax.text(0.02, 1.04, gate_txt,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color="#333333", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f8ff", ec="#99aacc",
                      alpha=0.92, lw=0.8),
            clip_on=False)

    # color legend
    leg_patches = [
        mpatches.Patch(color=_C_LIGHT,   label="2/2 stable zero  ← criterion"),
        mpatches.Patch(color="#fff3cd", label="2/2 ever-zero, not stable"),
        mpatches.Patch(color="#ffe5cc", label="1/2 ever-zero"),
        mpatches.Patch(color="#f9c0b0", label="0/2 ever-zero"),
    ]
    ax.legend(handles=leg_patches, fontsize=7.5, loc="lower right",
              framealpha=0.92, edgecolor="#cccccc")

# ── Panel B — noise sensitivity ───────────────────────────────────────────────

def _draw_panel_b(ax: plt.Axes, df_summ: pd.DataFrame) -> None:
    ax.set_title("B.  Training-noise sensitivity (seeds 13–14, gate τ=0.7 α=2)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Training-noise condition\n(noise applied during training only; "
                  "evaluation always deterministic)", fontsize=8)
    ax.set_ylabel("Count (out of 2 seeds)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_facecolor("white")
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)

    n_seeds = 2
    x_pos   = np.arange(len(_NOISE_ORDER))
    bar_w   = 0.32

    for xi, cond in enumerate(_NOISE_ORDER):
        row = df_summ[df_summ["condition"] == cond]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        ez  = int(row["ever_zero"])
        sz  = int(row["stable_zero"])
        is_sel  = (cond == "BASELINE")
        edge_c  = _C_SEL if is_sel else "#888888"
        edge_lw = 2.4   if is_sel else 1.0

        # Left bar — ever reached zero (light green)
        ax.bar(xi - bar_w / 2, ez, bar_w,
               color=_C_LIGHT, edgecolor=edge_c, linewidth=edge_lw, zorder=3,
               label="ever reached zero" if xi == 0 else None)
        # Right bar — stable zero (dark green)
        ax.bar(xi + bar_w / 2, sz, bar_w,
               color=_C_DARK,  edgecolor=edge_c, linewidth=edge_lw, zorder=3,
               label="stable zero (≥2 consec. zero-error checkpoints)" if xi == 0 else None)

        # count labels
        ax.text(xi - bar_w / 2, ez + 0.05, f"{ez}/{n_seeds}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.text(xi + bar_w / 2, sz + 0.05, f"{sz}/{n_seeds}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

        if is_sel:
            ax.text(xi, -0.38, "★ selected\n(baseline)", ha="center", va="top",
                    fontsize=7, color=_C_SEL, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([_NOISE_LABEL[c] for c in _NOISE_ORDER],
                       fontsize=8, multialignment="center")
    ax.set_ylim(-0.65, n_seeds + 0.6)
    ax.set_yticks([0, 1, 2])
    ax.axhline(n_seeds, color="#aaaaaa", linewidth=0.7, linestyle=":")

    ax.legend(fontsize=7.5, loc="upper right",
              framealpha=0.92, edgecolor="#cccccc")

# ── Panel C — 7H confirmatory trajectories ───────────────────────────────────

def _draw_panel_c(ax: plt.Axes, df_branch: pd.DataFrame,
                  df_summ: pd.DataFrame) -> None:
    ax.set_title("C.  Final confirmatory trajectories — fresh seeds 15–18",
                 loc="left", fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Training epoch", fontsize=9)
    ax.set_ylabel("Train full-route AR word errors\n"
                  "(log1p scale; tick labels = raw counts)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_facecolor("white")

    # collect trajectories
    all_errs  = []
    traj_data = {}
    traj_col  = next((c for c in ["trajectory", "trajectory_e100_e200"]
                      if c in df_branch.columns), None)
    for _, row in df_branch.iterrows():
        if traj_col is None:
            continue
        seed = int(row["seed"])
        epochs, errors = _parse_trajectory(row[traj_col])
        traj_data[seed] = (epochs, errors, row)
        all_errs.extend(errors)

    max_err = max(all_errs) if all_errs else 64

    # log1p ticks
    tick_raw = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    tick_raw = [v for v in tick_raw if v <= max_err * 1.15]
    ax.set_yticks([np.log1p(v) for v in tick_raw])
    ax.set_yticklabels([str(v) for v in tick_raw])

    ymax = np.log1p(max_err) * 1.22
    ax.set_ylim(-ymax * 0.04, ymax)

    all_epochs = sorted({ep for (eps, _, _) in traj_data.values() for ep in eps})
    ax.set_xlim(all_epochs[0] - 3, all_epochs[-1] + 3)
    ax.set_xticks(all_epochs[::2])

    # zero-error reference line
    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(all_epochs[-1] + 1, np.log1p(0) + ymax * 0.015,
            "0 errors = train exact-match 1.000",
            fontsize=6.5, color="#888888", ha="right", va="bottom")

    # trajectories
    for seed, (epochs, errors, row) in sorted(traj_data.items()):
        color  = _SEED_COLS.get(seed, "#888888")
        stable = bool(row["stable_zero"]) if "stable_zero" in row.index else False
        lw     = 2.4 if stable else 1.4
        ls     = "-" if stable else "--"
        best_e = int(row["best_errors"]) if "best_errors" in row.index else "?"
        label  = (f"seed {seed}  —  best observed errors: {best_e}  [stable zero]"
                  if stable else
                  f"seed {seed}  —  best observed errors: {best_e}")
        y_log  = [np.log1p(e) for e in errors]
        ax.plot(epochs, y_log, color=color, linewidth=lw, linestyle=ls,
                zorder=3, solid_capstyle="round", label=label)
        ax.plot(epochs, y_log, ".", color=color, markersize=4,
                linestyle="none", zorder=4)

    # "best = minimum" note
    ax.text(0.01, 0.01,
            "best = minimum train AR word-error count\n"
            "across evaluated checkpoints e100–e200",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color="#777777", style="italic")

    # PASS result box (derived from 7H summary TSV)
    def _summ_result(criterion: str) -> tuple[str, str]:
        r = df_summ[df_summ["criterion"] == criterion]
        if len(r) == 0:
            return "?", "?"
        return str(r.iloc[0].get("observed", "?")), str(r.iloc[0].get("result", "?"))

    obs_le1,  res_le1  = _summ_result("primary_reach_le1")
    obs_zero, res_zero = _summ_result("secondary_ever_zero")
    obs_stab, res_stab = _summ_result("secondary_stable_zero")
    obs_all,  res_all  = _summ_result("overall")

    _chk = lambda r: "✓" if "PASS" in str(r).upper() else "✗"
    result_lines = [
        f"{_chk(res_le1)} Primary: {obs_le1} reached ≤1 train AR word errors",
        f"{_chk(res_zero)} Secondary: {obs_zero} reached exact zero",
        f"{_chk(res_stab)} Secondary: {obs_stab} had stable zero",
        ("→ ALL CRITERIA PASSED" if "PASS" in str(res_all).upper()
         else f"→ RESULT: {res_all}"),
    ]
    ax.text(0.97, 0.97, "\n".join(result_lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=_C_SEL, alpha=0.95, lw=1.8))

    ax.legend(fontsize=7.5, loc="upper right",
              bbox_to_anchor=(0.97, 0.72), framealpha=0.92, edgecolor="#cccccc")

# ── export tables ─────────────────────────────────────────────────────────────

def build_panel_c_data(df_branch: pd.DataFrame) -> pd.DataFrame:
    traj_col = next((c for c in ["trajectory", "trajectory_e100_e200"]
                     if c in df_branch.columns), None)
    if traj_col is None:
        return df_branch.copy()
    rows = []
    for _, row in df_branch.iterrows():
        seed = int(row["seed"])
        epochs, errors = _parse_trajectory(row[traj_col])
        for ep, er in zip(epochs, errors):
            rows.append(dict(
                seed=seed, epoch=ep,
                train_errors_ar=er,
                log1p_errors=float(np.log1p(er)),
                ever_zero=bool(row["ever_zero"]),
                stable_zero=bool(row.get("stable_zero", False)),
                best_errors=int(row.get("best_errors", -1)),
            ))
    return pd.DataFrame(rows)

# ── figure assembly ───────────────────────────────────────────────────────────

def make_figure(d: dict, dpi: int) -> plt.Figure:
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        1, 3, figure=fig,
        left=0.05, right=0.98, top=0.85, bottom=0.18,
        wspace=0.35,
        width_ratios=[0.85, 0.95, 1.25],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    fig.text(0.5, 0.97,
             "Lichtheim3 — Gate/noise sensitivity and fresh-seed confirmation",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.92,
             ("Phase 7G2: gate parameter sweep  |  "
              "Phase 7G3: training-noise robustness  |  "
              "Phase 7H: final config confirmatory  |  "
              "H = 128, TF = 1.0, LR = 1e-3 → 1e-4 at e100, split_seed = 0"),
             ha="center", va="top", fontsize=8.5, color="#555555")

    _draw_panel_a(ax_a, d["7g2_summ"])
    _draw_panel_b(ax_b, d["7g3_summ"])
    _draw_panel_c(ax_c, d["7h_branch"], d["7h_summ"])

    fig.text(0.5, 0.025,
             ("Fresh seeds test training/initialization stochasticity on a fixed lexical split "
              "(split_seed = 0); they do NOT probe lexical-split robustness.  "
              "Panel B noise is applied during training only; all evaluations are deterministic.  "
              "Solid trajectories (Panel C) = stable zero (≥2 consecutive zero-error checkpoints).  "
              "Zero at one checkpoint does not guarantee zero at e200."),
             ha="center", va="bottom", fontsize=7.5, color="#666666")
    return fig

# ── output ────────────────────────────────────────────────────────────────────

def save_outputs(fig: plt.Figure, d: dict, out_dir: str, dpi: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "phase7gh_final_bottomline")
    for ext, kw in [
        (".png", dict(dpi=dpi, bbox_inches="tight", facecolor="white")),
        (".pdf", dict(bbox_inches="tight", facecolor="white")),
        (".svg", dict(bbox_inches="tight", facecolor="white")),
    ]:
        p = stem + ext
        fig.savefig(p, **kw)
        print(f"[figure] -> {p}")

    for tag, df in [
        ("panel_a", d["7g2_summ"].copy()),
        ("panel_b", d["7g3_summ"].copy()),
        ("panel_c", build_panel_c_data(d["7h_branch"])),
    ]:
        p = os.path.join(out_dir, f"phase7gh_{tag}_data.tsv")
        df.to_csv(p, sep="\t", index=False)
        print(f"[table]  -> {p}")

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"\n[load]  bundle-dir = {args.bundle_dir}")
    d = load_data(args.bundle_dir)
    print("\n[validate]")
    validate(d)
    print("\n[figure]")
    fig = make_figure(d, args.dpi)
    save_outputs(fig, d, args.out_dir, args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    print("\n=== OUTPUT CHECK ===")
    for fname in [
        "phase7gh_final_bottomline.png",
        "phase7gh_final_bottomline.pdf",
        "phase7gh_final_bottomline.svg",
        "phase7gh_panel_a_data.tsv",
        "phase7gh_panel_b_data.tsv",
        "phase7gh_panel_c_data.tsv",
    ]:
        fpath = os.path.join(args.out_dir, fname)
        ok = os.path.exists(fpath) and os.path.getsize(fpath) > 0
        print(f"  {'✓' if ok else '✗ MISSING'} {fpath}")


if __name__ == "__main__":
    main()
