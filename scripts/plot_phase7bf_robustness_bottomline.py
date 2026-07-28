#!/usr/bin/env python3
"""Phase 7C–7F robustness selection path — bottom-line figure.

Panel A: best deterministic train AR word errors per seed, organised by selection stage.
Panel B: Phase 7F H128 confirmatory trajectories, seeds 9–12 (log1p scale).

Inputs (all inside --bundle-dir):
  phase7c_multiseed_validation/_control/phase7c_all_candidate_seed.tsv
  phase7d_final_lr_schedule/_control/phase7d_candidate_arm_summary.tsv
  phase7e_prospective_validation/_control/phase7e_all_branches.tsv
  phase7f_b128_confirmatory/_control/phase7f_all_branches.tsv

Outputs (all inside --out-dir):
  phase7bf_robustness_bottomline.{png,pdf,svg}
  phase7bf_panel_a_data.tsv
  phase7bf_panel_b_data.tsv
  phase7bf_diagnostic_e200.tsv         ← best vs e200 comparison for B128 recipe

Usage:
  python scripts/plot_phase7bf_robustness_bottomline.py \\
      --bundle-dir outputs/plot_bundle_after_yair_20260727 \\
      --out-dir outputs/plot_bundle_after_yair_20260727/figures
"""
from __future__ import annotations
import argparse, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── palette (colorblind-friendly, consistent with Phase 7A scripts) ──────────
_C_B128  = "#009E73"   # green  — B128 selected candidate
_C_OTHER = "#aaaaaa"   # gray   — other 7C candidates
_C_LOWER = "#56B4E9"   # blue   — rejected 7D schedule arm
_SEED_COLS = {
    9:  "#E69F00",   # orange
    10: "#CC79A7",   # pink
    11: "#009E73",   # green  — stable zero
    12: "#0072B2",   # blue   — stable zero
}

# ── 7C candidate ordering and x-axis labels ───────────────────────────────────
# LR↓@eN = LR reduced from 1e-3 to 1e-4 at epoch N
# TF = teacher-forcing ratio (1.0 = full, 0.0 = free-run from start)
_CAND_ORDER = [
    "A_H064_tf1p0_sw100",
    "B_H128_tf1p0_sw100",
    "C_H256_tf1p0_sw100",
    "D_H064_tf0p0_sw120",
    "E_H256_tf1p0_sw120",
]
_CAND_LABEL = {
    "A_H064_tf1p0_sw100": "A\nH64 / TF1\nLR↓@e100",
    "B_H128_tf1p0_sw100": "B\nH128 / TF1\nLR↓@e100",
    "C_H256_tf1p0_sw100": "C\nH256 / TF1\nLR↓@e100",
    "D_H064_tf0p0_sw120": "D\nH64 / TF0\nLR↓@e120",
    "E_H256_tf1p0_sw120": "E\nH256 / TF1\nLR↓@e120",
}

# x-centre positions per group (gaps between phases produce visual separation)
_GROUP_X = {
    "A_H064_tf1p0_sw100": 0,
    "B_H128_tf1p0_sw100": 1,
    "C_H256_tf1p0_sw100": 2,
    "D_H064_tf0p0_sw120": 3,
    "E_H256_tf1p0_sw120": 4,
    "7D_cont":  6.5,
    "7D_low":   7.5,
    "7E_B128":  9.5,
    "7F_B128": 11.5,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_trajectory(s: str) -> tuple[list[int], list[int]]:
    """'100:43,105:11,...' → ([100,105,...], [43,11,...])"""
    epochs, errors = [], []
    for tok in str(s).strip().split(","):
        e, n = tok.split(":")
        epochs.append(int(e))
        errors.append(int(n))
    return epochs, errors


def _parse_seed_best(s: str) -> list[int]:
    """'s1:0,s2:0,s3:1,s4:1' → [0, 0, 1, 1]"""
    return [int(tok.split(":")[1]) for tok in str(s).strip().split(",")]


def _jitter(x_center: float, n: int, width: float = 0.28) -> list[float]:
    if n == 1:
        return [x_center]
    return list(np.linspace(x_center - width / 2, x_center + width / 2, n))


def _e200_from_traj(s: str) -> int | None:
    """Return train AR errors at epoch 200, or None if e200 not in trajectory."""
    for tok in str(s).strip().split(","):
        e, n = tok.split(":")
        if int(e) == 200:
            return int(n)
    return None

# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7C–7F robustness bottom-line figure")
    p.add_argument("--bundle-dir", required=True, dest="bundle_dir",
                   help="Path to plot_bundle_after_yair_20260727/")
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
        "7c": _load(bundle, "phase7c_multiseed_validation/_control/phase7c_all_candidate_seed.tsv"),
        "7d": _load(bundle, "phase7d_final_lr_schedule/_control/phase7d_candidate_arm_summary.tsv"),
        "7e": _load(bundle, "phase7e_prospective_validation/_control/phase7e_all_branches.tsv"),
        "7f": _load(bundle, "phase7f_b128_confirmatory/_control/phase7f_all_branches.tsv"),
    }

# ── validation ───────────────────────────────────────────────────────────────

def validate(d: dict[str, pd.DataFrame]) -> None:
    errs = []
    df7c = d["7c"]
    cands   = sorted(df7c["candidate_id"].unique())
    seeds7c = sorted(df7c["seed"].unique())
    if len(df7c) != 25:
        errs.append(f"7C: expected 25 rows, got {len(df7c)}")
    if len(cands) != 5:
        errs.append(f"7C: expected 5 candidates, got {len(cands)}")
    if set(seeds7c) != {0, 1, 2, 3, 4}:
        errs.append(f"7C: expected seeds 0–4, got {seeds7c}")

    df7d = d["7d"]
    b128_arms = sorted(df7d[df7d["candidate_id"] == "B_H128_tf1p0_sw100"]["arm"].unique())
    if "continue_1e-4" not in b128_arms:
        errs.append(f"7D: missing 'continue_1e-4'; found {b128_arms}")
    if "lower_to_1e-5" not in b128_arms:
        errs.append(f"7D: missing 'lower_to_1e-5'; found {b128_arms}")

    df7e = d["7e"]
    b128_7e = df7e[df7e["candidate_id"] == "B128_BENCHMARK"]
    if len(b128_7e) != 4:
        errs.append(f"7E: expected 4 B128 rows, got {len(b128_7e)}")
    if set(b128_7e["seed"].unique()) != {5, 6, 7, 8}:
        errs.append(f"7E: expected seeds 5–8")

    df7f = d["7f"]
    if len(df7f) != 4:
        errs.append(f"7F: expected 4 rows, got {len(df7f)}")
    if set(df7f["seed"].unique()) != {9, 10, 11, 12}:
        errs.append(f"7F: expected seeds 9–12")

    if errs:
        print("[validate] FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)

    n_le1  = int((df7f["best_train_errors_ar"] <= 1).sum())
    n_zero = int((df7f["best_train_errors_ar"] == 0).sum())
    n_stab = int(df7f["stable_zero"].sum()) if "stable_zero" in df7f.columns else "?"
    print(f"[validate] OK  — 7F: {n_le1}/4 ≤1 / {n_zero}/4 zero / {n_stab}/4 stable")

# ── Panel A data ─────────────────────────────────────────────────────────────

def build_panel_a_data(d: dict) -> pd.DataFrame:
    df7c, df7d, df7e, df7f = d["7c"], d["7d"], d["7e"], d["7f"]
    records = []

    for cand in _CAND_ORDER:
        sub = df7c[df7c["candidate_id"] == cand].sort_values("seed")
        for _, row in sub.iterrows():
            records.append(dict(
                phase="7C", group=cand,
                x_label=_CAND_LABEL[cand],
                seed=int(row["seed"]),
                best_errors=int(row["best_train_errors_ar"]),
                ever_zero=bool(row["ever_reached_zero_ar"]),
                is_b128=(cand == "B_H128_tf1p0_sw100"),
                is_selected_arm=True,
            ))

    for arm, group, is_sel in [
        ("continue_1e-4", "7D_cont", True),
        ("lower_to_1e-5", "7D_low",  False),
    ]:
        row7d = df7d[
            (df7d["candidate_id"] == "B_H128_tf1p0_sw100") &
            (df7d["arm"] == arm)
        ].iloc[0]
        lbl = ("B128 / 1e-4 cont.\n★ selected  (7D)" if is_sel
               else "B128 / 1e-5 lower\nrejected  (7D)")
        for i, e in enumerate(_parse_seed_best(row7d["seed_best_errors"]), 1):
            records.append(dict(
                phase="7D", group=group,
                x_label=lbl,
                seed=i,
                best_errors=int(e),
                ever_zero=(int(e) == 0),
                is_b128=True,
                is_selected_arm=is_sel,
            ))

    b128_7e = df7e[df7e["candidate_id"] == "B128_BENCHMARK"].sort_values("seed")
    for _, row in b128_7e.iterrows():
        records.append(dict(
            phase="7E", group="7E_B128",
            x_label="7E  prosp.\nB128 seeds 5–8",
            seed=int(row["seed"]),
            best_errors=int(row["best_train_errors_ar"]),
            ever_zero=bool(row["ever_reached_zero_ar"]),
            is_b128=True,
            is_selected_arm=True,
        ))

    for _, row in df7f.sort_values("seed").iterrows():
        records.append(dict(
            phase="7F", group="7F_B128",
            x_label="7F  conf.\nB128 seeds 9–12",
            seed=int(row["seed"]),
            best_errors=int(row["best_train_errors_ar"]),
            ever_zero=bool(row["ever_zero"]),
            is_b128=True,
            is_selected_arm=True,
        ))

    return pd.DataFrame(records)

# ── Panel B data ─────────────────────────────────────────────────────────────

def build_panel_b_data(df7f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df7f.sort_values("seed").iterrows():
        epochs, errors = _parse_trajectory(row["trajectory"])
        seed = int(row["seed"])
        for ep, er in zip(epochs, errors):
            rows.append(dict(
                seed=seed, epoch=ep,
                train_errors_ar=er,
                log1p_errors=float(np.log1p(er)),
                ever_zero=bool(row["ever_zero"]),
                stable_zero=bool(row["stable_zero"]),
                best_errors=int(row["best_train_errors_ar"]),
            ))
    return pd.DataFrame(rows)

# ── Diagnostic: best vs e200 for important B128 candidates ───────────────────

def build_diagnostic_tsv(d: dict) -> pd.DataFrame:
    """Compare best-observed vs endpoint-e200 errors for the B128 selection path.

    Retrospective only — does NOT change historical selection decisions.
    """
    df7c, df7d, df7e, df7f = d["7c"], d["7d"], d["7e"], d["7f"]
    rows = []

    # 7C: all 5 candidates (no trajectory column → e200 unavailable)
    for _, row in df7c.iterrows():
        cid = str(row["candidate_id"])
        rows.append(dict(
            phase="7C", group=cid,
            seed=int(row["seed"]),
            best_errors=int(row["best_train_errors_ar"]),
            e200_errors=None,
            e200_available=False,
            ever_zero=bool(row["ever_reached_zero_ar"]),
            stable_zero=None,
            note="no trajectory in 7C file",
        ))

    # 7D: B128 both arms (summary only → e200 unavailable)
    for arm in ["continue_1e-4", "lower_to_1e-5"]:
        r7d = df7d[(df7d["candidate_id"] == "B_H128_tf1p0_sw100") & (df7d["arm"] == arm)]
        if len(r7d) == 0:
            continue
        r7d = r7d.iloc[0]
        for i, e in enumerate(_parse_seed_best(r7d["seed_best_errors"]), 1):
            rows.append(dict(
                phase="7D", group=f"B128_{arm}",
                seed=i,
                best_errors=int(e),
                e200_errors=None,
                e200_available=False,
                ever_zero=(int(e) == 0),
                stable_zero=None,
                note="summary row only, no trajectory",
            ))

    # 7E: B128_BENCHMARK — extract e200 from trajectory
    b128_7e = df7e[df7e["candidate_id"] == "B128_BENCHMARK"]
    traj_col = next((c for c in ["trajectory_e100_e200", "trajectory"]
                     if c in b128_7e.columns), None)
    for _, row in b128_7e.iterrows():
        e200 = _e200_from_traj(row[traj_col]) if traj_col else None
        rows.append(dict(
            phase="7E", group="B128_BENCHMARK",
            seed=int(row["seed"]),
            best_errors=int(row["best_train_errors_ar"]),
            e200_errors=e200,
            e200_available=(e200 is not None),
            ever_zero=bool(row["ever_reached_zero_ar"]),
            stable_zero=None,
            note="",
        ))

    # 7F: confirmatory — extract e200 from trajectory
    traj_col_7f = "trajectory" if "trajectory" in df7f.columns else None
    for _, row in df7f.sort_values("seed").iterrows():
        e200 = _e200_from_traj(row[traj_col_7f]) if traj_col_7f else None
        stable = bool(row["stable_zero"]) if "stable_zero" in row.index else None
        rows.append(dict(
            phase="7F", group="B128_confirmatory",
            seed=int(row["seed"]),
            best_errors=int(row["best_train_errors_ar"]),
            e200_errors=e200,
            e200_available=(e200 is not None),
            ever_zero=bool(row["ever_zero"]),
            stable_zero=stable,
            note="",
        ))

    return pd.DataFrame(rows)


def _print_diagnostic_report(df_diag: pd.DataFrame) -> None:
    print("\n[diagnostic] B128 recipe — best-observed vs endpoint-e200 train AR errors")
    print("  (Retrospective only; historical selection is unchanged.)")
    for ph in ["7E", "7F"]:
        sub = df_diag[(df_diag["phase"] == ph) & df_diag["e200_available"]]
        if len(sub) == 0:
            continue
        print(f"\n  Phase {ph}:")
        for _, r in sub.iterrows():
            e200 = int(r["e200_errors"])
            best = int(r["best_errors"])
            relation = ("==" if e200 == best
                        else f">{best}" if e200 > best
                        else f"<{best} (unusually lower)")
            print(f"    seed {int(r['seed'])}: best={best}  e200={e200}  "
                  f"(e200 {relation}  ever_zero={r['ever_zero']}  "
                  f"stable_zero={r['stable_zero']})")
    # summary
    sub7f = df_diag[(df_diag["phase"] == "7F") & df_diag["e200_available"]]
    if len(sub7f) > 0:
        n_eq    = int((sub7f["e200_errors"] == sub7f["best_errors"]).sum())
        n_worse = int((sub7f["e200_errors"] > sub7f["best_errors"]).sum())
        n_le1_e200 = int((sub7f["e200_errors"] <= 1).sum())
        n_zero_e200 = int((sub7f["e200_errors"] == 0).sum())
        print(f"\n  7F summary ({len(sub7f)} seeds with e200 data):")
        print(f"    e200 == best  : {n_eq}/{len(sub7f)}")
        print(f"    e200 >  best  : {n_worse}/{len(sub7f)}  "
              "(seed improved after best, then regressed at e200)")
        print(f"    e200 ≤1 errors: {n_le1_e200}/{len(sub7f)}")
        print(f"    e200 = 0 exact: {n_zero_e200}/{len(sub7f)}")
        if n_le1_e200 == len(sub7f):
            print("  → B128 recipe remains competitive under e200 endpoint metric "
                  f"({n_le1_e200}/{len(sub7f)} seeds ≤1 at e200).")
        else:
            print(f"  → {n_le1_e200}/{len(sub7f)} seeds ≤1 at e200; "
                  "some seeds regressed between best and e200 — "
                  "stable-zero criterion is consequential.")

# ── Panel A drawing ──────────────────────────────────────────────────────────

def _phase_brackets(ax: plt.Axes) -> None:
    """Draw phase group labels + bracket lines above the axes using mixed transform."""
    trans = ax.get_xaxis_transform()   # x=data, y=axes fraction
    groups = [
        ("Phase 7C — multi-seed validation",  -0.65,  4.55, 2.0),
        ("7D — schedule",                       5.55,  8.20, 6.9),
        ("7E — prosp.",                         8.75, 10.25, 9.5),
        ("7F — conf.",                         10.75, 12.25, 11.5),
    ]
    y_line = 1.04
    y_txt  = 1.055
    for label, x0, x1, xmid in groups:
        ax.text(xmid, y_txt, label, ha="center", va="bottom",
                fontsize=7.5, color="#333333", fontweight="semibold",
                transform=trans, clip_on=False)
        ax.plot([x0, x1], [y_line, y_line], transform=trans,
                color="#999999", linewidth=0.9, solid_capstyle="butt", clip_on=False)
        for xv in (x0, x1):
            ax.plot([xv, xv], [y_line - 0.008, y_line], transform=trans,
                    color="#999999", linewidth=0.9, clip_on=False)


def _draw_panel_a(ax: plt.Axes, df_a: pd.DataFrame) -> None:
    ax.set_title(
        "A.  Selection path — best train AR word errors per seed\n"
        "     Each point = one training seed   |   LR↓@eN = LR switch (1e-3→1e-4) at epoch N",
        loc="left", fontsize=9, fontweight="bold", pad=8,
    )
    ax.set_ylabel("Best train AR word errors\n(0 = exact-match 1.000)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_facecolor("white")

    # reference bands
    ax.axhspan(-0.4, 0.45, color="#d4edda", alpha=0.40, zorder=0)
    ax.axhspan( 0.55, 1.45, color="#fff3cd", alpha=0.30, zorder=0)
    ax.axhline(0, color="#009E73", linewidth=0.9, linestyle="-", zorder=1, alpha=0.65)
    ax.axhline(1, color="#E69F00", linewidth=0.9, linestyle="--", zorder=1, alpha=0.55)

    # phase separator lines
    for xsep in [5.0, 8.5, 10.5]:
        ax.axvline(xsep, color="#cccccc", linewidth=1.0, linestyle=":", zorder=0)

    # scatter
    for grp in df_a["group"].unique():
        sub = df_a[df_a["group"] == grp].sort_values("seed")
        xc  = _GROUP_X[grp]
        xs  = _jitter(xc, len(sub))
        for xi, (_, row) in zip(xs, sub.iterrows()):
            is_b128 = bool(row["is_b128"])
            is_sel  = bool(row["is_selected_arm"])
            y = float(row["best_errors"])
            if is_b128 and is_sel:
                color = _C_B128;  ms = 9
            elif is_b128 and not is_sel:
                color = _C_LOWER; ms = 9
            else:
                color = _C_OTHER; ms = 8
            ax.plot(xi, y, "o", color=color, markersize=ms,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=0.6, zorder=3, alpha=0.92)

    # x-ticks
    tick_x, tick_l, seen = [], [], set()
    for grp in df_a["group"].unique():
        lbl = df_a[df_a["group"] == grp]["x_label"].iloc[0]
        if lbl not in seen:
            tick_x.append(_GROUP_X[grp])
            tick_l.append(lbl)
            seen.add(lbl)
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_l, fontsize=7.5, multialignment="center",
                       linespacing=1.35)

    ymax = float(df_a["best_errors"].max()) + 2.0
    ax.set_ylim(-0.9, max(ymax, 6))
    ax.set_xlim(-0.7, 12.6)

    _phase_brackets(ax)

    # legend (simplified)
    leg_handles = [
        mlines.Line2D([], [], marker="o", color=_C_B128,  markersize=9,
                      markeredgecolor="white", markeredgewidth=0.6,
                      linestyle="none", label="H128 / selected path (7C–7F)"),
        mlines.Line2D([], [], marker="o", color=_C_OTHER, markersize=8,
                      markeredgecolor="white", markeredgewidth=0.6,
                      linestyle="none", label="Other 7C candidates"),
        mlines.Line2D([], [], marker="o", color=_C_LOWER, markersize=9,
                      markeredgecolor="white", markeredgewidth=0.6,
                      linestyle="none", label="H128 / 1e-5 lower (7D rejected arm)"),
        mpatches.Patch(color="#d4edda", alpha=0.55, label="Exact zero band (0 errors)"),
        mpatches.Patch(color="#fff3cd", alpha=0.45, label="≤1 error band"),
    ]
    ax.legend(handles=leg_handles, fontsize=7.5, loc="upper right",
              framealpha=0.95, edgecolor="#cccccc")

# ── Panel B drawing ──────────────────────────────────────────────────────────

def _draw_panel_b(ax: plt.Axes, df_b: pd.DataFrame) -> None:
    ax.set_title("B.  H128 confirmatory trajectories — fresh seeds 9–12",
                 loc="left", fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("Training epoch", fontsize=9)
    ax.set_ylabel("Train full-route AR word errors\n(log1p scale; tick labels = raw counts)",
                  fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_facecolor("white")

    # log1p ticks
    all_errs = list(df_b["train_errors_ar"])
    max_err  = max(all_errs) if all_errs else 128
    tick_raw = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    tick_raw = [v for v in tick_raw if v <= max_err * 1.15]
    ax.set_yticks([np.log1p(v) for v in tick_raw])
    ax.set_yticklabels([str(v) for v in tick_raw])

    ymax = np.log1p(max_err) * 1.22
    ax.set_ylim(-ymax * 0.04, ymax)

    all_epochs = sorted(df_b["epoch"].unique())
    ax.set_xlim(all_epochs[0] - 3, all_epochs[-1] + 3)
    ax.set_xticks(all_epochs[::2])

    # zero-error reference line + label
    ax.axhline(0, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(all_epochs[-1] + 1, np.log1p(0) + ymax * 0.015,
            "0 errors = train exact-match 1.000",
            fontsize=6.5, color="#888888", ha="right", va="bottom")

    # trajectories
    for seed in sorted(df_b["seed"].unique()):
        sub    = df_b[df_b["seed"] == seed].sort_values("epoch")
        color  = _SEED_COLS.get(seed, "#888888")
        stable = bool(sub["stable_zero"].iloc[0])
        best_e = int(sub["best_errors"].iloc[0])
        lw     = 2.4 if stable else 1.4
        ls     = "-" if stable else "--"
        label  = (f"seed {seed} — best: {best_e}  [stable zero]" if stable
                  else f"seed {seed} — best: {best_e}")
        y_log  = [np.log1p(e) for e in sub["train_errors_ar"]]
        ax.plot(sub["epoch"], y_log, color=color, linewidth=lw,
                linestyle=ls, zorder=3, solid_capstyle="round", label=label)
        ax.plot(sub["epoch"], y_log, ".", color=color, markersize=4,
                linestyle="none", zorder=4)

    # result box
    df_summ  = df_b.drop_duplicates("seed")
    n_le1    = int((df_summ["best_errors"] <= 1).sum())
    n_zero   = int( df_summ["ever_zero"].sum())
    n_stable = int( df_summ["stable_zero"].sum())
    result_txt = (f"{n_le1}/4 reached ≤1 train AR word errors\n"
                  f"{n_zero}/4 reached exact zero (0 errors)\n"
                  f"{n_stable}/4 had stable zero")
    ax.text(0.97, 0.97, result_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="#009E73", alpha=0.95, lw=1.2))
    ax.text(0.97, 0.695, "stable zero = ≥2 consecutive evaluated\nzero-error checkpoints",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="#666666", style="italic")

    ax.legend(fontsize=7.5, loc="upper right",
              bbox_to_anchor=(0.96, 0.68), framealpha=0.92, edgecolor="#cccccc")

# ── figure assembly ───────────────────────────────────────────────────────────

def make_figure(d: dict, df_a: pd.DataFrame,
                df_b: pd.DataFrame, dpi: int) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(18, 8),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87,
                        bottom=0.24, wspace=0.30)

    fig.text(0.5, 0.97,
             "Lichtheim3 — From first ceiling to a robust training recipe",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.93,
             ("Phases 7C–7F: multi-seed validation → schedule selection → "
              "prospective test → confirmatory replication  |  "
              "H = 128, TF = 1.0, split_seed = 0"),
             ha="center", va="top", fontsize=8.5, color="#555555")

    _draw_panel_a(ax_a, df_a)
    _draw_panel_b(ax_b, df_b)

    fig.text(0.5, 0.02,
             ("Each point = one training seed.  "
              "split_seed = 0 fixed throughout; seeds vary training initialisation only, "
              "NOT the lexical split.  "
              "Solid trajectories (Panel B) = stable zero (≥2 consecutive zero-error "
              "evaluated checkpoints).  "
              "Zero at one checkpoint does not guarantee zero at e200."),
             ha="center", va="bottom", fontsize=7.5, color="#666666")
    return fig

# ── output ────────────────────────────────────────────────────────────────────

def save_outputs(fig: plt.Figure, df_a: pd.DataFrame, df_b: pd.DataFrame,
                 df_diag: pd.DataFrame, out_dir: str, dpi: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "phase7bf_robustness_bottomline")
    for ext, kw in [
        (".png", dict(dpi=dpi, bbox_inches="tight", facecolor="white")),
        (".pdf", dict(bbox_inches="tight", facecolor="white")),
        (".svg", dict(bbox_inches="tight", facecolor="white")),
    ]:
        p = stem + ext
        fig.savefig(p, **kw)
        print(f"[figure] -> {p}")

    for name, df in [
        ("phase7bf_panel_a_data.tsv", df_a),
        ("phase7bf_panel_b_data.tsv", df_b),
        ("phase7bf_diagnostic_e200.tsv", df_diag),
    ]:
        p = os.path.join(out_dir, name)
        df.to_csv(p, sep="\t", index=False)
        print(f"[table]  -> {p}")

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"\n[load]  bundle-dir = {args.bundle_dir}")
    d = load_data(args.bundle_dir)
    print("\n[validate]")
    validate(d)

    print("\n[build]")
    df_a    = build_panel_a_data(d)
    df_b    = build_panel_b_data(d["7f"])
    df_diag = build_diagnostic_tsv(d)
    print(f"  Panel A: {len(df_a)} rows")
    print(f"  Panel B: {len(df_b)} rows")
    print(f"  Diagnostic: {len(df_diag)} rows")

    _print_diagnostic_report(df_diag)

    print("\n[figure]")
    fig = make_figure(d, df_a, df_b, args.dpi)
    save_outputs(fig, df_a, df_b, df_diag, args.out_dir, args.dpi)
    if args.show:
        plt.show()
    plt.close(fig)

    print("\n=== OUTPUT CHECK ===")
    for fname in [
        "phase7bf_robustness_bottomline.png",
        "phase7bf_robustness_bottomline.pdf",
        "phase7bf_robustness_bottomline.svg",
        "phase7bf_panel_a_data.tsv",
        "phase7bf_panel_b_data.tsv",
        "phase7bf_diagnostic_e200.tsv",
    ]:
        fpath = os.path.join(args.out_dir, fname)
        ok = os.path.exists(fpath) and os.path.getsize(fpath) > 0
        print(f"  {'✓' if ok else '✗ MISSING'} {fpath}")


if __name__ == "__main__":
    main()
