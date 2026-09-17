#!/usr/bin/env python3
"""Directional-dose results package (READ-ONLY over the frozen outputs of the single --execute run).

Inputs:  summary_metrics_directional_dose.json, item_level_directional_dose.tsv (frozen fields only).
Outputs: figure_source_data/*.tsv|json, figures/*.png.  Every plotted value is written to
figure_source_data first and the figure is drawn by re-reading that file.  No model, no decode,
no fitted threshold.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSD, FIG = os.path.join(HERE, "figure_source_data"), os.path.join(HERE, "figures")
os.makedirs(FSD, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
STATES = ["W3_SRC", "W3_REP", "W4_SRC", "W4_REP"]
AK = ["a025", "a050", "a075", "a100"]
AV = {"a025": 0.25, "a050": 0.50, "a075": 0.75, "a100": 1.00}
CONVS = ["freear", "canonical"]
SCOL = {"W3_SRC": "#2a78d6", "W3_REP": "#eb6834", "W4_SRC": "#1baf7a", "W4_REP": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NUM = {}

S = json.load(open(os.path.join(HERE, "summary_metrics_directional_dose.json")))
R = S["results"]


def tsv(df, name):
    df.to_csv(os.path.join(FSD, name), sep="\t", index=False, lineterminator="\n", float_format="%.10g")


def read(name):
    return pd.read_csv(os.path.join(FSD, name), sep="\t")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- dose sequences (all strata)
rows = []
for sid in STATES:
    for c in CONVS:
        for st, blk in R[sid][c].items():
            ic = blk["immutable_controls"]
            n = ic["denominator"]
            rows.append({"state_id": sid, "convention": c, "stratum": st, "alpha": 0.0, "source": "immutable_S0",
                         "denominator": n, "exact_count": ic["S0_exact_count"],
                         "exact_proportion": ic["S0_exact_count"] / n if n else None,
                         "WRONG_TO_CORRECT": 0, "CORRECT_TO_WRONG": 0, "previous_S1_rescues": None,
                         "previous_S1_rescues_recovered": None, "recovery_fraction": None, "new_regressions": 0})
            for a in AK:
                cell = blk[a]
                rows.append({"state_id": sid, "convention": c, "stratum": st, "alpha": AV[a], "source": "directional_dose",
                             **{k: cell[k] for k in ("denominator", "exact_count", "exact_proportion", "WRONG_TO_CORRECT",
                                                     "CORRECT_TO_WRONG", "previous_S1_rescues",
                                                     "previous_S1_rescues_recovered", "recovery_fraction",
                                                     "new_regressions")}})
            for k in ("S1", "S3"):
                rows.append({"state_id": sid, "convention": c, "stratum": st, "alpha": None, "source": f"immutable_{k}",
                             "denominator": n, "exact_count": ic[f"{k}_exact_count"],
                             "exact_proportion": ic[f"{k}_exact_count"] / n if n else None})
dose = pd.DataFrame(rows)
tsv(dose, "dose_sequences_all_strata.tsv")

# ---------------------------------------------------------------- Figure 1: exact vs alpha (+S1, S3)
f1 = dose[(dose.stratum == "ALL_REPETITION_ITEMS")].copy()
tsv(f1, "fig1_exact_vs_alpha_all_items.tsv")
src = read("fig1_exact_vs_alpha_all_items.tsv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for k, c in enumerate(CONVS):
    ax = axes[k]
    for sid in STATES:
        d = src[(src.state_id == sid) & (src.convention == c) & src.source.isin(["immutable_S0", "directional_dose"])]
        d = d.sort_values("alpha")
        ax.plot(d.alpha, d.exact_proportion * 100, marker="o", color=SCOL[sid], linewidth=2, markersize=6, label=sid)
        s1 = src[(src.state_id == sid) & (src.convention == c) & (src.source == "immutable_S1")].exact_proportion.iloc[0]
        s3 = src[(src.state_id == sid) & (src.convention == c) & (src.source == "immutable_S3")].exact_proportion.iloc[0]
        ax.scatter([1.12], [s1 * 100], marker="*", s=90, color=SCOL[sid], zorder=3)
        ax.scatter([-0.12], [s3 * 100], marker="s", s=36, color=SCOL[sid], zorder=3)
    ax.set_xticks([-0.12, 0, 0.25, 0.5, 0.75, 1.0, 1.12], ["S3", "0\n(S0)", ".25", ".50", ".75", "1.00", "S1"], fontsize=8)
    ax.set_xlabel("α (native norm preserved); S3 / S1 = immutable controls", fontsize=8, color=INK)
    ax.set_title("GENUINE FREE-AR (primary)" if c == "freear" else "CANONICAL FORCED-LENGTH (secondary)",
                 fontsize=9, color=INK)
    style(ax)
axes[0].set_ylabel("isolated ventral exact (% of 29,571)", fontsize=8, color=INK)
axes[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.suptitle("Figure 1 — Exact repetition vs directional dose α (all items)", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig1_exact_vs_alpha.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: previous S1 rescue recovery
f2 = dose[(dose.stratum == "PREV_S1_RESCUES") & (dose.source == "directional_dose")][
    ["state_id", "convention", "alpha", "previous_S1_rescues", "previous_S1_rescues_recovered", "recovery_fraction"]]
tsv(f2, "fig2_prev_S1_rescue_recovery.tsv")
src = read("fig2_prev_S1_rescue_recovery.tsv")
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for sid in STATES:
    d = src[(src.state_id == sid) & (src.convention == "freear")].sort_values("alpha")
    x = [0.0] + list(d.alpha)
    y = [0.0] + list(d.recovery_fraction * 100)
    ax.plot(x, y, marker="o", color=SCOL[sid], linewidth=2, markersize=6, label=sid)
ax.axhline(100, color=MUTED, linewidth=1, linestyle="--")
ax.text(0.02, 101.5, "S1 (raw prototype) = 100% by definition", fontsize=7, color=MUTED)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylim(0, 108)
ax.set_xlabel("α", fontsize=8, color=INK)
ax.set_ylabel("previous S1 rescues recovered (%)", fontsize=8, color=INK)
ax.legend(fontsize=7, frameon=False)
style(ax)
fig.suptitle("Figure 2 — Recovery of previous S1 rescues vs α (free-AR)", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig2_prev_S1_recovery.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: W->C and C->W vs alpha
f3 = dose[(dose.stratum == "ALL_REPETITION_ITEMS") & (dose.source == "directional_dose")][
    ["state_id", "convention", "alpha", "WRONG_TO_CORRECT", "CORRECT_TO_WRONG"]]
tsv(f3, "fig3_transitions_vs_alpha.tsv")
src = read("fig3_transitions_vs_alpha.tsv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for k, col in enumerate(["WRONG_TO_CORRECT", "CORRECT_TO_WRONG"]):
    ax = axes[k]
    for sid in STATES:
        d = src[(src.state_id == sid) & (src.convention == "freear")].sort_values("alpha")
        ax.plot([0.0] + list(d.alpha), [0] + list(d[col]), marker="o", color=SCOL[sid], linewidth=2, markersize=6, label=sid)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("α", fontsize=8, color=INK)
    ax.set_title("S0 WRONG → CORRECT (rescues)" if k == 0 else "S0 CORRECT → WRONG (regressions)", fontsize=9, color=INK)
    style(ax)
axes[0].set_ylabel("items (of 29,571), free-AR", fontsize=8, color=INK)
axes[0].legend(fontsize=7, frameon=False)
fig.suptitle("Figure 3 — Rescues and regressions vs α", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig3_transitions_vs_alpha.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Figure 4: C_CORRECT_AND_NATIVE_LTM_WRONG
f4 = dose[(dose.stratum == "C_CORRECT_AND_NATIVE_LTM_WRONG") & dose.source.isin(
    ["immutable_S0", "directional_dose", "immutable_S1", "immutable_S3"])]
tsv(f4, "fig4_c_correct_native_wrong_dose.tsv")
src = read("fig4_c_correct_native_wrong_dose.tsv")
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for sid in STATES:
    d = src[(src.state_id == sid) & (src.convention == "freear") & src.source.isin(["immutable_S0", "directional_dose"])].sort_values("alpha")
    ax.plot(d.alpha, d.exact_proportion * 100, marker="o", color=SCOL[sid], linewidth=2, markersize=6, label=sid)
    s1 = src[(src.state_id == sid) & (src.convention == "freear") & (src.source == "immutable_S1")].exact_proportion.iloc[0]
    s3 = src[(src.state_id == sid) & (src.convention == "freear") & (src.source == "immutable_S3")].exact_proportion.iloc[0]
    ax.scatter([1.12], [s1 * 100], marker="*", s=90, color=SCOL[sid], zorder=3)
    ax.scatter([-0.12], [s3 * 100], marker="s", s=36, color=SCOL[sid], zorder=3)
ax.set_xticks([-0.12, 0, 0.25, 0.5, 0.75, 1.0, 1.12], ["S3", "0\n(S0)", ".25", ".50", ".75", "1.00", "S1"], fontsize=8)
ax.set_ylim(-3, 105)
ax.set_ylabel("exact within stratum (%), free-AR", fontsize=8, color=INK)
ax.legend(fontsize=7, frameon=False, loc="upper center")
style(ax)
fig.suptitle("Figure 4 — C_CORRECT_AND_NATIVE_LTM_WRONG: dose response", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig4_c_correct_native_wrong.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Figure 5: paired SRC vs REP
prow = []
for w in ("W3", "W4"):
    for c in CONVS:
        for a in AK:
            p = S["paired"][w][c][a]
            prow.append({"witness": w, "convention": c, "alpha": AV[a],
                         "SRC_exact": p["SRC"]["exact"], "REP_exact": p["REP"]["exact"],
                         "SRC_rescue_fraction": p["SRC"]["rescue_fraction"], "REP_rescue_fraction": p["REP"]["rescue_fraction"],
                         "SRC_regressions": p["SRC"]["regressions"], "REP_regressions": p["REP"]["regressions"],
                         "rescue_set_jaccard": p["rescue_set_jaccard"],
                         "prev_S1_recovery_fraction_REP_minus_SRC": p["prev_S1_recovery_fraction_REP_minus_SRC"],
                         **{f"SRC_to_REP_{k}": v for k, v in p["item_transitions_SRC_to_REP"].items()}})
tsv(pd.DataFrame(prow), "fig5_paired_src_rep.tsv")
src = read("fig5_paired_src_rep.tsv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for k, w in enumerate(("W3", "W4")):
    ax = axes[k]
    d = src[(src.witness == w) & (src.convention == "freear")].sort_values("alpha")
    for col, lab, colr, ls in (("SRC_rescue_fraction", "rescue fraction SOURCE", "#2a78d6", "-"),
                               ("REP_rescue_fraction", "rescue fraction POST_REPAIR", "#eb6834", "--")):
        ax.plot([0] + list(d.alpha), [0] + list(d[col] * 100), marker="o", color=colr, linestyle=ls, linewidth=2, label=lab)
    for i, row in d.iterrows():
        ax.text(row.alpha, max(row.SRC_rescue_fraction, row.REP_rescue_fraction) * 100 + 3,
                f"J={row.rescue_set_jaccard:.2f}", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylim(0, 90)
    ax.set_xlabel("α", fontsize=8, color=INK)
    ax.set_ylabel("native failures rescued (%), free-AR", fontsize=8, color=INK)
    ax.set_title(f"{w}: SOURCE vs POST_REPAIR (J = rescue-set Jaccard)", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    style(ax)
fig.suptitle("Figure 5 — Paired SOURCE vs POST_REPAIR dose response", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig5_paired_src_rep.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Figure 6: alpha=1 vs S1 vs S3
frow = []
for sid in STATES:
    for c in CONVS:
        x = S["alpha1_factorization"][sid][c]
        frow.append({"state_id": sid, "convention": c, "n": x["n"],
                     "S0_exact": x["exact"]["S0"], "S3_exact": x["exact"]["S3"], "alpha1_exact": x["exact"]["alpha1"],
                     "S1_exact": x["exact"]["S1"],
                     "S3_rescues": x["S3_rescues"], "alpha1_rescues": x["alpha1_rescues"], "S1_rescues": x["S1_rescues"],
                     "S3_regressions": x["S3_regressions"], "alpha1_regressions": x["alpha1_regressions"],
                     "S1_regressions": x["S1_regressions"],
                     "alpha1_rescues_also_S1_rescues": x["alpha1_rescues_also_S1_rescues"],
                     "S1_rescues_also_alpha1_rescues_fraction": x["S1_rescues_also_alpha1_rescues_fraction"],
                     "alpha1_vs_S1_item_disagreements": x["alpha1_vs_S1_item_disagreements"],
                     "alpha1_correct_S1_wrong": x["alpha1_correct_S1_wrong"],
                     "alpha1_wrong_S1_correct": x["alpha1_wrong_S1_correct"],
                     "alpha1_vs_S3_item_disagreements": x["alpha1_vs_S3_item_disagreements"],
                     "alpha1_minus_S1_exact": x["alpha1_minus_S1_exact"], "alpha1_minus_S3_exact": x["alpha1_minus_S3_exact"]})
tsv(pd.DataFrame(frow), "fig6_alpha1_vs_S1_S3.tsv")
src = read("fig6_alpha1_vs_S1_S3.tsv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
d = src[src.convention == "freear"].set_index("state_id").loc[STATES]
x = np.arange(4)
labs = [("S3_rescues", "S3 (native dir · proto norm)", "#eda100"), ("alpha1_rescues", "α=1 (proto dir · native norm)", "#1baf7a"),
        ("S1_rescues", "S1 (proto dir · proto norm)", "#eb6834")]
for k, (col_suffix, title) in enumerate((("rescues", "S0 WRONG → CORRECT"), ("regressions", "S0 CORRECT → WRONG"))):
    ax = axes[k]
    for j, (col, lab, colr) in enumerate(labs):
        col = col.replace("rescues", col_suffix)
        vals = d[col].values
        ax.bar(x + (j - 1) * 0.26, vals, width=0.24, color=colr, label=lab)
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * 0.26, v + 40, str(int(v)), ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, STATES, fontsize=8)
    ax.set_title(title, fontsize=9, color=INK)
    style(ax)
axes[0].set_ylabel("items (of 29,571), free-AR", fontsize=8, color=INK)
axes[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
fig.suptitle("Figure 6 — Direction × norm factorization: S3 vs α=1 vs S1", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig6_alpha1_vs_S1_S3.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- descriptive item-level tables (frozen fields)
use = ["state_id", "item_index", "phoneme_length", "S0_freear_exact_correct", "S1_freear_exact_correct", "cos_us_up", "theta_rad"]
for a in AK:
    use += [f"{a}_freear_exact_correct", f"{a}_canonical_exact_correct", f"{a}_freear_pred_length",
            f"{a}_freear_eos_before_target_length", f"{a}_freear_eos_after_target_length",
            f"{a}_freear_terminated_by_cap", f"{a}_freear_first_divergence_step"]
it = pd.read_csv(os.path.join(HERE, "item_level_directional_dose.tsv"), sep="\t", usecols=use, keep_default_na=False,
                 low_memory=False)
conv_rows, eos_rows, mono_rows, geo_rows = [], [], [], []
for sid in STATES:
    d = it[it.state_id == sid]
    for a in AK:
        conv_rows.append({"state_id": sid, "alpha": AV[a], "freear_vs_canonical_exact_disagreements":
                          int((d[f"{a}_freear_exact_correct"] != d[f"{a}_canonical_exact_correct"]).sum())})
        reg = d[(d.S0_freear_exact_correct == 1) & (d[f"{a}_freear_exact_correct"] == 0)]
        eb = reg[f"{a}_freear_eos_before_target_length"].astype(int)
        ea = reg[f"{a}_freear_eos_after_target_length"].astype(int)
        cap = reg[f"{a}_freear_terminated_by_cap"].astype(str) == "1"
        same = reg[f"{a}_freear_pred_length"].astype(int) == reg.phoneme_length.astype(int)
        fd = reg[f"{a}_freear_first_divergence_step"].astype(str)
        eos_rows.append({"state_id": sid, "alpha": AV[a], "regressions": len(reg),
                         "eos_before_target_length": int(eb.sum()), "eos_after_target_length": int(ea.sum()),
                         "terminated_by_cap": int(cap.sum()), "pred_length_equals_target_length": int(same.sum()),
                         "first_divergence_at_step0": int((fd == "0").sum())})
    seq = np.stack([d.S0_freear_exact_correct.values] + [d[f"{a}_freear_exact_correct"].values for a in AK], axis=1)
    mono_rows.append({"state_id": sid, "n": len(seq),
                      "correct_at_all_five_alphas_incl_S0": int(seq.all(axis=1).sum()),
                      "wrong_at_all_five": int((seq == 0).all(axis=1).sum()),
                      "S0_correct_then_any_dose_wrong": int(((seq[:, 0] == 1) & (seq[:, 1:] == 0).any(axis=1)).sum()),
                      "S0_correct_wrong_at_050_correct_at_100": int(((seq[:, 0] == 1) & (seq[:, 2] == 0) & (seq[:, 4] == 1)).sum()),
                      "S0_wrong_correct_at_025_wrong_at_050": int(((seq[:, 0] == 0) & (seq[:, 1] == 1) & (seq[:, 2] == 0)).sum()),
                      "S0_wrong_correct_at_100": int(((seq[:, 0] == 0) & (seq[:, 4] == 1)).sum()),
                      "S0_wrong_correct_at_025_and_100": int(((seq[:, 0] == 0) & (seq[:, 1] == 1) & (seq[:, 4] == 1)).sum()),
                      "items_with_any_correct_to_wrong_step_in_sequence": int((np.diff(seq, axis=1) < 0).any(axis=1).sum()),
                      "items_with_any_wrong_to_correct_step_in_sequence": int((np.diff(seq, axis=1) > 0).any(axis=1).sum())})
    qs = np.quantile(d.cos_us_up.astype(float), [0.1, 0.25, 0.5, 0.75, 0.9])
    geo_rows.append({"state_id": sid, **{f"cos_us_up_q{int(q*100)}": float(v) for q, v in zip([0.1, 0.25, 0.5, 0.75, 0.9], qs)},
                     "theta_rad_median": float(d.theta_rad.astype(float).median())})
tsv(pd.DataFrame(conv_rows), "table_convention_item_agreement.tsv")
tsv(pd.DataFrame(eos_rows), "table_regression_eos_profile_freear.tsv")
tsv(pd.DataFrame(mono_rows), "table_item_sequence_nonmonotonicity_freear.tsv")
tsv(pd.DataFrame(geo_rows), "table_geometry_descriptives.tsv")
NUM.update({"convention_agreement": conv_rows, "regression_eos_profile": eos_rows, "item_sequences": mono_rows,
            "geometry": geo_rows, "alpha1_factorization": frow, "paired": prow})
json.dump(NUM, open(os.path.join(FSD, "analysis_numbers.json"), "w"), indent=1, sort_keys=True, default=str)
print(json.dumps({"conv": conv_rows[:4], "eos": eos_rows[:4], "mono": mono_rows, "geo": geo_rows}, indent=1))
