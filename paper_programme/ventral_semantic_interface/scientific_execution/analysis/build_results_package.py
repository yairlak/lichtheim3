#!/usr/bin/env python3
"""Results package builder for the executed ventral semantic interface diagnostic.

READ-ONLY over the frozen outputs produced by the single `--execute` run:
  item_level_factorization.tsv, ar_diagnostic_native_freear.tsv, summary_metrics.json
No model is loaded; nothing is decoded; no new variable is computed beyond the frozen
fields (only counts, set overlaps, proportions and descriptive quantiles of frozen
columns).  Every plotted number is written to figure_source_data/ first and the figure
is drawn from that file.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scientific_execution/
FSD = os.path.join(HERE, "figure_source_data")
FIG = os.path.join(HERE, "figures")
STATES = ["W3_SRC", "W3_REP", "W4_SRC", "W4_REP"]
CONDS = ["S0", "S1", "S2", "S3"]
CONVS = ["freear", "canonical"]
CONV_LABEL = {"freear": "GENUINE FREE-AR (primary)", "canonical": "CANONICAL FORCED-LENGTH (secondary)"}
# reference categorical slots 1-4 (validated adjacent order), direct labels on every bar
COLOR = {"S0": "#2a78d6", "S1": "#eb6834", "S2": "#1baf7a", "S3": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

os.makedirs(FSD, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
NUM: dict = {}


def tsv(df: pd.DataFrame, name: str) -> str:
    p = os.path.join(FSD, name)
    df.to_csv(p, sep="\t", index=False, lineterminator="\n", float_format="%.10g")
    return p


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


it = pd.read_csv(os.path.join(HERE, "item_level_factorization.tsv"), sep="\t",
                 keep_default_na=False, low_memory=False)
ar = pd.read_csv(os.path.join(HERE, "ar_diagnostic_native_freear.tsv"), sep="\t",
                 keep_default_na=False, low_memory=False)
summ = json.load(open(os.path.join(HERE, "summary_metrics.json")))
for c in CONDS:
    for v in CONVS:
        it[f"{c}_{v}_exact_correct"] = it[f"{c}_{v}_exact_correct"].astype(int)
for col in ("in_C_population", "lexical_identity_correct", "phonology_correct", "s3_degenerate",
            "phoneme_length", "lexical_rank", "item_index"):
    it[col] = it[col].astype(int)
for col in ("cosine_target_shat", "cosine_top1", "cosine_top2", "retrieval_margin", "shat_norm",
            "retrieved_raw_glove_norm", "true_raw_glove_norm"):
    it[col] = it[col].astype(float)

# ---------------------------------------------------------------- integrity of the frozen outputs
checks = {
    "item_rows": int(len(it)), "item_rows_expected": 4 * 29571,
    "unique_state_item": int(it[["state_id", "item_index"]].drop_duplicates().shape[0]),
    "rows_per_state": {s: int((it.state_id == s).sum()) for s in STATES},
    "ar_rows": int(len(ar)),
    "ar_unique_state_item": int(ar[["state_id", "item_index"]].drop_duplicates().shape[0]),
    "s3_degenerate_total": int(it.s3_degenerate.sum()),
}
# summary <-> item-level consistency (every ALL / NATIVE_LTM_WRONG cell recomputed)
mism = 0
for s in STATES:
    d = it[it.state_id == s]
    for v in CONVS:
        for c in CONDS:
            cell = summ["results"][s][v]["ALL_REPETITION_ITEMS"][c]
            use = d if c != "S3" else d[d.s3_degenerate == 0]
            mism += int(cell["exact_count"] != int(use[f"{c}_{v}_exact_correct"].sum()))
            mism += int(cell["denominator"] != len(use))
    ar_s = ar[ar.state_id == s]
    mism += int(len(ar_s) != int((d.S0_freear_exact_correct == 0).sum()))
checks["summary_vs_item_level_mismatches"] = mism
NUM["output_integrity"] = checks

# ---------------------------------------------------------------- Fig 1: condition exact, both conventions
rows = []
for s in STATES:
    for v in CONVS:
        for c in CONDS:
            cell = summ["results"][s][v]["ALL_REPETITION_ITEMS"][c]
            rows.append({"state_id": s, "convention": v, "condition": c,
                         "denominator": cell["denominator"], "exact_count": cell["exact_count"],
                         "exact_proportion": cell["exact_proportion"],
                         "errors": cell["denominator"] - cell["exact_count"]})
f1 = pd.DataFrame(rows)
tsv(f1, "fig1_condition_exact_all_items.tsv")
NUM["fig1"] = f1.to_dict(orient="records")

# item-level agreement between conventions
agree = []
for s in STATES:
    d = it[it.state_id == s]
    for c in CONDS:
        agree.append({"state_id": s, "condition": c,
                      "items_disagreeing_freear_vs_canonical":
                          int((d[f"{c}_freear_exact_correct"] != d[f"{c}_canonical_exact_correct"]).sum()),
                      "predicted_phonology_disagreeing":
                          int((d[f"{c}_freear_predicted_phonology"] != d[f"{c}_canonical_predicted_phonology"]).sum())})
tsv(pd.DataFrame(agree), "table_convention_item_agreement.tsv")
NUM["convention_agreement"] = agree

fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharey=True)
for r, v in enumerate(CONVS):
    for k, s in enumerate(STATES):
        ax = axes[r, k]
        sub = pd.read_csv(os.path.join(FSD, "fig1_condition_exact_all_items.tsv"), sep="\t")
        sub = sub[(sub.state_id == s) & (sub.convention == v)]
        y = sub.exact_proportion.values * 100
        ax.bar(range(4), y, width=0.62, color=[COLOR[c] for c in sub.condition], edgecolor="white", linewidth=2)
        for x, (yy, e) in enumerate(zip(y, sub.errors.values)):
            ax.text(x, yy + 0.4, f"{yy:.2f}%\n{e} err", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(range(4), CONDS)
        ax.set_ylim(75, 104)
        style(ax)
        if r == 0:
            ax.set_title(s, fontsize=9, color=INK)
        if k == 0:
            ax.set_ylabel(("free-AR" if v == "freear" else "forced-length") + "\nexact (%)", fontsize=8, color=INK)
fig.suptitle("Figure 1 — Isolated ventral repetition exact match by semantic input (29,571 items; y-axis from 75%)",
             fontsize=10, color=INK)
fig.text(0.5, 0.005, "S0 native ŝ · S1 raw retrieved GloVe · S2 raw true GloVe · S3 radial (‖retrieved‖·ŝ/‖ŝ‖). "
         "Rows are separate readouts; never pooled.", ha="center", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(os.path.join(FIG, "fig1_condition_exact.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: rescue decomposition on native failures
rows, trans = [], []
for s in STATES:
    d = it[it.state_id == s]
    for v in CONVS:
        nf = d[d[f"S0_{v}_exact_correct"] == 0]
        nc = d[d[f"S0_{v}_exact_correct"] == 1]
        r1 = nf[f"S1_{v}_exact_correct"] == 1
        r2 = nf[f"S2_{v}_exact_correct"] == 1
        nd3 = nf.s3_degenerate == 0
        r3 = (nf[f"S3_{v}_exact_correct"] == 1) & nd3
        rec = {"state_id": s, "convention": v, "native_failures": len(nf),
               "S1_rescues": int(r1.sum()), "S2_rescues": int(r2.sum()), "S3_rescues": int(r3.sum()),
               "S1_and_S3_rescues": int((r1 & r3).sum()),
               "S1_only_not_S3": int((r1 & ~r3).sum()), "S3_only_not_S1": int((r3 & ~r1).sum()),
               "S2_only_not_S1": int((r2 & ~r1).sum()), "S1_only_not_S2": int((r1 & ~r2).sum()),
               "no_condition_rescues": int((~r1 & ~r2 & ~r3).sum()),
               "S3_over_S1_rescue_overlap_prop": float((r1 & r3).sum() / r1.sum()) if r1.sum() else None,
               "S1_regressions": int((nc[f"S1_{v}_exact_correct"] == 0).sum()),
               "S2_regressions": int((nc[f"S2_{v}_exact_correct"] == 0).sum()),
               "S3_regressions": int(((nc[f"S3_{v}_exact_correct"] == 0) & (nc.s3_degenerate == 0)).sum()),
               "native_correct": len(nc)}
        rows.append(rec)
        for c in ("S1", "S2", "S3"):
            t = summ["results"][s][v]["ALL_REPETITION_ITEMS"][c]["transitions_vs_S0"]
            trans.append({"state_id": s, "convention": v, "condition": c, **t,
                          "denominator": summ["results"][s][v]["ALL_REPETITION_ITEMS"][c]["denominator"]})
f2 = pd.DataFrame(rows)
tsv(f2, "fig2_native_failure_rescue_decomposition.tsv")
tsv(pd.DataFrame(trans), "table_transitions_all_items.tsv")
NUM["fig2"] = rows
NUM["transitions_all_items"] = trans

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
src = pd.read_csv(os.path.join(FSD, "fig2_native_failure_rescue_decomposition.tsv"), sep="\t")
src = src[src.convention == "freear"].set_index("state_id").loc[STATES]
x = np.arange(4)
ax = axes[0]
parts = [("S1_and_S3_rescues", "rescued by S1 and S3", COLOR["S3"]),
         ("S1_only_not_S3", "rescued by S1 only (not S3)", COLOR["S1"]),
         ("S2_only_not_S1", "rescued by S2 only (not S1)", COLOR["S2"]),
         ("no_condition_rescues", "rescued by none", "#b5b3ad")]
bottom = np.zeros(4)
for col, lab, colr in parts:
    vals = src[col].values.astype(float)
    ax.bar(x, vals, bottom=bottom, width=0.6, color=colr, edgecolor="white", linewidth=2, label=lab)
    for i, vv in enumerate(vals):
        if vv >= 60:
            ax.text(i, bottom[i] + vv / 2, f"{int(vv)}", ha="center", va="center", fontsize=7.5, color=INK)
    bottom += vals
for i, n in enumerate(src.native_failures.values):
    ax.text(i, n + 40, f"n={n}\nS2-only={int(src.S2_only_not_S1.values[i])} · none={int(src.no_condition_rescues.values[i])}",
            ha="center", va="bottom", fontsize=7, color=INK)
ax.set_ylim(0, 4300)
ax.set_xticks(x, STATES)
ax.set_ylabel("native S0 free-AR failures", fontsize=8, color=INK)
ax.set_title("Rescue of native failures (free-AR)", fontsize=9, color=INK)
ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
style(ax)
ax = axes[1]
w = 0.26
for j, c in enumerate(("S1", "S2", "S3")):
    vals = src[f"{c}_regressions"].values
    ax.bar(x + (j - 1) * w, vals, width=w - 0.02, color=COLOR[c], edgecolor="white", linewidth=1, label=c)
    for i, vv in enumerate(vals):
        ax.text(i + (j - 1) * w, vv + 25, str(int(vv)), ha="center", fontsize=7, color=INK)
ax.set_xticks(x, STATES)
ax.set_ylabel("CORRECT→WRONG vs S0 (all 29,571 items)", fontsize=8, color=INK)
ax.set_title("Regressions introduced (free-AR)", fontsize=9, color=INK)
ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3)
style(ax)
fig.suptitle("Figure 2 — Rescue vs regression decomposition relative to native S0", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(FIG, "fig2_rescue_decomposition.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Fig 3: C x native-LTM strata
STRATA = ["ALL_REPETITION_ITEMS", "C_POPULATION", "NONCANONICAL_HOMOPHONE_MEMBERS", "NATIVE_LTM_WRONG",
          "C_CORRECT_AND_NATIVE_LTM_WRONG", "C_WRONG_AND_NATIVE_LTM_WRONG",
          "C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT", "C_WRONG_BUT_RETRIEVED_PHONOLOGY_CORRECT",
          "LEXICAL_IDENTITY_CORRECT_AND_NATIVE_LTM_WRONG", "LEXICAL_IDENTITY_WRONG_AND_NATIVE_LTM_WRONG"]
rows = []
for s in STATES:
    for v in CONVS:
        for st in STRATA:
            for c in CONDS:
                cell = summ["results"][s][v][st][c]
                t = cell["transitions_vs_S0"] or {}
                rows.append({"state_id": s, "convention": v, "stratum": st, "condition": c,
                             "denominator": cell["denominator"], "exact_count": cell["exact_count"],
                             "exact_proportion": cell["exact_proportion"],
                             "WRONG_TO_CORRECT": t.get("WRONG_TO_CORRECT", "NA"),
                             "CORRECT_TO_WRONG": t.get("CORRECT_TO_WRONG", "NA"),
                             "CORRECT_TO_CORRECT": t.get("CORRECT_TO_CORRECT", "NA"),
                             "WRONG_TO_WRONG": t.get("WRONG_TO_WRONG", "NA")})
f3 = pd.DataFrame(rows)
tsv(f3, "fig3_strata_all.tsv")

# lexical vs phonological identity of retrieval, per native outcome (C population and homophones)
lp = []
for s in STATES:
    d = it[it.state_id == s]
    for pop, mask in (("C_POPULATION", d.in_C_population == 1), ("NONCANONICAL_HOMOPHONE_MEMBERS", d.in_C_population == 0)):
        for nat, m2 in (("NATIVE_WRONG", d.S0_freear_exact_correct == 0), ("NATIVE_CORRECT", d.S0_freear_exact_correct == 1)):
            sub = d[mask & m2]
            lp.append({"state_id": s, "population": pop, "native_freear": nat, "n": len(sub),
                       "retrieval_lexical_identity_correct": int(sub.lexical_identity_correct.sum()),
                       "retrieval_lexical_wrong_phonology_correct":
                           int(((sub.lexical_identity_correct == 0) & (sub.phonology_correct == 1)).sum()),
                       "retrieval_phonology_wrong": int((sub.phonology_correct == 0).sum())})
f3b = pd.DataFrame(lp)
tsv(f3b, "fig3_lexical_vs_phonological_retrieval.tsv")
NUM["fig3_lexical_vs_phonological"] = lp

fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=True)
KEY = ["C_CORRECT_AND_NATIVE_LTM_WRONG", "C_WRONG_AND_NATIVE_LTM_WRONG", "C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT",
       "LEXICAL_IDENTITY_WRONG_AND_NATIVE_LTM_WRONG"]
SHORT = ["C✓ · S0✗", "C✗ · S0✗", "C✗ · S0✓", "lexID✗ · S0✗\n(all rows)"]
src = pd.read_csv(os.path.join(FSD, "fig3_strata_all.tsv"), sep="\t")
for k, s in enumerate(STATES):
    ax = axes[k]
    sub = src[(src.state_id == s) & (src.convention == "freear")]
    for j, st in enumerate(KEY):
        cells = sub[sub.stratum == st].set_index("condition")
        n = int(cells.loc["S0", "denominator"])
        for q, c in enumerate(CONDS):
            val = cells.loc[c, "exact_proportion"]
            xpos = j + (q - 1.5) * 0.2
            if n > 0:
                ax.bar(xpos, float(val) * 100, width=0.18, color=COLOR[c], edgecolor="white", linewidth=1,
                       label=c if (j == 0 and k == 0) else None)
                if float(val) == 0.0:
                    ax.text(xpos, 1.5, "0", ha="center", fontsize=6, color=MUTED)
        ax.text(j, 104, f"n={n}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(4), SHORT, fontsize=6.5)
    ax.set_ylim(0, 112)
    ax.set_title(s, fontsize=9, color=INK)
    style(ax)
axes[0].set_ylabel("exact (%) within stratum, free-AR", fontsize=8, color=INK)
axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.14))
fig.suptitle("Figure 3 — C (historical lexical-identity contract) × native ventral outcome strata", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(FIG, "fig3_c_x_ltm_strata.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- retrieval-limited and S1 regressions
rl = []
for s in STATES:
    d = it[it.state_id == s]
    for v in CONVS:
        nf = d[d[f"S0_{v}_exact_correct"] == 0]
        lim = nf[(nf[f"S2_{v}_exact_correct"] == 1) & (nf[f"S1_{v}_exact_correct"] == 0)]
        nc = d[d[f"S0_{v}_exact_correct"] == 1]
        reg = nc[nc[f"S1_{v}_exact_correct"] == 0]
        rl.append({"state_id": s, "convention": v,
                   "native_fail_S2_correct_S1_wrong": len(lim),
                   "of_which_lexical_identity_wrong": int((lim.lexical_identity_correct == 0).sum()),
                   "of_which_retrieved_phonology_wrong": int((lim.phonology_correct == 0).sum()),
                   "of_which_in_C_population": int(lim.in_C_population.sum()),
                   "native_correct_S1_wrong": len(reg),
                   "S1_regressions_lexical_identity_wrong": int((reg.lexical_identity_correct == 0).sum()),
                   "S1_regressions_retrieved_phonology_wrong": int((reg.phonology_correct == 0).sum()),
                   "S1_regressions_in_C_population": int(reg.in_C_population.sum()),
                   "S1_wrong_with_lexical_identity_correct": int(((d[f"S1_{v}_exact_correct"] == 0)
                                                                  & (d.lexical_identity_correct == 1)).sum())})
        if v == "freear":
            cols = ["state_id", "item_index", "lexical_identity", "target_phonology", "in_C_population",
                    "C_contract_correct", "retrieved_lexical_identity", "retrieved_phonology",
                    "lexical_identity_correct", "phonology_correct", "S0_freear_exact_correct",
                    "S1_freear_predicted_phonology", "S2_freear_exact_correct"]
            pd.concat([lim.assign(kind="NATIVE_FAIL_S2_CORRECT_S1_WRONG")[["kind"] + cols],
                       reg.assign(kind="NATIVE_CORRECT_S1_WRONG")[["kind"] + cols]]).to_csv(
                os.path.join(FSD, f"table_retrieval_limited_items_{s}_freear.tsv"), sep="\t", index=False,
                lineterminator="\n")
tsv(pd.DataFrame(rl), "table_retrieval_limited_summary.tsv")
NUM["retrieval_limited"] = rl

# ---------------------------------------------------------------- geometry descriptives (frozen fields only)
GEO = ["cosine_target_shat", "cosine_top1", "cosine_top2", "retrieval_margin", "shat_norm",
       "retrieved_raw_glove_norm", "true_raw_glove_norm", "phoneme_length", "lexical_rank"]
geo = []
for s in STATES:
    d = it[it.state_id == s]
    groups = {
        "C_CORRECT_NATIVE_CORRECT": (d.in_C_population == 1) & (d.C_contract_correct == "1") & (d.S0_freear_exact_correct == 1),
        "C_CORRECT_NATIVE_WRONG": (d.in_C_population == 1) & (d.C_contract_correct == "1") & (d.S0_freear_exact_correct == 0),
        "C_CORRECT_NATIVE_WRONG_S3_RESCUED": (d.in_C_population == 1) & (d.C_contract_correct == "1") & (d.S0_freear_exact_correct == 0) & (d.S3_freear_exact_correct == 1),
        "C_CORRECT_NATIVE_WRONG_S3_NOT_RESCUED": (d.in_C_population == 1) & (d.C_contract_correct == "1") & (d.S0_freear_exact_correct == 0) & (d.S3_freear_exact_correct == 0),
        "NATIVE_CORRECT_S3_REGRESSED": (d.S0_freear_exact_correct == 1) & (d.S3_freear_exact_correct == 0),
        "NATIVE_CORRECT_S3_KEPT": (d.S0_freear_exact_correct == 1) & (d.S3_freear_exact_correct == 1),
        "C_WRONG": (d.in_C_population == 1) & (d.C_contract_correct == "0"),
        "NONCANONICAL_HOMOPHONE_MEMBERS": d.in_C_population == 0,
    }
    for g, m in groups.items():
        sub = d[m]
        rec = {"state_id": s, "group": g, "n": len(sub)}
        for col in GEO:
            vals = sub[col].astype(float)
            rec[f"{col}_median"] = float(vals.median()) if len(sub) else None
            rec[f"{col}_q25"] = float(vals.quantile(0.25)) if len(sub) else None
            rec[f"{col}_q75"] = float(vals.quantile(0.75)) if len(sub) else None
        ratio = sub.retrieved_raw_glove_norm / sub.shat_norm
        rec["retrieved_norm_over_shat_norm_median"] = float(ratio.median()) if len(sub) else None
        geo.append(rec)
tsv(pd.DataFrame(geo), "table_geometry_descriptives_freear.tsv")
NUM["geometry"] = geo

# ---------------------------------------------------------------- Fig 4: AR diagnostic
ar["divergence_step"] = ar.divergence_step.astype(int)
ar["gold_rank"] = ar.gold_rank.astype(int)
ar["margin_chosen_minus_gold"] = ar.margin_chosen_minus_gold.astype(float)
ar["n_positional_mismatches"] = ar.n_positional_mismatches.astype(int)
ar["n_generated_after_divergence"] = ar.n_generated_after_divergence.astype(int)
ar["phoneme_length"] = ar.phoneme_length.astype(int)
for col in ("divergence_is_eos_position", "margin_numerically_ambiguous",
            "diag_prefix_correction_corrected_exact", "diag_prefix_correction_corrected_terminated_by_cap"):
    ar[col] = ar[col].astype(int)
ar["pred_len"] = ar.S0_freear_predicted_phonology.map(lambda x: len(x.split()) if x else 0)

# separately labelled prefix-correction output (verbatim columns)
pc_cols = ["state_id", "item_index", "lexical_identity", "target_phonology", "S0_freear_predicted_phonology",
           "divergence_step"] + [c for c in ar.columns if c.startswith("diag_prefix_correction_")]
pc = pd.read_csv(os.path.join(HERE, "ar_diagnostic_native_freear.tsv"), sep="\t", keep_default_na=False,
                 dtype=str)[pc_cols]
pc.insert(0, "LABEL", "DIAGNOSTIC_ONLY_PREFIX_CORRECTION — NOT PERFORMANCE; NOT AN INFERENCE MECHANISM")
pc.to_csv(os.path.join(HERE, "DIAGNOSTIC_ONLY_PREFIX_CORRECTION_native_freear.tsv"), sep="\t", index=False,
          lineterminator="\n")

ars, step_rows, margin_rows = [], [], []
for s in STATES:
    a = ar[ar.state_id == s]
    n = len(a)
    first_div_before_final_eos = int((a.divergence_is_eos_position == 0).sum())
    ars.append({
        "state_id": s, "n_native_freear_failures": n,
        "divergence_at_eos_position": int(a.divergence_is_eos_position.sum()),
        "divergence_at_phoneme_position": first_div_before_final_eos,
        "gold_rank_2": int((a.gold_rank == 2).sum()), "gold_rank_3_to_5": int(((a.gold_rank >= 3) & (a.gold_rank <= 5)).sum()),
        "gold_rank_gt5": int((a.gold_rank > 5).sum()),
        "margin_median": float(a.margin_chosen_minus_gold.median()),
        "margin_q10": float(a.margin_chosen_minus_gold.quantile(0.10)),
        "margin_q25": float(a.margin_chosen_minus_gold.quantile(0.25)),
        "margin_q75": float(a.margin_chosen_minus_gold.quantile(0.75)),
        "margin_q90": float(a.margin_chosen_minus_gold.quantile(0.90)),
        "margin_lt_0p5": int((a.margin_chosen_minus_gold < 0.5).sum()),
        "margin_lt_1": int((a.margin_chosen_minus_gold < 1.0).sum()),
        "margin_lt_2": int((a.margin_chosen_minus_gold < 2.0).sum()),
        "margin_ge_5": int((a.margin_chosen_minus_gold >= 5.0).sum()),
        "margin_numerically_ambiguous": int(a.margin_numerically_ambiguous.sum()),
        "positional_mismatches_eq_1": int((a.n_positional_mismatches == 1).sum()),
        "positional_mismatches_ge_2": int((a.n_positional_mismatches >= 2).sum()),
        "positional_mismatches_mean": float(a.n_positional_mismatches.mean()),
        "pred_len_eq_target": int((a.pred_len == a.phoneme_length).sum()),
        "pred_len_lt_target": int((a.pred_len < a.phoneme_length).sum()),
        "pred_len_gt_target": int((a.pred_len > a.phoneme_length).sum()),
        "generated_after_divergence_median": float(a.n_generated_after_divergence.median()),
        "DIAG_prefix_correction_corrected_exact": int(a.diag_prefix_correction_corrected_exact.sum()),
        "DIAG_prefix_correction_corrected_exact_prop": float(a.diag_prefix_correction_corrected_exact.mean()) if n else None,
        "DIAG_prefix_correction_terminated_by_cap": int(a.diag_prefix_correction_corrected_terminated_by_cap.sum()),
    })
    for t, cnt in a.divergence_step.value_counts().sort_index().items():
        step_rows.append({"state_id": s, "divergence_step": int(t), "count": int(cnt)})
    edges = [-1e9, 0.5, 1, 2, 3, 5, 1e9]
    labels = ["[0,0.5)", "[0.5,1)", "[1,2)", "[2,3)", "[3,5)", "≥5"]
    cut = pd.cut(a.margin_chosen_minus_gold, bins=edges, labels=labels, right=False)
    for lab in labels:
        margin_rows.append({"state_id": s, "margin_bin_logit": lab, "count": int((cut == lab).sum())})
tsv(pd.DataFrame(ars), "fig4_ar_diagnostic_summary.tsv")
tsv(pd.DataFrame(step_rows), "fig4_first_divergence_step_counts.tsv")
tsv(pd.DataFrame(margin_rows), "fig4_margin_histogram.tsv")
NUM["ar"] = ars
NUM["ar_step_counts"] = step_rows
NUM["ar_margin_bins"] = margin_rows

# native-failure EOS protocol (frozen S0 fields)
eos = []
for s in STATES:
    d = it[(it.state_id == s) & (it.S0_freear_exact_correct == 0)]
    eos.append({"state_id": s, "n": len(d),
                "eos_emitted": int((d.S0_freear_eos_emitted.astype(int) == 1).sum()),
                "terminated_by_cap": int((d.S0_freear_terminated_by_cap.astype(str) == "1").sum()),
                "eos_before_target_length": int((d.S0_freear_eos_before_target_length.astype(int) == 1).sum()),
                "eos_after_target_length": int((d.S0_freear_eos_after_target_length.astype(int) == 1).sum())})
tsv(pd.DataFrame(eos), "table_native_failure_eos_protocol_freear.tsv")
NUM["eos_protocol"] = eos

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
steps = pd.read_csv(os.path.join(FSD, "fig4_first_divergence_step_counts.tsv"), sep="\t")
mh = pd.read_csv(os.path.join(FSD, "fig4_margin_histogram.tsv"), sep="\t")
asum = pd.read_csv(os.path.join(FSD, "fig4_ar_diagnostic_summary.tsv"), sep="\t").set_index("state_id").loc[STATES]
SCOL = {"W3_SRC": "#2a78d6", "W3_REP": "#eb6834", "W4_SRC": "#1baf7a", "W4_REP": "#eda100"}
ax = axes[0]
allsteps = sorted(steps.divergence_step.unique())
for q, s in enumerate(STATES):
    cnt = steps[steps.state_id == s].set_index("divergence_step")["count"].reindex(allsteps, fill_value=0)
    ax.bar(np.arange(len(allsteps)) + (q - 1.5) * 0.2, cnt.values, width=0.18, color=SCOL[s], label=s)
ax.set_xticks(range(len(allsteps)), allsteps)
ax.set_xlabel("first divergence step (0 = first phoneme)", fontsize=8, color=INK)
ax.set_ylabel("native free-AR failures", fontsize=8, color=INK)
ax.set_title("Where the first error occurs", fontsize=9, color=INK)
ax.legend(fontsize=7, frameon=False)
style(ax)
ax = axes[1]
bins = list(dict.fromkeys(mh.margin_bin_logit))
for q, s in enumerate(STATES):
    sub = mh[mh.state_id == s].set_index("margin_bin_logit").loc[bins]
    ax.bar(np.arange(len(bins)) + (q - 1.5) * 0.2, sub["count"].values, width=0.18, color=SCOL[s], label=s)
ax.set_xticks(range(len(bins)), bins, fontsize=7)
ax.set_xlabel("chosen − gold logit at first divergence (gold-prefix logits)", fontsize=8, color=INK)
ax.set_title("First-divergence margin", fontsize=9, color=INK)
style(ax)
ax = axes[2]
x = np.arange(4)
one = asum.positional_mismatches_eq_1.values
multi = asum.positional_mismatches_ge_2.values
ax.bar(x - 0.15, one, width=0.28, color="#2a78d6", label="1 positional mismatch")
ax.bar(x + 0.15, multi, width=0.28, color="#e34948", label="≥2 positional mismatches")
for i in range(4):
    ax.text(i - 0.15, one[i] + 20, str(int(one[i])), ha="center", fontsize=7, color=INK)
    ax.text(i + 0.15, multi[i] + 20, str(int(multi[i])), ha="center", fontsize=7, color=INK)
ax.set_ylim(0, 3900)
ax.set_xticks(x, [f"{st}\nfix {int(asum.DIAG_prefix_correction_corrected_exact.values[i])}/{int(asum.n_native_freear_failures.values[i])}"
                  for i, st in enumerate(STATES)], fontsize=6.5)
ax.set_title("Cascade extent\n(tick 'fix' = DIAGNOSTIC_ONLY prefix-corrected exact)", fontsize=8.5, color=INK)
ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=2)
style(ax)
fig.suptitle("Figure 4 — First-divergence diagnostic on native S0 genuine free-AR failures", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig4_ar_first_divergence.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- Fig 5: SOURCE vs POST_REPAIR
pair_rows, item_pairs = [], []
for w, (a_id, b_id) in {"W3": ("W3_SRC", "W3_REP"), "W4": ("W4_SRC", "W4_REP")}.items():
    A = it[it.state_id == a_id].set_index("item_index")
    B = it[it.state_id == b_id].set_index("item_index").loc[A.index]
    for v in CONVS:
        for c in CONDS:
            a, b = A[f"{c}_{v}_exact_correct"], B[f"{c}_{v}_exact_correct"]
            pair_rows.append({"witness": w, "convention": v, "condition": c,
                              "SRC_exact": int(a.sum()), "REP_exact": int(b.sum()),
                              "SRC_errors": int((a == 0).sum()), "REP_errors": int((b == 0).sum()),
                              "WRONG_TO_CORRECT_SRC_to_REP": int(((a == 0) & (b == 1)).sum()),
                              "CORRECT_TO_WRONG_SRC_to_REP": int(((a == 1) & (b == 0)).sum()),
                              "WRONG_TO_WRONG": int(((a == 0) & (b == 0)).sum())})
    ca = A[A.in_C_population == 1].C_contract_correct.astype(int)
    cb = B[B.in_C_population == 1].C_contract_correct.astype(int)
    item_pairs.append({"witness": w, "C_errors_SRC": int((ca == 0).sum()), "C_errors_REP": int((cb == 0).sum()),
                       "C_WRONG_TO_CORRECT": int(((ca == 0) & (cb == 1)).sum()),
                       "C_CORRECT_TO_WRONG": int(((ca == 1) & (cb == 0)).sum()),
                       "native_failure_set_jaccard_freear": float(
                           ((A.S0_freear_exact_correct == 0) & (B.S0_freear_exact_correct == 0)).sum()
                           / ((A.S0_freear_exact_correct == 0) | (B.S0_freear_exact_correct == 0)).sum()),
                       "S3_rescue_set_jaccard_freear": float(
                           ((A.S0_freear_exact_correct == 0) & (A.S3_freear_exact_correct == 1) & (B.S0_freear_exact_correct == 0) & (B.S3_freear_exact_correct == 1)).sum()
                           / (((A.S0_freear_exact_correct == 0) & (A.S3_freear_exact_correct == 1)) | ((B.S0_freear_exact_correct == 0) & (B.S3_freear_exact_correct == 1))).sum()),
                       "cosine_target_shat_median_SRC": float(A.cosine_target_shat.median()),
                       "cosine_target_shat_median_REP": float(B.cosine_target_shat.median()),
                       "shat_norm_median_SRC": float(A.shat_norm.median()), "shat_norm_median_REP": float(B.shat_norm.median()),
                       "native_freear_failures_cosine_target_median_SRC": float(A[A.S0_freear_exact_correct == 0].cosine_target_shat.median()),
                       "native_freear_failures_cosine_target_median_REP": float(B[B.S0_freear_exact_correct == 0].cosine_target_shat.median())})
f5 = pd.DataFrame(pair_rows)
tsv(f5, "fig5_source_vs_repaired_conditions.tsv")
tsv(pd.DataFrame(item_pairs), "fig5_source_vs_repaired_c_and_overlap.tsv")
NUM["fig5_conditions"] = pair_rows
NUM["fig5_c_overlap"] = item_pairs

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
src = pd.read_csv(os.path.join(FSD, "fig5_source_vs_repaired_conditions.tsv"), sep="\t")
for k, w in enumerate(("W3", "W4")):
    ax = axes[k]
    sub = src[(src.witness == w) & (src.convention == "freear")].set_index("condition").loc[CONDS]
    x = np.arange(4)
    ax.bar(x - 0.17, sub.SRC_errors.values, width=0.32, color="#2a78d6", label="SOURCE")
    ax.bar(x + 0.17, sub.REP_errors.values, width=0.32, color="#eb6834", label="POST_REPAIR (Arm-A)")
    for i in range(4):
        ax.text(i - 0.17, sub.SRC_errors.values[i] + 40, str(int(sub.SRC_errors.values[i])), ha="center", fontsize=7, color=INK)
        ax.text(i + 0.17, sub.REP_errors.values[i] + 40, str(int(sub.REP_errors.values[i])), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, CONDS)
    ax.set_ylabel("free-AR errors (of 29,571)", fontsize=8, color=INK)
    ax.set_title(f"{w}: SOURCE vs POST_REPAIR", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
fig.suptitle("Figure 5 — Paired SOURCE vs POST_REPAIR isolated ventral errors by semantic input (free-AR)", fontsize=10, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(FIG, "fig5_source_vs_repaired.png"), dpi=160)
plt.close(fig)

with open(os.path.join(FSD, "analysis_numbers.json"), "w") as f:
    json.dump(NUM, f, indent=1, sort_keys=True, default=str)
print(json.dumps({"integrity": checks}, indent=1))
