"""Sprint 5 — adapted feature-importance figures, tables and report generator.

    python -m scripts.behavioral_analysis.plot_feature_importance \
        --out_root reports/behavioral_wfe_fulllexicon_93a577f/feature_importance

Regenerates every Sprint-5 output from the canonical table alone: no torch, no
checkpoint, no model inference beyond the Ridge fits described in the frozen
spec.

Presentation (frozen): factors use a **neutral palette** and **red and blue are
not used at all** — these panels show model diagnostics, not real/pseudo
observations, so borrowing the reserved lexicality colours would misread as an
observation-level contrast. All four seed points are visible, within-seed
permutation spread is drawn as a distinct whisker so it cannot be confused with
between-seed spread, and ceiling-limited routes are labelled rather than given
an artificial zero.

The faithful Dager analysis (A11) is never read, rewritten or plotted here.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from scripts.behavioral_analysis import feature_importance as fi      # noqa: E402
from scripts.behavioral_analysis.common import (CANONICAL_TABLE,      # noqa: E402
                                                REPORT_ROOT, ROUTE_LABEL,
                                                ROUTES, SEED_MARKER, SEEDS,
                                                repo_relative)
from scripts.behavioral_analysis.io import (load_canonical,           # noqa: E402
                                            sha256_file, write_table)
from scripts.behavioral_analysis.plotting import save_figure          # noqa: E402

# Neutral factor palette — deliberately no red and no blue.
FACTOR_COLOR = {"route": "#2f3b40", "length": "#5c6b70",
                "lexicality": "#8a9499", "morphology": "#b9c1c4"}
FACTOR_LABEL = {"route": "Route", "length": "Phoneme length",
                "lexicality": "Lexicality/exposure", "morphology": "Morphology"}
NEUTRAL_ACCENT = "#3d4f57"

CLEAN_NOTE = (
    "LICHTHEIM_CLEAN comprises 671 trained real words (TRAINED_REAL_EXACT) and "
    "391 novel pseudowords (NOVEL_PSEUDOWORD). **Lexicality and training "
    "exposure are perfectly confounded in this set** — every Real item is "
    "trained and every Pseudo item is novel — so the factor is a "
    "lexicality/exposure contrast and the model cannot separate the two. Zipf "
    "frequency is undefined for pseudowords and is excluded; it is never "
    "imputed (see Sprint 3 for the trained-real frequency analysis).")

METHOD_NOTE = (
    "Ridge (alpha = 1.0, not tuned on the WFE), an 80/20 train/test split "
    "**grouped by item** so all three route rows of an item stay together and "
    "the identical item split is reused across all four seeds, and "
    "**factor-level grouped permutation importance** on the held-out items "
    "with 100 repeats (random_state 42): permuting a raw factor rebuilds every "
    "encoded, standardized and interaction column derived from it, and dummy "
    "or interaction columns are never permuted independently.")

FAITHFUL_NOTE = (
    "The faithful Dager feature importance (A11) is a **separate analysis** on "
    "1,200 source-labelled items, FULL route only, with the original coding and "
    "no route factor. It is not recomputed, replaced or pooled with these "
    "values, and the two are never placed on one quantitative axis.")


def _factor_legend(ax, factors, loc="upper right"):
    return ax.legend(handles=[
        Line2D([], [], color=FACTOR_COLOR[f], lw=7, label=FACTOR_LABEL[f])
        for f in factors], loc=loc, frameon=False, fontsize=8)


def _seed_legend(ax, loc="upper left"):
    return ax.legend(handles=[
        Line2D([], [], color="0.15", marker=SEED_MARKER[s], lw=0, ms=6,
               label=f"seed {s}") for s in SEEDS]
        + [Line2D([], [], color="0.45", lw=1.4,
                  label="within-seed permutation SD"),
           Line2D([], [], color="black", lw=2.4, label="mean over seeds")],
        loc=loc, frameon=False, fontsize=7.5)


# ------------------------------------------------------------- computation

def run_models(canon: pd.DataFrame, train_ids, test_ids):
    """Fit A, B and C for every seed.  One shared item split throughout."""
    out = {"main_fit": [], "main_coef": [], "main_reps": [], "main_summary": [],
           "int_fit": [], "int_coef": [], "int_drop": [],
           "route_fit": [], "route_coef": [], "route_reps": [],
           "route_summary": []}
    tr, te = set(train_ids), set(test_ids)
    for seed in SEEDS:
        df = fi.analysis_frame(canon, seed)
        train = df[df["item_id"].isin(tr)].reset_index(drop=True)
        test = df[df["item_id"].isin(te)].reset_index(drop=True)

        # ---- A: main effects (primary)
        m = fi.fit_model(train, test, fi.FACTORS)
        out["main_fit"].append({"model": "clean_main_effects", "seed": seed,
                                **_fit_row(m)})
        out["main_coef"].append(fi.coefficient_rows(
            m, model="clean_main_effects", seed=seed))
        reps = fi.permutation_importance(m, test, fi.FACTORS)
        reps.insert(0, "seed", seed)
        out["main_reps"].append(reps)
        out["main_summary"].append(fi.summarise_repeats(reps, seed=seed))

        # ---- B: predeclared interactions (secondary)
        mi = fi.fit_model(train, test, fi.FACTORS, fi.INTERACTION_BLOCKS)
        out["int_fit"].append({"model": "clean_interactions", "seed": seed,
                               **_fit_row(mi),
                               "delta_r2_vs_main": mi["test_r2"] - m["test_r2"],
                               "delta_mae_vs_main": m["test_mae"] - mi["test_mae"]})
        out["int_coef"].append(fi.coefficient_rows(
            mi, model="clean_interactions", seed=seed))
        for block in fi.INTERACTION_BLOCKS:
            kept = [b for b in fi.INTERACTION_BLOCKS if b != block]
            md = fi.fit_model(train, test, fi.FACTORS, kept)
            out["int_drop"].append({
                "seed": seed, "dropped_block": block,
                "n_blocks_retained": len(kept),
                "test_r2_without_block": md["test_r2"],
                "test_mae_without_block": md["test_mae"],
                "test_r2_full_interaction_model": mi["test_r2"],
                "test_mae_full_interaction_model": mi["test_mae"],
                "block_r2_utility": mi["test_r2"] - md["test_r2"],
                "block_mae_utility": md["test_mae"] - mi["test_mae"],
                "method": "drop-block refit; no post-hoc selection",
                "model_status": md["model_status"]})

        # ---- C: route-specific (secondary)
        for route in ROUTES:
            rtr = train[train["route"] == route].reset_index(drop=True)
            rte = test[test["route"] == route].reset_index(drop=True)
            mr = fi.fit_model(rtr, rte, fi.ROUTE_SPECIFIC_FACTORS)
            out["route_fit"].append({"model": "route_specific", "seed": seed,
                                     "route": route, **_fit_row(mr)})
            out["route_coef"].append(fi.coefficient_rows(
                mr, model="route_specific", seed=seed, route=route))
            rreps = fi.permutation_importance(mr, rte,
                                              fi.ROUTE_SPECIFIC_FACTORS)
            if len(rreps):
                rreps.insert(0, "seed", seed)
                rreps.insert(1, "route", route)
                out["route_reps"].append(rreps)
                out["route_summary"].append(
                    fi.summarise_repeats(rreps, seed=seed, route=route))
            else:
                out["route_summary"].append(pd.DataFrame([
                    {"seed": seed, "route": route, "factor": f,
                     "n_repeats": 0, "r2_drop_mean": np.nan,
                     "r2_drop_std": np.nan, "r2_drop_min": np.nan,
                     "r2_drop_max": np.nan, "mae_increase_mean": np.nan,
                     "mae_increase_std": np.nan, "mae_increase_min": np.nan,
                     "mae_increase_max": np.nan}
                    for f in fi.ROUTE_SPECIFIC_FACTORS]))
    return out


def _fit_row(m: dict) -> dict:
    return {k: m[k] for k in ("outcome", "n_train_rows", "n_test_rows",
                              "n_train_items", "n_test_items",
                              "n_test_nonzero", "ridge_alpha",
                              "outcome_status", "model_status",
                              "negative_test_r2",
                              "train_r2", "test_r2", "test_mae")}


# ---------------------------------------------------------------- figures

def plot_clean_importance(summary: pd.DataFrame, out_dir: str) -> dict:
    factors = fi.FACTORS
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, (col, sd, title) in zip(axes, [
            ("r2_drop_mean", "r2_drop_std",
             "Primary: held-out R² drop when the factor is permuted"),
            ("mae_increase_mean", "mae_increase_std",
             "Sensitivity: held-out MAE increase")]):
        for i, f in enumerate(factors):
            sub = summary[summary["factor"] == f]
            for _, r in sub.iterrows():
                x = i + (SEEDS.index(int(r["seed"])) - 1.5) * 0.16
                ax.vlines(x, r[col] - r[sd], r[col] + r[sd], color="0.45",
                          lw=1.4, zorder=2)
                ax.plot(x, r[col], marker=SEED_MARKER[int(r["seed"])], ms=6.5,
                        color=FACTOR_COLOR[f], markeredgecolor="0.15",
                        markeredgewidth=0.5, zorder=3)
            ax.hlines(float(sub[col].mean()), i - 0.32, i + 0.32,
                      color="black", lw=2.4, zorder=4)
        ax.set_xticks(range(len(factors)))
        ax.set_xticklabels([FACTOR_LABEL[f] for f in factors], fontsize=8.5)
        ax.axhline(0, color="grey", lw=0.9, ls="--")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Held-out R² drop")
    axes[1].set_ylabel("Held-out MAE increase")
    _seed_legend(axes[0], loc="upper left")
    fig.suptitle("Adapted grouped factor importance — clean WFE set, "
                 "three routes", y=1.02, fontsize=13)
    fig.tight_layout()
    caption = (
        "**Adapted grouped factor importance (clean WFE set, all three "
        "routes).** Left panel is the **primary** measure, the drop in held-out "
        "R² when a raw factor is permuted; right panel is the MAE sensitivity, "
        "reported because sparse zero-heavy outcomes make held-out R² unstable. "
        "One marker per seed (19, 20, 21, 22), each with a thin vertical bar "
        "showing the **within-seed permutation standard deviation over 100 "
        "repeats**; the spread between markers is the **between-seed** "
        "variation, and the two are deliberately drawn differently so they "
        "cannot be confused. The black bar is the mean over seeds. Grouped "
        "permutation importance is **unsigned** — no artificial sign is "
        "assigned to a multilevel factor such as route; coefficients are "
        "reported separately.\n\n" + METHOD_NOTE + "\n\n" + CLEAN_NOTE +
        "\n\nColours are a neutral factor palette: red and blue are reserved "
        "for real/pseudo observations elsewhere and are deliberately not used "
        "here.\n\n" + FAITHFUL_NOTE)
    return save_figure(fig, out_dir, "clean_adapted_factor_importance", caption)


def plot_route_specific(summary: pd.DataFrame, fits: pd.DataFrame,
                        out_dir: str) -> dict:
    factors = fi.ROUTE_SPECIFIC_FACTORS
    est = summary.dropna(subset=["r2_drop_mean"])
    lo, hi = ((float(est["r2_drop_mean"].min()), float(est["r2_drop_mean"].max()))
              if len(est) else (0.0, 1.0))
    pad = max(0.05 * (hi - lo), 1e-3)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), sharey=True)
    for ax, route in zip(axes, ROUTES):
        st = fits[fits["route"] == route]["model_status"].tolist()
        sub = summary[summary["route"] == route]
        drawn = False
        for i, f in enumerate(factors):
            s = sub[sub["factor"] == f]
            for _, r in s.iterrows():
                if not np.isfinite(r["r2_drop_mean"]):
                    continue
                drawn = True
                x = i + (SEEDS.index(int(r["seed"])) - 1.5) * 0.16
                ax.vlines(x, r["r2_drop_mean"] - r["r2_drop_std"],
                          r["r2_drop_mean"] + r["r2_drop_std"], color="0.45",
                          lw=1.4, zorder=2)
                ax.plot(x, r["r2_drop_mean"], marker=SEED_MARKER[int(r["seed"])],
                        ms=6.5, color=FACTOR_COLOR[f], markeredgecolor="0.15",
                        markeredgewidth=0.5, zorder=3)
            m = s["r2_drop_mean"].mean()
            if np.isfinite(m):
                ax.hlines(m, i - 0.32, i + 0.32, color="black", lw=2.4, zorder=4)
        ax.set_xticks(range(len(factors)))
        ax.set_xticklabels([FACTOR_LABEL[f] for f in factors], fontsize=8.5)
        ax.axhline(0, color="grey", lw=0.9, ls="--")
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.25, axis="y")
        uniq = sorted(set(st))
        ax.set_title(f"{ROUTE_LABEL[route]}\n{', '.join(uniq)}", fontsize=10)
        if not drawn:
            ax.text(0.5, 0.5, "\n".join(uniq) + "\nnot estimable —\nno importance"
                    " is drawn", transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color="0.25")
    axes[0].set_ylabel("Held-out R² drop")
    _seed_legend(axes[0], loc="upper left")
    fig.suptitle("Adapted route-specific factor importance — clean WFE set",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    caption = (
        "**Adapted route-specific factor importance (clean WFE set).** One "
        "panel per route, each fitted separately so the large route main effect "
        "cannot dominate the within-route interpretation. Primary measure is "
        "the drop in held-out R² when a raw factor is permuted; markers are the "
        "four seeds, thin bars the within-seed permutation standard deviation "
        "over 100 repeats, black bars the mean over seeds. Panels share one "
        "importance scale. **A route whose outcome is ceiling-limited or whose "
        "model is not estimable is labelled with its status and left blank — it "
        "is never given an artificial zero importance**, and negative held-out "
        "R² is retained rather than suppressed. Each panel title carries the "
        "model statuses observed across seeds.\n\n" + METHOD_NOTE + "\n\n" +
        CLEAN_NOTE + "\n\nNeutral factor palette; red and blue are not used.\n\n"
        + FAITHFUL_NOTE)
    return save_figure(fig, out_dir, "route_specific_factor_importance", caption)


# ------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default=CANONICAL_TABLE)
    ap.add_argument("--out_root",
                    default=os.path.join(REPORT_ROOT, "feature_importance"))
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    root = args.out_root
    d = {k: os.path.join(root, *p) for k, p in {
        "ctl": ("_control",),
        "jt": ("clean_joint", "tables"), "jf": ("clean_joint", "figures"),
        "it": ("clean_interactions", "tables"),
        "if_": ("clean_interactions", "figures"),
        "rt": ("route_specific", "tables"), "rf": ("route_specific", "figures"),
    }.items()}
    for p in d.values():
        os.makedirs(p, exist_ok=True)

    canon = load_canonical(args.canonical)
    items = fi.clean_items(canon)
    train_ids, test_ids = fi.split_items(items)
    written, tables = {}, []

    def _t(df, key, name, sort_by=None):
        path = write_table(df, os.path.join(d[key], name), sort_by=sort_by)
        tables.append(repo_relative(path))
        return path

    print(f"[split] grouped by item: {len(train_ids)} train / "
          f"{len(test_ids)} test items, reused across all four seeds")
    idx = items.set_index("item_id")
    for ids, name in ((train_ids, "fi_train_items.tsv"),
                      (test_ids, "fi_test_items.tsv")):
        sub = idx.loc[ids].reset_index()
        sub.insert(1, "split", "train" if "train" in name else "test")
        _t(sub, "ctl", name, sort_by=["item_id"])

    print("[models] fitting A (main), B (interactions), C (route-specific)")
    r = run_models(canon, train_ids, test_ids)

    # ---------------------------------------------------- Phase 3: clean joint
    main_fit = pd.DataFrame(r["main_fit"])
    _t(main_fit, "jt", "clean_main_model_fit.tsv", sort_by=["seed"])
    _t(pd.concat(r["main_coef"], ignore_index=True), "jt",
       "clean_main_model_coefficients.tsv", sort_by=["seed", "term"])
    summ = pd.concat(r["main_summary"], ignore_index=True)
    _t(summ, "jt", "clean_main_factor_importance.tsv",
       sort_by=["factor", "seed"])
    _t(pd.concat(r["main_reps"], ignore_index=True), "jt",
       "clean_main_factor_importance_repeats.tsv",
       sort_by=["factor", "seed", "repeat"])
    ranks = fi.rank_factors(summ)
    _t(ranks, "jt", "clean_main_factor_ranks.tsv", sort_by=["seed", "rank"])
    _t(fi.rank_stability(ranks), "jt", "clean_main_factor_rank_stability.tsv",
       sort_by=["factor"])
    seed_sum = pd.concat(
        [fi.seed_summary(summ, c) for c in ("r2_drop_mean", "mae_increase_mean")]
        + [fi.seed_summary(main_fit.assign(factor="(model)"), c)
           for c in ("test_r2", "test_mae")], ignore_index=True)
    _t(seed_sum, "jt", "clean_main_seed_summary.tsv",
       sort_by=["quantity", "factor"])
    ez = seed_sum[["factor", "quantity", "seed_values", "mean_over_seeds",
                   "exact_zero_seeds_mean", "seed21_included"]].copy()
    ez["exact_zero_seeds"] = "19; 20; 22"
    ez["replaces_primary"] = False
    _t(ez, "jt", "clean_main_exact_zero_sensitivity.tsv",
       sort_by=["quantity", "factor"])
    written["clean_adapted_factor_importance"] = plot_clean_importance(
        summ, d["jf"])

    # -------------------------------------------------- Phase 4: interactions
    int_fit = pd.DataFrame(r["int_fit"])
    _t(int_fit, "it", "interaction_model_fit.tsv", sort_by=["seed"])
    _t(pd.concat(r["int_coef"], ignore_index=True), "it",
       "interaction_model_coefficients.tsv", sort_by=["seed", "term"])
    inc = int_fit[["seed", "test_r2", "test_mae", "delta_r2_vs_main",
                   "delta_mae_vs_main", "model_status"]].copy()
    inc = inc.merge(main_fit[["seed", "test_r2", "test_mae"]], on="seed",
                    suffixes=("_interaction", "_main"))
    inc["improves_r2"] = inc["delta_r2_vs_main"] > 0
    inc["improves_mae"] = inc["delta_mae_vs_main"] > 0
    inc["positive_means"] = "improvement over the main-effects model"
    _t(inc, "it", "interaction_model_incremental_utility.tsv", sort_by=["seed"])
    drop = pd.DataFrame(r["int_drop"])
    _t(drop, "it", "interaction_block_drop_utility.tsv",
       sort_by=["dropped_block", "seed"])
    iez = fi.seed_summary(int_fit.assign(factor="(interaction model)"),
                          "delta_r2_vs_main")
    iez["exact_zero_seeds"] = "19; 20; 22"
    iez["replaces_primary"] = False
    _t(iez, "it", "interaction_exact_zero_sensitivity.tsv", sort_by=["factor"])

    n_improve = int(inc["improves_r2"].sum())
    decision = {"criterion": "meaningful held-out utility in at least two seeds",
                "seeds_with_positive_delta_r2": n_improve,
                "seeds_with_positive_delta_mae": int(inc["improves_mae"].sum()),
                "figure_created": n_improve >= 2,
                "status": "FIGURE_CREATED" if n_improve >= 2 else fi.NO_FIGURE}
    _t(pd.DataFrame([decision]), "it", "interaction_figure_decision.tsv")
    if decision["figure_created"]:
        print("[interactions] utility in >= 2 seeds -> figure")
        written["interaction_block_utility"] = _plot_interactions(
            drop, inc, d["if_"])
    else:
        print(f"[interactions] {fi.NO_FIGURE}")

    # ------------------------------------------------ Phase 5: route-specific
    rfit = pd.DataFrame(r["route_fit"])
    _t(rfit, "rt", "route_specific_model_fit.tsv", sort_by=["route", "seed"])
    _t(pd.concat(r["route_coef"], ignore_index=True), "rt",
       "route_specific_coefficients.tsv", sort_by=["route", "seed", "term"])
    rsum = pd.concat(r["route_summary"], ignore_index=True)
    _t(rsum, "rt", "route_specific_factor_importance.tsv",
       sort_by=["route", "factor", "seed"])
    _t(pd.concat(r["route_reps"], ignore_index=True) if r["route_reps"]
       else pd.DataFrame(columns=["seed", "route", "factor", "repeat",
                                  "r2_drop", "mae_increase"]),
       "rt", "route_specific_factor_importance_repeats.tsv",
       sort_by=["route", "factor", "seed", "repeat"])
    rranks = fi.rank_factors(rsum.dropna(subset=["r2_drop_mean"]),
                             group_cols=("seed", "route"))
    _t(rranks if len(rranks) else pd.DataFrame(
        columns=["seed", "route", "factor", "r2_drop_mean", "rank"]),
       "rt", "route_specific_factor_ranks.tsv",
       sort_by=["route", "seed", "rank"] if len(rranks) else None)
    rez = pd.concat([fi.seed_summary(rsum[rsum["route"] == rt], "r2_drop_mean")
                     .assign(route=rt) for rt in ROUTES], ignore_index=True)
    rez["exact_zero_seeds"] = "19; 20; 22"
    rez["replaces_primary"] = False
    _t(rez, "rt", "route_specific_exact_zero_sensitivity.tsv",
       sort_by=["route", "factor"])
    written["route_specific_factor_importance"] = plot_route_specific(
        rsum, rfit, d["rf"])

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({"canonical_table": repo_relative(args.canonical),
                       "canonical_table_sha256": sha256_file(args.canonical),
                       "out_root": repo_relative(root),
                       "n_train_items": len(train_ids),
                       "n_test_items": len(test_ids),
                       "split_grouped_by": "item_id",
                       "split_reused_across_seeds": True,
                       "ridge_alpha": fi.RIDGE_ALPHA,
                       "permutation_repeats": fi.PERM_REPEATS,
                       "interaction_figure": decision["status"],
                       "figures": {k: {kk: repo_relative(vv)
                                       for kk, vv in v.items()}
                                   for k, v in written.items()},
                       "tables": sorted(tables),
                       "model_inference_performed": False,
                       "faithful_a11_touched": False}, f, indent=2)
    print(f"\nSprint-5 feature-importance outputs written to "
          f"{repo_relative(root)}")
    return 0


def _plot_interactions(drop: pd.DataFrame, inc: pd.DataFrame,
                       out_dir: str) -> dict:
    blocks = fi.INTERACTION_BLOCKS
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, b in enumerate(blocks):
        sub = drop[drop["dropped_block"] == b]
        for _, r in sub.iterrows():
            ax.plot(i, r["block_r2_utility"],
                    marker=SEED_MARKER[int(r["seed"])], ms=7,
                    color=NEUTRAL_ACCENT, alpha=0.85, zorder=3)
        ax.hlines(float(sub["block_r2_utility"].mean()), i - 0.28, i + 0.28,
                  color="black", lw=2.4, zorder=4)
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels([b.replace("_x_", " × ") for b in blocks], fontsize=8.5)
    ax.axhline(0, color="grey", lw=0.9, ls="--")
    ax.set_ylabel("Held-out R² lost when the block is dropped")
    ax.grid(alpha=0.25, axis="y")
    _seed_legend(ax, loc="best")
    fig.suptitle("Predeclared interaction blocks — held-out drop-block utility",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    caption = (
        "**Held-out drop-block utility of the five predeclared interaction "
        "blocks.** Each value is the held-out R² lost when that block alone is "
        "removed and the model is refitted; positive means the block carries "
        "held-out information. Blocks were **declared before fitting** and no "
        "post-hoc selection was performed; a large coefficient alone is never "
        "treated as evidence that a block matters. One marker per seed, black "
        "bar the mean.\n\n" + METHOD_NOTE + "\n\n" + CLEAN_NOTE +
        "\n\nNeutral palette; red and blue are not used.")
    return save_figure(fig, out_dir, "interaction_block_utility", caption)


if __name__ == "__main__":
    sys.exit(main())
