"""Sprint 3 — frequency figures, tables and audit; full report generator.

    python -m scripts.behavioral_analysis.plot_frequency \
        --out_root reports/behavioral_wfe_fulllexicon_93a577f/frequency

Regenerates every Sprint-3 output from the canonical table alone.  No torch, no
checkpoint, no model inference.

Presentation (frozen): frequency uses a NEUTRAL colour scale.  Red and blue are
reserved for real/pseudo lexicality and never appear here.  All four seed
points are shown; continuous Zipf is primary and high/low secondary; confidence
and gate are drawn on separate panels.
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

from scripts.behavioral_analysis import frequency as fq               # noqa: E402
from scripts.behavioral_analysis.common import (CANONICAL_TABLE,      # noqa: E402
                                                CEILING_SEEDS, REPORT_ROOT,
                                                ROUTE_LABEL, ROUTES, SEEDS,
                                                SEED_MARKER, repo_relative)
from scripts.behavioral_analysis.io import (load_canonical,           # noqa: E402
                                            sha256_file, write_table)
from scripts.behavioral_analysis.plotting import save_figure          # noqa: E402

# Neutral frequency palette — deliberately not red/blue.
NEUTRAL_DARK = "#2f4b57"
NEUTRAL_MID = "#5c8394"
NEUTRAL_LIGHT = "#9dbac5"
PRIMARY = "TRAINED_REAL_FREQUENCY_PRIMARY"
SENSITIVITY = "TRAINED_REAL_FREQUENCY_SENSITIVITY"


def _seed_legend(ax, loc="best"):
    return ax.legend(handles=[
        Line2D([], [], color=NEUTRAL_DARK, marker=SEED_MARKER[s], lw=0, ms=7,
               label=f"seed {s}") for s in SEEDS],
        loc=loc, frameon=False, fontsize=8, ncol=2)


# ------------------------------------------------------------- audit

def distribution_audit(canon: pd.DataFrame, out_dir: str) -> dict:
    """Phase-2 distribution and confound tables."""
    cov = fq.standardized_covariates(canon, PRIMARY)
    verify = fq.verify_frequency_classes(cov)
    z = cov["zipf"].to_numpy(float)
    q = np.percentile(z, [25, 50, 75])
    dist = pd.DataFrame([{
        "dataset_regime": PRIMARY, "n": len(z), "mean": z.mean(),
        "sd": z.std(ddof=1), "min": z.min(), "q25": q[0], "median": q[1],
        "q75": q[2], "max": z.max(),
        "n_low": verify["n_low"], "n_high": verify["n_high"],
        "n_in_excluded_gap": verify["n_in_excluded_gap"],
        "n_mismatched_stored_labels": verify["n_mismatched_labels"],
        "corr_zipf_length": float(np.corrcoef(
            z, cov["phoneme_length"].to_numpy(float))[0, 1]),
    }])
    write_table(dist, os.path.join(
        out_dir, "trained_real_frequency_distribution.tsv"))

    base = canon[(canon["seed"] == SEEDS[0]) & (canon["route"] == "full")]
    prim = base[base["in_TRAINED_REAL_FREQUENCY_PRIMARY"]]
    by_len = (prim.assign(zipf=pd.to_numeric(prim["zipf_frequency"]))
              .groupby("target_length")
              .agg(n_items=("item_id", "nunique"), mean_zipf=("zipf", "mean"),
                   sd_zipf=("zipf", "std"), min_zipf=("zipf", "min"),
                   max_zipf=("zipf", "max"),
                   n_low=("frequency_class", lambda s: int((s == "low").sum())),
                   n_high=("frequency_class", lambda s: int((s == "high").sum())))
              .reset_index().rename(columns={"target_length": "phoneme_length"}))
    write_table(by_len, os.path.join(
        out_dir, "trained_real_frequency_by_length.tsv"))

    by_mor = (prim.assign(zipf=pd.to_numeric(prim["zipf_frequency"]))
              .groupby("morphology")
              .agg(n_items=("item_id", "nunique"), mean_zipf=("zipf", "mean"),
                   sd_zipf=("zipf", "std"),
                   n_low=("frequency_class", lambda s: int((s == "low").sum())),
                   n_high=("frequency_class", lambda s: int((s == "high").sum())))
              .reset_index())
    mor_code = (prim["morphology"] == "simple").astype(float).to_numpy()
    corr_mor = float(np.corrcoef(
        pd.to_numeric(prim["zipf_frequency"]).to_numpy(float), mor_code)[0, 1])
    by_mor["corr_zipf_simple_coding"] = corr_mor
    write_table(by_mor, os.path.join(
        out_dir, "trained_real_frequency_by_morphology.tsv"))

    cells = (prim.groupby(["frequency_class", "target_length"])
             .size().reset_index(name="n_items")
             .rename(columns={"target_length": "phoneme_length"}))
    cells["cell_flag"] = np.where(cells["n_items"] < 10, "VERY_SMALL_CELL",
                                  np.where(cells["n_items"] < 20, "SMALL_CELL",
                                           "OK"))
    write_table(cells, os.path.join(out_dir, "frequency_cell_counts.tsv"),
                sort_by=["frequency_class", "phoneme_length"])

    rows = []
    for regime, label in ((PRIMARY, "TRAINED_REAL_EXACT"),
                          ("UNTRAINED_REAL", "UNTRAINED_REAL")):
        c = fq.standardized_covariates(canon, regime)
        zz = c["zipf"].to_numpy(float)
        rows.append({"exposure_group": label, "n": len(zz),
                     "mean_zipf": zz.mean(), "sd_zipf": zz.std(ddof=1),
                     "min_zipf": zz.min(), "max_zipf": zz.max(),
                     "mean_phoneme_length": c["phoneme_length"].mean(),
                     "corr_zipf_length": float(np.corrcoef(
                         zz, c["phoneme_length"].to_numpy(float))[0, 1])})
    write_table(pd.DataFrame(rows), os.path.join(
        out_dir, "frequency_exposure_comparison.tsv"))
    return verify


# ---------------------------------------------------- primary figure

def plot_primary(slopes: pd.DataFrame, boot: pd.DataFrame,
                 hilo: pd.DataFrame, out_dir: str) -> dict:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for ax, route in zip(axes[:3], ROUTES):
        sub = slopes[slopes["route"] == route]
        for _, r in sub.iterrows():
            ax.plot(0, r["zipf_slope"], marker=SEED_MARKER[int(r["seed"])],
                    ms=10, color=NEUTRAL_DARK, alpha=0.85, zorder=3)
        m = float(sub["zipf_slope"].mean())
        ax.hlines(m, -0.22, 0.22, color="black", lw=2.4, zorder=4)
        b = boot[(boot["route"] == route) & (boot["quantity"] == "zipf_slope")]
        if len(b):
            ax.vlines(0, b["ci_low"].iloc[0], b["ci_high"].iloc[0],
                      color=NEUTRAL_MID, lw=3.5, alpha=0.45, zorder=2)
        ax.axhline(0, color="grey", lw=0.9, ls="--")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(ROUTE_LABEL[route], fontsize=12)
        ax.grid(alpha=0.25, axis="y")
        if all(s == "ALL_ZERO_OUTCOME" for s in sub["model_status"]):
            ax.text(0, 0, "  ceiling:\n  ALL_ZERO_OUTCOME", ha="center",
                    va="bottom", fontsize=8, color="0.35")
    axes[0].set_ylabel("Zipf slope  (edit distance per SD of Zipf)")
    _seed_legend(axes[0], loc="lower left")
    axes[1].legend(handles=[
        Line2D([], [], color="black", lw=2.4, label="mean over seeds"),
        Line2D([], [], color=NEUTRAL_MID, lw=3.5, alpha=0.6,
               label="95% hierarchical bootstrap")],
        loc="lower left", frameon=False, fontsize=8)

    ax = axes[3]
    for i, route in enumerate(ROUTES):
        sub = hilo[hilo["route"] == route]
        for _, r in sub.iterrows():
            ax.plot(i, r["high_low_contrast_raw_edit_distance"],
                    marker=SEED_MARKER[int(r["seed"])], ms=8,
                    color=NEUTRAL_MID, alpha=0.85)
        ax.hlines(float(sub["high_low_contrast_raw_edit_distance"].mean()),
                  i - 0.2, i + 0.2, color="black", lw=2.2, zorder=4)
    ax.axhline(0, color="grey", lw=0.9, ls="--")
    ax.set_xticks(range(len(ROUTES)))
    ax.set_xticklabels([ROUTE_LABEL[r] for r in ROUTES])
    ax.set_title("secondary: low − high", fontsize=10)
    ax.set_ylabel("mean edit-distance difference")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Frequency effect by route — trained real words "
                 "(continuous Zipf primary)", y=1.02, fontsize=13)
    fig.tight_layout()
    ceiling = sorted({ROUTE_LABEL[r] for r in ROUTES
                      if (slopes[slopes["route"] == r]["model_status"]
                          == "ALL_ZERO_OUTCOME").all()})
    caption = (
        "**Frequency effect by route (trained real words).** Only WFE real "
        "words encountered during training with the same phonological form are "
        "included (n = 671); pronunciation variants and untrained real words "
        "are analysed separately, and pseudowords are never assigned a "
        "frequency. Panels 1–3 show the **primary** continuous-Zipf slope of "
        "raw Levenshtein edit distance on standardized Zipf, one marker per "
        "seed, the black bar the mean over seeds and the shaded bar the 95% "
        "hierarchical bootstrap interval (B = 10,000, random seed 20260730). "
        "**A negative slope means higher-frequency words have fewer errors.** "
        "Panel 4 is the secondary descriptive low-minus-high contrast, where a "
        "positive value means low-frequency words are harder."
        + (f" {' and '.join(ceiling)} produce zero errors on every item in "
           "every seed (ALL_ZERO_OUTCOME): the slope there is structurally "
           "zero and the measure cannot identify a frequency effect — this is "
           "not evidence that those routes lack frequency information."
           if ceiling else "")
        + " Colours are a neutral scale; red and blue are reserved for "
          "real/pseudo lexicality and are not used here.")
    return save_figure(fig, out_dir, "trained_real_frequency_by_route", caption)


# ------------------------------------------- confidence / gate figure

def plot_confidence_gate(cg: pd.DataFrame, boot: pd.DataFrame,
                         out_dir: str) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    panels = (("lexical_confidence", "Lexical confidence"), ("gate", "Gate g"))
    for ax, (outcome, title) in zip(axes, panels):
        sub = cg[cg["outcome"] == outcome]
        for _, r in sub.iterrows():
            ax.plot(0, r["zipf_slope"], marker=SEED_MARKER[int(r["seed"])],
                    ms=10, color=NEUTRAL_DARK, alpha=0.85, zorder=3)
        ax.hlines(float(sub["zipf_slope"].mean()), -0.22, 0.22, color="black",
                  lw=2.4, zorder=4)
        b = boot[boot["outcome"] == outcome]
        if len(b):
            ax.vlines(0, b["ci_low"].iloc[0], b["ci_high"].iloc[0],
                      color=NEUTRAL_MID, lw=3.5, alpha=0.45, zorder=2)
        ax.axhline(0, color="grey", lw=0.9, ls="--")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Zipf slope (per SD of Zipf)")
        ax.grid(alpha=0.25, axis="y")
    _seed_legend(axes[0], loc="best")
    axes[1].legend(handles=[
        Line2D([], [], color="black", lw=2.4, label="mean over seeds"),
        Line2D([], [], color=NEUTRAL_MID, lw=3.5, alpha=0.6,
               label="95% hierarchical bootstrap")],
        loc="best", frameon=False, fontsize=8)
    fig.suptitle("Frequency slopes for lexical confidence and gate — "
                 "trained real words", y=1.02, fontsize=12)
    fig.tight_layout()
    caption = (
        "**Frequency slopes for lexical confidence and gate (trained real "
        "words, n = 671).** Lexical confidence is the **top-1 cosine "
        "similarity between the encoded form and the semantic bank** — it is "
        "not a word probability and not phonological similarity. The gate is a "
        "word-level scalar belonging to the FULL-route computation and is a "
        "deterministic monotonic transform of that confidence "
        "(g = sigmoid(2.0·(confidence − 0.7))), so **the two panels are linked "
        "outcomes, not independent findings**; they are drawn on separate axes "
        "for that reason. One marker per seed, black bar the mean, shaded bar "
        "the 95% hierarchical bootstrap interval. Neutral colours: red and "
        "blue are reserved for lexicality.")
    return save_figure(fig, out_dir, "frequency_confidence_gate", caption)


# --------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", default=CANONICAL_TABLE)
    ap.add_argument("--out_root",
                    default=os.path.join(REPORT_ROOT, "frequency"))
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    root = args.out_root
    shared, prim_f, prim_t = (os.path.join(root, "tables"),
                              os.path.join(root, "primary", "figures"),
                              os.path.join(root, "primary", "tables"))
    cg_f, cg_t = (os.path.join(root, "gate_confidence", "figures"),
                  os.path.join(root, "gate_confidence", "tables"))
    sens = os.path.join(root, "sensitivity")
    for p in (shared, prim_f, prim_t, cg_f, cg_t, sens):
        os.makedirs(p, exist_ok=True)

    canon = load_canonical(args.canonical)
    written = {}

    print("[audit] frequency distribution and confounds")
    verify = distribution_audit(canon, shared)

    print("[primary] continuous Zipf slopes")
    slopes = fq.continuous_slopes(canon, PRIMARY)
    write_table(slopes, os.path.join(prim_t, "trained_real_frequency_slopes.tsv"),
                sort_by=["route", "seed"])
    rc = fq.route_contrasts(slopes)
    write_table(rc, os.path.join(
        prim_t, "trained_real_frequency_route_contrasts.tsv"),
        sort_by=["route_contrast", "seed"])
    boot = fq.bootstrap_slopes(canon, PRIMARY)
    write_table(boot, os.path.join(
        prim_t, "trained_real_frequency_bootstrap.tsv"),
        sort_by=["quantity", "route"])
    hilo = fq.high_low_contrasts(canon, PRIMARY)
    write_table(hilo, os.path.join(
        prim_t, "trained_real_high_low_descriptives.tsv"),
        sort_by=["route", "seed"])
    zl = fq.zipf_length_models(canon, PRIMARY)
    write_table(zl, os.path.join(prim_t, "trained_real_zipf_length_models.tsv"),
                sort_by=["route", "seed"])
    print("[primary] figure")
    written["trained_real_frequency_by_route"] = plot_primary(
        slopes, boot, hilo, prim_f)

    print("[gate/confidence] slopes and figure")
    cg = fq.confidence_gate_slopes(canon, PRIMARY)
    write_table(cg[cg["outcome"] == "lexical_confidence"],
                os.path.join(cg_t, "frequency_confidence_slopes.tsv"),
                sort_by=["seed"])
    write_table(cg[cg["outcome"] == "gate"],
                os.path.join(cg_t, "frequency_gate_slopes.tsv"),
                sort_by=["seed"])
    cgb = fq.bootstrap_confidence_gate(canon, PRIMARY)
    write_table(cgb, os.path.join(
        cg_t, "frequency_confidence_gate_bootstrap.tsv"), sort_by=["outcome"])
    written["frequency_confidence_gate"] = plot_confidence_gate(cg, cgb, cg_f)

    print("[sensitivity] pronunciation variants, exact-zero seeds, untrained")
    sv = fq.continuous_slopes(canon, SENSITIVITY)
    sv_sum = fq.summarise(sv, "zipf_slope", ["dataset_regime", "route"])
    prim_sum = fq.summarise(slopes, "zipf_slope", ["dataset_regime", "route"])
    write_table(pd.concat([prim_sum, sv_sum], ignore_index=True),
                os.path.join(sens, "pronunciation_variant_sensitivity.tsv"),
                sort_by=["dataset_regime", "route"])
    write_table(prim_sum, os.path.join(sens, "exact_zero_seed_sensitivity.tsv"),
                sort_by=["route"])
    un = fq.continuous_slopes(canon, "UNTRAINED_REAL")
    write_table(un, os.path.join(sens, "untrained_real_frequency_slopes.tsv"),
                sort_by=["route", "seed"])
    un_hl = fq.high_low_contrasts(canon, "UNTRAINED_REAL")
    write_table(un_hl, os.path.join(sens, "untrained_real_high_low.tsv"),
                sort_by=["route", "seed"])

    base = canon[(canon["seed"] == SEEDS[0]) & (canon["route"] == "full")]
    allreal = base[(base["source_lexicality"] == "real")
                   & base["in_FAITHFUL_WFE_ALL"]]
    desc = (allreal.assign(zipf=pd.to_numeric(allreal["zipf_frequency"]))
            .groupby("lichtheim_exposure_status")
            .agg(n_items=("item_id", "nunique"), mean_zipf=("zipf", "mean"),
                 sd_zipf=("zipf", "std"), min_zipf=("zipf", "min"),
                 max_zipf=("zipf", "max"),
                 n_low=("frequency_class", lambda s: int((s == "low").sum())),
                 n_high=("frequency_class", lambda s: int((s == "high").sum())))
            .reset_index())
    desc["note"] = ("descriptive only; the mixed 800-item set is never the "
                    "primary Lichtheim3 frequency claim")
    write_table(desc, os.path.join(
        sens, "faithful_all_real_frequency_description.tsv"),
        sort_by=["lichtheim_exposure_status"])

    print("[word error] tables and model status")
    we = fq.high_low_contrasts(canon, PRIMARY)[
        ["dataset_regime", "seed", "route", "n_low", "n_high",
         "low_mean_word_error", "high_mean_word_error",
         "high_low_contrast_word_error"]]
    write_table(we, os.path.join(
        shared, "trained_real_frequency_word_error.tsv"),
        sort_by=["route", "seed"])
    status_rows = []
    for route in ROUTES:
        sub = slopes[slopes["route"] == route]
        n_err = int(sub["n_erroneous_items"].sum())
        if n_err == 0:
            st, fit = "ALL_ZERO_OUTCOME", False
            why = ("zero word errors in every seed: no finite logistic MLE "
                   "exists; fit not attempted; no absence of frequency "
                   "encoding is claimed")
        elif n_err < len(SEEDS) * 10:
            st, fit = "SPARSE_ERROR_LIMITED", False
            why = ("too few errors for a stable logistic fit; descriptive "
                   "high/low rates and bootstrap contrasts reported instead")
        else:
            st, fit = "OK", True
            why = "both outcome classes present with sufficient errors"
        status_rows.append({"dataset_regime": PRIMARY, "route": route,
                            "n_erroneous_items_total_over_seeds": n_err,
                            "model_status": st, "logistic_fit_attempted": fit,
                            "reason": why})
    write_table(pd.DataFrame(status_rows), os.path.join(
        shared, "frequency_word_error_model_status.tsv"), sort_by=["route"])

    if args.manifest:
        with open(args.manifest, "w") as f:
            json.dump({"canonical_table": repo_relative(args.canonical),
                       "canonical_table_sha256": sha256_file(args.canonical),
                       "out_root": repo_relative(root),
                       "frequency_class_verification": verify,
                       "figures": {k: {kk: repo_relative(vv)
                                       for kk, vv in v.items()}
                                   for k, v in written.items()},
                       "model_inference_performed": False}, f, indent=2)
    print(f"\nSprint-3 frequency outputs written to {repo_relative(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
