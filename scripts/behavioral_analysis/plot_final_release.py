"""Final release — render A09/A10/A11 and assemble the publication tree.

    python -m scripts.behavioral_analysis.plot_final_release \
        --out_root reports/behavioral_wfe_fulllexicon_93a577f/final_release

Editorial only.  The three faithful figures are drawn **from their stored
authoritative tables**, so no estimate is recomputed and A11's frozen Ridge /
split / permutation / sign policy is preserved by construction.  Every other
final figure is a byte-identical copy of an already-validated file.

Visual conventions are inherited unchanged: Real = red, Pseudoword = blue and
those two colours encode nothing else; morphology is line style (complex solid,
simple dashed); feature importance uses a neutral palette; captions carry the
science and plot titles do not.
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

from scripts.behavioral_analysis import final_release as fr             # noqa: E402
from scripts.behavioral_analysis.common import (LENGTHS,                # noqa: E402
                                                LEXICALITY_COLOR,
                                                LEXICALITY_LABEL,
                                                REPORT_ROOT, SEED_MARKER,
                                                SEEDS, repo_relative)
from scripts.behavioral_analysis.io import write_table                  # noqa: E402
from scripts.behavioral_analysis.plotting import save_figure            # noqa: E402

SOURCE_LABEL = {"real": "Real (source label)", "pseudo": "Pseudoword (source label)"}
NEUTRAL = "#2f3b40"
NEUTRAL_MID = "#5c6b70"

FAITHFUL_CAVEAT = (
    "This is a **faithful** stimulus-level replication on all 1,200 original "
    "WFE items with their **source** Real/Pseudo labels, FULL route only. "
    "**Source labels are not training exposure**: 122 of the 800 source-real "
    "words were never in the Lichtheim3 training lexicon and 9 source "
    "pseudowords collide with it, so these panels must not be read as "
    "trained-versus-novel effects. The exposure-audited analyses are the "
    "adapted clean-set figures.")

RENDER_NOTE = (
    "Rendered for this release directly from the stored, previously validated "
    "table; **no value was recomputed**.")


def _lex_legend(ax, loc="upper left"):
    return ax.legend(handles=[
        Line2D([], [], color=LEXICALITY_COLOR[k], lw=2.6, label=SOURCE_LABEL[k])
        for k in ("real", "pseudo")], loc=loc, frameon=False, fontsize=8)


# ------------------------------------------------------------------- A09

def plot_a09(tab: pd.DataFrame, out_dir: str) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    ymax = float(tab["mean_edit_distance"].max()) * 1.15 or 1.0
    for ax, mor, style in zip(axes, ("complex", "simple"), ("-", "--")):
        for lex in ("real", "pseudo"):
            sub = tab[(tab["morphology"] == mor) & (tab["lexicality"] == lex)]
            for s in SEEDS:
                ss = sub[sub["seed"] == s].sort_values("length")
                ax.plot(ss["length"], ss["mean_edit_distance"], ls=":", lw=0.7,
                        marker=SEED_MARKER[s], ms=3.2, alpha=0.45,
                        color=LEXICALITY_COLOR[lex])
            m = sub.groupby("length", as_index=False)["mean_edit_distance"].mean()
            ax.plot(m["length"], m["mean_edit_distance"], ls=style, lw=2.6,
                    marker="o", ms=6.5, color=LEXICALITY_COLOR[lex])
        ax.set_title(f"{mor.capitalize()} ({'solid' if mor == 'complex' else 'dashed'})",
                     fontsize=11)
        ax.set_xlabel("Phoneme length")
        ax.set_xticks(LENGTHS)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean raw Levenshtein edit distance")
    _lex_legend(axes[0])
    axes[1].legend(handles=[
        Line2D([], [], color="0.35", ls=":", lw=0.8, marker="o", ms=3.2,
               label="individual seed"),
        Line2D([], [], color="0.35", lw=2.6, marker="o", ms=6.5,
               label="mean over seeds")], loc="upper left", frameon=False,
        fontsize=8)
    fig.suptitle("Faithful Figure 2A — edit distance by source lexicality, "
                 "morphology and length", y=1.02, fontsize=12.5)
    fig.tight_layout()
    caption = (
        "**Faithful Figure 2A (A09).** Mean raw Levenshtein edit distance "
        "against phoneme length for the FULL route, split by source lexicality "
        "(red = Real, blue = Pseudoword) and by morphology, with the frozen "
        "Dager line styles — complex solid, simple dashed. Thin dotted lines "
        "are the four individual seeds (19, 20, 21, 22); thick lines are the "
        "mean over seeds. Length 6 is absent from the WFE by construction. "
        "Errors appear only at the long lengths: at length 9 the source-pseudo "
        "mean is 0.0947 and the source-real mean 0.0199, while lengths 3, 4 and "
        "5 are exactly zero in both conditions.\n\n" + FAITHFUL_CAVEAT +
        "\n\n" + RENDER_NOTE)
    return save_figure(fig, out_dir, fr.LEGACY_STEM["A09"], caption)


# ------------------------------------------------------------------- A10

def plot_a10(tab: pd.DataFrame, out_dir: str) -> dict:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for lex in ("real", "pseudo"):
        sub = tab[tab["lexicality"] == lex]
        agg = (sub.groupby("relative_position", as_index=False)
               ["error_rate_per_item"].mean().sort_values("relative_position"))
        for L in sorted(sub["length"].unique()):
            ss = sub[sub["length"] == L].sort_values("relative_position")
            ax.plot(ss["relative_position"], ss["error_rate_per_item"],
                    lw=0.7, alpha=0.35, color=LEXICALITY_COLOR[lex])
        ax.plot(agg["relative_position"], agg["error_rate_per_item"],
                lw=2.8, marker="o", ms=6, color=LEXICALITY_COLOR[lex])
    ax.set_xlabel("Relative position in the word")
    ax.set_ylabel("Positional errors per item")
    ax.grid(alpha=0.25)
    _lex_legend(ax)
    ax.legend(handles=[
        Line2D([], [], color="0.35", lw=0.7, alpha=0.6, label="individual length"),
        Line2D([], [], color="0.35", lw=2.8, marker="o", ms=6,
               label="mean over lengths")], loc="upper center", frameon=False,
        fontsize=8)
    fig.suptitle("Faithful Figure 2C — serial-position error profile",
                 y=1.0, fontsize=12.5)
    fig.tight_layout()
    caption = (
        "**Faithful Figure 2C (A10).** Positional error rate per item against "
        "relative position in the word, FULL route, split by source lexicality "
        "(red = Real, blue = Pseudoword). Thin lines are the individual "
        "phoneme lengths (3, 4, 5, 7, 8, 9); thick lines are the mean over "
        "lengths. Errors concentrate at the late positions: at length 9 the "
        "source-pseudo rate reaches 0.0417 at the final position against "
        "0.0087 for source-real. Overall means are 0.00295 (pseudo) and "
        "0.00129 (real).\n\n"
        "Method, frozen: **faithful zip-mismatch positions (Dager "
        "`Error_Indices`) with no Levenshtein alignment**, pooled across the "
        "four seeds by design (`n_items_x_seeds`). This is a **different "
        "estimand** from the adapted clean serial-position figure, which uses "
        "the exposure-audited 1,062-item set and is reported separately; the "
        "two are never merged.\n\n" + FAITHFUL_CAVEAT + "\n\n" + RENDER_NOTE)
    return save_figure(fig, out_dir, fr.LEGACY_STEM["A10"], caption)


# ------------------------------------------------------------------- A11

def plot_a11(tab: pd.DataFrame, out_dir: str) -> dict:
    feats = ["cont__Length", "cat__Lexicality_real", "cat__Morphology_simple"]
    nice = {"cont__Length": "Length", "cat__Lexicality_real": "Lexicality (real)",
            "cat__Morphology_simple": "Morphology (simple)"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, col, title in zip(axes, ("permutation_importance_mean",
                                     "ridge_coefficient"),
                              ("Permutation importance",
                               "Ridge coefficient")):
        for i, f in enumerate(feats):
            sub = tab[tab["feature_transformed"] == f]
            for _, r in sub.iterrows():
                ax.plot(i, r[col], marker=SEED_MARKER[int(r["seed"])], ms=8,
                        color=NEUTRAL, alpha=0.85, zorder=3)
            ax.hlines(float(sub[col].mean()), i - 0.26, i + 0.26, color="black",
                      lw=2.4, zorder=4)
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels([nice[f] for f in feats], fontsize=9)
        ax.axhline(0, color="grey", lw=0.9, ls="--")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Permutation importance (historical convention)")
    axes[1].set_ylabel("Ridge coefficient")
    axes[0].legend(handles=[
        Line2D([], [], color=NEUTRAL, marker=SEED_MARKER[s], lw=0, ms=7,
               label=f"seed {s}") for s in SEEDS]
        + [Line2D([], [], color="black", lw=2.4, label="mean over seeds")],
        loc="best", frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Faithful Figure 2B — feature importance (FAITHFUL, "
                 "not the adapted analysis)", y=1.02, fontsize=12.5)
    fig.tight_layout()
    caption = (
        "**Faithful Figure 2B (A11).** Ridge coefficients and permutation "
        "importance for the three original Dager features on all 1,200 "
        "source-labelled WFE items, FULL route. One marker per seed, black bar "
        "the mean over seeds. **Length leads in all four seeds** (permutation "
        "importance 0.0299, 0.0027, 0.0255, 0.0017); lexicality and morphology "
        "are near zero and change sign across seeds.\n\n"
        "Frozen parameters, preserved exactly: **Ridge alpha = 1.0**, 80/20 "
        "split `random_state = 42`, `permutation_importance(n_repeats = 100, "
        "random_state = 42)`, no interactions, no p-values, and the "
        "**historical signed convention**. No model was refitted for this "
        "release.\n\n"
        "**This is the FAITHFUL analysis (A11), not the adapted A15.** The two "
        "estimate different quantities on different item sets with different "
        "route scope, split unit and permutation semantics; they are never "
        "pooled and are never placed on a common quantitative axis. Neutral "
        "colours are used deliberately — red and blue are reserved for "
        "real/pseudo observations and would misread here.\n\n"
        + FAITHFUL_CAVEAT + "\n\n" + RENDER_NOTE)
    return save_figure(fig, out_dir, fr.LEGACY_STEM["A11"], caption)


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",
                    default=os.path.join(REPORT_ROOT, "final_release"))
    ap.add_argument("--spec", default=fr.SPEC_JSON)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    root = args.out_root
    d = {k: os.path.join(root, *p) for k, p in {
        "fe_f": ("formatted_existing", "figures"),
        "fe_t": ("formatted_existing", "tables"),
        "fig_m": ("figures", "main"), "fig_s": ("figures", "supplementary"),
        "cap_m": ("captions", "main"), "cap_s": ("captions", "supplementary"),
        "tab": ("tables",), "ctl": ("_control",),
    }.items()}
    for p in d.values():
        os.makedirs(p, exist_ok=True)

    spec = fr.load_spec(args.spec)
    written, copies = {}, []  # noqa: E501

    # ------------------------------------------- A09 / A10 / A11 formatting
    print("[formatted_existing] rendering A09/A10/A11 from stored tables")
    renderers = {"A09": plot_a09, "A10": plot_a10, "A11": plot_a11}
    for row in fr.FORMATTED_ROWS:
        tab = fr.legacy_table(row)
        # The plotting table is a BYTE-IDENTICAL copy of the authoritative
        # source, not a re-serialization: re-writing it through pandas would
        # reformat floats and make the release table differ byte-wise from the
        # table every earlier validation hashed.
        copies.append(fr.copy_with_provenance(
            os.path.join(fr.LEGACY_DIR, fr.LEGACY_TABLES[row]),
            os.path.join(d["fe_t"], fr.LEGACY_TABLES[row])))
        # the caption stays beside its figure, so the release-copy step finds
        # <stem>_caption.md next to <stem>.png exactly as for every other figure
        written[row] = renderers[row](tab, d["fe_f"])

    # ------------------------------------------------- selected release copies
    print("[figures] copying selected MAIN and SUPPLEMENTARY figures")
    index = fr.select_figures(spec)
    for r in index:
        src_stem = r["source_stem"]
        fig_dir = d["fig_m"] if r["category"] == "MAIN" else d["fig_s"]
        cap_dir = d["cap_m"] if r["category"] == "MAIN" else d["cap_s"]
        for ext in ("png", "pdf", "svg"):
            src = f"{src_stem}.{ext}"
            if not os.path.exists(src):
                raise FileNotFoundError(f"missing source figure: {src}")
            copies.append(fr.copy_with_provenance(
                src, os.path.join(fig_dir, f"{r['release_stem']}.{ext}")))
        cap = f"{src_stem}_caption.md"
        if not os.path.exists(cap):
            raise FileNotFoundError(f"missing source caption: {cap}")
        copies.append(fr.copy_with_provenance(
            cap, os.path.join(cap_dir, f"{r['release_stem']}_caption.md")))

    # ---------------------------------------------------------------- indexes
    print("[tables] figure index and summary tables")
    idx = pd.DataFrame(index + fr.excluded_figures())
    write_table(idx, os.path.join(d["tab"], "final_figure_index.tsv"))
    write_table(fr.status_summary(spec),
                os.path.join(d["tab"], "analysis_status_summary.tsv"),
                sort_by=["analysis_id"])
    write_table(fr.dataset_regime_summary(),
                os.path.join(d["tab"], "dataset_regime_summary.tsv"),
                sort_by=["dataset_regime"])
    prov = json.load(open(os.path.join(
        REPORT_ROOT, "behavioral_analysis_provenance.json")))
    write_table(fr.checkpoint_summary(prov),
                os.path.join(d["tab"], "checkpoint_summary.tsv"),
                sort_by=["seed"])

    manifest = {
        "release": "final publication and closure",
        "spec": repo_relative(args.spec),
        "formatted_existing": {k: {kk: repo_relative(vv)
                                   for kk, vv in v.items()}
                               for k, v in written.items()},
        "legacy_source_sha256": fr.legacy_source_hashes(),
        "n_main_figures": sum(r["category"] == "MAIN" for r in index),
        "n_supplementary_figures": sum(r["category"] == "SUPPLEMENTARY"
                                       for r in index),
        "release_copies": copies,
        "all_copies_identical": all(c["equality"] == "IDENTICAL"
                                    for c in copies),
        "sources_moved_or_deleted": False,
        "model_inference_performed": False,
        "new_scientific_value_added": False,
        "ssp_status": fr.SSP_STATUS,
        "composite_figure_created": False,
    }
    path = args.manifest or os.path.join(d["ctl"], "final_release_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"\nFinal release written to {repo_relative(root)} "
          f"({manifest['n_main_figures']} main, "
          f"{manifest['n_supplementary_figures']} supplementary, "
          f"{len(copies)} copies all identical="
          f"{manifest['all_copies_identical']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
