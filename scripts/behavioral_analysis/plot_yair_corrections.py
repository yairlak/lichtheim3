"""Working figures for the Yair corrections pass.

Every value plotted is read from a table written by `yair_corrections.py`; no
figure recomputes a statistic.  Presentation conventions follow `plotting.py`:
red = Real words, blue = Pseudowords and those two colours encode nothing else,
exposure categories use the neutral grey palette, and each figure is written as
PNG/PDF/SVG next to a standalone caption.

These are **working** figures for discussion.  They are not promoted into the
final release by this module.
"""
from __future__ import annotations

import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .common import (EXPOSURE_ACCENT, EXPOSURE_COLOR, EXPOSURE_ORDER, LENGTHS,
                     LEXICALITY_COLOR, LEXICALITY_LABEL, ROUTE_LABEL, ROUTES,
                     SEED_MARKER, SEEDS, SERIAL_POSITION_METHOD)
from .plotting import save_figure
from .yair_corrections import CORRECTIONS_ROOT, SUCCESS_GROUPS, out_path

FIG_DIR = os.path.join(CORRECTIONS_ROOT, "figures")
TAB_DIR = os.path.join(CORRECTIONS_ROOT, "tables")

LENGTH_NOTE = ("WFE contains no 6-phoneme items by construction; the gap on "
               "the x-axis is real and is not interpolated across.")


def _tab(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TAB_DIR, name), sep="\t")


# ------------------------------------------------- yc1 word error by length

def figure_word_error_by_length() -> Dict[str, str]:
    seed_t = _tab("word_error_by_length_seed.tsv")
    summ = _tab("word_error_by_length_summary.tsv")
    counts = _tab("word_error_by_length_item_counts.tsv")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.9), sharey=True)
    for ax, r in zip(axes, ROUTES):
        for lex in ("real", "pseudo"):
            col = LEXICALITY_COLOR[lex]
            s = summ[(summ["route"] == r) & (summ["source_lexicality"] == lex)]
            s = s.sort_values("phoneme_length")
            x = s["phoneme_length"].to_numpy(float)
            ax.fill_between(x, s["ci_low"], s["ci_high"], color=col, alpha=.12,
                            lw=0, zorder=1)
            ax.plot(x, s["mean_word_error_rate_across_seeds"], "-", color=col,
                    lw=2.4, zorder=3)
            for sd in SEEDS:
                d = seed_t[(seed_t["route"] == r)
                           & (seed_t["source_lexicality"] == lex)
                           & (seed_t["seed"] == sd)].sort_values("phoneme_length")
                ax.plot(d["phoneme_length"], d["word_error_rate"],
                        SEED_MARKER[sd], color=col, ms=4.2, mfc="none",
                        mew=1.1, alpha=.85, zorder=4)
        ax.set_title(ROUTE_LABEL[r], fontsize=12, pad=8)
        ax.set_xticks(LENGTHS)
        ax.set_xlabel("target length (phonemes)", fontsize=9.5)
        ax.set_xlim(2.6, 9.4)
        ax.grid(alpha=.22, lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("whole-word error rate", fontsize=10.5)
    axes[0].set_ylim(-0.02, 0.62)

    # Item counts are a property of the stimulus set, identical in all three
    # route panels, so they are printed once under the leftmost panel only.
    ax0 = axes[0]
    ax0.annotate("n items per length bin (identical in all three panels):",
                 (0.0, -0.30), xycoords="axes fraction", fontsize=8,
                 color="#333", ha="left", annotation_clip=False)
    for lex, dy, lab in (("real", -0.375, "real"),
                         ("pseudo", -0.435, "pseudo")):
        ax0.annotate(f"{lab}", (-0.02, dy), xycoords="axes fraction",
                     fontsize=8, color=LEXICALITY_COLOR[lex], ha="right",
                     annotation_clip=False)
        for L in LENGTHS:
            n = counts[(counts["source_lexicality"] == lex)
                       & (counts["phoneme_length"] == L)]["n_items"]
            if len(n):
                ax0.annotate(f"{int(n.iloc[0])}", (L, dy),
                             xycoords=("data", "axes fraction"), ha="center",
                             fontsize=8, color=LEXICALITY_COLOR[lex],
                             annotation_clip=False)

    lex_handles = [Line2D([], [], color=LEXICALITY_COLOR[k], lw=2.6,
                          label=f"{LEXICALITY_LABEL[k]} — across-seed mean")
                   for k in ("real", "pseudo")]
    lex_handles += [
        Patch(facecolor="#999", alpha=.30,
              label="95 % bootstrap interval on the across-seed mean"),
        Line2D([], [], color="#555", lw=0, marker="o", mfc="none", ms=6,
               label="one seed (open markers: o 19, s 20, ^ 21, D 22)")]
    axes[2].legend(handles=lex_handles, fontsize=7.8, loc="upper left",
                   frameon=True, framealpha=.93, handlelength=2.2)
    fig.suptitle("Whole-word error rate by exact target length "
                 "(LICHTHEIM_CLEAN: 671 trained real, 391 novel pseudo)",
                 fontsize=12.5, y=1.005)
    fig.tight_layout(rect=(0, 0.13, 1, 1))

    caption = f"""
# Whole-word error rate by exact target length and route

**Primary metric: whole-word error rate** (1 - exact match), not edit distance
and not a slope. Population: `LICHTHEIM_CLEAN`, 671 `TRAINED_REAL_EXACT` (red)
and 391 `NOVEL_PSEUDOWORD` (blue). Open markers are the four individual seeds
(19 circle, 20 square, 21 triangle, 22 diamond); the solid line is the
across-seed mean; the band is the frozen `cell_mean_bootstrap` 95 % interval
(B = 10,000, seed 20260730, seeds resampled then items). Item counts per
lexicality x length bin are printed under each panel. No smoothing is applied.

{LENGTH_NOTE}

**What the panels show.** FULL and WM sit at the floor almost everywhere: zero
whole-word errors for real words at every length, and zero for pseudowords up to
length 8, with the first non-zero values appearing only at length 9 (FULL 0.057,
WM 0.072). LTM is the only route with a substantial length effect, and it is
confined to pseudowords: 0.074 at length 3 rising monotonically to 0.534 at
length 9, while LTM real words stay at or near zero (maximum 0.027 at length 9).

Reading the same data as mean edit distance compresses this into a shallow slope
and hides the fact that two of the three routes are at ceiling. That is why word
error rate is the primary axis here.

Mean raw edit distance is carried in
`tables/word_error_by_length_seed.tsv` and
`tables/word_error_by_length_summary.tsv` as
`mean_raw_edit_distance_severity_only`. It describes **how severe** a failure is
once a word is already wrong; it is not a second measure of how often the model
fails.

**Limitation.** Inside `LICHTHEIM_CLEAN`, lexicality and training exposure
coincide exactly (real == trained-exact, pseudo == novel), so this figure cannot
separate the two. The faithful-population companion table
`tables/word_error_by_length_faithful_companion.tsv` shows the different mixture
the faithful labels produce; the two populations are never pooled.
"""
    return save_figure(fig, FIG_DIR, "yc1_word_error_by_length", caption)


# --------------------------------------- yc2 faithful source-real error audit

def figure_faithful_real_errors() -> Dict[str, str]:
    by_exp = _tab("faithful_real_error_by_exposure.tsv")
    summ = _tab("faithful_real_error_summary.tsv")
    rec = _tab("faithful_real_error_recurrence.tsv")

    present = [e for e in EXPOSURE_ORDER
               if e in set(by_exp[by_exp["n_error_events"] > 0]
                           ["lichtheim_exposure_status"])]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))

    ax = axes[0]
    x = np.arange(len(ROUTES))
    ax.bar(x - 0.19, [summ[summ.route == r]["n_error_events_seed_x_item"].iloc[0]
                      for r in ROUTES], 0.36, color=EXPOSURE_COLOR,
           label="error EVENTS (seed x item; an item failing in\n"
                 "3 seeds counts 3 times)")
    ax.bar(x + 0.19, [summ[summ.route == r]["n_unique_erroneous_items"].iloc[0]
                      for r in ROUTES], 0.36, color=EXPOSURE_ACCENT,
           label="unique ITEMS ever wrong (counted once,\n"
                 "however many seeds fail)")
    for i, r in enumerate(ROUTES):
        row = summ[summ.route == r].iloc[0]
        ax.annotate(f"{int(row['n_error_events_seed_x_item'])}",
                    (i - 0.19, row["n_error_events_seed_x_item"]),
                    ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{int(row['n_unique_erroneous_items'])}",
                    (i + 0.19, row["n_unique_erroneous_items"]),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([ROUTE_LABEL[r] for r in ROUTES])
    ax.set_ylabel("count", fontsize=10)
    ax.set_title("A  How many of the 800 source-real items fail",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    ax.set_ylim(0, 150)
    ax.grid(axis="y", alpha=.22, lw=.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    bottom = np.zeros(len(ROUTES))
    shades = ["#3d3d3d", "#6e6e6e", "#9e9e9e", "#c4c4c4"]
    for j, exp in enumerate(present):
        vals = []
        for r in ROUTES:
            row = by_exp[(by_exp.route == r)
                         & (by_exp.lichtheim_exposure_status == exp)]
            vals.append(float(row["n_error_events"].iloc[0]) if len(row) else 0.0)
        vals = np.array(vals)
        ax.bar(x, vals, 0.55, bottom=bottom, color=shades[j % len(shades)],
               label=f"{exp} (stratum n="
                     f"{int(by_exp[by_exp.lichtheim_exposure_status==exp]['n_items_in_stratum'].iloc[0])})")
        for i, v in enumerate(vals):
            if v > 0:
                sub = by_exp[(by_exp.route == ROUTES[i])
                             & (by_exp.lichtheim_exposure_status == exp)]
                items = int(sub["n_unique_erroneous_items"].iloc[0])
                inside = v >= 12
                ax.annotate(f"{int(v)} events\n({items} "
                            f"{'item' if items == 1 else 'items'})",
                            (x[i], bottom[i] + v / 2 if inside
                             else bottom[i] + v + 3),
                            ha="center",
                            va="center" if inside else "bottom",
                            fontsize=7.4,
                            color="white" if (inside and j == 0) else "#111")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([ROUTE_LABEL[r] for r in ROUTES])
    ax.set_ylabel("error events (seed x item)", fontsize=10)
    ax.set_ylim(0, 165)
    ax.set_title("B  Which exposure stratum the errors come from",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.0, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.22, lw=.6)
    ax.set_axisbelow(True)

    ax = axes[2]
    for r, ls in zip(ROUTES, ("-", "--", ":")):
        d = rec[rec.route == r].sort_values("n_seeds_with_error")
        ax.plot(d["n_seeds_with_error"], d["n_items"], ls, marker="o",
                color=EXPOSURE_COLOR if r != "ltm" else "#000", lw=1.8,
                ms=5, label=ROUTE_LABEL[r])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("number of seeds in which the item fails", fontsize=9.5)
    ax.set_ylabel("unique items", fontsize=10)
    ax.set_title("C  Do the same items fail across seeds?", fontsize=10.5,
                 loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.22, lw=.6)
    ax.set_axisbelow(True)

    fig.suptitle("Errors on the 800 WFE source-real items "
                 "(faithful stimulus label, all seeds and routes)",
                 fontsize=12.5, y=1.01)
    fig.tight_layout()

    caption = """
# Errors on the 800 WFE source-real items

Population: all 800 items the WFE stimulus set labels *real*, in
`FAITHFUL_WFE_ALL`, across 4 seeds and 3 routes (9,600 seed x route x item
rows). Exposure strata use the neutral grey palette; red and blue are reserved
for lexicality elsewhere and are deliberately not used here.

**Panel A.** Errors are overwhelmingly concentrated in LTM: 126 error events on
70 unique items, against 8 events on 5 items for FULL and 7 events on 5 items
for WM. Both counts are reported because they answer different questions -
events count seed x item failures, unique items count how much of the stimulus
set is ever affected.

**Panel B.** For FULL and WM, **100 %** of source-real errors come from
`UNTRAINED_REAL` - words the stimulus set calls real but which were never in the
Lichtheim3 training lexicon. For LTM, 86.5 % come from `UNTRAINED_REAL`
(109 events / 57 items out of 122 stratum items), 11.1 % from
`TRAINED_REAL_EXACT` (14 events / 12 items out of 671) and 2.4 % from
`TRAINED_REAL_PRON_VARIANT` (3 events / 1 item out of 7).

**Errors do remain in `TRAINED_REAL_EXACT`, but only in LTM**, at a 0.52 %
event rate. FULL and WM make no errors at all on trained-exact real words.

**Panel C.** Failures are only partly item-consistent. In LTM, 53 % of erroneous
items fail in exactly one seed and only 7 items (10 %) fail in all four. For
FULL and WM no item fails in all four seeds.

**Limitations.** This is descriptive. The `TRAINED_REAL_PRON_VARIANT` stratum has
7 items and the FULL/WM error sets have 5 items each - far too small for any
claim beyond enumeration. Association with length and low frequency is reported
in `tables/faithful_real_error_descriptive_bins.tsv` as observed rates and is
**not** adjusted for the exposure confound: inside the faithful real label,
untrained words are both rarer and differently distributed over length, so the
apparent frequency and length effects partly re-express exposure. No confirmatory
model is fitted in this pass.

Exhaustive per-event listing with the literal source columns, including EOS
class derived with the frozen `eos_diagnostics.classify_eos`:
`tables/faithful_real_error_events.tsv`.
"""
    return save_figure(fig, FIG_DIR, "yc2_faithful_real_error_composition",
                       caption)


# ------------------------------------ yc3 LTM successful-pseudoword audit

def figure_ltm_pseudoword_success() -> Dict[str, str]:
    items = _tab("ltm_pseudoword_item_success.tsv")
    groups = _tab("ltm_pseudoword_group_summary.tsv")

    order = ["ALWAYS_SUCCESSFUL", "MIXED_SUCCESS", "ALWAYS_FAILED"]
    shades = {"ALWAYS_SUCCESSFUL": "#3d3d3d", "MIXED_SUCCESS": "#8c8c8c",
              "ALWAYS_FAILED": "#c4c4c4"}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6))

    ax = axes[0]
    n = [int(groups[groups.success_group == g]["n_items"].iloc[0]) for g in order]
    ax.bar(range(3), n, 0.6, color=[shades[g] for g in order],
           edgecolor="#222", lw=.7)
    for i, v in enumerate(n):
        ax.annotate(f"{v}\n({v / sum(n) * 100:.1f} %)", (i, v), ha="center",
                    va="bottom", fontsize=8.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["4/4\nalways", "1-3/4\nmixed", "0/4\nnever"],
                       fontsize=9)
    ax.set_ylabel("novel pseudowords", fontsize=10)
    ax.set_ylim(0, max(n) * 1.22)
    ax.set_title("A  LTM exact-match success across seeds", fontsize=10.5,
                 loc="left")
    ax.grid(axis="y", alpha=.22, lw=.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    for i, g in enumerate(order):
        v = items[items.success_group == g]["target_length"].to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0)
                   .uniform(-.16, .16, len(v)), v, s=9, color=shades[g],
                   edgecolor="#333", linewidth=.25, alpha=.75)
        ax.plot([i - .3, i + .3], [v.mean()] * 2, color="#b00", lw=2.2,
                zorder=5)
        ax.annotate(f"mean {v.mean():.2f}", (i, v.mean()), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#b00")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["always", "mixed", "never"], fontsize=9)
    ax.set_ylabel("target length (phonemes)", fontsize=10)
    ax.set_yticks(LENGTHS)
    ax.set_title("B  Target length by success group", fontsize=10.5, loc="left")
    ax.grid(axis="y", alpha=.22, lw=.6)
    ax.set_axisbelow(True)

    ax = axes[2]
    for i, g in enumerate(order):
        v = items[items.success_group == g]["mean_lexical_confidence"].to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(1)
                   .uniform(-.16, .16, len(v)), v, s=9, color=shades[g],
                   edgecolor="#333", linewidth=.25, alpha=.75)
        ax.plot([i - .3, i + .3], [v.mean()] * 2, color="#b00", lw=2.2,
                zorder=5)
        ax.annotate(f"mean {v.mean():.3f}", (i, v.mean()), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#b00")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["always", "mixed", "never"], fontsize=9)
    ax.set_ylabel("lexical confidence (top-1 cosine)", fontsize=10)
    ax.set_title("C  Lexical confidence by success group", fontsize=10.5,
                 loc="left")
    ax.grid(axis="y", alpha=.22, lw=.6)
    ax.set_axisbelow(True)
    ax.annotate("gate = sigmoid(2.0 x (lexical_confidence - 0.7)) is a monotone "
                "function of this\nvariable, so it is reported as auxiliary and "
                "never as independent evidence",
                xy=(0.0, -0.20), xycoords="axes fraction", fontsize=7.2,
                color="#444", linespacing=1.35, va="top")

    fig.suptitle("Which novel pseudowords does the LTM-only route reproduce "
                 "exactly? (391 items, 4 seeds)", fontsize=12.5, y=1.01)
    fig.tight_layout()

    caption = """
# LTM-only success on the 391 novel pseudowords

Population: the 391 `NOVEL_PSEUDOWORD` items, **LTM route only**, classified by
how many of the four seeds reproduce the form exactly. The three classes are
exhaustive and mutually exclusive by construction.

**Panel A.** 201 items (51.4 %) are reproduced exactly by all four seeds,
173 (44.3 %) by one to three seeds, and 17 (4.3 %) by none. The LTM route
therefore reproduces a majority of novel pseudowords perfectly and consistently -
it is not a route that simply fails on unfamiliar forms.

**Panel B.** Target length separates the groups more sharply than anything else
available: mean 5.10 phonemes for always-successful, 7.01 for mixed, 8.53 for
always-failed. Every point is one item; the red bar is the group mean.

**Panel C.** Lexical confidence differs in the same direction but weakly:
0.562 / 0.540 / 0.507. **The gate is not shown as a separate panel because it is
a deterministic monotone function of this same variable**
(`gate = sigmoid(2.0 x (lexical_confidence - 0.7))`), so plotting both would
present one variable twice as if it were two pieces of evidence. Group gate means
are in `tables/ltm_pseudoword_feature_summary.tsv`, flagged as auxiliary.

## What this figure does NOT do

**It does not identify why LTM succeeds on some novel pseudowords and not
others.** It answers "which items succeed?" and nothing more. Length and lexical
confidence are **associated descriptors** of the success groups, not causes and
not a mechanism: they are themselves correlated with each other, the groups were
formed from the outcome, and no intervention or controlled comparison was made.
A descriptive difference in a group formed by its own outcome cannot establish
what produced that outcome.

**Lexical confidence is not retrieval, and no bank vector reaches the decoder.**
Lexical confidence is the top-1 cosine similarity between the item's `s_hat` and
the frozen semantic bank. That similarity is used only to compute the FULL gate
scalar. **The top-1 semantic-bank neighbour is never injected into the decoder**:
the LTM decoder is initialised from `h0 = tanh(sem_to_h0(s_hat))` using the raw
`s_hat`, and the normalised query used against the bank is a separate tensor that
does not modify `s_hat`. A higher-confidence pseudoword is therefore not one that
had a stored form supplied to it.

**No lexicalization claim is made**, and no new feature was computed for this
figure.

**Missing validated measures.** The three features that would sharpen this
comparison - phonotacticity, distance to the training lexicon, and
suffix/phonemic complexity - **do not exist as validated documented features**
for WFE items anywhere in `scripts/`, `reports/`, `outputs/` or `docs/`. They are
recorded as `UNAVAILABLE_VALIDATED_MEASURE` in
`tables/ltm_pseudoword_unavailable_measures.tsv`. No proxy was invented.

**Limitation.** Length, confidence and success are mutually entangled; the
always-failed group has 17 items.
"""
    return save_figure(fig, FIG_DIR, "yc3_ltm_pseudoword_success", caption)


# ------------------------------- yc4 faithful serial position by route

def figure_faithful_serial_position_by_route() -> Dict[str, str]:
    raw = _tab("faithful_figure2C_by_route.tsv")
    curves = _tab("faithful_figure2C_by_route_interpolated.tsv")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=True)
    for ax, r in zip(axes, ROUTES):
        for lex in ("real", "pseudo"):
            col = LEXICALITY_COLOR[lex]
            c = curves[(curves.route == r) & (curves.source_lexicality == lex)]
            c = c.sort_values("relative_position")
            ax.plot(c["relative_position"], c["interpolated_error_rate"], "-",
                    color=col, lw=2.2, alpha=.85, zorder=3)
            d = raw[(raw.route == r) & (raw.source_lexicality == lex)]
            for L in sorted(d["phoneme_length"].unique()):
                dl = d[d.phoneme_length == L].sort_values("relative_position")
                ax.plot(dl["relative_position"], dl["error_rate_per_item"],
                        "-", color=col, lw=.7, alpha=.32, zorder=2)
                ax.plot(dl["relative_position"], dl["error_rate_per_item"],
                        "o", color=col, ms=3.1, alpha=.55, zorder=4)
        ax.set_title(ROUTE_LABEL[r], fontsize=12, pad=8)
        ax.set_xlabel("relative position  (i-1)/(L-1)", fontsize=9.5)
        ax.grid(alpha=.22, lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("positional error rate per item x seed", fontsize=10)
    handles = [Line2D([], [], color=LEXICALITY_COLOR[k], lw=2.2,
                      label=f"{LEXICALITY_LABEL[k]} (weighted curve)")
               for k in ("real", "pseudo")]
    handles += [Line2D([], [], color="#666", lw=.8, marker="o", ms=3.2,
                       alpha=.6, label="one target length (empirical)")]
    axes[2].legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("Faithful serial-position error profile, split by route "
                 "(FAITHFUL_WFE_ALL: 800 source-real, 400 source-pseudo)",
                 fontsize=12.5, y=1.005)
    fig.tight_layout()

    caption = f"""
# Faithful serial-position error profile by route

**This does not replace the existing faithful Figure 2C.** Those files are
read-only in this pass; these are new files under the `yc4_` prefix.

**Method, recovered and verified rather than assumed.** The original producing
driver is not in the current tree, but its logic was promoted verbatim into
`compute.serial_position_tables` and `compute.zip_mismatch_positions`. Applying
that frozen function to the faithful subset at `route == "full"` reproduces the
frozen `faithful_figure2C_table.tsv` with **maximum absolute difference
8.8e-17 on the error rate and exactly 0 on the counts** - see
`tables/faithful_figure2C_reproduction_check.tsv`. The by-route extension is
written only if that gate passes, and the WM and LTM panels use the identical
function.

**Precise positional estimand.** For each (lexicality, length) cell and each
1-based position i: the numerator is the number of item x seed rows whose
predicted symbol at i differs from the target symbol at i under **zip alignment
with the prediction re-padded to the target length**; the denominator is the
number of item x seed rows in that cell. This is Dager's `Error_Indices` and is
**not** a Levenshtein alignment - no insertion or deletion is ever realigned, so
a single early deletion makes every later position count as an error. A trimmed
prediction is re-padded to `<PAD>`, which reproduces Dager's blanking after EOS.
The four seeds are **pooled**, not averaged. Relative position is
`(i-1)/(L-1)`; lengths below 2 are skipped. {SERIAL_POSITION_METHOD}.

**Display.** Faint points and thin lines are the empirical per-length values.
The thick curve is the item-count-weighted PCHIP interpolation and is shown only
alongside the points that produced it, never on its own.

**Labels.** Red and blue are the **faithful stimulus labels** real/pseudo
(800/400), not exposure categories. 122 of the 800 source-real items were never
in the training lexicon and 9 source pseudowords collide with it, so this figure
must not be read as trained versus untrained.

**What the panels show.** The rising profile is essentially an LTM phenomenon.
FULL and WM stay near the floor across the whole word for both stimulus classes;
LTM shows the characteristic climb, and it is far steeper for pseudowords. Under
zip alignment part of that climb is mechanical: once a position is wrong, later
positions are compared against a shifted target, so error accumulates by
construction. That is a property of the faithful estimand, not a separate
finding.
"""
    return save_figure(fig, FIG_DIR,
                       "yc4s_faithful_serial_position_all_routes", caption)


def figure_faithful_serial_position_wm_ltm() -> Dict[str, str]:
    """Simplified two-route presentation: the dorsal/ventral contrast only.

    FULL is omitted here because it sits on the floor across the whole word and
    adds no readable information; it remains in the supplementary three-route
    figure `yc4s_faithful_serial_position_all_routes`.
    """
    raw = _tab("faithful_figure2C_by_route.tsv")
    curves = _tab("faithful_figure2C_by_route_interpolated.tsv")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), sharey=True)
    for ax, r in zip(axes, ("wm", "ltm")):
        for lex in ("real", "pseudo"):
            col = LEXICALITY_COLOR[lex]
            d = raw[(raw.route == r) & (raw.source_lexicality == lex)]
            # per-length traces are visually subordinated: thin, pale, behind
            for L in sorted(d["phoneme_length"].unique()):
                dl = d[d.phoneme_length == L].sort_values("relative_position")
                ax.plot(dl["relative_position"], dl["error_rate_per_item"],
                        "-", color=col, lw=.7, alpha=.22, zorder=2)
                ax.plot(dl["relative_position"], dl["error_rate_per_item"],
                        "o", color=col, ms=3.4, alpha=.45, zorder=3,
                        mec="none")
            c = curves[(curves.route == r)
                       & (curves.source_lexicality == lex)]
            c = c.sort_values("relative_position")
            ax.plot(c["relative_position"], c["interpolated_error_rate"], "-",
                    color=col, lw=3.0, zorder=5)
        ax.set_title(f"{ROUTE_LABEL[r]}  "
                     f"({'dorsal' if r == 'wm' else 'ventral'} route)",
                     fontsize=12.5, pad=8)
        ax.set_xlabel("relative position within the word   (i-1)/(L-1)",
                      fontsize=10)
        ax.grid(alpha=.22, lw=.6)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.03, 1.03)
    axes[0].set_ylabel("positional error rate  (per item x seed)", fontsize=10.5)
    axes[0].annotate("word onset", (0.0, -0.085), xycoords="axes fraction",
                     fontsize=8, color="#555", ha="left")
    axes[0].annotate("word offset", (1.0, -0.085), xycoords="axes fraction",
                     fontsize=8, color="#555", ha="right")

    handles = [Line2D([], [], color=LEXICALITY_COLOR[k], lw=3.0,
                      label=f"{LEXICALITY_LABEL[k]} — pooled profile")
               for k in ("real", "pseudo")]
    handles += [Line2D([], [], color="#777", lw=.8, marker="o", ms=3.4,
                       alpha=.5,
                       label="one target length (empirical points, background)")]
    axes[1].legend(handles=handles, fontsize=8.4, loc="upper left",
                   frameon=True, framealpha=.93)
    fig.suptitle("Where in the word do errors occur? Dorsal versus ventral "
                 "route", fontsize=13, y=1.005)
    fig.tight_layout()

    caption = """
# Serial-position error profile: WM versus LTM

A simplified two-route reading of the faithful serial-position analysis. FULL is
omitted because it sits on the floor across the whole word and adds nothing
readable; the three-route version is kept as
`yc4s_faithful_serial_position_all_routes`.

**The estimator is the verified faithful one, unchanged.** Applying the frozen
`compute.serial_position_tables` to the faithful subset at `route == "full"`
reproduces the frozen `faithful_figure2C_table.tsv` to a maximum absolute
difference of **8.8e-17** on the error rate and **exactly 0** on the counts
(`tables/faithful_figure2C_reproduction_check.tsv`). The WM and LTM panels use
that identical function.

**Four properties of the estimand, all of which matter for reading the curves:**

1. **Seeds are pooled, not averaged.** The denominator is item x seed rows in
   the (lexicality, length) cell, so every point is one pooled rate over the
   four checkpoints.
2. **Zip mismatch, not Levenshtein alignment.** Position i of the prediction is
   compared with position i of the target. Nothing is realigned.
3. **No insertion or deletion is ever realigned**, so a single early deletion
   makes every later position count as an error. Part of the rise is therefore
   mechanical accumulation, not independent evidence of late-position fragility.
4. **Post-EOS re-padding.** A prediction trimmed at EOS is re-padded to `<PAD>`
   up to the target length, which recovers Dager's blanking-after-EOS, so
   positions after an early stop count as mismatches.

**Display.** Pale points and thin lines in the background are the empirical
per-length values, retained so nothing is hidden behind the summary. The thick
line is the item-count-weighted PCHIP interpolation across lengths and is never
shown without those points.

**Labels.** Red and blue are the **faithful stimulus labels** real (800) and
pseudo (400), not exposure categories. 122 of the source-real items were never
in the training lexicon and 9 source pseudowords collide with it, so this figure
must not be read as trained versus untrained.

**What it shows.** WM stays near the floor across the whole word for both
stimulus classes, with only a mild late rise for pseudowords. LTM shows the
characteristic climb, far steeper for pseudowords. The dorsal route holds serial
position; the ventral route degrades along the word.
"""
    return save_figure(fig, FIG_DIR, "yc4_faithful_serial_position_wm_ltm",
                       caption)


def run() -> Dict[str, Dict[str, str]]:
    return {
        "yc1": figure_word_error_by_length(),
        "yc2": figure_faithful_real_errors(),
        "yc3": figure_ltm_pseudoword_success(),
        "yc4": figure_faithful_serial_position_wm_ltm(),
        "yc4s": figure_faithful_serial_position_by_route(),
    }


if __name__ == "__main__":
    for k, v in run().items():
        print(k, v["png"])
