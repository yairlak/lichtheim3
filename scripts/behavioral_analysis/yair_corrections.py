"""Yair corrections pass — reaggregation of the frozen predictions.

Implements the specification frozen in
reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/_control/
yair_corrections_spec.json, written before any result here was computed.

**This is not a new analysis programme.** Every number is a reaggregation of the
canonical seed x item x route table. No checkpoint is loaded, no inference is
run, no model is trained, and no new statistical model is introduced: the only
estimators used are the already-validated frozen ones promoted in
`bootstrap.py`, `compute.py` and `eos_diagnostics.py`.

Five tasks:

  T1  whole-word error rate by exact length, route and lexicality (clean set)
  T2  exhaustive error audit of the 800 faithful source-real items
  T3  LTM-only success classification of the 391 novel pseudowords
  T4  faithful serial-position profile split by route, gated on an exact
      reproduction of the frozen faithful Figure 2C
  T5  route-specific feature-importance **estimability audit** (no refit)

Faithful and clean populations are never pooled in one estimate.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import compute
from .bootstrap import cell_mean_bootstrap
from .common import (EXPECTED_CLEAN_COUNTS, ITEM_MANIFEST, LENGTHS,
                     REPORT_ROOT, ROUTES, SEEDS)
from .eos_diagnostics import classify_eos
from .io import load_canonical, write_table

CORRECTIONS_ROOT = os.path.join(REPORT_ROOT, "yair_corrections")
LEXICALITIES = ["real", "pseudo"]

# T3 success classes; exhaustive and mutually exclusive over 0..4 seeds.
ALWAYS_SUCCESS = "ALWAYS_SUCCESSFUL"
MIXED = "MIXED_SUCCESS"
ALWAYS_FAILED = "ALWAYS_FAILED"
SUCCESS_GROUPS = [ALWAYS_SUCCESS, MIXED, ALWAYS_FAILED]

UNAVAILABLE = "UNAVAILABLE_VALIDATED_MEASURE"
NOT_ESTIMABLE = "NOT_ESTIMABLE_CEILING_OR_SPARSE_OUTCOME"
ESTIMABLE = "ESTIMABLE"

# The gate is a deterministic monotone function of lexical confidence
# (sigmoid(2.0 * (confidence - 0.7))), so the two are one variable reported
# twice.  Recorded here so no table can present them as independent evidence.
GATE_IS_FUNCTION_OF_CONFIDENCE = (
    "gate = sigmoid(2.0 * (lexical_confidence - 0.7)); auxiliary linked "
    "variable, not independent evidence")

ERROR_EVENT_COLUMNS = [
    "seed", "epoch", "route", "item_id", "target", "prediction",
    "target_length", "predicted_length", "exact_match", "word_error",
    "raw_edit_distance", "normalized_edit_distance",
    "substitutions", "deletions", "insertions",
    "eos_position", "source_lexicality", "lichtheim_exposure_status",
    "morphology", "size", "condition", "zipf_frequency", "frequency_class",
    "gate", "lexical_confidence",
]


def out_path(*parts: str) -> str:
    return os.path.join(CORRECTIONS_ROOT, *parts)


def _lexicality_of(canon: pd.DataFrame) -> pd.Series:
    return canon["source_lexicality"]


# ===================================================== T1 word error by length

def word_error_by_length(clean: pd.DataFrame
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-seed cells, across-seed summary, item counts).

    Primary metric is `word_error`.  `raw_edit_distance` travels in the same
    rows as post-failure severity and is never the headline.
    """
    seed_rows, summary_rows, count_rows = [], [], []
    for r in ROUTES:
        for lex in LEXICALITIES:
            for L in LENGTHS:
                cell = clean[(clean["route"] == r)
                             & (clean["source_lexicality"] == lex)
                             & (clean["target_length"] == L)]
                if cell.empty:
                    continue
                per_seed = {}
                for s in SEEDS:
                    sub = cell[cell["seed"] == s].sort_values("item_id")
                    per_seed[s] = sub["word_error"].to_numpy(float)
                lo, hi = cell_mean_bootstrap(per_seed, SEEDS)
                for s in SEEDS:
                    sub = cell[cell["seed"] == s]
                    seed_rows.append({
                        "route": r, "source_lexicality": lex,
                        "phoneme_length": L, "seed": s,
                        "n_items": int(sub["item_id"].nunique()),
                        "n_word_errors": int(sub["word_error"].sum()),
                        "word_error_rate": float(sub["word_error"].mean()),
                        "mean_raw_edit_distance_severity_only":
                            float(sub["raw_edit_distance"].mean()),
                    })
                summary_rows.append({
                    "route": r, "source_lexicality": lex, "phoneme_length": L,
                    "n_items": int(cell["item_id"].nunique()),
                    "n_items_x_seeds": int(len(cell)),
                    "mean_word_error_rate_across_seeds":
                        float(cell["word_error"].mean()),
                    "min_seed_word_error_rate":
                        float(min(v.mean() for v in per_seed.values())),
                    "max_seed_word_error_rate":
                        float(max(v.mean() for v in per_seed.values())),
                    "ci_low": lo, "ci_high": hi,
                    "mean_raw_edit_distance_severity_only":
                        float(cell["raw_edit_distance"].mean()),
                })
    one = clean[(clean["seed"] == SEEDS[0]) & (clean["route"] == ROUTES[0])]
    for lex in LEXICALITIES:
        for L in LENGTHS:
            n = int(((one["source_lexicality"] == lex)
                     & (one["target_length"] == L)).sum())
            count_rows.append({"source_lexicality": lex, "phoneme_length": L,
                               "n_items": n})
    return (pd.DataFrame(seed_rows), pd.DataFrame(summary_rows),
            pd.DataFrame(count_rows))


def faithful_word_error_companion(canon: pd.DataFrame) -> pd.DataFrame:
    """Faithful-population companion, kept strictly separate from the clean set.

    Shows why the faithful figure differs: the faithful `real` label pools 671
    trained-exact words with 122 untrained real words and 7 pronunciation
    variants, which is an exposure mixture, not a lexicality contrast.
    """
    faith = canon[canon["in_FAITHFUL_WFE_ALL"]]
    rows = []
    for r in ROUTES:
        for lex in LEXICALITIES:
            for L in LENGTHS:
                cell = faith[(faith["route"] == r)
                             & (faith["source_lexicality"] == lex)
                             & (faith["target_length"] == L)]
                if cell.empty:
                    continue
                rows.append({
                    "population": "FAITHFUL_WFE_ALL",
                    "route": r, "source_lexicality": lex, "phoneme_length": L,
                    "n_items": int(cell["item_id"].nunique()),
                    "mean_word_error_rate_across_seeds":
                        float(cell["word_error"].mean()),
                    "exposure_mixture": "; ".join(
                        f"{k}={v}" for k, v in
                        cell[cell["seed"] == SEEDS[0]]
                        ["lichtheim_exposure_status"].value_counts().items()),
                })
    return pd.DataFrame(rows)


# ============================================ T2 faithful source-real audit

def faithful_real_error_audit(canon: pd.DataFrame):
    """(exhaustive error events, summary, by-exposure, recurrence)."""
    real = canon[canon["in_FAITHFUL_WFE_ALL"]
                 & (canon["source_lexicality"] == "real")].copy()
    real["eos_class"] = [classify_eos(o, int(L)) for o, L
                         in zip(real["eos_position"], real["target_length"])]
    events = real[real["word_error"] == 1].copy()
    events = events[ERROR_EVENT_COLUMNS + ["eos_class"]]
    events = events.sort_values(["route", "seed", "item_id"])

    n_items = int(real["item_id"].nunique())
    n_rows = int(len(real))
    summary_rows = []
    for r in ROUTES:
        rr = real[real["route"] == r]
        ee = events[events["route"] == r]
        summary_rows.append({
            "route": r,
            "n_source_real_items": int(rr["item_id"].nunique()),
            "n_seed_x_item_rows": int(len(rr)),
            "n_error_events_seed_x_item": int(len(ee)),
            "n_unique_erroneous_items": int(ee["item_id"].nunique()),
            "event_error_rate": float(len(ee) / len(rr)) if len(rr) else np.nan,
            "unique_item_error_rate":
                float(ee["item_id"].nunique() / rr["item_id"].nunique()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.attrs["n_source_real_items"] = n_items
    summary.attrs["n_rows"] = n_rows

    by_exp_rows = []
    for r in ROUTES:
        rr = real[real["route"] == r]
        ee = events[events["route"] == r]
        tot_events = len(ee)
        tot_items = ee["item_id"].nunique()
        for exp in sorted(real["lichtheim_exposure_status"].unique()):
            base = rr[rr["lichtheim_exposure_status"] == exp]
            sub = ee[ee["lichtheim_exposure_status"] == exp]
            by_exp_rows.append({
                "route": r, "lichtheim_exposure_status": exp,
                "n_items_in_stratum": int(base["item_id"].nunique()),
                "n_seed_x_item_rows": int(len(base)),
                "n_error_events": int(len(sub)),
                "n_unique_erroneous_items": int(sub["item_id"].nunique()),
                "share_of_route_error_events":
                    float(len(sub) / tot_events) if tot_events else np.nan,
                "share_of_route_unique_erroneous_items":
                    (float(sub["item_id"].nunique() / tot_items)
                     if tot_items else np.nan),
                "within_stratum_event_error_rate":
                    float(len(sub) / len(base)) if len(base) else np.nan,
            })
    by_exposure = pd.DataFrame(by_exp_rows)

    rec_rows = []
    for r in ROUTES:
        ee = events[events["route"] == r]
        per_item = ee.groupby("item_id")["seed"].nunique()
        for k in range(1, len(SEEDS) + 1):
            rec_rows.append({
                "route": r, "n_seeds_with_error": k,
                "n_items": int((per_item == k).sum()),
                "share_of_erroneous_items":
                    (float((per_item == k).sum() / len(per_item))
                     if len(per_item) else np.nan),
            })
    recurrence = pd.DataFrame(rec_rows)
    return events, summary, by_exposure, recurrence


def trained_real_exact_ltm_errors(canon: pd.DataFrame,
                                  manifest_path: str) -> pd.DataFrame:
    """The `TRAINED_REAL_EXACT` items the LTM route gets wrong, one row per item.

    Orthography is joined from the frozen item manifest purely for readability;
    no value is recomputed.  One row per item with the failing seeds collapsed
    into a list, so the table reads as "which trained words does LTM lose".
    No frequency model is fitted here and none is implied.
    """
    d = canon[(canon["route"] == "ltm")
              & (canon["lichtheim_exposure_status"] == "TRAINED_REAL_EXACT")
              & (canon["word_error"] == 1)].copy()
    man = pd.read_csv(manifest_path, sep="\t")[["item_id", "word"]]
    d = d.merge(man, on="item_id", how="left", validate="many_to_one")
    rows = []
    for item, g in d.groupby("item_id"):
        g = g.sort_values("seed")
        rows.append({
            "item_id": item,
            "word": g["word"].iloc[0],
            "zipf_frequency": float(g["zipf_frequency"].iloc[0]),
            "frequency_class": g["frequency_class"].iloc[0],
            "phoneme_length": int(g["target_length"].iloc[0]),
            "morphology": g["morphology"].iloc[0],
            "target": g["target"].iloc[0],
            "n_failing_seeds": int(len(g)),
            "failing_seeds": ",".join(str(int(s)) for s in g["seed"]),
            "predictions_by_seed": " | ".join(
                f"{int(s)}: {p}" for s, p in zip(g["seed"], g["prediction"])),
            "substitutions": ",".join(str(int(v)) for v in g["substitutions"]),
            "deletions": ",".join(str(int(v)) for v in g["deletions"]),
            "insertions": ",".join(str(int(v)) for v in g["insertions"]),
            "raw_edit_distance": ",".join(f"{v:g}" for v in g["raw_edit_distance"]),
            "total_substitutions": int(g["substitutions"].sum()),
            "total_deletions": int(g["deletions"].sum()),
            "total_insertions": int(g["insertions"].sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["n_failing_seeds", "zipf_frequency"], ascending=[False, True])


def faithful_real_descriptive_bins(canon: pd.DataFrame) -> pd.DataFrame:
    """Observed error rates by length and by frequency class.  Descriptive.

    Not adjusted for the exposure confound: inside the faithful real label,
    untrained words are both rarer and differently distributed over length.
    """
    real = canon[canon["in_FAITHFUL_WFE_ALL"]
                 & (canon["source_lexicality"] == "real")]
    rows = []
    for r in ROUTES:
        rr = real[real["route"] == r]
        for L in LENGTHS:
            c = rr[rr["target_length"] == L]
            if len(c):
                rows.append({"route": r, "binning": "phoneme_length",
                             "bin": str(L), "n_seed_x_item_rows": int(len(c)),
                             "n_items": int(c["item_id"].nunique()),
                             "event_error_rate": float(c["word_error"].mean())})
        for fc in sorted(rr["frequency_class"].dropna().unique()):
            c = rr[rr["frequency_class"] == fc]
            rows.append({"route": r, "binning": "frequency_class",
                         "bin": str(fc), "n_seed_x_item_rows": int(len(c)),
                         "n_items": int(c["item_id"].nunique()),
                         "event_error_rate": float(c["word_error"].mean())})
    out = pd.DataFrame(rows)
    out["caveat"] = ("descriptive only; not adjusted for the exposure confound "
                     "inside the faithful real label")
    return out


# ================================== T3 LTM successful-pseudoword audit

def ltm_pseudoword_success(canon: pd.DataFrame):
    """(item-level success table, group summary, feature summary)."""
    d = canon[(canon["lichtheim_exposure_status"] == "NOVEL_PSEUDOWORD")
              & (canon["route"] == "ltm")].copy()
    d["eos_class"] = [classify_eos(o, int(L)) for o, L
                      in zip(d["eos_position"], d["target_length"])]

    rows = []
    for item, g in d.groupby("item_id"):
        g = g.sort_values("seed")
        n_ok = int(g["exact_match"].sum())
        assert len(g) == len(SEEDS), (item, len(g))
        group = (ALWAYS_SUCCESS if n_ok == len(SEEDS)
                 else ALWAYS_FAILED if n_ok == 0 else MIXED)
        failed = g[g["exact_match"] == 0]
        rec = {
            "item_id": item, "success_group": group,
            "n_seeds_exact": n_ok, "n_seeds": len(g),
            "target": g["target"].iloc[0],
            "target_length": int(g["target_length"].iloc[0]),
            "morphology": g["morphology"].iloc[0],
            "size": g["size"].iloc[0],
            "mean_lexical_confidence": float(g["lexical_confidence"].mean()),
            "mean_gate_auxiliary": float(g["gate"].mean()),
            "n_failed_seeds": int(len(failed)),
            "mean_substitutions_failed_seeds":
                float(failed["substitutions"].mean()) if len(failed) else np.nan,
            "mean_deletions_failed_seeds":
                float(failed["deletions"].mean()) if len(failed) else np.nan,
            "mean_insertions_failed_seeds":
                float(failed["insertions"].mean()) if len(failed) else np.nan,
            "mean_raw_edit_distance_failed_seeds":
                float(failed["raw_edit_distance"].mean()) if len(failed) else np.nan,
            "n_premature_eos_failed_seeds":
                int((failed["eos_class"] == "PREMATURE_EOS").sum()),
        }
        for s in SEEDS:
            gs = g[g["seed"] == s]
            rec[f"seed{s}_exact_match"] = int(gs["exact_match"].iloc[0])
            rec[f"seed{s}_prediction"] = gs["prediction"].iloc[0]
            rec[f"seed{s}_raw_edit_distance"] = float(
                gs["raw_edit_distance"].iloc[0])
            rec[f"seed{s}_eos_class"] = gs["eos_class"].iloc[0]
        rows.append(rec)
    items = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)

    gsum = []
    for grp in SUCCESS_GROUPS:
        sub = items[items["success_group"] == grp]
        gsum.append({
            "success_group": grp, "n_items": int(len(sub)),
            "share_of_391": float(len(sub) / len(items)),
            "n_seeds_exact_definition":
                {"ALWAYS_SUCCESSFUL": "4/4", "MIXED_SUCCESS": "1-3/4",
                 "ALWAYS_FAILED": "0/4"}[grp],
        })
    groups = pd.DataFrame(gsum)

    feats = []
    numeric = [("target_length", "already-validated"),
               ("mean_lexical_confidence", "already-validated"),
               ("mean_gate_auxiliary", GATE_IS_FUNCTION_OF_CONFIDENCE),
               ("mean_substitutions_failed_seeds", "already-validated"),
               ("mean_deletions_failed_seeds", "already-validated"),
               ("mean_insertions_failed_seeds", "already-validated"),
               ("mean_raw_edit_distance_failed_seeds", "already-validated"),
               ("n_premature_eos_failed_seeds", "already-validated")]
    for grp in SUCCESS_GROUPS:
        sub = items[items["success_group"] == grp]
        for col, note in numeric:
            v = sub[col].dropna()
            feats.append({
                "success_group": grp, "feature": col,
                "n_items_with_value": int(len(v)),
                "mean": float(v.mean()) if len(v) else np.nan,
                "median": float(v.median()) if len(v) else np.nan,
                "min": float(v.min()) if len(v) else np.nan,
                "max": float(v.max()) if len(v) else np.nan,
                "status": "OK", "note": note})
    features = pd.DataFrame(feats)
    return items, groups, features


def unavailable_measures_table() -> pd.DataFrame:
    """Measures §4 asks about that do not exist as validated features.

    Searched across scripts/, reports/, outputs/ and docs/.  Nothing is
    substituted: an absent measure is recorded, not approximated.
    """
    rows = [
        ("phonotacticity",
         "no phonotactic probability or legality score exists for WFE items in "
         "any validated table; the only mentions are mechanism captions stating "
         "that phonotactic influence was NOT tested"),
        ("distance_to_training_lexicon",
         "no graded phonological distance to the training lexicon exists; "
         "audits/wfe_pseudo_phonology_in_training.tsv records exact membership "
         "collisions only, which is already encoded in lichtheim_exposure_status"),
        ("suffix_or_phonemic_complexity",
         "the validated morphology field is a binary simple/complex stimulus "
         "label; no suffix inventory or graded phonemic-complexity score exists"),
    ]
    return pd.DataFrame([
        {"requested_measure": m, "status": UNAVAILABLE,
         "search_scope": "scripts/, reports/, outputs/, docs/",
         "proxy_computed": False, "evidence": e} for m, e in rows])


# ==================================== T4 faithful serial position by route

def faithful_serial_position_by_route(canon: pd.DataFrame,
                                      frozen_2c_path: str):
    """(raw by route, interpolated by route, reproduction check).

    The reproduction check is a hard gate: the frozen `serial_position_tables`
    applied to the faithful subset at route == "full" must reproduce the frozen
    faithful Figure 2C table before the WM and LTM extension is trusted.
    """
    faith = canon[canon["in_FAITHFUL_WFE_ALL"]].copy()
    raw, curves = compute.serial_position_tables(faith)

    frozen = pd.read_csv(frozen_2c_path, sep="\t")
    mine = raw[raw["route"] == "full"][
        ["source_lexicality", "phoneme_length", "position_index_1based",
         "relative_position", "n_items_x_seeds", "error_rate_per_item"]]
    mine = mine.rename(columns={"source_lexicality": "lexicality",
                                "phoneme_length": "length",
                                "position_index_1based": "position_1based"})
    m = frozen.merge(mine, on=["lexicality", "length", "position_1based"],
                     suffixes=("_frozen", "_recomputed"), how="outer",
                     indicator=True)
    check_rows = []
    for col in ("relative_position", "n_items_x_seeds", "error_rate_per_item"):
        d = np.abs(m[f"{col}_frozen"] - m[f"{col}_recomputed"])
        check_rows.append({
            "quantity": col, "n_compared": int(len(m)),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "rows_only_in_frozen": int((m["_merge"] == "left_only").sum()),
            "rows_only_in_recomputed": int((m["_merge"] == "right_only").sum()),
        })
    check = pd.DataFrame(check_rows)
    check["reproduces_frozen_figure2C"] = bool(
        (check["max_abs_diff"] < 1e-12).all()
        and (check["rows_only_in_frozen"] == 0).all()
        and (check["rows_only_in_recomputed"] == 0).all())
    check["estimand"] = (
        "zip-mismatch positional error, prediction re-padded to target length, "
        "no Levenshtein alignment, seeds pooled")
    return raw, curves, check


# =============================== T5 feature-importance estimability audit

def fi_route_estimability(fit_path: str, imp_path: str) -> pd.DataFrame:
    """Audit of the existing validated route-specific FI.  No model is fitted."""
    fit = pd.read_csv(fit_path, sep="\t")
    imp = pd.read_csv(imp_path, sep="\t")
    rows = []
    for _, r in fit.iterrows():
        ok = r["model_status"] == "OK"
        sub = imp[(imp["seed"] == r["seed"]) & (imp["route"] == r["route"])]
        rows.append({
            "route": r["route"], "seed": int(r["seed"]),
            "outcome": r["outcome"],
            "n_train_items": int(r["n_train_items"]),
            "n_test_items": int(r["n_test_items"]),
            "n_test_nonzero": int(r["n_test_nonzero"]),
            "test_nonzero_density":
                float(r["n_test_nonzero"] / r["n_test_rows"]),
            "ridge_alpha": float(r["ridge_alpha"]),
            "original_outcome_status": r["outcome_status"],
            "original_model_status": r["model_status"],
            "negative_test_r2": bool(r["negative_test_r2"]),
            "train_r2": r["train_r2"], "test_r2": r["test_r2"],
            "estimability_verdict": ESTIMABLE if ok else NOT_ESTIMABLE,
            "importance_reported": ok,
            "importance_is_zero_by_fiat": False,
            "n_factor_rows_available": int(len(sub)),
            "note": ("importance read from the validated A15 run" if ok else
                     "no importance is reported; a non-estimable cell is never "
                     "assigned zero importance"),
        })
    return pd.DataFrame(rows).sort_values(["route", "seed"])


def fi_outcome_density(clean: pd.DataFrame) -> pd.DataFrame:
    """Share of non-zero outcomes per route x population — governs estimability."""
    rows = []
    for r in ROUTES:
        for label, sub in (("LICHTHEIM_CLEAN_all", clean),
                           ("clean_real", clean[clean["source_lexicality"] == "real"]),
                           ("clean_pseudo", clean[clean["source_lexicality"] == "pseudo"])):
            c = sub[sub["route"] == r]
            for out in ("raw_edit_distance", "word_error"):
                v = c[out].to_numpy(float)
                rows.append({
                    "route": r, "population": label, "outcome": out,
                    "n_rows": int(len(v)),
                    "n_nonzero": int((v > 0).sum()),
                    "nonzero_density": float((v > 0).mean()),
                    "variance": float(v.var()),
                })
    return pd.DataFrame(rows)


def _write_trained_real_md(t: pd.DataFrame, path: str) -> str:
    """Compact Markdown twin of the trained-real LTM error table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L = ["# `TRAINED_REAL_EXACT` words the LTM route gets wrong\n",
         f"\n**{len(t)} words** out of 671, "
         f"{int(t['n_failing_seeds'].sum())} seed x item error events. "
         "These are the only trained-exact real words any route loses; FULL and "
         "WM make no errors at all on this stratum.\n",
         "\nEdit-operation columns are per failing seed, in the order of "
         "`failing_seeds`. **No frequency model is fitted**: Zipf is shown as a "
         "descriptive column only.\n",
         "\n| word | Zipf | len | target | seeds | prediction(s) | sub/del/ins |\n",
         "|---|---|---|---|---|---|---|\n"]
    for _, r in t.iterrows():
        ops = " / ".join([r["substitutions"], r["deletions"], r["insertions"]])
        L.append(f"| **{r['word']}** | {r['zipf_frequency']:.2f} | "
                 f"{r['phoneme_length']} | `{r['target']}` | "
                 f"{r['failing_seeds']} | {r['predictions_by_seed']} | "
                 f"{ops} |\n")
    L.append("\nFull machine-readable version: "
             "`trained_real_exact_ltm_errors.tsv`.\n")
    with open(path, "w") as f:
        f.write("".join(L))
    return path


# ================================================================ driver

def run(canon_path: str = None) -> Dict[str, str]:
    """Compute every table in the pass.  Returns {name: written path}."""
    canon = load_canonical(canon_path)
    clean = canon[canon["in_LICHTHEIM_CLEAN"]].copy()
    written: Dict[str, str] = {}

    # --- structural assertions; no scientific value is hardcoded downstream
    one = clean[(clean["seed"] == SEEDS[0]) & (clean["route"] == "full")]
    counts = one["lichtheim_exposure_status"].value_counts().to_dict()
    assert counts["TRAINED_REAL_EXACT"] == EXPECTED_CLEAN_COUNTS["real"]
    assert counts["NOVEL_PSEUDOWORD"] == EXPECTED_CLEAN_COUNTS["pseudo"]
    faith_one = canon[canon["in_FAITHFUL_WFE_ALL"]
                      & (canon["seed"] == SEEDS[0]) & (canon["route"] == "full")]
    assert int((faith_one["source_lexicality"] == "real").sum()) == 800
    assert sorted(canon["target_length"].unique()) == LENGTHS

    # --- T1
    seed_t, summ_t, cnt_t = word_error_by_length(clean)
    written["word_error_by_length_seed"] = write_table(
        seed_t, out_path("tables", "word_error_by_length_seed.tsv"),
        sort_by=["route", "source_lexicality", "phoneme_length", "seed"])
    written["word_error_by_length_summary"] = write_table(
        summ_t, out_path("tables", "word_error_by_length_summary.tsv"),
        sort_by=["route", "source_lexicality", "phoneme_length"])
    written["word_error_by_length_item_counts"] = write_table(
        cnt_t, out_path("tables", "word_error_by_length_item_counts.tsv"),
        sort_by=["source_lexicality", "phoneme_length"])
    written["word_error_faithful_companion"] = write_table(
        faithful_word_error_companion(canon),
        out_path("tables", "word_error_by_length_faithful_companion.tsv"),
        sort_by=["route", "source_lexicality", "phoneme_length"])

    # --- T2
    ev, summ, by_exp, rec = faithful_real_error_audit(canon)
    written["faithful_real_error_events"] = write_table(
        ev, out_path("tables", "faithful_real_error_events.tsv"),
        sort_by=["route", "seed", "item_id"])
    written["faithful_real_error_summary"] = write_table(
        summ, out_path("tables", "faithful_real_error_summary.tsv"),
        sort_by=["route"])
    written["faithful_real_error_by_exposure"] = write_table(
        by_exp, out_path("tables", "faithful_real_error_by_exposure.tsv"),
        sort_by=["route", "lichtheim_exposure_status"])
    written["faithful_real_error_recurrence"] = write_table(
        rec, out_path("tables", "faithful_real_error_recurrence.tsv"),
        sort_by=["route", "n_seeds_with_error"])
    written["faithful_real_descriptive_bins"] = write_table(
        faithful_real_descriptive_bins(canon),
        out_path("tables", "faithful_real_error_descriptive_bins.tsv"),
        sort_by=["route", "binning", "bin"])
    tre = trained_real_exact_ltm_errors(canon, ITEM_MANIFEST)
    written["trained_real_exact_ltm_errors"] = write_table(
        tre, out_path("tables", "trained_real_exact_ltm_errors.tsv"))
    written["trained_real_exact_ltm_errors_md"] = _write_trained_real_md(
        tre, out_path("tables", "trained_real_exact_ltm_errors.md"))

    # --- T3
    items, groups, features = ltm_pseudoword_success(canon)
    written["ltm_pseudoword_item_success"] = write_table(
        items, out_path("tables", "ltm_pseudoword_item_success.tsv"),
        sort_by=["item_id"])
    written["ltm_pseudoword_group_summary"] = write_table(
        groups, out_path("tables", "ltm_pseudoword_group_summary.tsv"),
        sort_by=["success_group"])
    written["ltm_pseudoword_feature_summary"] = write_table(
        features, out_path("tables", "ltm_pseudoword_feature_summary.tsv"),
        sort_by=["feature", "success_group"])
    written["ltm_pseudoword_unavailable_measures"] = write_table(
        unavailable_measures_table(),
        out_path("tables", "ltm_pseudoword_unavailable_measures.tsv"),
        sort_by=["requested_measure"])

    # --- T4
    frozen_2c = os.path.join(
        os.path.dirname(os.path.dirname(REPORT_ROOT)), "outputs",
        "behavioral_wfe_fulllexicon_93a577f", "behavioral_analysis",
        "faithful_replication", "faithful_figure2C_table.tsv")
    raw2c, curves2c, check2c = faithful_serial_position_by_route(canon,
                                                                 frozen_2c)
    written["faithful_figure2C_reproduction_check"] = write_table(
        check2c, out_path("tables", "faithful_figure2C_reproduction_check.tsv"),
        sort_by=["quantity"])
    if not bool(check2c["reproduces_frozen_figure2C"].iloc[0]):
        raise RuntimeError(
            "faithful Figure 2C reproduction gate FAILED; the by-route "
            "extension is not written")
    written["faithful_figure2C_by_route"] = write_table(
        raw2c, out_path("tables", "faithful_figure2C_by_route.tsv"),
        sort_by=["route", "source_lexicality", "phoneme_length",
                 "position_index_1based"])
    written["faithful_figure2C_by_route_interpolated"] = write_table(
        curves2c,
        out_path("tables", "faithful_figure2C_by_route_interpolated.tsv"),
        sort_by=["route", "source_lexicality", "relative_position"])

    # --- T5
    fi_dir = os.path.join(REPORT_ROOT, "feature_importance", "route_specific",
                          "tables")
    written["fi_route_estimability"] = write_table(
        fi_route_estimability(
            os.path.join(fi_dir, "route_specific_model_fit.tsv"),
            os.path.join(fi_dir, "route_specific_factor_importance.tsv")),
        out_path("tables", "fi_route_estimability.tsv"),
        sort_by=["route", "seed"])
    written["fi_outcome_density"] = write_table(
        fi_outcome_density(clean), out_path("tables", "fi_outcome_density.tsv"),
        sort_by=["route", "population", "outcome"])
    return written


if __name__ == "__main__":
    for k, v in run().items():
        print(f"{k:45s} {v}")
