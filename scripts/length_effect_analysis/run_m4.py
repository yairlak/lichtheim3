"""M4 Phase C driver — fits every probe declared in `m4_probe_protocol.md`.

Analysis-only: no model, no checkpoint, no decoder, no token generation.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.length_effect_analysis.m4_probes import (      # noqa: E402
    ALL_STAGES, ALPHAS, BOOT_B, BOOT_SEED, CONFIRMATORY, ELIGIBLE, EXTENSION,
    INSTR, LENGTH_NOTE, LTM_PROBE_STAGES, M4, MAX_POSITION, N_FOLDS,
    PROBE_STAGES, SEEDS, STAGE1, STAGES, WM_PROBE_STAGES, Bootstrap, Features,
    RidgeMultiOutput, actual_output_rows, assign_folds, cell_weights, git,
    item_error, load_item_table, macro_f1, ols_slope, pct, phoneme_targets,
    row_weights, run_ordered_probe, sha)

SHORT = (3, 4, 5)
LONG = (7, 8, 9)


def length_group(n):
    return "short (3-5)" if n in SHORT else "long (7-9)"


# ------------------------------------------------------------ unordered probe

def count_vectors(items, n_classes, Y):
    C = np.zeros((len(items), n_classes))
    for i in range(len(items)):
        L = int(items["phoneme_length"].iloc[i])
        for p in range(L):
            C[i, Y[i, p]] += 1.0        # repeated phonemes accumulate
    return C


def run_unordered(items, folds, C, feats, seed, stage):
    X = feats.get(stage, seed)
    fold = folds["fold"].to_numpy()
    pred = np.full_like(C, np.nan)
    base = np.full_like(C, np.nan)
    chosen = {}
    for k in range(N_FOLDS):
        te, tr = fold == k, fold != k
        inner = [f for f in range(N_FOLDS) if f != k]
        mse = {a: [] for a in ALPHAS}
        for j in inner:
            itr = tr & (fold != j)
            ite = fold == j
            mdl = RidgeMultiOutput(X[itr], C[itr])
            for a in ALPHAS:
                mse[a].append(float(((mdl.predict(X[ite], a) - C[ite]) ** 2).mean()))
        means = {a: float(np.mean(v)) for a, v in mse.items()}
        best = min(ALPHAS, key=lambda a: (means[a], a))
        chosen[k] = best
        mdl = RidgeMultiOutput(X[tr], C[tr])
        pred[te] = mdl.predict(X[te], best)
        base[te] = C[tr].mean(0)            # training-fold mean-count baseline
    return pred, base, chosen


def cos_rows(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-12)


# =================================================================== main

def main() -> int:
    t0 = time.time()
    os.makedirs(M4, exist_ok=True)
    items = load_item_table()
    folds = assign_folds(items)
    Y, phon_vocab = phoneme_targets(items)
    feats = Features(items["row_index"].to_numpy())

    # ------------------------------------------------ C3 probe_folds.tsv
    pf = folds[["item_id", "exposure_status", "phoneme_length", "fold"]].copy()
    pf["split_unit"] = "item_id"
    pf["fold_assignment_rule"] = ("rank of item_id within exposure_status x "
                                  "phoneme_length stratum, mod 5 (deterministic, "
                                  "no RNG)")
    pf["identical_across_seeds"] = True
    pf["identical_across_stages"] = True
    pf.to_csv(os.path.join(M4, "probe_folds.tsv"), sep="\t", index=False)
    print(f"folds: {len(pf)} eligible items, {pf['fold'].nunique()} folds")

    # ------------------------------------------------ C4/C5 ordered probe
    oof, alphas_used = [], []
    for seed in SEEDS:
        for stage in PROBE_STAGES:
            df, ch = run_ordered_probe(items, folds, Y, feats, seed, stage)
            df["variant"] = "primary"
            oof.append(df)
            alphas_used += [{"seed": seed, "stage": stage, "variant": "primary",
                             "outer_fold": k, "alpha": a} for k, a in ch.items()]
            print(f"  [{seed}] {stage:28s} err="
                  f"{1 - df['correct'].mean():.4f}  ({time.time()-t0:.0f}s)")
    oof = pd.concat(oof, ignore_index=True)

    act = actual_output_rows(items)
    act["variant"] = "primary"
    act["fold"] = folds.set_index("item_id").loc[act["item_id"], "fold"].to_numpy()
    act["target_class"] = -1
    act["predicted_class"] = -1
    act["decision_margin"] = np.nan
    ordered_all = pd.concat([oof, act[oof.columns]], ignore_index=True)
    ordered_all.to_csv(os.path.join(M4, "ordered_probe_oof_predictions.tsv"),
                       sep="\t", index=False)

    # ------------------------------------------------ sensitivity variants
    sens = []
    for seed in SEEDS:
        for stage in PROBE_STAGES:
            d, ch = run_ordered_probe(items, folds, Y, feats, seed, stage,
                                      weighted=False)
            d["variant"] = "unweighted"
            sens.append(d)
            alphas_used += [{"seed": seed, "stage": stage, "variant": "unweighted",
                             "outer_fold": k, "alpha": a} for k, a in ch.items()]
        d, ch = run_ordered_probe(items, folds, Y, feats, seed, "s_hat", pca=128)
        d["variant"] = "s_hat_pca128"
        sens.append(d)
        alphas_used += [{"seed": seed, "stage": "s_hat", "variant": "s_hat_pca128",
                         "outer_fold": k, "alpha": a} for k, a in ch.items()]
        d, ch = run_ordered_probe(items, folds, Y, feats, seed, "s_hat",
                                  normalise=True)
        d["variant"] = "s_hat_normalised_q"
        sens.append(d)
        alphas_used += [{"seed": seed, "stage": "s_hat",
                         "variant": "s_hat_normalised_q", "outer_fold": k,
                         "alpha": a} for k, a in ch.items()]
        print(f"  [{seed}] sensitivity variants done ({time.time()-t0:.0f}s)")
    sens = pd.concat(sens, ignore_index=True)
    # every OOF prediction is saved; the sensitivity variants go to their own
    # file so the required table stays the primary analysis alone
    sens.to_csv(os.path.join(M4, "ordered_probe_oof_predictions_sensitivity.tsv"),
                sep="\t", index=False)
    pd.DataFrame(alphas_used).to_csv(
        os.path.join(M4, "ordered_probe_selected_alphas.tsv"), sep="\t",
        index=False)

    everything = pd.concat([ordered_all, sens], ignore_index=True)

    # ------------------------------------------------ summary table
    rows = []
    for (seed, stage, variant, exp), g in everything.groupby(
            ["seed", "stage", "variant", "exposure_status"]):
        for lg in ("all", "short (3-5)", "long (7-9)"):
            gg = g if lg == "all" else g[g["phoneme_length"].map(length_group) == lg]
            if not len(gg):
                continue
            probe = STAGES[stage][1] == "probe"
            rows.append({
                "seed": seed, "stage": stage, "stage_label": STAGES[stage][0],
                "stage_kind": STAGES[stage][1], "variant": variant,
                "exposure_status": exp, "length_group": lg,
                "n_items": gg["item_id"].nunique(), "n_positions": len(gg),
                "token_error": 1.0 - gg["correct"].mean(),
                "top1_accuracy": gg["correct"].mean(),
                "macro_f1": (macro_f1(gg["target_class"].to_numpy(),
                                      gg["predicted_class"].to_numpy())
                             if probe else np.nan),
                "mean_decision_margin": (float(gg["decision_margin"].mean())
                                         if probe else np.nan),
            })
    summary = pd.DataFrame(rows).sort_values(
        ["variant", "stage", "exposure_status", "length_group", "seed"])
    summary["measurement_type"] = np.where(
        summary["stage_kind"] == "probe",
        "diagnostic OOF linear readout (linearly accessible information)",
        "actual trained-model output (not a probe)")
    summary.to_csv(os.path.join(M4, "ordered_probe_summary.tsv"), sep="\t",
                   index=False)

    # ------------------------------------------------ length slopes
    ie = item_error(everything.assign(
        stage=everything["stage"] + "|" + everything["variant"]))
    ie[["stage", "variant"]] = ie["stage"].str.split("|", expand=True)
    slope_rows = []
    for (seed, stage, variant, exp), g in ie.groupby(
            ["seed", "stage", "variant", "exposure_status"]):
        slope_rows.append({
            "seed": seed, "stage": stage, "stage_label": STAGES[stage][0],
            "stage_kind": STAGES[stage][1], "variant": variant,
            "exposure_status": exp, "n_items": len(g),
            "length_slope_token_error_per_phoneme":
                ols_slope(g["phoneme_length"], g["token_error"]),
            "mean_token_error": float(g["token_error"].mean()),
            "length_note": LENGTH_NOTE})
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(os.path.join(M4, "ordered_probe_length_slopes.tsv"), sep="\t",
                  index=False)

    # ------------------------------------------------ exact-length table
    ex = ie.groupby(["seed", "stage", "variant", "exposure_status",
                     "phoneme_length"], as_index=False).agg(
        n_items=("item_id", "nunique"), token_error=("token_error", "mean"))
    ex["stage_label"] = ex["stage"].map(lambda s: STAGES[s][0])
    ex["stage_kind"] = ex["stage"].map(lambda s: STAGES[s][1])
    ex["length_note"] = LENGTH_NOTE
    ex.to_csv(os.path.join(M4, "ordered_probe_exact_length.tsv"), sep="\t",
              index=False)

    # ------------------------------------------------ C6 unordered probe
    C = count_vectors(items, len(phon_vocab), Y)
    uo_rows, uo_sum = [], []
    unordered_stages = ["ltm_encoder_hidden", "s_hat", "ltm_decoder_h0",
                        "wm_encoder_hidden"]
    for seed in SEEDS:
        for stage in unordered_stages:
            pred, base, ch = run_unordered(items, folds, C, feats, seed, stage)
            cp, cb = cos_rows(pred, C), cos_rows(base, C)
            mae = np.abs(pred - C).mean(1) / np.maximum(C.mean(1), 1e-12)
            mab = np.abs(base - C).mean(1) / np.maximum(C.mean(1), 1e-12)
            for i in range(len(items)):
                uo_rows.append({
                    "seed": seed, "stage": stage,
                    "item_id": items["item_id"].iloc[i],
                    "exposure_status": items["exposure_status"].iloc[i],
                    "phoneme_length": int(items["phoneme_length"].iloc[i]),
                    "fold": int(folds["fold"].iloc[i]),
                    "cosine_pred_target": cp[i], "cosine_baseline_target": cb[i],
                    "normalised_mae": mae[i], "normalised_mae_baseline": mab[i]})
            d = pd.DataFrame(uo_rows[-len(items):])
            for exp, g in d.groupby("exposure_status"):
                for lg in ("all", "short (3-5)", "long (7-9)"):
                    gg = g if lg == "all" else g[g["phoneme_length"].map(length_group) == lg]
                    uo_sum.append({
                        "seed": seed, "stage": stage,
                        "stage_label": STAGES[stage][0],
                        "exposure_status": exp, "length_group": lg,
                        "n_items": len(gg),
                        "cosine_pred_target": float(gg["cosine_pred_target"].mean()),
                        "cosine_baseline_target": float(gg["cosine_baseline_target"].mean()),
                        "cosine_improvement_over_training_fold_mean_baseline":
                            float((gg["cosine_pred_target"]
                                   - gg["cosine_baseline_target"]).mean()),
                        "normalised_mae": float(gg["normalised_mae"].mean()),
                        "normalised_mae_baseline": float(gg["normalised_mae_baseline"].mean()),
                        "selected_alphas": json.dumps(ch)})
        print(f"  [{seed}] unordered done ({time.time()-t0:.0f}s)")
    pd.DataFrame(uo_rows).to_csv(
        os.path.join(M4, "unordered_probe_oof_predictions.tsv"), sep="\t",
        index=False)
    uo = pd.DataFrame(uo_sum)
    uo["note"] = ("Item-level count-vector recovery. Gold-prefix premotor is "
                  "deliberately excluded: it is timestep-specific and has "
                  "already consumed the true phonemes. Cosine is NOT comparable "
                  "in units to ordered classification accuracy.")
    uo.to_csv(os.path.join(M4, "unordered_probe_summary.tsv"), sep="\t",
              index=False)

    # ------------------------------------------------ C7 decoder utilisation
    prim = ie[ie["variant"].isin(["primary"])]
    du = prim[prim["stage"].isin(["ltm_decoder_h0", "ltm_premotor_gold_prefix",
                                  "ltm_actual_gold_prefix_output"])]
    du = du.groupby(["seed", "stage", "exposure_status", "phoneme_length"],
                    as_index=False).agg(n_items=("item_id", "nunique"),
                                        token_error=("token_error", "mean"))
    du["accuracy"] = 1 - du["token_error"]
    du = du.pivot_table(index=["seed", "exposure_status", "phoneme_length",
                               "n_items"], columns="stage",
                        values="accuracy").reset_index()
    du = du.rename(columns={
        "ltm_decoder_h0": "oof_accuracy_from_ltm_decoder_h0",
        "ltm_premotor_gold_prefix": "oof_accuracy_from_ltm_premotor_gold_prefix",
        "ltm_actual_gold_prefix_output": "actual_ltm_gold_prefix_accuracy"})
    du["premotor_probe_minus_actual_output"] = (
        du["oof_accuracy_from_ltm_premotor_gold_prefix"]
        - du["actual_ltm_gold_prefix_accuracy"])
    du["h0_probe_minus_premotor_probe"] = (
        du["oof_accuracy_from_ltm_decoder_h0"]
        - du["oof_accuracy_from_ltm_premotor_gold_prefix"])
    du["note"] = ("h0 is a global item vector; premotor is timestep-specific "
                  "under the gold prefix and has already consumed the true "
                  "preceding phonemes. Probe columns are linearly accessible "
                  "information, not information the model uses.")
    du.to_csv(os.path.join(M4, "decoder_utilisation.tsv"), sep="\t", index=False)

    # ------------------------------------------------ C9 bootstrap
    boot_stages = ALL_STAGES
    conf = items[items["exposure_status"].isin(CONFIRMATORY)].reset_index(drop=True)
    conf_ids = conf["item_id"].tolist()
    pos_of = {x: i for i, x in enumerate(conf_ids)}
    err = np.full((len(boot_stages), len(SEEDS), len(conf_ids)), np.nan)
    piv = prim.set_index(["stage", "seed", "item_id"])["token_error"]
    for si, st in enumerate(boot_stages):
        for sj, sd in enumerate(SEEDS):
            v = piv.loc[(st, sd)]
            err[si, sj] = v.reindex(conf_ids).to_numpy()
    assert np.isfinite(err).all(), "missing OOF item error somewhere"
    strata = (conf["exposure_status"] + "|"
              + conf["phoneme_length"].astype(str)).to_numpy()
    boot = Bootstrap(err, conf["phoneme_length"].to_numpy(), strata,
                     conf["exposure_status"].to_numpy())
    B = boot.run()
    print(f"  bootstrap done ({time.time()-t0:.0f}s)")

    # ------------------------------------------------ C10 stage contrasts
    def seed_vals(stage, exp, what):
        g = prim[(prim["stage"] == stage) & (prim["exposure_status"] == exp)]
        out = []
        for sd in SEEDS:
            gg = g[g["seed"] == sd]
            out.append(ols_slope(gg["phoneme_length"], gg["token_error"])
                       if what == "slope" else float(gg["token_error"].mean()))
        return out

    con = []

    def add(kind, exp, stage, arr, seedv, extra=""):
        lo, hi = pct(arr)
        con.append({"contrast_kind": kind, "exposure_status": exp,
                    "stage": stage, "stage_label": STAGES[stage][0]
                    if stage in STAGES else stage,
                    "seed19": seedv[0], "seed20": seedv[1], "seed21": seedv[2],
                    "seed22": seedv[3], "seed_mean": float(np.mean(seedv)),
                    "bootstrap_mean": float(np.nanmean(arr)),
                    "ci_low": lo, "ci_high": hi,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0), "note": extra})

    for exp in CONFIRMATORY:
        for si, st in enumerate(boot_stages):
            add("length_slope", exp, st, B[exp]["slope"][:, si],
                seed_vals(st, exp, "slope"), LENGTH_NOTE)
            add("mean_token_error", exp, st, B[exp]["mean"][:, si],
                seed_vals(st, exp, "mean"))
    # novel minus trained, paired
    for si, st in enumerate(boot_stages):
        d = B["NOVEL_PSEUDOWORD"]["slope"][:, si] - B["TRAINED_REAL_EXACT"]["slope"][:, si]
        sv = [a - b for a, b in zip(seed_vals(st, "NOVEL_PSEUDOWORD", "slope"),
                                    seed_vals(st, "TRAINED_REAL_EXACT", "slope"))]
        add("novel_minus_trained_slope", "NOVEL-TRAINED", st, d, sv, LENGTH_NOTE)

    transitions = [("ltm_encoder_hidden", "s_hat", "encoder -> s_hat"),
                   ("s_hat", "ltm_decoder_h0", "s_hat -> h0"),
                   ("ltm_decoder_h0", "ltm_premotor_gold_prefix",
                    "h0 -> premotor (GOLD PREFIX CONTEXT BEGINS)"),
                   ("ltm_premotor_gold_prefix", "ltm_actual_gold_prefix_output",
                    "premotor probe -> actual output")]
    for a, b, lbl in transitions:
        ia, ib = boot_stages.index(a), boot_stages.index(b)
        for exp in CONFIRMATORY:
            dm = B[exp]["mean"][:, ib] - B[exp]["mean"][:, ia]
            sv = [x - y for x, y in zip(seed_vals(b, exp, "mean"),
                                        seed_vals(a, exp, "mean"))]
            add(f"delta_mean_token_error [{lbl}]", exp, b, dm, sv,
                "positive = the later stage has MORE held-out error")
            ds = B[exp]["slope"][:, ib] - B[exp]["slope"][:, ia]
            sv = [x - y for x, y in zip(seed_vals(b, exp, "slope"),
                                        seed_vals(a, exp, "slope"))]
            add(f"delta_length_slope [{lbl}]", exp, b, ds, sv, LENGTH_NOTE)

    # LTM vs WM on novel forms
    for lt, wm in (("ltm_encoder_hidden", "wm_encoder_hidden"),
                   ("ltm_premotor_gold_prefix", "wm_premotor_gold_prefix")):
        il, iw = boot_stages.index(lt), boot_stages.index(wm)
        for exp in CONFIRMATORY:
            dm = B[exp]["mean"][:, il] - B[exp]["mean"][:, iw]
            sv = [x - y for x, y in zip(seed_vals(lt, exp, "mean"),
                                        seed_vals(wm, exp, "mean"))]
            add(f"ltm_minus_wm_mean_token_error [{lt} vs {wm}]", exp, lt, dm, sv,
                "positive = LTM stage has MORE held-out error than WM control")
            ds = B[exp]["slope"][:, il] - B[exp]["slope"][:, iw]
            sv = [x - y for x, y in zip(seed_vals(lt, exp, "slope"),
                                        seed_vals(wm, exp, "slope"))]
            add(f"ltm_minus_wm_length_slope [{lt} vs {wm}]", exp, lt, ds, sv,
                LENGTH_NOTE)

    contrasts = pd.DataFrame(con)
    contrasts.to_csv(os.path.join(M4, "stage_contrasts.tsv"), sep="\t",
                     index=False)

    # ------------------------------------------------ provenance / validation
    prov = {
        "phase": "M4 Phase C - stagewise representational localisation",
        "analysis_only": True, "model_inference": False, "decoder_executed": False,
        "tokens_generated": False, "training_performed": False,
        "weights_modified": False, "architecture_changed": False,
        "repository_head": git("rev-parse", "HEAD"),
        "repository_dirty": bool(git("status", "--porcelain").strip()),
        "protocol_frozen_before_fitting": "m4_probe_protocol.md",
        "inputs": {
            "representations_npz": sha(os.path.join(INSTR, "representations.npz")),
            "item_summary": sha(os.path.join(INSTR, "item_summary.tsv")),
            "timestep_metrics": sha(os.path.join(INSTR, "timestep_metrics.tsv")),
            **{f"ltm_encoder_hidden_seed{s}":
               sha(os.path.join(STAGE1, f"ltm_encoder_hidden_seed{s}.npy"))
               for s in SEEDS}},
        "seeds": SEEDS, "seed_21_excluded": False,
        "populations": {"confirmatory": list(CONFIRMATORY),
                        "extension": list(EXTENSION),
                        "n_eligible_items": int(len(items))},
        "folds": {"n_folds": N_FOLDS, "split_unit": "item_id",
                  "rule": "rank within exposure x length stratum, mod 5",
                  "identical_across_seeds_and_stages": True},
        "alpha_grid": list(ALPHAS),
        "alpha_selection": "nested leave-one-training-fold-out inside training data only",
        "preprocessing_fitted_on": "training folds only",
        "weighting": "exposure_status x phoneme_length cells equal total training weight",
        "bootstrap": {"B": BOOT_B, "rng_seed": BOOT_SEED,
                      "interval": "95% percentile",
                      "hierarchy": "seeds with replacement, then items within "
                                   "exposure x length strata",
                      "probes_refitted_inside_replicates": False,
                      "conditional_on_fitted_oof_probes": True},
        "observed_lengths": [3, 4, 5, 7, 8, 9], "length_note": LENGTH_NOTE,
        "vocabulary_discipline": ("probe results are reported as linearly "
                                  "accessible information, never as information "
                                  "the model uses"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(M4, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    print(f"done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
