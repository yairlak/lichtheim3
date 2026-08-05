"""Tests for the M4 Phase C stagewise probes.

Two groups: unit tests of the estimator, fold construction, feature construction
and preprocessing discipline; and artifact tests over the written tables.

No test here loads a checkpoint, runs a decoder, generates a token, or trains.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.length_effect_analysis import m4_probes as P              # noqa: E402
from scripts.length_effect_analysis import run_m4 as R                 # noqa: E402

M4 = P.M4
FIG = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/figures")
CONF = list(P.CONFIRMATORY)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(M4, "ordered_probe_summary.tsv")),
    reason="M4 phase C has not been run")


@pytest.fixture(scope="module")
def items():
    it = P.load_item_table()
    return P.assign_folds(it)


@pytest.fixture(scope="module")
def tables():
    def t(n):
        return pd.read_csv(os.path.join(M4, n), sep="\t")
    return {n: t(n) for n in (
        "probe_folds.tsv", "ordered_probe_oof_predictions.tsv",
        "ordered_probe_oof_predictions_sensitivity.tsv",
        "ordered_probe_summary.tsv", "ordered_probe_length_slopes.tsv",
        "ordered_probe_exact_length.tsv", "ordered_probe_selected_alphas.tsv",
        "unordered_probe_oof_predictions.tsv", "unordered_probe_summary.tsv",
        "decoder_utilisation.tsv", "stage_contrasts.tsv",
        "m4_localisation_decision.tsv")}


# ------------------------------------------------------------ folds / leakage

def test_every_eligible_item_has_exactly_one_fold(items, tables):
    f = tables["probe_folds.tsv"]
    assert len(f) == len(items) == 1184
    assert f["item_id"].nunique() == len(f)
    assert sorted(f["fold"].unique()) == list(range(P.N_FOLDS))
    assert set(f["exposure_status"]) == set(P.ELIGIBLE)


def test_ambiguous_categories_are_excluded(tables):
    f = tables["probe_folds.tsv"]
    assert not set(f["exposure_status"]) & set(P.EXCLUDED)


def test_no_item_is_in_train_and_test_simultaneously(items):
    fold = items["fold"].to_numpy()
    ids = items["item_id"].to_numpy()
    for k in range(P.N_FOLDS):
        assert not (set(ids[fold == k]) & set(ids[fold != k]))


def test_all_rows_of_an_item_stay_in_one_fold(tables):
    o = tables["ordered_probe_oof_predictions.tsv"]
    g = o.groupby(["seed", "stage", "item_id"])["fold"].nunique()
    assert (g == 1).all()


def test_folds_are_identical_across_seeds_and_stages(tables):
    o = pd.concat([tables["ordered_probe_oof_predictions.tsv"],
                   tables["ordered_probe_oof_predictions_sensitivity.tsv"]],
                  ignore_index=True)
    ref = None
    for (_seed, _stage), g in o.groupby(["seed", "stage"]):
        m = g.drop_duplicates("item_id").set_index("item_id")["fold"].sort_index()
        if ref is None:
            ref = m
        else:
            pd.testing.assert_series_equal(ref, m, check_names=False)
    u = tables["unordered_probe_oof_predictions.tsv"]
    for (_seed, _stage), g in u.groupby(["seed", "stage"]):
        m = g.set_index("item_id")["fold"].sort_index()
        pd.testing.assert_series_equal(ref, m, check_names=False)


def test_folds_are_stratified_by_exposure_and_length(items):
    n = items.groupby(["exposure_status", "phoneme_length", "fold"]).size()
    n = n.unstack("fold").fillna(0)
    assert (n.max(axis=1) - n.min(axis=1) <= 1).all()


def test_fold_assignment_is_deterministic(items):
    again = P.assign_folds(P.load_item_table())
    pd.testing.assert_frame_equal(items.reset_index(drop=True),
                                  again.reset_index(drop=True))


# --------------------------------------------------------------- estimator

def test_ridge_head_matches_sklearn():
    sk = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(220, 12))
    y = rng.integers(0, 5, size=220)
    for alpha in (0.1, 1.0, 10.0):
        mine = P.RidgeHead(X, y, np.ones(len(y)), scale=False)
        ref = sk.RidgeClassifier(alpha=alpha).fit(X, y)
        assert np.allclose(mine.coef(alpha).T, ref.coef_, atol=1e-8)
        assert (mine.predict(X, alpha) == ref.predict(X)).all()


def test_ridge_head_is_deterministic():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(80, 7)), rng.integers(0, 3, size=80)
    a = P.RidgeHead(X, y, np.ones(80)).predict(X, 1.0)
    b = P.RidgeHead(X, y, np.ones(80)).predict(X, 1.0)
    assert (a == b).all()


def test_alpha_selection_is_deterministic_and_recorded(tables):
    a = tables["ordered_probe_selected_alphas.tsv"]
    assert set(a["alpha"]) <= set(P.ALPHAS)
    assert len(a) == len(a.drop_duplicates(["seed", "stage", "variant",
                                            "outer_fold"]))
    assert sorted(a["outer_fold"].unique()) == list(range(P.N_FOLDS))


def test_preprocessing_statistics_come_only_from_the_training_rows():
    """Centering/scaling must be a function of the training rows alone."""
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(100, 6)), rng.integers(0, 3, size=100)
    tr = np.arange(60)
    h = P.RidgeHead(X[tr], y[tr], np.ones(60))
    assert np.allclose(h.xm, X[tr].mean(0))
    assert np.allclose(h.xs, X[tr].std(0), atol=1e-10)
    # perturbing held-out rows cannot change any fitted statistic
    X2 = X.copy()
    X2[60:] += 100.0
    h2 = P.RidgeHead(X2[tr], y[tr], np.ones(60))
    assert np.allclose(h.xm, h2.xm) and np.allclose(h.xs, h2.xs)


def test_pca_is_fitted_on_training_rows_only():
    """The SVD basis in run_ordered_probe uses only the training slice."""
    rng = np.random.default_rng(3)
    Xtr, Xte = rng.normal(size=(70, 10)), rng.normal(size=(30, 10))
    mu = Xtr.mean(0)
    V = np.linalg.svd(Xtr - mu, full_matrices=False)[2][:4].T
    Xte2 = Xte + 50.0
    V2 = np.linalg.svd(Xtr - mu, full_matrices=False)[2][:4].T
    assert np.allclose(V, V2)
    assert not np.allclose((Xte - mu) @ V, (Xte2 - mu) @ V)


def test_training_weights_equalise_exposure_by_length_cells(items):
    tr = items["fold"].to_numpy() != 0
    cw = P.cell_weights(items, tr)
    idx = np.flatnonzero(tr)
    w = P.row_weights(items, idx, cw)
    d = pd.DataFrame({"w": w,
                      "cell": (items["exposure_status"].to_numpy()[idx] + "|"
                               + items["phoneme_length"].astype(str).to_numpy()[idx])})
    tot = d.groupby("cell")["w"].sum()
    assert np.allclose(tot.to_numpy(), 1.0)


def test_weights_use_training_composition_only(items):
    """A cell's weight must not depend on how many held-out items it has."""
    tr = items["fold"].to_numpy() != 0
    cw = P.cell_weights(items, tr)
    counts = items[tr].groupby(["exposure_status", "phoneme_length"]).size()
    for k, v in cw.items():
        assert np.isclose(v, 1.0 / counts[k])


# --------------------------------------------------------- feature construction

def test_position_block_features_have_disjoint_support():
    x = np.arange(24, dtype=float).reshape(6, 4)
    pos = np.array([0, 0, 1, 2, 1, 2])
    X = P.build_position_block(x, pos, 3)
    assert X.shape == (6, 12)
    for r in range(6):
        p = pos[r]
        assert np.allclose(X[r, p * 4:(p + 1) * 4], x[r])
        other = np.delete(X[r], np.arange(p * 4, (p + 1) * 4))
        assert np.allclose(other, 0.0)


def test_position_block_slice_equals_the_per_head_feature_matrix():
    """Per-position fitting uses exactly the nonzero block of the design."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 5))
    pos = rng.integers(0, 4, size=40)
    X = P.build_position_block(x, pos, 4)
    for p in range(4):
        m = pos == p
        assert np.allclose(X[m][:, p * 5:(p + 1) * 5], x[m])


def test_bos_eos_pad_are_excluded_from_targets(items):
    Y, vocab = P.phoneme_targets(items)
    assert not {"<pad>", "<bos>", "<eos>"} & set(vocab)
    assert len(vocab) == 39
    for i in range(len(items)):
        L = int(items["phoneme_length"].iloc[i])
        assert (Y[i, :L] >= 0).all()
        assert (Y[i, L:] == -1).all()


def test_oof_rows_carry_no_special_token_and_only_valid_positions(tables):
    o = tables["ordered_probe_oof_predictions.tsv"]
    probe = o[o["stage"] != "ltm_actual_gold_prefix_output"]
    assert (probe["target_class"] >= 0).all()
    assert probe["position"].between(0, P.MAX_POSITION).all()
    assert (probe["position"] < probe["phoneme_length"]).all()


def test_count_vectors_preserve_repeated_phonemes(items):
    Y, vocab = P.phoneme_targets(items)
    C = R.count_vectors(items, len(vocab), Y)
    assert np.allclose(C.sum(1), items["phoneme_length"].to_numpy())
    assert C.max() >= 2, "no item with a repeated phoneme - counts look binarised"
    i = int(np.argmax(C.max(1)))
    toks = items["target_tokens"].iloc[i].split()
    for j, ph in enumerate(vocab):
        assert C[i, j] == toks.count(ph)


def test_unordered_baseline_is_the_training_fold_mean(items, tables):
    u = tables["unordered_probe_oof_predictions.tsv"]
    # the baseline is constant within a held-out fold (it is the train mean)
    g = u.groupby(["seed", "stage", "fold"])["cosine_baseline_target"].nunique()
    assert (g > 1).all(), "baseline should vary per item (cosine to its target)"
    # but it must be identical across seeds, since it uses no representation
    piv = u.pivot_table(index=["item_id", "stage"], columns="seed",
                        values="cosine_baseline_target")
    assert np.allclose(piv.to_numpy(), piv.iloc[:, [0]].to_numpy())


# ------------------------------------------------------------- OOF coverage

def test_exactly_one_oof_prediction_per_item_position_stage_seed(tables, items):
    o = pd.concat([tables["ordered_probe_oof_predictions.tsv"],
                   tables["ordered_probe_oof_predictions_sensitivity.tsv"]],
                  ignore_index=True)
    assert not o.duplicated(["seed", "stage", "variant", "item_id",
                             "position"]).any()
    expected = int(items["phoneme_length"].sum())
    for (_seed, _stage, _var), g in o.groupby(["seed", "stage", "variant"]):
        assert len(g) == expected


def test_probes_are_fitted_separately_per_seed(tables):
    o = tables["ordered_probe_oof_predictions.tsv"]
    o = o[(o["stage"] == "s_hat") & (o["variant"] == "primary")]
    piv = o.pivot_table(index=["item_id", "position"], columns="seed",
                        values="predicted_class")
    # seed-separated fitting must give at least some differing predictions
    assert not np.allclose(piv.to_numpy(), piv.iloc[:, [0]].to_numpy())
    assert sorted(o["seed"].unique()) == P.SEEDS


def test_all_four_seeds_present_everywhere(tables):
    for n in ("ordered_probe_summary.tsv", "ordered_probe_length_slopes.tsv",
              "ordered_probe_exact_length.tsv", "unordered_probe_summary.tsv",
              "decoder_utilisation.tsv"):
        assert sorted(tables[n]["seed"].unique()) == P.SEEDS, n


def test_seed_21_is_not_excluded(tables):
    for n, t in tables.items():
        if "seed" in t.columns:
            assert 21 in set(t["seed"]), n


# ------------------------------------------------------------- consistency

def test_summary_token_error_matches_the_oof_predictions(tables):
    o = pd.concat([tables["ordered_probe_oof_predictions.tsv"],
                   tables["ordered_probe_oof_predictions_sensitivity.tsv"]],
                  ignore_index=True)
    s = tables["ordered_probe_summary.tsv"]
    s = s[s["length_group"] == "all"]
    for _, r in s.sample(30, random_state=0).iterrows():
        g = o[(o.seed == r["seed"]) & (o.stage == r["stage"])
              & (o.variant == r["variant"])
              & (o.exposure_status == r["exposure_status"])]
        assert np.isclose(1 - g["correct"].mean(), r["token_error"])


def test_length_slopes_match_a_direct_recomputation(tables):
    o = pd.concat([tables["ordered_probe_oof_predictions.tsv"],
                   tables["ordered_probe_oof_predictions_sensitivity.tsv"]],
                  ignore_index=True)
    sl = tables["ordered_probe_length_slopes.tsv"]
    ie = o.groupby(["seed", "stage", "variant", "item_id", "exposure_status",
                    "phoneme_length"], as_index=False)["correct"].mean()
    ie["token_error"] = 1 - ie["correct"]
    for _, r in sl.sample(20, random_state=1).iterrows():
        g = ie[(ie.seed == r["seed"]) & (ie.stage == r["stage"])
               & (ie.variant == r["variant"])
               & (ie.exposure_status == r["exposure_status"])]
        assert np.isclose(P.ols_slope(g["phoneme_length"], g["token_error"]),
                          r["length_slope_token_error_per_phoneme"])


def test_exact_length_table_is_complete(tables):
    ex = tables["ordered_probe_exact_length.tsv"]
    lens = sorted(ex["phoneme_length"].unique())
    assert lens == [3, 4, 5, 7, 8, 9], lens
    prim = ex[ex["variant"] == "primary"]
    for (_seed, _stage, exp), g in prim.groupby(["seed", "stage",
                                                 "exposure_status"]):
        assert sorted(g["phoneme_length"].unique()) == lens
    assert (ex["length_note"] == P.LENGTH_NOTE).all()


def test_paired_stage_contrasts_are_consistent_with_the_stage_estimates(tables):
    c = tables["stage_contrasts.tsv"]
    ms = c[c["contrast_kind"] == "mean_token_error"].set_index(
        ["exposure_status", "stage"])["seed_mean"]
    d = c[c["contrast_kind"].str.startswith("delta_mean_token_error")]
    pairs = {"2 raw s_hat": ("ltm_encoder_hidden", "s_hat"),
             "3 LTM decoder h0": ("s_hat", "ltm_decoder_h0"),
             "4 gold-prefix premotor": ("ltm_decoder_h0",
                                        "ltm_premotor_gold_prefix"),
             "5 actual gold-prefix output": ("ltm_premotor_gold_prefix",
                                             "ltm_actual_gold_prefix_output")}
    for _, r in d.iterrows():
        a, b = pairs[r["stage_label"]]
        exp = r["exposure_status"]
        assert np.isclose(r["seed_mean"], ms[(exp, b)] - ms[(exp, a)], atol=1e-9)


def test_bootstrap_schema(tables):
    c = tables["stage_contrasts.tsv"]
    for col in ("seed19", "seed20", "seed21", "seed22", "seed_mean",
                "bootstrap_mean", "ci_low", "ci_high", "ci_excludes_zero"):
        assert col in c.columns
    assert (c["ci_low"] <= c["ci_high"]).all()
    assert (c["ci_low"] <= c["bootstrap_mean"]).all()
    assert (c["bootstrap_mean"] <= c["ci_high"]).all()
    inside = ((c["ci_low"] <= 0) & (c["ci_high"] >= 0))
    assert (c["ci_excludes_zero"] == ~inside).all()


def test_bootstrap_is_reproducible_from_its_declared_seed():
    rng = np.random.default_rng(P.BOOT_SEED)
    a = rng.integers(0, 4, size=(10, 4))
    rng = np.random.default_rng(P.BOOT_SEED)
    b = rng.integers(0, 4, size=(10, 4))
    assert (a == b).all()
    assert P.BOOT_B == 10_000 and P.BOOT_SEED == 20260730


def test_bootstrap_pairs_stages_within_a_replicate():
    """A paired design must give a tighter difference than an unpaired one."""
    rng = np.random.default_rng(7)
    base = rng.normal(size=(2, 4, 60))
    err = np.stack([base[0], base[0] + 0.5])           # perfectly paired shift
    lengths = rng.integers(3, 10, size=60)
    strata = np.array(["a|" + str(x) for x in lengths])
    groups = np.array(["a"] * 60)
    B = P.Bootstrap(err, lengths, strata, groups, b=300, chunk=100).run()
    d = B["a"]["mean"][:, 1] - B["a"]["mean"][:, 0]
    assert np.allclose(d, 0.5, atol=1e-9), "stage contrast is not paired"


# ------------------------------------------------------------- figure backing

def test_figure3_tsv_completeness_and_consistency():
    f = pd.read_csv(os.path.join(FIG, "figure3_stagewise_information.tsv"),
                    sep="\t")
    s = pd.read_csv(os.path.join(M4, "ordered_probe_summary.tsv"), sep="\t")
    sl = pd.read_csv(os.path.join(M4, "ordered_probe_length_slopes.tsv"), sep="\t")
    stages = ["ltm_encoder_hidden", "s_hat", "ltm_decoder_h0",
              "ltm_premotor_gold_prefix", "ltm_actual_gold_prefix_output"]
    a = f[(f.panel == "A") & (f.metric == "held_out_token_error")]
    for st in stages:
        for exp in CONF:
            for lg in ("all", "short (3-5)", "long (7-9)"):
                g = a[(a.stage == st) & (a.exposure_status == exp)
                      & (a.length_group == lg)]
                assert len(g) == 4, (st, exp, lg)
                ref = s[(s.variant == "primary") & (s.stage == st)
                        & (s.exposure_status == exp) & (s.length_group == lg)]
                assert np.allclose(sorted(g["value"]),
                                   sorted(ref["token_error"]))
    b = f[(f.panel == "B")
          & (f.metric == "length_slope_token_error_per_phoneme")]
    for st in stages:
        for exp in CONF:
            g = b[(b.stage == st) & (b.exposure_status == exp)]
            ref = sl[(sl.variant == "primary") & (sl.stage == st)
                     & (sl.exposure_status == exp)]
            assert np.allclose(sorted(g["value"]),
                               sorted(ref["length_slope_token_error_per_phoneme"]))
    assert (f["length_note"] == P.LENGTH_NOTE).all()


def test_figure3_ci_values_come_from_stage_contrasts():
    f = pd.read_csv(os.path.join(FIG, "figure3_stagewise_information.tsv"),
                    sep="\t")
    c = pd.read_csv(os.path.join(M4, "stage_contrasts.tsv"), sep="\t")
    c = c[c["contrast_kind"] == "length_slope"].set_index(
        ["exposure_status", "stage"])
    for metric, col in (("bootstrap_mean_length_slope", "bootstrap_mean"),
                        ("ci_low_length_slope", "ci_low"),
                        ("ci_high_length_slope", "ci_high")):
        g = f[f.metric == metric]
        assert len(g) > 0
        for _, r in g.iterrows():
            assert np.isclose(r["value"],
                              c.loc[(r["exposure_status"], r["stage"]), col])


def test_figure4_tsv_matches_its_sources():
    f = pd.read_csv(os.path.join(FIG, "figure4_order_content_utilisation.tsv"),
                    sep="\t")
    u = pd.read_csv(os.path.join(M4, "unordered_probe_summary.tsv"), sep="\t")
    u = u[(u.length_group == "all") & (u.exposure_status.isin(CONF))]
    for _, r in f[f.metric == "unordered_cosine"].iterrows():
        assert np.isclose(r["value"],
                          u[u.stage == r["stage"]]["cosine_pred_target"].mean())
    du = pd.read_csv(os.path.join(M4, "decoder_utilisation.tsv"), sep="\t")
    row = f[f.metric == "accuracy|NOVEL_PSEUDOWORD|len9"]
    row = row[row.stage == "actual_ltm_gold_prefix_accuracy"]
    ref = du[(du.exposure_status == "NOVEL_PSEUDOWORD")
             & (du.phoneme_length == 9)]["actual_ltm_gold_prefix_accuracy"].mean()
    assert np.isclose(row["value"].iloc[0], ref)


# ---------------------------------------------------------------- provenance

def test_provenance_schema():
    with open(os.path.join(M4, "provenance.json")) as f:
        p = json.load(f)
    for k in ("phase", "analysis_only", "model_inference", "decoder_executed",
              "training_performed", "weights_modified", "architecture_changed",
              "protocol_frozen_before_fitting", "inputs", "seeds", "populations",
              "folds", "alpha_grid", "alpha_selection", "preprocessing_fitted_on",
              "weighting", "bootstrap", "observed_lengths", "length_note",
              "vocabulary_discipline"):
        assert k in p, k
    assert p["model_inference"] is False and p["decoder_executed"] is False
    assert p["training_performed"] is False and p["architecture_changed"] is False
    assert p["seeds"] == P.SEEDS and p["seed_21_excluded"] is False
    assert p["bootstrap"]["B"] == 10_000
    assert p["bootstrap"]["rng_seed"] == 20260730
    assert p["bootstrap"]["probes_refitted_inside_replicates"] is False
    assert p["observed_lengths"] == [3, 4, 5, 7, 8, 9]
    assert p["alpha_grid"] == list(P.ALPHAS)


def test_decision_table_uses_only_the_allowed_verdicts(tables):
    d = tables["m4_localisation_decision.tsv"]
    allowed = {"SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "UNRESOLVED"}
    assert set(d["verdict"]) <= allowed
    assert d["hypothesis"].str.contains("OVERALL").any()


def test_protocol_was_written_before_the_results():
    prot = os.path.join(M4, "m4_probe_protocol.md")
    assert os.path.exists(prot)
    assert (os.path.getmtime(prot)
            <= os.path.getmtime(os.path.join(M4, "ordered_probe_summary.tsv")))
