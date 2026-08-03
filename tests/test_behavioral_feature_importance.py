"""Tests for the Sprint-5 adapted feature-importance analysis (A15).

No checkpoint is loaded and no model inference occurs (group J).  The faithful
Dager analysis (A11) is never written to and never pooled with these values
(group B).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis import common
from scripts.behavioral_analysis import feature_importance as fi
from scripts.behavioral_analysis import plot_feature_importance as pfi
from scripts.behavioral_analysis.io import load_canonical

PKG_DIR = os.path.join(ROOT, "scripts", "behavioral_analysis")
REPORT = os.path.join(ROOT, "reports", "behavioral_wfe_fulllexicon_93a577f")
Q = os.path.join(REPORT, "feature_importance")
CTL = os.path.join(Q, "_control")
JT, JF = os.path.join(Q, "clean_joint", "tables"), os.path.join(Q, "clean_joint", "figures")
IT, IF = (os.path.join(Q, "clean_interactions", "tables"),
          os.path.join(Q, "clean_interactions", "figures"))
RT, RF = (os.path.join(Q, "route_specific", "tables"),
          os.path.join(Q, "route_specific", "figures"))
A11_DIR = os.path.join(ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
                       "behavioral_analysis", "faithful_replication")

FI_MODULES = ["feature_importance.py", "plot_feature_importance.py"]

requires_canonical = pytest.mark.skipif(
    not os.path.exists(common.CANONICAL_TABLE), reason="canonical table absent")
requires_outputs = pytest.mark.skipif(
    not os.path.exists(os.path.join(JT, "clean_main_model_fit.tsv")),
    reason="Sprint-5 outputs not generated")


@pytest.fixture(scope="module")
def canon():
    return load_canonical(common.CANONICAL_TABLE)


@pytest.fixture(scope="module")
def items(canon):
    return fi.clean_items(canon)


@pytest.fixture(scope="module")
def split(items):
    return fi.split_items(items)


@pytest.fixture(scope="module")
def frame(canon):
    return fi.analysis_frame(canon, 19)


def _read(p):
    return pd.read_csv(p, sep="\t")


def _text(*paths):
    out = []
    for p in paths:
        with open(p) as f:
            out.append(f.read())
    return "\n".join(out)


def _norm(s):
    s = s.replace("`", "").replace("*", "")
    s = re.sub(r"(?m)^\s*>\s?", "", s)
    return " ".join(s.split())


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _code_only(module: str) -> str:
    """Source with docstrings and comments removed.

    The modules *document* the constraints they obey ("no checkpoint", "Zipf is
    undefined for pseudowords", the path of the faithful analysis), so a raw
    substring scan would flag the prose that states the rule as a violation of
    it.  These guards must look at what the code does, not at what it says.
    """
    src = open(os.path.join(PKG_DIR, module)).read()
    lines = src.split("\n")
    drop = set()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            d = body[0].value
            drop.update(range(d.lineno - 1, (d.end_lineno or d.lineno)))
    kept = [l for i, l in enumerate(lines) if i not in drop]
    return "\n".join(l.split("#")[0] if l.lstrip().startswith("#") else l
                     for l in kept)


# ==========================================================  A  dataset =====

@requires_canonical
def test_A_clean_set_is_1062_items(items):
    assert len(items) == 1062
    assert items["item_id"].nunique() == 1062


@requires_canonical
def test_A_clean_composition_671_real_391_pseudo(items):
    assert int((items["lexicality"] == "real").sum()) == 671
    assert int((items["lexicality"] == "pseudo").sum()) == 391


@requires_canonical
def test_A_four_seeds_and_three_routes(canon):
    assert sorted(canon["seed"].unique()) == [19, 20, 21, 22] == common.SEEDS
    assert sorted(canon["route"].unique()) == sorted(common.ROUTES)


@requires_canonical
def test_A_seed21_included_everywhere(canon):
    assert 21 in set(canon["seed"])
    assert len(fi.analysis_frame(canon, 21)) == 1062 * 3


@requires_canonical
def test_A_rows_per_seed_are_3186(canon):
    for seed in common.SEEDS:
        assert len(fi.analysis_frame(canon, seed)) == 3186


@requires_outputs
def test_A_all_four_seeds_in_every_published_table():
    for root in (JT, IT, RT):
        for name in sorted(os.listdir(root)):
            df = _read(os.path.join(root, name))
            if "seed" not in df.columns or df.empty:
                continue
            assert sorted(df["seed"].unique()) == common.SEEDS, name


# ==================================================  B  identifiability =====

def test_B_lexicality_and_exposure_never_entered_together():
    for group in (fi.FACTORS, fi.ROUTE_SPECIFIC_FACTORS):
        assert not ({"lexicality", "lichtheim_exposure_status"} <= set(group))
        assert not ({"lexicality", "exposure_status"} <= set(group))


@requires_canonical
def test_B_confounding_is_real_and_documented(items):
    x = pd.crosstab(items["lexicality"], items["lichtheim_exposure_status"])
    assert int((x.to_numpy() != 0).sum()) == 2, "must be perfectly confounded"
    spec = json.load(open(os.path.join(
        CTL, "feature_importance_analysis_spec.json")))
    assert spec["identifiability"]["lexicality_exposure_perfectly_confounded"]
    assert spec["identifiability"]["factor_label"] == "lexicality/exposure contrast"


def test_B_zipf_absent_from_clean_all_item_fi():
    assert "zipf_frequency" not in fi.FACTORS
    assert not any("zipf" in f.lower() or "frequency" in f.lower()
                   for f in fi.FACTORS + fi.ROUTE_SPECIFIC_FACTORS)
    assert "zipf" not in _code_only("feature_importance.py").lower()


@requires_canonical
def test_B_pseudowords_never_receive_a_frequency(canon, frame):
    assert "zipf_frequency" not in frame.columns
    d = canon[canon["in_LICHTHEIM_CLEAN"]]
    z = pd.to_numeric(d[d["source_lexicality"] == "pseudo"]["zipf_frequency"],
                      errors="coerce")
    assert bool(z.isna().all()), "no pseudoword may carry a Zipf value"


@requires_outputs
def test_B_faithful_and_adapted_have_distinct_paths():
    assert os.path.isdir(A11_DIR)
    assert not Q.startswith(A11_DIR) and not A11_DIR.startswith(Q)
    for base, _, names in os.walk(Q):
        for n in names:
            assert "figure2B" not in n, "adapted must not reuse faithful names"


@requires_outputs
def test_B_faithful_never_pooled_with_adapted():
    txt = _norm(_text(os.path.join(Q, "faithful_vs_adapted.md"),
                      os.path.join(Q, "feature_importance_results.md"))).lower()
    assert "never placed on one quantitative axis" in txt
    assert "not a pooled result" in txt
    for banned in ("pooled importance", "combined feature importance",
                   "average of a11 and a15"):
        assert banned not in txt


def test_B_no_module_writes_into_the_faithful_directory():
    for m in FI_MODULES:
        code = _code_only(m)
        assert "faithful_replication" not in code, m
        assert "figure2B" not in code, m


# ===========================================================  C  split ======

@requires_canonical
def test_C_split_is_grouped_by_item_and_disjoint(split):
    train, test = split
    assert set(train).isdisjoint(set(test))
    assert len(train) + len(test) == 1062
    assert len(test) == int(round(0.2 * 1062)) == 212


@requires_canonical
def test_C_all_route_rows_of_an_item_stay_together(canon, split):
    train, test = split
    tr, te = set(train), set(test)
    for seed in common.SEEDS:
        df = fi.analysis_frame(canon, seed)
        a = df[df["item_id"].isin(tr)]
        b = df[df["item_id"].isin(te)]
        assert len(a) == len(train) * 3 and len(b) == len(test) * 3
        for part in (a, b):
            counts = part.groupby("item_id")["route"].nunique()
            assert bool((counts == 3).all())
        assert set(a["item_id"]).isdisjoint(set(b["item_id"]))


@requires_canonical
def test_C_same_split_reused_across_seeds(canon, items, split):
    """The split is drawn from items only, so it cannot vary with seed."""
    a = fi.split_items(items)
    b = fi.split_items(fi.clean_items(canon))
    assert a == b == split


@requires_outputs
def test_C_published_split_matches_the_recomputed_split(split):
    train, test = split
    tr = _read(os.path.join(CTL, "fi_train_items.tsv"))
    te = _read(os.path.join(CTL, "fi_test_items.tsv"))
    assert sorted(tr["item_id"]) == list(train)
    assert sorted(te["item_id"]) == list(test)
    assert set(tr["split"]) == {"train"} and set(te["split"]) == {"test"}
    assert set(tr["item_id"]).isdisjoint(set(te["item_id"]))


def test_C_no_row_level_split_helper_exists():
    src = _code_only("feature_importance.py")
    tree = ast.parse(src)
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "train_test_split" not in called
    assert "train_test_split" not in src


# ==============================================  D  permutation semantics ===

@requires_canonical
def test_D_item_factor_permuted_across_items_not_within(frame):
    """Every route row of an item keeps one shared permuted value."""
    rng = np.random.RandomState(0)
    for factor in fi.ITEM_FACTORS:
        out = fi._permute_raw(frame, factor, rng)
        assert bool((out.groupby("item_id")[factor].nunique() == 1).all())
        # the multiset of item-level values is preserved
        before = sorted(frame.drop_duplicates("item_id")[factor].tolist())
        after = sorted(out.drop_duplicates("item_id")[factor].tolist())
        assert before == after
        assert list(out["item_id"]) == list(frame["item_id"])


@requires_canonical
def test_D_route_permuted_within_item_preserving_one_of_each(frame):
    rng = np.random.RandomState(0)
    out = fi._permute_raw(frame, "route", rng)
    g = out.groupby("item_id")["route"]
    assert bool((g.nunique() == 3).all())
    assert sorted(out["route"].value_counts().to_dict().values()) == [1062] * 3
    assert list(out["item_id"]) == list(frame["item_id"])


@requires_canonical
def test_D_permutation_actually_changes_something(frame):
    rng = np.random.RandomState(1)
    for factor in fi.FACTORS:
        out = fi._permute_raw(frame, factor, rng)
        assert not out[factor].equals(frame[factor]), factor


@requires_canonical
def test_D_derived_columns_are_rebuilt_from_the_permuted_raw_factor(frame):
    """Permuting `length` must move length_z AND every interaction using it."""
    design = fi.Design(fi.FACTORS, fi.INTERACTION_BLOCKS).fit(frame)
    base = design.transform(frame)
    rng = np.random.RandomState(2)
    perm = design.transform(fi._permute_raw(frame, "length", rng))
    cols = design.columns_
    li = cols.index("length_z")
    assert not np.allclose(base[:, li], perm[:, li])
    inter = [i for i, c in enumerate(cols) if ":" in c and "length_z" in c]
    assert inter, "interaction terms using length must exist"
    for i in inter:
        assert not np.allclose(base[:, i], perm[:, i]), cols[i]
    # and the interaction column is still exactly the product of its parts
    for i in inter:
        a, b = cols[i].split(":")
        assert np.allclose(perm[:, i],
                           perm[:, cols.index(a)] * perm[:, cols.index(b)])


@requires_canonical
def test_D_untouched_factors_keep_their_columns(frame):
    design = fi.Design(fi.FACTORS).fit(frame)
    base = design.transform(frame)
    rng = np.random.RandomState(3)
    perm = design.transform(fi._permute_raw(frame, "morphology", rng))
    li = design.columns_.index("length_z")
    assert np.allclose(base[:, li], perm[:, li])


def test_D_no_independent_dummy_or_interaction_permutation():
    src = _code_only("feature_importance.py")
    fn = src[src.index("def _permute_raw"):src.index("def permutation_importance")]
    for banned in ("length_z", "_x_", "columns_", "transform(", "dummy",
                   "coef_"):
        assert banned not in fn, f"_permute_raw touches {banned}"
    assert "raise ValueError" in fn, "unknown factors must be rejected"


def test_D_permutation_rejects_unknown_factors():
    df = pd.DataFrame({"item_id": ["a", "a", "a"], "route": ["full", "wm", "ltm"],
                       "length": [3, 3, 3]})
    with pytest.raises(ValueError):
        fi._permute_raw(df, "not_a_factor", np.random.RandomState(0))


# ==================================================  E  model choices =======

def test_E_frozen_hyperparameters():
    assert fi.RIDGE_ALPHA == 1.0
    assert fi.SPLIT_RANDOM_STATE == 42
    assert fi.SPLIT_TEST_SIZE == 0.2
    assert fi.PERM_REPEATS == 100
    assert fi.PERM_RANDOM_STATE == 42
    assert fi.REFERENCE_LEVELS == {"route": "wm", "lexicality": "pseudo",
                                   "morphology": "complex"}


def test_E_no_hyperparameter_search_anywhere():
    for m in FI_MODULES:
        code = _code_only(m)
        # API names only: the word "tuned" appears in captions that state the
        # alpha was NOT tuned, which is the opposite of a violation.
        for banned in ("GridSearchCV", "RandomizedSearchCV", "RidgeCV",
                       "cross_val_score", "GridSearch", "alphas=",
                       "param_grid", "best_params_"):
            assert banned not in code, f"{m} references {banned}"


@requires_outputs
def test_E_published_tables_record_the_frozen_alpha():
    for name, root in (("clean_main_model_fit.tsv", JT),
                       ("interaction_model_fit.tsv", IT),
                       ("route_specific_model_fit.tsv", RT)):
        df = _read(os.path.join(root, name))
        assert (df["ridge_alpha"] == 1.0).all(), name


@requires_outputs
def test_E_repeat_tables_have_100_repeats_per_seed_and_factor():
    reps = _read(os.path.join(JT, "clean_main_factor_importance_repeats.tsv"))
    counts = reps.groupby(["seed", "factor"]).size()
    assert bool((counts == fi.PERM_REPEATS).all()), counts.to_dict()
    summ = _read(os.path.join(JT, "clean_main_factor_importance.tsv"))
    assert (summ["n_repeats"] == fi.PERM_REPEATS).all()


@requires_canonical
def test_E_permutation_is_deterministic(canon, split):
    train_ids, test_ids = split
    df = fi.analysis_frame(canon, 19)
    train = df[df["item_id"].isin(set(train_ids))].reset_index(drop=True)
    test = df[df["item_id"].isin(set(test_ids))].reset_index(drop=True)
    m = fi.fit_model(train, test, fi.FACTORS)
    a = fi.permutation_importance(m, test, fi.FACTORS)
    b = fi.permutation_importance(m, test, fi.FACTORS)
    pd.testing.assert_frame_equal(a, b)


# ====================================================  F  sign policy =======

@requires_outputs
def test_F_grouped_importance_is_unsigned_and_separate_from_coefficients():
    imp = _read(os.path.join(JT, "clean_main_factor_importance.tsv"))
    coef = _read(os.path.join(JT, "clean_main_model_coefficients.tsv"))
    assert "coefficient" not in imp.columns
    assert not any("sign" in c for c in imp.columns)
    assert "coefficient" in coef.columns and "factor" not in coef.columns
    assert bool(coef["signed"].all())


@requires_outputs
def test_F_route_is_never_collapsed_into_one_signed_number():
    coef = _read(os.path.join(JT, "clean_main_model_coefficients.tsv"))
    terms = set(coef["term"])
    assert {"route_full", "route_ltm"} <= terms
    assert "route" not in terms, "route must stay two contrasts vs the reference"
    for seed, g in coef.groupby("seed"):
        assert len(g[g["term"].str.startswith("route_")]) == 2, seed


@requires_outputs
def test_F_reference_levels_are_recorded_with_the_coefficients():
    coef = _read(os.path.join(JT, "clean_main_model_coefficients.tsv"))
    assert bool(coef["reference_levels"].str.contains("route=wm").all())
    assert bool(coef["reference_levels"].str.contains("lexicality=pseudo").all())
    assert bool(coef["reference_levels"].str.contains("morphology=complex").all())


# ================================================  G  outcome statuses ======

def test_G_all_zero_outcome_is_not_fitted():
    df = pd.DataFrame({"item_id": [f"i{i}" for i in range(9)],
                       "route": ["full", "wm", "ltm"] * 3,
                       "lexicality": ["real"] * 9, "length": [3, 4, 5] * 3,
                       "morphology": ["simple"] * 9,
                       "raw_edit_distance": [0.0] * 9, "word_error": [0] * 9})
    m = fi.fit_model(df, df, fi.FACTORS)
    assert m["model_status"] == fi.STATUS_ALL_ZERO
    assert m["estimator"] is None and m["coefficients"] == {}
    assert fi.permutation_importance(m, df, fi.FACTORS).empty


@requires_outputs
def test_G_negative_test_r2_is_retained_and_flagged():
    rows = []
    for name, root in (("clean_main_model_fit.tsv", JT),
                       ("route_specific_model_fit.tsv", RT)):
        rows.append(_read(os.path.join(root, name)))
    df = pd.concat(rows, ignore_index=True)
    neg = df[df["test_r2"] < 0]
    assert bool(neg["negative_test_r2"].all())
    assert df["negative_test_r2"].equals(
        (df["test_r2"] < 0).fillna(False))       # never silently dropped
    assert set(df["model_status"]) <= {
        fi.STATUS_OK, fi.STATUS_ALL_ZERO, fi.STATUS_NEAR_ZERO,
        fi.STATUS_NEG_R2, fi.STATUS_INSUFFICIENT, fi.STATUS_NON_ESTIMABLE,
        fi.STATUS_NUMERICAL}


@requires_outputs
def test_G_mae_sensitivity_is_always_available():
    for name, root in (("clean_main_model_fit.tsv", JT),
                       ("route_specific_model_fit.tsv", RT)):
        df = _read(os.path.join(root, name))
        assert df["test_mae"].notna().all(), name
    imp = _read(os.path.join(RT, "route_specific_factor_importance.tsv"))
    assert "mae_increase_mean" in imp.columns
    # where R2 is non-estimable the MAE sensitivity still exists
    assert imp[imp["r2_drop_mean"].isna()]["mae_increase_mean"].notna().any()


@requires_outputs
def test_G_outcome_and_model_status_are_both_recorded():
    df = _read(os.path.join(RT, "route_specific_model_fit.tsv"))
    assert {"outcome_status", "model_status"} <= set(df.columns)
    near = df[df["outcome_status"] == fi.STATUS_NEAR_ZERO]
    assert len(near) and (near["model_status"] == fi.STATUS_NON_ESTIMABLE).all()


# ==============================================  H  visual conventions ======

FIGURES = [(JF, "clean_adapted_factor_importance"),
           (RF, "route_specific_factor_importance"),
           (IF, "interaction_block_utility")]


@requires_outputs
@pytest.mark.parametrize("out_dir,stem", FIGURES)
def test_H_figure_formats_and_caption(out_dir, stem):
    decision = _read(os.path.join(IT, "interaction_figure_decision.tsv"))
    if stem == "interaction_block_utility" and not bool(
            decision["figure_created"].iloc[0]):
        pytest.skip(fi.NO_FIGURE)
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        assert os.path.exists(p) and os.path.getsize(p) > 0, p
    assert os.path.getsize(os.path.join(out_dir, f"{stem}_caption.md")) > 0


def test_H_no_red_or_blue_in_the_fi_palette():
    for name, colour in pfi.FACTOR_COLOR.items():
        assert colour.startswith("#")
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        assert abs(r - g) < 40 and abs(g - b) < 40, f"{name} is not neutral"
    src = _code_only("plot_feature_importance.py")
    assert "LEXICALITY_COLOR" not in src
    assert '"red"' not in src and '"blue"' not in src


@requires_outputs
def test_H_captions_state_the_frozen_method_and_the_faithful_distinction():
    for out_dir, stem in FIGURES[:2]:
        cap = _norm(_text(os.path.join(out_dir, f"{stem}_caption.md"))).lower()
        assert "grouped by item" in cap
        assert "alpha = 1.0" in cap
        assert "100 repeats" in cap
        assert "lexicality and training exposure are perfectly confounded" in cap
        assert "671" in cap and "391" in cap
        assert "separate analysis" in cap and "faithful" in cap
        assert "red and blue" in cap


@requires_outputs
def test_H_all_four_seeds_are_available_to_every_figure():
    for path in (os.path.join(JT, "clean_main_factor_importance.tsv"),
                 os.path.join(RT, "route_specific_factor_importance.tsv"),
                 os.path.join(IT, "interaction_block_drop_utility.tsv")):
        assert sorted(_read(path)["seed"].unique()) == common.SEEDS, path


@requires_outputs
def test_H_ceiling_limited_routes_are_labelled_not_zeroed():
    imp = _read(os.path.join(RT, "route_specific_factor_importance.tsv"))
    fits = _read(os.path.join(RT, "route_specific_model_fit.tsv"))
    bad = fits[fits["model_status"] == fi.STATUS_NON_ESTIMABLE]
    assert len(bad), "this cohort has non-estimable route cells"
    for _, r in bad.iterrows():
        cell = imp[(imp["route"] == r["route"]) & (imp["seed"] == r["seed"])]
        assert cell["r2_drop_mean"].isna().all(), \
            "non-estimable cells must be NaN, never an artificial zero"
    cap = _norm(_text(os.path.join(
        RF, "route_specific_factor_importance_caption.md"))).lower()
    assert "never given an artificial zero importance" in cap


@requires_outputs
def test_H_within_and_between_seed_variability_are_distinguished():
    imp = _read(os.path.join(JT, "clean_main_factor_importance.tsv"))
    assert {"r2_drop_std", "r2_drop_min", "r2_drop_max"} <= set(imp.columns)
    summ = _read(os.path.join(JT, "clean_main_seed_summary.tsv"))
    assert {"range", "seed_values"} <= set(summ.columns)
    assert not bool(summ["is_hierarchical_bootstrap"].any())
    assert bool((summ["seed_interval_label"]
                 == fi.SEED_INTERVAL_LABEL).all())
    cap = _norm(_text(os.path.join(
        JF, "clean_adapted_factor_importance_caption.md"))).lower()
    assert "within-seed permutation standard deviation" in cap
    assert "between-seed" in cap


def test_H_seed_interval_is_not_called_a_hierarchical_bootstrap():
    out = fi.seed_interval([0.1, 0.2, 0.3, 0.4])
    assert out["is_hierarchical_bootstrap"] is False
    assert out["seed_interval_label"] == "seed-resampling interval over four checkpoints"
    assert out["n_resamples"] == 10000 and out["random_seed"] == 20260730
    for m in FI_MODULES:
        src = open(os.path.join(PKG_DIR, m)).read().lower()
        idx = src.find("hierarchical bootstrap")
        while idx >= 0:
            ctx = src[max(0, idx - 120):idx]
            assert any(k in ctx for k in ("not ", "never", "deliberately")), \
                f"{m} calls something a hierarchical bootstrap"
            idx = src.find("hierarchical bootstrap", idx + 1)


# ============================================  I  prior-sprint regression ===

LIVING = {"reports/behavioral_wfe_fulllexicon_93a577f/README.md",
          "reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv"}


def _verify(manifest, allow=frozenset()):
    bad = []
    with open(manifest) as f:
        for line in f:
            h, _, rel = line.strip().partition("  ")
            if not rel or rel in allow:
                continue
            p = os.path.join(ROOT, rel)
            if not os.path.exists(p) or _sha(p) != h:
                bad.append(rel)
    return bad


@pytest.mark.parametrize("rel,allow", [
    ("outputs/behavioral_wfe_fulllexicon_93a577f/full_wfe_evaluation/_control/"
     "production_scientific_outputs_FINAL.sha256", frozenset()),
    ("reports/behavioral_wfe_fulllexicon_93a577f/validation/"
     "sprint1_outputs.sha256", LIVING),
    ("reports/behavioral_wfe_fulllexicon_93a577f/morphology/validation/"
     "morphology_outputs.sha256", LIVING),
    ("reports/behavioral_wfe_fulllexicon_93a577f/frequency/validation/"
     "frequency_outputs.sha256", LIVING),
    ("reports/behavioral_wfe_fulllexicon_93a577f/error_taxonomy/validation/"
     "error_taxonomy_outputs.sha256", frozenset()),
])
def test_I_prior_manifests_unchanged(rel, allow):
    m = os.path.join(ROOT, rel)
    if not os.path.exists(m):
        pytest.skip("manifest absent")
    assert not _verify(m, allow)


@requires_canonical
def test_I_canonical_table_unchanged():
    prov = json.load(open(os.path.join(
        REPORT, "behavioral_analysis_provenance.json")))
    assert _sha(common.CANONICAL_TABLE) == prov["canonical_table_sha256"]


def test_I_faithful_a11_outputs_unchanged():
    pre = os.path.join(CTL, "feature_importance_preflight.json")
    if not os.path.exists(pre):
        pytest.skip("preflight absent")
    recorded = json.load(open(pre))["faithful_a11_outputs_sha256"]
    assert recorded, "A11 hashes must have been recorded"
    for rel, h in recorded.items():
        assert _sha(os.path.join(ROOT, rel)) == h, rel


# =====================================================  J  no inference =====

@pytest.mark.parametrize("module", FI_MODULES)
def test_J_no_torch_or_eval_imports(module):
    tree = ast.parse(open(os.path.join(PKG_DIR, module)).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("torch", "models", "evaluate", "train"):
        assert banned not in imported


@pytest.mark.parametrize("module", FI_MODULES)
def test_J_no_checkpoint_load_or_absolute_path(module):
    src = _code_only(module)
    for banned in ("import torch", "external_eval", "load_model_and_vocab",
                   "torch.load", "state_dict", "checkpoint_path",
                   ".pt\"", ".pt'", "/Users/", "/home/"):
        assert banned not in src, banned


@requires_outputs
def test_J_manifest_declares_no_inference():
    p = os.path.join(CTL, "feature_importance_output_manifest.json")
    if not os.path.exists(p):
        pytest.skip("manifest absent")
    m = json.load(open(p))
    assert m["model_inference_performed"] is False
    assert m["faithful_a11_touched"] is False
    assert m["split_grouped_by"] == "item_id"
    assert m["split_reused_across_seeds"] is True


@requires_outputs
def test_J_no_causal_or_architectural_claim():
    docs = []
    for base, _, names in os.walk(Q):
        docs += [os.path.join(base, n) for n in names if n.endswith(".md")]
    txt = _norm(_text(*sorted(docs))).lower()
    negations = ("no ", "not ", "never", "cannot", "neither", "nothing")
    for phrase in ("causes", "caused by", "should be changed",
                   "we recommend changing", "explains the mechanism"):
        i = txt.find(phrase)
        while i >= 0:
            ctx = txt[max(0, i - 160):i]
            assert any(n in ctx for n in negations), f"unqualified: {phrase}"
            i = txt.find(phrase, i + len(phrase))
