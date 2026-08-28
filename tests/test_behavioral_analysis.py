"""Tests for the tracked behavioral-analysis package.

No checkpoint is loaded and no model inference occurs anywhere in this file or
in the package it exercises (enforced by group F).

Expected values are read from the validated reference tables rather than
duplicated as constants, so the tests cannot drift from the frozen results.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis import bootstrap, common, compute, plotting
from scripts.behavioral_analysis.io import clean_subset, load_canonical

PKG_DIR = os.path.join(ROOT, "scripts", "behavioral_analysis")
FIG_DIR = os.path.join(ROOT, "reports", "behavioral_wfe_fulllexicon_93a577f",
                       "figures")
# Validated analysis reference tables.  Prefer the tracked copies under
# reports/.../repro_inputs/ so the cross-checks run from a fresh clone; fall
# back to the original (gitignored) outputs/ tree when it is present, so an
# existing working copy behaves exactly as before.  The files are
# byte-identical in both locations - see REPRO_INPUTS.tsv.
_TRACKED_REFERENCE_DIR = os.path.join(
    ROOT, "reports", "behavioral_wfe_fulllexicon_93a577f", "repro_inputs")
_OUTPUTS_REFERENCE_DIR = os.path.join(
    ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
    "behavioral_analysis")
REFERENCE_DIR = (_TRACKED_REFERENCE_DIR
                 if os.path.isdir(_TRACKED_REFERENCE_DIR)
                 else _OUTPUTS_REFERENCE_DIR)

requires_canonical = pytest.mark.skipif(
    not os.path.exists(common.CANONICAL_TABLE),
    reason="validated canonical table not present")
requires_figures = pytest.mark.skipif(
    not os.path.exists(os.path.join(FIG_DIR,
                                    "clean_length_slopes_by_seed.tsv")),
    reason="published plotting tables not generated")


@pytest.fixture(scope="module")
def canon():
    return load_canonical(common.CANONICAL_TABLE)


@pytest.fixture(scope="module")
def clean(canon):
    return clean_subset(canon)


# ======================================================  A  canonical input ==

@requires_canonical
def test_A_canonical_shape_and_keys(canon):
    assert len(canon) == common.EXPECTED_CANONICAL_ROWS == 14400
    assert sorted(canon["seed"].unique()) == common.SEEDS == [19, 20, 21, 22]
    assert sorted(canon["route"].unique()) == sorted(common.ROUTES)
    assert not canon.duplicated(["seed", "item_id", "route"]).any()
    assert canon["item_id"].nunique() == common.EXPECTED_FAITHFUL_ITEMS


@requires_canonical
def test_A_seed_21_present_everywhere(canon):
    assert 21 in set(canon["seed"])
    for route in common.ROUTES:
        sub = canon[(canon["seed"] == 21) & (canon["route"] == route)]
        assert sub["item_id"].nunique() == 1200


@requires_canonical
def test_A_clean_set_composition(clean):
    one = clean[(clean["seed"] == 19) & (clean["route"] == "full")]
    counts = one["lichtheim_exposure_status"].value_counts().to_dict()
    assert len(one) == 1062
    assert counts["TRAINED_REAL_EXACT"] == 671
    assert counts["NOVEL_PSEUDOWORD"] == 391
    assert set(counts) == {"TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD"}


@requires_canonical
def test_A_operations_reconstruct_edit_distance(canon):
    total = canon["insertions"] + canon["deletions"] + canon["substitutions"]
    assert (total == canon["raw_edit_distance"]).all()


# ==================================================  B  plotting tables =====

@requires_figures
@pytest.mark.parametrize("name,cols", [
    ("yair_clean_length_by_route.tsv",
     {"route", "source_lexicality", "phoneme_length", "seed", "n_items",
      "mean_raw_edit_distance", "mean_across_seeds", "ci_low", "ci_high"}),
    ("clean_length_slopes_by_seed.tsv",
     {"seed", "source_lexicality", "route", "n_items", "intercept",
      "length_slope", "model_status"}),
    ("clean_route_length_contrasts.tsv",
     {"seed", "source_lexicality", "wm_length_slope", "ltm_length_slope",
      "full_length_slope", "ltm_minus_wm"}),
    ("yair_clean_serial_position_interpolated.tsv",
     {"route", "source_lexicality", "relative_position",
      "interpolated_error_rate"}),
    ("gate_by_clean_lexicality.tsv",
     {"seed", "source_lexicality", "n_items", "mean_gate",
      "mean_lexical_confidence"}),
    ("gate_by_exposure_status.tsv",
     {"seed", "exposure_status", "n_items", "mean_gate",
      "mean_lexical_confidence"}),
])
def test_B_expected_columns(name, cols):
    df = pd.read_csv(os.path.join(FIG_DIR, name), sep="\t")
    assert cols <= set(df.columns), sorted(cols - set(df.columns))


@requires_figures
@pytest.mark.parametrize("name", [
    "yair_clean_length_by_route.tsv", "clean_length_slopes_by_seed.tsv",
    "clean_route_length_contrasts.tsv", "gate_by_clean_lexicality.tsv",
    "gate_by_exposure_status.tsv"])
def test_B_all_four_seeds(name):
    df = pd.read_csv(os.path.join(FIG_DIR, name), sep="\t")
    assert sorted(df["seed"].unique()) == [19, 20, 21, 22]


@requires_figures
def test_B_lengths_exclude_six():
    df = pd.read_csv(os.path.join(FIG_DIR, "yair_clean_length_by_route.tsv"),
                     sep="\t")
    assert sorted(df["phoneme_length"].unique()) == common.LENGTHS
    assert 6 not in set(df["phoneme_length"])


@requires_figures
def test_B_routes_and_lexicality_levels():
    for name in ("yair_clean_length_by_route.tsv",
                 "yair_clean_serial_position_interpolated.tsv",
                 "clean_length_slopes_by_seed.tsv"):
        df = pd.read_csv(os.path.join(FIG_DIR, name), sep="\t")
        assert sorted(df["route"].unique()) == sorted(common.ROUTES)
        assert sorted(df["source_lexicality"].unique()) == ["pseudo", "real"]


@requires_figures
def test_B_no_hidden_item_filtering():
    df = pd.read_csv(os.path.join(FIG_DIR, "yair_clean_length_by_route.tsv"),
                     sep="\t")
    per_cell = df[(df["route"] == "full") & (df["seed"] == 19)]
    assert int(per_cell[per_cell["source_lexicality"] == "real"]["n_items"].sum()) == 671
    assert int(per_cell[per_cell["source_lexicality"] == "pseudo"]["n_items"].sum()) == 391


@requires_figures
def test_B_exposure_table_covers_all_six_categories():
    df = pd.read_csv(os.path.join(FIG_DIR, "gate_by_exposure_status.tsv"),
                     sep="\t")
    assert set(df["exposure_status"]) == set(common.EXPOSURE_ORDER)


# ==================================================  C  bootstrap ===========

def test_C_frozen_bootstrap_configuration():
    assert common.BOOTSTRAP_REPLICATES == 10000
    assert common.BOOTSTRAP_SEED == 20260730
    assert common.BOOTSTRAP_CI_LEVEL == 95


def test_C_bootstrap_is_deterministic():
    rng = np.random.default_rng(0)
    x = {"s": rng.normal(size=40)}
    y = {(sd, rt, "s"): rng.normal(size=40)
         for sd in (1, 2) for rt in ("wm", "ltm")}
    kw = dict(b=200, chunk=50)
    a = bootstrap.hierarchical_bootstrap(x, y, [1, 2],
                                         lambda p: p[("ltm", "s")], **kw)
    b = bootstrap.hierarchical_bootstrap(x, y, [1, 2],
                                         lambda p: p[("ltm", "s")], **kw)
    assert a == b
    c = bootstrap.hierarchical_bootstrap(x, y, [1, 2],
                                         lambda p: p[("ltm", "s")],
                                         seed=common.BOOTSTRAP_SEED + 1, **kw)
    assert c["bootstrap_mean"] != a["bootstrap_mean"]


@requires_figures
def test_C_bootstrap_records_b_and_seed():
    df = pd.read_csv(os.path.join(FIG_DIR, "clean_bootstrap_results.tsv"),
                     sep="\t")
    assert set(df["n_replicates"]) == {common.BOOTSTRAP_REPLICATES}
    assert set(df["random_seed"]) == {common.BOOTSTRAP_SEED}
    assert set(df["ci_definition"]) == {"95% percentile interval"}


def test_C_ols_slope_matches_closed_form():
    x = np.array([3.0, 4, 5, 7, 8, 9])
    y = 2.5 * x + 1.25
    b0, b1 = bootstrap.ols_slope(x, y)
    assert b1 == pytest.approx(2.5)
    assert b0 == pytest.approx(1.25)
    assert np.isnan(bootstrap.ols_slope(np.ones(5), np.arange(5.0))[1])


# ==================================  D  scientific conventions ==============

def test_D_red_and_blue_are_reserved_for_lexicality():
    assert common.LEXICALITY_COLOR == {"real": "red", "pseudo": "blue"}
    # exposure categories must not reuse the lexicality colours
    for colour in (common.EXPOSURE_COLOR, common.EXPOSURE_ACCENT):
        assert colour.lower() not in ("red", "blue", "#ff0000", "#0000ff")
    src = open(os.path.join(PKG_DIR, "plotting.py")).read()
    exposure_fn = src.split("def plot_gate_exposure")[1]
    assert '"red"' not in exposure_fn and '"blue"' not in exposure_fn
    assert "LEXICALITY_COLOR" not in exposure_fn


def test_D_faithful_serial_position_uses_zip_mismatch():
    # a leading deletion smears mismatches across the whole word: the
    # signature of the positional zip rule, not of an alignment
    pos = compute.zip_mismatch_positions("S EH K", "EH K")
    assert pos == [1, 2, 3]
    # a single substitution marks exactly one position
    assert compute.zip_mismatch_positions("K AE T", "K EH T") == [2]


def test_D_no_editops_in_faithful_serial_position():
    src = open(os.path.join(PKG_DIR, "compute.py")).read()
    serial = src.split("def serial_position_tables")[1]
    for banned in ("editops", "Levenshtein"):
        assert banned not in serial, f"{banned} must not drive Figure 2C"


def test_D_gate_is_full_route_and_word_level(canon=None):
    src = open(os.path.join(PKG_DIR, "compute.py")).read()
    gate_fn = src.split("def gate_tables")[1]
    assert 'route"] == "full"' in gate_fn, "gate must be read from the full route"
    assert "FULL-route" in gate_fn or "full-route" in gate_fn.lower()


def test_D_terminology_is_not_misleading():
    for fname in ("common.py", "plotting.py", "compute.py"):
        src = open(os.path.join(PKG_DIR, fname)).read().lower()
        assert "phonological similarity" not in src.replace(
            "not phonological similarity", "")
        assert "word probability" not in src
        assert "probability that the stimulus is a word" not in src.replace(
            "not a probability that the stimulus is a word", "")


def test_D_seed_policy_includes_21():
    assert common.SEEDS == [19, 20, 21, 22]
    assert 21 in common.SEEDS
    assert 21 not in common.CEILING_SEEDS      # sensitivity set only


# ==============================  E  numerical non-regression ================

@requires_figures
def test_E_slopes_match_validated_reference():
    new = pd.read_csv(os.path.join(FIG_DIR, "clean_length_slopes_by_seed.tsv"),
                      sep="\t")
    ref = pd.read_csv(os.path.join(REFERENCE_DIR, "tables",
                                   "clean_length_slopes_by_seed.tsv"), sep="\t")
    ref = ref[ref["source_lexicality"].isin(["real", "pseudo"])]
    m = new.merge(ref, on=["seed", "source_lexicality", "route"],
                  suffixes=("_new", "_ref"))
    assert len(m) == 24
    assert (m["length_slope_new"] - m["length_slope_ref"]).abs().max() == 0.0


@requires_figures
def test_E_contrasts_match_validated_reference():
    new = pd.read_csv(os.path.join(FIG_DIR, "clean_route_length_contrasts.tsv"),
                      sep="\t")
    ref = pd.read_csv(os.path.join(REFERENCE_DIR, "tables",
                                   "clean_route_length_contrasts.tsv"), sep="\t")
    ref = ref[ref["source_lexicality"].isin(["real", "pseudo"])]
    m = new.merge(ref, on=["seed", "source_lexicality"],
                  suffixes=("_new", "_ref"))
    assert len(m) == 8
    assert (m["ltm_minus_wm_new"] - m["ltm_minus_wm_ref"]).abs().max() == 0.0


@requires_figures
def test_E_clean_item_counts_match_reference():
    new = pd.read_csv(os.path.join(FIG_DIR, "yair_clean_length_by_route.tsv"),
                      sep="\t")
    ref = pd.read_csv(os.path.join(REFERENCE_DIR, "tables",
                                   "clean_item_counts_by_length.tsv"), sep="\t")
    got = (new[(new["route"] == "full") & (new["seed"] == 19)]
           .set_index(["source_lexicality", "phoneme_length"])["n_items"])
    for _, row in ref.iterrows():
        assert int(got.loc[(row["source_lexicality"],
                            row["target_length"])]) == int(row["n_items"])


@requires_figures
def test_E_gate_clean_matches_reference():
    new = pd.read_csv(os.path.join(FIG_DIR, "gate_by_clean_lexicality.tsv"),
                      sep="\t")
    ref = pd.read_csv(os.path.join(REFERENCE_DIR, "statistics",
                                   "gate_results.tsv"), sep="\t")
    ref = ref[ref["grouping"] == "lexicality"].copy()
    ref["source_lexicality"] = ref["group"].map({"Real words": "real",
                                                 "Pseudowords": "pseudo"})
    m = new.merge(ref, on=["seed", "source_lexicality"],
                  suffixes=("_new", "_ref"))
    assert len(m) == 8
    assert (m["mean_gate_new"] - m["mean_gate_ref"]).abs().max() == 0.0
    assert (m["mean_lexical_confidence_new"]
            - m["mean_lexical_confidence_ref"]).abs().max() == 0.0


@requires_figures
def test_E_serial_position_matches_reference():
    new = pd.read_csv(os.path.join(
        FIG_DIR, "yair_clean_serial_position_interpolated.tsv"), sep="\t")
    ref = pd.read_csv(os.path.join(
        REFERENCE_DIR, "tables",
        "yair_clean_serial_position_interpolated.tsv"), sep="\t")
    m = new.merge(ref, on=["route", "source_lexicality", "relative_position"],
                  suffixes=("_new", "_ref"))
    assert len(m) == 600
    assert (m["interpolated_error_rate_new"]
            - m["interpolated_error_rate_ref"]).abs().max() == 0.0


# ==========================================  F  no inference ================

ANALYSIS_MODULES = ["common.py", "io.py", "bootstrap.py", "compute.py",
                    "plotting.py", "make_figures.py", "validate_outputs.py",
                    "build_canonical_table.py", "close_production_manifest.py"]


@pytest.mark.parametrize("module", ANALYSIS_MODULES)
def test_F_no_model_or_checkpoint_imports(module):
    tree = ast.parse(open(os.path.join(PKG_DIR, module)).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("torch", "models", "evaluate", "train"):
        assert banned not in imported, f"{module} imports {banned}"


@pytest.mark.parametrize("module", ANALYSIS_MODULES)
def test_F_no_checkpoint_loading_calls(module):
    tree = ast.parse(open(os.path.join(PKG_DIR, module)).read())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                called.add(fn.attr)
            elif isinstance(fn, ast.Name):
                called.add(fn.id)
    for banned in ("load_model_and_vocab", "run_wfe_eval", "evaluate_items",
                   "autoregressive_decode_batch"):
        assert banned not in called, f"{module} calls {banned}"


def test_F_package_does_not_reference_external_eval():
    for module in ANALYSIS_MODULES:
        src = open(os.path.join(PKG_DIR, module)).read()
        assert "external_eval" not in src


def test_F_no_absolute_user_paths_in_package():
    for module in ANALYSIS_MODULES + ["__init__.py"]:
        src = open(os.path.join(PKG_DIR, module)).read()
        assert "/Users/" not in src, f"{module} contains an absolute user path"
        assert "/home/" not in src
