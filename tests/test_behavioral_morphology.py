"""Tests for the Sprint-2 morphology analysis.

No checkpoint is loaded and no model inference occurs (group F).  Expected
values are read from the generated tables and the canonical table rather than
duplicated as constants.
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis import common, morphology as mo
from scripts.behavioral_analysis import plot_morphology as pm
from scripts.behavioral_analysis.io import load_canonical

PKG_DIR = os.path.join(ROOT, "scripts", "behavioral_analysis")
REPORT = os.path.join(ROOT, "reports", "behavioral_wfe_fulllexicon_93a577f")
M = os.path.join(REPORT, "morphology")
FA_TAB = os.path.join(M, "faithful_replication", "tables")
CL_TAB = os.path.join(M, "clean_adapted", "tables")
SHARED = os.path.join(M, "tables")

requires_canonical = pytest.mark.skipif(
    not os.path.exists(common.CANONICAL_TABLE),
    reason="canonical table not present")
requires_outputs = pytest.mark.skipif(
    not os.path.exists(os.path.join(CL_TAB,
                                    "clean_morphology_seed_contrasts.tsv")),
    reason="Sprint-2 outputs not generated")


@pytest.fixture(scope="module")
def canon():
    return load_canonical(common.CANONICAL_TABLE)


# ==============================================  A  datasets and cells ======

@requires_canonical
def test_A_faithful_is_1200_items(canon):
    d = canon[canon["in_FAITHFUL_WFE_ALL"] & (canon["route"] == "full")
              & (canon["seed"] == common.SEEDS[0])]
    assert d["item_id"].nunique() == 1200


@requires_canonical
def test_A_clean_is_1062_with_671_real_and_391_pseudo(canon):
    d = canon[canon["in_LICHTHEIM_CLEAN"] & (canon["route"] == "full")
              & (canon["seed"] == common.SEEDS[0])]
    assert len(d) == 1062
    vc = d["lichtheim_exposure_status"].value_counts().to_dict()
    assert vc["TRAINED_REAL_EXACT"] == 671
    assert vc["NOVEL_PSEUDOWORD"] == 391


@requires_outputs
def test_A_all_four_seeds_in_every_primary_table():
    for path in ("faithful_morphology_seed_contrasts.tsv",
                 "faithful_morphology_length_interactions.tsv"):
        df = pd.read_csv(os.path.join(FA_TAB, path), sep="\t")
        assert sorted(df["seed"].unique()) == [19, 20, 21, 22]
    for path in ("clean_morphology_seed_contrasts.tsv",
                 "clean_morphology_length_interactions.tsv",
                 "clean_morphology_route_contrasts.tsv"):
        df = pd.read_csv(os.path.join(CL_TAB, path), sep="\t")
        assert sorted(df["seed"].unique()) == [19, 20, 21, 22]


@requires_outputs
def test_A_clean_has_three_routes_faithful_has_full_only():
    cl = pd.read_csv(os.path.join(CL_TAB,
                                  "clean_morphology_seed_contrasts.tsv"),
                     sep="\t")
    assert sorted(cl["route"].unique()) == sorted(common.ROUTES)
    fa = pd.read_csv(os.path.join(FA_TAB,
                                  "faithful_morphology_seed_contrasts.tsv"),
                     sep="\t")
    assert sorted(fa["route"].unique()) == ["full"]
    plot = pd.read_csv(os.path.join(
        FA_TAB, "faithful_length_lexicality_morphology_plot.tsv"), sep="\t")
    assert sorted(plot["route"].unique()) == ["full"]


@requires_outputs
def test_A_morphology_labels_exactly_simple_and_complex():
    for path in (os.path.join(SHARED, "faithful_morphology_cell_counts.tsv"),
                 os.path.join(SHARED, "clean_morphology_cell_counts.tsv")):
        df = pd.read_csv(path, sep="\t")
        assert set(df["morphology"].unique()) == {"simple", "complex"}


@requires_outputs
def test_A_lengths_are_exactly_the_six_wfe_lengths():
    for path in (os.path.join(SHARED, "faithful_morphology_cell_counts.tsv"),
                 os.path.join(SHARED, "clean_morphology_cell_counts.tsv")):
        df = pd.read_csv(path, sep="\t")
        assert sorted(df["phoneme_length"].unique()) == [3, 4, 5, 7, 8, 9]
        assert 6 not in set(df["phoneme_length"])


@requires_outputs
def test_A_cell_counts_sum_to_dataset_sizes():
    fa = pd.read_csv(os.path.join(SHARED,
                                  "faithful_morphology_cell_counts.tsv"),
                     sep="\t")
    assert int(fa["n_items"].sum()) == 1200
    cl = pd.read_csv(os.path.join(SHARED, "clean_morphology_cell_counts.tsv"),
                     sep="\t")
    per_route = cl.groupby("route")["n_items"].sum()
    assert set(per_route.unique()) == {1062}


# ============================================  B  contrast definitions ======

@requires_outputs
def test_B_morphology_contrast_is_simple_minus_complex():
    df = pd.read_csv(os.path.join(CL_TAB,
                                  "clean_morphology_seed_contrasts.tsv"),
                     sep="\t")
    recomputed = (df["mean_simple_raw_edit_distance"]
                  - df["mean_complex_raw_edit_distance"])
    assert np.allclose(df["morphology_contrast_raw_edit_distance"],
                       recomputed, equal_nan=True)


@requires_outputs
def test_B_interaction_is_simple_slope_minus_complex_slope():
    df = pd.read_csv(os.path.join(
        CL_TAB, "clean_morphology_length_interactions.tsv"), sep="\t")
    recomputed = df["simple_length_slope"] - df["complex_length_slope"]
    assert np.allclose(df["morphology_length_interaction"], recomputed,
                       equal_nan=True)


def test_B_sign_conventions_on_synthetic_data():
    """positive contrast = simple worse; positive interaction = simple steeper."""
    rows = []
    for seed in common.SEEDS:
        for L in common.LENGTHS:
            for mor, base, slope in (("simple", 1.0, 0.5), ("complex", 0.0, 0.1)):
                rows.append({"seed": seed, "route": "full", "item_id": f"{mor}{L}",
                             "source_lexicality": "real", "morphology": mor,
                             "target_length": L,
                             "raw_edit_distance": base + slope * L,
                             "word_error": 0,
                             "in_FAITHFUL_WFE_ALL": True,
                             "in_LICHTHEIM_CLEAN": True})
    df = pd.DataFrame(rows)
    con = mo.seed_contrasts(df, "LICHTHEIM_CLEAN")
    real = con[con["source_lexicality"] == "real"]
    assert (real["morphology_contrast_raw_edit_distance"] > 0).all()
    inter = mo.seed_length_interactions(df, "LICHTHEIM_CLEAN")
    ri = inter[inter["source_lexicality"] == "real"]
    assert (ri["morphology_length_interaction"] > 0).all()
    assert np.allclose(ri["simple_length_slope"], 0.5)
    assert np.allclose(ri["complex_length_slope"], 0.1)


@requires_outputs
def test_B_route_contrast_sign_conventions():
    con = pd.read_csv(os.path.join(CL_TAB,
                                   "clean_morphology_seed_contrasts.tsv"),
                      sep="\t")
    rc = pd.read_csv(os.path.join(CL_TAB,
                                  "clean_morphology_route_contrasts.tsv"),
                     sep="\t")
    assert set(rc["route_contrast"]) == {"ltm_minus_wm", "full_minus_wm",
                                         "ltm_minus_full"}
    row = rc[(rc["route_contrast"] == "ltm_minus_wm")].iloc[0]
    seed, lex = int(row["seed"]), row["source_lexicality"]
    def val(route):
        return float(con[(con["seed"] == seed) & (con["route"] == route)
                         & (con["source_lexicality"] == lex)]
                     ["morphology_contrast_raw_edit_distance"].iloc[0])
    assert row["morphology_contrast_difference"] == pytest.approx(
        val("ltm") - val("wm"))


def test_B_exact_zero_seed_set():
    assert common.CEILING_SEEDS == [19, 20, 22]
    assert 21 not in common.CEILING_SEEDS
    assert 21 in common.SEEDS


# ========================================================  C  bootstrap =====

def test_C_bootstrap_configuration_unchanged():
    assert common.BOOTSTRAP_REPLICATES == 10000
    assert common.BOOTSTRAP_SEED == 20260730
    assert common.BOOTSTRAP_CI_LEVEL == 95


@requires_outputs
def test_C_bootstrap_tables_record_b_and_seed():
    for path in (os.path.join(FA_TAB, "faithful_morphology_bootstrap.tsv"),
                 os.path.join(CL_TAB, "clean_morphology_bootstrap.tsv")):
        df = pd.read_csv(path, sep="\t")
        assert set(df["n_replicates"]) == {10000}
        assert set(df["random_seed"]) == {20260730}
        assert set(df["ci_definition"]) == {"95% percentile interval"}


def test_C_mean_difference_bootstrap_is_deterministic():
    rng = np.random.default_rng(3)
    x_by = {"simple": rng.normal(size=30), "complex": rng.normal(size=30)}
    y_by = {(s, "full", m): rng.normal(size=30)
            for s in common.SEEDS for m in ("simple", "complex")}
    a = mo._bootstrap_mean_difference(x_by, y_by, "full", b=400, chunk=100)
    b = mo._bootstrap_mean_difference(x_by, y_by, "full", b=400, chunk=100)
    assert a == b


@requires_outputs
def test_C_seed21_retained_in_primary_cohort():
    for path in (os.path.join(CL_TAB, "clean_morphology_seed_contrasts.tsv"),
                 os.path.join(FA_TAB,
                              "faithful_morphology_seed_contrasts.tsv")):
        df = pd.read_csv(path, sep="\t")
        assert 21 in set(df["seed"])
    sens = pd.read_csv(os.path.join(
        CL_TAB, "clean_morphology_exact_zero_sensitivity.tsv"), sep="\t")
    assert sens["seed21_included"].all()
    assert "exact_zero_seeds_mean" in sens.columns


# ==============================================  D  visual conventions ======

def test_D_lexicality_colours_reserved():
    assert common.LEXICALITY_COLOR == {"real": "red", "pseudo": "blue"}
    src = open(os.path.join(PKG_DIR, "plot_morphology.py")).read()
    # morphology must be encoded by line style only, never by colour
    assert "MORPH_STYLE" in src
    assert pm.MORPH_STYLE == {"complex": "-", "simple": "--"}
    for banned in ('morphology": "red"', 'morphology": "blue"'):
        assert banned not in src


def test_D_complex_is_solid_and_simple_is_dashed():
    assert pm.MORPH_STYLE["complex"] == "-"
    assert pm.MORPH_STYLE["simple"] == "--"


def test_D_no_other_variable_uses_red_or_blue():
    src = open(os.path.join(PKG_DIR, "plot_morphology.py")).read()
    # every literal red/blue must come from LEXICALITY_COLOR
    assert src.count('"red"') == 0 and src.count('"blue"') == 0
    assert "LEXICALITY_COLOR[lex]" in src


# =====================================================  E  non-regression ===

def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


# Two Sprint-1 files are *living documents* that every later sprint is meant to
# extend: the report README (narrative index) and the analysis matrix (status of
# each planned analysis).  Sprint 2 legitimately adds a morphology section to the
# first and flips A12/A13 to ALREADY_VALIDATED in the second.  Everything else
# Sprint 1 produced — every figure, plotting table, caption, manifest and
# provenance record — must stay byte-identical.
SPRINT1_LIVING_DOCUMENTS = {
    "reports/behavioral_wfe_fulllexicon_93a577f/README.md",
    "reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv",
}


@pytest.mark.skipif(not os.path.exists(os.path.join(
    REPORT, "validation", "sprint1_outputs.sha256")),
    reason="Sprint-1 manifest not present")
def test_E_sprint1_scientific_outputs_unchanged():
    """No Sprint-1 figure, plotting table or manifest may move in Sprint 2."""
    manifest = os.path.join(REPORT, "validation", "sprint1_outputs.sha256")
    bad, living = [], []
    for line in open(manifest):
        h, _, rel = line.strip().partition("  ")
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or _sha(p) != h:
            (living if rel in SPRINT1_LIVING_DOCUMENTS else bad).append(rel)
    assert not bad, f"Sprint-1 scientific outputs modified: {bad[:5]}"
    # the living documents are allowed to change, but nothing else may hide here
    assert set(living) <= SPRINT1_LIVING_DOCUMENTS


@pytest.mark.skipif(not os.path.exists(os.path.join(
    REPORT, "validation", "sprint1_outputs.sha256")),
    reason="Sprint-1 manifest not present")
def test_E_sprint1_figures_and_plotting_tables_byte_identical():
    """The stricter half of the guarantee, stated separately so it cannot rot."""
    manifest = os.path.join(REPORT, "validation", "sprint1_outputs.sha256")
    checked, bad = 0, []
    for line in open(manifest):
        h, _, rel = line.strip().partition("  ")
        if "/figures/" not in rel:
            continue
        checked += 1
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or _sha(p) != h:
            bad.append(rel)
    assert checked >= 20, f"expected the Sprint-1 figure set, saw {checked}"
    assert not bad, f"Sprint-1 figure artefacts modified: {bad[:5]}"


@pytest.mark.skipif(not os.path.exists(os.path.join(
    ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
    "full_wfe_evaluation", "_control",
    "production_scientific_outputs_FINAL.sha256")),
    reason="production manifest not present")
def test_E_production_outputs_unchanged():
    manifest = os.path.join(
        ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
        "full_wfe_evaluation", "_control",
        "production_scientific_outputs_FINAL.sha256")
    bad = []
    for line in open(manifest):
        h, _, rel = line.strip().partition("  ")
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or _sha(p) != h:
            bad.append(rel)
    assert not bad, f"production outputs modified: {bad[:5]}"


@requires_canonical
def test_E_canonical_table_matches_sprint1_provenance():
    import json
    prov = json.load(open(os.path.join(
        REPORT, "behavioral_analysis_provenance.json")))
    assert _sha(common.CANONICAL_TABLE) == prov["canonical_table_sha256"]


# ========================================================  F  no inference ==

MORPH_MODULES = ["morphology.py", "plot_morphology.py"]


@pytest.mark.parametrize("module", MORPH_MODULES)
def test_F_no_torch_or_eval_imports(module):
    tree = ast.parse(open(os.path.join(PKG_DIR, module)).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("torch", "models", "evaluate", "train"):
        assert banned not in imported


@pytest.mark.parametrize("module", MORPH_MODULES)
def test_F_no_checkpoint_loading(module):
    src = open(os.path.join(PKG_DIR, module)).read()
    for banned in ("external_eval", "load_model_and_vocab", ".pt\"", "'.pt'"):
        assert banned not in src
    assert "/Users/" not in src and "/home/" not in src


# =====================================================  G  small-cell flags =

def test_G_flag_thresholds_are_frozen():
    assert mo.VERY_SMALL_CELL_MAX == 10
    assert mo.SMALL_CELL_MAX == 20
    assert mo.cell_flag(9) == "VERY_SMALL_CELL"
    assert mo.cell_flag(10) == "SMALL_CELL"
    assert mo.cell_flag(19) == "SMALL_CELL"
    assert mo.cell_flag(20) == "OK"


@requires_outputs
def test_G_no_item_excluded_by_flags():
    cl = pd.read_csv(os.path.join(SHARED, "clean_morphology_cell_counts.tsv"),
                     sep="\t")
    full = cl[cl["route"] == "full"]
    # flagged cells are retained, not dropped: totals still reconcile
    assert int(full["n_items"].sum()) == 1062
    assert (full["cell_flag"] != "OK").sum() >= 1, "expected flagged cells kept"
    assert full[full["cell_flag"] != "OK"]["n_items"].min() > 0
