"""Tests for the Sprint-3 frequency analysis.

No checkpoint is loaded and no model inference occurs (group I).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis import common, frequency as fq
from scripts.behavioral_analysis import plot_frequency as pf
from scripts.behavioral_analysis.io import load_canonical

PKG_DIR = os.path.join(ROOT, "scripts", "behavioral_analysis")
REPORT = os.path.join(ROOT, "reports", "behavioral_wfe_fulllexicon_93a577f")
Q = os.path.join(REPORT, "frequency")
PRIM_T = os.path.join(Q, "primary", "tables")
CG_T = os.path.join(Q, "gate_confidence", "tables")
SENS = os.path.join(Q, "sensitivity")
SHARED = os.path.join(Q, "tables")
PRIMARY = "TRAINED_REAL_FREQUENCY_PRIMARY"
SENSITIVITY = "TRAINED_REAL_FREQUENCY_SENSITIVITY"

requires_canonical = pytest.mark.skipif(
    not os.path.exists(common.CANONICAL_TABLE), reason="canonical table absent")
requires_outputs = pytest.mark.skipif(
    not os.path.exists(os.path.join(PRIM_T,
                                    "trained_real_frequency_slopes.tsv")),
    reason="Sprint-3 outputs not generated")


@pytest.fixture(scope="module")
def canon():
    return load_canonical(common.CANONICAL_TABLE)


# =============================================================  A  sets =====

@requires_canonical
def test_A_primary_is_671_trained_real_exact(canon):
    d = fq.regime_subset(canon, PRIMARY)
    one = d[(d["seed"] == 19) & (d["route"] == "full")]
    assert one["item_id"].nunique() == 671
    assert set(one["lichtheim_exposure_status"]) == {"TRAINED_REAL_EXACT"}


@requires_canonical
def test_A_sensitivity_is_678_with_seven_variants(canon):
    d = fq.regime_subset(canon, SENSITIVITY)
    one = d[(d["seed"] == 19) & (d["route"] == "full")]
    assert one["item_id"].nunique() == 678
    vc = one["lichtheim_exposure_status"].value_counts().to_dict()
    assert vc["TRAINED_REAL_EXACT"] == 671
    assert vc["TRAINED_REAL_PRON_VARIANT"] == 7


@requires_canonical
def test_A_untrained_real_is_122(canon):
    d = fq.regime_subset(canon, "UNTRAINED_REAL")
    one = d[(d["seed"] == 19) & (d["route"] == "full")]
    assert one["item_id"].nunique() == 122


@requires_canonical
@pytest.mark.parametrize("regime", [PRIMARY, SENSITIVITY, "UNTRAINED_REAL",
                                    "FAITHFUL_ALL_REAL"])
def test_A_no_pseudowords_in_any_frequency_regime(canon, regime):
    d = fq.regime_subset(canon, regime)
    assert set(d["source_lexicality"]) == {"real"}


@requires_outputs
def test_A_all_four_seeds_and_three_routes():
    s = pd.read_csv(os.path.join(PRIM_T, "trained_real_frequency_slopes.tsv"),
                    sep="\t")
    assert sorted(s["seed"].unique()) == [19, 20, 21, 22]
    assert sorted(s["route"].unique()) == sorted(common.ROUTES)


def test_A_exact_zero_seed_set():
    assert common.CEILING_SEEDS == [19, 20, 22]
    assert 21 in common.SEEDS and 21 not in common.CEILING_SEEDS


# =======================================================  B  frequency ======

@requires_canonical
@pytest.mark.parametrize("regime", [PRIMARY, SENSITIVITY, "UNTRAINED_REAL"])
def test_B_zipf_finite_and_thresholds_hold(canon, regime):
    cov = fq.standardized_covariates(canon, regime)
    assert np.isfinite(cov["zipf"]).all()
    v = fq.verify_frequency_classes(cov)
    assert v["n_in_excluded_gap"] == 0
    assert v["n_mismatched_labels"] == 0
    lo = cov[cov["frequency_class"] == "low"]["zipf"]
    hi = cov[cov["frequency_class"] == "high"]["zipf"]
    assert (lo <= fq.ZIPF_LOW_MAX).all()
    assert (hi >= fq.ZIPF_HIGH_MIN).all()


def test_B_thresholds_are_frozen():
    assert fq.ZIPF_LOW_MAX == 3.5
    assert fq.ZIPF_HIGH_MIN == 4.0


@requires_canonical
def test_B_zipf_identical_across_seeds_and_routes(canon):
    d = fq.regime_subset(canon, SENSITIVITY)
    assert int((d.groupby("item_id")["zipf_frequency"].nunique() > 1).sum()) == 0


# ==================================================  C  sign conventions ====

def test_C_negative_slope_means_frequency_helps():
    """Synthetic: errors fall as Zipf rises => negative slope."""
    rows = []
    for seed in common.SEEDS:
        for route in ("full", "ltm"):          # 'full' is the standardization anchor
            for i in range(40):
                z = 3.0 + i * 0.05
                rows.append({"seed": seed, "route": route, "item_id": f"i{i}",
                             "source_lexicality": "real", "zipf_frequency": z,
                             "target_length": 5,
                             "raw_edit_distance": 10.0 - 2.0 * z,
                             "word_error": 0, "frequency_class":
                                 "low" if z <= 3.5 else ("high" if z >= 4.0 else "x"),
                             "lichtheim_exposure_status": "TRAINED_REAL_EXACT",
                             "in_TRAINED_REAL_FREQUENCY_PRIMARY": True})
    df = pd.DataFrame(rows)
    s = fq.continuous_slopes(df, PRIMARY)
    assert (s["zipf_slope"] < 0).all()


def test_C_high_low_positive_means_low_is_harder():
    rows = []
    for seed in common.SEEDS:
        for route in ("full", "ltm"):          # 'full' is the standardization anchor
            for i, (z, err) in enumerate([(3.0, 2.0)] * 20 + [(5.0, 1.0)] * 20):
                rows.append({"seed": seed, "route": route, "item_id": f"i{i}",
                             "source_lexicality": "real", "zipf_frequency": z,
                             "target_length": 5, "raw_edit_distance": err,
                             "word_error": 0,
                             "frequency_class": "low" if z <= 3.5 else "high",
                             "lichtheim_exposure_status": "TRAINED_REAL_EXACT",
                             "in_TRAINED_REAL_FREQUENCY_PRIMARY": True})
    df = pd.DataFrame(rows)
    h = fq.high_low_contrasts(df, PRIMARY)
    assert (h["high_low_contrast_raw_edit_distance"] > 0).all()


@requires_outputs
def test_C_route_contrast_orientations_documented_and_consistent():
    r = pd.read_csv(os.path.join(
        PRIM_T, "trained_real_frequency_route_contrasts.tsv"), sep="\t")
    assert set(r["route_contrast"]) == {"ltm_minus_wm", "full_minus_wm",
                                        "ltm_minus_full"}
    assert np.allclose(r["raw_route_slope_difference"],
                       r["slope_route_A"] - r["slope_route_B"])
    assert np.allclose(r["frequency_benefit_route_difference"],
                       -r["raw_route_slope_difference"])
    assert r["raw_difference_meaning"].str.contains("slope").all()
    assert r["benefit_meaning"].str.contains("larger").all()


# ====================================================  D  standardization ===

@requires_canonical
def test_D_covariates_are_item_level_and_regime_fixed(canon):
    a = fq.standardized_covariates(canon, PRIMARY)
    b = fq.standardized_covariates(canon, PRIMARY)
    pd.testing.assert_frame_equal(a, b)
    assert a["zipf_mean"].nunique() == 1 and a["zipf_sd"].nunique() == 1
    assert a["standardized_zipf"].mean() == pytest.approx(0.0, abs=1e-12)
    assert a["standardized_zipf"].std(ddof=0) == pytest.approx(1.0, abs=1e-12)
    assert len(a) == 671


def test_D_missing_standardization_anchor_raises():
    """No silent NaN covariates when the anchor slice is absent."""
    df = pd.DataFrame([{"seed": 19, "route": "ltm", "item_id": "i0",
                        "source_lexicality": "real", "zipf_frequency": 4.0,
                        "target_length": 5, "raw_edit_distance": 0.0,
                        "word_error": 0, "frequency_class": "high",
                        "lichtheim_exposure_status": "TRAINED_REAL_EXACT",
                        "in_TRAINED_REAL_FREQUENCY_PRIMARY": True}])
    with pytest.raises(ValueError, match="standardization anchor"):
        fq.standardized_covariates(df, PRIMARY)


@requires_canonical
def test_D_same_covariates_reused_across_seeds_and_routes(canon):
    """The module must not re-standardize per seed or per route."""
    src = open(os.path.join(PKG_DIR, "frequency.py")).read()
    assert src.count("def standardized_covariates") == 1
    body = src.split("def standardized_covariates")[1].split("\ndef ")[0]
    assert 'seed"] == SEEDS[0]' in body and 'route"] == "full"' in body
    cov = fq.standardized_covariates(canon, PRIMARY)
    assert cov["item_id"].is_unique


# =======================================================  E  ceiling ========

@requires_outputs
def test_E_all_zero_outcomes_are_labelled():
    s = pd.read_csv(os.path.join(PRIM_T, "trained_real_frequency_slopes.tsv"),
                    sep="\t")
    for _, r in s.iterrows():
        if r["total_raw_edit_distance"] == 0 and r["n_erroneous_items"] == 0:
            assert r["model_status"] == "ALL_ZERO_OUTCOME"
            assert r["zipf_slope"] == 0.0


@requires_outputs
def test_E_logistic_not_attempted_under_separation_or_sparsity():
    st = pd.read_csv(os.path.join(
        SHARED, "frequency_word_error_model_status.tsv"), sep="\t")
    for _, r in st.iterrows():
        if r["model_status"] in ("ALL_ZERO_OUTCOME", "SPARSE_ERROR_LIMITED"):
            assert not r["logistic_fit_attempted"]
            assert isinstance(r["reason"], str) and r["reason"]


def test_E_zero_slope_is_not_called_evidence_of_absence():
    spec = json.load(open(os.path.join(
        Q, "_control", "frequency_analysis_spec.json")))
    assert spec["ceiling_policy"]["absence_claim_permitted"] is False
    assert spec["no_zero_slope_as_proof_of_absence"] is True
    results = os.path.join(Q, "frequency_results.md")
    if os.path.exists(results):
        # collapse whitespace: the statement must be present as prose, but the
        # document is hard-wrapped so it may straddle a line break
        text = " ".join(open(results).read().lower().split())
        assert "no absence of frequency encoding is claimed" in text
        assert "proof of absence" in text


# =======================================================  F  bootstrap ======

def test_F_bootstrap_configuration():
    assert common.BOOTSTRAP_REPLICATES == 10000
    assert common.BOOTSTRAP_SEED == 20260730
    assert common.BOOTSTRAP_CI_LEVEL == 95


@requires_outputs
def test_F_bootstrap_tables_record_b_and_seed():
    for p in (os.path.join(PRIM_T, "trained_real_frequency_bootstrap.tsv"),
              os.path.join(CG_T, "frequency_confidence_gate_bootstrap.tsv")):
        df = pd.read_csv(p, sep="\t")
        assert set(df["n_replicates"]) == {10000}
        assert set(df["random_seed"]) == {20260730}


@requires_canonical
def test_F_bootstrap_is_deterministic(canon):
    a = fq.bootstrap_confidence_gate(canon, PRIMARY)
    b = fq.bootstrap_confidence_gate(canon, PRIMARY)
    pd.testing.assert_frame_equal(a, b)


@requires_outputs
def test_F_seed21_present_and_exact_zero_kept_separate():
    s = pd.read_csv(os.path.join(PRIM_T, "trained_real_frequency_slopes.tsv"),
                    sep="\t")
    assert 21 in set(s["seed"])
    sens = pd.read_csv(os.path.join(SENS, "exact_zero_seed_sensitivity.tsv"),
                       sep="\t")
    assert "exact_zero_seeds_mean" in sens.columns
    assert sens["seed21_included"].all()


# ===============================================  G  visual conventions =====

def test_G_no_red_or_blue_in_frequency_figures():
    src = open(os.path.join(PKG_DIR, "plot_frequency.py")).read()
    assert '"red"' not in src and '"blue"' not in src
    assert "LEXICALITY_COLOR" not in src
    for c in (pf.NEUTRAL_DARK, pf.NEUTRAL_MID, pf.NEUTRAL_LIGHT):
        assert c.startswith("#")


def test_G_confidence_and_gate_are_separate_panels():
    src = open(os.path.join(PKG_DIR, "plot_frequency.py")).read()
    body = src.split("def plot_confidence_gate")[1].split("\ndef ")[0]
    assert "subplots(1, 2" in body
    assert "twinx" not in body


@requires_outputs
def test_G_all_four_seed_points_available_to_the_figure():
    for p, col in ((os.path.join(PRIM_T, "trained_real_frequency_slopes.tsv"),
                    "zipf_slope"),
                   (os.path.join(CG_T, "frequency_confidence_slopes.tsv"),
                    "zipf_slope"),
                   (os.path.join(CG_T, "frequency_gate_slopes.tsv"),
                    "zipf_slope")):
        df = pd.read_csv(p, sep="\t")
        assert sorted(df["seed"].unique()) == [19, 20, 21, 22]
        assert col in df.columns


def test_G_continuous_zipf_is_declared_primary():
    spec = json.load(open(os.path.join(
        Q, "_control", "frequency_analysis_spec.json")))
    assert spec["estimands"]["continuous_frequency_slope"]["primary"] is True
    assert spec["estimands"]["high_low_contrast"]["secondary"] is True


def test_G_confidence_not_mislabelled():
    for f in ("frequency.py", "plot_frequency.py"):
        src = open(os.path.join(PKG_DIR, f)).read().lower()
        assert "word probability" not in src.replace("not a word probability", "")
        assert "phonological similarity" not in src.replace(
            "not phonological similarity", "")


# ==================================================  H  non-regression ======

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


LIVING = {
    "reports/behavioral_wfe_fulllexicon_93a577f/README.md",
    "reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv",
}


def _verify(manifest, allow=frozenset()):
    bad = []
    for line in open(manifest):
        h, _, rel = line.strip().partition("  ")
        if rel in allow:
            continue
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or _sha(p) != h:
            bad.append(rel)
    return bad


@pytest.mark.skipif(not os.path.exists(os.path.join(
    REPORT, "validation", "sprint1_outputs.sha256")), reason="no manifest")
def test_H_sprint1_scientific_outputs_unchanged():
    assert not _verify(os.path.join(REPORT, "validation",
                                    "sprint1_outputs.sha256"), allow=LIVING)


@pytest.mark.skipif(not os.path.exists(os.path.join(
    REPORT, "morphology", "validation", "morphology_outputs.sha256")),
    reason="no manifest")
def test_H_morphology_outputs_unchanged():
    assert not _verify(os.path.join(REPORT, "morphology", "validation",
                                    "morphology_outputs.sha256"))


@pytest.mark.skipif(not os.path.exists(os.path.join(
    ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f", "full_wfe_evaluation",
    "_control", "production_scientific_outputs_FINAL.sha256")),
    reason="no manifest")
def test_H_production_outputs_unchanged():
    assert not _verify(os.path.join(
        ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
        "full_wfe_evaluation", "_control",
        "production_scientific_outputs_FINAL.sha256"))


@requires_canonical
def test_H_canonical_table_unchanged():
    prov = json.load(open(os.path.join(
        REPORT, "behavioral_analysis_provenance.json")))
    assert _sha(common.CANONICAL_TABLE) == prov["canonical_table_sha256"]


# ===================================================  I  no inference =======

FREQ_MODULES = ["frequency.py", "plot_frequency.py"]


@pytest.mark.parametrize("module", FREQ_MODULES)
def test_I_no_torch_or_eval_imports(module):
    tree = ast.parse(open(os.path.join(PKG_DIR, module)).read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("torch", "models", "evaluate", "train"):
        assert banned not in imported


@pytest.mark.parametrize("module", FREQ_MODULES)
def test_I_no_checkpoint_or_absolute_paths(module):
    src = open(os.path.join(PKG_DIR, module)).read()
    for banned in ("external_eval", "load_model_and_vocab", "/Users/", "/home/"):
        assert banned not in src
