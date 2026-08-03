"""Tests for the Sprint-4 error taxonomy and premature-EOS diagnostics.

No checkpoint is loaded and no model inference occurs (group J).

Frozen expected values live in one compact fixture,
reports/.../error_taxonomy/_control/error_taxonomy_reference_values.json,
rather than being scattered through the assertions.  That file is an immutable
expectation: it must never be regenerated to make a failing test pass.
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

from scripts.behavioral_analysis import common
from scripts.behavioral_analysis import eos_diagnostics as eos
from scripts.behavioral_analysis import error_taxonomy as et
from scripts.behavioral_analysis import plot_error_taxonomy as pet
from scripts.behavioral_analysis.io import load_canonical

PKG_DIR = os.path.join(ROOT, "scripts", "behavioral_analysis")
REPORT = os.path.join(ROOT, "reports", "behavioral_wfe_fulllexicon_93a577f")
Q = os.path.join(REPORT, "error_taxonomy")
FAITH_T = os.path.join(Q, "faithful", "tables")
FAITH_F = os.path.join(Q, "faithful", "figures")
CLEAN_T = os.path.join(Q, "clean", "tables")
CLEAN_F = os.path.join(Q, "clean", "figures")
EOS_T = os.path.join(Q, "eos", "tables")
EOS_F = os.path.join(Q, "eos", "figures")
STRATA = os.path.join(Q, "strata", "tables")
EXAMPLES = os.path.join(Q, "examples")
CONTROL = os.path.join(Q, "_control")
REFERENCE = os.path.join(CONTROL, "error_taxonomy_reference_values.json")

OPS = et.OPERATIONS

requires_canonical = pytest.mark.skipif(
    not os.path.exists(common.CANONICAL_TABLE), reason="canonical table absent")
requires_outputs = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_T,
                                    "clean_error_taxonomy_cells.tsv")),
    reason="Sprint-4 outputs not generated")
requires_reference = pytest.mark.skipif(
    not os.path.exists(REFERENCE), reason="reference fixture absent")


@pytest.fixture(scope="module")
def canon():
    return load_canonical(common.CANONICAL_TABLE)


@pytest.fixture(scope="module")
def ref():
    with open(REFERENCE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def items(canon):
    return eos.item_level(canon, "LICHTHEIM_CLEAN")


def _read(path):
    return pd.read_csv(path, sep="\t")


def _text(*paths):
    out = []
    for p in paths:
        with open(p) as f:
            out.append(f.read())
    return "\n".join(out)


def _norm(s):
    """Collapse whitespace and drop markdown emphasis so prose checks are
    insensitive to `code spans` and **bold**, which carry no meaning here."""
    return " ".join(s.replace("`", "").replace("*", "").split())


# Phrases such as "premature EOS causes the length effect" must never be
# ASSERTED, but the reports are required to DENY them explicitly.  A bare
# substring search cannot tell the two apart, so proximity to a negation is
# what distinguishes a forbidden claim from a mandated disclaimer.
NEGATIONS = ("no claim", "not claimed", "never", "no statement", "does not",
             "do not", "is not", "are not", "cannot", "without", "no causal",
             "not causal", "neither", "avoided", "must be avoided", "no ",
             "not ")


def _asserts(txt: str, phrase: str, window: int = 160) -> bool:
    """True if `phrase` occurs with no negation in the preceding window."""
    start = 0
    while True:
        i = txt.find(phrase, start)
        if i < 0:
            return False
        before = txt[max(0, i - window):i]
        if not any(n in before for n in NEGATIONS):
            return True
        start = i + len(phrase)


# ==========================================  A  cohort and analysis sets =====

@requires_canonical
def test_A_canonical_shape_and_seeds(canon):
    assert len(canon) == common.EXPECTED_CANONICAL_ROWS
    assert sorted(canon["seed"].unique()) == common.SEEDS == [19, 20, 21, 22]
    assert sorted(canon["route"].unique()) == sorted(common.ROUTES)


@requires_canonical
def test_A_clean_set_is_671_real_and_391_pseudo(canon):
    d = et.regime_subset(canon, "LICHTHEIM_CLEAN")
    one = d[(d["seed"] == 19) & (d["route"] == "full")]
    assert int((one["source_lexicality"] == "real").sum()) == 671
    assert int((one["source_lexicality"] == "pseudo").sum()) == 391
    assert set(one[one["source_lexicality"] == "real"]
               ["lichtheim_exposure_status"]) == {"TRAINED_REAL_EXACT"}
    assert set(one[one["source_lexicality"] == "pseudo"]
               ["lichtheim_exposure_status"]) == {"NOVEL_PSEUDOWORD"}


@requires_canonical
def test_A_faithful_set_is_all_1200_full_route(canon):
    d = et.regime_subset(canon, "FAITHFUL_WFE_ALL")
    assert set(d["route"]) == {"full"}
    assert d[d["seed"] == 19]["item_id"].nunique() == 1200


@requires_canonical
def test_A_no_exposure_stratum_is_dropped(canon):
    d = et.regime_subset(canon, "ALL_WITH_EXPOSURE_STRATA")
    one = d[(d["seed"] == 19) & (d["route"] == "full")]
    assert set(one["lichtheim_exposure_status"]) == set(common.EXPOSURE_ORDER)
    assert len(one) == 1200


@requires_outputs
def test_A_all_four_seeds_present_in_every_published_table():
    for root in (FAITH_T, CLEAN_T, EOS_T, STRATA):
        for name in sorted(os.listdir(root)):
            df = _read(os.path.join(root, name))
            if "seed" not in df.columns:
                continue
            assert sorted(df["seed"].unique()) == common.SEEDS, name


@requires_outputs
def test_A_no_trained_real_item_silently_filtered():
    cells = _read(os.path.join(CLEAN_T, "clean_error_taxonomy_cells.tsv"))
    real = cells[(cells["seed"] == 19) & (cells["route"] == "full")
                 & (cells["source_lexicality"] == "real")]
    assert int(real["n_items"].sum()) == 671


# =========================================  B  Levenshtein convention ========

@requires_canonical
def test_B_operations_reconstruct_edit_distance(canon):
    tot = canon[OPS].sum(axis=1).to_numpy()
    assert np.array_equal(tot, canon["raw_edit_distance"].to_numpy())


def test_B_exactly_three_operations_no_fourth():
    assert OPS == ["substitutions", "deletions", "insertions"]
    assert len(OPS) == 3
    for name in ("eos", "premature", "truncation", "early_stop"):
        assert not any(name in op for op in OPS)


def test_B_editops_backend_is_recorded():
    assert et.EDITOPS_BACKEND == "Levenshtein.editops"
    assert et.EDITOPS_VERSION == "0.27.3"


@pytest.mark.parametrize("module", ["error_taxonomy.py", "eos_diagnostics.py",
                                    "plot_error_taxonomy.py"])
def test_B_no_alternative_aligner_is_called(module):
    """Stored counts are used directly; nothing re-aligns the sequences."""
    src = open(os.path.join(PKG_DIR, module)).read()
    tree = ast.parse(src)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                called.add(fn.attr)
            elif isinstance(fn, ast.Name):
                called.add(fn.id)
    for banned in ("editops", "opcodes", "distance", "ratio", "align",
                   "get_opcodes", "SequenceMatcher", "levenshtein"):
        assert banned not in called, f"{module} calls {banned}"
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for banned in ("Levenshtein", "rapidfuzz", "difflib", "editdistance"):
        assert banned not in imported, f"{module} imports {banned}"


@requires_outputs
def test_B_tie_breaking_and_horizon_limits_are_stated():
    txt = _text(os.path.join(Q, "error_taxonomy_results.md"),
                os.path.join(CLEAN_F,
                             "clean_error_taxonomy_by_route_caption.md"))
    txt = _norm(txt).lower()
    assert "tie-breaking" in txt
    assert "without changing the total edit distance" in txt
    assert "terminal insertions" in txt and "unobservable" in txt


# ============================  C  EOS convention and its observability =======

def test_C_classification_boundaries():
    assert eos.classify_eos(0, 5) == eos.PREMATURE
    assert eos.classify_eos(4, 5) == eos.PREMATURE
    assert eos.classify_eos(5, 5) == eos.ON_TIME       # unreachable in practice
    assert eos.classify_eos(6, 5) == eos.LATE          # unreachable in practice
    assert eos.classify_eos(None, 5) == eos.NOT_OBSERVED
    assert eos.classify_eos(float("nan"), 5) == eos.NOT_OBSERVED
    assert eos.classify_eos("", 5) == eos.NOT_OBSERVED
    assert eos.classify_eos("x", 5) == eos.UNAVAILABLE


def test_C_missing_eos_is_never_relabelled_on_time():
    """EOS_NOT_OBSERVED must never be turned into ON_TIME_EOS."""
    for L in (3, 4, 5, 7, 8, 9):
        for missing in (None, float("nan"), ""):
            assert eos.classify_eos(missing, L) != eos.ON_TIME
            assert eos.classify_eos(missing, L) != eos.LATE
            assert eos.classify_eos(missing, L) == eos.NOT_OBSERVED


def test_C_premature_requires_a_positively_observed_early_eos():
    """Only a numeric observation strictly below L yields PREMATURE_EOS."""
    assert eos.classify_eos(2, 5) == eos.PREMATURE
    assert eos.classify_eos(None, 5) != eos.PREMATURE
    assert eos.classify_eos("", 5) != eos.PREMATURE
    assert np.isnan(eos.eos_shortfall(None, 5))
    assert np.isnan(eos.eos_shortfall(5, 5))
    assert eos.eos_shortfall(2, 5) == 3.0


@requires_canonical
def test_C_no_observed_eos_at_or_beyond_the_boundary(items):
    """Structural: the readout window ends at L-1, so L and beyond cannot occur."""
    assert int((items["eos_class"] == eos.ON_TIME).sum()) == 0
    assert int((items["eos_class"] == eos.LATE).sum()) == 0
    obs = items[items["eos_class"] == eos.PREMATURE]
    assert (obs["eos_position"].astype(float)
            < obs["target_length"].astype(float)).all()
    assert (obs["eos_position"].astype(float)
            == obs["predicted_length"].astype(float)).all()


@requires_outputs
def test_C_reports_define_eos_not_observed_as_horizon_limited():
    txt = _norm(_text(
        os.path.join(Q, "error_taxonomy_results.md"),
        os.path.join(Q, "length_effect_mechanism_handoff.md"),
        os.path.join(Q, "eos_convention.md"),
        os.path.join(EOS_F, "premature_eos_by_route_caption.md"))).lower()
    assert "no eos was observed within the instrumented evaluation horizon" in txt
    assert "ambiguous" in txt


@requires_outputs
def test_C_reports_state_on_time_and_late_are_unobservable():
    txt = _norm(_text(
        os.path.join(Q, "error_taxonomy_results.md"),
        os.path.join(Q, "length_effect_mechanism_handoff.md"),
        os.path.join(EOS_F, "premature_eos_by_route_caption.md"))).lower()
    assert "structurally unobservable" in txt
    assert "does not mean the decoder never stops correctly" in txt
    assert "does not mean late stopping never occurs" in txt
    assert "only premature_eos is positively observable" in txt


@requires_outputs
def test_C_missing_eos_never_called_correct_stopping():
    """No document may equate EOS_NOT_OBSERVED with correct stopping."""
    docs = [os.path.join(Q, "error_taxonomy_results.md"),
            os.path.join(Q, "length_effect_mechanism_handoff.md"),
            os.path.join(Q, "eos_convention.md"),
            os.path.join(Q, "error_taxonomy_analysis_spec.md"),
            os.path.join(EOS_F, "premature_eos_by_route_caption.md")]
    txt = _norm(_text(*docs)).lower()
    for banned in ("eos_not_observed means correct stop",
                   "eos_not_observed is correct stopping",
                   "missing eos means the model stopped correctly",
                   "no eos means correct stopping"):
        assert banned not in txt
    assert "never read as correct stopping" in txt or \
           "never read as evidence of correct stopping" in txt


def test_C_class_labels_are_not_renamed():
    assert eos.PREMATURE == "PREMATURE_EOS"
    assert eos.ON_TIME == "ON_TIME_EOS"
    assert eos.LATE == "LATE_EOS"
    assert eos.NOT_OBSERVED == "EOS_NOT_OBSERVED"
    assert eos.UNAVAILABLE == "EOS_UNAVAILABLE"


@requires_outputs
def test_C_published_class_counts_flag_observability():
    df = _read(os.path.join(EOS_T, "premature_eos_class_counts.tsv"))
    for cls in (eos.ON_TIME, eos.LATE):
        sub = df[df["eos_class"] == cls]
        assert len(sub) and not sub["structurally_observable"].any()
        assert int(sub["n"].sum()) == 0
    assert df[df["eos_class"] == eos.PREMATURE]["structurally_observable"].all()
    amb = df[df["eos_class"] == eos.NOT_OBSERVED]["note"].astype(str)
    assert amb.str.contains("ambiguous").all()


# ==================================================  D  estimand definitions =

@requires_outputs
def test_D_primary_operation_means_are_unconditional():
    cells = _read(os.path.join(CLEAN_T, "clean_error_taxonomy_cells.tsv"))
    for op in OPS:
        expected = cells[f"total_{op}"] / cells["n_items"]
        assert np.allclose(cells[f"mean_{op}_per_item"], expected)
    err = cells[cells["n_erroneous_items"] > 0]
    for op in OPS:
        assert np.allclose(err[f"mean_{op}_per_erroneous_item"],
                           err[f"total_{op}"] / err["n_erroneous_items"])


@requires_outputs
def test_D_zero_operation_cells_carry_no_manufactured_proportion():
    comp = _read(os.path.join(CLEAN_T,
                              "clean_error_taxonomy_composition.tsv"))
    zero = comp[comp["total_edit_operations"] == 0]
    assert (zero["composition_status"] == et.NO_COMPOSITION).all()
    for op in OPS:
        assert zero[f"proportion_{op}"].isna().all()
    ok = comp[comp["composition_status"] == "OK"]
    assert (ok["total_edit_operations"] > 0).all()
    assert np.allclose(ok[[f"proportion_{op}" for op in OPS]].sum(axis=1), 1.0)


@requires_canonical
def test_D_all_item_shortfall_does_not_relabel_eos_class(items):
    """Zero shortfall for a summary must not turn an item into PREMATURE_EOS."""
    zeroed = items[items["eos_shortfall_all_items"] == 0.0]
    assert (zeroed["eos_class"] != eos.PREMATURE).all()
    assert (zeroed["premature_eos"] == 0).all()
    prem = items[items["eos_class"] == eos.PREMATURE]
    assert (prem["eos_shortfall_all_items"] == prem["eos_shortfall"]).all()
    assert (prem["eos_shortfall"] >= 1).all()
    # the raw class column still distinguishes the two zero populations
    assert set(zeroed["eos_class"]) <= {eos.NOT_OBSERVED, eos.UNAVAILABLE}


@requires_canonical
def test_D_conditional_shortfall_denominator_is_premature_only(items):
    rates = eos.by_seed(items)
    for _, r in rates.iterrows():
        sub = items[(items["seed"] == r["seed"]) & (items["route"] == r["route"])
                    & (items["source_lexicality"] == r["source_lexicality"])]
        prem = sub[sub["eos_class"] == eos.PREMATURE]
        assert int(r["n_premature"]) == len(prem)
        if len(prem):
            assert np.isclose(r["conditional_mean_eos_shortfall"],
                              prem["eos_shortfall"].mean())
        else:
            assert np.isnan(r["conditional_mean_eos_shortfall"])
        assert np.isclose(r["mean_eos_shortfall_per_item"],
                          sub["eos_shortfall_all_items"].mean())


@requires_outputs
def test_D_slope_statuses_are_declared_not_forced():
    sl = _read(os.path.join(EOS_T, "premature_eos_length_slopes.tsv"))
    assert set(sl["model_status"]) <= {eos.STATUS_OK, eos.STATUS_ALL_ZERO,
                                       eos.STATUS_ALL_ONE,
                                       eos.STATUS_INSUFFICIENT,
                                       eos.STATUS_NON_ESTIMABLE}
    assert not sl["logistic_forced"].any()
    assert (sl["model_type"].str.contains("descriptive")).all()
    zero = sl[sl["n_premature"] == 0]
    assert (zero["model_status"] == eos.STATUS_ALL_ZERO).all()
    assert (zero["length_slope"] == 0.0).all()


@requires_outputs
def test_D_small_cells_are_flagged_never_excluded():
    expo = _read(os.path.join(STRATA, "exposure_error_taxonomy.tsv"))
    assert set(expo["lichtheim_exposure_status"]) == set(common.EXPOSURE_ORDER)
    small = expo[expo["n_items"] <= common.SMALL_STRATUM_MAX_N]
    assert len(small) and small["descriptive_only"].all()
    assert (small["cell_flag"] == "VERY_SMALL_CELL").all()


# =====================================  E  EOS / deletion overlap integrity ==

def test_E_overlap_does_not_infer_eos_from_deletions():
    """The 2x2 must read eos_class, never derive EOS status from counts."""
    src = open(os.path.join(PKG_DIR, "eos_diagnostics.py")).read()
    fn = src[src.index("def deletion_overlap"):src.index("def summarise")]
    assert "premature_eos" in fn and "has_deletion" in fn
    for banned in ("classify_eos(", "deletions >", "deletions ==",
                   "eos_position", "target_length"):
        assert banned not in fn, f"deletion_overlap references {banned}"


@requires_canonical
def test_E_eos_and_deletion_are_independent_measurements(items):
    """Neither column is a function of the other in the item-level table."""
    assert items["premature_eos"].equals(
        (items["eos_class"] == eos.PREMATURE).astype(int))
    assert items["has_deletion"].equals((items["deletions"] > 0).astype(int))
    # there exist deletion-bearing items with no observed premature EOS
    assert int(((items["has_deletion"] == 1)
                & (items["premature_eos"] == 0)).sum()) > 0


@requires_outputs
def test_E_overlap_counts_sum_to_the_denominator():
    for name in ("premature_eos_deletion_overlap.tsv",
                 "premature_eos_deletion_overlap_by_length.tsv"):
        ov = _read(os.path.join(EOS_T, name))
        total = (ov["premature_eos_and_deletion"]
                 + ov["premature_eos_without_deletion"]
                 + ov["deletion_without_premature_eos"] + ov["neither"])
        assert (total == ov["n_items"]).all(), name
        assert ov["not_causal"].all(), name


@requires_outputs
def test_E_overlap_probabilities_are_labelled_non_causal():
    txt = _norm(_text(os.path.join(Q, "error_taxonomy_results.md"),
                      os.path.join(Q, "length_effect_mechanism_handoff.md"))
                ).lower()
    assert "neither probability is causal" in txt
    for banned in ("premature eos causes the length effect",
                   "premature eos causes the route length effect",
                   "deletion errors are premature eos",
                   "deletions are premature eos"):
        assert not _asserts(txt, banned), banned
    # the disclaimers themselves must be present
    assert "premature eos causes the route length effect" in txt


# ==========================================================  F  bootstrap ====

def test_F_bootstrap_configuration_is_frozen():
    assert common.BOOTSTRAP_REPLICATES == 10000
    assert common.BOOTSTRAP_SEED == 20260730
    assert common.BOOTSTRAP_CI_LEVEL == 95


@requires_outputs
def test_F_bootstrap_table_records_b_and_seed():
    b = _read(os.path.join(CLEAN_T, "clean_error_taxonomy_bootstrap.tsv"))
    assert (b["n_replicates"] == common.BOOTSTRAP_REPLICATES).all()
    assert (b["random_seed"] == common.BOOTSTRAP_SEED).all()
    assert (b["ci_definition"] == "95% percentile interval").all()
    assert (b["ci_low"] <= b["ci_high"]).all()


@requires_canonical
def test_F_bootstrap_is_deterministic(canon):
    a = et.bootstrap_operations(canon, "LICHTHEIM_CLEAN")
    b = et.bootstrap_operations(canon, "LICHTHEIM_CLEAN")
    pd.testing.assert_frame_equal(a, b)


@requires_outputs
def test_F_seed21_kept_and_exact_zero_reported_separately():
    for name in ("clean_error_taxonomy_summary.tsv",
                 "clean_error_taxonomy_route_contrasts_summary.tsv"):
        s = _read(os.path.join(CLEAN_T, name))
        assert s["seed21_included"].all(), name
        assert "exact_zero_seeds_mean" in s.columns, name
        assert s["seed_values"].str.contains("21:").all(), name


# =======================================================  G  presentation ====

FIGURES = [(FAITH_F, "faithful_figure8a_error_types"),
           (CLEAN_F, "clean_error_taxonomy_by_route"),
           (CLEAN_F, "clean_error_taxonomy_full_wm_zoom"),
           (EOS_F, "premature_eos_by_route")]


@requires_outputs
@pytest.mark.parametrize("out_dir,stem", FIGURES)
def test_G_every_figure_has_three_formats_and_a_caption(out_dir, stem):
    for ext in ("png", "pdf", "svg"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        assert os.path.exists(p) and os.path.getsize(p) > 0, p
    cap = os.path.join(out_dir, f"{stem}_caption.md")
    assert os.path.exists(cap) and os.path.getsize(cap) > 0


@requires_outputs
def test_G_absolute_scale_primary_exists_alongside_the_zoom():
    primary = os.path.join(CLEAN_F, "clean_error_taxonomy_by_route.png")
    zoom = os.path.join(CLEAN_F, "clean_error_taxonomy_full_wm_zoom.png")
    assert os.path.exists(primary), "the common-scale primary must be retained"
    z = _read(os.path.join(CLEAN_T, "clean_error_taxonomy_zoom_rule.tsv"))
    assert not bool(z["zoom_replaces_primary"].iloc[0])
    assert os.path.exists(zoom) == bool(z["zoom_figure_produced"].iloc[0])


@requires_outputs
def test_G_zoom_exists_only_because_the_frozen_ratio_exceeds_ten():
    z = _read(os.path.join(CLEAN_T, "clean_error_taxonomy_zoom_rule.tsv"))
    ratio = float(z["ratio"].iloc[0])
    fires = bool(z["rule_fires"].iloc[0])
    assert fires == (ratio > pet.ZOOM_RATIO_TRIGGER)
    assert pet.ZOOM_RATIO_TRIGGER == 10.0
    assert bool(z["frozen_before_results"].iloc[0])
    assert os.path.exists(os.path.join(
        CLEAN_F, "clean_error_taxonomy_full_wm_zoom.png")) == fires
    cap = os.path.join(CLEAN_F, "clean_error_taxonomy_full_wm_zoom_caption.md")
    if fires:
        assert "does not replace" in _norm(_text(cap)).lower()


@requires_canonical
def test_G_zoom_rule_recomputes_from_the_canonical_table(canon):
    cells = et.clean_cells(canon)
    z = pet.zoom_rule(cells)
    pub = _read(os.path.join(CLEAN_T, "clean_error_taxonomy_zoom_rule.tsv"))
    assert np.isclose(float(z["ratio"].iloc[0]), float(pub["ratio"].iloc[0]))
    assert bool(z["rule_fires"].iloc[0]) == bool(pub["rule_fires"].iloc[0])


def test_G_operation_type_is_not_encoded_by_red_or_blue():
    assert set(pet.OPERATION_HATCH) == set(OPS)
    for op, hatch in pet.OPERATION_HATCH.items():
        assert "red" not in hatch and "blue" not in hatch
    src = open(os.path.join(PKG_DIR, "plot_error_taxonomy.py")).read()
    assert 'LEXICALITY_COLOR[lex]' in src
    # no operation-keyed colour map may exist
    for op in OPS:
        assert f'"{op}": "red"' not in src and f'"{op}": "blue"' not in src
    assert common.LEXICALITY_COLOR == {"real": "red", "pseudo": "blue"}


@requires_outputs
def test_G_captions_name_the_colour_convention():
    txt = _norm(_text(
        os.path.join(CLEAN_F, "clean_error_taxonomy_by_route_caption.md"),
        os.path.join(EOS_F, "premature_eos_by_route_caption.md"))).lower()
    assert "colour encodes lexicality" in txt
    assert "hatch" in txt


@requires_outputs
def test_G_faithful_caption_separates_source_label_from_exposure():
    cap = _norm(_text(os.path.join(
        FAITH_F, "faithful_figure8a_error_types_caption.md"))).lower()
    assert "is not training exposure" in cap
    assert "122" in cap


# ==============================================  H  numerical non-regression =

@requires_outputs
@requires_reference
def test_H_faithful_condition_structure(ref):
    f = _read(os.path.join(FAITH_T, "faithful_condition_error_types.tsv"))
    exp = ref["faithful_figure8a"]
    order = (f[f["seed"] == 19].sort_values("condition_order")["condition"]
             .tolist())
    assert order == exp["condition_order"]
    assert len(order) == 12
    assert (f["n_items"] == exp["n_items_per_condition"]).all()
    for cond, ops in exp["mean_over_seeds"].items():
        sub = f[f["condition"] == cond]
        for op, want in ops.items():
            assert np.isclose(sub[f"mean_{op}_per_item"].mean(), want), \
                f"{cond}/{op}"


@requires_outputs
@requires_reference
def test_H_clean_cells_match_frozen_reference(ref):
    cells = _read(os.path.join(CLEAN_T, "clean_error_taxonomy_cells.tsv"))
    for key, exp in ref["clean_cells_mean_over_seeds"].items():
        route, lex, grp = key.split("|")
        g = cells[(cells["route"] == route)
                  & (cells["source_lexicality"] == lex)
                  & (cells["broad_length"] == grp)]
        assert int(g["n_items"].iloc[0]) == exp["n_items"], key
        assert int(g["n_erroneous_items"].sum()) == \
            exp["n_erroneous_items_total"], key
        assert np.isclose(g["total_edit_operations"].sum(),
                          exp["total_edit_operations"]), key
        for op in OPS:
            assert np.isclose(g[f"mean_{op}_per_item"].mean(), exp[op]), \
                f"{key}/{op}"


@requires_canonical
@requires_reference
def test_H_route_level_eos_counts_reproduce_from_item_data(canon, ref):
    items = eos.item_level(canon, "LICHTHEIM_CLEAN")
    by_route = items.groupby("route")["premature_eos"].sum().to_dict()
    for route, want in ref["premature_eos_events_by_route"].items():
        assert int(by_route[route]) == want, route
    assert int(items["premature_eos"].sum()) == \
        ref["premature_eos_events_total"] == 87
    counts = items["eos_class"].value_counts().to_dict()
    for cls, want in ref["eos_class_totals"].items():
        assert int(counts.get(cls, 0)) == want, cls


@requires_canonical
@requires_reference
def test_H_ltm_eos_length_slopes_match_reference(canon, ref):
    items = eos.item_level(canon, "LICHTHEIM_CLEAN")
    sl = eos.length_slopes(items)
    ltm = sl[(sl["route"] == "ltm") & (sl["source_lexicality"] == "pseudo")]
    for _, r in ltm.iterrows():
        want = ref["ltm_pseudo_eos_length_slopes_by_seed"][str(int(r["seed"]))]
        assert np.isclose(r["length_slope"], want), int(r["seed"])
        assert r["model_status"] == eos.STATUS_OK
    assert (ltm["length_slope"] > 0).all(), "positive in all four seeds"


@requires_outputs
@requires_reference
def test_H_overlap_totals_match_reference(ref):
    ov = _read(os.path.join(EOS_T, "premature_eos_deletion_overlap.tsv"))
    for key, exp in ref["eos_deletion_overlap_pooled"].items():
        route, lex = key.split("|")
        g = ov[(ov["route"] == route) & (ov["source_lexicality"] == lex)]
        assert len(g) == 1, key
        for col, want in exp.items():
            assert int(g[col].iloc[0]) == want, f"{key}/{col}"


@requires_outputs
@requires_reference
def test_H_zoom_rule_matches_reference(ref):
    z = _read(os.path.join(CLEAN_T, "clean_error_taxonomy_zoom_rule.tsv"))
    exp = ref["zoom_rule"]
    assert np.isclose(float(z["ratio"].iloc[0]), exp["ratio"])
    assert np.isclose(float(z["max_mean_ltm_operation_per_item"].iloc[0]),
                      exp["max_mean_ltm"])
    assert bool(z["rule_fires"].iloc[0]) is exp["rule_fires"]


@requires_outputs
@requires_canonical
def test_H_seed22_examples_obey_the_frozen_sorting(items):
    """Re-derive the selection from the canonical table, not from the file."""
    err = _read(os.path.join(EXAMPLES,
                             "seed22_illustrative_pseudoword_errors.tsv"))
    assert set(err["seed"]) == {22}
    assert (err["raw_edit_distance"] > 0).all()

    d = items[(items["seed"] == 22) & (items["source_lexicality"] == "pseudo")
              & (items["word_error"] == 1)].copy()
    d["_s"] = d["eos_shortfall"].fillna(-1.0)   # missing shortfall sorts last
    for route, g in err.groupby("route"):
        assert len(g) <= 20, route
        want = (d[d["route"] == route]
                .sort_values(["raw_edit_distance", "_s", "item_id"],
                             ascending=[False, False, True])
                .head(20)["item_id"].tolist())
        assert sorted(g["item_id"]) == sorted(want), route

    pre = _read(os.path.join(EXAMPLES,
                             "seed22_illustrative_premature_eos.tsv"))
    assert set(pre["seed"]) == {22}
    assert set(pre["eos_class"]) == {eos.PREMATURE}
    assert (pre["eos_shortfall"] >= 1).all()
    p = items[(items["seed"] == 22) & (items["source_lexicality"] == "pseudo")
              & (items["premature_eos"] == 1)]
    for route, g in pre.groupby("route"):
        want = (p[p["route"] == route]
                .sort_values(["eos_shortfall", "raw_edit_distance", "item_id"],
                             ascending=[False, False, True])
                .head(20)["item_id"].tolist())
        assert sorted(g["item_id"]) == sorted(want), route


# =====================================  I  prior-sprint non-regression =======

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


# Declared living documents: each sprint extends these two files by policy.
LIVING = {
    "reports/behavioral_wfe_fulllexicon_93a577f/README.md",
    "reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv",
}


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


@pytest.mark.skipif(not os.path.exists(os.path.join(
    REPORT, "validation", "sprint1_outputs.sha256")), reason="no manifest")
def test_I_sprint1_scientific_outputs_unchanged():
    assert not _verify(os.path.join(REPORT, "validation",
                                    "sprint1_outputs.sha256"), allow=LIVING)


@pytest.mark.skipif(not os.path.exists(os.path.join(
    REPORT, "morphology", "validation", "morphology_outputs.sha256")),
    reason="no manifest")
def test_I_morphology_outputs_unchanged():
    assert not _verify(os.path.join(REPORT, "morphology", "validation",
                                    "morphology_outputs.sha256"), allow=LIVING)


@pytest.mark.skipif(not os.path.exists(os.path.join(
    REPORT, "frequency", "validation", "frequency_outputs.sha256")),
    reason="no manifest")
def test_I_frequency_outputs_unchanged():
    assert not _verify(os.path.join(REPORT, "frequency", "validation",
                                    "frequency_outputs.sha256"), allow=LIVING)


@pytest.mark.skipif(not os.path.exists(os.path.join(
    ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f", "full_wfe_evaluation",
    "_control", "production_scientific_outputs_FINAL.sha256")),
    reason="no manifest")
def test_I_production_outputs_unchanged():
    assert not _verify(os.path.join(
        ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
        "full_wfe_evaluation", "_control",
        "production_scientific_outputs_FINAL.sha256"))


@requires_canonical
def test_I_canonical_table_unchanged():
    with open(os.path.join(REPORT,
                           "behavioral_analysis_provenance.json")) as f:
        prov = json.load(f)
    assert _sha(common.CANONICAL_TABLE) == prov["canonical_table_sha256"]


# =======================================================  J  no inference ====

ET_MODULES = ["error_taxonomy.py", "eos_diagnostics.py",
              "plot_error_taxonomy.py"]


@pytest.mark.parametrize("module", ET_MODULES)
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


@pytest.mark.parametrize("module", ET_MODULES)
def test_J_no_checkpoint_or_absolute_paths(module):
    src = open(os.path.join(PKG_DIR, module)).read()
    # The bare word "checkpoint" is expected — the docstrings say "no
    # checkpoint" — so the ban targets the concrete ways one could be loaded.
    for banned in ("external_eval", "load_model_and_vocab", "/Users/",
                   "/home/", ".pt\"", ".pt'", "torch.load", "checkpoint_path",
                   "load_checkpoint", "state_dict"):
        assert banned not in src, banned


def test_J_manifest_declares_no_inference():
    p = os.path.join(CONTROL, "error_taxonomy_output_manifest.json")
    if not os.path.exists(p):
        pytest.skip("manifest not generated")
    with open(p) as f:
        m = json.load(f)
    assert m["model_inference_performed"] is False
    assert m["eos_and_levenshtein_kept_separate"] is True
    assert m["editops_backend"] == "Levenshtein.editops"


def test_J_no_causal_or_architectural_claim_in_any_sprint4_document():
    docs = []
    for base, _, names in os.walk(Q):
        docs += [os.path.join(base, n) for n in names if n.endswith(".md")]
    txt = _norm(_text(*sorted(docs))).lower()
    for banned in ("premature eos causes", "early eos causes",
                   "we recommend changing the architecture",
                   "the ltm architecture should be changed",
                   "should be redesigned", "explains the length effect"):
        assert not _asserts(txt, banned), banned


def test_J_negation_detector_distinguishes_claim_from_disclaimer():
    """Guards the guard: _asserts must not be satisfied by any string."""
    assert _asserts("premature eos causes the length effect.", "causes")
    assert not _asserts(
        "no claim is made that premature eos causes the length effect.",
        "causes")
    assert not _asserts(
        "we never state that premature eos causes the length effect.",
        "causes")
