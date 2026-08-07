"""Tests for the Yair corrections pass.

Everything here reads frozen predictions or the tables derived from them.  No
test loads a checkpoint, imports torch, or runs inference — a dedicated test
asserts that the two new modules cannot do so either.
"""
from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.behavioral_analysis import yair_corrections as YC       # noqa: E402
from scripts.behavioral_analysis.common import (EXPECTED_CLEAN_COUNTS,  # noqa: E402
                                                LENGTHS, REPORT_ROOT, ROUTES,
                                                SEEDS)
from scripts.behavioral_analysis.io import load_canonical             # noqa: E402

TAB = os.path.join(YC.CORRECTIONS_ROOT, "tables")
FIGS = os.path.join(YC.CORRECTIONS_ROOT, "figures")
CTRL = os.path.join(YC.CORRECTIONS_ROOT, "_control")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(TAB, "word_error_by_length_summary.tsv")),
    reason="yair corrections pass has not been run")


@pytest.fixture(scope="module")
def canon():
    return load_canonical()


@pytest.fixture(scope="module")
def clean(canon):
    return canon[canon["in_LICHTHEIM_CLEAN"]].copy()


def tab(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TAB, name), sep="\t")


# ------------------------------------------------------- population counts

def test_clean_counts_are_671_and_391(clean):
    one = clean[(clean["seed"] == SEEDS[0]) & (clean["route"] == "full")]
    c = one["lichtheim_exposure_status"].value_counts().to_dict()
    assert c["TRAINED_REAL_EXACT"] == EXPECTED_CLEAN_COUNTS["real"] == 671
    assert c["NOVEL_PSEUDOWORD"] == EXPECTED_CLEAN_COUNTS["pseudo"] == 391
    assert len(one) == 1062


def test_faithful_source_real_total_is_800(canon):
    one = canon[canon["in_FAITHFUL_WFE_ALL"] & (canon["seed"] == SEEDS[0])
                & (canon["route"] == "full")]
    assert int((one["source_lexicality"] == "real").sum()) == 800
    assert int((one["source_lexicality"] == "pseudo").sum()) == 400


def test_four_seeds_and_three_routes_everywhere():
    for name, seed_col in (("word_error_by_length_seed.tsv", True),
                           ("faithful_real_error_summary.tsv", False),
                           ("faithful_figure2C_by_route.tsv", False),
                           ("fi_route_estimability.tsv", True)):
        d = tab(name)
        assert sorted(d["route"].unique()) == sorted(ROUTES), name
        if seed_col:
            assert sorted(d["seed"].unique()) == SEEDS, name


def test_exact_length_bins_are_the_wfe_lengths():
    d = tab("word_error_by_length_seed.tsv")
    assert sorted(d["phoneme_length"].unique()) == LENGTHS == [3, 4, 5, 7, 8, 9]
    assert 6 not in set(d["phoneme_length"])


def test_exposure_categories_are_the_frozen_vocabulary(canon):
    d = tab("faithful_real_error_by_exposure.tsv")
    assert set(d["lichtheim_exposure_status"]) <= set(
        canon["lichtheim_exposure_status"].unique())
    # source-real items can only carry the three real exposure strata
    assert set(d["lichtheim_exposure_status"]) == {
        "TRAINED_REAL_EXACT", "TRAINED_REAL_PRON_VARIANT", "UNTRAINED_REAL"}


# ------------------------------------------------ T1 word error aggregation

def test_word_error_rates_recompute_from_the_canonical_table(clean):
    d = tab("word_error_by_length_seed.tsv")
    for _, r in d.sample(24, random_state=0).iterrows():
        cell = clean[(clean["route"] == r["route"])
                     & (clean["source_lexicality"] == r["source_lexicality"])
                     & (clean["target_length"] == r["phoneme_length"])
                     & (clean["seed"] == r["seed"])]
        assert len(cell) == r["n_items"]
        assert np.isclose(cell["word_error"].mean(), r["word_error_rate"])
        assert int(cell["word_error"].sum()) == r["n_word_errors"]


def test_summary_is_consistent_with_the_per_seed_rows():
    s = tab("word_error_by_length_seed.tsv")
    m = tab("word_error_by_length_summary.tsv")
    for _, r in m.iterrows():
        sub = s[(s.route == r["route"])
                & (s.source_lexicality == r["source_lexicality"])
                & (s.phoneme_length == r["phoneme_length"])]
        assert len(sub) == len(SEEDS)
        assert np.isclose(sub["word_error_rate"].mean(),
                          r["mean_word_error_rate_across_seeds"])
        assert np.isclose(sub["word_error_rate"].min(),
                          r["min_seed_word_error_rate"])
        assert np.isclose(sub["word_error_rate"].max(),
                          r["max_seed_word_error_rate"])
        assert r["ci_low"] <= r["ci_high"]


def test_word_error_rate_is_a_proportion():
    for name in ("word_error_by_length_seed.tsv",
                 "word_error_by_length_summary.tsv"):
        d = tab(name)
        col = ("word_error_rate" if "word_error_rate" in d.columns
               else "mean_word_error_rate_across_seeds")
        assert d[col].between(0.0, 1.0).all()


def test_item_counts_match_the_clean_population(clean):
    c = tab("word_error_by_length_item_counts.tsv")
    assert c["n_items"].sum() == 1062
    one = clean[(clean["seed"] == SEEDS[0]) & (clean["route"] == "full")]
    for _, r in c.iterrows():
        n = int(((one["source_lexicality"] == r["source_lexicality"])
                 & (one["target_length"] == r["phoneme_length"])).sum())
        assert n == r["n_items"]


def test_faithful_and_clean_are_never_pooled():
    """The companion table must be labelled and separate, never merged in."""
    comp = tab("word_error_by_length_faithful_companion.tsv")
    assert set(comp["population"]) == {"FAITHFUL_WFE_ALL"}
    clean_tab = tab("word_error_by_length_summary.tsv")
    assert "population" not in clean_tab.columns
    # the faithful real cell must contain more items than the clean real cell
    fr = comp[(comp.route == "ltm") & (comp.source_lexicality == "real")]
    cr = clean_tab[(clean_tab.route == "ltm")
                   & (clean_tab.source_lexicality == "real")]
    assert fr["n_items"].sum() == 800
    assert cr["n_items"].sum() == 671


# ------------------------------------------------ T2 faithful real audit

def test_error_events_are_exactly_the_word_error_rows(canon):
    real = canon[canon["in_FAITHFUL_WFE_ALL"]
                 & (canon["source_lexicality"] == "real")]
    expected = int(real["word_error"].sum())
    ev = tab("faithful_real_error_events.tsv")
    assert len(ev) == expected
    assert (ev["word_error"] == 1).all()
    assert (ev["exact_match"] == 0).all()
    assert set(ev["source_lexicality"]) == {"real"}
    assert not ev.duplicated(["seed", "route", "item_id"]).any()


def test_event_and_unique_item_counts_are_both_reported():
    s = tab("faithful_real_error_summary.tsv")
    for col in ("n_error_events_seed_x_item", "n_unique_erroneous_items",
                "event_error_rate", "unique_item_error_rate"):
        assert col in s.columns
    ev = tab("faithful_real_error_events.tsv")
    for _, r in s.iterrows():
        sub = ev[ev.route == r["route"]]
        assert len(sub) == r["n_error_events_seed_x_item"]
        assert sub["item_id"].nunique() == r["n_unique_erroneous_items"]
        assert r["n_unique_erroneous_items"] <= r["n_error_events_seed_x_item"]


def test_by_exposure_shares_sum_to_one_per_route():
    d = tab("faithful_real_error_by_exposure.tsv")
    for r in ROUTES:
        sub = d[(d.route == r) & (d["n_error_events"] > 0)]
        assert np.isclose(sub["share_of_route_error_events"].sum(), 1.0)


def test_recurrence_partitions_the_erroneous_items():
    rec = tab("faithful_real_error_recurrence.tsv")
    ev = tab("faithful_real_error_events.tsv")
    for r in ROUTES:
        assert (rec[rec.route == r]["n_items"].sum()
                == ev[ev.route == r]["item_id"].nunique())
    assert sorted(rec["n_seeds_with_error"].unique()) == [1, 2, 3, 4]


def test_error_events_carry_the_literal_source_columns(canon):
    ev = tab("faithful_real_error_events.tsv")
    for c in YC.ERROR_EVENT_COLUMNS:
        assert c in ev.columns, c
        assert c in canon.columns, c
    assert "eos_class" in ev.columns


def test_error_event_values_are_unmodified_copies(canon):
    ev = tab("faithful_real_error_events.tsv")
    key = ["seed", "route", "item_id"]
    m = ev.merge(canon, on=key, suffixes=("_ev", "_c"))
    assert len(m) == len(ev)
    for c in ("target", "prediction", "raw_edit_distance", "substitutions",
              "deletions", "insertions", "target_length", "zipf_frequency"):
        a, b = m[f"{c}_ev"], m[f"{c}_c"]
        if a.dtype.kind in "fc":
            assert np.allclose(a.fillna(-999), b.fillna(-999))
        else:
            assert (a.fillna("") == b.fillna("")).all()


# --------------------------------------------- T3 LTM pseudoword success

def test_success_groups_are_exhaustive_and_mutually_exclusive():
    items = tab("ltm_pseudoword_item_success.tsv")
    assert len(items) == 391
    assert items["item_id"].nunique() == 391
    assert set(items["success_group"]) <= set(YC.SUCCESS_GROUPS)
    g = tab("ltm_pseudoword_group_summary.tsv")
    assert set(g["success_group"]) == set(YC.SUCCESS_GROUPS)
    assert int(g["n_items"].sum()) == 391
    assert np.isclose(g["share_of_391"].sum(), 1.0)
    # each item is in exactly one group
    assert (items.groupby("item_id")["success_group"].nunique() == 1).all()


def test_success_group_matches_the_seed_counts():
    items = tab("ltm_pseudoword_item_success.tsv")
    for _, r in items.iterrows():
        n_ok = sum(int(r[f"seed{s}_exact_match"]) for s in SEEDS)
        assert n_ok == r["n_seeds_exact"]
        expected = (YC.ALWAYS_SUCCESS if n_ok == 4
                    else YC.ALWAYS_FAILED if n_ok == 0 else YC.MIXED)
        assert r["success_group"] == expected


def test_ltm_success_recomputes_from_the_canonical_table(canon):
    d = canon[(canon["lichtheim_exposure_status"] == "NOVEL_PSEUDOWORD")
              & (canon["route"] == "ltm")]
    ref = d.groupby("item_id")["exact_match"].sum()
    items = tab("ltm_pseudoword_item_success.tsv").set_index("item_id")
    assert (items["n_seeds_exact"] == ref.reindex(items.index)).all()


def test_gate_is_flagged_as_auxiliary_not_independent_evidence():
    f = tab("ltm_pseudoword_feature_summary.tsv")
    gate = f[f["feature"] == "mean_gate_auxiliary"]
    assert len(gate) == 3
    assert gate["note"].str.contains("not independent evidence").all()
    assert "auxiliary" in "".join(gate["note"].tolist())


def test_gate_really_is_a_monotone_function_of_confidence(canon):
    """The claim in the caption is verified, not asserted."""
    d = canon[canon["route"] == "full"].dropna(subset=["gate",
                                                       "lexical_confidence"])
    g = 1.0 / (1.0 + np.exp(-2.0 * (d["lexical_confidence"] - 0.7)))
    assert np.allclose(g, d["gate"], atol=1e-6)


def test_unavailable_measures_are_recorded_and_no_proxy_computed():
    u = tab("ltm_pseudoword_unavailable_measures.tsv")
    assert set(u["requested_measure"]) == {
        "phonotacticity", "distance_to_training_lexicon",
        "suffix_or_phonemic_complexity"}
    assert (u["status"] == YC.UNAVAILABLE).all()
    assert (~u["proxy_computed"]).all()
    # No feature standing in for an unavailable measure may appear.  Matched on
    # whole measure names and on the specific concepts, not on loose substrings:
    # `mean_raw_edit_distance_failed_seeds` is an error-severity field and is
    # not a distance-to-training-lexicon proxy.
    feats = set(tab("ltm_pseudoword_feature_summary.tsv")["feature"])
    forbidden = ("phonotact", "training_lexicon", "lexicon_distance",
                 "suffix", "phonemic_complexity")
    for m in u["requested_measure"]:
        assert m not in feats
    for f in feats:
        assert not any(tok in f for tok in forbidden), f


def test_no_lexicalization_claim_in_the_outputs():
    for name in os.listdir(FIGS):
        if not name.endswith("_caption.md"):
            continue
        with open(os.path.join(FIGS, name)) as f:
            text = f.read().lower()
        if "lexicaliz" in text or "lexicalis" in text:
            # any caption that uses the word at all must carry an explicit
            # disclaimer; both spellings and both wordings are accepted
            disclaimers = ("no lexicalization conclusion",
                           "no lexicalisation conclusion",
                           "no lexicalization claim is made",
                           "no lexicalisation claim is made")
            assert any(d in text for d in disclaimers), name


# ------------------------------------------- T4 faithful serial position

def test_figure2C_reproduction_gate_passed():
    c = tab("faithful_figure2C_reproduction_check.tsv")
    assert bool(c["reproduces_frozen_figure2C"].iloc[0])
    assert (c["max_abs_diff"] < 1e-12).all()
    assert (c["rows_only_in_frozen"] == 0).all()
    assert (c["rows_only_in_recomputed"] == 0).all()
    assert set(c["quantity"]) == {"relative_position", "n_items_x_seeds",
                                  "error_rate_per_item"}


def test_figure2C_full_rows_equal_the_frozen_table():
    frozen = pd.read_csv(os.path.join(
        ROOT, "outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/"
              "faithful_replication/faithful_figure2C_table.tsv"), sep="\t")
    mine = tab("faithful_figure2C_by_route.tsv")
    mine = mine[mine["route"] == "full"]
    m = frozen.merge(
        mine, left_on=["lexicality", "length", "position_1based"],
        right_on=["source_lexicality", "phoneme_length",
                  "position_index_1based"])
    assert len(m) == len(frozen) == 72
    assert np.allclose(m["error_rate_per_item_x"], m["error_rate_per_item_y"],
                       atol=1e-12)


def test_figure2C_by_route_covers_all_routes_and_lengths():
    d = tab("faithful_figure2C_by_route.tsv")
    assert sorted(d["route"].unique()) == sorted(ROUTES)
    assert set(d["source_lexicality"]) == {"real", "pseudo"}
    for r in ROUTES:
        for lex in ("real", "pseudo"):
            sub = d[(d.route == r) & (d.source_lexicality == lex)]
            assert sorted(sub["phoneme_length"].unique()) == LENGTHS
            for L in LENGTHS:
                assert len(sub[sub.phoneme_length == L]) == L


def test_figure2C_denominator_is_items_x_seeds(canon):
    d = tab("faithful_figure2C_by_route.tsv")
    faith = canon[canon["in_FAITHFUL_WFE_ALL"]]
    for _, r in d.sample(20, random_state=2).iterrows():
        n = len(faith[(faith.route == r["route"])
                      & (faith.source_lexicality == r["source_lexicality"])
                      & (faith.target_length == r["phoneme_length"])])
        assert n == r["n_items_x_seeds"]
        assert np.isclose(r["error_count"] / n, r["error_rate_per_item"])


def test_existing_faithful_figure2C_files_are_not_overwritten():
    """The new files must live under yc4_ and leave the originals alone."""
    orig = os.path.join(
        ROOT, "outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/"
              "faithful_replication")
    assert os.path.exists(os.path.join(orig, "faithful_figure2C_table.tsv"))
    for ext in ("png", "pdf", "svg"):
        assert os.path.exists(os.path.join(FIGS,
                              f"yc4_faithful_serial_position_by_route.{ext}"))
        assert not os.path.exists(os.path.join(FIGS,
                                  f"faithful_figure2C.{ext}"))


# ------------------------------------------------ T5 FI estimability

def test_non_estimable_fi_is_never_reported_as_zero_importance():
    d = tab("fi_route_estimability.tsv")
    bad = d[d["estimability_verdict"] == YC.NOT_ESTIMABLE]
    assert len(bad) > 0, "the audit must actually contain non-estimable cells"
    assert (~bad["importance_reported"]).all()
    assert (~bad["importance_is_zero_by_fiat"]).all()
    assert (~d["importance_is_zero_by_fiat"]).all()


def test_fi_verdict_follows_the_original_model_status():
    d = tab("fi_route_estimability.tsv")
    for _, r in d.iterrows():
        want = (YC.ESTIMABLE if r["original_model_status"] == "OK"
                else YC.NOT_ESTIMABLE)
        assert r["estimability_verdict"] == want


def test_fi_audit_matches_the_existing_validated_run():
    src = pd.read_csv(os.path.join(
        REPORT_ROOT, "feature_importance/route_specific/tables/"
                     "route_specific_model_fit.tsv"), sep="\t")
    d = tab("fi_route_estimability.tsv")
    assert len(d) == len(src)
    m = d.merge(src, on=["route", "seed"])
    assert (m["original_model_status"] == m["model_status"]).all()
    assert (m["n_test_nonzero_x"] == m["n_test_nonzero_y"]).all()


def test_negative_test_r2_is_preserved_not_clipped():
    d = tab("fi_route_estimability.tsv")
    neg = d[d["negative_test_r2"]]
    assert len(neg) > 0
    assert (neg["test_r2"] < 0).all()


def test_outcome_density_is_consistent_with_the_clean_table(clean):
    d = tab("fi_outcome_density.tsv")
    for _, r in d[d["population"] == "LICHTHEIM_CLEAN_all"].iterrows():
        c = clean[clean["route"] == r["route"]]
        v = c[r["outcome"]].to_numpy(float)
        assert len(v) == r["n_rows"]
        assert np.isclose((v > 0).mean(), r["nonzero_density"])


# ------------------------------------------------------- provenance rules

def test_modules_never_load_a_model_or_import_torch():
    for mod in ("yair_corrections.py", "plot_yair_corrections.py"):
        path = os.path.join(ROOT, "scripts/behavioral_analysis", mod)
        with open(path) as f:
            tree = ast.parse(f.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "torch" not in imported, mod
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        names = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        forbidden = {"load_model_and_vocab", "load_state_dict",
                     "autoregressive_decode_batch", "route_predictions"}
        assert not (calls | names) & forbidden, mod


def test_every_table_derives_only_from_frozen_sources():
    """Row counts must be reachable from the canonical table alone."""
    canon = load_canonical()
    assert len(canon) == 14400
    ev = tab("faithful_real_error_events.tsv")
    real = canon[canon["in_FAITHFUL_WFE_ALL"]
                 & (canon["source_lexicality"] == "real")]
    assert len(ev) == int(real["word_error"].sum())
    assert len(tab("ltm_pseudoword_item_success.tsv")) == 391


def test_spec_was_frozen_before_the_tables():
    spec = os.path.join(CTRL, "yair_corrections_spec.json")
    assert os.path.exists(spec)
    with open(spec) as f:
        s = json.load(f)
    assert s["frozen_before_results"] is True
    assert s["is_new_analysis_programme"] is False
    assert [t["id"] for t in s["tasks"]] == ["T1", "T2", "T3", "T4", "T5"]
    assert (os.path.getmtime(spec)
            <= os.path.getmtime(os.path.join(TAB,
                                             "word_error_by_length_summary.tsv")))


def test_no_scientific_value_is_hardcoded_in_the_compute_module():
    """Frozen structural counts may appear; scientific results may not."""
    path = os.path.join(ROOT, "scripts/behavioral_analysis/yair_corrections.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    code = ast.Module(body=[], type_ignores=[])
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]      # strip docstrings
    consts = {n.value for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and isinstance(n.value, float)}
    # the only float literals allowed are the gate parameters and tolerances
    assert consts <= {0.2, 2.0, 0.7, 1e-12, 0.0, 1.0}, consts
    ints = {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, int)
            and not isinstance(n.value, bool)}
    assert 800 in ints           # structural assertion, not a result
    assert 671 not in ints and 391 not in ints   # come from EXPECTED_CLEAN_COUNTS


def test_figures_and_captions_exist():
    for stem in ("yc1_word_error_by_length",
                 "yc2_faithful_real_error_composition",
                 "yc3_ltm_pseudoword_success",
                 "yc4_faithful_serial_position_by_route"):
        for ext in ("png", "pdf", "svg"):
            assert os.path.exists(os.path.join(FIGS, f"{stem}.{ext}")), stem
        assert os.path.exists(os.path.join(FIGS, f"{stem}_caption.md")), stem


# ============================= revision pass: polish, audits, diagram spec

AUDIT_SZ = os.path.join(YC.CORRECTIONS_ROOT, "stable_zero_audit")
AUDIT_AR = os.path.join(YC.CORRECTIONS_ROOT, "architecture_audit")


def test_trained_real_exact_ltm_error_table_is_complete(canon):
    """The 12-word table must be exactly the trained-exact LTM failures."""
    t = tab("trained_real_exact_ltm_errors.tsv")
    ref = canon[(canon["route"] == "ltm")
                & (canon["lichtheim_exposure_status"] == "TRAINED_REAL_EXACT")
                & (canon["word_error"] == 1)]
    assert len(t) == ref["item_id"].nunique()
    assert t["n_failing_seeds"].sum() == len(ref)
    assert set(t["item_id"]) == set(ref["item_id"])
    for c in ("word", "zipf_frequency", "phoneme_length", "target",
              "failing_seeds", "predictions_by_seed", "substitutions",
              "deletions", "insertions"):
        assert c in t.columns, c
    assert t["word"].notna().all()
    assert t["zipf_frequency"].notna().all()
    assert os.path.exists(os.path.join(TAB, "trained_real_exact_ltm_errors.md"))


def test_trained_real_error_table_values_are_unmodified(canon):
    t = tab("trained_real_exact_ltm_errors.tsv")
    ref = canon[(canon["route"] == "ltm") & (canon["word_error"] == 1)]
    for _, r in t.iterrows():
        g = ref[ref["item_id"] == r["item_id"]].sort_values("seed")
        assert r["failing_seeds"] == ",".join(str(int(s)) for s in g["seed"])
        assert r["target"] == g["target"].iloc[0]
        assert int(r["phoneme_length"]) == int(g["target_length"].iloc[0])
        assert np.isclose(r["zipf_frequency"], g["zipf_frequency"].iloc[0])


def test_no_frequency_model_is_fitted_for_the_trained_real_table():
    with open(os.path.join(TAB, "trained_real_exact_ltm_errors.md")) as f:
        md = f.read()
    assert "No frequency model is fitted" in md
    path = os.path.join(ROOT, "scripts/behavioral_analysis/yair_corrections.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    names = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not names & {"Ridge", "OLS", "ols", "polyfit", "lstsq", "linregress"}


def test_yc1_item_counts_are_printed_once_not_per_panel():
    """The count block must be attached to one axes only."""
    src = open(os.path.join(
        ROOT, "scripts/behavioral_analysis/plot_yair_corrections.py")).read()
    body = src[src.index("def figure_word_error_by_length"):
               src.index("def figure_faithful_real_errors")]
    block = body[body.index("n items per length bin"):]
    assert "ax0.annotate" in block
    # the per-length count loop must not run inside the per-route loop
    loop = body[body.index("for ax, r in zip(axes, ROUTES)"):
                body.index("n items per length bin")]
    assert "counts[" not in loop


def test_yc1_caption_documents_seeds_mean_and_band():
    with open(os.path.join(FIGS, "yc1_word_error_by_length_caption.md")) as f:
        c = f.read()
    for phrase in ("individual seeds", "across-seed mean", "bootstrap"):
        assert phrase in c, phrase
    assert "3" in c and "9" in c


def test_yc3_caption_states_it_does_not_identify_why():
    with open(os.path.join(FIGS, "yc3_ltm_pseudoword_success_caption.md")) as f:
        c = f.read()
    assert "does not identify why" in c
    assert "associated descriptors" in c
    assert "never injected into the decoder" in c
    assert "No lexicalization claim is made" in c


def test_yc4_wm_ltm_figure_exists_and_three_route_is_supplementary():
    for ext in ("png", "pdf", "svg"):
        assert os.path.exists(os.path.join(
            FIGS, f"yc4_faithful_serial_position_wm_ltm.{ext}"))
        assert os.path.exists(os.path.join(
            FIGS, f"yc4s_faithful_serial_position_all_routes.{ext}"))
    with open(os.path.join(
            FIGS, "yc4_faithful_serial_position_wm_ltm_caption.md")) as f:
        c = f.read()
    for phrase in ("Seeds are pooled", "Zip mismatch", "Levenshtein",
                   "Post-EOS re-padding"):
        assert phrase in c, phrase
    assert "8.8e-17" in c


# ---------------------------------------------------------- stable-zero audit

@pytest.mark.skipif(not os.path.isdir(AUDIT_SZ), reason="audit not run")
def test_stable_zero_streaks_recompute_from_the_trajectory():
    traj = pd.read_csv(os.path.join(AUDIT_SZ, "stable_zero_trajectory.tsv"),
                       sep="\t")
    streaks = pd.read_csv(os.path.join(AUDIT_SZ, "stable_zero_streaks.tsv"),
                          sep="\t")
    from scripts.audit_stable_zero import streaks_of_zeros
    for seed in sorted(traj["seed"].unique()):
        d = traj[traj["seed"] == seed].sort_values("epoch")
        ref = streaks_of_zeros(list(d["epoch"]), list(d["train_ar_errors_full"]))
        got = streaks[streaks["seed"] == seed]
        assert len(got) == len(ref)
        for r, (_, g) in zip(ref, got.iterrows()):
            assert r["start_epoch"] == g["first_checkpoint_of_streak"]
            assert r["length"] == g["length"]


@pytest.mark.skipif(not os.path.isdir(AUDIT_SZ), reason="audit not run")
def test_stable_zero_verdicts_are_internally_consistent():
    v = pd.read_csv(os.path.join(AUDIT_SZ, "stable_zero_verdicts.tsv"), sep="\t")
    assert sorted(v["seed"].unique()) == SEEDS
    assert sorted(v["X"].unique()) == [2, 3, 5]
    for _, r in v.iterrows():
        assert r["criterion_met"] == (r["longest_zero_streak"] >= r["X"])
        if r["criterion_met"]:
            assert r["selected_epoch"] <= r["stop_epoch_earliest_knowable"]
        else:
            assert pd.isna(r["selected_epoch"])
    # a stricter X can never be met where a looser one is not
    for seed in SEEDS:
        s = v[v["seed"] == seed].sort_values("X")
        met = s["criterion_met"].tolist()
        assert met == sorted(met, reverse=True)


@pytest.mark.skipif(not os.path.isdir(AUDIT_SZ), reason="audit not run")
def test_stable_zero_does_not_infer_missing_evaluations():
    v = pd.read_csv(os.path.join(AUDIT_SZ, "stable_zero_verdicts.tsv"), sep="\t")
    assert (~v["missing_evaluations_inferred"]).all()
    assert v["evaluation_grid_regular"].all()
    assert (v["n_evaluated_checkpoints"] == 20).all()
    with open(os.path.join(AUDIT_SZ, "stable_zero_audit.json")) as f:
        s = json.load(f)
    assert s["training_performed"] is False
    assert s["inference_performed"] is False
    assert s["checkpoint_loaded"] is False


@pytest.mark.skipif(not os.path.isdir(AUDIT_SZ), reason="audit not run")
def test_stable_zero_x2_reproduces_the_cohort_selection():
    c = pd.read_csv(os.path.join(AUDIT_SZ, "stable_zero_cross_check.tsv"),
                    sep="\t")
    for _, r in c.iterrows():
        if r["cohort_selection_reason"] == \
                "first_checkpoint_of_first_stable_zero_streak":
            assert r["agrees_with_cohort_selection"], r["seed"]


# --------------------------------------------------------- architecture audit

@pytest.mark.skipif(not os.path.isdir(AUDIT_AR), reason="audit not run")
def test_architecture_audit_matches_the_committed_code():
    with open(os.path.join(AUDIT_AR, "architecture_audit.json")) as f:
        a = json.load(f)
    code, ck = a["code_facts"], a["checkpoint_facts"]
    assert code["wm_premotor_projection_is_bare_linear"] is True
    assert code["ltm_premotor_projection_is_bare_linear"] is True
    assert code["motor_projection_is_bare_linear"] is True
    assert ck["cfg_ltm"]["ltm_encoder_mode"] == "unigru_last_hidden"
    assert ck["has_reverse_gru_parameters"] == []
    assert ck["parameter_shapes"]["motor.proj.weight"] == [42, 128]
    assert ck["parameter_shapes"]["wm.to_premotor.weight"] == [128, 128]
    assert ck["parameter_shapes"]["ltm.dec_to_premotor.weight"] == [128, 128]
    assert ck["model_constructed"] is False
    assert ck["forward_pass_run"] is False
    assert ck["weights_modified"] is False


@pytest.mark.skipif(not os.path.isdir(AUDIT_AR), reason="audit not run")
def test_architecture_audit_shapes_agree_with_the_live_modules():
    """Cross-check the recorded shapes against freshly constructed modules."""
    import torch.nn as nn
    with open(os.path.join(AUDIT_AR, "architecture_audit.json")) as f:
        ck = json.load(f)["checkpoint_facts"]
    lin = nn.Linear(ck["premotor_dim"], 42)
    assert list(lin.weight.shape) == ck["parameter_shapes"]["motor.proj.weight"]
    gru = nn.GRU(ck["cfg_ltm"]["phon_embed_dim"], ck["cfg_ltm"]["enc_hidden"],
                 batch_first=True, bidirectional=False)
    assert list(gru.weight_ih_l0.shape) == \
        ck["parameter_shapes"]["ltm.encoder.weight_ih_l0"]
    assert not [k for k in gru.state_dict() if "_reverse" in k]


def test_activation_checker_detects_a_wrapped_projection():
    """The bare-linear checker must be able to fail, or it proves nothing."""
    from scripts.audit_architecture import _returns_bare_projection
    src = ("import torch\n"
           "class A:\n    def f(self, x):\n        return {'p': self.p(x)}\n"
           "class B:\n    def f(self, x):\n        return torch.tanh(self.p(x))\n"
           "class C:\n    def f(self, x):\n        return self.p(x)\n"
           "class D:\n    def f(self, x):\n        return {'p': torch.relu(self.p(x))}\n")
    t = ast.parse(src)
    assert _returns_bare_projection(t, "A", "f", "p") is True
    assert _returns_bare_projection(t, "B", "f", "p") is False
    assert _returns_bare_projection(t, "C", "f", "p") is True
    assert _returns_bare_projection(t, "D", "f", "p") is False


@pytest.mark.skipif(not os.path.isdir(AUDIT_AR), reason="audit not run")
def test_architecture_note_answers_the_four_questions():
    with open(os.path.join(AUDIT_AR, "architecture_audit.md")) as f:
        m = f.read()
    for q in ("Is each premotor projection linear?",
              "Why is a common premotor space needed",
              "Is mixing before or after phoneme logits?",
              "What would break if the premotor projection were removed?"):
        assert q in m, q
    assert "No architecture was modified" in m


@pytest.mark.skipif(not os.path.isdir(AUDIT_AR), reason="audit not run")
def test_diagram_spec_replaces_the_stale_bigru_depiction():
    p = os.path.join(AUDIT_AR, "architecture_diagram_spec.md")
    assert os.path.exists(p)
    with open(p) as f:
        s = f.read()
    assert "bidirectional=False" in s
    assert "unigru_last_hidden" in s
    assert "no bank vector reaches the decoder" in s.lower() \
        or "no bank vector enters the decoder" in s.lower()
    assert "one scalar per word" in s
    assert "No diagram is rendered or committed" in s
    # the spec must not itself be a rendered diagram
    for ext in (".png", ".svg", ".pdf"):
        assert not os.path.exists(
            os.path.join(AUDIT_AR, f"architecture_diagram{ext}"))
