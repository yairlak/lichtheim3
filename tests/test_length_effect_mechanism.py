"""Focused tests for the M1/M2/M3/M5 instrumentation and analyses."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.length_effect_analysis import instrument as I
from scripts.length_effect_analysis import analyse_mechanism as A

OUT = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f")
INSTR = os.path.join(OUT, "instrumented")

requires_run = pytest.mark.skipif(
    not os.path.exists(os.path.join(INSTR, "item_summary.tsv")),
    reason="instrumented run not present")


def _read(p):
    return pd.read_csv(p, sep="\t")


# =================================================== instrumentation fidelity

@requires_run
def test_instrumentation_does_not_alter_canonical_predictions():
    eq = _read(os.path.join(INSTR, "canonical_equivalence.tsv"))
    assert len(eq) == 9600, "4 seeds x 1200 items x 2 instrumented routes"
    assert eq["prediction_match"].all(), "token differences found"
    assert eq["eos_match"].all(), "EOS position differences found"


@requires_run
def test_preflight_recorded_pass():
    p = os.path.join(INSTR, "preflight_equivalence.json")
    d = json.load(open(p))
    assert d["EXACT_TOKEN_EQUIVALENCE"] == "PASS"
    assert d["n_differences"] == 0
    assert set(d["lengths_covered"]) >= {3, 5, 7, 9}


# =========================================================== metric semantics

def test_target_rank_competition_ranking():
    lg = torch.tensor([3.0, 1.0, 5.0, 5.0, 0.0])
    r, tl, bn, mg = I._rank_margin(lg, torch.tensor(0))
    assert int(r) == 3            # two strictly greater logits -> rank 3
    assert float(tl) == 3.0
    assert float(bn) == 5.0
    assert float(mg) == -2.0
    r2, *_ = I._rank_margin(lg, torch.tensor(2))
    assert int(r2) == 1, "tied top logits must not inflate the rank"


def test_signed_target_margin_orientation():
    lg = torch.tensor([10.0, 2.0, 1.0])
    _, _, _, mg = I._rank_margin(lg, torch.tensor(0))
    assert float(mg) == 8.0 and float(mg) > 0
    _, _, _, mg2 = I._rank_margin(lg, torch.tensor(1))
    assert float(mg2) == -8.0


def test_entropy_natural_log_and_bounds():
    v = 42
    uniform = torch.zeros(v)
    assert abs(float(I._entropy(uniform)) - float(np.log(v))) < 1e-5
    peaked = torch.full((v,), -1e4)
    peaked[0] = 1e4
    assert float(I._entropy(peaked)) < 1e-6
    keep = torch.arange(3, v)
    assert abs(float(I._entropy(uniform, keep)) - float(np.log(v - 3))) < 1e-5


def test_edit_distance_matches_reference_cases():
    assert A.edit_distance(list("abc"), list("abc")) == 0
    assert A.edit_distance(list("abc"), list("abd")) == 1
    assert A.edit_distance(list("abc"), []) == 3
    assert A.edit_distance([], list("ab")) == 2
    assert A.edit_distance(list("kitten"), list("sitting")) == 3


# ============================================================ hazard semantics

@requires_run
def test_first_error_hazard_denominator():
    ev = _read(os.path.join(OUT, "m1_origin_propagation/first_error_events.tsv"))
    hz = _read(os.path.join(OUT, "m1_origin_propagation/first_error_hazard.tsv"))
    e = "FIRST_TOKEN_MISMATCH"
    sub = hz[(hz.event == e) & (hz.route == "ltm")
             & (hz.exposure_status == "NOVEL_PSEUDOWORD") & (hz.seed == 19)]
    src = ev[(ev.route == "ltm") & (ev.exposure_status == "NOVEL_PSEUDOWORD")
             & (ev.seed == 19)]
    for _, r in sub.iterrows():
        t = int(r.position)
        at_risk = src[(src.length > t) & (src[e].isna() | (src[e] >= t))]
        assert len(at_risk) == r.n_at_risk
        assert int((at_risk[e] == t).sum()) == r.n_events
        if r.n_at_risk:
            assert abs(r.hazard - r.n_events / r.n_at_risk) < 1e-12


@requires_run
def test_hazard_at_risk_never_exceeds_group_size():
    hz = _read(os.path.join(OUT, "m1_origin_propagation/first_error_hazard.tsv"))
    ev = _read(os.path.join(OUT, "m1_origin_propagation/first_error_events.tsv"))
    # `event` must be part of the grouping: the table stacks three event types,
    # so a diff without it would cross an event boundary and look like growth.
    for (event, seed, route, expo), g in hz.groupby(
            ["event", "seed", "route", "exposure_status"]):
        n = len(ev[(ev.seed == seed) & (ev.route == route)
                   & (ev.exposure_status == expo)])
        g = g.sort_values("position")
        assert g.n_at_risk.max() <= n
        assert (g.n_at_risk.diff().dropna() <= 0).all(), \
            f"risk set must not grow within {event}/{seed}/{route}/{expo}"


# ======================================================== prefix source labels

@requires_run
def test_prefix_source_labelling_is_exhaustive_and_consistent():
    ts = pd.read_csv(os.path.join(INSTR, "timestep_metrics.tsv"), sep="\t",
                     usecols=["route", "decode_mode", "prefix_source"])
    combos = set(map(tuple, ts.drop_duplicates().values))
    assert ("ltm", "gold_prefix", "gold") in combos
    assert ("wm", "gold_prefix", "gold") in combos
    assert ("full", "gold_prefix", "gold") in combos
    assert ("ltm", "autoregressive", "ltm_generated") in combos
    for r in ("wm", "ltm", "full"):
        assert (r, "autoregressive", "full_generated") in combos
    # a WM-only AR stream was deliberately NOT run
    assert ("wm", "autoregressive", "wm_generated") not in combos
    assert set(ts.prefix_source.unique()) == {"gold", "ltm_generated",
                                              "full_generated"}


@requires_run
def test_common_prefix_route_logits_share_one_prefix():
    """WM, LTM and FULL rows under full_generated must align 1:1 per position."""
    ts = pd.read_csv(os.path.join(INSTR, "timestep_metrics.tsv"), sep="\t",
                     usecols=["seed", "item_id", "route", "prefix_source",
                              "timestep", "target_token"])
    fp = ts[ts.prefix_source == "full_generated"]
    c = fp.groupby(["seed", "item_id", "timestep"]).route.nunique()
    assert (c == 3).all(), "each FULL-prefix position must carry all three routes"
    t = fp.groupby(["seed", "item_id", "timestep"]).target_token.nunique()
    assert (t == 1).all(), "the target token must agree across routes"


@requires_run
def test_wm_only_ar_stream_not_run():
    man = json.load(open(os.path.join(INSTR, "run_manifest.json")))
    assert man["wm_only_ar_stream_run"] is False
    assert man["streams"] == ["A_gold_prefix", "B_ltm_ar", "C_full_ar"]


# ============================================================== top-k ordering

@requires_run
def test_topk_deterministic_and_sorted():
    nb = _read(os.path.join(INSTR, "lexical_neighbors.tsv"))
    g = nb.groupby(["seed", "item_id"])
    assert (g["rank"].max() == 20).all()
    assert (g.size() == 20).all()
    for _, sub in list(g)[:200]:
        s = sub.sort_values("rank")
        assert (s.cosine.diff().dropna() <= 1e-9).all(), "cosine must be non-increasing"
        ties = s[s.cosine.duplicated(keep=False)]
        for _, tg in ties.groupby("cosine"):
            assert (tg.bank_row.diff().dropna() > 0).all(), \
                "cosine ties must be ordered by ascending bank row"


@requires_run
def test_top1_matches_item_summary():
    nb = _read(os.path.join(INSTR, "lexical_neighbors.tsv"))
    it = _read(os.path.join(INSTR, "item_summary.tsv"))
    t1 = nb[nb["rank"] == 1].set_index(["seed", "item_id"])
    for r in it.sample(200, random_state=0).itertuples():
        row = t1.loc[(r.seed, r.item_id)]
        assert row.bank_row == r.top1_neighbor_id
        assert abs(row.cosine - r.top1_similarity) < 1e-6


# ================================================ homophones / duplicates

def test_bank_audit_homophone_handling():
    p = os.path.join(OUT, "m3_lexical_attraction/bank_structure_audit.json")
    if not os.path.exists(p):
        pytest.skip("bank audit absent")
    d = json.load(open(p))
    assert d["bank_rows"] == 29571
    assert d["duplicate_orthographic_entries"] == 0
    assert d["unique_phonological_forms"] < d["bank_rows"], \
        "homophones must reduce the count of distinct phonological forms"
    assert d["multiple_bank_entries_can_share_one_phonological_form"] is True
    assert d["matches_checkpoint_hash"] is True
    assert d["model_forward_called"] is False


@requires_run
def test_attraction_categories_deterministic_and_exclusive():
    d = _read(os.path.join(OUT, "m3_lexical_attraction/lexical_attraction_items.tsv"))
    allowed = {"COMPLETE_TRAINING_WORD_LEXICALIZATION", "TOP1_ATTRACTION",
               "TOPK_ATTRACTION", "PARTIAL_ATTRACTION",
               "NO_DETECTED_LEXICAL_ATTRACTION"}
    assert set(d.attraction_category.unique()) <= allowed
    assert len(d) == d.groupby(["seed", "item_id"]).ngroups, "one row per item-seed"
    # a correct item can never be an attraction category
    assert (d[d.correct == 1].attraction_category
            == "NO_DETECTED_LEXICAL_ATTRACTION").all()


# ================================================= route rescue categories

@requires_run
def test_word_level_route_categories_partition_the_cohort():
    wl = _read(os.path.join(OUT, "m5_dorsal_rescue/word_level_route_outcomes.tsv"))
    for seed, g in wl.groupby("seed"):
        assert g.n.sum() == 1200, f"seed {seed} categories must partition 1,200 items"
    allowed = {"BOTH_ROUTES_CORRECT", "BOTH_ROUTES_WRONG",
               "WM_CORRECT_LTM_WRONG_FULL_CORRECT", "WM_CORRECT_LTM_WRONG_FULL_WRONG",
               "WM_WRONG_LTM_CORRECT_FULL_CORRECT", "WM_WRONG_LTM_CORRECT_FULL_WRONG"}
    assert set(wl.route_outcome_category.unique()) <= allowed


@requires_run
def test_position_level_categories_partition_positions():
    pl = _read(os.path.join(OUT, "m5_dorsal_rescue/position_level_rescue_summary.tsv"))
    p = _read(os.path.join(OUT, "m5_dorsal_rescue/position_level_common_prefix.tsv"))
    assert pl.n.sum() == len(p)
    allowed = {"BOTH_LOCAL_CORRECT", "BOTH_LOCAL_WRONG_FULL_CORRECT",
               "BOTH_LOCAL_WRONG_FULL_WRONG",
               "LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_CORRECT",
               "LTM_LOCAL_WRONG_WM_LOCAL_CORRECT_FULL_WRONG",
               "LTM_LOCAL_CORRECT_WM_LOCAL_WRONG_FULL_CORRECT",
               "LTM_LOCAL_CORRECT_WM_LOCAL_WRONG_FULL_WRONG"}
    assert set(pl.position_rescue_category.unique()) <= allowed


# ==================================================== schema and provenance

@requires_run
def test_schema_and_provenance_complete():
    prov = json.load(open(os.path.join(INSTR, "provenance.json")))
    for k in ("checkpoint_training_commit", "behavioral_evaluation_commit",
              "mechanism_analysis_commit_or_dirty_state", "checkpoints",
              "dataset_hashes", "lexicon_hashes", "vocabulary_hash",
              "architecture_configuration", "decode_mode", "prefix_modes",
              "top_k", "noise_settings", "readout_mode",
              "instrumented_script_sha256", "metric_conventions"):
        assert k in prov, k
    assert prov["checkpoint_training_commit"] == "93a577fd9822955fa272ee733fa7e2acf81f1333"
    assert prov["behavioral_evaluation_commit"] == "e876b755d0475ed11e5fbc0419a0bd8860dfd325"
    assert prov["architecture_configuration"]["gate_alpha"] == 2.0
    assert prov["architecture_configuration"]["gate_threshold"] == 0.7
    assert prov["architecture_configuration"]["ltm_encoder_mode"] == "unigru_last_hidden"
    assert prov["noise_settings"]["apply_noise"] is False
    assert prov["training_performed"] is False
    assert prov["weights_modified"] is False
    assert prov["architecture_changed"] is False
    assert prov["canonical_equivalence"] == "PASS"
    assert len(prov["checkpoints"]) == 4, "seed 21 must not be excluded"
    assert "21" in prov["checkpoints"]


@requires_run
def test_item_summary_covers_all_seeds_and_items():
    it = _read(os.path.join(INSTR, "item_summary.tsv"))
    assert len(it) == 4800
    assert sorted(it.seed.unique()) == [19, 20, 21, 22]
    assert it.groupby("seed").item_id.nunique().eq(1200).all()


@requires_run
def test_figures_have_backing_tsv_and_three_formats():
    fd = os.path.join(OUT, "figures")
    for stem in ("figure1_origin_propagation", "figure2_attraction_rescue"):
        for ext in ("png", "pdf", "svg"):
            p = os.path.join(fd, f"{stem}.{ext}")
            assert os.path.exists(p) and os.path.getsize(p) > 0, p
        assert os.path.getsize(os.path.join(fd, f"{stem}_caption.md")) > 0
        t = _read(os.path.join(fd, f"{stem}.tsv"))
        assert len(t) > 0 and "panel" in t.columns
