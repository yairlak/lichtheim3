"""Tests for the joint-run analysis tools (FINAL-2 diagnostics).

Both tools are analysis-only: they read completed runs and never train.  The
tests pin the pure accounting helpers (exposures, slopes, clipping statistics
over LOGGED rows) and check that the gradient auditor takes no optimizer step
and leaves the model bit-identical.
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.analyze_joint_dynamics import (      # noqa: E402
    STEPS_PER_C_PASS, STEPS_PER_R_PASS, analyse, clip_stats, exposures,
    milestone_rows, series, slope_per_100_exposures,
)
from scripts.naming_comprehension.grad_interference_audit import (     # noqa: E402
    Auditor, cosine, group_of,
)
from scripts.naming_comprehension.train_joint_scratch import (         # noqa: E402
    FINAL_FULL_MODE, JointScratchTrainer,
)

FINAL1 = os.path.join(ROOT, "outputs", "joint_scratch",
                      "final1_j0_seed22_final_full")

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")


# ==============================================  dynamics: pure helpers  ====

def test_exposure_accounting_matches_the_canonical_populations():
    assert STEPS_PER_R_PASS == -(-29_571 // 64) == 463
    assert STEPS_PER_C_PASS == -(-27_981 // 64) == 438
    assert exposures(324_100, STEPS_PER_R_PASS) == pytest.approx(700.0)
    assert exposures(138_900, STEPS_PER_R_PASS) == pytest.approx(300.0)
    assert exposures(324_100, STEPS_PER_C_PASS) == pytest.approx(740.0, abs=0.5)


def test_slope_is_per_100_exposures_and_endpoint_based():
    pts = [(0, 0.0), (46_300, 0.10), (92_600, 0.15)]
    # 0 -> 46,300 steps is exactly 100 exposures, so the slope is the delta
    assert slope_per_100_exposures(pts, 0, 46_300) == pytest.approx(0.10)
    assert slope_per_100_exposures(pts, 46_300, 92_600) == pytest.approx(0.05)
    assert slope_per_100_exposures(pts, 0, 92_600) == pytest.approx(0.075)
    assert math.isnan(slope_per_100_exposures(pts, 0, 0))


def test_series_dedups_and_sorts_and_skips_nan():
    rows = [{"step": "20", "k": "0.2"}, {"step": "10", "k": "0.1"},
            {"step": "20", "k": "0.9"}, {"step": "30", "k": ""},
            {"step": "40", "k": "nan"}]
    assert series(rows, "k") == [(10, 0.1), (20, 0.2)]


def test_clip_stats_describe_logged_rows_only():
    rows = [{"step": str(i * 100), "grad_norm": g} for i, g in
            enumerate(["0.5", "2.0", "4.0", "1.0"])]
    s = clip_stats(rows, clip=1.0)
    assert s["n_logged"] == 4
    assert s["fraction_over_clip"] == 0.5          # 2.0 and 4.0 only
    assert s["max_pre_clip_norm"] == 4.0
    # scales min(1/g, 1) = [1, .5, .25, 1] -> sorted [.25, .5, 1, 1] -> 0.75
    assert s["median_update_scale"] == pytest.approx(0.75)
    assert "logged rows only" in s["note"]
    assert clip_stats([], 1.0)["n_logged"] == 0


def test_milestone_rows_are_those_with_full_population_columns():
    rows = [{"step": "1", "full_naming_exact": "nan", "full_comp_top1": "nan"},
            {"step": "2", "full_naming_exact": "0.05", "full_comp_top1": "0.04"}]
    assert [int(r["step"]) for r in milestone_rows(rows)] == [2]


# ===============================  dynamics: against the real FINAL-1 run  ===

@pytest.mark.skipif(not os.path.isdir(FINAL1), reason="FINAL-1 run absent")
def test_analyse_reproduces_the_reported_final1_milestones():
    a = analyse(FINAL1, every=46_300)
    assert a["c_align_weight"] in (0.0, None)      # baseline objective
    ms = {m["step"]: m for m in a["milestones"]}
    assert set(ms) >= {138_900, 231_500, 324_100}
    # exact values reported in the FINAL-2 diagnostic
    assert ms[231_500]["full_comp_top1"] == pytest.approx(0.05271434, abs=1e-7)
    assert ms[231_500]["full_naming_exact"] == pytest.approx(0.05806364, abs=1e-7)
    assert ms[231_500]["full_rep_ltm"] == pytest.approx(0.85330222, abs=1e-7)
    assert ms[324_100]["full_rep_full"] == pytest.approx(1.0)
    assert ms[324_100]["full_rep_errors"] == pytest.approx(0.0, abs=1e-3)
    # naming decelerates: later windows are strictly slower than stage 1
    sl = a["slopes"]["naming_exact"]
    assert sl["stage1 (LR 1e-3)"] > sl["231k-324k"]
    # pervasive clipping in the logged sample
    assert a["clipping"]["ALL"]["fraction_over_clip"] == pytest.approx(1.0)


# ==========================================  gradient auditor behaviour  ====

def test_group_of_covers_the_ventral_pathway():
    assert group_of("ltm.encoder.weight_ih_l0") == "ltm.encoder"
    assert group_of("ltm.to_semantic.0.weight") == "to_semantic"
    assert group_of("ltm.sem_to_h0.weight") == "sem_to_h0"
    assert group_of("ltm.dec_to_premotor.bias") == "dec_to_premotor"
    assert group_of("ltm.decoder.weight_hh_l0") == "ltm.decoder"
    assert group_of("phon_embed.weight") == "phon_embed"
    assert group_of("nothing.like.this") == "other"


def test_cosine_handles_zero_vectors():
    v = torch.tensor([1.0, 0.0])
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, -v) == pytest.approx(-1.0)
    assert math.isnan(cosine(v, torch.zeros(2)))


def test_auditor_takes_no_optimizer_step_and_leaves_weights_untouched():
    tr = JointScratchTrainer(regime="j0", seed=22, c_align_weight=1.0, **TINY)
    before = {k: v.detach().clone() for k, v in tr.model.state_dict().items()}
    opt_before = len(tr.optim.state)

    aud = Auditor(tr, n_batches=2)
    g_ret = aud.gradient("C_retrieval_raw")
    g_align = aud.gradient("C_align")
    norms = {k: float(aud.flat(g).norm()) for k, g in
             (("ret", g_ret), ("align", g_align))}
    assert norms["ret"] > 0 and norms["align"] > 0

    after = tr.model.state_dict()
    assert not [k for k in before if not torch.equal(before[k], after[k])], \
        "the auditor modified model weights"
    assert len(tr.optim.state) == opt_before, "the auditor stepped the optimizer"
    # C-side components must not reach the naming-only pathway
    for g in ("sem_to_h0", "ltm.decoder", "dec_to_premotor"):
        assert float(aud.flat(g_align, g).norm()) == 0.0
        assert float(aud.flat(g_ret, g).norm()) == 0.0


def test_auditor_batches_are_fixed_and_reproducible():
    a = Auditor(JointScratchTrainer(regime="j0", seed=22, **TINY), n_batches=3)
    b = Auditor(JointScratchTrainer(regime="j0", seed=22, **TINY), n_batches=3)
    assert a.idx == b.idx
    c = Auditor(JointScratchTrainer(regime="j0", seed=23, **TINY), n_batches=3)
    assert a.idx != c.idx


def test_summed_component_matches_the_trainer_objective():
    """The auditor's SUMMED scalar must equal what train_step optimizes."""
    for w in (0.0, 1.0):
        tr = JointScratchTrainer(regime="j0", seed=22, c_align_weight=w, **TINY)
        aud = Auditor(tr, n_batches=1)
        with torch.no_grad():
            summed = float(aud.components(0)["SUMMED"])
        rec = tr.train_step()          # consumes the same cursor-0 batches
        assert rec["joint_total"] == pytest.approx(summed, rel=1e-5)
