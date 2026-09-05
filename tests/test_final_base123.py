"""Acceptance tests for the FINAL base-123 high-capacity joint block.

The block's whole claim rests on one thing: within a seed pair, the ONLY
intentional difference between the H256 and H512 arms is ventral width.  These
tests pin that, pin the audited historical recipe the arms share, and pin the
new stopping rule and its resume behaviour.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    CANONICAL_BATCH_SIZE, CANONICAL_HIDDEN, CANONICAL_LOSS_WEIGHTS,
    CEILING_CONSECUTIVE_REQUIRED, EXPECTED_CANONICAL_C_HASH,
    EXPECTED_CANONICAL_C_N, FREE_AR_MAX_STEPS, INTERLEAVED_123, LAMBDA_C,
    LAMBDA_N, LR_BOUNDARY_STEPS, LR_STAGE1, LR_STAGE2, MACRO_CYCLE_STEPS,
    OPT_POLICY_SHARED, RATIO_123, TAU, JointScratchTrainer, at_ceiling,
    lr_for_step, main,
)

JOB = "scripts/cluster/jeanzay/final_base123_production.slurm"
BENCH = "scripts/cluster/jeanzay/final_base123_benchmark.slurm"

R_PASS, C_PASS = 463, 438

TINY = dict(regime="j0", device="cpu", max_words=400, batch_size=8,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, subset_mode="final_full",
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt",
            schedule=INTERLEAVED_123)

ARGS = ["--regime", "j0", "--subset-mode", "final_full", "--device", "cpu",
        "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
        "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "0",
        "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--no-subset-hash-check",
        "--schedule", "interleaved_123"]


def make(seed=22, wm=128, enc=128, dec=128, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(seed=seed, wm_hidden=wm, enc_hidden=enc,
                               dec_hidden=dec, **kw)


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


# =========================================  1. width independence  =========

def test_ventral_width_never_resizes_the_dorsal_route():
    a, b = make(enc=256, dec=256), make(enc=512, dec=512)
    for tr, w in ((a, 256), (b, 512)):
        assert tr.widths == {"wm_hidden": 128, "ltm_enc_hidden": w,
                             "ltm_dec_hidden": w}
        assert tr.cfg.wm.hidden == 128
        assert tr.model.wm.encoder.hidden_size == 128
        assert tr.model.wm.decoder.hidden_size == 128
        assert tr.model.ltm.encoder.hidden_size == w
        assert tr.model.ltm.decoder.hidden_size == w
        assert tr.model.ltm.sem_to_h0.out_features == w
        assert tr.model.ltm.dec_to_premotor.in_features == w
    ca, cb = a.parameter_census(), b.parameter_census()
    assert ca["wm_dorsal"] == cb["wm_dorsal"] == 165_504
    assert ca["shared_phon_embed"] == cb["shared_phon_embed"]
    assert ca["shared_motor"] == cb["shared_motor"]
    assert cb["ltm_ventral_total"] > ca["ltm_ventral_total"]


def test_there_is_no_single_hidden_size_knob():
    """A generic width flag could resize the dorsal route by accident."""
    import argparse
    from scripts.naming_comprehension import train_joint_scratch as m
    p = m.build_parser() if hasattr(m, "build_parser") else None
    text = open(os.path.join(
        ROOT, "scripts/naming_comprehension/train_joint_scratch.py"),
        encoding="utf-8").read()
    assert '"--wm-hidden"' in text and '"--enc-hidden"' in text \
        and '"--dec-hidden"' in text
    assert '"--hidden"' not in text and '"--hidden-size"' not in text


def test_defaults_reproduce_the_audited_canonical_architecture():
    tr = make()
    assert tr.widths == {"wm_hidden": CANONICAL_HIDDEN,
                         "ltm_enc_hidden": CANONICAL_HIDDEN,
                         "ltm_dec_hidden": CANONICAL_HIDDEN}


# ======================================  2. the audited historical recipe ==

def test_the_shared_recipe_is_the_final3p_one():
    """Everything the two arms hold in common, as resolved by the audit."""
    for enc, dec in ((256, 256), (512, 512)):
        tr = make(enc=enc, dec=dec)
        assert tr.optimizer_policy == OPT_POLICY_SHARED
        assert tr.task_optims is None, "exactly one shared AdamW"
        assert tr.schedule == INTERLEAVED_123 and tr.ratio == RATIO_123
        assert tr.c_align_weight == 0.0
        assert tr.dec_weight == CANONICAL_LOSS_WEIGHTS["dec"] == 0.5
        assert tr.cfg.loss.rep == 1.0 and tr.cfg.loss.align == 1.0
        assert tr.cfg.loss.wm == 0.5 and tr.cfg.loss.gate == 0.05
        assert tr.cfg.wm.interference_noise == 0.0
        assert tr.cfg.ltm.ventral_noise == 0.0
        assert tr.lr_policy["kind"] == "two_stage_rep_cursor"
    assert (LAMBDA_C, LAMBDA_N, TAU) == (0.087, 1.0, 0.10)
    assert (LR_STAGE1, LR_STAGE2) == (1e-3, 1e-4)
    assert CANONICAL_BATCH_SIZE == 64
    assert MACRO_CYCLE_STEPS == 6 and RATIO_123 == (1, 2, 3)


def test_exactly_one_optimizer_and_every_parameter_registered_once():
    tr = make(enc=512, dec=512)
    opt = tr.optimizer_for("repetition")
    assert opt is tr.optimizer_for("naming") is tr.optimizer_for("comprehension")
    seen = [p for g in opt.param_groups for p in g["params"]]
    ids = [id(p) for p in seen]
    assert len(ids) == len(set(ids)), "a parameter is registered twice"
    model_ids = {id(p) for p in tr.model.parameters()}
    assert set(ids) == model_ids, "optimizer scope != model parameters"


def test_lr_clock_is_the_repetition_cursor_not_the_global_step():
    """The publication-critical detail: the boundary is 100 R exposures."""
    assert LR_BOUNDARY_STEPS == 46_300 == 100 * R_PASS
    # the argument counts COMPLETED R batches, so with 46,299 done the batch
    # about to run is the 46,300th (still stage 1) and the next is stage 2
    assert lr_for_step(LR_BOUNDARY_STEPS - 1, LR_BOUNDARY_STEPS) == LR_STAGE1
    assert lr_for_step(LR_BOUNDARY_STEPS, LR_BOUNDARY_STEPS) == LR_STAGE2
    # under 1:2:3 the R clock means N and C are far ahead at the drop
    cycles = LR_BOUNDARY_STEPS                       # 1 R batch per cycle
    assert cycles * 2 / R_PASS == 200.0              # N exposures
    assert cycles * 3 / C_PASS == pytest.approx(317.1233, abs=1e-4)


def test_samplers_are_the_audited_ones():
    tr = make()
    assert tr.streams["repetition"].weights is not None, \
        "R is log-frequency weighted with replacement"
    for name in ("naming", "comprehension", "pool"):
        assert tr.streams[name].weights is None, f"{name} must be unweighted"
    assert tr.streams["repetition"].n == tr.streams["naming"].n


# ===========================================  3. 1:2:3 accounting  =========

def test_macro_cycle_holds_exactly_one_r_two_n_three_c():
    from collections import Counter
    tr = make()
    for c in range(50):
        cyc = [tr.task_for_step(c * 6 + k) for k in range(6)]
        assert Counter(cyc) == {"repetition": 1, "naming": 2,
                                "comprehension": 3}


def test_cursors_advance_in_the_ratio_and_pool_rides_r():
    tr = make()
    for _ in range(6 * 5):
        tr.train_step()
    c = tr.cursors
    assert (c["repetition"], c["naming"], c["comprehension"]) == (5, 10, 15)
    assert c["pool"] == c["repetition"], "the dorsal pool rides every R step"


def test_u_accounting_is_exact_and_c_is_not_exactly_3u():
    """C has 27,981 targets, so C exposures are 3u x (463/438), NOT 3u.
    The 1:2:3 is a presentation/update ratio, not an exposure ratio."""
    for u in (100, 500):
        cycles = u * R_PASS
        assert cycles * 1 / R_PASS == u
        assert cycles * 2 / R_PASS == 2 * u
        c_exp = cycles * 3 / C_PASS
        assert c_exp != 3 * u
        assert c_exp == pytest.approx(3 * u * R_PASS / C_PASS, abs=1e-6)
    assert 500 * R_PASS * MACRO_CYCLE_STEPS == 1_389_000
    assert 500 * R_PASS * 3 / C_PASS == pytest.approx(1585.6164, abs=1e-4)


# =============================================  4. stopping rule  ==========

def test_ceiling_predicate_requires_all_four_readouts():
    full = {"full_rep_full": 1.0, "full_rep_freear": 1.0,
            "full_naming_exact": 1.0, "full_comp_top1": 1.0}
    assert at_ceiling(full)
    for k in full:
        bad = dict(full); bad[k] = 0.999999
        assert not at_ceiling(bad), k
        missing = dict(full); missing.pop(k)
        assert not at_ceiling(missing), k
    assert CEILING_CONSECUTIVE_REQUIRED == 2


def test_streak_persists_across_resume(tmp_path):
    out = str(tmp_path / "runs")
    assert main(ARGS + ["--seed", "22", "--out-dir", out, "--run-id", "s",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    ck = torch.load(os.path.join(out, "s", "checkpoints", "step_00000012.pt"),
                    map_location="cpu", weights_only=False)
    assert ck["consecutive_ceiling"] == 0
    tr = make()
    tr.consecutive_ceiling = 1
    sd = tr.state_dict()
    assert sd["consecutive_ceiling"] == 1
    tr2 = make()
    tr2.load_state_dict(sd)
    assert tr2.consecutive_ceiling == 1, "a requeue must not reset the streak"


# ===================================  5. free-AR + route health  ===========

def test_free_ar_repetition_never_consults_target_length():
    tr = make()
    assert FREE_AR_MAX_STEPS == 12
    longest = max(len(e.phonemes) for e in tr.entries)
    assert FREE_AR_MAX_STEPS > longest + 1
    eos = tr.vocab.eos_id
    real = tr.model.motor

    class NoEos(torch.nn.Module):
        def forward(self, x):
            o = real(x)
            o[..., eos] = -1e9
            return o

    tr.model.motor = NoEos()
    try:
        r = tr.free_ar_repetition(list(range(64)), routes=("full",))
        assert r["full"] == 0.0, "no-EOS output must never count as exact"
    finally:
        tr.model.motor = real


def test_route_health_metrics_are_produced():
    tr = make()
    g = tr.gate_statistics(list(range(64)))
    for k in ("gate_mean", "gate_std", "gate_frac_below_0.05",
              "gate_frac_above_0.95"):
        assert k in g and 0.0 <= g[k] <= 1.0 or k == "gate_std"
    r = tr.free_ar_repetition(list(range(32)), routes=("full", "wm", "ltm"))
    assert set(r) == {"full", "wm", "ltm"}


# ==========================================  6. populations  ===============

def test_population_hashes_are_the_immutable_final_ones():
    tr = JointScratchTrainer(
        regime="j0", seed=22, device="cpu", max_words=30000,
        lexicon_path="data/lexicon_en_glove_covered.tsv",
        dorsal_pool_size=4000, batch_size=64, subset_mode="final_full",
        subset_per_band=822, subset_size=32, lr_boundary_steps=46300,
        allow_glove_fallback=False, require_subset_hash=True,
        schedule=INTERLEAVED_123, enc_hidden=256, dec_hidden=256)
    assert len(tr.entries) == 29_571
    assert len(tr.naming_idx) == 29_571
    assert len(tr.comp_idx) == EXPECTED_CANONICAL_C_N == 27_981
    assert tr.comp_hash == EXPECTED_CANONICAL_C_HASH
    assert tr.streams["repetition"].per_epoch == R_PASS
    assert tr.streams["comprehension"].per_epoch == C_PASS


# ======================================  7. resume / architecture guard ====

def test_resume_refuses_a_different_architecture(tmp_path):
    out = str(tmp_path / "runs")
    assert main(ARGS + ["--seed", "22", "--out-dir", out, "--run-id", "a",
                        "--enc-hidden", "64", "--dec-hidden", "64",
                        "--max-steps", "12", "--save-every", "12"]) == 0
    ck = torch.load(os.path.join(out, "a", "checkpoints", "step_00000012.pt"),
                    map_location="cpu", weights_only=False)
    with pytest.raises(RuntimeError, match="ARCHITECTURE MISMATCH"):
        make(enc=128, dec=128).load_state_dict(ck)


def test_split_run_equals_uninterrupted_run(tmp_path):
    out = str(tmp_path / "runs")
    common = ["--seed", "22", "--out-dir", out, "--enc-hidden", "64",
              "--dec-hidden", "64"]
    assert main(ARGS + common + ["--run-id", "whole", "--max-steps", "24",
                                 "--save-every", "24"]) == 0
    assert main(ARGS + common + ["--run-id", "split", "--max-steps", "12",
                                 "--save-every", "12"]) == 0
    assert main(ARGS + common + ["--run-id", "split", "--max-steps", "24",
                                 "--save-every", "24", "--resume",
                                 os.path.join(out, "split", "checkpoints",
                                              "step_00000012.pt")]) == 0
    a = torch.load(os.path.join(out, "whole", "checkpoints",
                                "step_00000024.pt"), map_location="cpu",
                   weights_only=False)
    b = torch.load(os.path.join(out, "split", "checkpoints",
                                "step_00000024.pt"), map_location="cpu",
                   weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert a["cursors"] == b["cursors"]
    assert a["widths"] == b["widths"]


# ==============================================  8. job contracts  =========

def test_production_job_defines_exactly_the_eight_runs():
    t = script()
    # run ids are composed from two explicit, index-aligned arrays; check the
    # mapping itself, and that all eight combinations are derivable
    assert "WIDTHS=(256 256 256 256 512 512 512 512)" in t
    assert "SEEDS=( 19  20  21  22  19  20  21  22)" in t
    assert 'RUN_ID="final_base123_h${VENTRAL}_s${SEED}"' in t
    widths = [256, 256, 256, 256, 512, 512, 512, 512]
    seeds = [19, 20, 21, 22, 19, 20, 21, 22]
    assert len(set(zip(widths, seeds))) == 8, "array mapping is not a bijection"
    assert sorted(set(widths)) == [256, 512]
    assert sorted(set(seeds)) == [19, 20, 21, 22]
    assert "--array=0-7" in t
    assert "IDX >= 0 && IDX < 8" in t
    assert "WM=128" in t
    assert "U_MAX=500" in t
    assert "MAX_STEPS=1389000" in t
    assert "FULL_EVAL_AT=" in t
    assert "--stop-at-ceiling" in t
    assert "--schedule interleaved_123" in t or 'SCHEDULE=interleaved_123' in t
    assert "#SBATCH --requeue" in t
    # no rescue mechanism may appear on an executable line
    ex = "\n".join(l for l in t.splitlines()
                   if l.strip() and not l.lstrip().startswith("#"))
    for forbidden in ("--optimizer-policy", "--dec-weight", "--lr-repetition",
                      "--lr-naming", "--lr-comprehension", "--c-align-weight",
                      "interleaved_223", "grouped_rn_c", "task_separated",
                      "--phase-transition", "--resume-from-parent"):
        assert forbidden not in ex, forbidden


def test_production_job_evaluates_on_the_u_grid():
    t = script()
    for u in (0, 25, 50, 100, 150, 200, 300, 400, 500):
        assert str(u * R_PASS * MACRO_CYCLE_STEPS) in t, f"u={u}"


def test_benchmark_job_exercises_the_real_driver_at_both_widths():
    t = script(BENCH)
    assert "train_joint_scratch.py" in t
    assert "WIDTHS=(256 512)" in t
    assert "--wm-hidden 128" in t
    assert '--enc-hidden "$V" --dec-hidden "$V"' in t
    assert "--array=0-1" in t
    assert "$JOBSCRATCH" in t, "the benchmark must not touch production dirs"


def test_earlier_jobs_are_untouched():
    for name, must in (("final3p_run.slurm", "SCHEDULE=interleaved_123"),
                       ("final9e_durability_r190_to_r470.slurm",
                        "FINAL_STEP=912110"),
                       ("cap3_route_capacity.slurm", "EXPECTED_C_N=27981")):
        assert must in script(f"scripts/cluster/jeanzay/{name}"), name
