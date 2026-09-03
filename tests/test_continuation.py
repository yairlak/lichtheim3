"""Acceptance tests for an exact CONTINUATION of a joint-scratch trajectory.

FINAL-3P is carried from R110 to R150 by resuming step_00305580.pt in place,
with every scientific field unchanged.  The only fields that differ between
launches are run-control: the final budget and the future milestones.

These tests pin that distinction:
  * continuing with an EXTENDED budget is bitwise identical to one
    uninterrupted run of the same recipe (weights, AdamW moments, cursors,
    macro-cycle position, RNG);
  * budget and milestone changes are accepted as continuation metadata;
  * any SCIENTIFIC mismatch is still refused;
  * earlier history (metrics.tsv, losses.tsv, the first launch's config and
    provenance) is preserved, not overwritten;
  * the R130 / R150 milestone accounting, and that the continuation runs
    wholly at the post-boundary learning rate.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    FINAL_FULL_MODE, INTERLEAVED_123, LR_STAGE2, MACRO_CYCLE_STEPS,
    RATIO_123, JointScratchTrainer, lr_for_step, main,
)

SMOKE = ["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
         "--schedule", "interleaved_123", "--device", "cpu",
         "--max-words", "400", "--batch-size", "8", "--dorsal-pool-size", "32",
         "--lr-boundary-steps", "6", "--eval-every", "0", "--log-every", "0",
         "--glove-path", "tests/_no_such_glove_file.txt",
         "--allow-glove-fallback", "--no-subset-hash-check"]


def run_main(out, run_id, *, epochs_steps, save_every, resume=None,
             full_eval_at=None):
    """One launch of the driver, budgeted in raw steps (smoke-sized)."""
    argv = SMOKE + ["--out-dir", out, "--run-id", run_id,
                    "--max-steps", str(epochs_steps),
                    "--save-every", str(save_every)]
    if full_eval_at:
        argv += ["--full-eval-at", full_eval_at]
    if resume:
        argv += ["--resume", resume]
    assert main(argv) == 0


def model_of(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def ckpt_path(out, run_id, step):
    return os.path.join(out, run_id, "checkpoints", f"step_{step:08d}.pt")


# =========================================================  M4 / M5 maths  ==

def test_continuation_milestone_accounting():
    """R130 -> 361,140 / N260 / C~412.26 and R150 -> 416,700 / N300 / C~475.68."""
    r_pass, c_pass = 463, 438
    r, n, c = RATIO_123
    expected = {                     # R exp -> (cycles, steps, N exp, C exp)
        110: (50_930, 305_580, 220.0, 348.8356),      # M3, the start point
        130: (60_190, 361_140, 260.0, 412.2603),      # M4
        150: (69_450, 416_700, 300.0, 475.6849),      # M5
    }
    for r_exp, (cycles, steps, n_exp, c_exp) in expected.items():
        assert r_exp * r_pass == cycles
        assert cycles * MACRO_CYCLE_STEPS == steps
        assert cycles * r / r_pass == pytest.approx(r_exp)
        assert cycles * n / r_pass == pytest.approx(n_exp)
        assert cycles * c / c_pass == pytest.approx(c_exp, abs=1e-3)
    # the continuation adds 111,120 steps beyond M3
    assert expected[150][1] - expected[110][1] == 111_120


def test_continuation_is_entirely_after_the_lr_boundary():
    boundary = 46_300                    # 100 R exposures
    for r_exp in (110, 130, 150):
        assert lr_for_step(r_exp * 463, boundary) == LR_STAGE2


def test_continuation_slurm_job_declares_the_extension():
    path = os.path.join(ROOT, "scripts", "cluster", "jeanzay",
                        "final3p_continue_r150.slurm")
    text = open(path, encoding="utf-8").read()
    assert "EPOCHS=150" in text
    assert "FULL_EVAL_AT=361140,416700" in text
    assert "M3_STEP=305580" in text and "M5_STEP=416700" in text
    # same run directory, same recipe, never a restart
    assert 'RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"' in text
    for frozen in ("SCHEDULE=interleaved_123", "SEED=22",
                   "SUBSET_MODE=final_full", "--resume"):
        assert frozen in text
    for forbidden in ("--c-align-weight", "--lr-boundary-steps",
                      "--batch-size", "--allow-glove-fallback"):
        assert forbidden not in text
    # requeue must stay possible: the guard distinguishes first launch from
    # continuation-after-preemption rather than demanding an exact step
    assert "LATEST_STEP == M3_STEP" in text and "LATEST_STEP < M5_STEP" in text


# ============================  continuation == uninterrupted (the key one) ==

def test_continuation_with_extended_budget_is_bitwise_identical(tmp_path):
    """Stop at a smaller budget, relaunch with a LARGER budget from the
    checkpoint, and land bitwise where one uninterrupted run would."""
    out = str(tmp_path / "runs")
    # (a) uninterrupted: one launch straight to 18 steps
    run_main(out, "whole", epochs_steps=18, save_every=18)
    # (b) stopped at 7 (mid-cycle), then continued with a bigger budget
    run_main(out, "split", epochs_steps=7, save_every=7)
    run_main(out, "split", epochs_steps=18, save_every=18,
             resume=ckpt_path(out, "split", 7))

    a = model_of(ckpt_path(out, "whole", 18))
    b = model_of(ckpt_path(out, "split", 18))

    sa, sb = a["model_state_dict"], b["model_state_dict"]
    bad = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not bad, f"continuation diverged on {len(bad)} tensors: {bad[:4]}"

    # optimizer moments (exp_avg / exp_avg_sq) and step counters preserved
    oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    assert set(oa) == set(ob) and oa, "no optimizer state was carried over"
    for pid in oa:
        for key, va in oa[pid].items():
            vb = ob[pid][key]
            if torch.is_tensor(va):
                assert torch.equal(va, vb), f"moment {key} differs for param {pid}"
            else:
                assert va == vb, f"optimizer scalar {key} differs"

    # cursors, macro-cycle position and RNG
    assert a["cursors"] == b["cursors"]
    assert a["global_step"] == b["global_step"] == 18
    assert a["global_step"] % MACRO_CYCLE_STEPS == b["global_step"] % MACRO_CYCLE_STEPS
    assert torch.equal(a["rng_states"]["torch"], b["rng_states"]["torch"])


def test_continuation_preserves_earlier_history(tmp_path):
    """metrics.tsv / losses.tsv accumulate; the first launch's config and
    provenance are not overwritten by the continuation."""
    out = str(tmp_path / "runs")
    run_main(out, "hist", epochs_steps=6, save_every=6, full_eval_at="6")
    run_dir = os.path.join(out, "hist")
    first_metrics = open(os.path.join(run_dir, "metrics.tsv")).read()
    first_cfg = json.load(open(os.path.join(run_dir, "config.json")))
    first_prov = json.load(open(os.path.join(run_dir, "provenance.json")))
    assert first_cfg["total_steps"] == 6

    run_main(out, "hist", epochs_steps=12, save_every=12, full_eval_at="12",
             resume=ckpt_path(out, "hist", 6))

    # earlier rows still present, in order, with new ones appended
    now_metrics = open(os.path.join(run_dir, "metrics.tsv")).read()
    assert now_metrics.startswith(first_metrics)
    rows = list(csv.DictReader(open(os.path.join(run_dir, "metrics.tsv")),
                               delimiter="\t"))
    steps = [int(r["step"]) for r in rows]
    assert 6 in steps and 12 in steps
    # exactly one header line survived the second launch
    assert open(os.path.join(run_dir, "metrics.tsv")).read().count("rep_epoch") == 1

    # the first launch's records are untouched...
    assert json.load(open(os.path.join(run_dir, "config.json"))) == first_cfg
    assert json.load(open(os.path.join(run_dir, "provenance.json"))) == first_prov
    # ... and the continuation wrote its own, recording where it resumed
    cont_cfg = os.path.join(run_dir, "config_from_step_00000006.json")
    cont_prov = os.path.join(run_dir, "provenance_from_step_00000006.json")
    assert os.path.exists(cont_cfg) and os.path.exists(cont_prov)
    assert json.load(open(cont_cfg))["total_steps"] == 12
    p = json.load(open(cont_prov))
    assert p["resumed_at_step"] == 6 and p["resumed_from"]
    assert p["budget_this_launch"]["total_steps"] == 12
    assert p["resume_provenance"], "resume event not recorded"


def test_passed_milestones_do_not_refire_on_continuation(tmp_path):
    """A milestone already crossed cannot be evaluated twice: the driver only
    fires on an exact step match and global_step never decreases."""
    out = str(tmp_path / "runs")
    run_main(out, "mile", epochs_steps=6, save_every=6, full_eval_at="6")
    rows = list(csv.DictReader(open(os.path.join(out, "mile", "metrics.tsv")),
                               delimiter="\t"))
    assert sum(1 for r in rows if int(r["step"]) == 6) >= 1
    before = sum(1 for r in rows if int(r["step"]) == 6)

    run_main(out, "mile", epochs_steps=12, save_every=12,
             full_eval_at="6,12", resume=ckpt_path(out, "mile", 6))
    rows = list(csv.DictReader(open(os.path.join(out, "mile", "metrics.tsv")),
                               delimiter="\t"))
    assert sum(1 for r in rows if int(r["step"]) == 6) == before, \
        "a passed milestone re-fired after resume"


# ======================================  guards: recipe vs run-control  =====

def _tiny(**over):
    kw = dict(device="cpu", max_words=400,
              lexicon_path="data/lexicon_en_glove_covered.tsv",
              dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
              subset_per_band=822, subset_size=32, lr_boundary_steps=6,
              allow_glove_fallback=True, require_subset_hash=False,
              glove_path="tests/_no_such_glove_file.txt",
              schedule=INTERLEAVED_123)
    kw.update(over)
    return JointScratchTrainer(regime=kw.pop("regime", "j0"),
                               seed=kw.pop("seed", 22), **kw)


def test_budget_and_milestones_are_not_scientific_fields(tmp_path):
    """The resume guard must NOT police run-control: continuing with a bigger
    budget and new milestones is exactly what a continuation is."""
    a = _tiny()
    a.train_step()
    p = tmp_path / "c.pt"
    torch.save(a.state_dict(), str(p))
    ck = torch.load(str(p), weights_only=False)
    # nothing budget-like is compared by the guard
    _tiny().load_state_dict(ck, source="t")          # must not raise
    assert "total_steps" not in ck and "full_eval_at" not in ck
    # the LR boundary IS scientific and is restored FROM the checkpoint,
    # so a continuation cannot move it via the CLI
    other = _tiny(lr_boundary_steps=999_999)
    other.load_state_dict(ck, source="t")
    assert other.lr_boundary_steps == a.lr_boundary_steps == 6


def test_scientific_mismatch_still_refuses_resume(tmp_path):
    a = _tiny()
    a.train_step()
    p = tmp_path / "s.pt"
    torch.save(a.state_dict(), str(p))
    ck = torch.load(str(p), weights_only=False)
    for kw, match in ((dict(schedule="summed"), "schedule"),
                      (dict(c_align_weight=1.0), "c_align_weight"),
                      (dict(seed=23), "seed")):
        with pytest.raises(RuntimeError, match=match):
            _tiny(**kw).load_state_dict(ck, source="t")
