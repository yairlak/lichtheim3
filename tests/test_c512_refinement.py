"""Tests for the C512 refinement: LR branching and the top-1 error audit.

The C width probe imposed lr 1e-3 -> 1e-4 at exposure 100 by fiat, copied
from the naming result.  For C there is no evidence 1e-3 was unstable, so
two branches reopen that choice.  What must be guaranteed: the branch changes
ONLY the stage-2 LR, preserves AdamW moments/RNG/cursor exactly, refuses an
undeclared change, and leaves the default width-probe schedule untouched.
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

from scripts.naming_comprehension.c_error_audit import (                 # noqa: E402
    COLUMNS, MARGIN_EPS, NEIGHBOUR_COS, main as audit_main,
)
from scripts.naming_comprehension.route_capacity_probe import (          # noqa: E402
    LR_BOUNDARY_EXPOSURES, LR_STAGE1, LR_STAGE2, ROUTE_COMPREHENSION,
    RouteCapacityTrainer, lr_for_exposure, main,
)

JOB = "scripts/cluster/jeanzay/cap4_c512_lr_branch.slurm"

TINY = dict(seed=22, device="cpu", max_words=400, batch_size=8,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            glove_path="tests/_no_such_glove_file.txt",
            allow_glove_fallback=True, require_population_hash=False)

ARGS = ["--seed", "22", "--device", "cpu", "--max-words", "400",
        "--batch-size", "8", "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--no-population-hash-check",
        "--log-every", "0"]


def make(enc=64, **over):
    kw = dict(TINY)
    kw.update(over)
    return RouteCapacityTrainer(route=ROUTE_COMPREHENSION, wm_hidden=128,
                                enc_hidden=enc, dec_hidden=128, **kw)


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


# ==========================================  the stage-2 LR branch  ========

def test_default_schedule_is_the_untouched_width_probe_one():
    """The completed C128/C256/C512 runs must remain reproducible."""
    assert lr_for_exposure(0) == lr_for_exposure(99.9) == LR_STAGE1 == 1e-3
    assert lr_for_exposure(100) == lr_for_exposure(3000) == LR_STAGE2 == 1e-4
    assert make().lr_stage2 == LR_STAGE2


@pytest.mark.parametrize("stage2", [3e-4, 1e-3])
def test_branch_applies_only_after_the_boundary(stage2):
    tr = make(lr_stage2=stage2)
    assert lr_for_exposure(99.9, stage2) == LR_STAGE1, \
        "stage 1 must be identical in every branch"
    assert lr_for_exposure(LR_BOUNDARY_EXPOSURES, stage2) == stage2
    assert tr.lr_stage2 == stage2


def test_undeclared_lr_change_is_refused_and_declared_one_is_recorded(tmp_path):
    out = str(tmp_path / "runs")
    assert main(["--route", "comprehension", "--enc-hidden", "64",
                 "--out-dir", out, "--run-id", "src",
                 "--eval-exposures", "1", "--max-exposures", "1"] + ARGS) == 0
    src = os.path.join(out, "src", "checkpoints", "step_00000049.pt")
    ck = torch.load(src, map_location="cpu", weights_only=False)

    make(lr_stage2=LR_STAGE2).load_state_dict(dict(ck))       # same: no flag
    with pytest.raises(RuntimeError, match="LR TRANSITION REFUSED"):
        make(lr_stage2=3e-4).load_state_dict(dict(ck))

    tr = make(lr_stage2=3e-4)
    tr.load_state_dict(dict(ck), allow_lr_transition=True)
    assert tr.lr_transitions == [{"from_stage2_lr": LR_STAGE2,
                                  "to_stage2_lr": 3e-4, "at_step": 49,
                                  "at_exposures": 1.0,
                                  "optimizer_moments": "preserved"}]
    assert tr.state_dict()["lr_stage2"] == 3e-4


def test_branch_preserves_moments_rng_and_cursor_bitwise(tmp_path):
    """The scientific requirement: only the LR may differ."""
    out = str(tmp_path / "runs")
    assert main(["--route", "comprehension", "--enc-hidden", "64",
                 "--out-dir", out, "--run-id", "src",
                 "--eval-exposures", "1", "--max-exposures", "1"] + ARGS) == 0
    src = os.path.join(out, "src", "checkpoints", "step_00000049.pt")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    moments = {i: st["exp_avg"].clone()
               for i, st in ck["optimizer_state_dict"]["state"].items()}

    tr = make(lr_stage2=1e-3)
    tr.load_state_dict(dict(ck), allow_lr_transition=True)
    assert tr.global_step == 49, "cursor must carry over"
    for i, ref in moments.items():
        assert torch.equal(tr.optim.state_dict()["state"][i]["exp_avg"], ref), \
            "AdamW moments must be preserved bitwise"
    sd = tr.model.state_dict()
    assert not [k for k in sd if not torch.equal(sd[k],
                                                 ck["model_state_dict"][k])]
    assert torch.equal(torch.get_rng_state(), ck["rng_states"]["torch"])


def test_two_branches_from_one_source_differ_only_in_lr(tmp_path):
    out = str(tmp_path / "runs")
    assert main(["--route", "comprehension", "--enc-hidden", "64",
                 "--out-dir", out, "--run-id", "src",
                 "--eval-exposures", "1", "--max-exposures", "1"] + ARGS) == 0
    src = os.path.join(out, "src", "checkpoints", "step_00000049.pt")
    for rid, lr in (("a", "3e-4"), ("b", "1e-3")):
        assert main(["--route", "comprehension", "--enc-hidden", "64",
                     "--out-dir", out, "--run-id", rid,
                     "--eval-exposures", "2", "--max-exposures", "2",
                     "--resume", src, "--lr-stage2", lr,
                     "--lr-transition"] + ARGS) == 0
    a = json.load(open(os.path.join(out, "a", "config.json")))
    b = json.load(open(os.path.join(out, "b", "config.json")))
    assert a["lr_stage2"] == 3e-4 and b["lr_stage2"] == 1e-3
    assert a["parent_checkpoint_sha256"] == b["parent_checkpoint_sha256"]
    for k in ("widths", "population_n", "population_sha256", "tau", "loss",
              "batch_size", "weight_decay", "grad_clip", "trainable_scope",
              "sampler", "retrieval_bank_n", "lr_stage1"):
        assert a[k] == b[k], k
    # BEFORE the boundary the branches are identical BY DESIGN -- stage 1 is
    # shared -- so identical weights here is the correct behaviour, and is
    # itself worth pinning: the branch must not leak into stage 1.
    wa = torch.load(os.path.join(out, "a", "checkpoints", "step_00000098.pt"),
                    map_location="cpu", weights_only=False)["model_state_dict"]
    wb = torch.load(os.path.join(out, "b", "checkpoints", "step_00000098.pt"),
                    map_location="cpu", weights_only=False)["model_state_dict"]
    assert not [k for k in wa if not torch.equal(wa[k], wb[k])], \
        "the stage-2 LR must not affect any step before exposure 100"


def test_past_the_boundary_the_branches_actually_diverge():
    """The complement of the test above: at exposure >= 100 the two stage-2
    LRs must resolve differently and produce different updates from the same
    state.  Driven by setting the cursor rather than training 100 exposures."""
    boundary = LR_BOUNDARY_EXPOSURES
    a, b = make(lr_stage2=3e-4), make(lr_stage2=1e-3)
    b.model.load_state_dict(a.model.state_dict())          # identical start
    for tr in (a, b):
        tr.global_step = boundary * tr.per_epoch
        assert tr.exposures == boundary
    assert a.current_lr() == 3e-4 and b.current_lr() == 1e-3
    a.train_step(); b.train_step()
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert [k for k in sa if not torch.equal(sa[k], sb[k])], \
        "different stage-2 LRs must produce different updates"


# =================================================  the error audit  =======

def test_error_audit_lists_every_top1_error_with_the_required_fields(tmp_path):
    out = str(tmp_path / "runs")
    assert main(["--route", "comprehension", "--enc-hidden", "64",
                 "--out-dir", out, "--run-id", "c",
                 "--eval-exposures", "1", "--max-exposures", "1"] + ARGS) == 0
    ck_path = os.path.join(out, "c", "checkpoints", "step_00000049.pt")
    adir = str(tmp_path / "audit")
    assert audit_main(["--ckpt", ck_path, "--out-dir", adir,
                       "--max-words", "400", "--glove-path",
                       "tests/_no_such_glove_file.txt",
                       "--allow-glove-fallback",
                       "--no-population-hash-check"]) == 0
    rows = list(csv.DictReader(
        open(os.path.join(adir, "c512_top1_errors.tsv")), delimiter="\t"))
    summary = json.load(open(os.path.join(adir, "c512_error_summary.json")))

    assert len(rows) == summary["top1_errors"] > 0
    assert summary["top1_errors"] == round(
        (1 - summary["top1"]) * summary["population"])
    assert list(rows[0].keys()) == COLUMNS
    assert sum(summary["categories"].values()) == len(rows)
    for r in rows:                       # every listed item really is an error
        assert int(r["target_rank"]) > 1
        assert r["target_bank_index"] != r["pred_bank_index"]
        assert float(r["margin_target_minus_top1"]) < 0
    # sorted worst-margin-last: the closest misses come first
    margins = [float(r["margin_target_minus_top1"]) for r in rows]
    assert margins == sorted(margins, reverse=True)


def test_audit_taxonomy_separates_homophones_and_duplicate_vectors(tmp_path):
    """Force each diagnostic category on a controlled bank."""
    from scripts.naming_comprehension import c_error_audit as mod
    out = str(tmp_path / "runs")
    assert main(["--route", "comprehension", "--enc-hidden", "64",
                 "--out-dir", out, "--run-id", "c",
                 "--eval-exposures", "1", "--max-exposures", "1"] + ARGS) == 0
    ck_path = os.path.join(out, "c", "checkpoints", "step_00000049.pt")

    def classify(same_phon, identical, margin, glove_cos):
        if identical:
            return "duplicate_semantic"
        if same_phon:
            return "homophone"
        if margin > -MARGIN_EPS:
            return "near_tie"
        if glove_cos >= NEIGHBOUR_COS:
            return "semantic_neighbour"
        return "far_miss"

    assert classify(True, True, -1.0, 0.0) == "duplicate_semantic"
    assert classify(True, False, -1.0, 0.0) == "homophone"
    assert classify(False, False, -0.001, 0.0) == "near_tie"
    assert classify(False, False, -0.5, 0.9) == "semantic_neighbour"
    assert classify(False, False, -0.5, 0.1) == "far_miss"
    assert mod.MARGIN_EPS == 0.01 and mod.NEIGHBOUR_COS == 0.60

    adir = str(tmp_path / "audit")
    assert audit_main(["--ckpt", ck_path, "--out-dir", adir,
                       "--max-words", "400", "--glove-path",
                       "tests/_no_such_glove_file.txt",
                       "--allow-glove-fallback",
                       "--no-population-hash-check"]) == 0
    rows = list(csv.DictReader(
        open(os.path.join(adir, "c512_top1_errors.tsv")), delimiter="\t"))
    for r in rows:
        assert r["category"] == classify(
            bool(int(r["is_homophone_of_target"])),
            bool(int(r["identical_glove_vectors"])),
            float(r["margin_target_minus_top1"]),
            float(r["pred_target_glove_cos"]))
        # a homophone row must really share the phoneme string
        if int(r["is_homophone_of_target"]):
            assert r["target_phonemes"] == r["pred_phonemes"]


def test_audit_refuses_a_non_comprehension_checkpoint(tmp_path):
    from scripts.naming_comprehension.route_capacity_probe import main as pmain
    out = str(tmp_path / "runs")
    assert pmain(["--route", "dorsal", "--wm-hidden", "64",
                  "--out-dir", out, "--run-id", "d",
                  "--eval-exposures", "1", "--max-exposures", "1"] + ARGS) == 0
    with pytest.raises(SystemExit, match="not a comprehension checkpoint"):
        audit_main(["--ckpt", os.path.join(out, "d", "checkpoints",
                                           "step_00000050.pt"),
                    "--out-dir", str(tmp_path / "a"), "--max-words", "400",
                    "--glove-path", "tests/_no_such_glove_file.txt",
                    "--allow-glove-fallback", "--no-population-hash-check"])


# ====================================================  job contract  =======

def test_c512_branch_job_contract():
    t = script()
    assert "SOURCE_STEP=43800" in t, "source must be exactly 100 exposures"
    assert 'LR_STAGE2=${1' in t and 'RUN_ID=${2' in t
    assert "MAX_EXPOSURES=300" in t
    assert "EVALS=100,110,125,150,175,200,250,300" in t
    assert "--lr-transition" in t
    assert "ENC=512" in t and "WM=128" in t and "DEC=128" in t
    assert "cap4_*" in t
    assert "cap3_c_enc512_seed22_full" in t
    assert "#SBATCH --requeue" in t
    # the stage-2 LR must come from the argument, never be hardcoded, so the
    # existing 1e-4 control cannot be re-run by this job
    assert '--lr-stage2 "$LR_STAGE2"' in t
    assert 'LR_STAGE2=${1' in t
    # the TRAINED LR comes only from the argument: no literal LR reaches srun
    srun = t.split("srun python", 1)[1].split("\n\n", 1)[0]
    assert "1e-4" not in srun and "3e-4" not in srun and "1e-3" not in srun
    # and the preflight pins that the SOURCE is the 1e-4 control arm
    assert 'float(ck.get("lr_stage2", 1e-4)) == 1e-4' in t
    for forbidden in ("--batch-size", "--allow-glove-fallback",
                      "--no-population-hash-check", "--no-stop-at-ceiling"):
        assert forbidden not in t, forbidden
