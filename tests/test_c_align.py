"""Acceptance tests for the FINAL-2A comprehension-alignment intervention.

FINAL-2A adds the C stream's OWN semantic-target alignment to the summed
objective, reusing the s_hat already computed for retrieval and the canonical
`losses.alignment_loss` the R stream uses:

    C contribution = LAMBDA_C * retrieval_CE + c_align_weight * alignment

The alignment term is weighted by `c_align_weight` ALONE, never by LAMBDA_C.
`c_align_weight` defaults to 0.0, so the frozen FINAL-1 objective is what runs
unless the flag is given.

These tests pin: baseline equivalence at 0.0 (bitwise over a short
deterministic run), the exact scalar added at 1.0, the parameter pathway the
new term may and may not reach, bitwise exact resume under 1.0, and the
provenance record.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from losses import alignment_loss                                       # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    FINAL_FULL_MODE, LAMBDA_C, TAU, JointScratchTrainer,
)
from scripts.naming_comprehension.train_tasks import (                  # noqa: E402
    comprehension_forward, retrieval_loss,
)

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")


def make_trainer(regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(regime=regime, seed=seed, **kw)


def params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# =========================================  1. baseline equivalence at 0.0  ==

def test_default_weight_is_zero():
    assert make_trainer().c_align_weight == 0.0
    from scripts.naming_comprehension.train_joint_scratch import build_parser
    assert build_parser().parse_args(["--regime", "j0"]).c_align_weight == 0.0


def test_weight_zero_is_bitwise_identical_to_final1():
    """A short deterministic run with the flag absent vs explicitly 0.0 must
    be bitwise identical -- FINAL-1 semantics are preserved exactly."""
    a = make_trainer()                       # flag absent -> default 0.0
    b = make_trainer(c_align_weight=0.0)     # explicit 0.0
    for _ in range(8):
        ra, rb = a.train_step(), b.train_step()
        assert ra["joint_total"] == rb["joint_total"]
        assert ra["grad_norm"] == rb["grad_norm"]
    pa, pb = params(a.model), params(b.model)
    bad = [k for k in pa if not torch.equal(pa[k], pb[k])]
    assert not bad, f"{len(bad)} tensors differ, e.g. {bad[:4]}"


def test_weight_zero_logs_nan_and_computes_no_align():
    rec = make_trainer().train_step()
    assert rec["c_align"] != rec["c_align"]              # NaN, never fabricated
    assert rec["c_align_weighted"] != rec["c_align_weighted"]
    assert rec["retrieval_ce"] == rec["retrieval_ce"]    # retrieval still real


def test_weight_one_changes_the_trajectory():
    """Sanity: the intervention is not a no-op."""
    a, b = make_trainer(), make_trainer(c_align_weight=1.0)
    for _ in range(5):
        a.train_step(); b.train_step()
    pa, pb = params(a.model), params(b.model)
    assert any(not torch.equal(pa[k], pb[k]) for k in pa)


# ==============================================  2. exact scalar added  =====

def test_weight_one_adds_exactly_the_canonical_alignment_loss():
    """joint(w=1) - joint(w=0) == alignment_loss(s_hat_C, GloVe_C), computed
    on the SAME C batch and the SAME s_hat, at identical parameters."""
    a = make_trainer()                          # w = 0.0
    b = make_trainer(c_align_weight=1.0)        # w = 1.0, same seed/state
    pa, pb = params(a.model), params(b.model)
    assert all(torch.equal(pa[k], pb[k]) for k in pa), "trainers not paired"

    # the C batch both trainers will draw at cursor 0
    c = b.batch("comprehension")
    with torch.no_grad():
        s_hat = comprehension_forward(b.model, c["enc_in"], c["enc_mask"])
        expected_align = float(alignment_loss(s_hat, c["semantic"]))
        expected_ret = float(retrieval_loss(
            s_hat, b.model.ltm.semantic_bank, c["bank_idx"], TAU))

    ra, rb = a.train_step(), b.train_step()
    assert rb["c_align"] == pytest.approx(expected_align, rel=1e-6)
    assert rb["retrieval_ce"] == pytest.approx(expected_ret, rel=1e-6)
    assert rb["joint_total"] - ra["joint_total"] == pytest.approx(
        expected_align, rel=1e-5, abs=1e-6)


def test_alignment_is_not_scaled_by_lambda_c():
    """The recorded weighted contribution is w * align, NOT LAMBDA_C * w * align."""
    w = 0.5
    rec = make_trainer(c_align_weight=w).train_step()
    assert rec["c_align_weighted"] == pytest.approx(w * rec["c_align"])
    assert rec["retrieval_weighted"] == pytest.approx(
        LAMBDA_C * rec["retrieval_ce"])
    # a LAMBDA_C-scaled implementation would be ~11x smaller
    assert rec["c_align_weighted"] != pytest.approx(
        LAMBDA_C * w * rec["c_align"], rel=1e-3)


def test_half_weight_is_half_the_added_scalar():
    a = make_trainer()
    b = make_trainer(c_align_weight=1.0)
    h = make_trainer(c_align_weight=0.5)
    ra, rb, rh = a.train_step(), b.train_step(), h.train_step()
    assert (rh["joint_total"] - ra["joint_total"]) == pytest.approx(
        0.5 * (rb["joint_total"] - ra["joint_total"]), rel=1e-5)


# ==================================  3./4. parameter pathway of the term  ===

ENCODER_SIDE = ("phon_embed.", "ltm.encoder.", "ltm.to_semantic.")
NAMING_ONLY = ("ltm.sem_to_h0.", "ltm.decoder.", "ltm.dec_to_premotor.")


def test_c_align_term_reaches_only_the_comprehension_pathway():
    """The C-alignment term ALONE must send gradient to phon_embed, the LTM
    encoder and to_semantic, and to nothing else -- in particular not through
    sem_to_h0 / the LTM decoder / motor / WM."""
    tr = make_trainer(c_align_weight=1.0)
    c = tr.batch("comprehension")
    s_hat = comprehension_forward(tr.model, c["enc_in"], c["enc_mask"])
    tr.model.zero_grad(set_to_none=True)
    alignment_loss(s_hat, c["semantic"]).backward()

    got = {n for n, p in tr.model.named_parameters()
           if p.grad is not None and torch.any(p.grad != 0)}
    for prefix in ENCODER_SIDE:
        assert any(n.startswith(prefix) for n in got), \
            f"C-align sends no gradient to {prefix}"
    for prefix in NAMING_ONLY:
        assert not any(n.startswith(prefix) for n in got), \
            f"C-align unexpectedly reaches naming-only pathway {prefix}"
    for prefix in ("wm.", "motor.", "gate."):
        assert not any(n.startswith(prefix) for n in got), \
            f"C-align unexpectedly reaches {prefix}"


def test_full_step_still_trains_every_parameter_group():
    """The whole summed update (unchanged elsewhere) must still reach the
    naming/dorsal pathways -- the pathway restriction above is a property of
    the C term alone, not of the step."""
    tr = make_trainer(c_align_weight=1.0)
    before = params(tr.model)
    tr.train_step()
    after = params(tr.model)
    for prefix in ENCODER_SIDE + NAMING_ONLY + ("wm.", "motor."):
        assert any(k.startswith(prefix) and not torch.equal(before[k], after[k])
                   for k in before), f"{prefix} did not train"


# ==========================================  5. exact resume under w=1.0  ===

def test_exact_resume_bitwise_with_c_align(tmp_path):
    a = make_trainer(c_align_weight=1.0)
    for _ in range(7):                       # mid-epoch, past the LR boundary
        a.train_step()
    ck = tmp_path / "mid.pt"
    torch.save(a.state_dict(), str(ck))
    for _ in range(5):
        a.train_step()

    b = make_trainer(c_align_weight=1.0)
    b.load_state_dict(torch.load(str(ck), weights_only=False), source="test")
    assert b.global_step == 7
    for _ in range(5):
        b.train_step()

    pa, pb = params(a.model), params(b.model)
    bad = [k for k in pa if not torch.equal(pa[k], pb[k])]
    assert not bad, f"resume diverged on {len(bad)} tensors, e.g. {bad[:4]}"
    assert a.cursors == b.cursors


def test_resume_refuses_across_a_change_of_c_objective(tmp_path):
    a = make_trainer(c_align_weight=1.0)
    a.train_step()
    p = tmp_path / "calign.pt"
    torch.save(a.state_dict(), str(p))
    with pytest.raises(RuntimeError, match="c_align_weight"):
        make_trainer(c_align_weight=0.0).load_state_dict(
            torch.load(str(p), weights_only=False), source="t")
    # ... and a FINAL-1 checkpoint (no such key) counts as 0.0
    ck = torch.load(str(p), weights_only=False)
    ck.pop("c_align_weight")
    make_trainer(c_align_weight=0.0).load_state_dict(ck, source="t")
    with pytest.raises(RuntimeError, match="c_align_weight"):
        make_trainer(c_align_weight=1.0).load_state_dict(ck, source="t")


# ==================================================  6. provenance record  ==

def test_settings_and_checkpoint_record_the_weight():
    tr = make_trainer(c_align_weight=1.0)
    s = tr.resolved_settings()
    assert s["c_align_weight"] == 1.0
    assert "alignment_loss" in s["c_stream_objective"]
    assert str(LAMBDA_C) in s["c_stream_objective"]
    ck = tr.state_dict()
    assert ck["c_align_weight"] == 1.0
    base = make_trainer().resolved_settings()
    assert base["c_align_weight"] == 0.0
    assert "FINAL-1" in base["c_stream_objective"]


def test_positive_weight_without_the_c_stream_is_refused():
    """A regime that never draws a C batch would make the flag a silent no-op."""
    with pytest.raises(RuntimeError, match="comprehension stream"):
        make_trainer(regime="n_only", c_align_weight=1.0)
    with pytest.raises(ValueError):
        make_trainer(c_align_weight=-1.0)


def test_losses_tsv_carries_the_new_columns(tmp_path):
    import csv
    from scripts.naming_comprehension.train_joint_scratch import main
    out = str(tmp_path / "runs")
    rc = main(["--regime", "j0", "--seed", "22", "--subset-mode", "final_full",
               "--device", "cpu", "--max-steps", "2", "--c-align-weight", "1.0",
               "--eval-every", "0", "--save-every", "0", "--log-every", "1",
               "--max-words", "400", "--batch-size", "8",
               "--dorsal-pool-size", "32", "--lr-boundary-steps", "6",
               "--out-dir", out, "--run-id", "ca",
               "--glove-path", "tests/_no_such_glove_file.txt",
               "--allow-glove-fallback", "--no-subset-hash-check"])
    assert rc == 0
    rows = list(csv.DictReader(open(os.path.join(out, "ca", "logs", "losses.tsv")),
                               delimiter="\t"))
    assert rows and {"retrieval_ce", "retrieval_weighted", "c_align",
                     "c_align_weighted", "grad_norm"} <= set(rows[0])
    assert float(rows[0]["c_align"]) > 0.0
    import json
    prov = json.load(open(os.path.join(out, "ca", "provenance.json")))
    assert prov["c_align_weight"] == 1.0
    cfg = json.load(open(os.path.join(out, "ca", "config.json")))
    assert cfg["c_align_weight"] == 1.0
