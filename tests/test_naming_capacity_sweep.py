"""Acceptance tests for the CAP-1 ventral-width naming diagnostic.

The sweep's validity rests on three properties: the width knob changes ONLY
the ventral decoder side; every width sees the identical counter-addressed
item order; and nothing outside the naming path ever moves.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.naming_capacity_sweep import (          # noqa: E402
    LR, NAMING_PATH_PREFIXES, WEIGHT_DECAY, NamingCapacityTrainer,
    assert_frozen_untouched, main, param_census,
)

TINY = dict(seed=22, device="cpu", max_words=400, batch_size=8,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            glove_path="tests/_no_such_glove_file.txt",
            allow_glove_fallback=True)

ARGS = ["--seed", "22", "--device", "cpu", "--max-words", "400",
        "--batch-size", "8", "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--log-every", "0"]


def make(width, **over):
    kw = dict(TINY)
    kw.update(over)
    return NamingCapacityTrainer(width=width, **kw)


def test_width_knob_touches_only_the_ventral_decoder_side():
    a, b = make(128), make(512)
    assert a.cfg.wm.hidden == b.cfg.wm.hidden == 128
    assert a.cfg.ltm.enc_hidden == b.cfg.ltm.enc_hidden == 128
    assert a.cfg.ltm.dec_hidden == 128 and b.cfg.ltm.dec_hidden == 512
    assert b.model.ltm.sem_to_h0.out_features == 512
    assert b.model.ltm.decoder.hidden_size == 512
    assert b.model.ltm.dec_to_premotor.in_features == 512
    ca, cb = param_census(a.model), param_census(b.model)
    for grp in ("wm", "ltm_encoder", "phon_embed", "motor"):
        assert ca[grp] == cb[grp], grp
    for grp in ("ltm_sem_to_h0", "ltm_decoder", "ltm_dec_to_premotor"):
        assert cb[grp] > ca[grp], grp


def test_item_order_is_identical_across_widths():
    a, b = make(128), make(256)
    for k in (0, 1, 7, 50):
        assert a.stream.indices(k) == b.stream.indices(k)


def test_only_naming_path_parameters_move():
    tr = make(64)
    before = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
    for _ in range(3):
        tr.train_step()
    moved = {n for n, p in tr.model.named_parameters()
             if not torch.equal(p.detach(), before[n])}
    assert moved, "training must move something"
    for n in moved:
        assert n.startswith(NAMING_PATH_PREFIXES), n
    assert_frozen_untouched(tr.model, tr.frozen_ref)          # must not raise
    # and the assertion genuinely fires on a violation
    with torch.no_grad():
        tr.model.wm.encoder.weight_hh_l0.add_(1.0)
    with pytest.raises(RuntimeError, match="non-naming"):
        assert_frozen_untouched(tr.model, tr.frozen_ref)


def test_resume_is_exact(tmp_path):
    out = str(tmp_path / "runs")
    assert main(["--width", "64", "--out-dir", out, "--run-id", "whole",
                 "--eval-exposures", "0,2", "--max-exposures", "2"] + ARGS) == 0
    assert main(["--width", "64", "--out-dir", out, "--run-id", "split",
                 "--eval-exposures", "0,1", "--max-exposures", "1"] + ARGS) == 0
    assert main(["--width", "64", "--out-dir", out, "--run-id", "split",
                 "--eval-exposures", "2", "--max-exposures", "2", "--resume",
                 os.path.join(out, "split", "checkpoints", "step_00000050.pt")]
                + ARGS) == 0
    a = torch.load(os.path.join(out, "whole", "checkpoints", "step_00000100.pt"),
                   map_location="cpu", weights_only=False)
    b = torch.load(os.path.join(out, "split", "checkpoints", "step_00000100.pt"),
                   map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]


def test_resume_refuses_a_different_width(tmp_path):
    out = str(tmp_path / "runs")
    assert main(["--width", "64", "--out-dir", out, "--run-id", "w",
                 "--eval-exposures", "1", "--max-exposures", "1"] + ARGS) == 0
    with pytest.raises(RuntimeError, match="width"):
        make(128).load_state_dict(torch.load(
            os.path.join(out, "w", "checkpoints", "step_00000050.pt"),
            map_location="cpu", weights_only=False))


def test_recipe_constants_are_the_declared_ones():
    assert LR == 1e-3 and WEIGHT_DECAY == 1e-5
    tr = make(64)
    assert tr.optim.param_groups[0]["lr"] == LR
    assert tr.naming_idx == list(range(len(tr.entries))), \
        "naming population must be the full lexicon, no homophone restriction"
