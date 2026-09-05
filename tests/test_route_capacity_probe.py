"""Acceptance tests for the CAP-3 route-isolated capacity probes.

The experiment's validity rests on four claims, each tested here:
  1. the three widths are INDEPENDENT -- changing one never resizes another;
  2. each probe trains only its own route and freezes everything else, so a
     width result cannot be produced by some other route solving the task;
  3. the comprehension probe uses the canonical 27,981 population against the
     full 29,571 bank, with strict top-1 as the criterion;
  4. every width shares one LR schedule, never tuned per width.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.route_capacity_probe import (          # noqa: E402
    GRAD_CLIP, LR_BOUNDARY_EXPOSURES, LR_STAGE1, LR_STAGE2, ROUTE_DORSAL,
    ROUTE_COMPREHENSION, SCOPE, TAU, WEIGHT_DECAY, WIDTHS,
    WM_FREE_AR_MAX_STEPS, METRIC_COLUMNS, RouteCapacityTrainer,
    lr_for_exposure, main, param_census,
)
from scripts.naming_comprehension.train_joint_scratch import (           # noqa: E402
    EXPECTED_CANONICAL_C_HASH, EXPECTED_CANONICAL_C_N,
)

JOB = "scripts/cluster/jeanzay/cap3_route_capacity.slurm"
N256 = "scripts/cluster/jeanzay/cap2_n256_lr1e4.slurm"

TINY = dict(seed=22, device="cpu", max_words=400, batch_size=8,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            glove_path="tests/_no_such_glove_file.txt",
            allow_glove_fallback=True, require_population_hash=False)

ARGS = ["--seed", "22", "--device", "cpu", "--max-words", "400",
        "--batch-size", "8", "--glove-path", "tests/_no_such_glove_file.txt",
        "--allow-glove-fallback", "--no-population-hash-check",
        "--log-every", "0"]


def make(route, wm=128, enc=128, dec=128, **over):
    kw = dict(TINY)
    kw.update(over)
    return RouteCapacityTrainer(route=route, wm_hidden=wm, enc_hidden=enc,
                                dec_hidden=dec, **kw)


def script(path=JOB):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


# ==========================================  1. width independence  ========

@pytest.mark.parametrize("knob,grown,untouched", [
    ("enc", "ltm_encoder", ("wm", "ltm_decoder", "ltm_sem_to_h0",
                            "ltm_dec_to_premotor")),
    ("wm", "wm", ("ltm_encoder", "ltm_decoder", "ltm_sem_to_h0",
                  "ltm_dec_to_premotor")),
    ("dec", "ltm_decoder", ("wm", "ltm_encoder")),
])
def test_one_width_never_resizes_another(knob, grown, untouched):
    base = make(ROUTE_COMPREHENSION)
    wide = make(ROUTE_COMPREHENSION, **{knob: 512})
    cb, cw = param_census(base.model), param_census(wide.model)
    assert cw[grown] > cb[grown], f"{knob}=512 must grow {grown}"
    for g in untouched:
        assert cw[g] == cb[g], f"{knob}=512 silently resized {g}"


def test_widths_are_recorded_separately_and_default_to_canonical():
    tr = make(ROUTE_COMPREHENSION, enc=256)
    assert tr.widths == {"wm_hidden": 128, "ltm_enc_hidden": 256,
                         "ltm_dec_hidden": 128}
    assert tr.cfg.wm.hidden == 128 and tr.cfg.ltm.dec_hidden == 128
    assert tr.model.ltm.encoder.hidden_size == 256
    assert tr.model.wm.encoder.hidden_size == 128
    assert tr.model.ltm.decoder.hidden_size == 128, \
        "naming's 512 decoder must NOT be carried into the C probe"


def test_actual_module_dims_follow_each_knob():
    d = make(ROUTE_DORSAL, wm=512)
    assert d.model.wm.encoder.hidden_size == 512
    assert d.model.wm.decoder.hidden_size == 512
    assert d.model.wm.to_premotor.in_features == 512
    assert d.model.ltm.encoder.hidden_size == 128
    assert d.model.ltm.decoder.hidden_size == 128


# ================================================  2. route isolation  =====

@pytest.mark.parametrize("route", [ROUTE_COMPREHENSION, ROUTE_DORSAL])
def test_only_the_route_under_test_trains(route):
    tr = make(route)
    before = {n: p.detach().clone() for n, p in tr.model.named_parameters()}
    for _ in range(3):
        tr.train_step()
    moved = {n for n, p in tr.model.named_parameters()
             if not torch.equal(p.detach(), before[n])}
    assert moved, "training must move something"
    for n in moved:
        assert n.startswith(SCOPE[route]), f"{route} moved {n}"
    tr.assert_frozen()
    with torch.no_grad():                      # the guard must really fire
        next(p for n, p in tr.model.named_parameters()
             if not n.startswith(SCOPE[route])).add_(1.0)
    with pytest.raises(RuntimeError, match="out-of-scope"):
        tr.assert_frozen()


def test_comprehension_never_touches_the_production_decoder():
    tr = make(ROUTE_COMPREHENSION)
    for n in ("ltm.decoder.weight_hh_l0", "ltm.sem_to_h0.weight",
              "ltm.dec_to_premotor.weight", "motor.proj.weight",
              "wm.encoder.weight_hh_l0"):
        assert n in tr.frozen_ref, n
        assert not dict(tr.model.named_parameters())[n].requires_grad, n


def test_dorsal_never_touches_the_ltm_route():
    tr = make(ROUTE_DORSAL)
    ltm = [n for n in tr.frozen_names if n.startswith("ltm.")
           and "phon_embed" not in n]
    assert ltm, "the LTM route must be frozen for the dorsal probe"
    for n in ltm:
        assert not dict(tr.model.named_parameters())[n].requires_grad, n
    # and the dorsal loss must not call the LTM route at all
    assert "ltm" not in SCOPE[ROUTE_DORSAL]


# ==================================  3. population + criterion  ============

def test_comprehension_population_is_the_frozen_canonical_one():
    """Runs on the REAL lexicon: 27,981 canonical targets, full 29,571 bank."""
    tr = RouteCapacityTrainer(route=ROUTE_COMPREHENSION, wm_hidden=128,
                              enc_hidden=128, dec_hidden=128, seed=22,
                              device="cpu")
    assert len(tr.train_idx) == EXPECTED_CANONICAL_C_N == 27_981
    assert tr.population_hash == EXPECTED_CANONICAL_C_HASH
    assert len(tr.entries) == 29_571, "retrieval bank must stay the full lexicon"
    assert tr.per_epoch == 438


def test_comprehension_metrics_are_strict_top1_plus_diagnostics():
    tr = make(ROUTE_COMPREHENSION)
    row = tr.evaluate()
    for k in ("top1", "top5", "rank_mean", "rank_median", "target_cos_mean",
              "margin_mean", "retrieval_ce"):
        assert k in row, k
    assert 0.0 <= row["top1"] <= 1.0
    assert row["top1"] <= row["top5"], "top1 must be the strict criterion"
    assert row["_ceiling"] == (row["top1"] == 1.0)


def test_dorsal_reports_lexical_and_pseudoword_readouts():
    tr = make(ROUTE_DORSAL)
    row = tr.evaluate()
    for k in ("lex_exact", "wm_ce", "pseudo_exact", "pseudo_exact_short",
              "pseudo_exact_long"):
        assert k in row, k
    assert row["_ceiling"] == (row["lex_exact"] == 1.0), \
        "ceiling is lexical exact-match, not the pseudoword diagnostic"


# ==============================================  4. shared schedule  =======

def test_lr_schedule_is_two_stage_and_width_independent():
    assert (LR_STAGE1, LR_STAGE2, LR_BOUNDARY_EXPOSURES) == (1e-3, 1e-4, 100)
    assert lr_for_exposure(0) == lr_for_exposure(99.9) == LR_STAGE1
    assert lr_for_exposure(100) == lr_for_exposure(3000) == LR_STAGE2
    # every width and route resolves the same LR at the same exposure
    for route, kw in ((ROUTE_COMPREHENSION, "enc"), (ROUTE_DORSAL, "wm")):
        lrs = {make(route, **{kw: w}).current_lr() for w in (128, 256)}
        assert lrs == {LR_STAGE1}


def test_recipe_constants_and_no_lambda_c():
    import inspect
    from scripts.naming_comprehension import route_capacity_probe as m
    assert (WEIGHT_DECAY, GRAD_CLIP, TAU) == (1e-5, 1.0, 0.10)
    assert WIDTHS == (128, 256, 512)
    src = inspect.getsource(m.RouteCapacityTrainer.loss_on)
    assert "0.087" not in src and "LAMBDA_C" not in src, \
        "the isolated probe must use the unweighted objective"


def test_resume_is_exact_and_refuses_a_different_width(tmp_path):
    out = str(tmp_path / "runs")
    common = ["--route", "comprehension", "--enc-hidden", "64",
              "--out-dir", out, "--max-exposures"]
    assert main(common + ["2", "--run-id", "whole",
                          "--eval-exposures", "0,2"] + ARGS) == 0
    assert main(common + ["1", "--run-id", "split",
                          "--eval-exposures", "0,1"] + ARGS) == 0
    assert main(common + ["2", "--run-id", "split", "--eval-exposures", "2",
                          "--resume", os.path.join(
                              out, "split", "checkpoints", "step_00000049.pt")]
                + ARGS) == 0
    a = torch.load(os.path.join(out, "whole", "checkpoints", "step_00000098.pt"),
                   map_location="cpu", weights_only=False)
    b = torch.load(os.path.join(out, "split", "checkpoints", "step_00000098.pt"),
                   map_location="cpu", weights_only=False)
    sa, sb = a["model_state_dict"], b["model_state_dict"]
    assert not [k for k in sa if not torch.equal(sa[k], sb[k])]
    with pytest.raises(RuntimeError, match="this trainer is"):
        make(ROUTE_COMPREHENSION, enc=128).load_state_dict(a)


# ====================================================  job contracts  ======

def test_cap3_job_contract():
    t = script()
    assert "--route" in t and 'ROUTE=${1' in t
    assert 'WIDTH=${2' in t and 'RUN_ID=${3' in t
    assert "cap3_*" in t
    assert "#SBATCH --requeue" in t and "--benchmark" in t
    assert "sort | tail -1" in t
    assert "EXPECTED_C_N=27981" in t
    assert "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50" in t
    # a probe must never override the shared recipe
    for forbidden in ("--batch-size", "--allow-glove-fallback",
                      "--no-population-hash-check", "--lexicon-path",
                      "--no-stop-at-ceiling"):
        assert forbidden not in t, forbidden


def test_n256_job_contract():
    t = script(N256)
    assert "7ba12babb0d9327d56f579041bb20f5a98571f0167892a94b81499cf6c970377" in t
    assert "WIDTH=256" in t and "step_00046300.pt" in t
    assert '--lr "1e-4"' in t or "LR=1e-4" in t
    assert "--lr-transition" in t
    assert "cap2_n256" in t
    assert "#SBATCH --requeue" in t


def test_earlier_jobs_are_untouched():
    for name, must in (("cap2_h512_lr_branch.slurm", "WIDTH=512"),
                       ("final9e_durability_r190_to_r470.slurm",
                        "FINAL_STEP=912110"),
                       ("final9p_ratio223_r130.slurm", "MAX_STEPS=458370")):
        t = script(f"scripts/cluster/jeanzay/{name}")
        assert must in t, name


# =====================  corrections: sampler + free-AR metric  =============

def test_dorsal_uses_the_canonical_repetition_sampler():
    """The dorsal probe must reuse the sampler that established the existing
    WM128 repetition ceiling -- log-frequency weighted, WITH REPLACEMENT --
    not a uniform permutation borrowed from the C probe."""
    tr = make(ROUTE_DORSAL)
    assert tr.stream.weights is not None, "dorsal must be frequency-weighted"
    assert "log-frequency weighted" in tr.sampler_note
    assert "with replacement" in tr.sampler_note
    assert "freq_temp=1.0" in tr.sampler_note
    w = tr.stream.weights
    assert float(w.sum()) == pytest.approx(1.0)
    ranks = [tr.entries[i].rank for i in tr.train_idx]
    order = sorted(range(len(ranks)), key=lambda k: ranks[k])
    assert w[order[0]] > w[order[-1]], "rank 1 must carry the largest weight"
    # with replacement a full pass repeats items and omits others; an
    # unweighted permutation covers each item exactly once
    drawn = [i for k in range(tr.per_epoch) for i in tr.stream.indices(k)]
    assert len(drawn) == len(tr.train_idx)
    assert len(set(drawn)) < len(drawn), \
        "a with-replacement pass must contain duplicates"


def test_dorsal_stream_is_batch_identical_to_the_joint_repetition_stream():
    """Runs on the REAL lexicon: the probe must draw exactly the batches the
    canonical repetition stream draws, or it is not the historical recipe."""
    from scripts.naming_comprehension.train_joint_scratch import (
        CounterStream, derive_stream_seeds)
    from data.lexicon import logfreq_weights
    import numpy as np
    tr = RouteCapacityTrainer(route=ROUTE_DORSAL, wm_hidden=128,
                              enc_hidden=128, dec_hidden=128, seed=22,
                              device="cpu")
    w = logfreq_weights([e.rank for e in tr.entries]) ** float(
        tr.cfg.data.freq_temp)
    w = np.clip(w, 1e-6, None)
    ref = CounterStream("repetition", list(range(len(tr.entries))), 64,
                        derive_stream_seeds(22)["repetition"],
                        weights=w / w.sum())
    assert tr.per_epoch == ref.per_epoch == 463
    for k in (0, 1, 5, 462, 463, 1000):
        assert tr.stream.indices(k) == ref.indices(k), k


def test_comprehension_sampler_is_unchanged_by_the_dorsal_correction():
    """Guard: the C jobs are already running on this driver.  The dorsal
    sampler change must not touch the C stream."""
    tr = make(ROUTE_COMPREHENSION)
    assert tr.stream.weights is None
    assert tr.sampler_note == "unweighted permutation per pass (canonical C)"


def test_free_ar_cap_is_global_and_clears_the_longest_form():
    assert WM_FREE_AR_MAX_STEPS == 12
    tr = make(ROUTE_DORSAL)
    longest = max(len(e.phonemes) for e in tr.entries)
    assert WM_FREE_AR_MAX_STEPS > longest + 1,         "the cap must not truncate a correct answer"


def test_dorsal_reports_both_conventions_and_ceilings_on_the_canonical_one():
    tr = make(ROUTE_DORSAL)
    row = tr.evaluate()
    for k in ("lex_exact", "lex_exact_freear", "tf_token_acc",
              "pseudo_exact", "pseudo_exact_freear",
              "pseudo_exact_short", "pseudo_exact_long",
              "pseudo_exact_freear_short", "pseudo_exact_freear_long"):
        assert k in row, k
        assert k in METRIC_COLUMNS[ROUTE_DORSAL], f"{k} missing from the TSV"
    # ceiling must stay the canonical metric, so history stays comparable
    assert row["_ceiling"] == (row["lex_exact"] == 1.0)


def test_free_ar_never_consults_gold_length():
    """A model that never emits EOS must score 0 under free-AR for every
    item, whatever its length -- the forced-length window cannot rescue it."""
    tr = make(ROUTE_DORSAL)
    eos = tr.vocab.eos_id
    real_motor = tr.model.motor

    class NoEos(torch.nn.Module):
        def forward(self, x):
            out = real_motor(x)
            out[..., eos] = -1e9          # EOS can never win
            return out

    tr.model.motor = NoEos()
    try:
        forms = [e.phonemes for e in tr.entries[:64]]
        assert sum(tr._wm_exact_free(forms)) == 0
    finally:
        tr.model.motor = real_motor
