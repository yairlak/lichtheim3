"""Tests for the stage-1 encoder-only representation extraction (M4 Phase A/B).

Two groups:

  live      one checkpoint is loaded and its encoder run on a small batch, to
            prove the extracted tensor really is the GRU `h[-1]` consumed by
            `to_semantic`, that repeated extraction is deterministic, and that
            the decoder guard fires when a decoder path is touched.

  artifact  the written outputs are checked for shape, dtype, seed coverage,
            item-order alignment, equivalence against the canonical saved
            arrays, and provenance schema.

No test here trains, decodes or generates a token.
"""
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

from evaluate.hooks import make_batch                                  # noqa: E402
from scripts.length_effect_analysis import extract_encoder_stage1 as X  # noqa: E402

OUT = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/instrumented/"
                         "stage1_encoder_extraction")
INSTR = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/instrumented")
SEEDS = [19, 20, 21, 22]

pytestmark = pytest.mark.skipif(
    not os.path.isdir(OUT),
    reason="stage-1 encoder extraction has not been run")


# --------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def loaded():
    """One checkpoint plus a small batch.  Loaded once for the whole module."""
    from scripts.external_eval import load_model_and_vocab
    rel, _, exp_sha = X.CKPTS[19]
    path = os.path.join(ROOT, X.BUNDLE, rel)
    if not os.path.exists(path):
        pytest.skip("checkpoint bundle not present")
    model, vocab, _ = load_model_and_vocab(path, "cpu")
    model.eval()
    _, forms = X.load_items(vocab)
    batch = make_batch(forms[:16], vocab, "cpu")
    return model, vocab, batch


@pytest.fixture(scope="module")
def artifacts():
    val = pd.read_csv(os.path.join(OUT, "extraction_validation.tsv"), sep="\t")
    idx = pd.read_csv(os.path.join(OUT, "item_index.tsv"), sep="\t")
    with open(os.path.join(OUT, "provenance.json")) as f:
        prov = json.load(f)
    return val, idx, prov


# ------------------------------------------------------------ live: identity

def test_extracted_tensor_is_the_gru_last_hidden_fed_to_to_semantic(loaded):
    """The saved value is bit-identical to `h[-1]` and to `to_semantic`'s input."""
    model, _, batch = loaded
    with torch.inference_mode():
        out = X.extract_batch(model, batch)
        independent = X.ltm_encoder_last_hidden(
            model.ltm, batch["enc_in"], batch["enc_mask"])
    assert torch.equal(out["ltm_encoder_hidden"], independent)

    # and it is literally what the first projection layer receives
    captured = []
    hook = model.ltm.to_semantic[0].register_forward_pre_hook(
        lambda _m, inp: captured.append(inp[0].detach().clone()))
    try:
        with torch.inference_mode():
            model.ltm.encode(batch["enc_in"], batch["enc_mask"])
    finally:
        hook.remove()
    assert len(captured) == 1
    assert torch.equal(captured[0], out["ltm_encoder_hidden"])


def test_projection_of_extracted_tensor_reproduces_s_hat(loaded):
    """`to_semantic(ltm_encoder_hidden)` == the canonical `s_hat`, exactly."""
    model, _, batch = loaded
    with torch.inference_mode():
        out = X.extract_batch(model, batch)
        recon = model.ltm.to_semantic(out["ltm_encoder_hidden"])
    assert torch.equal(recon, out["s_hat"])


def test_no_reverse_direction_gru_tensor(loaded):
    """The LTM encoder is unidirectional; no `_reverse` parameters exist."""
    model, _, _ = loaded
    enc = model.ltm.encoder
    assert enc.bidirectional is False
    assert enc.num_layers == 1
    assert enc.hidden_size == 128
    assert [k for k in enc.state_dict() if "_reverse" in k] == []
    # the projection's fan-in equals the *unidirectional* hidden size
    assert model.ltm.to_semantic[0].in_features == enc.hidden_size


def test_extracted_shape_is_n_by_128(loaded):
    model, _, batch = loaded
    with torch.inference_mode():
        out = X.extract_batch(model, batch)
    assert tuple(out["ltm_encoder_hidden"].shape) == (16, 128)
    assert tuple(out["s_hat"].shape) == (16, 300)
    assert tuple(out["ltm_decoder_h0"].shape) == (16, 128)
    assert tuple(out["wm_encoder_hidden"].shape) == (16, 128)


def test_repeated_extraction_is_deterministic(loaded):
    model, _, batch = loaded
    with torch.inference_mode():
        a = X.extract_batch(model, batch)
        b = X.extract_batch(model, batch)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_architecture_assertions_pass(loaded):
    model, _, _ = loaded
    arch = X.architecture_assertions(model)
    assert arch["ltm_encoder_mode"] == "unigru_last_hidden"
    assert arch["ltm_encoder_bidirectional"] is False
    assert arch["ltm_encoder_hidden_size"] == 128
    assert arch["ltm_ventral_noise"] == 0.0
    assert arch["wm_interference_noise"] == 0.0


# --------------------------------------------------------- live: decoder guard

def test_guard_permits_encoder_only_extraction(loaded):
    """The real extraction path runs to completion inside the guard."""
    model, _, batch = loaded
    with X.DecoderGuard(model) as g, torch.inference_mode():
        X.extract_batch(model, batch)
    assert g.violations == []
    assert "motor" in g.installed
    assert "ltm.decoder" in g.installed
    assert "wm.decoder" in g.installed


@pytest.mark.parametrize("call", [
    "ltm_decode", "wm_decode", "motor", "model_forward",
])
def test_guard_fires_on_every_decoder_entry_point(loaded, call):
    """A guard that never fires would prove nothing; each path must raise."""
    model, _, batch = loaded
    with X.DecoderGuard(model) as g, torch.inference_mode():
        with pytest.raises(RuntimeError, match="decoder guard violated"):
            if call == "ltm_decode":
                model.ltm.decode_from_s_hat(
                    torch.zeros(16, 300), batch["dec_in"])
            elif call == "wm_decode":
                model.wm.decode_from_state(
                    torch.zeros(1, 16, 128), batch["dec_in"])
            elif call == "motor":
                model.motor(torch.zeros(16, 3, 128))
            else:
                model(batch["enc_in"], batch["enc_mask"], batch["dec_in"])
    assert g.violations, "guard recorded no violation"


def test_no_decoder_invocation_recorded_in_the_real_run(artifacts):
    _, _, prov = artifacts
    assert prov["decoder_executed"] is False
    assert prov["tokens_generated"] is False
    assert prov["autoregressive_decoding"] is False
    assert prov["motor_projection_applied"] is False
    for seed in SEEDS:
        g = prov["decoder_guard"][str(seed)]
        assert g["violations"] == []
        assert {"ltm.decoder", "wm.decoder", "motor", "model.forward"} \
            <= set(g["guarded_entry_points"])


# ------------------------------------------------------- artifact: coverage

def test_four_seed_coverage_and_shapes(artifacts):
    _, idx, _ = artifacts
    assert sorted(idx["seed"].unique().tolist()) == SEEDS
    for seed in SEEDS:
        a = np.load(os.path.join(OUT, f"ltm_encoder_hidden_seed{seed}.npy"))
        assert a.shape == (1200, 128), (seed, a.shape)
        assert a.dtype == np.float32
        assert np.isfinite(a).all()


def test_exact_item_order_alignment(artifacts):
    _, idx, _ = artifacts
    summary = pd.read_csv(os.path.join(INSTR, "item_summary.tsv"), sep="\t")
    for seed in SEEDS:
        sub = idx[idx["seed"] == seed].sort_values("row_index")
        ref = summary[summary["seed"] == seed].reset_index(drop=True)
        assert sub["row_index"].tolist() == list(range(1200))
        assert sub["item_id"].tolist() == ref["item_id"].tolist()
        assert sub["phoneme_length"].tolist() == ref["phoneme_length"].tolist()
        assert sub["exposure_status"].tolist() == ref["exposure_status"].tolist()


def test_no_missing_or_duplicate_item_ids(artifacts):
    _, idx, _ = artifacts
    for seed in SEEDS:
        sub = idx[idx["seed"] == seed]
        assert len(sub) == 1200
        assert sub["item_id"].nunique() == 1200
    # identical item set across seeds
    sets = [set(idx[idx["seed"] == s]["item_id"]) for s in SEEDS]
    assert all(s == sets[0] for s in sets)


def test_item_index_carries_the_expected_checkpoint_hashes(artifacts):
    _, idx, _ = artifacts
    for seed in SEEDS:
        got = idx[idx["seed"] == seed]["checkpoint_sha256"].unique().tolist()
        assert got == [X.CKPTS[seed][2]]


# ---------------------------------------------------- artifact: equivalence

@pytest.mark.parametrize("tensor", ["s_hat", "ltm_decoder_h0",
                                    "wm_encoder_hidden"])
def test_recomputed_tensor_matches_the_canonical_saved_array(tensor):
    saved = np.load(os.path.join(INSTR, "representations.npz"))
    new = np.load(os.path.join(OUT, "recomputed_encoder_quantities.npz"))
    for seed in SEEDS:
        a = new[f"{tensor}_seed{seed}"]
        b = saved[f"{tensor}_seed{seed}"]
        assert a.shape == b.shape, (tensor, seed)
        assert np.allclose(a, b, rtol=X.RTOL, atol=X.ATOL), (tensor, seed)


def test_validation_table_is_complete_and_passing(artifacts):
    val, _, _ = artifacts
    assert sorted(val["seed"].unique().tolist()) == SEEDS
    assert set(val["tensor"]) == {"s_hat", "ltm_decoder_h0",
                                  "wm_encoder_hidden", "ltm_encoder_hidden"}
    assert len(val) == 16
    assert val["shape_match"].all()
    assert val["allclose"].all()
    cmp = val[val["tensor"] != "ltm_encoder_hidden"]
    assert (cmp["rtol"] == X.RTOL).all() and (cmp["atol"] == X.ATOL).all()
    assert (cmp["max_abs_diff"] <= X.ATOL).all()


def test_predeclared_tolerances_were_not_changed(artifacts):
    _, _, prov = artifacts
    assert prov["tolerances"]["rtol"] == 1e-6
    assert prov["tolerances"]["atol"] == 1e-7
    assert prov["tolerances"]["changed_after_seeing_results"] is False


# ----------------------------------------------------- artifact: provenance

def test_provenance_schema(artifacts):
    _, _, prov = artifacts
    required = {"phase", "execution_type", "decoder_executed", "training_performed",
                "weights_modified", "architecture_changed",
                "checkpoint_training_commit", "repository_head", "checkpoints",
                "dataset_hashes", "script_sha256", "batch_size", "device",
                "torch_version", "inference_mode", "model_eval_mode",
                "noise_settings", "architecture_assertions", "decoder_guard",
                "identity_proof", "tolerances",
                "encoder_extraction_equivalence", "elapsed_seconds", "seeds",
                "n_items_per_seed"}
    assert required <= set(prov)
    assert prov["training_performed"] is False
    assert prov["weights_modified"] is False
    assert prov["architecture_changed"] is False
    assert prov["inference_mode"] is True and prov["model_eval_mode"] is True
    assert prov["n_items_per_seed"] == 1200
    assert prov["seeds"] == SEEDS
    assert prov["encoder_extraction_equivalence"] == "PASS"
    for seed in SEEDS:
        assert prov["checkpoints"][str(seed)]["sha256"] == X.CKPTS[seed][2]
        assert prov["architecture_assertions"][str(seed)][
            "ltm_encoder_bidirectional"] is False


def test_provenance_describes_the_run_as_an_execution_not_as_no_inference(artifacts):
    """The wording requirement: this is extraction *with no decoder execution*."""
    _, _, prov = artifacts
    t = prov["execution_type"].lower()
    assert "encoder-only" in t
    assert "no decoder execution" in t
    assert "no inference" not in t


def test_the_original_representations_npz_was_not_overwritten(artifacts):
    _, _, prov = artifacts
    assert prov["source_representations_npz_sha256"] == X.sha(
        os.path.join(INSTR, "representations.npz"))
    z = np.load(os.path.join(INSTR, "representations.npz"))
    assert "ltm_encoder_hidden_seed19" not in z.files
    assert len(z.files) == 20
