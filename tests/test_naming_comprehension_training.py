"""Focused tests for the Phase 2A single-task training infrastructure.

Verifies, on tiny synthetic models only (no checkpoint, no data file):
  - exact comprehension and naming parameter scopes;
  - phon_embed frozen in both, motor.proj frozen for naming N0;
  - parameters outside scope stay bit-identical after a real optimizer step;
  - canonical bank-index mapping is correct and order-sensitive;
  - retrieval loss is correct on a toy bank;
  - C0 is exactly the existing alignment loss;
  - naming forward is exactly motor(ltm.decode_from_s_hat(...));
  - the repetition path and total_loss are untouched.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import Config, DataConfig, LTMConfig, TrainConfig            # noqa: E402
from data.lexicon import LexEntry                                       # noqa: E402
from data.phonemes import build_vocab                                   # noqa: E402
from losses import alignment_loss, total_loss                           # noqa: E402
from models.dual_route import DualRouteModel                            # noqa: E402
from scripts.naming_comprehension.train_tasks import (                  # noqa: E402
    ALWAYS_FROZEN, TRAINABLE_PREFIXES, assert_dorsal_untouched,
    changed_parameters, comprehension_forward, comprehension_objective,
    dorsal_fingerprint, fresh_optimizer, make_batches, naming_forward,
    naming_objective, parameter_fingerprint, repetition_snapshot,
    retrieval_loss, set_trainable_scope, trainable_parameters,
    unique_phonology_indices, homophone_indices, verify_bank_mapping)

SEM_DIM = 16


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Config(
        data=DataConfig(use_real=False, glove_path=None, semantic_dim=SEM_DIM,
                        max_words=50, seed=0),
        ltm=LTMConfig(phon_embed_dim=8, enc_hidden=16, dec_hidden=16,
                      ltm_encoder_mode="unigru_last_hidden"),
        train=TrainConfig(device="cpu", seed=0),
    )
    vocab = build_vocab()
    model = DualRouteModel(cfg, vocab, premotor_dim=12)
    model.set_semantic_bank(torch.randn(10, SEM_DIM))
    model.eval()
    return model, vocab, cfg


def _tiny_entries():
    """6 words: 4 unique phonologies + one homophone pair (indices 2 and 5)."""
    rng = np.random.default_rng(0)
    specs = [("aa", [5, 6]), ("bb", [7, 8]), ("cc", [9, 10]),
             ("dd", [11, 12, 13]), ("ee", [14, 15]), ("ff", [9, 10])]
    return [LexEntry(word=w, phonemes=p,
                     semantic=rng.standard_normal(SEM_DIM).astype(np.float32),
                     freq=1.0, rank=i + 1)
            for i, (w, p) in enumerate(specs)]


# ------------------------------------------------------- parameter scopes

def test_comprehension_scope_is_exact():
    model, _, _ = _tiny_model()
    trainable = set_trainable_scope(model, "comprehension")
    assert trainable == sorted([
        "ltm.encoder.bias_hh_l0", "ltm.encoder.bias_ih_l0",
        "ltm.encoder.weight_hh_l0", "ltm.encoder.weight_ih_l0",
        "ltm.to_semantic.0.bias", "ltm.to_semantic.0.weight",
        "ltm.to_semantic.2.bias", "ltm.to_semantic.2.weight",
    ])
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert "phon_embed.weight" in frozen
    assert {"motor.proj.weight", "motor.proj.bias"} <= frozen
    assert not any(n.startswith("wm.") and p.requires_grad
                   for n, p in model.named_parameters())


def test_naming_scope_is_exact_N0():
    model, _, _ = _tiny_model()
    trainable = set_trainable_scope(model, "naming")
    assert trainable == sorted([
        "ltm.sem_to_h0.weight", "ltm.sem_to_h0.bias",
        "ltm.decoder.weight_ih_l0", "ltm.decoder.weight_hh_l0",
        "ltm.decoder.bias_ih_l0", "ltm.decoder.bias_hh_l0",
        "ltm.dec_to_premotor.weight", "ltm.dec_to_premotor.bias",
    ])
    # N0 must not include the encoder side, the shared readout or the embedding
    assert not any(n.startswith(("ltm.encoder.", "ltm.to_semantic.")) for n in trainable)
    params = dict(model.named_parameters())
    for n in ALWAYS_FROZEN:
        assert not params[n].requires_grad, f"{n} must be frozen"


def test_decoder_prefix_does_not_capture_dec_to_premotor_by_accident():
    # "ltm.decoder." and "ltm.dec_to_premotor." share the "ltm.dec" stem;
    # both are wanted for naming but must be listed explicitly, not by stem.
    assert "ltm.decoder." in TRAINABLE_PREFIXES["naming"]
    assert "ltm.dec_to_premotor." in TRAINABLE_PREFIXES["naming"]
    assert not any(p == "ltm.dec" for p in TRAINABLE_PREFIXES["naming"])


def test_phon_embed_is_one_shared_object_and_frozen_in_both_tasks():
    model, _, _ = _tiny_model()
    assert model.phon_embed is model.wm.phon_embed is model.ltm.phon_embed
    for task in ("comprehension", "naming"):
        set_trainable_scope(model, task)
        assert not model.phon_embed.weight.requires_grad
        assert not model.wm.phon_embed.weight.requires_grad
        assert not model.ltm.phon_embed.weight.requires_grad


def test_unknown_task_rejected():
    model, _, _ = _tiny_model()
    with pytest.raises(ValueError, match="Unknown task"):
        set_trainable_scope(model, "multitask")


# ------------------------------- out-of-scope parameters stay bit-identical

@pytest.mark.parametrize("task", ["comprehension", "naming"])
def test_out_of_scope_parameters_bit_identical_after_optimizer_step(task):
    model, vocab, cfg = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    model.set_semantic_bank(bank_raw.clone())

    trainable = set_trainable_scope(model, task)
    optim = fresh_optimizer(model, lr=1e-2, weight_decay=1e-2)
    before = parameter_fingerprint(model)

    batch = next(make_batches(entries, list(range(len(entries))), bank_raw,
                              vocab, batch_size=6, device="cpu"))
    if task == "comprehension":
        loss = comprehension_objective(model, batch, "c3", tau=0.1,
                                       lambda_ret=1.0)["total"]
    else:
        loss = naming_objective(model, batch, vocab.pad_id)["total"]
    optim.zero_grad()
    loss.backward()
    optim.step()

    changed = changed_parameters(model, before)
    assert changed, "sanity: the in-scope parameters must actually move"
    assert set(changed) <= set(trainable), (
        f"parameters outside scope changed: {sorted(set(changed) - set(trainable))}")
    # WM weights specifically must be untouched (phon_embed + motor frozen)
    assert not any(n.startswith("wm.") for n in changed)
    assert "phon_embed.weight" not in changed
    assert "motor.proj.weight" not in changed


def test_fresh_optimizer_only_holds_trainable_parameters():
    model, _, _ = _tiny_model()
    set_trainable_scope(model, "naming")
    optim = fresh_optimizer(model, lr=1e-3, weight_decay=0.0)
    held = {id(p) for group in optim.param_groups for p in group["params"]}
    assert held == {id(p) for p in trainable_parameters(model)}
    assert len(held) == 8
    assert id(model.phon_embed.weight) not in held
    assert id(model.motor.proj.weight) not in held


# -------------------------------------------------- bank index mapping

def test_unique_phonology_and_homophone_partition():
    entries = _tiny_entries()
    uniq = unique_phonology_indices(entries)
    homo = homophone_indices(entries)
    assert uniq == [0, 1, 3, 4]          # 'cc'/'ff' share [9,10]
    assert homo == [2, 5]
    assert set(uniq) & set(homo) == set()
    assert len(uniq) + len(homo) == len(entries)


def test_verify_bank_mapping_accepts_canonical_and_rejects_permutation():
    entries = _tiny_entries()
    bank = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    verify_bank_mapping(entries, bank, list(range(len(entries))))   # no raise
    permuted = bank[torch.tensor([1, 0, 2, 3, 4, 5])]
    with pytest.raises(RuntimeError, match="Bank mapping mismatch"):
        verify_bank_mapping(entries, permuted, list(range(len(entries))))


def test_batches_carry_explicit_bank_index_and_matching_targets():
    entries = _tiny_entries()
    bank = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    vocab = build_vocab()
    pop = unique_phonology_indices(entries)          # [0,1,3,4]
    batches = list(make_batches(entries, pop, bank, vocab, batch_size=3,
                                device="cpu"))
    seen = []
    for b in batches:
        seen += b["bank_idx"].tolist()
        for k, i in enumerate(b["bank_idx"].tolist()):
            # target gathered by bank index, not by row position
            assert torch.equal(b["semantic"][k], bank[i])
            assert b["words"][k] == entries[i].word
    assert seen == pop, "flat pass must cover the population exactly once, in order"


def test_flat_batching_covers_population_once_when_shuffled():
    entries = _tiny_entries()
    bank = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    vocab = build_vocab()
    pop = list(range(len(entries)))
    seen = []
    for b in make_batches(entries, pop, bank, vocab, batch_size=4,
                          device="cpu", shuffle=True, seed=3):
        seen += b["bank_idx"].tolist()
    assert sorted(seen) == pop and len(seen) == len(pop)   # no replacement


def test_batch_conventions_match_repository_bos_eos_pad():
    entries = _tiny_entries()
    bank = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    vocab = build_vocab()
    b = next(make_batches(entries, [3], bank, vocab, batch_size=1, device="cpu"))
    form = entries[3].phonemes
    assert b["enc_in"][0, :len(form) + 1].tolist() == form + [vocab.eos_id]
    assert b["dec_in"][0, :len(form) + 1].tolist() == [vocab.bos_id] + form
    assert b["dec_tgt"][0, :len(form) + 1].tolist() == form + [vocab.eos_id]
    assert b["enc_mask"][0, :len(form) + 1].all()


# ---------------------------------------------------------- objectives

def test_retrieval_loss_on_toy_bank_matches_manual_cross_entropy():
    torch.manual_seed(0)
    bank = F.normalize(torch.randn(7, SEM_DIM), dim=-1)
    s_hat = torch.randn(3, SEM_DIM)
    tgt = torch.tensor([2, 5, 0])
    tau = 0.1
    got = retrieval_loss(s_hat, bank, tgt, tau)
    manual = F.cross_entropy((F.normalize(s_hat, dim=-1) @ bank.t()) / tau, tgt)
    assert torch.allclose(got, manual, atol=1e-7)


def test_retrieval_loss_is_near_zero_for_perfect_retrieval():
    bank = F.normalize(torch.eye(8), dim=-1)
    # s_hat exactly on the target rows, scaled: cosine is scale-invariant
    s_hat = 4.0 * bank[torch.tensor([1, 6])]
    loss = retrieval_loss(s_hat, bank, torch.tensor([1, 6]), tau=0.05)
    assert float(loss) < 1e-3
    worse = retrieval_loss(s_hat, bank, torch.tensor([0, 0]), tau=0.05)
    assert float(worse) > float(loss)


def test_retrieval_loss_rejects_nonpositive_tau():
    bank = F.normalize(torch.randn(4, SEM_DIM), dim=-1)
    with pytest.raises(ValueError, match="tau must be"):
        retrieval_loss(torch.randn(2, SEM_DIM), bank, torch.tensor([0, 1]), tau=0.0)


def test_bank_receives_no_gradient():
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    model.set_semantic_bank(bank_raw.clone())
    assert not model.ltm.semantic_bank.requires_grad
    set_trainable_scope(model, "comprehension")
    b = next(make_batches(entries, [0, 1], bank_raw, vocab, 2, "cpu"))
    comprehension_objective(model, b, "c3", 0.1, 1.0)["total"].backward()
    assert model.ltm.semantic_bank.grad is None


def test_c0_is_exactly_the_existing_alignment_loss():
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    model.set_semantic_bank(bank_raw.clone())
    set_trainable_scope(model, "comprehension")
    b = next(make_batches(entries, [0, 1, 3], bank_raw, vocab, 3, "cpu"))
    out = comprehension_objective(model, b, "c0", tau=0.1, lambda_ret=1.0)
    s_hat = comprehension_forward(model, b["enc_in"], b["enc_mask"])
    assert torch.equal(out["c0"], alignment_loss(s_hat, b["semantic"]))
    assert torch.equal(out["total"], out["c0"])       # C0 adds nothing
    assert "retrieval" not in out


def test_c3_equals_c0_plus_lambda_times_retrieval():
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    model.set_semantic_bank(bank_raw.clone())
    set_trainable_scope(model, "comprehension")
    b = next(make_batches(entries, [0, 1, 3], bank_raw, vocab, 3, "cpu"))
    lam = 0.37
    out = comprehension_objective(model, b, "c3", tau=0.1, lambda_ret=lam)
    assert torch.allclose(out["total"], out["c0"] + lam * out["retrieval"], atol=1e-7)


def test_c3_with_lambda_zero_is_numerically_c0():
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    model.set_semantic_bank(bank_raw.clone())
    set_trainable_scope(model, "comprehension")
    b = next(make_batches(entries, [0, 1], bank_raw, vocab, 2, "cpu"))
    c3 = comprehension_objective(model, b, "c3", tau=0.1, lambda_ret=0.0)
    assert torch.allclose(c3["total"], c3["c0"], atol=1e-7)


def test_naming_forward_is_exactly_existing_machinery():
    model, vocab, _ = _tiny_model()
    sem = torch.randn(3, SEM_DIM)
    dec_in = torch.tensor([[vocab.bos_id, 5, 6]] * 3)
    got = naming_forward(model, sem, dec_in)
    expected = model.motor(model.ltm.decode_from_s_hat(sem, dec_in))
    assert torch.equal(got, expected)
    assert got.shape == (3, 3, vocab.size)


def test_comprehension_forward_is_exactly_ltm_encode():
    model, vocab, _ = _tiny_model()
    enc_in = torch.tensor([[5, 6, vocab.eos_id]])
    enc_mask = torch.ones_like(enc_in, dtype=torch.bool)
    assert torch.equal(comprehension_forward(model, enc_in, enc_mask),
                       model.ltm.encode(enc_in, enc_mask))


def test_naming_objective_uses_existing_seq_ce_convention():
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    set_trainable_scope(model, "naming")
    b = next(make_batches(entries, [0, 1, 3], bank_raw, vocab, 3, "cpu"))
    out = naming_objective(model, b, vocab.pad_id)
    logits = naming_forward(model, b["semantic"], b["dec_in"])
    manual = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             b["dec_tgt"].reshape(-1), ignore_index=vocab.pad_id)
    assert torch.allclose(out["total"], manual, atol=1e-7)


def test_naming_input_is_raw_unnormalised_glove():
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    vocab = build_vocab()
    b = next(make_batches(entries, [0, 1], bank_raw, vocab, 2, "cpu"))
    norms = b["semantic"].norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms)), \
        "naming input must be RAW GloVe, not L2-normalised"


# ------------------------------- repetition path / total_loss untouched

def test_total_loss_and_repetition_forward_unchanged():
    """The existing repetition graph must remain bit-compatible: importing the
    task module must not alter total_loss or model.forward behaviour."""
    model, vocab, cfg = _tiny_model()
    enc_in = torch.tensor([[5, 6, vocab.eos_id]])
    enc_mask = torch.ones_like(enc_in, dtype=torch.bool)
    dec_in = torch.tensor([[vocab.bos_id, 5, 6]])
    dec_tgt = torch.tensor([[5, 6, vocab.eos_id]])
    out = model(enc_in, enc_mask, dec_in)
    for k in ("logits", "wm_logits", "ltm_logits", "s_hat", "gate"):
        assert k in out
    batch = {"dec_tgt": dec_tgt, "semantic": torch.randn(1, SEM_DIM)}
    losses = total_loss(out, batch, cfg.loss, vocab.pad_id,
                        usage_prior=cfg.gating.usage_prior)
    assert set(losses) == {"total", "rep", "align", "dec", "wm", "gate"}
    assert torch.isfinite(losses["total"])


def test_no_model_api_added():
    assert not hasattr(DualRouteModel, "forward_comprehension")
    assert not hasattr(DualRouteModel, "forward_naming")
    assert not hasattr(DualRouteModel, "train_task")


# ------------------------------------------- repetition safeguard hooks

@pytest.mark.parametrize("task", ["comprehension", "naming"])
def test_dorsal_fingerprint_guard_passes_after_in_scope_step(task):
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    model.set_semantic_bank(bank_raw.clone())
    set_trainable_scope(model, task)
    optim = fresh_optimizer(model, lr=1e-2, weight_decay=1e-2)
    before = dorsal_fingerprint(model)
    b = next(make_batches(entries, list(range(len(entries))), bank_raw, vocab,
                          6, "cpu"))
    loss = (comprehension_objective(model, b, "c3", 0.1, 1.0)["total"]
            if task == "comprehension" else naming_objective(model, b, vocab.pad_id)["total"])
    optim.zero_grad(); loss.backward(); optim.step()
    assert_dorsal_untouched(model, before)          # must not raise


def test_dorsal_guard_detects_a_tampered_wm_weight():
    model, _, _ = _tiny_model()
    before = dorsal_fingerprint(model)
    with torch.no_grad():
        model.wm.to_premotor.weight.add_(1e-6)
    with pytest.raises(RuntimeError, match="Dorsal/shared parameters changed"):
        assert_dorsal_untouched(model, before)


def test_repetition_snapshot_shape_and_determinism():
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    model.set_semantic_bank(bank_raw.clone())
    idx = list(range(len(entries)))
    s1 = repetition_snapshot(model, vocab, entries, idx, bank_raw, "cpu",
                             batch_size=4)
    s2 = repetition_snapshot(model, vocab, entries, idx, bank_raw, "cpu",
                             batch_size=4)
    assert s1["n_items"] == len(entries)
    assert s1 == s2                                   # deterministic
    prim = s1["primary_readout"]
    assert set(prim["exact_match"]) == {"full", "wm", "ltm"}
    for v in prim["exact_match"].values():
        assert 0.0 <= v <= 1.0
    assert set(s1["auxiliary_teacher_forced"]["exact_match"]) == {"full", "wm", "ltm"}


def test_repetition_primary_readout_is_the_canonical_ar_convention():
    """The primary repetition metric must be the repository's canonical AR
    evaluation, not a locally invented one."""
    from scripts.evaluate_train_lexicon_ceiling import DECODE_AR, EVAL_NOTE_AR
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    snap = repetition_snapshot(model, vocab, entries, [0, 1, 3], bank_raw, "cpu")
    prim = snap["primary_readout"]
    assert prim["decode_mode"] == DECODE_AR
    assert prim["note"] == EVAL_NOTE_AR
    assert "evaluate_forms_ar" in prim["convention"]
    assert prim["wm_noise"] is False


def test_repetition_snapshot_matches_canonical_evaluator_exactly():
    """Byte-for-byte agreement with a direct call to the canonical evaluator."""
    from scripts.evaluate_train_lexicon_ceiling import evaluate_forms_ar
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    idx = [0, 1, 3, 4]
    snap = repetition_snapshot(model, vocab, entries, idx, bank_raw, "cpu",
                               include_teacher_forced=False)
    rows = evaluate_forms_ar(model, vocab, [entries[i] for i in idx], "cpu",
                             routes=("full", "wm", "ltm"), wm_noise=False)
    for r in ("full", "wm", "ltm"):
        expected = sum(x[f"{r}_exact_match"] for x in rows) / len(rows)
        assert snap["primary_readout"]["exact_match"][r] == expected


def test_repetition_snapshot_leaves_model_in_eval_and_unchanged():
    model, vocab, _ = _tiny_model()
    entries = _tiny_entries()
    bank_raw = torch.stack([torch.as_tensor(e.semantic) for e in entries])
    before = parameter_fingerprint(model)
    model.eval()
    repetition_snapshot(model, vocab, entries, [0, 1, 2], bank_raw, "cpu")
    assert not model.training
    assert changed_parameters(model, before) == []
