"""NON-SCIENTIFIC tests of the read-only C-align gradient diagnostic (toy model, synthetic lexicon).
No real state is touched; no optimizer.step() is called anywhere."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lexicon import LexEntry  # noqa: E402
from data.phonemes import build_vocab  # noqa: E402
from models.dual_route import DualRouteModel  # noqa: E402
from scripts.c_align_design import c_align_gradient_diagnostic as G  # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import canonical_config  # noqa: E402


class FakeOpt:
    def __init__(self, model):
        params = list(model.parameters())
        self._sd = {"param_groups": [{"betas": (0.9, 0.999), "eps": 1e-8, "params": list(range(len(params)))}],
                    "state": {i: {"step": torch.tensor(100.0), "exp_avg": torch.zeros_like(p),
                                  "exp_avg_sq": torch.full_like(p, 1e-4)} for i, p in enumerate(params)}}

    def state_dict(self):
        return self._sd


@pytest.fixture(scope="module")
def toy():
    torch.manual_seed(0)
    vocab = build_vocab()
    cfg = canonical_config(0, "cpu", max_words=50, lexicon_path="synthetic", dorsal_pool_size=1, batch_size=8,
                           glove_path=None, wm_hidden=8, enc_hidden=16, dec_hidden=16)
    model = DualRouteModel(cfg, vocab)
    rng = np.random.default_rng(0)
    entries = [LexEntry(word=f"w{k}", phonemes=[int(x) for x in rng.integers(3, vocab.size, size=int(rng.integers(2, 6)))],
                        semantic=(rng.standard_normal(300) * 0.4).astype(np.float32), freq=1.0, rank=k + 1)
               for k in range(30)]
    bank_raw = torch.stack([torch.tensor(e.semantic) for e in entries]).float()
    model.set_semantic_bank(bank_raw)
    model.train(True)
    tr = SimpleNamespace(vocab=vocab, entries=entries, bank_raw=bank_raw, cfg=cfg, model=model, optim=FakeOpt(model))
    return tr


def test_blocks_cover_expected_params_and_seb(toy):
    b = G.Blocks(toy.model)
    assert b.names["phon_embed"] == ["phon_embed.weight"]
    assert all(n.startswith("ltm.encoder.") for n in b.names["ltm_encoder"])
    assert b.names["to_semantic_2"] == ["ltm.to_semantic.2.weight", "ltm.to_semantic.2.bias"]


def test_batch_metrics_read_only_equivalence_and_fields(toy):
    from gate_x_lesion.hooks import state_dict_sha256
    b = G.Blocks(toy.model)
    before = state_dict_sha256(toy.model)
    adam = G.adam_denominator(toy, b)
    rec = G.batch_metrics(toy.model, toy, list(range(8)), b, adam)
    assert state_dict_sha256(toy.model) == before
    assert all(p.grad is None for p in toy.model.parameters())
    assert np.isfinite(rec["L_C_retrieval_raw"]) and np.isfinite(rec["L_C_align_unit"])
    assert abs(rec["L_C_align_components_sum_minus_total"]) < 1e-6
    # C-step alignment and R-step L_align are the same function of the same items
    assert rec["abs_diff_C_align_vs_R_align"] < 1e-6
    assert rec["SEB__equiv_rel_l2_diff"] < 1e-5
    assert abs(rec["SEB__norm_ret_eff"] - G.LAMBDA_C * rec["SEB__norm_ret_raw"]) < 1e-12
    assert abs(rec["SEB__norm_align_w0.1"] - 0.1 * rec["SEB__norm_align_unit"]) < 1e-12
    assert -1.0 <= rec["SEB__cos_ret_eff_vs_align_unit"] <= 1.0
    assert 0 < rec["clip_coef_C_step_V6"] <= 1.0
    for k in ("adam_proxy_SEB__C_ret_eff", "adam_proxy_SEB__C_align_w0.1", "adam_proxy_SEB__R_align_unit"):
        assert np.isfinite(rec[k])
    # retrieval/alignment never reach the decoder: SEB excludes it, and a decoder grad would appear in global norms only
    rec2 = G.batch_metrics(toy.model, toy, list(range(8)), b, adam)
    assert rec2["SEB__norm_ret_raw"] == rec["SEB__norm_ret_raw"]


def test_summaries_and_rule(toy):
    b = G.Blocks(toy.model)
    rows = [{"state_id": "T", "batch_index": i, **G.batch_metrics(toy.model, toy, list(range(i, i + 8)), b, None)}
            for i in range(0, 24, 8)]
    s = G.state_summary(rows)
    c = s["all_438"]["SEB"]["cos_ret_eff_vs_align_unit"]
    assert {"mean", "median", "sd", "min", "max", "q05", "q25", "q75", "q95", "frac_lt_0", "frac_abs_lt_0.1"} <= set(c)
    fake = {"A": {"all_438": {"SEB": {"rho_ref_w0.1_over_ret_eff": {"median": 1.0}, "equiv_rel_l2_diff": {"median": 0.0},
                                      "any_nonfinite": 0, "n_batches_allzero_ret_raw": 0, "n_batches_allzero_align_unit": 0},
                              **{k: {"any_nonfinite": 0} for k in G.BLOCK_PREFIXES}}}}
    assert G.weight_rule(fake, {"g": {"pass": True}})["outcome"] == "RECOMMEND_c_align_weight=0.1"
    fake["A"]["all_438"]["SEB"]["rho_ref_w0.1_over_ret_eff"]["median"] = 11.0
    assert "REQUIRES_CENTRAL_ARBITRATION" in G.weight_rule(fake, {"g": {"pass": True}})["outcome"]
    fake["A"]["all_438"]["SEB"]["rho_ref_w0.1_over_ret_eff"]["median"] = 0.09
    assert "REQUIRES_CENTRAL_ARBITRATION" in G.weight_rule(fake, {"g": {"pass": True}})["outcome"]
    fake["A"]["all_438"]["SEB"]["rho_ref_w0.1_over_ret_eff"]["median"] = 1.0
    fake["A"]["all_438"]["SEB"]["equiv_rel_l2_diff"]["median"] = 1e-3
    assert "REQUIRES_CENTRAL_ARBITRATION" in G.weight_rule(fake, {"g": {"pass": True}})["outcome"]


def test_no_optimizer_step_anywhere_in_diagnostic_source():
    import inspect
    import re
    src = re.sub(r"#.*|\"\"\"[\s\S]*?\"\"\"", "", inspect.getsource(G))
    assert ".step(" not in src and ".backward(" not in src and "zero_grad" not in src
