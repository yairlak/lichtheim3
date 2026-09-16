"""NON-SCIENTIFIC implementation tests for the ventral semantic interface diagnostic.

Mechanics are tested on a randomly initialised toy DualRouteModel with a synthetic
lexicon, so no scientific population, checkpoint behaviour or aggregate result is ever
used as a test oracle.  Real artifacts are only hashed (fast) or, with VI_SLOW=1,
reconstructed structurally.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lexicon import LexEntry                                           # noqa: E402
from data.phonemes import build_vocab                                       # noqa: E402
from models.dual_route import DualRouteModel                                # noqa: E402
from scripts.naming_comprehension.frozen_probe import semantic_greedy_decode  # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import canonical_config  # noqa: E402
from gating_diagnostics.gate_probe import ar_decode_forced_length, ar_decode_free  # noqa: E402
from ventral_interface import CONDITIONS, CONVENTIONS, ROUTE               # noqa: E402
from ventral_interface import conditions as C                              # noqa: E402
from ventral_interface import decode as D                                  # noqa: E402
from ventral_interface import schema as S                                  # noqa: E402
from ventral_interface.evaluate import evaluate_state                      # noqa: E402
from ventral_interface.injection import (                                  # noqa: E402
    ForbiddenRouteAccess, SemanticInjection, injected, state_dict_sha256, ventral_only)

CFG_PATH = os.path.join(ROOT, "paper_programme", "ventral_semantic_interface",
                        "ventral_interface_frozen_config.json")
SLOW = os.environ.get("VI_SLOW") == "1"


# ------------------------------------------------------------------ toy fixtures
@pytest.fixture(scope="module")
def toy():
    torch.manual_seed(0)
    vocab = build_vocab()
    cfg = canonical_config(0, "cpu", max_words=40, lexicon_path="synthetic",
                           dorsal_pool_size=1, batch_size=4, glove_path=None,
                           wm_hidden=8, enc_hidden=16, dec_hidden=16)
    model = DualRouteModel(cfg, vocab).eval()
    rng = np.random.default_rng(0)
    entries = []
    for k in range(30):
        L = int(rng.integers(2, 6))
        ph = [int(x) for x in rng.integers(3, vocab.size, size=L)]
        entries.append(LexEntry(word=f"w{k}", phonemes=ph,
                                semantic=(rng.standard_normal(300) * 0.4).astype(np.float32),
                                freq=1.0, rank=k + 1))
    for k, src in ((30, 3), (31, 7)):          # homophones: same phonology, different word
        entries.append(LexEntry(word=f"w{k}", phonemes=list(entries[src].phonemes),
                                semantic=(rng.standard_normal(300) * 0.4).astype(np.float32),
                                freq=1.0, rank=k + 1))
    bank_raw = torch.stack([torch.tensor(e.semantic) for e in entries]).float()
    model.set_semantic_bank(bank_raw)
    return SimpleNamespace(model=model, vocab=vocab, entries=entries, bank_raw=bank_raw)


def _forms(t, idx):
    return [t.entries[i].phonemes for i in idx]


# ------------------------------------------------------------ artifacts (fast)
def _cfg():
    with open(CFG_PATH) as f:
        return json.load(f)


@pytest.mark.skipif(not os.path.exists("/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/archives"),
                    reason="authoritative artifacts not present on this machine")
def test_authoritative_artifacts_hash_and_identities():
    from scripts.ventral_interface.run_ventral_interface_factorization import sha256_file
    from gate_x_lesion.identity import reconstructed_state_sha256
    cfg = _cfg()
    p = cfg["repository_parent"]
    for s in cfg["states"]:
        src = sha256_file(os.path.join(p, s["source_checkpoint_path"]))
        assert src == s["source_checkpoint_sha256"]
        assert sha256_file(os.path.join(p, s["source_checkpoint_archival_copy"])) == src
        head = sha256_file(os.path.join(p, s["repaired_head_path"])) if s["repaired_head_path"] else ""
        assert (head or None) == s["repaired_head_sha256"]
        assert reconstructed_state_sha256(src, head) == s["reconstructed_state_identity"]
    assert sha256_file(os.path.join(ROOT, cfg["data"]["lexicon_path"])) == cfg["data"]["lexicon_sha256"]


def test_frozen_state_set_is_exactly_four():
    ids = [s["state_id"] for s in _cfg()["states"]]
    assert ids == ["W3_SRC", "W3_REP", "W4_SRC", "W4_REP"]


def test_frozen_decoding_constants_match_code():
    cfg = _cfg()
    from ventral_interface.evaluate import DECODE_BATCH, NAMING_VALIDITY_MAX_STEPS, NAMING_BATCH
    from ventral_interface.population import RETRIEVAL_BATCH
    assert D.FREE_AR_MAX_STEPS == cfg["decoding"]["freear_max_steps"] == 12
    assert DECODE_BATCH == cfg["decoding"]["decode_batch"]
    assert RETRIEVAL_BATCH == cfg["decoding"]["retrieval_batch"]
    assert NAMING_VALIDITY_MAX_STEPS == cfg["decoding"]["naming_validity_max_steps"] == 256
    assert NAMING_BATCH == cfg["decoding"]["naming_validity_batch"]
    assert ROUTE == cfg["decoding"]["route"] == "ltm"
    assert C.S3_DEGENERATE_NORM == cfg["tolerances"]["s3_degenerate_norm"]


# ------------------------------------------------------------- vector semantics
def test_raw_bank_is_unnormalized_and_bank_is_normalized(toy):
    assert torch.equal(toy.model.ltm.semantic_bank,
                       torch.nn.functional.normalize(toy.bank_raw, dim=-1))
    assert not torch.allclose(toy.bank_raw.norm(dim=-1), torch.ones(len(toy.entries)))


def test_s1_equals_s2_literally_when_same_row(toy):
    idx = list(range(len(toy.entries)))
    top1 = [i if i % 2 == 0 else (i + 1) % len(idx) for i in idx]
    g = C.gate_b(toy.bank_raw, idx, top1)
    assert g["pass"] and g["n_equal_row"] == len([i for i in idx if top1[i] == i])
    s1, s2 = C.raw_retrieved(toy.bank_raw, top1), C.raw_true(toy.bank_raw, idx)
    for i in idx:
        if top1[i] == i:
            assert torch.equal(s1[i], s2[i])
    assert g["global_max_abs_diff"] == 0.0


def test_s3_preserves_direction_sets_norm_and_flags_degeneracy(toy):
    torch.manual_seed(1)
    v = torch.randn(10, 300)
    v[3] = 0.0
    target = torch.linspace(2.0, 14.0, 10)
    s3, deg = C.radial(v, target)
    assert deg.tolist() == [k == 3 for k in range(10)]
    assert torch.equal(s3[3], v[3])
    nd = ~deg
    cos = torch.nn.functional.cosine_similarity(s3[nd].double(), v[nd].double(), dim=-1)
    assert float(cos.min()) >= 1 - 1e-6
    assert torch.allclose(s3[nd].norm(dim=-1), target[nd], rtol=1e-5)
    assert C.gate_d(v, s3, deg, toy.model.ltm.semantic_bank)["pass"]


def test_gate_d_detects_direction_change(toy):
    torch.manual_seed(2)
    v = torch.randn(6, 300)
    bad = torch.randn(6, 300)
    assert not C.gate_d(v, bad, torch.zeros(6, dtype=torch.bool), toy.model.ltm.semantic_bank)["pass"]


# ------------------------------------------------------ downstream identity (E)
def _spy(model):
    calls = {"sem_to_h0": [], "decoder": 0, "dec_to_premotor": 0, "motor": 0}
    hs = [model.ltm.sem_to_h0.register_forward_pre_hook(
              lambda m, i: calls["sem_to_h0"].append(i[0].detach().clone())),
          model.ltm.decoder.register_forward_pre_hook(
              lambda m, i: calls.__setitem__("decoder", calls["decoder"] + 1)),
          model.ltm.dec_to_premotor.register_forward_pre_hook(
              lambda m, i: calls.__setitem__("dec_to_premotor", calls["dec_to_premotor"] + 1)),
          model.motor.register_forward_pre_hook(
              lambda m, i: calls.__setitem__("motor", calls["motor"] + 1))]
    return calls, hs


@pytest.mark.parametrize("conv", CONVENTIONS)
def test_all_conditions_use_same_downstream_path_with_only_vector_changed(toy, conv):
    idx = list(range(8))
    forms = _forms(toy, idx)
    S1 = C.raw_retrieved(toy.bank_raw, [(i + 1) % 32 for i in range(32)])
    S2 = C.raw_true(toy.bank_raw, range(32))
    rn = S1.norm(dim=-1)
    counts = {}
    with torch.inference_mode(), ventral_only(toy.model):
        for cond in CONDITIONS:
            inj = {"S0": SemanticInjection("S0"), "S1": SemanticInjection("S1", fixed=S1),
                   "S2": SemanticInjection("S2", fixed=S2),
                   "S3": SemanticInjection("S3", target_norm=rn)}[cond]
            calls, hs = _spy(toy.model)
            try:
                out = D.decode_condition(toy.model, toy.vocab, forms, idx, inj, conv, "cpu")
            finally:
                for h in hs:
                    h.remove()
            n = out["n_steps"]
            counts[cond] = (calls["decoder"], calls["dec_to_premotor"], calls["motor"],
                            len(calls["sem_to_h0"]))
            assert calls["decoder"] == calls["dec_to_premotor"] == calls["motor"] == n
            got = calls["sem_to_h0"][0]
            live = out["live_s_hat"]
            want = {"S0": live, "S1": S1[idx], "S2": S2[idx],
                    "S3": C.radial(live, rn[idx])[0]}[cond]
            assert torch.equal(got, want), cond
    if conv == "canonical":                   # fixed step count: identical call counts
        assert len(set(counts.values())) == 1


def test_s0_passthrough_equals_unhooked_historical_decoder(toy):
    idx = list(range(10))
    forms = _forms(toy, idx)
    with torch.inference_mode():
        for conv, fn in (("freear", ar_decode_free), ("canonical", ar_decode_forced_length)):
            ref = fn(toy.model, toy.vocab, forms, "cpu", routes=("ltm",))["ltm"]
            out = D.decode_condition(toy.model, toy.vocab, forms, idx,
                                     SemanticInjection("S0"), conv, "cpu")
            assert out["preds"] == ref


def test_s2_injection_matches_naming_greedy_decoder_matched_conventions(toy):
    idx = list(range(len(toy.entries)))
    S2 = C.raw_true(toy.bank_raw, idx)
    with torch.inference_mode():
        out = D.decode_condition(toy.model, toy.vocab, _forms(toy, idx), idx,
                                 SemanticInjection("S2", fixed=S2), "freear", "cpu", max_steps=256)
        ref, _ = semantic_greedy_decode(toy.model, S2, toy.vocab, 256)
    assert out["preds"] == ref


@pytest.mark.parametrize("route", ["full", "wm", "fixed05"])
def test_guard_blocks_fusion_gate_and_dorsal(toy, route):
    from evaluate.hooks import make_batch
    from gating_diagnostics.gate_probe import _route_step_logits
    b = make_batch([list(f) for f in _forms(toy, range(4))], toy.vocab, "cpu")
    with torch.inference_mode(), ventral_only(toy.model):
        with pytest.raises(ForbiddenRouteAccess):
            _route_step_logits(toy.model, b["enc_in"], b["enc_mask"], b["dec_in"], route)


def test_execute_path_source_never_names_forbidden_routes():
    import ventral_interface.decode as dm
    import ventral_interface.evaluate as em
    import ventral_interface.injection as im
    import ventral_interface.population as pm
    from scripts.ventral_interface import run_ventral_interface_factorization as drv
    forbidden = re.compile(r"route\s*=\s*\"(full|wm|fixed05)\"|routes\s*=\s*ROUTES\b|"
                           r"fixed_mix_logits|capture_gate_field|wm_logits|forced_gate|"
                           r"collect_item_level")
    def code(src):                            # comments and docstrings removed
        return re.sub(r"#.*|\"\"\"[\s\S]*?\"\"\"", "", src)

    for mod in (dm, em, pm):
        assert not forbidden.search(code(inspect.getsource(mod))), mod.__name__
    assert not forbidden.search(code(inspect.getsource(drv.run_execute)))
    # the injection module supplies vectors only; it contains no decoding logic
    assert "decode" not in code(inspect.getsource(im))


def test_execute_refuses_without_matching_contract_hash():
    from scripts.ventral_interface import run_ventral_interface_factorization as drv
    cfg = drv.load_config()
    with pytest.raises(SystemExit):
        drv.run_execute(cfg, None)
    with pytest.raises(SystemExit):
        drv.run_execute(cfg, "0" * 64)


# ------------------------------------------------------------- AR diagnostic
def test_first_divergence_and_eos_fields():
    eos = 2
    form = [5, 6, 7]
    assert D.first_divergence([5, 6, 7, 2, 9], form, eos) is None
    assert D.first_divergence([5, 9, 7, 2], form, eos) == 1
    assert D.first_divergence([5, 6, 2], form, eos) == 2              # early EOS
    assert D.first_divergence([5, 6, 7, 8, 2], form, eos) == 3        # over-generation
    f = D.eos_fields([5, 6, 2], form, eos, "freear")
    assert f["eos_before_target_length"] == 1 and f["terminated_by_cap"] == 0
    f = D.eos_fields([5, 6, 7, 8, 2], form, eos, "freear")
    assert f["eos_after_target_length"] == 1 and f["pred_length"] == 4
    f = D.eos_fields([5] * 12, form, eos, "freear")
    assert f["terminated_by_cap"] == 1 and f["first_eos_step"] == D.NA
    assert D.eos_fields([5, 6, 7, 8], form, eos, "canonical")["terminated_by_cap"] == D.NA


def test_prefix_correction_is_labelled_and_starts_from_gold(toy):
    form = toy.entries[0].phonemes
    raw = list(form[:1]) + [3] * 11                                 # diverges at t=1
    vec = toy.bank_raw[0]
    with torch.inference_mode():
        r = D.prefix_corrected_continuation(toy.model, toy.vocab, form, raw, 1, vec, "cpu")
    assert r["label"] == D.DIAGNOSTIC_LABEL == "DIAGNOSTIC_ONLY_PREFIX_CORRECTION"
    toks = r["corrected_predicted_phonology"].split()
    gold = [toy.vocab.itos[p] for p in form[:2]]
    assert toks[:2] == gold[:len(toks[:2])]
    # EOS-position divergence: correcting to EOS yields exactly the form
    raw2 = list(form) + [4]
    with torch.inference_mode():
        r2 = D.prefix_corrected_continuation(toy.model, toy.vocab, form, raw2, len(form), vec, "cpu")
    assert r2["corrected_exact"] == 1 and r2["corrected_second_divergence_step"] == D.NA


def test_native_diagnostic_consistency_on_toy(toy):
    idx = list(range(len(toy.entries)))
    forms = _forms(toy, idx)
    with torch.inference_mode():
        out = D.decode_condition(toy.model, toy.vocab, forms, idx,
                                 SemanticInjection("S0"), "freear", "cpu")
        recs = D.native_freear_diagnostic(toy.model, toy.vocab, forms, idx, idx, out,
                                          out["live_s_hat"], "cpu")
    n_fail = sum(1 for k, f in enumerate(forms) if out["preds"][k] != list(f))
    assert len(recs) == n_fail and n_fail > 0
    for r in recs:
        assert r["step_vs_goldprefix_logit_max_abs_dev"] <= D.LOGIT_PATH_TOL
        assert (r["gold_rank"] >= 2 and r["margin_chosen_minus_gold"] >= 0) \
            or r["margin_numerically_ambiguous"] == 1
        assert r["diag_prefix_correction_label"] == D.DIAGNOSTIC_LABEL


# ---------------------------------------------------------------- schemas
def test_schema_deterministic_and_strict():
    rows = [{c: (k if i % 2 else f"x{k}") for i, c in enumerate(S.ITEM_LEVEL_COLUMNS)}
            for k in range(3)]
    assert S.tsv_bytes(S.ITEM_LEVEL_COLUMNS, rows) == S.tsv_bytes(S.ITEM_LEVEL_COLUMNS, rows)
    assert S.schema_descriptor() == S.schema_descriptor()
    assert len(set(S.ITEM_LEVEL_COLUMNS)) == len(S.ITEM_LEVEL_COLUMNS)
    with pytest.raises(RuntimeError):
        S.tsv_bytes(S.ITEM_LEVEL_COLUMNS, [dict(rows[0], extra=1)])
    bad = dict(rows[0]); bad.pop("item_index")
    with pytest.raises(RuntimeError):
        S.tsv_bytes(S.ITEM_LEVEL_COLUMNS, [bad])
    for req in ("C_CORRECT_AND_NATIVE_LTM_WRONG", "C_WRONG_AND_NATIVE_LTM_WRONG",
                "C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT"):
        assert req in S.STRATA


def test_synthetic_end_to_end_is_non_scientific_and_schema_exact(toy):
    tr = SimpleNamespace(vocab=toy.vocab, entries=toy.entries, bank_raw=toy.bank_raw)
    meta = {"witness_id": "TOY", "state_id": "TOY_SRC", "source_or_repaired": "SOURCE",
            "seed": 0, "source_u": 0, "source_checkpoint_sha256": "0" * 64,
            "repaired_head_sha256": None, "reconstructed_state_identity": "0" * 64}
    before = state_dict_sha256(toy.model)
    out = evaluate_state(toy.model, tr, meta, None, synthetic=True)
    assert out["non_scientific"] is True
    assert state_dict_sha256(toy.model) == before
    assert set(out["gates"]) == {"R", "A", "B", "D", "C", "SHAT", "H", "E"}
    for r in out["rows"]:
        assert list(r) == [c for c in r] and set(r) == set(S.ITEM_LEVEL_COLUMNS)
        if r["in_C_population"] == 0:
            assert r["C_contract_correct"] == D.NA
    for r in out["ar_diagnostic"]:
        assert set(r) == set(S.AR_DIAGNOSTIC_COLUMNS)
    S.tsv_bytes(S.ITEM_LEVEL_COLUMNS, out["rows"])
    S.tsv_bytes(S.AR_DIAGNOSTIC_COLUMNS, out["ar_diagnostic"])
    summ = S.summarize(out["rows"])
    blob = json.dumps(summ, sort_keys=True)
    assert "diag" not in blob                   # prefix correction never in primary metrics
    assert blob == json.dumps(S.summarize(out["rows"]), sort_keys=True)
    homo = [r for r in out["rows"] if r["homophone_group_size"] > 1]
    assert len(homo) == 4 and sum(r["in_C_population"] for r in homo) == 2


def test_evaluate_state_refuses_missing_archived_values_outside_synthetic(toy):
    tr = SimpleNamespace(vocab=toy.vocab, entries=toy.entries, bank_raw=toy.bank_raw)
    with pytest.raises(RuntimeError):
        evaluate_state(toy.model, tr, {}, None, synthetic=False)


# ------------------------------------------------------------ slow: real states
@pytest.mark.skipif(not SLOW, reason="set VI_SLOW=1 (reconstructs the four real states)")
def test_preflight_twice_is_identical_and_passes():
    from scripts.ventral_interface import run_ventral_interface_factorization as drv
    os.chdir(ROOT)
    cfg = drv.load_config()
    a = drv.run_preflight(cfg, write=False)
    b = drv.run_preflight(cfg, write=False)
    assert a["PREFLIGHT"] == "PASS"
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
