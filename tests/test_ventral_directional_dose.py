"""NON-SCIENTIFIC tests for the directional-dose diagnostic.

Numerical tests use synthetic vectors; pipeline tests use a randomly initialised toy
DualRouteModel with a synthetic lexicon whose "immutable controls" are generated in the test.
No real state is decoded under any scientific alpha here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lexicon import LexEntry                                            # noqa: E402
from data.phonemes import build_vocab                                        # noqa: E402
from models.dual_route import DualRouteModel                                 # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import canonical_config  # noqa: E402
from ventral_directional_dose import ALPHAS, ALPHA_KEYS, CONVENTIONS         # noqa: E402
from ventral_directional_dose import dose_math as M                          # noqa: E402
from ventral_directional_dose import schema as S                             # noqa: E402
from ventral_directional_dose.controls import load_immutable_controls, sha256_file  # noqa: E402
from ventral_directional_dose.evaluate import GateFailure, execute_state, preflight_state  # noqa: E402
from ventral_directional_dose.supplier import DoseInjection, sem_to_h0_probe  # noqa: E402
from ventral_interface.decode import decode_condition                        # noqa: E402
from ventral_interface.injection import (                                    # noqa: E402
    ForbiddenRouteAccess, SemanticInjection, state_dict_sha256, ventral_only)

CFG = os.path.join(ROOT, "paper_programme", "ventral_semantic_directional_dose",
                   "ventral_directional_dose_frozen_config.json")


def _pair(cos_target: float, D: int = 300, seed: int = 0, ns: float = 7.0, nv: float = 5.0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(D, generator=g, dtype=torch.float64)
    uv = v / v.norm()
    z = torch.randn(D, generator=g, dtype=torch.float64)
    z = z - (z @ uv) * uv
    z = z / z.norm()
    u = cos_target * uv + math.sqrt(max(0.0, 1 - cos_target ** 2)) * z
    return (u * ns).float().unsqueeze(0), (uv * nv).float().unsqueeze(0)


def _angle(a, b):
    return float(M.angle_between(a, b)[0])


# ------------------------------------------------------------------ 1-10 numerical specification
@pytest.mark.parametrize("cos_target", [-0.9, -0.2, 0.3, 0.74, 0.99])
def test_01_ordinary_slerp_endpoints_and_angular_fraction(cos_target):
    s, v = _pair(cos_target)
    theta = _angle(s, v)
    for a in ALPHAS:
        sa = M.dose_vectors(s, v, a)
        assert abs(_angle(s, sa) - a * theta) <= 1e-6
    assert _angle(M.dose_vectors(s, v, 1.0), v) <= 1e-6


def test_02_norm_preservation():
    for seed in range(20):
        s, v = _pair(0.6, seed=seed, ns=3.0 + seed, nv=11.0)
        for a in ALPHAS:
            sa = M.dose_vectors(s, v, a)
            assert abs(float(sa.double().norm() / s.double().norm()) - 1) <= 1e-6


def test_03_float64_arithmetic_then_single_float32_cast():
    s, v = _pair(0.5)
    sa = M.dose_vectors(s, v, 0.5)
    assert sa.dtype == torch.float32
    s64, v64 = s.double(), v.double()
    us, up = s64 / s64.norm(), v64 / v64.norm()
    d = torch.clamp((us * up).sum(), -1, 1)
    r = up - d * us
    th = torch.arccos(d)
    u = torch.cos(0.5 * th) * us + torch.sin(0.5 * th) * r / r.norm()
    u = u / u.norm()
    assert torch.equal(sa, (s64.norm() * u).to(torch.float32))


def test_04_alpha1_prototype_direction_native_norm():
    s, v = _pair(0.3, ns=9.0, nv=4.0)
    s1 = M.dose_vectors(s, v, 1.0)
    assert 1 - float(M._unit64(s1)[0] @ M._unit64(v)[0]) <= 1e-6
    assert abs(float(s1.double().norm()) - float(s.double().norm())) / float(s.double().norm()) <= 1e-6


def test_05_alpha1_is_not_S1_when_norms_differ():
    s, v = _pair(0.3, ns=9.0, nv=4.0)
    s1 = M.dose_vectors(s, v, 1.0)
    assert not torch.allclose(s1, v, atol=1e-3)
    assert abs(float(s1.norm()) - float(v.norm())) > 1.0


def test_06_near_collinear_nlerp():
    s, v = _pair(1.0, ns=6.0, nv=3.0)
    s = s + torch.tensor(1e-9)
    geom = M.base_geometry(s, v)
    assert bool(geom["near_collinear"][0]) and not bool(geom["ordinary"][0])
    for a in ALPHAS:
        sa = M.dose_vectors(s, v, a)
        us, up = M._unit64(s)[0], M._unit64(v)[0]
        w = (1 - a) * us + a * up
        exp = (s.double().norm() * w / w.norm()).float()
        assert torch.equal(sa[0], exp)
        assert M.case_labels(geom) == [M.NEAR_COLLINEAR]


def test_07_zero_shat_hard_stop():
    s, v = _pair(0.5)
    with pytest.raises(M.DoseHardStop) as e:
        M.dose_vectors(torch.zeros_like(s), v, 0.5)
    assert e.value.kind == "ZERO_SHAT"


def test_08_zero_prototype_hard_stop():
    s, v = _pair(0.5)
    with pytest.raises(M.DoseHardStop) as e:
        M.dose_vectors(s, torch.zeros_like(v), 0.5)
    assert e.value.kind == "ZERO_PROTOTYPE"


@pytest.mark.parametrize("perturb", [0.0, 1e-9])
def test_09_near_antipodal_hard_stop(perturb):
    s, v = _pair(1.0, ns=6.0, nv=3.0)
    anti = -s + perturb
    geom = M.base_geometry(anti, v)
    assert bool(geom["near_antipodal"][0])
    for a in ALPHAS:
        with pytest.raises(M.DoseHardStop) as e:
            M.dose_vectors(anti, v, a)
        assert e.value.kind == "NEAR_ANTIPODAL"


def test_10_alpha0_exact_tensor_short_circuit():
    s, v = _pair(0.5)
    assert M.dose_vectors(s, v, 0.0) is s
    inj = DoseInjection(0.0, v)
    inj.bind([0])
    live = s.clone()
    assert inj._supply(live) is live
    with pytest.raises(ValueError):
        M.dose_vectors(s, v, 0.4)
    with pytest.raises(ValueError):
        DoseInjection(0.3, v)


# ------------------------------------------------------------------ toy pipeline fixtures
@pytest.fixture(scope="module")
def toy():
    torch.manual_seed(0)
    vocab = build_vocab()
    cfg = canonical_config(0, "cpu", max_words=40, lexicon_path="synthetic", dorsal_pool_size=1,
                           batch_size=4, glove_path=None, wm_hidden=8, enc_hidden=16, dec_hidden=16)
    model = DualRouteModel(cfg, vocab).eval()
    rng = np.random.default_rng(0)
    entries = []
    for k in range(40):
        L = int(rng.integers(2, 6))
        entries.append(LexEntry(word=f"w{k}", phonemes=[int(x) for x in rng.integers(3, vocab.size, size=L)],
                                semantic=(rng.standard_normal(300) * 0.4).astype(np.float32), freq=1.0, rank=k + 1))
    bank_raw = torch.stack([torch.tensor(e.semantic) for e in entries]).float()
    model.set_semantic_bank(bank_raw)
    tr = SimpleNamespace(vocab=vocab, entries=entries, bank_raw=bank_raw)
    meta = {"witness_id": "TOY", "state_id": "TOY_SRC", "source_or_repaired": "SOURCE", "seed": 0,
            "source_u": 0, "source_checkpoint_sha256": "0" * 64, "repaired_head_sha256": "NA",
            "reconstructed_state_identity": "1" * 64}
    from ventral_interface.population import retrieval_all_rows
    top1 = [int(x) for x in retrieval_all_rows(model, tr)["metrics"]["top1_idx"]]
    n = len(entries)
    idx = list(range(n))
    forms = [e.phonemes for e in entries]
    ctrl_decodes = {}
    with torch.inference_mode():
        for name, inj_fn in (("S0", lambda: SemanticInjection("S0")),
                             ("S1", lambda: SemanticInjection("S1", fixed=bank_raw[torch.tensor(top1)].contiguous()))):
            for c in CONVENTIONS:
                # S1 fixed table is indexed by item row via bind(idx)
                ctrl_decodes[(name, c)] = decode_condition(model, vocab, forms, idx, inj_fn(), c, "cpu")["preds"]
    controls = []
    for i, e in enumerate(entries):
        r = {k: str(v) for k, v in meta.items()}
        r.update({"item_index": str(i), "lexical_identity": e.word, "canonical_C_index": str(i),
                  "canonical_C_identity": e.word, "in_C_population": "1",
                  "target_phonology": " ".join(vocab.itos[p] for p in e.phonemes),
                  "phoneme_length": str(len(e.phonemes)), "homophone_group": str(i), "homophone_group_size": "1",
                  "retrieved_index": str(top1[i]), "retrieved_lexical_identity": entries[top1[i]].word,
                  "retrieved_phonology": " ".join(vocab.itos[p] for p in entries[top1[i]].phonemes),
                  "C_contract_correct": str(int(top1[i] == i)), "lexical_identity_correct": str(int(top1[i] == i)),
                  "phonology_correct": str(int(entries[top1[i]].phonemes == e.phonemes))})
        for c in CONVENTIONS:
            s0 = int(ctrl_decodes[("S0", c)][i] == list(e.phonemes))
            s1 = int(ctrl_decodes[("S1", c)][i] == list(e.phonemes))
            r[f"S0_{c}_exact_correct"] = str(s0)
            r[f"S1_{c}_exact_correct"] = str(s1)
            r[f"S2_{c}_exact_correct"] = "1"
            r[f"S3_{c}_exact_correct"] = "0"
            r[f"S0_{c}_predicted_phonology"] = " ".join(vocab.itos[p] for p in ctrl_decodes[("S0", c)][i])
            r[f"prev_S1_rescue_{c}"] = str(int(s0 == 0 and s1 == 1))
        controls.append(r)
    return SimpleNamespace(model=model, tr=tr, meta=meta, controls=controls, top1=top1)


def _retr(model, tr):
    from ventral_interface.population import retrieval_all_rows
    return [int(x) for x in retrieval_all_rows(model, tr)["metrics"]["top1_idx"]]


# ------------------------------------------------------------------ 11-16 pipeline
def test_11_immutable_retrieval_index_reuse(toy):
    pre = preflight_state(toy.model, toy.tr, toy.meta, toy.controls, None, retrieval_fn=_retr)
    assert pre["retrieved"] == toy.top1
    bad = [dict(r) for r in toy.controls]
    bad[0]["retrieved_index"] = str((toy.top1[0] + 1) % len(bad))
    with pytest.raises(GateFailure) as e:
        preflight_state(toy.model, toy.tr, toy.meta, bad, None, retrieval_fn=_retr)
    assert e.value.gate == "DOSE-J_retrieval_identity"


def test_12_supplier_reaches_the_frozen_to_semantic_hook(toy):
    idx = list(range(8))
    forms = [toy.tr.entries[i].phonemes for i in idx]
    table = toy.tr.bank_raw[torch.tensor(toy.top1)]
    for a in (0.0,) + ALPHAS:
        inj = DoseInjection(a, table)
        seen = []
        h = toy.model.ltm.to_semantic.register_forward_hook(lambda m, i, o: seen.append(1))
        try:
            with torch.inference_mode(), sem_to_h0_probe(toy.model, inj) as probe:
                decode_condition(toy.model, toy.tr.vocab, forms, idx, inj, "canonical", "cpu")
        finally:
            h.remove()
        assert inj.n_calls == len(seen) > 0 and inj.n_supply == inj.n_calls
        assert probe.n_calls > 0 and probe.n_mismatch == 0


def test_13_identical_downstream_decoder_calls(toy):
    idx = list(range(10))
    forms = [toy.tr.entries[i].phonemes for i in idx]
    table = toy.tr.bank_raw[torch.tensor(toy.top1)]
    counts = {}
    for a in (0.0,) + ALPHAS:
        inj = DoseInjection(a, table)
        calls = {"dec": 0, "prem": 0, "motor": 0}
        hs = [toy.model.ltm.decoder.register_forward_pre_hook(lambda m, i: calls.__setitem__("dec", calls["dec"] + 1)),
              toy.model.ltm.dec_to_premotor.register_forward_pre_hook(lambda m, i: calls.__setitem__("prem", calls["prem"] + 1)),
              toy.model.motor.register_forward_pre_hook(lambda m, i: calls.__setitem__("motor", calls["motor"] + 1))]
        try:
            with torch.inference_mode(), ventral_only(toy.model):
                out = decode_condition(toy.model, toy.tr.vocab, forms, idx, inj, "canonical", "cpu")
        finally:
            for h in hs:
                h.remove()
        counts[a] = (calls["dec"], calls["prem"], calls["motor"], out["n_steps"])
        if a != 0.0:
            expected = M.dose_vectors(out["live_s_hat"], table[idx], a)
            assert torch.equal(inj.last_supplied, expected)
    assert len(set(counts.values())) == 1


@pytest.mark.parametrize("route", ["full", "wm", "fixed05"])
def test_14_forbidden_routes_blocked(toy, route):
    from evaluate.hooks import make_batch
    from gating_diagnostics.gate_probe import _route_step_logits
    b = make_batch([list(e.phonemes) for e in toy.tr.entries[:4]], toy.tr.vocab, "cpu")
    with torch.inference_mode(), ventral_only(toy.model):
        with pytest.raises(ForbiddenRouteAccess):
            _route_step_logits(toy.model, b["enc_in"], b["enc_mask"], b["dec_in"], route)


def test_15_16_end_to_end_toy_parameters_schema_determinism(toy):
    before = state_dict_sha256(toy.model)
    pre = preflight_state(toy.model, toy.tr, toy.meta, toy.controls, before, retrieval_fn=_retr)
    out = execute_state(toy.model, toy.tr, toy.meta, toy.controls, pre)
    assert state_dict_sha256(toy.model) == before
    assert set(out["gates"]) >= {"DOSE-C_norm_preservation", "DOSE-D_alpha1_direction", "DOSE-E_angular",
                                 "DOSE-F_shared_path", "DOSE-G_no_forbidden_access", "DOSE-I_parameters",
                                 "DOSE-J_live_shat"}
    assert all(g["pass"] for g in out["gates"].values())
    for r in out["rows"]:
        assert list(r) == list(r) and set(r) == set(S.ITEM_COLUMNS)
    assert len(set(S.ITEM_COLUMNS)) == len(S.ITEM_COLUMNS)
    assert not any("a000" in c for c in S.ITEM_COLUMNS)
    assert S.tsv_bytes(S.ITEM_COLUMNS, out["rows"]) == S.tsv_bytes(S.ITEM_COLUMNS, out["rows"])
    summ = S.summarize(out["rows"])
    assert json.dumps(summ, sort_keys=True) == json.dumps(S.summarize(out["rows"]), sort_keys=True)
    for c in CONVENTIONS:
        allc = summ["TOY_SRC"][c]["ALL_REPETITION_ITEMS"]
        for a in ALPHAS:
            cell = allc[ALPHA_KEYS[a]]
            assert sum(cell[t] for t in S.TRANSITIONS) == cell["denominator"]
            assert cell["previous_S1_rescues"] == sum(int(r[f"prev_S1_rescue_{c}"]) for r in toy.controls)
    assert set(summ["TOY_SRC"]["freear"]) == set(S.STRATA)
    f = S.alpha1_factorization(out["rows"])
    assert f["TOY_SRC"]["freear"]["n"] == len(toy.controls)
    # alpha=1 uses the native norm, not the S1 vector
    for r in out["rows"]:
        assert abs(float(r["a100_s_alpha_norm"]) - float(r["shat_norm"])) / float(r["shat_norm"]) <= 1e-6


def test_preflight_never_decodes_a_scientific_alpha(toy, monkeypatch):
    import ventral_directional_dose.evaluate as E
    seen = []
    real = E.DoseInjection

    class Spy(real):
        def __init__(self, alpha, table):
            seen.append(alpha)
            super().__init__(alpha, table)
    monkeypatch.setattr(E, "DoseInjection", Spy)
    preflight_state(toy.model, toy.tr, toy.meta, toy.controls, None, retrieval_fn=_retr)
    assert seen and set(seen) == {0.0}


def test_preflight_hard_stop_zero_shat_blocks_before_any_decode(toy, monkeypatch):
    import ventral_directional_dose.evaluate as E
    monkeypatch.setattr(E, "encode_all", lambda *a, **k: torch.zeros(len(toy.tr.entries), 300))
    decoded = []
    monkeypatch.setattr(E, "decode_condition", lambda *a, **k: decoded.append(1))
    with pytest.raises(M.DoseHardStop) as e:
        preflight_state(toy.model, toy.tr, toy.meta, toy.controls, None, retrieval_fn=_retr)
    assert e.value.kind == "ZERO_SHAT" and not decoded and e.value.geometry["counts"]["ZERO_SHAT"] == 40


# ------------------------------------------------------------------ 17 immutable controls
def test_17_previous_outputs_read_only_and_hash_checked(tmp_path):
    cfg = json.load(open(CFG))
    ic = cfg["immutable_controls"]["item_level"]
    path = os.path.join(ROOT, ic["path"])
    before = sha256_file(path)
    assert before == ic["sha256"]
    ctrl = load_immutable_controls(path, ic["sha256"])
    assert sorted(ctrl) == ["W3_REP", "W3_SRC", "W4_REP", "W4_SRC"]
    assert all(len(v) == 29571 for v in ctrl.values())
    assert sha256_file(path) == before
    with pytest.raises(RuntimeError):
        load_immutable_controls(path, "0" * 64)
    for key in ("closed_contract", "summary", "ar_diagnostic"):
        assert sha256_file(os.path.join(ROOT, cfg["immutable_controls"][key]["path"])) == cfg["immutable_controls"][key]["sha256"]


# ------------------------------------------------------------------ 18-20 execute refusals
def _frozen_env(tmp_path):
    from scripts.ventral_directional_dose import run_directional_dose as drv
    contract = tmp_path / "contract.md"
    contract.write_text("CONTRACT_STATUS=FROZEN\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files_sha256": {}}))
    cfg = dict(drv.load_config())
    cfg["contract_path"] = str(contract)
    return drv, cfg, hashlib.sha256(contract.read_bytes()).hexdigest(), str(manifest)


def test_18_execute_refuses_without_matching_contract_hash(tmp_path):
    drv, cfg, sha, man = _frozen_env(tmp_path)
    with pytest.raises(SystemExit):
        drv.check_execute_preconditions(None, "a" * 40, cfg, manifest_path=man)
    with pytest.raises(SystemExit):
        drv.check_execute_preconditions("0" * 64, "a" * 40, cfg, manifest_path=man)
    real = drv.load_config()
    with pytest.raises(SystemExit):                     # the real contract is not FROZEN yet / hash differs
        drv.check_execute_preconditions("0" * 64, "a" * 40, real)


def test_18b_execute_refuses_unfrozen_status_even_if_frozen_is_mentioned(tmp_path):
    drv, cfg, _, man = _frozen_env(tmp_path)
    c = tmp_path / "amended.md"
    c.write_text("CONTRACT_STATUS=AMENDED_PER_CENTRAL_NOT_YET_FROZEN\n* becomes `CONTRACT_STATUS=FROZEN` later\n")
    cfg["contract_path"] = str(c)
    sha = hashlib.sha256(c.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="not FROZEN"):
        drv.check_execute_preconditions(sha, "a" * 40, cfg, manifest_path=man)


def test_19_execute_refuses_if_worktree_dirty(tmp_path):
    drv, cfg, sha, man = _frozen_env(tmp_path)
    fake = lambda *a: "a" * 40 if a[0] == "rev-parse" else " M something.py"  # noqa: E731
    with pytest.raises(SystemExit, match="not clean"):
        drv.check_execute_preconditions(sha, "a" * 40, cfg, git_fn=fake, exec_dir=str(tmp_path / "absent"),
                                        manifest_path=man)
    clean = lambda *a: "a" * 40 if a[0] == "rev-parse" else ""  # noqa: E731
    with pytest.raises(SystemExit, match="HEAD"):
        drv.check_execute_preconditions(sha, "b" * 40, cfg, git_fn=clean, exec_dir=str(tmp_path / "absent"),
                                        manifest_path=man)
    ok = drv.check_execute_preconditions(sha, "a" * 40, cfg, git_fn=clean, exec_dir=str(tmp_path / "absent"),
                                         manifest_path=man)
    assert ok["contract_sha256"] == sha


def test_20_execute_refuses_if_output_dir_exists(tmp_path):
    drv, cfg, sha, man = _frozen_env(tmp_path)
    clean = lambda *a: "a" * 40 if a[0] == "rev-parse" else ""  # noqa: E731
    exists = tmp_path / "scientific_execution"
    exists.mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        drv.check_execute_preconditions(sha, "a" * 40, cfg, git_fn=clean, exec_dir=str(exists), manifest_path=man)


def test_manifest_hash_mismatch_refused(tmp_path):
    drv, cfg, sha, _ = _frozen_env(tmp_path)
    man = tmp_path / "bad_manifest.json"
    man.write_text(json.dumps({"files_sha256": {"ventral_directional_dose/__init__.py": "0" * 64}}))
    with pytest.raises(SystemExit, match="manifest"):
        drv.check_execute_preconditions(sha, "a" * 40, cfg, manifest_path=str(man))


def test_frozen_config_constants_match_code():
    import ventral_directional_dose as P
    cfg = json.load(open(CFG))
    assert tuple(cfg["alphas"]) == P.ALPHAS == (0.25, 0.5, 0.75, 1.0)
    t = cfg["tolerances"]
    assert (t["EPS_NORM"], t["EPS_ORTHO"], t["TOL_NORM_REL"], t["TOL_ENDPOINT_COS"], t["TOL_ANGLE_RAD"],
            t["TOL_MONOTONE_RAD"], t["TOL_LIVE_SHAT"]) == (P.EPS_NORM, P.EPS_ORTHO, P.TOL_NORM_REL,
                                                          P.TOL_ENDPOINT_COS, P.TOL_ANGLE_RAD,
                                                          P.TOL_MONOTONE_RAD, P.TOL_LIVE_SHAT)
    assert [s["state_id"] for s in cfg["states"]] == list(P.STATES)
    from ventral_interface.decode import FREE_AR_MAX_STEPS
    assert FREE_AR_MAX_STEPS == cfg["decoding"]["freear_max_steps"] == 12
