"""Synthetic / toy tests for the Lesioning V2 operator.

No P1-P4 checkpoint is loaded and no nonzero lesion is run on a final model.
Every behavioural check uses a small synthetic module.
"""
from __future__ import annotations

import ast
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
sys.path.insert(0, REPO)

from paper_programme.lesioning_v2.lesion_operator import (  # noqa: E402
    battery, context, guard, masks, noise, seeds, sd_procedure)
from paper_programme.lesioning_v2.lesion_operator.sites import (  # noqa: E402
    L1, L2, L3, NEVER_LESIONED, SITES)

SHA_A = "a" * 64
SHA_B = "b" * 64


class ToyModel(torch.nn.Module):
    """A miniature stand-in with the same tensor names as the real model."""

    def __init__(self):
        super().__init__()
        self.phon_embed = torch.nn.Embedding(42, 64)
        self.wm = torch.nn.Module()
        self.wm.encoder = torch.nn.GRU(64, 128, batch_first=True)
        self.wm.phon_embed = self.phon_embed
        self.ltm = torch.nn.Module()
        self.ltm.encoder = torch.nn.GRU(64, 512, batch_first=True)
        self.ltm.sem_to_h0 = torch.nn.Linear(300, 512)
        self.ltm.decoder = torch.nn.GRU(64, 512, batch_first=True)
        self.ltm.phon_embed = self.phon_embed


# 1-3. exact logical-edge counts -------------------------------------------
def test_logical_edge_count_L1():
    assert L1.n_logical_edges == 128 * 64 == 8_192
    assert L1.n_scalar_coefficients == 24_576
    assert L1.expected_shape == (384, 64)


def test_logical_edge_count_L2():
    assert L2.n_logical_edges == 512 * 64 == 32_768
    assert L2.n_scalar_coefficients == 98_304
    assert L2.expected_shape == (1536, 64)


def test_logical_edge_count_L3():
    assert L3.n_logical_edges == 512 * 300 == 153_600
    assert L3.n_scalar_coefficients == 153_600
    assert L3.expected_shape == (512, 300)


# 4. synchronized 3-block GRU masking --------------------------------------
def test_gru_three_block_masking_is_synchronized():
    perm = masks.edge_permutation(SHA_A, L1, 0)
    logical = masks.logical_mask(L1, perm, 8)
    phys = masks.physical_mask(L1, logical)
    H = L1.hidden
    assert phys.shape == (3 * H, 64)
    for b in range(3):
        assert torch.equal(phys[b * H:(b + 1) * H], logical)
    zeros = (logical == 0).nonzero()
    h, d = int(zeros[0][0]), int(zeros[0][1])
    assert all(phys[b * H + h, d] == 0.0 for b in range(3))


def test_linear_site_mask_is_not_tiled():
    perm = masks.edge_permutation(SHA_A, L3, 0)
    phys = masks.physical_mask(L3, masks.logical_mask(L3, perm, 5))
    assert phys.shape == (512, 300)


# 5-6. nesting and exact counts --------------------------------------------
@pytest.mark.parametrize("site", [L1, L2, L3])
def test_nested_prefix_across_all_severities(site):
    perm = masks.edge_permutation(SHA_A, site, 3)
    removed = []
    for k in range(1, 16):
        m = masks.logical_mask(site, perm, k).flatten()
        removed.append(set((m == 0).nonzero().flatten().tolist()))
    for a, b in zip(removed, removed[1:]):
        assert a <= b, "severities are not nested"
    assert len(removed[-1]) == round(0.30 * site.n_logical_edges)


@pytest.mark.parametrize("site", [L1, L2, L3])
def test_exact_count_at_every_severity(site):
    perm = masks.edge_permutation(SHA_A, site, 0)
    for k in range(1, 16):
        m = masks.logical_mask(site, perm, k)
        want = round(0.30 * (k / 15) * site.n_logical_edges)
        assert int((m == 0).sum()) == want == masks.n_removed(site, k)


def test_k0_removes_nothing():
    perm = masks.edge_permutation(SHA_A, L1, 0)
    assert int((masks.logical_mask(L1, perm, 0) == 0).sum()) == 0
    assert masks.n_removed(L1, 0) == 0


def test_severity_formula_and_p_max():
    assert masks.N_SEVERITY_LEVELS == 15
    assert masks.CONNECTIVITY_P_MAX == 0.30
    assert masks.severity_fraction(15) == 1.0
    assert masks.connectivity_fraction(15) == pytest.approx(0.30)


# 7-10. determinism and independence ---------------------------------------
def test_mask_reproduces_deterministically():
    a = masks.build_mask(SHA_A, L1, 2, 7)[L1.connectivity_param]
    b = masks.build_mask(SHA_A, L1, 2, 7)[L1.connectivity_param]
    assert torch.equal(a, b)


def test_realizations_differ():
    a = masks.edge_permutation(SHA_A, L1, 0)
    b = masks.edge_permutation(SHA_A, L1, 1)
    assert not torch.equal(a, b)


def test_states_differ():
    a = masks.edge_permutation(SHA_A, L1, 0)
    b = masks.edge_permutation(SHA_B, L1, 0)
    assert not torch.equal(a, b)


def test_sites_differ():
    a = masks.logical_mask(L1, masks.edge_permutation(SHA_A, L1, 0), 5)
    b = masks.logical_mask(L2, masks.edge_permutation(SHA_A, L2, 0), 5)
    assert a.shape != b.shape
    pa = seeds.mask_payload(SHA_A, "L1", 0)
    pb = seeds.mask_payload(SHA_A, "L2", 0)
    assert seeds.derive_seed(pa) != seeds.derive_seed(pb)


def test_severity_task_and_decoder_never_enter_mask_identity():
    p = seeds.mask_payload(SHA_A, "L1", 0)
    for token in ("severity", "task", "decod", "item", "k=", "free_ar"):
        assert token not in p.lower()


# 11-13. untouchables -------------------------------------------------------
def test_biases_weight_hh_and_phon_embed_are_untouched():
    m = ToyModel()
    before = {n: context.parameter_digest(m, (n,)) for n in NEVER_LESIONED
              if n in dict(m.state_dict())}
    cell = masks.build_mask(SHA_A, L1, 0, 15)
    with context.connectivity_lesion(m, cell):
        for n, d in before.items():
            assert context.parameter_digest(m, (n,)) == d, f"{n} changed"
    for n, d in before.items():
        assert context.parameter_digest(m, (n,)) == d


def test_never_lesioned_list_covers_the_expected_tensors():
    for n in ("wm.encoder.weight_hh_l0", "ltm.encoder.weight_hh_l0",
              "ltm.sem_to_h0.bias", "ltm.decoder.weight_ih_l0",
              "ltm.decoder.weight_hh_l0", "phon_embed.weight",
              "ltm.to_semantic.2.weight", "motor.proj.weight"):
        assert n in NEVER_LESIONED
    for s in SITES.values():
        assert s.connectivity_param not in NEVER_LESIONED


# 14. L3 touches sem_to_h0 only ---------------------------------------------
def test_L3_lesion_touches_only_sem_to_h0_weight():
    m = ToyModel()
    dec_before = context.parameter_digest(m, ("ltm.decoder.weight_ih_l0",
                                              "ltm.decoder.weight_hh_l0"))
    bias_before = context.parameter_digest(m, ("ltm.sem_to_h0.bias",))
    cell = masks.build_mask(SHA_A, L3, 0, 15)
    assert set(cell) == {"ltm.sem_to_h0.weight"}
    with context.connectivity_lesion(m, cell):
        assert context.parameter_digest(
            m, ("ltm.decoder.weight_ih_l0", "ltm.decoder.weight_hh_l0")) == dec_before
        assert context.parameter_digest(m, ("ltm.sem_to_h0.bias",)) == bias_before
        w = m.ltm.sem_to_h0.weight
        assert int((w == 0).sum()) >= masks.n_removed(L3, 15)


# 15-18. activation noise ---------------------------------------------------
def test_item_noise_base_is_frozen_and_reproducible():
    a = noise.base_draw(SHA_A, "L1", 0, "bank_7", (1, 128))
    b = noise.base_draw(SHA_A, "L1", 0, "bank_7", (1, 128))
    assert torch.equal(a, b)
    c = noise.base_draw(SHA_A, "L1", 0, "bank_8", (1, 128))
    assert not torch.equal(a, c)


def test_same_base_noise_across_all_severities():
    u = noise.base_draw(SHA_A, "L2", 1, "bank_3", (1, 512))
    sd = 0.4
    etas = [noise.eta(u, k, sd) for k in range(1, 16)]
    for k, e in zip(range(1, 16), etas):
        assert torch.allclose(e, (k / 15) * sd * u)
    nz = u.abs() > 1e-9
    ratio = (etas[-1][nz] / etas[0][nz])
    assert torch.allclose(ratio, torch.full_like(ratio, 15.0), atol=1e-4)


def test_noise_identity_excludes_severity_task_and_decoder():
    p = seeds.noise_payload(SHA_A, "L1", 0, "bank_1")
    for token in ("severity", "task", "decod", "canonical", "free_ar"):
        assert token not in p.lower()
    assert "bank_1" in p


def test_max_half_width_is_one_site_sd_and_is_not_an_sd():
    sd = 0.37
    assert noise.max_half_width(sd, 15) == pytest.approx(sd)
    assert noise.distribution_sd(sd, 15) == pytest.approx(sd / 3 ** 0.5)
    u = noise.base_draw(SHA_A, "L1", 0, "bank_0", (200_000,))
    assert float(u.min()) >= -1.0 and float(u.max()) <= 1.0
    e = noise.eta(u, 15, sd)
    assert float(e.abs().max()) <= sd + 1e-6
    assert float(e.std()) == pytest.approx(sd / 3 ** 0.5, rel=0.02)


def test_symmetric_uniform_base():
    u = noise.base_draw(SHA_A, "L3", 2, "bank_5", (200_000,))
    assert float(u.mean()) == pytest.approx(0.0, abs=0.01)
    assert float(u.std()) == pytest.approx(1 / 3 ** 0.5, rel=0.02)


# 17. pairing across paired conditions --------------------------------------
def test_same_realization_gives_same_mask_and_noise_across_tasks():
    """Task and decoder are absent from both identities, so both are shared."""
    m1 = masks.build_mask(SHA_A, L1, 4, 9)[L1.connectivity_param]
    m2 = masks.build_mask(SHA_A, L1, 4, 9)[L1.connectivity_param]
    assert torch.equal(m1, m2)
    n1 = noise.base_draw(SHA_A, "L1", 4, "bank_2", (1, 128))
    n2 = noise.base_draw(SHA_A, "L1", 4, "bank_2", (1, 128))
    assert torch.equal(n1, n2)


# 19-21. restoration, leakage, intact identity ------------------------------
def test_parameters_restored_exactly():
    m = ToyModel()
    before = context.parameter_digest(m)
    with context.connectivity_lesion(m, masks.build_mask(SHA_A, L2, 0, 12)):
        assert context.parameter_digest(m) != before
    assert context.parameter_digest(m) == before


def test_no_leakage_between_successive_cells():
    m = ToyModel()
    before = context.parameter_digest(m)
    for k in range(1, 16):
        with context.connectivity_lesion(m, masks.build_mask(SHA_A, L1, 0, k)):
            pass
        assert context.parameter_digest(m) == before, f"leak after k={k}"


def test_restoration_is_exact_not_reconstructed():
    """A zeroed weight must come back, which division by the mask cannot do."""
    m = ToyModel()
    w = m.ltm.sem_to_h0.weight
    orig = w.detach().clone()
    with context.connectivity_lesion(m, masks.build_mask(SHA_A, L3, 0, 15)):
        assert int((w == 0).sum()) > 0
    assert torch.equal(w.detach(), orig)


def test_k0_cell_is_identity():
    m = ToyModel()
    before = context.parameter_digest(m)
    cell = masks.build_mask(SHA_A, L1, 0, 0)
    with context.connectivity_lesion(m, cell, verify=False):
        assert context.parameter_digest(m) == before
    assert context.parameter_digest(m) == before
    assert guard.is_intact_cell(0) and not guard.is_intact_cell(1)


def test_context_detects_an_untouchable_change():
    m = ToyModel()
    bad = {"ltm.sem_to_h0.bias": torch.zeros(512)}
    with pytest.raises(context.PristineViolation):
        with context.connectivity_lesion(m, bad):
            pass


# 22-23. training path and execution guard ----------------------------------
def test_no_optimizer_or_backward_path_in_the_operator():
    for root, _, files in os.walk(PKG):
        if "tests" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(root, f)).read())
            called = {n.func.attr for n in ast.walk(tree)
                      if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute)}
            for bad in ("backward", "zero_grad", "requires_grad_"):
                assert bad not in called, f"{f} calls {bad}()"
            ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            assert not any("optim" in x.lower() for x in ids), f"{f} names an optimizer"


def test_runner_refuses_execution_without_authorization(tmp_path, monkeypatch):
    monkeypatch.delenv(guard.AUTH_ENV, raising=False)
    with pytest.raises(guard.ExecutionRefused):
        guard.check_authorized("any-matrix-hash")


def test_authorization_must_name_this_exact_matrix(tmp_path, monkeypatch):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({
        "token": guard.REQUIRED_TOKEN, "go_for_scientific_execution": True,
        "run_matrix_sha256": "SOME_OTHER_MATRIX"}))
    monkeypatch.setenv(guard.AUTH_ENV, str(auth))
    with pytest.raises(guard.ExecutionRefused) as e:
        guard.check_authorized("THE_REAL_MATRIX")
    assert "DIFFERENT run matrix" in str(e.value)


def test_authorization_requires_the_exact_token(tmp_path, monkeypatch):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({
        "token": "NOT_THE_TOKEN", "go_for_scientific_execution": True,
        "run_matrix_sha256": "M"}))
    monkeypatch.setenv(guard.AUTH_ENV, str(auth))
    with pytest.raises(guard.ExecutionRefused):
        guard.check_authorized("M")


def test_valid_authorization_is_accepted(tmp_path, monkeypatch):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({
        "token": guard.REQUIRED_TOKEN, "go_for_scientific_execution": True,
        "run_matrix_sha256": "M"}))
    monkeypatch.setenv(guard.AUTH_ENV, str(auth))
    guard.check_authorized("M")          # must not raise


def test_compiled_posture_is_no():
    assert guard.GO_FOR_SCIENTIFIC_EXECUTION is False


# ------------------------------------------------ frozen design surfaces ---
def test_sd_procedure_is_frozen_and_hashable():
    d = sd_procedure.procedure_descriptor()
    assert d["sd_ddof"] == 1 and d["sd_n_items"] == 2048
    assert d["sd_sample_seed"] == 7 and d["sd_batch_size"] == 256
    assert d["no_lesion"] is True and d["no_activation_noise"] is True
    assert d["site_historically_calibrated"]["L3"] is False
    assert sd_procedure.procedure_hash() == sd_procedure.procedure_hash()


def test_sd_population_is_deterministic_and_sorted():
    a = sd_procedure.population_for(29_571)
    b = sd_procedure.population_for(29_571)
    assert a == b == sorted(a) and len(a) == 2048


def test_battery_tiers_and_exclusions():
    assert [e.key for e in battery.PRIMARY] == [
        "rep_canonical_full", "rep_freear_full", "naming_exact", "c_top1"]
    assert all(e.route == "full" for e in battery.PRIMARY)
    assert all(e.route in ("wm", "ltm") for e in battery.DIAGNOSTIC)
    for e in battery.ALL_ENDPOINTS:
        assert e.population and e.numerator and e.denominator
        assert e.exactness and e.undefined_handling
    assert "pseudoword_lesion_battery" in battery.EXCLUDED


def test_run_matrix_is_frozen_and_consistent():
    m = json.load(open(os.path.join(PKG, "contract",
                                    "LESIONING_V2_RUN_MATRIX.json")))
    assert m["n_lesion_cells"] == (12 + 12 + 12 + 4) * 3 * 15 == 1800
    assert m["n_intact_cells"] == 4 * 3
    assert m["source_states_included"] is False
    assert m["pseudoword_primary_lesion_battery"] is False
    assert all(c["state_id"].endswith("_POST_REPAIR") for c in m["cells"])
    for c in m["cells"]:
        site = SITES[c["site"]]
        assert c["n_logical_edges_removed"] == masks.n_removed(site,
                                                               c["severity_k"])


# ============================================================= closure ======
#  Canonical Jean-Zay artifacts (torch 2.6.0). These pin the binding hashes so
#  a later edit to the matrix or the layout record cannot pass unnoticed.
# ============================================================================
CANON = os.path.join(PKG, "contract", "CANONICAL_CLUSTER_ARTIFACTS.json")
RUN_MATRIX_SEMANTIC_SHA = \
    "cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a"


def _sha_file(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_canonical_artifact_record_exists():
    c = json.load(open(CANON))
    assert c["run_matrix_semantic_sha256"] == RUN_MATRIX_SEMANTIC_SHA
    assert c["scientific_lesion_executed"] is False
    assert c["go_for_scientific_execution"] is False
    assert c["intact_sd_procedure_hash"] == sd_procedure.procedure_hash()


def test_present_canonical_artifacts_match_their_binding_hashes():
    c = json.load(open(CANON))
    for name, rec in c["artifacts"].items():
        p = os.path.join(PKG, "contract", name)
        if rec["present_in_repo"]:
            assert os.path.exists(p), f"{name} declared present but missing"
            assert _sha_file(p) == rec["file_sha256"], f"{name} hash drift"
        else:
            assert not os.path.exists(p), \
                f"{name} is declared transfer-required but a local file exists"


def test_run_matrix_semantic_hash_is_the_cluster_one():
    from paper_programme.lesioning_v2.scripts.build_run_matrix import matrix_sha256
    m = json.load(open(os.path.join(PKG, "contract",
                                    "LESIONING_V2_RUN_MATRIX.json")))
    assert m["matrix_sha256"] == RUN_MATRIX_SEMANTIC_SHA
    assert matrix_sha256(m) == RUN_MATRIX_SEMANTIC_SHA     # recomputed, not trusted
    assert m["state_sha_resolved"] is True
    assert m["n_cells"] == 1812 and m["n_lesion_cells"] == 1800


def test_authorization_must_bind_the_cluster_matrix_hash(tmp_path, monkeypatch):
    """An authorization for the pre-cluster matrix must NOT license this one."""
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({
        "token": guard.REQUIRED_TOKEN, "go_for_scientific_execution": True,
        "run_matrix_sha256":
            "b4d3c98816b46f59afb3e32f081fa073dd5a48cd27f2279fdaccb1f47ca6887c"}))
    monkeypatch.setenv(guard.AUTH_ENV, str(stale))
    with pytest.raises(guard.ExecutionRefused):
        guard.check_authorized(RUN_MATRIX_SEMANTIC_SHA)


def test_sd_constants_absent_means_execution_still_refused():
    """Until the SD constants file is transferred, no real run can proceed."""
    p = os.path.join(PKG, "contract", "LESIONING_V2_INTACT_SD_CONSTANTS.json")
    assert not os.path.exists(p)


def test_gru_layout_record_is_the_canonical_2_6_0_one():
    g = json.load(open(os.path.join(PKG, "contract",
                                    "LESIONING_V2_GRU_LAYOUT.json")))
    assert g["torch_version"] == "2.6.0"
    assert g["three_contiguous_H_blocks"] is True
    assert g["logical_mask_tiles_identically"] is True
    assert g["operator_depends_on_gate_names"] is False
    assert g["documented_human_readable_order"] == "r,z,n"
    assert g["verified"] is True
