"""Synthetic execution-integration tests. No P1-P4 nonzero lesion is run."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
sys.path.insert(0, REPO)

from paper_programme.lesioning_v2.execution import (  # noqa: E402
    aggregate, authorization, cells, injection, preflight)
from paper_programme.lesioning_v2.lesion_operator import masks  # noqa: E402
from paper_programme.lesioning_v2.lesion_operator.sites import L1, L3  # noqa: E402

MATRIX = json.load(open(os.path.join(PKG, "contract",
                                     "LESIONING_V2_RUN_MATRIX.json")))
CLUSTER_SHA = authorization.CLUSTER_RUN_MATRIX_SHA256
SUPERSEDED_SHA = authorization.SUPERSEDED_RUN_MATRIX_SHA256
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                      capture_output=True, text=True).stdout.strip()


def toy_row(**kw):
    r = {"state_id": "PX_POST_REPAIR", "state_sha256": "c" * 64, "site": "L1",
         "severity_k": 3, "severity_s": 0.2, "connectivity_fraction": 0.06,
         "n_logical_edges_total": 8192, "n_logical_edges_removed": 492,
         "realization": 0, "mask_seed_digest": "d" * 64,
         "activation_seed_namespace": "ns", "sd_constant_ref": "PX/L1",
         "cell_kind": "LESION"}
    r.update(kw)
    return r


# ---------------------------------------------- A-F matrix mapping ---------
def test_A_all_rows_map_exactly_once():
    ids = [cells.cell_identity(r) for r in MATRIX["cells"]]
    assert len(ids) == 1812 == len(set(ids))


def test_B_no_duplicate_cell_identity():
    ids = [cells.cell_identity(r) for r in MATRIX["cells"]]
    assert len(ids) - len(set(ids)) == 0


def test_C_no_missing_cell_identity():
    keys = [cells.cell_key(r) for r in MATRIX["cells"]]
    assert len(set(keys)) == len(MATRIX["cells"])
    # identity and human key agree 1:1
    pairs = {(cells.cell_identity(r), cells.cell_key(r)) for r in MATRIX["cells"]}
    assert len({p[0] for p in pairs}) == len({p[1] for p in pairs}) == 1812


def test_D_twelve_intact_controls():
    assert sum(1 for r in MATRIX["cells"] if r["severity_k"] == 0) == 12


def test_E_eighteen_hundred_nonzero_lesions():
    assert sum(1 for r in MATRIX["cells"] if r["severity_k"] > 0) == 1800


def test_F_runner_consumes_matrix_and_never_regenerates():
    src = open(os.path.join(PKG, "scripts", "run_lesion_v2.py")).read()
    assert "preflight.load_matrix" in src
    assert "build_run_matrix" not in src
    assert "matrix_regenerated_by_runner" in src


def test_matrix_semantic_sha_unchanged():
    assert MATRIX["matrix_sha256"] == CLUSTER_SHA
    assert all(r["state_id"].endswith("_POST_REPAIR") for r in MATRIX["cells"])


# ------------------------------------------------ G item schema -------------
FROZEN_ITEM_FIELDS = (
    "state_id", "state_sha256", "site", "severity_k", "severity_s",
    "connectivity_fraction", "n_logical_edges_total", "n_logical_edges_removed",
    "realization", "mask_seed_digest", "item_id", "activation_seed_digest",
    "task", "decoding_convention", "target", "prediction", "correct")


def _items(n=3, **kw):
    base = {f: 0 for f in FROZEN_ITEM_FIELDS}
    base.update({"task": "repetition", "decoding_convention": "GENUINE_FREE_AR",
                 "state_id": "PX_POST_REPAIR", "site": "L1", "severity_k": 3,
                 "realization": 0, "correct": 1})
    base.update(kw)
    return [dict(base, item_id=f"bank_{i}") for i in range(n)]


def test_G_item_schema_round_trip(tmp_path):
    row = toy_row()
    d = cells.write_cell(str(tmp_path), row, _items(4), {"rows": []},
                         {"p": 1}, restoration_verified=True)
    lines = [json.loads(x) for x in
             open(os.path.join(d, "items.jsonl")) if x.strip()]
    assert len(lines) == 4
    for r in lines:
        assert set(FROZEN_ITEM_FIELDS) <= set(r), "frozen field dropped"


def test_no_silent_new_scientific_field():
    src = open(os.path.join(PKG, "execution", "evaluators.py")).read()
    for f in ("item_id", "task", "decoding_convention", "target",
              "prediction", "correct"):
        assert f'"{f}"' in src


# --------------------------------------- H aggregation hygiene --------------
def test_H_aggregate_uses_only_complete_cells(tmp_path):
    root = str(tmp_path / "res")
    cells.write_cell(root, toy_row(realization=0), _items(2),
                     {"rows": []}, {}, restoration_verified=True)
    # an unfinalized staging-like directory must be ignored
    partial = os.path.join(root, "PX_POST_REPAIR/L1/k03/r01")
    os.makedirs(partial, exist_ok=True)
    open(os.path.join(partial, "items.jsonl"), "w").write("{}\n")
    agg = aggregate.aggregate(root)
    assert agg["n_complete_cells"] == 1
    assert agg["incomplete_cells_excluded"] is True
    assert agg["rows"][0]["n"] == 2


def test_aggregate_detects_tampering(tmp_path):
    root = str(tmp_path / "res")
    d = cells.write_cell(root, toy_row(), _items(2), {"rows": []}, {},
                         restoration_verified=True)
    open(os.path.join(d, "items.jsonl"), "a").write("{}\n")
    with pytest.raises(cells.CellError):
        aggregate.aggregate(root)


def test_aggregate_computes_only_the_frozen_metric():
    """Only exact_match is computed. Checked over identifiers, not prose:
    the docstring legitimately names what the module refuses to do."""
    import ast
    src = open(os.path.join(PKG, "execution", "aggregate.py")).read()
    assert "exact_match" in src
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ("ttest", "pvalue", "significance", "bootstrap", "chi2",
                "scipy", "statsmodels"):
        assert not any(bad in x.lower() for x in names | attrs), \
            f"aggregate computes {bad}"
    # and the only derived quantity assigned is exact_match
    assigned = {t.slice.value for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Subscript)
                and isinstance(getattr(t, "slice", None), ast.Constant)}
    assert assigned <= {"exact_match", "expected_cells", "complete"}


# ------------------------------------- exactly-once / retry -----------------
def test_finalized_cell_cannot_be_overwritten(tmp_path):
    root = str(tmp_path / "res")
    row = toy_row()
    cells.write_cell(root, row, _items(1), {"rows": []}, {},
                     restoration_verified=True)
    with pytest.raises(cells.CellError):
        cells.write_cell(root, row, _items(1), {"rows": []}, {},
                         restoration_verified=True)


def test_failure_before_finalize_leaves_no_complete_cell(tmp_path):
    root = str(tmp_path / "res")
    row = toy_row()
    with pytest.raises(cells.CellError):
        cells.write_cell(root, row, _items(1), {"rows": []}, {},
                         restoration_verified=False)   # restoration failed
    assert not cells.is_complete(root, row)
    assert aggregate.aggregate(root)["n_complete_cells"] == 0


def test_retry_allowed_only_for_incomplete_cells(tmp_path):
    root = str(tmp_path / "res")
    row = toy_row()
    assert cells.retry_allowed(root, row) is True
    cells.write_cell(root, row, _items(1), {"rows": []}, {},
                     restoration_verified=True)
    assert cells.retry_allowed(root, row) is False


def test_retry_decision_is_performance_blind():
    import inspect
    src = inspect.getsource(cells.retry_allowed)
    for bad in ("acc", "loss", "correct", "exact", "metric", "curve"):
        assert bad not in src.lower().replace("performance-blind", "")


def test_synthetic_retry_invariance_of_identity_and_seeds():
    """A retried cell reuses byte-identical mask and noise."""
    row = toy_row()
    a = masks.build_mask(row["state_sha256"], L1, row["realization"],
                         row["severity_k"])[L1.connectivity_param]
    b = masks.build_mask(row["state_sha256"], L1, row["realization"],
                         row["severity_k"])[L1.connectivity_param]
    assert torch.equal(a, b)
    assert cells.cell_identity(row) == cells.cell_identity(dict(row))


def test_duplicate_cell_finalize_refused(tmp_path):
    root = str(tmp_path / "res")
    cells.write_cell(root, toy_row(), _items(1), {"rows": []}, {},
                     restoration_verified=True)
    with pytest.raises(cells.CellError):
        cells.write_cell(root, toy_row(), _items(1), {"rows": []}, {},
                         restoration_verified=True)


# ---------------------------------------- authorization rejections ----------
def _auth(tmp_path, **kw):
    a = {"token": authorization.REQUIRED_TOKEN,
         "go_for_scientific_execution": True,
         "execution_commit": HEAD, "run_matrix_sha256": CLUSTER_SHA}
    a.update(kw)
    p = tmp_path / "auth.json"
    p.write_text(json.dumps(a))
    return str(p)


def test_auth_missing_artifact(monkeypatch):
    monkeypatch.delenv(authorization.AUTH_ENV, raising=False)
    with pytest.raises(authorization.AuthorizationRefused):
        authorization.authorize(HEAD, CLUSTER_SHA)


def test_auth_wrong_token(tmp_path):
    a = json.load(open(_auth(tmp_path, token="NOPE")))
    with pytest.raises(authorization.AuthorizationRefused):
        authorization.validate(a, HEAD, CLUSTER_SHA)


def test_auth_false_go(tmp_path):
    a = json.load(open(_auth(tmp_path, go_for_scientific_execution=False)))
    with pytest.raises(authorization.AuthorizationRefused):
        authorization.validate(a, HEAD, CLUSTER_SHA)


def test_auth_wrong_matrix(tmp_path):
    a = json.load(open(_auth(tmp_path, run_matrix_sha256="f" * 64)))
    with pytest.raises(authorization.AuthorizationRefused):
        authorization.validate(a, HEAD, CLUSTER_SHA)


def test_auth_superseded_matrix(tmp_path):
    a = json.load(open(_auth(tmp_path, run_matrix_sha256=SUPERSEDED_SHA)))
    with pytest.raises(authorization.AuthorizationRefused) as e:
        authorization.validate(a, HEAD, CLUSTER_SHA)
    assert "SUPERSEDED" in str(e.value)


def test_auth_wrong_execution_commit(tmp_path):
    a = json.load(open(_auth(tmp_path, execution_commit="0" * 40)))
    with pytest.raises(authorization.AuthorizationRefused) as e:
        authorization.validate(a, HEAD, CLUSTER_SHA)
    assert "DIFFERENT execution commit" in str(e.value)


def test_auth_missing_field(tmp_path):
    a = json.load(open(_auth(tmp_path)))
    del a["execution_commit"]
    with pytest.raises(authorization.AuthorizationRefused):
        authorization.validate(a, HEAD, CLUSTER_SHA)


def test_no_valid_production_authorization_exists_in_repo():
    for root, _, files in os.walk(PKG):
        for f in files:
            if not f.endswith(".json"):
                continue
            try:
                d = json.load(open(os.path.join(root, f)))
            except Exception:
                continue
            if not (isinstance(d, dict)
                    and d.get("token") == authorization.REQUIRED_TOKEN):
                continue
            # a committed artifact must FAIL full validation against this
            # HEAD and the authoritative matrix -- being merely non-GO is not
            # enough to call it safe
            with pytest.raises(authorization.AuthorizationRefused):
                authorization.validate(d, HEAD, CLUSTER_SHA)


# ------------------------------------------ preflight rejections ------------
def test_preflight_rejects_existing_output_namespace(tmp_path):
    out = tmp_path / "res"
    out.mkdir()
    with pytest.raises(preflight.PreflightError) as e:
        preflight.run(str(out), require_authorization=False,
                      allow_missing_transfer=True)
    assert "already exists" in str(e.value)


def test_preflight_requires_transfer_artifacts_for_real_run(tmp_path):
    with pytest.raises(preflight.PreflightError) as e:
        preflight.run(str(tmp_path / "res"), require_authorization=True,
                      allow_missing_transfer=False)
    assert "transfer-required artifact absent" in str(e.value) \
        or "not clean" in str(e.value)


def test_preflight_rejects_wrong_transfer_sha(tmp_path):
    td = tmp_path / "ctl"
    td.mkdir()
    (td / "LESIONING_V2_STATE_VERIFICATION.json").write_text("{}")
    with pytest.raises(preflight.PreflightError) as e:
        preflight.run(str(tmp_path / "res"), require_authorization=False,
                      transfer_dir=str(td), allow_missing_transfer=True)
    assert "SHA mismatch" in str(e.value)


def test_preflight_dry_run_passes_and_validates_config(tmp_path):
    rep = preflight.run(str(tmp_path / "res"), require_authorization=False,
                        allow_missing_transfer=True)
    assert rep["matrix_sha256"] == CLUSTER_SHA
    assert rep["scientific_configuration"] == "UNCHANGED"
    assert rep["no_training_path"] is True
    assert rep["authorized"] is False
    assert rep["package_integrity"] == "OK"


def test_evaluator_identity_is_stable_and_covers_science_files():
    a = preflight.evaluator_identity()
    b = preflight.evaluator_identity()
    assert a == b
    for rel in ("lesion_operator/masks.py", "lesion_operator/noise.py",
                "lesion_operator/battery.py", "execution/injection.py",
                "execution/evaluators.py"):
        assert rel in a


def test_preflight_requires_canonical_torch_for_real_run(tmp_path):
    if torch.__version__ == preflight.REQUIRED_TORCH:
        pytest.skip("already on the canonical runtime")
    with pytest.raises(preflight.PreflightError):
        preflight.run(str(tmp_path / "res"), require_authorization=True,
                      allow_missing_transfer=True)


# --------------------------------------------- injection semantics ----------
class TinyGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wm = torch.nn.Module()
        self.wm.encoder = torch.nn.GRU(4, 6, batch_first=True)
        self.ltm = torch.nn.Module()
        self.ltm.encoder = torch.nn.GRU(4, 6, batch_first=True)
        self.ltm.decoder = torch.nn.GRU(4, 6, batch_first=True)


def test_injection_perturbs_the_encoder_hidden_state():
    m = TinyGRU()
    x = torch.randn(3, 5, 4)
    with torch.no_grad():
        _, h0 = m.wm.encoder(x)
        with injection.activation_injection(m, "L1", lambda s: torch.ones_like(s)):
            _, h1 = m.wm.encoder(x)
    assert torch.allclose(h1, h0 + 1.0)


def test_injection_removes_hooks_on_exit():
    m = TinyGRU()
    x = torch.randn(2, 5, 4)
    with injection.activation_injection(m, "L2", lambda s: torch.ones_like(s)):
        pass
    with torch.no_grad():
        _, a = m.ltm.encoder(x)
        _, b = m.ltm.encoder(x)
    assert torch.equal(a, b)


def test_L3_injection_targets_the_decoder_initial_state():
    m = TinyGRU()
    emb, h0 = torch.randn(2, 3, 4), torch.randn(1, 2, 6)
    with torch.no_grad():
        base, _ = m.ltm.decoder(emb, h0)
        with injection.activation_injection(m, "L3",
                                            lambda s: torch.ones_like(s)):
            pert, _ = m.ltm.decoder(emb, h0)
    assert not torch.allclose(base, pert)


def test_k0_injection_is_exactly_identity():
    m = TinyGRU()
    x = torch.randn(2, 5, 4)
    with torch.no_grad():
        _, a = m.wm.encoder(x)
        with injection.activation_injection(m, "L1", None):
            _, b = m.wm.encoder(x)
    assert torch.equal(a, b)


def test_batch_eta_is_item_frozen_and_severity_scaled():
    ids = ["bank_0", "bank_1"]
    f3 = injection.batch_eta_fn("a" * 64, "L1", 0, ids, 3, 0.5)
    f15 = injection.batch_eta_fn("a" * 64, "L1", 0, ids, 15, 0.5)
    s = torch.zeros(1, 2, 6)
    e3, e15 = f3(s), f15(s)
    nz = e3.abs() > 1e-12
    assert torch.allclose(e15[nz] / e3[nz],
                          torch.full_like(e15[nz], 5.0), atol=1e-4)
    assert torch.equal(f3(s), e3)          # stable across calls


def test_batch_eta_rejects_item_order_mismatch():
    f = injection.batch_eta_fn("a" * 64, "L1", 0, ["bank_0"], 5, 0.5)
    with pytest.raises(injection.InjectionError):
        f(torch.zeros(1, 4, 6))


# ------------------------------------------------- execution plan -----------
def test_execution_plan_has_no_scientific_results():
    p = json.load(open(os.path.join(PKG, "contract",
                                    "LESIONING_V2_EXECUTION_PLAN.json")))
    assert p["total_rows"] == 1812
    assert p["unique_executable_cell_identities"] == 1812
    assert p["duplicate_cell_identities"] == 0
    assert p["missing_cell_identities"] == 0
    assert p["intact_control_rows"] == 12
    assert p["nonzero_lesion_rows"] == 1800
    assert p["post_repair_only"] is True
    assert p["matrix_regenerated_by_runner"] is False
    assert p["scientific_results_included"] is False
    for bad in ("accuracy", "exact_match", "result", "prediction"):
        assert bad not in json.dumps(p).lower().replace("scientific_results_included", "")


def test_no_scientific_results_namespace_in_repo():
    assert not os.path.exists(os.path.join(PKG, "results"))
