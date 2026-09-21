"""Batch-ID repair: invariance and continuation tests.

Synthetic / static only. ZERO real P1-P4 nonzero lesion forward.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "paper_programme",
                                "v7_prelesion_validation", "scripts"))

from paper_programme.lesioning_v2.execution import (  # noqa: E402
    cells, continuity, injection, lesioned_eval, preflight)

SHA = "e" * 64
SITE = "L1"
REAL = 0
K = 7
SD = 0.4921
MATRIX = preflight.load_matrix()


def eta_for(ids, shape, k=K):
    f = injection.batch_eta_fn(SHA, SITE, REAL, list(ids), k, SD)
    return f(torch.zeros(*shape))


# ===================================================== A-F mapping ==========
def test_A_batch_len_matches_activation_batch_dimension():
    for n in (1, 17, 64, 128, 257):
        ids = [f"bank_{i}" for i in range(n)]
        out = eta_for(ids, (1, n, 6))
        assert out.shape[-2] == n == len(ids)


def test_A_mismatch_is_refused():
    f = injection.batch_eta_fn(SHA, SITE, REAL, ["bank_0", "bank_1"], K, SD)
    with pytest.raises(injection.InjectionError):
        f(torch.zeros(1, 3, 6))


def test_B_C_D_order_preserved_no_offset_no_permutation():
    ids = [f"bank_{i}" for i in range(8)]
    full = eta_for(ids, (1, 8, 6))[0]
    for j, iid in enumerate(ids):
        single = eta_for([iid], (1, 1, 6))[0, 0]
        assert torch.equal(full[j], single), f"row {j} is not item {iid}"
    # a permuted id list must permute the rows correspondingly
    perm = list(reversed(ids))
    pfull = eta_for(perm, (1, 8, 6))[0]
    for j, iid in enumerate(perm):
        assert torch.equal(pfull[j], eta_for([iid], (1, 1, 6))[0, 0])
    assert not torch.equal(full, pfull)


def test_E_first_and_last_population_items_map_correctly():
    pop = list(range(29571))
    first, last = f"bank_{pop[0]}", f"bank_{pop[-1]}"
    ch = lesioned_eval.chunks(pop, 128)
    assert f"bank_{ch[0][0]}" == first
    assert f"bank_{ch[-1][-1]}" == last
    a = eta_for([first], (1, 1, 6))[0, 0]
    b = eta_for([f"bank_{i}" for i in ch[0]], (1, len(ch[0]), 6))[0, 0]
    assert torch.equal(a, b)
    z = eta_for([f"bank_{i}" for i in ch[-1]], (1, len(ch[-1]), 6))[0, -1]
    assert torch.equal(z, eta_for([last], (1, 1, 6))[0, 0])


def test_F_final_partial_batch_maps_correctly():
    pop = list(range(29571))
    ch = lesioned_eval.chunks(pop, 128)
    assert len(ch[-1]) == 29571 % 128 != 0
    assert sum(len(c) for c in ch) == 29571


def test_G_every_global_item_appears_exactly_once_per_pass():
    pop = list(range(29571))
    for size in (64, 128, 256, 512):
        seen = [i for c in lesioned_eval.chunks(pop, size) for i in c]
        assert seen == pop


def test_H_canonical_and_free_ar_share_the_same_global_identity():
    """Identity depends on the item, never on the decoding convention."""
    from paper_programme.lesioning_v2.lesion_operator import seeds
    p = seeds.noise_payload(SHA, SITE, REAL, "bank_1234")
    for token in ("canonical", "free_ar", "forced", "retrieval", "naming"):
        assert token not in p.lower()
    # same eta regardless of which convention's chunk size produced the batch
    a = eta_for([f"bank_{i}" for i in range(128)], (1, 128, 6))[0, 5]
    b = eta_for([f"bank_{i}" for i in range(256)], (1, 256, 6))[0, 5]
    assert torch.equal(a, b)


# ===================================================== I invariance =========
@pytest.mark.parametrize("size", [1, 17, 64, 128, 257])
def test_I_batch_size_invariance_bitwise(size):
    """A fixed item's eta is bitwise identical under every partition."""
    pop = list(range(600))
    ref = {}
    for c in lesioned_eval.chunks(pop, size):
        ids = [f"bank_{i}" for i in c]
        out = eta_for(ids, (1, len(c), 6))[0]
        for j, i in enumerate(c):
            ref[i] = out[j].clone()
    base = {i: eta_for([f"bank_{i}"], (1, 1, 6))[0, 0] for i in (0, 1, 299,
                                                                 598, 599)}
    for i, v in base.items():
        assert torch.equal(ref[i], v), f"item {i} changed under batching {size}"


def test_I_first_middle_final_partial_batches_agree():
    pop = list(range(257))
    per_size = {}
    for size in (1, 17, 64, 128, 257):
        d = {}
        for c in lesioned_eval.chunks(pop, size):
            out = eta_for([f"bank_{i}" for i in c], (1, len(c), 6))[0]
            for j, i in enumerate(c):
                d[i] = out[j]
        per_size[size] = d
    ref = per_size[1]
    for size, d in per_size.items():
        for i in (0, 128, 255, 256):
            assert torch.equal(d[i], ref[i]), f"item {i} differs at size {size}"


# ===================================================== J pairing ============
def test_J_task_and_readout_do_not_change_eta():
    ids = [f"bank_{i}" for i in range(32)]
    a = eta_for(ids, (1, 32, 6))
    b = eta_for(ids, (1, 32, 6))
    assert torch.equal(a, b)
    from paper_programme.lesioning_v2.lesion_operator import seeds
    p = seeds.noise_payload(SHA, SITE, REAL, "bank_0")
    for token in ("task", "endpoint", "readout", "decod"):
        assert token not in p.lower()


# ============================================ autoregressive invariant ======
class TinyAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wm = torch.nn.Module()
        self.wm.encoder = torch.nn.GRU(4, 6, batch_first=True)


def test_autoregressive_eta_is_identical_at_every_step():
    """The hook fires once per AR step on the same batch; eta must not move."""
    m = TinyAR()
    ids = [f"bank_{i}" for i in range(5)]
    f = injection.batch_eta_fn(SHA, SITE, REAL, ids, K, SD)
    seen = []
    x = torch.randn(5, 3, 4)
    with torch.no_grad():
        with injection.activation_injection(m, "L1",
                                            lambda s: seen.append(f(s)) or seen[-1]):
            for _ in range(12):                 # FREE_AR_MAX_STEPS
                m.wm.encoder(x)
    assert len(seen) == 12
    for s in seen[1:]:
        assert torch.equal(s, seen[0]), "eta changed across AR steps"


def test_different_items_get_independent_deterministic_eta():
    ids = [f"bank_{i}" for i in range(6)]
    out = eta_for(ids, (1, 6, 6))[0]
    for i in range(6):
        for j in range(i + 1, 6):
            assert not torch.equal(out[i], out[j])
    assert torch.equal(eta_for(ids, (1, 6, 6)), eta_for(ids, (1, 6, 6)))


def test_no_cursor_in_injection_or_driver():
    """No stateful advancing construct. Checked over identifiers, not prose:
    the docstrings legitimately explain why a cursor would be wrong."""
    import ast
    for rel in ("execution/injection.py", "execution/lesioned_eval.py"):
        tree = ast.parse(open(os.path.join(PKG, rel)).read())
        ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        # what actually matters is that NO state survives a hook firing:
        # no iterator advancement, no nonlocal rebinding, no global counter.
        assert "next" not in called, f"{rel} advances an iterator"
        assert "count" not in attrs, f"{rel} uses itertools.count"
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Nonlocal)], \
            f"{rel} rebinds nonlocal state"
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)], \
            f"{rel} rebinds global state"
        for bad in ("cursor", "counter", "index_state", "_pos"):
            assert not any(bad == x.lower() for x in ids | attrs), \
                f"{rel} holds a cursor-like name: {bad}"
    # and behaviourally: repeated firings on one batch are identical
    # (proven by test_autoregressive_eta_is_identical_at_every_step)


# ================================================ recovered batch sizes =====
def test_batch_sizes_are_recovered_from_executable_code():
    s = lesioned_eval.evaluator_batch_sizes()
    from scripts.evaluate_train_lexicon_ceiling import BATCH_SIZE
    assert s["CANONICAL_FORCED_LENGTH_AR"] == BATCH_SIZE == 128
    assert s["GENUINE_FREE_AR"] == 256
    assert s["SEMANTIC_GREEDY_AR_GLOBAL_CAP"] == 512
    assert s["STRICT_TOP1_RETRIEVAL"] == 64


def test_chunks_are_contiguous_and_order_preserving():
    assert lesioned_eval.chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        lesioned_eval.chunks([1], 0)


# ==================================================== k=0 preservation ======
def test_k0_path_still_calls_the_unchanged_evaluator():
    src = open(os.path.join(PKG, "scripts", "run_lesion_v2.py")).read()
    assert "if k == 0:" in src
    k0 = src[src.index("if k == 0:"):src.index("else:")]
    assert "evaluators.evaluate_endpoints(" in k0
    assert "lesioned_eval" not in k0, "k=0 routed through nonzero batching"
    assert "injection.activation_injection(model, row[\"site\"], None)" in k0


def test_frozen_science_files_byte_identical_to_base():
    import subprocess
    base = "0b22b90455a30b8d2ee1ca86df0fc96955e4e542"
    for rel in ("execution/injection.py", "execution/evaluators.py",
                "lesion_operator/seeds.py", "lesion_operator/noise.py",
                "lesion_operator/masks.py", "lesion_operator/context.py",
                "lesion_operator/sites.py",
                "contract/LESIONING_V2_RUN_MATRIX.json"):
        p = f"paper_programme/lesioning_v2/{rel}"
        out = subprocess.run(["git", "show", f"{base}:{p}"], cwd=REPO,
                             capture_output=True)
        assert out.returncode == 0, f"{rel} missing at base"
        here = open(os.path.join(PKG, rel), "rb").read()
        assert hashlib.sha256(out.stdout).hexdigest() == \
            hashlib.sha256(here).hexdigest(), f"{rel} DRIFTED"


def test_matrix_semantic_sha_unchanged():
    assert MATRIX["matrix_sha256"] == \
        "cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a"


# ==================================================== continuation ==========
def _k0_rows():
    return [r for r in MATRIX["cells"] if int(r["severity_k"]) == 0]


def _items(n=2):
    return [{"item_id": f"bank_{i}", "task": "repetition",
             "decoding_convention": "GENUINE_FREE_AR", "correct": 1}
            for i in range(n)]


def _build_namespace(tmp_path, rows=None, extra=None):
    root = str(tmp_path / "res")
    for r in (rows if rows is not None else _k0_rows()):
        cells.write_cell(root, r, _items(), {"rows": []}, {"p": 1},
                         restoration_verified=True)
    if extra is not None:
        cells.write_cell(root, extra, _items(), {"rows": []}, {"p": 1},
                         restoration_verified=True)
    return root


def test_valid_twelve_control_namespace_is_accepted(tmp_path):
    root = _build_namespace(tmp_path)
    man = continuity.generate(root, MATRIX)
    assert man["n_complete_k0_cells"] == 12
    assert man["n_complete_nonzero_cells"] == 0
    assert len(man["cells"]) == 12
    assert man["original_execution_commit"] == \
        continuity.ORIGINAL_EXECUTION_COMMIT
    continuity.validate_for_continuation(man, root, MATRIX)


def test_missing_k0_rejected(tmp_path):
    root = _build_namespace(tmp_path, rows=_k0_rows()[:11])
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.generate(root, MATRIX)
    assert "expected 12" in str(e.value)


def test_any_complete_nonzero_cell_rejected(tmp_path):
    nz = next(r for r in MATRIX["cells"] if int(r["severity_k"]) == 1)
    root = _build_namespace(tmp_path, extra=nz)
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.generate(root, MATRIX)
    assert "k>0" in str(e.value)


def test_foreign_cell_rejected(tmp_path):
    root = _build_namespace(tmp_path)
    alien = dict(_k0_rows()[0])
    alien["state_id"] = "PZ_POST_REPAIR"
    cells.write_cell(root, alien, _items(), {"rows": []}, {},
                     restoration_verified=True)
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.generate(root, MATRIX)
    assert "matches no authoritative" in str(e.value)


def test_altered_marker_rejected(tmp_path):
    root = _build_namespace(tmp_path)
    man = continuity.generate(root, MATRIX)
    d = os.path.join(root, man["cells"][0]["relative_location"])
    with open(os.path.join(d, "items.jsonl"), "a") as fh:
        fh.write("{}\n")
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.generate(root, MATRIX)
    assert "differs from its marker hash" in str(e.value)


def test_wrong_k0_hash_rejected_by_validation(tmp_path):
    root = _build_namespace(tmp_path)
    man = continuity.generate(root, MATRIX)
    man["cells"][0]["items_jsonl_sha256"] = "0" * 64
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.validate_for_continuation(man, root, MATRIX)
    assert "stale manifest or altered cell" in str(e.value)


def test_stale_manifest_from_other_root_rejected(tmp_path):
    root = _build_namespace(tmp_path)
    man = continuity.generate(root, MATRIX)
    other = str(tmp_path / "other")
    shutil.copytree(root, other)
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.validate_for_continuation(man, other, MATRIX)
    assert "different result root" in str(e.value)


def test_manifest_for_other_matrix_rejected(tmp_path):
    root = _build_namespace(tmp_path)
    man = continuity.generate(root, MATRIX)
    man["run_matrix_sha256"] = "f" * 64
    with pytest.raises(continuity.ContinuityError):
        continuity.validate_for_continuation(man, root, MATRIX)


def test_duplicate_identity_rejected(tmp_path):
    root = _build_namespace(tmp_path)
    r = _k0_rows()[0]
    d = cells.cell_dir(root, r)
    shutil.copytree(d, d + "_dup")
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.generate(root, MATRIX)
    assert "duplicate scientific cell identity" in str(e.value)


def test_default_fresh_mode_still_rejects_existing_namespace(tmp_path):
    root = _build_namespace(tmp_path)
    with pytest.raises(preflight.PreflightError) as e:
        preflight.run(root, require_authorization=False,
                      allow_missing_transfer=True)
    assert "already exists" in str(e.value)
    assert "continuity-manifest" in str(e.value)


def test_continuation_requires_an_existing_namespace(tmp_path):
    man_path = str(tmp_path / "m.json")
    json.dump({"run_matrix_sha256": MATRIX["matrix_sha256"],
               "result_root": str(tmp_path / "nope"), "cells": []},
              open(man_path, "w"))
    with pytest.raises(preflight.PreflightError) as e:
        preflight.run(str(tmp_path / "nope"), require_authorization=False,
                      allow_missing_transfer=True, continuity_manifest=man_path)
    assert "does not exist" in str(e.value)


def test_continuation_accepts_a_valid_namespace(tmp_path):
    root = _build_namespace(tmp_path)
    man_path = str(tmp_path / "m.json")
    json.dump(continuity.generate(root, MATRIX), open(man_path, "w"))
    rep = preflight.run(root, require_authorization=False,
                        allow_missing_transfer=True,
                        continuity_manifest=man_path)
    assert rep["mode"] == "CONTINUATION"
    assert rep["existing_complete_k0_cells"] == 12


def test_complete_cells_are_skipped_and_never_overwritten(tmp_path):
    root = _build_namespace(tmp_path)
    for r in _k0_rows():
        assert cells.is_complete(root, r)
        assert cells.retry_allowed(root, r) is False
        with pytest.raises(cells.CellError):
            cells.write_cell(root, r, _items(), {"rows": []}, {},
                             restoration_verified=True)


def test_continuity_module_writes_nothing_into_the_namespace():
    import ast
    src = open(os.path.join(PKG, "execution", "continuity.py")).read()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            assert n.func.attr not in ("remove", "rmtree", "unlink", "rename",
                                       "makedirs", "mkdir"), \
                f"continuity performs a write operation: {n.func.attr}"
    assert "open(" in src        # reads only


def test_generator_refuses_output_inside_the_namespace(tmp_path):
    import subprocess
    root = _build_namespace(tmp_path)
    out = subprocess.run(
        [sys.executable,
         os.path.join(PKG, "scripts", "generate_continuity_manifest.py"),
         "--result-root", root, "--out", os.path.join(root, "m.json")],
        capture_output=True, text=True)
    assert out.returncode == 2
    assert "OUTSIDE" in out.stderr


def test_continuity_decision_is_performance_blind():
    src = open(os.path.join(PKG, "execution", "continuity.py")).read()
    import ast
    ids = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Attribute)}
    for bad in ("accuracy", "exact_match", "loss", "metric"):
        assert not any(bad in x.lower() for x in ids | attrs)
