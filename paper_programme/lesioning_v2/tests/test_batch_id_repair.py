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
    # a copied namespace is neither the manifest's parent nor a shard beneath it
    assert "neither the manifest's result root nor a shard directory" \
        in str(e.value)


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


# ============================================================================
#  REAL 12-SHARD TOPOLOGY (adversarial audit)
#
#  The real namespace is NOT twelve controls in one flat root. It is
#      parent/shard_00 .. parent/shard_11
#  with ONE COMPLETE k=0 control per shard, produced by twelve independent
#  runner invocations using --out-dir parent/shard_XX --shard X --n-shards 12.
#  The other eleven controls must NEVER be required inside shard_X.
# ============================================================================
SHARD_GROUPS = sorted({(r["state_id"], r["site"]) for r in MATRIX["cells"]})
N_SHARDS = 12


def _group_of(row):
    return (row["state_id"], row["site"])


def _shard_of(row):
    return SHARD_GROUPS.index(_group_of(row)) % N_SHARDS


def _build_real_topology(tmp_path, skip=None, swap=None):
    """parent/shard_XX/<that shard's own k0 control>."""
    parent = str(tmp_path / "l3_lesion_v2_results_0b22b904")
    for r in _k0_rows():
        x = _shard_of(r)
        if skip is not None and x == skip:
            continue
        target = swap[x] if (swap and x in swap) else r
        cells.write_cell(os.path.join(parent, f"shard_{x:02d}"), target,
                         _items(), {"rows": []}, {"p": 1},
                         restoration_verified=True)
    return parent


def test_real_topology_has_one_control_per_shard(tmp_path):
    parent = _build_real_topology(tmp_path)
    assert len(SHARD_GROUPS) == 12
    for x in range(N_SHARDS):
        d = os.path.join(parent, f"shard_{x:02d}")
        assert len(cells.read_complete_cells(d)) == 1


def test_global_manifest_generates_from_the_parent(tmp_path):
    parent = _build_real_topology(tmp_path)
    man = continuity.generate(parent, MATRIX)
    assert man["n_complete_k0_cells"] == 12
    assert man["n_complete_nonzero_cells"] == 0
    # locations are shard-relative, proving the tree was walked
    assert all(c["relative_location"].startswith("shard_")
               for c in man["cells"])
    assert all(c["original_execution_commit"] ==
               "0b22b90455a30b8d2ee1ca86df0fc96955e4e542" for c in man["cells"])


@pytest.mark.parametrize("x", list(range(N_SHARDS)))
def test_per_shard_continuation_accepted(tmp_path, x):
    """Exactly what the future runner does: global manifest, shard out-dir."""
    parent = _build_real_topology(tmp_path)
    man = continuity.generate(parent, MATRIX)
    out = os.path.join(parent, f"shard_{x:02d}")
    res = continuity.validate_for_continuation(man, out, MATRIX)
    assert res["mode"] == "SHARD"
    assert res["local_complete_k0"] == 1
    assert res["shard_root"] == f"shard_{x:02d}"


@pytest.mark.parametrize("x", [0, 5, 11])
def test_shard_recognizes_its_own_control_and_skips_it(tmp_path, x):
    parent = _build_real_topology(tmp_path)
    out = os.path.join(parent, f"shard_{x:02d}")
    mine = [r for r in _k0_rows() if _shard_of(r) == x]
    assert len(mine) == 1
    assert cells.is_complete(out, mine[0])
    assert cells.retry_allowed(out, mine[0]) is False
    with pytest.raises(cells.CellError):
        cells.write_cell(out, mine[0], _items(), {"rows": []}, {},
                         restoration_verified=True)


@pytest.mark.parametrize("x", [0, 7, 11])
def test_other_shards_controls_are_not_required_locally(tmp_path, x):
    parent = _build_real_topology(tmp_path)
    man = continuity.generate(parent, MATRIX)
    out = os.path.join(parent, f"shard_{x:02d}")
    others = [c for c in man["cells"]
              if c["relative_location"].split(os.sep)[0] != f"shard_{x:02d}"]
    assert len(others) == 11
    for c in others:
        assert not os.path.exists(os.path.join(out, c["cell_key"]))
    continuity.validate_for_continuation(man, out, MATRIX)   # still accepted


def test_nonzero_rows_for_that_shard_remain_eligible(tmp_path):
    parent = _build_real_topology(tmp_path)
    out = os.path.join(parent, "shard_00")
    grp = SHARD_GROUPS[0]
    nz = [r for r in MATRIX["cells"]
          if _group_of(r) == grp and int(r["severity_k"]) > 0]
    assert len(nz) == 15 * 12 or len(nz) == 15 * 4
    for r in nz:
        assert not cells.is_complete(out, r)
        assert cells.retry_allowed(out, r) is True


def test_wrong_local_k0_hash_rejected(tmp_path):
    parent = _build_real_topology(tmp_path)
    man = continuity.generate(parent, MATRIX)
    target = next(c for c in man["cells"]
                  if c["relative_location"].startswith("shard_03"))
    target["items_jsonl_sha256"] = "0" * 64
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.validate_for_continuation(
            man, os.path.join(parent, "shard_03"), MATRIX)
    assert "stale manifest or altered cell" in str(e.value)


def test_control_from_another_shard_substituted_is_rejected(tmp_path):
    """shard_02 physically holds shard_03's control."""
    rows = {_shard_of(r): r for r in _k0_rows()}
    parent = _build_real_topology(tmp_path, swap={2: rows[3]})
    with pytest.raises(continuity.ContinuityError):
        # the global scan now sees a duplicate identity / missing row
        continuity.generate(parent, MATRIX)


def test_global_manifest_missing_another_shard_k0_is_rejected(tmp_path):
    parent = _build_real_topology(tmp_path, skip=9)
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.generate(parent, MATRIX)
    assert "expected 12" in str(e.value)


def test_partial_global_manifest_refused_even_for_a_healthy_shard(tmp_path):
    parent = _build_real_topology(tmp_path)
    man = continuity.generate(parent, MATRIX)
    man["cells"] = [c for c in man["cells"]
                    if not c["relative_location"].startswith("shard_09")]
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.validate_for_continuation(
            man, os.path.join(parent, "shard_00"), MATRIX)
    assert "partial global manifest" in str(e.value)


def test_manifest_declaring_nonzero_complete_is_rejected(tmp_path):
    parent = _build_real_topology(tmp_path)
    man = continuity.generate(parent, MATRIX)
    man["n_complete_nonzero_cells"] = 1
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.validate_for_continuation(
            man, os.path.join(parent, "shard_00"), MATRIX)
    assert "COMPLETE k>0" in str(e.value)


def test_complete_nonzero_cell_inside_a_shard_is_rejected(tmp_path):
    parent = _build_real_topology(tmp_path)
    man = continuity.generate(parent, MATRIX)
    grp = SHARD_GROUPS[0]
    nz = next(r for r in MATRIX["cells"]
              if _group_of(r) == grp and int(r["severity_k"]) == 1)
    cells.write_cell(os.path.join(parent, "shard_00"), nz, _items(),
                     {"rows": []}, {}, restoration_verified=True)
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.validate_for_continuation(
            man, os.path.join(parent, "shard_00"), MATRIX)
    assert "k>0" in str(e.value)


def test_out_root_outside_the_manifest_parent_is_rejected(tmp_path):
    parent = _build_real_topology(tmp_path)
    man = continuity.generate(parent, MATRIX)
    elsewhere = str(tmp_path / "somewhere_else")
    os.makedirs(elsewhere)
    with pytest.raises(continuity.ContinuityError) as e:
        continuity.validate_for_continuation(man, elsewhere, MATRIX)
    assert "neither the manifest's result root nor a shard directory" in str(e.value)


def test_old_controls_are_never_moved_or_rewritten(tmp_path):
    """Validation must leave every byte and path untouched."""
    parent = _build_real_topology(tmp_path)
    before = {}
    for dp, _, fs in os.walk(parent):
        for f in fs:
            p = os.path.join(dp, f)
            before[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    man = continuity.generate(parent, MATRIX)
    for x in range(N_SHARDS):
        continuity.validate_for_continuation(
            man, os.path.join(parent, f"shard_{x:02d}"), MATRIX)
    after = {}
    for dp, _, fs in os.walk(parent):
        for f in fs:
            p = os.path.join(dp, f)
            after[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert before == after, "continuity validation modified the namespace"


# ------------------------------------------- science-bearing file coverage --
def test_lesioned_eval_is_in_the_evaluator_identity():
    assert "execution/lesioned_eval.py" in preflight.EVALUATOR_IDENTITY_FILES
    assert "execution/lesioned_eval.py" in preflight.evaluator_identity()


def test_evaluator_identity_changes_if_lesioned_eval_changes(tmp_path):
    """The identity must actually bind this file's bytes."""
    p = os.path.join(PKG, "execution", "lesioned_eval.py")
    original = open(p, "rb").read()
    before = preflight.evaluator_identity()["_combined"]
    try:
        with open(p, "ab") as fh:
            fh.write(b"\n# audit probe\n")
        assert preflight.evaluator_identity()["_combined"] != before
    finally:
        with open(p, "wb") as fh:
            fh.write(original)
    assert preflight.evaluator_identity()["_combined"] == before


def test_lesioned_eval_is_in_package_integrity():
    sums = open(os.path.join(PKG, "SHA256SUMS")).read()
    assert "execution/lesioned_eval.py" in sums
    assert "execution/continuity.py" in sums


def test_lesioned_eval_is_in_the_no_training_scan():
    src = open(os.path.join(PKG, "execution", "preflight.py")).read()
    scan = src[src.index("# ---- K. no training path"):]
    assert "EVALUATOR_IDENTITY_FILES" in scan
    assert "execution/lesioned_eval.py" in preflight.EVALUATOR_IDENTITY_FILES
    assert "execution/continuity.py" in scan


# ============================================================================
#  CLOSURE — the real cluster-generated continuity manifest
#
#  Pins the imported artifact so it cannot drift. The outer file SHA and the
#  internal manifest_sha256 are DIFFERENT quantities by design: the first
#  hashes the final JSON bytes, the second is the generator's logical digest
#  over the manifest body. Neither is derived from the other.
# ============================================================================
REAL_MANIFEST = os.path.join(
    PKG, "LESIONING_V2_INTACT_CONTROL_CONTINUITY_MANIFEST.json")
MANIFEST_FILE_SHA = \
    "a800a75f79a15ce099a3f2e35b1068bab8fd3ec252547329d07575d9a68e1467"
MANIFEST_INTERNAL_SHA = \
    "5fc1fd38f3ff6933f7474306255cdf166499da2920bb0648259d844c818943a9"
BASE_EXECUTION_COMMIT = "0b22b90455a30b8d2ee1ca86df0fc96955e4e542"


def _real_manifest():
    return json.load(open(REAL_MANIFEST))


def test_closure_manifest_outer_file_sha_is_exact():
    got = hashlib.sha256(open(REAL_MANIFEST, "rb").read()).hexdigest()
    assert got == MANIFEST_FILE_SHA, "the imported manifest was modified"


def test_closure_manifest_internal_digest_is_exact_and_recomputes():
    m = _real_manifest()
    assert m["manifest_sha256"] == MANIFEST_INTERNAL_SHA
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    rec = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    assert rec == MANIFEST_INTERNAL_SHA


def test_closure_outer_and_internal_digests_are_distinct_concepts():
    assert MANIFEST_FILE_SHA != MANIFEST_INTERNAL_SHA


def test_closure_manifest_declares_twelve_controls_and_no_nonzero():
    m = _real_manifest()
    assert m["n_complete_k0_cells"] == 12
    assert m["n_complete_nonzero_cells"] == 0
    assert len(m["cells"]) == 12


def test_closure_manifest_binds_the_authoritative_matrix():
    assert _real_manifest()["run_matrix_sha256"] == \
        "cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a"


def test_closure_every_control_carries_the_ORIGINAL_execution_commit():
    """k=0 provenance must stay the old commit; it is never homogenised."""
    m = _real_manifest()
    assert m["original_execution_commit"] == BASE_EXECUTION_COMMIT
    for c in m["cells"]:
        assert c["original_execution_commit"] == BASE_EXECUTION_COMMIT
        assert int(c["severity_k"]) == 0
        assert c["status"] == "COMPLETE"
        assert c["restoration_verified"] is True


def test_closure_manifest_passes_the_repository_validator():
    declared = continuity._check_manifest_structure(_real_manifest(), MATRIX)
    assert len(declared) == 12
    k0 = {cells.cell_identity(r) for r in MATRIX["cells"]
          if int(r["severity_k"]) == 0}
    assert set(declared) == k0


def test_closure_one_control_per_shard_root():
    m = _real_manifest()
    shards = [c["relative_location"].split("/")[0] for c in m["cells"]]
    assert sorted(shards) == [f"shard_{i:02d}" for i in range(12)]


def test_closure_cell_keys_agree_with_authoritative_rows():
    byid = {cells.cell_identity(r): r for r in MATRIX["cells"]}
    for c in _real_manifest()["cells"]:
        assert c["cell_key"] == cells.cell_key(byid[c["cell_identity"]])


def test_closure_manifest_records_the_real_results_parent():
    assert _real_manifest()["result_root"].endswith(
        "l3_lesion_v2_results_0b22b904")


def test_closure_no_pending_placeholder_remains():
    assert not os.path.exists(os.path.join(
        PKG, "contract",
        "LESIONING_V2_INTACT_CONTROL_CONTINUITY_MANIFEST.PENDING.json"))
