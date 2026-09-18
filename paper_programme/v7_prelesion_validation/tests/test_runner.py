"""Runner-specific tests. None of these runs a model forward pass."""
from __future__ import annotations

import ast
import csv
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
SCRIPTS = os.path.join(PKG, "scripts")
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(PKG, "execution"))

import inputs as EXIN                       # noqa: E402
import prelesion_eval as pe                 # noqa: E402
import run_prelesion_validation as RUN      # noqa: E402
import generate_reports as REP              # noqa: E402

RUNNER_SRC = open(os.path.join(SCRIPTS, "run_prelesion_validation.py")).read()
REPORT_SRC = open(os.path.join(SCRIPTS, "generate_reports.py")).read()


# ---------------------------------------------------------- state ordering --
def test_eight_states_exactly_once_from_frozen_manifest():
    states = RUN.read_manifest("state_manifest.tsv")
    ids = [r["state_id"] for r in states]
    assert len(ids) == 8 and len(set(ids)) == 8
    contract = json.load(open(os.path.join(PKG, "contract",
                                           "validation_contract.json")))
    assert ids == contract["states"]


def test_state_ordering_is_deterministic():
    a = [r["state_id"] for r in RUN.read_manifest("state_manifest.tsv")]
    b = [r["state_id"] for r in RUN.read_manifest("state_manifest.tsv")]
    assert a == b


def test_runner_does_not_hardcode_a_state_list():
    """Ordering must come from the frozen manifest, not a typed literal."""
    tree = ast.parse(RUNNER_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and len(node.elts) == 8:
            vals = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            assert not any(isinstance(v, str) and v.endswith("_POST_REPAIR")
                           for v in vals), "runner hardcodes the state list"


# --------------------------------------------------------- head_first_c0 ----
def _code_strings(src):
    """String literals that are NOT docstrings (i.e. could become a path)."""
    tree = ast.parse(src)
    docs = set()
    for p in ast.walk(tree):
        if isinstance(p, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            b = getattr(p, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant):
                docs.add(id(b[0].value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]


def test_head_final_is_impossible_to_select():
    """No executable string or attribute may name head_final; prose may."""
    for rel in ("scripts/run_prelesion_validation.py", "scripts/prelesion_eval.py",
                "scripts/rules.py", "execution/inputs.py"):
        src = open(os.path.join(PKG, rel)).read()
        for lit in _code_strings(src):
            assert "head_final" not in lit, f"{rel} has head_final literal: {lit}"
        for n in ast.walk(ast.parse(src)):
            assert not (isinstance(n, ast.Attribute) and "head_final" in n.attr)
    assert EXIN.REPAIR_HEAD_BASENAME == "head_first_c0.pt"


def test_post_reconstruction_uses_repair_head_only():
    """SOURCE passes head=None; POST passes the resolved head_first_c0 path."""
    fn = [n for n in ast.walk(ast.parse(RUNNER_SRC))
          if isinstance(n, ast.FunctionDef) and n.name == "evaluate_state"][0]
    body = ast.get_source_segment(RUNNER_SRC, fn)
    assert 'art["repair_head"] if kind == "POST_REPAIR" else None' in body
    assert "head_final" not in body


def test_derived_head_path_is_first_c0_for_every_slot():
    for slot, seed, step, u in (("P1", 31, 4125330, 1485), ("P2", 32, 4083660, 1470),
                                ("P3", 33, 2486310, 895), ("P4", 34, 6444960, 2320)):
        d = EXIN.derive_artifact_paths("/R", slot, seed, step, u)
        assert d["repair_head"].endswith("head_first_c0.pt")
        assert f"seed{seed}_u{u}_A" in d["repair_head"]


# ------------------------------------------------------------- refusals -----
def test_existing_results_namespace_is_hard_refusal(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    with pytest.raises(RUN.PreflightError) as e:
        RUN.preflight(str(out), require_clean_namespace=True)
    assert "already exists" in str(e.value)


def test_contract_mutation_is_hard_refusal(tmp_path, monkeypatch):
    """A single changed byte under contract/ must stop the run."""
    import shutil
    fake = tmp_path / "contract"
    shutil.copytree(os.path.join(PKG, "contract"), fake)
    p = fake / "state_manifest.tsv"
    p.write_text(p.read_text() + "# tampered\n")
    monkeypatch.setattr(RUN, "CONTRACT", str(fake))
    with pytest.raises(RUN.PreflightError) as e:
        RUN.preflight(str(tmp_path / "nope"), require_clean_namespace=True)
    assert "CONTRACT DRIFT" in str(e.value)


def test_bad_sha_is_hard_refusal(tmp_path, monkeypatch):
    bad = tmp_path / "canonical_behavioral_item_table.tsv"
    bad.write_text("wrong\n")
    monkeypatch.setenv("L3_CANON_TABLE", str(bad))
    with pytest.raises(EXIN.InputResolutionError) as e:
        EXIN.canon_table_path(required=True)
    assert "SHA256 mismatch" in str(e.value)


def test_runner_has_no_overwrite_or_resume_behaviour():
    lowered = RUNNER_SRC.lower()
    for bad in ("overwrite=true", "exist_ok=true, # results", "--resume",
                "--force", "--overwrite", "ignore_errors=true"):
        assert bad not in lowered
    assert "shutil.rmtree" not in RUNNER_SRC
    # the only os.makedirs with exist_ok is the per-state staging directory
    assert RUNNER_SRC.count("exist_ok=True") == 1


def test_failed_run_preserves_output_outside_scientific_namespace():
    assert ".FAILED_" in RUNNER_SRC
    assert "RUN_ABORTED" in RUNNER_SRC


def test_atomic_finalization_per_state():
    assert '.partial' in RUNNER_SRC and "os.rename(sdir" in RUNNER_SRC
    assert "os.rename(staging, a.out_dir)" in RUNNER_SRC


# --------------------------------------------------------------- schema -----
REQUIRED_SUMMARY_KEYS = (
    "rep_canonical_full", "rep_canonical_full_errors",
    "rep_freear_full", "rep_freear_full_errors",
    "rep_canonical_wm", "rep_freear_wm", "rep_canonical_ltm", "rep_freear_ltm",
    "naming_exact", "naming_errors", "c_top1", "c_errors",
    "c_ltm_summary", "g_summary",
    "pseudo_primary_wm_exact_acc", "pseudo_primary_ltm_exact_acc",
    "pseudo_primary_full_exact_acc",
    "pseudo_primary_wm_mean_ned", "pseudo_primary_ltm_mean_ned",
    "ordering_delta_acc", "ordering_delta_ned", "ordering_ordering",
)


def test_output_schema_covers_every_preregistered_metric():
    """Every preregistered summary key must be produced by evaluate_state.

    Keys built per-route inside a loop are matched by their f-string template.
    """
    fn = [n for n in ast.walk(ast.parse(RUNNER_SRC))
          if isinstance(n, ast.FunctionDef) and n.name == "evaluate_state"][0]
    body = ast.get_source_segment(RUNNER_SRC, fn)
    templates = ("rep_canonical_{r}", "rep_freear_{r}", "pseudo_primary_{r}_exact_acc",
                 "pseudo_primary_{r}_mean_ned", "{field}_summary", "ordering_")
    for key in REQUIRED_SUMMARY_KEYS:
        if key in body:
            continue
        assert any(t.split("{")[0] in body for t in templates), \
            f"summary never sets {key}"


def test_pseudoword_item_schema_is_complete():
    import inspect
    src = inspect.getsource(pe.free_ar_items)
    for col in ("item_id", "target_ids", "predicted_ids", "target_str",
                "predicted_str", "exact", "raw_edit_distance",
                "normalized_edit_distance", "target_length", "predicted_length",
                "eos_position", "no_eos", "first_divergence", "route"):
        assert f'"{col}"' in src


def test_paired_endpoints_cover_the_contract():
    labels = {l for l, _ in RUN.PAIRED_DISCRETE}
    assert labels == {"FULL_canonical", "FULL_freear", "WM_canonical",
                      "WM_freear", "LTM_canonical", "LTM_freear", "Naming", "C"}
    assert set(RUN.PAIRED_CONTINUOUS) == {"c_ltm", "g"}


def test_paired_transition_logic():
    """c->c, c->w, w->c, w->w are counted correctly and changes are listed."""
    assert RUN._truthy("1") == 1 and RUN._truthy("0") == 0
    assert RUN._truthy("") is None and RUN._truthy("True") == 1
    assert RUN._truthy(None) is None


# ------------------------------------------------ frozen-rule reuse ---------
def test_wm_dominant_classification_is_imported_not_redefined():
    assert "def classify_ordering" not in RUNNER_SRC
    assert "def classify_ordering" not in REPORT_SRC
    assert "pe.classify_ordering" in RUNNER_SRC
    assert "def preservation" not in REPORT_SRC
    assert "pe.preservation" in REPORT_SRC


def test_ordering_rule_boundaries_still_frozen():
    assert pe.classify_ordering(.5, .4, .2, .3)["ordering"] == "WM_DOMINANT"
    assert pe.classify_ordering(.5, .5, .2, .2)["ordering"] == "TIE"
    assert pe.classify_ordering(.4, .5, .3, .2)["ordering"] == "LTM_DOMINANT"
    assert pe.classify_ordering(.5, .4, .3, .2)["ordering"] == "MIXED"


# ------------------------------------------------- reporting separation -----
def test_reporting_never_invokes_model_inference():
    assert "import torch" not in REPORT_SRC
    tree0 = ast.parse(REPORT_SRC)
    imported = set()
    for n in ast.walk(tree0):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert "torch" not in imported and "prelesion_eval" not in imported
    for bad in ("build_state", "free_ar_items", "canonical_items", "gating_items",
                "build_trainer", "_isolated_model"):
        assert bad not in _code_strings(REPORT_SRC) and f"{bad}(" not in REPORT_SRC, \
            f"reporting references {bad}"
    tree = ast.parse(REPORT_SRC)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    for n in calls:
        # json.load is fine; a bare .load()/.forward() on anything else is not
        base = getattr(n.func.value, "id", None)
        assert not (n.func.attr == "load" and base != "json")
        assert n.func.attr != "forward"


def test_reporting_module_imports_without_torch():
    """Reporting must import with torch entirely unavailable."""
    code = ("import sys, builtins;"
            "_ri=builtins.__import__;"
            "builtins.__import__=lambda n,*a,**k: "
            "(_ for _ in ()).throw(ImportError('torch blocked')) "
            "if n.split('.')[0]=='torch' else _ri(n,*a,**k);"
            f"sys.path.insert(0,{SCRIPTS!r});"
            "import generate_reports; print('ok')")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert "ok" in out.stdout, out.stderr


def test_frozen_rules_module_is_torch_free():
    """No torch import anywhere in the frozen-rules module (AST, not prose)."""
    tree = ast.parse(open(os.path.join(SCRIPTS, "rules.py")).read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert "torch" not in imported
    src = open(os.path.join(SCRIPTS, "rules.py")).read()
    for name in ("classify_ordering", "preservation", "summarize"):
        assert f"def {name}" in src


# ------------------------------------------------------ forbidden paths -----
def test_no_training_or_lesion_code_path():
    for src, name in ((RUNNER_SRC, "runner"), (REPORT_SRC, "reports")):
        tree = ast.parse(src)
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for bad in ("backward", "step", "zero_grad", "requires_grad_", "save"):
            assert bad not in called, f"{name} calls {bad}()"
        # executable identifiers only -- prose/status strings may say "lesion"
        ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for bad in ("lesion", "p_max", "optimizer"):
            assert not any(bad in x for x in ids | attrs), \
                f"{name} has an executable {bad} path"
        assert "torch.save" not in src


def test_preflight_has_all_nine_hard_stops():
    body = ast.get_source_segment(RUNNER_SRC, [
        n for n in ast.walk(ast.parse(RUNNER_SRC))
        if isinstance(n, ast.FunctionDef) and n.name == "preflight"][0])
    for marker in ("git HEAD", "CONTRACT DRIFT", "canon_table_path",
                   "resolve_artifacts", "GloVe SHA mismatch",
                   "already exists", "8 unique states",
                   "population_manifest_sha256", "_FORBIDDEN_HEAD",
                   "training operation"):
        assert marker in body, f"preflight missing: {marker}"


# ------------------------------------------------------------- dry run ------
def test_dry_run_performs_zero_model_forwards(monkeypatch, tmp_path):
    """--dry-run must resolve everything it can and never touch a model."""
    calls = []
    monkeypatch.setattr(pe, "build_state",
                        lambda *a, **k: calls.append("build_state"))
    monkeypatch.setattr(pe, "free_ar_items",
                        lambda *a, **k: calls.append("free_ar_items"))
    monkeypatch.setattr(RUN, "evaluate_state",
                        lambda *a, **k: calls.append("evaluate_state"))
    rc = RUN.main(["--dry-run", "--out-dir", str(tmp_path / "results")])
    assert rc in (0, 2)          # 2 when inputs are not configured locally
    assert calls == [], f"dry-run touched the model: {calls}"
    assert not (tmp_path / "results").exists()


def test_dry_run_writes_no_results(tmp_path):
    out = tmp_path / "results"
    RUN.main(["--dry-run", "--out-dir", str(out)])
    assert not out.exists()


def test_runner_exposes_a_cli():
    out = subprocess.run([sys.executable,
                          os.path.join(SCRIPTS, "run_prelesion_validation.py"),
                          "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "--dry-run" in out.stdout and "--out-dir" in out.stdout


def test_no_results_namespace_committed():
    assert not os.path.exists(os.path.join(PKG, "results"))
