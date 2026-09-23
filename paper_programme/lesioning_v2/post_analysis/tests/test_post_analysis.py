"""Synthetic post-analysis tests. No real scientific values are used anywhere.

Nothing here loads a model, runs a forward pass, applies a lesion, or writes
into a results namespace.
"""
from __future__ import annotations

import ast
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PA = os.path.normpath(os.path.join(HERE, ".."))
PKG = os.path.normpath(os.path.join(PA, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
sys.path.insert(0, REPO)

from paper_programme.lesioning_v2.execution import cells, preflight  # noqa: E402
from paper_programme.lesioning_v2.execution import aggregate as frozen_agg  # noqa: E402
from paper_programme.lesioning_v2.lesion_operator import battery  # noqa: E402
from paper_programme.lesioning_v2.post_analysis import (  # noqa: E402
    aggregate_multishard as agg, curves, endpoints, io_utils, validate)
from paper_programme.lesioning_v2.post_analysis.io_utils import AnalysisError  # noqa: E402

MATRIX = preflight.load_matrix()
GROUPS = sorted({(r["state_id"], r["site"]) for r in MATRIX["cells"]})


# --------------------------------------------------- endpoint identity -----
def test_route_is_load_bearing_and_frozen_key_would_collapse():
    assert endpoints.route_is_load_bearing() is True
    collapsed = endpoints.collapsed_by_frozen_key()
    assert collapsed[("repetition", "CANONICAL_FORCED_LENGTH_AR")] == (
        "rep_canonical_full", "rep_canonical_ltm", "rep_canonical_wm")
    assert collapsed[("repetition", "GENUINE_FREE_AR")] == (
        "rep_freear_full", "rep_freear_ltm", "rep_freear_wm")


def test_post_analysis_key_extends_the_frozen_key_by_route_only():
    assert set(endpoints.POST_ANALYSIS_KEY) - set(endpoints.FROZEN_EXECUTION_KEY) \
        == {"route"}
    assert set(endpoints.FROZEN_EXECUTION_KEY) <= set(endpoints.POST_ANALYSIS_KEY)


def test_endpoints_resolve_through_battery_only():
    for e in battery.ALL_ENDPOINTS:
        got = endpoints.resolve(e.task, e.route, e.decoding_convention)
        assert got.key == e.key and got.tier == e.tier


def test_unknown_triple_is_refused():
    with pytest.raises(endpoints.EndpointError):
        endpoints.resolve("repetition", "no_such_route", "GENUINE_FREE_AR")


def test_primary_and_diagnostic_sets_are_exactly_the_frozen_ones():
    assert endpoints.PRIMARY_KEYS == ("rep_canonical_full", "rep_freear_full",
                                      "naming_exact", "c_top1")
    assert endpoints.DIAGNOSTIC_KEYS == ("rep_canonical_wm", "rep_freear_wm",
                                         "rep_canonical_ltm", "rep_freear_ltm")
    assert not set(endpoints.PRIMARY_KEYS) & set(endpoints.DIAGNOSTIC_KEYS)


# ------------------------------------------- THE REGRESSION (CENTRAL sec 8) --
def _rep_rows(full_correct, wm_correct, ltm_correct, n=10):
    """Three repetition routes, same task+decoder, deliberately different acc."""
    rows = []
    for route, ncorrect in (("full", full_correct), ("wm", wm_correct),
                            ("ltm", ltm_correct)):
        for i in range(n):
            rows.append({"item_id": f"bank_{i}", "task": "repetition",
                         "route": route,
                         "decoding_convention": "CANONICAL_FORCED_LENGTH_AR",
                         "correct": 1 if i < ncorrect else 0,
                         "state_id": "PX", "site": "L1", "severity_k": 3,
                         "realization": 0})
    return rows


def test_route_aware_aggregation_keeps_three_distinct_endpoints():
    """The discovered bug, demonstrated: 9/10, 5/10, 2/10 must stay separate."""
    rows = _rep_rows(9, 5, 2, n=10)
    acc = {}
    for r in rows:
        e = endpoints.resolve(r["task"], r["route"], r["decoding_convention"])
        a = acc.setdefault(e.key, {"n": 0, "correct": 0, "tier": e.tier})
        a["n"] += 1
        a["correct"] += r["correct"]
    assert set(acc) == {"rep_canonical_full", "rep_canonical_wm",
                        "rep_canonical_ltm"}
    assert acc["rep_canonical_full"]["correct"] / acc["rep_canonical_full"]["n"] == 0.9
    assert acc["rep_canonical_wm"]["correct"] / acc["rep_canonical_wm"]["n"] == 0.5
    assert acc["rep_canonical_ltm"]["correct"] / acc["rep_canonical_ltm"]["n"] == 0.2
    # each keeps its OWN denominator; routes are never pooled
    assert all(a["n"] == 10 for a in acc.values())
    assert acc["rep_canonical_full"]["tier"] == "PRIMARY"
    assert acc["rep_canonical_wm"]["tier"] == "DIAGNOSTIC"


def test_frozen_route_omitting_key_would_collapse_them():
    """Why post-analysis cannot use SUMMARY_KEY directly. The frozen aggregator
    is NOT modified to make this pass -- the collapse is the documented reason."""
    rows = _rep_rows(9, 5, 2, n=10)
    merged = {}
    for r in rows:
        key = tuple(r[k] for k in frozen_agg.SUMMARY_KEY)
        a = merged.setdefault(key, {"n": 0, "correct": 0})
        a["n"] += 1
        a["correct"] += r["correct"]
    assert len(merged) == 1, "the frozen key should merge all three routes"
    only = next(iter(merged.values()))
    assert only["n"] == 30 and only["correct"] == 16
    blended = only["correct"] / only["n"]
    assert abs(blended - 16 / 30) < 1e-12
    for true_acc in (0.9, 0.5, 0.2):
        assert abs(blended - true_acc) > 0.01
    assert "route" not in frozen_agg.SUMMARY_KEY


def test_metric_parity_with_frozen_definition_where_route_is_unambiguous():
    """naming has one route, so post-analysis and the frozen metric agree."""
    rows = [{"item_id": f"bank_{i}", "task": "naming", "route": "full",
             "decoding_convention": "SEMANTIC_GREEDY_AR_GLOBAL_CAP",
             "correct": 1 if i < 7 else 0, "state_id": "PX", "site": "L1",
             "severity_k": 3, "realization": 0} for i in range(10)]
    mine = sum(r["correct"] for r in rows) / len(rows)
    merged = {}
    for r in rows:
        key = tuple(r[k] for k in frozen_agg.SUMMARY_KEY)
        a = merged.setdefault(key, {"n": 0, "correct": 0})
        a["n"] += 1
        a["correct"] += r["correct"]
    frozen = next(iter(merged.values()))
    assert mine == frozen["correct"] / frozen["n"] == 0.7


# ---------------------------------------------- synthetic namespace ---------
def _endpoint_items(seed=0):
    out = []
    for i, e in enumerate(battery.ALL_ENDPOINTS):
        n = 5
        for j in range(n):
            out.append({"item_id": f"bank_{j}", "task": e.task, "route": e.route,
                        "decoding_convention": e.decoding_convention,
                        "correct": 1 if (j + i + seed) % 2 == 0 else 0})
    return out


def _build(tmp_path, drop=None, dup=False, extra_endpoint=False):
    parent = str(tmp_path / "l3_lesion_v2_results_0b22b904")
    for idx, (state, site) in enumerate(GROUPS):
        shard = os.path.join(parent, f"shard_{idx:02d}")
        rows = [r for r in MATRIX["cells"]
                if r["state_id"] == state and r["site"] == site]
        for row in rows:
            if drop is not None and cells.cell_identity(row) == drop:
                continue
            items = _endpoint_items(int(row["severity_k"]))
            if extra_endpoint and int(row["severity_k"]) == 1 and idx == 0:
                items.append({"item_id": "bank_0", "task": "repetition",
                              "route": "bogus",
                              "decoding_convention": "GENUINE_FREE_AR",
                              "correct": 1})
            cells.write_cell(shard, row, items, {"rows": []}, {"p": 1},
                             restoration_verified=True)
        if dup and idx == 0:
            d = cells.cell_dir(shard, rows[0])
            import shutil
            shutil.copytree(d, d + "_copy")
    return parent


@pytest.fixture(scope="module")
def namespace(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("ns"))


def test_discovers_exactly_twelve_shards(namespace):
    assert len(agg.discover_shards(namespace)) == 12


def test_missing_shard_rejected(tmp_path, namespace):
    import shutil
    copy = str(tmp_path / "copy")
    shutil.copytree(namespace, copy)
    shutil.rmtree(os.path.join(copy, "shard_07"))
    with pytest.raises(AnalysisError) as e:
        agg.discover_shards(copy)
    assert "expected exactly" in str(e.value)


def test_full_build_matches_the_expected_census(namespace):
    res = agg.build(namespace)
    assert res["census"]["n_cells"] == 1812
    assert res["census"]["n_k0"] == 12
    assert res["census"]["n_nonzero"] == 1800
    assert res["lifecycle"] == {"staging": 0, "failed": 0}
    assert len(res["rows"]) == 14496 == agg.EXPECTED_ROWS


def test_canonical_row_count_is_asserted():
    assert agg.EXPECTED_ROWS == 1812 * 8 == 14496


def test_every_cell_has_exactly_eight_endpoints(namespace):
    res = agg.build(namespace, verify=False)
    per = {}
    for r in res["rows"]:
        k = (r["state_id"], r["site"], r["severity_k"], r["realization"])
        per.setdefault(k, set()).add(r["endpoint"])
    assert len(per) == 1812
    assert all(len(v) == 8 for v in per.values())
    assert all(v == set(endpoints.PRIMARY_KEYS) | set(endpoints.DIAGNOSTIC_KEYS)
               for v in per.values())


def test_realization_counts_are_twelve_twelve_twelve_four(namespace):
    res = agg.build(namespace, verify=False)
    per = {}
    for r in res["rows"]:
        if int(r["severity_k"]) > 0:
            per.setdefault(r["state_id"], set()).add(r["realization"])
    assert {k: len(v) for k, v in per.items()} == {
        "P1_POST_REPAIR": 12, "P2_POST_REPAIR": 12,
        "P3_POST_REPAIR": 12, "P4_POST_REPAIR": 4}


def test_missing_cell_rejected(tmp_path):
    victim = cells.cell_identity(MATRIX["cells"][5])
    parent = _build(tmp_path, drop=victim)
    with pytest.raises(AnalysisError) as e:
        agg.build(parent, verify=False)
    assert "no COMPLETE cell" in str(e.value)


def test_duplicate_cell_rejected(tmp_path):
    parent = _build(tmp_path, dup=True)
    with pytest.raises(AnalysisError) as e:
        agg.build(parent, verify=False)
    assert "duplicate cell identities" in str(e.value)


def test_unknown_endpoint_tuple_rejected(tmp_path):
    parent = _build(tmp_path, extra_endpoint=True)
    with pytest.raises((AnalysisError, endpoints.EndpointError)):
        agg.build(parent, verify=False)


def test_foreign_cell_rejected(tmp_path, namespace):
    import shutil
    copy = str(tmp_path / "c2")
    shutil.copytree(namespace, copy)
    alien = dict(MATRIX["cells"][0])
    alien["state_id"] = "PZ_POST_REPAIR"
    cells.write_cell(os.path.join(copy, "shard_00"), alien,
                     _endpoint_items(), {"rows": []}, {},
                     restoration_verified=True)
    with pytest.raises(AnalysisError) as e:
        agg.build(copy, verify=False)
    assert "matches no authoritative row" in str(e.value)


def test_cell_integrity_failure_is_detected(tmp_path, namespace):
    import shutil
    copy = str(tmp_path / "c3")
    shutil.copytree(namespace, copy)
    d = cells.cell_dir(os.path.join(copy, "shard_00"), MATRIX["cells"][0])
    with open(os.path.join(d, "items.jsonl"), "a") as fh:
        fh.write("{}\n")
    with pytest.raises(cells.CellError):
        agg.build(copy, verify=True)


# ----------------------------------------------------------- curves --------
def test_k0_dispersion_is_na_never_zero(namespace):
    res = agg.build(namespace, verify=False)
    curve = curves.curve_summary(res["rows"])
    k0 = [r for r in curve if r["severity_k"] == 0]
    assert len(k0) == 12 * 8
    for r in k0:
        assert r["n_realizations"] == 1
        assert r["sd_exact_match"] is None
        assert r["mean_exact_match"] == r["min_exact_match"] == r["max_exact_match"]


def test_nonzero_uses_real_realization_counts(namespace):
    res = agg.build(namespace, verify=False)
    curve = curves.curve_summary(res["rows"])
    for r in curve:
        if r["severity_k"] == 0:
            continue
        want = 4 if r["state_id"] == "P4_POST_REPAIR" else 12
        assert r["n_realizations"] == want


def test_all_sixteen_severities_present(namespace):
    res = agg.build(namespace, verify=False)
    curve = curves.curve_summary(res["rows"])
    assert sorted({r["severity_k"] for r in curve}) == list(range(16))


def test_baselines_keep_all_twelve_records_verbatim(namespace):
    res = agg.build(namespace, verify=False)
    base = curves.intact_baselines(res["rows"])
    assert len({(r["state_id"], r["site"]) for r in base}) == 12
    assert len(base) == 12 * 8


def test_k0_cross_site_consistency_is_reported_not_enforced(namespace):
    res = agg.build(namespace, verify=False)
    cons = curves.k0_cross_site_consistency(curves.intact_baselines(res["rows"]))
    assert len(cons) == 4 * 8
    for c in cons:
        assert c["n_sites"] == 3
        assert isinstance(c["identical_across_sites"], bool)


def test_k15_table_lists_individual_realizations(namespace):
    res = agg.build(namespace, verify=False)
    k15 = curves.max_severity_table(res["rows"])
    assert len(k15) == 12 * 8
    for r in k15:
        want = 4 if r["state_id"] == "P4_POST_REPAIR" else 12
        assert r["n_realizations"] == want
        assert len(r["realization_values"].split("|")) == want


def test_no_inferential_quantity_is_computed():
    src = open(os.path.join(PA, "curves.py")).read()
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ("ttest", "pvalue", "pval", "chi2", "bootstrap", "auc",
                "scipy", "statsmodels", "confidence"):
        assert not any(bad in x.lower() for x in names | attrs)


# ------------------------------------------------------- read-only ----------
def test_out_dir_inside_results_parent_is_refused(tmp_path):
    root = str(tmp_path / "res")
    os.makedirs(root)
    with pytest.raises(AnalysisError):
        io_utils.guard_out_dir(os.path.join(root, "analysis"), root)
    with pytest.raises(AnalysisError):
        io_utils.guard_out_dir(root, root)
    assert io_utils.guard_out_dir(str(tmp_path / "out"), root)


def test_analysis_never_writes_into_the_namespace(tmp_path, namespace):
    import hashlib
    import shutil
    copy = str(tmp_path / "ro")
    shutil.copytree(namespace, copy)
    before = {}
    for dp, _, fs in os.walk(copy):
        for f in fs:
            p = os.path.join(dp, f)
            before[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    agg.build(copy)
    after = {}
    for dp, _, fs in os.walk(copy):
        for f in fs:
            p = os.path.join(dp, f)
            after[p] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    assert before == after


def test_no_model_or_training_path_in_post_analysis():
    for f in sorted(os.listdir(PA)):
        if not f.endswith(".py"):
            continue
        src = open(os.path.join(PA, f)).read()
        tree = ast.parse(src)
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                imported.add(n.module.split(".")[0])
        assert "torch" not in imported, f"{f} imports torch"
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for bad in ("backward", "zero_grad", "requires_grad_", "load_state_dict"):
            assert bad not in called, f"{f} calls {bad}()"
        for bad in ("build_state", "free_ar_items", "build_trainer",
                    "connectivity_lesion", "activation_injection"):
            assert f"{bad}(" not in src, f"{f} references {bad}"


def test_frozen_execution_files_untouched():
    import subprocess
    base = "10f68c5fcd53338cc090e80722259fc07476fca0"
    out = subprocess.run(
        ["git", "diff", "--name-only", base, "--",
         "paper_programme/lesioning_v2/lesion_operator",
         "paper_programme/lesioning_v2/execution",
         "paper_programme/lesioning_v2/scripts",
         "paper_programme/lesioning_v2/contract",
         "paper_programme/lesioning_v2/SHA256SUMS"],
        cwd=REPO, capture_output=True, text=True)
    assert out.stdout.strip() == "", f"frozen files changed: {out.stdout}"


# --------------------------------------------------------- determinism ------
def test_output_ordering_is_deterministic(namespace):
    a = agg.build(namespace, verify=False)["rows"]
    b = agg.build(namespace, verify=False)["rows"]
    key = lambda r: (r["state_id"], r["site"], r["severity_k"],
                     r["realization"], r["endpoint"])
    assert [key(r) for r in a] == [key(r) for r in b]
    assert [key(r) for r in a] == sorted(key(r) for r in a)


def test_tsv_write_is_deterministic(tmp_path, namespace):
    res = agg.build(namespace, verify=False)
    p1 = str(tmp_path / "a.tsv")
    p2 = str(tmp_path / "b.tsv")
    h1 = io_utils.write_tsv(p1, res["rows"], agg.CANONICAL_COLUMNS)
    h2 = io_utils.write_tsv(p2, res["rows"], agg.CANONICAL_COLUMNS)
    assert h1 == h2


def test_validation_payload_is_complete(namespace, tmp_path):
    res = agg.build(namespace, verify=False)
    base = curves.intact_baselines(res["rows"])
    cons = curves.k0_cross_site_consistency(base)
    val = validate.build(namespace, res, res["rows"], cons, {"x.tsv": "abc"})
    for k in ("execution_status", "validity_status", "results_parent",
              "execution_commits", "run_matrix_sha256", "n_matrix_cells",
              "n_complete_cells", "n_unique_cell_identities", "n_k0",
              "n_nonzero", "n_staging", "n_failed", "primary_endpoints",
              "diagnostic_endpoints", "states", "sites", "severity_levels",
              "realizations_nonzero", "generated_artifact_sha256",
              "analysis_schema_observation"):
        assert k in val
    assert val["run_matrix_sha256_match"] is True
    assert val["statistical_inference_performed"] is False
    assert val["results_namespace_written_to"] is False
    assert val["analysis_schema_observation"]["route_is_load_bearing"] is True
