"""Regression tests for REPORTING_ATTEMPT_1 (KeyError: 'cc').

paired_report iterated every key of each slot in SOURCE_POST_PAIRED.json and
treated it as a binary transition table. Four of the thirteen entries per slot
are not: `n_items_paired` (metadata), the two `*_delta_summary` dicts and the
two `*_n_changed` scalars. Reporting now iterates an explicit frozen tuple of
the eight binary endpoints and fails closed on schema drift.

No model is run and no scientific result is modified by these tests.
"""
from __future__ import annotations

import ast
import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
SCRIPTS = os.path.join(PKG, "scripts")
sys.path.insert(0, SCRIPTS)

import generate_reports as REP        # noqa: E402

REPORT_SRC = open(os.path.join(SCRIPTS, "generate_reports.py")).read()


def _transition(cc=10, cw=0, wc=1, ww=5, ids=None):
    return {"cc": cc, "cw": cw, "wc": wc, "ww": ww,
            "changed_item_ids": ids if ids is not None else ["bank_7"],
            "undefined": 0}


def frozen_like_paired() -> dict:
    """The EXACT key set the cluster run produced, per slot."""
    out = {}
    for slot in REP.SLOTS:
        d = {"n_items_paired": 29571}
        for ep in REP.PAIRED_TRANSITION_ENDPOINTS:
            d[ep] = _transition()
        d["c_ltm_delta_summary"] = {"n": 29571, "mean": 0.0012, "sd": 0.004,
                                    "min": -0.01, "p01": -0.008, "p05": -0.005,
                                    "p25": 0.0, "p50": 0.0, "p75": 0.002,
                                    "p95": 0.007, "p99": 0.009, "max": 0.02}
        d["c_ltm_n_changed"] = 14822
        d["g_delta_summary"] = dict(d["c_ltm_delta_summary"])
        d["g_n_changed"] = 14822
        out[slot] = d
    return out


@pytest.fixture
def results_dir(tmp_path):
    r = tmp_path / "results"
    r.mkdir()
    json.dump(frozen_like_paired(), open(r / "SOURCE_POST_PAIRED.json", "w"))
    return str(r)


# ---------------------------------------------------------------- 1 and 2 ---
def test_paired_report_succeeds_on_the_frozen_schema(results_dir):
    md = REP.paired_report(results_dir, {})
    assert "V7 SOURCE -> POST_REPAIR PAIRED ANALYSIS" in md
    for slot in REP.SLOTS:
        assert f"## {slot}" in md


def test_renders_exactly_the_eight_binary_endpoints_per_slot(results_dir):
    md = REP.paired_report(results_dir, {})
    assert len(REP.PAIRED_TRANSITION_ENDPOINTS) == 8
    for ep in REP.PAIRED_TRANSITION_ENDPOINTS:
        # one table row per slot
        assert md.count(f"| {ep} |") == len(REP.SLOTS)


def test_endpoint_order_is_the_frozen_tuple_not_file_order(results_dir):
    """Rendering must not depend on dict/key ordering in the JSON."""
    paired = frozen_like_paired()
    shuffled = {s: dict(reversed(list(d.items()))) for s, d in paired.items()}
    json.dump(shuffled, open(os.path.join(results_dir,
                                          "SOURCE_POST_PAIRED.json"), "w"))
    md = REP.paired_report(results_dir, {})
    first = md[md.index("## P1"):md.index("## P2")]
    positions = [first.index(f"| {ep} |") for ep in REP.PAIRED_TRANSITION_ENDPOINTS]
    assert positions == sorted(positions), "endpoints not in frozen tuple order"


# ------------------------------------------------------------- 3, 4 and 5 ---
def test_metadata_is_not_treated_as_a_transition_table(results_dir):
    md = REP.paired_report(results_dir, {})
    assert "| n_items_paired |" not in md
    assert "items paired: 29571" in md          # reported as metadata instead


def test_delta_summaries_are_not_treated_as_transition_tables(results_dir):
    md = REP.paired_report(results_dir, {})
    for key in REP.PAIRED_CONTINUOUS_SUMMARIES:
        assert f"| {key} |" not in md
        assert key in md                        # still reported, as continuous
    # and never rendered with transition semantics: scope to ONE slot's
    # continuous block (the next slot's table header legitimately says c->c)
    p1 = md[md.index("## P1"):md.index("## P2")]
    cont = p1[p1.index("continuous endpoints"):]
    assert "c->c" not in cont and "|" not in cont.split("\n")[1]


def test_n_changed_scalars_are_not_treated_as_transition_tables(results_dir):
    md = REP.paired_report(results_dir, {})
    for key in REP.PAIRED_CONTINUOUS_SCALARS:
        assert f"| {key} |" not in md
        assert f"{key}: 14822" in md


def test_continuous_entries_are_excluded_from_the_endpoint_tuple():
    for key in (REP.PAIRED_CONTINUOUS_SUMMARIES + REP.PAIRED_CONTINUOUS_SCALARS
                + REP.PAIRED_METADATA):
        assert key not in REP.PAIRED_TRANSITION_ENDPOINTS


# ------------------------------------------------------------- 6 and 7 ------
def test_missing_expected_endpoint_fails_closed(results_dir):
    paired = frozen_like_paired()
    del paired["P2"]["Naming"]
    json.dump(paired, open(os.path.join(results_dir,
                                        "SOURCE_POST_PAIRED.json"), "w"))
    with pytest.raises(REP.ReportSchemaError) as e:
        REP.paired_report(results_dir, {})
    assert "Naming" in str(e.value) and "missing" in str(e.value)


def test_endpoint_missing_transition_keys_fails_closed(results_dir):
    paired = frozen_like_paired()
    del paired["P3"]["C"]["cc"]
    json.dump(paired, open(os.path.join(results_dir,
                                        "SOURCE_POST_PAIRED.json"), "w"))
    with pytest.raises(REP.ReportSchemaError) as e:
        REP.paired_report(results_dir, {})
    assert "cc" in str(e.value)


def test_non_dict_endpoint_fails_closed(results_dir):
    paired = frozen_like_paired()
    paired["P4"]["WM_freear"] = 17
    json.dump(paired, open(os.path.join(results_dir,
                                        "SOURCE_POST_PAIRED.json"), "w"))
    with pytest.raises(REP.ReportSchemaError):
        REP.paired_report(results_dir, {})


def test_missing_slot_fails_closed(results_dir):
    paired = frozen_like_paired()
    del paired["P4"]
    json.dump(paired, open(os.path.join(results_dir,
                                        "SOURCE_POST_PAIRED.json"), "w"))
    with pytest.raises(REP.ReportSchemaError):
        REP.paired_report(results_dir, {})


def test_missing_paired_file_fails_closed(tmp_path):
    with pytest.raises(REP.ReportSchemaError):
        REP.paired_report(str(tmp_path), {})


def test_endpoints_are_not_discovered_by_probing_for_cc():
    """The frozen tuple must be explicit, never inferred from the data."""
    src = REP.REPORT_SRC if hasattr(REP, "REPORT_SRC") else REPORT_SRC
    fn = [n for n in ast.walk(ast.parse(src))
          if isinstance(n, ast.FunctionDef) and n.name == "paired_report"][0]
    body = ast.get_source_segment(src, fn)
    assert "PAIRED_TRANSITION_ENDPOINTS" in body
    assert 'if "cc" in' not in body and "'cc' in" not in body
    assert "sorted(d.items())" not in body, "still iterating raw file keys"


# ------------------------------------------------------------------ 8 -------
def test_classification_output_is_unchanged():
    """The classification path is untouched by this reporting fix."""
    def st(order):
        return {"ordering_ordering": order, "rep_canonical_wm": 0.9,
                "rep_canonical_ltm": 0.9, "pseudo_primary_n": 378,
                "pseudo_primary_wm_no_eos": 0, "pseudo_primary_ltm_no_eos": 0,
                "pseudo_primary_full_no_eos": 0}
    states = {}
    for slot in REP.SLOTS:
        states[f"{slot}_SOURCE"] = st("WM_DOMINANT")
        states[f"{slot}_POST_REPAIR"] = st("WM_DOMINANT")
    cls = REP.classify(states)
    assert cls["PRELESION_VALIDATION_STATUS"] == "V7_LESION_READY"
    assert cls["ARM_A_PSEUDOWORD_PRESERVATION"] == "YES"
    states["P2_POST_REPAIR"] = st("LTM_DOMINANT")
    assert REP.classify(states)["PRELESION_VALIDATION_STATUS"] == \
        "V7_NOT_LESION_READY"
    states["P2_POST_REPAIR"] = st("MIXED")
    assert REP.classify(states)["PRELESION_VALIDATION_STATUS"] == \
        "V7_LESION_READY_WITH_QUALIFICATION"


def test_classification_rules_still_imported_not_redefined():
    assert "def classify_ordering" not in REPORT_SRC
    assert "def preservation" not in REPORT_SRC
    assert "pe.preservation" in REPORT_SRC


# ------------------------------------------------------------------ 9 -------
def test_reporting_modifies_no_scientific_result_file(tmp_path):
    """Files listed in FILE_SHA256SUMS must be byte-identical afterwards."""
    import hashlib
    r = tmp_path / "results"
    r.mkdir()
    (r / "SOURCE_POST_PAIRED.json").write_text(
        json.dumps(frozen_like_paired()))
    payload = b"frozen scientific bytes\n"
    (r / "STATE_SUMMARY.tsv").write_bytes(payload)
    lines = []
    for rel in ("SOURCE_POST_PAIRED.json", "STATE_SUMMARY.tsv"):
        h = hashlib.sha256((r / rel).read_bytes()).hexdigest()
        lines.append(f"{h}  {rel}")
    (r / "FILE_SHA256SUMS").write_text("\n".join(lines) + "\n")

    before = REP.verify_frozen_results(str(r))
    REP.paired_report(str(r), {})            # rendering must not write
    after = REP.verify_frozen_results(str(r))
    assert before == after
    assert (r / "STATE_SUMMARY.tsv").read_bytes() == payload


def test_verify_frozen_results_detects_tampering(tmp_path):
    import hashlib
    r = tmp_path / "results"
    r.mkdir()
    (r / "x.tsv").write_bytes(b"a\n")
    h = hashlib.sha256(b"a\n").hexdigest()
    (r / "FILE_SHA256SUMS").write_text(f"{h}  x.tsv\n")
    assert REP.verify_frozen_results(str(r))
    (r / "x.tsv").write_bytes(b"tampered\n")
    with pytest.raises(REP.ReportSchemaError) as e:
        REP.verify_frozen_results(str(r))
    assert "ALTERED" in str(e.value)


def test_reporting_writes_only_under_reports_subdir():
    fn = [n for n in ast.walk(ast.parse(REPORT_SRC))
          if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    body = ast.get_source_segment(REPORT_SRC, fn)
    assert 'out = os.path.join(a.results, "reports")' in body
    for call in ("open(os.path.join(out", "json.dump(cls, open(os.path.join(out"):
        assert call in body
    assert "verify_frozen_results" in body


def test_reporting_still_runs_no_model():
    tree = ast.parse(REPORT_SRC)
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    assert "torch" not in imported and "prelesion_eval" not in imported
