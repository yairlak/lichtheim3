"""Fail-closed validator for a future scientific run's output.

Checks a candidate run summary against `EXPECTED_OUTPUT_MANIFEST.json`. Exits
non-zero on the FIRST unmet requirement: a run missing any required section,
field, shard or record count is INCOMPLETE and must not be reported.

Nothing here executes science; it only inspects artifacts a run would have
produced. It is exercised in tests against synthetic payloads.

    python3 scripts/gate_x_lesion/validate_output_manifest.py <summary.json>
    python3 scripts/gate_x_lesion/validate_output_manifest.py <summary.json> \
        <manifest.json> --test-only        # quarantined smoke fixture only
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, "paper_programme", "gate_x_lesion_recovery")
MANIFEST = os.path.join(BASE, "EXPECTED_OUTPUT_MANIFEST.json")

SECTIONS = ("PROVENANCE", "VALIDITY", "RAW_PAIRED_RESULTS", "ROUTE_DIAGNOSTICS",
            "ROBUSTNESS", "CLASSIFICATION", "ITEM_LEVEL_AUDIT", "RUN_COMPLETION")


class OutputIncomplete(RuntimeError):
    """A required artifact, field or count is missing. The run may not be reported."""


def load_manifest(path: str = MANIFEST) -> dict:
    with open(path) as f:
        return json.load(f)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise OutputIncomplete(msg)


def validate(summary: dict, manifest: Optional[dict] = None,
             *, shard_paths: Optional[List[str]] = None,
             expect_scientific: bool = True) -> Dict[str, object]:
    """Validate `summary`. Raises `OutputIncomplete` on the first failure.

    `expect_scientific` defaults to True and is NEVER relaxed for a real run: the
    scientific path keeps every pinned value and count. A TEST_ONLY harness may
    pass `expect_scientific=False` together with a strictly test-only `manifest`
    fixture, which exercises this same code with smoke-scale counts. Quarantined
    output can therefore never be accepted against the scientific manifest.
    """
    m = manifest or load_manifest()

    # Quarantined output may never masquerade as scientific output.
    quarantined = bool(summary.get("TEST_ONLY")
                       or summary.get("NOT_SCIENTIFIC_RESULT")
                       or summary.get("smoke"))
    if expect_scientific and quarantined:
        raise OutputIncomplete(
            "summary is marked TEST_ONLY / NOT_SCIENTIFIC_RESULT and cannot be "
            "validated as scientific output")
    if not expect_scientific and not quarantined:
        raise OutputIncomplete(
            "expect_scientific=False requires the summary to be marked TEST_ONLY")

    # --- every required section must be present ------------------------------
    for sec in SECTIONS:
        _require(sec in summary, f"missing required section: {sec}")

    # --- an aborted run must not carry a classification ----------------------
    completion = summary["RUN_COMPLETION"]
    status = completion.get("implementation_failure_status")
    if status not in (None, "NONE"):
        _require(not summary.get("CLASSIFICATION"),
                 f"implementation_failure_status={status} but a CLASSIFICATION was "
                 f"emitted; no classification may follow an implementation failure")
        raise OutputIncomplete(
            f"run aborted with implementation_failure_status={status}; "
            f"the run is not reportable")

    # --- required fields per section ----------------------------------------
    for sec in ("PROVENANCE", "RUN_COMPLETION"):
        for field in m[sec]["fields"]:
            _require(field in summary[sec], f"{sec}: missing field {field}")
            _require(summary[sec][field] is not None,
                     f"{sec}: field {field} is null")

    # --- pinned provenance values -------------------------------------------
    for k, v in m["PROVENANCE"]["pinned_values"].items():
        got = summary["PROVENANCE"].get(k)
        _require(got == v, f"PROVENANCE.{k} = {got!r}, contract pins {v!r}")

    # --- required completion values -----------------------------------------
    for k, v in m["RUN_COMPLETION"]["required_values"].items():
        got = completion.get(k)
        _require(got == v, f"RUN_COMPLETION.{k} = {got!r}, must be {v!r}")

    # --- record counts -------------------------------------------------------
    _require(len(summary["VALIDITY"]) == m["VALIDITY"]["expected_records"],
             f"VALIDITY: {len(summary['VALIDITY'])} records, expected "
             f"{m['VALIDITY']['expected_records']}")
    _require(len(summary["RAW_PAIRED_RESULTS"])
             == m["RAW_PAIRED_RESULTS"]["expected_records"],
             f"RAW_PAIRED_RESULTS: {len(summary['RAW_PAIRED_RESULTS'])} records, "
             f"expected {m['RAW_PAIRED_RESULTS']['expected_records']}")

    # --- validity records ----------------------------------------------------
    for rec in summary["VALIDITY"]:
        for field in m["VALIDITY"]["fields"]:
            _require(field in rec, f"VALIDITY record missing {field}")
        _require(rec["validity_reason"] in m["VALIDITY"]["validity_reason_enum"],
                 f"VALIDITY: unknown reason {rec['validity_reason']!r}")

    # --- robustness must carry no significance machinery ---------------------
    for rec in summary["ROBUSTNESS"]:
        _require(rec.get("rule_id") == m["ROBUSTNESS"]["rule_id"],
                 f"ROBUSTNESS: rule_id {rec.get('rule_id')!r} is not the frozen rule")
        _require(rec.get("verdict") in m["ROBUSTNESS"]["verdict_enum"],
                 f"ROBUSTNESS: unknown verdict {rec.get('verdict')!r}")
        for banned in m["ROBUSTNESS"]["must_not_contain"]:
            _require(banned not in rec,
                     f"ROBUSTNESS: forbidden field {banned} present — the frozen "
                     f"rule uses no significance test and no pooling")

    # --- classification ------------------------------------------------------
    for rec in summary["CLASSIFICATION"]:
        for field in m["CLASSIFICATION"]["fields"]:
            _require(field in rec, f"CLASSIFICATION record missing {field}")
        _require(rec["CANONICAL_OUTCOME"] in m["CLASSIFICATION"]["outcome_enum"],
                 f"bad CANONICAL_OUTCOME {rec['CANONICAL_OUTCOME']!r}")
        _require(rec["FREE_AR_OUTCOME"] in m["CLASSIFICATION"]["outcome_enum"],
                 f"bad FREE_AR_OUTCOME {rec['FREE_AR_OUTCOME']!r}")
        _require(rec["JOINT_OUTCOME"] in m["CLASSIFICATION"]["joint_enum"],
                 f"bad JOINT_OUTCOME {rec['JOINT_OUTCOME']!r}")
        # the joint label must be consistent with the two stage letters
        if rec["CANONICAL_OUTCOME"] == rec["FREE_AR_OUTCOME"]:
            expect = "CONCORDANT_" + rec["CANONICAL_OUTCOME"]
        else:
            expect = "MIXED_DECODING"
        _require(rec["JOINT_OUTCOME"] == expect,
                 f"JOINT_OUTCOME {rec['JOINT_OUTCOME']!r} inconsistent with "
                 f"({rec['CANONICAL_OUTCOME']}, {rec['FREE_AR_OUTCOME']}); "
                 f"expected {expect}")

    # --- shards --------------------------------------------------------------
    if shard_paths is not None:
        want = m["required_shards"]["expected_item_level_count"]
        _require(len(shard_paths) == want,
                 f"item-level shards: {len(shard_paths)}, expected {want}")
        # Match the quarantine MARKER, not just the full path: a relative path
        # such as "NOT_SCIENTIFIC_RESULT/x.tsv" must not slip through.
        marker = os.path.basename(m["must_not_write_to"].rstrip("/"))
        for p in shard_paths:
            _require(marker not in p.split(os.sep) and marker not in p.split("/"),
                     f"scientific shard inside the quarantine namespace: {p}")
            _require(m["must_not_write_to"] not in p,
                     f"scientific shard inside the quarantine namespace: {p}")

    return {"ok": True, "sections": list(SECTIONS)}


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) not in (1, 2):
        print(__doc__)
        return 2
    with open(args[0]) as f:
        summary = json.load(f)
    man = load_manifest(args[1]) if len(args) == 2 else None
    scientific = "--test-only" not in flags
    try:
        validate(summary, man, expect_scientific=scientific)
    except OutputIncomplete as e:
        print(f"HARD STOP: output incomplete — {e}", file=sys.stderr)
        return 1
    print("output manifest validation PASSED"
          + ("" if scientific else "  [TEST_ONLY fixture]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
