"""Focused tests for the stable-zero reproduction entrypoint.

These exercise argument parsing, the missing-input error path and output
directory behaviour only.  Nothing here loads a checkpoint, GloVe, NWR/SWP data
or trains anything; the one test that actually draws the figure reads only
tracked audit tables.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import reproduce                                        # noqa: E402

STEM = "mf2_stable_zero_bottom_line"
EXPECTED = {f"{STEM}.png", f"{STEM}.pdf", f"{STEM}.svg", f"{STEM}_caption.md"}


# ------------------------------------------------------- argument parsing

def test_target_is_required():
    with pytest.raises(SystemExit):
        reproduce.build_parser().parse_args([])


def test_out_dir_is_required():
    with pytest.raises(SystemExit):
        reproduce.build_parser().parse_args(["stable-zero"])


def test_unknown_target_is_rejected():
    with pytest.raises(SystemExit):
        reproduce.build_parser().parse_args(["no-such-target",
                                             "--out-dir", "x"])


def test_parses_stable_zero_with_out_dir():
    ns = reproduce.build_parser().parse_args(["stable-zero",
                                              "--out-dir", "somewhere"])
    assert ns.target == "stable-zero"
    assert ns.out_dir == "somewhere"


# --------------------------------------------------------- declared inputs

def test_declared_inputs_are_repository_relative_and_tracked():
    inputs = reproduce.TARGETS["stable-zero"]["inputs"]
    assert inputs, "stable-zero must declare its inputs"
    for rel in inputs:
        assert not os.path.isabs(rel)
        assert "/Users/" not in rel and "/lustre/" not in rel
        assert not rel.startswith("outputs/"), (
            "reproduction inputs must not depend on the gitignored outputs/ tree")
        assert os.path.exists(os.path.join(ROOT, rel)), rel


def test_no_missing_inputs_in_a_complete_checkout():
    assert reproduce.missing_inputs("stable-zero") == []


# ------------------------------------------------------ missing-input path

def test_missing_input_reports_nonzero_and_names_the_file(tmp_path,
                                                          monkeypatch, capsys):
    monkeypatch.setitem(
        reproduce.TARGETS["stable-zero"], "inputs",
        ["reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
         "stable_zero_audit/definitely_absent.tsv"])
    rc = reproduce.main(["stable-zero", "--out-dir", str(tmp_path / "out")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "definitely_absent.tsv" in err
    assert "cannot reproduce" in err
    assert not (tmp_path / "out").exists(), (
        "nothing may be written when a required input is missing")


# --------------------------------------------------- output directory use

def test_creates_out_dir_and_writes_only_the_requested_artifacts(tmp_path):
    out = tmp_path / "nested" / "stable_zero"
    assert not out.exists()
    rc = reproduce.main(["stable-zero", "--out-dir", str(out)])
    assert rc == 0
    assert out.is_dir(), "the output directory must be created"
    assert set(os.listdir(out)) == EXPECTED


def test_reproduced_caption_matches_the_tracked_canonical_caption(tmp_path):
    out = tmp_path / "cap"
    assert reproduce.main(["stable-zero", "--out-dir", str(out)]) == 0
    tracked = os.path.join(
        ROOT, "reports/behavioral_wfe_fulllexicon_93a577f/yair_corrections/"
              "meeting_figures", f"{STEM}_caption.md")
    with open(tracked, "rb") as f:
        expected = f.read()
    with open(out / f"{STEM}_caption.md", "rb") as f:
        assert f.read() == expected
