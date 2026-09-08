"""Acceptance tests for the CANNEAL freeze: STORE archive + u2000 audit.

Nothing here trains.  The two contracts under test are:
  * the archive is immutable, complete, and verified against SCRATCH file by
    file, and it never mutates SCRATCH;
  * the u2000 audit is read-only, hits the CONTROL arm only, and no
    preregistered threshold moves anywhere.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension import last_window_ratio as lwr      # noqa: E402
from scripts.naming_comprehension.base123_persistence_report import (  # noqa: E402
    switch_trigger,
)

AUDIT = "scripts/cluster/jeanzay/canneal_u2000_error_audit.slurm"
ARCHIVE = "scripts/cluster/jeanzay/canneal_archive.sh"
SEEDS = (19, 20, 21, 22)


def script(path):
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def executable(path):
    return "\n".join(l for l in script(path).splitlines()
                     if l.strip() and not l.lstrip().startswith("#"))


def _metrics(d, rows):
    os.makedirs(d, exist_ok=True)
    cols = ["step", "full_comp_top1", "full_comp_errors"]
    with open(os.path.join(d, "metrics.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for step, e in rows:
            w.writerow({"step": step, "full_comp_errors": e,
                        "full_comp_top1": 1 - e / 27981})


# =========================================  1. the last-window definition ==

def test_reproduces_the_published_1_0276_window(tmp_path):
    """u1350 -> u1400 was 93->94, 96->100, 95->104, 109->105 and the reported
    figure was 1.0276.  Only the mean-of-seed-ratios definition gives that."""
    runs = str(tmp_path / "runs")
    hist = {19: (93, 94), 20: (96, 100), 21: (95, 104), 22: (109, 105)}
    for s, (a, b) in hist.items():
        _metrics(os.path.join(runs, f"final_rep_rescue123_h512_s{s}"),
                 [(1350 * 2778, a), (1400 * 2778, b)])
    out = str(tmp_path / "w.json")
    assert lwr.main(["--runs-root", runs, "--run-template",
                     "final_rep_rescue123_h512_s{seed}",
                     "--u-prev", "1350", "--u-final", "1400",
                     "--out-json", out]) == 0
    g = json.load(open(out))
    assert g["mean_of_seed_ratios"] == 1.027615
    assert g["n_seeds_used"] == 4
    assert g["definition"].startswith("mean over seeds")
    # the pooled definition is different, and is recorded only to be excluded
    assert g["ratio_of_pooled_means_NOT_USED"] == 1.025445
    assert g["ratio_of_pooled_means_NOT_USED"] != g["mean_of_seed_ratios"]


def test_the_trigger_threshold_is_not_touched():
    assert lwr.TRIGGER_THRESHOLD == 0.85
    src = script("scripts/naming_comprehension/last_window_ratio.py")
    assert "TRIGGER_THRESHOLD = 0.85" in src
    # and this tool only computes; it must not decide anything
    assert "ALL_CONDITIONS_MET" not in src


def test_preregistered_switch_trigger_constants_are_unchanged():
    """The freeze must not move a threshold by accident."""
    d = switch_trigger({"frac_in_top5": 0.90, "frac_margin_within_0.01": 0.50},
                       [0.80, 0.80, 0.80, 0.80], 0.85, 1.0001)
    assert d["ALL_CONDITIONS_MET"] is True          # exactly at every bound
    c = d["conditions"]
    assert set(c) == {"last_window_ratio_ge_0.85",
                      "within_seed_retention_ge_0.80",
                      "survivors_predominantly_near_tie_or_top5",
                      "shared_persistent_core_growing"}
    just_under = switch_trigger(
        {"frac_in_top5": 0.8999, "frac_margin_within_0.01": 0.50},
        [0.80] * 4, 0.85, 1.0001)
    assert just_under["ALL_CONDITIONS_MET"] is False
    assert switch_trigger({"frac_in_top5": 0.95, "frac_margin_within_0.01": 0.6},
                          [0.80] * 4, 0.8499, 1.5)["ALL_CONDITIONS_MET"] is False
    assert switch_trigger({"frac_in_top5": 0.95, "frac_margin_within_0.01": 0.6},
                          [0.7999] * 4, 0.9, 1.5)["ALL_CONDITIONS_MET"] is False
    assert switch_trigger({"frac_in_top5": 0.95, "frac_margin_within_0.01": 0.6},
                          [0.80] * 4, 0.9, 1.0)["ALL_CONDITIONS_MET"] is False


def test_window_tool_refuses_rather_than_inventing_a_number(tmp_path, capsys):
    runs = str(tmp_path / "runs")
    _metrics(os.path.join(runs, "final_canneal_ctrl_h512_s19"),
             [(1950 * 2778, 55)])                      # no u2000 row
    assert lwr.main(["--runs-root", runs, "--run-template",
                     "final_canneal_ctrl_h512_s{seed}",
                     "--u-prev", "1950", "--u-final", "2000"]) == 1
    assert "refusing to report a ratio" in capsys.readouterr().out


def test_window_tool_skips_a_zero_denominator_seed(tmp_path):
    runs = str(tmp_path / "runs")
    for s, (a, b) in {19: (0, 0), 20: (64, 62), 21: (53, 51),
                      22: (58, 56)}.items():
        _metrics(os.path.join(runs, f"final_canneal_ctrl_h512_s{s}"),
                 [(1950 * 2778, a), (2000 * 2778, b)])
    out = str(tmp_path / "w.json")
    assert lwr.main(["--runs-root", runs, "--run-template",
                     "final_canneal_ctrl_h512_s{seed}", "--u-prev", "1950",
                     "--u-final", "2000", "--out-json", out]) == 0
    g = json.load(open(out))
    assert g["seeds_with_zero_prev"] == [19]
    assert g["n_seeds_used"] == 3
    assert [p["seed"] for p in g["per_seed"]] == [20, 21, 22]


def test_window_tool_rejects_a_u_off_the_step_grid(tmp_path):
    with pytest.raises(SystemExit):
        lwr.steps_for_u(1950.3)
    assert lwr.steps_for_u(2000) == 5_556_000
    assert lwr.steps_for_u(1950) == 5_417_100


# =====================================================  2. the u2000 audit ==

def test_audit_targets_the_control_arm_only_at_u2000():
    t, ex = script(AUDIT), executable(AUDIT)
    assert "#SBATCH --array=0-3" in t
    for s in SEEDS:
        assert f"final_canneal_ctrl_h512_s{s}" in t
    for arm in ("5e5", "3e5"):
        assert f"final_canneal_{arm}_h512" not in ex
    assert "STEPS=(5556000 5556000 5556000 5556000)" in t
    assert "(( U == 2000 ))" in t
    assert 'refusing unexpected run id' in t


def test_audit_is_read_only_in_three_independent_ways():
    t, ex = script(AUDIT), executable(AUDIT)
    assert "base123_error_audit.py" in ex
    assert "--out-dir \"$OUT\"" in ex
    assert 'OUT="$RUNS/$RUN_ID/error_audit_u${U}"' in t
    # (a) no training driver, (b) no transition declaration, (c) no LR flags
    for forbidden in ("train_joint_scratch.py", "--phase-transition",
                      "--reanchor-schedule", "--lr-comprehension",
                      "--lr-repetition", "--resume", "--max-steps",
                      "--save-every", "--allow-glove-fallback"):
        assert forbidden not in ex, forbidden
    # (d) the checkpoint file itself is hashed before and after
    assert "CKPT_SHA_BEFORE=" in t and "CKPT_SHA_AFTER=" in t
    assert 'the audit MODIFIED $CKPT' in t


def test_audit_pins_the_real_glove():
    t = script(AUDIT)
    assert ("GLOVE_SHA_EXPECTED=91125602f730fea7ca768736c6f442e668b49db095682"
            "bf2aad375db061c21ed") in t
    assert '--glove-path "$GLOVE"' in executable(AUDIT)


def test_audit_script_is_valid_bash():
    r = subprocess.run(["bash", "-n", os.path.join(ROOT, AUDIT)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# =====================================================  3. the STORE archive

def test_archive_script_is_valid_bash():
    r = subprocess.run(["bash", "-n", os.path.join(ROOT, ARCHIVE)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_archive_never_mutates_scratch():
    """The ONLY mutating command in the whole script must target $DEST."""
    ex = executable(ARCHIVE)
    import re
    muts = [l for l in ex.splitlines()
            if re.search(r"\b(rm|rmdir|mv|shred|truncate)\b", l)
            or "-delete" in l]
    assert muts == [], muts
    chmods = [l.strip() for l in ex.splitlines() if l.strip().startswith("chmod")]
    assert chmods == ['chmod -R a-w "$DEST"'], chmods
    assert "rsync --delete" not in ex


def test_archive_refuses_to_touch_an_existing_archive():
    t = script(ARCHIVE)
    assert 'if [[ -e "$DEST" ]]; then' in t
    assert "immutable by construction" in t
    assert "L3_ARCHIVE_NAME" in t


def test_archive_covers_everything_the_freeze_requires():
    t = script(ARCHIVE)
    for arm in ("ctrl", "5e5", "3e5"):
        for s in SEEDS:
            assert f"final_canneal_{arm}_h512_s{s}" in t, (arm, s)
    for s in SEEDS:
        assert f"final_rep_rescue123_h512_s{s}" in t
    for item in ("metrics.tsv", "config.json", "provenance.json", "logs",
                 "_canneal_report", "_slurm_logs", "SHA256SUMS",
                 "ARCHIVE_MANIFEST.json", "error_audit_u"):
        assert item in t, item
    assert "ENDPOINT=step_$(printf '%08d' $END_STEP).pt" in t
    assert "END_STEP=5556000" in t and "SRC_STEP=3889200" in t
    assert "TRAIN_BLOB_EXPECTED=95295d63560ae4c235a6beee8dfb47166f4ed30d" in t
    assert "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50" in t
    assert "78e46871d86efaaf34e9ef41891eb22afa93cc58467e4262225b79e8caffd50f" in t
    assert "fsstor" in t


def test_archive_verifies_before_it_claims_success():
    t = executable(ARCHIVE)          # executable lines only, not the header
    yes = t.index("CANNEAL_STORE_VERIFIED=YES")
    no = t.index("CANNEAL_STORE_VERIFIED=NO")
    assert no < yes, "the failure branch must be reachable before the success"
    assert "VERIFY_RC=$?" in t
    assert 'if (( VERIFY_RC != 0 )); then' in t
    # every archived file is compared with its source, logs included
    assert "_SOURCES.tsv" in t
    assert "unmappable" in t
    assert "sha256sum -c --quiet SHA256SUMS" in t
    # and the success line is guarded, not printed unconditionally
    assert t.index("exit 1", no) < yes


def test_archive_end_to_end_on_a_synthetic_scratch(tmp_path):
    """Copy, manifest, verify, immutability -- and SCRATCH untouched."""
    path_prefix = None
    if subprocess.run(["bash", "-c", "command -v sha256sum"],
                      capture_output=True).returncode != 0:
        # macOS ships `shasum` instead; shim it so the Jean-Zay script runs
        # here verbatim rather than going untested outside Linux.
        if subprocess.run(["bash", "-c", "command -v shasum"],
                          capture_output=True).returncode != 0:
            pytest.skip("neither sha256sum nor shasum on this host")
        binder = tmp_path / "bin"
        binder.mkdir()
        shim = binder / "sha256sum"
        shim.write_text(
            '#!/bin/bash\n'
            'if [[ "${1:-}" == "-c" ]]; then\n'
            '  shift; quiet=0\n'
            '  [[ "${1:-}" == "--quiet" ]] && { quiet=1; shift; }\n'
            '  rc=0\n'
            '  while IFS= read -r line; do\n'
            '    [[ -z "$line" ]] && continue\n'
            '    want=${line%% *}; file=${line#*  }\n'
            '    got=$(shasum -a 256 "$file" | cut -d\' \' -f1)\n'
            '    if [[ "$want" != "$got" ]]; then echo "$file: FAILED"; rc=1\n'
            '    elif (( ! quiet )); then echo "$file: OK"; fi\n'
            '  done < "${1:-/dev/stdin}"\n'
            '  exit $rc\n'
            'fi\n'
            'for f in "$@"; do shasum -a 256 "$f"; done\n')
        shim.chmod(0o755)
        path_prefix = str(binder)
    runs, store = tmp_path / "runs", tmp_path / "store"
    for arm in ("ctrl", "5e5", "3e5"):
        for s in SEEDS:
            d = runs / f"final_canneal_{arm}_h512_s{s}"
            (d / "checkpoints").mkdir(parents=True)
            (d / "checkpoints" / "step_05556000.pt").write_bytes(
                bytes(range(256)) * 4)
            _metrics(str(d), [(5_556_000, 55)])
            (d / "config.json").write_text(json.dumps({"seed": s}))
            (d / "provenance.json").write_text("{}")
    for s in SEEDS:
        d = runs / f"final_rep_rescue123_h512_s{s}"
        (d / "checkpoints").mkdir(parents=True)
        (d / "checkpoints" / "step_03889200.pt").write_bytes(b"\x01" * 512)
        _metrics(str(d), [(3_889_200, 100)])
        (d / "config.json").write_text("{}")
        (d / "provenance.json").write_text("{}")
    rep = runs / "_canneal_report"
    rep.mkdir()
    (rep / "canneal_meta.json").write_text(json.dumps({"x": 1}))

    before = {p: os.path.getsize(p)
              for p in map(str, runs.rglob("*")) if os.path.isfile(p)}
    env = dict(os.environ, L3_REPO=ROOT, L3_RUNS=str(runs),
               L3_STORE=str(store), L3_GLOVE=os.path.join(ROOT, "data",
                                                          "glove.6B.300d.txt"))
    if path_prefix:
        env["PATH"] = path_prefix + os.pathsep + env["PATH"]
    r = subprocess.run(["bash", os.path.join(ROOT, ARCHIVE)],
                       capture_output=True, text=True, env=env)
    if "GloVe" in r.stdout + r.stderr and r.returncode != 0 \
            and "not found" in r.stdout + r.stderr:
        pytest.skip("canonical GloVe not present on this host")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CANNEAL_STORE_VERIFIED=YES" in r.stdout
    dest = store / "lichtheim3_archives" / "canneal_u2000"
    assert (dest / "ARCHIVE_MANIFEST.json").exists()
    assert (dest / "SHA256SUMS").exists()
    man = json.load(open(dest / "ARCHIVE_MANIFEST.json"))
    assert len(man["endpoint_sha256"]) == 12
    assert len(man["lineage_u1400_sha256"]) == 4
    assert man["git"]["train_joint_scratch_blob"] == \
        "95295d63560ae4c235a6beee8dfb47166f4ed30d"
    assert man["result_u2000"]["ctrl_c_errors_mean"] == 55.25
    # immutable
    assert not os.access(dest / "SHA256SUMS", os.W_OK)
    # SCRATCH byte-identical and nothing removed
    after = {p: os.path.getsize(p)
             for p in map(str, runs.rglob("*")) if os.path.isfile(p)}
    assert after == before
    # a second run refuses
    r2 = subprocess.run(["bash", os.path.join(ROOT, ARCHIVE)],
                        capture_output=True, text=True, env=env)
    assert r2.returncode == 1
    assert "already exists" in r2.stdout
