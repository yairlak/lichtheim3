"""V7 FRESH STEP-0 CEILING REPLICATION -- orchestration library (NOT science).

Preregistration: docs/analysis/FRESH_STEP0_CEILING_PREREG_V7.md

This module contains ZERO scientific code.  It derives, from the frozen
retained-recipe ledger, the exact command-line legs that reproduce the
historical from-scratch H512 joint trajectory through the UNCHANGED official
driver `train_joint_scratch.py`, and it implements the non-scientific
plumbing the V7 cohort needs on Jean-Zay:

  * `grid()`            the 5u detector grid (13,890 steps), 780 points + step 0
  * `legs()`            the seven legs of the frozen ledger (LR transitions at
                        u750/u850, schedule re-anchors at u1200/u1400/u2000/u3000)
  * `plan()`            first-launch / requeue / boundary flag derivation
  * `verify_checkpoint` refuse to continue a checkpoint that is not the
                        preregistered recipe at its exact ledger position
  * `dedupe_metrics`    last-row-per-step view of metrics.tsv (a row whose
                        checkpoint save was interrupted is superseded by the
                        re-realised row; the frozen selector then sees exactly
                        one row per detector step)
  * `watch`             online FIRST-HIT freeze: stop the driver as soon as the
                        checkpoint of the first Rcan=Rfree=N=0 detector point is
                        fully written (C never enters the rule)
  * `seed_audit`        seed-token scan used by the preregistration and by the
                        job's remote preflight

Every constant here is COPIED from the executable evidence listed in the
preregistration; the tests assert them against the driver's own arithmetic
and against the archived provenance of the seed-19/20 witness lineage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import sys
import time
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------------------------------------------- horizon --
STEPS_PER_U = 2778            # 463 R batches x 6-step macro-cycle (verified vs driver)
DET_U = 5
DET_STEP = STEPS_PER_U * DET_U            # 13,890
MAX_U = 3900
END_STEP = STEPS_PER_U * MAX_U            # 10,834,200
N_DET = MAX_U // DET_U                    # 780 (excluding step 0)
SAVE_EVERY = 69450                         # unchanged official safety cadence
EVAL_EVERY = DET_STEP                      # dev cadence == detector cadence (as V6)
CEILING_REQUIRED = 2                       # SETTLE Amendment-2 / V6 training-ceiling semantics

# ------------------------------------------------------------------ recipe --
WM, ENC, DEC = 128, 512, 512
LR_R = LR_N = 3e-5
LR_C_MATURE = 1e-4
LR_BOUNDARY_REP_BATCHES = 46_300
TWO_STAGE = {"kind": "two_stage_rep_cursor", "stage1": 1e-3, "stage2": 1e-4,
             "boundary_rep_batches": LR_BOUNDARY_REP_BATCHES}
TASK_3E5 = {"kind": "task_specific", "repetition": 3e-5, "naming": 3e-5,
            "comprehension": 3e-5}
TASK_CHIGH = {"kind": "task_specific", "repetition": 3e-5, "naming": 3e-5,
              "comprehension": 1e-4}

EXPECTED = {
    "widths": {"wm_hidden": WM, "ltm_enc_hidden": ENC, "ltm_dec_hidden": DEC},
    "schedule": "interleaved_123",
    "schedule_ratio": [1, 2, 3],
    "optimizer_policy": "shared_adamw",
    "dec_weight": 0.5,
    "c_align_weight": 0.0,
    "subset_mode": "final_full",
    "comprehension_population_n": 27_981,
    "comprehension_population_sha256":
        "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
    "naming_population_n": 29_571,
    "naming_population_sha256":
        "78e46871d86efaaf34e9ef41891eb22afa93cc58467e4262225b79e8caffd50f",
    "lexicon_file_sha256":
        "ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66",
    "glove_sha256":
        "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed",
    "glove_fallback": 0,
    "format": "lichtheim3.joint_scratch.v1",
}
STREAM_SEED_STRIDE = 1_000_003
STREAM_SEED_OFFSET = {"repetition": 0, "pool": 1, "comprehension": 2, "naming": 3}
SCHEDULE_SEED_OFFSET = 4

# ---------------------------------------------------------------- ledger ----
# The COMPLETE retained trajectory of the seed-19/20 mature witnesses, read
# from checkpoint-level `phase_transitions` (see preregistration section 7).
#   u750  step 2,083,500  two_stage -> task_specific 3e-5/3e-5/3e-5   (moments unchanged)
#   u850  step 2,361,300  C 3e-5 -> 1e-4                                (moments unchanged)
#   u1200 step 3,333,600  schedule re-anchor 0         -> 3,333,600    (LR unchanged)
#   u1400 step 3,889,200  schedule re-anchor 3,333,600 -> 3,889,200
#   u2000 step 5,556,000  schedule re-anchor 3,889,200 -> 5,556,000
#   u3000 step 8,334,000  schedule re-anchor 5,556,000 -> 8,334,000
#   u3600 (no event: the V6 seed-19/22 continuation re-anchored nothing)
LEDGER = [
    # name        start_u  end_u  lr_policy    reanchor
    ("L1_two_stage",    0,   750,  TWO_STAGE,   False),
    ("L2_all3e5",     750,   850,  TASK_3E5,    False),
    ("L3_chigh",      850,  1200,  TASK_CHIGH,  False),
    ("L4_reanchor",  1200,  1400,  TASK_CHIGH,  True),
    ("L5_reanchor",  1400,  2000,  TASK_CHIGH,  True),
    ("L6_reanchor",  2000,  3000,  TASK_CHIGH,  True),
    ("L7_reanchor",  3000,  3900,  TASK_CHIGH,  True),
]

# ---------------------------------------------------------------- cohort ----
PRIORITY_SEEDS = {"p1": 31, "p2": 32, "p3": 33, "p4": 34}     # frozen (prereg s.2)
RUN_ID_FMT = "fresh_ceiling_v7_{slot}_s{seed}"
SEED_TOKEN_RE = r"(?<![0-9A-Za-z])(?:s|seed[_ =:-]?|seed)({seed})(?![0-9])"


def run_id(slot: str, seed: int) -> str:
    if slot not in PRIORITY_SEEDS or PRIORITY_SEEDS[slot] != int(seed):
        raise ValueError(f"slot {slot!r} does not carry seed {seed}")
    return RUN_ID_FMT.format(slot=slot, seed=int(seed))


# ----------------------------------------------------------------- helpers --
def steps(u: float) -> int:
    s = u * STEPS_PER_U
    if int(s) != s:
        raise ValueError(f"u={u} is not an integer number of steps")
    return int(s)


def grid(ledger=None) -> List[int]:
    """Every full-lexicon detector point: 0 (start evaluation) and each 5u."""
    end = END_STEP if ledger is None else ledger[-1]["end"]
    step = DET_STEP if ledger is None else ledger[0].get("det_step", DET_STEP)
    return list(range(0, end + 1, step))


def legs(ledger=None) -> List[dict]:
    if ledger is not None:
        return [dict(x) for x in ledger]
    out = []
    for name, u0, u1, lr, re_anchor in LEDGER:
        out.append({"name": name, "start": steps(u0), "end": steps(u1),
                    "start_u": u0, "end_u": u1, "lr_policy": dict(lr),
                    "reanchor": bool(re_anchor)})
    return out


def leg_for_step(step: int, ledger=None) -> dict:
    """The leg whose half-open interval [start, end) contains `step`; a step
    exactly on a boundary belongs to the leg that STARTS there (the boundary
    launch is where its declaration is made)."""
    for leg in legs(ledger):
        if leg["start"] <= step < leg["end"]:
            return leg
    raise ValueError(f"step {step} is outside every leg")


def expected_anchor_at(step: int, ledger=None) -> int:
    """schedule_anchor_step a checkpoint AT `step` must carry, i.e. the last
    re-anchor strictly BEFORE `step` (a boundary checkpoint still carries the
    PREVIOUS anchor: the new one is declared when that checkpoint is resumed)."""
    anchor = 0
    for leg in legs(ledger):
        if leg["reanchor"] and leg["start"] < step:
            anchor = leg["start"]
    return anchor


def expected_lr_policy_at(step: int, ledger=None) -> dict:
    """lr_policy a checkpoint AT `step` must carry (same boundary convention)."""
    pol = legs(ledger)[0]["lr_policy"]
    for leg in legs(ledger):
        if leg["start"] < step:
            pol = leg["lr_policy"]
    return dict(pol)


def expected_transitions_before(step: int, ledger=None) -> List[dict]:
    """The ledger entries a checkpoint AT `step` must already carry, in order."""
    out = []
    prev_lr, prev_anchor = legs(ledger)[0]["lr_policy"], 0
    for leg in legs(ledger)[1:]:
        if leg["start"] >= step:
            break
        changed = []
        if leg["lr_policy"] != prev_lr:
            changed.append("lr_policy")
        if leg["reanchor"]:
            changed.append("schedule_anchor")
        out.append({"transition_step": leg["start"], "changed": changed,
                    "old_lr_policy": dict(prev_lr), "new_lr_policy": dict(leg["lr_policy"]),
                    "old_schedule_anchor_step": prev_anchor,
                    "schedule_anchor_step": (leg["start"] if leg["reanchor"] else prev_anchor),
                    "moment_initialization": "unchanged"})
        prev_lr = leg["lr_policy"]
        if leg["reanchor"]:
            prev_anchor = leg["start"]
    return out


def lr_flags(policy: dict) -> List[str]:
    if policy["kind"] == "two_stage_rep_cursor":
        return []
    return ["--lr-repetition", repr(float(policy["repetition"])),
            "--lr-naming", repr(float(policy["naming"])),
            "--lr-comprehension", repr(float(policy["comprehension"]))]


def plan(latest_step: Optional[int], ledger=None) -> dict:
    """Derive the next launch from the run's OWN latest checkpoint step.

    latest_step None  -> first launch from scratch (leg 1), start evaluation
    latest_step == a leg start (and not END) -> boundary launch: resume that
        checkpoint and DECLARE the leg's transition (LR change and/or
        --reanchor-schedule), exactly once, via --phase-transition
    start < latest_step < end -> plain requeue inside the leg: same LR flags,
        no declaration (the driver would refuse a re-declaration anyway when
        nothing changed, and --reanchor-schedule would WRONGLY re-anchor)
    latest_step >= END -> training complete
    """
    L = legs(ledger)
    end = L[-1]["end"]
    if latest_step is None:
        leg = L[0]
        return {"status": "FIRST_LAUNCH", "leg": leg["name"], "resume": None,
                "max_steps": leg["end"], "flags": lr_flags(leg["lr_policy"]) + ["--eval-at-start"],
                "declare": False, "reanchor": False}
    s = int(latest_step)
    if s >= end:
        return {"status": "COMPLETE", "leg": None, "resume": s, "max_steps": end,
                "flags": [], "declare": False, "reanchor": False}
    leg = leg_for_step(s, ledger)
    at_boundary = (s == leg["start"]) and leg is not L[0]
    flags = lr_flags(leg["lr_policy"])
    declare = False
    if at_boundary:
        prev = L[L.index(leg) - 1]
        if leg["reanchor"]:
            flags = flags + ["--reanchor-schedule"]
        if leg["reanchor"] or leg["lr_policy"] != prev["lr_policy"]:
            flags = flags + ["--phase-transition"]
            declare = True
    return {"status": "BOUNDARY_LAUNCH" if at_boundary else "REQUEUE",
            "leg": leg["name"], "resume": s, "max_steps": leg["end"], "flags": flags,
            "declare": declare, "reanchor": bool(at_boundary and leg["reanchor"])}


# ------------------------------------------------------- checkpoint gate ----
def _need(cond: bool, msg: str, bad: List[str]) -> None:
    if not cond:
        bad.append(msg)


def verify_checkpoint(ck: dict, seed: int, expect_step: Optional[int] = None,
                      ledger=None, tiny: bool = False) -> List[str]:
    """Every reason this checkpoint must NOT be continued as a V7 run.  Empty
    list == PASS.  Pure function of the loaded dict (no torch needed).

    `tiny=True` is for the local engineering smoke on a 400-word fixture: it
    skips the population / hash / GloVe / detector-grid checks (which encode
    the full lexicon) but keeps every ledger, seed, width, schedule and
    optimizer check.  Production never passes it."""
    bad: List[str] = []
    step = int(ck.get("global_step", -1))
    boundary = int(legs(ledger)[0]["lr_policy"].get("boundary_rep_batches", LR_BOUNDARY_REP_BATCHES))
    _need(ck.get("format") == EXPECTED["format"], f"format {ck.get('format')!r}", bad)
    _need(int(ck.get("seed", -1)) == int(seed), f"seed {ck.get('seed')} != {seed}", bad)
    if expect_step is not None:
        _need(step == int(expect_step), f"global_step {step} != {expect_step}", bad)
    _need(step % 6 == 0, "not on a macro-cycle boundary", bad)
    _need(tiny or step % DET_STEP == 0, "not on the 5u detector grid", bad)
    _need(dict(ck.get("widths") or {}) == EXPECTED["widths"], f"widths {ck.get('widths')}", bad)
    _need(ck.get("schedule") == EXPECTED["schedule"], f"schedule {ck.get('schedule')!r}", bad)
    _need(list(ck.get("schedule_ratio") or []) == EXPECTED["schedule_ratio"], "ratio", bad)
    _need(int(ck.get("schedule_seed", -1)) == seed * STREAM_SEED_STRIDE + SCHEDULE_SEED_OFFSET,
          "schedule_seed", bad)
    _need(dict(ck.get("stream_seeds") or {}) ==
          {n: seed * STREAM_SEED_STRIDE + o for n, o in STREAM_SEED_OFFSET.items()},
          "stream_seeds", bad)
    _need(ck.get("optimizer_policy") == EXPECTED["optimizer_policy"], "optimizer_policy", bad)
    _need("optimizer_state_dict" in ck and "optimizer_states" not in ck,
          "not exactly one shared AdamW state", bad)
    _need(float(ck.get("dec_weight", 0.5)) == EXPECTED["dec_weight"], "dec_weight", bad)
    _need(float(ck.get("c_align_weight") or 0.0) == EXPECTED["c_align_weight"], "c_align", bad)
    _need(ck.get("subset_mode") == EXPECTED["subset_mode"], "subset_mode", bad)
    if not tiny:
        _need(int(ck.get("comprehension_population_n", 0)) == EXPECTED["comprehension_population_n"],
              "C population n", bad)
        _need(ck.get("comprehension_population_sha256") == EXPECTED["comprehension_population_sha256"],
              "C population hash", bad)
        _need(int(ck.get("naming_population_n", 0)) == EXPECTED["naming_population_n"], "N population n", bad)
        _need(ck.get("naming_population_sha256") == EXPECTED["naming_population_sha256"], "N population hash", bad)
        _need(ck.get("lexicon_file_sha256") == EXPECTED["lexicon_file_sha256"], "lexicon hash", bad)
        _need(int(ck.get("glove_fallback", 1)) == 0, "glove_fallback != 0", bad)
    _need(int(ck.get("lr_boundary_steps", -1)) == boundary, "lr_boundary_steps", bad)
    # ledger position
    _need(dict(ck.get("lr_policy") or {}) == expected_lr_policy_at(step, ledger),
          f"lr_policy {ck.get('lr_policy')} != ledger {expected_lr_policy_at(step, ledger)}", bad)
    _need(int(ck.get("schedule_anchor_step", -1)) == expected_anchor_at(step, ledger),
          f"anchor {ck.get('schedule_anchor_step')} != ledger {expected_anchor_at(step, ledger)}", bad)
    want = expected_transitions_before(step, ledger)
    got = list(ck.get("phase_transitions") or [])
    _need(len(got) == len(want), f"{len(got)} transitions recorded, ledger expects {len(want)}", bad)
    for g, w in zip(got, want):
        _need(int(g.get("transition_step", -1)) == w["transition_step"]
              and list(g.get("changed") or []) == w["changed"]
              and dict(g.get("new_lr_policy") or {}) == w["new_lr_policy"]
              and int(g.get("schedule_anchor_step", -1)) == w["schedule_anchor_step"]
              and g.get("moment_initialization") == "unchanged",
              f"transition at {g.get('transition_step')} differs from the ledger", bad)
    # cursors: on a cycle boundary the 1:2:3 ledger is exact
    c = ck.get("cursors") or {}
    if step % 6 == 0 and c:
        r = step // 6
        _need(int(c.get("repetition", -1)) == r, "R cursor", bad)
        _need(int(c.get("naming", -1)) == 2 * r, "N cursor", bad)
        _need(int(c.get("comprehension", -1)) == 3 * r, "C cursor", bad)
        _need(int(c.get("pool", -1)) == r, "pool cursor", bad)
    return bad


# ------------------------------------------------------------ metrics ------
N_POP = 29_571


def read_rows(metrics_tsv: str) -> List[dict]:
    with open(metrics_tsv, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _f(row: dict, key: str):
    v = row.get(key)
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def is_detector_row(row: dict) -> bool:
    return _f(row, "full_rep_errors") is not None


def eligible(row: dict) -> bool:
    """FROZEN V6 rule, restated (the frozen first_hit_selector.py is the
    authority; this must agree with it and the tests assert that it does)."""
    rcan, rfree, nam = _f(row, "full_rep_errors"), _f(row, "full_rep_freear_errors"), _f(row, "full_naming_exact")
    if rcan is None or rfree is None or nam is None:
        return False
    n_err = int(round((1.0 - nam) * N_POP))
    return int(round(rcan)) == 0 and int(round(rfree)) == 0 and n_err == 0


def dedupe_metrics(src: str, dst: str) -> dict:
    """Write `dst` = `src` with, for every DETECTOR step, only its LAST row.

    Why last: the driver appends the detector row BEFORE saving that step's
    checkpoint.  If the allocation dies in between, the row is an orphan of a
    discarded segment; the requeue re-realises the segment and appends a new
    row for the same step, followed by the surviving checkpoint.  The last row
    is therefore always the one that describes the checkpoint on disk.  Rows
    are never edited and `src` is never touched."""
    rows = read_rows(src)
    with open(src, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    last_index: Dict[int, int] = {}
    for i, r in enumerate(rows):
        if is_detector_row(r):
            last_index[int(float(r["step"]))] = i
    keep = [i for i, r in enumerate(rows)
            if not is_detector_row(r) or last_index[int(float(r["step"]))] == i]
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for i in keep:
            w.writerow(rows[i])
    dropped = [int(float(rows[i]["step"])) for i in range(len(rows)) if i not in set(keep)]
    return {"rows_in": len(rows), "rows_out": len(keep), "orphan_detector_rows_dropped": dropped,
            "n_detector_steps": len(last_index)}


def first_hit(rows: Sequence[dict]) -> Optional[dict]:
    pts = sorted((r for r in rows if is_detector_row(r)), key=lambda r: int(float(r["step"])))
    for r in pts:
        if eligible(r):
            return r
    return None


# --------------------------------------------------------------- watcher ---
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_complete(path: str, step: int, settle_s: float = 20.0) -> bool:
    """A checkpoint counts as WRITTEN when its size has been stable for
    `settle_s` seconds and torch can load it with the expected global_step."""
    if not os.path.exists(path):
        return False
    s1 = os.path.getsize(path)
    time.sleep(settle_s)
    if not os.path.exists(path) or os.path.getsize(path) != s1 or s1 == 0:
        return False
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck["global_step"]) == int(step)
    except Exception:
        return False


def watch(run_dir: str, pid: int, marker: str, poll_s: float = 30.0,
          seed: Optional[int] = None) -> int:
    """Poll metrics.tsv; on the FIRST eligible detector row whose checkpoint is
    completely written, write `marker` and SIGTERM `pid` (the srun step).  The
    steps the driver takes after the checkpoint save are discarded, never
    evaluated, and never selectable: the frozen selector sees only detector
    rows, and the marker pins the step.  Returns 0 on freeze, 1 if the driver
    exited first, without a hit."""
    metrics = os.path.join(run_dir, "metrics.tsv")
    ck_dir = os.path.join(run_dir, "checkpoints")
    while True:
        if os.path.exists(marker):
            return 0
        hit = first_hit(read_rows(metrics)) if os.path.exists(metrics) else None
        if hit is not None:
            step = int(float(hit["step"]))
            ck = os.path.join(ck_dir, f"step_{step:08d}.pt")
            if checkpoint_complete(ck, step):
                rec = {"seed": seed, "run_id": os.path.basename(run_dir.rstrip("/")),
                       "first_hit_step": step, "first_hit_u": step / STEPS_PER_U,
                       "checkpoint": ck, "checkpoint_sha256": sha256_file(ck),
                       "detector_row": {k: hit.get(k) for k in (
                           "step", "full_rep_errors", "full_rep_freear_errors",
                           "full_naming_exact", "full_comp_errors", "full_rep_ltm", "gate_mean")},
                       "rule": "FIRST detector point in chronological order with "
                               "Rcan=0 AND Rfree=0 AND Naming=0; C excluded",
                       "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "driver_pid_signalled": pid}
                tmp = marker + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(rec, f, indent=1)
                os.replace(tmp, marker)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                print(f"[v7-watch] FIRST-HIT frozen at step {step} (u{step / STEPS_PER_U:.0f}); "
                      f"sha256 {rec['checkpoint_sha256'][:12]}...; driver stopped", flush=True)
                return 0
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return 1
        time.sleep(poll_s)


# ------------------------------------------------------------ seed audit ---
def seed_audit(seeds: Sequence[int], paths: Sequence[str], texts: Sequence[str] = ()) -> dict:
    """Scan directory names under `paths` and the given text blobs for seed
    tokens (s31, seed31, seed_31, seed=31, 'seed': 31).  Returns per-seed hits."""
    hits: Dict[int, List[str]] = {int(s): [] for s in seeds}
    pats = {int(s): re.compile(SEED_TOKEN_RE.format(seed=int(s))) for s in seeds}
    for root in paths:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for name in dirnames + filenames:
                for s, p in pats.items():
                    if p.search(name):
                        hits[s].append(os.path.join(dirpath, name))
            if dirpath.count(os.sep) - root.count(os.sep) >= 3:
                dirnames[:] = []
    for label_text in texts:
        for s, p in pats.items():
            for m in p.finditer(label_text):
                hits[s].append(f"text:{label_text[max(0, m.start()-40):m.end()+40]!r}")
    return {"seeds": list(map(int, seeds)), "hits": hits,
            "clean": [s for s in map(int, seeds) if not hits[s]]}


# ------------------------------------------------------------------ CLI -----
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("grid"); p.add_argument("--no-zero", action="store_true")
    p = sub.add_parser("plan"); p.add_argument("--latest-step", type=int, default=None)
    p = sub.add_parser("verify-ckpt"); p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--expect-step", type=int, default=None)
    p = sub.add_parser("dedupe"); p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p = sub.add_parser("watch"); p.add_argument("--run-dir", required=True); p.add_argument("--pid", type=int, required=True)
    p.add_argument("--marker", required=True); p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=None)
    p = sub.add_parser("seed-audit"); p.add_argument("--seeds", required=True)
    p.add_argument("--paths", nargs="*", default=[]); p.add_argument("--text-files", nargs="*", default=[])
    p.add_argument("--out-json", default=None)
    p = sub.add_parser("ledger")
    a = ap.parse_args(argv)

    if a.cmd == "grid":
        g = grid()
        print(",".join(str(s) for s in (g[1:] if a.no_zero else g)))
        return 0
    if a.cmd == "plan":
        d = plan(a.latest_step)
        print(f"V7_STATUS={d['status']}")
        print(f"V7_LEG={d['leg']}")
        print(f"V7_RESUME_STEP={'' if d['resume'] is None else d['resume']}")
        print(f"V7_MAX_STEPS={d['max_steps']}")
        print(f"V7_FLAGS={' '.join(d['flags'])}")
        print(f"V7_DECLARE={int(d['declare'])}")
        print(f"V7_REANCHOR={int(d['reanchor'])}")
        return 0
    if a.cmd == "verify-ckpt":
        import torch
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        bad = verify_checkpoint(ck, a.seed, a.expect_step)
        for b in bad:
            print(f"FATAL verify-ckpt: {b}", file=sys.stderr)
        if not bad:
            print(f"[v7] checkpoint OK step={ck['global_step']} seed={ck['seed']} "
                  f"anchor={ck.get('schedule_anchor_step')} lr={ck['lr_policy']} "
                  f"transitions={len(ck.get('phase_transitions') or [])}")
        return 0 if not bad else 1
    if a.cmd == "dedupe":
        print(json.dumps(dedupe_metrics(a.src, a.dst)))
        return 0
    if a.cmd == "watch":
        return watch(a.run_dir, a.pid, a.marker, a.poll, a.seed)
    if a.cmd == "seed-audit":
        seeds = [int(s) for s in a.seeds.split(",")]
        texts = [open(t, errors="replace").read() for t in a.text_files if os.path.exists(t)]
        res = seed_audit(seeds, a.paths, texts)
        if a.out_json:
            json.dump(res, open(a.out_json, "w"), indent=1)
        print(json.dumps({"clean": res["clean"], "n_hits": {s: len(h) for s, h in res["hits"].items()}}))
        return 0 if set(res["clean"]) == set(seeds) else 2
    if a.cmd == "ledger":
        for leg in legs():
            print(f"{leg['name']:14s} u{leg['start_u']:>5}->u{leg['end_u']:<5} "
                  f"steps {leg['start']:>10,} -> {leg['end']:<10,} lr={leg['lr_policy']} "
                  f"reanchor={leg['reanchor']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
