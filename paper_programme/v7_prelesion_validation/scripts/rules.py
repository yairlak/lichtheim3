"""FROZEN decision rules for the V7 pre-lesion validation.

Deliberately torch-free and dependency-free so that REPORTING can import the
rules without being able to run a model. Both `prelesion_eval` (execution) and
`generate_reports` (reporting) import from here; neither redefines a rule.

Contract sections 6-8; frozen at design commit 2d240e1f.
"""
from __future__ import annotations

from typing import Sequence


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def first_divergence(pred: Sequence[int], tgt: Sequence[int]):
    for i in range(min(len(pred), len(tgt))):
        if pred[i] != tgt[i]:
            return i
    return None if len(pred) == len(tgt) else min(len(pred), len(tgt))


def summarize(values: Sequence[float]) -> dict:
    """n / mean / sample SD / min / p01..p99 / max, in float64."""
    import statistics
    v = sorted(float(x) for x in values)
    if not v:
        return {"n": 0}
    def pct(p: float) -> float:
        if len(v) == 1:
            return v[0]
        k = p * (len(v) - 1)
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (k - lo) * (v[hi] - v[lo])
    return {"n": len(v), "mean": sum(v) / len(v),
            "sd": statistics.stdev(v) if len(v) > 1 else 0.0,
            "min": v[0], "p01": pct(.01), "p05": pct(.05), "p25": pct(.25),
            "p50": pct(.50), "p75": pct(.75), "p95": pct(.95), "p99": pct(.99),
            "max": v[-1]}


def classify_ordering(acc_wm: float, acc_ltm: float,
                      ned_wm: float, ned_ltm: float) -> dict:
    """Frozen rule.  No post-hoc tolerance; float64 throughout."""
    d_acc = float(acc_wm) - float(acc_ltm)
    d_ned = float(ned_ltm) - float(ned_wm)
    if d_acc == 0.0 and d_ned == 0.0:
        label = "TIE"
    elif d_acc >= 0.0 and d_ned >= 0.0:
        label = "WM_DOMINANT"
    elif d_acc <= 0.0 and d_ned <= 0.0:
        label = "LTM_DOMINANT"
    else:
        label = "MIXED"
    return {"delta_acc": d_acc, "delta_ned": d_ned, "ordering": label}


def preservation(source_label: str, post_label: str) -> str:
    if source_label == "WM_DOMINANT" and post_label == "WM_DOMINANT":
        return "PRESERVED"
    if source_label != "WM_DOMINANT" and post_label == "WM_DOMINANT":
        return "PRESERVED_FROM_NONDOMINANT_SOURCE"
    if source_label == "WM_DOMINANT" and post_label != "WM_DOMINANT":
        return "NOT_PRESERVED"
    return "NO_EXPECTED_PATTERN_AT_SOURCE"
