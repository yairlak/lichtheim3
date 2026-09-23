"""Descriptive curve, baseline and k=15 tables. No inference of any kind.

Dispersion convention, stated once and applied everywhere:

    k>0   mean = arithmetic mean over that cell's realizations
          sd   = SAMPLE standard deviation, ddof=1 (undefined, hence NA, if
                 only one realization exists)
          min/max = over realizations

    k=0   the single preserved intact control. n_realizations = 1,
          mean = min = max = the observed value, and sd is NA -- never 0.
          A zero would read as "measured and found to have no spread", which
          is a different and false claim.

No p-values, confidence intervals, hypothesis tests, fitted thresholds,
selected severities or AUC are computed here.
"""
from __future__ import annotations

import statistics
from typing import Dict, List, Sequence

from paper_programme.lesioning_v2.post_analysis import io_utils

CURVE_COLUMNS = (
    "state_id", "site", "severity_k", "severity_s", "endpoint", "tier",
    "route", "task", "decoding_convention", "n_realizations",
    "mean_exact_match", "sd_exact_match", "min_exact_match", "max_exact_match",
    "n_items",
)
BASELINE_COLUMNS = (
    "state_id", "site", "endpoint", "tier", "route", "task",
    "decoding_convention", "n_items", "correct", "exact_match",
)
K15_COLUMNS = (
    "state_id", "site", "endpoint", "tier", "route", "task",
    "decoding_convention", "n_realizations", "realization_values",
    "mean_exact_match", "sd_exact_match", "min_exact_match", "max_exact_match",
)
SD_CONVENTION = "sample standard deviation, ddof=1; NA when n_realizations < 2"
MAX_SEVERITY_K = 15


def _group(rows: Sequence[Dict], keys: Sequence[str]) -> Dict[tuple, List[Dict]]:
    out: Dict[tuple, List[Dict]] = {}
    for r in rows:
        out.setdefault(tuple(str(r[k]) for k in keys), []).append(r)
    return out


def curve_summary(cell_rows: Sequence[Dict]) -> List[Dict]:
    keys = ("state_id", "site", "severity_k", "endpoint")
    out: List[Dict] = []
    for _, g in _group(cell_rows, keys).items():
        vals = [io_utils.fnum(r["exact_match"]) for r in g]
        vals = [v for v in vals if v is not None]
        k = int(g[0]["severity_k"])
        n = len(vals)
        sd = (statistics.stdev(vals) if n >= 2 else None)
        items = {int(r["n"]) for r in g}
        out.append({
            "state_id": g[0]["state_id"], "site": g[0]["site"],
            "severity_k": k, "severity_s": g[0]["severity_s"],
            "endpoint": g[0]["endpoint"], "tier": g[0]["tier"],
            "route": g[0]["route"], "task": g[0]["task"],
            "decoding_convention": g[0]["decoding_convention"],
            "n_realizations": n,
            "mean_exact_match": (sum(vals) / n) if n else None,
            "sd_exact_match": sd,
            "min_exact_match": min(vals) if vals else None,
            "max_exact_match": max(vals) if vals else None,
            "n_items": (items.pop() if len(items) == 1 else
                        "|".join(map(str, sorted(items)))),
        })
    out.sort(key=lambda r: (r["state_id"], r["site"], r["severity_k"],
                            r["endpoint"]))
    return out


def intact_baselines(cell_rows: Sequence[Dict]) -> List[Dict]:
    """All 12 state x site k=0 records, verbatim. Never collapsed or averaged."""
    out = [{
        "state_id": r["state_id"], "site": r["site"], "endpoint": r["endpoint"],
        "tier": r["tier"], "route": r["route"], "task": r["task"],
        "decoding_convention": r["decoding_convention"],
        "n_items": r["n"], "correct": r["correct"],
        "exact_match": io_utils.fnum(r["exact_match"]),
    } for r in cell_rows if int(r["severity_k"]) == 0]
    out.sort(key=lambda r: (r["state_id"], r["site"], r["endpoint"]))
    return out


def k0_cross_site_consistency(baselines: Sequence[Dict]) -> List[Dict]:
    """At k=0 the mask is all-ones and the perturbation null, so for one state
    the result should be identical at L1/L2/L3. Reported as a check; the
    site-specific records are kept verbatim regardless of the outcome."""
    out = []
    for key, g in _group(baselines, ("state_id", "endpoint")).items():
        vals = sorted({r["exact_match"] for r in g})
        out.append({
            "state_id": g[0]["state_id"], "endpoint": g[0]["endpoint"],
            "n_sites": len(g), "distinct_values": len(vals),
            "identical_across_sites": len(vals) == 1,
            "values": "|".join(f"{v:.10g}" for v in vals),
        })
    out.sort(key=lambda r: (r["state_id"], r["endpoint"]))
    return out


def max_severity_table(cell_rows: Sequence[Dict]) -> List[Dict]:
    """k=15 convenience table. One pre-defined point on the full curve; NOT a
    success criterion and never a substitute for the k=0..15 report."""
    sel = [r for r in cell_rows if int(r["severity_k"]) == MAX_SEVERITY_K]
    out: List[Dict] = []
    for _, g in _group(sel, ("state_id", "site", "endpoint")).items():
        g = sorted(g, key=lambda r: int(r["realization"]))
        vals = [io_utils.fnum(r["exact_match"]) for r in g]
        n = len(vals)
        out.append({
            "state_id": g[0]["state_id"], "site": g[0]["site"],
            "endpoint": g[0]["endpoint"], "tier": g[0]["tier"],
            "route": g[0]["route"], "task": g[0]["task"],
            "decoding_convention": g[0]["decoding_convention"],
            "n_realizations": n,
            "realization_values": "|".join(f"{v:.10g}" for v in vals),
            "mean_exact_match": sum(vals) / n if n else None,
            "sd_exact_match": statistics.stdev(vals) if n >= 2 else None,
            "min_exact_match": min(vals) if vals else None,
            "max_exact_match": max(vals) if vals else None,
        })
    out.sort(key=lambda r: (r["state_id"], r["site"], r["endpoint"]))
    return out
