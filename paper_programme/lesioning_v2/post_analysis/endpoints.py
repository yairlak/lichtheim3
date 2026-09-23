"""Endpoint identity for post-analysis.

Endpoint identity is recovered from `battery.ALL_ENDPOINTS` keyed by the full
scientific triple

    (task, route, decoding_convention)

and never inferred from task/decoder alone. That matters because three
repetition endpoints share one (task, decoding_convention) pair and differ only
by route:

    rep_canonical_full / rep_canonical_wm / rep_canonical_ltm
    rep_freear_full    / rep_freear_wm    / rep_freear_ltm

The frozen execution aggregator's SUMMARY_KEY omits `route` and would merge
each of those triples into a single row, mixing PRIMARY with DIAGNOSTIC
readouts. Post-analysis therefore groups on the triple. This is an
analysis-integration correction: no endpoint, metric or result changes, and
`execution/aggregate.py` is neither modified nor used for repetition.
"""
from __future__ import annotations

from typing import Dict, Tuple

from paper_programme.lesioning_v2.lesion_operator import battery

#: The post-analysis grouping key: the frozen SUMMARY_KEY plus `route`.
POST_ANALYSIS_KEY = ("state_id", "site", "severity_k", "realization",
                     "task", "route", "decoding_convention")

#: The frozen execution key, recorded so the difference is explicit.
FROZEN_EXECUTION_KEY = ("state_id", "site", "severity_k", "realization",
                        "task", "decoding_convention")

EXPECTED_ENDPOINTS_PER_CELL = 8
PRIMARY_KEYS = tuple(e.key for e in battery.PRIMARY)
DIAGNOSTIC_KEYS = tuple(e.key for e in battery.DIAGNOSTIC)


class EndpointError(RuntimeError):
    pass


def _index() -> Dict[Tuple[str, str, str], "battery.Endpoint"]:
    out: Dict[Tuple[str, str, str], battery.Endpoint] = {}
    for e in battery.ALL_ENDPOINTS:
        triple = (e.task, e.route, e.decoding_convention)
        if triple in out:
            raise EndpointError(f"frozen battery has duplicate triple {triple}")
        out[triple] = e
    return out


ENDPOINT_BY_TRIPLE = _index()


def resolve(task: str, route: str, decoding_convention: str):
    """The frozen Endpoint for one item row's scientific triple."""
    triple = (task, route, decoding_convention)
    e = ENDPOINT_BY_TRIPLE.get(triple)
    if e is None:
        raise EndpointError(
            f"unknown (task, route, decoding_convention) tuple {triple}; "
            "post-analysis refuses to invent an endpoint identity")
    return e


def route_is_load_bearing() -> bool:
    """True iff some (task, decoding_convention) pair maps to >1 endpoint.

    When true, the frozen execution SUMMARY_KEY cannot separate endpoints and
    post-analysis must group on the triple instead.
    """
    seen: Dict[Tuple[str, str], int] = {}
    for e in battery.ALL_ENDPOINTS:
        k = (e.task, e.decoding_convention)
        seen[k] = seen.get(k, 0) + 1
    return any(v > 1 for v in seen.values())


def collapsed_by_frozen_key() -> Dict[Tuple[str, str], Tuple[str, ...]]:
    """Which endpoint keys the frozen execution key would merge together."""
    groups: Dict[Tuple[str, str], list] = {}
    for e in battery.ALL_ENDPOINTS:
        groups.setdefault((e.task, e.decoding_convention), []).append(e.key)
    return {k: tuple(sorted(v)) for k, v in groups.items() if len(v) > 1}
