"""Gate / route-competence diagnostics (POST_STAGE / PAPER_PROGRAMME).

Read-only instrumentation for Experiments 1 and 2 of the
LICHTHEIM3 — GATING / ROUTE DIAGNOSTICS workstream.

Nothing in this package trains, tunes, or writes a parameter.  `models/gating.py`
is never modified: the forced-gate condition is applied either as a post-hoc
recombination of the logits `forward()` already returns (`fixed_mix_logits`), or
through a forward hook that is removed on exit (`forced_gate`).  The two paths
are independent and are cross-checked against each other in the test suite.
"""
from .gate_probe import (
    ROUTES, HISTORICAL_FREE_AR_MAX_STEPS, CompetenceCategory, forced_gate,
    fixed_mix_logits, capture_gate_field, ar_decode_forced_length,
    ar_decode_free, competence_category, collect_item_level,
)

__all__ = [
    "ROUTES", "HISTORICAL_FREE_AR_MAX_STEPS", "CompetenceCategory",
    "forced_gate", "fixed_mix_logits", "capture_gate_field",
    "ar_decode_forced_length", "ar_decode_free", "competence_category",
    "collect_item_level",
]
