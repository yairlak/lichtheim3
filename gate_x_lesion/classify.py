"""The frozen end-to-end classification pipeline. CENTRAL final decision, §§1-2.

Order is fixed and may not be collapsed:

    1. CANONICAL classified alone      -> CANONICAL_OUTCOME
    2. FREE_AR   classified alone      -> FREE_AR_OUTCOME
    3. JOINT only afterward            -> JOINT_OUTCOME

Inputs are already-computed, per-route-family quantities:
  * the SHARED diagnostic-validity label per severity (O-3, `validity.py`) — the
    SAME label is used for both conventions, by construction;
  * the frozen robustness verdict per severity x convention (O-4, `robustness.py`).

Direction -> letter mapping, stated once so it cannot drift:

    Delta = accuracy_native - accuracy_fixed05
    ROBUST_POSITIVE  (NATIVE better)   -> A
    ROBUST_NEGATIVE  (FIXED05 better)  -> B
    no valid severity robust           -> D
    valid severities robust in BOTH directions -> F_HETEROGENEOUS  (the O-4 veto)

THE O-4 VETO
------------
For a fixed route family and decoding convention: if one diagnostically valid
severity supports a robust direction and another diagnostically valid severity
satisfies the FULL robust rule in the opposite direction, then A/B/D may NOT be
declared — the classification is `OUTCOME_F_HETEROGENEOUS`.

Only a severity that is (a) diagnostically valid AND (b) fully robust can
contribute a direction. A severity that merely *looks* opposite-signed without
meeting the full robust rule does not trigger the veto, and a diagnostically
INVALID severity cannot trigger it at all.

All diagnostically valid severities remain visible in the output. No severity is
ever retrospectively promoted as "the lesion result".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from gate_x_lesion.outcomes import (JOINT_MIXED_DECODING, OUTCOME_A, OUTCOME_B,
                                    OUTCOME_D, OUTCOME_F, OUTCOME_UNDETERMINED,
                                    classify_joint, concordant)
from gate_x_lesion.robustness import (DIRECTION_NEGATIVE, DIRECTION_NONE,
                                      DIRECTION_POSITIVE, NOT_ROBUST,
                                      ROBUST_NEGATIVE, ROBUST_POSITIVE,
                                      RobustnessVerdict)
from gate_x_lesion.validity import SharedValidityVerdict, ValidityReason

#: Frozen mapping from a robust direction to the project's outcome letter.
DIRECTION_TO_LETTER = {
    DIRECTION_POSITIVE: OUTCOME_A,      # NATIVE advantage
    DIRECTION_NEGATIVE: OUTCOME_B,      # FIXED05 advantage
}


@dataclass(frozen=True)
class SeverityClassification:
    """One severity, under one decoding convention, for one route family."""
    lam: float
    diagnostically_valid: bool
    validity_reason: str
    robustness: Optional[RobustnessVerdict]

    @property
    def contributes_direction(self) -> str:
        """Only a VALID and FULLY ROBUST severity contributes a direction."""
        if not self.diagnostically_valid or self.robustness is None:
            return DIRECTION_NONE
        if not self.robustness.robust:
            return DIRECTION_NONE
        return self.robustness.direction


def classify_convention(severities: Sequence[SeverityClassification]
                        ) -> Dict[str, object]:
    """Classify ONE route family under ONE decoding convention."""
    valid = [s for s in severities if s.diagnostically_valid]
    if not valid:
        return {
            "outcome": OUTCOME_UNDETERMINED,
            "valid_severities": [],
            "invalid_severities": {s.lam: s.validity_reason for s in severities},
            "directions": {},
            "veto_applied": False,
            "reason": "no diagnostically valid severity; a null was never tested "
                      "and is not asserted",
        }

    directions = {s.lam: s.contributes_direction for s in valid}
    robust_dirs = {d for d in directions.values() if d != DIRECTION_NONE}

    # --- the O-4 veto, checked BEFORE any letter is emitted ------------------
    if DIRECTION_POSITIVE in robust_dirs and DIRECTION_NEGATIVE in robust_dirs:
        return {
            "outcome": OUTCOME_F,
            "valid_severities": sorted(directions),
            "invalid_severities": {s.lam: s.validity_reason
                                   for s in severities if not s.diagnostically_valid},
            "directions": directions,
            "veto_applied": True,
            "reason": "diagnostically valid severities in this route family satisfy "
                      "the FULL robust rule in OPPOSITE directions; A/B/D are vetoed "
                      "(CENTRAL final decision, O-4 veto)",
        }

    if len(robust_dirs) == 1:
        d = robust_dirs.pop()
        outcome = DIRECTION_TO_LETTER[d]
        reason = (f"robust {d} on "
                  f"{sorted(k for k, v in directions.items() if v == d)}; "
                  f"no opposite-signed robust valid severity")
    else:
        outcome = OUTCOME_D
        reason = "no diagnostically valid severity satisfies the frozen robust rule"

    return {
        "outcome": outcome,
        "valid_severities": sorted(directions),
        "invalid_severities": {s.lam: s.validity_reason
                               for s in severities if not s.diagnostically_valid},
        "directions": directions,
        "veto_applied": False,
        "reason": reason,
    }


def classify_route_family(by_convention: Dict[str, Sequence[SeverityClassification]]
                          ) -> Dict[str, object]:
    """Full three-stage classification for one route family.

    Stages 1 and 2 are computed independently; stage 3 reads only their letters.
    """
    missing = [c for c in ("CANONICAL", "FREE_AR") if c not in by_convention]
    if missing:
        raise ValueError(f"missing conventions: {missing}")

    canonical = classify_convention(by_convention["CANONICAL"])
    free_ar = classify_convention(by_convention["FREE_AR"])
    joint = classify_joint(canonical["outcome"], free_ar["outcome"])

    return {
        "CANONICAL_OUTCOME": canonical["outcome"],
        "FREE_AR_OUTCOME": free_ar["outcome"],
        "JOINT_OUTCOME": joint["joint_outcome"],
        "stages": {"CANONICAL": canonical, "FREE_AR": free_ar},
        "joint": joint,
    }
