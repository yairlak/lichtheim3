"""Outcome classification, frozen by CENTRAL micro-amendment 3 (contract §7).

Pure functions over records.  No torch, no model, no I/O — so the rules can be (and are)
unit-tested on synthetic records alone, with no possibility of a scientific lesion
result leaking into a test.

The order is fixed and cannot be collapsed:

    1. CANONICAL forced-length AR, classified alone
    2. FREE_AR, classified alone
    3. JOINT_OUTCOME, only afterward

Three rules do the real work, and each exists to block a specific way of over-claiming:

* **the heterogeneity veto** — A/B/D may not be declared when another diagnostically
  valid severity in the same route family shows a robust opposite-signed effect.  This
  blocks "the effect is A, apart from one severity".
* **no retrospective promotion** — the reported unit is the SET of valid severities and
  their signs, never a single chosen severity.  This blocks "at the appropriate
  severity...".
* **convention disagreement is visible** — a CANONICAL/FREE_AR split forces a
  heterogeneous joint outcome rather than silently adopting one convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------- outcome letters

OUTCOME_A = "A"                                  # robust NATIVE advantage
OUTCOME_B = "B"                                  # robust FIXED05 advantage
OUTCOME_D = "D"                                  # robust null
OUTCOME_F = "F_HETEROGENEOUS"                    # valid severities disagree in sign
#: Not a letter: the absence of any diagnostically valid severity.  Reported as such
#: rather than being collapsed into D, which would assert a null that was never tested.
OUTCOME_UNDETERMINED = "UNDETERMINED_NO_VALID_SEVERITY"

LETTERS = (OUTCOME_A, OUTCOME_B, OUTCOME_D, OUTCOME_F)

CONVENTIONS = ("CANONICAL", "FREE_AR")

#: Sign convention, inherited from `gating_diagnostics.analysis.mcnemar`:
#: net_change_in_correct = errors_recovered - errors_gained, with a = NATIVE, b = FIXED05.
SIGN_NATIVE_ADVANTAGE = -1
SIGN_FIXED05_ADVANTAGE = +1
SIGN_NONE = 0


# ------------------------------------------------------------------ frozen defaults

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_VALID_SEEDS = 3
DEFAULT_MIN_CONCORDANT_SEEDS = 3
DEFAULT_N_SEEDS = 4
DEFAULT_MIN_CHANGED_ITEMS = 1
DEFAULT_MIN_NATIVE_EXACT = 0.01
DEFAULT_MAX_MODAL_SHARE = 0.5


@dataclass(frozen=True)
class SeedRecord:
    """One `(state, route, lambda, seed, convention)` cell."""
    lam: float
    seed: int
    n_changed_vs_intact: int
    native_exact_match: float
    modal_prediction_share: float
    net_change_in_correct: int
    p_exact_mcnemar: Optional[float]

    def is_valid(self, *, min_changed_items: int = DEFAULT_MIN_CHANGED_ITEMS,
                 min_native_exact: float = DEFAULT_MIN_NATIVE_EXACT,
                 max_modal_share: float = DEFAULT_MAX_MODAL_SHARE) -> bool:
        """Diagnostically valid = neither inert nor saturating (contract §7.4)."""
        not_inert = self.n_changed_vs_intact >= min_changed_items
        not_saturated = (self.native_exact_match >= min_native_exact
                         and self.modal_prediction_share <= max_modal_share)
        return bool(not_inert and not_saturated)

    def sign(self, *, alpha: float = DEFAULT_ALPHA) -> int:
        if self.p_exact_mcnemar is None or self.p_exact_mcnemar >= alpha:
            return SIGN_NONE
        if self.net_change_in_correct < 0:
            return SIGN_NATIVE_ADVANTAGE
        if self.net_change_in_correct > 0:
            return SIGN_FIXED05_ADVANTAGE
        return SIGN_NONE


@dataclass(frozen=True)
class SeverityVerdict:
    """The rolled-up verdict for one `(state, route, lambda, convention)` cell."""
    lam: float
    diagnostically_valid: bool
    robust: bool
    sign: int
    n_valid_seeds: int
    n_concordant_seeds: int
    detail: str = ""


def evaluate_severity(records: Sequence[SeedRecord], *,
                      pooled_p: Optional[float] = None,
                      pooled_net: Optional[int] = None,
                      alpha: float = DEFAULT_ALPHA,
                      min_valid_seeds: int = DEFAULT_MIN_VALID_SEEDS,
                      min_concordant_seeds: int = DEFAULT_MIN_CONCORDANT_SEEDS,
                      **validity) -> SeverityVerdict:
    """Roll one severity's lesion seeds up into a verdict (contract §7.4, robustness).

    Validity: at least `min_valid_seeds` of the seeds are diagnostically valid.
    Robustness: at least `min_concordant_seeds` seeds are significant at `alpha` with
    the SAME sign, AND the seed-pooled test agrees in significance and sign.
    """
    if not records:
        raise ValueError("evaluate_severity requires at least one SeedRecord")
    lams = {r.lam for r in records}
    if len(lams) != 1:
        raise ValueError(f"records mix severities: {sorted(lams)}")
    lam = records[0].lam

    n_valid = sum(1 for r in records if r.is_valid(**validity))
    valid = n_valid >= min_valid_seeds

    signs = [r.sign(alpha=alpha) for r in records]
    best_sign, best_n = SIGN_NONE, 0
    for s in (SIGN_NATIVE_ADVANTAGE, SIGN_FIXED05_ADVANTAGE):
        n = sum(1 for x in signs if x == s)
        if n > best_n:
            best_sign, best_n = s, n

    robust = bool(valid and best_sign != SIGN_NONE and best_n >= min_concordant_seeds)
    if robust and pooled_p is not None:
        pooled_sign = (SIGN_NONE if pooled_p >= alpha or pooled_net is None
                       else (SIGN_NATIVE_ADVANTAGE if pooled_net < 0
                             else SIGN_FIXED05_ADVANTAGE if pooled_net > 0
                             else SIGN_NONE))
        if pooled_sign != best_sign:
            robust = False

    return SeverityVerdict(
        lam=lam, diagnostically_valid=valid, robust=robust,
        sign=best_sign if robust else SIGN_NONE,
        n_valid_seeds=n_valid, n_concordant_seeds=best_n,
        detail=f"{n_valid}/{len(records)} seeds valid; "
               f"{best_n}/{len(records)} concordant")


def classify_convention(verdicts: Sequence[SeverityVerdict]) -> Dict[str, object]:
    """Classify ONE route family under ONE decoding convention (contract §7.5-§7.7).

    Only diagnostically valid severities are considered, and ALL of them are.  No
    severity is promoted, dropped or reweighted.
    """
    valid = [v for v in verdicts if v.diagnostically_valid]
    if not valid:
        return {
            "outcome": OUTCOME_UNDETERMINED,
            "valid_severities": [],
            "signs": {},
            "veto_applied": False,
            "reason": "no diagnostically valid severity; a null was never tested and "
                      "is not asserted",
        }

    signs = {v.lam: v.sign for v in valid}
    robust_signs = {v.sign for v in valid if v.robust}

    # --- the heterogeneity veto, checked BEFORE any letter is emitted -------------
    if SIGN_NATIVE_ADVANTAGE in robust_signs and SIGN_FIXED05_ADVANTAGE in robust_signs:
        return {
            "outcome": OUTCOME_F,
            "valid_severities": sorted(signs),
            "signs": signs,
            "veto_applied": True,
            "reason": "diagnostically valid severities in this route family show "
                      "robust opposite-signed NATIVE-vs-FIXED05 effects; A/B/D are "
                      "vetoed (contract §7.6)",
        }

    if robust_signs == {SIGN_NATIVE_ADVANTAGE}:
        outcome, reason = OUTCOME_A, "robust NATIVE advantage, no opposite-signed valid severity"
    elif robust_signs == {SIGN_FIXED05_ADVANTAGE}:
        outcome, reason = OUTCOME_B, "robust FIXED05 advantage, no opposite-signed valid severity"
    else:
        outcome, reason = OUTCOME_D, "no diagnostically valid severity shows a robust effect"

    return {
        "outcome": outcome,
        "valid_severities": sorted(signs),
        "signs": signs,
        "veto_applied": False,
        "reason": reason,
    }


def classify_joint(canonical: str, free_ar: str) -> Dict[str, object]:
    """Stage 3.  Runs only after both conventions are classified independently."""
    for letter in (canonical, free_ar):
        if letter not in LETTERS + (OUTCOME_UNDETERMINED,):
            raise ValueError(f"unknown outcome {letter!r}")

    if canonical == OUTCOME_F or free_ar == OUTCOME_F:
        return {"joint_outcome": OUTCOME_F, "canonical": canonical, "free_ar": free_ar,
                "reason": "a stage-level classification is heterogeneous"}
    if canonical != free_ar:
        return {"joint_outcome": OUTCOME_F, "canonical": canonical, "free_ar": free_ar,
                "reason": "CANONICAL and FREE_AR disagree; the disagreement remains "
                          "visible and forces a heterogeneous joint classification "
                          "(contract §7.8)"}
    return {"joint_outcome": canonical, "canonical": canonical, "free_ar": free_ar,
            "reason": "both decoding conventions agree"}


def classify_route_family(by_convention: Dict[str, Sequence[SeverityVerdict]]
                          ) -> Dict[str, object]:
    """Full three-stage classification for one route family."""
    missing = [c for c in CONVENTIONS if c not in by_convention]
    if missing:
        raise ValueError(f"missing conventions: {missing}")
    stages = {c: classify_convention(by_convention[c]) for c in CONVENTIONS}
    joint = classify_joint(stages["CANONICAL"]["outcome"], stages["FREE_AR"]["outcome"])
    return {"stages": stages, "joint": joint}
