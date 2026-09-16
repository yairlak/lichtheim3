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
from typing import Callable, Dict, List, Optional, Sequence, Tuple


class UnfrozenRobustnessError(RuntimeError):
    """Raised when classification is attempted without a CENTRAL-frozen robustness rule.

    O-4 is NOT determined by any frozen authority (closure pass, §5).  The predecessor
    GATING workstream supplies the *primitives* — exact paired McNemar on discordant
    pairs, Holm correction across the planned family, a materiality floor of
    |Δ accuracy| = 0.002, a power floor of n >= 30 per cell — but it has **no severity
    axis** and therefore no rule for combining evidence across severities, and its §9
    explicitly forbids cross-witness significance testing.

    Rather than let an unfrozen default masquerade as preregistration, every entry
    point that needs "robust" requires the rule to be passed in explicitly.
    """

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

    #: Which witness this block came from.  Required by any rule that must respect
    #: GATING §9 (no cross-witness significance test) by classifying per witness.
    state_id: Optional[str] = None
    #: NATIVE minus FIXED05 accuracy for this block, for materiality-based rules.
    delta_accuracy: float = 0.0

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


@dataclass(frozen=True)
class RobustnessDecision:
    """What a robustness rule returns: is the effect robust, and with which sign."""
    robust: bool
    sign: int
    detail: str = ""


#: A robustness rule maps the blocks of one route x severity (plus optional pooled
#: statistics) to a RobustnessDecision.  No rule is frozen; see CANDIDATE_ROBUSTNESS_RULES.
RobustnessRule = Callable[..., RobustnessDecision]


def evaluate_severity(records: Sequence[SeedRecord], *,
                      robustness_rule: RobustnessRule,
                      pooled_p: Optional[float] = None,
                      pooled_net: Optional[int] = None,
                      diagnostically_valid: Optional[bool] = None,
                      min_valid_seeds: int = DEFAULT_MIN_VALID_SEEDS,
                      **validity) -> SeverityVerdict:
    """Roll one severity's lesion blocks up into a verdict.

    `robustness_rule` is REQUIRED and has no default: O-4 is not frozen, and a default
    here would be an unfrozen rule wearing preregistration's clothes.  Pass one of
    `CANDIDATE_ROBUSTNESS_RULES` only for testing, or the CENTRAL-frozen rule once it
    exists.

    `diagnostically_valid` may be supplied directly from `gate_x_lesion.validity`
    (the authoritative O-3 path).  The `min_valid_seeds` fallback is retained only for
    synthetic tests of this function in isolation.
    """
    if not records:
        raise ValueError("evaluate_severity requires at least one SeedRecord")
    if robustness_rule is None:
        raise UnfrozenRobustnessError(
            "no robustness rule supplied; O-4 is not frozen by any authority")
    lams = {r.lam for r in records}
    if len(lams) != 1:
        raise ValueError(f"records mix severities: {sorted(lams)}")
    lam = records[0].lam

    n_valid = sum(1 for r in records if r.is_valid(**validity))
    valid = (bool(diagnostically_valid) if diagnostically_valid is not None
             else n_valid >= min_valid_seeds)

    decision = robustness_rule(records, pooled_p=pooled_p, pooled_net=pooled_net)
    robust = bool(valid and decision.robust)

    return SeverityVerdict(
        lam=lam, diagnostically_valid=valid, robust=robust,
        sign=decision.sign if robust else SIGN_NONE,
        n_valid_seeds=n_valid,
        n_concordant_seeds=sum(1 for r in records
                               if r.sign(alpha=DEFAULT_ALPHA) == decision.sign),
        detail=f"{n_valid}/{len(records)} seeds valid; {decision.detail}")


# ------------------------------------------------- CANDIDATE robustness rules
# NONE OF THESE IS FROZEN.  They exist so the pipeline can be exercised and so
# CENTRAL has concrete, prospective options to choose between.  Selecting one is a
# scientific decision reserved to CENTRAL (closure pass §5).

def _sign_of(net: int) -> int:
    return (SIGN_NATIVE_ADVANTAGE if net < 0
            else SIGN_FIXED05_ADVANTAGE if net > 0 else SIGN_NONE)


def candidate_R1_per_witness_significance(records, *, pooled_p=None, pooled_net=None,
                                          alpha: float = DEFAULT_ALPHA,
                                          min_seeds_per_witness: int = 3):
    """CANDIDATE R1 — per-witness McNemar significance, unanimous across witnesses.

    Within EACH witness, >= `min_seeds_per_witness` of its lesion seeds must be
    significant at `alpha` with the same sign; both witnesses must agree on that sign.
    Never pools witnesses into one test, so it respects GATING §9.

    Weakness: on 29,571 paired items an exact McNemar is significant at discordance
    splits that are scientifically trivial, so R1 can call a handful of items "robust".
    """
    by_state: Dict[object, List[SeedRecord]] = {}
    for r in records:
        by_state.setdefault(getattr(r, "state_id", None), []).append(r)

    per_state_signs = []
    for _sid, recs in by_state.items():
        signs = [r.sign(alpha=alpha) for r in recs]
        best, n = SIGN_NONE, 0
        for s in (SIGN_NATIVE_ADVANTAGE, SIGN_FIXED05_ADVANTAGE):
            k = sum(1 for x in signs if x == s)
            if k > n:
                best, n = s, k
        per_state_signs.append(best if n >= min_seeds_per_witness else SIGN_NONE)

    uniq = set(per_state_signs)
    if len(uniq) == 1 and SIGN_NONE not in uniq:
        s = per_state_signs[0]
        return RobustnessDecision(True, s, f"R1: all {len(by_state)} witnesses agree (sign {s})")
    return RobustnessDecision(False, SIGN_NONE,
                              f"R1: witness signs {per_state_signs} not unanimous")


def candidate_R2_significance_plus_inherited_materiality(
        records, *, pooled_p=None, pooled_net=None, alpha: float = DEFAULT_ALPHA,
        min_seeds_per_witness: int = 3, materiality: float = 0.002):
    """CANDIDATE R2 — R1 plus the GATING materiality floor.

    Identical to R1, except a block counts only if it also moves accuracy by at least
    `materiality`. That number is NOT invented here: 0.002 is the |Δ accuracy| floor
    already frozen in the GATING contract's outcome B
    (`EXPERIMENT_CONTRACT.md:279,488`). Excludes statistically-significant-but-trivial
    effects, which is R1's main failure mode.
    """
    kept = [r for r in records if abs(getattr(r, "delta_accuracy", 0.0)) >= materiality]
    if not kept:
        return RobustnessDecision(False, SIGN_NONE,
                                  f"R2: no block reaches |Δacc| >= {materiality}")
    d = candidate_R1_per_witness_significance(
        kept, alpha=alpha, min_seeds_per_witness=min_seeds_per_witness)
    return RobustnessDecision(d.robust, d.sign, f"R2({d.detail})")


def candidate_R3_unanimous_sign_no_test(records, *, pooled_p=None, pooled_net=None):
    """CANDIDATE R3 — unanimous sign across all 8 blocks, no significance test at all.

    Robust iff every block's `net_change_in_correct` is non-zero and shares one sign.
    Uses no p-value anywhere, so it cannot conflict with GATING §9; and it introduces
    no effect-size threshold.

    Weakness: a net of +/-1 item in each of the 8 blocks would qualify, so R3 can also
    certify numerically tiny effects — it trades R1's sensitivity for unanimity, not
    for magnitude.
    """
    signs = {_sign_of(r.net_change_in_correct) for r in records}
    if len(signs) == 1 and SIGN_NONE not in signs:
        s = signs.pop()
        return RobustnessDecision(True, s, f"R3: all {len(records)} blocks sign {s}")
    return RobustnessDecision(False, SIGN_NONE, f"R3: block signs {signs} not unanimous")


CANDIDATE_ROBUSTNESS_RULES = {
    "R1_per_witness_significance": candidate_R1_per_witness_significance,
    "R2_significance_plus_inherited_materiality":
        candidate_R2_significance_plus_inherited_materiality,
    "R3_unanimous_sign_no_test": candidate_R3_unanimous_sign_no_test,
}

#: There is deliberately no FROZEN_ROBUSTNESS_RULE. Until CENTRAL selects one, any
#: attempt to classify must pass a rule explicitly and label it as a candidate.
FROZEN_ROBUSTNESS_RULE = None


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
