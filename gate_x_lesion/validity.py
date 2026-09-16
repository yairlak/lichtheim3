"""O-3 — machine-decidable diagnostic validity.

The accepted scientific rule (CENTRAL, closure directive §4).  A route x severity is
diagnostically valid only if, across **2 checkpoints x 4 lesion seeds = 8 blocks**:

  1. the TARGETED isolated route drops by >= 0.10 exact-match versus intact, in at
     least 6 of the 8 blocks;
  2. the UNTARGETED isolated route is unchanged within deterministic evaluator
     tolerance;
  3. no shared downstream readout / checkpoint parameter is mutated;
  4. for dorsal perturbation, `c_LTM` and `g` are unchanged within deterministic
     numerical tolerance — a violation is an IMPLEMENTATION FAILURE, not a result.

Validity NEVER depends on the NATIVE-vs-FIXED05 comparison.  Nothing in this module
reads a fusion contrast; the functions below take isolated-route and gate quantities
only, so the dependency is impossible rather than merely discouraged.

Relation to GATING §9.  Criterion 1 counts blocks; it is a **replication/diagnostic
count, not a significance test**, so it does not conflict with the frozen prohibition
on cross-witness significance testing (`EXPERIMENT_CONTRACT.md` §9: "No cross-witness
significance test. Four witnesses from three seeds of one historical cohort are not an
independent population").  That prohibition does bind O-4; see `outcomes.py`.

Deterministic tolerance — RESOLVED FROM PRIOR AUTHORITY, not invented here
-------------------------------------------------------------------------
"within deterministic evaluator tolerance" is **bitwise equality, tolerance exactly
0.0**, inherited from the frozen GATING lineage:

  * `EXPERIMENT_CONTRACT.md` (gating) §5: "A determinism check — re-run one batch,
    require bitwise equality — runs before the main pass and hard-stops on failure."
  * `scripts/gating_diagnostics/run_gate_route_audit.py:168-180` (`assert_determinism`)
    compares `gate` and the exact-match columns with `!=` and hard-stops on any
    difference.
  * frozen `summary_metrics.json`: `determinism_max_gate_dev == 0.0` in **all eight**
    states.

Empirically confirmed reachable on this architecture: T8 measures dorsal
`max|Δg| = 0.000e+00` and `max|Δc_LTM| = 0.000e+00` with isolated-LTM predictions
identical, at every frozen severity.

TWO FIELDS REMAIN UNRESOLVED (see `UNRESOLVED_O3` below).  They have **no defaults**:
`decide_route_severity_validity` requires them explicitly and raises otherwise, so the
pipeline fails closed rather than silently adopting a guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------- frozen constants

#: Criterion 1: minimum exact-match drop of the TARGETED isolated route vs intact.
TARGETED_MIN_DROP = 0.10

#: Criterion 1: how many of the 8 blocks must show that drop.
MIN_BLOCKS_WITH_DROP = 6

#: The block grid: 2 checkpoints x 4 lesion seeds.
N_CHECKPOINTS = 2
N_LESION_SEEDS = 4
N_BLOCKS = N_CHECKPOINTS * N_LESION_SEEDS          # == 8

#: Criteria 2 and 4: "within deterministic evaluator tolerance".
#: Resolved from prior authority (see module docstring) — bitwise, exactly zero.
DETERMINISTIC_TOLERANCE = 0.0
DETERMINISTIC_TOLERANCE_KIND = "bitwise"
DETERMINISTIC_TOLERANCE_SOURCE = (
    "gating EXPERIMENT_CONTRACT.md §5; "
    "scripts/gating_diagnostics/run_gate_route_audit.py:168-180 (assert_determinism); "
    "frozen summary_metrics.json determinism_max_gate_dev == 0.0 in all 8 states")

#: The route each perturbation targets, and the route it must leave alone.
_OTHER = {"wm_encoder_state": "ltm_encoder_state",
          "ltm_encoder_state": "wm_encoder_state"}
_FAMILY = {"wm_encoder_state": "dorsal", "ltm_encoder_state": "ventral"}


#: Fields NOT determined by any frozen authority.  Must be supplied by CENTRAL.
UNRESOLVED_O3 = {
    "validity_convention_scope": (
        "Is route x severity validity computed ONCE from shared route diagnostics, or "
        "SEPARATELY per decoding convention? The accepted rule states the 8-block grid "
        "as 2 checkpoints x 4 lesion seeds and does not mention the decoding "
        "convention, yet isolated-route exact-match exists under BOTH canonical and "
        "free-AR. Allowed values: 'shared' | 'per_convention'."),
    "implementation_failure_block_semantics": (
        "When criterion 3 or 4 is violated (an IMPLEMENTATION FAILURE, not a result), "
        "what happens to the 8-block denominator? Allowed values: "
        "'abort_run' (the whole condition is void and nothing is reported) | "
        "'exclude_block_and_shrink_denominator' (require 6 of the surviving blocks) | "
        "'exclude_block_keep_denominator' (still require 6 of 8, so a failed block "
        "counts against validity)."),
}


class UnresolvedPolicyError(RuntimeError):
    """Raised when a rule CENTRAL has not frozen would have to be guessed."""


# ------------------------------------------------------------------- data model

@dataclass(frozen=True)
class BlockObservation:
    """One (checkpoint, lesion_seed) block for one route x severity.

    All quantities are ISOLATED-ROUTE or GATE quantities measured against the intact
    control. No NATIVE/FIXED05 contrast appears here, by design.
    """
    state_id: str
    lesion_seed: int
    route: str
    lam: float
    convention: str

    #: exact-match of the isolated TARGETED route: intact, then lesioned.
    targeted_exact_intact: float
    targeted_exact_lesioned: float

    #: max |difference| of the isolated UNTARGETED route's per-item exact-match
    #: against intact. 0.0 means bitwise unchanged.
    untargeted_max_abs_delta: float

    #: True iff every shared downstream readout / checkpoint parameter is unmutated
    #: (state_dict hash identical around the lesion context).
    shared_params_unmutated: bool

    #: Dorsal only: max |Δ| of c_LTM and g against intact. Ignored for ventral.
    max_abs_delta_c_ltm: float = 0.0
    max_abs_delta_gate: float = 0.0

    #: Preferred source for the targeted drop. When supplied, the drop is computed as
    #: ONE division of an integer count difference, which is the numerically correct
    #: form: taking the difference of two separately-rounded accuracies introduces
    #: representation error right at the decision boundary (e.g. 0.90 - 0.10 evaluates
    #: to 0.09999999999999987, which would fail a `>= 0.10` test spuriously).
    n_items: Optional[int] = None
    n_correct_intact: Optional[int] = None
    n_correct_lesioned: Optional[int] = None

    # -- criteria -----------------------------------------------------------
    @property
    def targeted_drop(self) -> float:
        """Exact-match drop of the targeted isolated route, intact minus lesioned.

        Computed from counts when available (one division of an integer difference),
        otherwise from the two accuracies. Note that on the canonical population an
        exact tie with the threshold is arithmetically impossible:
        0.10 x 29571 = 2957.1 is not an integer, so no integer count difference can
        land exactly on TARGETED_MIN_DROP.
        """
        if (self.n_items and self.n_correct_intact is not None
                and self.n_correct_lesioned is not None):
            return (self.n_correct_intact - self.n_correct_lesioned) / self.n_items
        return self.targeted_exact_intact - self.targeted_exact_lesioned

    def criterion_1_targeted_drop(self) -> bool:
        return self.targeted_drop >= TARGETED_MIN_DROP

    def criterion_2_untargeted_unchanged(self) -> bool:
        return self.untargeted_max_abs_delta <= DETERMINISTIC_TOLERANCE

    def criterion_3_no_shared_mutation(self) -> bool:
        return bool(self.shared_params_unmutated)

    def criterion_4_dorsal_gate_unchanged(self) -> bool:
        """Only constrains dorsal perturbation; vacuously true for ventral."""
        if _FAMILY[self.route] != "dorsal":
            return True
        return (self.max_abs_delta_c_ltm <= DETERMINISTIC_TOLERANCE
                and self.max_abs_delta_gate <= DETERMINISTIC_TOLERANCE)

    def implementation_failure(self) -> Optional[str]:
        """Criteria 3 and 4 are wiring guarantees. A violation is a BUG, not a null."""
        if not self.criterion_3_no_shared_mutation():
            return (f"shared downstream/checkpoint parameter mutated "
                    f"({self.state_id} {self.route} lam={self.lam} "
                    f"seed={self.lesion_seed})")
        if not self.criterion_4_dorsal_gate_unchanged():
            return (f"dorsal perturbation moved c_LTM (|Δ|={self.max_abs_delta_c_ltm}) "
                    f"or g (|Δ|={self.max_abs_delta_gate}) beyond the deterministic "
                    f"tolerance {DETERMINISTIC_TOLERANCE} "
                    f"({self.state_id} seed={self.lesion_seed} lam={self.lam})")
        return None

    def criterion_2_failure(self) -> Optional[str]:
        if not self.criterion_2_untargeted_unchanged():
            return (f"untargeted route {_OTHER[self.route]} moved "
                    f"(|Δ|={self.untargeted_max_abs_delta}) beyond the deterministic "
                    f"tolerance {DETERMINISTIC_TOLERANCE}")
        return None


@dataclass(frozen=True)
class ValidityVerdict:
    route: str
    lam: float
    convention: Optional[str]
    diagnostically_valid: bool
    n_blocks_considered: int
    n_blocks_with_drop: int
    implementation_failures: Tuple[str, ...]
    criterion_2_failures: Tuple[str, ...]
    reason: str


# --------------------------------------------------------------------- decision

def decide_route_severity_validity(
        blocks: Sequence[BlockObservation], *,
        validity_convention_scope: str,
        implementation_failure_block_semantics: str,
        convention: Optional[str] = None) -> ValidityVerdict:
    """Decide diagnostic validity for one route x severity.

    Both policy arguments are REQUIRED and have no defaults: they are the two O-3
    fields no frozen authority determines. Calling without them, or with a value
    outside the allowed set, raises `UnresolvedPolicyError`.
    """
    if validity_convention_scope not in ("shared", "per_convention"):
        raise UnresolvedPolicyError(
            f"validity_convention_scope={validity_convention_scope!r} is not a "
            f"CENTRAL-frozen value. {UNRESOLVED_O3['validity_convention_scope']}")
    allowed = ("abort_run", "exclude_block_and_shrink_denominator",
               "exclude_block_keep_denominator")
    if implementation_failure_block_semantics not in allowed:
        raise UnresolvedPolicyError(
            f"implementation_failure_block_semantics="
            f"{implementation_failure_block_semantics!r} is not a CENTRAL-frozen "
            f"value. {UNRESOLVED_O3['implementation_failure_block_semantics']}")
    if not blocks:
        raise ValueError("decide_route_severity_validity requires at least one block")

    routes = {b.route for b in blocks}
    lams = {b.lam for b in blocks}
    if len(routes) != 1 or len(lams) != 1:
        raise ValueError(f"blocks mix route/severity: routes={routes} lambdas={lams}")
    route, lam = blocks[0].route, blocks[0].lam

    if validity_convention_scope == "per_convention":
        convs = {b.convention for b in blocks}
        if len(convs) != 1:
            raise ValueError(
                f"per_convention scope requires one convention per call, got {convs}")
        convention = blocks[0].convention

    impl_failures = tuple(m for m in (b.implementation_failure() for b in blocks) if m)
    c2_failures = tuple(m for m in (b.criterion_2_failure() for b in blocks) if m)

    if impl_failures and implementation_failure_block_semantics == "abort_run":
        raise RuntimeError(
            "HARD STOP: implementation failure in a validity block — this is a wiring "
            "failure, not a scientific result:\n  " + "\n  ".join(impl_failures))

    if implementation_failure_block_semantics == "exclude_block_and_shrink_denominator":
        considered = [b for b in blocks if b.implementation_failure() is None]
        required = MIN_BLOCKS_WITH_DROP
    else:                      # exclude_block_keep_denominator
        considered = [b for b in blocks if b.implementation_failure() is None]
        required = MIN_BLOCKS_WITH_DROP

    n_drop = sum(1 for b in considered if b.criterion_1_targeted_drop())

    ok_1 = n_drop >= required
    ok_2 = not c2_failures
    ok_34 = not impl_failures

    valid = bool(ok_1 and ok_2 and ok_34)
    if valid:
        reason = (f"{n_drop}/{len(considered)} blocks show a targeted drop "
                  f">= {TARGETED_MIN_DROP}; untargeted route and (if dorsal) "
                  f"c_LTM/g bitwise unchanged; no shared parameter mutated")
    else:
        bits = []
        if not ok_1:
            bits.append(f"only {n_drop}/{len(considered)} blocks reach the "
                        f"{TARGETED_MIN_DROP} targeted drop (need {required})")
        if not ok_2:
            bits.append(f"{len(c2_failures)} untargeted-route violation(s)")
        if not ok_34:
            bits.append(f"{len(impl_failures)} implementation failure(s)")
        reason = "; ".join(bits)

    return ValidityVerdict(
        route=route, lam=lam, convention=convention, diagnostically_valid=valid,
        n_blocks_considered=len(considered), n_blocks_with_drop=n_drop,
        implementation_failures=impl_failures, criterion_2_failures=c2_failures,
        reason=reason)
