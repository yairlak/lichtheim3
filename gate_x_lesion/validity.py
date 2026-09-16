"""O-3 — FROZEN shared diagnostic validity. CENTRAL final decision, implemented verbatim.

    O3_VALIDITY_CONVENTION_SCOPE       = SHARED
    O3_IMPLEMENTATION_FAILURE_SEMANTICS = ABORT_RUN

A route x severity receives ONE shared diagnostic-validity label.  For criteria
involving decoded isolated-route performance, the requirement must hold under
**BOTH** CANONICAL and FREE_AR.  Consequences, stated so they cannot drift:

  * the decoder convention does not select different valid severities;
  * the same valid-severity set is used for the subsequent CANONICAL and FREE_AR
    NATIVE-vs-FIXED05 classifications;
  * a severity that satisfies the competence-separation criterion under only one
    decoding convention is NOT diagnostically valid for the shared causal
    comparison.

Structural criteria are shared by construction.

THE COMPETENCE CRITERION
------------------------
Across 2 witnesses x 4 lesion seeds = 8 blocks, the TARGETED isolated route must
drop by >= 0.10 exact-match relative to intact in at least 6 of 8 blocks —
**separately under CANONICAL and under FREE_AR**, and both must pass.

TWO FAILURE CLASSES THAT MUST NEVER BE CONFLATED
------------------------------------------------
`SCIENTIFIC_PERTURBATION_VALIDITY_FAILURE`
    The >= 0.10 targeted-route criterion failed for THIS route x severity.  The
    severity is marked scientifically invalid, the reason is recorded, and the
    overall experiment CONTINUES.

`IMPLEMENTATION_FAILURE`
    A structural/wiring guarantee was violated.  The scientific run ABORTS.  No
    scientific outcome classification may be emitted.  There is NO denominator
    shrinking and NO block exclusion — a failed block is never quietly dropped.
    Implementation failures include at least:
      * mutation of shared downstream / checkpoint parameters;
      * failure to restore checkpoint identity;
      * untargeted-route change inconsistent with the intervention contract;
      * dorsal perturbation changing c_LTM or g beyond the frozen exact tolerance;
      * NATIVE/FIXED05 receiving non-identical matched lesion tensors.

An implementation failure is NEVER silently transformed into an invalid severity.

DETERMINISTIC EQUALITY TOLERANCE — exact, inherited, unchanged
--------------------------------------------------------------
0.0, bitwise, from the frozen GATING lineage:
  * gating `EXPERIMENT_CONTRACT.md` §5 ("require bitwise equality");
  * `scripts/gating_diagnostics/run_gate_route_audit.py:168-180` (assert_determinism);
  * frozen `summary_metrics.json`: `determinism_max_gate_dev == 0.0` in all 8 states.

THE >= 0.10 BOUNDARY IS COMPUTED FROM INTEGER COUNTS
-----------------------------------------------------
Retained from the closure pass and NOT reverted: evaluating the drop as the
difference of two separately-rounded accuracies puts representation error exactly
on the decision boundary (`0.90 - 0.10 == 0.09999999999999998`, which fails
`>= 0.10` spuriously).  The drop is computed from integer counts via one exact
rational comparison.  On the canonical population an exact tie cannot arise
anyway: `0.10 x 29571 = 2957.1` is not an integer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------- frozen constants

VALIDITY_CONVENTION_SCOPE = "SHARED"
IMPLEMENTATION_FAILURE_SEMANTICS = "ABORT_RUN"

#: Criterion 1: minimum exact-match drop of the TARGETED isolated route vs intact.
TARGETED_MIN_DROP = 0.10
TARGETED_MIN_DROP_EXACT = Fraction(1, 10)

#: Criterion 1: how many of the 8 blocks must show that drop, per convention.
MIN_BLOCKS_WITH_DROP = 6

N_WITNESSES = 2
N_LESION_SEEDS = 4
N_BLOCKS = N_WITNESSES * N_LESION_SEEDS          # == 8

CONVENTIONS = ("CANONICAL", "FREE_AR")

#: "within deterministic evaluator tolerance" — exact, bitwise.
DETERMINISTIC_TOLERANCE = 0.0
DETERMINISTIC_TOLERANCE_KIND = "bitwise"
DETERMINISTIC_TOLERANCE_SOURCE = (
    "gating EXPERIMENT_CONTRACT.md §5; "
    "scripts/gating_diagnostics/run_gate_route_audit.py:168-180 (assert_determinism); "
    "frozen summary_metrics.json determinism_max_gate_dev == 0.0 in all 8 states")

_OTHER = {"wm_encoder_state": "ltm_encoder_state",
          "ltm_encoder_state": "wm_encoder_state"}
_FAMILY = {"wm_encoder_state": "dorsal", "ltm_encoder_state": "ventral"}


# ------------------------------------------------------------------ reason enum

class ValidityReason:
    VALID = "VALID"
    TARGET_ROUTE_DROP_FAILED_CANONICAL = "TARGET_ROUTE_DROP_FAILED_CANONICAL"
    TARGET_ROUTE_DROP_FAILED_FREE_AR = "TARGET_ROUTE_DROP_FAILED_FREE_AR"
    TARGET_ROUTE_DROP_FAILED_BOTH = "TARGET_ROUTE_DROP_FAILED_BOTH"
    ALL = (VALID, TARGET_ROUTE_DROP_FAILED_CANONICAL,
           TARGET_ROUTE_DROP_FAILED_FREE_AR, TARGET_ROUTE_DROP_FAILED_BOTH)


class ImplementationFailure:
    SHARED_PARAM_MUTATED = "IMPLEMENTATION_FAILURE_SHARED_PARAM_MUTATED"
    CHECKPOINT_NOT_RESTORED = "IMPLEMENTATION_FAILURE_CHECKPOINT_NOT_RESTORED"
    UNTARGETED_ROUTE_CHANGED = "IMPLEMENTATION_FAILURE_UNTARGETED_ROUTE_CHANGED"
    DORSAL_GATE_NULL_VIOLATED = "IMPLEMENTATION_FAILURE_DORSAL_GATE_NULL_VIOLATED"
    LESION_TENSOR_MISMATCH = "IMPLEMENTATION_FAILURE_LESION_TENSOR_MISMATCH"
    ALL = (SHARED_PARAM_MUTATED, CHECKPOINT_NOT_RESTORED, UNTARGETED_ROUTE_CHANGED,
           DORSAL_GATE_NULL_VIOLATED, LESION_TENSOR_MISMATCH)


class ImplementationFailureAbort(RuntimeError):
    """Structural violation. The scientific run aborts; no classification is emitted."""

    def __init__(self, failures: Sequence[str]):
        self.failures = tuple(failures)
        super().__init__(
            "HARD STOP — IMPLEMENTATION_FAILURE, scientific run aborted "
            "(no denominator shrinking, no block exclusion, no classification):\n  "
            + "\n  ".join(failures))


class ScientificPerturbationValidityFailure(Exception):
    """Marker type. NOT raised: competence failure never aborts the experiment."""


# ------------------------------------------------------------------- data model

@dataclass(frozen=True)
class BlockObservation:
    """One (witness, lesion_seed, convention) block for one route x severity.

    Carries ISOLATED-ROUTE and GATE quantities only. No NATIVE/FIXED05 contrast is
    representable here, so validity cannot depend on the fusion comparison.
    """
    state_id: str
    lesion_seed: int
    route: str
    lam: float
    convention: str

    #: Targeted isolated-route exact-match, as integer counts (preferred).
    n_items: int
    n_correct_intact: int
    n_correct_lesioned: int

    #: Structural guarantees. Any False/violation is an IMPLEMENTATION_FAILURE.
    untargeted_max_abs_delta: float = 0.0
    shared_params_unmutated: bool = True
    checkpoint_identity_restored: bool = True
    matched_lesion_tensors_identical: bool = True

    #: Dorsal only: max |delta| of c_LTM and g vs intact. Vacuous for ventral.
    max_abs_delta_c_ltm: float = 0.0
    max_abs_delta_gate: float = 0.0

    # -- competence criterion ------------------------------------------------
    @property
    def targeted_drop(self) -> Fraction:
        """Exact drop, one rational from integer counts. Never a float subtraction."""
        if self.n_items <= 0:
            raise ValueError(f"{self.state_id}: n_items must be positive")
        return Fraction(int(self.n_correct_intact) - int(self.n_correct_lesioned),
                        int(self.n_items))

    def criterion_1_targeted_drop(self) -> bool:
        return self.targeted_drop >= TARGETED_MIN_DROP_EXACT

    # -- structural criteria -------------------------------------------------
    def implementation_failures(self) -> Tuple[str, ...]:
        where = (f"{self.state_id} seed={self.lesion_seed} {self.route} "
                 f"lam={self.lam} {self.convention}")
        out: List[str] = []
        if not self.shared_params_unmutated:
            out.append(f"{ImplementationFailure.SHARED_PARAM_MUTATED}: {where}")
        if not self.checkpoint_identity_restored:
            out.append(f"{ImplementationFailure.CHECKPOINT_NOT_RESTORED}: {where}")
        if not self.matched_lesion_tensors_identical:
            out.append(f"{ImplementationFailure.LESION_TENSOR_MISMATCH}: "
                       f"NATIVE and FIXED05 received non-identical matched lesion "
                       f"tensors at {where}")
        if self.untargeted_max_abs_delta > DETERMINISTIC_TOLERANCE:
            out.append(f"{ImplementationFailure.UNTARGETED_ROUTE_CHANGED}: "
                       f"{_OTHER[self.route]} moved by "
                       f"{self.untargeted_max_abs_delta} > "
                       f"{DETERMINISTIC_TOLERANCE} at {where}")
        if _FAMILY[self.route] == "dorsal":
            if (self.max_abs_delta_c_ltm > DETERMINISTIC_TOLERANCE
                    or self.max_abs_delta_gate > DETERMINISTIC_TOLERANCE):
                out.append(f"{ImplementationFailure.DORSAL_GATE_NULL_VIOLATED}: "
                           f"|d c_LTM|={self.max_abs_delta_c_ltm}, "
                           f"|d g|={self.max_abs_delta_gate} > "
                           f"{DETERMINISTIC_TOLERANCE} at {where}")
        return tuple(out)


@dataclass(frozen=True)
class ConventionBlockResult:
    convention: str
    n_blocks: int
    n_blocks_with_drop: int
    passed: bool
    per_block: Tuple[Tuple[str, int, float, bool], ...]   # (state, seed, drop, ok)


@dataclass(frozen=True)
class SharedValidityVerdict:
    route: str
    lam: float
    diagnostically_valid: bool
    reason: str
    canonical: ConventionBlockResult
    free_ar: ConventionBlockResult
    structural_ok: bool
    detail: str

    @property
    def scientific_perturbation_validity_failure(self) -> bool:
        return (not self.diagnostically_valid) and self.structural_ok


# --------------------------------------------------------------------- decision

def _convention_result(blocks: Sequence[BlockObservation],
                       convention: str) -> ConventionBlockResult:
    per = tuple((b.state_id, b.lesion_seed, float(b.targeted_drop),
                 b.criterion_1_targeted_drop()) for b in blocks)
    n_ok = sum(1 for _s, _k, _d, ok in per if ok)
    return ConventionBlockResult(
        convention=convention, n_blocks=len(blocks), n_blocks_with_drop=n_ok,
        passed=n_ok >= MIN_BLOCKS_WITH_DROP, per_block=per)


def decide_shared_validity(canonical_blocks: Sequence[BlockObservation],
                           free_ar_blocks: Sequence[BlockObservation],
                           *, require_full_grid: bool = True
                           ) -> SharedValidityVerdict:
    """ONE shared diagnostic-validity label for a route x severity.

    Raises `ImplementationFailureAbort` on any structural violation in EITHER
    convention — checked first, and never downgraded to an invalid severity.
    A competence failure returns an invalid verdict and does NOT raise.
    """
    if not canonical_blocks or not free_ar_blocks:
        raise ValueError("both CANONICAL and FREE_AR blocks are required "
                         "(O3_VALIDITY_CONVENTION_SCOPE=SHARED)")

    all_blocks = list(canonical_blocks) + list(free_ar_blocks)
    routes = {b.route for b in all_blocks}
    lams = {b.lam for b in all_blocks}
    if len(routes) != 1 or len(lams) != 1:
        raise ValueError(f"blocks mix route/severity: routes={routes} lambdas={lams}")
    route, lam = all_blocks[0].route, all_blocks[0].lam

    for blocks, conv in ((canonical_blocks, "CANONICAL"), (free_ar_blocks, "FREE_AR")):
        bad = {b.convention for b in blocks} - {conv}
        if bad:
            raise ValueError(f"{conv} block set contains conventions {bad}")
        if require_full_grid and len(blocks) != N_BLOCKS:
            raise ValueError(
                f"{conv}: expected the full {N_WITNESSES}x{N_LESION_SEEDS} grid of "
                f"{N_BLOCKS} blocks, got {len(blocks)}. No denominator shrinking.")

    # --- structural first: abort before anything else is computed -------------
    failures: List[str] = []
    for b in all_blocks:
        failures.extend(b.implementation_failures())
    if failures:
        raise ImplementationFailureAbort(failures)

    canon = _convention_result(canonical_blocks, "CANONICAL")
    free = _convention_result(free_ar_blocks, "FREE_AR")

    if canon.passed and free.passed:
        reason = ValidityReason.VALID
        valid = True
    elif not canon.passed and not free.passed:
        reason = ValidityReason.TARGET_ROUTE_DROP_FAILED_BOTH
        valid = False
    elif not canon.passed:
        reason = ValidityReason.TARGET_ROUTE_DROP_FAILED_CANONICAL
        valid = False
    else:
        reason = ValidityReason.TARGET_ROUTE_DROP_FAILED_FREE_AR
        valid = False

    detail = (f"CANONICAL {canon.n_blocks_with_drop}/{canon.n_blocks} blocks with "
              f"drop >= {TARGETED_MIN_DROP}; FREE_AR {free.n_blocks_with_drop}/"
              f"{free.n_blocks}; need >= {MIN_BLOCKS_WITH_DROP} under BOTH "
              f"(O3_VALIDITY_CONVENTION_SCOPE=SHARED)")
    if not valid:
        detail += ("; SCIENTIFIC_PERTURBATION_VALIDITY_FAILURE for this route x "
                   "severity only — the experiment continues")

    return SharedValidityVerdict(
        route=route, lam=lam, diagnostically_valid=valid, reason=reason,
        canonical=canon, free_ar=free, structural_ok=True, detail=detail)
