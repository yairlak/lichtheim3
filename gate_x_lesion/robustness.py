"""O-4 — the FROZEN robustness rule. CENTRAL final decision, implemented verbatim.

    O4_ROBUSTNESS_RULE = REPLICATED_SIGN_PLUS_MATERIALITY_NO_SIGNIFICANCE_TEST
    O4_STATISTICAL_UNIT = NONE
    O4_TEST             = NONE
    O4_SEED_HANDLING    = REPLICATION_UNIT_4_LESION_SEEDS_WITHIN_EACH_WITNESS
    O4_WITNESS_HANDLING = TWO_WITNESSES_MUST_SATISFY_RULE_INDEPENDENTLY
    O4_ALPHA            = NOT_APPLICABLE
    O4_MULTIPLICITY     = NOT_APPLICABLE
    O4_MAGNITUDE_FLOOR  = 0.002 ABSOLUTE ACCURACY

There is NO significance test anywhere in this module: no p-value, no alpha, no
multiplicity correction, no bootstrap, no pooling of the two witnesses.  This is
asserted by `tests/test_gate_x_lesion_final_rules.py::test_O4_15_*`.

SIGN CONVENTION — read carefully, it is the opposite of the legacy column
---------------------------------------------------------------------------
CENTRAL defines, per witness `w` and lesion seed `s`:

    Delta[w, s] = accuracy_native[w, s] - accuracy_fixed05[w, s]

so **POSITIVE means NATIVE is better** (which maps to outcome `A`) and NEGATIVE
means FIXED05 is better (outcome `B`).

The legacy `net_change_in_correct` column inherited from the GATING analysis is
`errors_recovered - errors_gained`, i.e. `n_correct_fixed05 - n_correct_native`,
which is the NEGATED numerator of Delta.  The two must never be conflated; this
module takes Delta directly and never reads the legacy column.

THE RULE
--------
A direction is ROBUST for a fixed (route x severity x decoding convention) iff:

1. BOTH witnesses independently support the same direction;
2. within EACH witness, >= 3 of the 4 lesion seeds have Delta with that direction
   (Delta == 0 is neutral and counts toward neither);
3. for EACH witness separately, the mean over exactly the 4 frozen seeds has that
   same direction and |mean| >= 0.002;
4. no lesion seed in either witness shows an OPPOSITE-signed effect with
   |Delta| >= 0.002 (a single material reversal blocks ROBUST);
5. no pooling of W3 and W4 may rescue a failed witness-level rule.

EXACT ARITHMETIC AT THE 0.002 BOUNDARY
--------------------------------------
The formal rule is `>= 0.002 ABSOLUTE ACCURACY`.  It is **not** "59 items":

    0.002 x 29571 = 59.142        (per seed)
    0.002 x 4 x 29571 = 236.568   (sum over the 4 seeds of a witness mean)

Neither is an integer, so any integer-count implementation must take the CEILING
at the boundary, not truncate and not round.  `witness_direction_from_counts`
does this with exact rational arithmetic (`fractions.Fraction`), so it is
mathematically equivalent to the accuracy comparison for any population size,
including the case where `0.002 * N` happens to be an integer (then the integer
threshold is that value itself, because the comparison is `>=`).

Prefer the count-based entry point in the pipeline: accuracies are
`n_correct / N`, and comparing a mean of separately-rounded ratios against a
threshold can misfire exactly at the boundary.  The float entry point exists for
synthetic tests that state Delta directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Optional, Sequence, Tuple

# ----------------------------------------------------------------- constants

ROBUSTNESS_RULE_ID = "REPLICATED_SIGN_PLUS_MATERIALITY_NO_SIGNIFICANCE_TEST"
O4_STATISTICAL_UNIT = "NONE"
O4_TEST = "NONE"
O4_SEED_HANDLING = "REPLICATION_UNIT_4_LESION_SEEDS_WITHIN_EACH_WITNESS"
O4_WITNESS_HANDLING = "TWO_WITNESSES_MUST_SATISFY_RULE_INDEPENDENTLY"
O4_ALPHA = "NOT_APPLICABLE"
O4_MULTIPLICITY = "NOT_APPLICABLE"

#: Prospective materiality floor, absolute accuracy.  NOT empirically derived
#: from any lesion outcome; declared before any non-zero science was run.
MATERIALITY_FLOOR = 0.002
MATERIALITY_FLOOR_EXACT = Fraction(2, 1000)

#: The replication unit.  Exactly these four seeds, within each witness.
N_LESION_SEEDS = 4
#: Minimum same-direction seeds within a witness.
MIN_DIRECTIONAL_SEEDS = 3
#: Both witnesses must independently support the direction.
N_WITNESSES = 2

DIRECTION_POSITIVE = "POSITIVE"      # Delta > 0  -> NATIVE advantage  -> outcome A
DIRECTION_NEGATIVE = "NEGATIVE"      # Delta < 0  -> FIXED05 advantage -> outcome B
DIRECTION_NONE = "NONE"

ROBUST_POSITIVE = "ROBUST_POSITIVE"
ROBUST_NEGATIVE = "ROBUST_NEGATIVE"
NOT_ROBUST = "NOT_ROBUST"


# ------------------------------------------------------------------- results

@dataclass(frozen=True)
class WitnessSupport:
    """One witness's verdict over its 4 lesion seeds."""
    state_id: str
    positive_seed_count: int
    negative_seed_count: int
    neutral_seed_count: int
    mean_delta: float
    material_reversal: bool
    direction: str
    reason: str

    def supports(self, direction: str) -> bool:
        return self.direction == direction and direction != DIRECTION_NONE


@dataclass(frozen=True)
class RobustnessVerdict:
    verdict: str                       # ROBUST_POSITIVE | ROBUST_NEGATIVE | NOT_ROBUST
    direction: str
    witnesses: Tuple[WitnessSupport, ...]
    reason: str

    @property
    def robust(self) -> bool:
        return self.verdict in (ROBUST_POSITIVE, ROBUST_NEGATIVE)


# ------------------------------------------------------- witness-level rule

def _witness_direction_exact(deltas: Sequence[Fraction], state_id: str,
                             floor: Fraction) -> WitnessSupport:
    """The rule, evaluated in exact rational arithmetic."""
    if len(deltas) != N_LESION_SEEDS:
        raise ValueError(
            f"witness {state_id}: the replication unit is exactly "
            f"{N_LESION_SEEDS} lesion seeds, got {len(deltas)}")

    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    neu = sum(1 for d in deltas if d == 0)
    mean = sum(deltas, Fraction(0)) / N_LESION_SEEDS

    # Condition 4, evaluated per candidate direction.
    reversal_against_pos = any(d <= -floor for d in deltas)
    reversal_against_neg = any(d >= floor for d in deltas)

    if (pos >= MIN_DIRECTIONAL_SEEDS and mean >= floor
            and not reversal_against_pos):
        return WitnessSupport(state_id, pos, neg, neu, float(mean), False,
                              DIRECTION_POSITIVE,
                              f"{pos}/{N_LESION_SEEDS} positive seeds, "
                              f"mean {float(mean):+.6f} >= {float(floor)}, "
                              f"no material reversal")

    if (neg >= MIN_DIRECTIONAL_SEEDS and mean <= -floor
            and not reversal_against_neg):
        return WitnessSupport(state_id, pos, neg, neu, float(mean), False,
                              DIRECTION_NEGATIVE,
                              f"{neg}/{N_LESION_SEEDS} negative seeds, "
                              f"mean {float(mean):+.6f} <= -{float(floor)}, "
                              f"no material reversal")

    bits = []
    if pos < MIN_DIRECTIONAL_SEEDS and neg < MIN_DIRECTIONAL_SEEDS:
        bits.append(f"neither direction reaches {MIN_DIRECTIONAL_SEEDS}/"
                    f"{N_LESION_SEEDS} seeds (pos={pos}, neg={neg}, neutral={neu})")
    if abs(mean) < floor:
        bits.append(f"|mean {float(mean):+.6f}| < materiality {float(floor)}")
    elif pos >= MIN_DIRECTIONAL_SEEDS and mean < floor:
        bits.append(f"positive seed majority but mean {float(mean):+.6f} "
                    f"does not reach +{float(floor)}")
    elif neg >= MIN_DIRECTIONAL_SEEDS and mean > -floor:
        bits.append(f"negative seed majority but mean {float(mean):+.6f} "
                    f"does not reach -{float(floor)}")
    if pos >= MIN_DIRECTIONAL_SEEDS and reversal_against_pos:
        bits.append(f"material reversal: a seed has Delta <= -{float(floor)}")
    if neg >= MIN_DIRECTIONAL_SEEDS and reversal_against_neg:
        bits.append(f"material reversal: a seed has Delta >= +{float(floor)}")

    material_reversal = ((pos >= MIN_DIRECTIONAL_SEEDS and reversal_against_pos)
                         or (neg >= MIN_DIRECTIONAL_SEEDS and reversal_against_neg))
    return WitnessSupport(state_id, pos, neg, neu, float(mean), material_reversal,
                          DIRECTION_NONE, "; ".join(bits) or "no direction supported")


def witness_direction(deltas: Sequence[float], state_id: str = "?") -> WitnessSupport:
    """Witness-level rule from Delta values given directly (synthetic/testing path).

    Uses the exact binary value of each float, so a Delta stated as exactly
    `0.002` or `-0.002` compares equal to the floor, as CENTRAL requires
    (cases O4-6, O4-7, O4-8, O4-9).
    """
    return _witness_direction_exact(
        [Fraction(float(d)) for d in deltas], state_id,
        Fraction(float(MATERIALITY_FLOOR)))


def witness_direction_from_counts(correct_native: Sequence[int],
                                  correct_fixed05: Sequence[int],
                                  n_items: int,
                                  state_id: str = "?") -> WitnessSupport:
    """Witness-level rule from exact integer item counts. PREFERRED in the pipeline.

    `Delta[w,s] = (correct_native[s] - correct_fixed05[s]) / n_items`, evaluated
    as an exact rational. Mathematically equivalent to the accuracy comparison,
    with no floating-point boundary defect.

    The equivalent integer thresholds on the canonical population are
    `ceil(0.002 * 29571) = 60` per seed and `ceil(0.002 * 4 * 29571) = 237` for
    the sum over a witness's four seeds — but those numbers are DERIVED here, not
    hard-coded, because the formal rule is `>= 0.002 absolute accuracy`.
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if len(correct_native) != len(correct_fixed05):
        raise ValueError("native/fixed05 count vectors differ in length")
    deltas = [Fraction(int(a) - int(b), int(n_items))
              for a, b in zip(correct_native, correct_fixed05)]
    return _witness_direction_exact(deltas, state_id, MATERIALITY_FLOOR_EXACT)


def integer_threshold_for(n_items: int, n_seeds: int = 1) -> int:
    """Smallest integer count difference that meets `>= 0.002` absolute accuracy.

    For a single seed, `|n_native - n_fixed05| >= ceil(0.002 * N)`.
    For a witness mean over `n_seeds`, `|sum of differences| >= ceil(0.002 * n_seeds * N)`.
    CEILING, not rounding: on the canonical population `0.002 * 29571 = 59.142`,
    so 59 items is NOT material and 60 is.
    """
    exact = MATERIALITY_FLOOR_EXACT * n_seeds * n_items
    return math.ceil(exact)


# -------------------------------------------------------- global frozen rule

def frozen_robustness(deltas_by_witness: Dict[str, Sequence[float]]
                      ) -> RobustnessVerdict:
    """THE frozen O-4 rule. Pure, deterministic, no significance test.

    `deltas_by_witness` maps each witness id to its 4 Delta values, in lesion-seed
    order. Exactly two witnesses are required; pooling them is never performed.
    """
    if len(deltas_by_witness) != N_WITNESSES:
        raise ValueError(
            f"the frozen rule requires exactly {N_WITNESSES} witnesses, each "
            f"evaluated independently; got {sorted(deltas_by_witness)}")

    supports = tuple(witness_direction(d, state_id=w)
                     for w, d in sorted(deltas_by_witness.items()))

    for direction, verdict in ((DIRECTION_POSITIVE, ROBUST_POSITIVE),
                               (DIRECTION_NEGATIVE, ROBUST_NEGATIVE)):
        if all(s.supports(direction) for s in supports):
            return RobustnessVerdict(
                verdict, direction, supports,
                f"both witnesses independently support {direction}: "
                + "; ".join(f"{s.state_id}: {s.reason}" for s in supports))

    return RobustnessVerdict(
        NOT_ROBUST, DIRECTION_NONE, supports,
        "witness-level support is not unanimous (no pooling may rescue it): "
        + "; ".join(f"{s.state_id} -> {s.direction} ({s.reason})" for s in supports))


def frozen_robustness_from_counts(counts_by_witness: Dict[str, dict],
                                  n_items: int) -> RobustnessVerdict:
    """Frozen rule from exact integer counts. `{witness: {native: [...], fixed05: [...]}}`."""
    if len(counts_by_witness) != N_WITNESSES:
        raise ValueError(
            f"the frozen rule requires exactly {N_WITNESSES} witnesses")
    supports = tuple(
        witness_direction_from_counts(c["native"], c["fixed05"], n_items, state_id=w)
        for w, c in sorted(counts_by_witness.items()))

    for direction, verdict in ((DIRECTION_POSITIVE, ROBUST_POSITIVE),
                               (DIRECTION_NEGATIVE, ROBUST_NEGATIVE)):
        if all(s.supports(direction) for s in supports):
            return RobustnessVerdict(
                verdict, direction, supports,
                f"both witnesses independently support {direction}")
    return RobustnessVerdict(
        NOT_ROBUST, DIRECTION_NONE, supports,
        "witness-level support is not unanimous (no pooling may rescue it)")
