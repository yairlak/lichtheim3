"""Emit `gxlr_conditions.final.json` FROM THE LIVE CONSTANTS.

Generating the manifest from the imported modules rather than hand-writing it means
the serialised rule set cannot drift from the executable rule set: if a constant
changed, this file changes with it and the FINAL_CONTRACT_HASH moves.

Deterministic: sorted keys, no timestamps, no run-dependent values.

    python3 scripts/gate_x_lesion/emit_final_conditions.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gate_x_lesion import noise, outcomes, robustness, validity      # noqa: E402
from gate_x_lesion.classify import DIRECTION_TO_LETTER               # noqa: E402
from gate_x_lesion.identity import STATE_IDENTITY_DOMAIN             # noqa: E402
from gate_x_lesion.sd import (SD_BATCH_SIZE, SD_DDOF, SD_N_ITEMS,    # noqa: E402
                              SD_SAMPLE_SEED)
from gate_x_lesion.targets import FROZEN_ROUTES                      # noqa: E402
from scripts.gate_x_lesion.run_gate_x_lesion import (                # noqa: E402
    QUARANTINE_DIR, SCIENTIFIC_BATCH_SIZE, SCIENTIFIC_DIR, FROZEN_SEEDS)

OUT = os.path.join(ROOT, "paper_programme", "gate_x_lesion_recovery",
                   "gxlr_conditions.final.json")

POPULATION_N = 29571


def build() -> dict:
    return {
        "schema": "gxlr_conditions.final/v1",
        "title": "GATE x LESION / RECOVERY — FINAL FROZEN RULE SET",
        "authority": "CENTRAL STEERING, authoritative final rule decision",
        "status": "FINAL_RULES_FROZEN_AWAITING_EXECUTION_GO",
        "go_for_scientific_execution": False,
        "no_canonical_non_zero_science_has_run": True,
        "supersedes": {
            "gxlr_conditions.frozen.json": "diagnostic_validity, robustness",
            "gxlr_conditions.amendment1.json":
                "O3_diagnostic_validity.unresolved_requiring_central, "
                "O4_robustness.candidates_not_frozen (candidates were never frozen; "
                "CENTRAL has now decided)",
        },
        "lineage": {
            "design_freeze_commit": "545ea436fa8f33480d74185a48a61874c04a17eb",
            "implementation_commit": "975560e6e2f1737f8915f56cf330223fd023be7c",
            "design_amendment_commit": "2a70f76b13a4d7909c56e8b15a0bf2f1942851ce",
            "implementation_amendment_commit": "4e65b14d6cfedb7eb73a9ff10d4a0869457c7e53",
        },

        "O3_shared_diagnostic_validity": {
            "O3_VALIDITY_CONVENTION_SCOPE": validity.VALIDITY_CONVENTION_SCOPE,
            "O3_IMPLEMENTATION_FAILURE_SEMANTICS":
                validity.IMPLEMENTATION_FAILURE_SEMANTICS,
            "one_shared_label_per_route_severity": True,
            "must_hold_under_both_conventions": list(validity.CONVENTIONS),
            "decoder_convention_does_not_select_valid_severities": True,
            "targeted_min_drop": validity.TARGETED_MIN_DROP,
            "min_blocks_with_drop": validity.MIN_BLOCKS_WITH_DROP,
            "n_blocks": validity.N_BLOCKS,
            "n_witnesses": validity.N_WITNESSES,
            "n_lesion_seeds": validity.N_LESION_SEEDS,
            "deterministic_tolerance": validity.DETERMINISTIC_TOLERANCE,
            "deterministic_tolerance_kind": validity.DETERMINISTIC_TOLERANCE_KIND,
            "deterministic_tolerance_source": validity.DETERMINISTIC_TOLERANCE_SOURCE,
            "drop_computed_from_integer_counts": True,
            "drop_boundary_note":
                "0.10 x 29571 = 2957.1; 2957 items fails and 2958 passes. The drop is "
                "one exact rational from integer counts, never a subtraction of two "
                "floating accuracies (which gives 0.09999999999999998 for a "
                "900/1000 -> 800/1000 block and fails >= 0.10 spuriously).",
            "no_denominator_shrinking": True,
            "no_block_exclusion": True,
            "validity_reasons": list(validity.ValidityReason.ALL),
            "implementation_failures": list(validity.ImplementationFailure.ALL),
            "failure_class_semantics": {
                "SCIENTIFIC_PERTURBATION_VALIDITY_FAILURE":
                    "the >= 0.10 targeted-route criterion failed for THIS route x "
                    "severity only; the severity is invalid, the reason is recorded, "
                    "and the experiment CONTINUES",
                "IMPLEMENTATION_FAILURE":
                    "a structural/wiring guarantee was violated; the scientific run "
                    "ABORTS and no classification is emitted; never downgraded to an "
                    "invalid severity",
            },
            "validity_never_reads_fusion_contrast": True,
            "implementation": "gate_x_lesion/validity.py::decide_shared_validity",
        },

        "O4_robustness": {
            "O4_ROBUSTNESS_RULE": robustness.ROBUSTNESS_RULE_ID,
            "O4_STATISTICAL_UNIT": robustness.O4_STATISTICAL_UNIT,
            "O4_TEST": robustness.O4_TEST,
            "O4_SEED_HANDLING": robustness.O4_SEED_HANDLING,
            "O4_WITNESS_HANDLING": robustness.O4_WITNESS_HANDLING,
            "O4_ALPHA": robustness.O4_ALPHA,
            "O4_MULTIPLICITY": robustness.O4_MULTIPLICITY,
            "O4_MAGNITUDE_FLOOR": "0.002_ABSOLUTE_ACCURACY",
            "materiality_floor": robustness.MATERIALITY_FLOOR,
            "materiality_is_prospective_not_empirically_derived": True,
            "delta_definition": "Delta[w,s] = accuracy_native[w,s] - accuracy_fixed05[w,s]",
            "sign_convention": {
                "POSITIVE": "Delta > 0 -> NATIVE advantage -> outcome A",
                "NEGATIVE": "Delta < 0 -> FIXED05 advantage -> outcome B",
                "ZERO": "neutral; counts toward neither direction",
                "warning": "This is the NEGATION of the legacy net_change_in_correct "
                           "column (errors_recovered - errors_gained); the frozen rule "
                           "never reads that column.",
            },
            "min_directional_seeds": robustness.MIN_DIRECTIONAL_SEEDS,
            "n_lesion_seeds": robustness.N_LESION_SEEDS,
            "n_witnesses": robustness.N_WITNESSES,
            "conditions": [
                "both witnesses independently support the same direction",
                "within each witness, >= 3/4 lesion seeds have Delta with that direction",
                "within each witness, mean over exactly the 4 seeds has that direction "
                "and |mean| >= 0.002",
                "no seed in either witness shows an opposite-signed effect with "
                "|Delta| >= 0.002",
                "no pooling of W3 and W4 may rescue a failed witness-level rule",
                "no item-level significance test",
                "no seed-level significance test",
                "no cross-witness significance test",
            ],
            "verdicts": [robustness.ROBUST_POSITIVE, robustness.ROBUST_NEGATIVE,
                         robustness.NOT_ROBUST],
            "delta_never_rounded_before_comparison": True,
            "integer_count_equivalence": {
                "rule_is_accuracy_not_items": True,
                "per_seed_threshold_items": robustness.integer_threshold_for(POPULATION_N, 1),
                "four_seed_sum_threshold_items":
                    robustness.integer_threshold_for(POPULATION_N,
                                                     robustness.N_LESION_SEEDS),
                "derivation": "0.002 x 29571 = 59.142 -> ceil 60 per seed; "
                              "0.002 x 4 x 29571 = 236.568 -> ceil 237 for the "
                              "four-seed sum. CEILING, not rounding; thresholds are "
                              "DERIVED from the 0.002 accuracy rule, never hard-coded.",
                "must_not_be_described_as_59_items": True,
            },
            "implementation": "gate_x_lesion/robustness.py::frozen_robustness "
                              "(and ::frozen_robustness_from_counts, preferred)",
        },

        "O4_veto": {
            "rule": "For a fixed route family and decoding convention: if one "
                    "diagnostically valid severity supports a robust direction and "
                    "another diagnostically valid severity satisfies the FULL ROBUST "
                    "rule in the opposite direction, A/B/D may NOT be declared; the "
                    "classification is OUTCOME_F_HETEROGENEOUS.",
            "only_valid_and_fully_robust_severities_contribute_a_direction": True,
            "non_robust_opposite_does_not_veto": True,
            "invalid_severity_cannot_veto": True,
            "all_valid_severities_remain_visible": True,
            "no_retrospective_promotion": True,
        },

        "classification": {
            "order": ["CANONICAL", "FREE_AR", "JOINT"],
            "conventions_classified_independently_first": True,
            "outcome_letters": {
                "A": "robust NATIVE advantage",
                "B": "robust FIXED05 advantage",
                "D": "robust null",
                outcomes.OUTCOME_F: "valid severities robust in opposite directions",
                outcomes.OUTCOME_UNDETERMINED:
                    "no diagnostically valid severity; a null was never tested",
            },
            "direction_to_letter": dict(sorted(DIRECTION_TO_LETTER.items())),
            "joint_concordant_prefix": outcomes.JOINT_CONCORDANT_PREFIX,
            "joint_mixed_decoding": outcomes.JOINT_MIXED_DECODING,
            "joint_rule": "same classification X in both conventions -> CONCORDANT_X; "
                          "they differ materially -> MIXED_DECODING",
            "concordant_label_examples": {
                "A": outcomes.concordant(outcomes.OUTCOME_A),
                "B": outcomes.concordant(outcomes.OUTCOME_B),
                "D": outcomes.concordant(outcomes.OUTCOME_D),
                outcomes.OUTCOME_F: outcomes.concordant(outcomes.OUTCOME_F),
            },
            "favourable_convention_never_overwrites_the_other": True,
            "implementation": "gate_x_lesion/classify.py::classify_route_family",
        },

        "unchanged_experimental_design": {
            "routes": list(FROZEN_ROUTES),
            "lambdas": list(noise.FROZEN_LAMBDAS),
            "lesion_seeds": list(FROZEN_SEEDS),
            "batch_size": SCIENTIFIC_BATCH_SIZE,
            "population_n": POPULATION_N,
            "fusion_conditions": ["NATIVE", "FIXED05"],
            "decoding_conventions": list(validity.CONVENTIONS),
            "rng_identity_fields": list(noise.IDENTITY_FIELDS),
            "rng_identity_excludes": list(noise.IDENTITY_EXCLUDED),
            "eps_domain": noise.EPS_DOMAIN.decode("ascii"),
            "state_identity_domain": STATE_IDENTITY_DOMAIN,
            "sd": {"definition": "torch.Tensor.std() of flattened pooled intact "
                                 "activations (the 'std' column, NOT 'rms')",
                   "ddof": SD_DDOF, "n_items": SD_N_ITEMS,
                   "sample_seed": SD_SAMPLE_SEED, "batch_size": SD_BATCH_SIZE,
                   "shape": "scalar per (state, route)"},
            "gate": {"alpha": 2.0, "gate_threshold": 0.7},
            "timing": "per_item_frozen",
        },

        "output_namespaces": {
            "scientific_execution": SCIENTIFIC_DIR,
            "scientific_execution_reserved_until_central_go": True,
            "quarantine": QUARANTINE_DIR,
            "namespaces_must_not_overlap": True,
        },

        "prohibitions": [
            "no training or fine-tuning", "no recovery training",
            "no semantic attractor", "no Lesioning V2", "no connectivity damage",
            "no change to gate architecture, alpha, gate_threshold, or fixed05",
            "no severity outside {0.25, 0.50, 1.00}",
            "no severity chosen or dropped by NATIVE-vs-FIXED05 behaviour",
            "no checkpoint substitution", "no contact with V7 / Phase 9 state",
            "no mutation of 993f2e73ecd0b9de2c3318255afbaa159948b09a",
            "no significance test of any kind in the frozen robustness rule",
            "no pooling of the two witnesses",
            "no denominator shrinking or block exclusion",
            "no classification after an implementation failure",
        ],
    }


def main() -> int:
    payload = build()
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
