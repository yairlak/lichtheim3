#!/bin/sh
# =====================================================================
#  GATE x LESION / RECOVERY — PREPARED SCIENTIFIC EXECUTION COMMAND
#
#                          *** NOT RUN ***
#
#  FINAL_DESIGN_DECISIONS_COMPLETE                = YES
#  FINAL_RULE_IMPLEMENTATION_COMPLETE             = YES
#  GO_FOR_SCIENTIFIC_EXECUTION                    = NO
#  AWAITING_CENTRAL_FINAL_SCIENTIFIC_EXECUTION_GO = YES
#
#  This file records the exact command CENTRAL would issue after final
#  approval.  It is deliberately NOT executable (no +x bit) and has never
#  been executed.  No canonical non-zero science has run: the largest
#  non-zero-severity population evaluated anywhere is a 24-item
#  quarantined subset under NOT_SCIENTIFIC_RESULT/.
#
#  INTEGRATION REPAIR APPLIED: the runner now produces the full frozen
#  eight-section output package by CALLING the frozen O-3 / O-4 / classifier
#  modules, and the mandatory post-run validator can accept it.  The previous
#  authorization (hash f2b3f561…) is VOID for this repaired runner.
#
#  The runner refuses on its own, before any model is loaded, if any of
#  these is not satisfied — verified from the CLI:
#    * --final-contract-hash absent or not matching a fresh recomputation
#    * --batch-size other than 256                        (F-8)
#    * non-zero severity without --i-have-central-go
#    * --limit without --smoke
#    * a severity outside {0.25, 0.50, 1.00}
# =====================================================================
#
#  FROZEN RULE SET
#
#    FINAL_CONTRACT_HASH
#      d5e9eb41adb1574b83ca9a308e976e8f324af93ba0f226ab38a53bda17eea4fd
#    derivation: paper_programme/gate_x_lesion_recovery/FINAL_CONTRACT_MANIFEST.txt
#                scripts/gate_x_lesion/compute_final_contract_hash.py
#
#    O3_VALIDITY_CONVENTION_SCOPE        = SHARED
#    O3_IMPLEMENTATION_FAILURE_SEMANTICS = ABORT_RUN
#      targeted isolated route must drop >= 0.10 exact-match vs intact in
#      >= 6 of 8 blocks (2 witnesses x 4 lesion seeds), under BOTH
#      CANONICAL and FREE_AR, for ONE shared validity label.
#      Competence failure  -> SCIENTIFIC_PERTURBATION_VALIDITY_FAILURE
#                             (that route x severity only; run continues)
#      Structural failure  -> IMPLEMENTATION_FAILURE (run ABORTS;
#                             no denominator shrinking, no block exclusion,
#                             no classification)
#
#    O4_ROBUSTNESS_RULE =
#      REPLICATED_SIGN_PLUS_MATERIALITY_NO_SIGNIFICANCE_TEST
#      Delta[w,s] = accuracy_native - accuracy_fixed05  (POSITIVE = NATIVE)
#      both witnesses independently; >= 3/4 seeds per witness;
#      |mean over the 4 seeds| >= 0.002 with the same direction;
#      no seed with an opposite-signed |Delta| >= 0.002;
#      NO pooling, NO significance test, NO alpha, NO multiplicity.
#
#    O4 VETO: a diagnostically valid severity that is fully ROBUST in the
#      opposite direction forbids A/B/D -> OUTCOME_F_HETEROGENEOUS.
#
#    CLASSIFICATION: CANONICAL and FREE_AR independently FIRST, then
#      JOINT = CONCORDANT_<X> if equal, else MIXED_DECODING.
#
#  WITNESSES — reconstruction recipes, both component hashes verified
#
#    GXLR_W3_REP  W3_REP  V6_seed19_u3825
#      base  probe_v6_completion_20260912/sources/s19_step_10625850.pt
#            a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c
#      head  archives/prospective_rn_detector_v6_completion_20260912/
#              seed19_u3825/arm_a/head_first_c0.pt
#            8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc
#      RECONSTRUCTED_STATE_SHA256
#            9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1
#
#    GXLR_W4_REP  W4_REP  V6_seed20_u3040
#      base  probe_v6_completion_20260912/sources/s20_step_08445120.pt
#            0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3
#      head  archives/prospective_rn_detector_v6_completion_20260912/
#              seed20_u3040/arm_a/head_first_c0.pt
#            724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03
#      RECONSTRUCTED_STATE_SHA256
#            e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6
#
#    localizer: frozen_head_probe._isolated_model(tr, "p_last_hinge",
#               head["state"]); head["state"] keys == {2.weight, 2.bias}
#
#  DESIGN — nothing here is altered by the final rule freeze
#    routes      wm_encoder_state (dorsal), ltm_encoder_state (ventral)
#    lambdas     0.25, 0.50, 1.00
#    seeds       0, 1, 2, 3
#    batch size  256                                        (F-8, pinned)
#    population  all 29,571 canonical real words  (NO --limit)
#    fusion      NATIVE and FIXED05 — always both, read from one
#                route="full" forward, so they cannot diverge at the
#                intervention
#    decoding    canonical forced-length AR AND genuine free-AR
#                (free-AR is on by default; --no-free-ar is NOT passed)
#    noise       epsilon ~ U(-1,+1), RNG identity exactly
#                (RECONSTRUCTED_STATE_SHA256, route, lesion_seed, item_id);
#                lambda and fusion condition excluded;
#                eta(lambda) = lambda * SD * epsilon, no redraw;
#                per_item_frozen across decoder steps
#    SD          re-measured per state on its own INTACT model by the
#                audited recipe (std, ddof=1, all axes pooled, 2048-item
#                sorted sample at seed 7), frozen before any lesion
#    output      paper_programme/gate_x_lesion_recovery/scientific_execution/
#                reserved, currently ABSENT, disjoint from the quarantine
#                namespace NOT_SCIENTIFIC_RESULT/
#
# =====================================================================

set -eu
cd /Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-gate-x-lesion

python3 scripts/gate_x_lesion/run_gate_x_lesion.py \
    --manifest paper_programme/gate_x_lesion_recovery/checkpoint_manifest.proposed.tsv \
    --state-id ALL \
    --routes wm_encoder_state,ltm_encoder_state \
    --lambdas 0.25,0.5,1.0 \
    --lesion-seeds 0,1,2,3 \
    --batch-size 256 \
    --device cpu \
    --final-contract-hash d5e9eb41adb1574b83ca9a308e976e8f324af93ba0f226ab38a53bda17eea4fd \
    --i-have-central-go

# Post-run completeness gate — MANDATORY before any interpretation.
# Fails closed on any missing shard, section, field, count or pinned value:
#   python3 scripts/gate_x_lesion/validate_output_manifest.py \
#       paper_programme/gate_x_lesion_recovery/scientific_execution/summary_ALL.json

# =====================================================================
#  NOT RUN.  Awaiting CENTRAL final scientific-execution go.
# =====================================================================
