#!/bin/sh
# =====================================================================
#  GATE x LESION / RECOVERY — PREPARED SCIENTIFIC EXECUTION COMMAND
#
#                          *** NOT RUN ***
#
#  GO_FOR_SCIENTIFIC_EXECUTION     = NO
#  AWAITING_CENTRAL_FINAL_GO_NO_GO = YES
#
#  This file is a record of the exact command CENTRAL would issue after
#  final approval.  It is deliberately NOT executable (no +x bit) and has
#  never been executed.  Running it performs the full canonical scientific
#  lesion experiment, which is forbidden until CENTRAL gives go.
#
#  The runner independently refuses to proceed: non-zero severity on the
#  full canonical population hard-stops unless --i-have-central-go is
#  passed, which CENTRAL alone supplies.
# =====================================================================
#
#  What the command encodes, all frozen:
#
#    two exact states     W3_REP (V6_seed19_u3825), W4_REP (V6_seed20_u3040)
#                         resolved from checkpoint_manifest.proposed.tsv as
#                         RECONSTRUCTION RECIPES (base + applied head), with
#                         both component SHA256 verified before use:
#                           W3 base a5f21de9…aad76c  head 8865ba95…7afcfc
#                           W4 base 0657f410…dc79f3  head 724ed4c6…b97203
#                         composite state_sha256:
#                           W3 9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1
#                           W4 e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6
#    population           all 29,571 canonical real words — NO --limit, which
#                         the runner refuses outside quarantine anyway
#    routes               wm_encoder_state (dorsal), ltm_encoder_state (ventral)
#    lambdas              0.25, 0.50, 1.00   (nothing added, nothing dropped)
#    lesion seeds         0, 1, 2, 3
#    fusion               NATIVE and FIXED05 — always both; they are read from
#                         one route="full" forward, so they cannot diverge at
#                         the intervention
#    decoding             canonical forced-length AR AND genuine free-AR —
#                         free-AR is on by default; --no-free-ar is NOT passed
#    base-noise semantics epsilon ~ U(-1,+1), RNG identity exactly
#                         (state_sha256, route, lesion_seed, item_id);
#                         lambda and fusion condition excluded;
#                         eta(lambda) = lambda * SD * epsilon, no redraw;
#                         per_item_frozen across decoder steps
#    SD                   re-measured per state on its own INTACT model by the
#                         audited historical recipe (std, ddof=1, all axes
#                         pooled, 2048-item sorted sample at seed 7), frozen
#                         before any lesion
#    output namespace     paper_programme/gate_x_lesion_recovery/scientific_execution/
#                         — reserved and empty; disjoint from the quarantine
#                         namespace NOT_SCIENTIFIC_RESULT/
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
    --i-have-central-go

# =====================================================================
#  NOT RUN.  Awaiting CENTRAL final go/no-go.
# =====================================================================
