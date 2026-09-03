#!/bin/bash
#SBATCH --job-name=l3_final7p
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# FINAL-7P — grouped RN | C optimizer banks.
#
#   optimizer_RN  steps on repetition (with its attached 0.5 pool CE) AND on
#                 naming
#   optimizer_C   steps on comprehension
#
# ONE shared network; both banks reference the SAME Parameter objects and own
# independent exp_avg / exp_avg_sq / per-parameter step counters.  Optimizer
# grouping is the ONLY intervention.
#
# Motivation (a hypothesis this experiment FALSIFIES, not an established
# cause): parameter overlap is asymmetric.  Comprehension trains the
# phonological encoder / semantic pathway and never trains the LTM decoder
# core; naming trains the semantic -> decoder/output pathway; repetition spans
# both.  Under the historical shared optimizer the decoder-side statistics
# were therefore largely an R+N history.  FINAL-6P separated R from N there,
# and naming (~.042 flat from R130 to R220) and ventral LTM repetition
# (.346 -> .245) both degraded, while separating C from R on the encoder side
# produced a durable C gain (.107 -> .135 by R220 against a shared control at
# ~.077).  FINAL-7P tests that dissociation directly.
#
# MATCHED OPTIMIZER-TOPOLOGY FAMILY, all from the same R100 checkpoint:
#     control    one shared bank            R+N+C
#     FINAL-6P   three banks                R | N | C
#     FINAL-7P   two banks                  RN | C
#
# Matched references at R130:
#     control   naming .054276  C .076695  dev LTM ~.55   full rep .999865
#     FINAL-6P  naming .041798  C .107430  dev LTM .3461  full rep .999966
# Prediction if the functional-stream account is right: C retains most of
# FINAL-6P's gain while naming and LTM move back toward the control.
#
# MOMENT INITIALISATION: at the declared transition BOTH banks are loaded with
# an independent deep copy of the inherited shared state -- moments and step
# counters preserved, nothing zeroed, no aliasing between banks.
#
# LEARNING RATES: the LR policy is NOT touched.  After R100 the historical
# two-stage schedule already yields 1e-4 for every task, so no --lr-* flag is
# passed and optimizer grouping is the only difference.
#
#   source R100 = 277,800 steps = 200 N exp, ~317.12 C exp
#   G1     R110 = 305,580 steps = 220 N exp, ~348.84 C exp
#   G2     R120 = 333,360 steps = 240 N exp, ~380.55 C exp
#   G3     R130 = 361,140 steps = 260 N exp, ~412.26 C exp   <- HARD STOP
#
# Everything else frozen: interleaved 1:2:3, deterministic macro-cycle,
# seed 22, stream seeds, lambda_C 0.087, lambda_N 1.0, c_align_weight 0,
# tau 0.10, all three objectives, pool CE at 0.5 on R steps, clip 1.0,
# wd 1e-5, batch 64, full populations, architecture, all parameters
# trainable, evaluation definitions, cursors and RNG.
#
# Contract: export L3_EXPECTED_COMMIT=<sha> before sbatch.
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-$WORK/lichtheim3/lichtheim3}
EXPECTED_BRANCH=${L3_EXPECTED_BRANCH:-feat/joint-multitask-scratch}
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || { echo "FATAL: export L3_EXPECTED_COMMIT"; exit 1; }
cd "$REPO"

BRANCH=$(git branch --show-current); HEAD=$(git rev-parse HEAD)
[[ "$BRANCH" == "$EXPECTED_BRANCH" ]] || { echo "FATAL: branch $BRANCH != $EXPECTED_BRANCH"; exit 1; }
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD $HEAD != $L3_EXPECTED_COMMIT"; exit 1; }
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked working tree dirty"; exit 1; }
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }
echo "== contract OK: $BRANCH @ $HEAD"

# ---- FINAL-7P configuration ----------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
OPT_POLICY=grouped_rn_c_adamw
DEVICE=cuda
EPOCHS=130                                    # REPETITION epochs -> 361,140 steps
FULL_EVAL_AT=305580,333360,361140             # G1 / G2 / G3
EVAL_EVERY=13890                              # unchanged dev cadence (5 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence (10 R exp)

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
PARENT_RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"
RUN_ID="final7p_r100_rn_c_banks_seed${SEED}_${SUBSET_MODE}"
PARENT_STEP=277800
FINAL_STEP=361140

PARENT_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $PARENT_STEP).pt"
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }
[[ -f "$PARENT_CKPT" ]] || { echo "FATAL: parent checkpoint $PARENT_CKPT not found — FINAL-7P branches from it and never starts a fresh run"; exit 1; }

# ---- first launch vs requeue ---------------------------------------------
# First launch : resume the EXTERNAL shared-AdamW parent at R100 and declare
#                shared_adamw -> grouped_rn_c_adamw (which clones the shared
#                state into both banks).
# Requeue      : resume FINAL-7P's OWN newest checkpoint, which already holds
#                two distinct banks -- no re-declaration, no re-cloning.
# Newest by STEP NUMBER (zero-padded names sort numerically), not by mtime.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
PHASE_ARGS=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$PARENT_CKPT"
    PHASE_ARGS=(--phase-transition)
    echo "== first FINAL-7P launch: branching from $PARENT_RUN_ID at step $PARENT_STEP"
    echo "== declaring shared_adamw -> grouped_rn_c_adamw (clone into RN and C banks)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=${OWN_STEP#step_}
    OWN_STEP=$((10#$OWN_STEP))
    if (( OWN_STEP >= FINAL_STEP )); then
        echo "== nothing to do: FINAL-7P already reached the R130 hard stop"; exit 0
    fi
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming FINAL-7P's own two-bank checkpoint at step $OWN_STEP"
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" \
    --optimizer-policy "$OPT_POLICY" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$RESUME_FROM" "${PHASE_ARGS[@]}"

echo "== FINAL-7P complete (hard stop R130 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== ancestry + phase record: $RUNS/$RUN_ID/provenance.json"
echo "== note: m_div_RN is 0 by construction here (R and N share the RN bank);"
echo "         m_div_RC and m_div_NC both measure RN-vs-C divergence"
echo "== matched comparisons:"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID --baseline $RUNS/$PARENT_RUN_ID      # shared control"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID \\"
echo "       --baseline $RUNS/final6p_r100_sepmoments_seed${SEED}_${SUBSET_MODE}   # 3-bank"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final7p/"
