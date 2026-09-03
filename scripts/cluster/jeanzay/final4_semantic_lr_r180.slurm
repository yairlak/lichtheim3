#!/bin/bash
#SBATCH --job-name=l3_final4
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
# FINAL-4 — task-specific semantic learning rates, phase 2 of a staged
# from-scratch recipe.
#
#   Phase 1 (frozen history): random init -> R150 under the FINAL-3 recipe,
#            ending at step 416,700 of final3p_i123_seed22_final_full.
#   Phase 2 (this job): from step 416,700 onward, one fixed learning rate per
#            task on that task's own optimizer step:
#                eta_R = 1e-4   eta_N = 3e-4   eta_C = 3e-4
#
# This is NOT an exact continuation: the LR policy changes, so the driver
# refuses the resume unless --phase-transition is declared, and it records the
# transition (step, old policy, new policy, source checkpoint and commit, new
# commit, exposures, timestamp) in the checkpoint and provenance.
#
# Why LR and not lambda_C: the offline actual-AdamW-update audit at this exact
# checkpoint measured one-step ||delta theta|| of R 0.01312, N 0.00884,
# C 0.00787 -- C is NOT update-starved despite its ~8x smaller raw gradient,
# because Adam's preconditioner cancels the static 0.087 scaling.  Raising
# lambda_C to 1.0 moved the actual update only x1.41 and saturated clipping,
# whereas LR scales the applied update exactly (x3.0000 measured, C stays
# unclipped).  So lambda_C stays 0.087 and only the LR policy changes.
#
# EVERYTHING else is unchanged and still guarded: seed 22, interleaved 1:2:3,
# deterministic shuffled macro-cycle, full populations and hashes,
# lambda_C 0.087, lambda_N 1.0, c_align_weight 0, tau 0.10, R objective with
# the pool riding every R step at 0.5, ONE shared AdamW with its moments
# carried over untouched, grad clip 1.0, batch 64, data cursors, RNG state,
# architecture, evaluation definitions.
#
#   source  R150 = 416,700 steps = 300 N exp, ~475.68 C exp
#   M6      R160 = 444,480 steps = 320 N exp, ~507.40 C exp
#   M7      R170 = 472,260 steps = 340 N exp, ~539.11 C exp
#   M8      R180 = 500,040 steps = 360 N exp, ~570.82 C exp   <- HARD STOP
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

# ---- FINAL-4 configuration ------------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
DEVICE=cuda
EPOCHS=180                                    # REPETITION epochs -> 500,040 steps
ETA_R=1e-4
ETA_N=3e-4
ETA_C=3e-4
FULL_EVAL_AT=444480,472260,500040             # M6 / M7 / M8
EVAL_EVERY=13890                              # unchanged dev cadence (5 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence (10 R exp)
RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"   # SAME trajectory/run dir
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
CKPTS="$RUNS/$RUN_ID/checkpoints"
SOURCE_STEP=416700
FINAL_STEP=500040

# ---- this job continues the R150 trajectory; it never starts one ----------
SRC_CKPT="$CKPTS/step_$(printf '%08d' $SOURCE_STEP).pt"
[[ -f "$SRC_CKPT" ]] || { echo "FATAL: source checkpoint $SRC_CKPT not found"; exit 1; }

LATEST=$(ls -t "$CKPTS/"step_*.pt 2>/dev/null | head -1 || true)
LATEST_STEP=$(basename "$LATEST" .pt); LATEST_STEP=${LATEST_STEP#step_}
LATEST_STEP=$((10#$LATEST_STEP))

# The phase transition is declared EXACTLY ONCE, on the first launch.  After
# it, the checkpoint already carries the task-specific policy, so a requeue
# resumes normally and must NOT re-declare anything.
PHASE_ARGS=()
if (( LATEST_STEP < SOURCE_STEP )); then
    echo "FATAL: latest checkpoint step $LATEST_STEP precedes the R150 source"; exit 1
elif (( LATEST_STEP == SOURCE_STEP )); then
    echo "== first FINAL-4 launch: declaring the phase transition at step $SOURCE_STEP"
    PHASE_ARGS=(--phase-transition)
elif (( LATEST_STEP < FINAL_STEP )); then
    echo "== requeue: resuming FINAL-4 at step $LATEST_STEP (no new declaration)"
else
    echo "== nothing to do: latest step $LATEST_STEP has reached the R180 hard stop"; exit 0
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" \
    --lr-repetition "$ETA_R" --lr-naming "$ETA_N" --lr-comprehension "$ETA_C" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$LATEST" "${PHASE_ARGS[@]}"

echo "== FINAL-4 complete (hard stop R180 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== phase record: $RUNS/$RUN_ID/provenance_from_step_00416700.json"
echo "== analyse with:"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID --baseline \$STORE/lichtheim3_final1/final1_j0_seed22_final_full"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final4/"
