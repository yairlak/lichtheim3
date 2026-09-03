#!/bin/bash
#SBATCH --job-name=l3_f3p_r150
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
# FINAL-3P CONTINUATION — the SAME trajectory carried from R110 to R150.
#
# Scientific question: does the post-LR-transition improvement seen between
# M2 and M3 persist for another 40 R exposures, or was M3 only a short
# consolidation transient?
#
# This is NOT a new experiment and NOT a restart.  It resumes
# step_00305580.pt in place, into the SAME run directory, with every
# scientific field unchanged and validated against the checkpoint by
# load_state_dict: regime, seed, populations and hashes, schedule
# (interleaved_123), schedule seed and ratio, c_align_weight 0, stream seeds,
# and lr_boundary_steps (restored FROM the checkpoint, so the LR convention
# cannot drift).  Model weights, AdamW moments, task cursors, macro-cycle
# position and all RNG states carry over untouched, so the continued
# trajectory is bitwise identical to an uninterrupted run of the same recipe.
#
# The only fields that differ are run-control, which the resume guard
# deliberately does not police: the final budget and the FUTURE milestones.
#
#   already done   M1  R50  = 138,900 steps
#                  M2  R100 = 277,800 steps   (LR boundary)
#                  M3  R110 = 305,580 steps   <- continuation starts here
#   this job       M4  R130 = 361,140 steps = 260 N exp, ~412.26 C exp
#                  M5  R150 = 416,700 steps = 300 N exp, ~475.68 C exp
#
# HARD STOP at 416,700 (R150).  No automatic continuation beyond it.
# LR is 1e-4 throughout (the boundary was crossed at R100).
#
# Preemption-safe: on requeue the latest checkpoint may be anywhere in
# (305,580, 416,700]; the guard below accepts that and resumes it normally.
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

# ---- continuation configuration (recipe identical to FINAL-3P) ------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
DEVICE=cuda
EPOCHS=150                                    # REPETITION epochs -> 416,700 steps
FULL_EVAL_AT=361140,416700                    # M4 (R130) and M5 (R150)
EVAL_EVERY=13890                              # unchanged dev cadence (5 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence (10 R exp)
RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
CKPTS="$RUNS/$RUN_ID/checkpoints"
M3_STEP=305580
M5_STEP=416700

# ---- the continuation must extend THIS trajectory ------------------------
M3_CKPT="$CKPTS/step_$(printf '%08d' $M3_STEP).pt"
[[ -f "$M3_CKPT" ]] || { echo "FATAL: M3 checkpoint $M3_CKPT not found — this job continues the FINAL-3P run, it does not start one"; exit 1; }

LATEST=$(ls -t "$CKPTS/"step_*.pt 2>/dev/null | head -1 || true)
[[ -n "${LATEST:-}" ]] || { echo "FATAL: no checkpoint to resume"; exit 1; }
LATEST_STEP=$(basename "$LATEST" .pt); LATEST_STEP=${LATEST_STEP#step_}
LATEST_STEP=$((10#$LATEST_STEP))

# First launch resumes exactly M3; a requeued launch resumes wherever it got
# to.  Anything before M3 would be a different (earlier) trajectory point and
# is refused rather than silently re-running finished work.
if (( LATEST_STEP < M3_STEP )); then
    echo "FATAL: latest checkpoint is step $LATEST_STEP, before M3 ($M3_STEP)"; exit 1
elif (( LATEST_STEP == M3_STEP )); then
    echo "== first continuation launch: resuming M3 at step $M3_STEP"
elif (( LATEST_STEP < M5_STEP )); then
    echo "== continuation after preemption: resuming at step $LATEST_STEP"
else
    echo "== nothing to do: latest step $LATEST_STEP has reached the R150 hard stop"; exit 0
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$LATEST"

echo "== FINAL-3P continuation complete (hard stop 150 R exposures): $RUNS/$RUN_ID"
echo "== M1/M2/M3 history is preserved: metrics.tsv and losses.tsv are append-only,"
echo "   and this launch wrote config_from_step_*.json / provenance_from_step_*.json"
echo "== analyse with:"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID --baseline \$STORE/lichtheim3_final1/final1_j0_seed22_final_full"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final3p_r150/"
