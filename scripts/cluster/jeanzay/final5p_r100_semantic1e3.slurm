#!/bin/bash
#SBATCH --job-name=l3_final5p
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
# FINAL-5P — who receives the R100 learning-rate drop?
#
# The historical rule dropped every task from 1e-3 to 1e-4 at R100, but that
# boundary was motivated by REPETITION consolidation.  FINAL-5P changes only
# who it applies to: R enters maintenance at 1e-4 while N and C KEEP the
# acquisition rate they already had immediately before the boundary.
#
#   Phase 1 (frozen history): random init -> R100 under the FINAL-3 recipe,
#            eta_R = eta_N = eta_C = 1e-3, ending at step 277,800 of
#            final3p_i123_seed22_final_full.
#   Phase 2 (this job): from step 277,800 onward,
#            eta_R = 1e-4, eta_N = 1e-3, eta_C = 1e-3.
#
# This is NOT the FINAL-4 manipulation.  FINAL-4 raised N/C only at R150,
# after they had already spent 50 R-exposures at the low rate and flattened,
# and it failed (naming net zero R150->R180, C +0.74pt).  Here the semantic
# rates are never lowered in the first place.
#
# Matched-state control, already measured on the SAME R100 checkpoint with
# every task at 1e-4 (the FINAL-3 branch):
#     R110  rep .999763  naming .048696  C top1 .072228
#     R130  rep .999865  naming .054276  C top1 .076695
# Same model, optimizer state, cursors, RNG and deterministic future data
# streams; the ONLY difference is who keeps 1e-3 after R100.
#
# NEW RUN DIRECTORY.  This branch must not be appended to the parent run,
# which already contains the R100->R180 history: it writes its own
# metrics.tsv, losses.tsv and checkpoints, and records its ancestry
# (parent run id, checkpoint, step, commit, inherited state) in provenance.
# The parent is only ever READ.
#
#   source R100 = 277,800 steps = 200 N exp, ~317.12 C exp
#   M5P-1  R110 = 305,580 steps = 220 N exp, ~348.84 C exp
#   M5P-2  R120 = 333,360 steps = 240 N exp, ~380.55 C exp
#   M5P-3  R130 = 361,140 steps = 260 N exp, ~412.26 C exp   <- HARD STOP
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

# ---- FINAL-5P configuration ----------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
DEVICE=cuda
EPOCHS=130                                    # REPETITION epochs -> 361,140 steps
ETA_R=1e-4                                    # repetition enters maintenance
ETA_N=1e-3                                    # naming stays in acquisition
ETA_C=1e-3                                    # comprehension stays in acquisition
FULL_EVAL_AT=305580,333360,361140             # M5P-1 / M5P-2 / M5P-3
EVAL_EVERY=13890                              # unchanged dev cadence (5 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence (10 R exp)

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
PARENT_RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"
RUN_ID="final5p_r100_semantic1e3_seed${SEED}_${SUBSET_MODE}"
PARENT_STEP=277800
FINAL_STEP=361140

PARENT_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $PARENT_STEP).pt"
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

# The branch must never be confused with the parent run.
[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }
# Never silently initialise from scratch: the R100 parent must exist.
[[ -f "$PARENT_CKPT" ]] || { echo "FATAL: parent checkpoint $PARENT_CKPT not found — FINAL-5P branches from it and never starts a fresh run"; exit 1; }

# ---- first launch vs requeue ---------------------------------------------
# First launch  : resume the EXTERNAL parent at R100 and declare the phase
#                 transition (two_stage -> task_specific).
# Requeue       : resume FINAL-5P's OWN newest checkpoint, no declaration.
# Never resume the parent's later (R110..R180) checkpoints: only the exact
# R100 file above is ever used as the branch point.
OWN_LATEST=$(ls -t "$OWN_CKPTS/"step_*.pt 2>/dev/null | head -1 || true)
PHASE_ARGS=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$PARENT_CKPT"
    PHASE_ARGS=(--phase-transition)
    echo "== first FINAL-5P launch: branching from $PARENT_RUN_ID at step $PARENT_STEP"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=${OWN_STEP#step_}
    OWN_STEP=$((10#$OWN_STEP))
    if (( OWN_STEP >= FINAL_STEP )); then
        echo "== nothing to do: FINAL-5P already reached the R130 hard stop"; exit 0
    fi
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming FINAL-5P's own checkpoint at step $OWN_STEP"
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
    --resume "$RESUME_FROM" "${PHASE_ARGS[@]}"

echo "== FINAL-5P complete (hard stop R130 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== ancestry + phase record: $RUNS/$RUN_ID/provenance.json"
echo "== compare against the matched OLD branch (all tasks at 1e-4 after R100):"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID --baseline $RUNS/$PARENT_RUN_ID"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final5p/"
