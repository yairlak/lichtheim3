#!/bin/bash
#SBATCH --job-name=l3_final7l_r700
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# FINAL-7L OVERNIGHT — the SAME from-scratch run carried from R220 to R700.
#
# This is not a new experiment and not a branch.  The recipe was predeclared
# before step 0 and runs to R700; R220 was an operational inspection gate, so
# resuming the same run unchanged continues that predeclared trajectory.
#
# NO phase transition, NO moment cloning, NO LR change, NO ratio/loss/topology
# change: the grouped RN|C banks are restored bit-exactly from this run's own
# R220 checkpoint, and the historical FINAL-3 run is never touched.
#
#   source R220 =   611,160 steps = 440 N exp,  ~697.67 C exp
#   R300        =   833,400 steps = 600 N exp,  ~951.37 C exp
#   R400        = 1,111,200 steps = 800 N exp, ~1268.49 C exp
#   R500        = 1,389,000 steps = 1000 N exp, ~1585.62 C exp
#   R600        = 1,666,800 steps = 1200 N exp, ~1902.74 C exp
#   R700        = 1,944,600 steps = 1400 N exp, ~2219.86 C exp  <- HARD STOP
#
# 1,333,440 added steps ~= 4.97 h of pure training at the measured V100
# throughput of 0.013427 s/step, plus five full-population evaluations and
# the dev cadence.  No metric-based stopping; the hard stop is R700.
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

REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
GROUPED=grouped_rn_c_adamw
DEVICE=cuda
EPOCHS=700                                    # REPETITION epochs -> 1,944,600 steps
FULL_EVAL_AT=833400,1111200,1389000,1666800,1944600
EVAL_EVERY=27780                              # unchanged dev cadence (10 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"   # SAME run
CKPTS="$RUNS/$RUN_ID/checkpoints"
GATE_STEP=611160
FINAL_STEP=1944600

# The gate's R220 endpoint must exist; this job continues that run and never
# creates one, and never reads the historical FINAL-3 checkpoints.
GATE_CKPT="$CKPTS/step_$(printf '%08d' $GATE_STEP).pt"
[[ -f "$GATE_CKPT" ]] || { echo "FATAL: gate endpoint $GATE_CKPT not found — run final7l_fromscratch_gate_r220.slurm first"; exit 1; }

OWN_LATEST=$(ls "$CKPTS/"step_*.pt 2>/dev/null | sort | tail -1)
LATEST_STEP=$(basename "$OWN_LATEST" .pt); LATEST_STEP=${LATEST_STEP#step_}
LATEST_STEP=$((10#$LATEST_STEP))

if (( LATEST_STEP >= FINAL_STEP )); then
    echo "== nothing to do: already reached the R700 hard stop"; exit 0
elif (( LATEST_STEP < GATE_STEP )); then
    echo "FATAL: newest checkpoint is step $LATEST_STEP, before the R220 gate"; exit 1
elif (( LATEST_STEP == GATE_STEP )); then
    echo "== overnight start: resuming the gate's own R220 endpoint (step $GATE_STEP)"
else
    echo "== requeue: resuming the overnight run at step $LATEST_STEP"
fi

# No --phase-transition: the checkpoint already carries
# optimizer_policy=grouped_rn_c_adamw and the two-stage LR policy in stage 2,
# so the requested configuration matches it and this is an ordinary resume --
# both banks restored from their own saved states, nothing cloned.
srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" --optimizer-policy "$GROUPED" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 500 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$OWN_LATEST"

echo "== FINAL-7L complete (hard stop R700 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== the whole R0->R700 history is in one run: metrics.tsv is append-only"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final7l/"
