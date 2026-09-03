#!/bin/bash
#SBATCH --job-name=l3_final7l_r1000
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# FINAL-7L OVERNIGHT (EXTENDED) — the SAME from-scratch run, R220 -> R1000.
#
# This is not a new experiment and not a branch.  It is the predeclared
# recipe carried further in the same run directory; R220 was an operational
# inspection gate.  Relative to final7l_continue_r220_to_r700.slurm the ONLY
# difference is the horizon (R1000 instead of R700) and the two extra
# milestones it adds; that older script is left untouched and still works.
#
# NO phase transition, NO moment cloning or reset, NO LR change, NO
# ratio/loss/topology/population change.  The grouped RN|C banks are restored
# bit-exactly from this run's own R220 checkpoint.
#
#     RN bank : repetition (with its 0.5 pool CE) and naming
#     C bank  : comprehension
#     LR      : 1e-4 -- the historical two-stage policy, long past its
#               repetition-cursor boundary of 46,300 (= R100)
#
#   source R220 =   611,160 steps =  440 N exp,  ~697.67 C exp
#   R300        =   833,400 steps =  600 N exp,  ~951.37 C exp
#   R400        = 1,111,200 steps =  800 N exp, ~1268.49 C exp
#   R500        = 1,389,000 steps = 1000 N exp, ~1585.62 C exp
#   R600        = 1,666,800 steps = 1200 N exp, ~1902.74 C exp
#   R700        = 1,944,600 steps = 1400 N exp, ~2219.86 C exp
#   R800        = 2,222,400 steps = 1600 N exp, ~2536.99 C exp
#   R900        = 2,500,200 steps = 1800 N exp, ~2854.11 C exp
#   R1000       = 2,778,000 steps = 2000 N exp, ~3171.23 C exp  <- HARD STOP
#
# 2,166,840 added steps.  At the 0.013427 s/step measured on this V100 for
# the interleaved schedule that is ~8.1 h of pure training; with 78 dev
# evaluations and 8 full-population evaluations expect roughly 9-10 h.  The
# walltime is deliberately generous rather than tight: the job is
# requeue-safe, so a timeout costs only a resubmission, and the true
# throughput of the grouped policy has not been separately measured (see the
# README note in the launch brief for how to read it off the gate's log).
#
# The evaluation and checkpoint cadences are IDENTICAL to the gate and to the
# R700 script.  Nothing here can perturb the training RNG trajectory:
# evaluation runs under preserved_rng and the batch streams are
# counter-addressed.
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
EPOCHS=1000                                   # REPETITION epochs -> 2,778,000 steps
FULL_EVAL_AT=833400,1111200,1389000,1666800,1944600,2222400,2500200,2778000
EVAL_EVERY=27780                              # unchanged dev cadence (10 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"   # SAME run
CKPTS="$RUNS/$RUN_ID/checkpoints"
GATE_STEP=611160
FINAL_STEP=2778000

# The gate's R220 endpoint must exist; this job continues that run, never
# creates one, and never reads the historical FINAL-3 checkpoints.
GATE_CKPT="$CKPTS/step_$(printf '%08d' $GATE_STEP).pt"
[[ -f "$GATE_CKPT" ]] || { echo "FATAL: gate endpoint $GATE_CKPT not found — run final7l_fromscratch_gate_r220.slurm first"; exit 1; }

# Newest by STEP NUMBER (zero-padded names sort numerically), never by mtime,
# so an archive or copy that rewrites timestamps cannot mislead the choice.
OWN_LATEST=$(ls "$CKPTS/"step_*.pt 2>/dev/null | sort | tail -1)
LATEST_STEP=$(basename "$OWN_LATEST" .pt); LATEST_STEP=${LATEST_STEP#step_}
LATEST_STEP=$((10#$LATEST_STEP))

if (( LATEST_STEP >= FINAL_STEP )); then
    echo "== nothing to do: already reached the R1000 hard stop"; exit 0
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

echo "== FINAL-7L complete (hard stop R1000 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== the whole R0->R1000 history is in one run: metrics.tsv is append-only"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final7l/"
