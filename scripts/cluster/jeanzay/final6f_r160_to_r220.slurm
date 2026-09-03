#!/bin/bash
#SBATCH --job-name=l3_final6f
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
# FINAL-6F — SHORT CONTINUATION of FINAL-6P, R160 -> R220.
#
# "6F" is an ANALYSIS LABEL ONLY.  Scientifically this is the exact same
# FINAL-6P regime carried further in the same run: three task-separated AdamW
# banks, all learning rates 1e-4, interleaved 1:2:3, every loss, population,
# seed and architecture unchanged.  NO phase transition, NO cloning from the
# shared R100 parent, NO optimizer reset, NO change of run topology -- the
# three specialised bank states are restored bit-exactly from the run's own
# checkpoint.  No scientific intervention occurs at R160.
#
# Why: through R160 the three-bank regime keeps improving comprehension
# (C top1 .1074 -> .1104 -> .1142 -> .1179, against a matched shared-Adam
# control at ~.0807) while naming stopped declining and ticked up
# (.04180 -> .04170 -> .04237 -> .04264) and dev LTM sat low (~.30-.35 vs
# ~.465 control).  The naming rebound is small and is NOT yet evidence of
# acquisition, but it does mean changing bank topology at R130 would have
# been premature.  This continuation distinguishes:
#     H6-recovery  naming bottomed around R130-R140 and is entering a delayed
#                  acquisition phase under its own AdamW statistics;
#     H6-N-plateau the uptick is local variation and naming stays near ~4%.
# It also asks whether C keeps climbing or saturates near ~12%, and whether
# LTM stabilises, recovers, or keeps deteriorating.
#
#   source R160 = 444,480 steps = 320 N exp, ~507.40 C exp
#   F1     R180 = 500,040 steps = 360 N exp, ~570.82 C exp
#   F2     R200 = 555,600 steps = 400 N exp, ~634.25 C exp
#   F3     R220 = 611,160 steps = 440 N exp, ~697.67 C exp   <- HARD STOP
#
# No metric-based automatic stopping: the hard stop is R220 and every
# milestone is inspected afterwards.  Bank grouping (RN|C or otherwise) is
# NOT changed here.
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

# ---- FINAL-6F configuration (identical recipe, longer budget) -------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
OPT_POLICY=task_separated_adamw
DEVICE=cuda
EPOCHS=220                                    # REPETITION epochs -> 611,160 steps
FULL_EVAL_AT=500040,555600,611160             # F1 / F2 / F3
EVAL_EVERY=13890                              # unchanged dev cadence (5 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence (10 R exp)

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
RUN_ID="final6p_r100_sepmoments_seed${SEED}_${SUBSET_MODE}"   # SAME run
CKPTS="$RUNS/$RUN_ID/checkpoints"
R160_STEP=444480
FINAL_STEP=611160

# The FINAL-6E endpoint must exist and is kept, never overwritten.
R160_CKPT="$CKPTS/step_$(printf '%08d' $R160_STEP).pt"
[[ -f "$R160_CKPT" ]] || { echo "FATAL: R160 endpoint $R160_CKPT not found — this job continues that run, it does not create one"; exit 1; }

# Newest by STEP NUMBER (zero-padded names sort numerically), not by mtime,
# so an archive or copy that rewrites timestamps cannot mislead the choice.
OWN_LATEST=$(ls "$CKPTS/"step_*.pt 2>/dev/null | sort | tail -1)
LATEST_STEP=$(basename "$OWN_LATEST" .pt); LATEST_STEP=${LATEST_STEP#step_}
LATEST_STEP=$((10#$LATEST_STEP))

if (( LATEST_STEP >= FINAL_STEP )); then
    echo "== nothing to do: FINAL-6F already reached the R220 hard stop"; exit 0
elif (( LATEST_STEP < R160_STEP )); then
    echo "FATAL: newest checkpoint is step $LATEST_STEP, before the R160 endpoint"; exit 1
elif (( LATEST_STEP == R160_STEP )); then
    echo "== first extension launch: resuming the run's own R160 endpoint (step $R160_STEP)"
else
    echo "== requeue: resuming FINAL-6F at step $LATEST_STEP"
fi

# No --phase-transition: the checkpoint already carries
# optimizer_policy=task_separated_adamw and the two-stage LR policy (already
# in stage 2, so every task runs at 1e-4), so the requested configuration
# matches it and the resume is an ordinary one -- the three banks are restored
# from their own saved states and nothing is cloned.
srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" \
    --optimizer-policy "$OPT_POLICY" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$OWN_LATEST"

echo "== FINAL-6F complete (hard stop R220 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== continuation record: $RUNS/$RUN_ID/provenance_from_step_00444480.json"
echo "== R100->R160 history is retained: metrics.tsv and losses.tsv are append-only"
echo "== compare against the matched shared-Adam control:"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID \\"
echo "       --baseline \$SCRATCH/lichtheim3_runs/final3p_i123_seed${SEED}_${SUBSET_MODE}"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final6f/"
