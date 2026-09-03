#!/bin/bash
#SBATCH --job-name=l3_final6e
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
# FINAL-6E — SHORT CONTINUATION of FINAL-6P, R130 -> R160.
#
# "6E" is an ANALYSIS LABEL ONLY.  Scientifically this is the exact same
# FINAL-6P regime carried further in the same run: three task-separated AdamW
# banks, all learning rates 1e-4, interleaved 1:2:3, every loss, population,
# seed and architecture unchanged.  There is NO phase transition, NO cloning
# from the R100 parent, NO optimizer reset and NO change of run topology --
# the three specialised bank states are restored bit-exactly from FINAL-6P's
# own checkpoint.
#
# Why: at R130 FINAL-6P showed the program's first genuine regime change on
# comprehension (C top1 .0901 -> .1012 -> .1074 against a matched shared-Adam
# control at .0767) while naming declined (.0434 -> .0424 -> .0418) and dev
# LTM fell to .346.  Those trajectories are only 60 N-exposures and ~95 C
# exposures long.  This continuation exists to separate:
#     H6a naming decline is transient and reverses;
#     H6b naming is persistently impaired by an N-specific bank;
#     H6c C's regime change saturates quickly;
#     H6d the independent R bank stabilises or rebuilds LTM repetition.
# Bank grouping (RN|C or otherwise) is NOT changed until these are answered.
#
#   source R130 = 361,140 steps = 260 N exp, ~412.26 C exp
#   E1     R140 = 388,920 steps = 280 N exp, ~443.97 C exp
#   E2     R150 = 416,700 steps = 300 N exp, ~475.68 C exp
#   E3     R160 = 444,480 steps = 320 N exp, ~507.40 C exp   <- HARD STOP
#
# No metric-based automatic stopping: the hard stop is R160 and every
# milestone is inspected afterwards.
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

# ---- FINAL-6E configuration (identical recipe, longer budget) -------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
OPT_POLICY=task_separated_adamw
DEVICE=cuda
EPOCHS=160                                    # REPETITION epochs -> 444,480 steps
FULL_EVAL_AT=388920,416700,444480             # E1 / E2 / E3
EVAL_EVERY=13890                              # unchanged dev cadence (5 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence (10 R exp)

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
RUN_ID="final6p_r100_sepmoments_seed${SEED}_${SUBSET_MODE}"   # SAME run
PARENT_RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"       # never resumed here
CKPTS="$RUNS/$RUN_ID/checkpoints"
R130_STEP=361140
FINAL_STEP=444480

# The FINAL-6P endpoint must exist and is kept, never overwritten.
R130_CKPT="$CKPTS/step_$(printf '%08d' $R130_STEP).pt"
[[ -f "$R130_CKPT" ]] || { echo "FATAL: FINAL-6P endpoint $R130_CKPT not found — this job continues that run, it does not create one"; exit 1; }

# Newest by STEP NUMBER (zero-padded names sort numerically), not by mtime,
# so an archive/copy that touches timestamps cannot mislead the choice.
OWN_LATEST=$(ls "$CKPTS/"step_*.pt 2>/dev/null | sort | tail -1)
LATEST_STEP=$(basename "$OWN_LATEST" .pt); LATEST_STEP=${LATEST_STEP#step_}
LATEST_STEP=$((10#$LATEST_STEP))

if (( LATEST_STEP >= FINAL_STEP )); then
    echo "== nothing to do: FINAL-6E already reached the R160 hard stop"; exit 0
elif (( LATEST_STEP < R130_STEP )); then
    echo "FATAL: newest checkpoint is step $LATEST_STEP, before the FINAL-6P R130 endpoint"; exit 1
elif (( LATEST_STEP == R130_STEP )); then
    echo "== first extension launch: resuming FINAL-6P's own R130 endpoint (step $R130_STEP)"
else
    echo "== requeue: resuming FINAL-6E at step $LATEST_STEP"
fi

# No --phase-transition: the checkpoint already carries
# optimizer_policy=task_separated_adamw and the two-stage LR policy, so the
# requested configuration matches it and the resume is an ordinary one --
# three banks restored from their own saved states, nothing cloned.
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

echo "== FINAL-6E complete (hard stop R160 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== continuation record: $RUNS/$RUN_ID/provenance_from_step_00361140.json"
echo "== R100->R130 history is retained: metrics.tsv and losses.tsv are append-only"
echo "== compare against the matched shared-Adam control:"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID --baseline $RUNS/$PARENT_RUN_ID"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final6e/"
