#!/bin/bash
#SBATCH --job-name=l3_final3p
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# FINAL-3P — Ueno-style interleaved 1:2:3 SHORT pilot, from scratch.
#
# Scientific question: does a 1:2:3 interleaved update schedule produce a
# qualitatively better full-lexicon acquisition regime for naming and
# comprehension while preserving repetition, making a path to 100/100/100
# plausible?
#
# The ONLY change vs FINAL-1 is the task schedule.  Frozen: seed 22,
# final_full populations (R 29,571 / N 29,571 / C 27,981 canonical /
# bank 29,571), lambda_C 0.087, lambda_N 1.0, tau 0.10, c_align_weight 0
# (FINAL-2A is closed), historical R total_loss + dorsal pool riding every R
# step, AdamW + weight decay, grad clip 1.0 per task step, batch 64, the same
# counter-addressed streams and seeds, architecture.
#
# Each optimizer step trains ONE task (zero_grad -> backward -> clip -> step)
# on the SHARED AdamW state; every six-step macro-cycle holds exactly 1 R,
# 2 N and 3 C, deterministically shuffled from (schedule seed, cycle index).
#
# LR follows REPETITION progress: 1e-3 while completed R batches < 46,300
# (= 100 R exposures), then 1e-4.  M2 sits exactly ON that boundary, and
# FINAL-1 showed substantial immediate consolidation just after it (LTM
# repetition .736 -> .868 within five exposures), so stopping there would risk
# a false negative.  The pilot therefore runs 10 further R exposures UNDER
# LR 1e-4 -- 27,780 extra steps, ~6 minutes at the measured V100 throughput.
#
# HARD PILOT ENDPOINT: 110 R exposures = 50,930 macro-cycles = 305,580
# optimizer steps.  There is NO automatic continuation.  If the result is a
# GO we resume this exact checkpoint; we never restart.
#
#   M1   50 R exp = 138,900 steps = 100 N exp, ~158.56 C exp   (stage 1)
#   M2  100 R exp = 277,800 steps = 200 N exp, ~317.12 C exp   (LR boundary)
#   M3  110 R exp = 305,580 steps = 220 N exp, ~348.84 C exp   (post-boundary)
#
# Contract: export L3_EXPECTED_COMMIT=<sha> before sbatch.  Resume refuses a
# checkpoint from a different schedule, population or C objective.
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

# ---- FINAL-3P configuration ----------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
DEVICE=cuda
EPOCHS=110                                    # REPETITION epochs -> 305,580 steps
FULL_EVAL_AT=138900,277800,305580             # M1 (50 R), M2 (100 R), M3 (110 R)
EVAL_EVERY=13890                              # dev cadence: every 5 R exposures
SAVE_EVERY=27780                              # safety checkpoints: every 10 R exp
RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}

[[ "$RUN_ID" != final1_* && "$RUN_ID" != final2a_* ]] || { echo "FATAL: run id collision"; exit 1; }

# ---- automatic exact resume (mid-cycle safe) ------------------------------
RESUME_ARGS=()
LATEST=$(ls -t "$RUNS/$RUN_ID/checkpoints/"step_*.pt 2>/dev/null | head -1 || true)
if [[ -n "${LATEST:-}" ]]; then
    echo "== resuming exactly from $LATEST"
    RESUME_ARGS=(--resume "$LATEST")
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    "${RESUME_ARGS[@]}"

echo "== FINAL-3P complete (hard endpoint 110 R exposures): $RUNS/$RUN_ID"
echo "== analyse with:"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID --baseline \$STORE/lichtheim3_final1/final1_j0_seed22_final_full"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final3p/"
