#!/bin/bash
#SBATCH --job-name=l3_final2a
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
# Lichtheim3 FINAL-2A — comprehension-stream semantic alignment.
#
# ONE intervention against the archived FINAL-1 baseline:
#
#     C contribution:  0.087 * retrieval_CE            (FINAL-1)
#                   -> 0.087 * retrieval_CE + 1.0 * alignment_loss(s_hat_C,
#                                                                  GloVe_C)
#
# using the canonical losses.alignment_loss the R stream already uses, on the
# s_hat already computed for retrieval.  The alignment term is weighted by
# c_align_weight ALONE, never by lambda_C.  EVERYTHING else is bit-frozen to
# FINAL-1: seed 22, summed j0, final_full populations (R 29,571 / N 29,571 /
# C 27,981 canonical / bank 29,571), lambda_C 0.087, lambda_N 1.0, tau 0.10,
# historical R total_loss + dorsal pool, AdamW, weight decay, grad clip 1.0,
# LR 1e-3 -> 1e-4 at step 46,300, batch 64, the same counter-addressed
# streams and seeds, 700 N-exposures = 324,100 steps, milestones at
# 138,900 / 231,500 / 324,100, same eval/checkpoint cadence.
#
# Scientific question: does giving the comprehension stream its own direct
# semantic-target alignment fix the full-scale ventral semantic optimization
# problem observed in FINAL-1?
#
# Baseline for every comparison is the completed run
# final1_j0_seed22_final_full, which is exactly paired (same seed, same batch
# sequences, same milestones).  This job writes to a DISTINCT run id and can
# never overwrite it.
#
# Contract: export L3_EXPECTED_COMMIT=<sha> before sbatch.  The job refuses to
# start on branch/commit mismatch or a dirty TRACKED tree; the driver asserts
# the populations and the frozen canonical-C hash, and requires real GloVe.
#
# Resume: checkpoints carry model, optimizer, stream cursors, all RNG states
# and c_align_weight; this script auto-resumes from the latest checkpoint, so
# resubmitting after walltime/preemption continues EXACTLY.  A checkpoint
# trained under a different C objective is refused, so FINAL-1 and FINAL-2A
# can never be spliced together.
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
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing (bash data/get_glove.sh)"; exit 1; }
echo "== contract OK: $BRANCH @ $HEAD"

# ---- FINAL-2A configuration (frozen; the ONLY delta vs FINAL-1 is C_ALIGN) --
REGIME=j0
SEED=22
SUBSET_MODE=final_full
DEVICE=cuda
EPOCHS=700                                    # 700 x 463 = 324,100 optimizer steps
C_ALIGN=1.0                                   # <-- the single intervention
FULL_EVAL_AT=138900,231500,324100             # 300 / 500 / 700 N-exposures
EVAL_EVERY=2315                               # dev-sample cadence (5 epochs)
SAVE_EVERY=4630                               # safety checkpoints (10 epochs)
RUN_ID="final2a_calign_seed${SEED}_${SUBSET_MODE}"
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}

# Guard: never write into the FINAL-1 run directory.
[[ "$RUN_ID" != final1_* ]] || { echo "FATAL: run id collides with FINAL-1"; exit 1; }

# ---- automatic exact resume from the latest checkpoint --------------------
RESUME_ARGS=()
LATEST=$(ls -t "$RUNS/$RUN_ID/checkpoints/"step_*.pt 2>/dev/null | head -1 || true)
if [[ -n "${LATEST:-}" ]]; then
    echo "== resuming exactly from $LATEST"
    RESUME_ARGS=(--resume "$LATEST")
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --c-align-weight "$C_ALIGN" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 100 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    "${RESUME_ARGS[@]}"

echo "== run complete: $RUNS/$RUN_ID"
echo "== archive with: cp -r $RUNS/$RUN_ID $STORE/lichtheim3_final2a/"
