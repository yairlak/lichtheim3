#!/bin/bash
#SBATCH --job-name=l3_final1
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
# Lichtheim3 FINAL-1 scientific run — ONE continuous, exactly-resumable job.
#
# Scientific question: does summed from-scratch J0 with the FINAL populations
# (R=29,571; N=29,571; C=canonical 27,981; bank=29,571) put full-lexicon
# naming onto a clearly better trajectory than the historical single-task
# full-lex naming run?  No decision is made inside the run; milestones at
# 300/500/700 N-exposures are evaluated on the FULL populations and
# checkpointed, and all three are inspected AFTER the run.
#
# Contract: export L3_EXPECTED_COMMIT=<sha> before sbatch.  The job refuses
# to start on branch/commit mismatch or a dirty TRACKED tree (untracked
# historical files do not block).  Population sizes and the frozen canonical
# hashes are asserted by the driver itself (final_full + real GloVe).
#
# Resume: the driver checkpoints contain model, optimizer, stream cursors and
# all RNG states (CUDA included); this script auto-resumes from the latest
# checkpoint, so resubmission after walltime/preemption continues EXACTLY.
#
# Walltime 08:00:00 is set from the measured V100 preflight (job 1643901):
# 324,100 steps x 0.04269 s/step = 3.84 h of training, plus ~140 dev-sample
# evaluations and 4 full-population evaluations -- roughly 2x margin.  A
# timeout is recoverable rather than fatal: resubmitting resumes exactly.
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

# ---- FINAL-1 configuration (frozen; edit only for a NEW experiment) -------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
DEVICE=cuda
EPOCHS=700                                    # 700 x 463 = 324,100 optimizer steps
FULL_EVAL_AT=138900,231500,324100             # 300 / 500 / 700 N-exposures
EVAL_EVERY=2315                               # dev-sample cadence (5 epochs)
SAVE_EVERY=4630                               # safety checkpoints (10 epochs)
RUN_ID="final1_${REGIME}_seed${SEED}_${SUBSET_MODE}"
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}     # $SCRATCH is purge-eligible:
                                              # archive finished runs to $STORE

# ---- automatic exact resume from the latest checkpoint --------------------
RESUME_ARGS=()
LATEST=$(ls -t "$RUNS/$RUN_ID/checkpoints/"step_*.pt 2>/dev/null | head -1 || true)
if [[ -n "${LATEST:-}" ]]; then
    echo "== resuming exactly from $LATEST"
    RESUME_ARGS=(--resume "$LATEST")
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 100 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    "${RESUME_ARGS[@]}"

echo "== run complete: $RUNS/$RUN_ID"
echo "== archive with: cp -r $RUNS/$RUN_ID $STORE/lichtheim3_final1/"
