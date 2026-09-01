#!/bin/bash
#SBATCH --job-name=l3_final
#SBATCH --account=<JEANZAY_ACCOUNT>          # e.g. abc@v100 / abc@a100 — fill in
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                          # or remove + use --device cpu below
#SBATCH --hint=nomultithread
#SBATCH --time=<JEANZAY_WALLTIME>             # e.g. 20:00:00 (respect partition limits)
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue                             # preempted/timed-out jobs may be requeued
##SBATCH --partition=<JEANZAY_PARTITION>      # only if your project requires one
##SBATCH --qos=<JEANZAY_QOS>                  # only if needed
##SBATCH -C <JEANZAY_CONSTRAINT>              # e.g. v100-32g / a100

# ===========================================================================
# Lichtheim3 FINAL scientific run template — exact-resume safe.
#
# The driver's checkpoints contain model, optimizer, all stream cursors and
# all RNG states; resuming from the latest checkpoint continues the run
# EXACTLY (verified by tests/test_joint_scratch.py and
# tests/test_final_populations.py).  Because of that, this script is
# idempotent: submit it again (or let --requeue rerun it) and it picks up
# from the latest checkpoint automatically.
#
# Fill in every <PLACEHOLDER> and the environment block before first use.
# ===========================================================================
set -euo pipefail

# ---- environment (fill in ONE of the two options) -------------------------
# module purge
# module load <JEANZAY_PYTORCH_MODULE>
# source <PATH_TO_YOUR_ENV>/bin/activate

REPO=${L3_REPO:-$HOME/lichtheim3}
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}     # $SCRATCH is fast but purge-eligible:
                                              # archive checkpoints to $STORE/$WORK
cd "$REPO"

# ---- run configuration (edit per experiment) ------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
DEVICE=cuda
EPOCHS=<TOTAL_REP_EPOCHS>                     # 1 epoch = 463 optimizer steps
RUN_ID="final1_${REGIME}_seed${SEED}_${SUBSET_MODE}"
EVAL_EVERY=2315
SAVE_EVERY=4630
# Optional strict determinism (measure its cost with the smoke A/B first):
# DETERMINISM="--torch-deterministic"; export CUBLAS_WORKSPACE_CONFIG=:4096:8
DETERMINISM=""

# ---- scientific runs REQUIRE real GloVe -----------------------------------
test -f data/glove.6B.300d.txt || {
    echo "ERROR: data/glove.6B.300d.txt missing — run: bash data/get_glove.sh"; exit 1; }

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
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 100 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    $DETERMINISM "${RESUME_ARGS[@]}"

echo "== run complete: $RUNS/$RUN_ID"
