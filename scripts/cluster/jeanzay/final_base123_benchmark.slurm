#!/bin/bash
#SBATCH --job-name=l3_base123_bench
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# ===========================================================================
# V100 preflight for the FINAL base-123 block.  Exercises the ACTUAL joint
# driver at the production widths -- not a capacity probe -- so the timing
# includes real 1:2:3 R/N/C steps with the dorsal pool riding every R step.
#
#   array 0 -> enc=dec=256      array 1 -> enc=dec=512
#
# It runs a SHORT from-scratch leg (3,000 steps = 500 macro-cycles, so ~500 R,
# 1,000 N and 1,500 C real updates) into $JOBSCRATCH, then ONE full-population
# evaluation, and reports both costs.  Nothing is written to the production
# run directory and no production run id is used.
#
# Read from the log:
#   - peak GPU memory (nvidia-smi line) -> V100 16GB fit
#   - seconds for the training leg      -> s/step -> hours to 1,389,000 steps
#   - seconds for the full evaluation   -> x9 scheduled evals
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-/lustre/fswork/projects/rech/llg/uss35bp/lichtheim3/lichtheim3}
cd "$REPO"
echo "== repo $REPO @ $(git rev-parse HEAD)"

WIDTHS=(256 512)
V=${WIDTHS[${SLURM_ARRAY_TASK_ID:-0}]}
BENCH_STEPS=${L3_BENCH_STEPS:-3000}
OUT="$JOBSCRATCH/base123_bench"

echo "== benchmarking wm=128 enc=$V dec=$V for $BENCH_STEPS steps"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Training leg WITHOUT any full evaluation, then a single full evaluation via
# --endpoint-eval, so the two costs are separable in the timestamps.
srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed 22 --subset-mode final_full \
    --schedule interleaved_123 \
    --wm-hidden 128 --enc-hidden "$V" --dec-hidden "$V" \
    --device cuda --max-steps "$BENCH_STEPS" \
    --eval-every 0 --save-every "$BENCH_STEPS" --log-every 250 \
    --endpoint-eval \
    --out-dir "$OUT" --run-id "bench_h${V}"

nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "== benchmark done for enc=dec=$V"
echo "== extrapolate: (training seconds / $BENCH_STEPS) * 1389000 / 3600 = hours to u=500"
echo "== add 9 x (one full-evaluation cost) for the scheduled u-grid"
