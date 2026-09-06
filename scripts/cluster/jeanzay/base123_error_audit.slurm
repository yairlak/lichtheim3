#!/bin/bash
#SBATCH --job-name=l3_b123_errs
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# ===========================================================================
# Read-only per-item R/N/C error audit of the FINAL base-123 endpoints.
# Trains nothing; writes only <run>/error_audit_u<U>/.
#
#   array 0-3 : H512 seeds 19,20,21,22 at u=500  (step 1,389,000)
#   array 4-7 : H512 seeds 19,20,21,22 at u=400  (step 1,111,200)
#
# The u400 pass is what makes the u400-vs-u500 overlap analysis possible,
# which is the discriminator between a moving error frontier and a stable
# systematic set.  Override the width with L3_AUDIT_WIDTH=256 to audit the
# H256 arm instead.
# ===========================================================================
set -euo pipefail
module purge
module load pytorch-gpu/py3/2.6.0

REPO=${L3_REPO:-/lustre/fswork/projects/rech/llg/uss35bp/lichtheim3/lichtheim3}
cd "$REPO"
RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
W=${L3_AUDIT_WIDTH:-512}

SEEDS=(19 20 21 22 19 20 21 22)
STEPS=(1389000 1389000 1389000 1389000 1111200 1111200 1111200 1111200)
IDX=${SLURM_ARRAY_TASK_ID:?must run as an array job}
SEED=${SEEDS[$IDX]}; STEP=${STEPS[$IDX]}
U=$(( STEP / (463*6) ))
RUN_ID="final_base123_h${W}_s${SEED}"
CKPT="$RUNS/$RUN_ID/checkpoints/step_$(printf '%08d' $STEP).pt"
OUT="$RUNS/$RUN_ID/error_audit_u${U}"

[[ -f "$CKPT" ]] || { echo "FATAL: $CKPT not found"; exit 1; }
echo "== auditing $RUN_ID at u=$U"
echo "== sha256 $(sha256sum "$CKPT" | cut -d' ' -f1)"

srun python scripts/naming_comprehension/base123_error_audit.py \
    --ckpt "$CKPT" --out-dir "$OUT" --device cuda

echo "== done -> $OUT"
