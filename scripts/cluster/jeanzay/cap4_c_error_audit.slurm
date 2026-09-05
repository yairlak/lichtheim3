#!/bin/bash
#SBATCH --job-name=l3_c_audit
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ===========================================================================
# Read-only per-item audit of every strict top-1 comprehension error.
# Trains nothing and writes nothing into the source run's own directories
# except a new error_audit/ folder.
#
# Default target is the best C checkpoint, Henc512 @1000 exposures
# (top1 .998642).  Override with arguments:
#     sbatch cap4_c_error_audit.slurm [ckpt_path] [out_dir]
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0

REPO=${L3_REPO:-$WORK/lichtheim3/lichtheim3}
cd "$REPO"
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
PARENT=${L3_C_RUN:-cap3_c_enc512_seed22_full}

CKPT=${1:-$RUNS/$PARENT/checkpoints/step_00438000.pt}
OUT=${2:-$RUNS/$PARENT/error_audit}

[[ -f "$CKPT" ]] || { echo "FATAL: checkpoint $CKPT not found"; exit 1; }
echo "== auditing $CKPT"
echo "== sha256 $(sha256sum "$CKPT" | cut -d' ' -f1)"
echo "== output  $OUT"

srun python scripts/naming_comprehension/c_error_audit.py \
    --ckpt "$CKPT" --out-dir "$OUT" --device cuda

echo "== done."
echo "== per-error table : $OUT/c512_top1_errors.tsv"
echo "== summary         : $OUT/c512_error_summary.json"
