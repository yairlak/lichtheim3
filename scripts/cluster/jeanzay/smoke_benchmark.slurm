#!/bin/bash
#SBATCH --job-name=l3_smoke
#SBATCH --account=<JEANZAY_ACCOUNT>          # e.g. abc@v100 / abc@a100 — fill in
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                          # remove this line for a CPU-only smoke
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
##SBATCH --partition=<JEANZAY_PARTITION>      # uncomment/fill only if your project requires one
##SBATCH --qos=<JEANZAY_QOS>                  # uncomment/fill only if needed (e.g. qos_gpu-dev)
##SBATCH -C <JEANZAY_CONSTRAINT>              # e.g. v100-16g / v100-32g / a100 — fill if needed

# ===========================================================================
# Lichtheim3 Jean Zay SMOKE benchmark — throughput + checkpoint + exact resume.
# NOT a scientific run.  ~15 min of compute.  Never touches outputs/.
#
# Before first use fill in the <PLACEHOLDERS> above and the environment block
# below; see scripts/cluster/jeanzay/README.md for the full checklist.
# ===========================================================================
set -euo pipefail

# ---- environment (fill in ONE of the two options) -------------------------
# Option A: Jean Zay module (check availability with `module avail pytorch`):
# module purge
# module load <JEANZAY_PYTORCH_MODULE>        # e.g. pytorch-gpu/py3/<version>
# Option B: your own conda/venv:
# source <PATH_TO_YOUR_ENV>/bin/activate

REPO=${L3_REPO:-$HOME/lichtheim3}             # where you cloned the repo
WORK=${L3_WORK:-$SCRATCH/lichtheim3_smoke}    # Jean Zay $SCRATCH is purge-eligible
mkdir -p "$WORK"
cd "$REPO"

# Strict-determinism support for the --torch-deterministic A/B below.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "== node: $(hostname)  date: $(date -Is)"
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

# ---- 1. throughput: CPU reference on this node ----------------------------
python scripts/cluster/benchmark_throughput.py --device cpu \
    --json-out "$WORK/bench_cpu.json"

# ---- 2. throughput: one CUDA GPU ------------------------------------------
python scripts/cluster/benchmark_throughput.py --device cuda \
    --json-out "$WORK/bench_cuda.json"

# ---- 3. determinism cost A/B on GPU ---------------------------------------
python scripts/cluster/benchmark_throughput.py --device cuda --torch-deterministic \
    --json-out "$WORK/bench_cuda_deterministic.json"

# ---- 4. checkpoint + exact-resume smoke (150 steps, resume to 300) --------
# Uses GloVe fallback: mechanics only, no scientific content.
SMOKE_DIR="$WORK/driver_smoke"
rm -rf "$SMOKE_DIR"
python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed 999 --subset-mode final_full \
    --device cuda --max-steps 150 --save-every 150 --eval-every 0 --log-every 50 \
    --out-dir "$SMOKE_DIR" --run-id smoke \
    --glove-path _no_such_glove.txt --allow-glove-fallback --no-subset-hash-check

python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed 999 --subset-mode final_full \
    --device cuda --max-steps 300 --save-every 150 --eval-every 0 --log-every 50 \
    --out-dir "$SMOKE_DIR" --run-id smoke \
    --resume "$SMOKE_DIR/smoke/checkpoints/step_00000150.pt" \
    --glove-path _no_such_glove.txt --allow-glove-fallback --no-subset-hash-check

echo "== smoke complete; reports in $WORK"
