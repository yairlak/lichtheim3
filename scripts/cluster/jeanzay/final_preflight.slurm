#!/bin/bash
#SBATCH --job-name=l3_final_preflight
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ===========================================================================
# Lichtheim3 FINAL preflight — ONE job validating the FINAL multitask driver
# on the already-validated Jean Zay environment.  NOT a scientific run.
#
# Verifies, end to end: environment, exact Git contract, real GloVe, the
# FINAL population tests, R/N/C/bank sizes, a CUDA final_full training smoke
# with checkpoint -> bitwise exact resume -> endpoint evaluation, and the
# matched CUDA throughput benchmark (incl. peak GPU memory).
#
# Contract: export L3_EXPECTED_COMMIT=<sha> before sbatch (the submit block
# in scripts/cluster/jeanzay/README.md does this).  The job FAILS on branch
# mismatch, HEAD mismatch, or a dirty TRACKED working tree.  Untracked
# historical files (old SLURM logs etc.) do not block it.
# ===========================================================================
set -euo pipefail

SUMMARY="${SCRATCH}/lichtheim3_preflight/summary_${SLURM_JOB_ID:-local}.txt"
mkdir -p "$(dirname "$SUMMARY")"
fail() { echo "PREFLIGHT: FAIL — $*" | tee -a "$SUMMARY"; exit 1; }
trap 'echo "PREFLIGHT: FAIL (aborted at line $LINENO)" | tee -a "$SUMMARY"' ERR
note() { echo "[preflight] $*" | tee -a "$SUMMARY"; }

# ---- 1. environment (validated Lichtheim3 Jean Zay setup) -----------------
module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-$WORK/lichtheim3/lichtheim3}
EXPECTED_BRANCH=${L3_EXPECTED_BRANCH:-feat/joint-multitask-scratch}
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || fail "L3_EXPECTED_COMMIT is not exported"
cd "$REPO"

note "user=$(whoami) node=$(hostname) date=$(date -Is)"
note "repo=$REPO"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" | tee -a "$SUMMARY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$SUMMARY" || true

# ---- 2. git contract ------------------------------------------------------
BRANCH=$(git branch --show-current)
HEAD=$(git rev-parse HEAD)
[[ "$BRANCH" == "$EXPECTED_BRANCH" ]] || fail "branch $BRANCH != $EXPECTED_BRANCH"
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || fail "HEAD $HEAD != $L3_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain -uno)" ]] || fail "tracked working tree is dirty"
note "git contract OK: $BRANCH @ $HEAD (tracked tree clean)"

# ---- 3. real GloVe --------------------------------------------------------
[[ -f data/glove.6B.300d.txt ]] || fail "data/glove.6B.300d.txt missing (bash data/get_glove.sh)"
note "GloVe present ($(du -h data/glove.6B.300d.txt | cut -f1))"

# ---- 4. FINAL population + driver tests -----------------------------------
python -m pytest tests/test_final_populations.py tests/test_joint_scratch.py -q \
    | tail -2 | tee -a "$SUMMARY"

# ---- 5. explicit population sizes (R / N / C / bank) ----------------------
python - <<'EOF' | tee -a "$SUMMARY"
import sys; sys.path.insert(0, ".")
from scripts.naming_comprehension.train_joint_scratch import (
    FINAL_FULL_MODE, JointScratchTrainer)
tr = JointScratchTrainer(
    regime="j0", seed=22, device="cpu", max_words=30000,
    lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=4000,
    batch_size=64, subset_mode=FINAL_FULL_MODE, subset_per_band=822,
    subset_size=64, lr_boundary_steps=46300, allow_glove_fallback=True,
    require_subset_hash=False, glove_path="_preflight_no_glove.txt")
R, N, C = tr.streams["repetition"].n, tr.streams["naming"].n, tr.streams["comprehension"].n
bank = int(tr.bank_raw.shape[0])
assert (R, N, C, bank) == (29571, 29571, 27981, 29571), (R, N, C, bank)
print(f"[preflight] populations OK: R={R} N={N} C={C} bank={bank}")
print(f"[preflight] C sha256: {tr.comp_hash}")
EOF

# ---- 6-10. CUDA final_full smoke: train -> ckpt -> exact resume -> eval ---
# Real GloVe + the frozen canonical-C hash assertion (require_hash is active
# by default for final_full at full size).  60 steps, checkpoint at 30,
# resume 30 -> 60 in a second process, then bitwise comparison + endpoint
# full-population evaluation on the resumed leg.
SMOKE="$SCRATCH/lichtheim3_preflight/driver_smoke_${SLURM_JOB_ID:-local}"
rm -rf "$SMOKE"

python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed 22 --subset-mode final_full --device cuda \
    --max-steps 60 --save-every 30 --eval-every 30 --probe-every 0 \
    --log-every 10 --out-dir "$SMOKE" --run-id runA

python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed 22 --subset-mode final_full --device cuda \
    --max-steps 60 --save-every 30 --eval-every 0 --probe-every 0 \
    --log-every 10 --endpoint-eval --out-dir "$SMOKE" --run-id runB \
    --resume "$SMOKE/runA/checkpoints/step_00000030.pt"

python - "$SMOKE" <<'EOF' | tee -a "$SUMMARY"
import sys, torch
smoke = sys.argv[1]
a = torch.load(f"{smoke}/runA/checkpoints/step_00000060.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{smoke}/runB/checkpoints/step_00000060.pt", map_location="cpu", weights_only=False)
sa, sb = a["model_state_dict"], b["model_state_dict"]
bad = [k for k in sa if not torch.equal(sa[k].cpu(), sb[k].cpu())]
assert not bad, f"resume diverged on {len(bad)} tensors: {bad[:4]}"
assert a["cursors"] == b["cursors"] == {k: 60 for k in a["cursors"]}
oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
obad = sum(1 for k in oa for kk, v in oa[k].items()
           if torch.is_tensor(v) and not torch.equal(v.cpu(), ob[k][kk].cpu()))
assert obad == 0, f"{obad} optimizer tensors differ"
import csv
rows = list(csv.DictReader(open(f"{smoke}/runB/metrics.tsv"), delimiter="\t"))
end = rows[-1]
for k in ("full_rep_full", "full_comp_top1", "full_naming_exact"):
    assert end[k] not in ("", "nan"), f"endpoint {k} missing"
print("[preflight] CUDA smoke OK: bitwise exact resume (model+optimizer+cursors), "
      "endpoint full-population evaluation present")
print(f"[preflight] endpoint @60 steps: full_rep={end['full_rep_full']} "
      f"full_naming={end['full_naming_exact']} full_comp_top1={end['full_comp_top1']}")
EOF

# ---- 11-12. matched CUDA throughput benchmark -----------------------------
python scripts/cluster/benchmark_throughput.py --device cuda \
    --json-out "$SCRATCH/lichtheim3_preflight/bench_cuda_${SLURM_JOB_ID:-local}.json" \
    | python -c "import json,sys; r=json.load(sys.stdin); print('\n'.join(f'[bench] {k}: {r[k]}' for k in ('device','cuda_device','seconds_per_step','steps_per_second','examples_per_second','projected_hours_per_100k_steps','peak_gpu_memory_mb')))" \
    | tee -a "$SUMMARY"

# ---- 13. verdict ----------------------------------------------------------
trap - ERR
echo "PREFLIGHT: PASS ($(date -Is))" | tee -a "$SUMMARY"
echo "summary: $SUMMARY"
