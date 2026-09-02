#!/bin/bash
#SBATCH --job-name=l3_f3p_smoke
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ===========================================================================
# FINAL-3P SHORT smoke for the NEW interleaved_123 schedule.  Minutes, not an
# hour.  NOT a scientific run: it writes only to $SCRATCH/lichtheim3_f3p_smoke.
#
# Verifies: git contract, real GloVe, final_full populations, a few hundred
# interleaved optimizer steps, observed task frequencies == 1:2:3, mid-cycle
# checkpoint + bitwise exact resume, and measured seconds/step projected to
# the FINAL-3P budget (277,800 steps).
#
# Contract: export L3_EXPECTED_COMMIT=<sha> before sbatch.
# ===========================================================================
set -euo pipefail

SUMMARY="${SCRATCH}/lichtheim3_f3p_smoke/summary_${SLURM_JOB_ID:-local}.txt"
mkdir -p "$(dirname "$SUMMARY")"
fail() { echo "F3P SMOKE: FAIL — $*" | tee -a "$SUMMARY"; exit 1; }
trap 'echo "F3P SMOKE: FAIL (aborted at line $LINENO)" | tee -a "$SUMMARY"' ERR
note() { echo "[smoke] $*" | tee -a "$SUMMARY"; }

module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-$WORK/lichtheim3/lichtheim3}
EXPECTED_BRANCH=${L3_EXPECTED_BRANCH:-feat/joint-multitask-scratch}
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || fail "L3_EXPECTED_COMMIT is not exported"
cd "$REPO"

BRANCH=$(git branch --show-current); HEAD=$(git rev-parse HEAD)
[[ "$BRANCH" == "$EXPECTED_BRANCH" ]] || fail "branch $BRANCH != $EXPECTED_BRANCH"
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || fail "HEAD $HEAD != $L3_EXPECTED_COMMIT"
[[ -z "$(git status --porcelain -uno)" ]] || fail "tracked working tree dirty"
[[ -f data/glove.6B.300d.txt ]] || fail "real GloVe missing"
note "contract OK: $BRANCH @ $HEAD"
python -c "import torch; print('[smoke] torch', torch.__version__, '|', torch.cuda.get_device_name(0))" | tee -a "$SUMMARY"

WORKDIR="$SCRATCH/lichtheim3_f3p_smoke/run_${SLURM_JOB_ID:-local}"
rm -rf "$WORKDIR"

# ---- 600 interleaved steps (= 100 macro-cycles), checkpoint at 303 --------
# 303 is deliberately NOT a multiple of 6, so the resume below is mid-cycle.
python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed 22 --subset-mode final_full --schedule interleaved_123 \
    --device cuda --max-steps 600 --save-every 303 --eval-every 0 \
    --log-every 50 --out-dir "$WORKDIR" --run-id runA

python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed 22 --subset-mode final_full --schedule interleaved_123 \
    --device cuda --max-steps 600 --save-every 303 --eval-every 0 \
    --log-every 50 --out-dir "$WORKDIR" --run-id runB \
    --resume "$WORKDIR/runA/checkpoints/step_00000303.pt"

python - "$WORKDIR" <<'EOF' | tee -a "$SUMMARY"
import csv, collections, sys, torch
w = sys.argv[1]
a = torch.load(f"{w}/runA/checkpoints/step_00000600.pt", map_location="cpu", weights_only=False)
b = torch.load(f"{w}/runB/checkpoints/step_00000600.pt", map_location="cpu", weights_only=False)
sa, sb = a["model_state_dict"], b["model_state_dict"]
bad = [k for k in sa if not torch.equal(sa[k].cpu(), sb[k].cpu())]
assert not bad, f"mid-cycle resume diverged on {len(bad)} tensors: {bad[:4]}"
oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
obad = sum(1 for k in oa for kk, v in oa[k].items()
           if torch.is_tensor(v) and not torch.equal(v.cpu(), ob[k][kk].cpu()))
assert obad == 0, f"{obad} optimizer tensors differ after resume"
assert a["cursors"] == b["cursors"], (a["cursors"], b["cursors"])
print("[smoke] mid-cycle exact resume OK (model + optimizer + cursors bitwise)")

# 600 steps = 100 complete macro-cycles -> exactly 100 R, 200 N, 300 C
cur = a["cursors"]
assert cur == {"repetition": 100, "pool": 100, "naming": 200,
               "comprehension": 300}, cur
print(f"[smoke] observed 1:2:3 confirmed: R={cur['repetition']} "
      f"N={cur['naming']} C={cur['comprehension']} pool={cur['pool']}")

rows = list(csv.DictReader(open(f"{w}/runA/logs/losses.tsv"), delimiter="\t"))
freq = collections.Counter(r["task"] for r in rows)
print(f"[smoke] logged task labels: {dict(freq)}")
EOF

# ---- throughput: 600 timed interleaved steps, no evaluation --------------
python - "$WORKDIR" <<'EOF' | tee -a "$SUMMARY"
import json, os, sys, time, torch
sys.path.insert(0, os.getcwd())
from scripts.naming_comprehension.train_joint_scratch import (
    FINAL_FULL_MODE, INTERLEAVED_123, JointScratchTrainer)
tr = JointScratchTrainer(
    regime="j0", seed=999, device="cuda", max_words=30000,
    lexicon_path="data/lexicon_en_glove_covered.tsv", dorsal_pool_size=4000,
    batch_size=64, subset_mode=FINAL_FULL_MODE, subset_per_band=822,
    subset_size=64, lr_boundary_steps=46300, allow_glove_fallback=False,
    require_subset_hash=True, glove_path="data/glove.6B.300d.txt",
    schedule=INTERLEAVED_123)
for _ in range(60):                      # warmup (full macro-cycles)
    tr.train_step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for _ in range(600):
    tr.train_step()
torch.cuda.synchronize()
dt = time.time() - t0
sps = 600 / dt
rep = {
    "seconds_per_step": round(dt / 600, 6),
    "steps_per_second": round(sps, 2),
    "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    "projected_hours_M1_138900_steps": round(138_900 / sps / 3600, 3),
    "projected_hours_M2_277800_steps": round(277_800 / sps / 3600, 3),
}
print("[bench] " + json.dumps(rep))
json.dump(rep, open(f"{sys.argv[1]}/throughput.json", "w"), indent=2)
EOF

trap - ERR
echo "F3P SMOKE: PASS ($(date -Is))" | tee -a "$SUMMARY"
echo "summary: $SUMMARY"
