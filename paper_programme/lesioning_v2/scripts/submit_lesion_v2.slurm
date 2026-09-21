#!/bin/bash
#SBATCH --job-name=l3_lesion_v2
#SBATCH --array=0-11
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# ---------------------------------------------------------------------------
# Lesioning V2 execution orchestration. NOT SUBMITTED.
#
# Decomposition: one array task per (state, site) = 4 x 3 = 12 shards. That
# boundary is chosen because it NEVER touches a scientific cell identity: mask
# and noise identities are pure functions of (state_sha256, site, realization)
# and (..., item_id), and contain neither severity nor task nor decoder nor any
# shard index. Grouping therefore cannot alter seeds, RNG ordering or outputs.
#
# Each shard walks its own subset of the authoritative matrix and finalizes
# cells atomically. Exactly-once is enforced by the cell lifecycle, not by the
# scheduler: a resubmitted shard skips every COMPLETE cell and never overwrites
# one. No adaptive scheduling on observed results.
# ---------------------------------------------------------------------------
set -euo pipefail
module load pytorch-gpu/py3/2.6.0

REPO=${REPO:?set REPO}
CTL=${L3_LESION_V2_CONTROL_DIR:?set the execution-control directory}
OUT=${L3_LESION_V2_RESULTS:?set the results namespace (OUTSIDE the worktree)}

export L3_LESION_V2_AUTHORIZATION="$CTL/authorization.json"

cd "$REPO"
python paper_programme/lesioning_v2/scripts/run_lesion_v2.py \
    --out-dir "$OUT" \
    --transfer-dir "$CTL" \
    --shard "$SLURM_ARRAY_TASK_ID" --n-shards 12
