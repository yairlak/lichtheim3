#!/bin/bash
# ===========================================================================
# V7 FRESH STEP-0 CEILING REPLICATION -- submission (run ON a Jean-Zay login
# node).  Deploys the pinned commit into a dedicated V7 checkout, verifies it,
# proves seed cleanliness on the cluster, refuses unsafe overwrites, then
# submits fresh_ceiling_v7.slurm as ONE four-task array plus a chain of
# dependent re-submissions (afterany) so a ~44 GPU-h no-hit seed completes
# across 20 h allocations without manual action.  Every element is the SAME
# idempotent script.
#
#   fresh_ceiling_v7_submit.sh <V7_EXECUTABLE_COMMIT> [N_CHAIN=4]
# ===========================================================================
set -euo pipefail
COMMIT=${1:?V7 executable commit (full sha)}
N_CHAIN=${2:-4}
BRANCH=fresh/step0-ceiling-v7
WORK=${WORK:-/lustre/fswork/projects/rech/llg/uss35bp}
SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/llg/uss35bp}
BASE="$WORK/l3_fresh_v7"
REPO="$BASE/repo"
LOGS="$BASE/logs"
GLOVE="$WORK/lichtheim3/lichtheim3/data/glove.6B.300d.txt"
RUNS="$SCRATCH/l3_fresh_v7_runs"
OFFICIAL_RUNS="$SCRATCH/lichtheim3_runs"
V6_RUNS="$SCRATCH/l3_prospective_v6_runs"
MAIN="$WORK/lichtheim3/lichtheim3"

module purge; module load pytorch-gpu/py3/2.6.0; module load git

# ---- dedicated checkout at the pinned commit ------------------------------
mkdir -p "$BASE" "$LOGS"
if [[ ! -d "$REPO/.git" && ! -f "$REPO/.git" ]]; then
    git -C "$MAIN" fetch origin "$BRANCH"
    git -C "$MAIN" worktree add --detach "$REPO" "$COMMIT"
else
    git -C "$REPO" fetch origin "$BRANCH"
    git -C "$REPO" checkout --detach "$COMMIT"
fi
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$COMMIT" ]] || { echo "FATAL: HEAD != $COMMIT"; exit 1; }
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked tree dirty"; exit 1; }
echo "== V7 checkout $REPO @ $COMMIT (clean)"
[[ -f "$GLOVE" ]] || { echo "FATAL: GloVe missing at $GLOVE"; exit 1; }
[[ "$(sha256sum "$GLOVE" | cut -d' ' -f1)" == "91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed" ]] || { echo "FATAL: GloVe sha"; exit 1; }

# ---- unsafe-overwrite refusal --------------------------------------------
for s in p1_s31 p2_s32 p3_s33 p4_s34; do
    if [[ -d "$RUNS/fresh_ceiling_v7_$s/checkpoints" && -n "$(ls -A "$RUNS/fresh_ceiling_v7_$s/checkpoints" 2>/dev/null)" ]]; then
        [[ "${L3_RESUBMIT:-0}" == "1" ]] || { echo "FATAL: $RUNS/fresh_ceiling_v7_$s already has checkpoints; export L3_RESUBMIT=1 to submit a resume chain"; exit 1; }
    fi
done
if [[ -n "$(squeue -u "$USER" -h -o '%j' | grep -x l3_fresh_v7 || true)" ]]; then
    echo "FATAL: an l3_fresh_v7 job is already queued/running"; squeue -u "$USER"; exit 1
fi

# ---- seed cleanliness proof on the cluster (read-only) --------------------
sacct -u "$USER" -S 2026-06-01 -X -n -o JobName%120,JobID%20,State%12 > "$LOGS/submit_sacct_jobnames.txt" 2>/dev/null || true
( ls -1 "$OFFICIAL_RUNS" "$V6_RUNS" 2>/dev/null; ls -1 "$RUNS" 2>/dev/null | grep -v "^fresh_ceiling_v7_" ) > "$LOGS/submit_run_namespaces.txt" || true
python scripts/naming_comprehension/fresh_ceiling_v7.py seed-audit --seeds 31,32,33,34 \
    --paths "$OFFICIAL_RUNS" "$V6_RUNS" "$WORK/lichtheim3" "$WORK/l3_prospective_v6" \
    --text-files "$LOGS/submit_sacct_jobnames.txt" "$LOGS/submit_run_namespaces.txt" \
    --out-json "$LOGS/submit_seed_audit.json" \
    || { echo "FATAL: a cohort seed is NOT clean on Jean-Zay -- apply the preregistered replacement rule first"; exit 1; }
echo "== seeds 31,32,33,34 clean on Jean-Zay"
mkdir -p "$RUNS"
echo "== SCRATCH free: $(df -Ph "$RUNS" | awk 'NR==2{print $4}')   (cohort worst case ~95 GB: 4 x 781 x ~30 MB)"

# ---- submit: one array + (N_CHAIN-1) dependent re-submissions -------------
cd "$LOGS"
EXPORTS="ALL,L3_REPO=$REPO,L3_EXPECTED_COMMIT=$COMMIT,L3_GLOVE=$GLOVE,L3_RUNS=$RUNS,L3_OFFICIAL_RUNS=$OFFICIAL_RUNS,L3_V6_RUNS=$V6_RUNS"
J1=$(sbatch --parsable --export="$EXPORTS" "$REPO/scripts/cluster/jeanzay/fresh_ceiling_v7.slurm"); J1=${J1%%;*}
CHAIN=("$J1"); PREV="$J1"
for (( k=2; k<=N_CHAIN; k++ )); do
    J=$(sbatch --parsable --dependency=afterany:"$PREV" --export="$EXPORTS" "$REPO/scripts/cluster/jeanzay/fresh_ceiling_v7.slurm"); J=${J%%;*}
    CHAIN+=("$J"); PREV="$J"
done
cat > "$LOGS/submission_${J1}.json" <<EOF
{"experiment": "fresh_ceiling_v7", "preregistration": "docs/analysis/FRESH_STEP0_CEILING_PREREG_V7.md",
 "executable_commit": "$COMMIT", "repo": "$REPO", "runs": "$RUNS", "logs": "$LOGS",
 "array": "0-3", "slots": {"0": "p1_s31", "1": "p2_s32", "2": "p3_s33", "3": "p4_s34"},
 "job_chain_afterany": [$(printf '"%s",' "${CHAIN[@]}" | sed 's/,$//')],
 "submitted": "$(date -Is)", "horizon_u": 3900, "detector_u": 5}
EOF
echo "V7_EXECUTABLE_COMMIT=$COMMIT"
echo "V7_JOB_ID=$J1"
echo "V7_JOB_CHAIN=${CHAIN[*]}"
echo "V7_COHORT=31,32,33,34"
echo "V7_HORIZON=u3900"
echo "V7_DETECTOR=5u"
echo "V7_STATUS=SUBMITTED"
