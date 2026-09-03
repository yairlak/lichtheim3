#!/bin/bash
#SBATCH --job-name=l3_final7a_audit
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# ===========================================================================
# FINAL-7A — READ-ONLY parameter-localisation audit at the matched R130 state.
#
# NO TRAINING.  No optimizer step, no backward pass, no checkpoint written to
# any scientific run directory.  Source checkpoints are opened read-only and
# fingerprinted before and after; the job fails if any of them changes.
#
# Question: the three optimizer topologies branch from one R100 state and at
# R130 dissociate --
#     control  (R+N+C shared)  naming .054276  C .076695  LTM .548612
#     FINAL-6P (R|N|C)         naming .041798  C .107430  LTM .340604
#     FINAL-7P (RN|C)          naming .052010  C .107037  LTM .242603
# so RN coupling rescues naming without rescuing LTM, while C separation
# tracks both the C gain and severe LTM loss.  This audit asks WHERE in
# parameter space those effects live, by displacement analysis and by
# transplanting one parameter group at a time between endpoints.
#
# CAUSAL CAVEAT (carried in every output): transplantation is a diagnostic
# intervention on ENDPOINT parameter states.  It localises endpoint dependence
# and is consistent with or against a mechanism; it does NOT show that
# training a group differently would reproduce the trajectory.
#
# Contract: export L3_EXPECTED_COMMIT=<sha> before sbatch.
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
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }
echo "== contract OK: $BRANCH @ $HEAD"

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
SEED=22; MODE=final_full
R100="$RUNS/final3p_i123_seed${SEED}_${MODE}/checkpoints/step_00277800.pt"
CONTROL="$RUNS/final3p_i123_seed${SEED}_${MODE}/checkpoints/step_00361140.pt"
SEP="$RUNS/final6p_r100_sepmoments_seed${SEED}_${MODE}/checkpoints/step_00361140.pt"
GROUPED="$RUNS/final7p_r100_rn_c_banks_seed${SEED}_${MODE}/checkpoints/step_00361140.pt"

for f in "$R100" "$CONTROL" "$SEP" "$GROUPED"; do
    [[ -f "$f" ]] || { echo "FATAL: checkpoint not found: $f"; exit 1; }
done
echo "== all four checkpoints present"

OUT=reports/final7_parameter_localization_r130

# The full-population sweep decodes 29,571 words autoregressively per
# condition.  Dropping the WM route (which no transplant can change, since no
# transplanted group feeds it except 'wm' itself) keeps the job short; pass
# --routes full,ltm,wm for the complete set.
srun python scripts/naming_comprehension/parameter_localization_audit.py \
    --r100 "$R100" --control "$CONTROL" --sep "$SEP" --grouped "$GROUPED" \
    --device cuda --eval-population full --routes full,ltm \
    --out-dir "$OUT"

echo "== FINAL-7A audit complete"
echo "== outputs: $REPO/$OUT/{audit.json,displacement.csv,conditions.csv}"
echo "== the report directory is inside the repo; copy it back with:"
echo "   scp -r jean-zay:$REPO/$OUT ."
