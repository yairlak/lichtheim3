#!/bin/bash
#SBATCH --job-name=l3_cap2_n256
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# CAP-2/N256 — can decoder width 256 also reach full-lexicon naming ceiling
# under the SAME optimization schedule that solved H512?
#
# CAP-2 showed H512 branched at 100 exposures with lr 1e-3 -> 1e-4 reaches
# exact 1.0 over all 29,571 words by 125 exposures (CE 6.7e-05, greedy token
# accuracy 1.0, WER 0).  H256 is the other width Yair asked to keep.  This
# job applies the identical transition to the genuine CAP-1 H256 trajectory.
#
# SOURCE (immutable input, transferred from the Mac):
#   CAP-1 cap1_w256_seed22 step_00046300.pt = exactly 100.0 exposures
#   exact .450475  CE .351851  TF tok .886094  greedy tok .589996
#   sha256 7ba12babb0d9327d56f579041bb20f5a98571f0167892a94b81499cf6c970377
# It is RESUMED, never restarted: model weights, AdamW moments, RNG and the
# data cursor all carry over; only the LR changes, and the driver refuses an
# undeclared change.
#
# INTERPRETIVE NOTE, to be stated in any write-up: H256 branches from exact
# .450475 where H512 branched from .839606.  A negative result therefore means
# "256 does not reach ceiling under the schedule validated for 512", NOT
# "256 lacks the capacity in principle" -- the branch point is part of the
# condition, and it was not re-tuned, by design.
#
# Frozen: width 256 (wm 128 / enc 128 / dec 256), seed 22, full 29,571
# population, counter-addressed order, batch 64, wd 1e-5, clip 1.0, teacher
# forcing, trainable scope, and the free-greedy evaluation (BOS -> greedy ->
# EOS/cap, target length never consulted).  Changed: LR only.
#
# Stops at exact 1.0; the milestone grid places 150 immediately after 125 so
# a single touch can be told apart from a stable ceiling.
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-$WORK/lichtheim3/lichtheim3}
cd "$REPO"
HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "== repo $REPO @ $HEAD"
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked tree dirty"; exit 1; }
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }
if [[ -n "${L3_EXPECTED_COMMIT:-}" ]]; then
    [[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD != expected"; exit 1; }
fi

WIDTH=256
SEED=22
LR=1e-4
MAX_EXPOSURES=3000
EVALS=100,125,150,175,200,250,300,400,500,750,1000,1500,2000,2500,3000
RUN_ID=${L3_RUN_ID:-cap2_n256_lr1e4_seed22_full}
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

case "$RUN_ID" in
    cap2_n256*) : ;;
    *) echo "FATAL: run id must start with cap2_n256 (never a CAP1/FINAL dir)"; exit 1 ;;
esac

SOURCE_DIR=${L3_CAP2_SOURCE:-$WORK/lichtheim3/cap2_source}
SOURCE_CKPT="$SOURCE_DIR/cap1_w256_step_00046300.pt"
SOURCE_SHA_EXPECTED="7ba12babb0d9327d56f579041bb20f5a98571f0167892a94b81499cf6c970377"
[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: source $SOURCE_CKPT not found — transfer it first"; exit 1; }
SOURCE_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
[[ "$SOURCE_SHA" == "$SOURCE_SHA_EXPECTED" ]] || {
    echo "FATAL: source sha $SOURCE_SHA != $SOURCE_SHA_EXPECTED — transfer corrupt"; exit 1; }
echo "== source OK: $SOURCE_CKPT sha256 $SOURCE_SHA"

python scripts/naming_comprehension/naming_capacity_sweep.py \
    --width "$WIDTH" --seed "$SEED" --device cuda \
    --out-dir "$JOBSCRATCH" --benchmark 200

# First launch resumes the immutable source and DECLARES lr 1e-3 -> 1e-4;
# a requeue resumes this run's own newest checkpoint, whose stored lr is
# already 1e-4, so nothing is re-declared.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$SOURCE_CKPT"
    EXTRA=(--lr-transition)
    echo "== first launch: branching from CAP-1 H256 @ 100 exposures"
    echo "== declaring lr 1e-3 -> $LR (AdamW moments preserved)"
else
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming from $(basename "$OWN_LATEST")"
fi

srun python scripts/naming_comprehension/naming_capacity_sweep.py \
    --width "$WIDTH" --seed "$SEED" --device cuda --lr "$LR" \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --eval-exposures "$EVALS" --max-exposures "$MAX_EXPOSURES" \
    --resume "$RESUME_FROM" --log-every 4630 "${EXTRA[@]}"

echo "== CAP-2/N256 finished: $RUNS/$RUN_ID"
echo "== grep 'CEILING REACHED' for exact 1.0 over all 29,571 words"
echo "== archive: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_cap2/"
