#!/bin/bash
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
# CAP-2 — overnight LR-stability branches for H512 full-lexicon naming.
#
# CAP-1 established that ventral width, not the objective, was the naming
# bottleneck: from scratch on all 29,571 words, H512 reached exact .839606
# (CE .084228, TF tok .973714) at only 100 exposures/item -- versus .0348 for
# H128 at 1500 and .0977 for the historical warm-started H128 at 3000 -- and
# then progressively COLLAPSED under the untuned constant lr 1e-3
# (.7429 @300, .5363 @500, .1564 @750, .0144 @1000, .0008 @1500), with
# TRAINING CE rising in step: an optimization instability, not overfitting.
#
# CAP-2 is the one-factor follow-up: from the SAME H512@100 checkpoint --
# model weights, AdamW moments, RNG and data cursor all preserved, optimizer
# NEVER reset -- three parallel branches that differ ONLY in the LR applied
# from exposure 100 on:
#
#     A  1e-3 -> 3e-4        B  1e-3 -> 1e-4        C  1e-3 -> 3e-5
#
# (the 1e-3 continuation already exists: it is the CAP-1 trajectory itself.)
# Frozen: H512 architecture (wm 128 / enc 128 / dec 512), seed 22, batch 64,
# wd 1e-5, clip 1.0, teacher forcing, loss, trainable scope (phon_embed,
# sem_to_h0, decoder, dec_to_premotor, motor), the full 29,571 population,
# the counter-addressed item order, and the committed free-greedy evaluation
# (BOS -> greedy -> EOS/cap, target length never consulted).
#
# Budget: up to 3000 exposures/item (1,389,000 optimizer steps; 1,342,700 in
# this leg).  Full-population evaluation + checkpoint at exposures
# 125,150,175,200,250,300,400,500,750,1000,1500,2000,2500,3000, plus the
# continuity re-evaluation at 100 on first launch.  A branch STOPS itself
# the first time exact_match == 1.0 over all 29,571 words (checkpoint saved,
# "CEILING REACHED" in the log); >= .95 keeps running -- the target is 100%.
#
# Usage (LR and run id are sbatch arguments; job-name set at submit time):
#   sbatch --job-name=l3_cap2_3e4 cap2_h512_lr_branch.slurm 3e-4 cap2_h512_lr3e4_seed22_full
#   sbatch --job-name=l3_cap2_1e4 cap2_h512_lr_branch.slurm 1e-4 cap2_h512_lr1e4_seed22_full
#   sbatch --job-name=l3_cap2_3e5 cap2_h512_lr_branch.slurm 3e-5 cap2_h512_lr3e5_seed22_full
#
# The job SELF-BENCHMARKS 200 V100 steps before training and prints s/step
# and the projected hours to 3000 exposures.  Walltime 10:00:00 covers even
# the Mac-CPU-speed worst case (0.0194 s/step -> 7.2 h + evals); the V100 is
# expected to be several times faster.  Requeue-safe: a requeued job resumes
# from this branch's own newest milestone checkpoint with no re-declaration.
#
# The source checkpoint is IMMUTABLE INPUT, transferred from the Mac; its
# SHA256 is pinned below and verified before anything runs.
# ===========================================================================
set -euo pipefail

LR_ARG=${1:?usage: sbatch cap2_h512_lr_branch.slurm <lr> <run_id>}
RUN_ID=${2:?usage: sbatch cap2_h512_lr_branch.slurm <lr> <run_id>}

module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-$WORK/lichtheim3/lichtheim3}
cd "$REPO"
HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "== repo $REPO @ $HEAD ($(git branch --show-current 2>/dev/null || echo detached))"
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked working tree dirty"; exit 1; }
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }
if [[ -n "${L3_EXPECTED_COMMIT:-}" ]]; then
    [[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD != L3_EXPECTED_COMMIT"; exit 1; }
fi

# ---- immutable source: the CAP-1 H512@100 checkpoint ----------------------
SOURCE_DIR=${L3_CAP2_SOURCE:-$WORK/lichtheim3/cap2_source}
SOURCE_CKPT="$SOURCE_DIR/step_00046300.pt"
SOURCE_SHA_EXPECTED="b49da0aab5f68b1bf286a7af99ed60f5199951bce159e359d6b51ed1aacbe233"
[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: source $SOURCE_CKPT not found — transfer it first"; exit 1; }
SOURCE_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
[[ "$SOURCE_SHA" == "$SOURCE_SHA_EXPECTED" ]] || {
    echo "FATAL: source sha $SOURCE_SHA != expected $SOURCE_SHA_EXPECTED — transfer corrupt"; exit 1; }
echo "== source OK: $SOURCE_CKPT sha256 $SOURCE_SHA"

# ---- configuration --------------------------------------------------------
WIDTH=512
SEED=22
DEVICE=cuda
MAX_EXPOSURES=3000
EVALS=0,100,125,150,175,200,250,300,400,500,750,1000,1500,2000,2500,3000
RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

case "$RUN_ID" in
    cap2_*) : ;;
    *) echo "FATAL: run id must start with cap2_ (never a FINAL-7L/9P directory)"; exit 1 ;;
esac

# ---- short V100 preflight benchmark (fresh model, no state touched) -------
python scripts/naming_comprehension/naming_capacity_sweep.py \
    --width "$WIDTH" --seed "$SEED" --device "$DEVICE" \
    --out-dir "$JOBSCRATCH" --benchmark 200

# ---- first launch vs requeue ---------------------------------------------
# First launch : resume the immutable source and DECLARE the lr transition.
# Requeue      : resume this branch's own newest milestone checkpoint; its
#                stored lr already equals $LR_ARG, so nothing is re-declared
#                (the driver refuses an undeclared change, so passing the
#                wrong LR to a requeue fails loudly instead of silently).
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$SOURCE_CKPT"
    EXTRA=(--lr-transition)
    echo "== first launch: branching from the immutable source @100 exposures"
    echo "== declaring lr 1e-3 -> $LR_ARG (AdamW moments preserved)"
else
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming $RUN_ID from $(basename "$OWN_LATEST")"
fi

srun python scripts/naming_comprehension/naming_capacity_sweep.py \
    --width "$WIDTH" --seed "$SEED" --device "$DEVICE" \
    --lr "$LR_ARG" \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --eval-exposures "$EVALS" --max-exposures "$MAX_EXPOSURES" \
    --resume "$RESUME_FROM" --log-every 4630 "${EXTRA[@]}"

echo "== CAP-2 branch $RUN_ID finished: $RUNS/$RUN_ID"
echo "== grep 'CEILING REACHED' to see whether exact 1.0 was hit"
echo "== provenance (parent sha, lr transition, moments): $RUNS/$RUN_ID/config.json"
echo "== archive: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_cap2/"
