#!/bin/bash
#SBATCH --job-name=l3_base123
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --requeue

# ===========================================================================
# FINAL base-123 — the first high-capacity joint baseline block.
#
# QUESTION: can increased VENTRAL CAPACITY ALONE move the original simple
# from-scratch shared-optimizer 1:2:3 joint regime into a successful R/N/C
# coexistence regime?  Historical H128 joint training failed and was patched
# with optimizer/schedule rescues (FINAL-6/7/9); the isolated capacity probes
# then showed the ventral route was severely under-capacity at 128.  This
# block therefore reverts to the AUDITED FINAL-3P recipe and changes only
# ventral width.
#
# 2 capacities x 4 historically predefined seeds = 8 runs, array 0-7:
#   idx 0-3 : enc=dec=256, seeds 19,20,21,22
#   idx 4-7 : enc=dec=512, seeds 19,20,21,22
#
# WM IS FIXED AT 128 in every arm: the dorsal probe found no gain above it
# (lex .999865 / pseudo .980 at 128 vs .9475 pseudo at 512).  Encoder and
# decoder are scaled TOGETHER, so the factor is a ventral capacity BUNDLE --
# this block does not separate encoder from decoder effects.
#
# AUDITED RECIPE, held constant across all eight runs (FINAL-3P lineage,
# commit 73982f3; verified to be the driver's defaults at this commit):
#   - from scratch, no source checkpoint, no inherited optimizer state
#   - ONE shared AdamW over all parameters for R, N and C
#   - interleaved 1:2:3, six-step macro-cycles, exactly 1 R + 2 N + 3 C,
#     order = deterministic shuffle of (schedule seed, cycle index)
#   - R sampler log-frequency weighted WITH REPLACEMENT (freq_temp 1.0);
#     N, C and the dorsal pool unweighted permutations
#   - R loss = historical total_loss (rep 1.0, align 1.0, dec 0.5, wm 0.5,
#     gate 0.05) + 0.5 * dorsal-pool CE, the pool riding every R step
#   - N loss = 1.0 * naming CE (teacher forced)
#   - C loss = 0.087 * full-bank retrieval CE, tau 0.10, c_align 0.0
#   - LR 1e-3 -> 1e-4 on the REPETITION CURSOR at 46,300 completed R batches
#     (= 100 R exposures), AdamW moments preserved across the change
#   - batch 64, clip 1.0 per task step, wd 1e-5, noise 0.0
#   - populations: R 29,571 / N 29,571 / C 27,981 canonical / bank 29,571
#
# LR-CLOCK CAVEAT, deliberately NOT corrected here (see the preregistration):
# the clock is REPETITION progress, so at the drop N has had ~200 and C ~317
# exposures at 1e-3.  Isolated H512 naming destabilised under sustained 1e-3.
# A naming rise-then-decay before u=100 with rising N CE, recovering after the
# drop, must be read FIRST as an LR-schedule signature, not as evidence that
# joint learning is intrinsically incompatible.
#
# BUDGET: u = R exposures.  u_max 500 = 231,500 macro-cycles = 1,389,000
# optimizer steps.  Note C exposures are 3u x (463/438), i.e. 1585.62 at
# u=500, NOT 1500: 1:2:3 is a presentation/update ratio and the C population
# is smaller.  No automatic continuation beyond u=500.
#
# STOPPING: only when TWO CONSECUTIVE scheduled full evaluations have
# canonical repetition, GENUINE free-AR repetition, free-AR naming and strict
# canonical C top-1 all exactly 1.0.  Any shortfall resets the streak, and the
# streak is checkpointed so a requeue cannot restart it.
#
# Usage:  sbatch --export=ALL,L3_EXPECTED_COMMIT=<sha> final_base123_production.slurm
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-/lustre/fswork/projects/rech/llg/uss35bp/lichtheim3/lichtheim3}
cd "$REPO"
HEAD=$(git rev-parse HEAD)
echo "== repo $REPO @ $HEAD ($(git branch --show-current))"
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked tree dirty"; exit 1; }
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || { echo "FATAL: export L3_EXPECTED_COMMIT"; exit 1; }
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD != L3_EXPECTED_COMMIT"; exit 1; }

# ---- explicit, collision-proof array mapping -----------------------------
WIDTHS=(256 256 256 256 512 512 512 512)
SEEDS=( 19  20  21  22  19  20  21  22)
IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 8 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
VENTRAL=${WIDTHS[$IDX]}
SEED=${SEEDS[$IDX]}
RUN_ID="final_base123_h${VENTRAL}_s${SEED}"

WM=128                                   # dorsal FIXED in every arm
ENC=$VENTRAL
DEC=$VENTRAL
U_MAX=500
MAX_STEPS=1389000                        # 500 x 463 x 6
FULL_EVAL_AT=0,69450,138900,277800,416700,555600,833400,1111200,1389000
EVAL_EVERY=13890                         # dev cadence, every 5 R exposures
SAVE_EVERY=69450                         # safety checkpoints, every u=25
RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

echo "== array $IDX -> $RUN_ID  (wm=$WM enc=$ENC dec=$DEC seed=$SEED)"
echo "== u_max=$U_MAX -> $MAX_STEPS steps; R=$U_MAX N=$((2*U_MAX)) C~1585.62 exposures"

case "$RUN_ID" in
    final_base123_*) : ;;
    *) echo "FATAL: refusing an unexpected run id"; exit 1 ;;
esac
for bad in final1_ final2a_ final3p_ final4_ final5p_ final6 final7 final8p_ final9 cap1_ cap2_ cap3_ cap4_; do
    [[ "$RUN_ID" != ${bad}* ]] || { echo "FATAL: run id collides with an existing campaign"; exit 1; }
done

# ---- requeue-safe resume from this run's own newest checkpoint ------------
# The u=0 full evaluation is produced by --eval-at-start together with the 0
# entry in FULL_EVAL_AT: the in-loop milestone check runs after train_step()
# and so can never observe step 0.  A requeue continues from the newest
# checkpoint by STEP NUMBER (zero-padded names sort numerically).
# Widths are pinned identically on resume, and the driver refuses a
# checkpoint whose architecture differs, so a mis-mapped array index cannot
# silently continue the wrong run.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
RESUME=()
if [[ -n "${OWN_LATEST:-}" ]]; then
    STEP=$(basename "$OWN_LATEST" .pt); STEP=$((10#${STEP#step_}))
    if (( STEP >= MAX_STEPS )); then
        echo "== nothing to do: $RUN_ID already reached u=$U_MAX"; exit 0
    fi
    RESUME=(--resume "$OWN_LATEST")
    echo "== requeue: resuming $RUN_ID at step $STEP"
else
    echo "== first launch: training $RUN_ID FROM SCRATCH (no source checkpoint)"
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed "$SEED" --subset-mode final_full \
    --schedule interleaved_123 \
    --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC" \
    --device cuda --max-steps "$MAX_STEPS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 500 \
    --eval-at-start --stop-at-ceiling --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" "${RESUME[@]}"

echo "== $RUN_ID finished: $RUNS/$RUN_ID"
echo "== grep MILESTONE for the u-grid full evaluations; STOP for the ceiling rule"
echo "== route health: full_rep_ltm, gate_mean, gate_frac_* in metrics.tsv --"
echo "   a 100/100/100 model with dead LTM repetition or a saturated gate is"
echo "   NOT a lesion-ready dual-route model and must be flagged."
