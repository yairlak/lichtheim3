#!/bin/bash
#SBATCH --job-name=l3_final9p
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
#SBATCH --requeue

# ===========================================================================
# FINAL-9P — repetition FREQUENCY: task ratio 1:2:3 -> 2:2:3.  ONE change.
#
# HYPOTHESIS (H9): under RN|C the comprehension trajectory moves the
# encoder/semantic surface faster than repetition can maintain the ventral
# mapping, because there is only ONE repetition update per two naming and
# three comprehension updates.  Doubling genuine R updates should supply more
# maintenance AFTER clipping, and more frequent RN-bank state updates, while
# leaving the comprehension mechanism itself untouched.
#
# WHY FREQUENCY AND NOT WEIGHT: FINAL-8P raised L_dec 0.5 -> 2.0 from this
# same R130 state and moved full LTM by only +.002029 (.169220 -> .171249),
# with 69/69 logged repetition steps clipped at a median pre-clip norm of
# 10.47 against a threshold of 1.0.  Increasing one component inside an
# already saturated, always-clipped step is not an effective way to add
# maintenance pressure.  FINAL-9P therefore buys it by the STEP.
# NOTE: FINAL-8P's dec=2.0 is deliberately NOT carried over -- L_dec is back
# at its canonical 0.5, so this is a pure frequency experiment.
#
# MATCHED DESIGN -- THIS IS THE POINT OF THE RUN.  The comparison is NOT at a
# matched global step: changing the macro-cycle changes how many optimizer
# steps a given amount of learning costs.  The run stops when NAMING and
# COMPREHENSION have had exactly the control's acquisition opportunity --
#
#     N = 320 exposures,  C = 507.3972603 exposures
#
# -- by which point repetition has had 190 exposures instead of the control's
# 160.  Acquisition opportunity is held fixed; only maintenance opportunity
# increases.  The task streams are counter-addressed, so at equal cursors each
# task has seen exactly the control's items in the control's order; only the
# interleaving and the number of R updates differ.
#
#   source R130 = step 361,140 = N 260, C 412.2603  (from FINAL-7L, read-only)
#   M1 = step 393,550 = R 150, N 280, C 443.9726   vs control step 388,920 (R140)
#   M2 = step 425,960 = R 170, N 300, C 475.6849   vs control step 416,700 (R150)
#   M3 = step 458,370 = R 190, N 320, C 507.3973   vs control step 444,480 (R160)
#
# 97,230 added steps (13,890 cycles of seven).  PRIMARY comparison: FINAL-9P
# M3 against the FINAL-7L control at R160 -- full LTM .169220, FULL rep
# .999966, naming .055122, C top1 .118902, C top5 .256495.  The PRIMARY
# readout is the LTM route; FULL repetition is only a safety metric because WM
# can mask ventral damage.  Also worth watching: whether the extra repetition
# steps harm naming despite sharing the RN bank.
#
# Frozen: architecture, seed 22, final_full populations, grouped_rn_c_adamw
# with RN = repetition+pool+naming and C = comprehension (both banks resumed
# exactly, no cloning or reset), LR 1e-4, L_dec 0.5, every other repetition
# weight, lambda_C 0.087, lambda_N 1.0, c_align 0, tau 0.10, batch 64,
# clip 1.0, wd 1e-5, task-local stream seeds, cursors, RNG, evaluation
# definitions.
#
# The run is budgeted in STEPS, not epochs: 463 R batches at two per seven-step
# cycle is not a whole number of cycles, and the driver refuses that rather
# than rounding.
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

# ---- FINAL-9P configuration ----------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_223                      # <-- the single intervention
OPT_POLICY=grouped_rn_c_adamw
DEVICE=cuda
MAX_STEPS=458370                              # matched-N/C endpoint (R190)
FULL_EVAL_AT=393550,425960,458370             # M1 / M2 / M3
EVAL_EVERY=16205                              # 2315 cycles = 10 R / 10 N exp
SAVE_EVERY=32410                              # 4630 cycles = 20 R / 20 N exp

# CADENCES.  Both fire on ABSOLUTE step multiples.  Because gcd(361140, 32410)
# is 4630 and 4630 is not a multiple of the seven-step cycle, no cadence can
# both land on the milestones and be a whole number of cycles; the milestones
# do not need it, because --full-eval-at checkpoints unconditionally at each
# of them.  So the cadences are chosen for what they are actually for:
#   EVAL_EVERY and SAVE_EVERY are both multiples of 7, so every dev evaluation
#   and every safety checkpoint falls at the SAME fixed phase of the
#   macro-cycle (position 4 after the anchor).  The dev curve is therefore
#   read at a constant phase instead of drifting across R/N/C positions.
#   Dev evals land at 372715, 388920, 405125, 421330, 437535, 453740.
#   Safety checkpoints land at 388920, 421330, 453740 -- one shortly before
#   each milestone, capping requeue loss at ~32410 steps.
# Cycle position is computed as (global_step - schedule_anchor_step) mod 7, so
# a mid-cycle checkpoint resumes at exactly the right position in the cycle.

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
PARENT_RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"
RUN_ID="final9p_r223_r130_seed${SEED}_${SUBSET_MODE}"
SOURCE_STEP=361140
FINAL_STEP=458370

PARENT_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }
[[ -f "$PARENT_CKPT" ]] || { echo "FATAL: source checkpoint $PARENT_CKPT not found — FINAL-9P branches from FINAL-7L's own R130 and never starts a fresh run"; exit 1; }

# ---- first launch vs requeue ---------------------------------------------
# First launch : resume FINAL-7L's R130 and declare interleaved_123 ->
#                interleaved_223.  The new macro-cycle count is anchored at
#                the transition step, so the post-transition task order is
#                well defined and reproduced exactly on any later resume.
# Requeue      : resume FINAL-9P's OWN newest checkpoint; the stored anchor
#                is carried forward and nothing is re-declared.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
PHASE_ARGS=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$PARENT_CKPT"
    PHASE_ARGS=(--phase-transition)
    echo "== first FINAL-9P launch: branching from $PARENT_RUN_ID at step $SOURCE_STEP"
    echo "== declaring schedule interleaved_123 -> $SCHEDULE (optimizer untouched)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=${OWN_STEP#step_}
    OWN_STEP=$((10#$OWN_STEP))
    if (( OWN_STEP >= FINAL_STEP )); then
        echo "== nothing to do: FINAL-9P already reached the matched endpoint"; exit 0
    fi
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming FINAL-9P at step $OWN_STEP"
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" --optimizer-policy "$OPT_POLICY" \
    --device "$DEVICE" --max-steps "$MAX_STEPS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$RESUME_FROM" "${PHASE_ARGS[@]}"

echo "== FINAL-9P complete (matched endpoint N320 / C507.3973, R190): $RUNS/$RUN_ID"
echo "== ancestry + schedule transition: $RUNS/$RUN_ID/provenance.json"
echo "== compare at MATCHED N/C, never at matched global step:"
echo "   FINAL-9P step 458370  vs  FINAL-7L step 444480"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final9p/"
