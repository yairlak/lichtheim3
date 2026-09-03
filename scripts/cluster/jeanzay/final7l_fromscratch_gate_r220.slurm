#!/bin/bash
#SBATCH --job-name=l3_final7l_gate
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# FINAL-7L GATE — the full recipe FROM RANDOM INITIALISATION, seed 22, to R220.
#
# The scientific recipe is PREDECLARED BEFORE STEP 0 and runs to R700:
#
#     random init  ->  shared_adamw @ LR 1e-3  ->  R100
#                  ->  grouped_rn_c_adamw @ LR 1e-4  ->  R700
#
# R220 is an OPERATIONAL inspection gate, not a scientific change.  Stopping
# here to inspect and checkpoint, then resuming the same run unchanged
# overnight (final7l_continue_r220_to_r700.slurm), does not alter the recipe.
# That is recorded in this run's provenance.
#
# NO historical checkpoint is ever a training source.  The FINAL-3 R100 file
# is used ONCE, read-only, for a replay COMPARISON after stage 1, and is
# never substituted for this run's own state.
#
# Stage 1  random init -> R100  (step 277,800): shared_adamw, LR 1e-3, which
#          the historical two-stage policy switches to 1e-4 exactly at
#          repetition cursor 46,300 = R100.
# Stage 2  R100 -> R220 (step 611,160): resumes THIS run's own R100
#          checkpoint and declares shared_adamw -> grouped_rn_c_adamw once,
#          cloning this run's shared moments into the RN and C banks.
#
#     RN bank : repetition (with its 0.5 pool CE) and naming
#     C bank  : comprehension
#
#   R100 = 277,800 steps = 200 N exp, ~317.12 C exp   (full eval + stage end)
#   R130 = 361,140 steps = 260 N exp, ~412.26 C exp   (full eval)
#   R160 = 444,480 steps = 320 N exp, ~507.40 C exp   (full eval)
#   R220 = 611,160 steps = 440 N exp, ~697.67 C exp   (full eval, GATE STOP)
#
# Frozen: architecture (hidden 128), final_full populations (R 29,571 /
# N 29,571 / C 27,981 canonical / bank 29,571), interleaved 1:2:3,
# deterministic macro-cycle, seed and stream seeds, historical repetition
# objective with 0.5 pool CE on R, lambda_C 0.087, lambda_N 1.0,
# c_align_weight 0, tau 0.10, batch 64, clip 1.0, wd 1e-5, all parameters
# trainable, evaluation definitions.  NO task-specific LR, no 3e-4, no
# post-R100 1e-3, no further decay.
#
# Requeue-safe at every point; a walltime timeout is recoverable by
# resubmitting, which resumes this run's own newest checkpoint.
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

# ---- FINAL-7L configuration ----------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
DEVICE=cuda
GROUPED=grouped_rn_c_adamw

R100_STEP=277800                              # stage 1 end / transition point
GATE_STEP=611160                              # R220, gate hard stop
STAGE1_EPOCHS=100                             # -> 277,800 steps
GATE_EPOCHS=220                               # -> 611,160 steps
STAGE1_FULL_EVAL=277800                       # R100
STAGE2_FULL_EVAL=361140,444480,611160         # R130 / R160 / R220

# Dev cadence every 10 R exposures (not 5): this run is 611,160 steps and the
# 3 h budget is protected by halving evaluation overhead.  Safety checkpoints
# at the same cadence.
EVAL_EVERY=27780
SAVE_EVERY=27780

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"
CKPTS="$RUNS/$RUN_ID/checkpoints"

# Read-only reference for the post-stage-1 replay check.  NEVER a training
# source; it is not passed to any training command.
HISTORICAL_R100_FOR_COMPARISON_ONLY="$RUNS/final3p_i123_seed${SEED}_${SUBSET_MODE}/checkpoints/step_00277800.pt"

COMMON=(--regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE"
        --schedule "$SCHEDULE" --device "$DEVICE"
        --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY"
        --log-every 500 --endpoint-eval
        --out-dir "$RUNS" --run-id "$RUN_ID")

# ---- where are we? --------------------------------------------------------
# Newest by STEP NUMBER (zero-padded names sort numerically), never by mtime.
LATEST=$(ls "$CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
if [[ -z "${LATEST:-}" ]]; then
    LATEST_STEP=-1
else
    LATEST_STEP=$(basename "$LATEST" .pt); LATEST_STEP=${LATEST_STEP#step_}
    LATEST_STEP=$((10#$LATEST_STEP))
fi

if (( LATEST_STEP >= GATE_STEP )); then
    echo "== nothing to do: the gate already reached R220 (step $LATEST_STEP)"; exit 0
fi

# ---- STAGE 1: random init -> R100, shared optimizer ----------------------
if (( LATEST_STEP < R100_STEP )); then
    if (( LATEST_STEP < 0 )); then
        echo "== STAGE 1: starting FROM RANDOM INITIALISATION (seed $SEED)"
        RESUME_ARGS=()
    else
        # Requeue inside stage 1: continue this run's own shared checkpoint.
        # Never restart from scratch once any checkpoint exists.
        echo "== STAGE 1 requeue: resuming own shared checkpoint at step $LATEST_STEP"
        RESUME_ARGS=(--resume "$LATEST")
    fi
    srun python scripts/naming_comprehension/train_joint_scratch.py \
        "${COMMON[@]}" --epochs "$STAGE1_EPOCHS" \
        --full-eval-at "$STAGE1_FULL_EVAL" "${RESUME_ARGS[@]}"

    # ---- R100 replay check: VALIDATION ONLY, never fatal, never substitutes
    NEW_R100="$CKPTS/step_$(printf '%08d' $R100_STEP).pt"
    if [[ -f "$NEW_R100" && -f "$HISTORICAL_R100_FOR_COMPARISON_ONLY" ]]; then
        echo "== R100 replay check against the historical run (comparison only)"
        python scripts/naming_comprehension/compare_checkpoints.py \
            --a "$NEW_R100" --b "$HISTORICAL_R100_FOR_COMPARISON_ONLY" \
            --json-out "$RUNS/$RUN_ID/r100_replay_check.json" || true
    else
        echo "== R100 replay check skipped (a checkpoint is unavailable)"
    fi
    LATEST_STEP=$R100_STEP
    LATEST="$CKPTS/step_$(printf '%08d' $R100_STEP).pt"
fi

# ---- STAGE 2: R100 -> R220, grouped RN|C --------------------------------
PHASE_ARGS=()
if (( LATEST_STEP == R100_STEP )); then
    PHASE_ARGS=(--phase-transition)
    echo "== STAGE 2 first launch: declaring shared_adamw -> $GROUPED at step $R100_STEP"
    echo "== moments are cloned from THIS run's own R100 state"
else
    echo "== STAGE 2 requeue: resuming own grouped checkpoint at step $LATEST_STEP"
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    "${COMMON[@]}" --epochs "$GATE_EPOCHS" \
    --optimizer-policy "$GROUPED" \
    --full-eval-at "$STAGE2_FULL_EVAL" \
    --resume "$LATEST" "${PHASE_ARGS[@]}"

echo "== FINAL-7L GATE complete (R220 / step $GATE_STEP): $RUNS/$RUN_ID"
echo "== replay check: $RUNS/$RUN_ID/r100_replay_check.json"
echo "== the overnight continuation resumes THIS run's own R220 checkpoint:"
echo "   sbatch --export=ALL,L3_EXPECTED_COMMIT final7l_continue_r220_to_r700.slurm"
