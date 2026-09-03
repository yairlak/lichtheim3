#!/bin/bash
#SBATCH --job-name=l3_final6p
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
# FINAL-6P — task-separated AdamW moment banks.
#
# Hypothesis under test: with ONE shared AdamW, the exp_avg / exp_avg_sq
# histories on shared parameters are built from interleaved gradients of all
# tasks, so the supposedly dedicated R/N/C steps ride a common optimizer
# memory.  The R150 audit is consistent with this: raw task gradients are
# near-orthogonal (R~C and R~N cosines ~0) while the ACTUAL AdamW deltas are
# strongly co-aligned (+0.73 and +0.84).  This is a hypothesis, not an
# established cause; FINAL-6P tests it directly.
#
#   Phase 1 (frozen history): random init -> R100 under the FINAL-3 recipe,
#            ending at step 277,800 of final3p_i123_seed22_final_full.
#   Phase 2 (this job): from step 277,800 onward, ONE shared network with
#            THREE AdamW banks (R / N / C) over the SAME Parameter objects.
#            Each bank owns its exp_avg, exp_avg_sq and per-parameter step
#            counters; only the current task's bank steps.  The R bank owns
#            the repetition step including the attached pool loss.
#
# MOMENT INITIALISATION: at the transition every bank is loaded with a
# bitwise DEEP COPY of the inherited shared state -- moments and step counters
# preserved, nothing zeroed, no aliasing between banks.  So this pilot asks
# what happens when optimizer memories become task-specific FROM R100 onward
# after sharing one history up to it.  It is NOT equivalent to training with
# separated banks from step 0; if it succeeds the production recipe must be
# replayed from random initialisation with separated banks from the start.
#
# LEARNING RATES: the LR policy is deliberately NOT touched.  After R100 the
# historical two-stage schedule already yields 1e-4 for every task, which is
# exactly the strongest post-R100 control, so passing no --lr-* flag makes
# optimizer-state separation the ONLY difference from that control.
#
# Matched control from the SAME R100 checkpoint (shared AdamW, all 1e-4):
#     R110  rep .999763  naming .048696  C top1 .072228
#     R130  rep .999865  naming .054276  C top1 .076695
#
# NEW RUN DIRECTORY: the parent already holds a different post-R100 history
# and FINAL-5P holds another; this branch writes its own metrics, losses and
# checkpoints, records its ancestry, and only ever READS the parent.
#
#   source R100 = 277,800 steps = 200 N exp, ~317.12 C exp
#   M6P-1  R110 = 305,580 steps = 220 N exp, ~348.84 C exp
#   M6P-2  R120 = 333,360 steps = 240 N exp, ~380.55 C exp
#   M6P-3  R130 = 361,140 steps = 260 N exp, ~412.26 C exp   <- HARD STOP
#
# Everything else frozen: interleaved 1:2:3, deterministic macro-cycle,
# seed 22, stream seeds, lambda_C 0.087, lambda_N 1.0, c_align_weight 0,
# tau 0.10, all three objectives, pool CE at 0.5 in the same backward,
# clip 1.0, batch 64, full populations, architecture, all parameters
# trainable, evaluation definitions, cursors and RNG.
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

# ---- FINAL-6P configuration ----------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
OPT_POLICY=task_separated_adamw
DEVICE=cuda
EPOCHS=130                                    # REPETITION epochs -> 361,140 steps
FULL_EVAL_AT=305580,333360,361140             # M6P-1 / M6P-2 / M6P-3
EVAL_EVERY=13890                              # unchanged dev cadence (5 R exp)
SAVE_EVERY=27780                              # unchanged safety cadence (10 R exp)

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
PARENT_RUN_ID="final3p_i123_seed${SEED}_${SUBSET_MODE}"
RUN_ID="final6p_r100_sepmoments_seed${SEED}_${SUBSET_MODE}"
PARENT_STEP=277800
FINAL_STEP=361140

PARENT_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $PARENT_STEP).pt"
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }
[[ -f "$PARENT_CKPT" ]] || { echo "FATAL: parent checkpoint $PARENT_CKPT not found — FINAL-6P branches from it and never starts a fresh run"; exit 1; }

# ---- first launch vs requeue ---------------------------------------------
# First launch : resume the EXTERNAL shared-AdamW parent at R100 and declare
#                the shared -> task_separated transition (which clones the
#                shared state into all three banks).
# Requeue      : resume FINAL-6P's OWN newest checkpoint, which already holds
#                three distinct banks -- no re-declaration and no re-cloning.
OWN_LATEST=$(ls -t "$OWN_CKPTS/"step_*.pt 2>/dev/null | head -1 || true)
PHASE_ARGS=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$PARENT_CKPT"
    PHASE_ARGS=(--phase-transition)
    echo "== first FINAL-6P launch: branching from $PARENT_RUN_ID at step $PARENT_STEP"
    echo "== declaring shared_adamw -> task_separated_adamw (clone into 3 banks)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=${OWN_STEP#step_}
    OWN_STEP=$((10#$OWN_STEP))
    if (( OWN_STEP >= FINAL_STEP )); then
        echo "== nothing to do: FINAL-6P already reached the R130 hard stop"; exit 0
    fi
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming FINAL-6P's own three-bank checkpoint at step $OWN_STEP"
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" \
    --optimizer-policy "$OPT_POLICY" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$RESUME_FROM" "${PHASE_ARGS[@]}"

echo "== FINAL-6P complete (hard stop R130 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== ancestry + phase record: $RUNS/$RUN_ID/provenance.json"
echo "== bank divergence is logged as m_div_RN / m_div_RC / m_div_NC in metrics.tsv"
echo "== compare against the matched shared-AdamW control (all 1e-4 after R100):"
echo "   python scripts/naming_comprehension/analyze_joint_dynamics.py \\"
echo "       --run $RUNS/$RUN_ID --baseline $RUNS/$PARENT_RUN_ID"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final6p/"
