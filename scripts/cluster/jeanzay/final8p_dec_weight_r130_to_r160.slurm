#!/bin/bash
#SBATCH --job-name=l3_final8p
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
# FINAL-8P — ventral-decode weight: L_dec 0.5 -> 2.0.  ONE diagnostic
# training intervention, branched from the genuine from-scratch FINAL-7L run.
#
# HYPOTHESIS (H8): under RN|C the encoder/semantic surface is shaped for
# retrieval faster than the ventral production pathway stays co-adapted to
# s_hat, so LTM-route repetition erodes while comprehension rises.  L_dec is
# the only term that couples the two surfaces in the failing direction -- it
# backpropagates through the decoder AND, via s_hat, through the encoder --
# which makes its weight the clean single lever.
#
# WHY THIS AND NOT SOMETHING ELSE: FINAL-7A showed the C phenotype transfers
# almost exactly with the encoder/semantic surface A (7P+A_control -> C
# .076695; control+A_7P -> C .107037), while LTM is strongly NON-additive
# across A and B (.548612 / .388624 / .151229 / .242603), i.e. matched halves
# beat mismatched halves -- a co-adaptation signature rather than a single
# broken group.  CAUSAL HYGIENE: those were ENDPOINT transplants.  They
# localise endpoint dependence and motivated H8; they did not prove a
# training mechanism.  THIS run is the test.
#
# THE ONLY SCIENTIFIC DIFFERENCE from FINAL-7L is cfg.loss.dec 0.5 -> 2.0.
# Frozen: architecture, seed 22, populations, interleaved 1:2:3, task order,
# grouped_rn_c_adamw with RN = repetition+pool+naming and C = comprehension,
# both banks resumed exactly (no cloning, no reset), LR 1e-4 from the
# untouched two-stage policy, lambda_C 0.087, lambda_N 1.0, c_align 0,
# tau 0.10, every other historical R loss weight, pool CE, batch 64, clip 1.0,
# wd 1e-5, data streams, cursors and RNG.  Repetition frequency and optimizer
# topology are NOT touched.
#
#   source R130 = 361,140 steps = 260 N exp, ~412.26 C exp
#   R140        = 388,920 steps = 280 N exp, ~443.97 C exp
#   R150        = 416,700 steps = 300 N exp, ~475.68 C exp
#   R160        = 444,480 steps = 320 N exp, ~507.40 C exp   <- HARD STOP
#
# 83,340 steps ~= 19 min pure training at 0.013427 s/step; ~30 min wall with
# three full-population evaluations.  No automatic continuation.
#
# MATCHED CONTROL: FINAL-7L itself ran R130 -> R160 at dec 0.5 over the same
# steps, data and state.  Its R160 full-population values are read from that
# run's metrics.tsv (full LTM is the PRIMARY readout; FULL repetition is only
# a safety metric, because WM can mask ventral damage).
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

# ---- FINAL-8P configuration ----------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_123
OPT_POLICY=grouped_rn_c_adamw
DEC_WEIGHT=2.0                                # <-- the single intervention
DEVICE=cuda
EPOCHS=160                                    # REPETITION epochs -> 444,480 steps
FULL_EVAL_AT=388920,416700,444480             # R140 / R150 / R160
EVAL_EVERY=13890                              # dev cadence, 5 R exposures
SAVE_EVERY=27780                              # safety checkpoints, 10 R exp

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
PARENT_RUN_ID="final7l_fromscratch_seed${SEED}_${SUBSET_MODE}"
RUN_ID="final8p_dec2_r130_seed${SEED}_${SUBSET_MODE}"
SOURCE_STEP=361140
FINAL_STEP=444480

PARENT_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }
[[ -f "$PARENT_CKPT" ]] || { echo "FATAL: source checkpoint $PARENT_CKPT not found — FINAL-8P branches from FINAL-7L's own R130 and never starts a fresh run"; exit 1; }

# ---- first launch vs requeue ---------------------------------------------
# First launch : resume FINAL-7L's R130 and declare dec 0.5 -> 2.0 (an
#                optimizer-preserving loss-weight transition: no cloning, no
#                reset, cursors and RNG untouched).
# Requeue      : resume FINAL-8P's OWN newest checkpoint, no declaration.
# Newest by STEP NUMBER (zero-padded names sort numerically), not by mtime.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
PHASE_ARGS=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$PARENT_CKPT"
    PHASE_ARGS=(--phase-transition)
    echo "== first FINAL-8P launch: branching from $PARENT_RUN_ID at step $SOURCE_STEP"
    echo "== declaring L_dec weight 0.5 -> $DEC_WEIGHT (optimizer untouched)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=${OWN_STEP#step_}
    OWN_STEP=$((10#$OWN_STEP))
    if (( OWN_STEP >= FINAL_STEP )); then
        echo "== nothing to do: FINAL-8P already reached the R160 hard stop"; exit 0
    fi
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming FINAL-8P at step $OWN_STEP"
fi

srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" --optimizer-policy "$OPT_POLICY" \
    --dec-weight "$DEC_WEIGHT" \
    --device "$DEVICE" --epochs "$EPOCHS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$RESUME_FROM" "${PHASE_ARGS[@]}"

echo "== FINAL-8P complete (hard stop R160 / step $FINAL_STEP): $RUNS/$RUN_ID"
echo "== ancestry + dec transition: $RUNS/$RUN_ID/provenance.json"
echo "== PRIMARY readout is the LTM ROUTE; FULL repetition is only a safety"
echo "   metric because WM can mask ventral damage."
echo "== matched control (dec 0.5) at the same steps:"
echo "   awk -F'\\t' 'NR==1{for(i=1;i<=NF;i++)h[\$i]=i} \$1==444480' \\"
echo "       $RUNS/$PARENT_RUN_ID/metrics.tsv"
echo "== archive with: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_final8p/"
