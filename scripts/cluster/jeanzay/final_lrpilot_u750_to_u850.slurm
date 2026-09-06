#!/bin/bash
#SBATCH --job-name=l3_lrpilot
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --requeue

# ===========================================================================
# LR PILOT — is 1e-4 now too high for final convergence?  Paired branch from
# the archived H512 u750 checkpoints.  ONE treatment factor: the LR.
#
#   array 0-3 : CONTROL   seeds 19,20,21,22   LR 1e-4  (unchanged)
#   array 4-7 : STAGE-3   seeds 19,20,21,22   LR 3e-5
#
# WHY.  At u750 the H512 arm is at C .9913-.9923 (233/215/239/243 errors of
# 27,981) and the residual is still MOVING, not a fixed core: u600->u750
# within-seed retention is only .5868, the four-seed sentinel core shrank
# 27 -> 16 -> 12, naming residuals churn completely (intersection 0 in every
# seed between u600 and u750), and there are no mathematically unavoidable C
# errors.  The preregistered margin / hard-negative trigger is NOT met.  What
# HAS changed is the rate: the exact u700->u750 residual ratio is 0.87835,
# against 0.600 for u300->u400 and 0.667 for u400->u500.  A frontier that
# keeps moving while progress decelerates and R/N/C all fluctuate near
# ceiling is the signature of a step size that has become too large for the
# final approach -- which is what this pilot tests.
#
# IMPLEMENTATION.  Both arms use the EXISTING task-specific LR policy with
# all three tasks set to the same value, declared once with
# --phase-transition.  No new optimization machinery was added.  Setting the
# three task LRs equal makes the policy a flat constant LR, and the CONTROL
# at 1e-4 is BITWISE IDENTICAL to simply continuing under the historical
# two-stage policy (whose stage 2 is 1e-4, and whose repetition cursor at
# u750 is 347,250, far past the 46,300 boundary).  That equivalence is
# verified in tests/test_lrpilot_u850.py, so the control is provably "just
# keep going" and the two arms differ only in a numeric LR.
#
# The transition NEVER resets optimizer state: load_state_dict restores the
# single shared AdamW and every moment, then the per-step LR is applied to
# the param groups.  RNG, all four stream cursors, the schedule anchor and
# the ceiling streak are restored unchanged.
#
# BRANCH SAFETY.  The canonical final_base123_h512_sXX directories are
# ARCHIVED AND IMMUTABLE.  These branches write to their own run ids and the
# job refuses any run id that is not final_lrpilot_*.  The source checkpoint
# is opened read-only and its SHA256 is logged and recorded in provenance.
#
# CEILING RULE unchanged: 5 DISTINCT scheduled full evaluations at strict
# ceiling.  The persisted streak at u750 is 0 and this pilot has only four
# milestones, so the rule CANNOT fire here.  Running to u850 is the expected
# outcome, not a failure.
#
# Usage: sbatch --export=ALL,L3_REPO,L3_EXPECTED_COMMIT \
#            scripts/cluster/jeanzay/final_lrpilot_u750_to_u850.slurm
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
module load git

[[ -n "${L3_REPO:-}" ]] || {
    echo "FATAL: L3_REPO must be exported to the pinned worktree"; exit 1; }
REPO="$L3_REPO"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || {
    echo "FATAL: L3_REPO=$REPO is not a git worktree"; exit 1; }
cd "$REPO"
HEAD=$(git rev-parse HEAD)
echo "== repo $REPO @ $HEAD"
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked tree dirty"; exit 1; }
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || { echo "FATAL: export L3_EXPECTED_COMMIT"; exit 1; }
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD != L3_EXPECTED_COMMIT"; exit 1; }

TRAIN_BLOB_EXPECTED=24742adf9f13c205673a3f8ff1f34bdce7a45e58
GOT_BLOB=$(git rev-parse HEAD:scripts/naming_comprehension/train_joint_scratch.py)
[[ "$GOT_BLOB" == "$TRAIN_BLOB_EXPECTED" ]] || {
    echo "FATAL: driver blob $GOT_BLOB != expected $TRAIN_BLOB_EXPECTED"; exit 1; }
echo "== training driver blob verified: $GOT_BLOB"

# ---- explicit, collision-proof array mapping -----------------------------
ARMS=(control control control control 3e5 3e5 3e5 3e5)
LRS=(1e-4 1e-4 1e-4 1e-4 3e-5 3e-5 3e-5 3e-5)
SEEDS=(19 20 21 22 19 20 21 22)
IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 8 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
ARM=${ARMS[$IDX]}; LR=${LRS[$IDX]}; SEED=${SEEDS[$IDX]}

WM=128; ENC=512; DEC=512
SOURCE_STEP=2083500                      # u750 = 750 x 463 x 6
MAX_STEPS=2361300                        # u850 = 850 x 463 x 6
FULL_EVAL_AT=2152950,2222400,2291850,2361300   # u775, u800, u825, u850
EVAL_EVERY=13890                         # unchanged
SAVE_EVERY=69450                         # unchanged; every milestone is a multiple
CEILING_REQUIRED=5                       # unchanged strengthened rule

RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
PARENT_RUN_ID="final_base123_h512_s${SEED}"
RUN_ID="final_lrpilot_${ARM}_h512_s${SEED}"
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"
SOURCE_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"

# The canonical lineages are immutable: a branch may only ever write to its
# own final_lrpilot_* directory.
case "$RUN_ID" in
    final_lrpilot_control_h512_s19|final_lrpilot_control_h512_s20|\
final_lrpilot_control_h512_s21|final_lrpilot_control_h512_s22|\
final_lrpilot_3e5_h512_s19|final_lrpilot_3e5_h512_s20|\
final_lrpilot_3e5_h512_s21|final_lrpilot_3e5_h512_s22) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUN_ID" != final_base123_* ]] || { echo "FATAL: would write into the canonical lineage"; exit 1; }
[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }

echo "== array $IDX -> $RUN_ID   arm=$ARM  LR=$LR  seed=$SEED"
echo "== branching from $PARENT_RUN_ID at u750 (step $SOURCE_STEP) -> u850"

[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: u750 source $SOURCE_CKPT not found"; exit 1; }
SOURCE_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
echo "== source: $SOURCE_CKPT"
echo "== source sha256: $SOURCE_SHA"
if [[ -n "${L3_U750_SHA:-}" ]]; then
    [[ "$SOURCE_SHA" == "$L3_U750_SHA" ]] || {
        echo "FATAL: source sha != L3_U750_SHA"; exit 1; }
fi

# ---- read-only preflight on the shared source ----------------------------
L3_CKPT="$SOURCE_CKPT" L3_STEP="$SOURCE_STEP" python - <<'PY'
import os, sys, torch
ck = torch.load(os.environ["L3_CKPT"], map_location="cpu", weights_only=False)
want = int(os.environ["L3_STEP"])
def need(c, m):
    if not c:
        print(f"FATAL preflight: {m}", file=sys.stderr); sys.exit(1)
need(ck.get("format") == "lichtheim3.joint_scratch.v1", "unexpected format")
need(int(ck["global_step"]) == want, f"step {ck['global_step']} != {want}")
need(ck.get("widths") == {"wm_hidden": 128, "ltm_enc_hidden": 512,
                          "ltm_dec_hidden": 512}, f"widths {ck.get('widths')}")
need(ck.get("schedule") == "interleaved_123", f"schedule {ck.get('schedule')!r}")
need(list(ck.get("schedule_ratio") or []) == [1, 2, 3], "ratio is not 1:2:3")
need(ck.get("optimizer_policy") == "shared_adamw", "not the single shared bank")
need("optimizer_state_dict" in ck and "optimizer_states" not in ck,
     "checkpoint does not carry exactly one shared AdamW state")
need(ck["lr_policy"]["kind"] == "two_stage_rep_cursor",
     f"source lr policy {ck['lr_policy']['kind']!r}")
need(abs(float(ck.get("lr")) - 1e-4) < 1e-12, f"source lr {ck.get('lr')} != 1e-4")
need(float(ck.get("c_align_weight") or 0.0) == 0.0, "c_align is not 0")
need(float(ck.get("dec_weight")) == 0.5, f"dec_weight {ck.get('dec_weight')}")
need(ck.get("subset_mode") == "final_full", "subset_mode")
need(int(ck.get("comprehension_population_n", 0)) == 27981, "C population")
need(ck.get("comprehension_population_sha256") ==
     "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
     "canonical C hash mismatch")
need(int(ck["cursors"]["repetition"]) == 347250,
     f"repetition cursor {ck['cursors']['repetition']} != 347250 (u750)")
e = ck["exposures"]
print(f"== preflight OK  step {ck['global_step']}  u={ck['global_step']/(463*6):.1f}")
print(f"   cursors R {ck['cursors']['repetition']} N {ck['cursors']['naming']} "
      f"C {ck['cursors']['comprehension']} pool {ck['cursors']['pool']}")
print(f"   exposures R {e['repetition']:.4f} N {e['naming']:.4f} C {e['comprehension']:.4f}")
print(f"   moments {len(ck['optimizer_state_dict']['state'])} tensors  "
      f"streak {ck.get('consecutive_ceiling')}  lr {ck.get('lr')}")
print(f"   written by commit {(ck.get('git') or {}).get('commit')}")
PY

# ---- first launch (branch, declare the LR) vs requeue (own checkpoint) ----
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$SOURCE_CKPT"
    EXTRA=(--phase-transition)
    echo "== first launch: branching the shared u750 state, declaring LR -> $LR"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=$((10#${OWN_STEP#step_}))
    if (( OWN_STEP >= MAX_STEPS )); then
        echo "== nothing to do: $RUN_ID already reached u850"; exit 0
    fi
    (( OWN_STEP >= SOURCE_STEP )) || { echo "FATAL: own checkpoint below the source"; exit 1; }
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming $RUN_ID at step $OWN_STEP (LR already $LR, nothing re-declared)"
fi

# The three task LRs are set to the SAME value, which makes the existing
# task-specific policy a flat constant LR.  No other flag differs between the
# two arms; no --dec-weight, --c-align-weight, --optimizer-policy or schedule
# override appears anywhere.
srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed "$SEED" --subset-mode final_full \
    --schedule interleaved_123 \
    --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC" \
    --device cuda --max-steps "$MAX_STEPS" \
    --lr-repetition "$LR" --lr-naming "$LR" --lr-comprehension "$LR" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 500 \
    --stop-at-ceiling --ceiling-consecutive-required "$CEILING_REQUIRED" \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$RESUME_FROM" "${EXTRA[@]}"

echo "== $RUN_ID finished: $RUN_DIR"
echo "== ancestry, source SHA and the LR transition: $RUN_DIR/provenance.json"
echo "== the canonical $PARENT_RUN_ID directory was READ ONLY throughout"
echo "== pair with: final_lrpilot_control_h512_s${SEED} vs final_lrpilot_3e5_h512_s${SEED}"
