#!/bin/bash
#SBATCH --job-name=l3_lrpilot1200
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --requeue
#
# ===========================================================================
# LR PILOT u850 -> u1200 — can a higher LR on COMPREHENSION ALONE accelerate
# the C residual while keeping the stability that the global 3e-5 arm gave?
#
#   array 0-3 : ALL-3e-5 control  seeds 19,20,21,22   R/N/C = 3e-5 / 3e-5 / 3e-5
#   array 4-7 : C-HIGH hybrid     seeds 19,20,21,22   R/N/C = 3e-5 / 3e-5 / 1e-4
#
# WHY.  The u750->u850 pilot showed the global drop to 3e-5 bought a large
# stability gain at essentially no cost in C: naming reached exact 1.0 for
# ALL four seeds at ALL four milestones, repetition errors fell (10.25 ->
# 6.50 mean), and LTM-only repetition rose ~11.15 pp (.827305 -> .938825),
# while mean C errors were statistically indistinguishable (186.25 vs 188.50,
# per-seed deltas +9/-10/+9/+1 -- mixed in sign).  So the low LR is clearly
# preferable overall, but there is no evidence the COMPREHENSION stream
# needed it.  This pilot restores 1e-4 on C updates only and asks whether the
# remaining ~190 C errors resolve faster while R/N/LTM keep their gains.
#
# ONE TREATMENT FACTOR: the comprehension task LR.  Repetition and naming
# stay at 3e-5 in both arms.
#
# IMPLEMENTATION -- NO DRIVER CHANGE.  The existing task-specific LR policy
# already expresses asymmetric per-task rates; that is what it was built for.
# Both arms declare their full three-task policy explicitly from the same
# source state, so the design is symmetric.  Note the driver REFUSES to
# resume a task-LR checkpoint without the three flags (it would silently
# revert to the two-stage policy), so passing them is mandatory, not
# cosmetic.  For the control the declared values equal the source's, so
# nothing changes and no new transition is recorded -- verified bitwise
# identical to the same run without --phase-transition.  For the hybrid only
# the comprehension entry differs and a second transition is recorded.
#
# Optimizer moments are NEVER reset: load_state_dict restores the single
# shared AdamW and every moment, then the per-step LR is applied to the param
# groups.  RNG, all four stream cursors, the schedule anchor and seed, and
# the ceiling streak are restored unchanged.
#
# BRANCH SAFETY.  The parent final_lrpilot_3e5_h512_sXX directories are
# READ-ONLY here.  These branches write only to final_lrpilot950_* ids; the
# job refuses anything else and explicitly refuses final_base123_* and the
# parent id.  The shared u850 source is opened read-only, its SHA256 logged
# and optionally pinned with L3_U850_SHA.
#
# CEILING unchanged: 5 DISTINCT scheduled full evaluations at strict ceiling
# (R canonical, R genuine free-AR, N free-AR, C strict top-1 all exactly 1).
# The source streak is 0 because R and C are not exact, and this pilot has
# four milestones... -- over the FOURTEEN milestones of this horizon the rule
# CAN fire, and an early stop is then a genuine stable-ceiling result rather
# than a truncation.  If it does not fire, running to u1200 is the expected
# outcome and is not a failure.
#
# Usage: sbatch --export=ALL,L3_REPO,L3_EXPECTED_COMMIT \
#            scripts/cluster/jeanzay/final_lrpilot_u850_to_u1200.slurm
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
echo "== training driver blob verified (unchanged since u750): $GOT_BLOB"

# ---- explicit, collision-proof array mapping -----------------------------
ARMS=(all3e5 all3e5 all3e5 all3e5 chigh chigh chigh chigh)
LR_C=(3e-5 3e-5 3e-5 3e-5 1e-4 1e-4 1e-4 1e-4)
SEEDS=(19 20 21 22 19 20 21 22)
IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 8 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
ARM=${ARMS[$IDX]}; SEED=${SEEDS[$IDX]}
LR_R=3e-5                                # repetition: identical in both arms
LR_N=3e-5                                # naming:     identical in both arms
LR_COMP=${LR_C[$IDX]}                    # THE treatment factor

WM=128; ENC=512; DEC=512
SOURCE_STEP=2361300                      # u850 = 850 x 463 x 6
MAX_STEPS=3333600                        # u1200 = 1200 x 463 x 6
# Every 25u from u875 to u1200 -- 14 scheduled full evaluations, all exact
# multiples of SAVE_EVERY.  The horizon is the ONLY thing changed from the
# approved u950 design; it is a longer observation window for the same
# two-arm experiment, not a new factor.  With 14 milestones the 5-distinct
# ceiling rule CAN now fire, which it could not over four.
FULL_EVAL_AT=2430750,2500200,2569650,2639100,2708550,2778000,2847450,2916900,2986350,3055800,3125250,3194700,3264150,3333600
EVAL_EVERY=13890                         # unchanged
SAVE_EVERY=69450                         # unchanged; every milestone is a multiple
CEILING_REQUIRED=5                       # unchanged strict rule

RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
PARENT_RUN_ID="final_lrpilot_3e5_h512_s${SEED}"
RUN_ID="final_lrpilot1200_${ARM}_h512_s${SEED}"
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"
SOURCE_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"

case "$RUN_ID" in
    final_lrpilot1200_all3e5_h512_s19|final_lrpilot1200_all3e5_h512_s20|\
final_lrpilot1200_all3e5_h512_s21|final_lrpilot1200_all3e5_h512_s22|\
final_lrpilot1200_chigh_h512_s19|final_lrpilot1200_chigh_h512_s20|\
final_lrpilot1200_chigh_h512_s21|final_lrpilot1200_chigh_h512_s22) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUN_ID" != final_base123_* ]] || { echo "FATAL: would write into the canonical lineage"; exit 1; }
[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }

echo "== array $IDX -> $RUN_ID   arm=$ARM  seed=$SEED"
echo "== task LRs: R=$LR_R  N=$LR_N  C=$LR_COMP   (C is the treatment factor)"
echo "== branching from $PARENT_RUN_ID at u850 (step $SOURCE_STEP) -> u1200"

[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: u850 source $SOURCE_CKPT not found"; exit 1; }
SOURCE_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
echo "== source: $SOURCE_CKPT"
echo "== source sha256: $SOURCE_SHA"
if [[ -n "${L3_U850_SHA:-}" ]]; then
    [[ "$SOURCE_SHA" == "$L3_U850_SHA" ]] || {
        echo "FATAL: source sha != L3_U850_SHA"; exit 1; }
fi

# ---- read-only preflight on the shared u850 source -----------------------
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
p = ck["lr_policy"]
need(p["kind"] == "task_specific", f"source lr policy {p['kind']!r}")
for t in ("repetition", "naming", "comprehension"):
    need(abs(float(p[t]) - 3e-5) < 1e-12,
         f"source {t} lr {p[t]} != 3e-5 -- this must be the 3e-5 arm")
need(float(ck.get("c_align_weight") or 0.0) == 0.0, "c_align is not 0")
need(float(ck.get("dec_weight")) == 0.5, f"dec_weight {ck.get('dec_weight')}")
need(ck.get("subset_mode") == "final_full", "subset_mode")
need(int(ck.get("comprehension_population_n", 0)) == 27981, "C population")
need(ck.get("comprehension_population_sha256") ==
     "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
     "canonical C hash mismatch")
need(int(ck["cursors"]["repetition"]) == 393550,
     f"repetition cursor {ck['cursors']['repetition']} != 393550 (u850)")
e = ck["exposures"]
print(f"== preflight OK  step {ck['global_step']}  u={ck['global_step']/(463*6):.1f}")
print(f"   cursors R {ck['cursors']['repetition']} N {ck['cursors']['naming']} "
      f"C {ck['cursors']['comprehension']} pool {ck['cursors']['pool']}")
print(f"   exposures R {e['repetition']:.4f} N {e['naming']:.4f} C {e['comprehension']:.4f}")
print(f"   lr_policy {p}")
print(f"   moments {len(ck['optimizer_state_dict']['state'])} tensors  "
      f"streak {ck.get('consecutive_ceiling')}  "
      f"transitions {len(ck.get('phase_transitions') or [])}")
print(f"   written by commit {(ck.get('git') or {}).get('commit')}")
PY

# ---- first launch (branch) vs requeue (own checkpoint) -------------------
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$SOURCE_CKPT"
    EXTRA=(--phase-transition)
    echo "== first launch: branching the shared u850 state"
    echo "== declaring R=$LR_R N=$LR_N C=$LR_COMP (control declares the source's"
    echo "   own values, so it records no new transition)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=$((10#${OWN_STEP#step_}))
    if (( OWN_STEP >= MAX_STEPS )); then
        echo "== nothing to do: $RUN_ID already reached u1200"; exit 0
    fi
    (( OWN_STEP >= SOURCE_STEP )) || { echo "FATAL: own checkpoint below the source"; exit 1; }
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming $RUN_ID at step $OWN_STEP (policy already set)"
fi

# R and N LRs come from the SAME variables in both arms; only $LR_COMP moves.
srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed "$SEED" --subset-mode final_full \
    --schedule interleaved_123 \
    --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC" \
    --device cuda --max-steps "$MAX_STEPS" \
    --lr-repetition "$LR_R" --lr-naming "$LR_N" --lr-comprehension "$LR_COMP" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 500 \
    --stop-at-ceiling --ceiling-consecutive-required "$CEILING_REQUIRED" \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$RESUME_FROM" "${EXTRA[@]}"

echo "== $RUN_ID finished: $RUN_DIR"
echo "== ancestry, source SHA and the LR policy: $RUN_DIR/provenance.json"
echo "== the parent $PARENT_RUN_ID directory was READ ONLY throughout"
echo "== pair with: final_lrpilot1200_all3e5_h512_s${SEED} vs final_lrpilot1200_chigh_h512_s${SEED}"
