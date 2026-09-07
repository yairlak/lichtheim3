#!/bin/bash
#SBATCH --job-name=l3_canneal
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-11
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --requeue
#
# ===========================================================================
# COMPREHENSION LR ANNEAL — three arms from the shared u1400 state.
# The ONLY treatment factor is the COMPREHENSION learning rate.
#
#   array 0-3  : ARM A control   seeds 19-22   R/N/C = 3e-5 / 3e-5 / 1e-4
#   array 4-7  : ARM B mid       seeds 19-22   R/N/C = 3e-5 / 3e-5 / 5e-5
#   array 8-11 : ARM C low       seeds 19-22   R/N/C = 3e-5 / 3e-5 / 3e-5
#
# WHY.  C acquisition has stopped paying: over the last 50u window
# (u1350 -> u1400) the residual did not improve -- 93->94, 96->100, 95->104,
# 109->105, mean ratio 1.0276 -- from a mean of 100.75 errors.  The
# hard-negative trigger is still FALSE (retention .70987, top5 .950372,
# margin-within-.01 .496278, shared-core ratio .6364), and the residual-
# feature analysis gives no length/frequency curriculum to aim at (the
# survivors are if anything SHORTER than the population).  2:2:3 was already
# tested and rejected (PREFER_223 = false: C worse in 4/4 seeds at all eight
# paired milestones).  What has NOT been tested is whether C LR = 1e-4, which
# earned its keep during acquisition, is now too large for terminal settling.
# This anneals it and nothing else.
#
# COMMON SCHEDULE ORIGIN -- the causal requirement.  All three arms re-anchor
# the macro-cycle at the branch step, INCLUDING the unchanged-LR control,
# which must be told to via --reanchor-schedule.  Without it the control
# would keep its historical cycle numbering while nothing else would, and the
# deterministic task ORDER would differ between arms for a reason unrelated
# to the treatment.  With it, all arms begin post-branch cycle index 0 under
# the same schedule seed, so two identically configured arms are BITWISE
# IDENTICAL (proved in tests/test_canneal_u2000.py).
#
# All three arms keep schedule 1:2:3, so at every milestone they share not
# only the global step but the exact R/N/C/pool cursors:
#
#   u1400 (source) step 3,889,200  R 648,200  N 1,296,400  C 1,944,600
#   u1425          step 3,958,650  R 659,775  N 1,319,550  C 1,979,325
#   ...  every 25u, 24 milestones ...
#   u2000 (end)    step 5,556,000  R 926,000  N 1,852,000  C 2,778,000
#
# +1,666,800 steps = 277,800 macro-cycles = 600 repetition exposures.
# Every milestone is an exact multiple of SAVE_EVERY.
#
# WALLTIME.  The closest measured precedent is u850 -> u1200: 972,300 steps
# with 14 full evaluations at this exact eval cadence, inside an 8 h
# allocation.  This block is 1.71x that (1,666,800 steps, 24 full
# evaluations), so ~13-14 h is the honest estimate and 12 h would have been
# too tight.  20 h is the qos_gpu-t3 ceiling and is requested here; --requeue
# plus the own-checkpoint resume below makes an overrun recoverable rather
# than fatal, and every 25u milestone is independently analysable, so a block
# that stops early still yields a usable matched comparison.
#
# Held fixed in ALL arms: schedule 1:2:3 and task ordering, one shared AdamW
# and every moment (never reset), RNG, all four cursors, R LR 3e-5, N LR
# 3e-5, lambda_C .087, lambda_N 1.0, tau .10, every R loss weight, dec .5,
# c_align 0, the dorsal pool riding every R step, wm128/enc512/dec512, batch
# 64, wd 1e-5, clip 1.0, populations and hashes, teacher forcing, evaluation
# definitions, and the committed 5-milestone strict ceiling rule.
#
# No loss, architecture, sampler, curriculum, hard-negative, margin or
# task-ratio change appears anywhere.
#
# Usage: sbatch --export=ALL,L3_REPO,L3_EXPECTED_COMMIT,L3_GLOVE \
#            scripts/cluster/jeanzay/final_canneal_u1400_to_u2000.slurm
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
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || { echo "FATAL: export L3_EXPECTED_COMMIT"; exit 1; }
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD != L3_EXPECTED_COMMIT"; exit 1; }

TRAIN_BLOB_EXPECTED=95295d63560ae4c235a6beee8dfb47166f4ed30d
GOT_BLOB=$(git rev-parse HEAD:scripts/naming_comprehension/train_joint_scratch.py)
[[ "$GOT_BLOB" == "$TRAIN_BLOB_EXPECTED" ]] || {
    echo "FATAL: driver blob $GOT_BLOB != expected $TRAIN_BLOB_EXPECTED"; exit 1; }
echo "== training driver blob verified: $GOT_BLOB"

# ---- canonical real GloVe (never fallback) --------------------------------
GLOVE=${L3_GLOVE:-/lustre/fswork/projects/rech/llg/uss35bp/lichtheim3/lichtheim3/data/glove.6B.300d.txt}
GLOVE_SHA_EXPECTED=91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed
[[ -f "$GLOVE" ]] || {
    echo "FATAL: real GloVe not found at $GLOVE (set L3_GLOVE)"; exit 1; }
GLOVE_SHA=$(sha256sum "$GLOVE" | cut -d' ' -f1)
[[ "$GLOVE_SHA" == "$GLOVE_SHA_EXPECTED" ]] || {
    echo "FATAL: GloVe sha256 $GLOVE_SHA != $GLOVE_SHA_EXPECTED"; exit 1; }
echo "== real GloVe verified: $GLOVE"

# ---- explicit, collision-proof array mapping -----------------------------
ARMS=(ctrl ctrl ctrl ctrl 5e5 5e5 5e5 5e5 3e5 3e5 3e5 3e5)
LR_CS=(1e-4 1e-4 1e-4 1e-4 5e-5 5e-5 5e-5 5e-5 3e-5 3e-5 3e-5 3e-5)
SEEDS=(19 20 21 22 19 20 21 22 19 20 21 22)
IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 12 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
ARM=${ARMS[$IDX]}; LR_C=${LR_CS[$IDX]}; SEED=${SEEDS[$IDX]}

WM=128; ENC=512; DEC=512
LR_R=3e-5; LR_N=3e-5                     # identical in all three arms
SCHEDULE=interleaved_123                 # identical in all three arms
SOURCE_STEP=3889200                      # u1400
MAX_STEPS=5556000                        # u2000
MS_STEP=69450                            # 25u worth of steps
FULL_EVAL_AT=""
for K in $(seq 1 24); do
    S=$(( SOURCE_STEP + MS_STEP * K ))
    FULL_EVAL_AT="${FULL_EVAL_AT:+$FULL_EVAL_AT,}$S"
done
EVAL_EVERY=13890                         # dev cadence, unchanged
SAVE_EVERY=69450                         # every milestone is a multiple
CEILING_REQUIRED=5                       # committed rule, unchanged

RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
PARENT_RUN_ID="final_rep_rescue123_h512_s${SEED}"
RUN_ID="final_canneal_${ARM}_h512_s${SEED}"
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"
SOURCE_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"

case "$RUN_ID" in
    final_canneal_ctrl_h512_s19|final_canneal_ctrl_h512_s20|\
final_canneal_ctrl_h512_s21|final_canneal_ctrl_h512_s22|\
final_canneal_5e5_h512_s19|final_canneal_5e5_h512_s20|\
final_canneal_5e5_h512_s21|final_canneal_5e5_h512_s22|\
final_canneal_3e5_h512_s19|final_canneal_3e5_h512_s20|\
final_canneal_3e5_h512_s21|final_canneal_3e5_h512_s22) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUN_ID" != final_base123_* && "$RUN_ID" != final_lrpilot* \
   && "$RUN_ID" != final_rep_* ]] || {
    echo "FATAL: would write into a parent lineage"; exit 1; }
[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }

echo "== array $IDX -> $RUN_ID   arm=$ARM  seed=$SEED"
echo "== task LRs: R=$LR_R  N=$LR_N  C=$LR_C   (C is the ONLY treatment factor)"
echo "== u1400 -> u2000, 24 milestones every 25u, schedule $SCHEDULE"

[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: u1400 source $SOURCE_CKPT not found"; exit 1; }
SOURCE_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
echo "== source: $SOURCE_CKPT"
echo "== source sha256: $SOURCE_SHA"

# ---- read-only preflight on the shared u1400 source ----------------------
L3_CKPT="$SOURCE_CKPT" L3_STEP="$SOURCE_STEP" python - <<'PY'
import os, sys, torch
ck = torch.load(os.environ["L3_CKPT"], map_location="cpu", weights_only=False)
want = int(os.environ["L3_STEP"])
def need(c, m):
    if not c:
        print(f"FATAL preflight: {m}", file=sys.stderr); sys.exit(1)
need(ck.get("format") == "lichtheim3.joint_scratch.v1", "unexpected format")
need(int(ck["global_step"]) == want, f"step {ck['global_step']} != {want}")
need(want % 6 == 0, "source is not on a 1:2:3 cycle boundary")
need(ck.get("widths") == {"wm_hidden": 128, "ltm_enc_hidden": 512,
                          "ltm_dec_hidden": 512}, f"widths {ck.get('widths')}")
need(ck.get("schedule") == "interleaved_123", f"schedule {ck.get('schedule')!r}")
need(list(ck.get("schedule_ratio") or []) == [1, 2, 3], "ratio is not 1:2:3")
need(ck.get("optimizer_policy") == "shared_adamw", "not the single shared bank")
need("optimizer_state_dict" in ck and "optimizer_states" not in ck,
     "checkpoint does not carry exactly one shared AdamW state")
p = ck["lr_policy"]
need(p["kind"] == "task_specific", f"lr policy {p['kind']!r}")
need(abs(float(p["repetition"]) - 3e-5) < 1e-12, f"R lr {p['repetition']}")
need(abs(float(p["naming"]) - 3e-5) < 1e-12, f"N lr {p['naming']}")
need(abs(float(p["comprehension"]) - 1e-4) < 1e-12,
     f"C lr {p['comprehension']} != 1e-4 -- this must be the 1:2:3 C-HIGH control")
need(float(ck.get("c_align_weight") or 0.0) == 0.0, "c_align is not 0")
need(float(ck.get("dec_weight")) == 0.5, f"dec_weight {ck.get('dec_weight')}")
need(ck.get("subset_mode") == "final_full", "subset_mode")
need(int(ck.get("comprehension_population_n", 0)) == 27981, "C population")
need(ck.get("comprehension_population_sha256") ==
     "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
     "canonical C hash mismatch")
c = ck["cursors"]
need(int(c["repetition"]) == 648200, f"R cursor {c['repetition']} != 648200")
need(int(c["naming"]) == 2 * 648200, "N cursor != 2R")
need(int(c["comprehension"]) == 3 * 648200, "C cursor != 3R")
need(int(c["pool"]) == 648200, "pool cursor != R cursor")
e = ck["exposures"]
print(f"== preflight OK  step {ck['global_step']}  u={ck['global_step']/(463*6):.1f}")
print(f"   cursors  R {c['repetition']}  N {c['naming']}  C {c['comprehension']}"
      f"  pool {c['pool']}")
print(f"   exposures R {e['repetition']:.4f} N {e['naming']:.4f} "
      f"C {e['comprehension']:.4f}")
print(f"   lr_policy {p}   anchor {ck.get('schedule_anchor_step')}")
print(f"   moments {len(ck['optimizer_state_dict']['state'])} tensors  "
      f"streak {ck.get('consecutive_ceiling')}")
print(f"   written by commit {(ck.get('git') or {}).get('commit')}")
PY

# ---- first launch (branch) vs requeue (own checkpoint) -------------------
# EVERY arm declares --reanchor-schedule so all three share one schedule
# origin.  The control's declared change is the re-anchor alone; the two
# anneal arms additionally declare the C-LR change.  Moments and RNG are
# never reset by either.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$SOURCE_CKPT"
    EXTRA=(--reanchor-schedule --phase-transition)
    echo "== first launch: branching the shared u1400 state"
    echo "== re-anchoring the macro-cycle at $SOURCE_STEP in ALL arms"
    echo "== declaring C LR -> $LR_C (R and N unchanged at 3e-5)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=$((10#${OWN_STEP#step_}))
    if (( OWN_STEP >= MAX_STEPS )); then
        echo "== nothing to do: $RUN_ID already reached u2000"; exit 0
    fi
    (( OWN_STEP >= SOURCE_STEP )) || { echo "FATAL: own checkpoint below the source"; exit 1; }
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming $RUN_ID at step $OWN_STEP (nothing re-declared)"
fi

# R and N LRs come from the SAME variables in every arm; only $LR_C moves.
srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed "$SEED" --subset-mode final_full \
    --schedule "$SCHEDULE" \
    --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC" \
    --device cuda --max-steps "$MAX_STEPS" \
    --lr-repetition "$LR_R" --lr-naming "$LR_N" --lr-comprehension "$LR_C" \
    --glove-path "$GLOVE" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 500 \
    --stop-at-ceiling --ceiling-consecutive-required "$CEILING_REQUIRED" \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$RESUME_FROM" "${EXTRA[@]}"

echo "== $RUN_ID finished: $RUN_DIR"
echo "== the parent $PARENT_RUN_ID directory was READ ONLY throughout"
echo "== if the strict ceiling fired, grep CEILING / STOP and preserve every checkpoint"
echo "== extraction: scripts/naming_comprehension/canneal_report.py"
