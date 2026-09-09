#!/bin/bash
#SBATCH --job-name=l3_settle
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
# LATE-STATE SETTLE TEST -- three arms from the shared u3000 CONTROL state.
# The ONLY treatment factor is the COMPREHENSION learning rate, and the
# treatment is the SAME downward bracket CANNEAL tested at u1400.
#
#   array 0-3  : ARM A control   seeds 19-22   R/N/C = 3e-5 / 3e-5 / 1e-4
#   array 4-7  : ARM B mid       seeds 19-22   R/N/C = 3e-5 / 3e-5 / 5e-5
#   array 8-11 : ARM C low       seeds 19-22   R/N/C = 3e-5 / 3e-5 / 3e-5
#
# WHY.  CHIGH closed the C-LR escalation search: across [3e-5, 2e-4] the late
# C decay rate is invariant (~ -0.7 errors / 50u in every arm) and the u3000
# endpoint is a smoothed tie (last-5 means 38.25 vs 38.25), with a monotone
# LTM cost as the only robust dose-response.  The mandatory read-only u3000
# audit then found:
#   * hard-tail trigger FALSE (retention .5577, four-way core 2 -> 1,
#     margin-within-.01 fraction .4966 -- one item short of the .50 bound);
#   * influx nearly stopped: 77-88% of current C errors were already wrong
#     at u2000, new-fraction only 12-23% over 1000u;
#   * zero homophone competitors, zero mathematically-unavoidable items;
#   * residuals are SHORT, HIGH-frequency words in dense semantic
#     neighbourhoods (their/they, weekday and number fields, morphology).
# That is a boundary layer whose fluctuation scale plausibly tracks the step
# size.  The SETTLE HYPOTHESIS: at this maturity, lowering the C step reduces
# and stabilises the error count.  CANNEAL tested the identical bracket at
# u1400 mid-acquisition and lowering LOST (+7.75 / +10.5 paired errors at
# 600u).  If the hypothesis is right the sign FLIPS here; if the tail is
# structural, the low arms freeze at ~current level with collapsed variance.
# Either outcome is decisive, and the frozen set in the second case is the
# stable target the preregistered hard-negative trigger has been waiting for.
#
# EXACT CANNEAL MIRROR.  Same arms (1e-4 / 5e-5 / 3e-5), same +600u horizon,
# same 24-milestone cadence, same seeds -- only the branch state differs
# (u3000 instead of u1400).  The state-dependence claim needs nothing else.
#
# COMMON SCHEDULE ORIGIN -- the causal requirement.  All three arms re-anchor
# the macro-cycle at the branch step, INCLUDING the unchanged-LR control,
# which must be told to via --reanchor-schedule.  With it, all arms begin
# post-branch cycle index 0 under the same schedule seed, so two identically
# configured arms are BITWISE IDENTICAL (proved in tests/test_settle_u3600.py).
#
# All three arms keep schedule 1:2:3, so at every milestone they share not
# only the global step but the exact R/N/C/pool cursors:
#
#   u3000 (source) step  8,334,000  R 1,389,000  N 2,778,000  C 4,167,000
#   u3025          step  8,403,450  R 1,400,575  N 2,801,150  C 4,201,725
#   ...  every 25u, 24 milestones ...
#   u3600 (end)    step 10,000,800  R 1,666,800  N 3,333,600  C 5,000,400
#
# +1,666,800 steps = 277,800 macro-cycles = 600 repetition exposures.
# Every milestone is an exact multiple of SAVE_EVERY.
#
# WALLTIME.  This block is shape-identical to CANNEAL u1400 -> u2000
# (600u, 24 full evaluations), which completed 12/12 inside one 20 h
# allocation.  --requeue plus the own-checkpoint resume below makes an
# overrun recoverable, and every 25u milestone is independently analysable.
#
# Held fixed in ALL arms: schedule 1:2:3 and task ordering, one shared AdamW
# and every moment (never reset), RNG, all four cursors, R LR 3e-5, N LR
# 3e-5, lambda_C .087, lambda_N 1.0, tau .10, every R loss weight, dec .5,
# c_align 0, the dorsal pool riding every R step, wm128/enc512/dec512, batch
# 64, wd 1e-5, clip 1.0, populations and hashes, teacher forcing, and every
# evaluation definition.  Ceiling semantics follow Amendment 2 of the frozen
# decision memo: stop on TWO consecutive full-lexicon 0/0/0/0 evaluations.
#
# No loss, architecture, sampler, curriculum, hard-negative, margin or
# task-ratio change appears anywhere.
#
# Usage: sbatch --export=ALL,L3_REPO,L3_EXPECTED_COMMIT,L3_GLOVE,L3_RUNS \
#            scripts/cluster/jeanzay/final_settle_u3000_to_u3600.slurm
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
echo "== training driver blob verified (unchanged since u1400): $GOT_BLOB"

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
SOURCE_STEP=8334000                      # u3000
MAX_STEPS=10000800                       # u3600
MS_STEP=69450                            # 25u worth of steps
FULL_EVAL_AT=""
for K in $(seq 1 24); do
    S=$(( SOURCE_STEP + MS_STEP * K ))
    FULL_EVAL_AT="${FULL_EVAL_AT:+$FULL_EVAL_AT,}$S"
done
EVAL_EVERY=13890                         # dev cadence, unchanged
SAVE_EVERY=69450                         # every milestone is a multiple
# Amendment 2 (pre-result): CEILING_CONFIRMED = TWO consecutive scheduled
# full-lexicon 0/0/0/0 evaluations (one 25u persistence interval).  A single
# perfect evaluation is a CEILING_HIT (existence proof; its checkpoint is
# written at that exact step) but does not stop the run.  Five consecutive
# remains reported as a stability diagnostic only.  The four component
# metrics are unchanged and exact.
CEILING_REQUIRED=2                       # confirmed-ceiling stop threshold

RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
PARENT_RUN_ID="final_chigh_ctrl_u3000_h512_s${SEED}"
RUN_ID="final_settle_${ARM}_h512_s${SEED}"
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"
SOURCE_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"

case "$RUN_ID" in
    final_settle_ctrl_h512_s19|final_settle_ctrl_h512_s20|\
final_settle_ctrl_h512_s21|final_settle_ctrl_h512_s22|\
final_settle_5e5_h512_s19|final_settle_5e5_h512_s20|\
final_settle_5e5_h512_s21|final_settle_5e5_h512_s22|\
final_settle_3e5_h512_s19|final_settle_3e5_h512_s20|\
final_settle_3e5_h512_s21|final_settle_3e5_h512_s22) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUN_ID" != final_base123_* && "$RUN_ID" != final_lrpilot* \
   && "$RUN_ID" != final_rep_* && "$RUN_ID" != final_canneal_* \
   && "$RUN_ID" != final_chigh_* ]] || {
    echo "FATAL: would write into a parent lineage"; exit 1; }
[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }

echo "== array $IDX -> $RUN_ID   arm=$ARM  seed=$SEED"
echo "== task LRs: R=$LR_R  N=$LR_N  C=$LR_C   (C is the ONLY treatment factor)"
echo "== u3000 -> u3600, 24 milestones every 25u, schedule $SCHEDULE"
echo "== CANNEAL mirror: same bracket, same 600u, branched at u3000 not u1400"

[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: u3000 source $SOURCE_CKPT not found"; exit 1; }
SOURCE_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
echo "== source: $SOURCE_CKPT"
echo "== source sha256: $SOURCE_SHA"

# ---- read-only preflight on the shared u3000 source ----------------------
L3_CKPT="$SOURCE_CKPT" L3_STEP="$SOURCE_STEP" L3_SEED="$SEED" python - <<'PY'
import os, sys, torch
ck = torch.load(os.environ["L3_CKPT"], map_location="cpu", weights_only=False)
want = int(os.environ["L3_STEP"])
seed = int(os.environ["L3_SEED"])
def need(c, m):
    if not c:
        print(f"FATAL preflight: {m}", file=sys.stderr); sys.exit(1)
need(ck.get("format") == "lichtheim3.joint_scratch.v1", "unexpected format")
need(int(ck["global_step"]) == want, f"step {ck['global_step']} != {want}")
need(want % 6 == 0, "source is not on a 1:2:3 cycle boundary")
need(int(ck.get("seed")) == seed, f"seed {ck.get('seed')} != {seed}")
need(ck.get("widths") == {"wm_hidden": 128, "ltm_enc_hidden": 512,
                          "ltm_dec_hidden": 512}, f"widths {ck.get('widths')}")
need(ck.get("schedule") == "interleaved_123", f"schedule {ck.get('schedule')!r}")
need(list(ck.get("schedule_ratio") or []) == [1, 2, 3], "ratio is not 1:2:3")
need(ck.get("optimizer_policy") == "shared_adamw", "not the single shared bank")
need("optimizer_state_dict" in ck and "optimizer_states" not in ck,
     "checkpoint does not carry exactly one shared AdamW state")
# schedule_seed is seed*1000003 + 4 and is SUPPOSED to differ between seeds.
need(int(ck.get("schedule_seed")) == seed * 1000003 + 4,
     f"schedule_seed {ck.get('schedule_seed')} != {seed*1000003+4}")
# lineage: this must be the CHIGH CONTROL arm, re-anchored at u2000.
need(int(ck.get("schedule_anchor_step")) == 5556000,
     f"anchor {ck.get('schedule_anchor_step')} != 5556000 -- not the chigh control")
p = ck["lr_policy"]
need(p["kind"] == "task_specific", f"lr policy {p['kind']!r}")
need(abs(float(p["repetition"]) - 3e-5) < 1e-12, f"R lr {p['repetition']}")
need(abs(float(p["naming"]) - 3e-5) < 1e-12, f"N lr {p['naming']}")
need(abs(float(p["comprehension"]) - 1e-4) < 1e-12,
     f"C lr {p['comprehension']} != 1e-4 -- this must be the CHIGH control, "
     "not one of its raised arms")
need(float(ck.get("c_align_weight") or 0.0) == 0.0, "c_align is not 0")
need(float(ck.get("dec_weight")) == 0.5, f"dec_weight {ck.get('dec_weight')}")
need(ck.get("subset_mode") == "final_full", "subset_mode")
need(int(ck.get("comprehension_population_n", 0)) == 27981, "C population")
need(ck.get("comprehension_population_sha256") ==
     "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
     "canonical C hash mismatch")
need(int(ck.get("consecutive_ceiling") or 0) < 5,
     "source already satisfies the ceiling rule -- do not continue it")
c = ck["cursors"]
need(int(c["repetition"]) == 1389000, f"R cursor {c['repetition']} != 1389000")
need(int(c["naming"]) == 2 * 1389000, "N cursor != 2R")
need(int(c["comprehension"]) == 3 * 1389000, "C cursor != 3R")
need(int(c["pool"]) == 1389000, "pool cursor != R cursor")
e = ck["exposures"]
print(f"== preflight OK  step {ck['global_step']}  u={ck['global_step']/(463*6):.1f}")
print(f"   cursors  R {c['repetition']}  N {c['naming']}  C {c['comprehension']}"
      f"  pool {c['pool']}")
print(f"   exposures R {e['repetition']:.4f} N {e['naming']:.4f} "
      f"C {e['comprehension']:.4f}")
print(f"   lr_policy {p}")
print(f"   schedule_seed {ck.get('schedule_seed')}  anchor {ck.get('schedule_anchor_step')}")
print(f"   moments {len(ck['optimizer_state_dict']['state'])} tensors  "
      f"streak {ck.get('consecutive_ceiling')}")
print(f"   written by commit {(ck.get('git') or {}).get('commit')}")
PY

# ---- first launch (branch) vs requeue (own checkpoint) -------------------
# EVERY arm declares --reanchor-schedule so all three share one schedule
# origin.  The control's declared change is the re-anchor alone; the two
# settle arms additionally declare the C-LR change.  Moments and RNG are
# never reset by either.  A requeued allocation re-declares NOTHING.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$SOURCE_CKPT"
    EXTRA=(--reanchor-schedule --phase-transition)
    echo "== first launch: branching the shared u3000 control state"
    echo "== re-anchoring the macro-cycle at $SOURCE_STEP in ALL arms"
    echo "== declaring C LR -> $LR_C (R and N unchanged at 3e-5)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=$((10#${OWN_STEP#step_}))
    if (( OWN_STEP >= MAX_STEPS )); then
        echo "== nothing to do: $RUN_ID already reached u3600"; exit 0
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
echo "== extraction: scripts/naming_comprehension/settle_report.py"
