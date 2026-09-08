#!/bin/bash
#SBATCH --job-name=l3_chigh
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
# COMPREHENSION LR RAISE -- three arms from the shared u2000 CONTROL state.
# The ONLY treatment factor is the COMPREHENSION learning rate.
#
#   array 0-3  : ARM A control   seeds 19-22   R/N/C = 3e-5 / 3e-5 / 1e-4
#   array 4-7  : ARM B c-high    seeds 19-22   R/N/C = 3e-5 / 3e-5 / 1.5e-4
#   array 8-11 : ARM C c-higher  seeds 19-22   R/N/C = 3e-5 / 3e-5 / 2e-4
#
# WHY.  The anneal answered its question in the opposite direction from the
# hypothesis.  Lowering C LR made C WORSE in every seed at u2000:
#
#   C LR 1e-4  (control)  C errors [52,62,51,56]  mean 55.25  LTM ~.8698
#   C LR 5e-5             C errors [57,67,61,67]  mean 63.00  LTM ~.9146
#   C LR 3e-5             C errors [63,69,61,70]  mean 65.75  LTM ~.9379
#
# so annealing buys LTM repetition and pays for it in comprehension, paired
# deltas +7.75 and +10.5 errors.  The u1350 -> u1400 "stall" that motivated
# the anneal was a short-window fluctuation: under the unchanged 1e-4 the mean
# fell 100.75 -> 55.25 between u1400 and u2000.  C is still the bottleneck and
# it is still moving, so the open question is now whether a LARGER C LR
# accelerates the remaining convergence.  This raises it and nothing else.
#
# COMMON SCHEDULE ORIGIN -- the causal requirement.  All three arms re-anchor
# the macro-cycle at the branch step, INCLUDING the unchanged-LR control,
# which must be told to via --reanchor-schedule.  Without it the control would
# keep its historical cycle numbering while nothing else would, and the
# deterministic task ORDER would differ between arms for a reason unrelated to
# the treatment.  With it, all arms begin post-branch cycle index 0 under the
# same schedule seed, so two identically configured arms are BITWISE IDENTICAL
# (proved in tests/test_chigh_u3000.py).
#
# WITHIN-SEED SCHEDULE SEED.  schedule_seed = seed*1000003 + 4, so it is
# 19000061 / 20000064 / 21000067 / 22000070 and is SUPPOSED to differ between
# seeds.  The invariant that matters is within-seed: for one seed, all three
# arms must draw the same task order.  An earlier report compared it across
# seeds and printed a confound that does not exist; chigh_report.py proves the
# correct invariant and emits a per-seed table.
#
# All three arms keep schedule 1:2:3, so at every milestone they share not
# only the global step but the exact R/N/C/pool cursors:
#
#   u2000 (source) step 5,556,000  R   926,000  N 1,852,000  C 2,778,000
#   u2025          step 5,625,450  R   937,575  N 1,875,150  C 2,812,725
#   ...  every 25u, 40 milestones ...
#   u3000 (end)    step 8,334,000  R 1,389,000  N 2,778,000  C 4,167,000
#
# +2,778,000 steps = 463,000 macro-cycles = 1000 repetition exposures:
# +463,000 R batches, +926,000 N, +1,389,000 C, +463,000 pool.
# Every milestone is an exact multiple of SAVE_EVERY.
#
# WALLTIME -- READ THIS BEFORE SUBMITTING.  The tightest measured precedent is
# u850 -> u1200: 972,300 steps with 14 full evaluations at this exact eval
# cadence inside 8 h, i.e. <= 82.3 s per repetition-exposure unit.  1000 u with
# 40 full evaluations is therefore ~23 h and DOES NOT FIT one allocation; 20 h
# is the qos_gpu-t3 ceiling.  Note that --requeue covers preemption and node
# failure but NOT a TIMEOUT.  The intended operation is two allocations: submit
# once, and when the first hits the wall submit THE SAME COMMAND again -- the
# branch/requeue block below detects each run's own latest checkpoint and
# resumes from it, re-declaring nothing.  Every 25u milestone is independently
# analysable, so a block cut short still yields a matched comparison.
#
# Held fixed in ALL arms: schedule 1:2:3 and task ordering, one shared AdamW
# and every moment (never reset), RNG, sampler cursors, R LR 3e-5, N LR 3e-5,
# lambda_C .087, lambda_N 1.0, tau .10, every R loss weight, dec .5, c_align 0,
# the dorsal pool riding every R step, wm128/enc512/dec512, batch 64, wd 1e-5,
# clip 1.0, populations and hashes, teacher forcing, evaluation definitions,
# and the committed 5-milestone strict ceiling rule.
#
# No loss, architecture, sampler, curriculum, hard-negative, margin or
# task-ratio change appears anywhere.
#
# Usage: sbatch --export=ALL,L3_REPO,L3_EXPECTED_COMMIT,L3_GLOVE,L3_RUNS \
#            scripts/cluster/jeanzay/final_chigh_u2000_to_u3000.slurm
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
ARMS=(ctrl ctrl ctrl ctrl 15e5 15e5 15e5 15e5 2e4 2e4 2e4 2e4)
LR_CS=(1e-4 1e-4 1e-4 1e-4 1.5e-4 1.5e-4 1.5e-4 1.5e-4 2e-4 2e-4 2e-4 2e-4)
SEEDS=(19 20 21 22 19 20 21 22 19 20 21 22)
IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 12 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
ARM=${ARMS[$IDX]}; LR_C=${LR_CS[$IDX]}; SEED=${SEEDS[$IDX]}

WM=128; ENC=512; DEC=512
LR_R=3e-5; LR_N=3e-5                     # identical in all three arms
SCHEDULE=interleaved_123                 # identical in all three arms
SOURCE_STEP=5556000                      # u2000
MAX_STEPS=8334000                        # u3000
MS_STEP=69450                            # 25u worth of steps
FULL_EVAL_AT=""
for K in $(seq 1 40); do
    S=$(( SOURCE_STEP + MS_STEP * K ))
    FULL_EVAL_AT="${FULL_EVAL_AT:+$FULL_EVAL_AT,}$S"
done
EVAL_EVERY=13890                         # dev cadence, unchanged
SAVE_EVERY=69450                         # every milestone is a multiple
CEILING_REQUIRED=5                       # committed rule, unchanged

RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
PARENT_RUN_ID="final_canneal_ctrl_h512_s${SEED}"
RUN_ID="final_chigh_${ARM}_u3000_h512_s${SEED}"
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"
SOURCE_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"

case "$RUN_ID" in
    final_chigh_ctrl_u3000_h512_s19|final_chigh_ctrl_u3000_h512_s20|\
final_chigh_ctrl_u3000_h512_s21|final_chigh_ctrl_u3000_h512_s22|\
final_chigh_15e5_u3000_h512_s19|final_chigh_15e5_u3000_h512_s20|\
final_chigh_15e5_u3000_h512_s21|final_chigh_15e5_u3000_h512_s22|\
final_chigh_2e4_u3000_h512_s19|final_chigh_2e4_u3000_h512_s20|\
final_chigh_2e4_u3000_h512_s21|final_chigh_2e4_u3000_h512_s22) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUN_ID" != final_base123_* && "$RUN_ID" != final_lrpilot* \
   && "$RUN_ID" != final_rep_* && "$RUN_ID" != final_canneal_* ]] || {
    echo "FATAL: would write into a parent lineage"; exit 1; }
[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }

echo "== array $IDX -> $RUN_ID   arm=$ARM  seed=$SEED"
echo "== task LRs: R=$LR_R  N=$LR_N  C=$LR_C   (C is the ONLY treatment factor)"
echo "== u2000 -> u3000, 40 milestones every 25u, schedule $SCHEDULE"

[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: u2000 source $SOURCE_CKPT not found"; exit 1; }
SOURCE_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
echo "== source: $SOURCE_CKPT"
echo "== source sha256: $SOURCE_SHA"

# ---- read-only preflight on the shared u2000 source ----------------------
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
# lineage: this must be the CONTROL arm of the anneal, re-anchored at u1400.
need(int(ck.get("schedule_anchor_step")) == 3889200,
     f"anchor {ck.get('schedule_anchor_step')} != 3889200 -- not the canneal control")
p = ck["lr_policy"]
need(p["kind"] == "task_specific", f"lr policy {p['kind']!r}")
need(abs(float(p["repetition"]) - 3e-5) < 1e-12, f"R lr {p['repetition']}")
need(abs(float(p["naming"]) - 3e-5) < 1e-12, f"N lr {p['naming']}")
need(abs(float(p["comprehension"]) - 1e-4) < 1e-12,
     f"C lr {p['comprehension']} != 1e-4 -- this must be the unannealed control")
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
need(int(c["repetition"]) == 926000, f"R cursor {c['repetition']} != 926000")
need(int(c["naming"]) == 2 * 926000, "N cursor != 2R")
need(int(c["comprehension"]) == 3 * 926000, "C cursor != 3R")
need(int(c["pool"]) == 926000, "pool cursor != R cursor")
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

# ---- first launch (branch) vs requeue/second allocation ------------------
# EVERY arm declares --reanchor-schedule so all three share one schedule
# origin.  The control's declared change is the re-anchor alone; the two
# raised arms additionally declare the C-LR change.  Moments, RNG and the
# sampler cursors are never reset by either.  A resumed allocation re-declares
# NOTHING: it continues its own run from its own latest checkpoint.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$SOURCE_CKPT"
    EXTRA=(--reanchor-schedule --phase-transition)
    echo "== first launch: branching the shared u2000 control state"
    echo "== re-anchoring the macro-cycle at $SOURCE_STEP in ALL arms"
    echo "== declaring C LR -> $LR_C (R and N unchanged at 3e-5)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=$((10#${OWN_STEP#step_}))
    if (( OWN_STEP >= MAX_STEPS )); then
        echo "== nothing to do: $RUN_ID already reached u3000"; exit 0
    fi
    (( OWN_STEP >= SOURCE_STEP )) || { echo "FATAL: own checkpoint below the source"; exit 1; }
    RESUME_FROM="$OWN_LATEST"
    echo "== second allocation: resuming $RUN_ID at step $OWN_STEP (nothing re-declared)"
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
echo "== if this allocation timed out short of $MAX_STEPS, submit the SAME sbatch again"
echo "== extraction: scripts/naming_comprehension/chigh_report.py"
