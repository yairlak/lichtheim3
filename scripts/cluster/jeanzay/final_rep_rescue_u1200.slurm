#!/bin/bash
#SBATCH --job-name=l3_rep_rescue
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
# REPETITION RESCUE — paired 1:2:3 vs 2:2:3 from the C-HIGH u1200 state.
# The ONLY treatment factor is TASK ALLOCATION.
#
#   array 0-3 : ARM A control  seeds 19-22   R:N:C = 1:2:3  (6 steps/cycle)
#   array 4-7 : ARM B rescue   seeds 19-22   R:N:C = 2:2:3  (7 steps/cycle)
#
# HYPOTHESIS.  LTM-only repetition equilibrates fast to the current training
# pressure: the global drop to 3e-5 lifted it .83 -> .94 by the FIRST
# evaluation and held; raising only the C learning rate pushed it .94 -> .87
# by the first evaluation and held.  It is an equilibrium, not an erosion.
# If that equilibrium is set by repetition REHEARSAL rate, doubling the
# repetition updates should raise it while keeping the faster C learning that
# LR_C = 1e-4 buys.  Because the shift may be fast, the first two paired
# milestones (25 and 50 control-u) are scientifically important.
#
# MATCHING IS BY MACRO-CYCLE COUNT, NEVER BY GLOBAL STEP.  At equal cycles
# the two arms take an IDENTICAL number of naming and comprehension updates
# on identical counter-addressed batches, while the rescue takes exactly
# twice the repetition updates.  Verified against the real scheduler at all
# eight milestones (see tests/test_rep_rescue_223.py):
#
#   Dcycles   CTRL step   RESC step |  dR ctrl  dR resc  dN both  dC both
#    11575     3403050     3414625  |   11575    23150    23150    34725
#    23150     3472500     3495650  |   23150    46300    46300    69450
#    34725     3541950     3576675  |   34725    69450    69450   104175
#    46300     3611400     3657700  |   46300    92600    92600   138900
#    57875     3680850     3738725  |   57875   115750   115750   173625
#    69450     3750300     3819750  |   69450   138900   138900   208350
#    81025     3819750     3900775  |   81025   162050   162050   243075
#    92600     3889200     3981800  |   92600   185200   185200   277800
#
# The arms therefore END AT DIFFERENT GLOBAL STEPS (3,889,200 vs 3,981,800).
# That is correct and intended; comparing them at a common step would be the
# error this design exists to avoid.
#
# TREATMENT, STATED ACCURATELY: this is INCREASED REPETITION-TASK REHEARSAL,
# INCLUDING ITS ATTACHED DORSAL POOL AUXILIARY -- not lexical repetition
# alone.  The pool rides every R step inside the same backward (pool:R = 1:1
# by construction), so the rescue receives twice the pool updates too.  That
# coupling is historical and is deliberately NOT decoupled; decoupling would
# itself be a second factor.
#
# NO DRIVER CHANGE.  The FINAL-9P schedule machinery already expresses this:
# 123 -> 223 is a declarable phase transition (summed <-> interleaved is
# still categorically refused), the anchor is re-set to the branch step so
# the new cycle index starts at 0, and task order stays a pure function of
# (schedule seed, cycle index) -- with the schedule seed itself verified
# unchanged on resume.  The source step 3,333,600 is divisible by 6, so the
# control continues exactly on a cycle boundary and the rescue's new anchor
# also begins on one.
#
# Held fixed in both arms: one shared AdamW and every moment, RNG, all four
# cursors, task LRs R 3e-5 / N 3e-5 / C 1e-4, lambda_C .087, lambda_N 1.0,
# tau .10, every R loss weight, dec .5, c_align 0, wm128/enc512/dec512,
# batch 64, wd 1e-5, clip 1.0, populations and hashes, teacher forcing,
# evaluation definitions, and the committed 5-milestone strict ceiling rule
# (unchanged pending Yair's answer on evaluation spacing).
#
# Usage: sbatch --export=ALL,L3_REPO,L3_EXPECTED_COMMIT \
#            scripts/cluster/jeanzay/final_rep_rescue_u1200.slurm
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
# ---- canonical real GloVe -------------------------------------------------
# data/glove.*.txt is gitignored, so a DETACHED WORKTREE contains no GloVe and
# the driver's default --glove-path ("data/glove.6B.300d.txt", relative to the
# cwd) resolves to a missing file.  Point at the canonical copy in the main
# checkout explicitly and verify it BEFORE any Python starts.  The file is
# never copied into the worktree, and pseudo-vector fallback is never enabled.
GLOVE=${L3_GLOVE:-/lustre/fswork/projects/rech/llg/uss35bp/lichtheim3/lichtheim3/data/glove.6B.300d.txt}
GLOVE_SHA_EXPECTED=91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed
[[ -f "$GLOVE" ]] || {
    echo "FATAL: real GloVe not found at $GLOVE (set L3_GLOVE)"; exit 1; }
GLOVE_SHA=$(sha256sum "$GLOVE" | cut -d' ' -f1)
[[ "$GLOVE_SHA" == "$GLOVE_SHA_EXPECTED" ]] || {
    echo "FATAL: GloVe sha256 $GLOVE_SHA != $GLOVE_SHA_EXPECTED"; exit 1; }
echo "== real GloVe verified: $GLOVE"
echo "== GloVe sha256: $GLOVE_SHA"
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || { echo "FATAL: export L3_EXPECTED_COMMIT"; exit 1; }
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD != L3_EXPECTED_COMMIT"; exit 1; }

TRAIN_BLOB_EXPECTED=95295d63560ae4c235a6beee8dfb47166f4ed30d
GOT_BLOB=$(git rev-parse HEAD:scripts/naming_comprehension/train_joint_scratch.py)
[[ "$GOT_BLOB" == "$TRAIN_BLOB_EXPECTED" ]] || {
    echo "FATAL: driver blob $GOT_BLOB != expected $TRAIN_BLOB_EXPECTED"; exit 1; }
echo "== training driver blob verified (unchanged since u750): $GOT_BLOB"

# ---- explicit, collision-proof array mapping -----------------------------
ARMS=(123 123 123 123 223 223 223 223)
SEEDS=(19 20 21 22 19 20 21 22)
IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 8 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
ARM=${ARMS[$IDX]}; SEED=${SEEDS[$IDX]}

WM=128; ENC=512; DEC=512
LR_R=3e-5; LR_N=3e-5; LR_C=1e-4          # identical in both arms
SOURCE_STEP=3333600                      # u1200 of the C-HIGH arm
DCYCLES=92600                            # 200 control-u worth of macro-cycles
MS_CYCLES=11575                          # paired milestone spacing (25 control-u)

# BOTH arms re-anchor the macro-cycle at the branch step.  The rescue would
# re-anchor anyway because its schedule changes; the control must be told to,
# via --reanchor-schedule, or it would keep its historical cycle numbering
# (cycle index 555,600 at the branch) while the rescue starts at 0.  That
# difference in cycle ORIGIN changes the deterministic task ORDER and is a
# nuisance factor unrelated to the treatment -- it is measurable: without the
# flag the control's weights differ from the re-anchored control's.  With it,
# both arms begin post-branch cycle index 0 under the same schedule seed, so
# the ONLY remaining difference is the macro-cycle COMPOSITION.
if [[ "$ARM" == "123" ]]; then
    SCHEDULE=interleaved_123; CS=6; RUN_ARM=rescue123
    PHASE=(--reanchor-schedule --phase-transition)   # declared no-op re-anchor
else
    SCHEDULE=interleaved_223; CS=7; RUN_ARM=rescue223
    PHASE=(--phase-transition)            # 1:2:3 -> 2:2:3, re-anchors itself
fi

MAX_STEPS=$(( SOURCE_STEP + CS * DCYCLES ))
FULL_EVAL_AT=""
for K in 1 2 3 4 5 6 7 8; do
    S=$(( SOURCE_STEP + CS * MS_CYCLES * K ))
    FULL_EVAL_AT="${FULL_EVAL_AT:+$FULL_EVAL_AT,}$S"
done
EVAL_EVERY=$(( CS * 2315 ))              # dev cadence, 5 control-u of cycles
SAVE_EVERY=$(( CS * MS_CYCLES ))         # safety saves land on the milestones
CEILING_REQUIRED=5                       # committed rule, unchanged

RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
PARENT_RUN_ID="final_lrpilot1200_chigh_h512_s${SEED}"
RUN_ID="final_rep_${RUN_ARM}_h512_s${SEED}"
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"
SOURCE_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"

case "$RUN_ID" in
    final_rep_rescue123_h512_s19|final_rep_rescue123_h512_s20|\
final_rep_rescue123_h512_s21|final_rep_rescue123_h512_s22|\
final_rep_rescue223_h512_s19|final_rep_rescue223_h512_s20|\
final_rep_rescue223_h512_s21|final_rep_rescue223_h512_s22) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUN_ID" != final_base123_* && "$RUN_ID" != final_lrpilot* ]] || {
    echo "FATAL: would write into a parent lineage"; exit 1; }
[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }

echo "== array $IDX -> $RUN_ID   arm=$ARM  seed=$SEED  schedule=$SCHEDULE"
echo "== cycles +$DCYCLES  steps/cycle $CS  -> max_steps $MAX_STEPS"
echo "== task LRs R=$LR_R N=$LR_N C=$LR_C (identical in both arms)"
echo "== paired milestones (global steps): $FULL_EVAL_AT"

[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: u1200 source $SOURCE_CKPT not found"; exit 1; }
SOURCE_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
echo "== source: $SOURCE_CKPT"
echo "== source sha256: $SOURCE_SHA"
if [[ -n "${L3_U1200_SHA:-}" ]]; then
    [[ "$SOURCE_SHA" == "$L3_U1200_SHA" ]] || {
        echo "FATAL: source sha != L3_U1200_SHA"; exit 1; }
fi

# ---- read-only preflight on the shared C-HIGH u1200 source ---------------
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
     f"C lr {p['comprehension']} != 1e-4 -- this must be the C-HIGH arm")
need(float(ck.get("c_align_weight") or 0.0) == 0.0, "c_align is not 0")
need(float(ck.get("dec_weight")) == 0.5, f"dec_weight {ck.get('dec_weight')}")
need(ck.get("subset_mode") == "final_full", "subset_mode")
need(int(ck.get("comprehension_population_n", 0)) == 27981, "C population")
need(ck.get("comprehension_population_sha256") ==
     "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
     "canonical C hash mismatch")
c = ck["cursors"]; e = ck["exposures"]
need(int(c["repetition"]) * 6 == want,
     f"R cursor {c['repetition']} inconsistent with a 1:2:3 lineage")
need(int(c["naming"]) == 2 * int(c["repetition"]), "N cursor != 2R")
need(int(c["comprehension"]) == 3 * int(c["repetition"]), "C cursor != 3R")
need(int(c["pool"]) == int(c["repetition"]), "pool cursor != R cursor")
print(f"== preflight OK  step {ck['global_step']}  u={ck['global_step']/(463*6):.1f}")
print(f"   cursors  R {c['repetition']}  N {c['naming']}  C {c['comprehension']}"
      f"  pool {c['pool']}")
print(f"   exposures R {e['repetition']:.4f} N {e['naming']:.4f} "
      f"C {e['comprehension']:.4f}")
print(f"   lr_policy {p}   anchor {ck.get('schedule_anchor_step')}")
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
    EXTRA=("${PHASE[@]}")
    echo "== first launch: branching the shared C-HIGH u1200 state"
    echo "== declaring schedule=$SCHEDULE re-anchored at $SOURCE_STEP"
    echo "   (both arms start post-branch macro-cycle index 0)"
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=$((10#${OWN_STEP#step_}))
    if (( OWN_STEP >= MAX_STEPS )); then
        echo "== nothing to do: $RUN_ID already reached its endpoint"; exit 0
    fi
    (( OWN_STEP >= SOURCE_STEP )) || { echo "FATAL: own checkpoint below the source"; exit 1; }
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming $RUN_ID at step $OWN_STEP (schedule already set)"
fi

# Task LRs are identical in both arms; only --schedule (and the resulting
# per-arm step arithmetic) differs.
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
echo "== COMPARE BY MACRO-CYCLE, NEVER BY GLOBAL STEP:"
echo "   paired extraction: scripts/naming_comprehension/paired_rescue_report.py"
