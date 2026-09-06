#!/bin/bash
#SBATCH --job-name=l3_b123_c750
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --requeue

# ===========================================================================
# FINAL base-123 H512 — minimal continuation u500 -> u750.  NO SCIENTIFIC
# CHANGE OF ANY KIND.  This is the SAME trajectory carried further.
#
# WHY.  At u500 the H512 arm sits at C .9813-.9838 (452-523 errors of
# 27,981) with R 8-15 and N 6-10 errors.  The C residual is still MOVING,
# not a stable core: over u400->u500 about 40-42% of the previous errors
# resolve (within-seed Jaccard .544-.563, 58-60% retained, only 8-13% new),
# and across seeds at u500 the pairwise Jaccard is only .109-.133 -- enriched
# over the ~8-9 item chance intersection, but overwhelmingly path-specific.
# N and R residuals are almost pure churn (N: zero persistent items within
# any seed and zero across-seed overlap; R: 2-6 persistent per seed, none
# common to all four).
#
# Every preregistered intervention trigger is therefore UNMET:
#   D (C margin)  needs a SMALL, STABLE, near-tie residual -- it is large,
#                 moving, and only 208/523 are within 0.01 margin (seed19).
#   C (2:2:3)     needs progressive LTM-repetition collapse -- LTM rose
#                 ~0.29 -> ~0.80 at the LR drop and holds at .834-.839.
#   B (RN|C)      needs task-locked oscillation/destabilisation -- absent.
#   E (LR/loss)   needs high-LR instability -- the historical drop produced
#                 a large improvement, not a degradation.
# The licensed action is A: continue unchanged and find out whether C keeps
# converging or develops a stable sub-ceiling core.
#
# SENTINEL.  27 C errors are common to ALL FOUR seeds at u500 (union 1506).
# They are heterogeneous and NOT all tiny-margin near-ties.  Their fate
# through u750 is a preregistered readout; it is a POST-HOC ANALYSIS ONLY and
# is never fed into the training objective.
#
# SAME RUN DIRECTORY, deliberately.  The driver's resume semantics make this
# the safest target and make overwriting the u0-u500 evidence impossible:
#   - metrics.tsv is opened in APPEND mode  -> one continuous u0..u750 curve
#   - config/provenance on a later launch are written as
#     config_from_step_00001389000.json, so the originals are untouched
#   - checkpoints are named by step, so step_01389000.pt and everything
#     before it can never be rewritten
#   - resuming the run's own newest checkpoint is exactly the requeue path
#     the driver was built and tested for, so first launch and requeue are
#     the SAME operation and no branch/warm-start code path is involved
#   - stream seeds are re-derived from the seed and then VERIFIED against the
#     checkpoint, so the samplers cannot silently differ
#
# STOPPING RULE: 5 consecutive DISTINCT scheduled full evaluations at
# 100/100/100/100 (raised from 2).  Within this leg there are exactly five
# milestones, so a stop can occur at u750 only if the model is at ceiling at
# u550 and stays there through u600/u650/u700/u750.  If ceiling begins later,
# the leg ends at u750 with a persisted streak < 5 and does NOT qualify as a
# stable ceiling; a further continuation would be needed.  This is intended.
# The u0-u500 block is independently archived in STORE regardless.
#
# STATE PRESERVED EXACTLY (all restored by load_state_dict): model, the ONE
# shared AdamW and every moment, all four stream cursors (repetition, pool,
# naming, comprehension), global_step, the interleaved_123 schedule anchor
# (so cycle phase continues), the ceiling streak, and all four RNG streams.
# The LR is RE-DERIVED from the repetition cursor, which at u500 is 231,500
# batches -- far past the 46,300 boundary -- so the whole leg runs at 1e-4
# and the LR clock is NOT restarted.
#
# Usage: sbatch --export=ALL,L3_EXPECTED_COMMIT=<sha> \
#            scripts/cluster/jeanzay/final_base123_h512_continue750.slurm
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
module load git

# L3_REPO is MANDATORY.  The first continuation attempt failed safely because
# a default checkout was used while L3_EXPECTED_COMMIT pointed elsewhere; the
# fallback is removed so the pinned worktree must be named explicitly and the
# job can never silently run from the historical checkout.
[[ -n "${L3_REPO:-}" ]] || {
    echo "FATAL: L3_REPO must be exported to the pinned continuation worktree"
    exit 1
}
REPO="$L3_REPO"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || {
    echo "FATAL: L3_REPO=$REPO is not a git worktree"; exit 1; }
cd "$REPO"
HEAD=$(git rev-parse HEAD)
echo "== repo $REPO @ $HEAD ($(git branch --show-current))"
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked tree dirty"; exit 1; }
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || { echo "FATAL: export L3_EXPECTED_COMMIT"; exit 1; }
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD != L3_EXPECTED_COMMIT"; exit 1; }

# The training driver must be the EXACT expected blob.  This is stronger than
# the commit pin: it survives unrelated commits and fails loudly if the driver
# is touched by anything else.
#
# The driver differs from the u0-u500 blob (e6c48ff8f4569c16af1773211d1d6a72
# d92307ea) in STOPPING/EVALUATION CONTROL ONLY: the ceiling requirement is
# now a CLI value defaulting to 5 instead of a hard-coded 2; a milestone step
# already counted toward the streak can no longer be counted twice; the
# milestone checkpoint is written AFTER the streak is updated so it persists
# the post-evaluation state; and a resume whose streak already meets the
# requirement takes zero optimizer steps.
# No optimizer, loss, sampler, schedule, LR or data path is touched, and the
# training updates produced before any stop are bitwise identical -- verified
# by running both versions and comparing checkpoints (see
# tests/test_base123_continue750.py).
TRAIN_BLOB_EXPECTED=24742adf9f13c205673a3f8ff1f34bdce7a45e58
GOT_BLOB=$(git rev-parse HEAD:scripts/naming_comprehension/train_joint_scratch.py)
[[ "$GOT_BLOB" == "$TRAIN_BLOB_EXPECTED" ]] || {
    echo "FATAL: training driver blob $GOT_BLOB != expected $TRAIN_BLOB_EXPECTED"
    exit 1; }
echo "== training driver blob verified: $GOT_BLOB"

# ---- explicit array mapping ----------------------------------------------
SEEDS=(19 20 21 22)
IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 4 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
SEED=${SEEDS[$IDX]}

WM=128; ENC=512; DEC=512          # H512 only; H256 is not continued
U_FROM=500
U_TO=750
SOURCE_STEP=1389000               # 500 x 463 x 6
MAX_STEPS=2083500                 # 750 x 463 x 6
FULL_EVAL_AT=1527900,1666800,1805700,1944600,2083500   # u550,600,650,700,750
EVAL_EVERY=13890                  # unchanged
SAVE_EVERY=69450                  # unchanged
CEILING_REQUIRED=5                # consecutive DISTINCT scheduled full evals
RUN_ID="final_base123_h512_s${SEED}"
RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"

case "$RUN_ID" in
    final_base123_h512_s19|final_base123_h512_s20|final_base123_h512_s21|final_base123_h512_s22) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUN_ID" != *h256* ]] || { echo "FATAL: H256 is not continued"; exit 1; }

echo "== array $IDX -> $RUN_ID  (wm=$WM enc=$ENC dec=$DEC seed=$SEED)"
echo "== continuing u$U_FROM -> u$U_TO  ($SOURCE_STEP -> $MAX_STEPS steps)"

[[ -d "$RUN_DIR" ]] || { echo "FATAL: $RUN_DIR not found — this continues an existing run and NEVER trains from scratch"; exit 1; }
SOURCE_CKPT="$OWN_CKPTS/step_$(printf '%08d' $SOURCE_STEP).pt"
[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: u500 endpoint $SOURCE_CKPT not found"; exit 1; }
echo "== u500 endpoint sha256: $(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)"

# ---- resume selection: the run's own newest checkpoint --------------------
# There is no scientific change, so first launch and requeue are the SAME
# operation.  On the first launch the newest checkpoint IS the u500 endpoint.
# Newest by STEP NUMBER (zero-padded names sort numerically), never mtime.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
[[ -n "${OWN_LATEST:-}" ]] || { echo "FATAL: no checkpoints in $OWN_CKPTS"; exit 1; }
OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=$((10#${OWN_STEP#step_}))

if (( OWN_STEP >= MAX_STEPS )); then
    echo "== nothing to do: $RUN_ID already reached u$U_TO"; exit 0
fi
if (( OWN_STEP < SOURCE_STEP )); then
    echo "FATAL: newest checkpoint is step $OWN_STEP, below the u$U_FROM endpoint $SOURCE_STEP."
    echo "       Refusing: this run directory is not in the expected state."
    exit 1
fi
if (( OWN_STEP == SOURCE_STEP )); then
    echo "== first launch of the continuation, resuming the u$U_FROM endpoint"
else
    echo "== requeue: resuming $RUN_ID at step $OWN_STEP"
fi

# ---- read-only preflight on the artefact we are about to continue --------
L3_CKPT="$OWN_LATEST" L3_SRC="$SOURCE_STEP" L3_MAX="$MAX_STEPS" python - <<'PY'
import os, sys, torch
ck = torch.load(os.environ["L3_CKPT"], map_location="cpu", weights_only=False)
src, mx = int(os.environ["L3_SRC"]), int(os.environ["L3_MAX"])
def need(c, m):
    if not c:
        print(f"FATAL preflight: {m}", file=sys.stderr); sys.exit(1)
step = int(ck["global_step"])
need(ck.get("format") == "lichtheim3.joint_scratch.v1", "unexpected format")
need(src <= step < mx, f"step {step} outside [{src}, {mx})")
need(ck.get("widths") == {"wm_hidden": 128, "ltm_enc_hidden": 512,
                          "ltm_dec_hidden": 512}, f"widths {ck.get('widths')}")
need(ck.get("schedule") == "interleaved_123", f"schedule {ck.get('schedule')!r}")
need(ck.get("optimizer_policy") == "shared_adamw",
     f"optimizer policy {ck.get('optimizer_policy')!r} -- must be the ONE shared bank")
need("optimizer_state_dict" in ck, "no shared AdamW state (banked checkpoint?)")
need("optimizer_states" not in ck, "checkpoint carries per-task banks")
need(float(ck.get("c_align_weight") or 0.0) == 0.0, "c_align is not 0")
need(float(ck.get("dec_weight")) == 0.5, f"dec_weight {ck.get('dec_weight')}")
need(ck.get("subset_mode") == "final_full", f"subset_mode {ck.get('subset_mode')!r}")
need(int(ck.get("comprehension_population_n", 0)) == 27981,
     f"C population {ck.get('comprehension_population_n')}")
need(ck.get("comprehension_population_sha256") ==
     "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
     "canonical C hash mismatch")
need(int(ck["lr_boundary_steps"]) == 46300, "LR boundary moved")
need(int(ck["cursors"]["repetition"]) > 46300,
     "repetition cursor is before the LR boundary -- stage 2 not reached")
need(not (ck.get("phase_transitions") or []), "checkpoint carries a phase transition")
e = ck["exposures"]
print(f"== preflight OK  step {step}  u={step/(463*6):.1f}")
print(f"   cursors R {ck['cursors']['repetition']} N {ck['cursors']['naming']} "
      f"C {ck['cursors']['comprehension']} pool {ck['cursors']['pool']}")
print(f"   exposures R {e['repetition']:.4f} N {e['naming']:.4f} C {e['comprehension']:.4f}")
print(f"   lr {ck.get('lr')} ({ck.get('lr_phase')!r})  moments "
      f"{len(ck['optimizer_state_dict']['state'])} tensors  "
      f"streak {ck.get('consecutive_ceiling')}")
print(f"   written by commit {(ck.get('git') or {}).get('commit')}")
PY

# NOTE: no --eval-at-start, and 1389000 is deliberately absent from
# FULL_EVAL_AT, so the already-completed u500 full evaluation is NOT redone.
# NO --endpoint-eval either: u750 (2,083,500) is already IN FULL_EVAL_AT, so
# the in-loop milestone branch evaluates and checkpoints it exactly once.
# With --endpoint-eval the driver deliberately re-runs the final full
# evaluation even when the same milestone just ran, which is what produced
# the duplicate step-1,389,000 rows in the u0-u500 metrics.  Each of u550,
# u600, u650, u700, u750 therefore yields EXACTLY ONE full-evaluation row.
# No --phase-transition, no --optimizer-policy, no --lr-*, no --dec-weight,
# no --c-align-weight: every scientific setting comes from the checkpoint and
# the driver's audited defaults, exactly as through u0-u500.
srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime j0 --seed "$SEED" --subset-mode final_full \
    --schedule interleaved_123 \
    --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC" \
    --device cuda --max-steps "$MAX_STEPS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 500 \
    --stop-at-ceiling \
    --ceiling-consecutive-required "$CEILING_REQUIRED" \
    --out-dir "$RUNS" --run-id "$RUN_ID" --resume "$OWN_LATEST"

echo "== $RUN_ID continuation finished: $RUN_DIR"
echo "== metrics.tsv now holds one continuous u0..u$U_TO trajectory"
echo "== later launches are recorded in config_from_step_*.json / provenance_from_step_*.json"
echo "== next: per-item error audits at u600 (step 1666800) and u750 (step 2083500)"
