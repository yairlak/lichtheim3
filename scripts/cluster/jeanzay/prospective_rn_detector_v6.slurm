#!/bin/bash
#SBATCH --job-name=l3_prosp_v6
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --requeue
#
# ===========================================================================
# V6 PROSPECTIVE R/N DETECTOR + FIRST-HIT TERMINAL REPAIR
#
# Preregistration: docs/analysis/PROSPECTIVE_RN_DETECTOR_PREREG_V6.md
# Pinned source commit: 2e559e503b9672a3b9e89951fb114abfb77e0ad2
#
# C is closed for the ceiling.  This pilot does NOT search C.  The training
# recipe is the official SETTLE control recipe, unchanged; the ONLY intended
# difference is the denser full-evaluation / checkpoint grid (5u instead of
# 25u), which upgrades EXISTING dev-cadence points (--eval-every 13890 = 5u)
# to full-lexicon detector points.  Each detector point writes a checkpoint,
# because the driver ties torch.save exactly to a full evaluation.
#
#   array 0 : seed 19  from official SETTLE ctrl u3600 -> u3900  (+300u)
#   array 1 : seed 22  from official SETTLE ctrl u3600 -> u3900  (+300u)
#   array 2 : seed 20  from official CHIGH  ctrl u3000 -> u3300  (+300u)
#
# Seeds 19/22 continue their OWN endpoint: lr_policy, dec_weight, optimizer
# policy, schedule and ratio already match, so the driver's `changed` set is
# EMPTY and NEITHER --reanchor-schedule NOR --phase-transition is passed.
# Re-anchoring would restart the macro-cycle and change the TASK ORDER, which
# this design forbids.
#
# Seed 20 branches from the shared u3000 control parent and therefore passes
# --reanchor-schedule --phase-transition on FIRST LAUNCH ONLY, exactly as the
# official SETTLE control arm did from that same parent.  A REQUEUED
# allocation resumes from this run's own latest checkpoint and re-declares
# NOTHING -- without that, a preemption would restart from the source and
# re-declare the transition.
#
# Integrity is verified by sha256 of every executed file (stronger than git
# metadata and independent of any remote git state).
#
# Usage: sbatch --export=ALL,L3_REPO,L3_GLOVE,L3_SRC,L3_RUNS \
#            scripts/cluster/jeanzay/prospective_rn_detector_v6.slurm
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0

PINNED_COMMIT=2e559e503b9672a3b9e89951fb114abfb77e0ad2

# ---- pinned executable, verified by CONTENT hash -------------------------
[[ -n "${L3_REPO:-}" ]] || { echo "FATAL: export L3_REPO"; exit 1; }
REPO="$L3_REPO"
[[ -d "$REPO" ]] || { echo "FATAL: $REPO not a directory"; exit 1; }
cd "$REPO"
echo "== repo $REPO (pinned source commit $PINNED_COMMIT)"

verify_sha() {   # verify_sha <relpath> <expected sha256>
    local f="$1" want="$2" got
    [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
    got=$(sha256sum "$f" | cut -d' ' -f1)
    [[ "$got" == "$want" ]] || {
        echo "FATAL: $f sha256 $got != expected $want"; exit 1; }
    echo "   verified $f"
}
echo "== verifying the official execution path (byte-identical to SETTLE)"
verify_sha scripts/naming_comprehension/train_joint_scratch.py \
    766244c1363d998f34fe3e4f725b57a9b3f95cf716876d71470d804cf962765c
verify_sha scripts/naming_comprehension/train_tasks.py \
    a867b6b5ec474fb79fcd18e769fa8d16d8bdc3de920f7704bfa9be2e7b907303
verify_sha scripts/naming_comprehension/frozen_probe.py \
    d4c7bb9289cef56d09d05b1cbcbd9b47f0fa32dda549b333d29d8a4753689d4f
verify_sha scripts/naming_comprehension/base123_error_audit.py \
    15008114d5e5ff3bde51b550193c07e62d5566347a29510ec9a75bb8ae910b37
verify_sha data/lexicon_en_glove_covered.tsv \
    ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66
echo "== verifying the FROZEN Arm-A implementation (V4/V5, unchanged)"
verify_sha scripts/naming_comprehension/ceiling_source_completion.py \
    298d1ce6454babd48aa90fe8cf09a96e7cfd2a764ef0961ecb6ba9177529416b
verify_sha scripts/naming_comprehension/gradient_training_probe.py \
    a7064708878365fc2875c3db6f903b3e370b16023bab12ce2be23c0f1ea5dcbf
verify_sha scripts/naming_comprehension/constrained_coexistence_v3.py \
    1c69a59c525f11f54946fe7fd42412be601b4b451a55706925ed516b4e495419
verify_sha scripts/naming_comprehension/frozen_head_probe.py \
    f2a30620aad79bba81bbd9e56ab0b1a38a74801c05e35b0f9e74f7bb63fb851b
verify_sha scripts/naming_comprehension/coexistence_probe.py \
    de715e3c573c44d00c0fd8bb56fef1e916c79b4f1e58af3965cb3f48d828ecef
verify_sha scripts/naming_comprehension/first_hit_selector.py \
    2482adb2ab89ced1249303f8e898a00e2a6927e40d22e1cc5f57d55db50a6c5c

# ---- canonical real GloVe (never fallback) --------------------------------
GLOVE=${L3_GLOVE:-/lustre/fswork/projects/rech/llg/uss35bp/lichtheim3/lichtheim3/data/glove.6B.300d.txt}
GLOVE_SHA_EXPECTED=91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed
[[ -f "$GLOVE" ]] || { echo "FATAL: GloVe not found at $GLOVE"; exit 1; }
GLOVE_SHA=$(sha256sum "$GLOVE" | cut -d' ' -f1)
[[ "$GLOVE_SHA" == "$GLOVE_SHA_EXPECTED" ]] || {
    echo "FATAL: GloVe sha256 $GLOVE_SHA != expected"; exit 1; }
echo "== real GloVe verified"

# ---- explicit, collision-proof array mapping -----------------------------
SEEDS=(19 22 20)
SRC_RUNS=(final_settle_ctrl_h512_s19 final_settle_ctrl_h512_s22 final_chigh_ctrl_u3000_h512_s20)
SRC_STEPS=(10000800 10000800 8334000)
END_STEPS=(10834200 10834200 9167400)
SRC_SHAS=(6070317ee5296996ea2b30bd191cbcd53b6a24991d3b49136408513d9510c58a
          f6cd1679cb8fd8b7d4c1352b9b3bdd77ec2ec4249e06af179b21ddc6cdbffae3
          171a8e2bc681382b4697c299956e4b76e45866ed341cd7de0774b944898832a4)
RUN_IDS=(prospective_rn_s19_u3600 prospective_rn_s22_u3600 prospective_rn_s20_u3000)
EXP_ANCHORS=(8334000 8334000 5556000)      # 5556000 proves the CHIGH control lineage
EXP_RCURSOR=(1666800 1666800 1389000)
BRANCHING=(0 0 1)                          # only seed 20 branches from a parent run

IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 3 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
SEED=${SEEDS[$IDX]}
SRC_RUN=${SRC_RUNS[$IDX]}
SOURCE_STEP=${SRC_STEPS[$IDX]}
MAX_STEPS=${END_STEPS[$IDX]}
SRC_SHA_EXPECTED=${SRC_SHAS[$IDX]}
RUN_ID=${RUN_IDS[$IDX]}
EXP_ANCHOR=${EXP_ANCHORS[$IDX]}
EXP_RCUR=${EXP_RCURSOR[$IDX]}
IS_BRANCH=${BRANCHING[$IDX]}

# ---- frozen recipe constants (identical to the official control) ---------
WM=128; ENC=512; DEC=512
LR_R=3e-5; LR_N=3e-5; LR_C=1e-4
SCHEDULE=interleaved_123
EVAL_EVERY=13890          # 5u dev cadence, UNCHANGED from the official run
SAVE_EVERY=69450          # unchanged; detector points also save via at_full
CEILING_REQUIRED=2        # unchanged training-ceiling semantics
STEPS_PER_U=2778
DET_STEP=13890            # 5u detector spacing
N_DET=60

SRC_BASE=${L3_SRC:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/l3_prospective_v6_runs}
SOURCE_CKPT="$SRC_BASE/$SRC_RUN/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"

# ---- refuse to write into any official lineage ---------------------------
case "$RUN_ID" in
    prospective_rn_s19_u3600|prospective_rn_s22_u3600|prospective_rn_s20_u3000) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUN_ID" != final_* ]] || { echo "FATAL: run id uses an official prefix"; exit 1; }
[[ "$RUNS" != *"/lichtheim3_runs"* ]] || {
    echo "FATAL: refusing to write inside the official runs namespace"; exit 1; }
[[ "$RUN_DIR" != "$SRC_BASE/$SRC_RUN" ]] || { echo "FATAL: run dir collides with source"; exit 1; }
mkdir -p "$OWN_CKPTS"

echo "== array $IDX -> $RUN_ID  seed=$SEED  branch_from_parent=$IS_BRANCH"
echo "== source: $SOURCE_CKPT"
[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: source not found"; exit 1; }
SRC_SHA=$(sha256sum "$SOURCE_CKPT" | cut -d' ' -f1)
echo "== source sha256: $SRC_SHA"
[[ "$SRC_SHA" == "$SRC_SHA_EXPECTED" ]] || {
    echo "FATAL: source sha256 != preregistered $SRC_SHA_EXPECTED"; exit 1; }
echo "== source sha256 matches the V6 preregistration"

# ---- exact detector grid, generated programmatically ---------------------
FULL_EVAL_AT=""
for K in $(seq 1 $N_DET); do
    S=$(( SOURCE_STEP + DET_STEP * K ))
    FULL_EVAL_AT="${FULL_EVAL_AT:+$FULL_EVAL_AT,}$S"
done
LAST_DET=$(( SOURCE_STEP + DET_STEP * N_DET ))
[[ "$LAST_DET" == "$MAX_STEPS" ]] || {
    echo "FATAL: grid end $LAST_DET != MAX_STEPS $MAX_STEPS"; exit 1; }
echo "== detector: $N_DET points every $DET_STEP steps (5u), $(( SOURCE_STEP + DET_STEP )) .. $LAST_DET"

# ---- read-only preflight on the source ----------------------------------
L3_CKPT="$SOURCE_CKPT" L3_STEP="$SOURCE_STEP" L3_SEED="$SEED" \
L3_ANCHOR="$EXP_ANCHOR" L3_RCUR="$EXP_RCUR" python - <<'PY'
import os, sys, torch
ck = torch.load(os.environ["L3_CKPT"], map_location="cpu", weights_only=False)
want, seed = int(os.environ["L3_STEP"]), int(os.environ["L3_SEED"])
anchor, rcur = int(os.environ["L3_ANCHOR"]), int(os.environ["L3_RCUR"])
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
need(int(ck.get("schedule_seed")) == seed * 1000003 + 4,
     f"schedule_seed {ck.get('schedule_seed')} != {seed*1000003+4}")
need(int(ck.get("schedule_anchor_step")) == anchor,
     f"anchor {ck.get('schedule_anchor_step')} != {anchor} -- wrong lineage")
need(ck.get("optimizer_policy") == "shared_adamw", "not the single shared bank")
need("optimizer_state_dict" in ck and "optimizer_states" not in ck,
     "checkpoint does not carry exactly one shared AdamW state")
need(float(ck.get("c_align_weight") or 0.0) == 0.0, "c_align is not 0")
need(float(ck.get("dec_weight")) == 0.5, f"dec_weight {ck.get('dec_weight')}")
need(ck.get("subset_mode") == "final_full", "subset_mode != final_full")
need(int(ck.get("comprehension_population_n", 0)) == 27981, "C population")
need(ck.get("comprehension_population_sha256") ==
     "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
     "canonical C hash mismatch")
p = ck["lr_policy"]
need(p.get("kind") == "task_specific", f"lr policy {p.get('kind')!r}")
need(abs(float(p["repetition"]) - 3e-5) < 1e-12, f"R lr {p['repetition']}")
need(abs(float(p["naming"]) - 3e-5) < 1e-12, f"N lr {p['naming']}")
need(abs(float(p["comprehension"]) - 1e-4) < 1e-12,
     f"C lr {p['comprehension']} != 1e-4 -- this must be the CONTROL arm, "
     "never a raised-C treatment")
c = ck["cursors"]
need(int(c["repetition"]) == rcur, f"R cursor {c['repetition']} != {rcur}")
need(int(c["naming"]) == 2 * rcur, "N cursor != 2R")
need(int(c["comprehension"]) == 3 * rcur, "C cursor != 3R")
need(int(c["pool"]) == rcur, "pool cursor != R cursor")
print(f"[preflight] OK step={ck['global_step']} seed={ck['seed']} "
      f"anchor={ck.get('schedule_anchor_step')} cursors={c}")
print(f"[preflight] lr_policy={p}  moments="
      f"{len(ck['optimizer_state_dict']['state'])} tensors")
PY
echo "== source preflight passed (control lineage proved)"

# ========================= 1. TRAINING (unchanged recipe) =================
# First launch vs requeue: a requeued allocation resumes from THIS run's own
# latest checkpoint and re-declares NOTHING.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
SKIP_TRAIN=0
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$SOURCE_CKPT"
    if [[ "$IS_BRANCH" == "1" ]]; then
        EXTRA=(--reanchor-schedule --phase-transition)
        echo "== first launch: branching the u3000 control parent; re-anchoring at $SOURCE_STEP"
    else
        echo "== first launch: PURE continuation of this seed's own u3600 endpoint"
        echo "== no re-anchor, no phase transition: the macro-cycle continues unchanged"
    fi
else
    OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=$((10#${OWN_STEP#step_}))
    if (( OWN_STEP >= MAX_STEPS )); then
        echo "== training already complete at step $OWN_STEP; going to post-processing"
        SKIP_TRAIN=1
    else
        (( OWN_STEP >= SOURCE_STEP )) || { echo "FATAL: own checkpoint below source"; exit 1; }
        RESUME_FROM="$OWN_LATEST"
        echo "== requeue: resuming $RUN_ID at step $OWN_STEP (nothing re-declared)"
    fi
fi

if [[ "$SKIP_TRAIN" == "0" ]]; then
    echo "== BEGIN training $(date -Is)"
    srun python -u scripts/naming_comprehension/train_joint_scratch.py \
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
    echo "== END training $(date -Is)"
fi

# ================= 2. FROZEN FIRST-HIT SELECTION (read-only) ==============
POST="$RUN_DIR/post"
mkdir -p "$POST"
set +e
python -u scripts/naming_comprehension/first_hit_selector.py \
    --metrics "$RUN_DIR/metrics.tsv" --run-dir "$RUN_DIR" --seed "$SEED" \
    --steps-per-u $STEPS_PER_U --out-json "$POST/first_hit_decision.json"
HIT_RC=$?
set -e

write_summary() {   # write_summary <status> <extra-json>
    L3_STATUS="$1" L3_EXTRA="$2" L3_SEED="$SEED" L3_RUN="$RUN_ID" \
    L3_SRCSHA="$SRC_SHA" L3_COMMIT="$PINNED_COMMIT" L3_POST="$POST" python - <<'PY'
import json, os, torch
post = os.environ["L3_POST"]
d = json.load(open(f"{post}/first_hit_decision.json"))
out = {"seed": int(os.environ["L3_SEED"]), "run_id": os.environ["L3_RUN"],
       "status": os.environ["L3_STATUS"],
       "source_checkpoint_sha256": os.environ["L3_SRCSHA"],
       "pinned_commit": os.environ["L3_COMMIT"],
       "torch": torch.__version__,
       "n_detector_points": d["n_detector_points"],
       "detector_first_step": d["first_step"],
       "detector_last_step": d["last_step"],
       "n_eligible_total": d["n_eligible_total"],
       "hit": d["hit"]}
out.update(json.loads(os.environ["L3_EXTRA"] or "{}"))
json.dump(out, open(f"{post}/RESULT_SUMMARY.json", "w"), indent=1)
print("RESULT_SUMMARY.json:", out["status"])
PY
}

if [[ "$HIT_RC" -ne 0 ]]; then
    echo "== NO_ELIGIBLE_HIT for seed $SEED: no Arm A, by the frozen rule"
    write_summary NO_ELIGIBLE_HIT '{}'
    sha256sum "$OWN_CKPTS"/*.pt > "$POST/CHECKPOINT_SHA256SUMS" 2>/dev/null || true
    echo "== task $IDX complete (no hit) $(date -Is)"
    exit 0
fi

# ============ 3. FIXED ARM A ON THE FIRST HIT (no alternatives) ===========
SEL_STEP=$(python -c "import json;print(json.load(open('$POST/first_hit_decision.json'))['selected_step'])")
SEL_U=$(python -c "import json;print(int(round(json.load(open('$POST/first_hit_decision.json'))['selected_u'])))")
SEL_C=$(python -c "import json;print(json.load(open('$POST/first_hit_decision.json'))['selected_metrics']['C_errors'])")
SEL_CKPT="$OWN_CKPTS/step_$(printf '%08d' $SEL_STEP).pt"
echo "== FIRST HIT: step $SEL_STEP (u$SEL_U), detector C=$SEL_C"
[[ -f "$SEL_CKPT" ]] || { echo "FATAL: selected checkpoint missing $SEL_CKPT"; exit 1; }
SEL_SHA=$(sha256sum "$SEL_CKPT" | cut -d' ' -f1)
echo "== selected checkpoint sha256: $SEL_SHA"

# 3a. source-specific cache rebuilt FROM THE SELECTED CHECKPOINT
python -u scripts/naming_comprehension/ceiling_source_completion.py extract \
    --seed "$SEED" --ckpt "$SEL_CKPT" --source-u "$SEL_U" \
    --out "$POST/cache_seed${SEED}_u${SEL_U}.pt" \
    --out-json "$POST/extract_validation.json" \
    --expect-c "$SEL_C" --expect-step "$SEL_STEP" \
    --code-commit "$PINNED_COMMIT" --code-dirty 0 --nt 10

# 3b. official battery on the UNMODIFIED selected source
python -u scripts/naming_comprehension/ceiling_source_completion.py battery \
    --seed "$SEED" --ckpt "$SEL_CKPT" --source-u "$SEL_U" --head SOURCE \
    --out-json "$POST/source_battery.json" --nt 10

# 3c. the frozen Arm A
python -u scripts/naming_comprehension/ceiling_source_completion.py train \
    --seed "$SEED" --ckpt "$SEL_CKPT" --source-u "$SEL_U" \
    --rep "$POST/cache_seed${SEED}_u${SEL_U}.pt" --out-dir "$POST" \
    --expect-step "$SEL_STEP" --code-commit "$PINNED_COMMIT" --code-dirty 0 --nt 10

ARM_DIR="$POST/seed${SEED}_u${SEL_U}_A"
HEAD_FILE="$ARM_DIR/head_first_c0.pt"
if [[ ! -f "$HEAD_FILE" ]]; then
    echo "== Arm A did not reach official C=0; recording REPAIR_NO_C0"
    write_summary REPAIR_NO_C0 "{\"selected_step\": $SEL_STEP, \"selected_u\": $SEL_U, \"selected_sha256\": \"$SEL_SHA\"}"
    exit 0
fi

# 3d. complete official battery on the repaired head
python -u scripts/naming_comprehension/ceiling_source_completion.py battery \
    --seed "$SEED" --ckpt "$SEL_CKPT" --source-u "$SEL_U" --head "$HEAD_FILE" \
    --out-json "$POST/repaired_battery.json" --nt 10

# ============================ 4. SUMMARY =================================
L3_POST="$POST" L3_ARM="$ARM_DIR" L3_SEED="$SEED" L3_RUN="$RUN_ID" \
L3_SRCSHA="$SRC_SHA" L3_SELSHA="$SEL_SHA" L3_SELSTEP="$SEL_STEP" \
L3_SELU="$SEL_U" L3_COMMIT="$PINNED_COMMIT" python - <<'PY'
import json, os, torch
post, arm = os.environ["L3_POST"], os.environ["L3_ARM"]
d  = json.load(open(f"{post}/first_hit_decision.json"))
sb = json.load(open(f"{post}/source_battery.json"))
rb = json.load(open(f"{post}/repaired_battery.json"))
tr = json.load(open(f"{arm}/trace.json"))
exact = (rb["c_errors"] == 0 and rb["rep_canonical_full_errors"] == 0
         and rb["rep_freear_full_errors"] == 0 and rb["naming_errors"] == 0)
keys = ("c_errors", "rep_canonical_full_errors", "rep_freear_full_errors",
        "naming_errors", "rep_canonical_ltm", "gate_mean")
out = {
    "seed": int(os.environ["L3_SEED"]), "run_id": os.environ["L3_RUN"],
    "status": ("PROSPECTIVE_MATURE_STATE_EXACT_WITNESS" if exact
               else "REPAIRED_BUT_NOT_FOUR_WAY_EXACT"),
    "hit": True,
    "source_checkpoint_sha256": os.environ["L3_SRCSHA"],
    "selected_step": int(os.environ["L3_SELSTEP"]),
    "selected_u": int(os.environ["L3_SELU"]),
    "selected_sha256": os.environ["L3_SELSHA"],
    "n_detector_points": d["n_detector_points"],
    "n_points_before_hit": d["n_points_before_hit"],
    "n_eligible_total": d["n_eligible_total"],
    "detector_at_hit": d["selected_metrics"],
    "source_battery": {k: sb[k] for k in keys},
    "repaired_battery": {k: rb[k] for k in keys},
    "LTM_source": sb["rep_canonical_ltm"],
    "LTM_repaired": rb["rep_canonical_ltm"],
    "LTM_delta": rb["rep_canonical_ltm"] - sb["rep_canonical_ltm"],
    "arm_a": {"status": tr["status"], "steps": tr["steps"],
              "first_c0_iter": tr["first_c0_iter"],
              "official_calls": tr["official_calls"],
              "wall_total_s": tr["wall_total_s"],
              "heads_sha256": tr["heads_sha256"]},
    "EXACT_100_100_100": bool(exact),
    "pinned_commit": os.environ["L3_COMMIT"], "torch": torch.__version__,
}
json.dump(out, open(f"{post}/RESULT_SUMMARY.json", "w"), indent=1)
print("EXACT_100_100_100 =", exact, " status =", out["status"])
PY

sha256sum "$OWN_CKPTS"/*.pt > "$POST/CHECKPOINT_SHA256SUMS" 2>/dev/null || true
echo "== task $IDX complete $(date -Is)"
