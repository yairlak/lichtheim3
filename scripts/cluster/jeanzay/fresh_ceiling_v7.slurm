#!/bin/bash
#SBATCH --job-name=l3_fresh_v7
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
#
# ===========================================================================
# V7 FRESH STEP-0 CEILING REPLICATION  --  four preregistered seeds
#
# Preregistration: docs/analysis/FRESH_STEP0_CEILING_PREREG_V7.md
# Orchestration:   scripts/naming_comprehension/fresh_ceiling_v7.py
#
# QUESTION (the only one): can the frozen detector -> FIRST-HIT -> fixed
# Arm-A -> exact-battery policy reproduce from FRESH STEP 0 under the frozen
# H512 joint recipe?  C IS CLOSED.  Nothing scientific is tuned here.
#
#   array 0 : slot P1  seed 31     array 2 : slot P3  seed 33
#   array 1 : slot P2  seed 32     array 3 : slot P4  seed 34
#
# Each task trains ONE fresh trajectory u0 -> u3900 (step 0 -> 10,834,200)
# through the UNCHANGED official driver, in the seven legs of the frozen
# retained-recipe ledger (LR transitions at u750 / u850, schedule re-anchors
# at u1200 / u1400 / u2000 / u3000), with a full-lexicon detector every 5u
# (13,890 steps; 780 points + step 0), a checkpoint at every detector point,
# an ONLINE FIRST-HIT freeze (Rcan=Rfree=N=0; C excluded), then the frozen
# V6 post-processing: first_hit_selector -> extract -> source battery ->
# fixed Arm A -> repaired battery -> RESULT_SUMMARY.json.
#
# One allocation is 20 h (qos_gpu-t3 ceiling); a no-hit seed needs ~44 GPU-h.
# The SAME script is therefore submitted as a chain of dependent arrays
# (see fresh_ceiling_v7_submit.sh); every launch is idempotent: it resumes
# the run's OWN newest checkpoint, derives the leg from the frozen ledger,
# declares a transition only at the exact boundary step, never re-anchors
# mid-leg, never re-runs a detector point, and after a FIRST-HIT freeze it
# never trains again.
#
# Usage: sbatch --export=ALL,L3_REPO,L3_EXPECTED_COMMIT,L3_GLOVE,L3_RUNS \
#            scripts/cluster/jeanzay/fresh_ceiling_v7.slurm
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
module load git

# ---- pinned executable: commit + clean tracked tree + content hashes -----
[[ -n "${L3_REPO:-}" ]] || { echo "FATAL: export L3_REPO (the pinned V7 checkout)"; exit 1; }
[[ -n "${L3_EXPECTED_COMMIT:-}" ]] || { echo "FATAL: export L3_EXPECTED_COMMIT"; exit 1; }
REPO="$L3_REPO"
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || { echo "FATAL: $REPO is not a git checkout"; exit 1; }
cd "$REPO"
HEAD=$(git rev-parse HEAD)
echo "== repo $REPO @ $HEAD"
[[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD $HEAD != L3_EXPECTED_COMMIT $L3_EXPECTED_COMMIT"; exit 1; }
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked tree dirty"; git status --porcelain -uno; exit 1; }

verify_sha() {   # verify_sha <relpath> <expected sha256>
    local f="$1" want="$2" got
    [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
    got=$(sha256sum "$f" | cut -d' ' -f1)
    [[ "$got" == "$want" ]] || { echo "FATAL: $f sha256 $got != expected $want"; exit 1; }
    echo "   verified $f"
}
echo "== verifying the official training/evaluation path (byte-identical to SETTLE/V6)"
verify_sha scripts/naming_comprehension/train_joint_scratch.py \
    766244c1363d998f34fe3e4f725b57a9b3f95cf716876d71470d804cf962765c
verify_sha scripts/naming_comprehension/train_tasks.py \
    a867b6b5ec474fb79fcd18e769fa8d16d8bdc3de920f7704bfa9be2e7b907303
verify_sha losses.py \
    382eed2335ee2add7a00f6de9b73a788cdef5e94155ae1970e38bd5d7dcefd80
verify_sha scripts/naming_comprehension/frozen_probe.py \
    d4c7bb9289cef56d09d05b1cbcbd9b47f0fa32dda549b333d29d8a4753689d4f
verify_sha scripts/naming_comprehension/base123_error_audit.py \
    15008114d5e5ff3bde51b550193c07e62d5566347a29510ec9a75bb8ae910b37
verify_sha data/lexicon_en_glove_covered.tsv \
    ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66
echo "== verifying the FROZEN Arm-A / selection implementation (V4/V5/V6, unchanged)"
verify_sha scripts/naming_comprehension/ceiling_source_completion.py \
    fc1a0865bba0810ba03a3a01a54feb234dd6cca7c341a94527c093f41f18088f
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
V7=scripts/naming_comprehension/fresh_ceiling_v7.py
[[ "$(git hash-object "$V7")" == "$(git rev-parse "HEAD:$V7")" ]] || { echo "FATAL: $V7 differs from HEAD"; exit 1; }
echo "   verified $V7 == HEAD blob $(git rev-parse "HEAD:$V7")"

# ---- canonical real GloVe (never fallback) --------------------------------
GLOVE=${L3_GLOVE:-/lustre/fswork/projects/rech/llg/uss35bp/lichtheim3/lichtheim3/data/glove.6B.300d.txt}
verify_sha "$GLOVE" 91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed
echo "== real GloVe verified"

# ---- frozen cohort, priority slots, horizon, cadence ----------------------
SLOTS=(p1 p2 p3 p4)
SEEDS=(31 32 33 34)
IDX=${SLURM_ARRAY_TASK_ID:?this script must run as an array job}
(( IDX >= 0 && IDX < 4 )) || { echo "FATAL: array index $IDX out of range"; exit 1; }
SLOT=${SLOTS[$IDX]}; SEED=${SEEDS[$IDX]}
RUN_ID="fresh_ceiling_v7_${SLOT}_s${SEED}"

WM=128; ENC=512; DEC=512
SCHEDULE=interleaved_123
MAX_U=3900
STEPS_PER_U=2778
END_STEP=10834200
DET_STEP=13890
N_DET=780
EVAL_EVERY=13890
SAVE_EVERY=69450
CEILING_REQUIRED=2
PINNED_COMMIT="$HEAD"

RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/l3_fresh_v7_runs}
OFFICIAL_RUNS=${L3_OFFICIAL_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
V6_RUNS=${L3_V6_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/l3_prospective_v6_runs}
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"
POST="$RUN_DIR/post"
MARKER="$POST/FIRST_HIT_FROZEN.json"

# ---- refuse to write into any official / V6 namespace --------------------
case "$RUN_ID" in
    fresh_ceiling_v7_p1_s31|fresh_ceiling_v7_p2_s32|fresh_ceiling_v7_p3_s33|fresh_ceiling_v7_p4_s34) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
[[ "$RUNS" != *"/lichtheim3_runs"* && "$RUNS" != *"/l3_prospective_v6_runs"* ]] || {
    echo "FATAL: refusing to write inside an official/V6 runs namespace"; exit 1; }
[[ "$RUN_ID" != final_* && "$RUN_ID" != prospective_* ]] || { echo "FATAL: official prefix"; exit 1; }

# ---- arithmetic self-check against the executable ------------------------
python - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.naming_comprehension import fresh_ceiling_v7 as v7
from scripts.naming_comprehension.train_joint_scratch import CounterStream, CANONICAL_N_WORDS, CANONICAL_BATCH_SIZE, cycle_steps_for
per = CounterStream("repetition", list(range(CANONICAL_N_WORDS)), CANONICAL_BATCH_SIZE, seed=1).per_epoch
assert per * cycle_steps_for("interleaved_123") == v7.STEPS_PER_U == 2778, per
assert v7.END_STEP == 10834200 and v7.DET_STEP == 13890 and v7.N_DET == 780 and len(v7.grid()) == 781
print("[v7] arithmetic verified against the driver: 2778 steps/u, grid 0..10,834,200 every 13,890")
PY
FULL_EVAL_AT=$(python "$V7" grid)

echo "== array $IDX -> $RUN_ID  slot=$SLOT seed=$SEED  horizon u$MAX_U ($END_STEP steps)  detector every ${DET_STEP} steps (5u), $N_DET points"
python "$V7" ledger

# ---- already terminal?  (idempotent chain element) ------------------------
if [[ -f "$POST/RESULT_SUMMARY.json" ]]; then
    echo "== $RUN_ID already has post/RESULT_SUMMARY.json: nothing to do"; exit 0
fi

latest_step() {   # newest own checkpoint by STEP NUMBER (zero-padded names sort numerically)
    local f; f=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
    [[ -n "$f" ]] || { echo ""; return; }
    f=$(basename "$f" .pt); echo $((10#${f#step_}))
}

# ---- FIRST-LAUNCH preflight: seed cleanliness on the cluster + storage ----
if [[ -z "$(latest_step)" && ! -f "$MARKER" ]]; then
    echo "== first launch of $RUN_ID: remote seed-cleanliness proof (no training before it passes)"
    mkdir -p "$POST"
    sacct -u "$USER" -S 2026-06-01 -X -n -o JobName%120,JobID%20,State%12 > "$POST/preflight_sacct_jobnames.txt" 2>/dev/null || true
    ( ls -1 "$OFFICIAL_RUNS" "$V6_RUNS" 2>/dev/null; ls -1 "$RUNS" 2>/dev/null | grep -v "^fresh_ceiling_v7_" ) > "$POST/preflight_run_namespaces.txt" || true
    python "$V7" seed-audit --seeds "$SEED" \
        --paths "$OFFICIAL_RUNS" "$V6_RUNS" "${WORK:-/lustre/fswork/projects/rech/llg/uss35bp}/lichtheim3" \
                "${WORK:-/lustre/fswork/projects/rech/llg/uss35bp}/l3_prospective_v6" \
        --text-files "$POST/preflight_sacct_jobnames.txt" "$POST/preflight_run_namespaces.txt" \
        --out-json "$POST/preflight_seed_audit.json" \
        || { echo "FATAL: seed $SEED is NOT clean on Jean-Zay (see $POST/preflight_seed_audit.json); apply the preregistered replacement rule BEFORE any training"; exit 1; }
    echo "== seed $SEED clean on Jean-Zay (official runs, V6 runs, WORK trees, accounting job names)"
    mkdir -p "$RUNS"
    FREE_GB=$(df -P "$RUNS" | awk 'NR==2{print int($4/1024/1024)}')
    echo "== free space at $RUNS: ${FREE_GB} GB (need >= 30 GB for this seed's 781 checkpoints x ~30 MB)"
    (( FREE_GB >= 30 )) || { echo "FATAL: insufficient SCRATCH space"; exit 1; }
    [[ ! -d "$RUN_DIR/checkpoints" || -z "$(ls -A "$RUN_DIR/checkpoints")" ]] || { echo "FATAL: unsafe overwrite: $RUN_DIR has checkpoints but latest_step failed"; exit 1; }
    mkdir -p "$OWN_CKPTS"
fi

# ============================ 1. TRAINING (leg loop) =======================
# Each iteration derives ONE launch from the ledger and the run's own newest
# checkpoint.  A watcher freezes the FIRST eligible detector point online.
mkdir -p "$POST"
while true; do
    if [[ -f "$MARKER" ]]; then
        echo "== FIRST-HIT already frozen ($MARKER): no further training"; break
    fi
    LATEST=$(latest_step)
    if [[ -n "$LATEST" ]]; then
        eval "$(python "$V7" plan --latest-step "$LATEST")"
    else
        eval "$(python "$V7" plan)"
    fi
    echo "== plan: status=$V7_STATUS leg=$V7_LEG resume=${V7_RESUME_STEP:-none} max_steps=$V7_MAX_STEPS flags=[$V7_FLAGS] declare=$V7_DECLARE reanchor=$V7_REANCHOR"
    if [[ "$V7_STATUS" == "COMPLETE" ]]; then
        echo "== training complete at step $LATEST (u$MAX_U): no FIRST-HIT during training"; break
    fi
    RESUME=()
    if [[ -n "${V7_RESUME_STEP:-}" ]]; then
        CK="$OWN_CKPTS/step_$(printf '%08d' "$V7_RESUME_STEP").pt"
        python "$V7" verify-ckpt --ckpt "$CK" --seed "$SEED" --expect-step "$V7_RESUME_STEP" \
            || { echo "FATAL: own checkpoint $CK fails the V7 recipe/ledger gate; refusing to continue"; exit 1; }
        RESUME=(--resume "$CK")
    fi
    # shellcheck disable=SC2206
    EXTRA=($V7_FLAGS)
    echo "== BEGIN leg $V7_LEG ($V7_STATUS) $(date -Is)"
    srun python -u scripts/naming_comprehension/train_joint_scratch.py \
        --regime j0 --seed "$SEED" --subset-mode final_full \
        --schedule "$SCHEDULE" \
        --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC" \
        --device cuda --max-steps "$V7_MAX_STEPS" \
        --glove-path "$GLOVE" \
        --full-eval-at "$FULL_EVAL_AT" \
        --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 500 \
        --stop-at-ceiling --ceiling-consecutive-required "$CEILING_REQUIRED" \
        --out-dir "$RUNS" --run-id "$RUN_ID" \
        "${RESUME[@]}" "${EXTRA[@]}" &
    TRAIN_PID=$!
    python -u "$V7" watch --run-dir "$RUN_DIR" --pid "$TRAIN_PID" --marker "$MARKER" --seed "$SEED" --poll 30 &
    WATCH_PID=$!
    set +e
    wait "$TRAIN_PID"; RC=$?
    set -e
    kill "$WATCH_PID" 2>/dev/null || true
    wait "$WATCH_PID" 2>/dev/null || true
    echo "== END leg $V7_LEG rc=$RC $(date -Is)"
    if [[ -f "$MARKER" ]]; then
        echo "== FIRST-HIT frozen online; training stopped by the watcher"; break
    fi
    if [[ "$RC" -ne 0 ]]; then
        echo "FATAL: driver exited rc=$RC without a FIRST-HIT freeze; a requeue/chain element will resume from the newest checkpoint"; exit "$RC"
    fi
done

# ===================== 2. FROZEN FIRST-HIT SELECTION ======================
python "$V7" dedupe --src "$RUN_DIR/metrics.tsv" --dst "$POST/metrics_dedup.tsv" > "$POST/metrics_dedup.json"
cat "$POST/metrics_dedup.json"
set +e
python -u scripts/naming_comprehension/first_hit_selector.py \
    --metrics "$POST/metrics_dedup.tsv" --run-dir "$RUN_DIR" --seed "$SEED" \
    --steps-per-u $STEPS_PER_U --out-json "$POST/first_hit_decision.json"
HIT_RC=$?
set -e

write_summary() {   # write_summary <status> <extra-json>
    L3_STATUS="$1" L3_EXTRA="$2" L3_SEED="$SEED" L3_RUN="$RUN_ID" L3_SLOT="$SLOT" \
    L3_COMMIT="$PINNED_COMMIT" L3_POST="$POST" python - <<'PY'
import json, os, torch
post = os.environ["L3_POST"]
d = json.load(open(f"{post}/first_hit_decision.json"))
out = {"seed": int(os.environ["L3_SEED"]), "slot": os.environ["L3_SLOT"], "run_id": os.environ["L3_RUN"],
       "status": os.environ["L3_STATUS"], "fresh_step0": True, "horizon_u": 3900,
       "pinned_commit": os.environ["L3_COMMIT"], "torch": torch.__version__,
       "n_detector_points": d["n_detector_points"], "detector_first_step": d["first_step"],
       "detector_last_step": d["last_step"], "n_eligible_total": d["n_eligible_total"], "hit": d["hit"]}
out.update(json.loads(os.environ["L3_EXTRA"] or "{}"))
json.dump(out, open(f"{post}/RESULT_SUMMARY.json", "w"), indent=1)
print("RESULT_SUMMARY.json:", out["status"])
PY
}

if [[ "$HIT_RC" -ne 0 ]]; then
    [[ ! -f "$MARKER" ]] || { echo "FATAL: watcher froze a hit but the frozen selector finds none"; exit 1; }
    echo "== NO_ELIGIBLE_HIT for seed $SEED by u$MAX_U: no Arm A, by the frozen rule"
    write_summary NO_ELIGIBLE_HIT_BY_U3900 '{}'
    sha256sum "$OWN_CKPTS"/*.pt > "$POST/CHECKPOINT_SHA256SUMS" 2>/dev/null || true
    echo "== task $IDX complete (no hit) $(date -Is)"; exit 0
fi

SEL_STEP=$(python -c "import json;print(json.load(open('$POST/first_hit_decision.json'))['selected_step'])")
SEL_U=$(python -c "import json;print(int(round(json.load(open('$POST/first_hit_decision.json'))['selected_u'])))")
SEL_C=$(python -c "import json;print(json.load(open('$POST/first_hit_decision.json'))['selected_metrics']['C_errors'])")
SEL_CKPT="$OWN_CKPTS/step_$(printf '%08d' $SEL_STEP).pt"
[[ -f "$SEL_CKPT" ]] || { echo "FATAL: selected checkpoint missing $SEL_CKPT"; exit 1; }
SEL_SHA=$(sha256sum "$SEL_CKPT" | cut -d' ' -f1)
echo "== FIRST HIT: step $SEL_STEP (u$SEL_U), detector C=$SEL_C, sha256 $SEL_SHA"
if [[ -f "$MARKER" ]]; then
    M_STEP=$(python -c "import json;print(json.load(open('$MARKER'))['first_hit_step'])")
    M_SHA=$(python -c "import json;print(json.load(open('$MARKER'))['checkpoint_sha256'])")
    [[ "$M_STEP" == "$SEL_STEP" && "$M_SHA" == "$SEL_SHA" ]] || {
        echo "FATAL: online freeze (step $M_STEP sha $M_SHA) != frozen selector (step $SEL_STEP sha $SEL_SHA)"; exit 1; }
    echo "== online freeze and frozen selector agree"
fi
python "$V7" verify-ckpt --ckpt "$SEL_CKPT" --seed "$SEED" --expect-step "$SEL_STEP" \
    || { echo "FATAL: selected source fails the recipe/ledger gate"; exit 1; }

# ============ 3. FIXED ARM A ON THE FIRST HIT (no alternatives) ===========
ARM_DIR="$POST/seed${SEED}_u${SEL_U}_A"
HEAD_FILE="$ARM_DIR/head_first_c0.pt"
if [[ -d "$ARM_DIR" && ! -f "$ARM_DIR/trace.json" ]]; then
    mv "$ARM_DIR" "${ARM_DIR}_partial_$(date +%s)"; echo "== moved an interrupted Arm-A directory aside"
fi
if [[ ! -f "$ARM_DIR/trace.json" ]]; then
    python -u scripts/naming_comprehension/ceiling_source_completion.py extract \
        --seed "$SEED" --ckpt "$SEL_CKPT" --source-u "$SEL_U" \
        --out "$POST/cache_seed${SEED}_u${SEL_U}.pt" --out-json "$POST/extract_validation.json" \
        --expect-c "$SEL_C" --expect-step "$SEL_STEP" \
        --code-commit "$PINNED_COMMIT" --code-dirty 0 --nt 10
    python -u scripts/naming_comprehension/ceiling_source_completion.py battery \
        --seed "$SEED" --ckpt "$SEL_CKPT" --source-u "$SEL_U" --head SOURCE \
        --out-json "$POST/source_battery.json" --nt 10
    python -u scripts/naming_comprehension/ceiling_source_completion.py train \
        --seed "$SEED" --ckpt "$SEL_CKPT" --source-u "$SEL_U" \
        --rep "$POST/cache_seed${SEED}_u${SEL_U}.pt" --out-dir "$POST" \
        --expect-step "$SEL_STEP" --code-commit "$PINNED_COMMIT" --code-dirty 0 --nt 10
fi
if [[ ! -f "$HEAD_FILE" ]]; then
    echo "== Arm A did not reach official C=0; recording REPAIR_NO_C0 (outcome B: audit the frozen terminal pipeline; no reselection)"
    write_summary REPAIR_NO_C0 "{\"selected_step\": $SEL_STEP, \"selected_u\": $SEL_U, \"selected_sha256\": \"$SEL_SHA\"}"
    sha256sum "$OWN_CKPTS"/*.pt > "$POST/CHECKPOINT_SHA256SUMS" 2>/dev/null || true
    exit 0
fi
[[ -f "$POST/repaired_battery.json" ]] || \
python -u scripts/naming_comprehension/ceiling_source_completion.py battery \
    --seed "$SEED" --ckpt "$SEL_CKPT" --source-u "$SEL_U" --head "$HEAD_FILE" \
    --out-json "$POST/repaired_battery.json" --nt 10

# ============================ 4. SUMMARY =================================
L3_POST="$POST" L3_ARM="$ARM_DIR" L3_SEED="$SEED" L3_RUN="$RUN_ID" L3_SLOT="$SLOT" \
L3_SELSHA="$SEL_SHA" L3_SELSTEP="$SEL_STEP" L3_SELU="$SEL_U" L3_COMMIT="$PINNED_COMMIT" python - <<'PY'
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
    "seed": int(os.environ["L3_SEED"]), "slot": os.environ["L3_SLOT"], "run_id": os.environ["L3_RUN"],
    "status": ("FRESH_STEP0_PROSPECTIVE_EXACT_WITNESS" if exact else "REPAIRED_BUT_NOT_FOUR_WAY_EXACT"),
    "fresh_step0": True, "horizon_u": 3900, "hit": True,
    "selected_step": int(os.environ["L3_SELSTEP"]), "selected_u": int(os.environ["L3_SELU"]),
    "selected_sha256": os.environ["L3_SELSHA"],
    "n_detector_points": d["n_detector_points"], "n_points_before_hit": d["n_points_before_hit"],
    "n_eligible_total": d["n_eligible_total"], "detector_at_hit": d["selected_metrics"],
    "source_battery": {k: sb[k] for k in keys}, "repaired_battery": {k: rb[k] for k in keys},
    "LTM_source": sb["rep_canonical_ltm"], "LTM_repaired": rb["rep_canonical_ltm"],
    "LTM_delta": rb["rep_canonical_ltm"] - sb["rep_canonical_ltm"],
    "arm_a": {"status": tr["status"], "steps": tr["steps"], "first_c0_iter": tr["first_c0_iter"],
              "official_calls": tr["official_calls"], "wall_total_s": tr["wall_total_s"],
              "heads_sha256": tr["heads_sha256"]},
    "EXACT_100_100_100": bool(exact),
    "pinned_commit": os.environ["L3_COMMIT"], "torch": torch.__version__,
}
json.dump(out, open(f"{post}/RESULT_SUMMARY.json", "w"), indent=1)
print("EXACT_100_100_100 =", exact, " status =", out["status"])
PY
sha256sum "$OWN_CKPTS"/*.pt > "$POST/CHECKPOINT_SHA256SUMS" 2>/dev/null || true
echo "== task $IDX complete $(date -Is)"
