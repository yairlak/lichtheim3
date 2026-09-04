#!/bin/bash
#SBATCH --job-name=l3_final9e
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# FINAL-9E — DURABILITY continuation of FINAL-9P.  NO scientific change.
#
# FINAL-9P (R:N:C 1:2:3 -> 2:2:3, from the from-scratch FINAL-7L R130) is the
# first intervention to produce a large, stable LTM rescue at a modest
# acquisition cost.  At the matched N320 / C507.397260 endpoint:
#
#     control 1:2:3 R160   full LTM .169220  FULL rep .999966
#                          naming .055122   C1 .118902  C5 .256495
#     FINAL-9P 2:2:3 R190  full LTM .358933  FULL rep 1.000000
#                          naming .052957   C1 .111862  C5 .244738
#
#     -> +.189713 LTM, -.002165 naming, -.007040 C1, -.011757 C5
#
# and the LTM trajectory was flat, not a transient: M1 .354367, M2 .359880,
# M3 .358933.  Meanwhile FINAL-7L at 1:2:3 reached R1000 with LTM .004092,
# essentially flat from R700.
#
# THIS RUN ASKS ONE QUESTION: does the rescue HOLD over substantially more
# naming and comprehension acquisition, or does it decay?  Nothing is being
# intervened on.  This is a pure continuation of the SAME trajectory in the
# SAME run directory: same model, same two optimizer banks, same cursors,
# same RNG, same schedule anchor, same LR, same losses, same populations.
# There is NO phase transition, NO bank cloning or reset, NO branch.
#
# MATCHED-N/C DURABILITY DESIGN.  The comparison is never at matched global
# step -- changing the macro-cycle changes what a given amount of learning
# costs in steps.  It is at matched ACQUISITION OPPORTUNITY, and the match is
# EXACT ON INTEGER CURSORS, not merely on rounded exposures:
#
#   source R190 = step 458370  cursors R/N/C = 87970 / 148160 / 222240
#   M4 = step 588010  R270 N400 C634.2465753  cursors 125010 / 185200 / 277800
#        vs control step 555600 (R200)        cursors  92600 / 185200 / 277800
#   M5 = step 750060  R370 N500 C792.8082192  cursors 171310 / 231500 / 347250
#        vs control step 694500 (R250)        cursors 115750 / 231500 / 347250
#   M6 = step 912110  R470 N600 C951.3698630  cursors 217610 / 277800 / 416700
#        vs control step 833400 (R300)        cursors 138900 / 277800 / 416700
#
# The N and C cursors are IDENTICAL integers at every milestone, so over this
# leg FINAL-9E and the control take the same number of naming steps (129640)
# and the same number of comprehension steps (194460) on the same items in
# the same order -- the streams are counter-addressed.  Only repetition
# differs: 129640 R steps against the control's 64820.  Acquisition
# opportunity is held fixed; only maintenance opportunity is doubled.
#
# 453740 added steps = 64820 cycles of seven.  HARD STOP at M6.  There is no
# automatic continuation beyond it.
#
# Known control values at R300 (step 833400), to be re-extracted from
# metrics.tsv rather than trusted here: FULL rep 1.0, naming .065030,
# C1 .147958, C5 .302991, full LTM .042474.
#
# Frozen: interleaved_223 ratio 2:2:3, grouped_rn_c_adamw (RN = repetition +
# pool + naming, C = comprehension), LR 1e-4, L_dec 0.5, lambda_C 0.087,
# lambda_N 1.0, c_align 0, tau 0.10, clip 1.0, wd 1e-5, batch 64, seed 22,
# final_full populations, architecture, task-local stream seeds, cursors,
# RNG and every evaluation definition.
#
# RELOCATABLE BY DESIGN.  FINAL-9P runs from an isolated Jean Zay checkout,
# so this script pins NO commit SHA and depends on no other checkout's HEAD.
# Point it anywhere with L3_REPO.  Code identity is enforced by CAPABILITY
# and by the SOURCE ARTEFACT itself (see the preflight below), which is a
# stronger guarantee than a SHA: it checks that the checkpoint we are about
# to continue really is the 2:2:3 state we think it is.  L3_EXPECTED_COMMIT
# and L3_EXPECTED_BRANCH remain available but are OPTIONAL.
# ===========================================================================
set -euo pipefail

module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-$WORK/lichtheim3/lichtheim3}
[[ -d "$REPO" ]] || { echo "FATAL: L3_REPO $REPO does not exist"; exit 1; }
cd "$REPO"

HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)
BRANCH=$(git branch --show-current 2>/dev/null || echo detached)
echo "== repo   : $REPO"
echo "== branch : $BRANCH"
echo "== HEAD   : $HEAD"

# Always enforced: a tracked-clean tree, so the running code is exactly what
# is committed here.  Untracked data files are irrelevant and ignored.
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked working tree dirty"; exit 1; }
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }

# Optional, for callers who do want to pin.  Never required.
if [[ -n "${L3_EXPECTED_BRANCH:-}" ]]; then
    [[ "$BRANCH" == "$L3_EXPECTED_BRANCH" ]] || { echo "FATAL: branch $BRANCH != $L3_EXPECTED_BRANCH"; exit 1; }
fi
if [[ -n "${L3_EXPECTED_COMMIT:-}" ]]; then
    [[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD $HEAD != $L3_EXPECTED_COMMIT"; exit 1; }
fi

# Capability check: an out-of-date checkout must fail loudly, not silently
# run a schedule it does not have.
python - <<'PY' || { echo "FATAL: this checkout has no interleaved_223 support"; exit 1; }
import sys
from scripts.naming_comprehension.train_joint_scratch import (
    INTERLEAVED_223, RATIO_223, SCHEDULES, cycle_steps_for)
assert INTERLEAVED_223 in SCHEDULES and RATIO_223 == (2, 2, 3)
assert cycle_steps_for(INTERLEAVED_223) == 7
print("== capability OK: interleaved_223, ratio (2, 2, 3), 7-step cycle")
PY

# ---- FINAL-9E configuration ----------------------------------------------
REGIME=j0
SEED=22
SUBSET_MODE=final_full
SCHEDULE=interleaved_223                      # UNCHANGED from FINAL-9P
OPT_POLICY=grouped_rn_c_adamw                 # UNCHANGED from FINAL-9P
DEVICE=cuda
ANCHOR_STEP=361140                            # where 1:2:3 -> 2:2:3 took effect
SOURCE_STEP=458370                            # FINAL-9P's own endpoint
FINAL_STEP=912110                             # HARD STOP at M6
MAX_STEPS=912110
FULL_EVAL_AT=588010,750060,912110             # M4 / M5 / M6
EVAL_EVERY=16205                              # 2315 cycles = 10 R / 10 N exp
SAVE_EVERY=32410                              # 4630 cycles = 20 R / 20 N exp

# CADENCES are deliberately IDENTICAL to FINAL-9P's, so metrics.tsv carries
# one continuous, evenly sampled curve across the whole 2:2:3 trajectory.
# Both are multiples of the seven-step cycle, so every dev evaluation and
# safety checkpoint falls at the same fixed phase of the macro-cycle
# (position 4 relative to the anchor) instead of drifting across R/N/C
# positions.  Neither cadence lands on a milestone -- gcd(361140, 32410) is
# 4630, which is not a multiple of seven, so no cadence can -- and neither
# needs to, because --full-eval-at checkpoints unconditionally at each
# milestone.  Over this leg: 28 dev evaluations, and 14 safety checkpoints
# at 486150 ... 907480, one 4630 steps before each milestone, which caps
# requeue loss at ~32410 steps.

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
RUN_ID="final9p_r223_r130_seed${SEED}_${SUBSET_MODE}"    # SAME run directory
RUN_DIR="$RUNS/$RUN_ID"
OWN_CKPTS="$RUN_DIR/checkpoints"
SOURCE_CKPT="$OWN_CKPTS/step_$(printf '%08d' $SOURCE_STEP).pt"

[[ -d "$RUN_DIR" ]] || { echo "FATAL: $RUN_DIR not found — FINAL-9E continues FINAL-9P and never starts a fresh run"; exit 1; }
[[ -f "$SOURCE_CKPT" ]] || { echo "FATAL: FINAL-9P endpoint $SOURCE_CKPT not found"; exit 1; }

# ---- resume selection: this run's own newest checkpoint -------------------
# There is no scientific change, so first launch and requeue are the SAME
# operation: continue this run from its own newest checkpoint.  On the first
# launch that newest checkpoint IS the FINAL-9P endpoint.  Newest by STEP
# NUMBER (zero-padded names sort numerically), never by mtime.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
[[ -n "${OWN_LATEST:-}" ]] || { echo "FATAL: no checkpoints in $OWN_CKPTS"; exit 1; }
OWN_STEP=$(basename "$OWN_LATEST" .pt); OWN_STEP=${OWN_STEP#step_}
OWN_STEP=$((10#$OWN_STEP))

if (( OWN_STEP >= FINAL_STEP )); then
    echo "== nothing to do: FINAL-9E already reached the M6 hard stop ($FINAL_STEP)"
    exit 0
fi
if (( OWN_STEP < SOURCE_STEP )); then
    echo "FATAL: newest checkpoint is step $OWN_STEP, below the FINAL-9P endpoint $SOURCE_STEP."
    echo "       Refusing to continue: this run directory is not in the expected state."
    exit 1
fi
echo "== continuing $RUN_ID from its own step $OWN_STEP (no branch, no transition)"

# ---- read-only preflight on the artefact we are about to continue --------
# Stronger than a commit pin: it asserts the SOURCE STATE itself.  Read-only;
# the checkpoint is only ever opened for reading until the driver resumes it.
# ---8<--- FINAL-9E SOURCE PREFLIGHT (executed verbatim by the test suite)
L3_CKPT="$OWN_LATEST" L3_SRC_STEP="$SOURCE_STEP" L3_FINAL="$FINAL_STEP" \
L3_ANCHOR="$ANCHOR_STEP" python - <<'PY'
import os, sys, torch
p = os.environ["L3_CKPT"]
src, final = int(os.environ["L3_SRC_STEP"]), int(os.environ["L3_FINAL"])
anchor = int(os.environ["L3_ANCHOR"])
ck = torch.load(p, map_location="cpu", weights_only=False)
def need(cond, msg):
    if not cond:
        print(f"FATAL preflight: {msg}", file=sys.stderr); sys.exit(1)
step = int(ck["global_step"])
need(ck.get("format") == "lichtheim3.joint_scratch.v1", "unexpected format")
need(src <= step < final, f"step {step} outside [{src}, {final})")
need(ck.get("schedule") == "interleaved_223", f"schedule {ck.get('schedule')!r}")
need(list(ck.get("schedule_ratio") or []) == [2, 2, 3], f"ratio {ck.get('schedule_ratio')}")
need(int(ck.get("schedule_anchor_step", -1)) == anchor, f"anchor {ck.get('schedule_anchor_step')}")
need((step - anchor) % 7 == 0, f"step {step} is not a whole cycle past the anchor")
need(ck.get("optimizer_policy") == "grouped_rn_c_adamw", f"policy {ck.get('optimizer_policy')!r}")
need("optimizer_states" in ck, "no per-bank optimizer states")
need(set(ck["optimizer_states"]) == {"rn", "comprehension"}, f"banks {set(ck['optimizer_states'])}")
need(float(ck.get("dec_weight")) == 0.5, f"dec_weight {ck.get('dec_weight')}")
need(ck.get("subset_mode") == "final_full", f"subset_mode {ck.get('subset_mode')!r}")
need(float(ck.get("c_align_weight") or 0.0) == 0.0, "c_align is not 0")
cur, e = ck["cursors"], ck["exposures"]
print(f"== source preflight OK  step {step}")
print(f"   cursors  R {cur['repetition']}  N {cur['naming']}  C {cur['comprehension']}")
print(f"   exposures R {e['repetition']:.6f}  N {e['naming']:.6f}  C {e['comprehension']:.7f}")
print(f"   schedule {ck['schedule']} ratio {ck['schedule_ratio']} anchor {ck['schedule_anchor_step']}")
print(f"   policy {ck['optimizer_policy']}  banks {sorted(ck['optimizer_states'])}  dec {ck['dec_weight']}")
print(f"   lr {ck.get('lr')}  phase {ck.get('lr_phase')!r}  transitions {len(ck.get('phase_transitions') or [])}")
print(f"   written by commit {(ck.get('git') or {}).get('commit')}")
PY
# --->8--- end FINAL-9E SOURCE PREFLIGHT

# NOTE: no --phase-transition, no --dec-weight, no --lr-* and no optimizer
# re-declaration appear below.  The schedule and policy flags passed here are
# identical to the checkpoint's, so the driver's guards see no change and no
# transition is recorded.
srun python scripts/naming_comprehension/train_joint_scratch.py \
    --regime "$REGIME" --seed "$SEED" --subset-mode "$SUBSET_MODE" \
    --schedule "$SCHEDULE" --optimizer-policy "$OPT_POLICY" \
    --device "$DEVICE" --max-steps "$MAX_STEPS" \
    --full-eval-at "$FULL_EVAL_AT" \
    --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --log-every 200 \
    --endpoint-eval \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --resume "$OWN_LATEST"

echo "== FINAL-9E complete (hard stop M6 / step $FINAL_STEP): $RUN_DIR"
echo "== no automatic continuation beyond M6."
echo "== PRIMARY readout is the LTM ROUTE; FULL repetition is only a safety"
echo "   metric, because WM can mask ventral damage."
echo "== compare at MATCHED N/C cursors, never at matched global step:"
echo "   9E 588010 vs 7L 555600 | 9E 750060 vs 7L 694500 | 9E 912110 vs 7L 833400"
echo "== archive UPDATE (same run dir, so rsync rather than a fresh copy):"
echo "   rsync -a $RUN_DIR/ \$STORE/lichtheim3_final9p/$RUN_ID/"
