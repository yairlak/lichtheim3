#!/bin/bash
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# CAP-4 — was the C512 learning-rate reduction premature?
#
# The CAP-3 width probe imposed lr 1e-3 -> 1e-4 at exposure 100 on every
# width, a schedule inherited from the NAMING result where 1e-3 demonstrably
# destabilised H512.  For comprehension there is no such evidence: the C512
# trajectory was still improving fast at the boundary (.957078 @25, .989314
# @50, .995068 @100, all under 1e-3) and then flattened under 1e-4
# (.998070 @150 ... .998642 @1000, ~38 top-1 errors out of 27,981).
#
# This job reopens that one choice, and nothing else.  Two branches from the
# genuine C512 checkpoint at EXACTLY 100 exposures:
#
#     A   1e-3 -> 3e-4 at 100        B   keep 1e-3 after 100
#
# The existing 1e-3 -> 1e-4 arm is the CAP-3 run itself; it is the control
# and is NOT re-run here.
#
# Everything else is carried over untouched and the driver enforces it:
# model weights, AdamW moments (never reset), RNG, data cursor, the canonical
# 27,981 population and its hash, the full 29,571 retrieval bank, unweighted
# retrieval CE, tau 0.10, batch 64, wd 1e-5, clip 1.0, encoder width 512 with
# wm/dec at the canonical 128, and the trainable scope.  Only the stage-2 LR
# differs, and a stage-2 LR that disagrees with the checkpoint is REFUSED
# unless --lr-transition is passed, so a silent change is impossible.
#
# Short by design: 300 exposures, evaluated at
# 100,110,125,150,175,200,250,300.  110 is deliberately early -- if 1e-3 is
# in fact unstable for C, it should show within ~10 exposures.  A branch
# stops itself the first time strict top-1 == 1.0 over all 27,981 targets
# (checkpoint saved first).  Each evaluation logs top1, top5, mean/median
# rank, target cosine, margin and retrieval CE; error counts follow from
# top1 x 27,981.
#
# Usage:
#   sbatch --job-name=l3_cap4_3e4 cap4_c512_lr_branch.slurm 3e-4 cap4_c512_lr3e4_seed22_full
#   sbatch --job-name=l3_cap4_1e3 cap4_c512_lr_branch.slurm 1e-3 cap4_c512_lr1e3_seed22_full
# ===========================================================================
set -euo pipefail

LR_STAGE2=${1:?usage: sbatch cap4_c512_lr_branch.slurm <stage2_lr> <run_id>}
RUN_ID=${2:?usage: sbatch cap4_c512_lr_branch.slurm <stage2_lr> <run_id>}

module purge
module load pytorch-gpu/py3/2.6.0
module load git

REPO=${L3_REPO:-$WORK/lichtheim3/lichtheim3}
cd "$REPO"
HEAD=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "== repo $REPO @ $HEAD"
[[ -z "$(git status --porcelain -uno)" ]] || { echo "FATAL: tracked tree dirty"; exit 1; }
test -f data/glove.6B.300d.txt || { echo "FATAL: real GloVe missing"; exit 1; }
if [[ -n "${L3_EXPECTED_COMMIT:-}" ]]; then
    [[ "$HEAD" == "$L3_EXPECTED_COMMIT" ]] || { echo "FATAL: HEAD != expected"; exit 1; }
fi

case "$RUN_ID" in
    cap4_*) : ;;
    *) echo "FATAL: run id must start with cap4_ (never a CAP1/2/3/FINAL dir)"; exit 1 ;;
esac

# ---- configuration -------------------------------------------------------
WM=128; ENC=512; DEC=128            # the CAP-3 C512 architecture, unchanged
SEED=22
SOURCE_STEP=43800                   # 100 exposures x 438 batches per pass
MAX_EXPOSURES=300
EVALS=100,110,125,150,175,200,250,300

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
PARENT_RUN_ID=${L3_CAP4_PARENT:-cap3_c_enc512_seed22_full}
PARENT_CKPT="$RUNS/$PARENT_RUN_ID/checkpoints/step_$(printf '%08d' $SOURCE_STEP).pt"
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

[[ "$RUN_ID" != "$PARENT_RUN_ID" ]] || { echo "FATAL: run id collides with the parent"; exit 1; }
[[ -f "$PARENT_CKPT" ]] || { echo "FATAL: C512@100 source $PARENT_CKPT not found"; exit 1; }
PARENT_SHA=$(sha256sum "$PARENT_CKPT" | cut -d' ' -f1)
echo "== source: $PARENT_CKPT"
echo "== source sha256: $PARENT_SHA"
# Optional pin: export L3_C512_SOURCE_SHA once you have read the value above,
# so both branches are provably continuing the same artefact.
if [[ -n "${L3_C512_SOURCE_SHA:-}" ]]; then
    [[ "$PARENT_SHA" == "$L3_C512_SOURCE_SHA" ]] || {
        echo "FATAL: source sha != L3_C512_SOURCE_SHA"; exit 1; }
fi

# ---- read-only preflight on the source artefact --------------------------
L3_CKPT="$PARENT_CKPT" L3_STEP="$SOURCE_STEP" python - <<'PY'
import os, sys, torch
ck = torch.load(os.environ["L3_CKPT"], map_location="cpu", weights_only=False)
want = int(os.environ["L3_STEP"])
def need(c, m):
    if not c:
        print(f"FATAL preflight: {m}", file=sys.stderr); sys.exit(1)
need(ck.get("format") == "lichtheim3.route_capacity.v1", "wrong format")
need(ck.get("route") == "comprehension", f"route {ck.get('route')!r}")
need(int(ck["global_step"]) == want, f"step {ck['global_step']} != {want}")
need(int(ck["per_epoch"]) == 438, f"per_epoch {ck['per_epoch']}")
need(ck["widths"] == {"wm_hidden": 128, "ltm_enc_hidden": 512,
                      "ltm_dec_hidden": 128}, f"widths {ck['widths']}")
need(int(ck["population_n"]) == 27981, f"population {ck['population_n']}")
need(ck["population_sha256"] ==
     "10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50",
     "canonical C hash mismatch")
need(float(ck["tau"]) == 0.10, f"tau {ck['tau']}")
need(float(ck.get("lr_stage2", 1e-4)) == 1e-4, "source is not the 1e-4 arm")
need("optimizer_state_dict" in ck and ck["optimizer_state_dict"]["state"],
     "no AdamW moments to preserve")
print(f"== preflight OK: step {ck['global_step']} "
      f"({ck['global_step']/ck['per_epoch']:.1f} exposures), "
      f"widths {ck['widths']}, pop {ck['population_n']}, tau {ck['tau']}, "
      f"moments {len(ck['optimizer_state_dict']['state'])} tensors")
PY

# ---- first launch vs requeue ---------------------------------------------
# First launch declares the stage-2 LR change from the source's 1e-4; a
# requeue resumes this branch's own newest checkpoint, whose stored stage-2
# LR already equals $LR_STAGE2, so nothing is re-declared.
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -z "${OWN_LATEST:-}" ]]; then
    RESUME_FROM="$PARENT_CKPT"
    EXTRA=(--lr-transition)
    echo "== first launch: branching C512 at 100 exposures"
    echo "== declaring stage-2 lr 1e-4 -> $LR_STAGE2 (AdamW moments preserved)"
else
    RESUME_FROM="$OWN_LATEST"
    echo "== requeue: resuming from $(basename "$OWN_LATEST")"
fi

srun python scripts/naming_comprehension/route_capacity_probe.py \
    --route comprehension --wm-hidden "$WM" --enc-hidden "$ENC" \
    --dec-hidden "$DEC" --seed "$SEED" --device cuda \
    --lr-stage2 "$LR_STAGE2" \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --eval-exposures "$EVALS" --max-exposures "$MAX_EXPOSURES" \
    --resume "$RESUME_FROM" --log-every 4380 "${EXTRA[@]}"

echo "== CAP-4 branch $RUN_ID finished: $RUNS/$RUN_ID"
echo "== grep 'CEILING REACHED' for strict top-1 == 1.0 over 27,981 targets"
echo "== control (1e-4) is the CAP-3 run itself: $RUNS/$PARENT_RUN_ID/metrics.tsv"
echo "== archive: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_cap4/"
