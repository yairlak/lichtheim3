#!/bin/bash
#SBATCH --job-name=l3_u1200_audit
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# ===========================================================================
# READ-ONLY per-item C/N/R error audit at u1200, plus the shared u850 source
# needed for the persistence chain.  Trains NOTHING and modifies no
# checkpoint; writes only <run>/error_audit_u<U>/.
#
#   array 0-3  : C-HIGH   u1200  seeds 19-22   step 3,333,600
#   array 4-7  : ALL-3e-5 u1200  seeds 19-22   step 3,333,600
#   array 8-11 : shared   u850   seeds 19-22   step 2,361,300
#                (final_lrpilot_3e5_h512_sXX -- the common ancestor of both
#                 u1200 arms, so u850 -> u1200 persistence is well defined)
#
# The u850 pass is what makes questions 1-3 answerable: what fraction of the
# u850 C errors are still wrong at u1200, what fraction of the u1200 residual
# is new, and what the within-seed retention is.
# ===========================================================================
set -euo pipefail
module purge
module load pytorch-gpu/py3/2.6.0

[[ -n "${L3_REPO:-}" ]] || { echo "FATAL: export L3_REPO"; exit 1; }
cd "$L3_REPO"
RUNS=${L3_RUNS:-/lustre/fsn1/projects/rech/llg/uss35bp/lichtheim3_runs}
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


RUNIDS=(final_lrpilot1200_chigh_h512_s19 final_lrpilot1200_chigh_h512_s20 \
        final_lrpilot1200_chigh_h512_s21 final_lrpilot1200_chigh_h512_s22 \
        final_lrpilot1200_all3e5_h512_s19 final_lrpilot1200_all3e5_h512_s20 \
        final_lrpilot1200_all3e5_h512_s21 final_lrpilot1200_all3e5_h512_s22 \
        final_lrpilot_3e5_h512_s19 final_lrpilot_3e5_h512_s20 \
        final_lrpilot_3e5_h512_s21 final_lrpilot_3e5_h512_s22)
STEPS=(3333600 3333600 3333600 3333600 3333600 3333600 3333600 3333600 \
       2361300 2361300 2361300 2361300)

IDX=${SLURM_ARRAY_TASK_ID:?must run as an array job}
(( IDX >= 0 && IDX < 12 )) || { echo "FATAL: index $IDX out of range"; exit 1; }
RUN_ID=${RUNIDS[$IDX]}; STEP=${STEPS[$IDX]}
U=$(( STEP / (463*6) ))
CKPT="$RUNS/$RUN_ID/checkpoints/step_$(printf '%08d' $STEP).pt"
OUT="$RUNS/$RUN_ID/error_audit_u${U}"

[[ -f "$CKPT" ]] || { echo "FATAL: $CKPT not found"; exit 1; }
echo "== auditing $RUN_ID at u=$U (step $STEP)"
echo "== sha256 $(sha256sum "$CKPT" | cut -d' ' -f1)"
echo "== READ-ONLY: writes only $OUT"

srun python scripts/naming_comprehension/base123_error_audit.py \
    --ckpt "$CKPT" --out-dir "$OUT" --device cuda \
    --glove-path "$GLOVE"

echo "== done -> $OUT"
