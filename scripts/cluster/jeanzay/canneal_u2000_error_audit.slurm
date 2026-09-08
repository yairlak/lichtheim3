#!/bin/bash
#SBATCH --job-name=l3_u2000_audit
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#
# ===========================================================================
# READ-ONLY per-item C/N/R error audit of the u2000 CANNEAL CONTROL endpoints.
# Trains NOTHING, takes no optimizer step and modifies no checkpoint; writes
# only <run>/error_audit_u2000/.
#
#   array 0-3 : final_canneal_ctrl_h512_s{19,20,21,22}
#               step 5,556,000 = u2000, C errors [52,62,51,56], mean 55.25
#
# The control arm ONLY.  The 5e-5 and 3e-5 anneal arms are archived but not
# audited: they lost to the control on C in every seed, so their residual is
# not the state any further work continues from.
#
# The audit rebuilds the trainer from the CHECKPOINT'S OWN config -- task LRs,
# optimizer policy, dec_weight, c_align_weight -- so no --phase-transition is
# needed and none is passed.  base123_error_audit.py fingerprints every
# parameter tensor before the readouts and re-checks it afterwards, raising if
# a single element moved; "read-only verified" in the log is that proof.
#
# Pairs with the already-archived u1400 audit under
# final_rep_rescue123_h512_s<seed>/error_audit_u1400/ to give the preregistered
# u1400 -> u2000 persistence chain.  NO trigger threshold is touched anywhere.
#
# Usage: sbatch --export=ALL,L3_REPO,L3_GLOVE,L3_RUNS \
#            scripts/cluster/jeanzay/canneal_u2000_error_audit.slurm
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

RUNIDS=(final_canneal_ctrl_h512_s19 final_canneal_ctrl_h512_s20 \
        final_canneal_ctrl_h512_s21 final_canneal_ctrl_h512_s22)
STEPS=(5556000 5556000 5556000 5556000)

IDX=${SLURM_ARRAY_TASK_ID:?must run as an array job}
(( IDX >= 0 && IDX < 4 )) || { echo "FATAL: index $IDX out of range"; exit 1; }
RUN_ID=${RUNIDS[$IDX]}; STEP=${STEPS[$IDX]}
U=$(( STEP / (463*6) ))
CKPT="$RUNS/$RUN_ID/checkpoints/step_$(printf '%08d' $STEP).pt"
OUT="$RUNS/$RUN_ID/error_audit_u${U}"

# Refuse to audit anything but the canneal CONTROL endpoints.
case "$RUN_ID" in
    final_canneal_ctrl_h512_s19|final_canneal_ctrl_h512_s20|\
final_canneal_ctrl_h512_s21|final_canneal_ctrl_h512_s22) : ;;
    *) echo "FATAL: refusing unexpected run id $RUN_ID"; exit 1 ;;
esac
(( U == 2000 )) || { echo "FATAL: derived u=$U, expected 2000"; exit 1; }

[[ -f "$CKPT" ]] || { echo "FATAL: $CKPT not found"; exit 1; }
CKPT_SHA_BEFORE=$(sha256sum "$CKPT" | cut -d' ' -f1)
echo "== auditing $RUN_ID at u=$U (step $STEP)"
echo "== sha256 before: $CKPT_SHA_BEFORE"
echo "== READ-ONLY: writes only $OUT"

srun python scripts/naming_comprehension/base123_error_audit.py \
    --ckpt "$CKPT" --out-dir "$OUT" --device cuda \
    --glove-path "$GLOVE"

# Second, independent read-only proof: the checkpoint file itself is untouched.
CKPT_SHA_AFTER=$(sha256sum "$CKPT" | cut -d' ' -f1)
[[ "$CKPT_SHA_AFTER" == "$CKPT_SHA_BEFORE" ]] || {
    echo "FATAL: the audit MODIFIED $CKPT"; exit 1; }
echo "== sha256 after:  $CKPT_SHA_AFTER  (checkpoint byte-identical)"
echo "== done -> $OUT"
