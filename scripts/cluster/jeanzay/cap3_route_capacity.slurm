#!/bin/bash
#SBATCH --account=llg@v100
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --requeue

# ===========================================================================
# CAP-3 — route-isolated capacity search, {128, 256, 512}, per Yair's request
# for "the same search for Ventral and dorsal".
#
# CAP-1/CAP-2 tested the ventral PRODUCTION route (LTM decoder width) and
# solved full-lexicon naming at 512.  CAP-3 runs the identical protocol on
# the two routes that probe did not touch:
#
#   comprehension : ltm.enc_hidden 128/256/512.  Path is strictly
#       phonology -> phon_embed -> ltm.encoder -> to_semantic -> s_hat ->
#       cosine retrieval against the FULL 29,571 bank.  The production
#       decoder never runs and is frozen.  Population = the frozen canonical
#       27,981 one-target-per-phonology set (hash checked below).  Criterion
#       = STRICT top-1 word ID; top-5/rank/cos/margin are diagnostics only.
#
#   dorsal        : wm.hidden 128/256/512.  Path is phonology -> WM encoder
#       -> WM decoder -> to_premotor -> motor, with the ENTIRE LTM route
#       frozen, so lexical retrieval cannot solve repetition.  Readouts:
#       route-isolated WM autoregressive repetition on the 29,571 lexical
#       items (task adequacy) and on the committed WFE pseudoword set,
#       split short/long by median length (the capacity diagnostic -- a real
#       WM generalises to novel forms, a memoriser does not).
#
# ONE FACTOR PER JOB.  The three widths are independent CLI arguments; a job
# sets exactly one away from the canonical 128 and the other two stay there.
# Naming's 512 decoder is deliberately NOT carried in: it would confound the
# comparison.  Shared by every job: seed 22, counter-addressed item order,
# AdamW, wd 1e-5, clip 1.0, batch 64, teacher forcing, and the CAP-2-validated
# two-stage schedule lr 1e-3 through exposure 100 then 1e-4 -- applied in-run,
# never tuned per width.
#
# Losses are each route's isolated UNWEIGHTED objective.  lambda_C = 0.087 is
# the JOINT-task weight on the sole C term in train_joint_scratch; it is not
# an intrinsic property of the C objective and is deliberately not carried
# into this probe, so probe CE is NOT comparable to joint retrieval_ce.
#
# Usage:  sbatch --job-name=<name> --time=<hh:mm:ss> cap3_route_capacity.slurm \
#             <route> <width> <run_id> <max_exposures> <eval_exposures>
# ===========================================================================
set -euo pipefail

ROUTE=${1:?usage: <route> <width> <run_id> <max_exposures> <eval_exposures>}
WIDTH=${2:?missing width}
RUN_ID=${3:?missing run id}
MAX_EXP=${4:?missing max exposures}
EVALS=${5:?missing eval exposures}

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
    cap3_*) : ;;
    *) echo "FATAL: run id must start with cap3_ (never a CAP1/CAP2/FINAL dir)"; exit 1 ;;
esac

# The frozen canonical C population, re-asserted here so a lexicon drift can
# never silently change what "comprehension" means.
EXPECTED_C_N=27981
EXPECTED_C_HASH=10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50
echo "== canonical C contract: n=$EXPECTED_C_N hash=$EXPECTED_C_HASH"
echo "   (the driver aborts on any mismatch; retrieval bank stays 29,571)"

# ---- exactly one width leaves the canonical 128 --------------------------
WM=128; ENC=128; DEC=128
case "$ROUTE" in
    comprehension) ENC=$WIDTH ;;
    dorsal)        WM=$WIDTH ;;
    *) echo "FATAL: route must be comprehension|dorsal, got $ROUTE"; exit 1 ;;
esac
echo "== $ROUTE probe: wm=$WM enc=$ENC dec=$DEC (one factor = $ROUTE width $WIDTH)"

RUNS=${L3_RUNS:-$SCRATCH/lichtheim3_runs}
OWN_CKPTS="$RUNS/$RUN_ID/checkpoints"

# ---- short V100 preflight benchmark --------------------------------------
python scripts/naming_comprehension/route_capacity_probe.py \
    --route "$ROUTE" --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC" \
    --out-dir "$JOBSCRATCH" --run-id bench --eval-exposures 0 \
    --max-exposures "$MAX_EXP" --benchmark 200

# ---- requeue-safe resume from this run's own newest checkpoint ------------
OWN_LATEST=$(ls "$OWN_CKPTS/"step_*.pt 2>/dev/null | sort | tail -1 || true)
EXTRA=()
if [[ -n "${OWN_LATEST:-}" ]]; then
    EXTRA=(--resume "$OWN_LATEST")
    echo "== requeue: resuming from $(basename "$OWN_LATEST")"
else
    echo "== first launch: training from scratch"
fi

srun python scripts/naming_comprehension/route_capacity_probe.py \
    --route "$ROUTE" --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC" \
    --seed 22 --device cuda \
    --out-dir "$RUNS" --run-id "$RUN_ID" \
    --eval-exposures "$EVALS" --max-exposures "$MAX_EXP" \
    --log-every 4630 "${EXTRA[@]}"

echo "== CAP-3 $ROUTE width $WIDTH finished: $RUNS/$RUN_ID"
echo "== grep 'CEILING REACHED' for strict top-1 == 1.0 (C) / lexical exact == 1.0 (WM)"
echo "== provenance: $RUNS/$RUN_ID/config.json"
echo "== archive: cp -r $RUNS/$RUN_ID \$STORE/lichtheim3_cap3/"
