# Jean Zay support for Lichtheim3 (Phase FINAL-1)

Cluster material for the FINAL multitask driver
(`scripts/naming_comprehension/train_joint_scratch.py`) on the **already
validated** Lichtheim3 Jean Zay environment. Earlier Lichtheim3 work on Jean
Zay validated CUDA training, checkpointing, optimizer/RNG resume (CUDA RNG
included), deterministic AR evaluation and V100 execution; nothing here
redesigns that setup — these scripts only validate and run the NEW driver.

## Validated environment (fixed)

| item | value |
|---|---|
| user / project | `uss35bp` / `llg` |
| WORK | `/lustre/fswork/projects/rech/llg/uss35bp` |
| SCRATCH | `/lustre/fsn1/projects/rech/llg/uss35bp` |
| STORE | `/lustre/fsstor/projects/rech/llg/uss35bp` |
| repo | `$WORK/lichtheim3/lichtheim3` |
| modules | `module purge; module load pytorch-gpu/py3/2.6.0; module load git` |
| SLURM (V100) | `--account=llg@v100 --partition=gpu_p13 --qos=qos_gpu-t3 --gres=gpu:1 --cpus-per-task=10 --hint=nomultithread` (no `--mem`) |

## Repository contract

Scientific jobs REQUIRE, and verify at job start:

- branch `feat/joint-multitask-scratch`;
- `HEAD == $L3_EXPECTED_COMMIT` (exported at submit time; see the block below);
- clean **tracked** working tree (`git status --porcelain -uno` empty —
  untracked historical files such as old SLURM logs do NOT block);
- real GloVe present; population sizes and the frozen canonical hashes
  (`C = 27,981`, sha `10c2f06e…`) are asserted by the driver itself.

## Files

| file | purpose |
|---|---|
| `final_preflight.slurm` | ONE compact preflight job: environment, git contract, GloVe, FINAL tests, R/N/C/bank sizes, CUDA final_full smoke with bitwise exact resume + endpoint eval, CUDA throughput benchmark with peak GPU memory, PASS/FAIL summary. NOT scientific. |
| `scientific_run.slurm` | FINAL-1: one continuous exactly-resumable job to 700 N-exposures, full-population milestone evals at 300/500/700 (no in-run decisions). Requeue-safe; resubmission resumes exactly. |
| `../benchmark_throughput.py` | Matched Mac/Jean-Zay throughput benchmark (identical command; only `--device` differs). |

## Preflight (run this once before FINAL-1)

Copy-paste on Jean Zay (fill only `<FINAL_SHA>` — the value is given with each
release message):

```bash
export L3_EXPECTED_COMMIT=<FINAL_SHA>
cd $WORK/lichtheim3/lichtheim3
git fetch origin
git checkout feat/joint-multitask-scratch
git pull --ff-only origin feat/joint-multitask-scratch
[ "$(git rev-parse HEAD)" = "$L3_EXPECTED_COMMIT" ] || echo "STOP: HEAD mismatch"
test -f data/glove.6B.300d.txt || bash data/get_glove.sh
cd scripts/cluster/jeanzay
sbatch --export=ALL,L3_EXPECTED_COMMIT final_preflight.slurm
```

Monitor / collect:

```bash
squeue -u uss35bp
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
tail -f l3_final_preflight_<JOBID>.out
cat $SCRATCH/lichtheim3_preflight/summary_<JOBID>.txt        # PASS/FAIL verdict
cat $SCRATCH/lichtheim3_preflight/bench_cuda_<JOBID>.json    # throughput numbers
```

## FINAL-1 scientific run (WAIT for explicit GO)

After the preflight PASSes and a walltime is chosen from the measured
throughput, edit `<WALLTIME>` in `scientific_run.slurm`, then:

```bash
export L3_EXPECTED_COMMIT=<FINAL_SHA>
cd $WORK/lichtheim3/lichtheim3/scripts/cluster/jeanzay
sbatch --export=ALL,L3_EXPECTED_COMMIT scientific_run.slurm
# resume after walltime/preemption: just resubmit the same command —
# the job finds the latest checkpoint and continues exactly.
```

## Platform decision rule (Mac reference already measured)

Mac CPU (FINAL-1A): **0.0672 s/step = 1.87 h per 100k summed optimizer
steps** (`bench` protocol: 30 warmup + 200 timed steps, eval and init
excluded; the JSON also reports `examples_per_second`, `trainer_init_seconds`,
`end_to_end_seconds`, and on CUDA `peak_gpu_memory_mb`). Use Jean Zay for
long runs only if its `projected_hours_per_100k_steps` is at most half the
Mac's; otherwise the Mac remains the reference platform and Jean Zay's value
is concurrency (parallel seeds/schedules). No GPU speed is assumed before
the preflight measurement.

## Determinism notes

- Training streams never read the global RNG; checkpoints save/restore
  Python/NumPy/Torch-CPU/Torch-CUDA RNG states; exact resume is bitwise
  (tests + preflight verify).
- Optional strict algorithm determinism: `--torch-deterministic` plus
  `export CUBLAS_WORKSPACE_CONFIG=:4096:8`. Never enabled silently; measure
  its cost first (`benchmark_throughput.py --device cuda --torch-deterministic`).
- CPU-vs-GPU bit-identity is not guaranteed by PyTorch; keep any one run's
  training and evaluation on a single device class.
