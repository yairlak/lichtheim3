# Jean Zay support for Lichtheim3 (Phase FINAL-1A)

Minimal cluster material for running the Phase-4 joint driver
(`scripts/naming_comprehension/train_joint_scratch.py`) on Jean Zay.
Nothing here is submitted automatically; you run every command yourself.

## Files

| file | purpose |
|---|---|
| `smoke_benchmark.slurm` | 30-minute smoke: CPU + GPU throughput, determinism A/B, checkpoint + exact-resume test. NOT scientific. |
| `scientific_run.slurm` | Template for a real run: requeue-safe, resumes exactly from the latest checkpoint on resubmission. |
| `../benchmark_throughput.py` | Matched Mac/Jean-Zay throughput benchmark (same command both platforms, only `--device` changes). |

## What you must fill in (no values are invented here)

In both `.slurm` files:

1. `<JEANZAY_ACCOUNT>` — your project accounting, e.g. `abc@v100` (from `idracct` / your project welcome mail).
2. The environment block — either `module load <JEANZAY_PYTORCH_MODULE>` (find one with `module avail pytorch`) or `source <your env>/bin/activate`. Requirements: `torch>=2.0`, `numpy`, `matplotlib`, `Levenshtein` (see `requirements.txt`).
3. Optional `<JEANZAY_PARTITION>` / `<JEANZAY_QOS>` / `<JEANZAY_CONSTRAINT>` lines — only if your allocation requires them; otherwise leave commented.
4. `scientific_run.slurm` only: `<JEANZAY_WALLTIME>` and `<TOTAL_REP_EPOCHS>`.
5. Paths: the scripts use `L3_REPO` (default `$HOME/lichtheim3`) and `$SCRATCH`-based work dirs. Checkpoints on `$SCRATCH` are purge-eligible — archive finished runs to `$STORE` or `$WORK`.

## One-time setup (login node)

```bash
# clone / sync the repo, then:
cd $L3_REPO
bash data/get_glove.sh          # real GloVe — REQUIRED for scientific runs
                                # (smoke/benchmark use --allow-glove-fallback)
python -m pytest tests/test_final_populations.py tests/test_joint_scratch.py -q
```

## Smoke benchmark (run this before any scientific job)

```bash
cd $L3_REPO/scripts/cluster/jeanzay
sbatch smoke_benchmark.slurm
```

Check status / logs:

```bash
squeue -u $USER
tail -f l3_smoke_<JOBID>.out
```

Collect results (from your Mac):

```bash
scp jean-zay:'$SCRATCH/lichtheim3_smoke/bench_*.json' .
```

The smoke verifies, on one job: CPU throughput, one-GPU CUDA throughput,
the cost of `--torch-deterministic`, checkpoint creation, and an
interrupted-then-resumed execution (150 steps → checkpoint → resume → 300).

## Scientific run

```bash
# edit scientific_run.slurm (regime/seed/epochs/walltime), then:
sbatch scientific_run.slurm
```

Resume after timeout/preemption/cancellation — just resubmit; the script
finds the latest checkpoint and continues exactly:

```bash
sbatch scientific_run.slurm
```

## Mac vs Jean Zay decision rule (matched benchmark protocol)

Same command on both platforms (only `--device` differs), 30 warmup steps,
200 timed steps, evaluation excluded from the timed window; the JSON reports
`seconds_per_step`, `steps_per_second`, `examples_per_second`,
`projected_hours_per_100k_steps`, and separately `trainer_init_seconds` and
`end_to_end_seconds`:

```bash
# Mac (reference, already measured in FINAL-1A):
python scripts/cluster/benchmark_throughput.py --device cpu

# Jean Zay (inside the smoke job): --device cpu, then --device cuda
```

Decision rule: use Jean Zay for long runs only if its
`projected_hours_per_100k_steps` is **at most half** of the Mac's measured
value (the model is ~433k parameters at batch 64 — GPU benefit is NOT assumed;
queue latency and $SCRATCH purges are real costs). Otherwise run on the Mac
and reserve Jean Zay for parallel multi-seed/multi-schedule sweeps, where
concurrency, not per-step speed, is the payoff.

## Determinism notes

- The driver's training streams never touch the global RNG; checkpoints save
  and restore Python/NumPy/Torch-CPU/Torch-CUDA RNG states.
- Bitwise CUDA reproducibility additionally needs
  `--torch-deterministic` **and** `export CUBLAS_WORKSPACE_CONFIG=:4096:8`
  (both wired into the templates). This may cost throughput — measure it with
  the smoke A/B and decide explicitly; it is never enabled silently.
- CPU→GPU and GPU-model→CPU-eval bit-identity is NOT guaranteed by PyTorch;
  cross-device comparisons must be tolerance-based, and any mixed-platform
  plan should keep training and evaluation on one device class per run.
