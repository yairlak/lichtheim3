# LESIONING V2 — FINAL EXECUTABLE PREFLIGHT

Implemented in `execution/preflight.py::run`. Every check fails closed and none
attempts repair. No scientific value is consulted anywhere.

| # | check | implementation | failure |
|---|---|---|---|
| A | authorization names the exact execution commit; `git rev-parse HEAD` must equal it | `authorization.validate` | `AuthorizationRefused` |
| B | `git status --porcelain` empty before a real run | `preflight.run` step B | `PreflightError` |
| C | run-matrix semantic SHA recomputes to `cd48e99c…1732a`; `b4d3c988…887c` explicitly rejected | step C + `_recompute_matrix_sha` | `PreflightError` |
| D | both transfer-required artifacts present and SHA-exact | step D, `TRANSFER_REQUIRED` | `PreflightError` |
| E | package `SHA256SUMS` verified completely | step E | `PreflightError` |
| F | P1-P4 POST_REPAIR identities against the transferred state verification; no SOURCE state; no `head_final` | step F + matrix scan | `PreflightError` |
| G | frozen constants: sites, p_max=0.30, 15 levels, mask/activation semantics, realization counts, seed namespace, SD procedure hash | step G | `PreflightError` |
| H | evaluator identity hash over the exact science-bearing files | `evaluator_identity()` | `PreflightError` |
| I | `torch == 2.6.0` for a real run | step I | `PreflightError` |
| J | SD constants produced by the frozen procedure and covering every cell | step J | `PreflightError` |
| K | no `backward` / optimizer / `step` / `zero_grad` / `requires_grad_` / save reachable | step K, AST scan | `PreflightError` |
| L | scientific result namespace absent | step L | `PreflightError` |

`--dry-run` runs A(partial: HEAD recorded, authorization not required) and
C, E, G, H, K, L in full, tolerates absent transfer artifacts, and performs
ZERO model forwards.

## Required transfer artifacts

    LESIONING_V2_STATE_VERIFICATION.json
      d216fe562396be19d918a1f7c433839e846ca94602c6aae1e21754c931f96349
    LESIONING_V2_INTACT_SD_CONSTANTS.json
      3c6dcd3595d7432d592bcecbc691d33ec117f816932771ec1fccc0b0001b0602

They are NOT reconstructed. Stage them from the external execution-control
directory via `--transfer-dir`, which keeps the source checkout clean for
check B. A wrong hash fails closed.

## Authorization artifact

    {
      "token": "CENTRAL_GO_FOR_SCIENTIFIC_EXECUTION_LESIONING_V2",
      "go_for_scientific_execution": true,
      "execution_commit": "<EXACT FINAL EXECUTION COMMIT>",
      "run_matrix_sha256":
        "cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a"
    }

Exposed via `L3_LESION_V2_AUTHORIZATION`, and kept OUTSIDE the worktree.
Rejection is tested for: missing artifact, wrong token, false GO, wrong matrix,
superseded matrix, wrong execution commit, missing field.

No valid production authorization exists in this repository; a test asserts it.

## Exact Jean-Zay commands (DO NOT RUN)

    module load pytorch-gpu/py3/2.6.0
    cd $REPO && git fetch --all
    git checkout paper-programme/lesioning-v2-execution-integration-repair
    git rev-parse HEAD                       # must equal authorization.execution_commit
    git status --porcelain                   # must be empty

    export L3_LESION_V2_CONTROL_DIR=$WORK/l3_lesion_v2_control
    export L3_LESION_V2_RESULTS=$SCRATCH/l3_lesion_v2_results_0b22b904   # OUTSIDE the worktree
    export L3_LESION_V2_AUTHORIZATION=$L3_LESION_V2_CONTROL_DIR/authorization.json

    # stage the two transfer-required artifacts into $L3_LESION_V2_CONTROL_DIR
    sha256sum $L3_LESION_V2_CONTROL_DIR/LESIONING_V2_STATE_VERIFICATION.json
    sha256sum $L3_LESION_V2_CONTROL_DIR/LESIONING_V2_INTACT_SD_CONSTANTS.json

    python -m pytest paper_programme/lesioning_v2/tests -q      # expect 110 passed

    python paper_programme/lesioning_v2/scripts/run_lesion_v2.py \
        --dry-run --out-dir "$L3_LESION_V2_RESULTS" \
        --transfer-dir "$L3_LESION_V2_CONTROL_DIR"

    # ONLY after a NEW CENTRAL authorization naming this commit:
    sbatch paper_programme/lesioning_v2/scripts/submit_lesion_v2.slurm
