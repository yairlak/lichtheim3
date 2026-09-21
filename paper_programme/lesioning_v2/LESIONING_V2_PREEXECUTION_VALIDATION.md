# LESIONING V2 — PRE-EXECUTION VALIDATION

    DATE  2026-09-21 (closed 2026-09-21 after cluster validation)
    SCIENTIFIC_LESION_EXECUTED = NO
    ALL THREE CLUSTER GATES PASSED under torch 2.6.0

## 1. What was validated here (local, synthetic)

    70 tests passed, 0 failed, 0 skipped
    (64 at implementation freeze + 6 closure tests pinning the
     canonical artifact hashes and the matrix-bound authorization)

Covering all 23 required checks:

| # | check | status |
|---|---|---|
| 1-3 | exact logical-edge counts L1 8,192 / L2 32,768 / L3 153,600 | PASS |
| 4 | synchronized 3-block GRU masking; linear site not tiled | PASS |
| 5 | nestedness k1 subset ... subset k15, all three sites | PASS |
| 6 | exact removed count at every severity | PASS |
| 7 | deterministic mask reproduction | PASS |
| 8 | different realizations differ | PASS |
| 9 | different states differ | PASS |
| 10 | different sites differ | PASS |
| 11 | biases untouched | PASS |
| 12 | `weight_hh` untouched | PASS |
| 13 | `phon_embed` untouched | PASS |
| 14 | L3 lesions `sem_to_h0.weight` only; decoder and bias intact | PASS |
| 15 | item-frozen activation base | PASS |
| 16 | same base noise across all 15 severities (ratio exactly 15:1) | PASS |
| 17 | same realization across paired task/decoder conditions | PASS |
| 18 | max half-width = SD_site; distribution SD = SD_site/sqrt(3) | PASS |
| 19 | exact parameter restoration (by stored clone) | PASS |
| 20 | no leakage across 15 successive contexts | PASS |
| 21 | k=0 intact identity | PASS |
| 22 | no optimizer / backward / zero_grad path anywhere in the operator | PASS |
| 23 | runner refuses execution without a valid authorization artifact | PASS |

Additional: untouchable targets are refused at context entry; authorization
must match both the token and this exact matrix hash; SD procedure and output
schema are frozen and hashable; run matrix internally consistent.

Runner behaviour:

    --dry-run            PREFLIGHT_OK, zero forwards, writes nothing
    (no flag, no auth)   refuses; currently at the SD gate, and at the
                         authorization guard once SD constants exist

## 2. Cluster gates — ALL PASSED (torch 2.6.0)

    IMPLEMENTATION_TESTS         = 64 PASS on the cluster
    GATE 1  verify_states.py     PASS   L1/L2/L3 tensors verified on the
                                        ACTUAL P1-P4 POST_REPAIR state_dicts
    GATE 2  verify_gru_layout.py PASS   under the canonical torch 2.6.0
    GATE 3  extract_intact_sd.py PASS   intact-only SD constants frozen

    L1_TENSOR_VERIFIED           = YES
    L2_TENSOR_VERIFIED           = YES
    L3_TENSOR_VERIFIED           = YES
    TORCH_GRU_LAYOUT_VERIFIED    = YES
    INTACT_SD_PROCEDURE_FROZEN   = YES
    INTACT_SD_CONSTANTS_FROZEN   = YES

GRU runtime under 2.6.0: `three_contiguous_H_blocks=true`,
`logical_mask_tiles_identically=true`, `operator_depends_on_gate_names=false`,
documented order `r,z,n`. The operator never consults that order — only the
(3H, D) shape invariant — and a test proves the tiling is invariant under
permutation of the three blocks.

### Frozen intact site SDs

    state              L1         L2         L3
    P1_POST_REPAIR     0.492585   0.524076   0.820609
    P2_POST_REPAIR     0.477633   0.535038   0.824531
    P3_POST_REPAIR     0.491430   0.572050   0.839828
    P4_POST_REPAIR     0.482054   0.489595   0.802448

    procedure hash 122ae40c600e639f875bdcb18e588bd32242981af0fa42c0e975a296bc851942

These are recorded for the written record. The OPERATIONAL source is the
transferred `LESIONING_V2_INTACT_SD_CONSTANTS.json` with its full per-constant
provenance; the values above must never be hand-entered into an execution path.

At k=15 the perturbation half-width equals the site SD above; the distribution
SD is that value divided by sqrt(3) (e.g. P1/L3: half-width 0.820609,
distribution SD 0.473779).

### Canonical artifacts and their binding hashes

    LESIONING_V2_RUN_MATRIX.json          751a71d6...de150   in repo
    LESIONING_V2_GRU_LAYOUT.json          762d9547...b49c37  in repo
    LESIONING_V2_STATE_VERIFICATION.json  d216fe56...f96349  TRANSFER REQUIRED
    LESIONING_V2_INTACT_SD_CONSTANTS.json 3c6dcd35...b0602   TRANSFER REQUIRED

    RUN_MATRIX_SEMANTIC_SHA256
    cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a

The run matrix and the GRU layout record were **reproduced byte-identically off
the cluster** from the frozen code, and their installed copies match the
canonical hashes exactly — an independent confirmation that the cluster ran this
code and nothing else.

The other two artifacts contain cluster-local absolute paths, deployed-head
digests, the population hash and descriptive statistics that cannot be derived
off-cluster. They were NOT reconstructed: fabricating a file to match a
published hash would defeat the purpose of the hash. They must be transferred
into `contract/` and verified against the hashes above before execution. A test
asserts they are absent while declared transfer-required, so the two states
cannot be confused.

Until `LESIONING_V2_INTACT_SD_CONSTANTS.json` is present, the runner refuses any
real run at the SD gate — the safe failure.

## 3. Exact Jean-Zay commands

    module load pytorch-gpu/py3/2.6.0
    cd <repo>
    git fetch --all
    git checkout paper-programme/lesioning-v2-implementation
    git rev-parse HEAD

    export L3_V7_RUN_ROOT=/lustre/fsn1/projects/rech/llg/uss35bp/l3_fresh_v7_runs

    # local-equivalent suite must still pass
    python -m pytest paper_programme/lesioning_v2/tests -q          # expect 64 passed

    # GATE 1 - hard stop on any mismatch
    python paper_programme/lesioning_v2/scripts/verify_states.py \
        --v7-run-root "$L3_V7_RUN_ROOT"

    # GATE 2 - record the layout under 2.6.0
    python paper_programme/lesioning_v2/scripts/verify_gru_layout.py

    # GATE 3 - intact only; refuses to overwrite an existing constants file
    python paper_programme/lesioning_v2/scripts/extract_intact_sd.py \
        --v7-run-root "$L3_V7_RUN_ROOT" \
        --expect-procedure-hash 122ae40c600e639f875bdcb18e588bd32242981af0fa42c0e975a296bc851942

    # rebuild the matrix against the cluster-verified identities
    python paper_programme/lesioning_v2/scripts/build_run_matrix.py \
        --state-sha-json paper_programme/lesioning_v2/contract/LESIONING_V2_STATE_VERIFICATION.json

    # dry run only. This must NOT be followed by a real run.
    python paper_programme/lesioning_v2/scripts/run_lesion_v2.py --dry-run

All three gates have now been run and passed. A real run still requires an
authorization artifact CENTRAL has not issued, and that artifact must bind

    run_matrix_sha256 = cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a

An authorization naming the earlier pre-cluster matrix (`b4d3c988...`) is
refused; a test pins that behaviour.

## 4. Hard-stop conditions (section 15), all implemented

Every one raises and returns nonzero; none attempts automatic repair.

  * P1-P4 tensor name/shape differs from the freeze -> `VerificationFailed`
  * torch GRU layout unverifiable -> `verify_gru_layout.py` exits 2
  * logical mask does not tile identically -> `physical_mask` raises
  * nestedness failure -> caught by test; masks are prefix by construction
  * task/decoder altering a realization -> structurally impossible (absent from
    the RNG identity) and asserted by test
  * pristine restoration failure -> `PristineViolation`
  * SD provenance differs -> `extract_intact_sd.py` exits 2 on hash mismatch
  * backward/optimizer reachable -> runner preflight AST scan raises
  * SOURCE state substituted -> runner preflight rejects non-POST_REPAIR
  * `head_final` referenced -> runner preflight AST scan rejects
  * results directory already present -> runner preflight refuses

## 5. Declared carry-forward limitations

Inherited from the accepted operator audit and unchanged here:

1. p_max=0.30 rests on D-05, which used SCALAR granularity; the evidence bounds
   the dose, not the response.
2. No site-specific dose evidence exists for L2, and none for L3.
3. D-05 was run on a non-V7 checkpoint.
4. An L3 lesion acts on naming and repetition through the same interface.
5. Nesting and the uniform noise distribution are Lichtheim3 adaptations, not
   attributable to Ueno.
