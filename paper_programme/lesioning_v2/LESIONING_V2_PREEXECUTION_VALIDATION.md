# LESIONING V2 — PRE-EXECUTION VALIDATION

    DATE  2026-09-21
    SCIENTIFIC_LESION_EXECUTED = NO

## 1. What was validated here (local, synthetic)

    64 tests passed, 0 failed, 0 skipped

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

## 2. What CANNOT be validated locally — the three cluster gates

The P1-P4 artifacts are on Jean Zay, not on this machine. These three gates are
implemented and ready but must run there.

    GATE 1  verify_states.py      actual P1-P4 state_dict names/shapes
    GATE 2  verify_gru_layout.py  torch 2.6.0 GRU layout
    GATE 3  extract_intact_sd.py  intact-only SD constants

GATE 2 has been run locally under torch 2.12.1 and PASSES
(`contract/LESIONING_V2_GRU_LAYOUT.json`), but the canonical environment is
2.6.0 and the record must be produced there. Note the operator does not depend
on the documented gate ORDER at all — only on the (3H, D) shape invariant — so
a layout change could not silently corrupt a mask; a test proves the tiling is
invariant under permutation of the three blocks.

Until all three gates pass on the cluster:

    L1_TENSOR_VERIFIED           = NO (pending GATE 1)
    L2_TENSOR_VERIFIED           = NO (pending GATE 1)
    L3_TENSOR_VERIFIED           = NO (pending GATE 1)
    TORCH_GRU_LAYOUT_VERIFIED    = NO (pending GATE 2 under 2.6.0)
    INTACT_SD_CONSTANTS_FROZEN   = NO (pending GATE 3)

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

Then STOP and return to CENTRAL with the three gate artifacts. A real run
requires an authorization artifact CENTRAL has not issued.

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
