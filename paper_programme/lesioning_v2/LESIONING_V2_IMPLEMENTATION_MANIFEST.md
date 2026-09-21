# LESIONING V2 — IMPLEMENTATION MANIFEST

    BRANCH    paper-programme/lesioning-v2-implementation
    WORKTREE  ../wt-lesioning-v2-implementation
    BASE      5d1849a0c1d01065ff926d324703e8b767c05223 (accepted operator audit)
    DATE      2026-09-21

    GO_FOR_IMPLEMENTATION       = YES (this pass)
    GO_FOR_SCIENTIFIC_EXECUTION = NO
    SCIENTIFIC_LESION_EXECUTED  = NO

No nonzero lesion was run on P1/P2/P3/P4. No training, no optimizer, no
backward. Every behavioural test uses a synthetic toy module.

## 1. Package layout

    paper_programme/lesioning_v2/
      lesion_operator/
        seeds.py          SHA-256 domain-separated seed derivation
        sites.py          frozen L1/L2/L3 bindings + NEVER_LESIONED list
        masks.py          logical-edge masks, exact count, nested prefix
        noise.py          symmetric-uniform per-item activation base
        context.py        reversible composite lesion context
        sd_procedure.py   FROZEN intact-SD recipe (hash below)
        battery.py        frozen real-word battery, primary vs diagnostic
        guard.py          hard scientific-execution guard
      scripts/
        verify_states.py      GATE 1 - actual P1-P4 state_dict verification
        verify_gru_layout.py  GATE 2 - torch GRU layout at runtime
        extract_intact_sd.py  GATE 3 - intact-only SD extraction
        build_run_matrix.py   run-matrix enumeration
        run_lesion_v2.py      guarded scientific runner (--dry-run)
      contract/
        LESIONING_V2_RUN_MATRIX.json
        LESIONING_V2_SD_PROCEDURE.json
        LESIONING_V2_OUTPUT_SCHEMA.json
        LESIONING_V2_GRU_LAYOUT.json

The package is `lesion_operator`, deliberately NOT `operator`: a package of
that name shadows the Python standard library `operator` module for anything
run from this directory, which broke tooling during implementation. Renamed
rather than left as a trap.

## 2. Frozen operator bindings

| site | connectivity tensor | shape | logical edges | activation target |
|---|---|---|---|---|
| L1 | `wm.encoder.weight_ih_l0` | (384, 64) | 8,192 | final `wm.encoder` h_n |
| L2 | `ltm.encoder.weight_ih_l0` | (1536, 64) | 32,768 | final `ltm.encoder` h_n, BEFORE `to_semantic` |
| L3 | `ltm.sem_to_h0.weight` | (512, 300) | 153,600 | `h0 = tanh(sem_to_h0(s_hat))`, AFTER tanh |

Never lesioned, enforced by `NEVER_LESIONED` and by a refusal at context entry:
every `weight_hh_l0`, every bias (including `ltm.sem_to_h0.bias`), the shared
`phon_embed`, `to_semantic`, the LTM decoder GRU, and `motor.proj`.

## 3. Masking

    MASK_GRANULARITY = LOGICAL_SOURCE_TARGET_EDGE
    MASK_SAMPLING    = EXACT_COUNT_UNIFORM_WITHOUT_REPLACEMENT
    MASK_NESTING     = NESTED_PREFIX
    BIAS_LESIONABLE  = NO
    CONNECTIVITY_P_MAX = 0.30
    s_k = k/15,  p_k = 0.30 * s_k,  n_k = round(p_k * N_j),  k = 1..15

`physical_mask` depends ONLY on the shape invariant (3H, D) and tiles the same
logical mask onto rows [0:H], [H:2H], [2H:3H]. It never references gate names
or their order; `verify_gru_layout.py` proves the result is invariant under
permutation of the three blocks.

One permutation per `(state_sha256, site, realization)`. Severity, task,
decoding convention and item are ABSENT from that identity, which is what makes
severities nested and masks shared across paired conditions.

## 4. Activation noise

    ACTIVATION_DISTRIBUTION = SYMMETRIC_UNIFORM
    u ~ Uniform(-1, +1), drawn once per (state, site, realization, item)
    eta_k = s_k * SD_site * u

Maximum HALF-WIDTH at k=15 is exactly one intact site SD. This is deliberately
not called "noise SD = 1 SD": the distribution SD at k=15 is
`SD_site / sqrt(3)` ~= 0.577 * SD_site, and `noise.distribution_sd()` returns
that explicitly.

The same base `u` is reused across all 15 severities and across every directly
compared task/decoder condition, and is held fixed for the entire
autoregressive trajectory of an item. No redraw per AR step.

## 5. Composite operator

`context.connectivity_lesion` implements contract section 8: verify pristine
digest, refuse any untouchable target, clone originals, apply the mask, yield,
then write the ORIGINAL tensors back and re-verify the full digest. Restoration
is by stored clone, never by dividing out the mask — a zeroed weight cannot be
recovered arithmetically, and a test asserts the exact value returns.

Nesting is a property of deterministic masks, not of cumulative mutation: a
test runs k=1..15 in sequence and asserts the live model returns to its exact
pristine digest after every cell.

## 6. Intact SD procedure

    procedure_hash = 122ae40c600e639f875bdcb18e588bd32242981af0fa42c0e975a296bc851942

Frozen BEFORE any measurement, inherited verbatim from the audited historical
recipe: pooled flattened intact activations, all axes pooled, ddof=1, one scalar
per (state, site), population `deterministic_sample(range(N), 2048, seed=7)`
sorted, batch 256, eval mode, CPU float32, teacher-forced intact forward.

**L3 is declared a NEW site**: the historical calibration covered the encoder
states only. The same statistical recipe is reused unchanged, and
`site_historically_calibrated["L3"] = false` is recorded in every L3 constant.

`extract_intact_sd.py` refuses to overwrite an existing constants file and
refuses to run if the procedure hash differs from the expected one.

## 7. Battery

PRIMARY (FULL/native — the core dual-route lesion claim concerns what damage
does to the intact system):
`rep_canonical_full`, `rep_freear_full`, `naming_exact`, `c_top1`.

DIAGNOSTIC (route-isolated decompositions, recorded at full item resolution):
`rep_canonical_wm`, `rep_freear_wm`, `rep_canonical_ltm`, `rep_freear_ltm`.

Every endpoint carries population, numerator, denominator, exactness definition,
decoding convention and undefined handling. Evaluators are bound to the
validated V7 implementations; none is reimplemented.

Excluded by the accepted design: pseudoword lesion battery, Ueno-adapted naming
as a primary endpoint, HF/LF split, seven-panel replication.

## 8. Run matrix

    matrix_sha256 = b4d3c98816b46f59afb3e32f081fa073dd5a48cd27f2279fdaccb1f47ca6887c
    cells 1,812 = 1,800 LESION + 12 INTACT_CONTROL
    lesion cells = (12+12+12+4) realizations x 3 sites x 15 severities
    intact cells = 4 states x 3 sites, one per (state, site), k=0

Only POST_REPAIR states. No SOURCE state appears. k=0 is an intact control and
is NOT counted among the 15 severities.

State identity is `composite_state_sha256`, derived from the frozen SOURCE
checkpoint SHA and the frozen `head_first_c0` SHA bound together, so a mask can
never migrate between a SOURCE state and its repaired counterpart. `head_final`
is never referenced; the runner's preflight rejects any executable reference to
it by AST scan.

## 9. Execution guard

`run_lesion_v2.py` refuses every nonzero lesion cell unless
`L3_LESION_V2_AUTHORIZATION` points at an artifact containing the exact token
`CENTRAL_GO_FOR_SCIENTIFIC_EXECUTION_LESIONING_V2`, `go_for_scientific_execution:
true`, AND the matching `run_matrix_sha256`. A stale authorization therefore
cannot license a changed matrix. There is no bypass flag. `--dry-run` performs
full provenance validation with zero model forwards and always remains
available.
