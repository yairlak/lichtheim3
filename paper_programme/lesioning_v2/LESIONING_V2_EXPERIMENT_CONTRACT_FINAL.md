# LESIONING V2 — FINAL EXPERIMENT CONTRACT

    STATUS  FROZEN FOR EXECUTION REVIEW — NOT EXECUTED
    BASE    5d1849a0c1d01065ff926d324703e8b767c05223
    DATE    2026-09-21

This is the executable specification. It ADDS to, and never replaces, the
accepted operator freeze (`LESIONING_V2_OPERATOR_FREEZE_CONTRACT.md`) and the
connectivity audit, both of which remain byte-unchanged from the audit commit.

## 1. Operator (inherited, unchanged)

    L1 = wm.encoder.weight_ih_l0    (384,64)     8,192 logical edges
    L2 = ltm.encoder.weight_ih_l0   (1536,64)    32,768 logical edges
    L3 = ltm.sem_to_h0.weight       (512,300)    153,600 logical edges

    MASK_GRANULARITY   LOGICAL_SOURCE_TARGET_EDGE
    MASK_SAMPLING      EXACT_COUNT_UNIFORM_WITHOUT_REPLACEMENT
    MASK_NESTING       NESTED_PREFIX
    BIAS_LESIONABLE    NO
    CONNECTIVITY_P_MAX 0.30
    s_k = k/15,  p_k = 0.30*s_k,  n_k = round(p_k*N_j),  k = 1..15

## 2. Activation

    u ~ Uniform(-1,+1) per (state, site, realization, item)
    eta_k = s_k * SD_site * u
    max HALF-WIDTH at k=15 = 1 intact site SD
    distribution SD at k=15 = SD_site / sqrt(3)

One draw per item, shared across all severities and all compared
task/decoder conditions, fixed for the whole AR trajectory.

## 3. RNG

    MASK  "L3_LESION_V2_V1|MASK|<state_sha256>|<site>|<realization_index>"
    NOISE "L3_LESION_V2_V1|NOISE|<state_sha256>|<site>|<realization_index>|<item_id>"

    utf-8 -> sha256 -> digest[:8] BIG-endian unsigned -> & (2^63-1)
    -> torch.Generator.manual_seed

Severity, task and decoding convention never enter either identity. Python
`hash()` is never used.

`<state_sha256>` is the composite identity
`sha256("L3_LESION_V2_V1|STATE|<state_id>|<source_sha>|<head_first_c0_sha>")`,
binding both halves so a mask cannot migrate between SOURCE and POST_REPAIR.

## 4. States and matrix

    PRIMARY STATES  P1_POST_REPAIR P2_POST_REPAIR P3_POST_REPAIR P4_POST_REPAIR
    SOURCE states   NOT part of V2 lesioning
    realizations    P1:12 P2:12 P3:12 P4:4
    severities      15 NONZERO; k=0 is an intact control, not a severity
    matrix          1,812 cells = 1,800 lesion + 12 intact
    matrix_sha256   b4d3c98816b46f59afb3e32f081fa073dd5a48cd27f2279fdaccb1f47ca6887c

## 5. Battery

PRIMARY (FULL/native): `rep_canonical_full`, `rep_freear_full`,
`naming_exact`, `c_top1`.

DIAGNOSTIC (route-isolated): `rep_canonical_wm`, `rep_freear_wm`,
`rep_canonical_ltm`, `rep_freear_ltm`.

Full definitions — population, numerator, denominator, exactness, decoding
convention, undefined handling — in `contract/LESIONING_V2_OUTPUT_SCHEMA.json`.
Evaluators are bound to the validated V7 implementations and are not
reimplemented.

Excluded: pseudoword lesion battery, Ueno-adapted naming as primary, HF/LF
split, seven-panel replication.

No metric may be added once lesion results exist, except derived reporting
explicitly labelled post hoc and non-scientific.

## 6. Intact baseline

k=0 cells carry zero connectivity removal and zero activation noise, and must
reproduce the authoritative intact POST_REPAIR behaviour exactly. They are
intact verification, not lesion calibration, and are excluded from the 15
severities.

## 7. Output

Item-level and summary schemas are frozen in
`contract/LESIONING_V2_OUTPUT_SCHEMA.json`. Aggregation across realizations
never deletes per-realization rows. No adaptive early stopping on any lesion
outcome.

## 8. Execution authorization

    GO_FOR_SCIENTIFIC_EXECUTION = NO

Nonzero lesion execution requires an authorization artifact carrying the token
`CENTRAL_GO_FOR_SCIENTIFIC_EXECUTION_LESIONING_V2`,
`go_for_scientific_execution: true`, and this contract's `matrix_sha256`.

## 9. Prerequisites before any execution

    GATE 1  verify_states.py      actual P1-P4 state_dict names/shapes
    GATE 2  verify_gru_layout.py  under torch 2.6.0
    GATE 3  extract_intact_sd.py  intact-only SD constants, procedure hash
            122ae40c600e639f875bdcb18e588bd32242981af0fa42c0e975a296bc851942

All three must pass on the cluster. Any mismatch is a hard stop with no
automatic repair.

## 10. Carry-forward limitations

p_max rests on D-05 under scalar granularity (dose bound, not response
prediction); no site-specific dose evidence for L2 or L3; D-05 used a non-V7
checkpoint; an L3 lesion reaches naming and repetition through one interface;
nesting and the uniform noise distribution are Lichtheim3 adaptations, not
Ueno's.
