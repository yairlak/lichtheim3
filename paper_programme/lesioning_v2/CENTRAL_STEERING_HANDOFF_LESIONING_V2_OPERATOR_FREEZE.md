# CENTRAL STEERING HANDOFF — LESIONING V2 OPERATOR FREEZE

    PASS     STATIC / READ-ONLY SCIENTIFIC AUDIT
    BASE     db06aad0aabc701b94eb75c87661c7b5de29c02b
    BRANCH   paper-programme/lesioning-v2-operator-freeze
    WORKTREE ../wt-lesioning-v2-operator-freeze
    DATE     2026-09-21

No lesion was run. No forward pass was performed through P1/P2/P3/P4. No
parameter, checkpoint or scientific result was modified.

## Frozen operator

    L1_AFFERENT_CONNECTIVITY=
    wm.encoder.weight_ih_l0            (384, 64)    8,192 logical edges

    L2_AFFERENT_CONNECTIVITY=
    ltm.encoder.weight_ih_l0           (1536, 64)   32,768 logical edges

    L3_AFFERENT_CONNECTIVITY=
    ltm.sem_to_h0.weight               (512, 300)   153,600 logical edges

    MASK_GRANULARITY=
    LOGICAL_SOURCE_TARGET_EDGE
    (GRU: one (hidden, input) edge zeroed in all three r/z/n gate blocks;
     sem_to_h0 Linear: one edge = one scalar coefficient)

    MASK_SAMPLING=
    EXACT_COUNT_UNIFORM_WITHOUT_REPLACEMENT

    MASK_NESTING=
    NESTED_PREFIX
    (Lichtheim3 adaptation; UENO_MASK_NESTING=NOT_ESTABLISHED_FROM_SOURCE)

    BIAS_LESIONABLE=
    NO

    CONNECTIVITY_P_MAX=
    0.30

    ACTIVATION_DISTRIBUTION=
    UNIFORM_MINUS1_PLUS1
    eta_k = s_k * SD_site * epsilon,  epsilon ~ Uniform(-1,+1)

    ACTIVATION_TIMING=
    POST_SITE_PRE_DOWNSTREAM
    one epsilon per item; same epsilon direction at all 15 severities; same
    across compared tasks/decoders; held fixed for the whole AR trajectory

    COMPOSITE_ORDER=
    RESTORE -> MASK -> FORWARD_THROUGH_MASKED_CONNECTIVITY -> SITE_STATE
    -> ADD_ACTIVATION_NOISE -> DOWNSTREAM_INTACT -> RESTORE_AND_VERIFY

    ACTIVATION_MAX=
    1.0_SD

    N_SEVERITY_LEVELS=
    15

    LESION_REALIZATIONS=
    P1:12 / P2:12 / P3:12 / P4:4

    OPERATOR_FREEZE_STATUS=
    READY_FOR_CENTRAL

    GO_FOR_IMPLEMENTATION=
    NO

    GO_FOR_SCIENTIFIC_EXECUTION=
    NO

## Why READY rather than BLOCKED

The five blocking criteria, each answered by evidence:

1. **Live code confirms the L1/L2/L3 mappings.** The production graph separates
   the three sites cleanly; in particular `ltm_route.py:153-155` shows semantic
   afference (`sem_to_h0`) and autoregressive phonological feedback
   (`decoder.weight_ih_l0`) as distinct inputs, so L3 needs no combination.
2. **Exact tensor names/shapes are known.** Widths 128/512/512 are declared in
   the V7 slurm (`:117`) and driver (`:60`) and *asserted per checkpoint* by
   `verify_checkpoint` (`fresh_ceiling_v7.py:279`), which the V7 driver runs on
   every selected source. Shapes were independently enumerated from a
   same-architecture local checkpoint, read-only.
3. **Mask granularity is implementable without ambiguity.** A logical (H, D)
   mask tiles across the three gate blocks; verified on synthetic tensors.
4. **p_max is prospectively defensible** — with one premise of CENTRAL's own
   argument rejected; see below.
5. **RNG, nesting and composite semantics are fully specified** in the contract,
   including exact byte order and truncation.

## Where I disagree with CENTRAL's p_max argument

Points 1–4 of §15 verify exactly against the D-05 artifacts. **Point 5 does
not.** It claims logical-edge masking "changes correlation structure, not
nominal edge fraction". That understates it: under scalar masking at p=0.30
only 34.3% of logical edges survive fully intact and ~63% are partially
degraded, whereas logical-edge masking removes 30% of edges completely and
leaves 70% pristine. These are qualitatively different perturbations, so the
D-05 curve does **not** transfer as a like-for-like prediction of behaviour at
the same p.

What survives is the weaker but sufficient claim: at p=0.30 both schemes zero
the same number of coefficients (7,373 of 24,576 at L1), so the evidence bounds
the **dose** even though it does not predict the **response**. Since all 15
levels are reported, a curve that saturates earlier or later remains fully
interpretable. I therefore freeze 0.30, with the transfer stated as approximate
rather than exact.

Two further limitations to carry forward: there is **no site-specific dose
evidence for L2 and none for L3** — p_max is transferred by design uniformity,
not evidence — and D-05 was run on a non-V7 checkpoint.

## Notable audit findings

  * **The H128 trap is real.** `train_joint_scratch.py:157` sets
    `CANONICAL_HIDDEN = 128` for wm/enc/dec, and the `--enc-hidden`/`--dec-hidden`
    defaults are that constant. V7 overrides both to 512. Reading the historical
    default as the final architecture would have produced wrong L2/L3 shapes.
  * **`phon_embed` is one shared module** (`dual_route.py:60,63-65`), surfacing
    three times in the state_dict. Lesioning it would damage the dorsal and
    ventral routes simultaneously — rejected on route-specificity grounds.
  * **The WM encoder→decoder path has no weight matrix** (`wm_route.py:105`); it
    is a hidden-state handoff and cannot be lesioned as connectivity.
  * **Historical D-05 WAS nested.** `derive_generator` keys on
    `(lesion_seed, target_id, role)` and not on the removal fraction
    (`masks.py:36-40`), so `randperm[:n]` is a prefix. Reproduced synthetically:
    614 ⊂ 1,229 ⊂ 1,843 ⊂ 3,686 ⊂ 7,373. This is precedent only — nesting is
    still an adaptation, not attributed to Ueno.
  * **All twenty D-05 behavioural values CENTRAL quoted verify exactly**, as do
    both matched-absolute-count comparisons (3,686 and 7,373 links).
  * **Ueno special-cased a clamped input layer** by removing *outgoing* links
    (supplement p. 15) — direct evidence that "incoming links" means whatever
    delivers signal into the damaged layer, reasoned functionally. None of
    L1/L2/L3 is a clamped input layer.

## Deliverables

    paper_programme/lesioning_v2/
      LESIONING_V2_CONNECTIVITY_MAPPING_AUDIT.md
      LESIONING_V2_OPERATOR_FREEZE_CONTRACT.md
      CENTRAL_STEERING_HANDOFF_LESIONING_V2_OPERATOR_FREEZE.md
      tests/test_operator_freeze_static.py        22 passed

No prior `LESIONING_V2_EXPERIMENT_CONTRACT` exists anywhere in the repository,
so nothing was amended or overwritten; this contract is the first of its name.

## Required before implementation

  * assert the three tensor names/shapes directly on the P1-P4 state_dicts on
    the cluster (this pass proved them from frozen code, not from those files);
  * re-assert the GRU layout under torch 2.6.0 (verified here on 2.12.1 against
    the documented API);
  * measure and freeze SD_site per state on each intact model, before any
    lesioned output exists.

    STOP. RETURN TO CENTRAL.
