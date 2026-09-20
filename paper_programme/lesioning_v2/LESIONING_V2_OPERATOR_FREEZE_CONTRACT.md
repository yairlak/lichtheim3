# LESIONING V2 — OPERATOR FREEZE CONTRACT

    STATUS      FROZEN FOR CENTRAL REVIEW — NOT IMPLEMENTED, NOT EXECUTED
    BASE        db06aad0aabc701b94eb75c87661c7b5de29c02b
    BRANCH      paper-programme/lesioning-v2-operator-freeze
    EVIDENCE    LESIONING_V2_CONNECTIVITY_MAPPING_AUDIT.md
    DATE        2026-09-21

This contract freezes the OPERATOR. It authorises nothing: implementation and
scientific execution both remain NO.

## 1. Sites and connectivity tensors

    L1  DORSAL / WM ENCODER
        L1_AFFERENT_CONNECTIVITY = wm.encoder.weight_ih_l0        (384, 64)
        logical edges 128 x 64 = 8,192      scalar coefficients 24,576

    L2  VENTRAL / LTM ENCODER
        L2_AFFERENT_CONNECTIVITY = ltm.encoder.weight_ih_l0       (1536, 64)
        logical edges 512 x 64 = 32,768     scalar coefficients 98,304

    L3  VENTRAL PRODUCTION
        L3_AFFERENT_CONNECTIVITY = ltm.sem_to_h0.weight           (512, 300)
        logical edges 512 x 300 = 153,600   scalar coefficients 153,600

Exactly one tensor per site. No factorial. `weight_hh_l0` is NOT lesioned at any
site: it is the site's own internal recurrent dynamics, which the activation
term already perturbs. `phon_embed` is NOT lesionable — it is shared by both
routes (`dual_route.py:60,63-65`) and would destroy route specificity.
`to_semantic`, `dec_to_premotor` and `motor.proj` are efferent, not afferent.
`ltm.decoder.weight_ih_l0` carries autoregressive phonological feedback, not
semantic afference, and is excluded from L3.

Widths are verified, not assumed: `fresh_ceiling_v7.slurm:117` (`WM=128;
ENC=512; DEC=512`) and `fresh_ceiling_v7.py:60,72,279`, which asserts
`ck["widths"] == {128, 512, 512}` for every V7 checkpoint.

## 2. Mask granularity

    MASK_GRANULARITY = LOGICAL_SOURCE_TARGET_EDGE
    BIAS_LESIONABLE  = NO

For a GRU `weight_ih_l0` of shape (3H, D): the logical mask has shape (H, D);
a masked edge is zeroed in ALL THREE gate blocks (r, z, n) at rows
`h`, `H+h`, `2H+h`. Gate order and layout per the documented torch API.

For `sem_to_h0` (`nn.Linear`): one logical edge = one scalar coefficient; the
mask has the tensor's own shape (512, 300).

Biases are not connectivity and are never masked.

## 3. Severity

    N_SEVERITY_LEVELS = 15
    s_k = k / 15                       for k = 1..15
    p_k = CONNECTIVITY_P_MAX * s_k
    A_k = s_k * SD_site

    CONNECTIVITY_P_MAX = 0.30
    ACTIVATION_MAX     = 1.0_SD

Removed logical-edge count at severity k:

    n_k = round(p_k * N_logical_edges)

## 4. Mask sampling and nesting

    MASK_SAMPLING = EXACT_COUNT_UNIFORM_WITHOUT_REPLACEMENT
    MASK_NESTING  = NESTED_PREFIX

One deterministic permutation of all N logical edges is drawn per

    (state identity, lesion site, lesion realization)

Severity k masks the FIRST n_k entries of that permutation. Hence severity k+1
contains every edge removed at k plus additional edges. Severity does NOT enter
the permutation's identity.

Nesting is a prospective Lichtheim3 dose-response choice. Ueno is silent:
`UENO_MASK_NESTING = NOT_ESTABLISHED_FROM_SOURCE`. Historical D-05 was nested as
an emergent property of its implementation (audit §5), which is precedent, not
authority.

### Mask scope and independence

The mask is FIXED for the entire behavioural battery of one lesion realization,
shared across all tasks, all items, and canonical/free-AR or any other directly
compared decoding convention.

    different states (P1/P2/P3/P4)  MUST NOT share a mask
    different sites (L1/L2/L3)      MUST NOT share a mask
    different realizations          MUST NOT share a mask

## 5. RNG hierarchy

Domain-separated, deterministic, and never Python `hash()`.

    MASK payload:
      "L3_LESION_V2_V1|MASK|<state_sha256>|<site>|<realization_index>"

    NOISE payload:
      "L3_LESION_V2_V1|NOISE|<state_sha256>|<site>|<realization_index>|<item_id>"

Derivation, specified exactly:

    digest = sha256(payload.encode("utf-8")).digest()      # 32 raw bytes
    seed   = int.from_bytes(digest[:8], byteorder="big", signed=False)
    seed  &= (2**63 - 1)                                   # non-negative int64

`<state_sha256>` is the SOURCE checkpoint SHA256 of that state, so P1/P2/P3/P4
are independent by construction. `<site>` is the literal tensor path of §1.

    severity                      MUST NOT enter the RNG identity
    task                          MUST NOT enter the RNG identity
    decoding convention           MUST NOT enter the RNG identity
    item_id                       enters ONLY the noise stream

This guarantees pairing across severity, tasks and decoding conventions.

## 6. Activation noise

    ACTIVATION_DISTRIBUTION = UNIFORM_MINUS1_PLUS1
    eta_k = s_k * SD_site * epsilon
    epsilon(state, site, realization, item) ~ Uniform(-1, +1)

One epsilon per item. The SAME epsilon direction at all 15 severity levels
(only the scalar s_k changes), the same epsilon across all directly compared
tasks and decoding conventions, and HELD FIXED throughout the entire
autoregressive decoding trajectory for that item. This preserves the pairing
lesson from GATE x LESION.

SD_site semantics, inherited verbatim from the audited historical recipe
(`gate_x_lesion/sd.py`):

    SD_DEFINITION    sample std of pooled flattened intact site activations
    SD_AXES          all activation axes pooled; no per-unit SD retained
    SD_DDOF          1 (PyTorch unbiased)
    SD_POPULATION    deterministic_sample(range(N), 2048, seed=7), sorted;
                     batch 256; teacher-forced intact forward
    SD_TENSOR_SHAPE  one scalar per (state, site); isotropic

SD is measured on each state's OWN intact model and frozen BEFORE any lesioned
output exists. It is NOT measured in this pass.

Uniform(-1,+1) is an **[INFERENCE]** from Ueno's word "range", never a claim
about Ueno, and is inherited from the audited Lichtheim3 lineage.

## 7. Activation target per site

    L1   final wm.encoder hidden state h_n (dorsal encoded state)
         wm_route.py:89   `_, h = self.encoder(packed)`
         perturb h BEFORE the dorsal decoder consumes it (:105)

    L2   final ltm.encoder hidden state h_n (ventral encoded state),
         BEFORE the downstream semantic projection
         ltm_route.py:139 `_, h = self.encoder(packed)`; pooled feeds
         to_semantic at :146

    L3   decoder initial state h0 = tanh(sem_to_h0(s_hat))
         ltm_route.py:153, AFTER the connectivity-masked projection and tanh,
         BEFORE decoder recurrent generation at :155

All three boundaries exist as explicit intermediate values in the live code.

## 8. Composite operator order

    COMPOSITE_ORDER = MASK_THEN_FORWARD_THEN_ACTIVATION_NOISE

    1. restore pristine model parameters
    2. apply the realization's fixed connectivity mask for this site
    3. run upstream computation THROUGH the masked connectivity
    4. obtain the target site activation/state (§7)
    5. add the item-specific activation perturbation eta_k
    6. run all downstream computation intact
    7. restore and verify pristine parameters after the evaluation context

Connectivity damage and activation damage therefore act on the SAME functional
site: disconnection changes the input reaching the site, activation noise
perturbs the resulting site representation. For L3:

    s_hat -> masked sem_to_h0 -> tanh -> h0 -> + eta_k -> decoder

This is architecturally coherent: masking precedes tanh, so the perturbation is
applied to the state the damaged projection actually produced.

    ACTIVATION_TIMING = POST_SITE_PRE_DOWNSTREAM, once per item, held fixed
                        for the whole decoding trajectory

## 9. Design scope

    PRIMARY_OPERATOR_FAMILY = ACTIVATION_NOISE_PLUS_AFFERENT_CONNECTIVITY_REMOVAL
    CORE_3_SITE_PROGRAMME   = FROZEN  (L1, L2, L3)
    LESION_REALIZATIONS     = P1:12 / P2:12 / P3:12 / P4:4
    REAL_WORD_PRIMARY_BATTERY        = YES
    PSEUDOWORD_PRIMARY_LESION_BATTERY = NO
    UENO_ADAPTED_NAMING_PRIMARY      = NO
    HF_LF_PRIMARY_ENDPOINT           = NO
    FULL_SEVEN_PANEL_UENO_REPLICATION = NO

## 10. Declared limitations

1. p_max=0.30 rests on D-05, which used SCALAR granularity; under logical-edge
   masking the same p zeroes the same number of coefficients but arranges them
   as complete edges. The evidence bounds the dose, it does not predict the
   response. (Audit §7.)
2. There is no site-specific dose evidence for L2, and none for L3. p_max is
   transferred by design uniformity and must be reported as such.
3. D-05 was run on a non-V7 checkpoint (`chigh_15e5_h512_s22_u3000`).
4. An L3 lesion acts on naming and repetition through the same interface,
   because both enter production at `h0 = tanh(sem_to_h0(s_hat))`.
5. Nesting and the noise distribution are Lichtheim3 adaptations, not Ueno.
6. P1-P4 state_dict shapes were established from frozen code plus the driver's
   own per-checkpoint width assertion and a same-architecture local checkpoint;
   they were not read off the P1-P4 files themselves in this pass. The
   implementation preflight must assert them directly on the cluster.
