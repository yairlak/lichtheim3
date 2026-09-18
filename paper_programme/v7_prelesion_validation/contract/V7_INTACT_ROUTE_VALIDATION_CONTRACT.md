# V7 INTACT ROUTE + PSEUDOWORD VALIDATION CONTRACT

    FROZEN BY   CENTRAL STEERING — FINAL ARBITRATION CLOSED (2026-09-18)
    BASE COMMIT 46e638628e2c610c640189523c7918ade60209ad
    BRANCH      feat/v7-prelesion-validation
    TORCH (V7)  2.6.0
    STATUS      APPROVED_FOR_FROZEN_INTACT_VALIDATION

This contract is frozen BEFORE any scientific output is inspected. After the
design commit: no metric definition change, no population change, no ordering
rule change, no classification rule change, and no rerun because an outcome is
inconvenient.

## 1. State convention (CENTRAL D1)

    POST_REPAIR_SCIENTIFIC_WITNESS = head_first_c0.pt   for P1, P2, P3, P4

    HEAD_FIRST_C0_ROLE = CANONICAL_POST_REPAIR_SCIENTIFIC_WITNESS
    HEAD_FINAL_ROLE    = LATER_DERIVED_ARM_A_CONVERGENCE_ARTIFACT
    HEAD_FINAL_BEHAVIOR = NOT_SCIENTIFICALLY_EVALUATED

    SOURCE      = the frozen FIRST-HIT checkpoint
    POST_REPAIR = SOURCE reconstructed IN MEMORY with the head_first_c0
                  deployed state ({"2.weight","2.bias"}) installed

No monolithic scientific checkpoint is ever materialised. No `head_final` state
is evaluated in this workstream.

Justification of record (Jean-Zay trace audit + tensor audit, both authoritative):

    slot  first_c0_iter  steps  first_c0 == final ?
    P1          30         30   YES  (latch coincides with convergence)
    P2          34         35   NO
    P3          67         68   NO
    P4          29         30   NO

This matches the frozen Arm-A mechanism exactly: `first_c0_theta` is latched once
at the first official C=0 (gradient_training_probe.py:229-232) and optimisation
then continues to margin convergence. Full per-state identities, including both
head file SHAs and the deployed-state digests, are in `state_manifest.tsv`.

The frozen official V7 outcomes — P1/P2/P3 `0/0/0/0`, P4 `1/1/0/0` — refer to
`head_first_c0.pt`.

## 2. Populations

### Real-word (unchanged)

    R = 29,571   repetition / free-AR population
    N = 29,571   naming population
    C = 27,981   canonical phonological targets, retrieved against the full
                 29,571 bank

    lexicon data/lexicon_en_glove_covered.tsv
    sha256  ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66
            (== EXPECTED["lexicon_file_sha256"] in the frozen V7 driver)
    29,571 entries / 29,571 unique orthographic / 27,981 unique phonological

### Pseudoword PRIMARY (CENTRAL D2)

    V7_COMMON_UNSEEN   N = 378
      = frozen 391 NOVEL_PSEUDOWORD
        MINUS the UNION of items present in ANY of the P1/P2/P3/P4
              4,000-item dorsal WM training pools (13 items)

    Used as PRIMARY for P1, P2, P3 and DESCRIPTIVELY for P4.
    Identical 378 items for every slot.

### Pseudoword SECONDARY SENSITIVITY

    V7_SEED_UNSEEN     P1=384  P2=390  P3=389  P4=387
      = 391 MINUS that seed's own overlaps.
    SECONDARY ONLY. May not replace the common-378 primary.

### Exposed stratum

    V7_DORSAL_POOL_EXPOSED  N = 13
    NOT unseen pseudowords. DESCRIPTIVE ONLY.

### Descriptive lexicality/exposure comparison

    TRAINED_REAL_EXACT  N = 671.  Descriptive only.
    Permitted terms: TRAINED-REAL-EXACT vs NOVEL-PSEUDOWORD;
                     LEXICALITY/EXPOSURE CONTRAST.
    NEVER a pure causal lexicality contrast.

    Provenance: canonical_behavioral_item_table.tsv
    sha256 8988aff6fac55ca36cb43ce758f5684f30ae10a6303bdbd7b0b9f462433d5a67
    14,400 rows = 1,200 items x 3 routes x 4 historical seeds; one unique
    stimulus row per item taken by first occurrence in file order (no
    outcome-based selection); item-level exposure labels unique for all 1,200.

    Dorsal pool generator (for the exclusion above):
      train_joint_scratch.py::build_pool_entries, seed = RAW train seed,
      n = 4000, min_len = 2, max_len = 9; equality by canonical phoneme-ID
      sequence (pool entries carry word="" and have no orthographic form).

## 3. Decode conventions

### Genuine free-AR (PRIMARY for pseudowords) — CENTRAL D4

    FREE_AR_CONTRACT = PASS
    implementation   train_joint_scratch.py::free_ar_repetition (unchanged)
    start token      vocab.bos_id
    termination      first EOS, wherever it falls; non-termination and
                     over-generation count as errors
    horizon          FREE_AR_MAX_STEPS = 12 (single GLOBAL constant)
    target length    NEVER used to terminate decoding. Gold length may enter
                     ONLY final scoring / diagnostics.
    routes           full -> "logits", wm -> "wm_logits", ltm -> "ltm_logits"
    scoring          exact sequence equality after trimming at first EOS

    Adequacy: primary stimuli are 3-9 phonemes, so 9 + EOS = 10 <= 12. No item
    is horizon-truncated by construction.

### Canonical (forced-length) repetition

    repetition_snapshot, unchanged. Truncates to gold length + 1. Legitimate
    for repetition; reported alongside free-AR, never substituted for it.

### Secondary historical decoder

    SECONDARY_HISTORICAL_TARGET_LENGTH_COMPATIBLE
    NOT_GENUINE_FREE_AR — must never be conflated with the primary.

## 4. Endpoints — all 8 states

    P1_SOURCE  P1_POST_REPAIR  P2_SOURCE  P2_POST_REPAIR
    P3_SOURCE  P3_POST_REPAIR  P4_SOURCE  P4_POST_REPAIR

Global real-word: R canonical FULL, R genuine free-AR FULL, Naming free greedy,
strict C top-1 — each exact count + error count.

Routes: WM-only canonical, WM-only free-AR, LTM-only canonical, LTM-only free-AR
— exact + errors.

Gating (real-word R): item-level c_LTM where defined, item-level g, and summary
n / mean / sample SD / min / p01 / p05 / p25 / p50 / p75 / p95 / p99 / max.

Native-vs-fixed05 intact: DESCRIPTIVE ONLY, included only if the existing frozen
gating evaluator is reused unchanged; otherwise omitted with reason recorded.
This does not reopen GATE x LESION.

Pseudoword (378 primary; routes WM_ONLY, LTM_ONLY, FULL_NATIVE): target phoneme
IDs/string, predicted IDs/string, exact, raw Levenshtein ED, normalized ED,
target length, predicted length, EOS position / no-EOS, first divergence
position where already supported, and c_LTM / g descriptively for FULL native.

## 5. SOURCE -> POST_REPAIR paired analysis

Item-level paired changes in: FULL canonical, FULL free-AR, WM canonical, WM
free-AR, LTM canonical, LTM free-AR, Naming, C top-1, c_LTM, g.

Discrete endpoints report correct->correct, correct->wrong, wrong->correct,
wrong->wrong, with exact item IDs for every changed item. Continuous endpoints
report item-level delta plus distribution summary.

Arm A is NOT assumed to affect only C. Results are separated into:

    WHAT_ARM_A_REPAIRS
    WHAT_ARM_A_PRESERVES
    WHAT_ARM_A_CHANGES_INCIDENTALLY

## 6. Frozen pseudoword ordering rule

Computed on the 378 primary, per state, in float64:

    delta_acc = exact_acc_WM  - exact_acc_LTM
    delta_ned = mean_NED_LTM  - mean_NED_WM     (positive favours WM)

    WM_DOMINANT   delta_acc >= 0 AND delta_ned >= 0 AND at least one > 0
    TIE           delta_acc == 0 AND delta_ned == 0
    MIXED         the two metrics favour opposite routes
    LTM_DOMINANT  delta_acc <= 0 AND delta_ned <= 0 AND at least one < 0

No tolerance is introduced post hoc. Exact accuracy is rational from 378 items.

## 7. Arm-A preservation rule (P1/P2/P3)

    PRESERVED                        SOURCE WM_DOMINANT and POST_REPAIR WM_DOMINANT
    PRESERVED_FROM_NONDOMINANT_SOURCE SOURCE not WM_DOMINANT, POST_REPAIR WM_DOMINANT
    NOT_PRESERVED                    SOURCE WM_DOMINANT, POST_REPAIR not
    NO_EXPECTED_PATTERN_AT_SOURCE    neither WM_DOMINANT

    ARM_A_PSEUDOWORD_PRESERVATION = YES
    only if all of P1/P2/P3 are PRESERVED or PRESERVED_FROM_NONDOMINANT_SOURCE
    and none is NOT_PRESERVED. P4 does not define this family conclusion.

## 8. Final classification rules

    V7_LESION_READY
      mechanical/provenance validity PASS
      AND P1/P2/P3 POST global batteries match the frozen V7 outcomes
      AND P1/P2/P3 POST pseudoword ordering = WM_DOMINANT
      AND Arm-A does not reverse WM dominance
      AND no basic route distinction is mechanically uninterpretable

    V7_LESION_READY_WITH_QUALIFICATION
      validity PASS and models remain usable for route lesioning, BUT one
      intact route/pseudoword property is TIE/MIXED or carries a nonfatal
      pathology flag such that the manuscript claim must be narrowed

    V7_NOT_LESION_READY
      provenance/evaluator validity FAILS
      OR one or more P1/P2/P3 POST models show a basic route distinction
         incompatible with the planned interpretation (e.g. robust
         LTM_DOMINANT ordering under the frozen primary metrics)
      OR required route outputs cannot be meaningfully evaluated

A negative status does NOT authorize model search.

### Mechanical validity requirements

All provenance gates pass; state reconstruction succeeds; repeated deterministic
smoke hashes match; no source/head/checkpoint mutated; no NaN/Inf in required
metrics; all required rows present; P1/P2/P3 POST global batteries exactly
reproduce the frozen V7 official outcomes; P4 reproduces its frozen `1/1/0/0`
counterexample.

### Route pathology flags (reported, never silently converted to a threshold)

Route evaluator crashes/undefined; all 378 predictions identical; 100% no-EOS
under the valid horizon; isolated real-word route exact accuracy exactly zero.
No performance floor is invented.

## 9. Row identity

Every result row carries: model slot; training seed; SOURCE/POST_REPAIR; source
checkpoint SHA; repair-head file SHA; deployed-head-state digest; evaluator
commit/blob; population name + source artifact SHA; route; decode convention;
item ID.

## 10. Execution discipline

Execute the eight states exactly once, as one orchestrated run whose state list
is frozen before launch. Real words use the full frozen populations; pseudowords
use exactly the 378 primary (per-seed sets secondary; 13 exposed descriptive;
671 descriptive). No lesions. No inspection of intermediate results to alter
later state evaluation. A failed infrastructure job may be resumed only if no
scientific output was partially accepted and resume semantics are provably
identical; all incidents recorded.

## 11. Frozen manifests

    state_manifest.tsv                              8 states
    stimulus_manifest_common_unseen_378.tsv         PRIMARY
    stimulus_manifest_seed_unseen.tsv               SECONDARY (1,550 rows)
    stimulus_manifest_dorsal_pool_exposed_13.tsv    DESCRIPTIVE
    stimulus_manifest_trained_real_exact_671.tsv    DESCRIPTIVE
    population_manifests.json
    validation_contract.json
