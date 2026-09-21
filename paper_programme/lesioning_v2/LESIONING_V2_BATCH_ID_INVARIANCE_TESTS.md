# LESIONING V2 — BATCH-ID INVARIANCE TEST RECORD

    148 tests passed (128 pre-existing + 20 new). Zero real P1-P4 nonzero forward.

## A-J as required by CENTRAL

| # | requirement | test | result |
|---|---|---|---|
| A | `len(batch_item_ids)` == activation batch dim | `test_A_batch_len_matches_activation_batch_dimension`, `test_A_mismatch_is_refused` | PASS |
| B | exact order preservation | `test_B_C_D_order_preserved_no_offset_no_permutation` | PASS |
| C | no off-by-one | same (each row re-derived singly and compared) | PASS |
| D | no cross-item permutation | same (reversed id list permutes rows correspondingly) | PASS |
| E | first and last population items map correctly | `test_E_first_and_last_population_items_map_correctly` | PASS |
| F | final partial batch maps correctly | `test_F_final_partial_batch_maps_correctly` (29,571 mod 128 != 0) | PASS |
| G | every global item exactly once per full pass | `test_G_every_global_item_appears_exactly_once_per_pass` (64/128/256/512) | PASS |
| H | canonical and free-AR share one global identity | `test_H_canonical_and_free_ar_share_the_same_global_identity` | PASS |
| I | batch-size invariance over 1/17/64/128/257 incl. first, middle, final partial | `test_I_batch_size_invariance_bitwise`, `test_I_first_middle_final_partial_batches_agree` | PASS |
| J | task/readout do not change eta | `test_J_task_and_readout_do_not_change_eta` | PASS |

Bitwise comparison throughout: `torch.equal`, never `allclose`.

## Autoregressive invariant

`test_autoregressive_eta_is_identical_at_every_step` drives 12 hook firings on
one batch (the real `FREE_AR_MAX_STEPS`) and asserts every captured eta is
bitwise identical to the first. `test_different_items_get_independent_deterministic_eta`
asserts distinct items differ pairwise and that repeat calls reproduce exactly.

`test_no_cursor_in_injection_or_driver` additionally forbids, by AST, any
iterator advancement, `itertools.count`, `nonlocal`/`global` rebinding, or
cursor-like name in the injection layer or the driver.

## Recovered batching

`test_batch_sizes_are_recovered_from_executable_code` asserts the sizes come
from the frozen code itself — `evaluate_train_lexicon_ceiling.BATCH_SIZE` and
the three signature defaults — and equal 128 / 256 / 512 / 64.
`test_chunks_are_contiguous_and_order_preserving` pins the partition.

## k=0 and byte identity

`test_k0_path_still_calls_the_unchanged_evaluator`,
`test_frozen_science_files_byte_identical_to_base` (re-hashes 8 files against
`git show BASE:<path>`), `test_matrix_semantic_sha_unchanged`.

## Continuation (§9)

Accepted: valid 12-control namespace. Refused: missing k=0, extra/foreign cell,
any COMPLETE k>0, altered marker, wrong hash, stale manifest from another root,
manifest for another matrix, duplicate identity, default fresh mode against an
existing namespace, output path inside the namespace. COMPLETE cells are
skipped and cannot be overwritten. The continuity decision is proven
performance-blind by AST.
