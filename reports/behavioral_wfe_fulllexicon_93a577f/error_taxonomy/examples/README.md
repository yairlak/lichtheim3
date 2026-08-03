# Seed-22 illustrative items

**These are deterministic illustrations, not a representative sample, and they carry no inference.** Seed 22 was declared in `../error_taxonomy_analysis_spec.md` before any item was inspected; nothing here was chosen after seeing the outcome, and no claim in `../error_taxonomy_results.md` rests on these rows.

Scope: clean-set novel pseudowords only (`NOVEL_PSEUDOWORD`), up to 20 rows per route.

## `seed22_illustrative_pseudoword_errors.tsv`

Erroneous items ordered by raw edit distance descending, then `eos_shortfall` descending with missing values last, then `item_id` ascending.

## `seed22_illustrative_premature_eos.tsv`

Items with an observed premature EOS, ordered by `eos_shortfall` descending, then raw edit distance descending, then `item_id` ascending.

## Reading the columns

`eos_position` is a **0-based index into the item's readout window** and equals the number of phonemes emitted before EOS; `expected_eos_position` is the target length L; `eos_shortfall = expected − observed`. The window holds only indices 0…L−1, so every observed EOS is premature and an on-time EOS is not representable. See `../eos_convention.md`.

A premature EOS is **not** the same event as a deletion: the two columns are independent measurements and rows may show either, both or neither.
