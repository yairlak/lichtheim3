# FINAL populations and homophone policy (Phase FINAL-1A)

Human decision (Yair-approved formulation, 2026-09-01), implemented by
`--subset-mode final_full` in `scripts/naming_comprehension/train_joint_scratch.py`.

## Populations

| task | population | N | definition |
|---|---|---|---|
| repetition (R) | full lexicon | 29,571 | unchanged historical recipe: log-frequency weighted, with replacement |
| naming (N) | full lexicon | 29,571 | flat, without replacement, per-epoch permutation |
| comprehension (C) | canonical phonology | 27,981 | ONE target per exact phonological form (below); flat, without replacement |
| retrieval bank | full lexicon | 29,571 | unchanged; all GloVe rows, L2-normalised |

Population hashes (SHA-256 over the ordered `position/bank_index/word/
phoneme_ids/rank` records, the repository's `subset_definition_hash`
convention) are frozen in the driver:

- canonical C population: `10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50`
- lexicon file: `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66`

## Canonical comprehension target policy

FINAL-0 established that homophones are **bit-identical comprehension
inputs** (stress-free ARPABET token IDs; the encoder sees only phoneme IDs +
mask), capping full-lexicon exact word-ID top-1 at 94.62% (27,981/29,571).

Policy (`train_tasks.canonical_phonology_indices`):

- equivalence classes are keyed by the exact phoneme token-ID sequence — the
  deterministic encoder input (EOS/padding are class-invariant);
- each class contributes its **highest-frequency** member: lowest
  `LexEntry.rank`, the authoritative frequency field (hybrid ranking:
  measured frequency core + deterministic Zipf-continued tail, see
  `data/build_lexicon_en.py`; the GloVe-covered lexicon preserves original
  ranks, which are unique and strictly increasing in file order);
- ties break to the lowest bank index (= file order). On the canonical
  lexicon ranks are unique so the tie-break is provably inert (tested);
- singleton phonologies are unchanged (26,682); the 1,299 homophone groups
  are reduced to one target each: 26,682 + 1,299 = **27,981**.

Excluded homophone lexical IDs (1,590 = 2,889 − 1,299) remain (a) full
retrieval-bank competitors/distractors, and (b) ordinary members of the
repetition and naming populations. They are simply not independent
comprehension targets.

**The primary FINAL comprehension criterion is 100% strict word-ID top-1
retrieval on the canonical 27,981-target population against the full
29,571-row bank.** This is mathematically attainable (tested:
`test_comprehension_targets_map_into_full_bank_and_100pct_is_attainable`).
Top-5/rank/margin remain diagnostics. Multi-positive retrieval,
homophone-aware primary scoring and top-k success criteria are deliberately
NOT implemented.

## Relation to Ueno et al.

Ueno et al. removed homophone items from their training corpus outright.
Ours is a related but **not identical** modern adaptation: we retain one
canonical target per identifiable phonological form, keep every homophone in
the retrieval bank as a competitor, and keep all homophones in the
repetition and naming populations.

## Evaluation cadence in `final_full`

Developmental (cadence) evaluation runs on fixed deterministic samples of
3,288 items per task (seed namespace 3,000,000+, disjoint from all training
streams); full-population evaluation (all 27,981 C targets, all 29,571 N and
R items) runs with `--endpoint-eval` and fills the `full_*` metric columns.
The out-of-subset repetition probe is explicitly disabled in this mode (the
populations cover the whole lexicon; recorded in provenance, reported as
NaN, never fabricated).
