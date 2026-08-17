"""Focused tests for the Phase 2C deterministic nested subset selector.

Verifies, on a synthetic lexicon only (no checkpoint, no data file):
  - exactly n = 4 * per_band items, with per_band items in each frozen band;
  - every selected item has unique phonology (no homophones can be drawn);
  - bit-identical reproduction across calls and across rebuilt lexicons;
  - the subset definition hash is order-sensitive;
  - the nesting property: a smaller subset is a strict subset of any larger
    one drawn from the same seed;
  - selection is not simply the most frequent head of each band;
  - the predeclared success criteria behave exactly as declared.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lexicon import LexEntry                                       # noqa: E402
from data.phonemes import build_vocab                                   # noqa: E402
from scripts.naming_comprehension.aggregate_cohort import (             # noqa: E402
    FREQ_BANDS, BAND_LABELS)
from scripts.naming_comprehension.train_tasks import (                  # noqa: E402
    C3_SUBSET_SUCCESS, NAMING_SUBSET_SUCCESS, band_of_rank,
    c3_subset_success, homophone_indices, naming_subset_success,
    nested_band_ordering, select_nested_subset, subset_definition_hash,
    subset_records, unique_phonology_indices)

SEM_DIM = 8
N_PER_BAND = 50
N_HOMOPHONE_PAIRS = 3


def _form(k: int) -> list:
    """Distinct 3-phoneme form per integer k (ids 3..41 are real phonemes)."""
    ids = []
    for _ in range(3):
        ids.append(3 + k % 39)
        k //= 39
    return ids


def _synth_entries(n_per_band: int = N_PER_BAND,
                   n_homophone_pairs: int = N_HOMOPHONE_PAIRS):
    """Synthetic lexicon spanning all four frozen bands, plus homophone pairs.

    Words are interleaved across bands so that bank-index order never
    coincides with band order.
    """
    rng = np.random.default_rng(0)
    entries = []
    k = 0
    for j in range(n_per_band):
        for bi, (lo, _hi) in enumerate(FREQ_BANDS):
            entries.append(LexEntry(
                word=f"w{bi}_{j}", phonemes=_form(k),
                semantic=rng.standard_normal(SEM_DIM).astype(np.float32),
                freq=1.0, rank=lo + j * 7 + 1))
            k += 1
    for h in range(n_homophone_pairs):
        shared = _form(50_000 + h)
        for c in range(2):
            entries.append(LexEntry(
                word=f"h{h}_{c}", phonemes=list(shared),
                semantic=rng.standard_normal(SEM_DIM).astype(np.float32),
                freq=1.0, rank=FREQ_BANDS[h % 4][0] + 3))
    return entries


# ------------------------------------------------------------- band logic

def test_band_of_rank_respects_frozen_half_open_bands():
    assert band_of_rank(1, FREQ_BANDS) == 0
    assert band_of_rank(999, FREQ_BANDS) == 0
    assert band_of_rank(1000, FREQ_BANDS) == 1       # half-open: upper exclusive
    assert band_of_rank(4999, FREQ_BANDS) == 1
    assert band_of_rank(5000, FREQ_BANDS) == 2
    assert band_of_rank(15000, FREQ_BANDS) == 3
    assert band_of_rank(0, FREQ_BANDS) is None       # rank 0 is outside


def test_bands_are_the_frozen_phase1a_definition():
    assert FREQ_BANDS == ((1, 1000), (1000, 5000), (5000, 15000), (15000, 10 ** 9))
    assert BAND_LABELS == ("1-1k", "1k-5k", "5k-15k", "15k-end")


# ------------------------------------------------------ size and structure

def test_subset_is_exactly_64_items():
    entries = _synth_entries()
    idx = select_nested_subset(entries, per_band=16, subset_seed=0)
    assert len(idx) == 64
    assert len(set(idx)) == 64


def test_subset_has_exactly_16_items_per_fixed_band():
    entries = _synth_entries()
    idx = select_nested_subset(entries, per_band=16, subset_seed=0)
    counts = {lab: 0 for lab in BAND_LABELS}
    for i in idx:
        counts[BAND_LABELS[band_of_rank(entries[i].rank, FREQ_BANDS)]] += 1
    assert counts == {lab: 16 for lab in BAND_LABELS}


def test_subset_contains_no_homophones():
    entries = _synth_entries()
    idx = select_nested_subset(entries, per_band=16, subset_seed=0)
    homo = set(homophone_indices(entries))
    assert homo, "sanity: the synthetic lexicon must contain homophones"
    assert not (set(idx) & homo)
    assert set(idx) <= set(unique_phonology_indices(entries))
    forms = [tuple(entries[i].phonemes) for i in idx]
    assert len(set(forms)) == len(forms)


def test_selection_pool_excludes_homophones_even_in_their_own_band():
    entries = _synth_entries()
    homo = set(homophone_indices(entries))
    ordering = nested_band_ordering(entries, subset_seed=0)
    for lab in BAND_LABELS:
        assert not (set(ordering[lab]) & homo)


def test_raises_when_a_band_cannot_supply_enough_items():
    entries = _synth_entries(n_per_band=4)
    with pytest.raises(RuntimeError, match="cannot select"):
        select_nested_subset(entries, per_band=16, subset_seed=0)


# ------------------------------------------------------------ determinism

def test_subset_is_deterministic_across_calls_and_rebuilds():
    a = select_nested_subset(_synth_entries(), per_band=16, subset_seed=0)
    b = select_nested_subset(_synth_entries(), per_band=16, subset_seed=0)
    assert a == b                                     # order included
    ordering1 = nested_band_ordering(_synth_entries(), 0)
    ordering2 = nested_band_ordering(_synth_entries(), 0)
    assert ordering1 == ordering2


def test_different_subset_seed_changes_the_selection():
    entries = _synth_entries()
    a = select_nested_subset(entries, per_band=16, subset_seed=0)
    b = select_nested_subset(entries, per_band=16, subset_seed=1)
    assert set(a) != set(b)


def test_selection_is_not_the_frequency_head_of_each_band():
    """A seeded permutation, not the top-16 most frequent words per band."""
    entries = _synth_entries()
    idx = select_nested_subset(entries, per_band=16, subset_seed=0)
    for bi, lab in enumerate(BAND_LABELS):
        in_band = sorted((entries[i].rank, i) for i in unique_phonology_indices(entries)
                         if band_of_rank(entries[i].rank, FREQ_BANDS) == bi)
        head = {i for _, i in in_band[:16]}
        chosen = {i for i in idx
                  if band_of_rank(entries[i].rank, FREQ_BANDS) == bi}
        assert chosen != head, f"band {lab} selection is the frequency head"


# ---------------------------------------------------------------- nesting

def test_smaller_subset_is_a_strict_subset_of_a_larger_one():
    entries = _synth_entries()
    small = select_nested_subset(entries, per_band=16, subset_seed=0)
    large = select_nested_subset(entries, per_band=40, subset_seed=0)
    assert set(small) < set(large)                     # strict subset
    assert len(large) == 160


def test_nesting_holds_across_a_chain_of_sizes():
    entries = _synth_entries()
    sizes = [4, 8, 16, 32, 50]
    sets = [set(select_nested_subset(entries, per_band=k, subset_seed=0))
            for k in sizes]
    for a, b in zip(sets, sets[1:]):
        assert a < b


def test_selection_is_the_prefix_of_a_size_independent_band_ordering():
    """The mechanism that makes nesting structural rather than incidental."""
    entries = _synth_entries()
    ordering = nested_band_ordering(entries, subset_seed=0)
    for per_band in (5, 16, 33):
        idx = select_nested_subset(entries, per_band=per_band, subset_seed=0)
        expected = [i for lab in BAND_LABELS for i in ordering[lab][:per_band]]
        assert idx == expected


# ------------------------------------------------------------------ hash

def test_subset_hash_is_order_sensitive():
    entries = _synth_entries()
    vocab = build_vocab()
    idx = select_nested_subset(entries, per_band=16, subset_seed=0)
    recs = subset_records(entries, idx, vocab)
    h1 = subset_definition_hash(recs)
    permuted = subset_records(entries, list(reversed(idx)), vocab)
    assert subset_definition_hash(permuted) != h1
    assert subset_definition_hash(subset_records(entries, idx, vocab)) == h1
    assert len(h1) == 64                               # sha256 hex


def test_subset_hash_changes_when_membership_changes():
    entries = _synth_entries()
    vocab = build_vocab()
    a = subset_records(entries, select_nested_subset(entries, 16, 0), vocab)
    b = subset_records(entries, select_nested_subset(entries, 16, 1), vocab)
    assert subset_definition_hash(a) != subset_definition_hash(b)


def test_subset_records_carry_the_required_definition_fields():
    entries = _synth_entries()
    vocab = build_vocab()
    idx = select_nested_subset(entries, per_band=16, subset_seed=0)
    recs = subset_records(entries, idx, vocab)
    assert len(recs) == 64
    required = {"position", "bank_index", "word", "phonemes", "phoneme_ids",
                "n_phonemes", "freq_rank", "band"}
    for pos, (r, i) in enumerate(zip(recs, idx)):
        assert required <= set(r)
        assert r["position"] == pos
        assert r["bank_index"] == i
        assert r["word"] == entries[i].word
        assert r["n_phonemes"] == len(entries[i].phonemes)
        assert r["freq_rank"] == entries[i].rank
        assert r["band"] in BAND_LABELS
        assert r["phonemes"] == " ".join(vocab.itos[p] for p in entries[i].phonemes)


# -------------------------------------------------- predeclared criteria

def test_c3_success_requires_all_three_conditions():
    ok = {"top1": 0.96, "target_rank_median": 1.0, "margin_mean": 0.01}
    assert c3_subset_success(ok)
    assert not c3_subset_success({**ok, "top1": 0.94})
    assert not c3_subset_success({**ok, "target_rank_median": 2.0})
    assert not c3_subset_success({**ok, "margin_mean": 0.0})     # strict >
    assert not c3_subset_success({**ok, "margin_mean": -0.01})


def test_naming_success_requires_both_conditions():
    ok = {"exact_match": 0.95, "whole_word_error_rate": 0.05}
    assert naming_subset_success(ok)
    assert not naming_subset_success({**ok, "exact_match": 0.9375})
    assert not naming_subset_success({**ok, "whole_word_error_rate": 0.06})


def test_success_thresholds_are_the_predeclared_values():
    assert C3_SUBSET_SUCCESS == {"top1_min": 0.95,
                                 "median_target_rank_max": 1.0,
                                 "margin_min_exclusive": 0.0}
    assert NAMING_SUBSET_SUCCESS == {"exact_min": 0.95, "wer_max": 0.05}
