"""Acceptance tests for the FINAL-1A population modes.

Covers the Phase FINAL-1A human decision (2026-09-01):

  * comprehension trains on ONE canonical lexical target per exact
    phonological encoder input (highest-frequency member per class,
    deterministic tie-break), expected N = 27,981 on the canonical lexicon;
  * naming and repetition train on the full 29,571-entry lexicon;
  * the retrieval bank stays the full 29,571-row GloVe bank;
  * the out-of-subset probe is explicitly disabled (never faked) when the
    populations cover the whole lexicon;
  * provenance names the populations, sizes and hashes;
  * exact resume still holds under the final_full population configuration;
  * naming evaluation consults no target length;
  * comprehension targets map by explicit bank index, and 100% strict
    word-ID top-1 is mathematically attainable on the canonical population.

Population-level tests run on the REAL canonical lexicon (GloVe disabled --
semantic provenance is irrelevant to phonological equivalence classes);
driver-level tests run on the small TINY lexicon like test_joint_scratch.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import default_config                                       # noqa: E402
from data.lexicon import LexEntry, build_lexicon                        # noqa: E402
from data.phonemes import build_vocab                                   # noqa: E402
from scripts.naming_comprehension.train_joint_scratch import (          # noqa: E402
    CANONICAL_N_WORDS, DEV_EVAL_SIZE, EXPECTED_CANONICAL_C_HASH,
    EXPECTED_CANONICAL_C_N, FINAL_FULL_MODE, JointScratchTrainer,
    deterministic_sample,
)
from scripts.naming_comprehension.train_tasks import (                  # noqa: E402
    canonical_phonology_indices, phonology_groups, subset_definition_hash,
    subset_records, evaluate_naming,
)
from scripts.naming_comprehension.frozen_probe import (                 # noqa: E402
    comprehension_metrics,
)

TINY = dict(device="cpu", max_words=400,
            lexicon_path="data/lexicon_en_glove_covered.tsv",
            dorsal_pool_size=32, batch_size=8, subset_mode=FINAL_FULL_MODE,
            subset_per_band=822, subset_size=32, lr_boundary_steps=6,
            allow_glove_fallback=True, require_subset_hash=False,
            glove_path="tests/_no_such_glove_file.txt")


def make_trainer(regime="j0", seed=22, **over):
    kw = dict(TINY)
    kw.update(over)
    return JointScratchTrainer(regime=regime, seed=seed, **kw)


@pytest.fixture(scope="module")
def canonical_entries():
    """The real canonical lexicon, GloVe disabled (phonology-only tests)."""
    cfg = default_config()
    cfg.data.lexicon_path = "data/lexicon_en_glove_covered.tsv"
    cfg.data.glove_path = None
    cfg.data.split_mode = "full_lexicon"
    cfg.data.val_fraction = 0.0
    return build_lexicon(cfg.data, build_vocab()).entries


# =====================================  canonical population definition  ====

def test_canonical_population_is_one_per_exact_phonology(canonical_entries):
    canon = canonical_phonology_indices(canonical_entries)
    forms = [tuple(canonical_entries[i].phonemes) for i in canon]
    assert len(forms) == len(set(forms)), "duplicate phonology among C targets"
    # ... and covers EVERY phonology exactly once.
    assert set(forms) == set(phonology_groups(canonical_entries).keys())


def test_canonical_population_expected_size(canonical_entries):
    assert len(canonical_entries) == CANONICAL_N_WORDS == 29_571
    canon = canonical_phonology_indices(canonical_entries)
    assert len(canon) == EXPECTED_CANONICAL_C_N == 27_981


def test_each_group_selects_its_highest_frequency_member(canonical_entries):
    groups = phonology_groups(canonical_entries)
    canon = set(canonical_phonology_indices(canonical_entries))
    for g in groups.values():
        chosen = [i for i in g if i in canon]
        assert len(chosen) == 1
        best_rank = min(canonical_entries[i].rank for i in g)
        assert canonical_entries[chosen[0]].rank == best_rank


def test_known_representatives(canonical_entries):
    canon = set(canonical_phonology_indices(canonical_entries))
    words = {canonical_entries[i].word for i in canon}
    # FINAL-0 examples: highest-frequency member of each verified group.
    assert "key" in words and "see" in words and "to" in words
    for excluded in ("quay", "qui", "sea", "two", "too"):
        idx = [i for i, e in enumerate(canonical_entries) if e.word == excluded]
        assert idx and idx[0] not in canon, f"{excluded} should not be a C target"


def test_tie_break_is_lowest_bank_index():
    """Ranks tie (synthetic) -> the lowest bank index wins deterministically."""
    sem = np.zeros(4, np.float32)
    entries = [
        LexEntry(word="b_first", phonemes=[5, 6], semantic=sem, freq=1.0, rank=7),
        LexEntry(word="a_second", phonemes=[5, 6], semantic=sem, freq=1.0, rank=7),
        LexEntry(word="unique", phonemes=[7, 8], semantic=sem, freq=1.0, rank=1),
    ]
    canon = canonical_phonology_indices(entries)
    assert canon == [0, 2]          # index 0 beats index 1 on the tie-break


def test_ranks_are_unique_so_tie_break_is_inert(canonical_entries):
    ranks = [e.rank for e in canonical_entries]
    assert len(set(ranks)) == len(ranks)
    # file order (bank index) is ascending rank, so min-rank == min-index
    assert all(b > a for a, b in zip(ranks, ranks[1:]))


def test_canonical_population_hash_is_stable(canonical_entries):
    vocab = build_vocab()
    canon = canonical_phonology_indices(canonical_entries)
    h = subset_definition_hash(subset_records(canonical_entries, canon, vocab))
    assert h == EXPECTED_CANONICAL_C_HASH
    # recomputation is deterministic
    assert h == subset_definition_hash(
        subset_records(canonical_entries, canon, vocab))


# ===============================================  driver population mode  ==

def test_final_full_populations_and_bank():
    tr = make_trainer()
    n = len(tr.entries)
    assert tr.streams["repetition"].n == n                     # R unchanged
    assert tr.streams["naming"].n == n                         # N = full
    assert tr.naming_idx == list(range(n))
    assert tr.streams["comprehension"].n == len(tr.comp_idx)   # C = canonical
    assert len(tr.comp_idx) == len(canonical_phonology_indices(tr.entries))
    assert tr.bank_raw.shape[0] == n                           # bank = full
    assert int(tr.model.ltm.semantic_bank.shape[0]) == n
    forms = [tuple(tr.entries[i].phonemes) for i in tr.comp_idx]
    assert len(forms) == len(set(forms)), "C target conflict remains"


def test_probe_disabled_not_faked_and_evaluate_does_not_crash():
    tr = make_trainer()
    assert tr.probe_idx == []
    assert "disabled" in tr.probe_note
    tr.train_step()
    row = tr.evaluate(with_probe=True)          # must not crash
    assert np.isnan(row["probe_rep_ltm"]) and np.isnan(row["probe_rep_full"])
    # dev-cadence metrics are real numbers
    assert np.isfinite(row["comp_top1"]) and np.isfinite(row["naming_exact"])


def test_endpoint_eval_reports_full_populations():
    tr = make_trainer()
    row = tr.evaluate(with_full_lexicon=True)
    for k in ("full_rep_full", "full_comp_top1", "full_naming_exact"):
        assert np.isfinite(row[k]), f"{k} missing from endpoint evaluation"


def test_resolved_settings_record_populations_and_hashes():
    tr = make_trainer()
    s = tr.resolved_settings()
    assert s["subset_mode"] == FINAL_FULL_MODE
    assert s["comprehension_population_name"] == "canonical_phonology"
    assert s["naming_population_name"] == "full_lexicon"
    assert s["comprehension_population"] == len(tr.comp_idx)
    assert s["naming_population"] == len(tr.entries)
    assert s["comprehension_population_sha256"] == tr.comp_hash
    assert s["naming_population_sha256"] == tr.naming_hash
    assert s["retrieval_bank_size"] == len(tr.entries)
    assert "Ueno" in s["homophone_policy"]
    ck = tr.state_dict()
    for key in ("comprehension_population_name", "comprehension_population_sha256",
                "naming_population_name", "naming_population_sha256",
                "probe_note"):
        assert key in ck


def test_legacy_modes_are_unchanged():
    tr = make_trainer(subset_mode="representative")
    assert tr.comp_idx is tr.subset_idx and tr.naming_idx is tr.subset_idx
    assert tr.dev_comp_idx is tr.subset_idx
    assert len(tr.probe_idx) == len(tr.subset_idx)


def test_final_full_requires_canonical_lexicon_when_hash_required():
    with pytest.raises(RuntimeError, match="canonical"):
        make_trainer(require_subset_hash=True)      # 400-word tiny lexicon


def test_deterministic_sample_is_pure_and_sorted():
    pop = list(range(100, 200))
    a = deterministic_sample(pop, 10, 42)
    assert a == deterministic_sample(pop, 10, 42)
    assert a == sorted(a) and len(set(a)) == 10
    assert a != deterministic_sample(pop, 10, 43)
    assert deterministic_sample(pop, 1000, 0) == pop
    assert len(deterministic_sample(list(range(30_000)), DEV_EVAL_SIZE, 0)) \
        == DEV_EVAL_SIZE


# ==========================================================  exact resume  ==

def _drive(tr, n):
    for _ in range(n):
        tr.train_step()


def test_exact_resume_under_final_full(tmp_path):
    torch.manual_seed(0)
    a = make_trainer()
    _drive(a, 7)                                     # mid-epoch, past LR boundary (6)
    ck = tmp_path / "mid.pt"
    torch.save(a.state_dict(), str(ck))
    _drive(a, 5)

    b = make_trainer()
    b.load_state_dict(torch.load(str(ck), weights_only=False), source="test")
    assert b.global_step == 7
    assert b.cursors == {"repetition": 7, "pool": 7,
                         "comprehension": 7, "naming": 7}
    _drive(b, 5)

    sa, sb = a.model.state_dict(), b.model.state_dict()
    bad = [k for k in sa if not torch.equal(sa[k], sb[k])]
    assert not bad, f"resume diverged on {len(bad)} tensors, e.g. {bad[:4]}"
    assert a.cursors == b.cursors and a.global_step == b.global_step


def test_resume_rejects_population_mode_mismatch(tmp_path):
    a = make_trainer()
    a.train_step()
    ck = tmp_path / "final.pt"
    torch.save(a.state_dict(), str(ck))
    b = make_trainer(subset_mode="representative")
    with pytest.raises(RuntimeError):
        b.load_state_dict(torch.load(str(ck), weights_only=False), source="t")


# =====================================================  metric semantics  ==

def test_naming_evaluation_has_no_target_length_leakage():
    """Predictions are a pure function of the semantic vector: rewriting the
    gold forms (the only place target length lives) must not change them."""
    tr = make_trainer()
    idx = tr.naming_idx[:16]
    base = evaluate_naming(tr.model, tr.vocab, tr.entries, tr.bank_raw, idx,
                           "cpu", max_steps=10, return_per_item=True)
    mangled = [LexEntry(word=e.word, phonemes=(e.phonemes * 3)[:9],
                        semantic=e.semantic, freq=e.freq, rank=e.rank)
               for e in tr.entries]
    other = evaluate_naming(tr.model, tr.vocab, mangled, tr.bank_raw, idx,
                            "cpu", max_steps=10, return_per_item=True)
    preds_a = [r["pred"] for r in base["_per_item"]]
    preds_b = [r["pred"] for r in other["_per_item"]]
    assert preds_a == preds_b, "naming predictions depend on the gold form"


def test_comprehension_targets_map_into_full_bank_and_100pct_is_attainable(
        canonical_entries):
    """With s_hat == the target GloVe row, every canonical target must rank 1
    in the FULL bank: strict word-ID top-1 = 100% is mathematically possible
    under the one-per-phonology population (checked on a slice for speed,
    including members of large homophone groups)."""
    rng = np.random.default_rng(0)
    bank = torch.tensor(rng.standard_normal((len(canonical_entries), 32)),
                        dtype=torch.float32)
    canon = canonical_phonology_indices(canonical_entries)
    probe = canon[:512] + canon[-512:]
    m = comprehension_metrics(bank[probe], bank, probe, batch_size=128)
    assert float(np.mean(m["top1"])) == 1.0
    assert int(np.max(m["target_rank"])) == 1
    assert list(m["top1_idx"][:8]) == probe[:8]     # explicit index mapping
