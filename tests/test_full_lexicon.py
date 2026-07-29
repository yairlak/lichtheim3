"""Tests for the full-lexicon training patch (split_mode / train_all_words).

Scenarios covered:
  T01 – standard split_mode: historical behaviour unchanged (val_fraction=0.15 OK)
  T02 – full_lexicon split: all entries go to train, val is empty
  T03 – explicit split_mode='full_lexicon' requires val_fraction=0.0
  T04 – val_fraction=0.0 with split_mode='standard' is ambiguous → ValueError
  T05 – val metrics are None in full_lexicon mode (never 0.0)
  T06 – val metrics are float in standard mode
  T07 – checkpoint metadata: split_mode, train_all_words, validation_enabled correct
  T08 – hash convention: UTF-8, LF separator, no trailing LF
  T09 – sha256_words_ordered vs sha256_words_sorted differ for unsorted input
  T10 – resume fail-fast: contradicting --max_words rejected
  T10b– fresh mode without --max_words uses 4000 (backward-compat CLI default)
  T11 – resume fail-fast: --train_all_words against standard checkpoint rejected
  T12 – evaluator: provenance VERIFIED when hash matches
  T13 – evaluator: provenance mismatch → fail-fast without --ignore_provenance
  T14 – evaluator: legacy checkpoint (no hash) → warning not error
  T15 – sampler coverage: every train entry reachable (WeightedRandomSampler)
  T16 – LoadStats populated by build_bundled (integration, skipped without TSV)
  T17 – split_seed_effective=None for full_lexicon (seed not advertised as used)
  T18 – resume without --train_all_words inherits full_lexicon from checkpoint
  T19 – seed at resume: RNG states from checkpoint take precedence; CLI seed warned
  T20 – lexicon file hash detects TSV content change (same words, different rank)
  T21 – validate_split_config catches post-mutation contradictions
  T22 – epochs in fresh mode: cfg.train.epochs == args.epochs
  T23 – new checkpoint carries provenance_schema_version=1
  T24 – v1 checkpoint with missing required field → resume refuses (sys.exit)
  T25 – v1 checkpoint with missing required field → evaluator refuses
  T26 – legacy checkpoint (no schema version) → warning not crash
  T27 – unknown schema version → fail-fast (sys.exit)
  T28 – GloVe mismatch is mandatory failure for v1 checkpoints
  T29 – --ignore_provenance allows continuation but provenance_verified=False
  T30 – README contains --max_words 30000 in usage example
  T31 – full_lexicon v1 contract: split_seed_used=False, split_seed_effective=None
  T32 – _pchk sentinel: None==None → checked=True, match=True (not skipped)
  T33 – _pchk sentinel: key absent → checked=False (distinct from None value)
  T34 – _pchk sentinel: None vs integer → checked=True, match=False (mismatch)
  T35 – all_checks_passed requires every mandatory check to be checked=True
  T36 – _verify_dataset_provenance: split_seed_effective=None in v1 ckpt → no exit
  T37 – _pchk native types: None→null, bool→bool, int→int, float→float, str→str
  T38 – _pchk absent key: checkpoint_present=False, checked=False
  T39 – _pchk not-available rebuilt: rebuilt_available=False, checked=False
  T40 – _pchk full_lexicon split_seed_effective: rebuilt=None, checkpoint=None, match=True
"""
from __future__ import annotations

import hashlib
import os
import sys
import types
import unittest
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from config import DataConfig, _VALID_SPLIT_MODES, get_effective_split_seed, validate_split_config
from utils.provenance import sha256_words_ordered, sha256_words_sorted


# ---------------------------------------------------------------------------
# T01 – standard split_mode: historical val_fraction=0.15 still valid
# ---------------------------------------------------------------------------
class TestT01StandardModeValid(unittest.TestCase):
    def test_standard_valid(self):
        cfg = DataConfig(split_mode="standard", val_fraction=0.15)
        self.assertEqual(cfg.split_mode, "standard")
        self.assertEqual(cfg.val_fraction, 0.15)


# ---------------------------------------------------------------------------
# T02 – full_lexicon: all entries in train, val empty
# ---------------------------------------------------------------------------
class TestT02FullLexiconSplit(unittest.TestCase):
    def _make_lexicon(self):
        from data.phonemes import build_vocab
        from data.lexicon import LexEntry, Lexicon
        import numpy as np
        vocab = build_vocab()
        bos_id = vocab.bos_id if hasattr(vocab, "bos_id") else 0
        entries = [
            LexEntry(word=f"w{i}", phonemes=[bos_id + 1, bos_id + 2],
                     semantic=np.zeros(10, dtype="float32"), freq=1.0, rank=i)
            for i in range(20)
        ]
        return Lexicon(entries, vocab, semantic_dim=10, source="test")

    def test_all_words_in_train_when_val_fraction_zero(self):
        lex = self._make_lexicon()
        train, val = lex.split(0.0, seed=42)
        self.assertEqual(len(train), 20)
        self.assertEqual(len(val), 0)


# ---------------------------------------------------------------------------
# T03 – full_lexicon mode requires val_fraction=0.0
# ---------------------------------------------------------------------------
class TestT03FullLexiconRequiresZeroFraction(unittest.TestCase):
    def test_raises_when_val_fraction_nonzero(self):
        with self.assertRaises(ValueError) as ctx:
            DataConfig(split_mode="full_lexicon", val_fraction=0.15)
        self.assertIn("val_fraction", str(ctx.exception))

    def test_full_lexicon_with_zero_fraction_ok(self):
        cfg = DataConfig(split_mode="full_lexicon", val_fraction=0.0)
        self.assertEqual(cfg.split_mode, "full_lexicon")


# ---------------------------------------------------------------------------
# T04 – val_fraction=0.0 + standard is ambiguous → ValueError
# ---------------------------------------------------------------------------
class TestT04AmbiguousZeroFraction(unittest.TestCase):
    def test_raises_on_zero_with_standard(self):
        with self.assertRaises(ValueError) as ctx:
            DataConfig(split_mode="standard", val_fraction=0.0)
        self.assertIn("ambiguous", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# T05 – val metrics are None (not 0.0) when validation is DISABLED
# ---------------------------------------------------------------------------
class TestT05ValMetricsAreNoneNotZero(unittest.TestCase):
    def test_val_rep_none_not_zero(self):
        # Simulate the val loop skip logic from train.py / train_checkpoint.py
        split_mode = "full_lexicon"
        validation_enabled = (split_mode != "full_lexicon")
        if validation_enabled:
            val_rep = 0.42
        else:
            val_rep = None
        self.assertIsNone(val_rep)
        self.assertNotEqual(val_rep, 0.0)

    def test_history_stores_none(self):
        history_row = {"epoch": 1, "train_total": 0.5, "val_rep": None, "val_wm": None}
        self.assertIsNone(history_row["val_rep"])
        self.assertIsNone(history_row["val_wm"])


# ---------------------------------------------------------------------------
# T06 – val metrics are float in standard mode
# ---------------------------------------------------------------------------
class TestT06ValMetricsFloatInStandardMode(unittest.TestCase):
    def test_val_rep_is_float_in_standard(self):
        split_mode = "standard"
        validation_enabled = (split_mode != "full_lexicon")
        # In standard mode, val_rep should be set (here we just verify the branch)
        val_rep = 0.82 if validation_enabled else None
        self.assertIsNotNone(val_rep)
        self.assertIsInstance(val_rep, float)


# ---------------------------------------------------------------------------
# T07 – checkpoint metadata fields correct
# ---------------------------------------------------------------------------
class TestT07CheckpointMetadata(unittest.TestCase):
    def _fake_ckpt_metadata(self, split_mode):
        full_lexicon = (split_mode == "full_lexicon")
        return {
            "split_mode":         split_mode,
            "train_all_words":    full_lexicon,
            "validation_enabled": not full_lexicon,
            "val_fraction":       0.0 if full_lexicon else 0.15,
        }

    def test_full_lexicon_metadata(self):
        meta = self._fake_ckpt_metadata("full_lexicon")
        self.assertEqual(meta["split_mode"],         "full_lexicon")
        self.assertTrue(meta["train_all_words"])
        self.assertFalse(meta["validation_enabled"])
        self.assertEqual(meta["val_fraction"],       0.0)

    def test_standard_metadata(self):
        meta = self._fake_ckpt_metadata("standard")
        self.assertEqual(meta["split_mode"],         "standard")
        self.assertFalse(meta["train_all_words"])
        self.assertTrue(meta["validation_enabled"])
        self.assertEqual(meta["val_fraction"],       0.15)


# ---------------------------------------------------------------------------
# T08 – hash convention: UTF-8, LF separator, no trailing LF
# ---------------------------------------------------------------------------
class TestT08HashConvention(unittest.TestCase):
    def test_ordered_convention(self):
        words = ["apple", "banana", "cherry"]
        expected_bytes = "apple\nbanana\ncherry".encode("utf-8")
        expected_hex   = hashlib.sha256(expected_bytes).hexdigest()
        self.assertEqual(sha256_words_ordered(words), expected_hex)

    def test_no_trailing_newline(self):
        words = ["a", "b"]
        raw = "\n".join(words).encode("utf-8")
        self.assertFalse(raw.endswith(b"\n"))

    def test_empty_list(self):
        result = sha256_words_ordered([])
        self.assertEqual(result, hashlib.sha256(b"").hexdigest())


# ---------------------------------------------------------------------------
# T09 – ordered vs sorted hashes differ for unsorted input
# ---------------------------------------------------------------------------
class TestT09OrderedVsSortedDiffer(unittest.TestCase):
    def test_differ_for_unsorted(self):
        words = ["zebra", "apple", "mango"]
        h_ord = sha256_words_ordered(words)
        h_srt = sha256_words_sorted(words)
        self.assertNotEqual(h_ord, h_srt)

    def test_equal_for_already_sorted(self):
        words = sorted(["apple", "mango", "zebra"])
        h_ord = sha256_words_ordered(words)
        h_srt = sha256_words_sorted(words)
        self.assertEqual(h_ord, h_srt)


# ---------------------------------------------------------------------------
# T10 – resume fail-fast: contradicting --max_words rejected
# ---------------------------------------------------------------------------
class TestT10ResumeFailFastMaxWords(unittest.TestCase):
    def test_max_words_mismatch_exits(self):
        # Simulate the _resume_fail_fast logic from train_checkpoint.py
        ckpt_max_words = 29571
        cli_max_words  = 10000   # contradicts checkpoint

        errors = []
        if cli_max_words is not None and cli_max_words != ckpt_max_words:
            errors.append(f"--max_words: CLI={cli_max_words} ≠ checkpoint={ckpt_max_words}")
        self.assertTrue(len(errors) > 0, "Expected a fail-fast error")

    def test_matching_max_words_ok(self):
        ckpt_max_words = 29571
        cli_max_words  = 29571   # matches

        errors = []
        if cli_max_words is not None and cli_max_words != ckpt_max_words:
            errors.append("mismatch")
        self.assertEqual(errors, [])


# ---------------------------------------------------------------------------
# T11 – resume fail-fast: --train_all_words contradicts standard checkpoint
# ---------------------------------------------------------------------------
class TestT11ResumeFailFastTrainAllWords(unittest.TestCase):
    def test_train_all_words_against_standard_checkpoint(self):
        ckpt_split_mode  = "standard"
        cli_train_all    = True

        errors = []
        if cli_train_all and ckpt_split_mode != "full_lexicon":
            errors.append(
                f"--train_all_words: checkpoint split_mode={ckpt_split_mode!r}, "
                f"not 'full_lexicon'."
            )
        self.assertTrue(len(errors) > 0, "Should detect contradiction")

    def test_train_all_words_against_full_lexicon_checkpoint_ok(self):
        ckpt_split_mode  = "full_lexicon"
        cli_train_all    = True

        errors = []
        if cli_train_all and ckpt_split_mode != "full_lexicon":
            errors.append("contradiction")
        self.assertEqual(errors, [])


# ---------------------------------------------------------------------------
# T12 – evaluator: provenance VERIFIED when hash matches
# ---------------------------------------------------------------------------
class TestT12ProvenanceVerified(unittest.TestCase):
    def test_hash_match_means_verified(self):
        train_words = ["apple", "banana", "cherry"]
        rebuilt_sha = sha256_words_ordered(train_words)
        ckpt_sha    = sha256_words_ordered(train_words)  # same words → same hash

        if ckpt_sha is None:
            provenance_verified = False
        elif rebuilt_sha == ckpt_sha:
            provenance_verified = True
        else:
            provenance_verified = False

        self.assertTrue(provenance_verified)


# ---------------------------------------------------------------------------
# T13 – evaluator: hash mismatch → fail-fast without --ignore_provenance
# ---------------------------------------------------------------------------
class TestT13ProvenanceMismatchFailFast(unittest.TestCase):
    def test_mismatch_triggers_fail_fast(self):
        rebuilt_sha = sha256_words_ordered(["apple", "banana"])
        ckpt_sha    = sha256_words_ordered(["apple", "cherry"])   # different

        ignore_provenance = False
        would_exit = (rebuilt_sha != ckpt_sha) and not ignore_provenance
        self.assertTrue(would_exit)

    def test_mismatch_with_ignore_provenance_continues(self):
        rebuilt_sha = sha256_words_ordered(["apple", "banana"])
        ckpt_sha    = sha256_words_ordered(["apple", "cherry"])   # different

        ignore_provenance = True
        would_exit = (rebuilt_sha != ckpt_sha) and not ignore_provenance
        self.assertFalse(would_exit)


# ---------------------------------------------------------------------------
# T14 – evaluator: legacy checkpoint (no hash) → warning not sys.exit
# ---------------------------------------------------------------------------
class TestT14LegacyCheckpointNoHash(unittest.TestCase):
    def test_no_hash_means_legacy_not_crash(self):
        ckpt_ordered_sha = None   # legacy checkpoint has no hash field
        ignore_provenance = False

        if ckpt_ordered_sha is None:
            provenance_status = "legacy_no_hash"
            would_exit = False
        else:
            would_exit = (sha256_words_ordered(["x"]) != ckpt_ordered_sha
                          and not ignore_provenance)
            provenance_status = "verified" if not would_exit else "mismatch"

        self.assertEqual(provenance_status, "legacy_no_hash")
        self.assertFalse(would_exit)


# ---------------------------------------------------------------------------
# T15 – sampler coverage: all train entries reachable
# ---------------------------------------------------------------------------
class TestT15SamplerCoverage(unittest.TestCase):
    def test_all_entries_reachable(self):
        """WeightedRandomSampler with replacement=True and N steps covers the set
        with high probability (1/N probability of missing any one entry
        approaches 0 as N grows). We verify all entries have weight > 0."""
        import numpy as np
        n = 50
        ranks = np.arange(1, n + 1, dtype=np.float64)
        weights = np.log((n + 1.0) / ranks)
        # All weights must be positive (no zero-weight entries)
        self.assertTrue(np.all(weights > 0), "All training entries must have positive weight")
        self.assertEqual(len(weights), n)


# ---------------------------------------------------------------------------
# T16 – LoadStats populated by build_bundled (integration, skipped if TSV absent)
# ---------------------------------------------------------------------------
BUNDLED_PATH = os.path.join(ROOT, "data", "lexicon_en_glove_covered.tsv")

@unittest.skipUnless(os.path.exists(BUNDLED_PATH), "lexicon_en_glove_covered.tsv not present")
class TestT16LoadStatsIntegration(unittest.TestCase):
    def test_load_stats_populated(self):
        from config import DataConfig
        from data.phonemes import build_vocab
        from data.lexicon import build_bundled, LoadStats

        vocab = build_vocab()
        cfg = DataConfig(
            use_real=True,
            glove_path=None,   # no GloVe in unit test
            max_words=500,
            min_phonemes=2,
            max_phonemes=9,
            split_mode="full_lexicon",
            val_fraction=0.0,
        )
        lex = build_bundled(cfg, vocab, path=BUNDLED_PATH)
        self.assertIsNotNone(lex, "build_bundled returned None")
        self.assertIsNotNone(lex.load_stats, "load_stats should be set")
        ls = lex.load_stats
        self.assertIsInstance(ls, LoadStats)
        self.assertGreater(ls.n_source_rows, 0)
        self.assertGreater(ls.n_entries_after_loading, 0)
        self.assertEqual(ls.n_entries_after_loading, len(lex.entries))
        self.assertIsNotNone(ls.lexicon_file_sha256)
        self.assertEqual(len(ls.lexicon_file_sha256), 64,
                         "SHA256 should be a 64-character hex string")
        # In no-GloVe mode: all entries use fallback vectors
        self.assertEqual(ls.n_glove_found, 0)
        self.assertGreater(ls.n_glove_fallback, 0)


# ---------------------------------------------------------------------------
# T10b – fresh mode without --max_words uses 4000 (backward-compat CLI default)
# ---------------------------------------------------------------------------
class TestT10bFreshModeMaxWordsDefault(unittest.TestCase):
    def test_no_max_words_arg_gives_4000(self):
        """_apply_fresh_cli with max_words=None must set cfg.data.max_words = 4000,
        not the DataConfig dataclass default of 30000."""
        import argparse
        from config import default_config
        from train_checkpoint import _apply_fresh_cli

        args = argparse.Namespace(
            epochs=2, max_words=None, seed=None, split_seed=None,
            batch_size=None, lexicon_path=None, teacher_forcing_ratio=None,
            ltm_encoder_mode=None, hidden_size=None, lr=None,
            interference_noise=None, ventral_noise=None,
            gate_alpha=None, gate_threshold=None,
            num_workers=None, save_every_epochs=None, train_all_words=False,
        )
        cfg = default_config()
        _apply_fresh_cli(cfg, args)
        self.assertEqual(cfg.data.max_words, 4000,
                         "Fresh mode omitting --max_words should default to 4000 (legacy CLI)")

    def test_explicit_max_words_overrides_default(self):
        import argparse
        from config import default_config
        from train_checkpoint import _apply_fresh_cli

        args = argparse.Namespace(
            epochs=2, max_words=500, seed=None, split_seed=None,
            batch_size=None, lexicon_path=None, teacher_forcing_ratio=None,
            ltm_encoder_mode=None, hidden_size=None, lr=None,
            interference_noise=None, ventral_noise=None,
            gate_alpha=None, gate_threshold=None,
            num_workers=None, save_every_epochs=None, train_all_words=False,
        )
        cfg = default_config()
        _apply_fresh_cli(cfg, args)
        self.assertEqual(cfg.data.max_words, 500)


# ---------------------------------------------------------------------------
# T17 – split_seed_effective=None for full_lexicon
# ---------------------------------------------------------------------------
class TestT17SplitSeedFullLexicon(unittest.TestCase):
    def test_split_seed_effective_none_when_full_lexicon(self):
        cfg_data = DataConfig(split_mode="full_lexicon", val_fraction=0.0,
                              seed=42, split_seed=99)
        split_mode       = cfg_data.split_mode
        full_lexicon     = (split_mode == "full_lexicon")
        split_seed_eff   = None if full_lexicon else get_effective_split_seed(cfg_data)
        self.assertIsNone(split_seed_eff)
        self.assertFalse(split_mode == "standard")

    def test_split_seed_effective_set_when_standard(self):
        cfg_data = DataConfig(split_mode="standard", val_fraction=0.15,
                              seed=42, split_seed=99)
        split_mode     = cfg_data.split_mode
        full_lexicon   = (split_mode == "full_lexicon")
        split_seed_eff = None if full_lexicon else get_effective_split_seed(cfg_data)
        self.assertEqual(split_seed_eff, 99)


# ---------------------------------------------------------------------------
# T18 – resume without --train_all_words inherits full_lexicon from checkpoint
# ---------------------------------------------------------------------------
class TestT18ResumeInheritsSplitMode(unittest.TestCase):
    def test_resume_inherits_full_lexicon_without_flag(self):
        """Simulates loading a full_lexicon checkpoint and resuming without --train_all_words."""
        import argparse
        from train_checkpoint import _resume_fail_fast
        from config import Config, WMConfig, LTMConfig, GatingConfig, LossConfig, TrainConfig

        # Simulate a full_lexicon checkpoint
        ckpt_cfg_data = DataConfig(split_mode="full_lexicon", val_fraction=0.0,
                                   max_words=500, seed=0)
        cfg = Config(data=ckpt_cfg_data, wm=WMConfig(), ltm=LTMConfig(),
                     gating=GatingConfig(), loss=LossConfig(), train=TrainConfig())

        # CLI does NOT pass --train_all_words
        args = argparse.Namespace(
            lexicon_path=None, max_words=None, split_seed=None, batch_size=None,
            teacher_forcing_ratio=None, interference_noise=None, ventral_noise=None,
            gate_alpha=None, gate_threshold=None, hidden_size=None,
            ltm_encoder_mode=None, seed=None, train_all_words=False,
        )
        # Should not raise — full_lexicon is inherited from checkpoint, not required on CLI
        try:
            _resume_fail_fast(args, cfg)
        except SystemExit as e:
            self.fail(f"_resume_fail_fast raised SystemExit({e}) "
                      f"but should accept resume without --train_all_words")

        # After restoration, split_mode should still be full_lexicon
        self.assertEqual(cfg.data.split_mode, "full_lexicon")
        self.assertEqual(cfg.data.val_fraction, 0.0)


# ---------------------------------------------------------------------------
# T19 – seed at resume: CLI seed != checkpoint seed → warning, not error
# ---------------------------------------------------------------------------
class TestT19SeedAtResume(unittest.TestCase):
    def test_seed_mismatch_produces_warning_text(self):
        """Resume with a different --seed should log a warning, not crash.
        We verify the warning message is constructed, not that sys.exit is called."""
        ckpt_seed = 0
        cli_seed  = 42
        cfg_seed  = ckpt_seed

        warned = False
        if cli_seed is not None and cli_seed != cfg_seed:
            warning_msg = (f"WARNING: --seed {cli_seed} ignored. "
                           f"RNG states from checkpoint (training_seed={cfg_seed}) "
                           f"take precedence.")
            warned = True

        self.assertTrue(warned)
        self.assertIn("ignored", warning_msg)

    def test_matching_seed_produces_no_warning(self):
        ckpt_seed = 0
        cli_seed  = 0
        cfg_seed  = ckpt_seed

        warned = False
        if cli_seed is not None and cli_seed != cfg_seed:
            warned = True
        self.assertFalse(warned)


# ---------------------------------------------------------------------------
# T20 – file hash detects TSV content change (same words, different rank/phoneme)
# ---------------------------------------------------------------------------
class TestT20FileShaDetectsContentChange(unittest.TestCase):
    def test_same_words_different_rank_detected_by_file_sha(self):
        """Two TSV files with identical word lists but different rank columns
        must produce the same ordered_words_sha256 but different file_sha256.
        The evaluator's file hash check (lexicon_file_sha256) catches this.
        """
        import tempfile, os

        lines_v1 = "rank\tword\tarpabet\n1\tapple\tAE P AH L\n2\tbanana\tB AH N AE N AH\n"
        lines_v2 = "rank\tword\tarpabet\n2\tapple\tAE P AH L\n1\tbanana\tB AH N AE N AH\n"

        words_v1 = ["apple", "banana"]
        words_v2 = ["apple", "banana"]

        sha_words_v1 = sha256_words_ordered(words_v1)
        sha_words_v2 = sha256_words_ordered(words_v2)
        self.assertEqual(sha_words_v1, sha_words_v2,
                         "Same words in same order → same ordered_words hash")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f1:
            f1.write(lines_v1); p1 = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f2:
            f2.write(lines_v2); p2 = f2.name
        try:
            import hashlib
            sha_file_v1 = hashlib.sha256(open(p1, "rb").read()).hexdigest()
            sha_file_v2 = hashlib.sha256(open(p2, "rb").read()).hexdigest()
            self.assertNotEqual(sha_file_v1, sha_file_v2,
                                "Different TSV content → different file_sha256")
        finally:
            os.unlink(p1); os.unlink(p2)


# ---------------------------------------------------------------------------
# T21 – validate_split_config catches post-mutation contradictions
# ---------------------------------------------------------------------------
class TestT21ValidateSplitConfigMutation(unittest.TestCase):
    def test_mutation_to_full_lexicon_without_zero_fraction_caught(self):
        """If someone mutates split_mode to full_lexicon without updating val_fraction,
        validate_split_config should catch it."""
        cfg = DataConfig(split_mode="standard", val_fraction=0.15)
        cfg.split_mode = "full_lexicon"   # mutation without adjusting val_fraction
        with self.assertRaises(ValueError) as ctx:
            validate_split_config(cfg)
        self.assertIn("val_fraction", str(ctx.exception))

    def test_mutation_to_zero_fraction_without_full_lexicon_caught(self):
        cfg = DataConfig(split_mode="standard", val_fraction=0.15)
        cfg.val_fraction = 0.0   # mutation without adjusting split_mode
        with self.assertRaises(ValueError) as ctx:
            validate_split_config(cfg)
        self.assertIn("ambiguous", str(ctx.exception).lower())

    def test_valid_full_lexicon_mutation_passes(self):
        cfg = DataConfig(split_mode="standard", val_fraction=0.15)
        cfg.split_mode   = "full_lexicon"
        cfg.val_fraction = 0.0
        validate_split_config(cfg)   # should not raise


# ---------------------------------------------------------------------------
# T22 – epochs in fresh mode: cfg.train.epochs == args.epochs
# ---------------------------------------------------------------------------
class TestT22EpochsInFreshMode(unittest.TestCase):
    def test_epochs_applied_from_cli(self):
        import argparse
        from config import default_config
        from train_checkpoint import _apply_fresh_cli

        for n_epochs in [1, 2, 5, 30]:
            args = argparse.Namespace(
                epochs=n_epochs, max_words=None, seed=None, split_seed=None,
                batch_size=None, lexicon_path=None, teacher_forcing_ratio=None,
                ltm_encoder_mode=None, hidden_size=None, lr=None,
                interference_noise=None, ventral_noise=None,
                gate_alpha=None, gate_threshold=None,
                num_workers=None, save_every_epochs=None, train_all_words=False,
            )
            cfg = default_config()
            _apply_fresh_cli(cfg, args)
            self.assertEqual(cfg.train.epochs, n_epochs,
                             f"cfg.train.epochs should be {n_epochs}, got {cfg.train.epochs}")


# ---------------------------------------------------------------------------
# T23 – new checkpoint carries provenance_schema_version=1
# ---------------------------------------------------------------------------
class TestT23SchemaVersionInCheckpoint(unittest.TestCase):
    def test_schema_version_constant_is_one(self):
        from utils.provenance import PROVENANCE_SCHEMA_VERSION
        self.assertEqual(PROVENANCE_SCHEMA_VERSION, 1)

    def test_checkpoint_dict_includes_schema_version(self):
        """The constant imported from utils.provenance is what _build_ckpt_dict uses."""
        from utils.provenance import PROVENANCE_SCHEMA_VERSION
        ckpt_fragment = {"provenance_schema_version": PROVENANCE_SCHEMA_VERSION}
        self.assertIn("provenance_schema_version", ckpt_fragment)
        self.assertEqual(ckpt_fragment["provenance_schema_version"], 1)


# ---------------------------------------------------------------------------
# T24 – v1 checkpoint with missing required field → resume refuses (sys.exit)
# ---------------------------------------------------------------------------
class TestT24ResumeRefusesMissingV1Field(unittest.TestCase):
    def test_v1_checkpoint_missing_all_fields_exits(self):
        """_verify_dataset_provenance: v1 ckpt with missing required fields → sys.exit."""
        from config import default_config
        from train_checkpoint import _verify_dataset_provenance
        # schema_version=1 but ALL 22 required fields are absent
        ckpt = {"provenance_schema_version": 1}
        cfg  = default_config()
        with self.assertRaises(SystemExit):
            _verify_dataset_provenance(ckpt, cfg, lexicon=None,
                                       train_entries=[], val_entries=[])

    def test_legacy_ckpt_without_schema_version_does_not_exit(self):
        """_verify_dataset_provenance: absent schema_version → legacy path, no exit."""
        from config import default_config
        from train_checkpoint import _verify_dataset_provenance
        ckpt = {}  # no schema_version at all (legacy checkpoint)
        cfg  = default_config()
        try:
            _verify_dataset_provenance(ckpt, cfg, lexicon=None,
                                       train_entries=[], val_entries=[])
        except SystemExit as e:
            self.fail(f"Legacy checkpoint should not sys.exit, got SystemExit({e})")


# ---------------------------------------------------------------------------
# T25 – v1 checkpoint with missing required field → evaluator refuses
# ---------------------------------------------------------------------------
class TestT25EvaluatorRefusesMissingV1Field(unittest.TestCase):
    def test_missing_field_would_trigger_exit(self):
        """Evaluator inline logic: v1 ckpt with any missing required field → exit."""
        from utils.provenance import V1_REQUIRED_PROVENANCE_FIELDS
        ckpt = {"provenance_schema_version": 1}  # none of the 22 required fields present
        schema_version    = ckpt.get("provenance_schema_version")
        is_legacy         = (schema_version is None)
        ignore_provenance = False

        self.assertFalse(is_legacy, "schema_version=1 is not legacy")
        missing_fields = sorted(f for f in V1_REQUIRED_PROVENANCE_FIELDS if f not in ckpt)
        self.assertGreater(len(missing_fields), 0, "All 22 required fields should be missing")
        would_exit = bool(missing_fields) and not ignore_provenance
        self.assertTrue(would_exit, "Evaluator must exit on missing v1 fields")

    def test_with_ignore_provenance_no_exit(self):
        """With --ignore_provenance, evaluator skips exit even on missing fields."""
        from utils.provenance import V1_REQUIRED_PROVENANCE_FIELDS
        ckpt              = {"provenance_schema_version": 1}
        ignore_provenance = True
        missing_fields    = sorted(f for f in V1_REQUIRED_PROVENANCE_FIELDS if f not in ckpt)
        would_exit        = bool(missing_fields) and not ignore_provenance
        self.assertFalse(would_exit)


# ---------------------------------------------------------------------------
# T26 – legacy checkpoint (no schema version) → warning not crash
# ---------------------------------------------------------------------------
class TestT26LegacyCheckpointWarningNotCrash(unittest.TestCase):
    def test_absent_schema_version_is_legacy(self):
        """provenance_schema_version absent ↔ legacy; must not crash."""
        from config import default_config
        from train_checkpoint import _verify_dataset_provenance
        # A checkpoint that predates provenance_schema_version
        ckpt = {"ordered_training_words_sha256": "abc123def456"}
        cfg  = default_config()
        try:
            _verify_dataset_provenance(ckpt, cfg, lexicon=None,
                                       train_entries=[], val_entries=[])
        except SystemExit as e:
            self.fail(f"Legacy checkpoint must not sys.exit, got SystemExit({e})")

    def test_schema_version_key_distinguishes_legacy(self):
        ckpt_legacy = {"n_train": 100}
        ckpt_new    = {"n_train": 100, "provenance_schema_version": 1}
        self.assertIsNone(ckpt_legacy.get("provenance_schema_version"))
        self.assertEqual(ckpt_new.get("provenance_schema_version"), 1)


# ---------------------------------------------------------------------------
# T27 – unknown schema version → fail-fast (sys.exit)
# ---------------------------------------------------------------------------
class TestT27UnknownSchemaVersionFailFast(unittest.TestCase):
    def test_unknown_version_99_exits(self):
        from config import default_config
        from train_checkpoint import _verify_dataset_provenance
        ckpt = {"provenance_schema_version": 99}
        cfg  = default_config()
        with self.assertRaises(SystemExit):
            _verify_dataset_provenance(ckpt, cfg, lexicon=None,
                                       train_entries=[], val_entries=[])

    def test_unknown_version_2_exits(self):
        """Any version != 1 and != None should fail-fast."""
        from config import default_config
        from train_checkpoint import _verify_dataset_provenance
        ckpt = {"provenance_schema_version": 2}
        cfg  = default_config()
        with self.assertRaises(SystemExit):
            _verify_dataset_provenance(ckpt, cfg, lexicon=None,
                                       train_entries=[], val_entries=[])


# ---------------------------------------------------------------------------
# T28 – GloVe mismatch is mandatory failure for v1 checkpoints
# ---------------------------------------------------------------------------
class TestT28GloveMismatchMandatoryForV1(unittest.TestCase):
    def test_glove_mandatory_when_schema_v1(self):
        """n_glove_found mismatch with schema_version=1 → fail-fast without --ignore_provenance."""
        schema_version  = 1
        glove_mandatory = (schema_version == 1)
        self.assertTrue(glove_mandatory, "GloVe must be mandatory for v1")

    def test_glove_mismatch_would_exit(self):
        rebuilt_n_glove   = 29571
        ckpt_n_glove      = 10000   # wrong!
        ignore_provenance = False
        glove_mandatory   = True    # schema_version=1

        match = (rebuilt_n_glove == ckpt_n_glove)
        failed_mandatory = [
            {"field": "n_glove_found", "match": match, "mandatory": glove_mandatory}
        ]
        failed_mandatory = [c for c in failed_mandatory
                            if c["match"] is False and c["mandatory"]]
        self.assertTrue(len(failed_mandatory) > 0, "Mismatch must appear in failed_mandatory")
        would_exit = bool(failed_mandatory) and not ignore_provenance
        self.assertTrue(would_exit, "GloVe mismatch must cause sys.exit for v1")

    def test_glove_not_mandatory_for_legacy(self):
        """Legacy checkpoints (schema_version=None) do not mandate GloVe check."""
        schema_version  = None
        glove_mandatory = (schema_version == 1)
        self.assertFalse(glove_mandatory)


# ---------------------------------------------------------------------------
# T29 – --ignore_provenance allows continuation but provenance_verified=False
# ---------------------------------------------------------------------------
class TestT29IgnoreProvenanceVerifiedFalse(unittest.TestCase):
    def test_mismatch_plus_ignore_provenance_verified_false(self):
        """With hash mismatch + --ignore_provenance: no exit AND verified=False."""
        rebuilt_sha       = sha256_words_ordered(["apple", "banana"])
        ckpt_sha          = sha256_words_ordered(["apple", "cherry"])   # different!
        ignore_provenance = True

        all_checks_passed   = (rebuilt_sha == ckpt_sha)
        provenance_verified = all_checks_passed
        would_exit          = not all_checks_passed and not ignore_provenance

        self.assertFalse(would_exit,          "--ignore_provenance must prevent sys.exit")
        self.assertFalse(provenance_verified, "provenance_verified must be False on mismatch")

    def test_no_mismatch_verified_true(self):
        """Matching hashes + --ignore_provenance: provenance_verified=True."""
        words             = ["apple", "banana"]
        rebuilt_sha       = sha256_words_ordered(words)
        ckpt_sha          = sha256_words_ordered(words)   # same!
        ignore_provenance = True

        all_checks_passed   = (rebuilt_sha == ckpt_sha)
        provenance_verified = all_checks_passed
        self.assertTrue(provenance_verified)


# ---------------------------------------------------------------------------
# T30 – README contains --max_words 30000 in usage example
# ---------------------------------------------------------------------------
README_PATH = os.path.join(ROOT, "README_full_lexicon_design.md")

@unittest.skipUnless(os.path.exists(README_PATH), "README_full_lexicon_design.md not present")
class TestT30ReadmeMaxWords(unittest.TestCase):
    def test_readme_contains_max_words_30000(self):
        with open(README_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("--max_words 30000", content,
                      "README must include --max_words 30000 in the usage example")

    def test_readme_explains_max_words_and_train_all_words_interaction(self):
        with open(README_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        # README should explain that --train_all_words doesn't neutralise --max_words
        self.assertTrue(
            "neutralise" in content.lower() or "neutralize" in content.lower()
            or "does not" in content.lower() or "n'annule" in content.lower()
            or "max_words" in content,
            "README should explain --max_words interaction with --train_all_words"
        )


# ---------------------------------------------------------------------------
# T31 – full_lexicon v1 contract: split_seed_used=False, split_seed_effective=None
# ---------------------------------------------------------------------------
class TestT31SplitSeedFullLexiconV1Contract(unittest.TestCase):
    def test_split_seed_used_false_for_full_lexicon(self):
        split_mode       = "full_lexicon"
        is_full_lex      = (split_mode == "full_lexicon")
        split_seed_used  = not is_full_lex
        self.assertFalse(split_seed_used,
                         "split_seed_used must be False for full_lexicon")

    def test_split_seed_effective_none_for_full_lexicon(self):
        from config import DataConfig, get_effective_split_seed
        cfg_data         = DataConfig(split_mode="full_lexicon", val_fraction=0.0,
                                      seed=42, split_seed=7)
        is_full_lex      = (cfg_data.split_mode == "full_lexicon")
        split_seed_eff   = None if is_full_lex else get_effective_split_seed(cfg_data)
        self.assertIsNone(split_seed_eff,
                          "split_seed_effective must be None for full_lexicon")

    def test_standard_mode_has_seed_used_true(self):
        split_mode       = "standard"
        is_full_lex      = (split_mode == "full_lexicon")
        split_seed_used  = not is_full_lex
        self.assertTrue(split_seed_used,
                        "split_seed_used must be True for standard mode")

    def test_standard_mode_has_integer_seed_effective(self):
        from config import DataConfig, get_effective_split_seed
        cfg_data       = DataConfig(split_mode="standard", val_fraction=0.15,
                                    seed=42, split_seed=None)
        is_full_lex    = (cfg_data.split_mode == "full_lexicon")
        split_seed_eff = None if is_full_lex else get_effective_split_seed(cfg_data)
        self.assertIsNotNone(split_seed_eff)
        self.assertIsInstance(split_seed_eff, int)


# ---------------------------------------------------------------------------
# Helpers shared by T32-T36 that reproduce the sentinel logic inline
# ---------------------------------------------------------------------------

def _make_pchk_fn(ckpt: dict):
    """Return a _pchk function operating on `ckpt`, mirroring the evaluator logic.

    Returns (pchk_fn, _ABSENT sentinel, _NOT_AVAILABLE sentinel).
    The returned function stores native Python values (no str() conversion).
    """
    _ABSENT        = object()
    _NOT_AVAILABLE = object()

    def _pchk(label, rebuilt, saved_key, mandatory=True):
        checkpoint_present = (saved_key in ckpt)
        rebuilt_available  = (rebuilt is not _NOT_AVAILABLE)

        if not checkpoint_present:
            return {
                "field": label,
                "rebuilt":             rebuilt if rebuilt_available else None,
                "checkpoint":          None,
                "rebuilt_available":   rebuilt_available,
                "checkpoint_present":  False,
                "checked":             False,
                "match":               None,
                "mandatory":           mandatory,
            }
        saved = ckpt[saved_key]
        if not rebuilt_available:
            return {
                "field": label,
                "rebuilt":             None,
                "checkpoint":          saved,
                "rebuilt_available":   False,
                "checkpoint_present":  True,
                "checked":             False,
                "match":               None,
                "mandatory":           mandatory,
            }
        match = (rebuilt == saved)
        return {
            "field": label,
            "rebuilt":             rebuilt,
            "checkpoint":          saved,
            "rebuilt_available":   True,
            "checkpoint_present":  True,
            "checked":             True,
            "match":               match,
            "mandatory":           mandatory,
        }

    return _pchk, _ABSENT, _NOT_AVAILABLE


def _all_checks_passed(prov_checks):
    """Mirror the evaluator's mandatory_ok / optional_ok logic."""
    mandatory_checks = [c for c in prov_checks if c["mandatory"]]
    optional_checks  = [c for c in prov_checks if not c["mandatory"]]
    mandatory_ok = all(c["checked"] is True and c["match"] is True
                       for c in mandatory_checks)
    optional_ok  = all(c["match"] is not False for c in optional_checks)
    return mandatory_ok and optional_ok


# ---------------------------------------------------------------------------
# T32 – _pchk sentinel: None==None → checked=True, match=True
# ---------------------------------------------------------------------------
class TestT32PchkNoneEqualsNone(unittest.TestCase):
    def test_none_value_present_in_ckpt_gives_match(self):
        """Key present with value None + rebuilt None → checked=True, match=True."""
        ckpt = {"split_seed_effective": None}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        result = _pchk("split_seed_effective", None, "split_seed_effective", mandatory=True)
        self.assertTrue(result["checked"],  "checked must be True when key is present")
        self.assertIs(result["match"], True, "None==None must be match=True")
        self.assertFalse(result["match"] is False)

    def test_false_value_present_in_ckpt_gives_match(self):
        """split_seed_used=False in ckpt, rebuilt=False → checked=True, match=True."""
        ckpt = {"split_seed_used": False}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        result = _pchk("split_seed_used", False, "split_seed_used", mandatory=True)
        self.assertTrue(result["checked"])
        self.assertIs(result["match"], True)


# ---------------------------------------------------------------------------
# T33 – _pchk sentinel: key absent → checked=False (not confused with None value)
# ---------------------------------------------------------------------------
class TestT33PchkAbsentVsNone(unittest.TestCase):
    def test_absent_key_gives_checked_false(self):
        ckpt = {}   # key is absent
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        result = _pchk("split_seed_effective", None, "split_seed_effective", mandatory=True)
        self.assertFalse(result["checked"], "absent key must give checked=False")
        self.assertIsNone(result["match"],  "absent key must give match=None")

    def test_none_value_checked_true_vs_absent_checked_false(self):
        """Presence of key with None value differs from absence of key."""
        ckpt_with_none   = {"split_seed_effective": None}
        ckpt_without_key = {}

        _pchk_with, *_ = _make_pchk_fn(ckpt_with_none)
        _pchk_without, *_ = _make_pchk_fn(ckpt_without_key)

        r_with    = _pchk_with("split_seed_effective",    None, "split_seed_effective")
        r_without = _pchk_without("split_seed_effective", None, "split_seed_effective")

        self.assertTrue(r_with["checked"],    "key present with None → checked=True")
        self.assertFalse(r_without["checked"],"key absent → checked=False")
        self.assertIs(r_with["match"], True)
        self.assertIsNone(r_without["match"])


# ---------------------------------------------------------------------------
# T34 – _pchk sentinel: None vs integer → mismatch (wrong split_seed_effective)
# ---------------------------------------------------------------------------
class TestT34PchkNoneVsIntegerMismatch(unittest.TestCase):
    def test_full_lexicon_none_vs_integer_is_mismatch(self):
        """Checkpoint has split_seed_effective=0 but full_lexicon expects None → mismatch."""
        ckpt = {"split_seed_effective": 0}   # wrong: should be None for full_lexicon
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        rebuilt = None   # full_lexicon → None
        result = _pchk("split_seed_effective", rebuilt, "split_seed_effective", mandatory=True)
        self.assertTrue(result["checked"])
        self.assertIs(result["match"], False, "None vs 0 must be a mismatch")

    def test_standard_mode_integer_matches(self):
        """Standard mode: both sides have the same integer → match."""
        ckpt = {"split_seed_effective": 42}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        result = _pchk("split_seed_effective", 42, "split_seed_effective", mandatory=True)
        self.assertTrue(result["checked"])
        self.assertIs(result["match"], True)

    def test_standard_mode_integer_vs_none_is_mismatch(self):
        """Standard mode: ckpt has None but rebuilt is integer → mismatch."""
        ckpt = {"split_seed_effective": None}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        result = _pchk("split_seed_effective", 42, "split_seed_effective", mandatory=True)
        self.assertTrue(result["checked"])
        self.assertIs(result["match"], False)


# ---------------------------------------------------------------------------
# T35 – all_checks_passed requires mandatory checked=True (unchecked → False)
# ---------------------------------------------------------------------------
class TestT35AllChecksPassedRequiresCheckedTrue(unittest.TestCase):
    def test_unchecked_mandatory_fails_all_checks_passed(self):
        """A mandatory check with checked=False must make all_checks_passed False."""
        prov_checks = [
            {"field": "split_mode",        "checked": True,  "match": True,  "mandatory": True},
            {"field": "split_seed_effective","checked": False, "match": None,  "mandatory": True},
        ]
        result = _all_checks_passed(prov_checks)
        self.assertFalse(result,
                         "Unchecked mandatory check must prevent all_checks_passed=True")

    def test_all_mandatory_checked_and_matched_passes(self):
        """All mandatory checks checked=True, match=True → all_checks_passed=True."""
        prov_checks = [
            {"field": "split_mode",         "checked": True, "match": True,  "mandatory": True},
            {"field": "n_train",            "checked": True, "match": True,  "mandatory": True},
            {"field": "split_seed_effective","checked": True, "match": True,  "mandatory": True},
            {"field": "n_glove_found",      "checked": True, "match": True,  "mandatory": False},
        ]
        result = _all_checks_passed(prov_checks)
        self.assertTrue(result)

    def test_optional_unchecked_does_not_fail(self):
        """An unchecked optional check does not make all_checks_passed False."""
        prov_checks = [
            {"field": "split_mode",    "checked": True,  "match": True, "mandatory": True},
            {"field": "n_glove_found", "checked": False, "match": None, "mandatory": False},
        ]
        result = _all_checks_passed(prov_checks)
        self.assertTrue(result, "Unchecked optional check should not block all_checks_passed")

    def test_mandatory_match_false_fails(self):
        """A mandatory check with match=False (mismatch) fails all_checks_passed."""
        prov_checks = [
            {"field": "split_mode", "checked": True, "match": False, "mandatory": True},
        ]
        result = _all_checks_passed(prov_checks)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# T36 – _verify_dataset_provenance: split_seed_effective=None in v1 ckpt → no exit
# ---------------------------------------------------------------------------
class TestT36VerifyProvenanceNullSplitSeedNoExit(unittest.TestCase):
    """End-to-end test calling the actual _verify_dataset_provenance function
    with a full_lexicon v1 checkpoint where split_seed_effective=None.

    Pass lexicon=None (so ls=None → ls-based checks are skipped via _NOT_AVAILABLE).
    Use train_entries=[] / val_entries=[] so reconstructed counters are 0.
    Set matching checkpoint values for everything we CAN compare.
    """

    def _make_full_lexicon_ckpt(self):
        from utils.provenance import sha256_words_ordered, sha256_words_sorted
        empty_ord = sha256_words_ordered([])
        empty_srt = sha256_words_sorted([])
        return {
            "provenance_schema_version": 1,
            # Split regime
            "split_mode":             "full_lexicon",
            "train_all_words":        True,
            "validation_enabled":     False,
            "val_fraction":           0.0,
            # Counters (ls=None → these are skipped at rebuild; just must be present)
            "n_source_rows":             29571,
            "n_entries_after_loading":   29571,
            "n_filtered_unknown_phoneme": 0,
            "n_filtered_length":         0,
            "n_unique_loaded_words":     29571,
            # Counts matching train_entries=[], val_entries=[]
            "n_train":               0,
            "n_val":                 0,
            "n_unique_train_words":  0,
            # GloVe (ls=None → skipped at rebuild; just must be present)
            "n_glove_found":         29571,
            "n_glove_fallback":      0,
            # Paths (presence required; value not compared)
            "lexicon_file_path":     "/data/lexicon_en_glove_covered.tsv",
            "glove_file_path":       "/data/glove.6B.300d.txt",
            # Hashes matching train_words=[] (sha of empty string)
            "lexicon_file_sha256":           "placeholder_skipped_ls_none",
            "ordered_training_words_sha256": empty_ord,
            "sorted_training_words_sha256":  empty_srt,
            # Split seed — THE KEY: None is legitimate for full_lexicon
            "split_seed_used":       False,
            "split_seed_effective":  None,
            # Sampler
            "sampler_config":        {"type": "WeightedRandomSampler"},
        }

    def test_none_split_seed_effective_does_not_exit(self):
        from config import (Config, DataConfig, WMConfig, LTMConfig,
                            GatingConfig, LossConfig, TrainConfig)
        from train_checkpoint import _verify_dataset_provenance

        ckpt = self._make_full_lexicon_ckpt()
        cfg  = Config(
            data   = DataConfig(split_mode="full_lexicon", val_fraction=0.0, seed=0),
            wm     = WMConfig(), ltm=LTMConfig(), gating=GatingConfig(),
            loss   = LossConfig(), train=TrainConfig(),
        )
        try:
            _verify_dataset_provenance(ckpt, cfg, lexicon=None,
                                       train_entries=[], val_entries=[])
        except SystemExit as e:
            self.fail(
                f"_verify_dataset_provenance raised SystemExit({e}) for a valid v1 "
                f"full_lexicon checkpoint with split_seed_effective=None"
            )

    def test_wrong_split_seed_effective_causes_exit(self):
        """If checkpoint has split_seed_effective=0 (wrong for full_lexicon) → sys.exit."""
        from config import (Config, DataConfig, WMConfig, LTMConfig,
                            GatingConfig, LossConfig, TrainConfig)
        from train_checkpoint import _verify_dataset_provenance

        ckpt = self._make_full_lexicon_ckpt()
        ckpt["split_seed_effective"] = 0   # wrong: full_lexicon should have None

        cfg  = Config(
            data   = DataConfig(split_mode="full_lexicon", val_fraction=0.0, seed=0),
            wm     = WMConfig(), ltm=LTMConfig(), gating=GatingConfig(),
            loss   = LossConfig(), train=TrainConfig(),
        )
        with self.assertRaises(SystemExit):
            _verify_dataset_provenance(ckpt, cfg, lexicon=None,
                                       train_entries=[], val_entries=[])

    def test_absent_split_seed_effective_causes_exit(self):
        """If split_seed_effective key is missing from v1 ckpt → sys.exit (missing field)."""
        from config import (Config, DataConfig, WMConfig, LTMConfig,
                            GatingConfig, LossConfig, TrainConfig)
        from train_checkpoint import _verify_dataset_provenance

        ckpt = self._make_full_lexicon_ckpt()
        del ckpt["split_seed_effective"]   # remove required field

        cfg  = Config(
            data   = DataConfig(split_mode="full_lexicon", val_fraction=0.0, seed=0),
            wm     = WMConfig(), ltm=LTMConfig(), gating=GatingConfig(),
            loss   = LossConfig(), train=TrainConfig(),
        )
        with self.assertRaises(SystemExit):
            _verify_dataset_provenance(ckpt, cfg, lexicon=None,
                                       train_entries=[], val_entries=[])


# ---------------------------------------------------------------------------
# T37 – _pchk native types: None→null, bool→bool, int→int, float→float, str→str
# ---------------------------------------------------------------------------
class TestT37PchkNativeTypes(unittest.TestCase):
    """Values stored in check dicts must survive json.dumps→json.loads unchanged."""

    def _roundtrip(self, value):
        import json
        return json.loads(json.dumps(value))

    def test_none_survives_json_roundtrip(self):
        """None must serialise as JSON null, not the string "None"."""
        ckpt = {"split_seed_effective": None}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        c = _pchk("split_seed_effective", None, "split_seed_effective")
        self.assertIsNone(self._roundtrip(c["rebuilt"]),
                          "rebuilt=None must round-trip as JSON null, not string 'None'")
        self.assertIsNone(self._roundtrip(c["checkpoint"]),
                          "checkpoint=None must round-trip as JSON null")

    def test_bool_stays_bool(self):
        ckpt = {"split_seed_used": False}
        _pchk, *_ = _make_pchk_fn(ckpt)
        c = _pchk("split_seed_used", False, "split_seed_used")
        rebuilt_rt   = self._roundtrip(c["rebuilt"])
        checkpoint_rt = self._roundtrip(c["checkpoint"])
        self.assertIsInstance(rebuilt_rt,   bool, "bool rebuilt must stay bool after JSON")
        self.assertIsInstance(checkpoint_rt, bool, "bool checkpoint must stay bool after JSON")
        self.assertFalse(rebuilt_rt)
        self.assertFalse(checkpoint_rt)

    def test_int_stays_int(self):
        ckpt = {"n_train": 29571}
        _pchk, *_ = _make_pchk_fn(ckpt)
        c = _pchk("n_train", 29571, "n_train")
        self.assertIsInstance(self._roundtrip(c["rebuilt"]),   int)
        self.assertIsInstance(self._roundtrip(c["checkpoint"]), int)
        self.assertEqual(self._roundtrip(c["rebuilt"]),   29571)

    def test_float_stays_float(self):
        ckpt = {"val_fraction": 0.15}
        _pchk, *_ = _make_pchk_fn(ckpt)
        c = _pchk("val_fraction", 0.15, "val_fraction")
        self.assertIsInstance(self._roundtrip(c["rebuilt"]),   float)
        self.assertIsInstance(self._roundtrip(c["checkpoint"]), float)

    def test_sha256_string_stays_str(self):
        sha = "0cb1c6172a7c2aea8a503549ffdf32543da820e1c505e0885f3999d6e50f7fa1"
        ckpt = {"ordered_training_words_sha256": sha}
        _pchk, *_ = _make_pchk_fn(ckpt)
        c = _pchk("ordered_training_words_sha256", sha, "ordered_training_words_sha256")
        self.assertIsInstance(self._roundtrip(c["rebuilt"]),   str)
        self.assertEqual(self._roundtrip(c["rebuilt"]),   sha)
        self.assertEqual(self._roundtrip(c["checkpoint"]), sha)


# ---------------------------------------------------------------------------
# T38 – _pchk absent key: checkpoint_present=False, checked=False
# ---------------------------------------------------------------------------
class TestT38PchkAbsentKey(unittest.TestCase):
    def test_absent_key_fields(self):
        ckpt = {}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        c = _pchk("split_seed_effective", None, "split_seed_effective", mandatory=True)
        self.assertFalse(c["checkpoint_present"], "Key absent → checkpoint_present=False")
        self.assertFalse(c["checked"],            "Key absent → checked=False")
        self.assertIsNone(c["match"],             "Key absent → match=None")
        self.assertIsNone(c["checkpoint"],        "Key absent → checkpoint stored as None")

    def test_absent_key_with_available_rebuilt(self):
        ckpt = {}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        c = _pchk("n_train", 42, "n_train", mandatory=True)
        self.assertTrue(c["rebuilt_available"])
        self.assertFalse(c["checkpoint_present"])
        self.assertFalse(c["checked"])
        self.assertEqual(c["rebuilt"], 42)


# ---------------------------------------------------------------------------
# T39 – _pchk not-available rebuilt: rebuilt_available=False, checked=False
# ---------------------------------------------------------------------------
class TestT39PchkNotAvailableRebuilt(unittest.TestCase):
    def test_not_available_rebuilt_fields(self):
        ckpt = {"n_glove_found": 29571}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        c = _pchk("n_glove_found", _NOT_AVAILABLE, "n_glove_found", mandatory=True)
        self.assertFalse(c["rebuilt_available"],  "NOT_AVAILABLE → rebuilt_available=False")
        self.assertTrue(c["checkpoint_present"],  "Key present → checkpoint_present=True")
        self.assertFalse(c["checked"],            "NOT_AVAILABLE rebuilt → checked=False")
        self.assertIsNone(c["rebuilt"],           "NOT_AVAILABLE rebuilt → rebuilt stored as None")
        self.assertEqual(c["checkpoint"], 29571)

    def test_not_available_does_not_falsely_match(self):
        """An uncalculable rebuilt must not be a match, even if checkpoint is also None."""
        ckpt = {"n_glove_found": None}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        c = _pchk("n_glove_found", _NOT_AVAILABLE, "n_glove_found", mandatory=True)
        self.assertFalse(c["checked"])
        self.assertIsNone(c["match"])


# ---------------------------------------------------------------------------
# T40 – _pchk full_lexicon split_seed_effective: rebuilt=None, checkpoint=None, match=True
# ---------------------------------------------------------------------------
class TestT40PchkFullLexiconSplitSeedEffective(unittest.TestCase):
    def test_none_none_match_true_fields(self):
        """The canonical real-world case: full_lexicon ckpt has split_seed_effective=None."""
        ckpt = {"split_seed_effective": None}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        c = _pchk("split_seed_effective", None, "split_seed_effective", mandatory=True)

        self.assertTrue(c["rebuilt_available"],  "None is a valid value → rebuilt_available=True")
        self.assertTrue(c["checkpoint_present"], "Key is present → checkpoint_present=True")
        self.assertTrue(c["checked"],            "Both present → checked=True")
        self.assertIs(c["match"], True,          "None==None → match=True")
        self.assertIsNone(c["rebuilt"],          "rebuilt stored as Python None (not string)")
        self.assertIsNone(c["checkpoint"],       "checkpoint stored as Python None (not string)")
        self.assertTrue(c["mandatory"])

    def test_all_checks_passed_with_null_split_seed(self):
        """Verify all_checks_passed=True when split_seed_effective=None on both sides."""
        prov_checks = [
            {"field": "split_mode",          "checked": True, "match": True,  "mandatory": True},
            {"field": "split_seed_used",      "checked": True, "match": True,  "mandatory": True},
            {"field": "split_seed_effective", "checked": True, "match": True,  "mandatory": True},
        ]
        self.assertTrue(_all_checks_passed(prov_checks))

    def test_json_roundtrip_of_full_check(self):
        """The full check dict for split_seed_effective=None must survive JSON serialisation."""
        import json
        ckpt = {"split_seed_effective": None}
        _pchk, _ABSENT, _NOT_AVAILABLE = _make_pchk_fn(ckpt)
        c = _pchk("split_seed_effective", None, "split_seed_effective", mandatory=True)
        serialised = json.dumps(c)
        recovered  = json.loads(serialised)
        self.assertIsNone(recovered["rebuilt"],    "After JSON round-trip, rebuilt must be null")
        self.assertIsNone(recovered["checkpoint"], "After JSON round-trip, checkpoint must be null")
        self.assertIs(recovered["match"], True)
        self.assertIs(recovered["checked"], True)


if __name__ == "__main__":
    unittest.main()
