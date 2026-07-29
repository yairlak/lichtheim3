# Full-Lexicon Training — Design Document

## Purpose

Enable training on the entire GloVe-covered lexicon (29,571 words) with no
validation hold-out, to measure the true memorisation ceiling of the
dual-route model.  This is a distinct *training regime* (not a change to the
model architecture or loss function).

---

## Split modes

| `split_mode`   | `val_fraction` | Validation | Use case |
|---|---|---|---|
| `"standard"`   | 0.15 (default) | 15 % held out | normal training |
| `"full_lexicon"` | 0.0 (required) | DISABLED | ceiling measurement |

`split_mode` is an explicit field in `DataConfig`; inferring it from
`val_fraction` alone is prohibited to prevent ambiguity.

### Activating full-lexicon mode

**Fresh run (all 29,571 words):**
```bash
python scripts/train_checkpoint.py \
    --lexicon_path data/lexicon_en_glove_covered.tsv \
    --train_all_words --max_words 30000 --epochs 30 --seed 0 \
    --ckpt checkpoints/full_lexicon_s0.pt
```

> **Important:** `--train_all_words` activates `split_mode="full_lexicon"` (no
> validation hold-out) but does **not** neutralise `--max_words`.  The default
> CLI value for `--max_words` in fresh mode is `4000` (backward-compatibility).
> To train on the full 29,571-word TSV you must explicitly pass
> `--max_words 30000` (or higher).  Expected counts: `n_train=29571`, `n_val=0`.

**Resume** (mode inherited from checkpoint — `--train_all_words` not required):
```bash
python scripts/train_checkpoint.py \
    --resume_from checkpoints/full_lexicon_s0.pt \
    --epochs 60 --ckpt checkpoints/full_lexicon_s0_e60.pt
```

---

## Absent validation — representation contract

When `split_mode="full_lexicon"`:

- The val epoch loop is **skipped entirely** (not run on an empty DataLoader).
- History rows contain `val_rep=None`, `val_wm=None` (never `0.0`).
- Training logs print `val: DISABLED` instead of a metric value.
- `plot_loss_history` skips `None` val points silently.

**Why `None` not `0.0`:** a `0.0` validation loss is indistinguishable from a
perfect score and would corrupt any automated ceiling check.

---

## Checkpoint provenance fields (new)

All new fields are stored at the top level of the checkpoint dict.

### Provenance schema version

| Field | Type | Description |
|---|---|---|
| `provenance_schema_version` | `int` | `1` for all checkpoints produced by the current code |

**Detection semantics:**

| `provenance_schema_version` | Meaning |
|---|---|
| absent (`None`) | Legacy checkpoint (pre-patch); strict provenance unavailable — warning emitted, execution continues |
| `1` | All 22 `V1_REQUIRED_PROVENANCE_FIELDS` present and verified |
| any other value | Checkpoint from a **newer** version of the code — **fail-fast** (`sys.exit(1)`) |

The 22 required fields for schema version 1 are defined in `utils/provenance.py`
as `V1_REQUIRED_PROVENANCE_FIELDS`.  Absence of any of them in a v1 checkpoint
is a hard error at both resume and evaluation (cannot be bypassed without
`--ignore_provenance`).

### Split regime

| Field | Type | Description |
|---|---|---|
| `split_mode` | `str` | `"standard"` or `"full_lexicon"` |
| `train_all_words` | `bool` | `True` iff `full_lexicon` |
| `validation_enabled` | `bool` | `False` iff `full_lexicon` |
| `val_fraction` | `float` | 0.0 for full_lexicon |

### Dataset counters (from `LoadStats`)

| Field | Description |
|---|---|
| `n_source_rows` | Data rows in the TSV (excl. header) |
| `n_entries_after_loading` | Words surviving all filters |
| `n_filtered_unknown_phoneme` | Dropped: unknown phoneme symbol |
| `n_filtered_length` | Dropped: outside [min, max] phonemes |
| `n_unique_loaded_words` | Unique words in the loaded set |
| `n_glove_found` | Words with a real GloVe vector |
| `n_glove_fallback` | Words using deterministic pseudo-vector |
| `n_train` / `n_val` | Size of each split |
| `n_unique_train_words` | Unique words in the training split |

### Hashes

| Field | Convention |
|---|---|
| `lexicon_file_sha256` | SHA256 of the raw TSV bytes |
| `ordered_training_words_sha256` | SHA256 of train words in TSV order |
| `sorted_training_words_sha256` | SHA256 of train words alphabetically |
| `val_split_sha256` | SHA256 of val words alphabetically |

**Hash convention (must not change once checkpoints reference these):**
- UTF-8 encoding
- one word per line
- LF (`\n`) separator
- NO trailing newline

Reference values for `data/lexicon_en_glove_covered.tsv` (29,571 words, all in train):

```
lexicon_file_sha256   = ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66
ordered_words_sha256  = 0cb1c6172a7c2aea8a503549ffdf32543da820e1c505e0885f3999d6e50f7fa1
sorted_words_sha256   = f9721c17f97d0f2a1afeb97ce917075db202f21ad471dcb8755a26226de7d63a
```

### Split seed

| Field | Value in full_lexicon | Value in standard |
|---|---|---|
| `split_seed_used` | `False` | `True` |
| `split_seed_effective` | `None` | integer |
| `split_seed_configured` | `None` | `cfg.data.split_seed` (or `None` → falls back to `seed`) |

### Git state

| Field | Description |
|---|---|
| `git_commit` | HEAD commit SHA (best-effort; `"unknown"` if unavailable) |
| `git_branch` | Current branch name |
| `git_dirty` | `True` if working tree is dirty; `"unknown"` if git unavailable |

### Sampler config

```json
{
  "type": "WeightedRandomSampler",
  "replacement": true,
  "num_samples": <n_train>,
  "frequency_weighted": true,
  "freq_temp": 1.0,
  "n_train": <n_train>
}
```

---

## Resume contract

At resume (`--resume_from`), the following are **always inherited from the checkpoint**
and must NOT be overridden by CLI arguments:

- `lexicon_path`, `max_words`, `split_seed`, `batch_size`
- `teacher_forcing_ratio`, `interference_noise`, `ventral_noise`
- `gate_alpha`, `gate_threshold`, `hidden_size`, `ltm_encoder_mode`

If any of these are provided on the CLI and differ from the checkpoint value,
the script exits with a clear error message.

Operational overrides (safe to change at resume):
- `--lr`, `--num_workers`, `--save_every_epochs`, `--epochs` (target total)

**`--seed` at resume:** RNG states are restored from the checkpoint.  If
`--seed` is provided and differs from `cfg.train.seed`, a WARNING is printed
and the CLI value is ignored.

**Hash verification at resume:** the script rebuilds the lexicon and split from
the restored config, then computes `ordered_training_words_sha256` and compares
it to the checkpoint.  A mismatch is a hard error.

---

## Evaluator provenance

`scripts/evaluate_train_lexicon_ceiling.py` performs per-field verification
using `provenance_schema_version` to determine the check set:

1. Read `provenance_schema_version` from the checkpoint.
2. **Legacy (absent):** emit WARNING, set `provenance_verified=False`, continue.
3. **Unknown (≠ 1):** `sys.exit(1)` (checkpoint from newer code).
4. **v1:** verify all 22 `V1_REQUIRED_PROVENANCE_FIELDS` are present; then
   compare 15 fields per-check: split_mode, train_all_words, validation_enabled,
   n_source_rows, n_entries_after_loading, n_train, n_val, n_unique_train_words,
   n_glove_found (**mandatory** for v1), n_glove_fallback (**mandatory** for v1),
   lexicon_file_sha256, ordered_training_words_sha256, sorted_training_words_sha256,
   split_seed_used, split_seed_effective.
5. **All pass → `provenance_verified=True`** in `metrics.json`.
6. **Any mandatory failure → `sys.exit(1)`** unless `--ignore_provenance`.
   With `--ignore_provenance`: continues, but `provenance_verified=False`.

`metrics.json` always contains a `"provenance"` key:
```json
{
  "provenance": {
    "verified": true,
    "status": "verified",
    "split_mode": "full_lexicon",
    "validation_enabled": false,
    "rebuilt_ordered_sha256": "...",
    "ckpt_ordered_sha256": "..."
  }
}
```

---

## `LoadStats` dataclass

Populated inline during the single pass through the TSV in `build_bundled`.
Stored as `lexicon.load_stats` so checkpoint code can read provenance counters
without re-parsing the file or re-loading GloVe (the 1 GB vector file is
loaded once during training, not at checkpoint save time).

```python
@dataclass
class LoadStats:
    n_source_rows: int = 0
    n_entries_after_loading: int = 0
    n_filtered_unknown_phoneme: int = 0
    n_filtered_length: int = 0
    n_unique_loaded_words: int = 0
    n_glove_found: int = 0
    n_glove_fallback: int = 0
    lexicon_file_path: str = ""
    glove_file_path: Optional[str] = None
    lexicon_file_sha256: Optional[str] = None
```

---

## Files modified / created

| File | Change |
|---|---|
| `config.py` | Added `split_mode: str = "standard"` to `DataConfig`; `__post_init__` validation |
| `data/lexicon.py` | Added `LoadStats` dataclass; rewrote `build_bundled` to populate it inline |
| `train.py` | `build_and_train` skips val loop when `split_mode="full_lexicon"`; `plot_loss_history` skips `None` val points |
| `utils/provenance.py` | New module: `sha256_file`, `sha256_words_ordered`, `sha256_words_sorted`, `git_state` |
| `scripts/train_checkpoint.py` | Full rewrite of `parse_args`, `_build_ckpt_dict`, `main` (fresh + resume modes) |
| `scripts/evaluate_train_lexicon_ceiling.py` | Hash verification, `--ignore_provenance`, `provenance` block in `metrics.json` |
| `tests/test_full_lexicon.py` | 31 test scenarios (T01–T31) |
| `README_full_lexicon_design.md` | This document |

---

## Backward compatibility

Old checkpoints (without `split_mode`, `split_seed`, provenance hashes,
`provenance_schema_version`) load correctly:

- `DataConfig(**ckpt["cfg_data"])` uses `split_mode="standard"` as default.
- `get_effective_split_seed` falls back to `data.seed` when `split_seed` is `None`.
- Absent `provenance_schema_version` → treated as legacy; WARNING printed, strict
  checks skipped, `provenance_verified=False` / `provenance_status="legacy"`.
- `cfg_train` setdefaults protect against missing `teacher_forcing_ratio`,
  `save_every_epochs`, `num_workers` fields in old checkpoints.

---

## Empirical determinism check (H scenario)

To verify that AR evaluation is deterministic (no stochastic elements at eval
time), run the evaluator twice and compare the hashes of
`item_level_predictions.tsv`:

```bash
python scripts/evaluate_train_lexicon_ceiling.py \
    --ckpt <ckpt> --decode autoregressive --out_dir outputs/ar_run1
python scripts/evaluate_train_lexicon_ceiling.py \
    --ckpt <ckpt> --decode autoregressive --out_dir outputs/ar_run2
sha256sum outputs/ar_run1/ar/item_level_predictions.tsv \
          outputs/ar_run2/ar/item_level_predictions.tsv
```

Both hashes must match.  They will match because:
- Model is in `eval()` mode (dropout off).
- WM noise is disabled by default (`CEILING_COLLECT_WM_NOISE = False`).
- `torch.no_grad()` applied throughout.
- `torch.use_deterministic_algorithms(True)` is NOT set — determinism relies
  on no noise, not on algorithm-level determinism guarantees.
