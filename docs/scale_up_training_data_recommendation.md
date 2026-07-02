# Scale-Up Training Data Recommendation

> Status: pre-run recommendation — based on static code/data audit.
> Update this document after running `scripts/audit_lexicon_and_glove.py`.

---

## Context

The current checkpoint (`checkpoints/lichtheim3.pt`) was trained on 4 000 words
for 30 epochs.  The scale-up target is **30 000 words** with real GloVe embeddings,
trained until full exact-match repetition on trained words is achieved (ceiling = 1.0).

Primary metric: **full exact-match on the training split** evaluated by
`scripts/evaluate_train_lexicon_ceiling.py`.

---

## GloVe requirement

Real GloVe embeddings are required for the ventral route to learn genuine semantic
structure rather than CRC32-seeded hash vectors.  Without GloVe, the "semantic
alignment" loss is effectively just a per-word unique-ID memorisation task.

**Do not launch the 30K training run before verifying GloVe is present.**

```bash
bash data/get_glove.sh   # ~820 MB download; produces data/glove.6B.300d.txt
```

---

## Source recommendation

**Use Lichtheim3's `data/lexicon_en.tsv` first.**

Rationale:
1. Already fully integrated — zero pipeline changes beyond GloVe availability.
2. 30 000 words pre-built, frequency-ranked, stress-stripped CMU ARPABET.
3. 100% phoneme-compatible with the 39-symbol Lichtheim3 inventory.
4. A GloVe-filtered variant can be created in under a minute via
   `scripts/create_glove_covered_lexicon.py`.

Consider SWP/Dager source **only if**:
- GloVe coverage of `lexicon_en.tsv` falls below ~80% at 30k after filtering; OR
- POS or morphological metadata is needed for evaluation;
- In that case, extract `get_curated_words` logic standalone (simple CMU dict filter)
  rather than importing the full SWP package.

See `outputs/scale_up_data_audit/swp_training_source_audit.md` and
`outputs/scale_up_data_audit/training_source_comparison.md`.

---

## Is `lexicon_en.tsv` clean enough?

Based on static inspection (rows 1–30000 read, phoneme inventory verified):

| Check | Finding |
|---|---|
| Total words | 30 000 ✓ |
| Phoneme format | Stress-stripped CMU ARPABET ✓ |
| Phoneme-length range | 2–9 (enforced at build time) ✓ |
| Phoneme-inventory compatibility | 100% (all 39 symbols covered) ✓ |
| Duplicate words | Likely zero (CMU dict is word-keyed; `pron.setdefault` used) |
| Non-alphabetic entries | Likely zero (build script filters `w.isalpha()`) |
| Rank ordering | 1=most frequent; Zipfian tail from rank ~9k+ |

**Conclusion: clean enough to use directly.** Run `scripts/audit_lexicon_and_glove.py`
for exact counts once GloVe is present.

---

## GloVe coverage — expected

GloVe 6B covers ~400k English words.  Expected coverage of `lexicon_en.tsv`:

| Scale | Expected coverage |
|---|---|
| Top 10k | ~95–99% — all common words present |
| Top 20k | ~88–95% — some proper nouns / obscure words missing |
| Top 30k | ~80–90% — tail includes hunspell words and proper nouns |

Run `scripts/audit_lexicon_and_glove.py` for exact numbers.

---

## Decision tree

```
GloVe coverage of top 30k:
  ≥ 95%  → use lexicon_en.tsv directly, max_words=30000
  85–95% → create filtered lexicon_en_glove_covered.tsv; use max_words=<covered_n>
  < 85%  → use filtered lexicon up to covered_n; fallback 20k or 10k if needed
            OR use SWP source with explicit GloVe filter
```

---

## Exact commands to run next (in order)

```bash
# 0. Get real GloVe embeddings (REQUIRED)
bash data/get_glove.sh

# 1. Audit lexicon and GloVe coverage
python scripts/audit_lexicon_and_glove.py
# -> outputs/scale_up_data_audit/lichtheim3_lexicon_audit.md
# -> outputs/scale_up_data_audit/glove_coverage_lexicon_en.md

# 2. Create GloVe-filtered lexicon (if coverage < 100%)
python scripts/create_glove_covered_lexicon.py
# -> data/lexicon_en_glove_covered.tsv

# 3. Train (DO NOT launch until steps 0-2 are complete)
python scripts/train_checkpoint.py \
    --lexicon_path data/lexicon_en_glove_covered.tsv \
    --max_words 30000 \
    --epochs 30 \
    --seed 0 \
    --ckpt checkpoints/lichtheim3_30k_glove.pt

# 4. Verify ceiling (target: full_exact_match = 1.000 on train split)
python scripts/evaluate_train_lexicon_ceiling.py \
    --ckpt checkpoints/lichtheim3_30k_glove.pt

# 5. If ceiling not reached: increase epochs and retrain
python scripts/train_checkpoint.py \
    --lexicon_path data/lexicon_en_glove_covered.tsv \
    --max_words 30000 \
    --epochs 60 \
    --seed 0 \
    --ckpt checkpoints/lichtheim3_30k_glove_60ep.pt
```

---

## Phase 2: Longer training (post 30-epoch run)

### Current status (30-epoch run, 30k/GloVe)

```
checkpoint: checkpoints/lichtheim3_30k_glove.pt
n_train: 25,136  n_val: 4,435
full train exact-match: 0.9962
remaining train errors: 96
```

See `outputs/train_lexicon_ceiling/error_analysis.md` for detailed diagnosis.

### Error pattern

- 96 / 25 136 training words still fail.
- ~52% have WM correct — gate routing issue, not WM capacity.
- ~10% have LTM correct — gate routing issue the other way.
- ~37% both routes wrong — need more training gradient.
- Almost all errors are edit-distance = 1 (single phoneme off).
- High-frequency surprises: `december` (rank 396), `concepts` (3103),
  `wikipedia` (3423) — these should resolve quickly with more epochs.
- Long tail (rank > 20 000, length = 9) dominates; expect gradual improvement.

### Recommended next command

Resume from the 30-epoch checkpoint for 30 more epochs (60 total):

```bash
python scripts/train_checkpoint.py \
    --lexicon_path data/lexicon_en_glove_covered.tsv \
    --max_words 30000 --epochs 60 --seed 0 \
    --resume_from checkpoints/lichtheim3_30k_glove.pt \
    --ckpt checkpoints/lichtheim3_30k_glove_e60.pt
```

Then verify ceiling:

```bash
python scripts/evaluate_train_lexicon_ceiling.py \
    --ckpt checkpoints/lichtheim3_30k_glove_e60.pt \
    --lexicon_path data/lexicon_en_glove_covered.tsv
```

Then re-run error analysis:

```bash
python scripts/analyze_train_ceiling_errors.py
```

### Success criterion

`full_exact_match` on train split = **1.0000** and `train_errors.tsv` empty.

---

## Phase 3: e60 vs e90 — new finding (2026-07-02)

### Results

| Checkpoint | Train errors | Full exact |
|---|---|---|
| e30 (`lichtheim3_30k_glove.pt`) | 96 | 0.9962 |
| e60 (`lichtheim3_30k_glove_e60.pt`) | 46 | 0.9982 |
| e90 (`lichtheim3_30k_glove_e90.pt`) | 48 | 0.9981 |

e90 is marginally **worse** than e60.

### Error overlap (e60 vs e90)

| Category | Count |
|---|---|
| Common errors (in both) | 4 |
| Fixed by e90 | 42 |
| New in e90 | 44 |

Only 4 / 46 errors are shared.  The error set is **moving, not converging**.

### Root cause diagnosis

**Primary: non-deterministic ceiling evaluation.**
`evaluate_train_lexicon_ceiling.py` was calling the WM route with `collect=True`,
which enables Gaussian interference noise (`wm_route.py:52-54`) even in
`model.eval()` mode.  Every evaluation run produced a different set of WM
predictions for border-case items, and the full (gated) route inherits this noise.
There was also a dead first loop calling `route_predictions` twice per batch.

**Secondary: optimizer warm-restart.**
`--resume_from` was recreating a fresh `AdamW` (moments = 0) at the same LR
(1e-3 = original training LR).  This creates a different gradient trajectory
than continuous training and can transiently "unlearn" border-case items.

**Both issues are now fixed** (see `outputs/train_resume_audit/resume_audit.md`):
1. `evaluate_train_lexicon_ceiling.py` now uses `collect=False` by default —
   ceiling evaluation is deterministic.
2. `train_checkpoint.py` now saves/restores optimizer state + RNG states.
3. `--lr` flag allows lower-LR continuation without cold optimizer restart.

### Do NOT do this next

```
# ← BLOCKED until determinism is verified
python scripts/train_checkpoint.py --epochs 120 --resume_from ...
```

### Do this next (in order)

```bash
# 1. Verify determinism of e60 ceiling (should now be stable)
python scripts/check_ceiling_eval_determinism.py \
    --ckpt checkpoints/lichtheim3_30k_glove_e60.pt

# 2. Re-evaluate e60 with fixed ceiling script (no WM noise)
python scripts/evaluate_train_lexicon_ceiling.py \
    --ckpt checkpoints/lichtheim3_30k_glove_e60.pt \
    --lexicon_path data/lexicon_en_glove_covered.tsv \
    --out_dir outputs/train_lexicon_ceiling_e60_fixed

# 3. Compare e60 fixed vs e90 to understand how much was eval noise vs real
python scripts/compare_ceiling_checkpoints.py \
    --a outputs/train_lexicon_ceiling_e60_fixed/train_errors.tsv \
    --b outputs/train_lexicon_ceiling_e90/train_errors.tsv \
    --label_a e60_fixed --label_b e90

# 4. Resume from e60 with proper optimizer state + lower LR
python scripts/train_checkpoint.py \
    --lexicon_path data/lexicon_en_glove_covered.tsv \
    --max_words 30000 --epochs 90 --seed 0 \
    --resume_from checkpoints/lichtheim3_30k_glove_e60.pt \
    --lr 1e-4 \
    --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e90_lowlr.pt

# 5. Ceiling evaluation with fixed script
python scripts/evaluate_train_lexicon_ceiling.py \
    --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e90_lowlr.pt \
    --lexicon_path data/lexicon_en_glove_covered.tsv \
    --out_dir outputs/train_lexicon_ceiling_e90_lowlr
```

### LR choice

Current training LR: `1e-3` (from `config.py`).

| Continuation option | LR | Rationale |
|---|---|---|
| Mild (try first) | `3.3e-4` (LR/3) | Reduce step-size while keeping momentum |
| Conservative | `1e-4` (LR/10) | Standard fine-tuning reduction |
| Aggressive | `3e-5` (LR/30) | Last resort for hard residuals |

**Recommended: `--lr 1e-4`** for the first low-LR continuation from e60.

---

## Fallback scales

If 30k all-GloVe coverage is below ~85%:

| Fallback | `--max_words` | `--lexicon_path` |
|---|---|---|
| ~20k | 20000 | `data/lexicon_en_glove_covered.tsv` |
| ~10k | 10000 | `data/lexicon_en_glove_covered.tsv` |
| Exact covered count | see `glove_coverage_lexicon_en.json` → `coverage_by_scale` | same |

Prefer the largest scale where **all** included words have GloVe, rather than using
pseudo-vectors for the tail.

---

## What NOT to do

- Do not use `max_words=30000` with `data/lexicon_en.tsv` (without GloVe filtering)
  if any words lack GloVe — those words will silently fall back to hash vectors.
- Do not launch training before the audit confirms GloVe coverage.
- Do not train on WFE or SSP data.
- Do not change architecture, losses, or gate.
