# WFE Retraining Recommendation — lichtheim3 Dual-Route Model

> Status: recommendation draft — to be reviewed before any retraining.
>
> Key constraint: WFE and SSP remain external diagnostic datasets.
> They must NOT be used as training data unless explicitly approved (Option D).

---

## Background

The current checkpoint (`checkpoints/lichtheim3.pt`) was trained on the top
**4 000 words** of the bundled lexicon (`lexicon_en.tsv`) for **30 epochs**
with default config (`seed=0`, `max_phonemes=9`).  WFE contains 800 real words
whose phoneme sequences may or may not overlap with this training set.  The
`lexicon_category` column in `outputs/external_eval/wfe/item_level_predictions.tsv`
classifies every WFE word into one of:

| Category | Interpretation |
|---|---|
| `real_word_seen_in_training_lexicon` | Memorised during training; accuracy here reflects training performance |
| `real_word_in_validation_split` | In the lexicon but held out; genuine within-lexicon generalisation test |
| `real_word_outside_4000_lexicon` | Not in the 4 000-word lexicon; depends on WM phonotactic generalisation |
| `pseudoword` | No lexical entry; must be handled by the dorsal buffer |

Before committing to a retraining strategy, run `scripts/wfe_condition_analysis.py`
and inspect the `wfe_real_word_coverage.tsv` report to see how many WFE real words
fall into each category.

---

## SSP priority note

SSP is secondary.  It tests sub-lexical phonotactic sensitivity on 3-phoneme
controlled sequences and should not drive retraining decisions for WFE.
Address WFE first; SSP can be revisited once WFE coverage is satisfactory.

---

## Retraining options

### Option A — Same 4k lexicon, more epochs

**Config:** `--max_words 4000 --epochs 60 --seed 0` (or seed grid)

**What changes:** The model sees the same 4 000 words for longer.  If
performance on `real_word_seen_in_training_lexicon` items is already near-ceiling
(expected after 30 epochs) then additional epochs are unlikely to help.  Can
improve `real_word_in_validation_split` accuracy slightly if the model is still
underfitting the general phonotactic patterns, but likely produces minimal gain
beyond ~30 epochs for a 4k vocabulary.

**Pros:**
- Zero data change; fully reproducible.
- Fastest to run.
- Cleanest interpretation (nothing changes except gradient steps).

**Cons:**
- Does not add coverage of WFE words outside the 4k lexicon.
- Risk of overfitting to 4k training words without improving generalisation.

**When to choose:** Only if the coverage report shows that most WFE real words
ARE already in the training split and errors concentrate on pseudowords /
out-of-lexicon items where more epochs cannot help.

**Scientific claim:** "Longer training with the same lexicon."

---

### Option B — Larger lexicon (max\_words = 8 000)

**Config:** `--max_words 8000 --epochs 30 --seed 0`

**What changes:** The model trains on the top 8 000 English words.  This doubles
the lexicon size, covering more of the WFE real-word items while staying well
under the full 30k vocabulary.  Frequency weights still follow log-rank, so the
top-4k words still receive the most gradient exposure.

**Pros:**
- More WFE real words shift from `real_word_outside_4000_lexicon` into
  `real_word_seen_in_training_lexicon` or `real_word_in_validation_split`.
- Richer phonotactic distribution for the dorsal route.
- Remains a genuine external evaluation (WFE words were not targeted).
- Training time roughly doubles but is still feasible on CPU.

**Cons:**
- Checkpoint is no longer directly comparable to the committed
  `figures/summary.json` (which used max\_words=4 000).
- Some WFE real words may still fall outside even 8 000 words.

**When to choose:** Recommended first step.  It is the most principled
expansion that keeps external validity: the lexicon grows uniformly by
frequency rank, not by targeting WFE words specifically.

**Scientific claim:** "Training on a larger representative lexicon."

---

### Option C — Full available lexicon (max\_words = 30 000)

**Config:** `--max_words 30000 --epochs 15 --seed 0`

**What changes:** The model trains on all 30 000 bundled words.  Most WFE real
words are likely present (if they exist in CMU dict at plausible frequency ranks).

**Pros:**
- Maximally realistic training distribution.
- Closest to the intended scientific scope of the paper.
- WFE evaluation becomes a true external test of a production-scale checkpoint.

**Cons:**
- Training time is ~7–8× longer than Option A on CPU.
- Requires re-running all existing evaluations (`figures/summary.json`,
  `outputs/external_eval/`) to update the benchmark.
- Some effects may weaken at scale (e.g. WM route length sensitivity may
  become harder to isolate against a rich lexical background).

**When to choose:** Use as the target state for a publication-ready checkpoint.
Not recommended as the first retraining step; validate with Option B first.

**Scientific claim:** "Full-lexicon training with representative frequency weighting."

---

### Option D — Targeted inclusion of WFE real words

**Config:** Adds WFE real words explicitly to the training lexicon.

> ⚠ **SCIENTIFIC WARNING — read before choosing this option.**
>
> If WFE real words are added to the training lexicon, WFE is **no longer an
> external evaluation**.  Any accuracy reported on WFE real words would reflect
> in-distribution (or at best held-out) performance within a lexicon that was
> deliberately shaped to include those items.  This would invalidate the core
> motivation of the WFE evaluation as a diagnostic of generalisation.
>
> Option D is only appropriate if:
> (a) the goal is to demonstrate that the model *can* learn these word forms
>     given sufficient training exposure (a capacity demonstration, not a
>     generalisation test); AND
> (b) the resulting evaluation is clearly labelled as "trained-included" rather
>     than "external diagnostic."
>
> Do NOT present Option D results alongside Option A/B/C results as if they
> were the same kind of evaluation.

**Implementation if approved:**
1. Extract WFE real words from `data/raw-nwr_swp/wfe.csv`.
2. Cross-reference with `data/lexicon_en.tsv` to obtain pronunciation + frequency rank.
3. For words absent from the lexicon, append them with a conservative low-frequency rank.
4. Retrain with this augmented lexicon.
5. Re-run WFE evaluation and clearly label all outputs as "WFE-included training."

**When to choose:** Only if the primary goal is demonstrating training-coverage
improvement, NOT generalisation.

---

## Recommendation

**Start with Option B (max\_words = 8000)**, followed by re-running both the
existing internal evaluations and the external WFE/SSP diagnostics.

Steps:
```bash
# Retrain with enlarged lexicon
python scripts/train_checkpoint.py --max_words 8000 --epochs 30 --seed 0 \
    --ckpt checkpoints/lichtheim3_8k.pt

# Re-run internal evaluations
python run_all.py --max_words 8000 --epochs 30

# Re-run external WFE evaluation
python scripts/external_eval.py --wfe_only \
    --ckpt checkpoints/lichtheim3_8k.pt

# WFE condition analysis
python scripts/wfe_condition_analysis.py
```

If coverage of `real_word_outside_4000_lexicon` items remains low after Option B,
escalate to Option C.  Do not choose Option D without explicit approval and
clear documentation that the evaluation is no longer external.

---

## Metrics to compare across options

After each retraining, compare:

| Metric | Expected direction | Notes |
|---|---|---|
| WFE `real_word_outside_4000_lexicon` exact match | Increase (B/C only) | Coverage improves as more words enter training |
| WFE `real_word_in_validation_split` exact match | Small increase | Better generalisation from richer distribution |
| WFE pseudoword exact match | Should stay ≈ flat | Pseudoword accuracy tests the WM buffer, not lexical coverage |
| `figures/summary.json` val_rep loss | Should decrease | Better overall generalisation |
| Double-dissociation direction | Must be preserved | Ventral lesion must still hurt words more than pseudowords |

---

## SSP secondary priority — what to do

SSP analysis can be run at any checkpoint as a secondary diagnostic.  No
retraining should be driven by SSP results alone.  If SSP sonority-accuracy
correlation is already near-zero (as in the committed run), this is expected
given the model's CV-templated pseudoword curriculum which does not enforce SSP.
A targeted SSP-sensitive training curriculum would require modifying the
pseudoword pool generator (`data/dataset.py::build_pool_loader`) to prefer
legal onset/coda clusters — outside the current scope.
