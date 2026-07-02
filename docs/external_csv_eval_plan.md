# External CSV Evaluation Plan — lichtheim3 Dual-Route Model

## Purpose

Run clean external diagnostics of the lichtheim3 dual-route model on two
independently produced datasets:

- **WFE** (`data/raw-nwr_swp/wfe.csv`): 1 200 items, 800 real words + 400
  pseudowords, from a controlled word-form experiment with a 12-condition
  factorial design.
- **SSP** (`data/raw-nwr_swp/ssp.csv`): 16 560 3-phoneme CCV/VCC sequences
  designed to probe Sonority Sequencing Principle compliance.
  **SSP is secondary priority for the current evaluation cycle; see §SSP note.**

**This is not a claim of cognitive replication.**  It is diagnostic analysis
of a specific model checkpoint under a documented, consistent evaluation regime.

---

## Constraints (must hold for all steps)

| Constraint | How enforced |
|---|---|
| Do not modify architecture | Scripts only call `model.forward()` / `model.route_logits()` |
| Do not modify losses | No gradient computation; `model.eval()` throughout |
| Do not modify the gate | Gate runs as-is via `model.forward()` |
| Do not retrain on eval datasets | CSV data never touches `DataLoader` or `Optimizer` |
| Do not mix CSVs with training lexicon | `data/eval_external/` is a separate directory, never passed to `build_lexicon()` |
| Do not overwrite original CSVs | `data/raw-nwr_swp/` is read-only; converted files go to `data/eval_external/` |

---

## Evaluation regime

All inference is **teacher-forced**:

```
enc_in  = [p₁, p₂, …, pₙ, EOS]          (input to encoder)
dec_in  = [BOS, p₁, p₂, …, pₙ]          (gold prefix to decoder at each step)
dec_tgt = [p₁, p₂, …, pₙ, EOS]          (target for loss / accuracy)
```

At position `t`, the decoder predicts `pₜ` given the GOLD phonemes `[p₁…pₜ₋₁]`,
not the model's own previous predictions.  This is identical to the regime used
by all `evaluate/*.py` in the repo and does **not** simulate free-recall error
propagation.  All outputs state this regime explicitly.

---

## Known caveats

| Caveat | Impact on this evaluation |
|---|---|
| GloVe absent (pseudo-hash vectors) | `lexical_confidence` and gate routing still computed; "semantic" labels must be avoided |
| Single-seed training, 4k-word lexicon | Results reflect one checkpoint at reduced scale |
| WM route uses injected noise (`collect=True`) | Per-position accuracy is stochastic; only one noise draw per item |
| Some committed `figures/summary.json` claims not reproduced by that run | This evaluation does not validate those claims |

---

## WFE condition key

Each WFE item has a condition code encoding up to four binary factors.

| Position | Real words | Pseudowords |
|---|---|---|
| 1 | R = real | P = pseudo |
| 2 | L = long (7–9 phones) / S = short (3–5 phones) | same |
| 3 | C = complex morphology / S = simple morphology | same |
| 4 | H = high frequency / L = low frequency | absent (no frequency column) |

Full condition decoding:

| Code | Lexicality | Size | Morphology | Freq group |
|---|---|---|---|---|
| RLCH | real | long | complex | high |
| RLCL | real | long | complex | low |
| RLSH | real | long | simple | high |
| RLSL | real | long | simple | low |
| RSCH | real | short | complex | high |
| RSCL | real | short | complex | low |
| RSSH | real | short | simple | high |
| RSSL | real | short | simple | low |
| PLC  | pseudo | long | complex | N/A |
| PLS  | pseudo | long | simple | N/A |
| PSC  | pseudo | short | complex | N/A |
| PSS  | pseudo | short | simple | N/A |

Frequency group is defined by splitting real-word items at the median
Zipf_Frequency within each size × morphology cell.  For pseudowords the
column is empty (not applicable).

---

## File structure

```
data/
  raw-nwr_swp/          ← ORIGINAL (read-only)
    wfe.csv
    ssp.csv
    phonemes.csv
  eval_external/        ← CONVERTED (created by scripts/convert_csvs.py)
    wfe_eval.tsv
    ssp_eval.tsv

checkpoints/
  lichtheim3.pt         ← created by scripts/train_checkpoint.py

scripts/
  train_checkpoint.py        ← Step 0: train + save checkpoint
  inspect_csvs.py            ← Step 1: vocabulary compatibility
  convert_csvs.py            ← Step 2: parse + filter + write TSVs
  external_eval.py           ← Steps 3–5: dry run + full WFE + SSP
  wfe_condition_analysis.py  ← Step 6: condition breakdown + error table

outputs/external_eval/          (gitignored — generated when scripts run)
  csv_inspection_report.json
  csv_inspection_summary.md
  dry_run_wfe/   dry_run_ssp/
  wfe/
    item_level_predictions.tsv
    metrics.json
    summary_table.tsv
    wfe_condition_breakdown.tsv   ← scripts/wfe_condition_analysis.py
    wfe_condition_breakdown.md
    wfe_real_word_coverage.tsv
    wfe_real_word_coverage.md
    train_seen_real_word_errors.tsv
    figures/
      accuracy_by_lexicality_{route}.png
      accuracy_by_length_{route}.png
      accuracy_by_size_{route}.png
      accuracy_by_morphology_{route}.png
      accuracy_by_condition_{route}.png
      edit_distance_by_lexicality_{route}.png
      frequency_effect_real_words_{route}.png
      route_accuracy_barplot_{lexicality}.png
  ssp/
    item_level_predictions.tsv
    metrics.json  summary_table.tsv  figures/
  external_eval_summary.json

docs/
  external_csv_eval_plan.md          ← this file
  external_csv_eval_results.md
  wfe_retraining_recommendation.md
```

---

## Step-by-step instructions

### Prerequisites

```bash
cd /path/to/lichtheim3
pip install -r requirements.txt    # torch, numpy, matplotlib (pandas needed too)
pip install pandas
```

### Step 0 — Check if a checkpoint already exists

```bash
ls checkpoints/lichtheim3.pt
```

If it does not exist, run:

```bash
# Reproduce the 30-epoch / 4k-word run that produced figures/summary.json
python scripts/train_checkpoint.py --max_words 4000 --epochs 30 --seed 0
```

This trains the model (architecture / losses / gate unchanged) and saves weights
to `checkpoints/lichtheim3.pt`.  It does NOT use the WFE or SSP CSVs.

### Step 1 — Inspect CSVs

```bash
python scripts/inspect_csvs.py
```

Outputs: `outputs/external_eval/csv_inspection_report.json`
         `outputs/external_eval/csv_inspection_summary.md`

Verifies that all phonemes in WFE / SSP `No_Stress` columns are in
the model's 39-symbol ARPABET vocabulary (`data/phonemes.py`).

### Step 2 — Convert CSVs

```bash
python scripts/convert_csvs.py
```

Outputs: `data/eval_external/wfe_eval.tsv`
         `data/eval_external/ssp_eval.tsv`

Parses Python-list-string phoneme columns, filters unknown phonemes,
flags out-of-training-distribution lengths (> 9 phones), writes clean TSVs.

### Step 3 — Dry runs

```bash
python scripts/external_eval.py --dry_run
```

Runs 10 WFE items + 20 SSP items. Outputs:
`outputs/external_eval/dry_run_wfe/`
`outputs/external_eval/dry_run_ssp/`

Check that:
- checkpoint loads without error
- phoneme sequences parse correctly
- teacher-forcing note appears in `metrics.json`
- predictions TSV looks reasonable

### Step 4 — Full WFE evaluation

```bash
python scripts/external_eval.py --wfe_only
```

### Step 5 — Full SSP evaluation

```bash
python scripts/external_eval.py --ssp_only
```

### Step 6 — Combined (re-run everything)

```bash
python scripts/external_eval.py
```

### Step 7 — WFE condition breakdown and error analysis

**Requires Step 4 to have run first** (needs `item_level_predictions.tsv`).

```bash
python scripts/wfe_condition_analysis.py
```

Outputs in `outputs/external_eval/wfe/`:
- `wfe_condition_breakdown.tsv` / `.md`
- `wfe_real_word_coverage.tsv` / `.md`
- `train_seen_real_word_errors.tsv`

---

## Metric definitions

| Metric | Definition |
|---|---|
| `exact_match` | 1 if full predicted phoneme sequence == gold sequence, else 0 |
| `phoneme_acc` | fraction of positions where `pred[t] == target[t]` (up to `len(target)`) |
| `edit_dist` | Levenshtein distance between predicted and target symbol sequences |
| `norm_edit` | `edit_dist / max(len_pred, len_target)` |

All metrics computed under teacher-forced decoding.

---

## Dataset notes

### WFE

- `No_Stress` column used as phoneme target (stress markers stripped).
- Format: Python list string `"['AH', 'T', 'EH']"` → parsed with `ast.literal_eval`.
- Pseudowords have no Frequency or Zipf_Frequency; frequency analyses restricted to
  `Lexicality == "real"` items.
- Items with unknown phonemes or parse failures excluded; flagged in TSV `notes` column.

### SSP

- All items are length-3 (CCV or VCC templates).
- Synthetic IDs `ssp_00000`–`ssp_16559` assigned (no orthographic form exists).
- SSP items NOT inserted into the Yair-L3 training lexicon.
- WM (dorsal) route is the primary route for SSP analysis (sub-lexical phonological
  processing); LTM/full results provided for completeness.
- Sonority column: higher values = greater sonority slope = more SSP-compliant.

### phonemes.csv

- Used only for vocabulary validation (all 39 ARPABET symbols confirmed against
  the model's inventory in `data/phonemes.py`).
- Not used for model feature computation.

---

## SSP note

SSP is **secondary priority** for the current evaluation cycle.  The 16 560
3-phoneme CCV/VCC sequences test sub-lexical phonotactic sensitivity but cannot
serve as a lexicality or frequency benchmark.  WFE analysis and the retraining
recommendation take priority.  SSP figures are produced by `external_eval.py`
but are not discussed in the main results document until WFE coverage improves.
See `docs/wfe_retraining_recommendation.md` for next steps.
