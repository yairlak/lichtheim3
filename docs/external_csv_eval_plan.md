# Yair-L3: External CSV Evaluation Plan

## Purpose

Run clean external diagnostics of the Yair-L3 dual-route model on two
independently produced datasets:

- **WFE** (`data/raw-nwr_swp/wfe.csv`): 1200 items, 800 real words + 400
  pseudowords, from a controlled word-form experiment.
- **SSP** (`data/raw-nwr_swp/ssp.csv`): 16,560 3-phoneme CCV/VCC sequences
  designed to test Sonority Sequencing Principle compliance.

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

## Known caveats from the audit

| Caveat | Impact on this evaluation |
|---|---|
| GloVe absent (pseudo-hash vectors) | `lexical_confidence` and gate routing are still computed; "semantic" labels should be avoided |
| Single-seed training, 4k-word lexicon | Results reflect one checkpoint at reduced scale |
| WM route accuracy uses injected noise (`collect=True`) | Per-position accuracy is stochastic; averaged over 1 forward pass per item (deterministic for LTM/full) |
| Some committed `figures/summary.json` claims not reproduced | This evaluation does not validate those claims |

---

## File structure

```
data/
  raw-nwr_swp/          ← ORIGINAL (read-only)
    wfe.csv
    ssp.csv
    phonemes.csv
  eval_external/        ← CONVERTED (created by scripts/convert_csvs.py)
    wfe_yair_l3_format.tsv
    ssp_yair_l3_format.tsv

checkpoints/
  lichtheim3.pt         ← created by scripts/train_checkpoint.py

scripts/
  train_checkpoint.py   ← trains model + saves checkpoint
  inspect_csvs.py       ← Step 2: CSV compatibility report
  convert_csvs.py       ← Step 3: parse + filter + write TSVs
  external_eval.py      ← Steps 4–6: dry run + full WFE + full SSP

outputs/external_eval/
  csv_inspection_report.json
  csv_inspection_summary.md
  dry_run_wfe/
    item_level_predictions.tsv
    metrics.json
  dry_run_ssp/
    item_level_predictions.tsv
    metrics.json
  wfe/
    item_level_predictions.tsv
    metrics.json
    summary_table.tsv
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
    metrics.json
    summary_table.tsv
    figures/
      accuracy_by_sonority_{route}.png
      edit_distance_by_sonority_{route}.png
      accuracy_by_type_CCV_VCC_{route}.png
      phoneme_accuracy_by_sonority_{route}.png
  external_eval_summary.json

docs/
  yair_l3_external_csv_eval_plan.md   ← this file
  yair_l3_external_csv_eval_results.md
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
Yair-L3's 39-symbol ARPABET vocabulary.

### Step 2 — Convert CSVs

```bash
python scripts/convert_csvs.py
```

Outputs: `data/eval_external/wfe_yair_l3_format.tsv`
         `data/eval_external/ssp_yair_l3_format.tsv`

Parses Python-list-string phoneme columns, filters unknown phonemes,
flags out-of-distribution lengths (>9), writes clean TSVs.

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

- Used only for vocabulary validation (all 39 Yair-L3 ARPABET symbols confirmed).
- Not used for model features (Yair-L3 uses its own articulatory feature matrix
  from `data/phonemes.py`).
