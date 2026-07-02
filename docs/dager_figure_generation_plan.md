# Dager-Style Figure Generation Plan

> Stable checkpoint: `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt`
> Lexicon: `data/lexicon_en_glove_covered.tsv` (29,571 words)
> Full/gated train ceiling: 1.0000 (0 errors)
>
> This plan defines what scripts to create or update to reproduce Dager/SWP-style
> figures using the current checkpoint.  It is ordered by priority — run figures 1–4
> first, as they use data already produced by existing eval scripts.

---

## Prerequisites

All figures below use the WFE and SSP eval outputs.  These must be (re-)generated
with the 30k/GloVe checkpoint before running any plotting scripts:

```bash
# 1. Convert raw CSVs (if not already done)
python scripts/convert_csvs.py

# 2. Run WFE + SSP evaluation with 30k/GloVe checkpoint
python scripts/external_eval.py \
    --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
    --out_dir outputs/external_eval_30k

# 3. WFE condition breakdown + model-centred figure
python scripts/wfe_condition_analysis.py \
    --pred outputs/external_eval_30k/wfe/item_level_predictions.tsv
```

---

## Figure 1 — WFE length × lexicality × morphology × frequency

**Scientific question**: does lichtheim3 show the same condition ordering as humans
on the WFE stimuli (real > pseudo; short > long; simple > complex; high-freq > low-freq)?

**Script to update**: `scripts/plot_wfe_dager_style.py` (new)

**Inputs**:
- `outputs/external_eval_30k/wfe/item_level_predictions.tsv`
- columns: `word`, `condition`, `full_exact_match`, `wm_exact_match`, `ltm_exact_match`,
  `lexicality`, `length_cat`, `complexity`, `frequency`, `lexicon_category`

**Outputs**:
- `outputs/external_eval_30k/figures/wfe_condition_accuracy.png`
  Grouped barplot: x = WFE condition code; bars = full/WM/LTM; hue = route
- `outputs/external_eval_30k/figures/wfe_factor_main_effects.png`
  2×2 panel: lexicality × length, lexicality × complexity, freq (real only)
- `outputs/external_eval_30k/figures/wfe_condition_accuracy.tsv`

**Implementation notes**:
- Group `item_level_predictions.tsv` by `condition` + route columns
- Four main panels matching Dager figure layout: real vs pseudo × long vs short
- Include all three route bars (full / WM / LTM) in every panel
- Add `EVAL_REGIME_NOTE` in figure subtitle (teacher-forced)
- Logistic regression (sklearn) of `full_exact_match` on: `length_phonemes`,
  `lexicality_is_real` (0/1), `complexity_is_complex` (0/1), `frequency_is_high` (0/1)
  → coefficient plot saved to `wfe_logistic_regression.png`

---

## Figure 2 — Regression / feature importance

**Scientific question**: which stimulus features best predict per-item accuracy,
and does this match the human pattern?

**Script**: same `scripts/plot_wfe_dager_style.py` (section 2)

**Implementation notes**:
- Logistic regression: `full_exact_match ~ length_phonemes + is_real + is_high_freq + is_complex`
  (real-word items only for freq; pseudo items coded freq=NaN, excluded from freq coeff)
- Output: coefficient bar chart with 95% bootstrap CIs
- Separate models per route (full / WM / LTM) to show route-specific sensitivity

---

## Figure 3 — Primacy/recency serial-position curve

**Scientific question**: does the dorsal (WM) route show a U-shaped serial-position
accuracy curve, as predicted by a capacity-limited buffer with primacy and recency cues?

**Script to create**: `scripts/plot_position_errors.py`

**Inputs**:
- Train split entries (loaded from checkpoint + lexicon, same as ceiling eval)
- Per-position correctness computed via `evaluate/hooks.per_position_correct`
- Route: WM isolated (`route_logits("wm")`) to avoid LTM masking the curve

**Algorithm**:
```
For each word w in train split:
    For each position p in 0..len(w)-1:
        relative_pos = p / (len(w) - 1)   # 0.0 = first, 1.0 = last
        record (relative_pos, correct[w][p], route)
Bin relative_pos into 10 quantile bins
Plot mean accuracy per bin for WM / LTM / full routes
```

**Outputs**:
- `outputs/train_ceiling_analysis/serial_position_curve.png`
  Line plot: x = relative position bin; y = mean per-position accuracy; lines = routes
- `outputs/train_ceiling_analysis/serial_position_data.tsv`

**Implementation notes**:
- Use `collect=False` (no WM noise) for deterministic evaluation
- Length-stratify (2–4 phonemes vs 5–7 vs 8–9) to show length × position interaction
- The dorsal pool pseudoword stream (also length-constrained) should produce the U-shape;
  real words through full route may show a flatter curve because LTM compensates

---

## Figure 4 — SSP sonority gradient

**Scientific question**: does lichtheim3 repeat CCV clusters better than VCC, or vice versa?
Does the dorsal route show a sonority-sequencing preference?

**Script to create**: `scripts/plot_ssp_dager_style.py`

**Inputs**:
- `outputs/external_eval_30k/ssp/item_level_predictions.tsv`
- `data/eval_external/ssp_eval.tsv` for cluster-type metadata

**Algorithm**:
- Parse SSP condition codes (expected: CCV, CVV, VCC, VC, etc.)
- Group by cluster type × route; compute mean exact-match and phoneme accuracy
- For each condition: plot full / WM / LTM accuracy

**Outputs**:
- `outputs/external_eval_30k/figures/ssp_condition_accuracy.png`
- `outputs/external_eval_30k/figures/ssp_condition_accuracy.tsv`

**Implementation notes**:
- SSP is secondary; run only after Figure 1 is verified
- If SSP condition column is absent in `ssp_eval.tsv`, extract cluster type from
  the phoneme string directly (first 2 phonemes = onset cluster)
- Note teacher-forced regime in figure subtitle

---

## Figure 5 — Route comparison across tasks (full / WM / LTM)

**Scientific question**: do the three routes show the expected dissociation pattern
(full ≈ LTM for real words; full ≈ WM for pseudowords)?

**Script**: `scripts/run_route_ablation_wfe.py` (new)

**Inputs**: `wfe_eval.tsv`

**Algorithm**:
```
For each item in WFE:
    score_full  = eval with route="full"
    score_wm    = eval with route="wm"
    score_ltm   = eval with route="ltm"
    record (item, condition, lexicality, score_full, score_wm, score_ltm)
```

**Outputs**:
- `outputs/external_eval_30k/figures/route_dissociation_wfe.png`
  Scatter: x = WM score; y = LTM score; colour = lexicality (real/pseudo)
- `outputs/external_eval_30k/figures/route_accuracy_by_lexicality.png`
  Barplot: x = lexicality; bars = full/WM/LTM

**Note**: this extends what `external_eval.py` already does (per-route columns
are already in `item_level_predictions.tsv`). The new script adds the ablation
framing and route-dissociation scatter.

---

## Figure 6 — Error-type breakdown (substitutions / insertions / deletions)

**Scientific question**: are the model's errors predominantly substitutions (wrong
phoneme) or structural (missing / extra phoneme)?  Does this vary by route?

**Script to create**: `scripts/plot_error_types.py`

**Inputs**: any `item_level_predictions.tsv` (WFE or train ceiling)

**Dependency**: `python-Levenshtein` or `editdistance` (pip-installable; no heavy deps)

**Algorithm**:
```python
from Levenshtein import editops
for each row where exact_match == 0:
    ops = editops(predicted_form, target_form)
    n_sub = sum(1 for op in ops if op[0] == "replace")
    n_ins = sum(1 for op in ops if op[0] == "insert")
    n_del = sum(1 for op in ops if op[0] == "delete")
```

**Outputs**:
- `outputs/external_eval_30k/figures/error_type_breakdown.png`
  Stacked bar: x = route (full / WM / LTM); y = mean error count; stacked = sub/ins/del
- `outputs/external_eval_30k/figures/error_type_by_condition.png`
  Same, split by WFE lexicality (real vs pseudo)
- `outputs/external_eval_30k/figures/error_types.tsv`

---

## Optional Figure 7 — RSA on WM / LTM representations

**Scientific question**: does the WM route organise representations by phonological
similarity, and does the LTM route organise them by semantic similarity?

**Script to create**: `scripts/run_rsa_analysis.py`

**Inputs**:
- Train or val split; WFE items
- `collect=True` forward pass to extract: WM state `h` and LTM `s_hat`

**Algorithm**:
```
For each item:
    h_i     = WM encoder hidden state  (via wm_out["state"])
    s_hat_i = LTM semantic encoding    (via ltm_out["s_hat"])
Compute:
    D_phon  = pairwise phoneme edit distance matrix
    D_sem   = pairwise GloVe cosine distance matrix
    D_wm    = pairwise cosine distance in h-space
    D_ltm   = pairwise cosine distance in s_hat-space
RSA:
    r_wm_phon  = Spearman(D_wm.flatten(),  D_phon.flatten())
    r_ltm_sem  = Spearman(D_ltm.flatten(), D_sem.flatten())
    r_ltm_phon = Spearman(D_ltm.flatten(), D_phon.flatten())
    r_wm_sem   = Spearman(D_wm.flatten(),  D_sem.flatten())
```

**Outputs**:
- `outputs/rsa_analysis/rsa_summary.json`
- `outputs/rsa_analysis/rsa_matrix_comparison.png`
  2×2 grid: WM vs phonological / WM vs semantic / LTM vs phonological / LTM vs semantic

**Notes**:
- Limit to ≤2000 items for memory (pairwise matrix is O(N²))
- WM state extraction requires `collect=True` in WMRecurrent; currently returns `wm_out["state"]`
- This script is deferred; priority 7

---

## Orchestrator script

`scripts/run_dager_style_eval.py` — single entry point that calls all of the above
in the right order and with the right checkpoint path.

```bash
python scripts/run_dager_style_eval.py \
    --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
    --out_dir outputs/dager_style_figures
```

Internally calls:
1. `external_eval.py` (WFE + SSP)
2. `wfe_condition_analysis.py`
3. `plot_wfe_dager_style.py`
4. `plot_position_errors.py`
5. `plot_error_types.py`
6. `run_route_ablation_wfe.py`
7. `plot_ssp_dager_style.py`

---

## Constraint reminders

- Do NOT modify architecture, losses, or gate.
- Do NOT retrain on WFE or SSP.
- Do NOT include WFE/SSP items in the semantic bank.
- Do NOT use "yair" in any filename or directory.
- Do NOT commit checkpoints, GloVe files, or large output files.
- All figures must note "teacher-forced decoding" in subtitle or caption.
