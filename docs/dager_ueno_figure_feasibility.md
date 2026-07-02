# Dager / Ueno Figure Feasibility Manifest

> Checkpoint: `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt`
> Lexicon: `data/lexicon_en_glove_covered.tsv` (29,571 words)
> Train split: 25,136 words · Val split: 4,435 words
> Full/gated train ceiling: **1.0000** (0 errors)
>
> Status codes:
> - **READY** — output available today from existing scripts
> - **ADAPT** — data/infrastructure exists; one new script required
> - **MISSING** — requires new model capability or architecture change

---

## Part A — Dager / SWP-style figures

| # | Figure / analysis | Scientific question | Required task | Required data | Required model capability | Status | Existing script | Proposed script | Caveat |
|---|---|---|---|---|---|---|---|---|---|
| A1 | Train ceiling — perfect real-word repetition | Does the model reach 100% exact-match on every trained word? | Repetition (teacher-forced) | Training lexicon split | Forward pass; evaluate all train entries | **READY** | `scripts/evaluate_train_lexicon_ceiling.py` | — | Teacher-forced only; not free generation |
| A2 | WFE accuracy by condition: length × lexicality × morphology × frequency | Does the model show the same condition ordering as humans? | Repetition (teacher-forced) | `data/eval_external/wfe_eval.tsv` | Accuracy grouped by WFE condition codes (RLCH / RLCL / … / PSS) | **READY** | `scripts/external_eval.py` + `scripts/wfe_condition_analysis.py` | — | Need to run with 30k/GloVe checkpoint |
| A3 | WFE model-centred category barplot (full / WM / LTM per category) | Which route dominates for real-word-seen / real-word-unseen / pseudoword? | Repetition | Same as A2 | Route isolation via `route_logits` | **READY** | `scripts/wfe_condition_analysis.py` (make_model_centered_figure) | — | 4 × lexicon-overlap categories |
| A4 | WFE feature importance / regression | Which variables (length, freq, lexicality) predict per-item accuracy? | Repetition | `item_level_predictions.tsv` from A2 | Logistic regression over item features | **ADAPT** | Partial: external_eval.py produces features | `scripts/plot_wfe_dager_style.py` | Frequency in WFE is categorical (high/low), not continuous |
| A5 | Primacy/recency serial-position curve | Does the dorsal route show U-shaped position accuracy? | Repetition | Train-split predictions | Per-position accuracy via `per_position_correct`; aggregate by relative position | **ADAPT** | `evaluate/hooks.py::per_position_correct` | `scripts/plot_position_errors.py` | Must use WM route isolated; full route is LTM-dominated for real words |
| A6 | SSP sonority condition breakdown (CCV / VCC / …) | Does the model respect sonority sequencing in pseudoword repetition? | Repetition | `data/eval_external/ssp_eval.tsv` | Accuracy grouped by SSP condition | **ADAPT** | `scripts/external_eval.py` (produces SSP eval) | `scripts/plot_ssp_dager_style.py` | SSP is secondary; Dager figures use SSP for dorsal route | 
| A7 | Error-type breakdown: substitutions / insertions / deletions | What kind of errors does the model make? | Repetition | Any `item_level_predictions.tsv` | Levenshtein alignment (editops) | **ADAPT** | — | `scripts/plot_error_types.py` | Requires `python-Levenshtein` or `editdistance` with backtrace |
| A8 | Ablation scatter: WM-only vs LTM-only error rates per word | Do the two routes fail on different words? Double-dissociation plot | Repetition | Train predictions | `route_logits("wm")`, `route_logits("ltm")` — already in external_eval and ceiling scripts | **ADAPT** | `scripts/external_eval.py` (produces per-route columns) | `scripts/plot_wfe_dager_style.py` (route scatter section) | Gate is heuristic; WM/LTM isolation via route argument, not a true lesion |
| A9 | Route damage / lesion — WM knockout | What happens to repetition when dorsal route is disabled? | Repetition with WM zeroed | Any eval data | Force g=1 (LTM only) via `route_logits("ltm")` | **ADAPT** | `scripts/external_eval.py --route ltm` (not yet exposed as arg) | `scripts/run_route_ablation_wfe.py` | Not true biological lesion; route isolation via argument, not weight damage |
| A10 | Route damage / lesion — LTM knockout | What happens when ventral route is disabled? | Repetition with LTM zeroed | Any eval data | Force g=0 (WM only) via `route_logits("wm")` | **ADAPT** | Same as A9 | `scripts/run_route_ablation_wfe.py` | Same caveat as A9 |
| A11 | Multi-seed robustness | Are WFE condition effects stable across random seeds? | Repetition | Same as A2; requires ≥3 checkpoints trained with different seeds | Multiple trained checkpoints | **MISSING** | — | `scripts/run_dager_style_eval.py` (with seed loop) | Requires training 2–3 additional seeds from scratch (~2–3 hours each on GPU) |

---

## Part B — Ueno et al.-style figures

> **Critical caveat**: lichtheim3 implements **repetition only**. Ueno et al. require
> repetition, comprehension, and naming/speaking. Figures that involve comprehension
> or naming cannot be faithfully replicated without adding new task heads. The table
> below flags each figure with the task it requires and whether it is currently possible.

| # | Figure / analysis | Scientific question | Required task | Required data | Required model capability | Status | Existing script | Proposed script | Caveat |
|---|---|---|---|---|---|---|---|---|---|
| B1 | Ueno Fig 2 — learning curve for repetition | Does the repetition score improve monotonically with training? | Repetition | Train history from checkpoint | Plot `train_rep` and `val_rep` from `history` list in checkpoint | **READY** | `train.py::plot_loss_history` | — | Only repetition curve available; comprehension / naming curves MISSING |
| B2 | Ueno Fig 2 — learning curve for comprehension | Does comprehension emerge alongside repetition? | Comprehension (semantic decision) | Not present | Semantic output head + comprehension loss | **MISSING** | — | — | Comprehension not implemented; `s_hat` could proxy but is not validated |
| B3 | Ueno Fig 2 — learning curve for naming / speaking | Does naming emerge alongside repetition? | Naming (semantic → phonological) | Not present | Semantic input pipeline + naming loss | **MISSING** | — | — | Naming not implemented; see capability gap analysis |
| B4 | Ueno Fig 2b — held-out / nonword generalization for repetition | Does the model generalise to unseen items? | Repetition | Validation split (4,435 words); WFE pseudowords | Forward pass on val split or pseudoword set | **ADAPT** | `scripts/evaluate_train_lexicon_ceiling.py --include_val` | — | Val split = unseen words (same phoneme set); WFE pseudowords = nonwords |
| B5 | Ueno Fig 3 — lesion profiles across dorsal/ventral damage levels | How does each route's damage map to performance across tasks? | All three tasks | Any eval data | Parametric gate override or weight scaling | **ADAPT (rep only)** | — | `scripts/run_route_ablation_wfe.py` | Repetition route ablation possible now; comprehension/naming MISSING |
| B6 | Ueno Fig 4 — recovery after iSMG / dorsal lesion | Does the ventral route compensate during retraining after lesion? | Repetition; recovery protocol | Lesioned checkpoint + retrain | Lesion-then-retrain loop (zero WM weights, continue training) | **MISSING** | — | — | Recovery protocol not implemented; would require custom train loop |
| B7 | Ueno Fig 5 — RSA along ventral / dorsal pathways | Do intermediate representations follow semantic / phonological organisation? | Representational analysis | Activations: WM state h, LTM s_hat, motor premotor | `collect=True` to extract h and s_hat; RSA vs GloVe / phoneme-edit-distance | **ADAPT** | — | `scripts/run_rsa_analysis.py` | Only two representational levels available (WM h, LTM s_hat); no layerwise hierarchy |
| B8 | Ueno Fig 6 — semantic naming errors by lesion location | Are semantic errors concentrated at ventral lesions? | Naming | Output phoneme sequences + semantic category labels | Naming task + semantic categories per word | **MISSING** | — | — | Naming not implemented; no semantic category labels in current lexicon |
| B9 | Ueno Fig 7 — ventral-only (no dorsal) control | What does the model do without the WM buffer? | Repetition, LTM only | Any eval data | `route_logits("ltm")` | **ADAPT** | `scripts/external_eval.py` (per-route columns) | `scripts/run_route_ablation_wfe.py` | Equivalent to g=1 everywhere; route isolation is clean |
| B10 | Ueno naming task (as named in paper) | Can the model produce the word's phonological form given a semantic cue? | Naming (semantic → phonological free decoding) | GloVe vector as input | `ltm.decode(glove_vec, ...)` with greedy autoregressive decoding | **MISSING (architecturally proximate)** | — | `scripts/run_naming_eval.py` | Model NOT trained with naming loss; encoder bypassed; greedy decode needed |
| B11 | Ueno — double dissociation between real words and nonwords | Do dorsal lesions spare real words while damaging nonwords? | Repetition + route isolation | WFE or train/pseudoword sets | `route_logits` isolation | **ADAPT** | `scripts/external_eval.py` (produces wm/ltm per-route accuracy) | — | Gate is heuristic, not a biological lesion |

---

## Priority ranking for next sprint

| Priority | Figure | Effort | Value |
|---|---|---|---|
| 1 | A2 / A3 — WFE condition × route breakdown with 30k checkpoint | Low (rerun existing scripts) | High |
| 2 | A5 — Primacy/recency serial-position curve | Medium (1 new script) | High — directly tests WM route capacity hypothesis |
| 3 | A7 — Error-type breakdown | Medium (1 new script + editops dep) | High — clinical relevance |
| 4 | A9/A10/B9 — Route ablation WFE | Medium (1 new script, flag-level) | High — Ueno Fig 3/7 partial proxy |
| 5 | A4/A8 — Regression / scatter | Medium (extend plot_wfe_dager_style.py) | Medium |
| 6 | B4 — Val split / nonword generalisation | Low (--include_val flag exists) | Medium |
| 7 | B7 — RSA | High (extract activations at scale; need RSA library) | Medium |
| 8 | A6 — SSP sonority | Low–Medium (ssp_eval.tsv exists) | Medium |
| 9 | A11 — Multi-seed | Very high (retrain) | Low (for now) |
| 10 | B2/B3/B6/B8/B10 — Comprehension/naming/recovery | Very high (architecture change) | Deferred |
