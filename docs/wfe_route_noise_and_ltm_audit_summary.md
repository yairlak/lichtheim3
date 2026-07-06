# WFE Route, Noise, and LTM Audit Summary

> **Purpose.** Summary of the WFE evaluation audit for Lichtheim3 / Yair-L3 at the 30 k GloVe-covered scale-up checkpoint.  Intended for discussion with Yair.  The document distinguishes established results, likely interpretations, caveats, and proposed next steps.

---

## 1. Scope and checkpoint

| Item | Value |
|---|---|
| Checkpoint | `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt` |
| Lexicon | 29,571 GloVe-covered words |
| Train split | 25,136 words |
| Held-out validation | 4,435 words |
| Ceiling (full/gated, train split) | exact match = **1.0000** |
| WFE status | External evaluation dataset; WFE itself was not used for training, but some real WFE words overlap with the training lexicon. |

### WFE item categorisation

WFE items are classified into four lexicon-overlap categories:

| Category | Meaning |
|---|---|
| `real_word_seen_in_training_lexicon` | Real word whose orthographic form or phoneme sequence appears in the 25 k train split |
| `real_word_in_validation_split` | Real word in the held-out 4 k split |
| `real_word_outside_4000_lexicon` | Legacy label for WFE real words not classified as train-seen or held-out validation in the current overlap audit; should be interpreted as “novel / outside current effective split” rather than literally tied to the 4k model. |
| `pseudoword` | Lexicality ≠ "real" in the WFE dataset |

**Two group modes are used throughout:**

- **Dager-comparable** (`dager_strict`): train-seen real words vs actual pseudowords only.  Held-out and novel real words are excluded from the main figures.  This closely mirrors the Dager/SWP experimental design.
- **Generalization** (`seen_vs_unseen`): train-seen real words vs all unseen forms (held-out real + novel real + pseudowords).  The "Unseen forms" group is a **familiarity** group, not a lexicality group.  Do not call these items "pseudowords."

---

## 2. Evaluation regimes: teacher-forced vs autoregressive

### Teacher-forced decoding (Regime A)

At each decoder step, the decoder receives the **gold** previous phoneme as input.  Errors do not propagate — a wrong prediction at position *t* does not corrupt position *t+1*.

- **Established result:** Teacher-forced gives an upper bound on per-step accuracy.
- **Use:** ceiling check, debugging, per-position fidelity probe.
- **Not appropriate for** behavioural comparison with human WFE / Dager data, where recall is free and errors accumulate.

### Autoregressive decoding (Regime B)

At each decoder step, the decoder receives its **own previous prediction** as input.  Errors propagate: a wrong phoneme at position *t* corrupts all subsequent positions.

- **Established result:** Autoregressive decoding is implemented and verified.
- **Use:** behaviorally plausible comparison for Dager/SWP figures.
- **Note:** we did not retrain.  The same checkpoint is used for both regimes.  The goal was to isolate the evaluation-regime effect on fixed model weights.

### Key TF vs AR result

- **Exact-match accuracy** is nearly unchanged between TF and AR (ceiling items remain correct; already-wrong items stay wrong).
- **Edit distance** increases under AR, especially for long pseudowords and unseen forms.
- **Likely interpretation:** AR mostly increases the *severity* of errors (more phonemes wrong per item) rather than the *number* of incorrect items.  Error propagation amplifies small mistakes into longer edit chains.

---

## 3. Length-effect slopes

### Definition

**Length-effect slope** = item-level OLS slope of an error metric regressed on word length (phonemes).

- Positive slope = longer words are harder.
- Two metrics: **error rate** (= 1 − exact match) and **edit distance**.
- Short = {3, 4, 5} phonemes; long = {7, 8, 9} phonemes.

### Results summary

**Train-seen real words:** full and WM routes are essentially at ceiling; no meaningful length effect in either metric under either decoding regime.

**Pseudowords (Dager-comparable):**

| Route | Metric | TF slope | AR slope |
|---|---|---|---|
| Full (gated) | edit distance | ≈ 0.021 | ≈ 0.048 |
| WM (dorsal) | edit distance | ≈ 0.015 | ≈ 0.026 |
| LTM (ventral) | edit distance | ≈ 0.198 | ≈ 0.377 |
| Full (gated) | error rate | ≈ 0.016 | ≈ 0.016 |
| WM (dorsal) | error rate | ≈ 0.012 | ≈ 0.012 |
| LTM (ventral) | error rate | ≈ 0.114 | ≈ 0.114 |

> **Note.** Slope values above are computed from the current audit and should be treated as approximate until verified against the final run outputs.

**Key findings:**
- LTM edit-distance slope for pseudowords (~0.2 TF, ~0.4 AR) is an order of magnitude larger than the WM slope (~0.015 TF, ~0.026 AR).
- AR decoding amplifies the edit-distance slope substantially more than the error-rate slope, for all three routes.
- The WM error-rate slope is modest even under AR decoding, consistent with WM being a robust phonological sequence memory.

---

## 4. WM/dorsal noise sweep

### Setup

Gaussian noise N(0, σ) is added to the WM route encoder hidden state *h* after encoding and before decoding.  This is an evaluation-time perturbation — model weights are not changed.

- **Moderate sweep:** σ ∈ {0.00, 0.01, 0.03, 0.05, 0.10}, 20 repeats per σ > 0.
- **Stress sweep:** σ ∈ {0.00, 0.10, 0.20, 0.50, 1.00}, 5 repeats per σ > 0.
- Seed deterministic.
- Noise applies to the WM route and to the WM component inside the full/gated route.  The LTM encoder is unaffected by WM noise by design.

### Results

**Pseudowords — WM exact-match accuracy vs noise:**

| σ | WM accuracy |
|---|---|
| 0.0 | ≈ 0.980 |
| 0.2 | ≈ 0.928 |
| 0.5 | ≈ 0.378 |
| 1.0 | ≈ 0.024 |

**Pseudowords — full/gated exact-match accuracy vs noise:**

| σ | Full accuracy |
|---|---|
| 0.0 | ≈ 0.973 |
| 0.2 | ≈ 0.955 |
| 0.5 | ≈ 0.778 |
| 1.0 | ≈ 0.265 |

> **Note.** LTM accuracy is unchanged across all σ, as expected.

**Established results:**
- The WM noise mechanism works: WM accuracy degrades monotonically with σ.
- Full/gated degrades more slowly than isolated WM at every noise level.

**Likely interpretation:**
- σ = 0.10: small effect; within the plausible noise range for normal cognition.
- σ = 0.20: WM shows meaningful degradation while still above chance; the most informative regime for auditing length effects.
- σ = 0.50, 1.00: strong degradation; these are stress tests, out-of-distribution relative to training.

**Caveat:** the model was trained with σ defined by `interference_noise` in the config.  Applying a different σ at evaluation is an extrapolation.  High-σ results describe what the model does under extrapolated conditions, not its trained operating regime.

---

## 5. Why full/gated is more robust than WM

### Gate formula

```
g = sigmoid(4 · (confidence − 0.5))
```

where `confidence = max cosine similarity of s_hat to the frozen semantic bank`.

- `g → 1`: LTM dominates (lexical route wins).
- `g → 0`: WM dominates (phonological buffer wins).
- **The gate has no learnable parameters.**

### Why the gate does not dynamically adapt to WM noise

**Established result:** the gate value `g` depends only on LTM confidence, which depends only on the LTM encoder.  Since the LTM encoder is not affected by WM noise, `g` is **invariant to σ**.  The gate is not detecting a noisy WM signal and compensating in real time.

### Mechanism of full robustness

**Established result:** the full/gated route outperforms isolated WM under noise at every noise level tested.

**Likely interpretation:**
- For **train-seen real words**: LTM confidence is high → g ≈ 1 → full ≈ LTM output.  Full robustness here is trivially explained: WM noise barely reaches the output.
- For **pseudowords**: LTM confidence is low → g < 0.5 → WM dominates, but g > 0.  Even a partial LTM contribution acts as a stable anchor.  The blended premotor signal `g·ltm + (1−g)·wm` is more stable than `wm` alone when WM is degraded.

**Audit evidence:**
- The full − WM accuracy gap rises with σ (figure: `full_vs_wm_accuracy_gap.png`).
- The rate of "full correct, WM wrong" rescue events rises with σ (figure: `route_rescue_vs_noise.png`).
- Rescue is stronger for short items than long items — longer pseudowords are harder to anchor even with partial LTM support.
- At σ = 1.0 both routes collapse for long pseudowords, reflecting the limits of partial LTM support.

**Caveat:** word-level rescue statistics can mask position-level complementarity.  A "rescued" item may be correct at different phoneme positions than the WM route would have been.  This would require aligned position-level analysis to disentangle.

---

## 6. Why LTM is more length-sensitive than WM

This is the most conceptually important finding.

### WM route (dorsal / phonological buffer)

The WM encoder is a unidirectional GRU with `pack_padded_sequence`.  It reads the entire phoneme sequence into a single bounded hidden state, then decodes step by step.  The capacity limit is architectural (fixed hidden size) but the route is a direct phonological sequence memory.

### LTM route (ventral / lexical-semantic)

The LTM encoder is a biGRU with masked mean-pooling.  It maps the phoneme input to a 300-d semantic vector `s_hat` aligned to GloVe space.  The decoder then initialises its hidden state from `s_hat` and regenerates the phoneme sequence.

**The key constraint:** the entire phonological form must pass through a single 300-d vector aligned to real-word GloVe embeddings.

### Why this creates length sensitivity for pseudowords

**Established result:** LTM edit-distance slope for pseudowords is ~10× larger than WM.  LTM position-wise accuracy declines across later phoneme positions.

**Likely interpretation:**
1. **Semantic bottleneck:** for real words, `s_hat` lands near a true lexical neighbor → the decoder has a reliable starting point.  For pseudowords, there is no true semantic target; `s_hat` lands in a diffuse region of the semantic bank near its closest (wrong) real-word neighbor.
2. **Lexicalization:** the LTM route may confidently reconstruct the phonology of a known word that resembles the pseudoword input, rather than the pseudoword itself.  The audit shows items with high LTM confidence but high edit distance — consistent with this.
3. **Length:** longer pseudowords carry more phonological structure that the bottleneck must compress.  With more detail to encode into a fixed 300-d vector, reconstruction degrades further from the true target.

**Audit evidence:**
- LTM edit distance rises sharply with pseudoword length; WM remains stable (figure: `ltm_vs_wm_edit_by_length.png`).
- LTM position-wise accuracy drops across later positions for pseudowords (figure: `ltm_position_accuracy_pseudowords.png`).
- LTM failures on long pseudowords include both substitutions and deletions, not simple uniform degradation (figure: `ltm_error_types_by_length.png`).
- LTM confidence for pseudowords clusters near the gate threshold (~0.5) and is less stable with length (figure: `ltm_confidence_vs_length.png`).

**Technical caveat — biGRU padding sensitivity:**  
The LTM biGRU does not use `pack_padded_sequence`.  In a padded batch, the backward direction starts from the rightmost PAD position and accumulates hidden state from zero-input steps.  This means `s_hat` shifts with the batch's maximum sequence length, independent of the item's actual length.  This is a known artifact.  A retrained architecture should add `pack_padded_sequence` to `LTMLexicon.encode()`.  The current checkpoint cannot be silently patched without retraining.

**Clarification — GloVe at inference time:**  
GloVe is NOT used as an input at WFE inference time.  The LTM encoder produces `s_hat`, aligned to the GloVe semantic space.  The semantic bank is used to compute lexical confidence / gate value, while `s_hat` is used to initialise the LTM decoder state `h0 = tanh(W · s_hat)`.  Pseudowords do not have GloVe vectors; the model maps their phoneme inputs to the nearest training-lexicon semantic neighbor.

---

## 7. Relation to Ueno / Dager conceptual framing

### What dual-route theory predicts

In the Dager/SWP framework, the lexical/LTM route supports familiar words, while the sublexical/WM route handles novel forms.  The WM route is expected to show length effects due to limited serial recall capacity.  The LTM route is expected to be length-invariant for familiar words.

In Ueno / Lichtheim2, the value of dual pathways is that the semantic/ventral route and the systematic phonological/articulatory route solve different mappings.  Ueno's result that a ventral-only control fails for nonword repetition supports the claim that a semantic pathway alone is insufficient for novel phonological generalization.

### Where the current model aligns

**Consistent with theory:**
- WM (dorsal) is more robust than LTM (ventral) for pseudowords: pseudowords have no true semantic target, so WM's direct phonological memory is more appropriate.
- LTM fragility on long pseudowords is not unexpected: the semantic bottleneck is not designed for phonological generalization.
- Gate successfully routes real words toward LTM and pseudowords toward WM.

### Where the current model is limited

**Established limitation:** the current "semantic" mechanism is an auxiliary GloVe alignment (static bank lookup), not a full comprehension/naming system.  The model is repetition-only.

- It cannot faithfully reproduce Ueno naming or comprehension figures.
- Any use of LTM output as a proxy for comprehension or naming is currently diagnostic only.
- The GloVe semantic bank provides a stable anchoring signal for real-word repetition, but it does not implement semantic understanding in the cognitive sense.

**Therefore:** finding that LTM is length-sensitive for pseudowords is not necessarily inconsistent with dual-route theory.  It reflects the architectural choice of using a semantic bottleneck as the ventral route.  A more faithful Ueno-like implementation would require a separate comprehension head and a naming pathway that uses semantic representations as input, not as an incidental alignment target.

---

## 8. Figures to show Yair

| Figure path | What it shows | Message | Caveat |
|---|---|---|---|
| `outputs/figures_tf_vs_ar/wfe_tf_vs_ar_dager_strict_combined_4panel.png` | TF vs AR, train-seen real vs pseudowords, full route; error rate and edit distance by length | TF and AR have similar exact-match; AR increases edit distance for long pseudowords | TF remains useful for ceiling/debug; AR is the behavioral comparison |
| `outputs/length_effect_analysis/length_effect_slopes_edit_dist.png` | Edit-distance length-effect slopes for full, WM, LTM by group and decode regime | Length effect is quantified; LTM is much more length-sensitive than WM for pseudowords | Slope values are point estimates with bootstrap CI; train-seen real is at ceiling |
| `outputs/length_effect_analysis/ar_minus_tf_delta_slopes.png` | AR − TF delta slopes for edit distance and error rate | AR mainly increases edit-distance length effect, not binary error rate; error propagation amplifies per-phoneme errors | Delta is at the no-noise baseline; noise interacts separately |
| `outputs/wm_noise_stress_wfe/wm_noise_global_accuracy.png` | Exact-match accuracy vs noise level; train-seen real and pseudowords; full and WM routes | WM noise mechanism works; full/gated degrades more slowly; LTM is flat | σ ≥ 0.5 is out of distribution |
| `outputs/wm_noise_stress_wfe/wm_noise_length_slope_edit_dist.png` | Edit-distance length-effect slope vs noise level; pseudowords and unseen forms; all routes | WM noise amplifies WM length effect; σ = 0.2 is informative; σ = 0.5/1.0 are stress tests | Each point is a mean across 5 repeats for the stress sweep; 1 SD bands shown. The moderate sweep used 20 repeats. |
| `outputs/route_ltm_audit/full_vs_wm_accuracy_gap.png` | Full accuracy − WM accuracy vs noise level; all groups | Full/gated advantage over isolated WM grows with noise | Advantage is a compound effect of gate and LTM stability; not pure LTM rescue |
| `outputs/route_ltm_audit/route_rescue_vs_noise.png` | Rate of "full correct, WM wrong" vs noise level; 3 groups | As WM degrades under noise, full increasingly rescues WM failures | Word-level rescue can hide position-level complementarity |
| `outputs/route_ltm_audit/gate_by_noise_group_length.png` | Gate value (mean ± SD) by group and length bin | Gate depends on LTM confidence, not WM noise; it is not dynamically compensating for noisy WM | Gate is the same value regardless of σ — routing strategy is frozen across the noise sweep |
| `outputs/route_ltm_audit/ltm_vs_wm_edit_by_length.png` | Mean edit distance by word length; full, WM, LTM; train-seen real and pseudowords | LTM edit distance rises sharply with pseudoword length; WM remains stable | No-noise, AR decoding only; includes biGRU padding artifact caveat |
| `outputs/route_ltm_audit/ltm_position_accuracy_pseudowords.png` | Position-wise phoneme accuracy (proportion correct at each position); full, WM, LTM; pseudowords | LTM accuracy declines across later phoneme positions, explaining its length sensitivity | Position alignment is by absolute position, not relative; longer items contribute fewer data points at later positions |
| `outputs/route_ltm_audit/ltm_confidence_vs_length.png` | LTM lexical confidence (max cosine sim to bank) by word length; train-seen real and pseudowords | Train-seen real words have high stable confidence; pseudowords hover near the gate threshold and become less stable with length | Confidence is proximity to the nearest real-word neighbor, not to the correct semantic target for pseudowords |
| `outputs/route_ltm_audit/ltm_confidence_vs_edit_distance.png` | Scatter: LTM confidence vs LTM edit distance; all groups | Some high-confidence items still have high edit distance — consistent with lexicalization toward a wrong neighbor | This is an associative observation; direct evidence of the specific wrong neighbor would require inspecting `sims` |
| `outputs/route_ltm_audit/ltm_error_types_by_length.png` | Sub/ins/del breakdown for LTM, pseudowords, short vs long | LTM failures on long pseudowords include both substitutions and deletions — not simple truncation | Error type decomposition depends on Levenshtein alignment, which is not unique when multiple alignments have the same cost |

---

## 9. Suggested presentation narrative

### Oral flow

**1. Start from the ceiling.**  
"The scale-up requirement is met: full/gated exact match reaches 1.0000 on the trained split of 25,136 GloVe-covered words."

**2. Clarify evaluation.**  
"I added autoregressive decoding alongside teacher-forced so we can compare to Dager/SWP conditions where repetition is free recall.  Importantly, this changes how we interpret the outputs — not the model."

**3. Main WFE result.**  
"Under autoregressive decoding, exact-match barely changes.  But edit distance increases for long pseudowords and unseen forms.  AR doesn't create new failures; it makes existing failures worse, because errors propagate."

**4. Length effect.**  
"The dorsal WM route has a modest length effect on pseudowords.  But the LTM ventral route has a length effect that is an order of magnitude larger.  That's unexpected if we think of LTM as the stable, lexically-anchored route."

**5. Noise.**  
"Adding WM noise confirms that the dorsal route can show stronger length sensitivity when degraded.  At σ = 0.2, WM degrades meaningfully while still making sense.  At σ = 0.5 and 1.0, we're in stress-test territory."

**6. Full robustness.**  
"Full/gated is not just WM.  When WM is noisy, the LTM component acts as a stable anchor.  Crucially, the gate doesn't know WM is noisy — it's routing based on LTM confidence, which is unchanged.  The robustness is structural, not adaptive."

**7. LTM puzzle.**  
"LTM is more length-sensitive for pseudowords because it passes everything through a semantic bottleneck.  For real words, that bottleneck is anchored by a real lexical neighbor.  For pseudowords, the nearest neighbor is the wrong word.  The longer the pseudoword, the more phonological detail gets lost in that bottleneck."

**8. Architecture implication.**  
"The current semantic route is useful for gating and for real-word repetition, but it's not yet a rigorous Ueno-like semantic pathway.  We need to be clear about what GloVe is doing: it's a lexical anchor, not semantic comprehension."

---

## 10. Proposed next changes / experiments

### Immediate — no retraining required

- **Gate override evaluation:** run the full route with fixed g ∈ {0, 0.25, 0.5, 0.75, 1.0} to quantify how much LTM contribution helps or hurts pseudoword repetition at each fixed gate level.
- **LTM nearest-neighbor inspection:** for high-confidence / high-edit-distance pseudowords, extract the top-1 semantic bank match and compare its phonology to the pseudoword.  Direct test of the lexicalization hypothesis.
- **Semantic confidence calibration:** plot LTM confidence distributions for each group and each length bin.  Check whether pseudoword confidence is systematically lower for long items.
- **Expose gate value in full forward pass:** add gate value collection to the standard WFE evaluation pipeline so gate values can be aligned to per-item predictions without rerunning a separate encoder pass.

### Later — retraining or architecture changes required

- **Fix LTM biGRU padding sensitivity:** add `pack_padded_sequence` to `LTMLexicon.encode()` and retrain.  This removes the confound between batch max length and `s_hat`.  The current checkpoint cannot be silently patched.
- **Retrain final model on full 29,571-word lexicon** after hyperparameters are fixed and architecture decisions are settled.
- **Separate lexical confidence from semantic comprehension:** the current gate uses a proxy (cosine proximity to GloVe bank) as the "semantic" signal.  A cleaner architecture would have a dedicated comprehension head that takes an explicit semantic input and could be trained separately.
- **Add explicit naming pathway:** if we want to reproduce Ueno naming/comprehension figures, the model needs a semantic-input → phonological-output pathway, distinct from the current bank-lookup gate.
- **Reconsider GloVe's role:** decide explicitly whether GloVe is:
  - alignment target only (train, then discard at inference);
  - semantic bank for gating only (current);
  - true semantic input for lexical representation;
  - or replaced by a more controlled semantic representation.
- **Make gate noise-sensitive:** the current gate is purely a function of LTM confidence.  A noise-adaptive gate would incorporate a WM reliability signal, allowing the model to down-weight a degraded WM route.  This requires retraining.
- **Consider a more lexicon-like LTM route** for familiar-word repetition, distinct from a conceptual semantic route, to align more closely with Ueno's dual-pathway interpretation.

---

## 11. Short conclusion

The current Lichtheim3 model at the 30 k scale-up checkpoint has a strong experimental audit trail.  It reaches the repetition ceiling and shows a plausible broad division between WM/dorsal and LTM/ventral routes.

The detailed audit reveals:

- **WM is very robust** — modest length effect, graceful degradation under noise, noise-resistant for short pseudowords.
- **LTM is fragile for long pseudowords** — strong length sensitivity, driven by the semantic bottleneck and likely lexicalization toward wrong real-word neighbors.
- **Full/gated partly compensates WM noise via a stable LTM anchor** — but the gate routing is not adaptive; compensation is structural, not dynamic.
- **The current "semantic" mechanism is not yet a full Ueno-style pathway** — GloVe provides lexical anchoring and gating, but not semantic comprehension or naming.

These findings should guide the next clean Lichtheim3 design: fix the biGRU padding artifact, clarify GloVe's role, separate lexical confidence from semantic comprehension, and consider a noise-sensitive gate for a more behaviorally plausible architecture.

---

*Document last updated: 2026-06.  Contact: lichtheim3 repository.*
