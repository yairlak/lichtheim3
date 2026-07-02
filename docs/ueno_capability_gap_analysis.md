# Ueno et al. Capability Gap Analysis

> Checkpoint: `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt`
> Architecture: DualRouteModel (wm_route.py + ltm_route.py + gating.py + motor.py)
> Lexicon: 29,571 real English words with GloVe semantics
>
> This document answers which Ueno et al. capabilities are present, absent, or
> proximate in the current lichtheim3 repo, without proposing architecture changes.

---

## 1. Is naming implemented?

**No.**

Naming (spoken word production from a picture / semantic concept) requires:
1. A semantic concept as INPUT (e.g. a GloVe vector, a picture embedding).
2. A phonological form as OUTPUT, generated autoregressively from that input.
3. A naming loss to train the model on this mapping.

None of these exist in the current pipeline. The model's ONLY task during training
and evaluation is **phonological repetition**: phoneme sequence in → phoneme sequence
out, with a semantic bottleneck in the LTM route.

**Architecturally proximate situation**: `LTMLexicon.decode(s_hat, dec_in)` can produce
phoneme logits from any 300-d semantic vector. If one bypasses the LTM encoder and
feeds a GloVe vector directly as `s_hat`, the decoder will produce a phoneme sequence.
However:
- The model was NOT trained with a naming loss; it learned to *reconstruct the input word's form*,
  not to map arbitrary semantic vectors to their closest word's form.
- `dec_in` is teacher-forced from the gold phonological sequence; free autoregressive decoding
  (which naming requires) is not implemented.
- There is no evaluation harness for naming accuracy.

**Verdict**: naming is absent from training and evaluation. A `scripts/run_naming_eval.py`
script could test whether the LTM decoder produces reasonable forms given gold GloVe inputs
(a partial proxy), but this would not constitute trained naming ability.

---

## 2. Is comprehension implemented?

**No.**

Comprehension (spoken word recognition / understanding) requires the model to map a
phonological input to a semantic decision — either a similarity judgment, a category
match, or a semantic-space nearest-neighbour output.

In the current model:
- `s_hat` (the LTM route's encoded semantic vector) is produced at every forward pass.
- `s_hat` is supervised via `L_align` to approach the word's GloVe vector.
- But `s_hat` is never used to make a comprehension decision; it only enters the gate's
  confidence estimate (max cosine similarity to the semantic bank).

**What would be needed**: a comprehension head that takes `s_hat` and outputs a
semantic decision (e.g., a softmax over a concept vocabulary, or cosine similarity to
a probe concept). This would also need comprehension training data and a comprehension loss.

**Verdict**: comprehension is not implemented. `s_hat` is an alignment target, not an
output. A nearest-neighbour retrieval from `s_hat` into the semantic bank could serve as
a rough proxy (like a "semantic identification" score), but it was not trained for
comprehension and would not be a faithful Ueno replication.

---

## 3. Is semantic-to-phonological production implemented?

**Partially, structurally only.**

The LTM route contains: `phoneme-sequence → s_hat (300-d) → phoneme-sequence`.
The second half (`s_hat → phoneme-sequence`) IS a semantic-to-phonological mapping.
It is trained implicitly via `L_dec` (ventral form reconstruction loss).

However:
- The model was trained in a *reconstruction* regime, not a *naming* regime. It learns
  to reproduce the word it just heard, using semantics as an intermediate code. It does
  NOT learn to produce the word for a semantic input it has never heard phonologically.
- Autoregressive (free) decoding from `s_hat` is not implemented; all decoding is teacher-forced.
- Without a naming loss, the decoder has no gradient signal for producing words it cannot
  reconstruct from a phonological input.

**Verdict**: the architecture supports semantic-to-phonological decoding in principle.
A new eval script could test this by feeding gold GloVe vectors to the LTM decoder with
greedy decoding — this could reveal whether the decoder has implicitly learned a useful
semantic-to-phonological map. It would NOT constitute a trained naming task.

---

## 4. What role does the GloVe vector play?

GloVe vectors serve exactly ONE role in the current model: **alignment target** for the
LTM encoder's `s_hat` output, via the `L_align` loss:

```
L_align = (1 - cosine_similarity(s_hat, glove_target)) + 0.1 * MSE(s_hat, glove_target)
```

GloVe is used:
- During training: as the target for `L_align`.
- During model setup: to populate `semantic_bank` (the frozen matrix of GloVe vectors
  used by the gate to compute `confidence = max cosine sim of s_hat to bank`).
- Not at inference time as a model INPUT.

GloVe is NOT:
- A forward-pass input (phoneme-to-phoneme is the only input→output path).
- A comprehension target.
- An output of the model.
- Available as a concept-input for naming.

---

## 5. Can the LTM route be used for naming without architecture changes?

**Possibly, as a diagnostic proxy, but not as a trained naming task.**

The steps would be:
1. Take a target GloVe vector `g` (e.g. the GloVe embedding for "cat").
2. Skip the LTM encoder; use `g` directly as `s_hat`.
3. Run `ltm.decode(g, dec_in)` with greedy autoregressive decoding
   (i.e., feed model's own previous output as `dec_in` at each step).
4. The output is the model's attempt to produce the phonological form of "cat".

This does NOT require architecture changes. It does require:
- A new eval script with greedy decoding (the current `make_batch` / `route_predictions`
  path is teacher-forced only).
- Deciding what GloVe vector to use as input (gold GloVe, centroid of a semantic category, etc.).

**Expected result**: the LTM decoder was trained to reconstruct forms it received as
input — so it has learned a `phoneme → s_hat → phoneme` map. When fed a GloVe vector
directly, the decoder may produce the correct word's form if `g ≈ s_hat` for that word
(which is what `L_align` encourages). For high-frequency words with tight alignment,
this proxy may be surprisingly accurate. For low-frequency or abstract words, quality
is uncertain.

**Verdict**: worth testing as a diagnostic script. Should NOT be called "naming" in
publications without explicit validation.

---

## 6. Which Ueno figures are impossible without adding new tasks?

The following Ueno figures cannot be generated from the current codebase:

| Ueno Figure | Why impossible |
|---|---|
| Fig 2 — comprehension learning curve | No comprehension task or loss |
| Fig 2 — naming learning curve | No naming task or loss |
| Fig 4 — recovery after dorsal lesion | No lesion-then-retrain protocol |
| Fig 6 — semantic naming errors by lesion | No naming; no semantic category labels |
| Any figure involving comprehension scores | No comprehension decision head |

---

## 7. Which Ueno-inspired figures are possible now?

The following Ueno-style analyses can be generated from the current codebase
without any architecture changes:

| Analysis | Method | Notes |
|---|---|---|
| Repetition learning curve | Plot `train_rep` / `val_rep` from checkpoint `history` | Already implemented in `train.py::plot_loss_history` |
| Trained-word vs held-out vs pseudoword repetition | ceiling eval on train / val splits + WFE pseudowords | `--include_val` flag; val split = held-out real words |
| Dorsal-only repetition profile | `route_logits("wm")` — WM route in isolation | Clean route isolation; not a biological lesion |
| Ventral-only repetition profile (Ueno Fig 7) | `route_logits("ltm")` — LTM route in isolation | Same |
| Double dissociation: real-word vs nonword | Compare full/WM/LTM accuracy on real vs pseudo items | Already computed in external_eval.py |
| Lexicality × frequency × length effects | WFE condition breakdown | Already computed; needs re-run with 30k checkpoint |
| Serial-position accuracy curve | `per_position_correct` aggregated by relative position | Requires new `plot_position_errors.py` |
| Representational similarity (WM space) | Extract WM encoder state `h` via `collect=True`; RSA vs phoneme edit distance | Requires new `run_rsa_analysis.py`; WM RSA only (no layerwise hierarchy) |
| Representational similarity (LTM space) | Extract `s_hat` via `collect=True`; RSA vs GloVe cosine similarity | Same script; tests whether `s_hat` organises semantically |
| Parametric route ablation (gate override) | Multiply WM or LTM premotor by a scalar ∈ [0,1] before gating | Requires new eval script; not the same as biological lesion depth |
| Naming proxy (diagnostic only) | Feed gold GloVe → LTM decoder greedy decoding | Not trained naming; labelled as proxy |

---

## 8. Summary verdict

| Capability | Status |
|---|---|
| Repetition (trained words) | Ceiling 1.000 ✓ |
| Repetition (held-out words) | Available via val split |
| Repetition (nonwords) | Available via WFE pseudowords |
| Comprehension | NOT implemented |
| Naming (trained) | NOT implemented |
| Naming (proxy via LTM decoder) | Architecturally possible; not yet tested |
| Semantic-to-phonological (trained) | NOT implemented |
| Route isolation (dorsal / ventral) | READY via route_logits |
| Parametric route damage | ADAPT (new eval script) |
| Recovery after lesion | NOT implemented |
| Serial-position curve | ADAPT (new plot script) |
| RSA on internal representations | ADAPT (new analysis script) |

**Do not claim Ueno replication unless comprehension and naming are implemented.**
The current repo supports Ueno-style *repetition and route-dissociation* analyses only.
