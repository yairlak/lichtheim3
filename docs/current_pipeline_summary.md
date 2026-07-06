# Lichtheim3: Current Pipeline Summary

Reference checkpoint: `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt`

---

## 1. Training Lexicon

| Field | Value |
|---|---|
| Source | CMU Pronouncing Dictionary filtered to GloVe-300 coverage |
| File | `data/lexicon_en_glove_covered.tsv` |
| Total words | 29,571 |
| Train split | 25,136 words (85 %) |
| Validation split | 4,435 words (15 %, held out throughout training) |
| Split seed | fixed (reproducible train/val partition) |

The split is fixed and deterministic. The validation split is never seen during
training and is not used to select hyperparameters — it exists to check
generalisation to untrained real words.

---

## 2. Repetition Task

The model is trained on a **phoneme-to-same-phoneme repetition task**:

```
input:   R EH R L IY   ("rarely")
output:  R EH R L IY
```

At inference, the model receives only the phoneme sequence. It does **not** receive
orthography, semantics, or any external cue about whether the sequence is a real word.
Lexical status is inferred implicitly by the LTM (ventral) route from its internal
encoding of phoneme patterns seen during training.

---

## 3. GloVe Role

GloVe 300-dimensional embeddings are used **only as an alignment target** for the LTM
encoder during training. They are **not an input to the model at inference**:

- Each word in the training lexicon has a GloVe vector.
- A loss term (`L_align`) pushes the LTM encoder's output (`s_hat`) toward the
  word's GloVe vector.
- At inference, GloVe vectors are stored in a frozen **semantic bank** used only by the
  gate's confidence calculation (cosine similarity of `s_hat` to bank entries).
- The model never receives a GloVe vector as input; it reads phonemes and produces
  phonemes.

GloVe is effectively the model's "lexicon" — the source of lexical identity that the
LTM route learns to align to.

---

## 4. Model Architecture

### 4.1 Shared phoneme embedding

A single `nn.Embedding(vocab_size, phon_embed_dim)` layer is shared by both routes.
It maps phoneme token IDs to dense feature vectors. It is trained jointly and
represents general phonetic features, not route-specific representations.

### 4.2 WM / Dorsal route  (`models/wm_route.py`)

The **working-memory / dorsal route** is a parametric recurrent encoder-decoder:

```
phoneme sequence
       ↓
  GRU encoder  (pack_padded_sequence → single hidden state h)
       ↓
  [optional interference noise on h, when training or collect=True]
       ↓
  GRU decoder  (teacher-forced: receives gold prefix at each step)
       ↓
  premotor representation  (B, S, premotor_dim)
```

Properties:
- **Unidirectional GRU encoder** — no padding sensitivity (uses `pack_padded_sequence`).
- **Capacity-limited**: the whole phoneme sequence is compressed into a single bounded
  hidden state `h`. Longer sequences are harder to pack faithfully.
- **Length-effect mechanism**: interference noise on `h` (active during training and
  during `collect=True` evaluation) corrupts the recalled state. Combined with
  capacity limits this produces primacy/recency effects and a length-accuracy gradient.
- **Lexical-frequency invariant**: the WM route learns sublexical phonotactics and
  generalises to novel sequences; it does not benefit from word-frequency weighting.
- **WM noise at inference**: disabled by default (`collect=False`) for deterministic
  evaluation. Enable with `--wm_noise` flag (or `collect=True`) for cognitive/noisy mode.

### 4.3 LTM / Ventral route  (`models/ltm_route.py`)

The **long-term memory / ventral route** is a semantic encoder-decoder:

```
phoneme sequence
       ↓
  biGRU encoder  (masked mean-pool of hidden states → s_hat)
       ↓
  Linear → s_hat  (300-d, aligned to GloVe during training)
       ↓
  cosine similarity to frozen semantic bank  → lexical confidence
       ↓
  GRU decoder  (initial hidden = tanh(Linear(s_hat)))
       ↓
  premotor representation  (B, S, premotor_dim)
```

Properties:
- **Bidirectional GRU encoder** — uses masked mean-pool of hidden states, but does
  **not** use `pack_padded_sequence`. This makes `s_hat` sensitive to the batch
  padding context (see §7 below).
- **Semantic alignment**: `s_hat` is pushed toward the GloVe vector of the input word
  by `L_align` during training, making the LTM route "know" which word it is processing.
- **Lexical activation field**: the cosine similarity of `s_hat` to the frozen semantic
  bank determines how well the current phoneme sequence matches a known word.
- **Frequency-sensitive**: because the frequency-weighted training sampler gives more
  exposure to high-frequency words, the LTM encoder learns them more robustly.
- **Pseudoword failure mode**: novel / out-of-vocabulary phoneme sequences produce a
  low-confidence `s_hat` → gate favours WM route → LTM output mostly irrelevant for
  pseudowords.

### 4.4 Gate  (`models/gating.py`)

The gate combines both routes into a single premotor signal:

```
g = sigmoid( alpha × (confidence − 0.5) )

premotor = g × LTM_premotor + (1 − g) × WM_premotor
```

- `confidence` = max cosine similarity of `s_hat` to the semantic bank (scalar per item).
- `alpha` = 4 (fixed; no learnable parameters in the gate).
- `g → 1`: gate favours LTM (real, known word with high cosine match).
- `g → 0`: gate favours WM (novel sequence, pseudoword, or poor LTM encoding).
- The gate is **not learned** and does not adapt. A wrong LTM prediction on a known
  word cannot be corrected by the gate if the gate's confidence is already high.

### 4.5 Shared output  (`models/motor.py`)

A single `Linear(premotor_dim, vocab_size)` layer maps the gated premotor signal to
phoneme logits. This layer is shared between both routes and the combined output.

---

## 5. Evaluation: Teacher-Forced (Current)

All current evaluations use **teacher-forced decoding**:

```
decoder input at step t:  gold phoneme at t-1  (not the model's prediction)
```

At each position, the model sees the correct preceding phoneme and produces a
distribution over the next phoneme. Accuracy is measured as whether the argmax
matches the gold phoneme.

**Implication**: errors do not propagate. If the model gets phoneme 2 wrong, it
still receives the correct phoneme 2 as input for step 3. Teacher-forced accuracy
is therefore an **upper bound** on free-generation (autoregressive) accuracy.

The train-ceiling result (`full_exact_match = 1.0000` on 25,136 training words)
reflects teacher-forced performance only.

---

## 6. Evaluation: Autoregressive (Implemented)

For comparison with Dager/SWP and for cognitive plausibility, the model can be
evaluated **autoregressively** using `--decode autoregressive`:

```
decoder input at step t:  model's own predicted phoneme at t-1
```

This is the regime used in most behavioural experiments (free recall, repetition
without feedback). In this mode:
- Errors propagate: a wrong phoneme at position 2 corrupts all subsequent positions.
- Length effects are amplified compared to teacher-forced.
- The serial-position curve (primacy/recency) reflects accumulated error, not
  per-position logit quality.

```bash
python scripts/external_eval.py \
    --ckpt checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt \
    --out_dir outputs/external_eval_30k \
    --decode autoregressive --wfe_only
# outputs: outputs/external_eval_30k/wfe_ar/
```

See `docs/evaluation_regimes.md` for the full regime specification.

---

## 7. Deterministic vs Cognitive/Noisy Evaluation

| Mode | `model.eval()` | `collect` | WM noise | LTM s_hat | Deterministic? |
|---|---|---|---|---|---|
| Deterministic (default) | ✓ | False | OFF | padding-sensitive | Yes |
| Cognitive / noisy | ✓ | True | ON | padding-sensitive | No |

Note: `model.eval()` alone does **not** disable WM noise. Noise is active when
`self.training OR collect`. The `collect=False` default in `external_eval.py` is what
makes the current evaluation deterministic.

---

## 8. WFE Real-Word Distinction

The WFE dataset labels stimuli as `lexicality = real` or `lexicality = pseudo`.
However, within the real-word set, lichtheim3 distinguishes:

| `lexicon_category` | Meaning | N (approx) |
|---|---|---|
| `real_word_seen_in_training_lexicon` | In the 25,136-word train split | majority of WFE real |
| `real_word_in_validation_split` | In the 4,435-word val split | some WFE real |
| `real_word_outside_4000_lexicon` | Real word absent from GloVe lexicon | a few WFE real |
| `pseudoword` | Not a real word | all WFE pseudo |

**For Dager/SWP comparison**: use only `real_word_seen_in_training_lexicon` as the
"real" category. Mixing in validation or novel real words inflates the apparent
real-word deficit because those items were never trained.

---

## 9. Current WFE Metrics

From `outputs/external_eval_30k/wfe/metrics.json` (deterministic, teacher-forced,
`collect=False`, `wm_noise=False`):

| Route | Exact-match accuracy |
|---|---|
| Full (gated) | ≈ 0.987 |
| WM (dorsal) | ≈ 0.987 |
| LTM (ventral) | ≈ 0.790 |

The WM and LTM overall accuracies are nearly identical for the full WFE set because
WFE real words (known to the model) dominate and both routes handle them well.
The divergence is more visible when broken down by `lexicon_category`.

---

## 10. "Rarely" Caveat: Padding-Sensitive LTM biGRU

**Confirmed discrepancy** (`scripts/debug_single_item_prediction.py`):
the word "rarely" (R EH R L IY) is in the training lexicon and produces correct
predictions in all ceiling-eval contexts, but produces a wrong LTM (and full/gated)
prediction in some external-eval batches.

| Batch context | WM | LTM | Full/gated |
|---|---|---|---|
| Solo (batch_size=1, no trailing PAD) | ✓ | ✗ | ✗ |
| Padded to WFE max length | ✓ | ✓ | ✓ |
| Actual external-eval batch (64 items) | ✓ | ✗ | ✗ |

**Root cause**: `LTMLexicon.encode()` runs a bidirectional GRU over the full padded
tensor without `pack_padded_sequence`. The backward GRU direction starts at the
rightmost PAD position in the batch and processes left. GRU bias terms produce a
non-zero hidden state from repeated zero-input (PAD) steps, which shifts `s_hat` via
the masked mean-pool. The amount of shift depends on the number of trailing PAD
columns, i.e., on the length of the longest sequence in the same batch.

The WM route is unaffected: it uses a unidirectional GRU with `pack_padded_sequence`.

**Recommended fix (retrained architecture only)**: wrap the biGRU in
`LTMLexicon.encode()` with `pack_padded_sequence` / `pad_packed_sequence`.
Do not apply this to the current checkpoint — it would invalidate the learned weights.

The current effect is small in aggregate (WFE full accuracy ≈ 0.987), but it means
LTM-route results from the external evaluator are **batch-order dependent** for a
subset of short words in long batches.
