# Lichtheim3: Architecture and Evaluation — Meeting Notes

> **Scope:** Current checkpoint only. Future proposals are labeled *Proposed (not implemented)*.
> **Constraints:** Do not retrain. Do not modify architecture, loss, gate, or checkpoint. Do not patch `pack_padded_sequence`. Do not delete old figures. Do not put unseen forms in main figures.

---

## 1. Current checkpoint and lexicon summary

**Checkpoint:** `checkpoints/lichtheim3_30k_glove_e60_to_e120_lowlr.pt`

- Trained on the 30,000 most-frequent English words (ARPABET phoneme sequences).
- Lexicon source: `data/lexicon_en.tsv` (or `data/lexicon_en_glove_covered.tsv` if used at training time — the checkpoint stores the source string in `ckpt["lexicon_source"]`).
- GloVe: 300-dimensional (6B tokens, `data/glove.6B.300d.txt`). GloVe vectors are used as the semantic alignment target during training and as the frozen semantic bank at inference.
- Word length filter: 2–9 phonemes.
- Validation split: 15% (stratified random, `seed=0`).
- Training schedule: AdamW, lr=1e-3 → low-lr continuation (epochs 60 → 120). Resumed from an earlier checkpoint (stored in `ckpt["resumed_from"]`).
- Default `teacher_forcing_ratio = 1.0` during training (fully teacher-forced).
- Default WM interference noise `σ = 0.10` during training.

**Lexicon size (approximate):**

| Split | N items |
|---|---:|
| Training | ~25,500 |
| Validation | ~4,500 |
| Semantic bank `B_lex` | = training words only (frozen at eval) |

These values are read from `ckpt["n_train"]` at load time and are the authoritative source.

---

## 2. Current implemented architecture

> Everything in this section reflects the current checkpoint. Proposed revisions appear only in Section 2.7.

### 2.1 Notation

| Symbol | Meaning |
|---|---|
| x = (x₁, …, xT) | input phoneme ID sequence |
| y<t = (y₁, …, y_{t−1}) | decoder prefix (gold for TF; model output for AR) |
| B, T, S | batch size, encoder length, decoder length |
| E = 64 | shared phoneme embedding dim (`phon_embed_dim`) |
| H\_WM = 128 | WM GRU hidden dim |
| H\_LTM\_enc = 256 | LTM encoder GRU hidden dim (per direction) |
| H\_LTM\_dec = 256 | LTM decoder GRU hidden dim |
| D = 300 | semantic / GloVe dim |
| P = 128 | premotor dim (`dual_route.py:34`) |
| V | vocabulary size (ARPABET phonemes + BOS/EOS/PAD) |
| σ | WM interference noise std (`WMConfig.interference_noise`, default 0.10) |
| α = 4.0 | gate sharpness (`GatingConfig.alpha`) |

### 2.2 Shared phoneme embedding

Both routes share a single embedding matrix (not frozen):

```
E_embed ∈ R^{V × E}
e_t = E_embed[x_t]    ∈ R^E
```

### 2.3 WM / dorsal route (`models/wm_route.py`)

**Encoder** — 1-layer unidirectional GRU with `pack_padded_sequence`:

```
lengths  = sum(enc_mask, dim=1)            # true token counts, padding-immune
h_WM     = GRU_enc_WM( e_{1:T} )          ∈ R^{H_WM}   (final hidden, scalar state)
```

**Noise** — applied to the encoder's final hidden state:

```
if (model.training OR collect=True) AND σ > 0:
    ε ~ N(0, σ²I),    ε ∈ R^{H_WM}
    h_WM_noisy = h_WM + ε
else:
    h_WM_noisy = h_WM                      (deterministic)
```

**Decoder** — 1-layer GRU, initialised from `h_WM_noisy`:

```
d_WM,t = GRU_dec_WM( e_{y<t},  h_WM_noisy )    ∈ R^{B × S × H_WM}
z_WM,t = W_WM  d_WM,t  +  b_WM                  ∈ R^{B × S × P}       (to_premotor)
```

### 2.4 LTM / ventral route (`models/ltm_route.py`)

**Encoder** — 1-layer bidirectional GRU; **no** `pack_padded_sequence`:

```
out_LTM ∈ R^{B × T × 2H_LTM_enc}    ← GRU_enc_LTM( e_{1:T} )
                                        (forward + backward, concatenated at each t)

pooled = sum_t( out_LTM,t × mask_t ) / sum_t( mask_t )    ∈ R^{B × 2H_LTM_enc}
```

⚠ **Known artifact:** the backward GRU pass starts from the rightmost padded position in the batch, not from the last real token. Therefore `pooled` (and consequently `s_hat`) shifts with the batch's maximum sequence length. This is a known consequence of the missing `pack_padded_sequence` call in the LTM encoder. **Do not patch in the current checkpoint.**

**Semantic projection** — 2-layer MLP with GELU:

```
s_hat = Linear(H_LTM_enc, D)( GELU( Linear(2H_LTM_enc, H_LTM_enc)( pooled ) ) )
      ∈ R^{B × D}    (D = 300)
```

Exact shape: `Linear(512, 256) → GELU → Linear(256, 300)` (`ltm_route.py:45–48`).

**Decoder** — 1-layer GRU, initialised from `s_hat`:

```
h0_LTM  = tanh( W_s  s_hat + b_s )              ∈ R^{B × H_LTM_dec}   (sem_to_h0)
d_LTM,t = GRU_dec_LTM( e_{y<t},  h0_LTM )       ∈ R^{B × S × H_LTM_dec}
z_LTM,t = W_LTM  d_LTM,t  +  b_LTM              ∈ R^{B × S × P}       (dec_to_premotor)
```

### 2.5 Semantic bank and confidence (`models/ltm_route.py`)

The bank `B_lex ∈ R^{n_train × D}` holds the frozen, L2-normalised GloVe vectors of all training-lexicon words. It is constructed once at checkpoint load and never updated.

```
q       = s_hat / ||s_hat||_2                         ∈ R^{B × D}
sims    = q  B_lex^T                                  ∈ R^{B × n_train}     (cosine similarities)
c_LTM   = max_i  sims_i                               ∈ R^B                  (top-1 confidence)
margin  = sims_(1) − sims_(2)                         ∈ R^B                  (top-1 minus top-2)
density = sum_i sigmoid( 20 · (sims_i − (c_LTM − 0.10)) )  ∈ R^B          (soft neighbor count)
```

`c_LTM` is the gate's only input. `margin` and `density` are diagnostic fields available in `lexical_field` output; they are not used by the gate or decoder.

### 2.6 Gate and premotor blend (`models/gating.py`)

The gate is a **scalar per item**, constant across all decoder steps:

```
g = sigmoid( α · (c_LTM − 0.5) )     ∈ R^B,    α = 4.0
```

- `g → 1`: high lexical confidence → route toward LTM (real words).
- `g → 0`: low lexical confidence → route toward WM (pseudowords, novel forms).
- **No learnable parameters. No input from the WM route.**

`g` is broadcast to `R^{B × S × 1}` for premotor blending:

```
z_full,t = g · z_LTM,t + (1 − g) · z_WM,t     ∈ R^{B × S × P}
```

### 2.7 Shared Motor Cortex and output (`models/motor.py`, `models/dual_route.py`)

A single shared linear layer projects all three premotor streams to logits:

```
logits_full,t = W_motor  z_full,t     ∈ R^{B × S × V}
logits_WM,t   = W_motor  z_WM,t       ∈ R^{B × S × V}
logits_LTM,t  = W_motor  z_LTM,t      ∈ R^{B × S × V}

p( y_t | y<t, x ) = softmax( logits_full,t )
```

`W_motor ∈ R^{V × P}` is **shared** across all three routes. The full route mixes premotor states before projection — it does not mix logits.

### 2.8 Summary: what is and is not implemented

| Component | Status |
|---|---|
| Shared phoneme embedding | **Implemented** |
| WM encoder with `pack_padded_sequence` | **Implemented** |
| WM noise on encoder final state | **Implemented** |
| LTM biGRU encoder (no pack) | **Implemented** — backward-pass artifact (see §2.4) |
| LTM semantic MLP (`s_hat`) | **Implemented** |
| Frozen GloVe semantic bank | **Implemented** |
| Parameter-free gate (c_LTM only) | **Implemented** |
| Premotor blending | **Implemented** |
| Shared Motor Cortex | **Implemented** |
| Learnable gate (β terms) | *Proposed — not implemented, requires retraining* |
| WM reliability input to gate | *Proposed — not implemented* |
| Comprehension/naming pathway | *Proposed — not implemented* |

---

## 3. Minimal equations (reference card)

**WM encoding:**
```
h_WM = GRU_enc_WM( E_embed[x_{1:T}] )    ∈ R^{H_WM}
```

**WM noise:**
```
h_WM_noisy = h_WM + ε,    ε ~ N(0, σ²I)    [training or collect=True]
           = h_WM                             [eval, collect=False]
```

**LTM encoding:**
```
pooled = masked_mean_pool( GRU_enc_LTM( E_embed[x_{1:T}] ) )    ∈ R^{2H_LTM_enc}
s_hat  = Linear_2( GELU( Linear_1( pooled ) ) )                   ∈ R^D
```

**Semantic bank confidence:**
```
c_LTM = max_i  cosine( s_hat, B_lex[i] )    ∈ [−1, 1]
```

**Gate:**
```
g = sigmoid( 4.0 · (c_LTM − 0.5) )    ∈ (0, 1)
```

**Premotor blend:**
```
z_full,t = g · z_LTM,t + (1 − g) · z_WM,t
```

**Shared motor softmax:**
```
p( y_t | y<t, x ) = softmax( W_motor  z_full,t )
```

**Teacher-forced decoding (TF):**
```
dec_in_t = y_{t−1}  (gold token)    for all t
```
Error cannot propagate across steps. Used as ceiling/debug probe only.

**Autoregressive decoding (AR):**
```
dec_in_t = argmax p( y_{t−1} | y<{t−1}, x )    (model's own previous output)
```
Errors propagate. This is the behavioral evaluation regime.

---

## 4. Noise: complete specification

### Where noise is applied

Noise is applied to the **WM encoder final hidden state `h_WM`** only, after the `pack_padded_sequence`-based GRU forward pass and before the WM decoder is initialised (`wm_route.py:53–54`).

Noise is **not** applied to:
- Phoneme embeddings
- LTM encoder, pooling, or `s_hat`
- Semantic bank `B_lex`
- Gate value `g`
- LTM decoder hidden state
- Motor Cortex weights

### When noise is active

| Context | `collect` flag | σ active? |
|---|---|---|
| Training (all routes) | `collect=False` by default; noise fires via `model.training=True` | **Yes** |
| WFE / SSP noise sweep (`external_eval.py`) | `collect=True` is passed explicitly | **Yes** |
| Train lexicon ceiling TF eval (default) | `collect=False` | **No** (deterministic) |
| Train lexicon ceiling TF eval with `--wm_noise` | `collect=True` for WM-isolated route only | Yes for WM route; No for WM inside full |
| Autoregressive eval (`--decode autoregressive`) | same as above | same as above |

`collect=True` applies noise to the **WM-isolated route only**. The WM component inside the full/gated route does not receive independent noise via `--wm_noise` — it shares the forward pass with LTM and therefore does not call the WM noise path separately.

### σ meaning

σ is the standard deviation of the additive Gaussian noise vector `ε ∈ R^{H_WM}`. A larger σ degrades the WM encoder's representation. Default `σ = 0.10` (`WMConfig.interference_noise`). In WFE noise sweeps, σ typically ranges from 0.0 to ≥1.0.

### Gate invariance to noise

The gate `g = sigmoid(4·(c_LTM − 0.5))` depends only on `c_LTM`, which depends only on the LTM route. Changing σ does not change `g`. Gate values for pseudowords are constant across all noise levels. Full-route noise robustness is structural (passive LTM anchor), not adaptive.

---

## 5. Evaluation policy

### Teacher-forced (TF) — ceiling / debug only

- Use: verify training convergence; detect training failures; debug architectural changes.
- Success criterion: `full_exact_match = 1.0000` on the training split, `train_errors.tsv` empty.
- **Do not report TF numbers as behavioral results.** TF accuracy does not measure free-generation performance; errors cannot propagate.
- Script: `scripts/evaluate_train_lexicon_ceiling.py` (default mode).

### Autoregressive (AR) — behavioral main regime

- Use: all figures intended to represent model behavior at a behavioral level.
- WFE figures use AR decoding.
- Error can propagate across decoder steps; performance is lower than TF ceiling. This is expected and correct.
- Script: `scripts/evaluate_train_lexicon_ceiling.py --decode autoregressive` (train/val lexicon) or `scripts/external_eval.py` (WFE / SSP pseudowords).

---

## 6. Figure policy for WFE evaluation

### Which item types appear in main figures

Main WFE figures must include **real words and pseudowords only**. Unseen real words (held-out validation words) must not appear in main figures for the current checkpoint. They can appear in supplementary or diagnostic figures with explicit labeling.

### X-axis

Use **Word length (phonemes)** as the primary x-axis variable. Do not use syllables or orthographic length.

### Required metrics (AR decoding)

| Metric | Definition |
|---|---|
| Exact match | `pred == target` (whole sequence, binary) |
| Error rate | `1 − exact_match` |
| Edit distance | Levenshtein distance (phoneme-level) between prediction and target |
| Normalized edit distance | `edit_distance / max(1, len(target))` |

Report all four. Normalized edit distance controls for length and allows cross-length comparison.

### Error bars

Error bars must be defined explicitly in every figure caption:
- **Preferred:** 95% bootstrap confidence interval over items (resampling within the length × item-type cell).
- **Alternative:** ±1 standard error of the mean across items in the cell.
- State which is used. Do not omit error bars for small-N cells; widen them or collapse bins instead.

---

## 7. Rarely anomaly: identical transcription, WM correct, LTM wrong, full follows LTM

### Observed pattern

On a small subset of items (predominantly pseudowords), the following dissociation is observed:

- The **transcription** presented to the model is identical to a training item (same phoneme sequence).
- The **WM-isolated route** produces the correct output.
- The **LTM-isolated route** produces an incorrect output (typically a real word that is a neighbor in GloVe space — a lexicalization error).
- The **full/gated route** follows the LTM output, not WM — even though WM is correct.

### Likely cause: LTM backward-pass padding artifact

The LTM biGRU encoder does not use `pack_padded_sequence`. The backward pass starts from the rightmost padded position in the batch, not the last real token. When a pseudoword is batched with longer items, the backward hidden state at position 1 traverses additional PAD tokens before reaching the form. This shifts `pooled` and therefore `s_hat` relative to what it would be for an isolated item.

Consequence: `s_hat` for pseudowords is noisy and shifts with batch composition. The confidence `c_LTM` can land above 0.5 by accident, driving `g > 0.5` and causing the full route to weight LTM more heavily than WM — even when WM is correct.

### What this is not

This is not a training-set contamination issue. The transcription identity is incidental (the pseudoword happens to share a sequence with a trained form), and the WM route handles it correctly. The failure is in the LTM pooling upstream of the gate.

### Do not patch the current checkpoint

Fixing this would require either:
1. Adding `pack_padded_sequence` to the LTM encoder and retraining, or
2. Sorting batches by length (same effect for the backward pass) and retraining.

Either change alters the model weights. **Do not apply any fix to the current checkpoint.** Document this anomaly and carry it forward as a known artifact when interpreting full/gated route LTM-pull effects on pseudowords.

---

## 8. Open checks before interpretation

The following checks must be completed before any behavioral claim is published or presented:

1. **Confirm checkpoint provenance.** Run `torch.load(ckpt, weights_only=False)` and print `ckpt["lexicon_source"]`, `ckpt["n_train"]`, `ckpt["resumed_from"]`. Confirm GloVe present (`ckpt["glove_present"]`).

2. **Confirm ceiling is met.** Run `evaluate_train_lexicon_ceiling.py` in TF mode and verify `full_exact_match = 1.0000` on the training split, `train_errors.tsv` empty. If ceiling is not met, behavioral AR comparisons are uninterpretable.

3. **Confirm AR eval runs deterministically (collect=False).** Run `--decode autoregressive` twice on the same split and verify identical outputs. If noise is accidentally enabled, results will differ.

4. **Confirm semantic bank is built from training words only.** Check that `bank = torch.stack([e.semantic for e in train_entries])`, not `all_entries`. Validation words must not appear in `B_lex`.

5. **Confirm WFE items are not in training lexicon.** Pseudowords and WFE stimuli must not overlap with `train_entries` on the word string. Check `set(wfe_words) ∩ set(e.word for e in train_entries)`.

6. **Check gate value distribution on WFE items.** Plot the histogram of `g` for real words and pseudowords separately. Verify that real words cluster near `g ≈ 1` and pseudowords near `g < 0.5`. If pseudowords have `g > 0.5` at baseline noise, the padding artifact (§7) is inflating LTM confidence.

7. **Document noise level used in each figure.** Every WFE figure must state whether σ = 0 (deterministic WM) or σ > 0, and whether `collect=True` was passed. Never mix noise conditions within a figure without explicit labeling.

8. **Verify no unseen forms in main figures.** Check that main figure item lists come from pseudowords generated for WFE, not from the validation split of the training lexicon.

---

## Code locations (quick reference)

| Equation | File | Line(s) |
|---|---|---|
| WM noise application | [models/wm_route.py](../models/wm_route.py) | 53–54 |
| Gate formula | [models/gating.py](../models/gating.py) | 45 |
| Premotor blend | [models/gating.py](../models/gating.py) | 51 |
| `s_hat` computation | [models/ltm_route.py](../models/ltm_route.py) | 61–66 |
| `h0_LTM = tanh(W_s s_hat)` | [models/ltm_route.py](../models/ltm_route.py) | 72 |
| Confidence `c_LTM` | [models/ltm_route.py](../models/ltm_route.py) | 94 |
| Motor projection | [models/dual_route.py](../models/dual_route.py) | 65 |
| Full config / defaults | [config.py](../config.py) | all |

---

*Document written 2026-07-09. Source of truth: `docs/current_and_proposed_architecture_equations.md`. Proposed future revisions (learnable gate, comprehension pathway) are not documented here; see §2.8 and the source document §3.*
