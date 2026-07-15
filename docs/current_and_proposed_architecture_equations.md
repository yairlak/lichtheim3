# Current and Proposed Architecture: Equations and Notation

> **Purpose.** Exact equations derived from the implemented code, plus a proposal for a revised gate.  Intended as a companion to `docs/wfe_route_noise_and_ltm_audit_summary.md` for discussion.

---

## 1. Current architecture implemented in this checkpoint

### Notation

| Symbol | Meaning |
|---|---|
| x = (x₁, …, xT) | input phoneme ID sequence |
| y<t = (y₁, …, y_{t−1}) | decoder prefix (gold for TF; model output for AR) |
| B, T, S | batch size, encoder length, decoder length |
| E = 64 | shared phoneme embedding dim (`phon_embed_dim`) |
| H_WM = 128 | WM GRU hidden dim |
| H_LTM_enc = 256 | LTM encoder GRU hidden dim (per direction) |
| H_LTM_dec = 256 | LTM decoder GRU hidden dim |
| D = 300 | semantic / GloVe dim |
| P = 128 | premotor dim |
| V | vocabulary size (phonemes) |
| σ | interference noise std (`interference_noise`, default 0.10) |
| α = 4.0 | gate sharpness (`GatingConfig.alpha`) |

---

### Shared phoneme embedding

Both routes share a single embedding matrix:

```
E_embed ∈ R^{V × E}
e_t = E_embed[x_t]                   ∈ R^E
```

---

### WM / dorsal route   (`models/wm_route.py`)

**Encoder** (1-layer unidirectional GRU with `pack_padded_sequence`):

```
lengths = sum(enc_mask, dim=1)        # true sequence lengths, padding-immune
h_WM = GRU_enc_WM( e_{1:T} )         # (1, B, H_WM)   final hidden state only
```

**Noise** (evaluation-time perturbation; training noise uses the same path):

```
if (training or collect) and σ > 0:
    ε ~ N(0, σ²I),    ε ∈ R^{H_WM}
    h_WM_noisy = h_WM + ε
else:
    h_WM_noisy = h_WM
```

Noise is applied to the **scalar encoder final state** (after packing), not to the
sequence or the embeddings.  The decoder receives the noisy state.

**Decoder** (1-layer GRU, initialised from h_WM_noisy):

```
d_WM,t = GRU_dec_WM( e_{y<t},  h_WM_noisy )    ∈ R^{B × S × H_WM}
z_WM,t = W_WM d_WM,t + b_WM                     ∈ R^{B × S × P}
```

where `W_WM ∈ R^{P × H_WM}` is the `to_premotor` linear layer.

---

### LTM / ventral route   (`models/ltm_route.py`)

**Encoder** (1-layer bidirectional GRU; **no** `pack_padded_sequence`):

```
out_LTM ∈ R^{B × T × 2H_LTM_enc}  ← GRU_enc_LTM( e_{1:T} )
                                       (forward + backward concatenated at each t)

pooled = sum_t( out_LTM,t * mask_t ) / sum_t( mask_t )    ∈ R^{B × 2H_LTM_enc}
```

⚠ The backward pass starts from the **rightmost padded position**, not the last
real token.  This means `pooled` (and therefore `s_hat`) shifts with the batch's
maximum sequence length — a known artifact.

**Semantic projection** (2-layer MLP):

```
s_hat = Linear(D, H_LTM_enc)( GELU( Linear(2H_LTM_enc, H_LTM_enc)( pooled ) ) )
      ∈ R^{B × D}          (D = 300)
```

**Semantic bank and lexical field** (`lexical_field`):

The bank **B_lex ∈ R^{n_words × D}** is a frozen, L2-normalised matrix of the
training lexicon's GloVe vectors.

```
q = s_hat / ||s_hat||_2                              ∈ R^{B × D}     (L2-norm)
sims = q  B_lex^T                                    ∈ R^{B × n_words}
c_LTM  = max_i  sims_i                               ∈ R^B            confidence
margin = sims_(1) - sims_(2)                         ∈ R^B            top1 − top2
density = sum_i sigmoid( 20 · (sims_i - (c_LTM − 0.10)) )  ∈ R^B   soft neighbor count
```

**Decoder** (1-layer GRU, initialised from s_hat):

```
h0_LTM = tanh( W_s  s_hat + b_s )    ∈ R^{B × H_LTM_dec}   (sem_to_h0)
d_LTM,t = GRU_dec_LTM( e_{y<t},  h0_LTM )                   ∈ R^{B × S × H_LTM_dec}
z_LTM,t = W_LTM  d_LTM,t + b_LTM                            ∈ R^{B × S × P}
```

where `W_LTM ∈ R^{P × H_LTM_dec}` is `dec_to_premotor`.

---

### Gate   (`models/gating.py`)

The gate is a **scalar per item, constant across decoder steps**:

```
g = sigmoid( α · (c_LTM − 0.5) )     ∈ R^B,    α = 4.0
```

Broadcasting to match the premotor sequence shape: `g ∈ R^{B × S × 1}` (expanded).

**No learnable parameters.  No input from the WM route.**

---

### Full / gated route combination   (`models/gating.py`, `models/dual_route.py`)

The gate blends **premotor states** (not logits, not hidden states):

```
z_full,t = g · z_LTM,t + (1 − g) · z_WM,t     ∈ R^{B × S × P}
```

**Motor Cortex** (`models/motor.py`) — a single shared linear layer:

```
logits_full,t   = W_motor  z_full,t    ∈ R^{B × S × V}
logits_WM,t     = W_motor  z_WM,t      ∈ R^{B × S × V}
logits_LTM,t    = W_motor  z_LTM,t     ∈ R^{B × S × V}
```

All three routes share the **same** `W_motor ∈ R^{V × P}`.  The full route does
**not** mix logits — it mixes premotor states, then passes the mixture through a
single common projection.

**Phoneme probability:**

```
p( y_t | y<t, x ) = softmax( logits_full,t )
```

---

### Summary diagram (Mermaid)

```mermaid
flowchart TD
    X["x (phoneme sequence)"]

    subgraph WM["WM / Dorsal route"]
        ENC_WM["GRU encoder\n(pack_padded)\n→ h_WM ∈ R^128"]
        NOISE["+ ε ~ N(0, σ²I)\n(if training or collect)"]
        DEC_WM["GRU decoder\n→ z_WM ∈ R^(S×128)"]
    end

    subgraph LTM["LTM / Ventral route"]
        ENC_LTM["biGRU encoder\n(masked mean-pool)\n→ pooled ∈ R^512"]
        MLP["MLP\n→ s_hat ∈ R^300"]
        BANK["Semantic bank B_lex\n(frozen GloVe, n_words×300)\n→ c_LTM = max cosine sim"]
        DEC_LTM["GRU decoder\n(h0 = tanh(W_s s_hat))\n→ z_LTM ∈ R^(S×128)"]
    end

    GATE["Gate\ng = sigmoid(4·(c_LTM − 0.5))\nblends premotor states"]
    MOTOR["Motor Cortex\nW_motor (shared, Linear)\n→ logits ∈ R^(S×V)"]
    OUT["p(y_t | y<t, x) = softmax(logits)"]

    X --> ENC_WM --> NOISE --> DEC_WM
    X --> ENC_LTM --> MLP --> BANK
    MLP --> DEC_LTM
    BANK -->|c_LTM| GATE
    DEC_WM -->|z_WM| GATE
    DEC_LTM -->|z_LTM| GATE
    GATE -->|z_full = g·z_LTM + (1−g)·z_WM| MOTOR --> OUT
```

---

## 2. Why the current gate is limited

### What the gate depends on

The gate value `g = sigmoid(4 · (c_LTM − 0.5))` is a function of **LTM
confidence only**.  It has no inputs from the WM route.

### Gate is invariant to WM noise

Consequence: adding interference noise to `h_WM` at evaluation time does not
change `g`.  The gate does not detect a degraded WM signal.  Routing is
identical whether σ = 0.0 or σ = 1.0.

This was confirmed experimentally: the gate values for pseudowords are constant
across all noise levels in the audit figures (`gate_by_noise_group_length.png`).

### Full/gated robustness is structural, not adaptive

When WM is noisy, the full/gated route is more robust because:

- For real words: `g ≈ 1` → full ≈ LTM, which is unaffected by WM noise.
- For pseudowords: `g < 0.5` but `g > 0` → partial LTM contribution provides a
  stable anchor that reduces the variance of `z_full` even when `z_WM` is noisy.

This is a passive benefit of mixing — not active compensation.

### What the gate cannot do

- It cannot route away from a noisy WM.
- It cannot adapt gate value to item difficulty, word length, or noise level.
- It cannot incorporate any WM signal (reliability estimate, entropy, margin).
- `g` is constant across all decoder steps (same item-level value at t=1 and t=S).

---

## 3. Proposed revised architecture

### Motivation

A more principled gate should reflect both lexical evidence (from LTM) and
phonological buffer reliability (from WM), and potentially word length.

### Candidate gate equation

Let:

```
c_LTM          = max cosine sim to semantic bank    (current confidence signal)
r_WM,t         = WM reliability estimate at step t,
                 e.g. margin of WM logits: max_logit − 2nd_max_logit,
                 or 1 − H(softmax(z_WM,t))  (entropy-based)
length(x)      = phoneme sequence length of input x
σ_eval         = applied interference noise level (0 at test if no noise)
```

Proposed gate:

```
g_t = sigmoid(
        β₀
      + β_L  · c_LTM
      − β_W  · r_WM,t
      + β_len · length(x)
      + β_σ  · σ_eval
      )
```

**Sign conventions (conceptual):**
- `+β_L · c_LTM`: higher lexical confidence → more LTM
- `−β_W · r_WM,t`: higher WM reliability → less LTM (WM can handle it)
- `+β_len · length(x)`: longer sequences → more LTM reliance (or the reverse if LTM is length-sensitive — the sign should be validated empirically)
- `+β_σ · σ_eval`: higher applied noise → more LTM

**Output combination** (unchanged):

```
z_t = (1 − g_t) · z_WM,t + g_t · z_LTM,t
p(y_t | y<t, x) = softmax( W_motor · z_t )
```

### Implementation notes

- This is a **proposal**, not current code.
- Signs of `β` terms are conceptual defaults and must be validated.
- If `r_WM,t` is position-level (entropy at step t), the gate becomes
  **position-level** — g varies across the decode sequence.
  The current gate is item-level (g constant over t).
- The β parameters could be learnable scalars (cheap) or a small MLP (more expressive).
- **Would require retraining** — the current checkpoint has a parameter-free gate;
  adding trainable β changes the training objective.

### Diagram addition (Mermaid)

```mermaid
flowchart TD
    X["x (phoneme sequence)"]
    DEC_WM["WM decoder\n→ z_WM,t"]
    DEC_LTM["LTM decoder\n→ z_LTM,t"]
    CONF["c_LTM = max cosine sim"]
    RELIABILITY["r_WM,t = WM reliability\n(margin or entropy of WM logits)"]
    LEN["length(x)"]
    GATE_PROP["Proposed gate\ng_t = sigmoid(β₀ + β_L·c_LTM − β_W·r_WM,t + β_len·len)"]
    MOTOR["Motor Cortex (shared)\n→ logits"]
    OUT["p(y_t | y<t, x)"]

    X --> DEC_WM --> RELIABILITY --> GATE_PROP
    X --> DEC_LTM --> CONF --> GATE_PROP
    X --> LEN --> GATE_PROP
    DEC_WM -->|z_WM,t| GATE_PROP
    DEC_LTM -->|z_LTM,t| GATE_PROP
    GATE_PROP -->|z_t = g_t·z_LTM + (1−g_t)·z_WM| MOTOR --> OUT
```

---

## 4. Semantic-route clarification

### GloVe is not an input at WFE inference time

At WFE evaluation, **no GloVe vector is ever fed to the model**.  GloVe served
two roles during training:

1. **Alignment target** for the semantic loss (`loss.align`): the LTM encoder was
   trained to map phoneme sequences to their GloVe 300-d vector.  After training,
   these GloVe vectors are no longer needed as input.

2. **Semantic bank construction**: before evaluation, the GloVe vectors for all
   training-lexicon words are stored in the frozen matrix `B_lex` (loaded once
   at checkpoint initialisation).  This is fixed, not updated at inference.

### Two separate roles of s_hat and the semantic bank

```
s_hat  →  sem_to_h0  →  h0_LTM  →  LTM decoder initial state
       (tells the decoder "which phoneme sequence to regenerate")

s_hat  →  L2-normalise  →  q  →  q @ B_lex^T  →  sims  →  c_LTM
       (measures lexical familiarity: how close is this form to any known word?)
       →  gate value g
```

The LTM encoder `s_hat` therefore simultaneously serves as:
- The decoder initialisation signal (content for form regeneration).
- The lexical confidence signal (proximity to the bank → gate input).

### This is not a full semantic pathway

The current LTM route implements **lexical anchoring**, not semantic comprehension:

- `s_hat` is a form-derived proxy for meaning; it is not a full conceptual representation.
- The semantic bank lookup measures neighborhood in GloVe space (familiarity /
  lexicality); it does not implement inference, composition, or naming.
- Pseudowords have no GloVe vector; the model maps them to the nearest real-word
  neighbor in s_hat space — this produces lexicalization errors, not novel
  phonological generalization.

### Future architecture: separate lexical confidence from semantic comprehension

A more rigorous Ueno-style ventral pathway would require:

- A **comprehension head**: semantic input (word meaning / pictured object) →
  semantic vector, trained separately from phonological encoding.
- A **naming pathway**: semantic vector → phoneme sequence, using semantic
  representations as an explicit input (not just an alignment target).
- Clear separation of `c_LTM` (lexical familiarity) from a semantic activation
  signal (e.g., distance to a separately computed conceptual representation).

Until these components are added, any naming / comprehension inference from the
LTM route should be treated as **diagnostic**, not as a faithful Ueno-style test.

---

## 5. Files inspected

| File | Purpose |
|---|---|
| [models/wm_route.py](../models/wm_route.py) | WM encoder/decoder, noise path |
| [models/ltm_route.py](../models/ltm_route.py) | LTM encoder, semantic bank, lexical field, decoder |
| [models/gating.py](../models/gating.py) | Gate formula and premotor blending |
| [models/dual_route.py](../models/dual_route.py) | Full model forward pass, route-isolated logits |
| [models/motor.py](../models/motor.py) | Shared Motor Cortex (Linear) |
| [config.py](../config.py) | All hyperparameters including α = 4.0, σ = 0.10 defaults |

---

## 6. Exact code locations for gate and route combination

| Equation | File | Line |
|---|---|---|
| `g = sigmoid(α·(conf − 0.5))` | [models/gating.py](../models/gating.py) | line 45 |
| `premotor = g * ltm + (1 − g) * wm` | [models/gating.py](../models/gating.py) | line 51 |
| `"logits": self.motor(gated["premotor"])` | [models/dual_route.py](../models/dual_route.py) | line 65 |
| Noise application to h_WM | [models/wm_route.py](../models/wm_route.py) | lines 53–54 |
| `s_hat` computation | [models/ltm_route.py](../models/ltm_route.py) | lines 61–66 |
| `h0_LTM = tanh(W_s s_hat)` | [models/ltm_route.py](../models/ltm_route.py) | line 72 |
| `confidence = top2[:, 0]` | [models/ltm_route.py](../models/ltm_route.py) | line 94 |

---

## 7. Uncertainties

- **Premotor dim = 128** (`premotor_dim` default in `DualRouteModel.__init__`) — confirmed from the code; not stored in config.
- **LTM MLP exact shape:** `Linear(512, 256) → GELU → Linear(256, 300)` — confirmed from `to_semantic` in `ltm_route.py:45–48`.
- **Gate noise invariance** is an analytical consequence of the code, verified experimentally in the audit.  No further uncertainty.
- **β terms in proposed gate** are conceptual defaults.  The sign of `β_len` is genuinely uncertain (current data show LTM is more length-sensitive for pseudowords; it is unclear whether giving the gate length information would help or hurt and in which direction).
- **The `usage_prior` field** in `GatingConfig` (default 0.5) is mentioned in the config as a regularizer prior but its effect on the current checkpoint is not audited here — it acts during training, not inference.

---

*Document created from code inspection of the `lichtheim3_30k_glove_e60_to_e120_lowlr.pt` checkpoint.  Proposed equations are not implemented.*
