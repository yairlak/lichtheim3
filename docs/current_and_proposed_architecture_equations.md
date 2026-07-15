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

## 3. Proposed revised architecture (Post-Yair-meeting, 2026-07-15)

> **NONE of the changes in this section are implemented.** The current checkpoint uses the equations in Section 1. This section documents proposals from the Yair meeting on 2026-07-15 for Phase 2 implementation and Phase 4+ gridsearch.

### 3.1 LTM encoder: uniGRU + last hidden (Yair proposal)

**Current:** biGRU + masked mean pooling (described in §1, confirmed in `models/ltm_route.py:42-66`).

**Proposed:**
```
PROPOSED: LTM encoder = 1-layer unidirectional GRU (bidirectional=False)
          Use last hidden state h[-1] instead of masked mean pool

out_LTM, h_LTM = GRU_enc_LTM( e_{1:T} )    # h_LTM: (1, B, H_LTM_enc)
pooled          = h_LTM.squeeze(0)            # (B, H_LTM_enc)    ← replaces masked mean pool

s_hat = Linear_2( GELU( Linear_1( pooled ) ) )
      ∈ R^{B × D}
```

With unified `H = H_WM = H_LTM_enc = H_LTM_dec`, both routes become symmetric:
```
PROPOSED: H ∈ {64, 128, 256}    (new gridsearch dimension)
          WM: GRU(E, H), last hidden → (1, B, H)
          LTM: GRU(E, H), last hidden → (1, B, H)
```

**Motivation (Yair):** Simpler, more symmetric with WM. Pack_padded_sequence and bidirectionality of LTM were not a design choice — they were defaults. UniGRU + last hidden matches WM's structure.

**Files to modify:** `models/ltm_route.py:42-66`, `config.py:LTMConfig`, `scripts/train_checkpoint.py` (add `--hidden_size` or `--ltm_enc_hidden`).

**Weight changes:** `to_semantic[0]` input dim changes from `2*H_LTM_enc` to `H_LTM_enc`; biGRU backward weights disappear. **Not backward-compatible with current checkpoint.**

### 3.2 LTM ventral noise (Yair proposal)

**Current:** No noise anywhere in LTM route (confirmed absent in `models/ltm_route.py`).

**Proposed:**
```
PROPOSED: after computing pooled (uniGRU last hidden), before to_semantic:
    if (model.training OR collect_ltm) AND σ_LTM > 0:
        ε_LTM ~ N(0, σ_LTM² I),    ε_LTM ∈ R^{H_LTM_enc}
        pooled_noisy = pooled + ε_LTM
    else:
        pooled_noisy = pooled                 (deterministic)

s_hat = to_semantic( pooled_noisy )
```

**New config field needed:** `LTMConfig.ventral_noise: float = 0.0`

**Motivation (Yair):** Symmetric with WM noise. Allows Phase 6 WM×LTM noise grid.

### 3.3 Gate with configurable threshold tau (Yair proposal)

**CURRENT (hard-coded literal):**
```
CURRENT: g = sigmoid( α · (c_LTM − 0.5) )    # 0.5 is a Python literal in gating.py:45
```

**PROPOSED:**
```
PROPOSED: g = sigmoid( α · (c_LTM − τ) )      # τ is a configurable hyperparameter

where:
    τ ∈ {0.3, 0.5, 0.7}   (Phase 7 gridsearch dimension)
    α ∈ {2.0, 4.0, 8.0}   (Phase 7 gridsearch dimension)
```

**Motivation (Yair):** `0.5` is not principled — it was a convenience value. `τ` controls the crossover point between LTM-dominant and WM-dominant routing and should be a hyperparameter.

**Files to modify:** `config.py:GatingConfig` (add `gate_threshold: float = 0.5`), `models/gating.py:45` (replace `0.5` with `self.cfg.gate_threshold`), `scripts/train_checkpoint.py` (add `--gate_threshold`).

**Status: NOT BLOCKING for Phase 4** (Phase 4 runs at τ=0.5, the current hard-coded default).

### 3.4 L2 normalization before cosine similarity — redundancy note

**Current code (`ltm_route.py:91`):** `q = F.normalize(s_hat, dim=-1)` — creates tensor `q`, does NOT modify `s_hat`.

**Yair observation:** `F.cosine_similarity` normalizes internally; the explicit L2 norm before it is redundant.

**Decision: do not remove.** The code is correct (harmless), `s_hat` is unmodified and used downstream (`decode()`, `alignment_loss()`), and removing the explicit norm would require verifying no downstream consumer expects it. Mathematical redundancy does not mean a bug.

### 3.5 Gate level: word-level confirmed (current), phoneme-level is a future proposal

**Confirmed from code** (`models/gating.py:43-46`):
```
CURRENT: conf.view(B, 1, 1) → expand(B, S, 1) → constant g across all S decoder steps
         Gate is WORD-LEVEL (item-level scalar).
```

A **phoneme-level gate** (g varying per timestep t) would require a fundamentally different architecture. Not proposed for Phase 4.

### 3.6 Future architecture directions (not in Phase 2-4 scope)

The original proposed learnable multi-input gate from the earlier document (β₀ + β_L·c_LTM − β_W·r_WM,t + β_len·len) is preserved here for reference but is deferred to Phase 12:

```mermaid
flowchart TD
    X["x (phoneme sequence)"]
    DEC_WM["WM decoder\n→ z_WM,t"]
    DEC_LTM["LTM decoder\n→ z_LTM,t"]
    CONF["c_LTM = max cosine sim"]
    RELIABILITY["r_WM,t = WM reliability\n(margin or entropy of WM logits)"]
    LEN["length(x)"]
    GATE_PROP["Proposed learnable gate (Phase 12)\ng_t = sigmoid(β₀ + β_L·c_LTM − β_W·r_WM,t + β_len·len)"]
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

## 7. Confirmed facts and resolved uncertainties (Phase 1 audit, 2026-07-15)

| Item | Status | Source |
|---|---|---|
| `premotor_dim = 128` | **Confirmed** — hardcoded default in `DualRouteModel.__init__`, NOT in config | `models/dual_route.py:34` |
| LTM MLP shape: `Linear(512,256)→GELU→Linear(256,300)` | **Confirmed** | `ltm_route.py:45-48` |
| Gate noise invariance | **Confirmed analytically** — `g` depends only on `c_LTM`, not on WM route | `gating.py:43-46` |
| `0.5` in gate is a Python literal, not configurable | **Confirmed** — no `gate_threshold` field in `GatingConfig` | `gating.py:45`, `config.py` |
| Gate is word-level (constant across all decoder steps) | **Confirmed** — `conf.view(B,1,1).expand(B,S,1)` | `gating.py:43-46` |
| L2 norm before cosine sim is redundant but harmless | **Confirmed** — `q = F.normalize(s_hat)` creates new tensor; `s_hat` unmodified | `ltm_route.py:91` |
| LTM encoder hidden `_` (backward h) is discarded in encode() | **Confirmed** — `out, _ = self.encoder(emb)` at line 62 | `ltm_route.py:62` |
| WM noise drawn S times per step with TF<1 | **Confirmed** — BLOCKING issue for Phase 6 with TF<1 | `train.py:60-63,78` |
| `optimizer_state_dict=None` in fresh mode | **Confirmed** — `build_and_train()` does not expose optimizer | `scripts/train_checkpoint.py:158,314` |
| β terms in learnable gate proposal | **Unresolved** — conceptual defaults; signs unvalidated | (Phase 12 scope) |
| `usage_prior=0.5` in GatingConfig | Acts during training (gate regularizer loss), not inference | `config.py:GatingConfig`, `losses.py:53-55` |

---

*Document last updated 2026-07-15 (Phase 1 audit). Section 1 = current implementation confirmed from code. Section 3 = proposed changes, none implemented. Source of truth for current implementation: code files listed in §5.*
