# M0 — exact architecture audit of the full-lexicon checkpoints

**Audited 2026-08-04 at HEAD `c29de360786afd26f548dffe37812928ee34f6af`, branch
`feat/full-lexicon-ceiling`.** No model forward pass was executed. Checkpoints
were opened read-only with `torch.load(..., map_location="cpu")` to read config
dictionaries and `state_dict` keys/shapes.

Evidence tags: `[CHECKPOINT]`, `[TRAINING CODE]`, `[EVALUATION CODE]`,
`[EXPERIMENT MANIFEST]`, `[BEHAVIORAL OUTPUT]`, `[STALE DOCUMENT]`,
`[INFERENCE]`, `[UNRESOLVED]`.

**The stale document `handoffs/.../04_Lichtheim3_architecture_equations_STALE_FIELDS.md`
was not used as a source for any statement below.** Every claim is traced to
executable code, checkpoint contents or output provenance.

---

## 0. Decisive provenance fact

`git diff 93a577fd9822955fa272ee733fa7e2acf81f1333..HEAD -- config.py models/`
returns **empty**. `[TRAINING CODE]`

The model-construction code at HEAD is byte-identical to the code at the
checkpoint training commit. Reading `models/*.py` and `config.py` at HEAD is
therefore reading the exact code that built and now reconstructs these
checkpoints. Only `scripts/external_eval.py` changed after training, and it is
identical at `e876b755` and HEAD.

---

## 1. Verified configuration claims

Every claim in the task prompt was checked against `cfg_*` dictionaries inside
all four `.pt` files. `[CHECKPOINT]`

| Claim | Verified value | Source key | Verdict |
|---|---|---|---|
| hidden size = 128 | `cfg_wm.hidden = 128`, `cfg_ltm.enc_hidden = 128`, `cfg_ltm.dec_hidden = 128` | `cfg_wm`, `cfg_ltm` | **CONFIRMED** |
| LTM encoder mode = `unigru_last_hidden` | `cfg_ltm.ltm_encoder_mode = "unigru_last_hidden"`, `bidirectional_encoder = False` | `cfg_ltm` | **CONFIRMED** |
| gate alpha = 2.0 | `cfg_gating.alpha = 2.0` | `cfg_gating` | **CONFIRMED** |
| gate threshold = 0.7 | `cfg_gating.gate_threshold = 0.7` | `cfg_gating` | **CONFIRMED** |
| gate usage prior = 0.5 | `cfg_gating.usage_prior = 0.5` | `cfg_gating` | **CONFIRMED** |
| gate loss weight = 0.05 | `cfg_loss.gate = 0.05` | `cfg_loss` | **CONFIRMED** |
| no training noise | `cfg_wm.interference_noise = 0.0`, `cfg_ltm.ventral_noise = 0.0` | `cfg_wm`, `cfg_ltm` | **CONFIRMED** |
| no evaluation noise | `wm_noise_enabled: false`, `apply_noise: false`, `deterministic_no_noise: true` | seed metrics.json `[BEHAVIORAL OUTPUT]` | **CONFIRMED** |
| deterministic AR evaluation | `decode_mode: "autoregressive"`, `@torch.no_grad()`, `model.eval()` | `external_eval.py:270, 645` `[EVALUATION CODE]` | **CONFIRMED** |
| forced-length readout | `pred_ids = dec_input[i, 1:n_steps+1]`, `n_steps = len(form)` | `external_eval.py:683-684` | **CONFIRMED** |

Additional verified constants: `premotor_dim = 128` `[CHECKPOINT]`;
`phon_embed_dim = 64`; `enc_layers = 1`; semantic dim **300** (from
`ltm.to_semantic.2.weight` shape `(300,128)` and `semantic_bank_shape [29571,300]`);
vocabulary size **42** (`phon_embed.weight (42,64)`, `motor.proj.weight (42,128)`).

### Checkpoint integrity

All four SHA256 values match the expected list exactly. All four record
`git_commit = 93a577fd9822955fa272ee733fa7e2acf81f1333`, `git_dirty = False`,
`provenance_schema_version = 1`, `n_train = 29571`, identical
`lexicon_file_sha256` and `ordered_training_words_sha256`. Epochs 155 / 130 /
145 / 140 for seeds 19 / 20 / 21 / 22. **Architecture and all config dicts are
identical across the four checkpoints**; `state_dict` key sets and every tensor
shape match. `[CHECKPOINT]`

---

## 2. Encoder directionality — decisive `state_dict` evidence

The `state_dict` contains **31 parameter tensors**. **No key contains
`_reverse`.** `[CHECKPOINT]`

```
ltm.encoder.weight_ih_l0   (384, 64)      384 = 3 x 128 (GRU gates), input 64 = phon_embed_dim
ltm.encoder.weight_hh_l0   (384, 128)
wm.encoder.weight_ih_l0    (384, 64)
wm.encoder.weight_hh_l0    (384, 128)
```

A bidirectional GRU would additionally emit `weight_ih_l0_reverse` and
`weight_hh_l0_reverse`. Their absence proves **both encoders are
unidirectional**, independently of the config field. This is the single most
important correction to the stale document, which presents biGRU + masked mean
as current. `[CHECKPOINT]`

---

## 3A. Inputs

`evaluate/hooks.py:31-54` `make_batch` `[EVALUATION CODE]`:

```
ei = form + [EOS]                 -> enc_in  row, enc_mask True over len(form)+1
di = [BOS] + form                 -> dec_in  row  (teacher-forced path only)
dt = form + [EOS]                 -> dec_tgt row
```

- Token IDs from `data/phonemes.py`: `SPECIALS = [PAD, BOS, EOS]` then
  `PHONEMES`, so `pad_id=0`, `bos_id=1`, `eos_id=2`; vocab size 42.
- **BOS is NOT passed to the encoder.** `enc_in` begins with the first phoneme.
- **Encoder lengths INCLUDE the EOS token**: `enc_mask` is True over
  `len(form)+1` positions.
- **WM and LTM receive identical token tensors** — both are called with the same
  `batch["enc_in"]`, `batch["enc_mask"]` (`dual_route.py:77-81`, and in the AR
  loop `external_eval.py:674-675`).
- `pack_padded_sequence(..., enforce_sorted=False)` is used by **both** encoders
  (`wm_route.py:87-88`, `ltm_route.py:137-138`), with
  `lengths = enc_mask.sum(1).clamp(min=1).cpu()`.
- `unigru_last_hidden` selects `h[-1]` — the last layer's hidden state at each
  item's **last valid position**, which given the construction above is the
  **EOS token position**. `ltm_route.py:139-140` `[TRAINING CODE]`
- **Padded batches are safe**: packing makes the returned `h` independent of
  padding, so batch composition cannot shift `s_hat`. This is the explicit fix
  relative to the historical `bigru_masked_mean` artifact described in
  `ltm_route.py:23-26`. Batch size 1 and mixed-length batches are therefore
  equivalent for both encoders. `[TRAINING CODE]`

---

## 3B. WM route (`models/wm_route.py`)

- `WMRecurrent`, lines 53-116. Encoder `nn.GRU(64, 128, batch_first=True)`,
  **unidirectional** (no `bidirectional=True` argument, line 59).
- Final state: `_, h = self.encoder(packed)` → `h` shape `(1, B, 128)`, line 89.
- Noise path, line 91: active **iff** `(self.training or apply_noise) and
  cfg.interference_noise > 0`. In the frozen WFE run `model.eval()` is set
  (`external_eval.py:270`), `apply_noise` is never passed as True in the AR path,
  and `cfg.interference_noise = 0.0`. **Noise is inactive for two independent
  reasons.** `[EVALUATION CODE]` `[CHECKPOINT]`
- Decoder: `nn.GRU(64, 128, batch_first=True)` initialised from `h`
  (`decode_from_state`, line 105).
- Decoder token input at step *t*: `self.phon_embed(dec_in)` — the shared
  embedding of the current prefix.
- Teacher-forcing branch: `dec_in = [BOS] + form` from `make_batch`.
  Autoregressive branch: `dec_input` grown by the route's own argmax
  (`external_eval.py:671-677`).
- Premotor: `to_premotor: Linear(128 → 128)`, line 61.
- Available in principle: encoder state `h`, decoder outputs `dout`, premotor,
  and route logits via the shared motor. **None of these is persisted by the
  frozen evaluator.**

---

## 3C. LTM route (`models/ltm_route.py`)

- `LTMLexicon`, lines 59-200. Encoder `nn.GRU(64, 128, bidirectional=False)`,
  lines 85-90 (the `unigru_last_hidden` branch); `enc_out_dim = enc_hidden = 128`.
- Last-hidden extraction, lines 136-140: pack → `_, h = self.encoder(packed)` →
  `pooled = h[-1]`, shape `(B, 128)`.
- Ventral noise, line 143: same two-condition guard; `ventral_noise = 0.0`, so
  **inactive**.
- `to_semantic = Sequential(Linear(128→128), GELU(), Linear(128→300))`,
  lines 92-95 → **`s_hat ∈ R^300`**, line 146.
- `sem_to_h0 = Linear(300 → 128)`; decoder `nn.GRU(64, 128)`;
  `dec_to_premotor = Linear(128 → 128)`, lines 98-100.
- Decoder init, line 153: **`h0 = tanh(sem_to_h0(s_hat)).unsqueeze(0)`**.
- Decoder input at step *t*: `self.phon_embed(dec_in)`, line 154 — the same
  shared embedding, over the current prefix.

### The two roles of `s_hat` — explicitly distinguished

| Role | Code | Vector used |
|---|---|---|
| **Initialise the LTM decoder** | `decode_from_s_hat`, line 153 | **raw, unnormalised `s_hat`** |
| **Query the frozen bank** | `lexical_field`, line 182 | `q = F.normalize(s_hat, dim=-1)`, a **separate tensor** |

The source docstring at lines 176-180 states this deliberately: normalising
creates `q` which "does NOT modify s_hat — s_hat is used downstream as-is in
decode_from_s_hat and alignment_loss". `[TRAINING CODE]`

**No retrieved bank vector is ever passed to the decoder.** `decode_from_s_hat`
takes only `s_hat` and `dec_in`; `semantic_bank` appears solely inside
`lexical_field`, whose outputs are the four scalars/vectors
`sims, confidence, margin, density`. Neither `forward` (lines 193-200) nor
`DualRouteModel.decode_from_states` (`dual_route.py:143`) routes a bank row into
the decoder. **The nearest neighbour is read off, never read in.** `[TRAINING CODE]`

Semantic-alignment target during training is referenced as `alignment_loss` with
`cfg_loss.align = 1.0`; the training objective aligns `s_hat` to the item's
GloVe vector. `[CHECKPOINT]` `[UNRESOLVED — exact loss form not re-audited in M0;
it is not needed for the evaluation-time mechanism and is deferred.]`

---

## 3D. Lexical / GloVe bank

- Construction, `external_eval.py:262-266` `[EVALUATION CODE]`:
  `bank = torch.stack([torch.tensor(e.semantic) for e in train_entries]).float()`
  where `train_entries` comes from `build_lexicon(cfg.data, vocab).split(...)`
  with the checkpoint's own `val_fraction` and effective split seed.
- `set_semantic_bank` **L2-normalises the bank rows** (`ltm_route.py:164`).
- Shape **`[29571, 300]`**, `semantic_bank_sha256 = 80a534a8…`; the entry set is
  exactly the **29,571-word training lexicon** (`n_train = 29571`,
  `n_val = 0`, `n_glove_found_at_training = 29571`,
  `n_glove_fallback_at_training = 0`). `[BEHAVIORAL OUTPUT]` `[CHECKPOINT]`
- **GloVe coverage is complete** for the bank: every training word had a GloVe
  vector; no fallback rows.
- Row → word mapping is the **ordered training word list**, pinned by
  `ordered_training_words_sha256 = 0cb1c617…` and identical across the four
  checkpoints. Row → phonological form is recoverable from the same lexicon
  entries. `[CHECKPOINT]`
- `lexical_field` (lines 167-190): `sims = q @ bank.T` (cosine, both sides
  normalised); `top2 = topk(sims, k=2)`; **`confidence = top2[:,0]`** (top-1
  cosine); `margin = top2[:,0] - top2[:,-1]`; `density = sigmoid(20·(sims −
  (confidence − 0.1))).sum(1)`.
- **Top-k IDs are NOT currently exposed**: `lexical_field` returns the full
  `sims` vector inside the dict, but the evaluator persists only the three
  scalars (`capture_gate_and_field`, `external_eval.py:136-196`). No neighbour
  identity reaches any stored file. `[EVALUATION CODE]`
- **GloVe is not supplied as an item-level input at inference.** The bank is
  built once at load time from the training lexicon; the WFE item's own GloVe
  vector is never read during evaluation. `[EVALUATION CODE]`

Duplicate orthography, duplicate phonological forms and homophones within the
bank are **not yet characterised**. `[UNRESOLVED — required before M3 lexical
attraction; it is a pure lexicon computation needing no inference.]`

**This bank is a phonology-to-lexical-neighbourhood readout aligned to GloVe
vectors. It is not a model of conceptual comprehension, and nothing in this
audit licenses that description.**

---

## 3E. Gate (`models/gating.py`)

Exact implemented equation, lines 49-51 `[TRAINING CODE]`:

```
conf = field["confidence"].view(B, 1, 1)
g    = sigmoid( alpha * (conf - gate_threshold) )      # alpha = 2.0, threshold = 0.7
g    = g.expand(B, S, 1)
```

Combination, line 56:

```
premotor = g * ltm_premotor + (1 - g) * wm_premotor
```

**Orientation: `g = 1` means LTM (ventral); `g = 0` means WM (dorsal).**
Unambiguous from line 56 and the module docstring at lines 6-8.

- Scalar shape before broadcasting: **`(B, 1, 1)`** → expanded to `(B, S, 1)`.
- **Item-level (word-level), constant across all decoder timesteps** — the
  expansion is over the sequence axis of a single per-item scalar. The docstring
  at lines 22-24 states this and the code proves it.
- **No WM information enters the gate.** `gate_value(wm, ltm, field)` uses `wm`
  only to read `B, S` for shaping (line 46); the value depends solely on
  `field["confidence"]`, itself a function of `s_hat` and the bank.
- **No learnable parameters**: `Gate` registers no `nn.Parameter`, and the
  `state_dict` contains no `gate.*` key. `[CHECKPOINT]`
- Fallback: if `field is None` the gate returns a constant 0.5 (line 48). This
  path is taken for WM-only and LTM-only routes because they never build a
  field — but in those routes the gate is never called at all (see 3G).
- `usage_prior = 0.5` and `cfg_loss.gate = 0.05` are **training-time**
  quantities: a regulariser pulling mean gate usage toward 0.5, weighted 0.05.
  **Neither has any effect at inference**: no loss is computed under
  `@torch.no_grad()`, and neither value appears in `gate_value`. `[CHECKPOINT]`
  `[TRAINING CODE]`

**The gate is a deterministic monotonic transformation of `confidence`. Gate and
confidence are one measurement, not two.**

---

## 3F. FULL mixture — exact level

`dual_route.py:85-93` and `gating.py:56` `[TRAINING CODE]`:

```
wm_premotor  = wm.to_premotor(  wm_decoder_out  )          in R^{B x S x 128}
ltm_premotor = ltm.dec_to_premotor( ltm_decoder_out )      in R^{B x S x 128}
g            = sigmoid(2.0 * (confidence - 0.7))           in R^{B x 1 x 1} -> (B,S,1)
premotor_FULL = g * ltm_premotor + (1 - g) * wm_premotor
logits_FULL   = motor.proj( premotor_FULL )                W in R^{42 x 128}, b in R^{42}
```

**FULL mixes premotor states.** It does **not** mix encoder states, decoder
hidden states, logits, probabilities or selected tokens. The motor projection
`motor.proj` is a **single shared `nn.Linear(128, 42)`** — one tensor pair in the
`state_dict`, used for FULL, WM-only and LTM-only alike (`dual_route.py:88-90`,
`168`, `176`).

### Fixed-prefix algebraic relation

Because `motor.proj` is affine and shared, for a **fixed decoder prefix**:

```
logits_FULL = W(g·z_LTM + (1−g)·z_WM) + b
            = g(W z_LTM + b) + (1−g)(W z_WM + b)          since g + (1−g) = 1
            = g · logits_LTM + (1−g) · logits_WM
```

So at a fixed timestep with a common prefix, **premotor mixing is exactly
equivalent to route-logit mixing**. This is what makes fixed-prefix
counterfactual re-mixing legitimate.

**This fixed-timestep equivalence does not make independently generated
autoregressive trajectories equivalent, because the route prefixes may differ
after the first divergence.** Once WM-AR and LTM-AR emit different tokens, their
later premotor states are conditioned on different histories and the identity
above no longer relates their trajectories.

---

## 3G. Route isolation

Evaluator flag → selected logits: `external_eval.py:670-679` calls
`model.route_logits(enc_in, enc_mask, dec_input, route=route)` inside a loop
`for route in routes:`, with `dec_input` **re-initialised to BOS at the top of
each route's block** (line 671). `dual_route.py:157-181` dispatches.

| Question | FULL | WM-only | LTM-only |
|---|---|---|---|
| unused route still computed? | n/a | **No** — only `self.wm(...)` runs (`dual_route.py:165-171`) | **No** — only `self.ltm(...)` runs (lines 172-179) |
| can the unused route affect output? | n/a | **No** — never evaluated | **No** — never evaluated |
| gate computed? | Yes (`forward`, line 85) | **No** | **No** |
| gate bypassed? | n/a | Yes — `motor(wm_premotor)` directly | Yes — `motor(ltm_premotor)` directly |
| shared motor still used? | Yes | **Yes** (line 168) | **Yes** (line 176) |
| feeds back its own argmax in AR? | Yes | **Yes** — own `dec_input` | **Yes** — own `dec_input` |
| decoder state / prefix shared? | **No** | **No** | **No** |
| forced-length + EOS identical? | Yes | Yes | Yes — the trimming block (lines 682-699) is outside the route branch |

**Verdicts:**

- **FULL: VERIFIED.**
- **WM-only: VERIFIED.** Fully isolated; the LTM route is not executed, no gate
  exists, the prefix is its own, and the shared motor is used as in training.
- **LTM-only: VERIFIED.** Same, with `want_field=collect` (False in the frozen
  production run), so not even the lexical field is computed.

One consequence to carry into M1–M5: because each route generates its own
prefix, **the three stored trajectories are not position-comparable.** Any
position-level route comparison requires a common prefix and therefore new
inference.

---

## 4. Evaluation audit

Evaluator: `scripts/external_eval.py` at commit **`e876b755d0475ed11e5fbc0419a0bd8860dfd325`**,
which is **identical to HEAD** (`git diff e876b755..HEAD -- scripts/external_eval.py`
is empty). Confirmed independently by each seed's
`metrics.json → evaluation_code_commit`. `[BEHAVIORAL OUTPUT]`

| Aspect | Finding | Source |
|---|---|---|
| `model.eval()` | called at load | `external_eval.py:270` |
| gradients | `@torch.no_grad()` on `autoregressive_decode_batch` and `capture_gate_and_field` | lines 644, 135 |
| dropout | none in any module (`models/*.py` contains no `nn.Dropout`) | `[TRAINING CODE]` |
| noise | `wm_noise=False` → `collect=False`, `apply_noise` never True; both σ = 0.0 | lines 670-671; `cfg_*` |
| random seeds / determinism | no sampling anywhere in the decode path; pure `argmax` | line 676 |
| AR loop | `dec_input` starts as BOS; `max_steps = max(len(f))` over the batch; one `argmax` appended per step | lines 671-677 |
| teacher-forcing path | separate function `eval_batch` (line 597), **not** used for the canonical AR cohort | `[EVALUATION CODE]` |
| route flags | `routes=("full","wm","ltm")`, looped, each with a fresh prefix | line 670 |
| maximum horizon | batch maximum target length | line 666 |
| forced length | per item `n_steps = len(form)`; window `dec_input[i, 1:n_steps+1]` | lines 683-684 |
| EOS capture | `_first_eos_position` on the **raw untrimmed window**, before trimming | line 690 |
| EOS trimming | break at first EOS; remaining positions dropped | lines 693-698 |
| tokens after early EOS | **discarded**, and count against edit distance as missing | line 695 |
| PAD removal | not needed — the window is sliced by target length | line 684 |
| prediction strings | `_argmax_to_phonemes` / id→symbol join | line 219 |
| target length | `len(form)`, phonemes only, **excludes** the appended EOS | line 683 |
| Levenshtein inputs | `_edit_distance` internal DP on symbol lists; operation counts by `Levenshtein.editops` 0.27.3 in the enrichment step | `metrics.json` |
| `Error_Indices` | zip-mismatch positions, **no alignment** | `metrics.json` |
| raw untrimmed tokens | **NOT persisted** — `dec_input` is local to the function | line 679 |
| logits | **NOT persisted** | line 676 |
| hidden / premotor states | **NOT persisted** (`collect=False` throughout) | line 671 |

**The existing serial-position figure must not be used for first-error
dynamics.** It aggregates zip-mismatch positions and applies PCHIP interpolation
to 100 points; it is neither an item-level first-event record nor a survival
denominator.

`capture_gate_and_field` (lines 135-196) runs **one extra deterministic
full-route forward on BOS only** to record `gate`, `confidence`, `margin`,
`density`. Since the gate is word-level and constant across timesteps, this is
sufficient for the gate but yields **no per-timestep or per-neighbour
information**.

---

## 5. Summary of what M0 establishes

1. The checkpoints, the model code and the evaluator are mutually consistent and
   fully pinned; HEAD reconstructs the checkpoints exactly.
2. Both encoders are unidirectional, proven by absent `_reverse` weights.
3. `s_hat` initialises the LTM decoder; a **normalised copy** queries the bank;
   **no bank vector enters the decoder**.
4. The gate is a parameter-free, word-level, deterministic function of top-1
   cosine confidence, with `g = 1 ⇒ LTM`.
5. FULL mixes **premotor states** before a **shared** affine motor projection,
   giving exact premotor/logit mixing equivalence **at a fixed prefix only**.
6. All three routes are genuinely isolated and each generates its **own**
   prefix — so the frozen outputs support word-level comparisons but **not**
   position-level route comparison.
7. The frozen run persisted **no logits, no hidden states, no premotor states, no
   raw untrimmed tokens and no bank neighbour identities**.

Nothing here explains the length effect, and M0 does not attempt to.
