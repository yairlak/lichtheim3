# LESIONING V2 — CONNECTIVITY MAPPING AUDIT

    PASS        STATIC / READ-ONLY SCIENTIFIC AUDIT
    BASE        db06aad0aabc701b94eb75c87661c7b5de29c02b
    BRANCH      paper-programme/lesioning-v2-operator-freeze
    DATE        2026-09-20

No lesion was run. No forward pass was performed through P1/P2/P3/P4. No
parameter was modified. Checkpoint reads were metadata/shape only.

Every statement is labelled:
**[VERIFIED]** in a cited source · **[INFERENCE]** our reading ·
**[ADAPTATION]** a Lichtheim3 decision, never attributed to Ueno ·
**[UNRESOLVED]** not recoverable.

---

## 1. Ueno source audit

Source of truth: `docs/lesioning/UENO_2011_PROTOCOL.md` (Lichtheim3
brain-damage lineage), which records direct reading of Ueno et al. 2011
*Neuron* 72:385–396 and its 22-page supplement, with file hashes in its
`PROVENANCE.md`. Quotes below are as recorded there.

### 1.1 The damage operation

**[VERIFIED]** (main text p. 388) "simulated damage included both the addition
of noise to the unit outputs as an analog of gray matter pathology (ranging
from 0.01 to 0.15 in equal intervals) and removal of the incoming links to the
damaged layer as an analog of white matter damage (ranging from 0.5% to 7.5%
in equal intervals)."

**[VERIFIED]** (Fig. 3 caption p. 388) "the proportion (%) of the incoming
links removed and the range of the noise (bracket) over the output of the
damaged layer."

So, confirmed against CENTRAL §3:

| requirement | status |
|---|---|
| noise + incoming-link removal combined | **[VERIFIED]** |
| noise = gray-matter analogue | **[VERIFIED]** |
| incoming-link removal = white-matter analogue | **[VERIFIED]** |
| 15 severity levels | **[VERIFIED]** (p. 388) |
| noise 0.01 → 0.15 | **[VERIFIED]** |
| incoming-link removal 0.5% → 7.5% | **[VERIFIED]** |

**[VERIFIED]** Noise acts on the **output** of the damaged layer; removal acts
on links **incoming to** it. Damage is paired: one severity index selects both.

### 1.2 What the source does NOT establish

    UENO_MASK_NESTING = NOT_ESTABLISHED_FROM_SOURCE

**[UNRESOLVED]** Neither text nor supplement states whether the links removed
at severity k are a superset of those removed at k−1. Nesting is **not**
attributed to Ueno anywhere in this programme.

**[UNRESOLVED]** The noise distribution. The word used throughout is "range";
"Gaussian", "uniform", "normal", "SD", "variance" do not occur in a noise
context in either document. Uniform is an **[INFERENCE]**, never a claim.

**[UNRESOLVED]** Whether recurrent self-connections were removed. CENTRAL's
instruction not to infer this from the word "incoming" is correct and is
honoured: nothing in the recorded source resolves it.

### 1.3 Sites — A / B / C as required by CENTRAL §3

Ueno's model is a layered interactive network of sigmoid units in [0,1], with
explicit inter-layer projections.

| | A. computational function of the Ueno site | B. what "incoming connections" operationally means | C. does the source support inter-layer afference / recurrent self-connections / both? |
|---|---|---|---|
| L1 analogue (dorsal/phonological buffer, iSMG) | phonological working store mediating repetition | the projections arriving at that layer from the preceding layer(s) | **inter-layer afference [VERIFIED]**; recurrent self-connection removal **[UNRESOLVED]** |
| L2 analogue (ventral input encoding, aSTG) | mapping phonological input toward meaning | the projections arriving at that layer | **inter-layer afference [VERIFIED]**; recurrence **[UNRESOLVED]** |
| L3 analogue (ventral production, vATL→output) | mapping semantic representation toward speech output | the projections arriving at the production layer, which carry the semantic pattern | **inter-layer afference [VERIFIED]**; recurrence **[UNRESOLVED]** |

**[VERIFIED]** One site type is explicitly special-cased (supplement p. 15): a
hard-clamped **input** layer "does not have input connections like the other
layers", so Ueno removed **outgoing** connections there instead. This is direct
evidence that "incoming links" means *the projections that deliver signal into
the damaged layer*, and that Ueno reasoned functionally about what carries
afference — not by tensor naming. **None of L1/L2/L3 is a clamped input layer**,
so the standard incoming-link rule applies to all three.

---

## 2. Live Lichtheim3 tensor audit

All evidence from the code at this worktree's base commit.

### 2.1 Architecture, verified not assumed

`models/dual_route.py:53-67` — the model owns ONE phoneme embedding and passes
the same object into both routes, plus one shared motor readout:

```
def __init__(self, cfg, vocab, premotor_dim: int = 128):      # :53
    self.phon_embed = nn.Embedding(vocab.size, cfg.ltm.phon_embed_dim, ...)  # :60
    self.wm  = WMRecurrent(cfg.wm, self.phon_embed, premotor_dim)            # :63
    self.ltm = LTMLexicon(cfg.ltm, self.phon_embed, cfg.data.semantic_dim,
                          premotor_dim, vocab.pad_id)                        # :64-65
    self.motor = MotorCortex(premotor_dim, vocab.size)                       # :67
```

`models/wm_route.py:57-61`
```
self.phon_embed = phon_embed                 # shared          :57
self.encoder = nn.GRU(emb, cfg.hidden, batch_first=True)       :59
self.decoder = nn.GRU(emb, cfg.hidden, batch_first=True)       :60
self.to_premotor = nn.Linear(cfg.hidden, premotor_dim)         :61
```
`models/wm_route.py:89` `_, h = self.encoder(packed)` then `:105`
`dout, _ = self.decoder(self.phon_embed(dec_in), h)` — the WM encoder→decoder
path is a **hidden-state handoff with no weight matrix**.

`models/ltm_route.py:78-100`
```
else:  # unigru_last_hidden
    self.encoder = nn.GRU(emb_dim, cfg.enc_hidden, num_layers=cfg.enc_layers,
                          batch_first=True, bidirectional=False)   :86-89
    enc_out_dim = cfg.enc_hidden                                   :90
self.to_semantic = nn.Sequential(
    nn.Linear(enc_out_dim, cfg.enc_hidden), nn.GELU(),
    nn.Linear(cfg.enc_hidden, semantic_dim))                       :92-95
self.sem_to_h0  = nn.Linear(semantic_dim, cfg.dec_hidden)          :98
self.decoder    = nn.GRU(emb_dim, cfg.dec_hidden, batch_first=True):99
self.dec_to_premotor = nn.Linear(cfg.dec_hidden, premotor_dim)     :100
```
`models/ltm_route.py:153-155`
```
h0 = torch.tanh(self.sem_to_h0(s_hat)).unsqueeze(0)   # (1, B, dec_hidden)
out, _ = self.decoder(emb, h0)
```

### 2.2 Widths — verified against the FINAL V7 configuration

CENTRAL warned against silently using the historical H128 LTM code. That trap
is real: `train_joint_scratch.py:157` declares
`CANONICAL_HIDDEN = 128  # wm.hidden == ltm.enc_hidden == ltm.dec_hidden`,
and the `--enc-hidden`/`--dec-hidden` argparse defaults are that constant
(`:2200-2203`). **V7 does not use those defaults.**

    scripts/cluster/jeanzay/fresh_ceiling_v7.slurm:117   WM=128; ENC=512; DEC=512
    scripts/cluster/jeanzay/fresh_ceiling_v7.slurm:225   --wm-hidden "$WM" --enc-hidden "$ENC" --dec-hidden "$DEC"
    scripts/naming_comprehension/fresh_ceiling_v7.py:60  WM, ENC, DEC = 128, 512, 512
    scripts/naming_comprehension/fresh_ceiling_v7.py:72  EXPECTED["widths"] = {...}
    scripts/naming_comprehension/fresh_ceiling_v7.py:279 _need(dict(ck["widths"]) == EXPECTED["widths"], ...)

`verify-ckpt` is run on the selected source in the V7 driver
(`fresh_ceiling_v7.slurm:299`), so **every** P1–P4 SOURCE checkpoint provably
carries widths `{wm_hidden:128, ltm_enc_hidden:512, ltm_dec_hidden:512}`.
`ltm_encoder_mode` is `CANONICAL_LTM_ENCODER_MODE = "unigru_last_hidden"`
(`:158`), `phon_embed_dim = 64` (`config.py:135`), `semantic_dim = 300`,
`premotor_dim = 128` (`dual_route.py:53`).

| expected by CENTRAL | verified | source |
|---|---|---|
| phoneme embedding 64 | **64** | `config.py:135` |
| WM hidden 128 | **128** | slurm:117, driver:60, asserted at :279 |
| LTM encoder hidden 512 | **512** | same |
| semantic 300 | **300** | `config.py` semantic_dim |
| LTM decoder hidden 512 | **512** | same |
| premotor 128 | **128** | `dual_route.py:53` |

**[VERIFIED]** Tensor names/shapes were additionally enumerated from a local
checkpoint of the *same driver and format* (`lichtheim3.joint_scratch.v1`,
`unigru_last_hidden`, widths 128/512/512) — read-only, metadata only, no model
constructed and no forward pass:

```
phon_embed.weight            (42, 64)      ltm.encoder.weight_ih_l0   (1536, 64)
wm.phon_embed.weight         (42, 64)      ltm.encoder.weight_hh_l0   (1536, 512)
ltm.phon_embed.weight        (42, 64)      ltm.to_semantic.0.weight   (512, 512)
wm.encoder.weight_ih_l0      (384, 64)     ltm.to_semantic.2.weight   (300, 512)
wm.encoder.weight_hh_l0      (384, 128)    ltm.sem_to_h0.weight       (512, 300)
wm.encoder.bias_ih_l0        (384,)        ltm.sem_to_h0.bias         (512,)
wm.encoder.bias_hh_l0        (384,)        ltm.decoder.weight_ih_l0   (1536, 64)
wm.decoder.weight_ih_l0      (384, 64)     ltm.decoder.weight_hh_l0   (1536, 512)
wm.to_premotor.weight        (128, 128)    ltm.dec_to_premotor.weight (128, 512)
                                           motor.proj.weight          (42, 128)
```

**Sharing, decisive for candidate rejection:** `phon_embed.weight`,
`wm.phon_embed.weight` and `ltm.phon_embed.weight` are the SAME module object
(constructed once at `dual_route.py:60`, passed to both routes) surfacing three
times in the state_dict. `motor.proj` is likewise shared — `models/motor.py`
calls it "the architectural commitment that there is *one* output channel for
speech, fed by two streams."

### 2.3 GRU equation audit

`torch.nn.GRU` documented equations (verified against the installed
interpreter, torch 2.12.1 locally; V7 recorded 2.6.0 — the layout is a stable
documented API contract and is re-asserted by a static test):

```
r_t = sigma(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
z_t = sigma(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
n_t = tanh (W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
h_t = (1 - z_t) * n_t + z_t * h_{t-1}
```

    weight_ih_l0 : (3*hidden, input_size)   gate order [r ; z ; n]
    weight_hh_l0 : (3*hidden, hidden)       gate order [r ; z ; n]

Scientifically:

    weight_ih_l0  EXTERNAL INPUT -> current GRU state      = afference
    weight_hh_l0  PREVIOUS HIDDEN STATE -> current state   = internal recurrent dynamics

No tensor is selected because its name contains "weight_ih". The selection
below is functional; `weight_hh` is rejected on the grounds that it is the
site's own internal dynamics, which the **activation** term already perturbs.

---

## 3. Candidate comparison (CENTRAL §17)

### L1 — DORSAL / WM ENCODER · functional target: phonological working store

| CANDIDATE_TENSOR | WHAT_SIGNAL_IT_CARRIES | AFFERENCE or INTERNAL | ROUTE-SPECIFIC? | SHARED? | SCIENTIFIC_ADVANTAGE | CONFOUND | KEEP/REJECT | EVIDENCE |
|---|---|---|---|---|---|---|---|---|
| `wm.encoder.weight_ih_l0` (384,64) | embedded phoneme input → dorsal store | **external afference** | yes | no | exactly Ueno's "incoming links": the signal entering the damaged layer | none identified | **KEEP** | wm_route.py:59,86-89 |
| `wm.encoder.weight_hh_l0` (384,128) | previous dorsal state → current | internal dynamics | yes | no | — | duplicates what activation noise already perturbs; not "incoming" in Ueno's sense | REJECT | wm_route.py:59 |
| both (ih+hh) | mixture | mixed | yes | no | — | conflates disconnection with state dysfunction; makes the composite operator non-identifiable | REJECT | D-05 §4 |
| `phon_embed.weight` (42,64) | one-hot phoneme → embedding | outgoing weights of a clamped input | **NO — shared by BOTH routes** | **yes** | Ueno's clamped-input analogue | lesioning it damages L1 and L2 simultaneously; destroys route specificity | REJECT | dual_route.py:60,63-65 |
| `wm.decoder.weight_ih_l0` | previous phoneme → dorsal decoder | afference of a *different* site | yes | no | — | production-side, not the encoder/store | REJECT | wm_route.py:60,105 |
| encoder→decoder path | dorsal state handoff | **no weight matrix exists** | — | — | — | not lesionable as connectivity | REJECT | wm_route.py:105 |

### L2 — VENTRAL / LTM ENCODER · functional target: phonological input → semantic encoding

| CANDIDATE_TENSOR | WHAT_SIGNAL_IT_CARRIES | AFFERENCE or INTERNAL | ROUTE-SPECIFIC? | SHARED? | SCIENTIFIC_ADVANTAGE | CONFOUND | KEEP/REJECT | EVIDENCE |
|---|---|---|---|---|---|---|---|---|
| `ltm.encoder.weight_ih_l0` (1536,64) | embedded phoneme input → ventral encoder | **external afference** | yes | no | the ventral analogue of L1; symmetric operator across the two encoders | none identified | **KEEP** | ltm_route.py:86-89,139 |
| `ltm.encoder.weight_hh_l0` (1536,512) | previous ventral state → current | internal dynamics | yes | no | — | same objection as L1 hh | REJECT | ltm_route.py:86-89 |
| `phon_embed.weight` | shared embedding | — | **NO** | **yes** | — | identical sharing confound as L1 | REJECT | dual_route.py:60 |
| `ltm.to_semantic.0/.2` | pooled encoder state → 300-d semantic | **downstream projection, efferent** | yes | no | — | not encoder afference; it is the encoder's *output* stage. `.2` is also the Arm-A repair head — lesioning it would confound the repair | REJECT | ltm_route.py:92-95,146 |

### L3 — VENTRAL PRODUCTION · functional target: semantic → phonological production

The production graph, read from `ltm_route.py:153-155`:

```
s_hat --[sem_to_h0 (512,300)]--> tanh --> h0 --> decoder GRU --> dec_to_premotor --> motor
                                                    ^
previous phoneme --[phon_embed]--> emb -------------'  (decoder.weight_ih_l0)
```

There are **two functionally distinct inputs** to production: the semantic
pattern (once, as the initial state) and the autoregressive phonological
feedback (at every step).

| CANDIDATE_TENSOR | WHAT_SIGNAL_IT_CARRIES | AFFERENCE or INTERNAL | ROUTE-SPECIFIC? | SHARED? | SCIENTIFIC_ADVANTAGE | CONFOUND | KEEP/REJECT | EVIDENCE |
|---|---|---|---|---|---|---|---|---|
| `ltm.sem_to_h0.weight` (512,300) | **semantic pattern → initial production state** | **external afference into production** | yes | no | the exact analogue of Ueno's vATL→production incoming links; damages transmission of meaning into speech output | none identified | **KEEP** | ltm_route.py:98,153 |
| `ltm.decoder.weight_ih_l0` (1536,64) | previously produced phoneme → decoder | afference, but of *phonological feedback*, not of meaning | yes | no | — | this is the model's autoregressive self-feedback, not the semantic input; damaging it models a different disorder | REJECT | ltm_route.py:99,155 |
| `ltm.decoder.weight_hh_l0` (1536,512) | previous decoder state → current | internal dynamics | yes | no | — | as above | REJECT | ltm_route.py:99 |
| `ltm.dec_to_premotor.weight` (128,512) | decoder state → premotor | **efferent** | yes | no | — | outgoing, not incoming | REJECT | ltm_route.py:100 |
| `sem_to_h0` + `decoder.weight_ih` | both | mixed | yes | no | — | CENTRAL forbids combining absent forcing graph evidence; the graph in fact *separates* them cleanly | REJECT | ltm_route.py:153-155 |

**Naming and repetition share this production interface.** `s_hat` is produced
by the ventral encoder during repetition and supplied from the target semantic
vector during naming, but in both cases production begins at
`h0 = tanh(sem_to_h0(s_hat))` (`ltm_route.py:153`). An L3 lesion therefore acts
on both tasks through the same tensor — a stated interpretive consequence, not
a defect.

---

## 4. Mask granularity

Two alternatives, as required:

    A. independent scalar coefficients over (3H, input_dim)
    B. logical source->target edge over (H, input_dim), the SAME mask tiled
       across the r/z/n gate blocks

A GRU's three gate rows for a given (hidden unit, input unit) pair are three
coefficients of **one anatomical connection**. Ueno removed *links*, not
coefficients. Under (A) at fraction p, a logical edge survives intact with
probability (1−p)³ — at p=0.30 only 34.3% of edges are fully intact, ~2.7% are
fully removed, and ~63% are *partially* degraded. That is a diffuse
weight-perturbation, not link removal. Under (B), exactly p of edges are fully
removed and the remainder are pristine — which is what "removal of the incoming
links" means.

    MASK_GRANULARITY = LOGICAL_SOURCE_TARGET_EDGE          [ADAPTATION]

For `sem_to_h0` (a plain `nn.Linear`) one logical edge **is** one scalar
coefficient, so no ambiguity arises at L3.

**Biases are not connectivity.** `BIAS_LESIONABLE = NO`. This also matches the
historical implementation, where `include_bias: bool = False` is annotated
"never true by default" (`lesion/spec.py:77`).

Logical-edge counts, confirmed against the verified widths:

| site | tensor | shape | logical edges | scalar coefficients |
|---|---|---|---|---|
| L1 | `wm.encoder.weight_ih_l0` | (384, 64) | 128 × 64 = **8,192** | **24,576** |
| L2 | `ltm.encoder.weight_ih_l0` | (1536, 64) | 512 × 64 = **32,768** | **98,304** |
| L3 | `ltm.sem_to_h0.weight` | (512, 300) | 512 × 300 = **153,600** | **153,600** |

These match CENTRAL's expected counts exactly and are confirmed by live shapes.

---

## 5. Historical D-05 audit (read-only; nothing rerun)

`scripts/lesion/run_premeeting_d05.py`, target `wm_encoder`, checkpoint
`chigh_15e5_h512_s22_u3000`, 8 seeds, scopes afferent / recurrent / both.

**[VERIFIED]** Spec used (`run_premeeting_d05.py:70-75`):
`mask_granularity="scalar_links"`, `mask_sampling="exact_count"`,
`include_bias=False`, `exploratory=True`.

**[VERIFIED]** eligible **scalar coefficients**, read from the raw TSV:
ih **24,576** · hh **49,152** · both **73,728** — matching CENTRAL exactly.

**[VERIFIED]** dorsal-route repetition (`rep_exact_wm`, mean of 8 seeds),
reproduced from `outputs/lesion/premeeting_d05/d05_raw_realizations.tsv`:

| p | afferent | recurrent | both |
|---|---|---|---|
| .025 | .9999 | 1.0000 | 1.0000 |
| .050 | .9996 | .9999 | .9990 |
| .075 | .9994 | .9994 | .9969 |
| .150 | .9907 | .9899 | .9569 |
| .300 | .8754 | .6536 | .4092 |

All twenty values match CENTRAL's quoted table exactly.

**[VERIFIED]** The matched-absolute-count caveat, confirmed from the recorded
`n_masked_links`: afferent p=.15 and recurrent p=.075 both remove **3,686**
coefficients (.9907 vs .9994); afferent p=.30 and recurrent p=.15 both remove
**7,373** (.8754 vs .9899). At matched absolute count the **afferent** scope is
the more damaging one. Used here only as pre-existing sensitivity evidence; no
scope was chosen from these outcomes.

**[VERIFIED] Historical severities WERE nested.** `derive_generator` keys the
RNG on `(lesion_seed, target_id, role)` only — **not** on the removal fraction
(`lesion/masks.py:36-40`) — and the mask is
`torch.randperm(total, generator=g)[:n_remove]` (`:151`). The permutation is
therefore identical across p and the removed set is a prefix. Reproduced
synthetically on a 24,576-element lattice: the removed sets at
p = .025 ⊂ .05 ⊂ .075 ⊂ .15 ⊂ .30, with counts 614 ⊂ 1,229 ⊂ 1,843 ⊂ 3,686 ⊂ 7,373.

    HISTORICAL_D05_NESTED = YES (emergent from implementation)

This is a precedent, not a justification: V2 nesting is adopted as a
prospective dose-response design choice **[ADAPTATION]**, and is *not*
attributed to Ueno (§1.2).

**[VERIFIED]** `derive_seed` already uses SHA-256, not Python `hash()`
(`lesion/masks.py:26-33`), with role-separated streams "mask"/"noise".

---

## 6. Activation-noise semantics (historical, verified)

From `gate_x_lesion/sd.py:1-36`, which cites
`HISTORICAL_LESION_OPERATOR_AUDIT.md` §2 recovering these from executable code
at `3af0ef9`:

| claim | verified value | source |
|---|---|---|
| SD_DEFINITION | `float(a.std())` of flattened pooled intact activations | sd.py:7-8, :106 |
| SD_AXES | all axes pooled; `.flatten()` then `torch.cat` | sd.py:9, :112 |
| SD_DDOF | 1, PyTorch unbiased | sd.py:10, `SD_DDOF = 1` |
| SD_POPULATION | `deterministic_sample(range(N), 2048, seed=7)`, sorted; batch 256; one teacher-forced intact forward per batch | sd.py:11-12, `SD_N_ITEMS=2048`, `SD_SAMPLE_SEED=7`, `SD_BATCH_SIZE=256` |
| SD_TENSOR_SHAPE | scalar per (state, site), isotropic | sd.py:13 |
| distribution | `epsilon ~ Uniform(-1, +1)` | `gate_x_lesion/noise.py:8` |

**[VERIFIED]** the historical noise stream is already SHA-256 counter-mode
keyed by `(state_sha256, route, lesion_seed, item_id)` (`noise.py:86-115`) —
domain-separated and free of Python `hash()`. This is the precedent the V2 RNG
hierarchy follows.

**Not recomputed here.** Measuring SD for P1–P4 requires intact forwards
through the final models, which this pass forbids. The V2 implementation must
measure and freeze those intact scales prospectively, before any lesioned
output exists.

---

## 7. Connectivity p_max — prospective arbitration

CENTRAL's five-point case, examined point by point.

1. **Ueno's 7.5% maximum was near-inert in Lichtheim3.** Supported: at p=.075
   afferent dorsal repetition is .9994 — a 0.06% decrement. A dose-response
   study whose maximum dose does nothing cannot support a claim.
2. **p=.15 → .9907, p=.30 → .8754 on the afferent scope.** Verified above.
3. **Therefore .30 is nontrivial but non-ablative.** Supported by (2): a ~12.5
   point decrement, far from floor.
4. **Not selected from future P1–P4 curves.** True — the evidence is a 2026-09-10
   exploratory run on a *different* checkpoint (`...h512_s22_u3000`), predating
   the V7 cohort entirely.
5. **Logical-edge masking changes correlation structure, not nominal fraction.**
   **This premise is the weak one and I do not accept it as stated.** As shown
   in §4, scalar masking at p=.30 leaves only 34.3% of edges fully intact and
   degrades ~63% partially; logical-edge masking at p=.30 removes 30% of edges
   completely and leaves 70% pristine. Those are qualitatively different
   perturbations, so the D-05 curve does **not** transfer as a like-for-like
   prediction of behaviour at the same p.

   What *does* transfer is the quantity of weight mass removed: at p=.30 both
   schemes zero exactly 7,373 of 24,576 coefficients at L1. The evidence
   therefore still bounds the *dose*, even though it does not predict the
   *response*.

Two further honest caveats, neither fatal:

  * D-05 measured **L1 only**, on the dorsal encoder. There is no site-specific
    dose evidence for L2, and none at all for L3 (`sem_to_h0` is a Linear
    projection of a 300-d semantic vector, structurally unlike a GRU afferent
    matrix). p_max is transferred to L2/L3 by **design uniformity**, not by
    evidence. It must be reported that way.
  * The historical evidence comes from a non-V7 checkpoint.

**Verdict.** Points 1–4 survive; point 5 does not, but the argument does not
depend on it once restated as a dose bound rather than a response prediction.
Because the design reports all 15 levels, a curve that saturates earlier or
later than D-05 remains fully interpretable — no conclusion depends on p_max
being exactly right, only on its spanning mild-to-substantial damage without
flooring the model. 0.30 does that on the one site where evidence exists, and
is the only value in the pre-existing record with that property.

    CONNECTIVITY_P_MAX = 0.30            [ADAPTATION, prospective]

Recorded limitation: the D-05 transfer is approximate because granularity
differs; L2/L3 doses are uniform-by-design, not evidence-derived.

---

## 8. Source-derived fact vs Lichtheim3 adaptation

| item | status |
|---|---|
| noise + incoming-link removal, paired, 15 levels | **[VERIFIED]** Ueno |
| noise on layer output; removal of links into the layer | **[VERIFIED]** Ueno |
| Ueno ranges 0.01–0.15 and 0.5%–7.5% | **[VERIFIED]** Ueno |
| removal applies to inter-layer afference | **[VERIFIED]** Ueno |
| removal of recurrent self-connections | **[UNRESOLVED]** |
| mask nesting across severity | **[UNRESOLVED]** in Ueno; **[ADAPTATION]** in V2 |
| noise distribution Uniform(−1,+1) | **[INFERENCE]** from "range"; **[ADAPTATION]** in V2, inherited from the audited Lichtheim3 lineage |
| afferent-only scope (ih, not hh) | **[ADAPTATION]**, argued functionally in §3 |
| logical-edge granularity | **[ADAPTATION]**, argued in §4 |
| p_max = 0.30 | **[ADAPTATION]**, argued in §7; Ueno's own maximum was 0.075 |
| activation scaled to a measured per-site SD | **[ADAPTATION]**; Ueno used absolute values on [0,1] sigmoid units |
