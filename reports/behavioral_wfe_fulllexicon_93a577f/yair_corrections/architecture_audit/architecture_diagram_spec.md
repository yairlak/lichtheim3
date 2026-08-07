# Updated architecture diagram — specification

**Specification only. No diagram is rendered or committed by this pass.**

Every element below is taken from `architecture_audit.md` / `.json`, which were
generated from the committed `models/*.py` and the frozen checkpoint's own
`cfg_*` dicts and parameter shapes. The purpose of the update is to replace the
**stale biGRU / masked-mean depiction** with the path that is actually in the
weights.

---

## 1. What must change relative to the current diagram

| current (stale) | replace with (executable) |
|---|---|
| LTM encoder drawn as **bidirectional GRU** | **unidirectional** 1-layer GRU, `bidirectional=False`; no `_reverse` parameters exist in the checkpoint |
| LTM pooling drawn as **masked mean over all positions** | `pack_padded_sequence` → **last valid hidden state** `h[-1]`, mode `unigru_last_hidden` |
| pooled vector feeding a single projection | `to_semantic = Linear(128,128) → GELU → Linear(128,300)` |
| bank vector implied to reach the decoder | bank contributes **one scalar** (`confidence`) to the gate; **no bank vector reaches the decoder** |
| gate drawn per timestep | gate is **one scalar per word**, broadcast over all positions |
| (often omitted) | the two **premotor projections** and the **single shared** motor readout |

If the existing diagram shows a backward arrow over the input phonemes, a "mean"
box, or an arrow from the lexicon into the decoder, all three are wrong for these
checkpoints.

## 2. Nodes, with the labels to print

Print dimensions on the edges, not inside the boxes.

**Input / shared**

| id | label | note to print |
|---|---|---|
| `IN` | `enc_in = form + <EOS>` | lengths include EOS |
| `EMB` | `phon_embed` `Embedding(42, 64)` | **shared by both routes**; not a lexicon |

**Dorsal (WM) route** — draw in the neutral/left column

| id | label |
|---|---|
| `WM_ENC` | `nn.GRU(64 → 128)` · packed · last hidden |
| `WM_DEC` | `nn.GRU(64 → 128)`, `h0 = h_WM` |
| `WM_PRE` | `to_premotor` `Linear(128 → 128)` **linear, bias, no activation** |

**Ventral (LTM) route** — draw in the right column

| id | label |
|---|---|
| `LTM_ENC` | `nn.GRU(64 → 128)` · **unidirectional** · packed · `h[-1]` |
| `TO_SEM` | `Linear(128→128) → GELU → Linear(128→300)` |
| `S_HAT` | `s_hat` (300-d) — GloVe-aligned, phonology-derived |
| `SEM2H0` | `sem_to_h0` `Linear(300 → 128)` then `tanh` |
| `LTM_DEC` | `nn.GRU(64 → 128)`, `h0 = tanh(sem_to_h0(s_hat))` |
| `LTM_PRE` | `dec_to_premotor` `Linear(128 → 128)` **linear, bias, no activation** |

**Lexical field — a side branch, never in the main path**

| id | label |
|---|---|
| `NORM` | `q = normalize(s_hat)` — **separate tensor; does not modify `s_hat`** |
| `BANK` | frozen GloVe bank, 29,571 × 300, L2-normalised, **no gradient** |
| `CONF` | `confidence = max cosine(q, bank)` — **one scalar per item** |

**Gate and readout**

| id | label |
|---|---|
| `GATE` | `g = σ(2.0 · (confidence − 0.7))` — **scalar per word, 0 learnable parameters** |
| `MIX` | `premotor = g·LTM + (1−g)·WM` (128-d) |
| `MOTOR` | `motor.proj` `Linear(128 → 42)` — **one shared readout** |
| `LOGITS` | phoneme logits (42) |

## 3. Edges

```
IN → EMB
EMB → WM_ENC → WM_DEC → WM_PRE → MIX          [edge label: 128]
EMB → LTM_ENC → TO_SEM → S_HAT
S_HAT → SEM2H0 → LTM_DEC → LTM_PRE → MIX      [edge label: 128]
S_HAT → NORM → (with BANK) → CONF → GATE      [dashed, side branch]
GATE → MIX                                     [dashed, annotated "scalar, constant across the word"]
MIX → MOTOR → LOGITS
```

Plus the two **route-isolated** readout edges, drawn thin and dashed to show
they share one map:

```
WM_PRE  → MOTOR → wm_logits
LTM_PRE → MOTOR → ltm_logits
```

And the autoregressive feedback edge:

```
LOGITS → argmax → appended to dec_in → WM_DEC and LTM_DEC   [feedback arrow]
```

## 4. Annotations that must appear

1. **`bidirectional=False`** on `LTM_ENC`, with `h[-1]` on its outgoing edge.
2. **"no bank vector enters the decoder"** on the `BANK`/`CONF` branch.
3. **"one scalar per word, constant across all phoneme positions"** on `GATE`.
4. **"mixing happens on premotor states, before the shared readout"** on `MIX`.
5. **"one shared 128 → 42 affine map, evaluated at three points"** on `MOTOR`.
6. **"forced-length readout: exactly L tokens; on-time and late EOS are
   structurally unobservable"** as a footnote on the feedback edge.
7. `ventral_noise = 0.0`, `interference_noise = 0.0` — deterministic in these
   checkpoints — as a small provenance line.

## 5. Visual grammar

- **Two columns**, dorsal left / ventral right, converging on `MIX`; this is the
  Lichtheim claim and should be the first thing read.
- **Solid** boxes and arrows = tensors on the main computational path.
- **Dashed** = the lexical-field side branch and the route-isolated readouts, so
  no reader infers that the bank feeds the decoder.
- **One fill colour per route**, plus a third for shared components (`EMB`,
  `MOTOR`). Do **not** reuse the red/blue of the behavioural figures: those are
  reserved for real vs pseudo and must not encode anything else.
- Trapezoids for learnable projections, rectangles for recurrent modules,
  a cylinder for the frozen bank, a small diamond for the gate.
- The frozen bank should be visually distinct (grey fill, snowflake or lock
  glyph) to mark that it carries no gradient.

## 6. Provenance block to print on the figure

```
cohort fulllexicon_93a577f · training commit 93a577f · seeds 19-22
ltm_encoder_mode = unigru_last_hidden · premotor_dim = 128 · vocab = 42
gate: alpha = 2.0, threshold = 0.7 · noise = 0.0 · deterministic AR decoding
```

## 7. Suggested source format

Mermaid `flowchart LR` or Graphviz `dot`, kept as text next to the rendered
output so the diagram is diffable and regenerable. Whatever is chosen, the
rendered file should sit beside its source and a caption naming
`architecture_audit.md` as the evidence.

## 8. Acceptance checklist before rendering

- [ ] no backward arrow over the input phonemes anywhere
- [ ] no "mean" or "pool over all positions" box
- [ ] no arrow from the bank into either decoder
- [ ] gate shown as a scalar, not a per-timestep vector
- [ ] both premotor projections drawn explicitly, labelled linear
- [ ] exactly one motor box, with three incoming premotor edges
- [ ] mixing shown before the motor box
- [ ] dimensions on edges match `architecture_audit.json`
- [ ] red/blue not used for route or component identity

## 9. Open choice for Yair

Whether the diagram should show the **training-time** objective arrows
(alignment loss on `s_hat` against GloVe, the two route decoder losses, and the
gate usage prior) or stay purely an **inference-path** diagram. The inference
path alone is simpler and is what every analysis in this project measures;
adding the loss arrows would explain why `s_hat` is GloVe-aligned at all. My
recommendation is a clean inference diagram plus a small inset listing the four
loss terms and their weights, but that is a presentation decision, not a
technical one.
