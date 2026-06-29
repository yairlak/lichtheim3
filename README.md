# Dual-Route Connectionist Model of Word Repetition

A small, modular, **interpretable** PyTorch codebase that implements a dual-stream
sequence-to-sequence model of spoken word repetition, grounded in the
**Dual-Stream Model of Speech Processing** (Hickok & Poeppel) and
**Complementary Learning Systems** (CLS) theory (McClelland, McNaughton & O'Reilly).

The model takes a sequence of one-hot phonemes and re-articulates it through a
single shared output layer (the *Motor Cortex bottleneck*). Two functionally
distinct routes feed that bottleneck:

| Route | Stream | Memory type | Implemented in |
|-------|--------|-------------|----------------|
| **WM Buffer** | Dorsal | Non-parametric, activity-based, capacity-limited | `models/wm_route.py` |
| **LTM Lexicon** | Ventral | Parametric, weight-based, semantic | `models/ltm_route.py` |

The point of this repo is **not** state-of-the-art accuracy. It is to show that
two architecturally honest routes, when made to compete through a gate, reproduce
four classic **human** behavioral signatures *for free*:

1. **Primacy & recency** (U-shaped serial-position curve) in the WM route.
2. **Lexical neighborhood density effects** in the gate / error rate.
3. **Sonority-graded errors** (confusions track phonetic distance).
4. A **double dissociation**: the ventral route is frequency-sensitive but
   length-invariant; the dorsal route is frequency-invariant but length-sensitive.

---

## Summary: architecture & principles

**Architecture — phonemes in, two routes, one shared output:**

- **Dorsal WM buffer** (`models/wm_route.py`): a *non-parametric*, capacity-limited
  activity buffer that re-articulates the phonemes actually heard (primacy gain +
  recency leak + interference noise; only tiny position weights). Sub-lexical and
  content-agnostic — it can repeat anything once, but fragilely.
- **Ventral LTM lexicon** (`models/ltm_route.py`): a *parametric*
  encoder → semantic vector → decoder associative memory that maps a word-form to
  meaning and regenerates the form. Lexical and weight-based — robust on known
  words, useless on meaningless ones.
- A **familiarity gate** (`models/gating.py`) routes each item — known word →
  lexicon, novel form → buffer — and both speak through one shared **Motor Cortex**
  layer (`models/motor.py`).

**Why dual-route processing matters.** Neither route alone is enough. A purely
lexical system repeats familiar words but *cannot repeat novel words or nonwords*
(there is no meaning to look up); a purely buffer-based system can echo anything
but is fragile, capacity- and length-limited, and never benefits from experience.
Human repetition needs both, working in parallel and competing — these are the
**dorsal and ventral language pathways** (Hickok & Poeppel) and the
fast-buffer / slow-lexicon division of **Complementary Learning Systems** theory.
Crucially, having two routes predicts how the system *breaks*: focal damage
produces dissociable deficits that a single pathway cannot explain.

**Phenomena it aims to explain.**

- **Lexicality / generalization:** known words ride the lexicon; nonwords and
  novel forms fall back on the buffer.
- **Frequency × length double dissociation:** the lexicon is frequency-sensitive
  but length-invariant; the buffer is frequency-invariant but length-sensitive.
- **Working-memory signatures:** U-shaped primacy & recency, neighborhood-density
  effects, and sonority-graded (phonetically similar) errors.
- **Aphasia by lesion (Ueno et al. 2011):** damaging the dorsal pathway impairs
  nonword repetition while sparing words (conduction aphasia); damaging the
  ventral pathway does the reverse.

Training uses a realistic, bundled **30k-word** English lexicon (real ARPABET
pronunciations, frequency-ranked) with **log-frequency** weighting; see
[Lexicon & frequency](#lexicon--frequency).

---

## Why each route is built the way it is

**Dorsal / WM is deliberately crippled, not trained to be good.**
It is a fixed-capacity buffer of `K` slots. Each phoneme is *written* (the actual
one-hot vector, not a learned lexical code) with a **primacy gain** that decays
across input position, the buffer **leaks** every step (recency), and stored
traces accumulate **Gaussian interference**. The only learnable parameters are
tiny *position* embeddings used for read-out — there is no content-addressable
weight anywhere, so the buffer physically *cannot* memorize a lexicon. This is
what makes it frequency-invariant and length-sensitive.

**Ventral / LTM is a proper associative memory.**
A recurrent encoder maps the phoneme sequence to a 300-d semantic vector that is
aligned (cosine + MSE) to a lexical semantic vector for the word (a real **GloVe**
vector if available, otherwise a stable deterministic pseudo-vector); a recurrent
decoder regenerates the phoneme form from that single semantic vector. Because a
word lives at one point in semantic space regardless of how many phonemes it has,
the ventral route is length-invariant; because frequent words get more gradient
(log-frequency weighting), it is frequency-sensitive.

**The Motor Cortex is a single shared layer.**
Both routes emit a pre-motor vector in phoneme space; the gate mixes them and the
*same* `Motor` linear+softmax articulates the phoneme. This is the bottleneck
through which both streams must speak.

---

## The gate (error-suppression / lexicality routing)

The two routes are combined by a single gate (`models/gating.py`):

```
premotor = g · ltm_premotor + (1 − g) · wm_premotor
g = sigmoid(alpha · (lexical_confidence − 0.5))
```

`lexical_confidence` is the LTM route's max cosine similarity to a known lexeme.
A known word → confident LTM → `g→1` → the ventral route speaks and the WM buffer
is suppressed; a non-word → low confidence → `g→0` → the dorsal buffer carries the
trial. This single mechanism drives every result below (the neighborhood, the
generalization split, and the lesion dissociation).

> The project deliberately ships **one** gating hypothesis. Earlier
> density-competition and learned-routing variants live in the git history if you
> want to compare them.

---

## Quick start

```bash
pip install -r requirements.txt

# train the model and run all evaluations end to end
python run_all.py --epochs 8
```

Outputs (loss curve, generalization, serial-position curves, neighborhood plots,
sonority confusion matrix, the 2x2 dissociation plot, and the lesion figures) are
written to `outputs/`.

---

## Lexicon & frequency

The repo ships a **realistic English lexicon** in `data/lexicon_en.tsv`: the
**30,000 most frequent words** with real **CMU ARPABET** pronunciations and a
frequency rank (the top ~9k carry a measured frequency rank; the long tail is
real CMUdict∩hunspell words with continued Zipfian rank). It loads offline with
no downloads. Semantic targets for the ventral route use **GloVe** if you place
`glove.6B.300d.txt` in `data/`, otherwise a stable deterministic pseudo-vector.

Training samples words by **log frequency** (`data.lexicon.logfreq_weights`):
word frequencies are Zipfian, so the log compresses the huge high-frequency tail,
matching practice in the word-repetition modelling literature. Rebuild the
lexicon from your own frequency list with:

```bash
python -m data.build_lexicon_en  RANKED_WORDS.txt  30000
```

---

## Loss

```
L_total =  λ_rep  * CE(repetition, target)          # both routes -> motor
         + λ_align* (1 - cos(s_hat, glove)) + MSE    # ventral semantic alignment
         + λ_dec  * CE(ltm_decode, target)           # ventral form reconstruction
         + λ_wm   * CE(wm_only, target)              # keep WM honest (aux)
         + λ_gate * gate_regularizer                  # route-usage prior
```

See `losses.py` and `config.py:LossConfig` for the default weights and the
rationale behind each term.

---

## Training curve + unseen-word generalization

`run_all.py` now saves a `training_loss.png` (train vs held-out repetition loss)
and runs `evaluate/generalization.py`, which repeats **trained vs novel
(held-out) words** through each route in isolation — the protocol from
Chang et al., *Modelling Word Repetition with Deep Neural Networks*
(arXiv:2506.13450). The expected dissociation: the **ventral** route shows a
large trained-minus-novel gap (lexical knowledge does not transfer to
non-words), the **dorsal** buffer is flat across familiarity but falls off with
length, and the **gated** model tracks the better route item-by-item.

### Runnable NumPy twin (`numpy_demo/`)

Because the full model is PyTorch, a dependency-light NumPy twin is provided that
trains end-to-end with hand-written backprop and reproduces the same story on any
machine — no torch required:

```bash
# trains once and writes every figure into an organized figures/ tree:
#   figures/train/      training_loss, logfreq_curriculum
#   figures/eval/       generalization, nonword_by_length, primacy_recency
#   figures/ablation/   ablation_severity, ablation_dissociation, ablation_length
python -m numpy_demo.make_figures

# or run the pieces individually (write to outputs/numpy_demo/):
python -m numpy_demo.run_demo
python -m numpy_demo.ablation
```

It trains on the same realistic lexicon with log-frequency weighting (a frequent
core of real words), and keeps the three commitments: a parametric ventral
autoencoder through a tight semantic bottleneck, a non-parametric
capacity-limited dorsal copy buffer, and an error-suppression gate that routes by
lexical *familiarity*. Results (≈14 s each on CPU): trained words ~1.0, held-out
real words ~0.94 (the lexicon generalizes), nonwords carried best by the **gated**
model; and the Ueno-style lesion dissociation — **dorsal lesion** spares words
(~1.0) but abolishes nonwords (→0), **ventral lesion** does the reverse.

---

## Lesion / ablation studies (Ueno et al. 2011, "Lichtheim 2")

Following Ueno, Saito, Rogers & Lambon Ralph (2011, *Neuron* 72:385–396),
damage is simulated by **removing a proportion of a pathway's units/links and
adding noise over its output**, titrating severity and averaging over random
"patients" (seeds), reported as mean ± SE.

```bash
python -m numpy_demo.ablation     # writes outputs/numpy_demo/ablation_*.png
```

Lesioning each route reproduces the classic **double dissociation**:

| Lesion | Word (trained) repetition | Nonword (novel) repetition | Aphasia analogue |
|--------|---------------------------|----------------------------|------------------|
| **Dorsal (WM/iSMG)** | spared (~1.0) | abolished (→ 0) | conduction aphasia |
| **Ventral (LTM/vATL)** | abolished (→ 0) | spared (~0.46) | lexical-semantic (SD-like) |

This mirrors Ueno et al.'s Figures 3–4 (and Fig 7: a ventral-only system cannot
repeat nonwords). Figures produced: `ablation_severity.png` (two-panel severity
curves), `ablation_dissociation.png` (the dissociation at a fixed severity), and
`ablation_length.png` (a dorsal lesion erases nonword repetition at every
length). The same study runs on the full PyTorch model via
`evaluate/ablation.py` (lesions applied with forward hooks; included in
`run_all.py`).

> Ueno, T., Saito, S., Rogers, T. T., & Lambon Ralph, M. A. (2011). Lichtheim 2:
> Synthesizing aphasia and the neural basis of language in a neurocomputational
> model of the dual dorsal-ventral language pathways. *Neuron, 72*(2), 385–396.

---

## Layout

```
lichtheim3/
├── config.py                # all hyperparameters as dataclasses
├── run_all.py               # train + evaluate end-to-end
├── train.py                 # training loop (+ loss-curve plot)
├── losses.py                # L_total assembly
├── data/
│   ├── phonemes.py          # ARPABET inventory + sonority/phonetic features
│   ├── lexicon.py           # bundled-lexicon loader + log-frequency weights
│   ├── lexicon_en.tsv       # 30k real words + ARPABET + frequency rank
│   ├── build_lexicon_en.py  # (re)build lexicon_en.tsv from a frequency list
│   └── dataset.py           # Dataset + log-frequency sampler + collate
├── models/
│   ├── wm_route.py          # dorsal buffer
│   ├── ltm_route.py         # ventral encoder/decoder
│   ├── motor.py             # shared motor bottleneck
│   ├── gating.py            # error-suppression gate
│   └── dual_route.py        # top-level model
├── evaluate/
│   ├── hooks.py             # activation capture
│   ├── primacy_recency.py   neighborhood.py   sonority.py
│   ├── dissociation.py      # frequency × length
│   ├── generalization.py    # trained vs unseen words
│   └── ablation.py          # Ueno-style lesions
├── numpy_demo/              # runnable, torch-free twin (+ ablation)
└── utils/
    ├── seed.py
    └── plotting.py
```
