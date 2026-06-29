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
| **WM serial-recall net** | Dorsal | Parametric recurrent encoder→decoder, capacity-limited | `models/wm_route.py` |
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

- **Dorsal WM route** (`models/wm_route.py`): a *parametric* recurrent
  serial-recall network (encoder→decoder, after Botvinick & Plaut 2006). It reads
  the sequence into a bounded recurrent state and re-articulates it, learning the
  auditory→motor map and phonotactics. Sub-lexical and generalizing — it repeats
  novel words and nonwords — but capacity/length-limited and noisy.
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

**Dorsal / WM is a parametric recurrent serial-recall network** (Botvinick &
Plaut, 2006). A recurrent **encoder** reads the phoneme sequence into a single,
bounded recurrent state; a recurrent **decoder** re-articulates it. Because it is
trained on the repetition task (and additionally on a frequency-flat stream of
pronounceable pseudowords — `config.TrainConfig.dorsal_pool_size`), it learns the
auditory→motor map
and the language's phonotactics, and acquires the *general* serial-recall
computation — so it generalizes to novel words and nonwords. Its frailties are
emergent, not hand-wired: packing the whole sequence into a fixed-size state plus
interference noise gives **capacity / length limits** and the **serial-position
curve**. It is **lexical-frequency-invariant** because a generalizing route gains
nothing from word identity — frequency-invariance now comes from *what it learns*,
not from having no weights.

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

# train the model (PyTorch) and run every evaluation end to end
python run_all.py                                  # quick run (small lexicon)
python run_all.py --epochs 15 --max_words 8000     # fuller run
```

This writes the **whole figure set** — loss curve, generalization, serial-position
curve, neighborhood, sonority confusion matrix, the frequency × length
dissociation, and the lesion figures — into an organized tree:

```
figures/train/   figures/eval/   figures/ablation/   figures/summary.json
```

For real GloVe semantic targets, fetch them once:

```bash
bash data/get_glove.sh        # -> data/glove.6B.300d.txt (auto-detected)
```

Tip: run `python -m tests.smoke_test` first — it trains a tiny model and runs the
evaluations in a few seconds, catching any environment issue before a full run.

---

## Lexicon & frequency

The repo ships a **realistic English lexicon** in `data/lexicon_en.tsv`: the
**30,000 most frequent words** with real **CMU ARPABET** pronunciations and a
frequency rank (the top ~9k carry a measured frequency rank; the long tail is
real CMUdict∩hunspell words with continued Zipfian rank). It loads offline with
no downloads. Semantic targets for the ventral route use **GloVe** if you place
`glove.6B.300d.txt` in `data/`, otherwise a stable deterministic pseudo-vector.

Frequency enters through **exposure**: words are *presented* in proportion to
their **log frequency** (`data.lexicon.logfreq_weights`) — the PyTorch dataset's
sampler draws words by these weights — rather than re-weighting the loss. Word
frequencies are Zipfian, so the log compresses
the huge high-frequency tail (raw frequency would mean essentially only ever
seeing *the/of/and*), matching practice in the word-repetition literature.
Rebuild the lexicon from your own frequency list with:

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

`run_all.py` saves a `training_loss.png` (train vs held-out repetition loss) and
runs `evaluate/generalization.py`, which repeats **trained vs novel (held-out)
words** through each route in isolation — the protocol from Chang et al.,
*Modelling Word Repetition with Deep Neural Networks* (arXiv:2506.13450). The
expected pattern is a dual-route **crossover**: the **ventral** route wins on
trained words but fails on novel forms (lexical knowledge does not transfer to
non-words), the **dorsal** route generalizes to novel forms but falls off with
length, and the **gated** model tracks the better route item-by-item.

---

## Lesion / ablation studies (Ueno et al. 2011, "Lichtheim 2")

Following Ueno, Saito, Rogers & Lambon Ralph (2011, *Neuron* 72:385–396),
`evaluate/ablation.py` simulates damage by **removing a proportion of a pathway's
units and adding noise over its output**, titrating severity and averaging over
random "patients" (seeds), reported as mean ± SE (figures in
`figures/ablation/`). Lesioning each route reproduces the classic
**double dissociation**:

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
│   ├── get_glove.sh         # fetch GloVe 6B 300d into data/
│   └── dataset.py           # Dataset + log-frequency sampler + pseudoword pool
├── models/
│   ├── wm_route.py          # dorsal recurrent serial-recall net
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
├── tests/smoke_test.py      # fast end-to-end sanity check
└── utils/
    ├── seed.py
    └── plotting.py
```
