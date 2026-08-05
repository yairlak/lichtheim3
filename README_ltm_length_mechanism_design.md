# Lichtheim3 — LTM length-effect mechanism: study design

**Phase M0 only.** This document states the target phenomenon, the candidate
hypotheses and the confirmatory/exploratory split. **No mechanistic result
exists yet, and none may be inferred from this document.**

The behavioural WFE analysis is finished, validated, versioned and closed at
`c29de360786afd26f548dffe37812928ee34f6af`. It is **not** recomputed, revised,
replaced or reinterpreted here.

---

## 1. Target phenomenon

The interaction **route × phoneme length × phonological exposure**, as
established by the closed behavioural analysis:

**Trained phonological forms** (`TRAINED_REAL_EXACT`, n = 671)
- FULL and WM are at **exact ceiling** — zero errors on every item in every
  seed, so their length slope is structurally 0.000.
- LTM has only a **very small** length slope (0.0007–0.0177 per phoneme).

**Untrained real words and genuinely novel pseudowords**
(`UNTRAINED_REAL` n = 122, `NOVEL_PSEUDOWORD` n = 391)
- LTM develops a **large** length effect (0.197–0.256 per phoneme on clean
  pseudowords).
- WM remains **much more robust** (0.007–0.037).
- FULL remains **close to ceiling** (0.003–0.020).

The primary contrast LTM − WM on clean pseudowords is positive in all four
seeds: +0.246, +0.183, +0.184, +0.205.

**The phenomenon is conditional.** It is *not* "LTM always has a length effect".
Any mechanistic account must explain why the ventral route's length dependence
appears for unfamiliar forms and is nearly absent for trained ones.

---

## 2. Candidate hypotheses

These are **candidates to be discriminated**, not a ranking. Several may hold
jointly; the behavioural data alone do not separate them.

**H1 — Lexical attraction.** LTM output for unfamiliar forms is pulled toward
phonologically or semantically neighbouring **training** words, and longer items
offer more opportunity for such capture.
*Falsifiable against*: predictions that are not closer to any bank neighbour
than chance, once target–neighbour proximity is controlled.

**H2 — Representational bottleneck.** `s_hat` is a fixed 300-d point regardless
of length; for unfamiliar forms it carries insufficient phonological detail, and
the deficit grows with the number of phonemes to be regenerated.
*Falsifiable against*: length-independent degradation of `s_hat`-derived
quantities.

**H3 — Autoregressive amplification.** Local per-step predictions are only
mildly degraded, but feeding back a wrong generated token compounds the error
across the remaining suffix, so the burden grows with remaining length.
*Falsifiable against*: gold-prefix per-step accuracy that already falls with
position at the same rate as the AR trajectory.

**H4 — Decoder-specific limitation.** The LTM decoder GRU, initialised once from
`tanh(sem_to_h0(s_hat))`, cannot sustain a long phoneme sequence irrespective of
how good `s_hat` is.
*Falsifiable against*: intact long-sequence regeneration when `s_hat` is
well-formed.

**H5 — Exposure / familiarity.** The determining variable is training exposure
to the phonological form, not lexical status. Untrained real words should behave
like pseudowords.
*Partially supported already*: under LTM, untrained real words (0.549 mean edit
distance) pattern with novel pseudowords (0.601), far from trained real words
(0.024). This is descriptive and does **not** identify a mechanism.

**H6 — Gate / blending compensation.** FULL stays near ceiling because the gate
routes unfamiliar items toward WM. This explains **FULL's robustness**; it does
**not** explain LTM's own length effect, and must not be confused with one.

**H7 — Evaluation or implementation artifact.** The forced-length readout, the
EOS convention, `editops` tie-breaking, or route-isolation semantics generate or
inflate the effect.
*Status after M0*: the audit found the three routes genuinely isolated, each
generating its own prefix, with identical forced-length and EOS handling, no
noise, no dropout and no sampling. **H7 is not eliminated** — the forced-length
horizon still makes terminal insertions and on-time/late EOS unobservable — but
no isolation or determinism defect was found.

---

## 3. Confirmatory versus exploratory

**Confirmatory** (pre-registered before any mechanistic result is inspected):
- first-error hazard as a function of target position, by route × exposure ×
  length, using the three separately pre-registered first-error definitions;
- post-divergence suffix burden, stratified by divergence type;
- position-level route comparison **under a common prefix only**;
- lexical-attraction categories with target–neighbour proximity controlled.

**Exploratory** (hypothesis-generating, labelled as such, never reported as
confirmatory):
- representational geometry of `s_hat` and the encoder hidden states;
- entropy and margin trajectories;
- fixed-prefix counterfactual gate mixtures;
- neighbourhood-density effects.

---

## 4. Primary datasets, metrics and seed policy

**Datasets** — the canonical 1,200-item WFE set with its frozen exposure
classification. Primary groups: `TRAINED_REAL_EXACT`, `NOVEL_PSEUDOWORD`.
Exposure extension: `UNTRAINED_REAL`. Other categories retained in raw outputs
but outside the confirmatory contrast.

**Metrics** — first-error position and hazard; post-divergence mismatch and
suffix Levenshtein burden; per-step target rank, signed margin and entropy;
route agreement under a common prefix; lexical-attraction categories.

**Seeds** — all four (19, 20, 21, 22) are primary. **Seed 21 is never excluded.**
Exact-zero sensitivity (19, 20, 22) is **secondary only** and never replaces the
four-seed result.

---

## 5. Standing constraints

**No architectural intervention before a causal decision.** No architecture
change, no retraining, no weight modification, no checkpoint conversion may be
proposed or performed until a mechanism has actually been identified and its
causal status established.

**Poor LTM pseudoword repetition alone does not justify a model change.** The
LTM route is *designed* to be a lexical-semantic memory; weak regeneration of
phonological forms absent from the training lexicon may be the intended
behaviour of that design rather than a defect. Establishing that something is
wrong requires more than showing the effect exists.

**Interpretive constraints carried from the audit:**
- The gate is a **deterministic transformation of confidence**; they are one
  measurement, not two independent lines of evidence.
- The GloVe-aligned bank is a lexical-neighbourhood readout. It is **not**
  conceptual comprehension.
- A substitution error is **not** evidence of lexical attraction by itself.
- Word-level route rescue is co-occurrence, **not** a demonstrated causal
  contribution.
- `ON_TIME_EOS` and `LATE_EOS` remain structurally unobservable under the frozen
  readout, and terminal insertions are underestimated.

---

## 6. Current status

M0 deliverables are complete: git and artifact verification, commit role map,
architecture audit, evaluation audit, output inventory, variable availability,
and the design of a minimal consolidated instrumented pass.

**No instrumented pass has been executed. No M1/M2/M3/M5 result exists.**
See `outputs/length_effect_mechanism_93a577f/m0/decision_log.md`.
