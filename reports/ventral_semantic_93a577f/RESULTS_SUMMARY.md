# Ventral / semantic probe — Phase B results

Frozen cohort: the **X=5 stable-zero** checkpoints only — **seed 19 / epoch 155**
and **seed 22 / epoch 140**. Raw `s_hat` extracted for all **29,571** training
words per checkpoint, encoder-only, `model.eval()`, `torch.inference_mode()`, no
gradient, no decoder, deterministic (re-run of a batch is bit-identical).
GloVe coverage is complete: `n_glove_found = 29571`, `n_glove_fallback = 0`.

## Target semantic identification (Figure 2)

| | seed 19 / e155 | seed 22 / e140 |
|---|---|---|
| **top-1 accuracy** | **2.65 %** | **2.54 %** |
| **top-5 accuracy** | **5.98 %** | **5.69 %** |
| **median target cosine** | **0.298** | **0.294** |
| **median margin** (target − best incorrect) | **−0.223** | **−0.224** |
| median target rank (of 29,571) | 1,504 | 1,581 |
| target within rank ≤ 100 | 19.3 % | 18.6 % |
| median best-incorrect cosine | 0.527 | 0.527 |

Chance top-1 = 1/29,571 = **0.0034 %**.

## PCA (Figure 1)

One 2-component PCA fitted on the L2-normalised GloVe bank **only**, then applied
unchanged to GloVe and to each seed's normalised `s_hat`.
**PC1 = 5.69 %**, **PC2 = 2.56 %**, **PC1 + PC2 = 8.25 %** of GloVe variance.

## What the two figures show

`s_hat` carries a real, reproducible amount of target-specific semantic
information: median cosine to the word's own GloVe vector is ≈ 0.30, top-1
retrieval is ≈ 2.6 % against a chance level of 0.0034 % (roughly 780× chance),
and the per-word target ranks agree closely across the two independently trained
seeds (Spearman ρ = 0.80). At the same time the ventral representation is
**not** an identification-grade semantic code: for a typical word some *other*
GloVe vector is nearer than its own (median margin ≈ −0.22, median target rank
≈ 1,500), so fewer than 3 % of words are retrieved correctly at top-1.
Figure 1 adds a structural observation: under a projection fitted on GloVe
alone, the normalised `s_hat` cloud is far more dispersed than the GloVe cloud
it is trained to match, occupying a wide annulus around the comparatively
compact GloVe distribution, and the 20 prospectively selected words show
systematic displacement from their targets in both seeds.

Two cautions on reading these panels. The 2-D view captures only 8.25 % of GloVe
variance, so Figure 1 is a projection, not a faithful picture of the 300-d
geometry, and apparent distances there should not be quantified. And these
numbers describe `s_hat` on **trained words only** — nothing here speaks to
pseudowords, and no causal claim is made about what produces the gap.

## Files

- `figures/fig1_glove_s_hat_pca_alignment.{png,pdf,svg}`
- `figures/fig2_target_semantic_identification.{png,pdf,svg}`
- `tables/per_word_identification.tsv` — 59,142 rows (2 seeds × 29,571 words):
  `word, lexicon_index, seed, epoch, target_cosine, best_wrong_cosine,
  target_rank, top1, top5, margin`
- `tables/pca_annotated_pairs.tsv` — the 20 annotated words per seed with GloVe
  and `s_hat` PC coordinates and their displacement
- `analysis_metadata.json` — normalisation, PCA fit population, checkpoints,
  annotation rule, input hashes
- `../../outputs/ventral_semantic_93a577f/` — raw `s_hat` arrays, normalised
  GloVe bank, word order, extraction provenance

**Annotation selection rule (prospective, model-independent):** 20 words at
evenly spaced positions in the lexicon's frequency-rank ordering,
`index = round(linspace(0, 29570, 20))`. No RNG, no model output involved.
Frequency rank correlates with training exposure, which is a stimulus property
known before any model was run; it is not a function of alignment or retrieval
performance.
