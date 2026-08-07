# LTM-only success on the 391 novel pseudowords

Population: the 391 `NOVEL_PSEUDOWORD` items, **LTM route only**, classified by
how many of the four seeds reproduce the form exactly. The three classes are
exhaustive and mutually exclusive by construction.

**Panel A.** 201 items (51.4 %) are reproduced exactly by all four seeds,
173 (44.3 %) by one to three seeds, and 17 (4.3 %) by none. The LTM route
therefore reproduces a majority of novel pseudowords perfectly and consistently -
it is not a route that simply fails on unfamiliar forms.

**Panel B.** Target length separates the groups more sharply than anything else
available: mean 5.10 phonemes for always-successful, 7.01 for mixed, 8.53 for
always-failed. Every point is one item; the red bar is the group mean.

**Panel C.** Lexical confidence differs in the same direction but weakly:
0.562 / 0.540 / 0.507. **The gate is not shown as a separate panel because it is
a deterministic monotone function of this same variable**
(`gate = sigmoid(2.0 x (lexical_confidence - 0.7))`), so plotting both would
present one variable twice as if it were two pieces of evidence. Group gate means
are in `tables/ltm_pseudoword_feature_summary.tsv`, flagged as auxiliary.

## What this figure does NOT do

**It does not identify why LTM succeeds on some novel pseudowords and not
others.** It answers "which items succeed?" and nothing more. Length and lexical
confidence are **associated descriptors** of the success groups, not causes and
not a mechanism: they are themselves correlated with each other, the groups were
formed from the outcome, and no intervention or controlled comparison was made.
A descriptive difference in a group formed by its own outcome cannot establish
what produced that outcome.

**Lexical confidence is not retrieval, and no bank vector reaches the decoder.**
Lexical confidence is the top-1 cosine similarity between the item's `s_hat` and
the frozen semantic bank. That similarity is used only to compute the FULL gate
scalar. **The top-1 semantic-bank neighbour is never injected into the decoder**:
the LTM decoder is initialised from `h0 = tanh(sem_to_h0(s_hat))` using the raw
`s_hat`, and the normalised query used against the bank is a separate tensor that
does not modify `s_hat`. A higher-confidence pseudoword is therefore not one that
had a stored form supplied to it.

**No lexicalization claim is made**, and no new feature was computed for this
figure.

**Missing validated measures.** The three features that would sharpen this
comparison - phonotacticity, distance to the training lexicon, and
suffix/phonemic complexity - **do not exist as validated documented features**
for WFE items anywhere in `scripts/`, `reports/`, `outputs/` or `docs/`. They are
recorded as `UNAVAILABLE_VALIDATED_MEASURE` in
`tables/ltm_pseudoword_unavailable_measures.tsv`. No proxy was invented.

**Limitation.** Length, confidence and success are mutually entangled; the
always-failed group has 17 items.
