# Stable-zero checkpoint selection — bottom line

Train autoregressive error count on the 29,571-word training lexicon (FULL
route) at every evaluated epoch, for the four full-lexicon seeds. The evaluation
grid is complete and regular: epochs 105-200 in steps of 5, 20 evaluations per
seed, with no gaps and nothing inferred. Filled black-edged markers are
zero-error evaluations; shaded bands are streaks of at least two consecutive
zeros.

**Rule illustrated.** The selected checkpoint is the **first** checkpoint of a
qualifying streak, but training can only stop once the **Xth** consecutive zero
has actually been observed - the two epochs are different and both are
annotated on seed 22.

**Longest zero streaks:** seed 19 = 6 (155-180), seed 20 = 2 (130-135),
seed 21 = 0, seed 22 = 13 (140-200).

**Criterion outcomes:** X = 2 -> 3/4 seeds pass; X = 3 -> 2/4; X = 5 -> 2/4.

Two things worth saying out loud at the meeting. First, **raising X would not
have changed a single selected checkpoint** in this cohort - seeds 19 and 22
keep 155 and 140 at every X - it only moves the earliest epoch at which you
could have stopped. Second, **seed 21 never reaches zero at all**; it was
selected by the fallback rule (earliest checkpoint with the minimum error count,
1 error at epoch 145) and passes no X.

Read from the audited trajectories only; nothing is recomputed and no training
or inference was run.
