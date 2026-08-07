# Serial-position error profile: WM versus LTM

A simplified two-route reading of the faithful serial-position analysis. FULL is
omitted because it sits on the floor across the whole word and adds nothing
readable; the three-route version is kept as
`yc4s_faithful_serial_position_all_routes`.

**The estimator is the verified faithful one, unchanged.** Applying the frozen
`compute.serial_position_tables` to the faithful subset at `route == "full"`
reproduces the frozen `faithful_figure2C_table.tsv` to a maximum absolute
difference of **8.8e-17** on the error rate and **exactly 0** on the counts
(`tables/faithful_figure2C_reproduction_check.tsv`). The WM and LTM panels use
that identical function.

**Four properties of the estimand, all of which matter for reading the curves:**

1. **Seeds are pooled, not averaged.** The denominator is item x seed rows in
   the (lexicality, length) cell, so every point is one pooled rate over the
   four checkpoints.
2. **Zip mismatch, not Levenshtein alignment.** Position i of the prediction is
   compared with position i of the target. Nothing is realigned.
3. **No insertion or deletion is ever realigned**, so a single early deletion
   makes every later position count as an error. Part of the rise is therefore
   mechanical accumulation, not independent evidence of late-position fragility.
4. **Post-EOS re-padding.** A prediction trimmed at EOS is re-padded to `<PAD>`
   up to the target length, which recovers Dager's blanking-after-EOS, so
   positions after an early stop count as mismatches.

**Display.** Pale points and thin lines in the background are the empirical
per-length values, retained so nothing is hidden behind the summary. The thick
line is the item-count-weighted PCHIP interpolation across lengths and is never
shown without those points.

**Labels.** Red and blue are the **faithful stimulus labels** real (800) and
pseudo (400), not exposure categories. 122 of the source-real items were never
in the training lexicon and 9 source pseudowords collide with it, so this figure
must not be read as trained versus untrained.

**What it shows.** WM stays near the floor across the whole word for both
stimulus classes, with only a mild late rise for pseudowords. LTM shows the
characteristic climb, far steeper for pseudowords. The dorsal route holds serial
position; the ventral route degrades along the word.
