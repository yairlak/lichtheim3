# Faithful serial-position error profile by route

**This does not replace the existing faithful Figure 2C.** Those files are
read-only in this pass; these are new files under the `yc4_` prefix.

**Method, recovered and verified rather than assumed.** The original producing
driver is not in the current tree, but its logic was promoted verbatim into
`compute.serial_position_tables` and `compute.zip_mismatch_positions`. Applying
that frozen function to the faithful subset at `route == "full"` reproduces the
frozen `faithful_figure2C_table.tsv` with **maximum absolute difference
8.8e-17 on the error rate and exactly 0 on the counts** - see
`tables/faithful_figure2C_reproduction_check.tsv`. The by-route extension is
written only if that gate passes, and the WM and LTM panels use the identical
function.

**Precise positional estimand.** For each (lexicality, length) cell and each
1-based position i: the numerator is the number of item x seed rows whose
predicted symbol at i differs from the target symbol at i under **zip alignment
with the prediction re-padded to the target length**; the denominator is the
number of item x seed rows in that cell. This is Dager's `Error_Indices` and is
**not** a Levenshtein alignment - no insertion or deletion is ever realigned, so
a single early deletion makes every later position count as an error. A trimmed
prediction is re-padded to `<PAD>`, which reproduces Dager's blanking after EOS.
The four seeds are **pooled**, not averaged. Relative position is
`(i-1)/(L-1)`; lengths below 2 are skipped. faithful zip-mismatch positions (Dager Error_Indices), relative position (index-1)/(length-1), PCHIP interpolation to 100 points, item-count weighted across lengths; no Levenshtein alignment is used.

**Display.** Faint points and thin lines are the empirical per-length values.
The thick curve is the item-count-weighted PCHIP interpolation and is shown only
alongside the points that produced it, never on its own.

**Labels.** Red and blue are the **faithful stimulus labels** real/pseudo
(800/400), not exposure categories. 122 of the 800 source-real items were never
in the training lexicon and 9 source pseudowords collide with it, so this figure
must not be read as trained versus untrained.

**What the panels show.** The rising profile is essentially an LTM phenomenon.
FULL and WM stay near the floor across the whole word for both stimulus classes;
LTM shows the characteristic climb, and it is far steeper for pseudowords. Under
zip alignment part of that climb is mechanical: once a position is wrong, later
positions are compared against a shifted target, so error accumulates by
construction. That is a property of the faithful estimand, not a separate
finding.
