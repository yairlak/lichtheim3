# A09 / A10 / A11 — audit before formatting

**Audited 2026-08-04 from the tracked analysis matrix, the tracked documentation
and the validated outputs themselves, before any formatting work.**
Machine-readable twin: `_control/a09_a10_a11_audit.json`.

All three rows are **faithful** replications of Dager et al. on
`FAITHFUL_WFE_ALL` (1,200 items, FULL route, original source labels). Every one
already has an authoritative, validated numerical table. **Nothing here requires
new scientific computation**; the outstanding work is rendering and provenance.

---

## Shared facts

| | |
|---|---|
| matrix status (all three) | `NEEDS_FORMATTING_ONLY` |
| regime | `FAITHFUL_WFE_ALL`, n = 1,200 |
| route scope | FULL only — no route factor exists in any of the three |
| seeds | 19, 20, 21, 22 (2A and 2B per seed; 2C pooled over seeds by design) |
| source directory | `outputs/behavioral_wfe_fulllexicon_93a577f/behavioral_analysis/faithful_replication/` |
| source README | `README_faithful_replication.md` (frozen parameters recorded there) |
| historical driver | **not present in the tracked package** — no tracked `.py` references `figure2A`, `figure2B` or `figure2C` |
| existing formats | PNG + PDF |
| missing formats | **SVG** |
| missing artefacts | **standalone caption files**; no copy under `reports/` |

The source labels are **WFE source labels, not training exposure**: 122 of the
800 source-real words were never in the Lichtheim3 training lexicon and 9 source
pseudowords collide with it. This caveat must appear in every caption.

---

## A09 — Faithful Figure 2A

| field | value |
|---|---|
| exact title | Faithful Figure 2A |
| metric | `raw_edit_distance` |
| source table | `faithful_figure2A_table.tsv` — 96 rows = 4 seeds × 2 lexicality × 2 morphology × 6 lengths |
| columns | `seed, lexicality, morphology, length, n_items, mean_edit_distance` |
| source figures | `faithful_figure2A.png`, `faithful_figure2A.pdf` |
| item counts | real 400 complex + 400 simple; pseudo 200 complex + 200 simple |
| plotting TSV exists? | **yes** — the table *is* the plotting table |
| code promotion needed? | **yes** — rendering only |

Existing numerical estimates (mean edit distance, averaged over seeds and
morphologies): lengths 3, 4, 5 are exactly **0.0000** for both lexicalities;
length 7 real 0.0020 / pseudo 0.0000; length 8 real 0.0078 / pseudo 0.0000;
length 9 real 0.0199 / pseudo **0.0947**.

Formatting deficiencies: no SVG; no caption; no exposure caveat attached to the
figure; not reachable from the tracked package.

**Classification: `PROMOTE_AND_FORMAT_EXISTING_ANALYSIS`.**

Non-regression check: every one of the 96 `mean_edit_distance` values, and the
`n_items` column, must match the source table exactly; the rendered figure must
be driven by that table and by nothing else.

---

## A10 — Faithful Figure 2C

| field | value |
|---|---|
| exact title | Faithful Figure 2C |
| metric | positional error rate (zip-mismatch, **no Levenshtein alignment**) |
| source table | `faithful_figure2C_table.tsv` — 72 rows = 2 lexicality × 6 lengths × position |
| columns | `lexicality, length, position_1based, relative_position, n_items_x_seeds, error_rate_per_item` |
| source figures | `faithful_figure2C.png`, `faithful_figure2C.pdf` |
| seed handling | pooled across seeds by design (`n_items_x_seeds`) |
| plotting TSV exists? | **yes** |
| code promotion needed? | **yes** — rendering only |

Existing numerical estimates: overall mean error rate per item — pseudo
0.00295, real 0.00129. At length 9 the profile rises at the late positions
(pseudo 0.0303 / 0.0227 / 0.0417 at relative positions 0.750 / 0.875 / 1.000;
real 0.0116 / 0.0145 / 0.0087).

Formatting deficiencies: no SVG; no caption; the zip-mismatch method and the
"no alignment" property are not stated on the figure.

**Classification: `PROMOTE_AND_FORMAT_EXISTING_ANALYSIS`.**

Non-regression check: all 72 `error_rate_per_item` and `n_items_x_seeds` values
must match exactly. **This is a distinct estimand from the Sprint-1 clean
serial-position figure** (different item set, different pooling) and the two are
never merged.

---

## A11 — Faithful feature importance (Dager Figure 2B)

| field | value |
|---|---|
| exact title | Faithful feature importance |
| metric | `raw_edit_distance` |
| source table | `faithful_figure2B_feature_importance.tsv` — 12 rows = 4 seeds × 3 features |
| columns | `seed, feature_transformed, ridge_coefficient, permutation_importance_mean, signed_importance` |
| features | `cont__Length`, `cat__Lexicality_real`, `cat__Morphology_simple` |
| source figures | `faithful_figure2B.png`, `faithful_figure2B.pdf` |
| frozen parameters | Ridge α = 1.0; 80/20 split `random_state=42`; `permutation_importance(n_repeats=100, random_state=42)`; no interactions; no p-values |
| sign convention | **historical `signed_importance`**, retained unchanged |
| plotting TSV exists? | **yes** |
| code promotion needed? | **yes** — rendering only |

Existing numerical estimates (permutation importance mean): `cont__Length`
0.0299 / 0.0027 / 0.0255 / 0.0017 for seeds 19 / 20 / 21 / 22 — **length leads
in all four seeds**; `cat__Lexicality_real` 0.0036 / −0.0006 / 0.0196 / −0.0012;
`cat__Morphology_simple` 0.0030 / 0.0004 / −0.0027 / 0.0005.

Formatting deficiencies: no SVG; no caption; the caption must carry the frozen
parameters, the historical sign convention, and the statement that this is
**not** the adapted A15 and is never pooled with it or drawn on a common
quantitative axis.

**Classification: `PROMOTE_AND_FORMAT_EXISTING_ANALYSIS`.**

Non-regression check: all 12 `ridge_coefficient`, `permutation_importance_mean`
and `signed_importance` values must match exactly. **The estimand is not
recomputed** — the release renders the stored table. Ridge α, the split policy,
the permutation policy and the sign convention are preserved by construction,
because no model is refitted.

---

## Verdict

| row | classification | new scientific computation? |
|---|---|---|
| A09 | `PROMOTE_AND_FORMAT_EXISTING_ANALYSIS` | **no** |
| A10 | `PROMOTE_AND_FORMAT_EXISTING_ANALYSIS` | **no** |
| A11 | `PROMOTE_AND_FORMAT_EXISTING_ANALYSIS` | **no** |

No row is `STOP_REQUIRES_NEW_ANALYSIS`. The release renders the three
authoritative tables through the tracked package, adds SVG and standalone
captions, and records provenance. **The historical outputs under `outputs/`
remain untouched.**
