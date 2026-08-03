# WFE behavioral analysis matrix

Planned and completed analyses for the full-lexicon cohort 93a577f, in sprint
order. The machine-readable twin is
`reports/behavioral_wfe_fulllexicon_93a577f/analysis_matrix.tsv`.

Status values: `ALREADY_VALIDATED`, `NEEDS_FORMATTING_ONLY`,
`NEEDS_COMPUTATION`, `OPTIONAL`, `OUT_OF_SCOPE`.

## Sprint order

1. reproducibility and provenance (A01–A03)
2. formatting existing figures (A04–A11)
3. morphology (A12–A13)
4. frequency (A14)
5. faithful feature importance (A11)
6. error taxonomy and premature EOS (A16–A18)
7. adapted analyses (A15)
8. optional SSP (A19)

Nothing beyond priority 2 was computed in Sprint 1.

## Matrix

| id | analysis | regime | n | routes | primary metric | faithful/adapted | status | prio | next action |
|---|---|---|---|---|---|---|---|---|---|
| A01 | Script versioning and code promotion | n/a | n/a | n/a | n/a | n/a | **ALREADY_VALIDATED** | 1 | none |
| A02 | Production manifest closure | n/a | 41 files | n/a | sha256 | n/a | **ALREADY_VALIDATED** | 1 | none |
| A03 | Documentation and portable provenance | n/a | n/a | n/a | n/a | n/a | **ALREADY_VALIDATED** | 1 | none |
| A04 | Clean length curves | LICHTHEIM_CLEAN | 1062 | full;wm;ltm | raw_edit_distance | adapted | **ALREADY_VALIDATED** | 2 | none |
| A05 | Clean length slopes and LTM-WM contrast | LICHTHEIM_CLEAN | 1062 | full;wm;ltm | raw_edit_distance | adapted | **ALREADY_VALIDATED** | 2 | none |
| A06 | Clean serial position | LICHTHEIM_CLEAN | 1062 | full;wm;ltm | positional error rate | faithful | **ALREADY_VALIDATED** | 2 | none |
| A07 | Gate and confidence, clean set | LICHTHEIM_CLEAN | 1062 | full (gate is full-route only) | gate | adapted | **ALREADY_VALIDATED** | 2 | none |
| A08 | Gate and confidence by exposure status | ALL_WITH_EXPOSURE_STRATA | 1200 | full | gate | adapted | **ALREADY_VALIDATED** | 2 | none |
| A09 | Faithful Figure 2A | FAITHFUL_WFE_ALL | 1200 | full | raw_edit_distance | faithful | **NEEDS_FORMATTING_ONLY** | 3 | promote plotting to tracked package |
| A10 | Faithful Figure 2C | FAITHFUL_WFE_ALL | 1200 | full | positional error rate | faithful | **NEEDS_FORMATTING_ONLY** | 3 | promote plotting to tracked package |
| A11 | Faithful feature importance | FAITHFUL_WFE_ALL | 1200 | full | raw_edit_distance | faithful | **NEEDS_FORMATTING_ONLY** | 5 | promote to tracked package |
| A12 | Clean morphology x length | LICHTHEIM_CLEAN | 1062 | full;wm;ltm | raw_edit_distance | adapted | **NEEDS_COMPUTATION** | 3 | sprint 2 |
| A13 | Faithful morphology x length | FAITHFUL_WFE_ALL | 1200 | full | raw_edit_distance | faithful | **NEEDS_COMPUTATION** | 3 | sprint 2 |
| A14 | Trained-real frequency | TRAINED_REAL_FREQUENCY_PRIMARY | 671 | full;wm;ltm | raw_edit_distance | adapted | **NEEDS_COMPUTATION** | 4 | sprint 3 |
| A15 | Adapted feature importance | LICHTHEIM_CLEAN | 1062 | full;wm;ltm | raw_edit_distance | adapted | **NEEDS_COMPUTATION** | 7 | sprint 5 |
| A16 | Faithful error taxonomy | FAITHFUL_WFE_ALL | 1200 | full | insertions;deletions;substitutions | faithful | **NEEDS_COMPUTATION** | 6 | sprint 4 |
| A17 | Clean error taxonomy | LICHTHEIM_CLEAN | 1062 | full;wm;ltm | insertions;deletions;substitutions | adapted | **NEEDS_COMPUTATION** | 6 | sprint 4 |
| A18 | Premature EOS | LICHTHEIM_CLEAN | 1062 | full;wm;ltm | premature_eos rate | adapted | **NEEDS_COMPUTATION** | 6 | sprint 4 |
| A19 | SSP / sonority | SSP dataset | 2859 | full;wm;ltm | raw_edit_distance | faithful | **OPTIONAL** | 8 | deferred |
| A20 | Neural representations | n/a | n/a | n/a | n/a | adapted | **OUT_OF_SCOPE** | 9 | separate project |
| A21 | Route ablations | n/a | n/a | n/a | n/a | adapted | **OUT_OF_SCOPE** | 9 | separate project |
| A22 | Causal length-effect mechanism | n/a | n/a | n/a | n/a | adapted | **OUT_OF_SCOPE** | 9 | separate project |

## Notes on scope

`OUT_OF_SCOPE` entries (A20–A22) are not deferred analyses of these outputs:
they require new experiments — hidden-state extraction, new inference for
ablations, or new manipulations for a causal account of the length effect — and
belong to a separate project.

A19 (SSP / sonority) has been deferred since the protocol freeze and is
genuinely optional: it uses a different stimulus set (2,859 CCV/VCC triphones)
and answers a different question.

Every `NEEDS_COMPUTATION` row must be computed with the same frozen
conventions recorded in `docs/behavioral_wfe_fulllexicon.md`; none of them may
change an analysis set, the seed policy or the bootstrap.
