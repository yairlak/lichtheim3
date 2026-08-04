# Final figure selection

Frozen in `final_release_spec.md` before any figure was copied. Machine-readable
index: `tables/final_figure_index.tsv`. Nothing is selected merely because it
exists, and **no new composite figure was created**.

Every release copy is byte-identical to its source, and **no source figure was
moved, overwritten or deleted**.

## MAIN (7)

| # | source analysis | regime | purpose | main finding | primary limitation |
|---|---|---|---|---|---|
| **F1** | A04 clean length curves | adapted | The central route-specific length effect | Trained real words: FULL and WM at exact ceiling, LTM only a very weak slope. Novel pseudowords: LTM develops a large length effect, WM stays far more robust, FULL near ceiling | lexicality and exposure are perfectly confounded in the clean set |
| **F2** | A05 length slopes and LTM−WM | adapted | Quantifies F1 and the primary contrast | LTM−WM pseudoword slope difference positive in all four seeds (+0.183 to +0.246 per phoneme) | trained real words ceiling-limited under FULL/WM, so their zero slope is structural |
| **F3** | A06 clean serial position | adapted | Where in the word the errors fall | LTM pseudowords rise strongly at late positions; FULL and WM much less | zip-mismatch positions, no alignment; pooled over seeds |
| **F4** | A17 clean error taxonomy | adapted | What kind of errors they are | Substitutions dominate, then deletions, then insertions; burden concentrated in long LTM pseudowords | editops tie-breaking moves counts between types without changing total distance; terminal insertions unobservable |
| **F5** | A18 premature EOS | adapted | The decoder-stopping diagnostic and its limits | 87 observed events, 82 in LTM, all on pseudowords; rate rises with length in LTM in all four seeds | ON_TIME/LATE EOS structurally unobservable; premature EOS on only a minority of erroneous items |
| **F6** | A14 trained-real frequency | adapted | Familiarity within trained words | LTM Zipf slope negative in all four seeds (mean −0.0130) | sparse errors; FULL and WM ALL_ZERO_OUTCOME by ceiling |
| **F7** | A15 adapted feature importance | adapted | Which factors predict errors | Route and lexicality/exposure lead jointly; length a stable third; morphology negligible | route vs lexicality/exposure ordering unresolved; importance unsigned |

These seven cover the six frozen narrative points, with the trained-versus-novel
distinction carried inside F1 and F2 by the lexicality split rather than by a
separate figure.

## SUPPLEMENTARY (12)

| # | source analysis | regime | why supplementary |
|---|---|---|---|
| S1 | A09 faithful Figure 2A | **faithful** | replication fidelity, not the model-specific question |
| S2 | A10 faithful Figure 2C | **faithful** | faithful counterpart of F3, different estimand |
| S3 | A11 faithful feature importance | **faithful** | faithful counterpart of F7; never pooled with it |
| S4 | A17 FULL/WM zoom | adapted | companion to F4; the frozen >10× rule fired (ratio 22.95) |
| S5 | A16 faithful Figure 8A | **faithful** | faithful counterpart of F4 |
| S6 | A12 adapted morphology | adapted | null result; no robust contrast |
| S7 | A13 faithful morphology | **faithful** | stimulus-level replication of the same null |
| S8 | A15 route-specific FI | adapted | detail behind F7; only LTM estimable |
| S9 | A15 interaction blocks | adapted | detail behind F7 |
| S10 | A14 frequency confidence/gate | adapted | linked outcomes, not independent evidence |
| S11 | A07 gate by clean lexicality | adapted | mechanism-adjacent descriptive |
| S12 | A08 gate by exposure status | adapted | descriptive; three strata have n ≤ 7 |

## MECHANISM_HANDOFF_ONLY

`error_taxonomy/length_effect_mechanism_handoff.md` — not a figure, carries no
estimate, and belongs to the separate mechanism project.

## NOT_SELECTED_REDUNDANT

The legacy PNG/PDF copies under
`outputs/.../behavioral_analysis/figures/` duplicate the Sprint-1 figures that
F1, F2, F3, S11 and S12 already carry in three formats with captions. They
remain in place, untouched, and are not copied into the release.

## VALIDATION_ONLY

No figure falls in this category: every validation artefact in this project is a
table, a manifest or a log.
