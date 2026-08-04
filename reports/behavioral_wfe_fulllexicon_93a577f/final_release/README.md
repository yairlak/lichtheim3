# Final release — WFE behavioral analysis

Publication-facing tree for the Lichtheim3 WFE behavioral program. **Editorial
only**: every number here already exists in a validated table produced by
Sprints 1–5 or by the analysis phase. No new analysis, no checkpoint inference,
no causal claim.

## Start here

| document | for |
|---|---|
| `yair_brief.md` | 1–2 pages, meeting-ready |
| `executive_summary.md` | full scientific synthesis, 14 sections |
| `robust_findings_and_limitations.md` | what is established and how strongly |
| `faithful_vs_adapted_summary.md` | why the two families are never mixed |
| `final_figure_selection.md` | why each figure is main, supplementary or excluded |
| `final_table_selection.md` | which tables are copied and which are pointed at |
| `a09_a10_a11_audit.md` | the pre-formatting audit of the three legacy rows |
| `final_release_spec.md` | the frozen release specification |

## Layout

```
figures/main/            F1..F7   PNG (300 dpi) + PDF + SVG
figures/supplementary/   S1..S12  PNG + PDF + SVG
captions/main/           one standalone caption per main figure
captions/supplementary/  one standalone caption per supplementary figure
tables/                  figure index, table index, key results, status,
                         regime and checkpoint summaries
formatted_existing/      A09/A10/A11 rendered from their stored tables,
                         with captions and byte-identical plotting tables
_control/                preflight, frozen spec, audit, release manifest
validation/              validation report, test log, inventory, hashes, diff review
```

## The seven main figures

1. **F1** length by route — the central route-specific length effect
2. **F2** length slopes and the LTM−WM contrast
3. **F3** serial-position profile
4. **F4** error taxonomy by route
5. **F5** premature EOS
6. **F6** trained-real frequency
7. **F7** adapted feature importance

## Reading rules

- **Faithful ≠ adapted.** Faithful uses the paper's original 1,200 items and
  source labels; adapted uses the exposure-audited 1,062-item clean set. Values
  from the two families are never pooled or placed on a common axis, and the
  adapted analysis is **not** a correction of the faithful one.
- **Source Real/Pseudo is not trained/novel.** 122 source-real items were never
  trained; 9 source-pseudo items collide with the training lexicon.
- **On the clean set lexicality and exposure are perfectly confounded**, so that
  factor is the *lexicality/exposure contrast* and nothing separates the two.
- **Red = Real, blue = Pseudoword**, reserved; route, operation type and factor
  importance use neutral palettes.
- Captions carry the science, including each analysis-set definition and the
  principal limitation. Plot titles do not.

## Regeneration

```bash
python -m scripts.behavioral_analysis.plot_final_release \
    --out_root reports/behavioral_wfe_fulllexicon_93a577f/final_release
```

Deterministic and byte-stable. Source reports are never moved, overwritten or
deleted; every release copy records source path, source hash, release hash and
an equality verdict in `_control/final_release_manifest.json`.

## Status

Core WFE behavioral analysis **complete**. **A19 (SSP / sonority) is optional,
deferred and unstarted.** The causal mechanism analysis is maintained as a
**separate project**; the factual handoff is
`../error_taxonomy/length_effect_mechanism_handoff.md`.
