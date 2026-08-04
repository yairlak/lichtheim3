# Final release — frozen specification

**Frozen 2026-08-04, before any formatting work.** Machine-readable twin:
`_control/final_release_spec.json`.

This is an **editorial, formatting, integration and provenance** release. It
adds **no new scientific value of any kind**. Every number in it already exists
in a validated table produced by Sprints 1–5 or by the analysis phase.

## 1. The no-new-analysis rule

Formatting may change. **Scientific values may not.** Specifically, this release
introduces no new estimator, bootstrap, contrast, analysis set, uncertainty
method, item filter or seed policy; recomputes nothing from checkpoint
predictions where an authoritative validated table exists; performs no
checkpoint inference; and makes no causal or architectural claim. SSP (A19)
stays **optional, deferred and unstarted**.

A09, A10 and A11 are **rendered from their stored authoritative tables**, never
refitted. For A11 this is what preserves Ridge α = 1.0, the 80/20
`random_state=42` split, `n_repeats=100 random_state=42` permutation and the
historical signed convention: no model is fitted at all.

## 2. Main versus supplementary

**MAIN** (target 5–7) must communicate, in this order:

1. the central route-specific length effect;
2. the trained-versus-novel distinction (carried inside 1 by lexicality);
3. the serial-position profile;
4. the operation taxonomy and the EOS limitation;
5. the frequency/familiarity result;
6. the adapted feature-importance summary.

Everything else that remains scientifically useful is **SUPPLEMENTARY**.
Categories are `MAIN`, `SUPPLEMENTARY`, `VALIDATION_ONLY`,
`MECHANISM_HANDOFF_ONLY`, `NOT_SELECTED_REDUNDANT`. Nothing is selected merely
because it exists.

Gate/confidence, morphology, faithful replications, the zoom panel and
sensitivity figures are supplementary unless the evidence demands otherwise.
**No new statistical composite figure is created.** A layout-only multi-panel
assembly would be permitted only if it combined existing final panels without
altering data, kept each panel separately available, was deterministic, rescaled
no axis in a way that hides route differences, and preserved each source
definition in the caption — **this release creates none.**

## 3. Naming and ordering

Final figures are numbered `F1`…`Fn` for MAIN and `S1`…`Sn` for SUPPLEMENTARY,
in the order fixed by §2. Release copies keep their **source stem** so a reader
can always find the origin, prefixed by the number:
`F3_yair_clean_serial_position.png`. Tables are indexed, not bulk-copied.
Ordering in every index is deterministic: MAIN before SUPPLEMENTARY, then by
figure number, then by source path.

## 4. Copy policy

The release tree **copies** selected files; it never moves or deletes a source.
Every copy records `source_path`, `source_sha256`, `release_sha256` and an
`equality` verdict. Tables are **indexed by pointer** where copying adds no
value; only tables that a reader needs in hand are copied.

## 5. Caption template

Every final figure carries a standalone caption with, in order: a bold title
sentence; what is plotted; the analysis-set definition with exact n; the seed
convention; the uncertainty convention; the **faithful/adapted label**; and the
principal limitation. Captions carry the science; **plot titles never do**.

## 6. Visual conventions (inherited, unchanged)

- **Real = red, Pseudoword = blue**, reserved: they never encode route,
  operation type or factor importance.
- Neutral palettes for route-only, operation-type and feature-importance
  figures.
- Morphology by line style — complex solid, simple dashed.
- Seed-level values visible wherever they are already part of the analysis; the
  across-seed summary is visually prominent.
- Legends inside figures where readable; **no scientific explanation in a plot
  title**.
- PNG at 300 dpi, plus PDF and SVG, for every final figure.

## 7. Terminology, frozen

- **Faithful** = original WFE/SWP labels and procedure, replication fidelity.
  **Adapted** = Lichtheim3 exposure-audited analysis. The adapted analysis is
  **never** called a correction of the faithful one, and the two are never
  pooled or placed on a common quantitative axis.
- Source Real/Pseudo is **not** trained/novel.
- On `LICHTHEIM_CLEAN`, lexicality and exposure are perfectly confounded; the
  factor is the **lexicality/exposure contrast**.
- Route labels: **FULL**, **WM**, **LTM**.
- Uncertainty: "95 % hierarchical bootstrap" only where items are genuinely
  resampled and the statistic recomputed (Sprints 1–4); the feature-importance
  interval is a **"seed-resampling interval over four checkpoints"**.
- Statuses: `ROBUST`, `CONSISTENT_BUT_SMALL`, `CEILING_LIMITED`,
  `SPARSE_ERROR_LIMITED`, `SPARSE_EOS_LIMITED`, `DESCRIPTIVE_ONLY`,
  `NON_ESTIMABLE`, `OPTIONAL_DEFERRED`.

## 8. Final output inventory

```
final_release/
  README.md  final_release_spec.md  executive_summary.md
  faithful_vs_adapted_summary.md  yair_brief.md
  robust_findings_and_limitations.md
  final_figure_selection.md  final_table_selection.md
  a09_a10_a11_audit.md  final_release_provenance.json
  final_release_commit_plan.md
  _control/    preflight, spec twin, audit twin, release manifest
  figures/main/  figures/supplementary/
  captions/main/ captions/supplementary/
  tables/      final_figure_index.tsv  final_table_index.tsv
               key_results_summary.tsv  analysis_status_summary.tsv
               dataset_regime_summary.tsv  checkpoint_summary.tsv
  formatted_existing/   A09/A10/A11 figures, captions and plotting tables
  validation/
```

Regeneration is deterministic and byte-stable:

```
python -m scripts.behavioral_analysis.plot_final_release \
    --out_root reports/behavioral_wfe_fulllexicon_93a577f/final_release
```
