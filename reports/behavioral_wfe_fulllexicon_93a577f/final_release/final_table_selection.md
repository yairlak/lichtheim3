# Final table selection

Machine-readable index: `tables/final_table_index.tsv` (24 rows: 12 MAIN,
12 SUPPLEMENTARY).

## Copy policy

The release **does not duplicate every existing TSV**. Three modes are used and
recorded per row:

- `COPIED_GENERATED` — summary tables created by this release itself (regime,
  checkpoint, key results, status, figure index).
- `COPIED_BYTE_IDENTICAL` — the three faithful A09/A10/A11 plotting tables,
  copied bit-for-bit from `outputs/.../faithful_replication/` so the release is
  self-contained for the figures it renders.
- `POINTER` — the table stays where its sprint validated it and the index points
  at it. This is the default: copying a validated table adds no value and would
  create a second thing to keep in sync.

## MAIN tables

| # | topic | analysis | mode |
|---|---|---|---|
| T1 | dataset and exposure audit | A01–A03 | generated |
| T2 | checkpoint audit | A01–A03 | generated |
| T3 | route-specific length slopes | A05 | pointer |
| T4 | LTM−WM contrasts | A05 | pointer |
| T5 | morphology contrasts | A12 | pointer |
| T6 | frequency slopes | A14 | pointer |
| T7 | error-operation summary | A17 | pointer |
| T8 | EOS summary | A18 | pointer |
| T9 | adapted feature-importance summary | A15 | pointer |
| T10 | key results across analyses | all | generated |
| T11 | analysis status | all | generated |
| T12 | figure index | all | generated |

## SUPPLEMENTARY tables

Detailed seed-level, composition, overlap, sensitivity and illustration tables
are indexed rather than copied: the three faithful plotting tables (byte-identical
copies), exposure-stratified operations, error composition, EOS by exact length,
the EOS/deletion 2 × 2, route-specific model statuses, interaction drop-block
utility, the frequency distribution audit, gate by exposure status, and the
seed-22 illustrative items.

## What is deliberately not indexed

Repeat-level permutation tables, bootstrap replicate tables and per-sprint
validation manifests. They are reproducibility artefacts rather than results;
they remain in their sprint directories and are covered by the six manifests.
