# ITEM_LEVEL_STORAGE

The frozen driver wrote `item_level_directional_dose.tsv` (164,156,690 bytes). The file exceeds the 100 MB per-file limit of the GitHub remote, so it is stored as follows:

| artifact | location | SHA256 |
|---|---|---|
| raw item-level TSV (authoritative bytes, read-only) | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/archives/ventral_directional_dose_execution_20260917/item_level_directional_dose.tsv` | `08fe29fe3a3f701a0b3e3f4ad4ac23ce1594349a830f4927d1713695dd0a4fcb` |
| committed deterministic gzip (`gzip -n -9`) | `scientific_execution/item_level_directional_dose.tsv.gz` | see `SHA256SUMS`; decompresses to exactly `08fe29fe…a4fcb` (validated: `gz_decompresses_to_raw_item_tsv`) |

Read-only archival copies of `summary_metrics_directional_dose.json`, `validity_gates_directional_dose.json` and `real_state_geometry_preflight.json` are in the same archive directory and are byte-identical to the committed files.

Reproduce: `gzip -dc item_level_directional_dose.tsv.gz | shasum -a 256`.
