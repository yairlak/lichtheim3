# SMOKE ARTIFACTS — NOT RESULTS

Produced by `run_gate_route_audit.py --smoke` to validate the pipeline end to end:
**400 items, one state (`W3_SRC`)**. They exist to show the machinery runs and that the
contract's power rule and the `STRUCTURALLY_UNTESTABLE` path fire correctly.

**They are not evidence and must not be cited, plotted, or summarised.**

## Two independent reasons these can never be read as results

1. **Quarantine is enforced in code, not by convention.** Every smoke artifact is written under
   this directory with a `_SMOKE_TEST_ONLY` marker, and `assert_quarantined` hard-stops if a
   smoke invocation would ever resolve to a full-result path or omit the marker. Pinned by
   `tests/test_gate_route_diagnostics.py::test_smoke_output_is_quarantined`. The full pass writes
   to `../figure_source_data/` and `../summary_metrics.json`, which smoke can no longer reach.

2. **These specific files are STALE.** They were produced by the **pre-AMENDMENT-2** runner and do
   not reflect the corrected code. They predate:
   * the free-AR cap correction (local `max_steps = 24` → imported historical `FREE_AR_MAX_STEPS = 12`),
     so their `freear_*` columns use a cap that is no longer used;
   * the `batch_dec_width` column and the two named gate means
     (`gate_mean_item_level` / `gate_mean_position_weighted_historical`);
   * the `_SMOKE_TEST_ONLY` naming convention.

   They are retained **only** as an audit trail of what was actually executed and when — which is
   also the source of the "400 items" statement used throughout the package. They are deliberately
   **not** regenerated: doing so would produce a new smoke artifact for no scientific purpose.
