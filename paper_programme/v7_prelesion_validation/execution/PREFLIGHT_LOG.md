# CLUSTER PREFLIGHT LOG — V7 PRE-LESION VALIDATION

    SCIENTIFIC_INFERENCE_EXECUTED = NO
    RESULT_NAMESPACE_CREATED      = NO

## PREFLIGHT_ATTEMPT_1

    outcome   INFRASTRUCTURE FAIL — git module absent on the node
    cause     environment, not code
    action    none required of the implementation

## PREFLIGHT_ATTEMPT_2

    runner commit 150ca569d8e11217a6ddd7c245643a7634261e6b
    tests         77 passed, 0 skipped
    dry-run       PREFLIGHT_FAIL, before any model load or forward pass:
                    scripts/run_prelesion_validation.py contains a
                    head_final string literal

    verdict   FALSE POSITIVE in hard stop 8.

    root cause
      `_is_docstring(node, rel)` reparsed the file into a NEW tree via
      `_AST_CACHE.setdefault(rel, ast.parse(...))` and then tested
      `body[0].value is node`, where `node` came from the tree parsed in the
      preflight loop. AST object identity is only meaningful WITHIN one parsed
      tree, so the `is` comparison could never hold. Every docstring was
      therefore classified as executable, and the runner's own module docstring
      -- which documents the prohibition -- tripped the check.

      The executable sentinel itself was already correct
      (`_FORBIDDEN_HEAD = "head_" + "final"`); only the exclusion logic was
      wrong. No scientific criterion, metric, state definition, input identity
      or namespace behaviour was involved.

## REPAIR (this commit)

    * `_is_docstring` and `_AST_CACHE` removed entirely.
    * `docstring_constant_ids(tree)` collects docstring Constant ids from a
      tree that is PASSED IN and never reparsed.
    * `scan_forbidden_head(source)` parses EXACTLY ONE tree and derives both
      the docstring set and the walk from that same tree.
    * The prohibition was not weakened; it was widened. Executable context now
      also rejects Name, keyword and import-alias references, alongside the
      existing Constant and Attribute checks.
    * The documentation string was NOT deleted; a test asserts the module
      docstring still documents the prohibition, so the check cannot be
      "fixed" by removing what it was meant to allow.

    Verified directly against hard stop 8 on all four execution files:
    PASS, no executable reference.

## Regression tests added

    module / function / class / async docstrings may mention it   -> allowed
    executable assignment, call argument, attribute, name,
        keyword, import alias, non-first string in a function     -> rejected
    the four real execution files pass the sentinel
    the scanner parses exactly one tree and the helper never reparses
    `_is_docstring` / `_AST_CACHE` are absent
    no selectable forbidden-head literal exists in execution code
    the module docstring still documents the prohibition

## Test counts

    local (artifacts absent)   79 passed, 10 skipped
      pre-existing suite unchanged at 40 passed, 10 skipped
    Jean Zay (inputs set)      89 passed, 0 skipped expected

## PREFLIGHT_ATTEMPT_3 / SCIENTIFIC RUN

    dry-run                       PASS (sentinel repair abf126df effective)
    SCIENTIFIC_RUN                COMPLETE
    SCIENTIFIC_RERUN_REQUIRED     NO
    FROZEN_RESULTS_HASH_CHECK     PASS_ALL (sha256sum -c FILE_SHA256SUMS)
    runner commit                 abf126df21de182569a65880cc05d3bf4560d715

## REPORTING_ATTEMPT_1

    outcome   FAILED_SCHEMA_ITERATION — KeyError: 'cc'
    stage     rendering V7_SOURCE_REPAIR_PAIRED_ANALYSIS.md only
    produced  V7_INTACT_ROUTE_VALIDATION_RESULTS.md      (complete)
              V7_PSEUDOWORD_VALIDATION_RESULTS.md        (complete)
              V7_SOURCE_REPAIR_PAIRED_ANALYSIS.md        (zero bytes)

    root cause
      `paired_report` iterated `sorted(d.items())` over every key of each slot
      in SOURCE_POST_PAIRED.json and treated each as a binary transition table.
      Only eight of the thirteen entries per slot are transition tables. The
      other five are `n_items_paired` (metadata), two `*_delta_summary` dicts
      and two `*_n_changed` scalars — none of which carry `cc`.

    scope     DERIVED REPORTING ONLY. No model, no scientific result, no
              contract, metric, classification rule, state definition, runner
              or evaluator was involved.

## REPORTING FIX (this commit)

    * `PAIRED_TRANSITION_ENDPOINTS` — an explicit frozen tuple of the eight
      binary endpoints. `paired_report` iterates ONLY that tuple, so rendering
      no longer depends on the file's key order.
    * Endpoints are NEVER discovered by probing for a "cc" key; a test forbids
      that and forbids the old `sorted(d.items())` iteration.
    * `validate_paired_schema` fails closed: a missing slot, a missing expected
      endpoint, a non-dict endpoint, or an endpoint missing any of
      cc/cw/wc/ww/undefined raises `ReportSchemaError`. Nothing is skipped.
    * The continuous quantities are reported AS continuous (n/mean/sd/min/p50/
      max and the n_changed scalars) and are never given transition semantics.
      They are not reinterpreted.
    * `n_items_paired` is rendered as metadata, not as an endpoint.
    * `verify_frozen_results` re-checks every file in FILE_SHA256SUMS BEFORE and
      AFTER reporting and refuses if any changed. Reports are written only
      under `results/reports/`, which FILE_SHA256SUMS does not cover.

    Test counts after the fix
      local   98 passed, 10 skipped
        contract suite  40 passed, 10 skipped   (unchanged)
        runner suite    39 passed               (unchanged)
        reporting suite 19 passed               (new)
      cluster 108 passed, 0 skipped expected
