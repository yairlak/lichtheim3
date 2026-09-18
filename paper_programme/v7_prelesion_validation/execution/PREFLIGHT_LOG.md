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
