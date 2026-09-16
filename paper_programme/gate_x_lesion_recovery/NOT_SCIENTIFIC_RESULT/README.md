# NOT_SCIENTIFIC_RESULT

Everything under this directory is **quarantined implementation-validation output**.

It is **not** a scientific result. It must not be reported, cited, plotted, summarised,
or compared against any frozen record, and it cannot be promoted to a result later.

## Why these files exist

They exist only to prove that the GXLR code path runs end to end: that a state
reconstructs from its recipe, that intact SD is measured, that a lesion context applies
and is removed, that the checkpoint is unmutated, and that the runner writes where it is
supposed to write. Nothing here was produced on a scientific population.

## Why they are not results

* the population is a deliberately **non-canonical truncated subset** (24 items) — the
  contract fixes the scientific population as all 29 571 canonical real words (§4), and
  every paired comparison and power rule is defined over it;
* only one lesion seed was used, not the frozen `{0, 1, 2, 3}`;
* a truncated pass cannot support the paired item-by-item NATIVE/FIXED05 statistics the
  contract preregisters;
* no outcome classification was performed on any of it.

## How the quarantine is enforced

* the runner refuses `--limit` without `--smoke`;
* the runner refuses non-zero severity on the full canonical population unless
  `--i-have-central-go` is passed, which CENTRAL alone supplies;
* every path written under `--smoke` must contain `NOT_SCIENTIFIC_RESULT` and every
  `.json`/`.tsv` must carry the `_SMOKE_TEST_ONLY` filename marker, or the run
  hard-stops;
* each `.tsv` carries the quarantine banner as its first line and each summary `.json`
  carries `"NOT_SCIENTIFIC_RESULT": true` and `"TEST_ONLY": true`;
* this namespace is disjoint from `../scientific_execution/`, which is reserved, empty,
  and cannot be written by a smoke invocation.

T11 in `tests/test_gate_x_lesion.py` pins all of the above.

## Status

```
GO_FOR_SCIENTIFIC_EXECUTION     = NO
AWAITING_CENTRAL_FINAL_GO_NO_GO = YES
```
