# EOS instrumentation convention — audited, not assumed

**Audited 2026-08-03 from the committed evaluator, its provenance strings and
its tests, before any EOS distribution was read.** Machine-readable twin:
`_control/eos_convention.json`.

Sources inspected: `scripts/external_eval.py` (`_first_eos_position`, the
autoregressive readout loop, the `eos_position_convention` provenance string),
`tests/test_behavioral_eval_patch.py` group B, the Sprint-1 output matrix, and
the per-seed production validation reports.

## 1. The eight required facts

| # | Question | Answer, with its source |
|---|---|---|
| 1 | Zero- or one-based? | **Zero-based.** `_first_eos_position` returns `enumerate` index `i`; its docstring states "0-based index INTO THE READOUT WINDOW … It is NOT a 1-based position". |
| 2 | Is BOS counted? | **No.** The window is `dec_input[i, 1:n_steps+1]`; index 0 holds BOS and is sliced away before the scan. |
| 3 | What does the position denote? | **The output-token index within the item's readout window**, equivalently **the number of phonemes emitted before EOS**. The docstring pins `eos_position == len(predicted_symbols)`. |
| 4 | Expected boundary for target length L? | **L.** A correct model emits L phonemes and then EOS, i.e. at window index L. |
| 5 | Missing-value convention? | `None` in the evaluator, serialized as the empty string (`MISSING = ""`), never `0` or `-1`. Asserted by `test_B_missing_value_is_empty_not_zero`. |
| 6 | Is EOS after the boundary observable? | **No.** See §2 — this is the decisive finding. |
| 7 | How is a correct-boundary EOS represented? | **It is not representable.** It falls outside the window and is recorded exactly like "no EOS at all". |
| 8 | Were predictions trimmed at first EOS before serialization? | **Yes**, but the position is captured from the RAW window *before* trimming, so trimming does not corrupt it. |

## 2. The observable horizon — decisive constraint

The readout window is `dec_input[i, 1:n_steps + 1]` with `n_steps = L`, i.e.
exactly **L generated tokens at 0-based indices 0 … L−1**. An EOS emitted at the
correct boundary would occupy window index L, one past the end of the slice.
The committed comment says so explicitly: an EOS "emitted only beyond n_steps
… is outside what this evaluation reads out and is therefore recorded as
absent, exactly like no EOS at all."

Consequences, which constrain every EOS claim in this sprint:

- **Every observed EOS is premature.** `observed ∈ [0, L−1] < L = expected`.
- **`ON_TIME_EOS` is structurally unobservable**, and so is `LATE_EOS`.
- **`EOS_NOT_OBSERVED` is ambiguous**: it conflates "the model stopped correctly
  at the boundary" with "the model never emitted EOS inside the horizon". These
  two cannot be separated from the stored instrumentation.

Empirically confirmed across all 4 seeds × 3 routes on the enriched production
tables: **121 observed EOS events, 0 with `position ≥ L`, and 0 where
`position ≠ predicted_length`.** The data match the audited convention exactly.

## 3. Frozen definitions

```
expected_eos_position = L                      (target phoneme length)
eos_offset            = observed − expected    (always negative when observed)
eos_shortfall         = expected − observed    (PREMATURE_EOS only; ≥ 1)
```

`eos_shortfall` is the number of target positions left unemitted: **positive
means EOS came early, larger means more of the target remained.** Because the
horizon truncates at L, `eos_shortfall ∈ [1, L]`.

## 4. Classification

| Class | Condition | Observable here? |
|---|---|---|
| `PREMATURE_EOS` | `0 ≤ observed < L` | **yes** — the only observable EOS class |
| `ON_TIME_EOS` | `observed == L` | **no** — outside the readout window |
| `LATE_EOS` | `observed > L` | **no** — outside the readout window |
| `EOS_NOT_OBSERVED` | no EOS in the window | yes, but **ambiguous** (correct stop or no stop) |
| `EOS_UNAVAILABLE` | field absent or non-numeric | yes (none occur in this cohort) |

## 5. What this convention forbids

- Reporting an on-time or late EOS rate — neither is measurable.
- Treating `EOS_NOT_OBSERVED` as evidence of correct stopping.
- Deriving a "fourth edit operation" from EOS. Premature EOS is a **decoder
  diagnostic**, entirely separate from the Levenshtein taxonomy: a deletion is
  not automatically a premature EOS, a premature EOS is not one deletion,
  several deletions may follow one early stop, and early stops may coexist with
  substitutions or insertions.

The forced-length readout additionally makes **terminal insertions beyond the
target horizon unobservable**, because the prediction can never exceed L tokens.
