# `TRAINED_REAL_EXACT` words the LTM route gets wrong

**12 words** out of 671, 14 seed x item error events. These are the only trained-exact real words any route loses; FULL and WM make no errors at all on this stratum.

Edit-operation columns are per failing seed, in the order of `failing_seeds`. **No frequency model is fitted**: Zipf is shown as a descriptive column only.

| word | Zipf | len | target | seeds | prediction(s) | sub/del/ins |
|---|---|---|---|---|---|---|
| **sculptural** | 2.90 | 9 | `S K AH L P CH ER AH L` | 19,21 | 19: S K AH L P AH L AH N | 21: S K AH L P SH AH N TH | 3,4 / 0,0 / 0,0 |
| **siegfried** | 3.03 | 7 | `S IY G F R IY D` | 21,22 | 21: S IY G W IY L D | 22: S IY G F L IY D | 3,1 / 0,0 / 0,0 |
| **porcupine** | 2.88 | 9 | `P AO R K Y AH P AY N` | 19 | 19: P AO R K Y AO N IH SH | 4 / 0 / 0 |
| **polynesian** | 2.90 | 9 | `P AA L IH N IY ZH AH N` | 20 | 20: P AA L IH ZH IY N AH N | 2 / 0 / 0 |
| **reprimand** | 2.90 | 9 | `R EH P R AH M AE N D` | 20 | 20: R EH P AH M R AE N D | 0 / 1 / 1 |
| **seashore** | 2.92 | 5 | `S IY SH AO R` | 22 | 22: S AW TH AO R | 2 / 0 / 0 |
| **parentheses** | 3.13 | 9 | `P ER EH N TH AH S IY Z` | 19 | 19: P ER EH N W AH L IH S | 4 / 0 / 0 |
| **rename** | 3.18 | 5 | `R IY N EY M` | 22 | 22: R AY N EY M | 1 / 0 / 0 |
| **placebo** | 3.48 | 7 | `P L AH S IY B OW` | 20 | 20: P L AH S IY B IY | 1 / 0 / 0 |
| **mindful** | 3.49 | 7 | `M AY N D F AH L` | 21 | 21: M AY N F L AH K | 3 / 0 / 0 |
| **locomotive** | 3.50 | 9 | `L OW K AH M OW T IH V` | 22 | 22: L OW K AH M OW T AH V | 1 / 0 / 0 |
| **lieutenant** | 4.44 | 8 | `L UW T EH N AH N T` | 19 | 19: L EH T AH N T EH N | 3 / 1 / 1 |

Full machine-readable version: `trained_real_exact_ltm_errors.tsv`.
