# VENTRAL_INTERFACE_RESULTS_RECAP — frozen S0–S3 factorization diagnostic (executed)

SCIENTIFIC_EXECUTION=COMPLETE · all frozen validity gates PASS on all four states · single execution attempt.

Labels used throughout: **[STRUCTURAL_CODE_FACT]**, **[EMPIRICAL_RESULT]**, **[INTERPRETATION]**, **[NOT_ESTABLISHED]**.
All numbers come from the frozen outputs (`item_level_factorization.tsv`, `summary_metrics.json`, `ar_diagnostic_native_freear.tsv`) through the read-only `analysis/build_results_package.py`. Every figure value is also stored in `figure_source_data/`.

| provenance | value |
|---|---|
| worktree / branch | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-interface` · `paper-programme/ventral-interface-diagnostic` |
| freeze commit (= HEAD at execution) | `4ad20048e20da84b9f22a92965098b84c2bf7dd6` |
| contract SHA256 | `a5ca7b83717b43ab77cedae0cd3e4bef17901ffff051f5ca90b9340b45bda5d3` |
| command | `python3 scripts/ventral_interface/run_ventral_interface_factorization.py --execute --contract-sha256 a5ca7b83717b43ab77cedae0cd3e4bef17901ffff051f5ca90b9340b45bda5d3` |
| run | attempt 1 of 1; 2026-09-16T22:06:57Z → 22:24:03Z (UTC); exit 0; `EXECUTE=COMPLETE`; logs in `logs/` |
| environment | Apple M2 Pro, macOS 14.2.1 arm64, python 3.11.15, torch 2.12.1, CPU, float32, deterministic algorithms |
| outputs SHA256 | item-level `37f2fb170e53596067ad1bb8a64189dabec6d808fd31a1ffff942649de175663` · summary `455ccaf07bd7e2adb6047c24ac40667ba9b3e1256bdfeeb6772589b12b610c80` · AR diagnostic `75106959da8156131fc9f19d4d053b8307825dbcb7f0a45ead48a8eb15fdf027` (read-only copies also stored in `archives/ventral_interface_execution_logs_20260917/`) |

---

## 1. EXECUTIVE RESULT

**[EMPIRICAL_RESULT]** The pattern is the same in all four states (W3/W4 × SOURCE/POST_REPAIR) and in both readouts:

* **S2, raw true GloVe, decodes every item correctly: 29,571/29,571** through the isolated ventral decoder in every state, under genuine free-AR and forced-length alike.
* **S1, the raw GloVe of the frozen C top-1 lexical row, rescues 99.86–100% of native S0 failures** (3195/3199, 3193/3193, 3642/3647, 3662/3662).
* **S3, radial (ŝ rescaled to the retrieved GloVe norm with direction unchanged), rescues only 11.96–13.38% of native failures** and **breaks 2,085–2,556 items that were correct under S0**. Its exact rate is therefore below native S0 in every state.
* S3 reproduces only 11.96–13.35% of S1 rescues. Every S3 rescue is also an S1 rescue, except 2 items in W4_SRC.
* **Arm-A repair** removes all C errors (34→0, 42→0) and all retrieval-limited S1 failures. The native ventral failure set is essentially unchanged (Jaccard 0.963 / 0.944).

**[INTERPRETATION]** The best-supported frozen family is **F2 (directional / prototype rescue)**. Swapping ŝ for its retrieved lexical prototype removes essentially all native isolated-ventral failures; rescaling ŝ does not. **F3 (retrieval-limited)** applies only to a very small SOURCE-only subpopulation, which Arm-A removes. F1 is contradicted by the counts. F4 did not occur. F5 is not supported as the dominant mechanism: substantial AR cascades exist, but the prototype input removes them under the same decoder.

## 2. SOURCE INTEGRITY

**[EMPIRICAL_RESULT]**
* **Before execution:** HEAD was exactly the freeze commit and descends from `79f4e5bd94a9c1f82594f4050028b39c220b82b0`. The tracked tree was clean, the contract hash matched, the freeze `SHA256SUMS` passed (17/17), and a fresh preflight passed with an identical report (`379f5ed1…`).
* **Inputs, re-hashed before and after execution, all exact:**
  * SOURCE `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` and `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3`
  * both read-only archival copies (identical bytes)
  * Arm-A heads `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` and `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03`
* Per-state parameter `state_dict` hashes are identical before and after each state (Gate E).
* **Preflight hash inside execute:** the preflight that `--execute` runs internally reported `a4f683fd…`, not `379f5ed1…`. A preflight-only check showed the only difference is `ventral_interface/evaluate.py` (a frozen-closure file) in the runtime-loaded module list, because `run_execute` imports it first. Every other key and the whole closure-hash map are identical. See `logs/03_internal_preflight_hash_note.txt`.

| state | seed/u/step | base SHA256 | head SHA256 | reconstructed_state_identity |
|---|---|---|---|---|
| W3_SRC | 19/3825/10625850 | `a5f21de9…aad76c` | NA | `6d7282854323b810727cd9c0c90cace6dd8a54d306408cafaa8c40c5455d10a0` |
| W3_REP | 19/3825/10625850 | `a5f21de9…aad76c` | `8865ba95…7afcfc` | `9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1` |
| W4_SRC | 20/3040/8445120 | `0657f410…dc79f3` | NA | `32928707acc81af215faa829ad94bba566804982bc192bb5116ea32d43d56e29` |
| W4_REP | 20/3040/8445120 | `0657f410…dc79f3` | `724ed4c6…b972d03` | `e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6` |

## 3. VALIDITY GATES (all PASS, all states)

| gate | W3_SRC | W3_REP | W4_SRC | W4_REP |
|---|---|---|---|---|
| BANK (hard stop before the recorded gates) | pass | pass | pass | pass |
| R: recomputed C errors == archived; top-1 rows == historical call | 34 == 34; 0 mismatch | 0 == 0; 0 | 42 == 42; 0 | 0 == 0; 0 |
| A: C-correct ⇒ lexical identity | 27,947 items, 0 violations | 27,981, 0 | 27,939, 0 | 27,981, 0 |
| B: S1 ≡ S2 when rows equal | 27,947 rows, max diff 0.0 | 27,981, 0.0 | 27,939, 0.0 | 27,981, 0.0 |
| D: S3 direction (cos ≥ 1−1e-6; top-1/2 preserved) | min cos 0.9999999999999993; 0 rank changes; 0 degenerate | …994; 0; 0 | …993; 0; 0 | …994; 0; 0 |
| C: matched Naming (cap 256) | historical 0 == archived 0; 0/29,571 item mismatches | 0; 0 | 0; 0 | 0; 0 |
| SHAT: live ŝ vs retrieval ŝ | 5.36e-6 ≤ 1e-5; 0 top-1 changes | 5.30e-6; 0 | 6.42e-6; 0 | 6.32e-6; 0 |
| H: S0 reproduces archived LTM errors (free-AR / canonical) | 3199/3199 | 3193/3193 | 3647/3647 | 3662/3662 |
| E: shared downstream path, no guard hit, params unchanged, encoder bitwise across steps | pass | pass | pass | pass |

**[STRUCTURAL_CODE_FACT]** For every condition, the downstream path is `gate_probe.ar_decode_{free,forced_length}(routes=('ltm',))` → `route_logits(ltm)` → `ltm.encode` (with the `to_semantic` hook) → `decode_from_s_hat` (`sem_to_h0`, `decoder`, `dec_to_premotor`) → `motor`. There is no gate, no FULL fusion and no dorsal readout.

## 4. POPULATIONS / HOMOPHONES

**[STRUCTURAL_CODE_FACT]**
* Rows: 29,571 lexical rows, split into 27,981 canonical C targets (27,981 phonological classes, 1,299 of them with more than one word) and 1,590 non-canonical homophone rows.
* Historical C correctness applies only to the C targets. It means exact lexical-row identity: argmax over all 29,571 rows equals the item's own row.

**[EMPIRICAL_RESULT]**
* **No C error is a homophone hit.** `C_WRONG_BUT_RETRIEVED_PHONOLOGY_CORRECT` is 0 in every state, and every SOURCE C error retrieves a word with a different phonology.
* **Non-canonical homophone rows:**
  * Retrieval never returns the row itself (lexical identity wrong for all 1,590). By construction its encoder input equals the canonical member's.
  * Retrieved phonology is correct for 1,582/1,590 (W3_SRC), 1,581/1,590 (W4_SRC) and 1,590/1,590 (both REP).
  * Median cos(ŝ, own GloVe) is 0.14, while median top-1 cosine is 0.72.
* Native S0 on those rows: correct 1,555 / 1,556 / 1,557 / 1,556.

## 5. PRIMARY FREE-AR FACTORIZATION (isolated ventral, all 29,571 items)

**[EMPIRICAL_RESULT]**

| state | S0 exact | S1 exact | S2 exact | S3 exact |
|---|---|---|---|---|
| W3_SRC | 26,372 (89.18%) | 29,529 (99.86%) | 29,571 (100%) | 24,684 (83.47%) |
| W3_REP | 26,378 (89.20%) | 29,571 (100%) | 29,571 (100%) | 24,650 (83.36%) |
| W4_SRC | 25,924 (87.67%) | 29,520 (99.83%) | 29,571 (100%) | 23,867 (80.71%) |
| W4_REP | 25,909 (87.62%) | 29,571 (100%) | 29,571 (100%) | 23,842 (80.63%) |

Transitions relative to S0 (all items; each row sums to 29,571):

| state | cond | WRONG→CORRECT | CORRECT→WRONG | CORRECT→CORRECT | WRONG→WRONG |
|---|---|---|---|---|---|
| W3_SRC | S1 | 3,195 | 38 | 26,334 | 4 |
| W3_SRC | S2 | 3,199 | 0 | 26,372 | 0 |
| W3_SRC | S3 | 397 | 2,085 | 24,287 | 2,802 |
| W3_REP | S1 | 3,193 | 0 | 26,378 | 0 |
| W3_REP | S2 | 3,193 | 0 | 26,378 | 0 |
| W3_REP | S3 | 382 | 2,110 | 24,268 | 2,811 |
| W4_SRC | S1 | 3,642 | 46 | 25,878 | 5 |
| W4_SRC | S2 | 3,647 | 0 | 25,924 | 0 |
| W4_SRC | S3 | 488 | 2,545 | 23,379 | 3,159 |
| W4_REP | S1 | 3,662 | 0 | 25,909 | 0 |
| W4_REP | S2 | 3,662 | 0 | 25,909 | 0 |
| W4_REP | S3 | 489 | 2,556 | 23,353 | 3,173 |

Rescue and regression are separated: S1's only regressions (38, 46) occur in SOURCE states and are all retrieval-identity errors (§9). S2 never regresses. S3's regressions outnumber its rescues by a factor of about 5.

## 6. FORCED-LENGTH FACTORIZATION (secondary)

**[EMPIRICAL_RESULT]** Every count in §5 is identical under canonical forced-length AR. At item level, exact-correct disagrees between the two conventions on **0 items** for every state and condition. The predicted strings differ only on wrong items: S0 447 / 452 / 493 / 498 and S3 708 / 717 / 821 / 832, because the canonical window truncates at len+1. **[INTERPRETATION]** On these states the conventions are therefore redundant as accuracy readouts. They remain reported separately, as frozen.

## 7. C × LTM DECOMPOSITION (free-AR; canonical identical)

**[EMPIRICAL_RESULT]**

| stratum | W3_SRC n · S0/S1/S2/S3 exact | W3_REP | W4_SRC | W4_REP |
|---|---|---|---|---|
| C_POPULATION | 27,981 · 24,817/27,947/27,981/23,197 | 27,981 · 24,822/27,981/27,981/23,164 | 27,981 · 24,367/27,939/27,981/22,384 | 27,981 · 24,353/27,981/27,981/22,366 |
| NONCANONICAL_HOMOPHONE_MEMBERS | 1,590 · 1,555/1,582/1,590/1,487 | 1,590 · 1,556/1,590/1,590/1,486 | 1,590 · 1,557/1,581/1,590/1,483 | 1,590 · 1,556/1,590/1,590/1,476 |
| **C_CORRECT_AND_NATIVE_LTM_WRONG** | **3,160** · 0/3,160/3,160/390 | **3,159** · 0/3,159/3,159/375 | **3,609** · 0/3,609/3,609/479 | **3,628** · 0/3,628/3,628/482 |
| **C_WRONG_AND_NATIVE_LTM_WRONG** | **4** · 0/0/4/0 | 0 | **5** · 0/0/5/2 | 0 |
| **C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT** (C lexical identity wrong; native S0 output phonology correct) | **30** · 30/0/30/28 | 0 | **37** · 37/0/37/25 | 0 |
| C_WRONG_BUT_RETRIEVED_PHONOLOGY_CORRECT (retrieval/homophone diagnostic) | 0 | 0 | 0 | 0 |
| LEXICAL_IDENTITY_CORRECT_AND_NATIVE_LTM_WRONG | 3,160 · 0/3,160/3,160/390 | 3,159 · 0/3,159/3,159/375 | 3,609 · 0/3,609/3,609/479 | 3,628 · 0/3,628/3,628/482 |
| LEXICAL_IDENTITY_WRONG_AND_NATIVE_LTM_WRONG | 39 · 0/35/39/7 | 34 · 0/34/34/7 | 38 · 0/33/38/9 | 34 · 0/34/34/7 |

**[EMPIRICAL_RESULT]**
* **Correct retrieval, failed decoding:** 99.86–100% of C-population native failures have a C-correct retrieval (3,160/3,164 = 99.87%; 3,159/3,159; 3,609/3,614 = 99.86%; 3,628/3,628). In most failures the encoder's ŝ retrieves the right lexical identity, yet the ventral decoder fails to produce its form from ŝ.
* **Wrong retrieval, correct decoding:** in SOURCE, 30 and 37 items have wrong C lexical identity but a correct native ventral output.
* **Homophone rows:** the 34–39 lexical-identity-wrong native failures are non-canonical homophone rows (plus the 4 and 5 C-wrong items in SOURCE). S1 rescues all of them whose retrieved row has the same phonology.

## 8. RADIAL VS DIRECTIONAL RESCUE (native failures, free-AR)

**[EMPIRICAL_RESULT]**

| state | native failures | S1 rescues | S3 rescues | S1 ∩ S3 | S3/S1 overlap | S1-only (not S3) | S3-only (not S1) | S2-only (not S1) | S1 regressions | S3 regressions |
|---|---|---|---|---|---|---|---|---|---|---|
| W3_SRC | 3,199 | 3,195 | 397 | 397 | 12.43% | 2,798 | 0 | 4 | 38 | 2,085 |
| W3_REP | 3,193 | 3,193 | 382 | 382 | 11.96% | 2,811 | 0 | 0 | 0 | 2,110 |
| W4_SRC | 3,647 | 3,642 | 488 | 486 | 13.34% | 3,156 | 2 | 5 | 46 | 2,545 |
| W4_REP | 3,662 | 3,662 | 489 | 489 | 13.35% | 3,173 | 0 | 0 | 0 | 2,556 |

(The 2 W4_SRC S3-only items are among the 5 S2-only items.)

* The majority of S1 rescues were **not** reproduced by S3; S3 reproduced 11.96–13.35% of them.
* S1-only rescues outnumber S1∩S3 by 6.5–7.4×.
* S3 is net harmful: −1,688, −1,728, −2,057 and −2,067 exact relative to S0.

**Descriptives** (frozen fields, medians, free-AR; `table_geometry_descriptives_freear.tsv`) **[EMPIRICAL_RESULT, descriptive only]**:
* **Native-wrong vs native-correct, C-correct items:**
  * cos(target, ŝ): 0.730 vs 0.750 (W3_SRC); 0.723 vs 0.742 (W4_SRC).
  * Lexical rank: 27,167 vs 13,253; 26,803 vs 13,066. Failures are rarer words.
  * ‖ŝ‖: 6.87 vs 6.92; 7.51 vs 7.49 (similar).
  * Retrieval margin: 0.265 vs 0.236; 0.259 vs 0.228 (not smaller for failures).
* **ŝ direction is far from its prototype even when retrieval and decoding both succeed:** median cos(target, ŝ) ≈ 0.74–0.75 on C-correct, native-correct items.
* **S3-rescued vs S3-not-rescued failures:** ‖ŝ‖ 8.08 vs 6.71 (W3_SRC); 8.61 vs 7.34 (W4_SRC). Median retrieved/ŝ norm ratio 0.78 vs 0.97; 0.74 vs 0.90.
* **S3-regressed vs S3-kept native-correct items:** ‖ŝ‖ 7.19 vs 6.89 (W3_SRC); 9.23 vs 7.37 (W4_SRC). Ratio 0.84 vs 0.94; 0.66 vs 0.88. S3 shrinks these vectors most strongly.
* REP states match to within ≤0.01.

**[INTERPRETATION]**
* Semantic-vector scale is not what separates native failures from successes. Imposing the prototype's norm helps a minority of failures, mainly those with the largest ‖ŝ‖, and disrupts many currently-correct items. The decoder's current behaviour on ŝ therefore depends on ŝ's own norm.
* What rescues is replacing ŝ's direction with the lexical prototype's. This is F2's allowed interpretation: movement toward the retrieved lexical prototype substantially increases decoder compatibility.

## 9. RETRIEVAL-LIMITED ITEMS

**[EMPIRICAL_RESULT]** (`table_retrieval_limited_summary.tsv`, `table_retrieval_limited_items_*`)
* **Native failures with S2 correct but S1 wrong:** W3_SRC 4, W4_SRC 5. All are C-population items with lexical identity wrong and retrieved phonology wrong; these are exactly the C_WRONG_AND_NATIVE_LTM_WRONG strata. REP states: 0.
* **S1 regressions (native correct, S1 wrong):** W3_SRC 38 (30 C-population, the `C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT` items, plus 8 non-canonical rows) and W4_SRC 46 (37 + 9). All have wrong lexical identity and wrong retrieved phonology. REP states: 0.
* **S1 wrong while lexical identity correct:** 0 items in every state, as Gate B implies.
* **Near-tie retrieval on C-wrong items:** median retrieval margin 0.0075 (W3_SRC) and 0.0072 (W4_SRC), against ≈0.24 overall.

**[INTERPRETATION]** F3 holds on 4 and 5 native failures (≈0.1% of failures) plus 38 and 46 S1 regressions in SOURCE only. Retrieval identity is limiting there, and Arm-A removes that limitation completely.

## 10. AUTOREGRESSIVE FAILURE DIAGNOSTIC (native S0 free-AR failures)

**[EMPIRICAL_RESULT]** (`ar_diagnostic_native_freear.tsv`, `fig4_*`)

| | W3_SRC | W3_REP | W4_SRC | W4_REP |
|---|---|---|---|---|
| failures | 3,199 | 3,193 | 3,647 | 3,662 |
| first divergence at step 0 | 1,184 (37.0%) | 1,182 (37.0%) | 1,299 (35.6%) | 1,298 (35.4%) |
| first divergence at EOS position (length error only) | 58 | 60 | 85 | 88 |
| gold rank 2 / 3–5 / >5 | 2,205 / 800 / 194 | 2,202 / 793 / 198 | 2,496 / 907 / 244 | 2,508 / 915 / 239 |
| margin (chosen−gold logit) median [q10, q25, q75, q90] | 3.19 [0.43, 1.26, 7.08, 13.60] | 3.19 [0.44, 1.28, 7.07, 13.56] | 3.21 [0.47, 1.31, 7.31, 13.58] | 3.27 [0.47, 1.28, 7.24, 13.54] |
| margin < 0.5 / < 1 / < 2 / ≥ 5 | 359 / 649 / 1,143 / 1,150 | 360 / 643 / 1,127 / 1,140 | 384 / 731 / 1,298 / 1,303 | 392 / 738 / 1,288 / 1,317 |
| margin_numerically_ambiguous | 0 | 0 | 0 | 0 |
| 1 positional mismatch / ≥ 2 | 248 / 2,951 (92.2%) | 241 / 2,952 (92.5%) | 320 / 3,327 (91.2%) | 323 / 3,339 (91.2%) |
| mean positional mismatches | 4.81 | 4.83 | 4.75 | 4.74 |
| predicted length = / < / > target | 873 / 1,345 / 981 | 866 / 1,344 / 983 | 1,062 / 1,400 / 1,185 | 1,064 / 1,409 / 1,189 |
| terminated by cap (no EOS in 12) | 1 | 1 | 3 | 3 |
| **DIAGNOSTIC_ONLY_PREFIX_CORRECTION → exact** | 1,968 (61.5%) | 1,956 (61.3%) | 2,232 (61.2%) | 2,252 (61.5%) |
| of which step-0 divergence / step ≥ 1 / EOS position | 40.2% / 74.0% / 100% | 40.0% / 73.7% / 100% | 39.5% / 73.2% / 100% | 39.5% / 73.6% / 100% |

**[EMPIRICAL_RESULT]**
* First errors occur early: 35–37% at the first phoneme, and about 72% within the first three positions (W3_SRC 2,318/3,199).
* Most failures cascade into multiple positional errors.
* First-divergence margins are broadly distributed: about 20% below 1 logit, about 36% at or above 5 logits.
* Under DIAGNOSTIC_ONLY_PREFIX_CORRECTION, correcting the single first divergent token and continuing greedily reaches the exact target in about 61% of failures. This is not model performance and not an inference mechanism.
* The distributions are nearly identical between SOURCE and POST_REPAIR.

**[INTERPRETATION]**
* In about 61% of failures, the remaining target continuation is recoverable from ŝ once the first error is corrected. The other about 39% contain further divergences; after a step-0 error only about 40% recover.
* Because S1/S2 remove essentially all failures under the identical decoder and AR loop, the early divergences and cascades are conditional on the ŝ input. They are not an input-independent limitation of AR deployment. F5 ("semantic interventions yield limited rescue") is therefore not satisfied.

## 11. SOURCE VS POST_REPAIR

**[STRUCTURAL_CODE_FACT]** Arm-A changes only `ltm.to_semantic.2.{weight,bias}`. The ventral decoder, `sem_to_h0`, encoder GRU, `to_semantic.0`, motor and gate are bitwise identical (lineage §5; preflight pair diff).

**[EMPIRICAL_RESULT]** (`fig5_*`)

| | W3 SRC → REP | W4 SRC → REP |
|---|---|---|
| C errors | 34 → 0 (34 fixed, 0 broken) | 42 → 0 (42 fixed, 0 broken) |
| S0 free-AR errors | 3,199 → 3,193 (64 W→C, 58 C→W) | 3,647 → 3,662 (97 W→C, 112 C→W) |
| native-failure set Jaccard | 0.963 | 0.944 |
| S1 errors | 42 → 0 | 51 → 0 |
| S2 errors | 0 → 0 | 0 → 0 |
| S3 errors | 4,887 → 4,921 (77 W→C, 111 C→W) | 5,704 → 5,729 (134 W→C, 159 C→W) |
| S3-rescue set Jaccard | 0.791 | 0.717 |
| S3/S1 rescue overlap | 12.43% → 11.96% | 13.34% → 13.35% |
| median cos(target, ŝ), all items | 0.74360 → 0.74346 | 0.73496 → 0.73470 |
| median ‖ŝ‖ | 6.896 → 6.892 | 7.460 → 7.453 |
| AR: step-0 share; median margin; prefix-fix exact | 37.0%→37.0%; 3.19→3.19; 61.5%→61.3% | 35.6%→35.4%; 3.21→3.27; 61.2%→61.5% |

**[INTERPRETATION]** Arm-A materially alters **retrieval identity**: C becomes perfect, and the retrieval-limited S1 subpopulation disappears. It does **not** materially alter the semantic-interface failure pattern. S0 native ventral errors, the S1-over-S3 rescue structure, the geometry medians and the AR first-divergence characteristics are all essentially unchanged. No decoder change is attributed to Arm-A; none exists structurally.

## 12. RESULT FAMILY

* **F2 DIRECTIONAL / PROTOTYPE RESCUE: best supported.** The evidence holds in both witnesses, in SOURCE and POST_REPAIR, and under both readouts. S1 rescues 3,195/3,199, 3,193/3,193, 3,642/3,647 and 3,662/3,662 native failures, while S3 reproduces only 11.96–13.35% of those rescues and introduces 2,085–2,556 regressions.
* **F3 RETRIEVAL-LIMITED: present but minor.** It covers SOURCE only (4 and 5 native failures, plus 38 and 46 S1 regressions) and is removed by Arm-A.
* **F1 RADIAL RESCUE: not supported.** The majority of S1 rescues were not reproduced by S3.
* **F4 ORACLE/NAMING INCONSISTENCY: not encountered.** Gate C passed with 0 item mismatches.
* **F5 AR AMPLIFICATION DOMINANT: not supported as dominant.** Early divergences and cascades are real, but the semantic intervention yields near-complete rescue under the same decoder.
* **F6 MIXED: not required.** The only secondary mechanism (F3) accounts for ≤0.14% of native failures and vanishes after repair.

## 13. WHAT IS ESTABLISHED

1. **[EMPIRICAL_RESULT]** The isolated ventral decoder of these four mature H512 states reproduces every lexical form (29,571/29,571) from raw true GloVe, under the repetition AR protocol (genuine free-AR and forced-length).
2. **[EMPIRICAL_RESULT]** At least 98.7% of all native isolated-ventral failures (and 99.86–100% of those in the C population) have correct lexical retrieval from the same ŝ. Replacing ŝ by the raw GloVe of its retrieved row removes them (all of them in POST_REPAIR).
3. **[EMPIRICAL_RESULT]** Matching ŝ's norm to that prototype without changing direction rescues about 12–13% of failures and breaks more than 2,000 correct items per state.
4. **[EMPIRICAL_RESULT]** The Arm-A head repair removes C and retrieval-identity errors but leaves the ventral failure set, rescue structure and AR diagnostics essentially unchanged.
5. **[EMPIRICAL_RESULT]** Native failures begin early, usually cascade, and are recoverable by one DIAGNOSTIC_ONLY prefix correction in about 61% of cases.
6. **[INTERPRETATION]** The residual native ventral repetition error is localized at the interface between encoder-produced ŝ and the ventral decoder: the decoder is compatible with lexical GloVe prototypes but not with the directional deviation of ŝ from them. Scale/calibration is not the principal factor.

## 14. WHAT IS NOT ESTABLISHED

* **[NOT_ESTABLISHED]** That a recurrent semantic attractor, or any refinement architecture, is **necessary**. S1 shows that a prototype input suffices; it does not show that iterative refinement is required, nor how much movement toward the prototype is needed (no interpolation was preregistered).
* **[NOT_ESTABLISHED]** That ŝ is "off-manifold". No operational manifold test was preregistered.
* **[NOT_ESTABLISHED]** *Why* the decoder fails on ŝ: which training factor (for example relative supervision from raw GloVe in Naming vs from ŝ in `L_dec`) produced the incompatibility. Only a structural code fact is known: the decoder is supervised from both.
* **[NOT_ESTABLISHED]** Generalization beyond these two seeds and these mature first-hit states, and any bearing on FULL-fusion repetition. FULL was not evaluated and was already at 0 errors historically.
* **[NOT_ESTABLISHED]** That prefix correction, scheduled sampling or any decoding change is a viable mechanism. The prefix-correction output is diagnostic only.
* **[NOT_ESTABLISHED]** That rarity (median lexical rank of failures about 2× that of successes) causes failures. This is descriptive only.

## 15. AT MOST TWO NEXT CANDIDATE INTERVENTIONS (not implemented; for CENTRAL arbitration)

1. **Semantic-refinement candidate.** Before `sem_to_h0`, move ŝ toward the lexical prototype of its own retrieved identity: a refinement or clean-up step on the ventral path that uses the frozen GloVe bank. It is tied directly to F2 (S1 ≈ 100% vs S3 ≈ 12%) and to C being ≥ 99.85% correct (100% post-repair). It is an architectural change, and CENTRAL must weigh it against the publication-path constraint. Its necessity is not established.
2. **Training-only candidate.** Make the existing ventral decoder compatible with encoder-produced ŝ without architecture change, targeting the directional ŝ–prototype discrepancy that S3 shows is not a norm problem. Examples are the relative supervision of the decoder from ŝ vs raw GloVe, or ŝ-to-GloVe directional alignment. The diagnostic readout would be the S0-vs-S1/S2 gap on this same frozen protocol. This is one pilot, not a sweep.

## 16. STOP CONDITION

The frozen diagnostic has been executed once and packaged. **STOP.** There was no training, no architecture change, no gate change, no lesion, no attractor and no follow-up semantic condition. The next scientific workstream awaits CENTRAL STEERING.
