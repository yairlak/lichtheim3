# VENTRAL_INTERFACE_EXPERIMENT_CONTRACT — LICHTHEIM3 VENTRAL SEMANTIC INTERFACE, FROZEN FACTORIZATION DIAGNOSTIC

Programme: POST_STAGE / PAPER_PROGRAMME. Decision authority: CENTRAL STEERING.

CONTRACT_STATUS=FROZEN

LINEAGE: `VENTRAL_INTERFACE_LINEAGE.md` (LINEAGE_STATUS=CLOSED, PROVENANCE_GATE=PASS)
Worktree / branch: `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-interface`, `paper-programme/ventral-interface-diagnostic`, based on `79f4e5bd94a9c1f82594f4050028b39c220b82b0`.
Frozen config: `ventral_interface_frozen_config.json`. Implementation hashes: `IMPLEMENTATION_MANIFEST.json`, `SHA256SUMS`.

This experiment is **diagnosis only**. It performs no training, architecture change, gate change, lesion, attractor, FULL fusion or dorsal readout. No scientific result existed when this contract was frozen; `--execute` had never been run.

---

## 1. Frozen state set (exactly four; no other checkpoint is authorized)

| state_id | witness | role | seed / u / step | base SHA256 | Arm-A head SHA256 | reconstructed_state_identity |
|---|---|---|---|---|---|---|
| W3_SRC | V6_seed19_u3825 | SOURCE | 19 / 3825 / 10625850 | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | NA | `6d7282854323b810727cd9c0c90cace6dd8a54d306408cafaa8c40c5455d10a0` |
| W3_REP | V6_seed19_u3825 | POST_REPAIR | 19 / 3825 / 10625850 | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` | `9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1` |
| W4_SRC | V6_seed20_u3040 | SOURCE | 20 / 3040 / 8445120 | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | NA | `32928707acc81af215faa829ad94bba566804982bc192bb5116ea32d43d56e29` |
| W4_REP | V6_seed20_u3040 | POST_REPAIR | 20 / 3040 / 8445120 | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03` | `e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6` |

* **Composite identity:** `sha256("gxlr-state-v1|" + base_sha + "|" + head_sha)`, with an empty head operand for SOURCE.
* **Reconstruction:** unmodified `scripts/gating_diagnostics/run_gate_route_audit.build_state` (manifest blob `e5f0bb4fa68ebe1b200a3fcabe3544ebbff5483b`). POST_REPAIR differs from SOURCE only in `ltm.to_semantic.2.{weight,bias}`, which preflight asserts.
* **Data:** lexicon `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66`; GloVe file `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`; raw bank tensor `4658e11e6a8f60a468472cc3fa62e71064e33da6790f924cc3005e602762ddb4`; canonical C population `10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50`.
* **Archived reference values** (from the V6 completion archive; hash-verified by preflight):

| state | C errors | Naming errors | free-AR LTM errors | canonical LTM errors |
|---|---|---|---|---|
| W3_SRC | 34 | 0 | 3199 | 3199 |
| W3_REP | 0 | 0 | 3193 | 3193 |
| W4_SRC | 42 | 0 | 3647 | 3647 |
| W4_REP | 0 | 0 | 3662 | 3662 |

These are existing historical records used only as reproduction gates (§6). They are not results of this experiment.

## 2. The four semantic inputs (the ONLY experimental variable)

For item row `i` in a given state, with `ŝ_i` the encoder output entering the ventral decoder:

| condition | vector entering `decode_from_s_hat` | code |
|---|---|---|
| **S0 NATIVE** | `ŝ_i`, passed through unchanged (identity supplier) | `SemanticInjection("S0")` |
| **S1 RAW_RETRIEVED** | `bank_raw[r_i]`, where `r_i` = the frozen historical C top-1 row (§3.4); raw / unnormalized | `SemanticInjection("S1", fixed=bank_raw[top1])` |
| **S2 RAW_TRUE** | `bank_raw[i]`, gathered by bank index exactly as `train_tasks.evaluate_naming` does | `SemanticInjection("S2", fixed=bank_raw)` |
| **S3 RADIAL** | `‖bank_raw[r_i]‖₂ · ŝ_i / ‖ŝ_i‖₂`, computed in float64 and cast to float32; `ŝ_i` is the live vector of that forward | `SemanticInjection("S3", target_norm=‖bank_raw[top1]‖)` |

**S3 degeneracy (preregistered).** If `‖ŝ_i‖₂ ≤ 1e-6` (float64 of the float32 vector), S3 is DEGENERATE for that item. The vector is left equal to `ŝ_i` so decoding stays defined, the item is flagged `s3_degenerate=1`, and it is **excluded from every S3 denominator and S3 transition count**. The count of degenerate items is reported. Retrieved raw norms are ≥ 2.61 on this bank, so the target norm is never degenerate.

**Injection mechanism.** One forward hook on `model.ltm.to_semantic` (whose output is `ŝ`, `ltm_route.py:146`) returns the condition vector, row by row, for **explicitly bound** item rows. The hook body is identical for every condition; only the supplier table differs. The hook fires at every AR step because the historical decoders re-run the encoder each step. It returns the same vector each time, which is checked.

**Forbidden:** interpolation, attractor, soft retrieval, top-k mixture, noise, any other transformation, and any condition-specific decoder branch.

## 3. Population and identity contract (reconstructed from frozen code)

### 3.1 Sets
* **P_R, lexical rows:** `range(29571)`. `item_index` = bank row = `tr.entries` position (entry-order SHA256 `ab2b193f98eee7fb270c3e708753aab7c58d5f1d59f7597182cd0b0154c68b76`). This is the historical repetition and Naming population, and **the evaluated item population of this experiment**. Every S0–S3 decode covers all 29,571 rows per state.
* **Phonological class of i:** all rows with identical `tuple(entries[i].phonemes)` (`train_tasks.phonology_groups`). There are 27,981 classes, of which 1,299 contain more than one row (homophone classes).
* **Canonical C target `c(i)`:** `min(class, key=(rank, index))` (`train_tasks.canonical_phonology_indices`).
* **P_C, canonical C targets:** `{i : c(i) == i}`, |P_C| = 27,981. The 1,590 rows with `c(i) ≠ i` are **non-canonical homophone members**.
* **Retrieval bank:** always all 29,571 rows, including non-canonical homophone rows as competitors.

### 3.2 Historical C correctness (the frozen C contract, not redefined)
Defined only for `i ∈ P_C`: `C_contract_correct(i) = 1` iff `argmax_j cos(normalize(ŝ_i), normalize(bank_raw)_j) == i` over all 29,571 rows. This is exactly `frozen_probe.comprehension_metrics.top1` as called by `train_tasks.evaluate_comprehension_subset(model, vocab, entries, bank_raw, comp_idx, "cpu", 512)`, i.e. the historical battery call. Ties resolve by `torch.argmax` (first maximal index). **It is exact lexical-row identity of the canonical target. Retrieving a homophone row is WRONG.** For `i ∉ P_C` the historical contract is undefined, so the field is `NA`.

### 3.3 Separately exposed booleans (all rows)
* `lexical_identity_correct = [r_i == i]`. On P_C this equals `C_contract_correct`; Gate A checks this.
* `phonology_correct = [phonemes(r_i) == phonemes(i)]`.
* `retrieved_is_homophone_not_target = [r_i ≠ i ∧ phonemes(r_i) == phonemes(i)]`.
* `in_C_population`, `canonical_C_index`, `canonical_C_identity`, `homophone_group` (= canonical row index of the class), `homophone_group_size`, `homophone_group_members` (words in bank order, `|`-joined).

### 3.4 Retrieval used for S1 and the retrieval fields
`r_i = comprehension_metrics(encode_all(model, vocab, all 29,571 forms, "cpu", 512), bank_raw, range(29571), 512).top1_idx[i]`. Gate R requires `r_i` to equal the historical C call's top-1 row for every `i ∈ P_C`.

* `cosine_top1`/`cosine_top2`/`top2_index`: normalized query against the row-normalized bank; top-2 is the argmax after masking the top-1 index.
* `retrieval_margin = cosine_top1 − cosine_top2`.
* `historical_target_margin`: the historical `margin` field (target cosine minus best other).
* `cosine_target_shat`: the historical `target_cos`.

### 3.5 Item metadata
* `phoneme_length = len(phonemes)`.
* `lexical_rank = LexEntry.rank`: the lexicon file's authoritative hybrid frequency-rank field (measured core plus deterministic Zipf-continued tail, `data/build_lexicon_en.py`); it is the field that defines the canonical C target.
* `lexical_frequency = NA`: `LexEntry.freq` is a Zipf value synthesized from rank (`_zipf_freq`), not a measured frequency, so it is not reported.

## 4. Decoding contract (identical for S0, S1, S2, S3)

Route: **isolated ventral only**, via `model.route_logits(enc_in, enc_mask, dec_in, route="ltm", collect=False, apply_noise=False)`. The computation is `ltm.encode` (hooked) → `decode_from_s_hat`: `h0 = tanh(sem_to_h0(s))`, `decoder` GRU over `phon_embed(dec_in)`, `dec_to_premotor` → `motor.proj` logits over 42 tokens. It includes no gate, no WM route and no FULL fusion; a runtime guard hard-stops if any of them executes.

Pinned from code (`gating_diagnostics/gate_probe.py`, blob `565b80962962c153ef47f081af1d9a9dab02ae0e`, mirroring `train_joint_scratch.JointScratchTrainer.free_ar_repetition` and `evaluate_train_lexicon_ceiling._ar_decode_batch`):

| element | GENUINE FREE-AR (**primary**) — `ar_decode_free(..., routes=("ltm",))` | CANONICAL FORCED-LENGTH AR (**secondary**) — `ar_decode_forced_length(..., routes=("ltm",))` |
|---|---|---|
| encoder input | `form + [EOS]`, pad 0, mask True on real tokens | same (`evaluate.hooks.make_batch`) |
| BOS | decoder starts from `[BOS]` (id 1) | same |
| greedy rule | `argmax` of the last-position logits (first index on ties) | same |
| feedback | the argmaxed token is appended to the decoder prefix (no teacher forcing) | same |
| max steps | **12** = `train_joint_scratch.FREE_AR_MAX_STEPS` (`train_joint_scratch.py:194`), imported by `gate_probe`, asserted `== 12` | `max(len(form) over batch) + 1` |
| stopping | loop breaks early only when **every** row contains EOS | none (runs all steps) |
| readout | `dec[k, 1:]` cut at the first EOS | window `dec[k, 1 : 1+len(form)+1]`, cut at the first EOS |
| target length | never consulted | consulted (per-item window) |
| EOS before target length | shorter prediction, wrong | same |
| EOS after target length / never | over-generation or non-termination (no EOS in 12) scored wrong | cannot be observed beyond the window; no EOS in window gives a (len+1)-token prediction, wrong |
| exact match | `prediction == form` (phoneme-id lists) | same |
| batch | 256 items in bank order | 256 |

The Naming convention (cap 10 in training evaluation, 256 in the witness batteries) is **not** used for S0–S3. It appears only in the separate Gate C validity check (§5).

Raw greedy tokens are recovered by a passive forward hook on `model.motor` that stores last-position logits per step. The raw tokens, cut at EOS, must equal the historical function's returned prediction for every row, or the run hard-stops.

## 5. S2 ↔ Naming validity control (Gate C, separate from primary S2)

The matched Naming conventions match the historical witness battery `coexistence_probe.full_battery`:
* Call: `train_tasks.evaluate_naming(model, vocab, entries, bank_raw, all 29,571 rows, "cpu", 256)`.
* Semantic input: raw `bank_raw[i]`.
* Decoding: `frozen_probe.semantic_greedy_decode` (BOS; greedy argmax; per-row EOS cut; break when all rows have EOS; cap **256**); batch 512.
* Scoring: exact phoneme match.

Gate C requires both of the following:
1. The historical call's error count equals the archived `naming_errors` (0 for all four states).
2. The S2 vector injected through this experiment's hook, decoded with `ar_decode_free(..., routes=("ltm",), max_steps=256)` at batch 512 in the same order, produces **per-item predictions identical** to the historical Naming predictions for all 29,571 items.

Failure means F4 (ORACLE/NAMING INCONSISTENCY): STOP.

## 6. Pre-execution validity gates (frozen; per state; order fixed; any failure → STOP, write `GATE_FAILURE.json`, no summary)

| gate | requirement |
|---|---|
| BANK | `model.ltm.semantic_bank` is bitwise equal to `F.normalize(bank_raw, dim=-1)` |
| **R** retrieval reproduction | historical C call error count == archived `c_errors`; retrieval row `r_i` == historical top-1 row for every `i ∈ P_C` |
| **A** C-correct identity | every `i ∈ P_C` with `C_contract_correct = 1` has retrieved lexical identity == target lexical identity (`r_i == i == c(i)`). Both `C_contract_correct` and `lexical_identity_correct` are stored; neither is silently equated with the other. |
| **B** raw vector equivalence | whenever `r_i == i`: `torch.equal(S1_i, S2_i)`; max absolute difference reported and required `== 0.0`. Failure → STOP interpretation. |
| **D** radial direction | for every nondegenerate item: `cos(S3, ŝ) ≥ 1 − 1e-6` (float64), and top-1 and top-2 bank indices unchanged, except for ties within `1e-6` cosine. Degenerate count reported. |
| **C** Naming path | §5 |
| **SHAT** | the live `ŝ` observed in the S0 free-AR decode (batch 256) vs the retrieval `ŝ` (batch 512): max abs deviation ≤ `1e-5` **and** identical top-1 row for every item |
| **H** native reproduction | S0 free-AR error count == archived `rep_freear_ltm_errors`; S0 canonical error count == archived `rep_canonical_ltm_errors` |
| **E** downstream identity | all four conditions and both conventions run through the same `gate_probe` function, route `ltm`, `sem_to_h0`, `decoder`, `dec_to_premotor`, `motor` and greedy code. The FULL/gate/dorsal guard is never hit. Parameter `state_dict` SHA256 is unchanged before vs after. The encoder output is bitwise stable across AR steps (max deviation `0.0`). Recorded tokens reproduce the historical predictions. |

Non-scientific tests (§10) additionally prove Gate E structurally on a toy model: the tensor entering `sem_to_h0` is exactly the supplied vector, and the call counts of every downstream module are identical across conditions.

## 7. Autoregressive diagnostic (native failures only; implemented, not yet run)

Scope: every item with S0 **genuine free-AR** exact_correct = 0, in every state.

1. **First divergence:** `t = min k` over target `form + [EOS]` such that the greedy token `≠` target token `k`. The prefix before `t` is gold by construction.
2. **Gold-prefix logits:** one teacher-forced forward of the isolated ventral route with `dec_in = [BOS] + form`, and the native `ŝ` injected as a fixed table (bitwise the vector seen in the free-AR decode).
3. **Recorded at t:**
   * `divergence_step`, `divergence_is_eos_position` (`t == len(form)`)
   * `gold_token`, `chosen_token`
   * `gold_rank` (1 + number of logits strictly above the gold logit), `gold_logit`, `chosen_logit`, `margin_chosen_minus_gold` (all from gold-prefix logits)
   * `top5_tokens`/`top5_logits`
   * `step_vs_goldprefix_logit_max_abs_dev` (free-AR step logits vs gold-prefix logits on the identical prefix)
   * `margin_numerically_ambiguous` (`|margin| ≤ max(that deviation, 1e-4)`)
4. **Cascade versus first error:**
   * `n_generated_after_divergence`: tokens generated after `t`, up to the first EOS.
   * `n_positional_mismatches`: aligned mismatches between prediction and form, over `max(len)` positions.
   * Only position `t` is the first divergence; everything after it is downstream cascade.
5. **DIAGNOSTIC_ONLY_PREFIX_CORRECTION.** The prefix is `[BOS] + generated[:t] + [target[t]]`. Greedy decoding then continues with the historical step function (`gate_probe._route_step_logits`, route `ltm`) until EOS or 12 generated tokens, at batch 1 with the fixed native vector. Recorded fields: `corrected_predicted_phonology`, `corrected_exact`, `corrected_second_divergence_step`, `corrected_terminated_by_cap`, `label = DIAGNOSTIC_ONLY_PREFIX_CORRECTION`. **This is not model performance and not an inference mechanism, and it never contributes to any primary metric or stratum.** It lives in a separate TSV and a separately labelled summary block.

**Numerical note (frozen).** On the real W3_SRC model (8 NON_SCIENTIFIC smoke items), free-AR step logits and gold-prefix logits agree bitwise from step 1 onward. They differ by ≤ ~2e-5 at step 0 (a length-1 versus length-n GRU call; logit magnitude ~80, relative ~2e-7). The frozen absolute tolerance is `LOGIT_PATH_TOL = 1e-4`.

## 8. Output schemas (frozen; `ventral_interface/schema.py`, `SCHEMA_VERSION = ventral-interface-schema-v1`)

### 8.1 `item_level_factorization.tsv`
One row per (state, item). 4 × 29,571 rows. Tab-separated with a fixed column order; any missing or extra key hard-stops.

* **Provenance:** `witness_id, state_id, source_or_repaired, seed, source_u, source_checkpoint_sha256, repaired_head_sha256 (NA for SOURCE), reconstructed_state_identity`
* **Identity:** `item_index, lexical_identity, target_phonology, phoneme_length, lexical_frequency (NA), lexical_rank, in_C_population, canonical_C_index, canonical_C_identity, homophone_group, homophone_group_size, homophone_group_members`
* **Retrieval / semantic:** `retrieved_index, retrieved_lexical_identity, retrieved_phonology, C_contract_correct (1/0/NA), lexical_identity_correct, phonology_correct, retrieved_is_homophone_not_target, cosine_target_shat, cosine_top1, cosine_top2, top2_index, retrieval_margin, historical_target_margin, shat_norm, retrieved_raw_glove_norm, true_raw_glove_norm, s1_equals_s2_row, s1_s2_max_abs_diff, s3_degenerate, s3_cos_to_shat, s3_top1_index, live_shat_vs_retrieval_shat_max_abs_dev`
* **Per condition ∈ {S0,S1,S2,S3} × convention ∈ {freear, canonical}**, prefix `{cond}_{conv}_`: `exact_correct, predicted_phonology, pred_length, eos_emitted, first_eos_step, eos_before_target_length, eos_after_target_length, terminated_by_cap (freear; NA for canonical), first_divergence_step, first_divergence_gold_token, first_divergence_pred_token`. For canonical, EOS fields are computed within the readout window.

Transitions are not stored per row. They are derived deterministically in the summary (§8.3).

### 8.2 `ar_diagnostic_native_freear.tsv`
`state_id, item_index, lexical_identity, target_phonology, phoneme_length, S0_freear_predicted_phonology, divergence_step, divergence_is_eos_position, gold_token, chosen_token, gold_rank, gold_logit, chosen_logit, margin_chosen_minus_gold, top5_tokens, top5_logits, step_vs_goldprefix_logit_max_abs_dev, margin_numerically_ambiguous, n_generated_after_divergence, n_positional_mismatches, diag_prefix_correction_label, diag_prefix_correction_corrected_predicted_phonology, diag_prefix_correction_corrected_exact, diag_prefix_correction_corrected_second_divergence_step, diag_prefix_correction_corrected_terminated_by_cap`

### 8.3 `summary_metrics.json`
Top-level keys: `schema_version, contract_sha256, provenance, gates, results, ar_diagnostic_native_freear`.

* **`results`:** `state_id → convention → stratum → condition → {denominator, exact_count, exact_proportion, transitions_vs_S0}`.
  * `transitions_vs_S0` = `{WRONG_TO_CORRECT, CORRECT_TO_WRONG, CORRECT_TO_CORRECT, WRONG_TO_WRONG}` for S1, S2 and S3; `null` for S0. Each is computed within the same state, convention and stratum, from S0 exact to Sx exact.
  * S3 denominators and transitions exclude S3-degenerate items.
* **`ar_diagnostic_native_freear`:** `state_id → {n_native_freear_failures, divergence_step_counts, divergence_is_eos_position_count, DIAGNOSTIC_ONLY_PREFIX_CORRECTION: {note, corrected_exact_count}}`.

**Strata.** "NATIVE_LTM_WRONG" means S0 exact_correct = 0 **under the same convention**.

| stratum | definition |
|---|---|
| ALL_REPETITION_ITEMS | all 29,571 rows |
| C_POPULATION | `in_C_population = 1` |
| NONCANONICAL_HOMOPHONE_MEMBERS | `in_C_population = 0` (C = NA) |
| NATIVE_LTM_WRONG | S0 wrong |
| **C_CORRECT_AND_NATIVE_LTM_WRONG** | `in_C = 1 ∧ C_contract_correct = 1 ∧ S0 wrong` |
| **C_WRONG_AND_NATIVE_LTM_WRONG** | `in_C = 1 ∧ C_contract_correct = 0 ∧ S0 wrong` |
| **C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT** | `in_C = 1 ∧ C_contract_correct = 0 ∧ S0 exact-correct` (the native ventral output phonology is correct although C lexical identity is wrong) |
| C_WRONG_BUT_RETRIEVED_PHONOLOGY_CORRECT | `in_C = 1 ∧ C_contract_correct = 0 ∧ phonology_correct = 1` (homophone retrieval). A supplement disambiguating the name above; it does not replace it. |
| LEXICAL_IDENTITY_CORRECT_AND_NATIVE_LTM_WRONG | all rows: `r_i == i ∧ S0 wrong` |
| LEXICAL_IDENTITY_WRONG_AND_NATIVE_LTM_WRONG | all rows: `r_i ≠ i ∧ S0 wrong` |

C strata are defined on P_C only, by the historical C contract. The lexical-identity strata cover all rows and never substitute for them.

## 9. Interpretation rules (preserved; applied by CENTRAL to the frozen summary)

The driver emits **no** F-label. Classification uses the primary convention (GENUINE FREE-AR) and reports the canonical convention separately. It is made per witness and state and must be consistent across W3 and W4 before any general claim.

* **F1 — RADIAL RESCUE.** S3 explains most S1 rescue. Interpretation: scale/calibration is a major factor. Do not prioritize a recurrent semantic attractor.
* **F2 — DIRECTIONAL / PROTOTYPE RESCUE.** S1 strongly rescues beyond S3. Interpretation: movement toward the lexical prototype increases decoder compatibility. Semantic refinement becomes plausible but is not proven necessary.
* **F3 — RETRIEVAL-LIMITED.** S2 succeeds but S1 fails because retrieval identity is wrong. Interpretation: identity/retrieval remains limiting.
* **F4 — ORACLE/NAMING INCONSISTENCY.** Matched S2 does not reproduce Naming (Gate C). STOP interpretation.
* **F5 — AR AMPLIFICATION DOMINANT.** Semantic interventions yield limited rescue while small early margins create large free-AR cascades. Decoder deployment robustness becomes a stronger candidate.
* **F6 — MIXED.** Substantially different mechanisms across subpopulations. Do not force one bottleneck.

Further binding rules:
* Any gate failure (§6) voids interpretation.
* "Off-manifold" must not be used as an established conclusion.
* DIAGNOSTIC_ONLY_PREFIX_CORRECTION outputs never count as rescue or performance.
* This contract fixes no numeric thresholds for "most" or "strongly"; that judgement belongs to CENTRAL on the frozen summary.

## 10. Implementation and execution protocol

| component | path |
|---|---|
| package | `ventral_interface/` (`__init__`, `conditions`, `injection`, `decode`, `population`, `schema`, `evaluate`) |
| driver | `scripts/ventral_interface/run_ventral_interface_factorization.py` |
| tests | `tests/test_ventral_interface.py` (toy model plus artifact hashes; `VI_SLOW=1` adds the double real preflight) |
| config | `paper_programme/ventral_semantic_interface/ventral_interface_frozen_config.json` |

Runtime requirements:
* `model.eval()` and `torch.inference_mode()` (plus the historical `@torch.no_grad`).
* `torch.use_deterministic_algorithms(True)`, CPU, float32.
* No optimizer, no backward, no parameter writes (hash-checked), no file writes to checkpoints or heads (files are hash-checked before use).

**`--preflight`** (the only mode run in the freeze pass):
* validates every hash, identity, archival copy and archived reference file;
* reconstructs all four states and checks structure, population counts and the SOURCE→POST_REPAIR parameter diff;
* runs NON_SCIENTIFIC mechanics on 8 fixed items of W3_SRC, recording booleans only, never predictions or correctness;
* verifies that every runtime-loaded worktree module is in the frozen closure;
* writes a deterministic report to `provenance/preflight_report.json`.

**`--execute`** is refused unless all of the following hold:
* `--contract-sha256` equals this file's SHA256;
* the file contains `CONTRACT_STATUS=FROZEN`;
* `git status --porcelain` is empty;
* a fresh preflight passes;
* `scientific_execution/` does not exist.

It then evaluates W3_SRC, W3_REP, W4_SRC and W4_REP in that order, with the gate order of `ventral_interface/evaluate.py`. Outputs go to `paper_programme/ventral_semantic_interface/scientific_execution/`.

Command, from the worktree root, after this contract's hash is recorded in `SHA256SUMS`:
```
python3 scripts/ventral_interface/run_ventral_interface_factorization.py --execute --contract-sha256 <SHA256 of this file>
```

## 11. Frozen tolerances and constants
`FREE_AR_MAX_STEPS = 12`; `DECODE_BATCH = 256`; `RETRIEVAL_BATCH = 512`; `NAMING_VALIDITY_MAX_STEPS = 256`; `NAMING_BATCH = 512`; `S3_DEGENERATE_NORM = 1e-6`; `GATE_D_COS_TOL = 1e-6`; `GATE_D_TIE_TOL = 1e-6`; `SHAT_MAX_ABS_DEV = 1e-5`; encoder AR-step deviation `0.0`; Gate B difference `0.0`; `LOGIT_PATH_TOL = 1e-4`; `TOP_K = 5`.

## 12. Notes for CENTRAL (no design change made)
1. **Stratum naming.** "C_WRONG_BUT_NATIVE_LTM_PHONOLOGY_CORRECT" is implemented literally (C lexical identity wrong, native ventral output correct). Because the name could also be read as "retrieved phonology correct", that reading is computed as the separate supplement `C_WRONG_BUT_RETRIEVED_PHONOLOGY_CORRECT`. Both are frozen and neither replaces the other.
2. **Implementation-integrity gates.** Gates R, SHAT and H are added. They only require reproduction of existing historical records and numerical consistency; they do not alter conditions A–E.
3. **Evaluated population.** All 29,571 lexical rows (the historical repetition population). C-based strata are restricted to the 27,981 canonical targets, as §3 requires.
