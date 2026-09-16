# VENTRAL_INTERFACE_LINEAGE — LICHTHEIM3 VENTRAL SEMANTIC INTERFACE, FROZEN FACTORIZATION DIAGNOSTIC

Programme: POST_STAGE / PAPER_PROGRAMME. Decision authority: CENTRAL STEERING.
Scope of this document: PROVENANCE + CODE-DEPENDENCY CLOSURE ONLY.
Nothing was trained, decoded, scored, lesioned or modified. The S0–S3 diagnostic was not run.
Written 2026-09-16 (audit pass in `wt-gate-x-lesion`). Carried into the dedicated diagnostic worktree `wt-ventral-interface` on 2026-09-16. Only worktree-dependent paths were updated (§0). Every scientific identity, full hash and provenance fact is unchanged.

# STATUS

LINEAGE_STATUS=CLOSED

PROVENANCE_GATE=PASS

Conditions attached to PASS (they apply to the next step, not to identity):
(a) the worktree holds untracked files from another workstream (§2). None of them is in the dependency closure, but the contract freeze must be committed on a state where they are resolved or explicitly excluded;
(b) the S0/S1/S2/S3 driver does not exist yet. Only its component closure is pinned (§6), and the driver must be frozen with the contract;
(c) the decode conventions listed in §8.4 are code facts that the contract must choose between. They are not provenance ambiguities.

---

## 0. Dedicated diagnostic worktree (carry-over, contract-freeze pass)

| field | value |
|---|---|
| dedicated worktree | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-ventral-interface` |
| branch | `paper-programme/ventral-interface-diagnostic` (new) |
| created from | `79f4e5bd94a9c1f82594f4050028b39c220b82b0` (= audited HEAD, verified exactly) |
| state at creation | `git status --short` empty; `git diff` empty; `git diff --cached` empty |
| audited worktree | `wt-gate-x-lesion` was left untouched (no delete, clean, stash, reset or edit) |
| GloVe in this worktree | `data/glove.6B.300d.txt` → symlink (git-ignored, one hop) → `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/lichtheim3/data/glove.6B.300d.txt`, SHA256 `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed` (unchanged) |
| SOURCE archival copies (read-only, `-r--r--r--`, byte-identical) | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/archives/ventral_interface_sources_20260916/s19_u3825_step_10625850.pt` = `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c`; `…/s20_u3040_step_08445120.pt` = `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3`. The authoritative identity remains the SHA256; the manifest paths still name the original working copies, which were only read. |
| lineage-pass evidence (copied) | `provenance/lineage_pass_structural_check/structural_check.{py,json}` (hashes as in §10) |

Sections §1–§11 below record the audit as performed in the audited worktree. Relative code paths resolve identically here: the tree is the same commit.

## 1. Repository / worktree / branch / HEAD

| field | value |
|---|---|
| main repository (git common dir) | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/lichtheim3/.git` |
| current worktree | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/wt-gate-x-lesion` |
| branch | `paper-programme/gate-x-lesion-recovery` |
| HEAD | `79f4e5bd94a9c1f82594f4050028b39c220b82b0` (2026-09-16 13:35 +0200, "fix(gate-x-lesion): wire frozen runner outputs to final validator contract") |
| upstream | none configured (`origin = https://github.com/yairlak/lichtheim3.git`) |
| interpreter | `/opt/homebrew/Caskroom/miniforge/base/envs/lichtheim3/bin/python3`, python 3.11.15, torch 2.12.1 (same as the V6 completion `ENVIRONMENT.txt` and the GATING provenance) |

Note: the parent directory `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3` is **not** a git repository. Relative artifact paths in the manifests resolve against it.

## 2. Cleanliness

* `git diff --stat`: empty. `git diff --cached --stat`: empty. **No tracked file is modified.**
* `git status --porcelain` restricted to every closure path in §6: **empty**.
* `git status --short` (untracked only, all from the GXLR workstream, none in the closure):

```
?? paper_programme/gate_x_lesion_recovery/SCIENTIFIC_INTERPRETATION_HANDOFF.md
?? paper_programme/gate_x_lesion_recovery/execution_run.log
?? paper_programme/gate_x_lesion_recovery/interpretation/
?? paper_programme/gate_x_lesion_recovery/scientific_execution/
?? scripts/gate_x_lesion/make_interpretation_artifacts.py
```

  plus this workstream's own new directory `paper_programme/ventral_semantic_interface/` (this file and `provenance/`).
* Ignored but scientific-output-relevant: `data/glove.6B.300d.txt`, a symlink (see §7).

Verdict: the worktree is **dirty (untracked only)**. The closure itself is clean. Scientific execution is stopped regardless, because this pass is not allowed to execute.

## 3. Git ancestry (full SHAs, all verified `merge-base --is-ancestor <c> HEAD` = YES)

| commit | role |
|---|---|
| `86ea3d329245ae2f0314d5c1f99be9ab575c7816` | Amendment V5 prereg (ceiling-assembly; `prereg_commit` in both heads) |
| `2e559e503b9672a3b9e89951fb114abfb77e0ad2` | V5 completion driver (applies frozen Arm-A at a non-endpoint source) |
| `ab25f7c93813d85272ac142cd6209dc30ae689ae` | Amendment V6 prereg + prospective R/N detector |
| `78f550570b82b3230cb621d9d000600ed192eb74` | V6 completion code commit; recorded as `code_commit` inside both head blobs, `code_dirty_paths=0`. Tip of `prospective/rn-detector-v6` (merge-base with HEAD = itself) |
| `f43ccd0971387ccfce006ddb0d6f985f1ee73441` | GATING frozen experiment commit |
| `993f2e73ecd0b9de2c3318255afbaa159948b09a` | GATING results commit. Tip of `paper-programme/gating-route-diagnostics` (merge-base with HEAD = itself) |
| `545ea436fa8f33480d74185a48a61874c04a17eb` | GXLR design freeze |
| `975560e6e2f1737f8915f56cf330223fd023be7c` | GXLR implementation |
| `4e65b14d6cfedb7eb73a9ff10d4a0869457c7e53` | GXLR IMPLEMENTATION_AMENDMENT 1 (`gate_x_lesion/identity.py`) |
| `c970c6629fc0f94f56516d2bd236bd3d2dea7706` | GXLR final rule freeze |

Referenced but not re-checked for ancestry (recorded only inside the head blobs): V4 prereg `054633cdf173c27296bdbc406775329bd0bc38a4`, V4 impl `3912d758e4864a2e8a297f38311c6162772562d8`.

**Code stability:** `git diff --stat 78f5505 HEAD -- models data utils evaluate scripts/naming_comprehension configs` is **empty**. Model, data, utils, evaluate and naming_comprehension code is byte-identical between the commit that produced the witnesses and HEAD. Only new packages were added afterwards (`gating_diagnostics/`, `gate_x_lesion/`, their scripts and tests).

## 4. Witnesses: artifacts, SHA256 computed now

Every hash below was computed from the file bytes on 2026-09-16 (`shasum -a 256`). The structural check re-hashed each file before and after loading it: unchanged.

### 4.1 SOURCE checkpoints

| witness | path (abs) | bytes | mtime | mode | SHA256 | role |
|---|---|---|---|---|---|---|
| V6_seed19_u3825 | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/probe_v6_completion_20260912/sources/s19_step_10625850.pt` | 31174770 | 2026-09-12T13:00:25 | `-rw-r--r--` | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | full training checkpoint; SOURCE state and POST_REPAIR base |
| V6_seed20_u3040 | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/probe_v6_completion_20260912/sources/s20_step_08445120.pt` | 31174386 | 2026-09-12T13:00:26 | `-rw-r--r--` | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | full training checkpoint; SOURCE state and POST_REPAIR base |

Upstream origin (recorded, not locally hashable): the V6 first-hit selector (`probe_v6_collect_20260912/s{19,20}_first_hit_decision.json`) selected
`/lustre/fsn1/projects/rech/llg/uss35bp/l3_prospective_v6_runs/prospective_rn_s19_u3600/checkpoints/step_10625850.pt` (step 10625850, u3825, point 45/60, the only eligible point) and
`/lustre/fsn1/projects/rech/llg/uss35bp/l3_prospective_v6_runs/prospective_rn_s20_u3000/checkpoints/step_08445120.pt` (step 8445120, u3040, point 8/60, the first of 26 eligible).
These local files are the only copies on this machine (a size-matched search found no others). They are **not archived** and are writable. Their identity rests on the SHA256 recorded at repair time (`extract_validation.json`, `source_battery.json`, `repaired_battery.json`, `RESULT_SUMMARY.json`), which the hashes computed now reproduce exactly.

### 4.2 Arm-A repaired heads (authoritative = archived `head_first_c0.pt`)

| witness | path (abs) | bytes | mode | SHA256 | role |
|---|---|---|---|---|---|
| seed19 | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/archives/prospective_rn_detector_v6_completion_20260912/seed19_u3825/arm_a/head_first_c0.pt` | 1849495 | `-r--r--r--` | `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` | replaces `ltm.to_semantic.2.{weight,bias}` |
| seed20 | `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/archives/prospective_rn_detector_v6_completion_20260912/seed20_u3040/arm_a/head_first_c0.pt` | 1849495 | `-r--r--r--` | `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03` | replaces `ltm.to_semantic.2.{weight,bias}` |

Hash-identical working copies (historical recorded paths, writable, non-authoritative):
`/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/probe_v6_completion_20260912/seed19/seed19_u3825_A/head_first_c0.pt` (`8865ba95…7afcfc`, identical) and
`/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/probe_v6_completion_20260912/seed20/seed20_u3040_A/head_first_c0.pt` (`724ed4c6…b972d03`, identical).

**Not the witness heads (excluded):** `seed19_u3825/arm_a/head_final.pt` `590cb143b1d56f139d49ecc77224fd04677de30212fafe40f0dfa548441ecc74` and `seed20_u3040/arm_a/head_final.pt` `64c4c6a3cdfbe87a6ce87d6b3082f946c30a5d1a3a4f8877fb6b9c89dd4dd6bb`. The repaired batteries, `RESULT_SUMMARY.json`, the GATING manifest and the GXLR manifest all name `head_first_c0` as the applied head.

Archive integrity: `shasum -a 256 -c archives/prospective_rn_detector_v6_completion_20260912/SHA256SUMS` passes for all 22 files.

Head blob metadata, read from the blobs themselves:

| field | seed19 | seed20 |
|---|---|---|
| `arm` | `A` | `A` |
| `seed` / `source_u` | 19 / 3825 | 20 / 3040 |
| `trainable` | `[ltm.to_semantic.2.weight, ltm.to_semantic.2.bias]` | same |
| `state` keys (shape, dtype) | `2.weight` (300,512) float32; `2.bias` (300,) float32 | same |
| `iterate` | 14 (= `first_c0_iter`) | 25 (= `first_c0_iter`) |
| `code_commit` / `code_dirty_paths` | `78f550570b82b3230cb621d9d000600ed192eb74` / 0 | same |
| `prereg_commit` | `86ea3d329245ae2f0314d5c1f99be9ab575c7816` | same |
| `label` | `DERIVED_GRADIENT_TRAINING_DIAGNOSTIC/NOT_OFFICIAL_MODEL/NOT_TRAINED_BY_JOINT_DRIVER` | same |

The head blob also carries `theta_float64`, which is **not** applied. The deployed evaluation applies the float32 `state` tensors only (see the archive's "fp64 fit, fp32 deployed evaluation").

## 5. POST_REPAIR reconstruction

**Storage form:** POST_REPAIR is **not** a stored checkpoint. It is reconstructed as SOURCE base checkpoint + Arm-A head.

**Canonical reconstruction** (used unchanged by the V6 battery, GATING and GXLR):
`scripts/gating_diagnostics/run_gate_route_audit.py::build_state` →
`frozen_head_probe.build_trainer(ckpt, "cpu", <worktree>/data/glove.6B.300d.txt)` (= `base123_error_audit.build`) →
`frozen_head_probe._isolated_model(tr, "p_last_hinge", head["state"])`: `copy.deepcopy(tr.model)`, load `ltm.to_semantic.state_dict()` with `2.weight`/`2.bias` replaced by clones of the head tensors, `model.eval()`.
Guards: the source SHA256 must equal the manifest before load, the head SHA256 must equal the manifest, and `set(head["state"]) == {"2.weight","2.bias"}`. The V6 battery driver `ceiling_source_completion.cmd_battery` applies the same `_isolated_model` call and key guard.

**Parameter-level verification** (read-only structural load, `provenance/lineage_pass_structural_check/structural_check.{py,json}`):

| check | W3 (seed19) | W4 (seed20) |
|---|---|---|
| model tensors in `state_dict` | 31 | 31 |
| same key set / same shapes SOURCE vs POST_REPAIR | yes / yes | yes / yes |
| tensors that differ | **only** `ltm.to_semantic.2.bias`, `ltm.to_semantic.2.weight` | **only** the same two |
| POST_REPAIR param == head `state` tensor (bitwise) | yes | yes |
| ventral decoder (`ltm.sem_to_h0.*`, `ltm.decoder.*`, `ltm.dec_to_premotor.*`) identical | yes | yes |
| LTM encoder GRU, `ltm.to_semantic.0.*` identical | yes | yes |
| dorsal `wm.*` identical | yes | yes |
| `motor.*`, `phon_embed.*` identical | yes | yes |
| gate | no parameters exist (`gate.*` is empty); config alpha 2.0 / threshold 0.7 / usage_prior 0.5 identical | same |
| `semantic_bank` buffer (non-persistent) identical SOURCE vs POST_REPAIR | yes | yes |
| source trainer model untouched by isolation | yes | yes |
| `model.training` | False / False | False / False |
| SOURCE `to_semantic.2.weight` tensor-SHA256 = archived `W0_to_semantic_2_weight` | `2d8e278a5f7bd3bd9729ad4cf8d2f53d2321ee1f9bd179780dc310407d11ad24` ✔ | `ccbb033d8dfcafddc8435fd836b7ae374906312e78bd99897e1e4078639a9d4f` ✔ |
| SOURCE `sem_to_h0.weight` tensor-SHA256 = archived value | `461545f4067373a7f0bbfd9f867dbf8f54e3ab1c7f801f161594704272fc13ea` ✔ | `ab2eb905be3f8db3ac72cfecfd6d4e6495c3a212b5ad70b708b1df07b7dab63f` ✔ |
| POST_REPAIR `to_semantic.2.weight` tensor-SHA256 (new record) | `e8f6f43bad574527fc63675434928d9c1d788b45f3872c73742158d5275f7578` | `40ccef4c50a34448424d80a9cf0a52d30ef0d8c7aa1c94edd7d7ec56dfd8c963` |

**Optimizer state:** the checkpoints carry `optimizer_state_dict`, and `JointScratchTrainer.load_state_dict` restores it into the trainer's AdamW. No evaluation path touches the optimizer, and `_isolated_model` copies only the model. It is irrelevant to evaluation outputs.

**No conversion.** Checkpoint `format == "lichtheim3.joint_scratch.v1"`. The live `DualRouteModel` is built at the checkpoint's own widths and loads it without renaming, reshaping or dtype conversion. `ltm_encoder_mode` is valid, and `n_glove_fallback == 0`, so no fallback path fired.

### 5.1 Reconstructed-state identities (pre-existing authoritative method, recomputed)

An authoritative composite hash already exists in this lineage. It is defined in `gate_x_lesion/identity.py` (blob `f8e2b8a62355036f74e5e28d063cc062b603a550`, IMPLEMENTATION_AMENDMENT 1 `4e65b14d…`) and frozen in `paper_programme/gate_x_lesion_recovery/checkpoint_manifest.proposed.tsv`:

```
RECONSTRUCTED_STATE_SHA256 = sha256_hex( ASCII("gxlr-state-v1|" + base_artifact_sha256 + "|" + applied_head_sha256) )
    (head operand = "" for a SOURCE state with no head)
```

Recomputed now from the **file hashes computed now**, and equal to the frozen manifest:

| state | base SHA256 | head SHA256 | gxlr-state-v1 |
|---|---|---|---|
| W3_SRC (V6 s19 u3825 SOURCE) | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | — | `6d7282854323b810727cd9c0c90cace6dd8a54d306408cafaa8c40c5455d10a0` |
| W3_REP (V6 s19 u3825 POST_REPAIR) | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` | `9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1` |
| W4_SRC (V6 s20 u3040 SOURCE) | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | — | `32928707acc81af215faa829ad94bba566804982bc192bb5116ea32d43d56e29` |
| W4_REP (V6 s20 u3040 POST_REPAIR) | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` | `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03` | `e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6` |

This identity is a function of the two **file** digests only. It is used here as the reconstructed-state name. No new identity scheme was invented.

## 6. SOURCE ↔ POST_REPAIR pair table

| field | W3 = V6_seed19_u3825 | W4 = V6_seed20_u3040 |
|---|---|---|
| seed (ckpt / head / battery) | 19 / 19 / 19 | 20 / 20 / 20 |
| global_step (ckpt / first-hit selector) | 10625850 / 10625850 | 8445120 / 8445120 |
| u (selector / head / manifest) | 3825.0 / 3825 / 3825 | 3040.0 / 3040 / 3040 |
| selection | preregistered FIRST eligible detector point (Rcan=Rfree=N=0; C excluded) | same |
| SOURCE SHA256 | `a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c` | `0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3` |
| Arm-A head SHA256 | `8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc` | `724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03` |
| POST_REPAIR identity | `9185aa5671e2ae3ab1e0860d52424b472f616d2515057d7ccbc6b383cf43dff1` | `e151b306f706e9db9828b3c9e742984a0ae836dbd97e73f86e4e2063bd7c5cd6` |
| regime / schedule / subset_mode | j0 / interleaved_123 / final_full | same |
| widths wm / ltm_enc / ltm_dec | 128 / 512 / 512 | same |
| ltm cfg | phon_embed 64, enc_layers 1, unigru_last_hidden, bidirectional False, ventral_noise 0.0 | same |
| wm interference_noise | 0.0 | 0.0 |
| gating | alpha 2.0, threshold 0.7, usage_prior 0.5 | same |
| semantic_dim | 300 | 300 |
| optimizer_policy / dec_weight / c_align_weight | shared_adamw / 0.5 / 0.0 | same |
| lexicon_path / lexicon_file_sha256 (ckpt) | `data/lexicon_en_glove_covered.tsv` / `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66` | same |
| comprehension population n / sha256 | 27981 / `10c2f06eda769bf620ca3dbb9889204e4431cac2bfe0d0f5dd37fa4df2bb9f50` | same |
| bank_raw tensor-SHA256 (live = archived) | `4658e11e6a8f60a468472cc3fa62e71064e33da6790f924cc3005e602762ddb4` | same |

**Pairing evidence chain (each link verified from artifacts, not documentation):**
1. The first-hit selector json selects the lustre step 10625850 / 8445120.
2. `extract_validation.json` records a local source with that `global_step` and `seed`, and SHA256 = the values in §4.1.
3. `arm_a/trace.json` + `RESULT_SUMMARY.json` `heads_sha256.first_c0` = the values in §4.2.
4. `repaired_battery.json` records `head_sha256` = `first_c0` and `source_sha256` = the same source.
5. The head blob's own `seed`/`source_u`/`arm`/`code_commit` agree.
6. The GATING manifest (blob `e5f0bb4fa68ebe1b200a3fcabe3544ebbff5483b`) and the GXLR proposed manifest (blob `00afe9b5cd00b7dc27798b853d1531b8056dcd02`) carry the same pairs.
7. Live reconstruction differs only at the intended head.

Both pairs are **DEMONSTRATED**. No pair was rejected.

## 7. Lexicon / semantic-bank / GloVe provenance

| item | path | bytes | SHA256 (now) | agrees with |
|---|---|---|---|---|
| lexicon (used) | `wt-gate-x-lesion/data/lexicon_en_glove_covered.tsv` (tracked, blob `5aaab2af7b6dfaabf8dffc241f7ab41324b6cb7b`) | 825434 | `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66` | ckpt `lexicon_file_sha256`; `extract_validation.lexicon_sha256` |
| lexicon_en.tsv (bundled default, not used) | `wt-gate-x-lesion/data/lexicon_en.tsv` | 838758 | `1aba9a2f972293f94626461454f0cbae709766108b6cbfe3b0495729896fd959` | — |
| GloVe raw | `wt-gate-x-lesion/data/glove.6B.300d.txt` → symlink → `wt-gating-diagnostics/data/glove.6B.300d.txt` → symlink → `wt-head-probe/data/glove.6B.300d.txt` → realpath `/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/lichtheim3/data/glove.6B.300d.txt` (git-ignored) | 1037962819 | `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed` | `extract_validation.glove_sha256` (both seeds) |

Load facts (live): 29571 source rows → 29571 entries, 0 unknown-phoneme or length filtered, 29571 unique words, `n_glove_found = 29571`, `n_glove_fallback = 0`. Canonical C population is 27981 with hash `10c2f06e…`, enforced by the trainer. Vocab size 42 (`<pad>`=0, `<bos>`=1, `<eos>`=2), `itos` sha256 `d88e9cc7dda9273f10f93c717e4e233835609d907cdd33a66bdfee0677099c8e`. Max form length 9. Entry order sha256 `ab2b193f98eee7fb270c3e708753aab7c58d5f1d59f7597182cd0b0154c68b76` (= the GATING `item_order_sha256`).
The checkpoint-stored `config.data.glove_path` is a lustre path. It is **not** used: `build_state` passes the worktree GloVe path explicitly.
The semantic bank is **not** frozen separately. `semantic_bank` is a non-persistent buffer rebuilt at load as `F.normalize(bank_raw)`, so its identity is the GloVe file + lexicon file + load code. It is pinned by the bank_raw tensor hash `4658e11e…`, and live `semantic_bank == F.normalize(bank_raw)` holds bitwise.

## 8. Evaluator / code dependency closure

All paths are relative to the worktree root. For every file, blob@HEAD == blob@`78f5505` unless marked NEW; the file SHA256 is of the working-tree file, which equals HEAD (closure clean).

| # | path | git blob | file SHA256 | key symbols | why in closure |
|---|---|---|---|---|---|
| 1 | `models/dual_route.py` | `5166df420e0f70994e76c68a3542f6a3abfe9809` | `dfed9c0083d9b5ec88bec091fa71e74614faeb55b28ced344849ba51ad4425da` | `DualRouteModel.forward`, `route_logits`, `set_semantic_bank` | model construction; route isolation (`route="ltm"`) |
| 2 | `models/ltm_route.py` | `548cc68735830986ee38801997c83b2781252d22` | `49f8cd49b76413c6eb3040c3b90cada2c3e1e31fee842286867bcb8de73b0073` | `LTMLexicon.encode`, `to_semantic`, `decode_from_s_hat`, `sem_to_h0`, `set_semantic_bank`, `lexical_field` | ŝ production; semantic→form decoder; bank normalization |
| 3 | `models/wm_route.py` | `e58b9670e708d419d88367f4cc8f27b92b4f4c3e` | `5933765b288e852ee4408058860db5a428c21361052141478dc190089b8b21a7` | `WMRecurrent` | construction/state-dict load (not used on isolated ventral readouts) |
| 4 | `models/gating.py` | `8b9244abeb1ad9150c481ab25b316d0eda3edf8e` | `3d87ddbf80f540c77262ab7bb310cee271ca849d5e406beeef374c8f7ff11677` | `build_gate` | construction only; FULL fusion excluded by design |
| 5 | `models/motor.py` | `cf8ffef02450451b7e84f5ce554bd82ac41bdd6f` | `5b8998e7e54b74d2c765e3f24403bad52cab034f4780b12d7910a16177f91906` | `MotorCortex.proj` | shared readout for every decode |
| 6 | `config.py` | `799b513dc1acc0caa352efb22a550895f907138f` | `9a8088982995a2b53edf4ee81d214ff1ab3603149dd9db9f73a307812cd80453` | `Config`, `LTMConfig`, `default_config` | architecture config |
| 7 | `losses.py` | `67ddf64f37d5e3210b79e9f4591f5ab80394c4a3` | `382eed2335ee2add7a00f6de9b73a788cdef5e94155ae1970e38bd5d7dcefd80` | `total_loss` (L_dec), `alignment_loss` | code fact on training supervision (§9); imported by trainer |
| 8 | `data/lexicon.py` | `4a5d69f81c24cc7fea9ad891f516d0617d7f49bd` | `0e4936d291f27d23d0532ff1aaa93a28b69caf582b37fca5cfb0d0f6ea578d0f` | `_load_glove_map`, `build_bundled`, `build_lexicon`, `LexEntry` | raw GloVe/lexicon loading; fallback logic |
| 9 | `data/phonemes.py` | `7bf5bb7f28b7f577de6d74341821d70cb98e701b` | `b093764b744a7be621763a87aacb94ebcd145a04ebaa647f6f261264d9aa1c98` | `Vocab`, `build_vocab`, PAD/BOS/EOS | token vocabulary, BOS/EOS ids |
| 10 | `data/lexicon_en_glove_covered.tsv` | `5aaab2af7b6dfaabf8dffc241f7ab41324b6cb7b` | `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66` | — | lexicon data |
| 11 | `data/glove.6B.300d.txt` (ignored, symlink chain §7) | — | `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed` | — | raw semantic vectors |
| 12 | `scripts/naming_comprehension/train_joint_scratch.py` | `95295d63560ae4c235a6beee8dfb47166f4ed30d` | `766244c1363d998f34fe3e4f725b57a9b3f95cf716876d71470d804cf962765c` | `JointScratchTrainer.__init__/load_state_dict/free_ar_repetition`, `canonical_config`, `FREE_AR_MAX_STEPS=12`, `NAMING_MAX_STEPS=10`, `EXPECTED_CANONICAL_C_HASH`, `HOMOPHONE_POLICY_NOTE` | model/bank build, guards, genuine free-AR, caps, canonical C |
| 13 | `scripts/naming_comprehension/train_tasks.py` | `0b3a1f19addcc2351edf75248ba31333ccd7cf59` | `a867b6b5ec474fb79fcd18e769fa8d16d8bdc3de920f7704bfa9be2e7b907303` | `canonical_phonology_indices`, `verify_bank_mapping`, `make_batches`, `naming_forward`, `naming_objective`, `retrieval_loss`, `evaluate_naming`, `evaluate_comprehension_subset`, `repetition_snapshot` | homophone/canonical C; naming from raw GloVe; canonical AR wrapper |
| 14 | `scripts/naming_comprehension/frozen_probe.py` | `8e54f361b7cd13647f31434d0477f391c38da547` | `d4c7bb9289cef56d09d05b1cbcbd9b47f0fa32dda549b333d29d8a4753689d4f` | `encode_all`, `comprehension_metrics` (top-1 idx), `semantic_greedy_decode` | ŝ batch encode; Comprehension top-1 retrieval; semantic-vector greedy decode |
| 15 | `scripts/naming_comprehension/frozen_head_probe.py` | `72b5f46d8a007adde37a4915912c1e9ed949a21b` | `f2a30620aad79bba81bbd9e56ab0b1a38a74801c05e35b0f9e74f7bb63fb851b` | `build_trainer`, `_isolated_model`, `sha256_tensor` | head reconstruction |
| 16 | `scripts/naming_comprehension/base123_error_audit.py` | `9884be44b7f86e2b3c125f5a8f1a61c9cda3a517` | `15008114d5e5ff3bde51b550193c07e62d5566347a29510ec9a75bb8ae910b37` | `build` | trainer rebuild at checkpoint's own config |
| 17 | `scripts/naming_comprehension/coexistence_probe.py` | `513d7ed1c8acadf8b31937210a3d8a5658b3c1de` | `de715e3c573c44d00c0fd8bb56fef1e916c79b4f1e58af3965cb3f48d828ecef` | `full_battery` | historical witness battery (reference numbers) |
| 18 | `scripts/naming_comprehension/ceiling_source_completion.py` | `4e6fa297cba53871158109feb50e05b43371f5e6` | `fc1a0865bba0810ba03a3a01a54feb234dd6cca7c341a94527c093f41f18088f` (= archived copy) | `cmd_battery` | historical witness reconstruction |
| 19 | `scripts/evaluate_train_lexicon_ceiling.py` | `089ab48c2cce8f8f11adff39b9284268c1835a98` | `352d9a4b950923b90fc2a7e8e03476ecb938f86850667b927648fa4f83b32a26` | `_ar_decode_batch`, `evaluate_forms_ar`, `make_batch` | canonical forced-length AR |
| 20 | `evaluate/hooks.py` | `96cf63d921790a853388b7cad5cc7ba46e25f966` | `113d0aeb367e0d181f5f3216ccd515df4658170293db44a5aad25ce5a022f074` | `route_predictions` | auxiliary TF readout imported by `repetition_snapshot` |
| 21 | `scripts/external_eval.py` | `1b4ca00cad28f1d40197c7769ac14343c1397f81` | `8931a1c92c716124862c170ec7854acc002befc1c4094051b406ef056b7174f1` | `_edit_distance` | naming edit metric |
| 22 | `utils/seed.py` | `7ad61801146090b0fae885b3637ec538d4cd9f22` | `aa7c8bbba3e036417bd54a9e8b67a1a09be156bc4542fa0adf10f62e4563f344` | `set_seed` | model-init RNG before state load |
| 23 | `utils/provenance.py` | `78681721041424b10e7c1ba85fbbdea638ab0bf8` | `c8bb5d4a58d6e48b29a80e83dbe8e3741bb5bdb23173af8fd9df554020b540a7` | `sha256_file` | imported by trainer |
| 24 | `scripts/naming_comprehension/train_multitask.py` | `6cf71dbdac98400b6faeecc6d4be59de514e18bc` | `d4963f59c51aa46c877d9abf85078352674dbbad32ece264516e9b44d1ea1d96` | imports only | import-time dependency of trainer |
| 25 | `scripts/naming_comprehension/aggregate_cohort.py` | `a90eed6a1a58308e82031c2afbf9398aa356621c` | `9c22adab9571aa96b68ac4b05ea557137e59677c3611ebeaee4540aedeca0c1b` | `FREQ_BANDS` | import-time dependency |
| 26 NEW | `gating_diagnostics/gate_probe.py` | `565b80962962c153ef47f081af1d9a9dab02ae0e` | `a60ef0c21b922ae9cf5a9a15bd8a7a7c170d275691bcdfa17bbdd681c7b6b02d` | `ar_decode_forced_length`, `ar_decode_free`, `_route_step_logits` | audited canonical + genuine free-AR route-isolated decoders (GATING/GXLR) |
| 27 NEW | `gating_diagnostics/__init__.py` | `616ff6601ecd89f39d9d68ff47b300c305f4fccb` | `e844d00fcd5908694f71d4c75bb4914eef701d1ebe5d0ae586115d197484f081` | exports | — |
| 28 NEW | `scripts/gating_diagnostics/run_gate_route_audit.py` | `fa0a259802e8b9b0caa48d7d4d5ac74b79c22ceb` | `79feb855d890e3c6a5681926b2ab851a807b6d0375f89757024247a55cc024ce` | `build_state`, `sha256_file`, `load_manifest` | canonical manifest-driven reconstruction |
| 29 NEW | `gate_x_lesion/identity.py` | `f8e2b8a62355036f74e5e28d063cc062b603a550` | `413e3648cc0e6bfb5375e610e542e955c4ca6d3b5ebf29ce2b43e8a729c7f2ad` | `reconstructed_state_sha256` | authoritative composite-state identity |
| 30 NEW | `paper_programme/gating_route_diagnostics/checkpoint_manifest.tsv` | `e5f0bb4fa68ebe1b200a3fcabe3544ebbff5483b` | `eeb6b9703ada869e443874fdfe49d714e231f4d2410d32fc4938a1d0fe848d54` | — | frozen witness manifest |
| 31 NEW | `paper_programme/gate_x_lesion_recovery/checkpoint_manifest.proposed.tsv` | `00afe9b5cd00b7dc27798b853d1531b8056dcd02` | `12cf15612ff7f0e6397b1d5e624f32832503096627da7ad7ed4d4c7ba93a328c` | — | frozen state identities |

Mapping to the 15 required closure items:
1 model construction/loading → 1,2,5,6,12,15,16,28.
2 ŝ → 2 (`encode`), 14 (`encode_all`).
3 semantic bank → 2 (`set_semantic_bank`), 12 (`bank_raw`), 13 (`verify_bank_mapping`).
4 raw GloVe/lexicon → 8,10,11,12.
5 Comprehension top-1 → 14 (`comprehension_metrics.top1_idx`).
6 Naming from raw semantics → 13 (`evaluate_naming`), 14 (`semantic_greedy_decode`).
7 isolated ventral repetition → 1 (`route_logits(route="ltm")`), 19, 26.
8 canonical forced-length AR → 19 (`_ar_decode_batch`), 26 (`ar_decode_forced_length`).
9 genuine free-AR → 12 (`free_ar_repetition`), 26 (`ar_decode_free`).
10 BOS / 11 EOS → 9 plus each decoder.
12 max-step conventions → 12, 14, 19, 26 (§8.4).
13 phoneme vocabulary → 9.
14 homophone/canonical C → 12, 13.
15 checkpoint/head reconstruction → 15, 28, 29.

**Not yet in closure:** the S0/S1/S2/S3 driver itself (vector injection into `decode_from_s_hat` under the repetition AR conventions). It does not exist. It must be written, audited and frozen with the experiment contract.

### 8.4 Decode conventions: code facts the contract must choose between

* **Canonical forced-length AR** (`evaluate_train_lexicon_ceiling._ar_decode_batch`; the same semantics in `gate_probe.ar_decode_forced_length`): BOS start, greedy for `max(len(form))+1` steps over the batch, per-item readout window `dec[1 : 1+len(form)+1]`, cut at first EOS. Consults target length.
* **Genuine free-AR** (`JointScratchTrainer.free_ar_repetition`; `gate_probe.ar_decode_free`): BOS start, global cap `FREE_AR_MAX_STEPS = 12`, batch loop breaks only when every row has emitted EOS, each row cut at its first EOS. A row without EOS yields all generated tokens.
* **Naming semantic greedy decode** (`frozen_probe.semantic_greedy_decode`): same BOS/greedy/EOS logic on `decode_from_s_hat(sem, ·)`. The cap is the caller's `max_steps`. Training-time evaluation passes `NAMING_MAX_STEPS = 10`. **The V6 witness batteries (`coexistence_probe.full_battery`) passed `evaluate_naming(..., "cpu", 256)`, i.e. `max_steps = 256`.** Because the longest form is 9 phonemes, per-item exact-match is invariant across caps ≥ 10 (a correct output needs ≤ 10 steps; any row without EOS by step 10 is wrong under every cap). EOS-emission rate and edit distance are **not** invariant. The contract must name the cap for S1–S3.
* Route-isolated LTM repetition re-runs the encoder at every step (`route_logits → ltm.forward → encode → decode_from_s_hat`). With `eval()`, `ventral_noise = 0.0` and a packed unidirectional GRU, ŝ does not depend on the decoder prefix. It is therefore structurally the same as a one-shot encode followed by `decode_from_s_hat(ŝ, ·)`. Archived `rebatch_max_dev_same_items_diff_chunking = 0.0`.

## 9. Semantic path: code facts (structural, no empirical interpretation)

* **ŝ production** (`ltm_route.py:127–147`): `phon_embed` (64) → unidirectional 1-layer GRU(64→512) on `pack_padded_sequence` → `pooled = h[-1]` (B,512) → `to_semantic = Linear(512,512) → GELU → Linear(512,300)` → **ŝ (B,300)**. Encoder input is `form + [EOS]`, mask True on real tokens, pad id 0. **ŝ is not normalized** before the decoder. Ventral noise is off (eval, `ventral_noise = 0.0`). Arm-A changes only the final `Linear(512,300)`.
* **semantic_bank** (`ltm_route.py:103,163–165`): non-persistent buffer = `F.normalize(bank_raw, dim=-1)`, shape (29571,300), row norms 1 ± 2.4e-7. Rebuilt at load and absent from `state_dict`.
* **Retrieval** (`frozen_probe.comprehension_metrics`): `q = normalize(ŝ)`, `bank_n = normalize(bank_raw)`, `sims = q @ bank_n.T`, `top1_idx = argmax(sims)` over the full 29571-row bank. Rows are normalized for retrieval. `lexical_field` uses the same normalized geometry for the gate.
* **Raw GloVe** lives in `JointScratchTrainer.bank_raw` = `stack(LexEntry.semantic)`. These are the float32 vectors parsed verbatim from `glove.6B.300d.txt` with no normalization (live row norms min 2.610, mean 6.531, max 14.011; tensor-SHA256 `4658e11e…`). `make_batches` gathers `semantic = bank_raw[bank_idx]` (raw).
* **Naming passes to the decoder** the RAW target GloVe: `evaluate_naming` → `sem = bank_raw[idx]` → `semantic_greedy_decode` → `model.ltm.decode_from_s_hat(sem, dec_input)` → `model.motor`. Training uses `naming_objective` → `naming_forward(model, batch["semantic"] (raw), dec_in)`.
* **sem_to_h0** (`ltm_route.py:150–156`): `h0 = tanh(Linear(300→512)(s)).unsqueeze(0)`, then decoder `GRU(64→512)` from h0 over `phon_embed(dec_in)` → `dec_to_premotor Linear(512→128)` → `motor.proj Linear(128→42)`. The vector enters **unnormalized** through an affine map and tanh, so the decoder input is not scale-invariant.
* **Naming vs isolated-LTM decoding:** both call the identical `decode_from_s_hat` + `motor`. **No transformation differs.** The only difference is the source of the 300-d vector: raw GloVe for Naming, encoder ŝ for isolated LTM repetition.
* **L_dec** (`losses.py:64`; `train_joint_scratch._interleaved_step`, task `repetition`): `out = model(enc_in, enc_mask, dec_in)` (TF = 1.0). `L_dec = CE(out["ltm_logits"], dec_tgt)`, where `ltm_logits = motor(ltm.decode_from_s_hat(ltm.encode(enc_in, enc_mask), dec_in))`. **The decoder is supervised from encoder-produced ŝ during training** (weight `dec = 0.5`). In the same R step, `L_align = (1 − cos) + 0.1·MSE(ŝ, raw GloVe)` (weight 1.0). Separately, N steps supervise the same decoder from raw GloVe (`LAMBDA_N = 1.0`), and C steps apply `LAMBDA_C · retrieval_CE` on normalized ŝ (`c_align_weight = 0.0`). Both checkpoints: regime j0, schedule interleaved_123, `dec_weight` 0.5.

## 10. Artifact compatibility validation

| requirement | result |
|---|---|
| SOURCE architecture matches live model code | ✔ loads into live `DualRouteModel` at ckpt widths; format `lichtheim3.joint_scratch.v1`; all trainer guards pass (regime, seed, subset hash, schedule, lr_policy, no phase transition) |
| head tensor names/shapes match SOURCE model | ✔ `{2.weight (300,512), 2.bias (300,)}` float32 ↔ `ltm.to_semantic.2.*` |
| same lexicon/semantic representation across witnesses | ✔ same lexicon SHA256, same GloVe SHA256, same bank_raw tensor hash, same C hash, same vocab |
| files load without mutation | ✔ SHA256 before == after, all four files |
| no silent fallback | ✔ `n_glove_fallback = 0` (trainer would raise otherwise); lexicon not synthetic (29571 entries asserted); `ltm_encoder_mode` explicit |
| no parameter-altering conversion | ✔ none |

Evidence (now at `provenance/lineage_pass_structural_check/`): `structural_check.py` (SHA256 `5c57d7b818156bdb8479905e3c9549656b093aa31931faa19e14c4a66e41aad7`) and its output `structural_check.json` (SHA256 `243c71ee549259fc7db2bdbe2976b83a38f5b4b0b793197aee14880e5f40449a`). The check loads and diffs structure only. It performs no decoding and no scoring.

## 11. Remaining ambiguity / risks (none blocks identity)

1. **[RESOLVED: dedicated clean worktree, §0]** **Worktree dirty (untracked GXLR outputs + this workstream's new directory).** Outside the closure. Resolve (commit or explicitly exclude) before the contract freeze commit.
2. **[RESOLVED: driver implemented and frozen with the contract]** **The S0–S3 driver does not exist.** Its components are pinned, but the driver must be frozen with the contract.
3. **[RESOLVED in contract: 256 applies to the Gate C Naming validity check only; S0–S3 all use the repetition conventions]** **The naming cap used by the historical witness batteries was 256, not 10** (§8.4). Exact-match is invariant, auxiliary metrics are not. The contract must choose the S1–S3 cap and the primary free-AR cap (12).
4. **[RESOLVED in the contract-freeze pass: read-only archival copies exist, §0]** **SOURCE checkpoints are unarchived, writable working copies.** Identity is closed by SHA256. A read-only archival copy is recommended; the lustre originals were not hashed from here.
5. **Relative-path dependence.** The ckpt `lexicon_path` is relative (`data/…`) and resolves against the process CWD. Execution must run from the worktree root. The trainer's canonical-lexicon asserts (29571 entries, C hash) fail closed otherwise.
6. **GloVe resolves through a 3-hop symlink chain across worktrees** into `lichtheim3/data/` (git-ignored). The hash matches the record; the fragility is operational, not an identity issue.
7. The GATING manifest's `frozen_gate_mean` for the `*_REP` rows is flagged `DEFECTIVE_source_model_measured` (CODE_AUDIT_GATE §7). It is irrelevant here (no fusion or gate readout), and noted so it is not reused.
