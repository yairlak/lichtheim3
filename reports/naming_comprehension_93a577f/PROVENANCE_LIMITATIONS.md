# Provenance limitations — naming/comprehension and warm-start runs

Factual record of what can and cannot be reconstructed exactly for the runs
curated in this report. It does not weaken or strengthen any scientific finding.

## Summary

Seven of the twelve retained runs were executed from a **dirty working tree**:
their own `run_summary.json` records `provenance.eval_git.dirty = true`. The
runs recorded this themselves; nothing was discovered after the fact and nothing
was concealed.

For all seven, `tracked_dirty = true` and `untracked_present = false` with
`untracked_paths = []`. The uncommitted delta was therefore confined to files
already tracked at the recorded commit, and no untracked file participated.

**No run stored a diff, a patch hash, a modified-file list, source-file hashes,
a script hash, or an argv/command line.** The exact byte-level working-tree
state at execution is not recoverable from the artifacts.

## The seven runs

| Run | Recorded commit | Family |
|---|---|---|
| `phase2b_c3_flat_seed22` | `d5f087b` | repetition cost after comprehension (Fig 4) |
| `phase2b_n0_naming_seed22` | `d5f087b` | repetition cost after naming (Fig 4) |
| `phase2f_n0_representative3288_seed22` | `bb45299` | naming acquisition N=3,288 (Fig 1) |
| `phase2g_n0_representative10000_seed22` | `14571e5` | naming acquisition N=10,000 (Figs 1, 2) |
| `phase2i_n0_repblockB_10000_seed22` | `80906b0` | disjoint Block B (Fig 2) |
| `phase2i_n0_repblockC_9571_seed22` | `80906b0` | disjoint Block C (Fig 2) |
| `phase2j_n0_sequential_A_then_B_seed22` | `80906b0` | sequential forgetting (Fig 5) |

The remaining five retained runs — `phase2d3`, `phase2h`, `phase3a`, `phase3b`,
`phase3c` — recorded `dirty = false`. They include Phase 3B and Phase 3C, which
carry the warm-start preservation–acquisition result.

## What is still known exactly

Every one of the seven records, independently and per run:

- the source checkpoint `seed_22_epoch_0140.pt` and its SHA-256
  `a15846cbf3c7df88ed289512bbb20cbefd2121d0deec1b39f363932a743da595`;
- the checkpoint training commit `93a577fd9822955fa272ee733fa7e2acf81f1333`;
- the lexicon SHA-256 `ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66`
  and the ordered-bank SHA-256;
- the recorded code commit, branch, and the dirty/tracked-dirty/untracked flags;
- the data seed, device, LTM encoder mode, naming decode cap and runtime;
- `torch_version` (five of the seven; absent for the two `phase2b` runs);
- a complete snapshot of the scientific configuration actually used —
  `objective`, `objective_config`, `budget`, `populations`, `subset` (including
  `subset_definition_sha256`), `scope_audit`, `trainable_parameters` and
  `always_frozen` — plus the full `snapshots` / `trajectory` series.

The reported numbers are therefore attributable to a fully specified experimental
configuration, a hash-identified starting checkpoint and a hash-identified
dataset. What is missing is the exact source text of the driver script.

## What cannot be reconstructed exactly

The uncommitted delta itself. It was not stored, and it cannot be recovered from
the artifacts.

## Forensic bounding from Git history

Read-only inspection of the commit sequence bounds the delta. In every case the
commit immediately following the recorded one touches only
`scripts/naming_comprehension/train_tasks.py` and its test file.

| Recorded commit | Next commit | Files changed by the next commit |
|---|---|---|
| `d5f087b` | `ef5ceee` | `train_tasks.py` (+323) |
| `bb45299` | `14571e5` | `train_tasks.py` (+87/−9), `test_naming_comprehension_subset.py` |
| `14571e5` | `80906b0` | `train_tasks.py` (+50/−12), `test_naming_comprehension_subset.py` |
| `80906b0` | `564e345` | `train_tasks.py` (+156/−6), `test_naming_comprehension_subset.py` |

For three of the seven, the run demonstrably used functionality that does not
exist in the committed code at the recorded commit:

- **`phase2b_c3_flat` and `phase2b_n0_naming`** (`d5f087b`). At `d5f087b`,
  `train_tasks.py` defines only the `preflight` subcommand and contains **no**
  reference to `run_summary.json` or `trajectory.tsv`. Both files exist for
  these runs. The `run` subcommand and both writers first appear at `ef5ceee`.
- **`phase2f`** (`bb45299`). `select_representative_subset` does not exist at
  `bb45299`; it first appears at `14571e5`. The run used representative-subset
  selection.
- **`phase2j`** (`80906b0`). `staged_init`, `retention_population` and
  `sequential` do not appear in `train_tasks.py` at `80906b0`; they first appear
  at `564e345`. The run emits `staged_init` and `retention_population`.

For these three cases the working tree at execution already contained the
changes that landed in the next commit. The most probable delta is that
next-commit diff to `train_tasks.py`. **This is an inference from feature
presence, not a recovered patch, and no exact patch is claimed.**

For `phase2g`, `phase2i_repblockB` and `phase2i_repblockC` no such positive
marker was found: every field their summaries emit already exists in the
committed code at their recorded commit. Their delta is bounded to tracked files
and, by the table above, most plausibly to `train_tasks.py`, but its content is
unknown.

## Classification

No run is exactly reconstructible, because no diff was stored. None is
unclassifiable, because all are bounded to tracked files with a complete
scientific configuration snapshot.

| Run | Class | Basis |
|---|---|---|
| `phase2b_c3_flat_seed22` | **B** | delta bounded; strongly identified as `d5f087b..ef5ceee` on `train_tasks.py` |
| `phase2b_n0_naming_seed22` | **B** | as above |
| `phase2f_n0_representative3288_seed22` | **B** | delta bounded; strongly identified as `bb45299..14571e5` |
| `phase2g_n0_representative10000_seed22` | **B** | delta bounded to tracked files; content unknown |
| `phase2i_n0_repblockB_10000_seed22` | **B** | delta bounded to tracked files; content unknown |
| `phase2i_n0_repblockC_9571_seed22` | **B** | delta bounded to tracked files; content unknown |
| `phase2j_n0_sequential_A_then_B_seed22` | **B** | delta bounded; strongly identified as `80906b0..564e345` |

**B = substantially reconstructible, small unknown dirty delta.**

`dirty = true` does not by itself establish that the executed code differed in
any way that changed a result. Equally, because the diff was not stored, exact
byte-level reproducibility cannot be claimed for these seven runs.

## Integrity of the published artifacts

The result artifacts themselves are integrity-protected independently of the
dirty-tree question. Every file curated in this report is listed with its
SHA-256 in `CURATED_FILES.sha256`, and each was verified byte-identical to its
source output before being committed. The figures were not regenerated. The
15 checkpoints produced by the retained runs are hash-referenced in
`CURATION_MANIFEST.tsv`.

So the artifacts are fixed and verifiable; what is uncertain is the exact source
text that produced them.

## Effect on interpretation

The figures in this report are descriptive summaries of runs whose objective,
population, budget, subset hash, starting checkpoint and dataset are all
recorded exactly. The dirty-tree finding does not identify any specific
discrepancy, and none of the affected runs reports a result that depends on a
contested implementation detail.

At the same time, a claim of exact bit-level reproduction should not be made for
these seven runs. Where a result matters most, the relevant comparison is the
one that is already clean: **Phase 3B and Phase 3C — the runs carrying the
preservation–acquisition result — both recorded `dirty = false`**, as did
`phase2d3` (the comprehension learning curve) and `phase2h` (the full-lexicon
naming endpoint).

The remaining scientific limitations of these experiments, including the
Phase 3B vs Phase 3C rehearsal-breadth and per-item-exposure confound, are
documented separately in `README.md`. They are independent of this file.

## What future reproduction should use

- Start from `seed_22_epoch_0140.pt`, verified against its SHA-256.
- Use `configs/canonical_93a577f.yaml` for the historical training recipe of the
  source checkpoint, and note that `config.py` defaults do not reproduce it
  (see `configs/canonical_93a577f_vs_defaults.tsv`).
- Drive adaptation runs from a **clean** tree, and record the commit, the argv,
  and — if the tree is not clean — the diff itself or its hash, so that this
  limitation does not recur.
- Treat the recorded `objective_config`, `budget`, `populations` and
  `subset_definition_sha256` in each `run_summary.json` as the authoritative
  specification of what was run.
