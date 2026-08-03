# Sprint 4 — diff review

**Base commit**: `b550580455046a8c420e4a62cefeeb435815804b` ·
**Files**: 68 · **Total**: 4.08 MB ·
**Diff sha256**: `ae3876da4b8c7e6904d885419a612589229eed3e0f18a8627a5a6217e171a8a6`
(`git diff | shasum -a 256`, tracked unstaged changes only)

> **Closure.** This review describes the working tree as it stood immediately
> before Commit A; the hash above is that pre-commit diff and is retained as the
> record of what was reviewed. Commit A is
> `62ae51b4018e7ab35f6e6a69eae282db3d917f27`
> (`feat(analysis): add WFE error taxonomy and EOS diagnostics`), containing 64
> files. This file and the other provenance/validation artefacts follow in
> Commit B. One prose-only correction was applied after this review and before
> Commit A: the EOS-slope / edit-distance-slope ratio was removed from the
> handoff summary, replaced there by co-occurrence counts (82 events, ≈ 22 % of
> erroneous and 43.4 % of deletion-bearing LTM pseudoword items, 874 total edit
> operations), and retained only in a technical note labelled as a
> non-decompositional comparison between outcomes with different units. No
> numerical table, figure, plotting TSV, estimator or analysis choice changed;
> deterministic regeneration remained 46/46 byte-identical.

## Composition

| Area | Change | Notes |
|---|---|---|
| `scripts/behavioral_analysis/error_taxonomy.py` | new | Levenshtein estimands, faithful/clean cells, route contrasts, bootstrap |
| `scripts/behavioral_analysis/eos_diagnostics.py` | new | EOS classification, shortfall, rates, length slopes, EOS/deletion 2 × 2 |
| `scripts/behavioral_analysis/plot_error_taxonomy.py` | new | 4 figures, 29 TSVs, seed-22 examples, output manifest |
| `tests/test_behavioral_error_taxonomy.py` | new | 67 tests, groups A–J |
| `docs/behavioral_wfe_fulllexicon.md` | modified | Sprint-4 usage section, EOS observability table, two new limitations |
| `docs/behavioral_wfe_analysis_matrix.md` | modified | A16–A18 → `ALREADY_VALIDATED`, sprint history |
| `reports/.../README.md` | modified | Sprint-4 index section, manifest table, living-file policy |
| `reports/.../analysis_matrix.tsv` | modified | A16, A17, A18, A22 status rows (4 lines) |
| `reports/.../error_taxonomy/` | new, 60 files | convention, spec, results, handoff, figures, tables, examples, provenance, validation |

By type inside `error_taxonomy/`: 4 SVG (2.45 MB), 4 PNG (0.82 MB), 4 PDF
(0.43 MB), 30 TSV, 9 MD, 7 JSON, 1 `.sha256`, 1 `.txt`.

## Checks

- `git diff --check`: **0 flagged lines**.
- **No file under `outputs/` or `archives/` staged**; no production prediction
  and no canonical item table staged.
- **No absolute local path** in any file to be staged.
- **No prior-sprint scientific artefact modified.** The Sprint-1, Sprint-2 and
  Sprint-3 manifests verify apart from the two declared living documents; the
  production manifest verifies in full (36/36) with no exemption.
- **Canonical table unchanged** — hash matches
  `behavioral_analysis_provenance.json`.
- **Deterministic regeneration**: 46 regenerated files compared byte for byte
  against the published tree, **0 differing**.
- **Tests**: 67 new, 174 across all behavioral-analysis suites, **412 passed /
  4 deselected** for `pytest tests/ -m "not slow"`.
- **Validation**: 26/26 checks PASS (`error_taxonomy_validation.json`).

## Two corrections made during this sprint

1. **Provenance diff hash.** The first generator hashed `git diff` captured with
   `text=True`, whose universal-newline translation dropped CR bytes, so the
   recorded hash did not match `git diff | shasum -a 256`. The generator now
   hashes the raw bytes git emits and the two agree.
2. **`analysis_matrix.tsv` line endings.** The committed file uses CRLF. The
   first edit rewrote it with LF, which made all 23 data rows appear changed.
   The file was restored and re-edited preserving CRLF, so the diff is now the
   4 intended rows only.

Neither correction touched a scientific value.

## Note on SVG size

The clean-taxonomy SVGs are large (0.87 MB and 0.85 MB) because hatch fills
expand into many path elements. This is presentation only — hatch is what keeps
operation type off the red/blue channel, which is reserved for lexicality. PNG
and PDF are unaffected.

## Not in this sprint

No feature importance, SSP, ablation or causal length-effect diagnostic; no
checkpoint inference; no retraining; no architecture change; no
evaluation-code, dataset or checkpoint change; no custom edit alignment; no
fourth Levenshtein operation; no conflation of deletion and premature EOS; no
post-hoc item exclusion; no seed exclusion; no new morphology or frequency
inference; no causal explanation; no architectural recommendation; no overwrite
of a prior figure.
