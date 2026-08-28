# Data

Inventory of the datasets Lichtheim3 depends on, what is tracked, and what must
be obtained separately. Machine-readable form:
[`../manifests/datasets.tsv`](../manifests/datasets.tsv).

Nothing in this directory was modified to write this file.

---

## `lexicon_en_glove_covered.tsv` — the canonical training lexicon

**29,571 words** (29,572 lines including the header). CMU Pronouncing Dictionary
entries filtered to full GloVe-300 coverage, built by
`scripts/create_glove_covered_lexicon.py`.

```
sha256  ae80918165e16b8cbdb58e16d0c9d1fff291773abffd7c0d786e6746024a6a66
```

This hash is asserted at evaluation time and is independently recorded by every
archived checkpoint and by every retained naming/comprehension run summary.
GloVe coverage is complete for this file: `n_glove_found = 29571`,
`n_glove_fallback = 0`.

**This is the lexicon every current result is based on.** Tracked; no action
needed.

## `lexicon_en.tsv` — legacy 30k lexicon (NON-CANONICAL)

The bundled 30,000-word frequency-ranked list (30,001 lines with header). It
predates the GloVe-covered lexicon and is **not** the training lexicon for any
current result. No provenance hash is recorded for it.

Retained for historical continuity only. Do not use it to reproduce anything in
this repository, and do not confuse the two: the older top-level documentation
described a generic "30k lexicon", which is this file, not the canonical one.

## GloVe 6B 300d — required, external, not tracked

`data/glove.6B.300d.txt` (~1.04 GB) is gitignored (`data/glove.*.txt`) and must
be downloaded:

```bash
bash data/get_glove.sh
```

Used as the LTM semantic alignment target during training, as the comprehension
retrieval bank (all 29,571 vectors), and as the naming input.

> **No checksum verification.** `get_glove.sh` does not verify the download, and
> **no trustworthy published checksum for the official archive is recorded
> anywhere in this repository**. None has been invented here. Adding one is an
> open reproducibility item. The source is the Stanford NLP GloVe 6B release;
> its terms are not stated in this repository.

## `raw-nwr_swp/` — NWR/SWP stimuli · **REDISTRIBUTION STATUS: UNRESOLVED**

```
phonemes.csv   sha256 2c8b7dffba1da1fb0c219b486a18f4204a88c2f458875da9bf2ea8bf6b3860bc
ssp.csv        sha256 c1b1970d1f5ef42ef5f7c718d0f089c5e7cc99fb25d9975cf35baf525f9e9925
wfe.csv        sha256 295d8e795927235f65141eb3a517611e418a706559d52db97ed47cdb9b0e0b43
```

Source: the SWP / NWR single-word-processing corpus of Daniel Dager and Robin
Sobczyk (`danieldager/swp-model`). These files supply the stimuli behind every
external behavioral evaluation (the F1–F7 and S1–S12 release).

**REDISTRIBUTION STATUS: UNRESOLVED.**

- No licence, permission, or redistribution statement is recorded anywhere in
  this repository.
- The separate Lichtheim2 replication repository documents byte-identical copies
  of these same three files as *private data that are not committed*, while this
  repository tracks them.
- **No permission is claimed here, and none is inferred.** Resolving this
  requires a decision from the data authors. Until then, do not redistribute
  these files further.

## `eval_external/` — derived evaluation TSVs

```
ssp_eval.tsv   746,333 bytes
wfe_eval.tsv   103,644 bytes
```

Produced from `raw-nwr_swp/` by `scripts/convert_csvs.py`; they define the
evaluation regimes used by the behavioral analyses. As derived artefacts they
**inherit the unresolved redistribution status above**.

---

## Summary

| File / directory | Tracked | External download | Redistribution | Needed for |
|---|---|---|---|---|
| `lexicon_en_glove_covered.tsv` | yes | no | no repository LICENSE | everything |
| `lexicon_en.tsv` | yes | no | no repository LICENSE | nothing current (legacy) |
| `glove.6B.300d.txt` | **no** | **yes (~1.04 GB)** | external project terms | training, RSA, naming/comprehension |
| `raw-nwr_swp/*.csv` | yes | no | **UNRESOLVED** | behavioral evaluation |
| `eval_external/*.tsv` | yes | no | **UNRESOLVED** (inherited) | behavioral evaluation |
