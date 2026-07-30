# Full-Lexicon Cohort Results

- Experiment: `fulllexicon_cohort_93a577f_seeds19_22`
- Training commit: `93a577fd9822955fa272ee733fa7e2acf81f1333`
- Dataset: 29,571 train words, 0 validation words (`split_mode=full_lexicon`)
- Evaluated epochs: 105–200 (every 5 epochs), 20 per seed, 80 total
- Evaluations validated: 80/80
- Decode: deterministic autoregressive

## Cohort criteria

| Criterion | Result |
|-----------|--------|
| Primary: at least one zero-error seed | **PASS** |
| All four seeds at most one error | **PASS** |
| At least two zero-error seeds | **PASS** |
| At least two stable-zero seeds | **PASS** |
| Complete secondary bundle | **PASS** |

Stable zero = at least two consecutive scheduled checkpoints (every 5 epochs) with zero errors.

## Seed-level results

| Seed | Best errors | First zero epoch | Zero checkpoints | Stable zero | Longest streak | Errors @ e200 | Selected epoch |
|---:|---:|---:|---:|:---:|---:|---:|---:|
| 19 | 0 | 140 | 7 | yes | 6 | 2 | 155 |
| 20 | 0 | 130 | 3 | yes | 2 | 2 | 130 |
| 21 | 1 | — | 0 | no | 0 | 1 | 145 |
| 22 | 0 | 140 | 13 | yes | 13 | 0 | 140 |

### Non-monotonic behavior

Seeds 19 and 20 reach zero errors before epoch 200 but end non-zero at epoch 200 (2 errors each).
Seed 19 has an isolated zero at epoch 140 followed by a stable-zero streak at epochs 155–180, then drifts
back to 2 errors. Checkpoint selection uses the first stable-zero streak onset, not the first zero epoch.

## Selected checkpoints

| Seed | Epoch | n_errors | exact_match | Stable-zero streak | SHA256 (full in TSV) |
|---:|---:|---:|---:|---|---|
| 19 | 155 | 0 | 1.000000 | 155,160,165,170,175,180 | `7d05f9c2…` |
| 20 | 130 | 0 | 1.000000 | 130,135 | `b44548b6…` |
| 21 | 145 | 1 | 0.999966 | — | `ab58092e…` |
| 22 | 140 | 0 | 1.000000 | 140,145,…,200 (13) | `a15846cb…` |

Full SHA256 hashes are in `selected_checkpoints.tsv`.

## Checkpoint policy

| Use case | Seeds |
|----------|-------|
| Main results (full cohort) | 19, 20, 21, 22 |
| Exact-zero sensitivity | 19, 20, 22 |
| Canonical illustrative checkpoint | 22 (epoch 140) |

## Archive

```
archives/fulllexicon_93a577f/lichtheim3_fulllexicon_93a577f_final_20260729.tar.gz
SHA256: 5ef9872f17b957ea5bc4589f490e7506f20e4c520666e4d35add52f9ed1086e5
```

Not committed to git: `.pt` checkpoint files, `item_level_predictions.tsv`, `train_errors.tsv`, `.tar.gz`.

## Commits

| Role | SHA |
|------|-----|
| `checkpoint_training_commit` | `93a577fd9822955fa272ee733fa7e2acf81f1333` |
| `documentation_commit` | See Git history for the commit containing this file |
