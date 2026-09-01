# Phase 4 Joint Multitask From Scratch — Seed22 Factorial Synthesis

Analysis-only synthesis of the completed 2×2 objective factorial. Every number
here is read from the run artifacts by
`scripts/naming_comprehension/analyze_joint_factorial.py`; nothing was
retrained, resumed or copied in by hand.

Regenerate with, from the repository root:

```bash
python scripts/naming_comprehension/analyze_joint_factorial.py \
    --seed 22 --output reports/joint_scratch_factorial_seed22 --wm-audit
```

(Drop `--wm-audit` to skip the few-minute full-lexicon re-evaluation.)

---

## Experimental design

Four conditions, all trained from **random initialization** at seed 22 to
**e440** (203,720 optimizer steps), sharing the same architecture, the same
29,571-word repetition lexicon, the same repetition and dorsal-pool batch
sequences, the same optimizer and the same LR schedule (1e-3 → 1e-4 at step
46,300). The **only** difference between cells is which additional
developmental objective is present:

| condition | retrieval | naming | loss added to the historical base |
|-----------|-----------|--------|-----------------------------------|
| H0        | OFF       | OFF    | — |
| C-only    | **ON**    | OFF    | `+ 0.087 · retrieval_CE` |
| N-only    | OFF       | **ON** | `+ 1.0 · naming_CE` |
| J0        | **ON**    | **ON** | both |

**Base objective** (identical everywhere):
`total_loss(R batch) + 0.5 · dorsal_pool_CE`, where `total_loss` is
`1.0·L_rep + 1.0·L_align + 0.5·L_dec + 0.5·L_wm + 0.05·L_gate`.

**Retrieval** — phonology → LTM encoder → `s_hat`, scored by cosine against the
frozen bank of all 29,571 true GloVe vectors at τ = 0.10; cross-entropy over
that 29,571-way choice. Trained on the frozen `subset3288` population.

**Naming** — true target GloVe → `sem_to_h0` → LTM decoder → `dec_to_premotor`
→ `motor.proj` → phoneme sequence, teacher forcing 1.0, same `subset3288`
population, evaluated by free greedy AR from BOS with no target-length leakage.

**Routes** — `WM` is the dorsal phonological buffer alone, `LTM` the ventral
lexical-semantic route alone, and `FULL` the gated combination of the two.
Reporting `LTM` separately from `FULL` is deliberate: a cost to the isolated
ventral route can be invisible in `FULL`, which can recruit the dorsal buffer.

---

## Endpoint results (e440)

Full 29,571-word training lexicon, canonical forced-length AR, route-isolated.

| Route | H0 | C-only | N-only | J0 |
|---|---|---|---|---|
| FULL | 0.999932 (2 err) | 0.999966 (1 err) | 0.999966 (1 err) | 0.999932 (2 err) |
| WM | 0.999932 (2 err) | 0.999932 (2 err) | 0.999493 (15 err) | 0.999966 (1 err) |
| LTM | 0.982348 (**522** err) | 0.958845 (**1,217** err) | 0.968483 (**932** err) | 0.944709 (**1,635** err) |

Semantic tasks on `subset3288`:

| Metric | H0 | C-only | N-only | J0 |
|---|---|---|---|---|
| comprehension top-1 | 0.1463 | **0.8735** | 0.1244 | 0.8266 |
| comprehension top-5 | 0.2582 | **0.9611** | 0.2165 | 0.9343 |
| comprehension median rank | 96.5 | **1** | 142 | **1** |
| comprehension margin | −0.1492 | **+0.0739** | −0.1616 | +0.0612 |
| naming exact | 0.0000 | 0.0003 | **1.0000** | **1.0000** |

---

## Ventral LTM cost decomposition

On the error-count scale, which is the least noisy endpoint summary:

| Quantity | Errors / 29,571 |
|---|---|
| Retrieval cost, naming OFF (C−H) | **+695** |
| Retrieval cost, naming ON (J−N) | **+703** |
| Naming cost, retrieval OFF (N−H) | **+410** |
| Naming cost, retrieval ON (J−C) | **+418** |
| Additive prediction | +1,105 |
| Observed J0 cost (J−H) | +1,113 |
| **Interaction (J − C − N + H)** | **+8** |

At seed22/e440, the ventral LTM costs are approximately additive: the
descriptive interaction is only +8 errors on a 29,571-word lexicon. Each
objective's cost is nearly unchanged by the presence of the other — retrieval
costs 695 alone and 703 alongside naming; naming costs 410 alone and 418
alongside retrieval.

---

## Acquisition asymmetry

**Comprehension** (first / sustained epoch at threshold):

| threshold | C-only | J0 |
|---|---|---|
| 20% | 25 / 25 | 35 / 35 |
| 50% | **50 / 50** | **90 / 90** |
| 80% | **145 / 145** | 310 / **340** |
| 90%, 95% | not reached | not reached |

**Naming** (first / sustained epoch):

| threshold | N-only | J0 |
|---|---|---|
| 20%, 50% | 5 / 5 | 5 / 5 |
| 80%, 95% | 10 / 10 | 10 / 10 |
| 99% | 25 / 60 | 20 / 20 |
| 100% | **105 / 105** | **105 / 105** |

Adding naming roughly halves the speed of retrieval acquisition (2.1× slower to
80%). Adding retrieval produces no material slowing of naming under the tested
schedule — every crossing coincides, and both cells hold exactly 100% for the
final 68 consecutive evaluations.

---

## N-only WM endpoint audit

The one anomaly in the four-cell table is N-only's **15** full-lexicon WM
errors, against 1–2 in the other three cells. Re-running the canonical
evaluator over all four final checkpoints reproduced every stored count exactly
(12/12 route × cell combinations), so this is not an evaluator or provenance
artefact.

The 15 items are not random:

- **All 15 are rescued by FULL** (100%); the single N-only FULL error
  (*separateness*) is an LTM error, not one of these.
- **Mean target length 8.27 phonemes** against a lexicon mean of 5.87; 12 of 15
  have length ≥ 8, where the lexicon base rate is 21%.
- Errors are **serial-order errors in consonant clusters**, edit distance 1–2:
  six disrupt an /S T/ cluster (*consists, consisting, dualistic, enlist,
  holistic, inelastic*), four are the *sleepwalk* family showing the same
  S L IY P W → S W IY P L transposition (*sleepwalk, sleepwalker, sleepwalking,
  sleepwear*), and the rest involve other clusters (*shoplifting, skinflint,
  scoundrel, definitive, felicitate*).
- **14 of the 15 lie outside `subset3288`**, which is why the 5-epoch
  developmental WM series sits at ceiling in all four cells while the
  full-lexicon endpoint separates them — the routine evaluator simply never
  probes these items.

**Classification: C — structured item effect.** It does not affect any
factorial contrast reported above (all of which concern LTM, comprehension and
naming), and it is not elevated to a main result on one seed.

---

## OBSERVATION

1. FULL repetition is essentially at ceiling in all four cells: 1–2 errors out
   of 29,571, and 1.0000 on the out-of-subset probe everywhere.
2. The two retrieval-trained cells reach substantial explicit comprehension
   (C-only 0.8735, J0 0.8266 top-1, both with median rank 1 and a positive
   margin); the two untrained cells stay at 0.12–0.15.
3. The two naming-trained cells reach and hold perfect naming (1.0000, edit
   distance 0.0000, EOS 1.0000, predicted length equal to target length).
4. There is essentially no spontaneous cross-task transfer: C-only naming is
   0.0003 (one word of 3,288), and N-only comprehension (0.1244) is slightly
   *below* H0 (0.1463) on every retrieval metric.
5. Naming slows retrieval: C-only exceeds J0 throughout development, by up to
   +0.204 top-1 around e60 and +0.047 at e440.
6. No material reciprocal slowdown of naming by retrieval is detectable:
   N-only and J0 crossings coincide at every threshold.
7. Isolated LTM repetition costs order H0 < N-only < C-only < J0 at all three
   population levels (subset3288, out-of-subset probe, full lexicon).
8. At seed22/e440, full-lexicon LTM costs are approximately additive
   (interaction +8 errors).

## INTERPRETATION

The dual-route system can co-develop repetition, lexical retrieval and naming
from random initialization while preserving near-ceiling FULL repetition.
However, the isolated ventral route pays persistent costs for both added
semantic objectives. At seed22/e440 these costs are approximately additive.
Acquisition is asymmetric: naming materially slows retrieval, whereas no
comparable reciprocal cost on naming acquisition is detectable.

The two objectives also show little spontaneous behavioral transfer in either
direction — training one does not deliver the other, and training naming alone
leaves explicit retrieval marginally worse than the untrained control.

## HYPOTHESIS

These are hypotheses motivated by the pattern, not results:

- Shared ventral parameters may create competing functional demands, with each
  objective consuming a portion of the route's capacity.
- Naming may impose an early, persistent reorganization: it is essentially
  solved by e10 and its CE falls to ~1e-4 thereafter, yet its ventral cost
  persists to e440.
- Retrieval may remain a longer-running optimization pressure, still descending
  at the horizon.
- The acquisition asymmetry may relate to these differing learning timescales
  rather than to any structural property of the two mappings.

## LIMITATION

- **One seed only** (seed22), one run per cell, **no variance estimate**; the
  +8-error interaction cannot be distinguished from noise without replication.
- All contrasts are **descriptive factorial contrasts**, not statistically
  established main effects or interactions.
- The earlier single-task C3 experiment was a warm start from a mature
  checkpoint and is **not an exact matched control** for C-only.
- Comprehension in every cell remains below the ≥95% single-task C3 reference,
  and was still rising at e440 in both retrieval-trained cells.
- **Behavioural evidence only** — no representational analysis (RSA, probing),
  so no claim is made about internal geometry or about the objectives using
  separate representations.
- Naming endpoints are **ceiling-limited**, so the endpoint naming interaction
  is uninformative and only developmental timing carries evidence there.
- The N-only WM endpoint anomaly is audited above and remains a single-seed
  observation.

---

## Files

```
reports/joint_scratch_factorial_seed22/
    README.md                                  this synthesis
    provenance.json                            per-cell checkpoint identity + alignment check
    data/
        factorial_trajectories_seed22.tsv      canonical table, 4 × 89 snapshots
        fig1_developmental.tsv                 Figure 1 source
        fig2_endpoint.tsv                      Figure 2 source (accuracy + error counts)
        fig3_decomposition.tsv                 Figure 3 source
        factorial_contrasts_e440.tsv           all descriptive contrasts
        acquisition_crossings.json             threshold crossings per cell
        wm_audit_per_item.tsv                  per-item WM/LTM/FULL errors, four cells
        wm_audit_summary.json                  WM audit counts and N-only detail
    figures/
        fig1_developmental_trajectories.{png,pdf}
        fig2_factorial_endpoint.{png,pdf}
        fig3_ltm_cost_decomposition.{png,pdf}
```

Source runs (read-only, unmodified):

| condition | run directories |
|---|---|
| H0 | `phase4a2a_h0_seed22_e160` + `phase4a2c_h0_seed22_e440` |
| C-only | `phase4a3b_c_only_seed22_e440` |
| N-only | `phase4a3c_n_only_seed22_e440` |
| J0 | `phase4a2b_j0_seed22_e160` + `phase4a2c_j0_seed22_e440` |

H0 and J0 trajectories are stitched from two segments. Within every run file the
final step carries two rows — the cadence evaluation and the endpoint evaluation
that adds the full-lexicon columns — so rows are collapsed by step keeping the
later, strictly more complete one. No value is interpolated or forward-filled;
missing metrics stay missing.
