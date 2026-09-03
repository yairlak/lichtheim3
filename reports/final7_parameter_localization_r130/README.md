# FINAL-7A — parameter-localisation audit at the matched R130 state

Read-only diagnostic. **No training, no optimizer step, no backward pass**, and
nothing written to any scientific run directory. Source checkpoints are opened
read-only and SHA-256-fingerprinted before and after; the audit aborts if any
of them changes.

> **Causal caveat, which applies to every number produced here.** Weight
> transplantation is a diagnostic intervention on **endpoint parameter
> states**. It localises *endpoint dependence* and is *consistent with* (or
> against) a mechanism. It does **not** show that training a group differently
> would reproduce the trajectory. Language in any write-up should stay at
> "localises", "suggests", "consistent with".

## The dissociation being localised

Three runs branch from one R100 checkpoint and differ **only** in
optimizer-state topology (architecture, data streams, future batches, LR 1e-4
after R100, interleaved 1:2:3, losses, populations and seed 22 all identical):

| R130 endpoint | naming | C top1 | C top5 | full LTM rep | full rep |
|---|---|---|---|---|---|
| control — one shared AdamW (R+N+C) | .054276 | .076695 | .178121 | .548612 | .999865 |
| FINAL-6P — three banks (R \| N \| C) | .041798 | .107430 | .235338 | .340604 | .999966 |
| FINAL-7P — two banks (RN \| C) | .052010 | .107037 | .235803 | .242603 | 1.000000 |

So R↔N optimizer coupling rescues naming but **not** LTM, while C separation
tracks both the comprehension gain **and** severe LTM degradation — and FINAL-7P
is *worse* on LTM than full separation. The audit asks whether the C gain and
the LTM loss live on **different parameter surfaces**.

## Method

**A. Displacement.** `Δθ = θ_R130 − θ_R100` per parameter group, with `‖Δθ‖`,
`‖Δθ‖/‖θ_R100‖`, and pairwise cosines between the three runs' displacement
directions — showing *where* the 6P/7P trajectories depart from the control.

**B/C/D. Transplants.** One group (or composite) at a time is taken from a
donor endpoint into a base endpoint and the result re-evaluated: control→7P
(what does 7P's phenotype depend on?), the reverse 7P→control (is a group
*sufficient*?), and 6P→7P (which surface carries the RN-sharing effect?).

Every condition installs a **fresh, complete** parameter assignment by copying
values into the model, so no condition can alias or leak into another
(pinned by tests). Evaluation uses the training driver's own definitions:
canonical forced-length AR repetition, free greedy AR naming from true GloVe,
and full-bank cosine retrieval over the canonical 27,981-target C population.

## Parameter groups

Groups are defined over the model's **canonical `named_parameters()` names**.
This matters: `state_dict()` also exposes the shared phoneme embedding as
`wm.phon_embed.weight` and `ltm.phon_embed.weight`, so working from
`state_dict()` would let a "wm" transplant silently drag the shared embedding
with it. The groups are exhaustive and disjoint (tested).

| group | parameters | reached by |
|---|---|---|
| `phon_embed` | `phon_embed.weight` | R, N, C |
| `ltm_encoder` | `ltm.encoder.*` | R, C |
| `to_semantic` | `ltm.to_semantic.*` | R, C |
| `sem_to_h0` | `ltm.sem_to_h0.*` | R, N |
| `ltm_decoder` | `ltm.decoder.*` | R, N |
| `dec_to_premotor` | `ltm.dec_to_premotor.*` | R, N |
| `motor` | `motor.proj.*` | R, N, pool |
| `wm` | `wm.encoder/decoder/to_premotor.*` | R, pool |

Composites: **A** encoder/semantic side (everything a C step touches:
`phon_embed + ltm_encoder + to_semantic`); **B** production side (what naming
and ventral repetition use and C never touches: `sem_to_h0 + ltm_decoder +
dec_to_premotor + motor`); **C** lower encoder alone; **D** `to_semantic`
alone; **E** production core without the shared readout; **F** whole ventral
route; **G** dorsal WM.

## Reproducing

```bash
python scripts/naming_comprehension/parameter_localization_audit.py \
    --r100    $SCRATCH/lichtheim3_runs/final3p_i123_seed22_final_full/checkpoints/step_00277800.pt \
    --control $SCRATCH/lichtheim3_runs/final3p_i123_seed22_final_full/checkpoints/step_00361140.pt \
    --sep     $SCRATCH/lichtheim3_runs/final6p_r100_sepmoments_seed22_final_full/checkpoints/step_00361140.pt \
    --grouped $SCRATCH/lichtheim3_runs/final7p_r100_rn_c_banks_seed22_final_full/checkpoints/step_00361140.pt \
    --device cuda --eval-population full --routes full,ltm \
    --out-dir reports/final7_parameter_localization_r130
```

`--eval-population sample --sample-size 4096` gives a fast screen;
`--conditions intact,single,composite,reverse,sixp_vs_sevenp` selects subsets.

## Outputs

- `audit.json` — inputs with per-checkpoint provenance (step, cursors,
  optimizer policy and bank layout, git commit, SHA-256), group and composite
  definitions, the displacement table, every condition's metrics, the
  read-only confirmation and the causal caveat.
- `displacement.csv` — audit A.
- `conditions.csv` — audits B/C/D: one row per weight state, with the exact
  parameter names transplanted.

## Reading the result (the four cases to distinguish)

1. **One encoder group is necessary for both the C gain and the LTM loss** →
   genuine C-vs-LTM weight competition; the next intervention should address
   rehearsal/frequency/gradient handling, not another bank topology.
2. **Different groups drive the C gain and the LTM loss** → a
   parameter-group-selective optimizer policy becomes strongly justified.
3. **Decoder/production weights explain 7P's extra LTM loss while encoder
   weights explain the C gain** → keep C separation on the encoder/semantic
   surface and treat the production surface differently.
4. **Transplants are non-local / non-additive and nothing localises cleanly**
   → report that plainly and do not force a mechanistic story.
