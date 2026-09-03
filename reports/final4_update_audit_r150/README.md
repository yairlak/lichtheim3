# FINAL-4 actual-AdamW-update audit at the FINAL-3 R150 checkpoint

Offline diagnostic that decided the FINAL-4 intervention. Analysis only: the
source run and checkpoint are read-only, no optimizer step is ever kept, and
no training happens.

Regenerate (`audit_r150.json` is this command's output):

```bash
python scripts/naming_comprehension/audit_update_geometry.py \
    --checkpoint outputs/joint_scratch/final3p_i123_seed22_final_full/checkpoints/step_00416700.pt \
    --batches 16 --self-check --json-out reports/final4_update_audit_r150/audit_r150.json
```

Source state: step 416,700; cursors R 69,450 / pool 69,450 / N 138,900 /
C 208,350 (= R150 / N300 / C475.6849); `interleaved_123` ratio [1,2,3];
`c_align_weight = 0.0`; commit `9969e95`.

## Method

Every measurement starts from an exact fresh clone of the checkpoint's model
**and** optimizer state, performs ONE hypothetical task optimizer step
(`zero_grad → forward → backward → clip → set LR → step`), records the actual
parameter delta, and discards the clone, so no batch or configuration
contaminates the next. Batches are the 16 deterministic ones each stream would
see next at the checkpoint's own cursors. `--self-check` asserts exact restore
(a repeated measurement is bit-identical), that `grad=None` parameters are
untouched, and that the one-step delta is linear in the learning rate.

## Headline result

| condition | pre-clip ‖g‖ | clipped | ‖Δθ‖ (median) |
|---|---|---|---|
| R @ 1e-4 | 1.481 | 100% | 0.01312 |
| N @ 1e-4 | 1.366 | 100% | 0.00884 |
| C @ λ=0.087, 1e-4 | 0.166 | 0% | **0.00787** |
| C @ λ=1.0, 1e-4 | 1.907 | 100% | 0.01114 (only ×1.41) |
| N @ 3e-4 | — | — | 0.02652 (×3.0000) |
| C @ 3e-4 | — | — | 0.02362 (×3.0000) |
| C @ 1e-3 | — | — | 0.07874 (×10.000) |

C is **not** update-starved: its actual step is 89% of N's despite a ~8×
smaller gradient, because Adam's preconditioner cancels the static λ scaling.
Per macro-cycle C already moves the most (3 × 0.00787 = 0.0236 vs N 0.0177,
R 0.0131). λ_C is therefore not the bottleneck at the update level, and LR is
the exact, predictable knob — which is why FINAL-4 changes only the LR policy
and keeps λ_C = 0.087.

Weight decay is negligible next to the adaptive term (~1e-7 vs ~1e-2).

## Update geometry

Raw task gradients are near-orthogonal at this frontier (R~C −0.01, R~N
+0.00/−0.03), but the *applied* updates are strongly co-aligned (R~C +0.83 /
+0.73, R~N +0.85 / +0.84): shared momentum dominates each step, so the tasks
ride one consensus direction and update-space interference is not the problem.
The system moves coherently but at only ~4–6e-5 relative step size.

## C diagnostic (reference only)

cos(retrieval gradient, alignment gradient) on the same C batches: +0.81
encoder side, +0.84 on `to_semantic`. **The current C update contains zero
alignment contribution** (`c_align_weight = 0`); this is recorded only to show
that the historical retrieval-vs-alignment conflict has dissolved at this
state, not as a candidate recipe.
