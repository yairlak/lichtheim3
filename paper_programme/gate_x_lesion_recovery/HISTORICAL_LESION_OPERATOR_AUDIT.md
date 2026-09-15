# HISTORICAL LESION OPERATOR AUDIT

**Workstream:** `POST_STAGE / PAPER_PROGRAMME — GATE × LESION / RECOVERY` (GXLR)
**Status:** RESOLVED FROM EXECUTABLE HISTORICAL CODE. No field remains `PENDING_LIVE_CODE_REVERIFICATION`.
**Purpose:** recover, from committed source rather than notation or memory, the exact
operator semantics that produced the pre-meeting SD-relative lesion atlas, so that the
GXLR experiment inherits them instead of re-inventing them.

All historical evidence is read from Git objects at commit

```
3af0ef9c00df5aba8200d56f566bb957d084109c
  2026-09-10  feat(lesion): pre-meeting lesion atlas + D-05 scope sensitivity (EXPLORATORY)
  branch: feat/brain-damage
```

This commit is the *only* branch head ancestry carrying `lesion/`; the GATING lineage
(`paper-programme/gating-route-diagnostics`, frozen at `993f2e73…`) does **not** contain
the `lesion/` package at all (verified: `git diff --stat 3af0ef9 993f2e73 -- lesion/
scripts/lesion/` reports 3 198 deletions, 0 insertions). The two lineages are disjoint;
their merge base is `96468c69`. GXLR is therefore a *port*, and every ported semantic is
recorded below with its source blob and line range.

---

## 1. The historical operator, end to end

`scripts/lesion/run_premeeting_atlas.py` @ `3af0ef9` is the executable definition of the
pre-meeting atlas. The chain is:

| step | file @ `3af0ef9` | lines | what it fixes |
|---|---|---|---|
| 1. SD is read from a frozen TSV | `scripts/lesion/run_premeeting_atlas.py` | 46–53 | `scales[target] = float(row["std"])` |
| 2. SD is written by the calibrator | `scripts/lesion/measure_activation_scales.py` | 84–100 | `"std": float(a.std())` |
| 3. SD enters the spec as a scalar | `lesion/spec.py` | 86, 177–187 | `activation_scales: Tuple[Tuple[str, float], ...]` |
| 4. amplitude is composed | `lesion/apply.py` | 169–176 | `amp = λ · scale · activation_scale_for(target)` |
| 5. noise is drawn and added | `lesion/masks.py` | 171–192 | `(rand·2 − 1) · amp` |
| 6. the site is selected | `lesion/apply.py` | 99–128 | GRU tuple slot, `_GRU_SLOT` |

The atlas records its own formula in provenance
(`run_premeeting_atlas.py:111`):

```
"noise": "A_j(lambda) = lambda * SD_j; eta ~ U(-A_j, +A_j)"
```

which is algebraically identical to CENTRAL's frozen nested form
`eta(λ) = λ · SD · epsilon`, `epsilon ~ U(−1, +1)`. **The micro-amendment does not change
the historical operator; it only fixes the draw order so that `epsilon` is shared across
severities.** See §4.

---

## 2. The six frozen SD fields

### `SD_DEFINITION`

**`float(a.std())` — the PyTorch sample standard deviation of the flattened, pooled
intact activation values at the site.** Not RMS, not mean-abs, not p99.

Evidence. `measure_activation_scales.py:86–99` computes several statistics into one row:

```python
a = torch.cat(chunks)
rms = float(a.pow(2).mean().sqrt())
rows.append({... "rms": rms, "mean_abs": float(a.abs().mean()),
             "std": float(a.std()), "max_abs": float(a.abs().max()), ...})
```

`run_premeeting_atlas.py:48–49` selects exactly one of them:

```python
for r in csv.DictReader(open(sf), delimiter="\t"):
    scales[r["target"]] = float(r["std"])
```

The `rms` column is computed and printed but **is not the atlas scale**. Anyone reading
only the calibrator would plausibly assume RMS (it is the column the console table
prints, `measure_activation_scales.py:114–118`). The atlas is authoritative: it is `std`.

> This is the single most consequential recovery in this audit. `std` and `rms` differ
> whenever the site has a non-zero mean, which a tanh-bounded GRU state generally does.

### `SD_AXES`

**All axes pooled. SD is computed over a fully flattened 1-D tensor; the batch dimension
is pooled with the hidden dimension. Unitwise statistics are NOT retained.**

Evidence, `measure_activation_scales.py:62–67`:

```python
def make_hook(tid, slot):
    def hook(_m, _i, out):
        t = out[slot] if isinstance(out, tuple) else out
        if torch.is_tensor(t):
            caught.setdefault(tid, []).append(t.detach().float().flatten())
    return hook
```

`.flatten()` per call, then `torch.cat(chunks)` (line 86) concatenates every batch into
one 1-D vector, then `.std()` reduces it to a scalar. There is no `dim=` argument
anywhere in the chain.

### `SD_DDOF`

**1 — PyTorch's default unbiased / sample standard deviation (`correction=1`).**

`torch.Tensor.std()` is called with no `correction` / `unbiased` argument
(`measure_activation_scales.py:95`), so the default applies. Verified empirically under
the live interpreter (torch 2.12.1): `torch.tensor([1.,2.,3.,4.]).std() == 1.2909944`,
which equals `statistics.stdev` (ddof=1) and not `statistics.pstdev` (ddof=0).

This is population SD **of nothing**: it is the sample SD of the pooled activation
values. With `n ≈ 2048 × H` the ddof choice is numerically negligible, but it is frozen
here so the value can never silently drift.

### `SD_POPULATION`

**A deterministic 2 048-item sample of the intact model's own entry list, drawn by
`deterministic_sample(range(len(entries)), n=2048, seed=7)`, evaluated in
`batch_size=256` chunks under a single teacher-forced forward per batch.**

Evidence, `measure_activation_scales.py:42–47` (defaults) and 56–58, 77–80:

```python
p.add_argument("--n-items", type=int, default=2048)
p.add_argument("--sample-seed", type=int, default=7)
p.add_argument("--batch-size", type=int, default=256)
...
idx = deterministic_sample(range(len(loaded.entries)), args.n_items, args.sample_seed)
forms = [loaded.entries[i].phonemes for i in idx]
...
with torch.no_grad():
    for lo in range(0, len(forms), args.batch_size):
        b = make_batch(forms[lo:lo + args.batch_size], loaded.vocab, args.device)
        loaded.model(b["enc_in"], b["enc_mask"], b["dec_in"])
```

`deterministic_sample` (`scripts/naming_comprehension/train_joint_scratch.py:234–244`,
identical text on both lineages) is *"first `n` of a seeded permutation, returned
sorted"* — a pure function of `(population, n, seed)` that never touches the global RNG.
Sorted output means **the SD population is in ascending entry-index order**, not in
permutation order.

Two properties of this forward matter and are preserved:

* it is **teacher-forced** (`make_batch` builds `dec_in` from the gold form,
  `evaluate/hooks.py:31–49`), not autoregressive. Each item contributes its encoder
  state exactly once per batch.
* it is a **`model(...)` call**, i.e. the full dual-route forward, so both encoders are
  driven for every item.

### `SD_TENSOR_SHAPE`

**Scalar. One float per (checkpoint, target). Not a per-unit vector, not a tensor.**

Evidence, `lesion/spec.py:84–86`:

```python
#: target_id -> measured activation scale, required when noise_scaling
#: == "relative" (decision D-15). Frozen before any sweep, never fitted.
activation_scales: Tuple[Tuple[str, float], ...] = ()
```

and `lesion/spec.py:177–187`, which returns `float(scales[target_id])` — a Python float,
consumed at `lesion/apply.py:171` as a scalar multiplier. The atlas writes it into the
row as the scalar `site_sd_intact` and the scalar `noise_amplitude_absolute = lam*sd_j`
(`run_premeeting_atlas.py:90–91`).

> **Answering the explicit question:** SD is a **single site/route scalar**, not a
> per-unit vector. The normalisation is isotropic across hidden units. GXLR preserves
> this. A per-unit normalisation would be a different operator and is out of scope.

### `ZERO_SD_BEHAVIOR`

**No special-casing, no guard, no epsilon floor. A zero scale propagates to a zero
amplitude, which `draw_noise` short-circuits to an exact zero tensor.**

Evidence. `lesion/spec.py:177–187` (`activation_scale_for`) has no zero test — it raises
only for a *missing* target, never for a zero value. `lesion/apply.py:169–176` gates on
`amp > 0.0`, so an exactly-zero SD means **no hook is registered at all**.
`lesion/masks.py:182–183` independently short-circuits:

```python
if amplitude == 0.0:
    out = torch.zeros(tuple(shape), **kw)
```

Note the contrast with `measure_activation_scales.py:99`, which *does* clamp
(`UENO_MAX_NOISE / max(rms, 1e-9)`) — but that clamp guards a **reporting column**
(`ueno_max_noise_over_rms`) and never touches the operator path. GXLR does not import
that clamp.

**Device / dtype.** SD is computed on CPU (`--device` default `"cpu"`,
`measure_activation_scales.py:46`) in **float32**: the hook casts with `.detach().float()`
(line 66) regardless of the site's own dtype, and `.std()` on a float32 tensor returns
float32, materialised to a Python float by `float(...)`. It is stored as a decimal string
in a TSV and re-read with `float(...)` (float64) at `run_premeeting_atlas.py:49`. The
noise tensor is then drawn in the *reference tensor's* dtype
(`lesion/apply.py:92–93` → `draw_noise(..., dtype=ref.dtype)`), i.e. float32 here.

---

## 3. The activation tensor actually collected

`_GRU_SLOT` (`lesion/apply.py:116–128`) selects which element of an `nn.GRU` return
tuple is the lesion site:

```python
_GRU_SLOT = {
    "wm_encoder": 1, "wm_encoder_state": 1,
    "ltm_encoder": 1, "ltm_encoder_state": 1,
    "wm_decoder": 0, "wm_decoder_state": 0,
    ...
}
```

Slot 1 is `h_n`. The same table is imported by the calibrator
(`measure_activation_scales.py:35`), so **the tensor whose SD is measured and the tensor
that is perturbed are the same tensor by construction.** GXLR keeps that coupling.

Historical target declarations (`lesion/targets.py:98–109`):

| target | `module_path` | declared activation |
|---|---|---|
| `wm_encoder_state` | `wm.encoder` | *"dorsal recurrent state h_n (1, B, H) — the bounded phonological store; tanh-bounded to [−1, 1]"* |
| `ltm_encoder_state` | `ltm.encoder` | *"ventral encoder last hidden state, pre-to_semantic; tanh-bounded to [−1, 1]"* |

**These notes are confirmed against the live W3/W4 architecture in `LIVE_CODE_AUDIT.md`
§2 and are correct for these checkpoints — but only because both carry
`ltm_encoder_mode = "unigru_last_hidden"`. Under the alternative
`bigru_masked_mean` mode the ventral route reads slot 0 and discards `h_n`, and slot 1
would be a silent no-op.** The live audit pins the mode; GXLR asserts it at run time.

---

## 4. What the micro-amendment changes, and what it does not

### Unchanged (inherited verbatim)

* SD definition, axes, ddof, population recipe, scalar shape, zero behaviour (§2).
* The site selection and GRU slot semantics (§3).
* The distribution: `uniform_symmetric`, `eta = (U(0,1)·2 − 1) · amplitude`
  (`lesion/masks.py:184–185`), i.e. `epsilon ~ U(−1, +1)`.
* The multiplication order `epsilon · amplitude` with `amplitude = λ · SD`, preserving
  the historical floating-point path.
* Non-mutation of the checkpoint: hooks only, removed in `finally`
  (`lesion/apply.py:180–185`).

### Changed, deliberately, by CENTRAL micro-amendment

1. **RNG identity.** Historical: one `torch.Generator` per `(lesion_seed, target_id,
   "noise")` (`lesion/masks.py:26–40`), consumed *sequentially*, so the draw depended on
   batch composition, batch order, call count, and — because `resolved_noise_amplitude()`
   fed the same stream — could not be shared across severities. GXLR: a per-item
   deterministic stream keyed by
   `(checkpoint_sha256, route, lesion_seed, item_id)`, with **λ and fusion condition
   excluded**. Consequences: batch size and batch order become irrelevant; the same
   `epsilon` is reused across `{0.25, 0.50, 1.00}`; NATIVE and FIXED05 provably consume
   the same tensor.

2. **Timing.** Historical atlas ran `noise_timing="per_step"`
   (`run_premeeting_atlas.py:80`), a fresh draw at every forward call. CENTRAL freezes
   `per_item_frozen`. The historical `per_item` path existed
   (`lesion/apply.py:68–96`) but inferred item identity from a *content fingerprint of
   the encoder input* (lines 43–65) — a heuristic that could collide. GXLR replaces
   inference with an explicit bound item-id list, which is exact.

3. **Severity grid.** Historical λ ∈ {0, 0.5, 1, 2, 4} over 8 seeds and 7 targets
   (`run_premeeting_atlas.py:27–32`). CENTRAL freezes λ ∈ {0.25, 0.50, 1.00}, seeds
   {0,1,2,3}, two routes. No severity is added, dropped, or chosen by outcome.

4. **Scope.** Connectivity damage, composite sites, mask scope/granularity (`D-05`) are
   **not ported**. GXLR implements activation noise only. `lesion/masks.py`'s mask
   machinery is deliberately left behind.

### Why the nesting is exact

`λ ∈ {0.25, 0.50, 1.00}` are exact binary powers of two. Scaling a float by a power of
two commutes with round-to-nearest (absent overflow/subnormal), so with
`amp(λ) = float32(λ · SD)`:

```
amp(0.50) == 2·amp(0.25)   and   amp(1.00) == 2·amp(0.50)   bitwise
⇒  epsilon·amp(0.50) == 2·(epsilon·amp(0.25))               bitwise, elementwise
```

Verified empirically over 1 000 float32 draws (see `LIVE_CODE_AUDIT.md` §5). The frozen
deterministic tolerance for T5 is therefore **exactly 0.0 (bitwise)**, and SD semantics
were *not* altered to obtain it.

---

## 5. Defects found in the historical operator (recorded, not inherited)

| # | defect | historical location | GXLR disposition |
|---|---|---|---|
| H-1 | `per_item` identity inferred from a float fingerprint of the encoder input; two distinct batches with equal shape, dtype, sum and first element would silently share a draw | `lesion/apply.py:43–65` | **Removed.** Item identity is bound explicitly per batch. |
| H-2 | noise depends on batch size and batch order, because one generator is consumed sequentially for whole-batch draws | `lesion/apply.py:68–96`, `masks.py:26–40` | **Removed.** Per-item streams; batching is an implementation detail with no scientific footprint. |
| H-3 | severities cannot share a realisation: a different λ re-enters the same sequential stream at a different point | `lesion/apply.py:169–176` | **Removed.** λ excluded from RNG identity (CENTRAL §6.2). |
| H-4 | the calibrator prints RMS but the atlas consumes `std`; a reader of either file alone infers the wrong operator | `measure_activation_scales.py:114–118` vs `run_premeeting_atlas.py:49` | **Recorded.** `std` is authoritative and is now asserted by name in the contract. |
| H-5 | `_GRU_SLOT` hard-codes slot 1 for `ltm_encoder_state`, which is wrong under `bigru_masked_mean` | `lesion/apply.py:116–128` | **Guarded.** GXLR asserts `ltm_encoder_mode == "unigru_last_hidden"` at run time and refuses otherwise. |
| H-6 | the atlas's own scales were measured on a *different* checkpoint (`chigh_15e5_h512_s22_u3000`) than any GXLR state | `run_premeeting_atlas.py:37,47` | **Not inherited.** SD is re-measured per GXLR state on that state's own intact model, by the same recipe. |

None of these defects affect the *definition* of SD, which is what this audit was
commissioned to recover. H-1..H-3 concern the draw, and are exactly what CENTRAL's
micro-amendment 2 replaces.

---

## 6. Provenance of every claim in this document

| claim | blob | path @ `3af0ef9` |
|---|---|---|
| SD = `std` | `git show 3af0ef9:scripts/lesion/run_premeeting_atlas.py` | 46–53 |
| SD statistics computed | `git show 3af0ef9:scripts/lesion/measure_activation_scales.py` | 84–100 |
| SD population recipe | same | 42–47, 56–58, 77–80 |
| SD flattening | same | 62–67, 86 |
| SD is a scalar | `git show 3af0ef9:lesion/spec.py` | 84–86, 177–187 |
| amplitude composition | `git show 3af0ef9:lesion/apply.py` | 169–176 |
| uniform symmetric draw | `git show 3af0ef9:lesion/masks.py` | 171–192 |
| zero-amplitude short circuit | same | 182–183 |
| RNG derivation | same | 26–40 |
| GRU slot | `git show 3af0ef9:lesion/apply.py` | 99–128 |
| target declarations | `git show 3af0ef9:lesion/targets.py` | 98–109 |
| restoration on exit | `git show 3af0ef9:lesion/apply.py` | 180–185 |
| `deterministic_sample` | `git show 3af0ef9:scripts/naming_comprehension/train_joint_scratch.py` | 234–244 |

**NO SCIENTIFIC LESION EXECUTION accompanies this audit.**
