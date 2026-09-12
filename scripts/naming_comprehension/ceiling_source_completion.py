"""Amendment-V5 completion driver: apply the FROZEN V4 Arm-A repair to a
preregistered NON-ENDPOINT official SETTLE control checkpoint.

Preregistration: docs/analysis/CEILING_ASSEMBLY_PREREG_V5.md
                 commit 86ea3d329245ae2f0314d5c1f99be9ab575c7816

This module is PATH PLUMBING ONLY.  It adds no lever and changes no rule:

  * the learning rule is `gradient_training_probe.train_run(arm="A", ...)`,
    imported unchanged -- gamma_raw, the frozen J_lin_h0 preconditioner, the
    length-normalised Armijo line search, the stopping rule, MAX_ITERS, the
    wall cap and the CG semantics all come from that module;
  * the evaluator is `frozen_head_probe.official_strict_errors` and the final
    battery is `coexistence_probe.full_battery`, both unchanged;
  * only `ltm.to_semantic.2.{weight,bias}` is trainable.

V5 section 6 requires every source-dependent quantity to be rebuilt FROM THE
SELECTED CHECKPOINT.  `train_run` already derives z0, D = 1 - tanh(z0)^2 and
the Jlinh0 Metric from the (theta0, A, c, X) it is handed, so pointing it at a
u3400 cache rebuilds the metric by construction; the u3600 cache is never read.

The V3/V3.1 witnesses were solved AT THE u3600 ENDPOINT, so they are not valid
reference points for an earlier milestone.  `theta_v3` is therefore set to
theta0, which makes the purely diagnostic cos/dist-to-V3 columns null (they are
logged only; they never enter the step).  That is recorded as VOID rather than
silently comparing across sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.constrained_coexistence_v3 import (  # noqa: E402
    GAMMA_RAW, FEAS_TOL_RAW, augment,
)
from scripts.naming_comprehension.frozen_head_probe import (  # noqa: E402
    CANON, N_BANK, N_COMP_POP, POOLED_DIM, SEM_DIM,
    build_trainer, official_strict_errors, sha256_file, sha256_tensor,
    _isolated_model,
)
from scripts.naming_comprehension.coexistence_probe import full_battery  # noqa: E402
from scripts.naming_comprehension.gradient_training_probe import (  # noqa: E402
    ARMIJO_C, BETA, CG_RELRES_FAIL, DERIVED_LABEL, MAX_BACKTRACKS, MAX_ITERS,
    R0, TAU, TRAINABLE_NAMES, WALL_CAP_S,
    _save_head, _theta_of, assert_unchanged, refuse_protected, train_run,
)

PREREG_COMMIT = "86ea3d329245ae2f0314d5c1f99be9ab575c7816"
V4_PREREG_COMMIT = "054633cdf173c27296bdbc406775329bd0bc38a4"
V4_IMPL_COMMIT = "3912d758e4864a2e8a297f38311c6162772562d8"
GLOVE = os.path.join(ROOT, "data", "glove.6B.300d.txt")
ARM = "A"


# ==================================================================== extract
def cmd_extract(a) -> int:
    """Rebuild the frozen cache FROM THE SELECTED SOURCE, with the same
    capture and reconciliation logic as `frozen_head_probe.cmd_extract`."""
    torch.set_num_threads(a.nt)
    refuse_protected(os.path.dirname(os.path.abspath(a.out)))
    before = sha256_file(a.ckpt)
    ck_meta = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if int(ck_meta["global_step"]) != a.expect_step:
        raise RuntimeError(f"global_step {ck_meta['global_step']} != {a.expect_step}")
    if int(ck_meta["seed"]) != a.seed:
        raise RuntimeError(f"seed {ck_meta['seed']} != {a.seed}")
    if ck_meta.get("comprehension_population_sha256") != \
            CANON["comprehension_population_sha256"]:
        raise RuntimeError("comprehension population hash mismatch")
    if ck_meta.get("lexicon_file_sha256") != CANON["lexicon_sha256"]:
        raise RuntimeError("lexicon hash mismatch")
    glove_sha = sha256_file(GLOVE)
    if glove_sha != CANON["glove_sha256"]:
        raise RuntimeError(f"glove hash mismatch {glove_sha}")

    tr, ckd = build_trainer(a.ckpt, "cpu", GLOVE)
    model = tr.model
    model.eval()
    idx = list(tr.comp_idx)
    if len(idx) != N_COMP_POP:
        raise RuntimeError(f"C population {len(idx)} != {N_COMP_POP}")
    if int(tr.bank_raw.shape[0]) != N_BANK:
        raise RuntimeError(f"bank {tr.bank_raw.shape[0]} != {N_BANK}")

    from scripts.naming_comprehension.frozen_probe import encode_all
    forms = [tr.entries[i].phonemes for i in idx]
    buf: List[torch.Tensor] = []
    h = model.ltm.to_semantic.register_forward_pre_hook(
        lambda mod, inp: buf.append(inp[0].detach().clone()))
    with torch.no_grad():
        s_hat = encode_all(model, tr.vocab, forms, "cpu", 512)
    h.remove()
    pooled = torch.cat(buf, 0)
    if pooled.shape != (N_COMP_POP, POOLED_DIM):
        raise RuntimeError(f"pooled shape {tuple(pooled.shape)}")

    ts = model.ltm.to_semantic
    with torch.no_grad():
        phi = F.gelu(ts[0](pooled))
        z = ts[2](phi)
    dev = float((z - s_hat).abs().max())
    if dev > 1e-4:
        raise RuntimeError(f"phi path deviates from s_hat by {dev:.3e}")

    base = official_strict_errors(s_hat, tr.bank_raw, idx)
    if base["errors"] != a.expect_c:
        raise RuntimeError(f"source strict C {base['errors']} != preregistered {a.expect_c}")

    # DETERMINISM, with batching held constant.
    #
    # V6 defect fix (assertion only; no scientific semantics change).  The
    # previous sampled check compared a re-encode of a scattered 512-item
    # subset against the corresponding rows of the full pass.  `encode_all`
    # pads every chunk to `max_enc = max(len(f) for f in chunk) + 1`, so
    # re-batching a scattered subset changes the padded width.  That check
    # therefore required BATCH-COMPOSITION INVARIANCE, not determinism -- a
    # property the science never relies on (the frozen cache, phi, Arm A and
    # the official evaluators all use one fixed canonical batching) and which
    # is not stably satisfiable: it passed in V5 on seed 21 and failed here on
    # seeds 19/20 purely through chunk-length coincidence.
    #
    # The determinism requirement is kept and asserted: re-encode the SAME
    # items with the SAME chunking and require bitwise equality, both for the
    # full population and for a sampled subset.  The cross-batching deviation
    # is still measured and recorded, as a DIAGNOSTIC, never as a gate.
    with torch.no_grad():
        s2 = encode_all(model, tr.vocab, forms, "cpu", 512)
    det_full = bool(torch.equal(s_hat, s2))
    g = torch.Generator().manual_seed(12345)
    sample = torch.randperm(N_COMP_POP, generator=g)[:512].tolist()
    sub_forms = [forms[i] for i in sample]
    with torch.no_grad():
        s3a = encode_all(model, tr.vocab, sub_forms, "cpu", 512)
        s3b = encode_all(model, tr.vocab, sub_forms, "cpu", 512)
    det_sample = bool(torch.equal(s3a, s3b))
    # diagnostic only: identical items, DIFFERENT chunk composition
    rebatch_dev = float((s3a - s_hat[sample]).abs().max())
    if not (det_full and det_sample):
        raise RuntimeError(f"extraction not deterministic (full={det_full} sample={det_sample})")

    head0 = {k: v.detach().clone() for k, v in ts.state_dict().items()}
    msd = ckd.get("model") or ckd.get("model_state_dict") or ckd.get("state_dict")
    Aw = [v for k, v in msd.items() if k.endswith("sem_to_h0.weight")][0]
    Ab = [v for k, v in msd.items() if k.endswith("sem_to_h0.bias")][0]
    blob = {"seed": a.seed, "target_idx": idx, "pooled": pooled, "phi": phi,
            "s_hat_reference": s_hat, "bank_raw": tr.bank_raw.cpu(),
            "head_state": head0, "checkpoint": a.ckpt, "checkpoint_sha256": before,
            "global_step": int(ckd["global_step"]), "source_u": a.source_u,
            "sem_to_h0_weight": Aw.detach().clone(), "sem_to_h0_bias": Ab.detach().clone()}
    torch.save(blob, a.out)
    assert_unchanged(a.ckpt, before)

    val = {
        "source_checkpoint": a.ckpt, "source_sha256": before,
        "global_step": int(ckd["global_step"]), "seed": int(ckd["seed"]),
        "source_u": a.source_u, "widths": ckd.get("widths"),
        "regime": ckd.get("regime"), "schedule": ckd.get("schedule"),
        "subset_mode": ckd.get("subset_mode"),
        "comprehension_population_n": len(idx), "retrieval_bank_n": int(tr.bank_raw.shape[0]),
        "comprehension_population_sha256": ckd.get("comprehension_population_sha256"),
        "lexicon_sha256": ckd.get("lexicon_file_sha256"), "glove_sha256": glove_sha,
        "canon_match": True,
        "strict_c_errors": base["errors"], "strict_c_expected": a.expect_c,
        "phi_path_max_dev_vs_s_hat": dev,
        "deterministic_reencode_full": det_full,
        "deterministic_reencode_sample_512": det_sample,
        "sample_determinism_semantics": "same items, SAME chunking, bitwise equal "
                                        "(batching held constant)",
        "rebatch_max_dev_same_items_diff_chunking": rebatch_dev,
        "rebatch_note": "DIAGNOSTIC ONLY, never a gate: identical items re-encoded "
                        "under a different chunk composition, where encode_all pads "
                        "each chunk to its own max length.  Batch-composition "
                        "invariance is not required by the frozen pipeline.",
        "item_order_sha256": hashlib.sha256(",".join(map(str, idx)).encode()).hexdigest(),
        "first5_target_idx": idx[:5], "last5_target_idx": idx[-5:],
        "hashes": {
            "pooled": sha256_tensor(pooled), "phi": sha256_tensor(phi),
            "s_hat_reference": sha256_tensor(s_hat),
            "bank_raw": sha256_tensor(tr.bank_raw.cpu()),
            "W0_to_semantic_2_weight": sha256_tensor(head0["2.weight"]),
            "b0_to_semantic_2_bias": sha256_tensor(head0["2.bias"]),
            "to_semantic_0_weight": sha256_tensor(head0["0.weight"]),
            "to_semantic_0_bias": sha256_tensor(head0["0.bias"]),
            "sem_to_h0_weight": sha256_tensor(Aw), "sem_to_h0_bias": sha256_tensor(Ab),
        },
        "cache_file": a.out, "cache_sha256": sha256_file(a.out),
        "dtype": str(pooled.dtype), "torch": torch.__version__,
        "code_commit": a.code_commit, "code_dirty_paths": a.code_dirty,
        "prereg_commit": PREREG_COMMIT,
    }
    json.dump(val, open(a.out_json, "w"), indent=1)
    print(f"== extract seed {a.seed} u{a.source_u}: C={base['errors']} "
          f"dev={dev:.3e} det_full={det_full} det_sample={det_sample}", flush=True)
    return 0


# ====================================================================== train
def cmd_train(a) -> int:
    torch.set_num_threads(a.nt)
    run_dir = os.path.join(a.out_dir, f"seed{a.seed}_u{a.source_u}_{ARM}")
    refuse_protected(run_dir)
    if os.path.exists(run_dir):
        raise RuntimeError(f"REFUSED: run dir exists: {run_dir}")
    os.makedirs(run_dir)
    ck_before = sha256_file(a.ckpt)
    blob = torch.load(a.rep, map_location="cpu", weights_only=False)
    if blob["checkpoint_sha256"] != ck_before:
        raise RuntimeError("cached representation does not match source checkpoint")
    if int(blob["global_step"]) != a.expect_step:
        raise RuntimeError("cache global_step mismatch")
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    msd = sd.get("model") or sd.get("model_state_dict") or sd.get("state_dict")
    A = [v for k, v in msd.items() if k.endswith("sem_to_h0.weight")][0].double()
    c = [v for k, v in msd.items() if k.endswith("sem_to_h0.bias")][0].double()
    X = augment(blob["phi"].double())
    bn = F.normalize(blob["bank_raw"].double(), dim=-1)
    tgt = torch.as_tensor(blob["target_idx"]).long()
    idx = [int(i) for i in blob["target_idx"]]
    theta0 = _theta_of(blob["head_state"]["2.weight"], blob["head_state"]["2.bias"])
    bank_raw = blob["bank_raw"]
    # V3 witnesses were solved at the u3600 endpoint; they are NOT a valid
    # reference for an earlier milestone.  theta_v3 = theta0 => the diagnostic
    # cos/dist-to-V3 columns are null.  They never enter the step.
    res = train_run(ARM, theta0, X, bn, tgt, A, c,
                    evaluator=lambda s: official_strict_errors(s, bank_raw, idx)["errors"],
                    theta_v3=theta0.clone(), theta_v3_best=None,
                    v3_active_bank=None, verbose=True)
    assert_unchanged(a.ckpt, ck_before)
    meta = {"seed": a.seed, "arm": ARM, "source_u": a.source_u,
            "prereg_commit": PREREG_COMMIT, "v4_prereg_commit": V4_PREREG_COMMIT,
            "v4_impl_commit": V4_IMPL_COMMIT,
            "code_commit": a.code_commit, "code_dirty_paths": a.code_dirty}
    heads = {}
    if res["first_c0_theta"] is not None:
        heads["first_c0"] = _save_head(os.path.join(run_dir, "head_first_c0.pt"),
                                       res["first_c0_theta"],
                                       {**meta, "iterate": res["first_c0_iter"]})
    heads["final"] = _save_head(os.path.join(run_dir, "head_final.pt"),
                                res["final_theta"], {**meta, "iterate": res["steps"]})
    rec = {k: v for k, v in res.items() if k not in ("first_c0_theta", "final_theta")}
    rec.update({"provenance": {
        **meta, "source_checkpoint": a.ckpt, "source_sha256": ck_before,
        "source_unchanged": True, "source_global_step": int(blob["global_step"]),
        "rep_cache": a.rep, "rep_sha256_file": sha256_file(a.rep),
        "v3_reference": "VOID_NON_ENDPOINT_SOURCE",
        "theta_v3_used": "theta0 (diagnostic cos/dist columns null)",
        "threads": a.nt, "torch": torch.__version__, "argv": sys.argv,
        "trainable": list(TRAINABLE_NAMES),
        "constants": {"GAMMA_RAW": GAMMA_RAW, "FEAS_TOL_RAW": FEAS_TOL_RAW,
                      "TAU": TAU, "R0": R0, "BETA": BETA, "ARMIJO_C": ARMIJO_C,
                      "MAX_BACKTRACKS": MAX_BACKTRACKS, "MAX_ITERS": MAX_ITERS,
                      "WALL_CAP_S": WALL_CAP_S, "CG_RELRES_FAIL": CG_RELRES_FAIL}},
        "heads_sha256": heads})
    json.dump(rec, open(os.path.join(run_dir, "trace.json"), "w"), indent=1)
    print(f"== seed {a.seed} u{a.source_u} arm {ARM}: {res['status']} "
          f"steps={res['steps']} first_c0={res['first_c0_iter']} "
          f"({res['wall_total_s']}s)", flush=True)
    return 0


# ==================================================================== battery
def cmd_battery(a) -> int:
    torch.set_num_threads(a.nt)
    before = sha256_file(a.ckpt)
    if a.head == "SOURCE":
        tr, _ = build_trainer(a.ckpt, "cpu", GLOVE)
        rec = full_battery(tr, tr.model, list(tr.comp_idx))
        rec.update({"head": "SOURCE_HEAD", "head_sha256": None})
    else:
        h = torch.load(a.head, map_location="cpu", weights_only=False)
        if set(h["state"]) != {"2.weight", "2.bias"}:
            raise RuntimeError("derived head must carry exactly the final-layer parameters")
        tr, _ = build_trainer(a.ckpt, "cpu", GLOVE)
        model = _isolated_model(tr, "p_last_hinge", h["state"])
        rec = full_battery(tr, model, list(tr.comp_idx))
        rec.update({"head": a.head, "head_sha256": sha256_file(a.head)})
    assert_unchanged(a.ckpt, before)
    rec.update({"seed": a.seed, "source_u": a.source_u, "source_checkpoint": a.ckpt,
                "source_sha256": before, "source_unchanged": True})
    json.dump(rec, open(a.out_json, "w"), indent=1)
    print(f"== battery seed {a.seed} u{a.source_u} [{os.path.basename(str(a.head))}]: "
          f"C={rec['c_errors']} Rcan={rec['rep_canonical_full_errors']} "
          f"Rfree={rec['rep_freear_full_errors']} N={rec['naming_errors']} "
          f"LTM={rec['rep_canonical_ltm']:.6f} gate={rec['gate_mean']:.6f}", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "train", "battery"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--source-u", type=int, required=True)
        p.add_argument("--nt", type=int, default=2)
        if name == "extract":
            p.add_argument("--out", required=True)
            p.add_argument("--out-json", required=True)
            p.add_argument("--expect-c", type=int, required=True)
            p.add_argument("--expect-step", type=int, required=True)
            p.add_argument("--code-commit", required=True)
            p.add_argument("--code-dirty", type=int, required=True)
        elif name == "train":
            p.add_argument("--rep", required=True)
            p.add_argument("--out-dir", required=True)
            p.add_argument("--expect-step", type=int, required=True)
            p.add_argument("--code-commit", required=True)
            p.add_argument("--code-dirty", type=int, required=True)
        else:
            p.add_argument("--head", required=True)
            p.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    return {"extract": cmd_extract, "train": cmd_train, "battery": cmd_battery}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
