#!/usr/bin/env python
"""GATE 3 — intact site-SD extraction. INTACT ONLY: no mask, no noise, no k>0.

Applies the frozen procedure in `operator/sd_procedure.py`, whose hash is
recorded alongside every constant it produces. Refuses to run if the procedure
hash differs from the one frozen in the contract.

    python extract_intact_sd.py --v7-run-root <root> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
sys.path.insert(0, REPO)

from paper_programme.lesioning_v2.lesion_operator import sd_procedure as SDP  # noqa: E402
from paper_programme.lesioning_v2.lesion_operator.sites import SITE_ORDER  # noqa: E402
from paper_programme.lesioning_v2.scripts.build_run_matrix import (  # noqa: E402
    FROZEN_IDENTITY, STATE_ORDER, composite_state_sha256)
from paper_programme.lesioning_v2.scripts.verify_states import paths_for  # noqa: E402


def site_activations(tr, model, indices, site: str):
    """Collect the intact site tensor for one batch-sliced population.

    L1  final wm.encoder h_n
    L2  final ltm.encoder h_n, BEFORE to_semantic
    L3  h0 = tanh(sem_to_h0(s_hat)), AFTER tanh
    """
    import torch
    from scripts.naming_comprehension.train_joint_scratch import build_batch

    chunks = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for lo in range(0, len(indices), SDP.SD_BATCH_SIZE):
                idx = list(indices[lo:lo + SDP.SD_BATCH_SIZE])
                b = build_batch(tr.entries, tr.bank_raw, tr.vocab, idx, "cpu")
                if site == "L1":
                    a = model.wm.encode(b["enc_in"], b["enc_mask"])
                elif site == "L2":
                    a = model.ltm.encode(b["enc_in"], b["enc_mask"])
                elif site == "L3":
                    s_hat = model.ltm.encode(b["enc_in"], b["enc_mask"])
                    a = torch.tanh(model.ltm.sem_to_h0(s_hat))
                else:                                    # pragma: no cover
                    raise ValueError(site)
                chunks.append(a.detach().reshape(-1).to(torch.float32).cpu())
    finally:
        model.train(was_training)
    return torch.cat(chunks)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-run-root", required=True)
    ap.add_argument("--expect-procedure-hash", default=None)
    ap.add_argument("--out", default=os.path.join(
        PKG, "contract", "LESIONING_V2_INTACT_SD_CONSTANTS.json"))
    a = ap.parse_args(argv)

    phash = SDP.procedure_hash()
    if a.expect_procedure_hash and a.expect_procedure_hash != phash:
        print(f"HARD_STOP procedure hash {phash} != expected "
              f"{a.expect_procedure_hash}", file=sys.stderr)
        return 2
    if os.path.exists(a.out):
        print(f"HARD_STOP refusing to overwrite {a.out}", file=sys.stderr)
        return 2

    import torch
    sys.path.insert(0, os.path.join(
        REPO, "paper_programme", "v7_prelesion_validation", "scripts"))
    import prelesion_eval as pe
    from scripts.naming_comprehension.ceiling_source_completion import GLOVE

    out = {"procedure_hash": phash, "procedure": SDP.procedure_descriptor(),
           "torch_version": torch.__version__, "constants": {}}
    for state in STATE_ORDER:
        ck, head = paths_for(a.v7_run_root, state)
        tr, model, prov = pe.build_state(ck, head, GLOVE)
        pop = SDP.population_for(len(tr.entries))
        pophash = SDP.population_hash(pop)
        for site in SITE_ORDER:
            vals = site_activations(tr, model, pop, site)
            sd = float(vals.std())                      # ddof=1, unbiased
            out["constants"][f"{state}/{site}"] = {
                "state_id": state, "site": site,
                "source_checkpoint_sha256":
                    FROZEN_IDENTITY[state]["source_checkpoint_sha256"],
                "repair_head_file_sha256":
                    FROZEN_IDENTITY[state]["repair_head_file_sha256"],
                "repair_head_deployed_state_sha256":
                    prov.get("repair_head_deployed_state_sha256"),
                "composite_state_sha256": composite_state_sha256(state),
                "tensor_definition": SDP.SD_SITE_TENSOR[site],
                "historically_calibrated_site":
                    SDP.SD_SITE_IS_HISTORICALLY_CALIBRATED[site],
                "n_sampled": len(pop), "population_hash": pophash,
                "n_activation_scalars": int(vals.numel()),
                "dtype": SDP.SD_DTYPE, "device": SDP.SD_DEVICE,
                "torch_version": torch.__version__,
                "sample_sd": sd,
                "descriptive_mean_non_operative": float(vals.mean()),
                "descriptive_rms_non_operative":
                    float((vals ** 2).mean().sqrt()),
                "operational_quantity": "sample_sd",
                "procedure_hash": phash,
                "lesion_applied": False, "activation_noise_applied": False,
            }
            print(f"{state}/{site}: sample_sd={sd:.6f}")
        pe.assert_source_unchanged(ck, pe.sha256_file(ck))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"INTACT_SD_CONSTANTS_FROZEN=YES\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
