#!/usr/bin/env python
"""GATE 1 — verify the ACTUAL P1-P4 POST_REPAIR state_dicts, read-only.

Reconstructs POST_REPAIR as SOURCE checkpoint + head_first_c0 (never
head_final) and asserts every frozen tensor name and shape. No behavioural
forward pass is performed. Any mismatch is a HARD STOP.

    python verify_states.py --v7-run-root <root> --out <json>
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

from paper_programme.lesioning_v2.lesion_operator.sites import (  # noqa: E402
    DEC_HIDDEN, ENC_HIDDEN, PHON_EMBED, REQUIRED_SHAPES, SEMANTIC, WM_HIDDEN)
from paper_programme.lesioning_v2.scripts.build_run_matrix import (  # noqa: E402
    FROZEN_IDENTITY, STATE_ORDER, composite_state_sha256)

sys.path.insert(0, os.path.join(
    REPO, "paper_programme", "v7_prelesion_validation", "execution"))

RUN_ID = {"P1_POST_REPAIR": ("p1", 31, 4125330, 1485),
          "P2_POST_REPAIR": ("p2", 32, 4083660, 1470),
          "P3_POST_REPAIR": ("p3", 33, 2486310, 895),
          "P4_POST_REPAIR": ("p4", 34, 6444960, 2320)}


class VerificationFailed(RuntimeError):
    pass


def paths_for(root: str, state: str):
    slot, seed, step, u = RUN_ID[state]
    run = os.path.join(root, f"fresh_ceiling_v7_{slot}_s{seed}")
    return (os.path.join(run, "checkpoints", f"step_{step:08d}.pt"),
            os.path.join(run, "post", f"seed{seed}_u{u}_A", "head_first_c0.pt"))


def verify(root: str) -> dict:
    import torch
    out = {}
    for state in STATE_ORDER:
        ck, head = paths_for(root, state)
        for p in (ck, head):
            if not os.path.exists(p):
                raise VerificationFailed(f"{state}: missing {p}")
        import hashlib

        def sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""):
                    h.update(c)
            return h.hexdigest()

        ident = FROZEN_IDENTITY[state]
        if sha(ck) != ident["source_checkpoint_sha256"]:
            raise VerificationFailed(f"{state}: SOURCE checkpoint SHA mismatch")
        if sha(head) != ident["repair_head_file_sha256"]:
            raise VerificationFailed(f"{state}: head_first_c0 SHA mismatch")

        blob = torch.load(ck, map_location="cpu", weights_only=False)
        msd = (blob.get("model_state_dict") or blob.get("model")
               or blob.get("state_dict"))
        if msd is None:
            raise VerificationFailed(f"{state}: no model state in checkpoint")
        hb = torch.load(head, map_location="cpu", weights_only=False)
        hstate = hb["state"]
        if set(hstate) != {"2.weight", "2.bias"}:
            raise VerificationFailed(
                f"{state}: repair head keys {sorted(hstate)} != "
                "{'2.weight','2.bias'}")

        shapes = {}
        for name, want in REQUIRED_SHAPES.items():
            if name not in msd:
                raise VerificationFailed(f"{state}: missing tensor {name}")
            got = tuple(msd[name].shape)
            if got != want:
                raise VerificationFailed(
                    f"{state}: {name} shape {got} != required {want}")
            shapes[name] = got

        # widths and sharing
        if tuple(msd["wm.encoder.weight_hh_l0"].shape)[1] != WM_HIDDEN:
            raise VerificationFailed(f"{state}: WM hidden != {WM_HIDDEN}")
        if tuple(msd["ltm.encoder.weight_hh_l0"].shape)[1] != ENC_HIDDEN:
            raise VerificationFailed(f"{state}: LTM enc hidden != {ENC_HIDDEN}")
        if tuple(msd["ltm.sem_to_h0.weight"].shape) != (DEC_HIDDEN, SEMANTIC):
            raise VerificationFailed(f"{state}: sem_to_h0 != {(DEC_HIDDEN, SEMANTIC)}")
        if tuple(msd["phon_embed.weight"].shape)[1] != PHON_EMBED:
            raise VerificationFailed(f"{state}: phon_embed dim != {PHON_EMBED}")
        shared = all(
            torch.equal(msd["phon_embed.weight"], msd[k])
            for k in ("wm.phon_embed.weight", "ltm.phon_embed.weight")
            if k in msd)
        if not shared:
            raise VerificationFailed(f"{state}: phon_embed is NOT shared")

        out[state] = {
            "source_checkpoint": ck, "repair_head": head,
            "source_checkpoint_sha256": ident["source_checkpoint_sha256"],
            "repair_head_file_sha256": ident["repair_head_file_sha256"],
            "composite_state_sha256": composite_state_sha256(state),
            "shapes": {k: list(v) for k, v in shapes.items()},
            "phon_embed_shared": True,
            "widths": {"wm_hidden": WM_HIDDEN, "ltm_enc_hidden": ENC_HIDDEN,
                       "ltm_dec_hidden": DEC_HIDDEN, "semantic": SEMANTIC,
                       "phon_embed": PHON_EMBED},
            "head_final_referenced": False,
            "verified": True,
        }
        print(f"{state}: VERIFIED")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-run-root", required=True)
    ap.add_argument("--out", default=os.path.join(
        PKG, "contract", "LESIONING_V2_STATE_VERIFICATION.json"))
    a = ap.parse_args(argv)
    try:
        res = verify(a.v7_run_root)
    except VerificationFailed as e:
        print(f"HARD_STOP {e}", file=sys.stderr)
        return 2
    import torch
    json.dump({"torch_version": torch.__version__, "states": res},
              open(a.out, "w"), indent=1)
    print(f"L1_TENSOR_VERIFIED=YES\nL2_TENSOR_VERIFIED=YES\nL3_TENSOR_VERIFIED=YES")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
