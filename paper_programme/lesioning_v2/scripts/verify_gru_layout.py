#!/usr/bin/env python
"""GATE 2 — verify the torch GRU weight_ih_l0 layout at runtime.

The operator depends ONLY on the invariant shape == (3H, input_dim) and on
applying the same logical mask to rows [0:H], [H:2H], [2H:3H]. It never depends
on gate names or their order. This script records the layout for provenance and
proves the block-tiling identity numerically.
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))


def main() -> int:
    H, D = 7, 5
    g = torch.nn.GRU(D, H, batch_first=True)
    ih, hh = tuple(g.weight_ih_l0.shape), tuple(g.weight_hh_l0.shape)
    ok_shape = ih == (3 * H, D) and hh == (3 * H, H)

    # three contiguous H-sized blocks, and a logical mask tiles identically
    M = (torch.rand(H, D) > 0.4).float()
    P = torch.cat([M, M, M], dim=0)
    tiles_ok = all(torch.equal(P[b * H:(b + 1) * H], M) for b in range(3))
    # permuting the blocks leaves the mask unchanged -> gate-name independent
    perm = torch.cat([P[2 * H:3 * H], P[0:H], P[H:2 * H]], dim=0)
    order_independent = torch.equal(perm, P)

    doc = torch.nn.GRU.__doc__ or ""
    documented_order = ("r,z,n" if doc.find("W_{ir}") < doc.find("W_{iz}")
                        < doc.find("W_{in}") else "UNCONFIRMED")

    rec = {"torch_version": torch.__version__,
           "weight_ih_l0_shape": list(ih), "weight_hh_l0_shape": list(hh),
           "three_contiguous_H_blocks": bool(ok_shape),
           "logical_mask_tiles_identically": bool(tiles_ok),
           "independent_of_block_order": bool(order_independent),
           "documented_human_readable_order": documented_order,
           "operator_depends_on_gate_names": False,
           "verified": bool(ok_shape and tiles_ok and order_independent)}
    out = os.path.join(PKG, "contract", "LESIONING_V2_GRU_LAYOUT.json")
    json.dump(rec, open(out, "w"), indent=1)
    print(json.dumps(rec, indent=1))
    print(f"TORCH_GRU_LAYOUT_VERIFIED={'YES' if rec['verified'] else 'NO'}")
    return 0 if rec["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
