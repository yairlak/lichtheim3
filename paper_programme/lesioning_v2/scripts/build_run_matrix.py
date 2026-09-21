#!/usr/bin/env python
"""Build the frozen Lesioning V2 run matrix. Pure enumeration; no model touched."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.normpath(os.path.join(PKG, "..", "..")))

from paper_programme.lesioning_v2.lesion_operator import battery, masks, seeds  # noqa: E402
from paper_programme.lesioning_v2.lesion_operator.sites import SITE_ORDER, SITES  # noqa: E402

#: Frozen V7 identities (pre-lesion state_manifest.tsv, design commit 2d240e1f).
#: POST_REPAIR = SOURCE checkpoint + head_first_c0 deployed state. head_final is
#: never referenced.
FROZEN_IDENTITY = {
    "P1_POST_REPAIR": {
        "source_checkpoint_sha256":
            "8e5188055ce5a3fda361b16b482b1010c124473806f07448fe0cd0fd9a7bc707",
        "repair_head_file_sha256":
            "f91ef0a4e17116ee0d3258f81fa28f34825141b3326b5c96a6f3594243456725"},
    "P2_POST_REPAIR": {
        "source_checkpoint_sha256":
            "8d537e02bf02ad119ec190ba2290991e0000768af8e56085606c3fb7d7efedca",
        "repair_head_file_sha256":
            "3e881de5699b90b340c96cfd033f7c819c356260ae1da700df16579fe1104f3e"},
    "P3_POST_REPAIR": {
        "source_checkpoint_sha256":
            "5f4d4d24c16cebc70dc1f87ad3c113a8696346c75e5121910d1bc2da65a95517",
        "repair_head_file_sha256":
            "0e529f2be90868be54b87313ed28ae231d5a37b230e184a8ac1b6559d563429b"},
    "P4_POST_REPAIR": {
        "source_checkpoint_sha256":
            "8ee73263e36d76ba6d815bf57c1d9f6398138280cd24927f64f2bf08e8b7c96a",
        "repair_head_file_sha256":
            "92052a3990b4a2750f9296f3e053d03b0eb876fe8c7b343aac41e4b8a98d7242"},
}


def composite_state_sha256(state_id: str) -> str:
    """Deterministic identity of a POST_REPAIR state, from frozen identities.

    Binding both halves means a mask can never silently migrate between a
    SOURCE state and its repaired counterpart.
    """
    ident = FROZEN_IDENTITY[state_id]
    payload = (f"L3_LESION_V2_V1|STATE|{state_id}"
               f"|{ident['source_checkpoint_sha256']}"
               f"|{ident['repair_head_file_sha256']}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


REALIZATIONS = {"P1_POST_REPAIR": 12, "P2_POST_REPAIR": 12,
                "P3_POST_REPAIR": 12, "P4_POST_REPAIR": 4}
STATE_ORDER = ("P1_POST_REPAIR", "P2_POST_REPAIR",
               "P3_POST_REPAIR", "P4_POST_REPAIR")
INCLUDE_INTACT_BASELINE = True


def build(state_sha: dict) -> dict:
    cells = []
    for state in STATE_ORDER:
        sha = state_sha.get(state) or composite_state_sha256(state)
        for site_name in SITE_ORDER:
            site = SITES[site_name]
            ks = ([0] if INCLUDE_INTACT_BASELINE else []) + list(range(1, 16))
            for r in range(REALIZATIONS[state]):
                for k in ks:
                    if k == 0 and r > 0:
                        continue          # one intact control per (state, site)
                    mp = seeds.mask_payload(sha, site_name, r)
                    cells.append({
                        "state_id": state,
                        "state_sha256": sha,
                        "site": site_name,
                        "connectivity_param": site.connectivity_param,
                        "severity_k": k,
                        "severity_s": masks.severity_fraction(k),
                        "connectivity_fraction": masks.connectivity_fraction(k),
                        "n_logical_edges_total": site.n_logical_edges,
                        "n_logical_edges_removed": masks.n_removed(site, k),
                        "realization": r,
                        "mask_seed_payload": mp,
                        "mask_seed_digest": seeds.digest_hex(mp),
                        "activation_seed_namespace": seeds.noise_payload(
                            sha, site_name, r, "<item_id>"),
                        "sd_constant_ref": f"{state}/{site_name}",
                        "tasks": list(battery.TASKS),
                        "decoding_conventions": list(battery.DECODING_CONVENTIONS),
                        "cell_kind": "INTACT_CONTROL" if k == 0 else "LESION",
                        "output_destination":
                            f"results/{state}/{site_name}/k{k:02d}/r{r:02d}",
                    })
    matrix = {
        "matrix_name": "LESIONING_V2_RUN_MATRIX",
        "operator_audit_commit": "5d1849a0c1d01065ff926d324703e8b767c05223",
        "primary_states": list(STATE_ORDER),
        "source_states_included": False,
        "sites": list(SITE_ORDER),
        "n_severity_levels": masks.N_SEVERITY_LEVELS,
        "severity_note": "15 NONZERO severities; k=0 is an intact control cell "
                         "and is NOT counted among them",
        "connectivity_p_max": masks.CONNECTIVITY_P_MAX,
        "realizations": dict(REALIZATIONS),
        "include_intact_baseline": INCLUDE_INTACT_BASELINE,
        "pseudoword_primary_lesion_battery": False,
        "primary_endpoints": [e.key for e in battery.PRIMARY],
        "diagnostic_endpoints": [e.key for e in battery.DIAGNOSTIC],
        "n_cells": len(cells),
        "n_lesion_cells": sum(1 for c in cells if c["cell_kind"] == "LESION"),
        "n_intact_cells": sum(1 for c in cells if c["cell_kind"] == "INTACT_CONTROL"),
        "cells": cells,
    }
    return matrix


def matrix_sha256(matrix: dict) -> str:
    body = {k: v for k, v in matrix.items() if k != "matrix_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-sha-json", default=None,
                    help="JSON mapping state_id -> composite state sha256 "
                         "(from verify_states.py on the cluster)")
    ap.add_argument("--out", default=os.path.join(PKG, "contract",
                                                  "LESIONING_V2_RUN_MATRIX.json"))
    a = ap.parse_args(argv)
    state_sha = json.load(open(a.state_sha_json)) if a.state_sha_json else {}
    m = build(state_sha)
    m["state_sha_resolved"] = True
    m["state_sha_source"] = ("cluster verify_states.py" if state_sha
                             else "frozen identities (deterministic)")
    m["frozen_identity"] = {k: dict(v) for k, v in FROZEN_IDENTITY.items()}
    m["composite_state_sha256"] = {s: composite_state_sha256(s)
                                   for s in STATE_ORDER}
    m["matrix_sha256"] = matrix_sha256(m)
    json.dump(m, open(a.out, "w"), indent=1)
    print(f"cells={m['n_cells']} lesion={m['n_lesion_cells']} "
          f"intact={m['n_intact_cells']}")
    print(f"state_sha_resolved={m['state_sha_resolved']}")
    print(f"matrix_sha256={m['matrix_sha256']}")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
