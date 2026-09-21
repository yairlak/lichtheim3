#!/usr/bin/env python
"""READ-ONLY generator for the 12-control continuity manifest.

Writes ONLY to --out, which must lie OUTSIDE the scientific result namespace.
Nothing inside the result root is created, modified or deleted.

    python generate_continuity_manifest.py \
        --result-root $SCRATCH/l3_lesion_v2_results \
        --out $WORK/l3_lesion_v2_control/LESIONING_V2_INTACT_CONTROL_CONTINUITY_MANIFEST.json
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

from paper_programme.lesioning_v2.execution import continuity, preflight  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Read-only continuity manifest.")
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    root = os.path.abspath(a.result_root)
    out = os.path.abspath(a.out)
    if out.startswith(root + os.sep) or out == root:
        print("REFUSED: --out must lie OUTSIDE the result namespace",
              file=sys.stderr)
        return 2
    if os.path.exists(out):
        print(f"REFUSED: refusing to overwrite {out}", file=sys.stderr)
        return 2

    before = sorted(os.walk(root).__next__()[1]) if os.path.isdir(root) else None
    try:
        m = continuity.generate(root, preflight.load_matrix())
    except continuity.ContinuityError as e:
        print(f"BLOCKED {e}", file=sys.stderr)
        return 3

    after = sorted(os.walk(root).__next__()[1])
    if before != after:
        print("REFUSED: result namespace changed during a read-only scan",
              file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
    print(f"EXISTING_K0_COMPLETE={m['n_complete_k0_cells']}/12")
    print(f"NONZERO_LESION_CELL_FINALIZED={m['n_complete_nonzero_cells']}")
    print(f"manifest_sha256={m['manifest_sha256']}")
    print(f"written (read-only scan): {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
