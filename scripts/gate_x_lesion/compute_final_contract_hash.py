"""Compute FINAL_CONTRACT_HASH over the scientific-contract inputs.

DERIVATION — exact and reproducible
-----------------------------------
1. Take the fixed, explicitly enumerated list of contract INPUT paths in
   `CONTRACT_INPUTS` below (repo-relative, POSIX separators).
2. Sort that list by byte value of the path string (`sorted()` on str), so the
   ordering is canonical and independent of filesystem order.
3. For each path in that order, compute the SHA256 of the file's exact bytes.
4. Build one manifest text in which each entry is exactly

       "<sha256>  <path>\\n"

   (two spaces, `shasum`-style), concatenated in the sorted order.
5. Prepend the domain-separation line `"GXLR-final-contract-v1\\n"`.
6. FINAL_CONTRACT_HASH = SHA256 of that manifest text, UTF-8 encoded.

The manifest text is written to `FINAL_CONTRACT_MANIFEST.txt` so the hash can be
re-derived and audited by hand.

DELIBERATELY EXCLUDED — so the hash is stable across runs
----------------------------------------------------------
* future scientific result files (`scientific_execution/**`);
* epsilon realizations and any lesion output;
* quarantined smoke artifacts (`NOT_SCIENTIFIC_RESULT/**`);
* timestamps, run_utc, wall times, hostnames, absolute paths;
* anything under `.git/`.

A missing input is a hard error: the hash is never computed over a partial set.

    python3 scripts/gate_x_lesion/compute_final_contract_hash.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "paper_programme/gate_x_lesion_recovery"

DOMAIN = "GXLR-final-contract-v1\n"

#: Every input required to EXECUTE the experiment. Enumerated explicitly — never
#: globbed — so that adding a file cannot silently change the contract identity.
CONTRACT_INPUTS = [
    # --- original design freeze artifacts ---
    f"{BASE}/GATE_X_LESION_EXPERIMENT_CONTRACT.md",
    f"{BASE}/HISTORICAL_LESION_OPERATOR_AUDIT.md",
    f"{BASE}/LIVE_CODE_AUDIT.md",
    f"{BASE}/GATE_X_LESION_CENTRAL_STEERING_HANDOFF.md",
    f"{BASE}/gxlr_conditions.frozen.json",
    # --- design amendments ---
    f"{BASE}/gxlr_conditions.amendment1.json",
    f"{BASE}/CENTRAL_FINAL_EXECUTION_READINESS_HANDOFF.md",
    f"{BASE}/IMPLEMENTATION_NOTES.md",
    # --- final rule freeze ---
    f"{BASE}/FINAL_RULE_FREEZE.md",
    f"{BASE}/gxlr_conditions.final.json",
    # --- W3/W4 provenance manifest ---
    f"{BASE}/checkpoint_manifest.proposed.tsv",
    # --- final O3/O4 rule representation (the executable rules themselves) ---
    "gate_x_lesion/validity.py",
    "gate_x_lesion/robustness.py",
    "gate_x_lesion/classify.py",
    "gate_x_lesion/outcomes.py",
    # --- identity, noise and site semantics the rules depend on ---
    "gate_x_lesion/identity.py",
    "gate_x_lesion/noise.py",
    "gate_x_lesion/targets.py",
    "gate_x_lesion/hooks.py",
    "gate_x_lesion/sd.py",
    "gate_x_lesion/evaluate.py",
    # --- expected-output contract ---
    f"{BASE}/EXPECTED_OUTPUT_MANIFEST.json",
    # --- execution script / version identity ---
    "scripts/gate_x_lesion/run_gate_x_lesion.py",
    "scripts/gate_x_lesion/emit_final_conditions.py",
    "scripts/gate_x_lesion/validate_output_manifest.py",
]

MANIFEST_OUT = os.path.join(ROOT, BASE, "FINAL_CONTRACT_MANIFEST.txt")
HASH_OUT = os.path.join(ROOT, BASE, "FINAL_CONTRACT_HASH.txt")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_manifest() -> str:
    missing = [p for p in CONTRACT_INPUTS
               if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        raise SystemExit(
            "HARD STOP: contract input(s) missing; FINAL_CONTRACT_HASH is never "
            "computed over a partial set:\n  " + "\n  ".join(missing))
    lines = [DOMAIN]
    for path in sorted(CONTRACT_INPUTS):
        lines.append(f"{sha256_file(os.path.join(ROOT, path))}  {path}\n")
    return "".join(lines)


def final_contract_hash(manifest: str) -> str:
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="recompute and verify against the recorded hash")
    a = ap.parse_args()

    manifest = build_manifest()
    digest = final_contract_hash(manifest)

    if a.check:
        if not os.path.isfile(HASH_OUT):
            raise SystemExit("HARD STOP: no recorded FINAL_CONTRACT_HASH to check")
        recorded = open(HASH_OUT).read().strip().split()[0]
        if recorded != digest:
            raise SystemExit(
                f"HARD STOP: FINAL_CONTRACT_HASH mismatch\n"
                f"  recorded   {recorded}\n  recomputed {digest}")
        print(f"FINAL_CONTRACT_HASH OK  {digest}")
        return 0

    with open(MANIFEST_OUT, "w") as f:
        f.write(manifest)
    with open(HASH_OUT, "w") as f:
        f.write(digest + "  FINAL_CONTRACT_HASH\n")

    print(manifest, end="")
    print(f"\nFINAL_CONTRACT_HASH = {digest}")
    print(f"wrote {MANIFEST_OUT}")
    print(f"wrote {HASH_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
