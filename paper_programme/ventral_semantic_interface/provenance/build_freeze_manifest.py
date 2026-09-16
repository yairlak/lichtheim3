#!/usr/bin/env python3
"""Build IMPLEMENTATION_MANIFEST.json and SHA256SUMS for the contract freeze.

Run from the worktree root AFTER `git add` of the freeze files.  The recorded git diff
is the staged diff against the audited base, excluding the two files this script
writes (they cannot contain their own hashes)."""
import hashlib
import json
import os
import subprocess

ROOT = os.getcwd()
PROG = "paper_programme/ventral_semantic_interface"
BASE = "79f4e5bd94a9c1f82594f4050028b39c220b82b0"
OUT_MANIFEST = f"{PROG}/IMPLEMENTATION_MANIFEST.json"
OUT_SUMS = f"{PROG}/SHA256SUMS"
EXCL = [f":!{OUT_MANIFEST}", f":!{OUT_SUMS}"]


def sh(*a):
    return subprocess.run(list(a), check=True, capture_output=True).stdout


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


files = sorted(sh("git", "diff", "--cached", "--name-only", BASE, "--", ".", *EXCL).decode().split())
patch = sh("git", "diff", "--cached", "--binary", BASE, "--", ".", *EXCL)
cfg = json.load(open(f"{PROG}/ventral_interface_frozen_config.json"))
manifest = {
    "workstream": "LICHTHEIM3 VENTRAL SEMANTIC INTERFACE — FROZEN FACTORIZATION DIAGNOSTIC",
    "branch": sh("git", "branch", "--show-current").decode().strip(),
    "audited_base_commit": BASE,
    "contract_status": "FROZEN",
    "scientific_execution": "NOT_RUN",
    "key_documents_sha256": {p: sha(p) for p in [
        f"{PROG}/VENTRAL_INTERFACE_LINEAGE.md",
        f"{PROG}/VENTRAL_INTERFACE_EXPERIMENT_CONTRACT.md",
        f"{PROG}/ventral_interface_frozen_config.json",
        f"{PROG}/provenance/preflight_report.json",
        "scripts/ventral_interface/run_ventral_interface_factorization.py",
        "tests/test_ventral_interface.py"]},
    "new_files_sha256": {p: sha(p) for p in files},
    "code_closure_sha256": {p: sha(p) for p in cfg["code_closure"]},
    "git_diff_vs_base": {
        "command": f"git diff --cached --binary {BASE} -- . ':!{OUT_MANIFEST}' ':!{OUT_SUMS}'",
        "name_status": sh("git", "diff", "--cached", "--name-status", BASE, "--", ".", *EXCL).decode().splitlines(),
        "stat": sh("git", "diff", "--cached", "--stat", BASE, "--", ".", *EXCL).decode().splitlines(),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "modified_preexisting_files": [l.split("\t", 1)[1] for l in sh(
            "git", "diff", "--cached", "--name-status", BASE, "--", ".", *EXCL).decode().splitlines()
            if not l.startswith("A")],
    },
    "external_artifacts_sha256": {
        "source_W3": "a5f21de9edb4b2090cb144c666ca4bd8e1f0572333000facb13f6c51cfaad76c",
        "source_W4": "0657f41030e54eac11816116af6c76b66d597c393fc72579a3755401c9dc79f3",
        "head_W3": "8865ba9539e4a445ffc8847a71be698779a98478aa5e6a1bce2393f3de7afcfc",
        "head_W4": "724ed4c6a84fba07b6d9f11d535b8a0eb726daff21b9def9a5e60d9c7b972d03",
        "glove": cfg["data"]["glove_sha256"],
        "archival_copies_dir": "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3/archives/ventral_interface_sources_20260916",
    },
}
with open(OUT_MANIFEST, "w") as f:
    json.dump(manifest, f, indent=1, sort_keys=True)
    f.write("\n")
sums = sorted(files + [OUT_MANIFEST])
with open(OUT_SUMS, "w") as f:
    for p in sums:
        f.write(f"{sha(p)}  {p}\n")
print(f"manifest {sha(OUT_MANIFEST)}  files={len(sums)}")
