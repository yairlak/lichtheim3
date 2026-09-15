"""Experiments 1 and 2 of the GATING / ROUTE DIAGNOSTICS workstream.

Executes the preregistered contract in
`paper_programme/gating_route_diagnostics/EXPERIMENT_CONTRACT.md`.

NO TRAINING.  NO TUNING.  Source checkpoints are opened read-only and their
SHA256 is re-verified after every state; the run aborts if one moved.

Model reconstruction is NOT reimplemented: it reuses
`frozen_head_probe.build_trainer` and `frozen_head_probe._isolated_model`
verbatim, which is exactly how the Phase-8 batteries built the source and
repaired models (`ceiling_source_completion.cmd_battery`).  The repaired state
produced here is therefore the same object the frozen witness metrics describe.

Usage:
    python3 scripts/gating_diagnostics/run_gate_route_audit.py --smoke
    python3 scripts/gating_diagnostics/run_gate_route_audit.py --state-id ALL
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gating_diagnostics import ROUTES, collect_item_level          # noqa: E402
from gating_diagnostics.analysis import summarize_state            # noqa: E402

REPO = "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3"
GLOVE = os.path.join(ROOT, "data", "glove.6B.300d.txt")
DEFAULT_MANIFEST = os.path.join(
    ROOT, "paper_programme", "gating_route_diagnostics", "checkpoint_manifest.tsv")
DEFAULT_OUT = os.path.join(ROOT, "paper_programme", "gating_route_diagnostics")
BOOTSTRAP_SEED = 20260915          # frozen in the contract, §6.3


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_manifest(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_state(row: dict, device: str = "cpu"):
    """Reconstruct one manifest state.  Returns (trainer, model, provenance)."""
    from scripts.naming_comprehension.frozen_head_probe import (
        build_trainer, _isolated_model)

    ckpt = os.path.join(REPO, row["artifact_path"])
    before = sha256_file(ckpt)
    if before != row["artifact_sha256"]:
        raise RuntimeError(
            f'HARD STOP: {row["state_id"]} checkpoint hash {before} != '
            f'manifest {row["artifact_sha256"]}')

    tr, ckd = build_trainer(ckpt, device, GLOVE)
    prov = {
        "state_id": row["state_id"], "arm": row["arm"],
        "checkpoint": ckpt, "checkpoint_sha256": before,
        "global_step": int(ckd.get("global_step", -1)),
        "seed": int(ckd.get("seed", -1)),
        "source_u": int(row["source_u"]),
        "gating_config": ckd["config"]["gating"],
        "lexicon_file_sha256": ckd.get("lexicon_file_sha256"),
        "head": None, "head_sha256": None,
    }

    if row["arm"] == "SOURCE_PRE_REPAIR":
        model = tr.model
    else:
        head_path = os.path.join(REPO, row["applies_head_path"])
        hsha = sha256_file(head_path)
        if hsha != row["applies_head_sha256"]:
            raise RuntimeError(f'HARD STOP: {row["state_id"]} head hash mismatch')
        h = torch.load(head_path, map_location="cpu", weights_only=False)
        if set(h["state"]) != {"2.weight", "2.bias"}:
            raise RuntimeError("derived head must carry exactly the final-layer parameters")
        model = _isolated_model(tr, "p_last_hinge", h["state"])
        prov["head"], prov["head_sha256"] = head_path, hsha
        prov["head_label"], prov["head_arm"] = h.get("label"), h.get("arm")

    model.eval()
    return tr, model, prov, before, ckpt


def assert_determinism(model, tr, device: str, n: int = 64) -> float:
    """Contract §5: re-decode one batch and require bitwise equality."""
    idx = list(range(min(n, len(tr.entries))))
    a = collect_item_level(model, tr.vocab, tr.entries, idx, device,
                           batch_size=n, free_ar=False)
    b = collect_item_level(model, tr.vocab, tr.entries, idx, device,
                           batch_size=n, free_ar=False)
    for ra, rb in zip(a, b):
        for k in ("gate", "canonical_full_exact", "canonical_wm_exact",
                  "canonical_ltm_exact", "canonical_fixed05_exact"):
            if ra[k] != rb[k]:
                raise RuntimeError(f"HARD STOP: non-deterministic decode at {ra['word']} ({k})")
    return max(abs(ra["gate"] - rb["gate"]) for ra, rb in zip(a, b))


def run_state(row: dict, out_dir: str, device: str, limit: Optional[int],
              batch_size: int, free_ar: bool) -> dict:
    sid = row["state_id"]
    print(f"\n=== {sid} [{row['arm']}] {row['witness_label']} ===", flush=True)
    tr, model, prov, before, ckpt = build_state(row, device)

    n_total = len(tr.entries)
    indices = list(range(n_total if limit is None else min(limit, n_total)))
    prov.update({
        "repetition_population_n": n_total,
        "evaluated_n": len(indices),
        "item_order_sha256": hashlib.sha256(
            ",".join(e.word for e in tr.entries[:len(indices)]).encode()).hexdigest(),
        "routes": list(ROUTES), "free_ar": free_ar,
        "determinism_max_gate_dev": assert_determinism(model, tr, device),
        "torch": torch.__version__,
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })

    def progress(done: int, total: int) -> None:
        print(f"  … {done}/{total}", end="\r", flush=True)

    rows = collect_item_level(model, tr.vocab, tr.entries, indices, device,
                              batch_size=batch_size, free_ar=free_ar,
                              progress=progress)
    print()

    after = sha256_file(ckpt)
    if after != before:
        raise RuntimeError(f"HARD STOP: source checkpoint mutated: {ckpt}")
    prov["source_unchanged"] = True

    shard_dir = os.path.join(out_dir, "figure_source_data")
    os.makedirs(shard_dir, exist_ok=True)
    shard = os.path.join(shard_dir, f"item_level_{sid}.tsv")
    with open(shard, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        wr.writeheader()
        wr.writerows(rows)

    summary = summarize_state(rows, seed=BOOTSTRAP_SEED)
    summary["provenance"] = prov
    summary["item_level_shard"] = os.path.relpath(shard, out_dir)
    summary["item_level_shard_sha256"] = sha256_file(shard)
    print(f"  wrote {shard}  ({len(rows)} items)")
    print(f"  gate mean={summary['gate']['mean']:.6f} sd={summary['gate']['sd']:.6f}  "
          f"cells={ {k: v['n'] for k, v in summary['competence'].items()} }")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--state-id", default="ALL",
                    help="a state_id from the manifest, or ALL")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N items (diagnostic runs)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-free-ar", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="200 items on one state; validates the pipeline end to end")
    a = ap.parse_args(argv)

    if a.smoke:
        a.limit = a.limit or 200
        if a.state_id == "ALL":
            a.state_id = "W3_SRC"

    manifest = load_manifest(a.manifest)
    states = manifest if a.state_id == "ALL" else \
        [r for r in manifest if r["state_id"] == a.state_id]
    if not states:
        print(f"no such state_id: {a.state_id}", file=sys.stderr)
        return 2

    os.makedirs(a.out_dir, exist_ok=True)
    summaries: Dict[str, dict] = {}
    for row in states:
        summaries[row["state_id"]] = run_state(
            row, a.out_dir, a.device, a.limit, a.batch_size, not a.no_free_ar)

    name = "summary_metrics.json" if (a.state_id == "ALL" and not a.smoke) \
        else f"summary_metrics_{a.state_id}{'_smoke' if a.smoke else ''}.json"
    out = os.path.join(a.out_dir, name)
    with open(out, "w") as f:
        json.dump({
            "contract": "paper_programme/gating_route_diagnostics/EXPERIMENT_CONTRACT.md",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "smoke": bool(a.smoke),
            "limit": a.limit,
            "states": summaries,
        }, f, indent=1)
    print(f"\nwrote {out}")
    if a.smoke:
        print("\nSMOKE RUN — NOT A RESULT. Inspect, then run the full pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
