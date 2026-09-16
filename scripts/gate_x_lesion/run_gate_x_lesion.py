"""GATE x LESION / RECOVERY — the preregistered experiment runner.

Executes `paper_programme/gate_x_lesion_recovery/GATE_X_LESION_EXPERIMENT_CONTRACT.md`.

NO TRAINING.  NO TUNING.  NO CONNECTIVITY DAMAGE.  Source artifacts are opened read-only
and their SHA256 is re-verified; the in-memory state_dict hash is compared before and
after every lesion context.

Execution safety, layered so that no single mistake can produce a fake result:

  * `--smoke` writes ONLY under a path containing NOT_SCIENTIFIC_RESULT, with a
    _SMOKE_TEST_ONLY filename marker and in-file metadata saying so;
  * `--limit` is REFUSED without `--smoke` (a truncated pass is a diagnostic, never a
    result — the contract fixes the population as all 29,571 canonical real words);
  * non-zero severity is REFUSED without `--smoke` unless `--i-have-central-go` is
    passed, which CENTRAL alone supplies after final go/no-go;
  * the scientific output namespace is disjoint from the quarantine namespace, and a
    smoke run that would write outside quarantine hard-stops.

Usage (quarantined validation — what this task is permitted to run):
    python3 scripts/gate_x_lesion/run_gate_x_lesion.py --smoke

Usage (scientific execution — PREPARED, NOT RUN, requires CENTRAL go):
    see EXECUTION_COMMAND in the handoff.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gate_x_lesion.evaluate import collect_item_level_lesioned        # noqa: E402
from gate_x_lesion.hooks import lesioned_route, state_dict_sha256     # noqa: E402
from gate_x_lesion.noise import EpsilonCache, FROZEN_LAMBDAS          # noqa: E402
from gate_x_lesion.sd import measure_intact_sd                        # noqa: E402
from gate_x_lesion.targets import FROZEN_ROUTES, assert_site_compatible  # noqa: E402
from scripts.gating_diagnostics.run_gate_route_audit import (          # noqa: E402
    build_state, load_manifest, sha256_file)

REPO = "/Users/louishayot/MVA/ENS-LSCP/Yair-Lichtheim3"
OUT_BASE = os.path.join(ROOT, "paper_programme", "gate_x_lesion_recovery")
DEFAULT_MANIFEST = os.path.join(OUT_BASE, "checkpoint_manifest.proposed.tsv")
CONDITIONS = os.path.join(OUT_BASE, "gxlr_conditions.frozen.json")

QUARANTINE_DIR = "NOT_SCIENTIFIC_RESULT"
SCIENTIFIC_DIR = "scientific_execution"
SMOKE_SUFFIX = "_SMOKE_TEST_ONLY"
SMOKE_DEFAULT_LIMIT = 24
SMOKE_DEFAULT_SEEDS = (0,)

FROZEN_SEEDS = (0, 1, 2, 3)

QUARANTINE_BANNER = (
    "NOT_SCIENTIFIC_RESULT — quarantined implementation-validation output. This file "
    "was produced on a deliberately non-canonical truncated subset to exercise the "
    "code path. It is NOT a scientific result, MUST NOT be reported, cited, or "
    "compared against any frozen record, and cannot be promoted to one."
)


# ------------------------------------------------------------------ safety gates

def resolve_outputs(smoke: bool) -> str:
    return os.path.join(OUT_BASE, QUARANTINE_DIR if smoke else SCIENTIFIC_DIR)


def assert_full_population(limit: Optional[int], smoke: bool) -> None:
    if limit is not None and not smoke:
        raise RuntimeError(
            f"HARD STOP: --limit {limit} without --smoke. A truncated pass is a "
            "diagnostic, never a scientific result: the contract fixes the population "
            "as all 29,571 canonical real words (§4), and the paired item-by-item "
            "comparisons are defined over it. --limit is reserved for quarantined use.")


def assert_execution_authorised(lambdas, smoke: bool, central_go: bool) -> None:
    """Refuse canonical scientific lesion execution without explicit CENTRAL go."""
    nonzero = [l for l in lambdas if float(l) != 0.0]
    if nonzero and not smoke and not central_go:
        raise RuntimeError(
            f"HARD STOP: non-zero severity {nonzero} on the full canonical population "
            "is SCIENTIFIC EXECUTION and requires CENTRAL final go/no-go. "
            "GO_FOR_SCIENTIFIC_EXECUTION = NO. Re-run with --smoke for quarantined "
            "validation, or with severity 0 for the authoritative intact null.")


def assert_quarantined(path: str, smoke: bool) -> None:
    if not smoke:
        sci = os.path.realpath(os.path.join(OUT_BASE, SCIENTIFIC_DIR))
        t = os.path.realpath(path)
        if not (t == sci or t.startswith(sci + os.sep)):
            raise RuntimeError(f"HARD STOP: scientific run would write outside "
                               f"{SCIENTIFIC_DIR}/: {t}")
        return
    base = os.path.realpath(os.path.join(OUT_BASE, QUARANTINE_DIR))
    t = os.path.realpath(path)
    if not (t == base or t.startswith(base + os.sep)):
        raise RuntimeError(
            f"HARD STOP: smoke run would write outside {QUARANTINE_DIR}/: {t}")
    if QUARANTINE_DIR not in t:
        raise RuntimeError(f"HARD STOP: quarantine path must contain "
                           f"{QUARANTINE_DIR}: {t}")
    if t.endswith((".json", ".tsv")) and SMOKE_SUFFIX not in os.path.basename(t):
        raise RuntimeError(f"HARD STOP: quarantined artifact must carry the "
                           f"{SMOKE_SUFFIX} marker: {t}")


# ------------------------------------------------------------------------ run

def run_state(row: dict, *, device: str, limit: Optional[int], batch_size: int,
              routes, lambdas, seeds, free_ar: bool, out_dir: str,
              smoke: bool) -> dict:
    sid = row["state_id"]
    label = row["gxlr_label"] or sid
    print(f"\n=== {label} [{sid}] {row['witness_label']} ===", flush=True)

    # Reconstruct via the frozen GATING recipe: base checkpoint + derived head.
    legacy = {"state_id": sid, "arm": row["arm"],
              "artifact_path": row["base_artifact_path"],
              "artifact_sha256": row["base_artifact_sha256"],
              "applies_head_path": row["applies_head_path"],
              "applies_head_sha256": row["applies_head_sha256"],
              "source_u": row["source_u"]}
    tr, model, prov, base_before, ckpt = build_state(legacy, device)

    state_sha = row["state_sha256"]
    recomputed = hashlib.sha256(
        ("gxlr-state-v1|" + row["base_artifact_sha256"] + "|"
         + row["applies_head_sha256"]).encode()).hexdigest()
    if recomputed != state_sha:
        raise RuntimeError(f"HARD STOP: composite state_sha256 mismatch for {sid}")

    for r in routes:
        assert_site_compatible(model, r)

    sd_before = state_dict_sha256(model)

    # --- intact SD, measured on THIS state's intact model, by the frozen recipe ---
    sd_info = measure_intact_sd(model, tr.vocab, tr.entries, routes=routes,
                                device=device)
    for r in routes:
        print(f"  intact SD[{r}] = {sd_info[r]['sd']:.9f} "
              f"(n_values={sd_info[r]['n_values']}, ddof=1)")

    n_total = len(tr.entries)
    indices = list(range(n_total if limit is None else min(limit, n_total)))
    item_order_sha = hashlib.sha256(
        ",".join(tr.entries[i].word for i in indices).encode()).hexdigest()

    # --- intact control (severity 0) ---
    t0 = time.perf_counter()
    intact_rows = collect_item_level_lesioned(
        model, tr.vocab, tr.entries, indices, device, hook=None,
        batch_size=batch_size, free_ar=free_ar)
    intact_wall = time.perf_counter() - t0
    intact_by_item = {r["item_index"]: r for r in intact_rows}
    print(f"  intact control: {len(intact_rows)} items in {intact_wall:.1f}s")

    cells: List[dict] = []
    all_rows: Dict[str, List[dict]] = {"intact": intact_rows}

    for route in routes:
        sd = sd_info[route]["sd"]
        for lam in lambdas:
            if float(lam) == 0.0:
                continue
            for seed in seeds:
                cache = EpsilonCache(state_sha, route, seed,
                                     sd_info[route]["hidden_size"])
                h_before = state_dict_sha256(model)
                t0 = time.perf_counter()
                with lesioned_route(model, route=route, state_sha256=state_sha,
                                    lesion_seed=seed, lam=lam, sd=sd,
                                    cache=cache) as hook:
                    rows = collect_item_level_lesioned(
                        model, tr.vocab, tr.entries, indices, device, hook=hook,
                        batch_size=batch_size, free_ar=free_ar,
                        intact_by_item=intact_by_item)
                wall = time.perf_counter() - t0
                h_after = state_dict_sha256(model)
                if h_after != h_before:
                    raise RuntimeError(
                        f"HARD STOP: state_dict mutated by lesion context "
                        f"{route} lam={lam} seed={seed}")

                key = f"{route}_lam{lam}_seed{seed}"
                all_rows[key] = rows
                cells.append(summarize_cell(rows, state_id=sid, route=route, lam=lam,
                                            seed=seed, sd=sd, wall=wall,
                                            n_hook_calls=hook.n_calls,
                                            n_eta_builds=hook.n_eta_builds,
                                            n_eps_computed=cache.n_computed,
                                            n_eps_served=cache.n_served))
                c = cells[-1]
                print(f"  {route:18s} lam={lam:<5} seed={seed}  "
                      f"disc_canon={c['canonical']['n_discordant_prediction']:<6} "
                      f"nat_acc={c['canonical']['acc_native']:.4f} "
                      f"fix_acc={c['canonical']['acc_fixed05']:.4f} "
                      f"dg={c['max_abs_delta_gate']:.3e} "
                      f"{wall:.1f}s", flush=True)

    sd_after = state_dict_sha256(model)
    if sd_after != sd_before:
        raise RuntimeError(f"HARD STOP: state_dict mutated during {sid}")
    if sha256_file(ckpt) != base_before:
        raise RuntimeError(f"HARD STOP: base checkpoint file mutated: {ckpt}")

    summary = {
        "state_id": sid, "gxlr_label": label,
        "witness_label": row["witness_label"],
        "state_sha256": state_sha,
        "base_artifact_sha256": row["base_artifact_sha256"],
        "applies_head_sha256": row["applies_head_sha256"],
        "state_kind": row["state_kind"],
        "head_localizer": row["head_localizer"],
        "ltm_encoder_mode": getattr(model.ltm.cfg, "ltm_encoder_mode", None),
        "gating_config": prov["gating_config"],
        "repetition_population_n": n_total,
        "evaluated_n": len(indices),
        "item_order_sha256": item_order_sha,
        "intact_sd": sd_info,
        "state_dict_sha256_before": sd_before,
        "state_dict_sha256_after": sd_after,
        "state_dict_unmutated": sd_before == sd_after,
        "base_checkpoint_unmutated": True,
        "intact_control": summarize_cell(intact_rows, state_id=sid, route="NONE",
                                         lam=0.0, seed=None, sd=0.0,
                                         wall=intact_wall),
        "cells": cells,
        "torch": torch.__version__,
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    write_shards(all_rows, out_dir, sid, smoke)
    return summary


def summarize_cell(rows: List[dict], *, state_id: str, route: str, lam: float,
                   seed, sd: float, wall: float, **extra) -> dict:
    out = {"state_id": state_id, "route": route, "lambda": lam, "lesion_seed": seed,
           "site_sd_intact": sd, "n_items": len(rows),
           "wall_seconds": round(wall, 2)}
    out.update(extra)
    for conv in ("canonical", "freear"):
        if f"{conv}_full_exact" not in rows[0]:
            continue
        nat = [r[f"{conv}_full_exact"] for r in rows]
        fix = [r[f"{conv}_fixed05_exact"] for r in rows]
        gained = sum(1 for r in rows if r[f"{conv}_error_introduced_by_fixed05"])
        prevented = sum(1 for r in rows if r[f"{conv}_error_prevented_by_fixed05"])
        preds = [r[f"{conv}_full_predicted"] for r in rows]
        modal = max((preds.count(p) for p in set(preds)), default=0)
        out[conv] = {
            "acc_native": sum(nat) / len(nat),
            "acc_fixed05": sum(fix) / len(fix),
            "acc_wm_isolated": sum(r[f"{conv}_wm_exact"] for r in rows) / len(rows),
            "acc_ltm_isolated": sum(r[f"{conv}_ltm_exact"] for r in rows) / len(rows),
            "n_discordant_prediction": sum(r[f"{conv}_discordant_prediction"] for r in rows),
            "n_discordant_exact": sum(r[f"{conv}_discordant_exact"] for r in rows),
            "errors_introduced_by_fixed05": gained,
            "errors_prevented_by_fixed05": prevented,
            "net_change_in_correct": prevented - gained,
            "modal_prediction_share": modal / len(rows),
            "n_changed_vs_intact": sum(
                r.get("n_route_conv_cells_changed_vs_intact", 0) for r in rows),
        }
    gv = [r["gate"] for r in rows if r.get("gate") is not None]
    cv = [r["c_LTM"] for r in rows if r.get("c_LTM") is not None]
    out["gate_mean"] = sum(gv) / len(gv) if gv else None
    out["c_LTM_mean"] = sum(cv) / len(cv) if cv else None
    dg = [abs(r["delta_gate"]) for r in rows if r.get("delta_gate") is not None]
    dc = [abs(r["delta_c_LTM"]) for r in rows if r.get("delta_c_LTM") is not None]
    out["max_abs_delta_gate"] = max(dg) if dg else 0.0
    out["max_abs_delta_c_LTM"] = max(dc) if dc else 0.0
    return out


def write_shards(all_rows: Dict[str, List[dict]], out_dir: str, sid: str,
                 smoke: bool) -> None:
    shard_dir = os.path.join(out_dir, "item_level")
    assert_quarantined(shard_dir, smoke)
    os.makedirs(shard_dir, exist_ok=True)
    for key, rows in all_rows.items():
        name = f"item_level_{sid}_{key}{SMOKE_SUFFIX if smoke else ''}.tsv"
        p = os.path.join(shard_dir, name)
        assert_quarantined(p, smoke)
        cols: List[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(p, "w", newline="") as f:
            if smoke:
                f.write(f"# {QUARANTINE_BANNER}\n")
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--state-id", default="ALL")
    ap.add_argument("--routes", default=",".join(FROZEN_ROUTES))
    ap.add_argument("--lambdas", default=",".join(str(x) for x in FROZEN_LAMBDAS))
    ap.add_argument("--lesion-seeds", default=",".join(str(x) for x in FROZEN_SEEDS))
    ap.add_argument("--limit", type=int, default=None,
                    help="QUARANTINED USE ONLY: requires --smoke")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-free-ar", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--i-have-central-go", action="store_true",
                    help="CENTRAL final go/no-go. Required for non-zero severity on "
                         "the full canonical population.")
    a = ap.parse_args(argv)

    routes = [r for r in a.routes.split(",") if r]
    lambdas = [float(x) for x in a.lambdas.split(",") if x]
    seeds = [int(x) for x in a.lesion_seeds.split(",") if x]

    bad = [l for l in lambdas if l != 0.0 and l not in FROZEN_LAMBDAS]
    if bad:
        raise RuntimeError(f"HARD STOP: severity {bad} is outside the frozen grid "
                           f"{list(FROZEN_LAMBDAS)}; the contract forbids adding one.")
    bad = [r for r in routes if r not in FROZEN_ROUTES]
    if bad:
        raise RuntimeError(f"HARD STOP: route {bad} outside {list(FROZEN_ROUTES)}")

    assert_full_population(a.limit, bool(a.smoke))
    assert_execution_authorised(lambdas, bool(a.smoke), bool(a.i_have_central_go))

    if a.smoke:
        a.limit = a.limit or SMOKE_DEFAULT_LIMIT
        seeds = seeds if len(seeds) < len(FROZEN_SEEDS) else list(SMOKE_DEFAULT_SEEDS)
        if a.state_id == "ALL":
            a.state_id = "W3_REP"

    rows = [r for r in load_manifest(a.manifest) if r["in_scope"] == "1"]
    states = rows if a.state_id == "ALL" else [r for r in rows
                                               if r["state_id"] == a.state_id]
    if not states:
        print(f"no in-scope state: {a.state_id}", file=sys.stderr)
        return 2

    out_dir = resolve_outputs(bool(a.smoke))
    assert_quarantined(out_dir, bool(a.smoke))
    os.makedirs(out_dir, exist_ok=True)

    summaries = {}
    for row in states:
        summaries[row["state_id"]] = run_state(
            row, device=a.device, limit=a.limit, batch_size=a.batch_size,
            routes=routes, lambdas=lambdas, seeds=seeds,
            free_ar=not a.no_free_ar, out_dir=out_dir, smoke=bool(a.smoke))

    name = (f"summary_{a.state_id}{SMOKE_SUFFIX}.json" if a.smoke
            else f"summary_{a.state_id}.json")
    out = os.path.join(out_dir, name)
    assert_quarantined(out, bool(a.smoke))
    payload = {
        "contract": "paper_programme/gate_x_lesion_recovery/GATE_X_LESION_EXPERIMENT_CONTRACT.md",
        "conditions": "paper_programme/gate_x_lesion_recovery/gxlr_conditions.frozen.json",
        "conditions_sha256": sha256_file(CONDITIONS),
        "smoke": bool(a.smoke),
        "TEST_ONLY": bool(a.smoke),
        "NOT_SCIENTIFIC_RESULT": bool(a.smoke),
        "quarantine_notice": QUARANTINE_BANNER if a.smoke else None,
        "limit": a.limit,
        "routes": routes, "lambdas": lambdas, "lesion_seeds": seeds,
        "states": summaries,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {out}")
    if a.smoke:
        print(f"\n*** {QUARANTINE_BANNER} ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
