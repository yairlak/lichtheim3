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

from gate_x_lesion.assemble import build_summary                      # noqa: E402
from gate_x_lesion.evaluate import collect_item_level_lesioned        # noqa: E402
from gate_x_lesion.hooks import lesioned_route, state_dict_sha256     # noqa: E402
from gate_x_lesion.identity import verify_manifest_state_identity    # noqa: E402
from gate_x_lesion.noise import (EpsilonCache, EPS_DOMAIN,            # noqa: E402
                                 FROZEN_LAMBDAS, IDENTITY_FIELDS)
from gate_x_lesion.identity import STATE_IDENTITY_DOMAIN             # noqa: E402
from gate_x_lesion.sd import (SD_BATCH_SIZE, SD_DDOF, SD_N_ITEMS,     # noqa: E402
                              SD_SAMPLE_SEED)
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
#: Smoke covers the FULL frozen grid (2 witnesses x 4 lesion seeds x 2 routes x
#: 3 lambdas x 2 conventions) on a deliberately tiny 24-item non-canonical
#: population.  That way the quarantined run exercises exactly the scientific code
#: path -- including the frozen O-3 8-block grid and the frozen O-4 two-witness /
#: four-seed replication unit -- and NO rule has to be relaxed for smoke.
SMOKE_DEFAULT_SEEDS = (0, 1, 2, 3)

FROZEN_SEEDS = (0, 1, 2, 3)

#: Frozen scientific expectations, mirrored from EXPECTED_OUTPUT_MANIFEST.json.
SCIENTIFIC_EXPECTED_SHARDS = 50        # 2 witnesses x (1 intact + 2x3x4 lesion)
SCIENTIFIC_EXPECTED_VALIDITY = 6       # 2 routes x 3 lambdas
SCIENTIFIC_EXPECTED_PAIRED = 96        # 2 x 4 x 2 x 3 x 2

#: F-8 (closure pass §6).  Reproducibility freeze, not a scientific hypothesis.
#: Both encoders pack their input, so the packed-GRU reduction order depends on batch
#: composition: at batch 512 the GATE moves by 1-2 ulp on 15/2048 items (max 1.19e-07)
#: while changing no prediction.  At 256 -- the value the frozen GATING record was
#: produced with -- the gate is bitwise identical.  Scientific mode therefore fails
#: closed on any other value, absent a formal CENTRAL amendment.
SCIENTIFIC_BATCH_SIZE = 256

#: The frozen rule set's identity. Scientific execution must name it explicitly, so
#: a run can never be launched against a contract the operator has not pinned.
FINAL_CONTRACT_HASH_FILE = os.path.join(OUT_BASE, "FINAL_CONTRACT_HASH.txt")

QUARANTINE_BANNER = (
    "NOT_SCIENTIFIC_RESULT — quarantined implementation-validation output. This file "
    "was produced on a deliberately non-canonical truncated subset to exercise the "
    "code path. It is NOT a scientific result, MUST NOT be reported, cited, or "
    "compared against any frozen record, and cannot be promoted to one."
)


# ------------------------------------------------------------------ safety gates

def _git(*args) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "-C", ROOT, *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "UNKNOWN"


def _git_head() -> str:
    return _git("rev-parse", "HEAD")


def _git_branch() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


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


def assert_scientific_batch_size(batch_size: int, smoke: bool) -> None:
    """F-8: scientific mode runs at batch 256 or not at all."""
    if smoke:
        return
    if int(batch_size) != SCIENTIFIC_BATCH_SIZE:
        raise RuntimeError(
            f"HARD STOP: --batch-size {batch_size} in scientific mode. F-8 pins the "
            f"scientific batch size at {SCIENTIFIC_BATCH_SIZE}, the value the frozen "
            f"GATING record was produced with: both encoders pack their input, so a "
            f"different batch size perturbs the packed-GRU reduction order and moves "
            f"the reported gate by 1-2 ulp. Changing it requires a formal CENTRAL "
            f"amendment, not a flag.")


def recorded_final_contract_hash() -> Optional[str]:
    if not os.path.isfile(FINAL_CONTRACT_HASH_FILE):
        return None
    return open(FINAL_CONTRACT_HASH_FILE).read().strip().split()[0]


def assert_final_contract_hash(supplied: Optional[str], smoke: bool) -> Optional[str]:
    """Scientific mode must pin the frozen rule set by hash, and it must match.

    Recomputed from the contract inputs at run time, so an edited rule file, an
    edited manifest or a stale pin all fail closed before any model is loaded.
    """
    if smoke:
        return None
    from scripts.gate_x_lesion.compute_final_contract_hash import (
        build_manifest, final_contract_hash)
    actual = final_contract_hash(build_manifest())
    recorded = recorded_final_contract_hash()
    if recorded is not None and recorded != actual:
        raise RuntimeError(
            f"HARD STOP: the frozen contract has changed since it was recorded.\n"
            f"  recorded   {recorded}\n  recomputed {actual}\n"
            f"Re-freeze the contract via a formal amendment before executing.")
    if supplied is None:
        raise RuntimeError(
            f"HARD STOP: scientific execution must pin the frozen rule set: pass "
            f"--final-contract-hash {actual}")
    if supplied != actual:
        raise RuntimeError(
            f"HARD STOP: --final-contract-hash {supplied} does not match the "
            f"contract as it stands ({actual}).")
    return actual


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

    # O-1: RECONSTRUCTED_STATE_SHA256 -- resolved and verified BEFORE any lesion,
    # and hoisted out of the route/lambda/seed loops.  A pure function of the two
    # artifact file digests; no runtime state can reach it.
    state_sha = verify_manifest_state_identity(row)

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
    audit: List[dict] = []
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
                # Structural invariants recorded (not raised) so the frozen O-3
                # module is the single place that decides ABORT semantics.
                shared_ok = (h_after == h_before)
                ckpt_ok = (sha256_file(ckpt) == base_before)
                # Exactly one eta build per bound batch: a rebuild mid-batch would
                # mean NATIVE and FIXED05 could see different matched tensors.
                n_batches = -(-len(indices) // batch_size)
                tensors_ok = (hook.n_eta_builds == min(n_batches, hook.n_binds)
                              and hook.n_eta_builds == hook.n_binds)

                key = f"{route}_lam{lam}_seed{seed}"
                all_rows[key] = rows
                audit.extend(collect_item_audit(rows, state_id=sid, route=route,
                                                lam=lam, seed=seed))
                cells.append(summarize_cell(rows, state_id=sid, route=route, lam=lam,
                                            seed=seed, sd=sd, wall=wall,
                                            n_hook_calls=hook.n_calls,
                                            n_eta_builds=hook.n_eta_builds,
                                            n_eps_computed=cache.n_computed,
                                            n_eps_served=cache.n_served,
                                            shared_params_unmutated=shared_ok,
                                            checkpoint_identity_restored=ckpt_ok,
                                            matched_lesion_tensors_identical=tensors_ok))
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
        "reconstructed_state_sha256": state_sha,
        "state_sha256": state_sha,
        "base_artifact_path": row["base_artifact_path"],
        "base_artifact_sha256": row["base_artifact_sha256"],
        "applies_head_path": row["applies_head_path"],
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
        "applied_head_unmutated": sha256_file(
            os.path.join(REPO, row["applies_head_path"])) == row["applies_head_sha256"],
        "torch": torch.__version__,
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    shard_paths = write_shards(all_rows, out_dir, sid, smoke)
    return summary, audit, shard_paths


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
        n_wm = sum(r[f"{conv}_wm_exact"] for r in rows)
        n_ltm = sum(r[f"{conv}_ltm_exact"] for r in rows)
        out[conv] = {
            # Integer counts are the PREFERRED representation: the frozen rules
            # consume them through exact rational arithmetic, so no accuracy
            # boundary is decided by a float subtraction.
            "n_correct_native": sum(nat),
            "n_correct_fixed05": sum(fix),
            "n_correct_wm_isolated": n_wm,
            "n_correct_ltm_isolated": n_ltm,
            "wm_changed_vs_intact": sum(
                r.get(f"{conv}_wm_changed_vs_intact", 0) for r in rows),
            "ltm_changed_vs_intact": sum(
                r.get(f"{conv}_ltm_changed_vs_intact", 0) for r in rows),
            "acc_native": sum(nat) / len(nat),
            "acc_fixed05": sum(fix) / len(fix),
            "acc_wm_isolated": n_wm / len(rows),
            "acc_ltm_isolated": n_ltm / len(rows),
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


def collect_item_audit(rows: List[dict], *, state_id: str, route: str, lam: float,
                       seed) -> List[dict]:
    """Preregistered discordant-item audit ONLY. No ranking, no selection."""
    out: List[dict] = []
    for r in rows:
        for conv in ("canonical", "freear"):
            if not r.get(f"{conv}_discordant_prediction"):
                continue
            out.append({
                "state_id": state_id, "route": route, "lambda": float(lam),
                "lesion_seed": seed, "convention": conv,
                "item_index": r["item_index"], "word": r["word"],
                "target_phonemes": r["target_phonemes"], "length": r["length"],
                "zipf_approx": r["zipf_approx"],
                "gate": r["gate"], "c_LTM": r["c_LTM"],
                "native_predicted": r[f"{conv}_full_predicted"],
                "fixed05_predicted": r[f"{conv}_fixed05_predicted"],
                "native_exact": r[f"{conv}_full_exact"],
                "fixed05_exact": r[f"{conv}_fixed05_exact"],
                "trace_native": r.get(f"{conv}_trace_native", ""),
                "trace_fixed05": r.get(f"{conv}_trace_fixed05", ""),
            })
    return out


def write_shards(all_rows: Dict[str, List[dict]], out_dir: str, sid: str,
                 smoke: bool) -> List[str]:
    shard_dir = os.path.join(out_dir, "item_level")
    assert_quarantined(shard_dir, smoke)
    os.makedirs(shard_dir, exist_ok=True)
    written: List[str] = []
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
        written.append(p)
    return written


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
    ap.add_argument("--final-contract-hash", default=None,
                    help="FINAL_CONTRACT_HASH of the frozen rule set. Required for "
                         "scientific execution; verified against a recomputation.")
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
    assert_scientific_batch_size(a.batch_size, bool(a.smoke))
    contract_hash = assert_final_contract_hash(a.final_contract_hash, bool(a.smoke))

    if a.smoke:
        a.limit = a.limit or SMOKE_DEFAULT_LIMIT
        seeds = list(SMOKE_DEFAULT_SEEDS)
        # state_id stays ALL: both witnesses are needed for the frozen O-4 rule.

    rows = [r for r in load_manifest(a.manifest) if r["in_scope"] == "1"]
    states = rows if a.state_id == "ALL" else [r for r in rows
                                               if r["state_id"] == a.state_id]
    if not states:
        print(f"no in-scope state: {a.state_id}", file=sys.stderr)
        return 2

    out_dir = resolve_outputs(bool(a.smoke))
    assert_quarantined(out_dir, bool(a.smoke))
    os.makedirs(out_dir, exist_ok=True)

    summaries, audit_all, shard_all = {}, [], []
    for row in states:
        st, audit, shards = run_state(
            row, device=a.device, limit=a.limit, batch_size=a.batch_size,
            routes=routes, lambdas=lambdas, seeds=seeds,
            free_ar=not a.no_free_ar, out_dir=out_dir, smoke=bool(a.smoke))
        summaries[row["state_id"]] = st
        audit_all.extend(audit)
        shard_all.extend(shards)

    name = (f"summary_{a.state_id}{SMOKE_SUFFIX}.json" if a.smoke
            else f"summary_{a.state_id}.json")
    out = os.path.join(out_dir, name)
    assert_quarantined(out, bool(a.smoke))

    # ---- assemble the frozen eight-section package -------------------------
    from scripts.gate_x_lesion.compute_final_contract_hash import (
        HASH_OUT, MANIFEST_OUT, build_manifest, final_contract_hash)
    manifest_text = build_manifest()
    cfg = {
        "final_contract_hash": contract_hash or final_contract_hash(manifest_text),
        "final_contract_manifest_sha256": hashlib.sha256(
            manifest_text.encode("utf-8")).hexdigest(),
        "code_commit": _git_head(),
        "branch": _git_branch(),
        "runner_identity": "scripts/gate_x_lesion/run_gate_x_lesion.py",
        "batch_size": a.batch_size,
        "population_n": (list(summaries.values())[0]["repetition_population_n"]
                         if not a.smoke else list(summaries.values())[0]["evaluated_n"]),
        "routes": routes, "lambdas": lambdas, "lesion_seeds": seeds,
        "sd_definition": "torch.Tensor.std() of flattened pooled intact activations "
                         "(the 'std' column, NOT 'rms')",
        "sd_ddof": SD_DDOF, "sd_n_items": SD_N_ITEMS,
        "sd_sample_seed": SD_SAMPLE_SEED, "sd_batch_size": SD_BATCH_SIZE,
        "rng_identity_fields": list(IDENTITY_FIELDS),
        "rng_identity_version": EPS_DOMAIN.decode("ascii"),
        "eps_domain": EPS_DOMAIN.decode("ascii"),
        "state_identity_domain": STATE_IDENTITY_DOMAIN,
        "torch_version": torch.__version__,
    }
    package = build_summary(
        state_summaries=list(summaries.values()), audit_records=audit_all,
        shard_paths=shard_all, cfg=cfg,
        expected_shards=SCIENTIFIC_EXPECTED_SHARDS,
        expected_validity=SCIENTIFIC_EXPECTED_VALIDITY,
        expected_paired=SCIENTIFIC_EXPECTED_PAIRED,
        shard_sha256={os.path.basename(p): sha256_file(p) for p in shard_all},
        # ALWAYS strict: the full 2x4 block grid is mandatory and there is no
        # denominator shrinking. Smoke covers the full grid at tiny N, so this is
        # never relaxed -- not even for quarantined validation.
        require_full_grid=True)

    package["contract"] = "paper_programme/gate_x_lesion_recovery/FINAL_RULE_FREEZE.md"
    package["conditions"] = "paper_programme/gate_x_lesion_recovery/gxlr_conditions.final.json"
    package["conditions_sha256"] = sha256_file(CONDITIONS)
    package["smoke"] = bool(a.smoke)
    package["TEST_ONLY"] = bool(a.smoke)
    package["NOT_SCIENTIFIC_RESULT"] = bool(a.smoke)
    package["quarantine_notice"] = QUARANTINE_BANNER if a.smoke else None
    if a.smoke:
        # The summary is serialized with sort_keys=True for determinism, which would
        # otherwise push "NOT_SCIENTIFIC_RESULT" past the head of the file. A key
        # beginning with "!" (0x21) sorts before every section name, so a quarantined
        # artifact still ANNOUNCES ITSELF in its first bytes.
        package["!! NOT_SCIENTIFIC_RESULT !!"] = QUARANTINE_BANNER
    package["limit"] = a.limit
    package["run_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Raw inputs to the assembler, retained so the eight-section package can be
    # re-derived and audited without re-running the experiment. Not part of the
    # frozen eight sections; the validator ignores extra keys.
    package["_state_summaries"] = list(summaries.values())
    package["_cfg"] = cfg

    with open(out, "w") as f:
        json.dump(package, f, indent=1, sort_keys=True, default=str)

    failed = package["RUN_COMPLETION"]["implementation_failure_status"] != "NONE"
    print(f"\nwrote {out}")
    if failed:
        print("\nHARD STOP: IMPLEMENTATION_FAILURE — scientific run aborted; "
              "no classification emitted.", file=sys.stderr)
        for m in package["RUN_COMPLETION"]["implementation_failures"]:
            print(f"  {m}", file=sys.stderr)
        return 3
    if a.smoke:
        print(f"\n*** {QUARANTINE_BANNER} ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
