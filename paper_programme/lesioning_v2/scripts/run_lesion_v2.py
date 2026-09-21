#!/usr/bin/env python
"""Lesioning V2 scientific runner — execution body, guarded.

    GO_FOR_SCIENTIFIC_EXECUTION = NO

Consumes ONLY `contract/LESIONING_V2_RUN_MATRIX.json`; it never regenerates the
matrix. Every nonzero lesion cell requires a CENTRAL authorization artifact that
binds BOTH the exact execution commit and the authoritative run-matrix hash.

`--dry-run` performs the full preflight and emits the execution plan with ZERO
model forwards, and is always available.

This program never trains: no backward, no optimizer, no checkpoint write.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
REPO = os.path.normpath(os.path.join(PKG, "..", ".."))
for p in (REPO, os.path.join(REPO, "paper_programme", "v7_prelesion_validation",
                             "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_programme.lesioning_v2.execution import (  # noqa: E402
    aggregate, authorization, cells, evaluators, injection, lesioned_eval,
    preflight)
from paper_programme.lesioning_v2.lesion_operator import (  # noqa: E402
    battery, context, masks)
from paper_programme.lesioning_v2.lesion_operator.sites import SITES  # noqa: E402


def _matrix() -> dict:
    return preflight.load_matrix()


def execution_plan(m: dict, rep: dict) -> dict:
    """Metadata/provenance only. No scientific prediction of any kind."""
    ids, keys = {}, {}
    for row in m["cells"]:
        cid, key = cells.cell_identity(row), cells.cell_key(row)
        ids.setdefault(cid, []).append(key)
        keys.setdefault(key, []).append(cid)
    dup = {k: v for k, v in ids.items() if len(v) > 1}
    return {
        "plan_name": "LESIONING_V2_EXECUTION_PLAN",
        "authoritative_matrix_sha256": m["matrix_sha256"],
        "matrix_regenerated_by_runner": False,
        "total_rows": len(m["cells"]),
        "unique_executable_cell_identities": len(ids),
        "duplicate_cell_identities": len(dup),
        "missing_cell_identities": len(m["cells"]) - len(ids),
        "intact_control_rows": sum(1 for c in m["cells"]
                                   if c["severity_k"] == 0),
        "nonzero_lesion_rows": sum(1 for c in m["cells"]
                                   if c["severity_k"] > 0),
        "post_repair_only": all(c["state_id"].endswith("_POST_REPAIR")
                                for c in m["cells"]),
        "evaluator_identity": rep.get("evaluator_identity"),
        "primary_endpoints": [e.key for e in battery.PRIMARY],
        "diagnostic_endpoints": [e.key for e in battery.DIAGNOSTIC],
        "scientific_results_included": False,
    }


def run_cell(row: dict, state_cache: dict, sd_constants: dict,
             out_root: str, transfer: dict) -> str:
    """Execute ONE authoritative matrix cell and finalize it atomically."""
    import prelesion_eval as pe

    site = SITES[row["site"]]
    k = int(row["severity_k"])
    state_id = row["state_id"]

    tr, model, prov = state_cache[state_id]
    ck = prov["source_checkpoint"]
    ck_before = pe.sha256_file(ck)
    pristine = context.parameter_digest(model)

    bank = list(range(len(tr.entries)))
    comp = list(tr.comp_idx)
    sd_site = float(sd_constants["constants"][f"{state_id}/{row['site']}"]
                    ["sample_sd"])

    cell_masks = masks.build_mask(row["state_sha256"], site,
                                  int(row["realization"]), k)

    if k == 0:
        # INTACT CONTROL -- the original path, unchanged. The null perturbation
        # still installs and removes the same hook, so the control traverses a
        # byte-identical evaluator path; it is deliberately NOT routed through
        # the nonzero batching driver.
        with context.connectivity_lesion(model, cell_masks):
            with injection.activation_injection(model, row["site"], None):
                item_rows = evaluators.evaluate_endpoints(
                    tr, model, battery.ALL_ENDPOINTS, bank, comp)
    else:
        # NONZERO -- each frozen evaluator is driven one chunk at a time at its
        # OWN batch size, with the hook installed per chunk from exactly that
        # chunk's GLOBAL item ids.
        def make_eta(batch_item_ids):
            return injection.batch_eta_fn(
                row["state_sha256"], row["site"], int(row["realization"]),
                batch_item_ids, k, sd_site)

        with context.connectivity_lesion(model, cell_masks):
            item_rows = lesioned_eval.evaluate_endpoints_lesioned(
                tr, model, battery.ALL_ENDPOINTS, bank, comp, row["site"],
                make_eta)

    restored = context.parameter_digest(model) == pristine
    pe.assert_source_unchanged(ck, ck_before)

    for r in item_rows:
        r.update({
            "state_id": state_id, "state_sha256": row["state_sha256"],
            "site": row["site"], "severity_k": k,
            "severity_s": row["severity_s"],
            "connectivity_fraction": row["connectivity_fraction"],
            "n_logical_edges_total": row["n_logical_edges_total"],
            "n_logical_edges_removed": row["n_logical_edges_removed"],
            "realization": row["realization"],
            "mask_seed_digest": row["mask_seed_digest"],
            "activation_seed_digest": row["activation_seed_namespace"],
        })

    summary = {}
    for r in item_rows:
        key = (state_id, row["site"], k, row["realization"], r["task"],
               r["decoding_convention"])
        s = summary.setdefault("|".join(map(str, key)),
                               {"state_id": state_id, "site": row["site"],
                                "severity_k": k,
                                "realization": row["realization"],
                                "task": r["task"],
                                "decoding_convention": r["decoding_convention"],
                                "n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(r["correct"])
    for s in summary.values():
        s["exact_match"] = s["correct"] / s["n"] if s["n"] else None

    provenance = {
        "cell_identity": cells.cell_identity(row),
        "matrix_row": {kk: row[kk] for kk in sorted(row)},
        "sd_site": sd_site, "sd_constant_ref": row["sd_constant_ref"],
        "source_checkpoint_sha256": ck_before,
        "repair_head_file_sha256": prov.get("repair_head_file_sha256"),
        "repair_head_deployed_state_sha256":
            prov.get("repair_head_deployed_state_sha256"),
        "parameter_restoration_verified": restored,
        "transfer_artifacts": transfer,
        "cell_kind": row["cell_kind"],
    }
    return cells.write_cell(out_root, row, item_rows,
                            {"rows": list(summary.values())}, provenance,
                            restoration_verified=restored)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lesioning V2 runner (guarded).")
    ap.add_argument("--out-dir", required=True,
                    help="scientific result namespace; must NOT already exist. "
                         "Keep it OUTSIDE the git worktree.")
    ap.add_argument("--transfer-dir", default=None,
                    help="external execution-control directory holding the two "
                         "transfer-required artifacts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continuity-manifest", default=None,
                    help="explicit, fail-closed continuation into an existing "
                         "namespace validated as exactly the 12 COMPLETE k=0 "
                         "controls. Without it an existing namespace is "
                         "refused, as before.")
    ap.add_argument("--plan-out", default=None)
    ap.add_argument("--shard", type=int, default=None,
                    help="scheduler shard index; partitions cells by "
                         "(state, site) ONLY. Never enters any seed or cell "
                         "identity, so sharding cannot alter outputs.")
    ap.add_argument("--n-shards", type=int, default=None)
    a = ap.parse_args(argv)

    try:
        rep = preflight.run(a.out_dir,
                            require_authorization=not a.dry_run,
                            transfer_dir=a.transfer_dir,
                            allow_missing_transfer=a.dry_run,
                            continuity_manifest=a.continuity_manifest)
    except (preflight.PreflightError,
            authorization.AuthorizationRefused) as e:
        print(f"PREFLIGHT_FAIL\n{e}", file=sys.stderr)
        return 2

    m = _matrix()
    plan = execution_plan(m, rep)
    print("PREFLIGHT_OK")
    for kk in ("head_commit", "matrix_sha256", "n_cells", "mode",
               "package_integrity",
               "scientific_configuration", "no_training_path",
               "output_namespace_absent", "tree_clean"):
        print(f"  {kk} = {rep.get(kk)}")
    print(f"  unique_cell_identities = "
          f"{plan['unique_executable_cell_identities']} / {plan['total_rows']}")

    if a.plan_out:
        plan["preflight"] = {k: v for k, v in rep.items()
                             if k != "transfer_artifacts"}
        json.dump(plan, open(a.plan_out, "w"), indent=1, sort_keys=True)
        print(f"  plan written: {a.plan_out}")

    if a.dry_run:
        print("DRY_RUN=1 : no model was loaded and no forward pass was run.")
        print("GO_FOR_SCIENTIFIC_EXECUTION=NO")
        return 0

    # --- authorized scientific execution ------------------------------------
    import prelesion_eval as pe
    from scripts.naming_comprehension.ceiling_source_completion import GLOVE

    sd_path = rep["transfer_artifacts"]["LESIONING_V2_INTACT_SD_CONSTANTS.json"]
    sd_constants = json.load(open(sd_path))
    sv_path = rep["transfer_artifacts"]["LESIONING_V2_STATE_VERIFICATION.json"]
    states = json.load(open(sv_path))["states"]

    rows = m["cells"]
    if a.shard is not None:
        if a.n_shards is None or not (0 <= a.shard < a.n_shards):
            print("invalid shard specification", file=sys.stderr)
            return 2
        groups = sorted({(r["state_id"], r["site"]) for r in rows})
        mine = {g for i, g in enumerate(groups) if i % a.n_shards == a.shard}
        rows = [r for r in rows if (r["state_id"], r["site"]) in mine]
        print(f"  shard {a.shard}/{a.n_shards}: {len(rows)} cells")

    state_cache = {}
    os.makedirs(a.out_dir, exist_ok=True)
    done = 0
    for row in rows:
        if cells.is_complete(a.out_dir, row):
            continue                       # never rerun a finalized cell
        sid = row["state_id"]
        if sid not in state_cache:
            s = states[sid]
            state_cache[sid] = pe.build_state(s["source_checkpoint"],
                                              s["repair_head"], GLOVE)
        run_cell(row, state_cache, sd_constants, a.out_dir,
                 rep["transfer_artifacts"])
        done += 1
        print(f"finalized {cells.cell_key(row)}  ({done}/{len(rows)})",
              flush=True)

    if a.shard is not None:
        print(f"shard {a.shard} complete: {done} cells finalized")
        return 0
    agg = aggregate.aggregate(a.out_dir, expected_cells=len(m["cells"]))
    json.dump(agg, open(os.path.join(a.out_dir, "AGGREGATE.json"), "w"),
              indent=1)
    print(f"COMPLETE cells: {agg['n_complete_cells']} / {len(m['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
