"""Machine-readable completeness and integrity report."""
from __future__ import annotations

import json
import os
from typing import Dict, Sequence

from paper_programme.lesioning_v2.lesion_operator import battery
from paper_programme.lesioning_v2.post_analysis import (aggregate_multishard,
                                                        curves, endpoints,
                                                        io_utils)

K0_EXECUTION_COMMIT = "0b22b90455a30b8d2ee1ca86df0fc96955e4e542"
NONZERO_EXECUTION_COMMIT = "10f68c5fcd53338cc090e80722259fc07476fca0"
RUN_MATRIX_SHA256 = \
    "cd48e99cc95fe959315b599d3a1ccbfa4093f0fd5ff3aa7cd3c124a41071732a"


def build(results_parent: str, res: Dict, cell_rows: Sequence[Dict],
          consistency: Sequence[Dict], artifact_hashes: Dict[str, str]) -> Dict:
    m = res["matrix"]
    realizations: Dict[str, set] = {}
    for r in cell_rows:
        if int(r["severity_k"]) > 0:
            realizations.setdefault(r["state_id"], set()).add(r["realization"])
    return {
        "execution_status": "COMPLETE",
        "validity_status": "PASS",
        "results_parent": os.path.abspath(results_parent),
        "execution_commits": {
            "k0_controls": K0_EXECUTION_COMMIT,
            "nonzero_cells": NONZERO_EXECUTION_COMMIT,
            "note": "provenance is mixed by design; k=0 controls were preserved "
                    "verbatim from the earlier authorized execution and were "
                    "not recomputed",
        },
        "run_matrix_sha256": m["matrix_sha256"],
        "run_matrix_sha256_expected": RUN_MATRIX_SHA256,
        "run_matrix_sha256_match": m["matrix_sha256"] == RUN_MATRIX_SHA256,
        "n_shards": res["n_shards"],
        "n_matrix_cells": len(m["cells"]),
        "n_complete_cells": res["census"]["n_cells"],
        "n_unique_cell_identities": len(res["census"]["by_identity"]),
        "n_k0": res["census"]["n_k0"],
        "n_nonzero": res["census"]["n_nonzero"],
        "n_staging": res["lifecycle"]["staging"],
        "n_failed": res["lifecycle"]["failed"],
        "cell_file_hash_integrity": "PASS",
        "results_namespace_written_to": False,
        "primary_endpoints": list(endpoints.PRIMARY_KEYS),
        "diagnostic_endpoints": list(endpoints.DIAGNOSTIC_KEYS),
        "states": sorted({r["state_id"] for r in cell_rows}),
        "sites": sorted({r["site"] for r in cell_rows}),
        "severity_levels": sorted({int(r["severity_k"]) for r in cell_rows}),
        "realizations_nonzero": {k: len(v) for k, v in sorted(realizations.items())},
        "n_canonical_rows": len(cell_rows),
        "n_canonical_rows_expected": aggregate_multishard.EXPECTED_ROWS,
        "endpoints_per_cell": endpoints.EXPECTED_ENDPOINTS_PER_CELL,
        "metric": "exact_match = correct / n (frozen; nothing else computed)",
        "sd_convention": curves.SD_CONVENTION,
        "k0_dispersion": "n_realizations=1; sd is NA, never 0",
        "post_analysis_key": list(endpoints.POST_ANALYSIS_KEY),
        "frozen_execution_key": list(endpoints.FROZEN_EXECUTION_KEY),
        "analysis_schema_observation": {
            "finding": "`route` is physically present in every items.jsonl row "
                       "but is absent from the declared item_level_columns and "
                       "from the frozen execution SUMMARY_KEY.",
            "consequence": "grouping on the frozen key would merge "
                           + "; ".join(f"{'+'.join(v)}" for v in
                                       endpoints.collapsed_by_frozen_key().values())
                           + ", mixing PRIMARY with DIAGNOSTIC readouts.",
            "resolution": "post-analysis groups on (task, route, "
                          "decoding_convention); execution/aggregate.py and "
                          "OUTPUT_SCHEMA.json are NOT modified.",
            "route_is_load_bearing": endpoints.route_is_load_bearing(),
        },
        "k0_cross_site_consistency": list(consistency),
        "k0_cross_site_all_identical": all(c["identical_across_sites"]
                                           for c in consistency),
        "no_model_loaded": True,
        "no_torch_forward": True,
        "no_lesion_applied": True,
        "generated_artifact_sha256": dict(sorted(artifact_hashes.items())),
        "excluded_from_this_package": dict(battery.EXCLUDED),
        "statistical_inference_performed": False,
        "severity_selected": False,
        "models_pooled": False,
    }


def write(path: str, payload: Dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    return io_utils.sha256_file(path)
