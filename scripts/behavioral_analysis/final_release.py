"""Final release — selection, copying and legacy-table loading.

Editorial layer only.  It computes **no statistic**: every value it touches
already exists in a validated table produced by Sprints 1-5 or by the analysis
phase.  A09, A10 and A11 are rendered from their stored authoritative tables and
are never refitted — which is precisely what preserves A11's Ridge alpha = 1.0,
its 80/20 random_state=42 split, its n_repeats=100 random_state=42 permutation
and its historical signed convention: no model is fitted, so nothing can drift.

The faithful outputs under outputs/.../faithful_replication/ are read-only here.
Nothing in this module writes outside the release tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Dict, List

import pandas as pd

from .common import REPO_ROOT, REPORT_ROOT, repo_relative

LEGACY_DIR = os.path.join(
    REPO_ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f",
    "behavioral_analysis", "faithful_replication")

LEGACY_TABLES = {
    "A09": "faithful_figure2A_table.tsv",
    "A10": "faithful_figure2C_table.tsv",
    "A11": "faithful_figure2B_feature_importance.tsv",
}
LEGACY_STEM = {"A09": "A09_faithful_figure2A",
               "A10": "A10_faithful_figure2C",
               "A11": "A11_faithful_figure2B"}

SPEC_JSON = os.path.join(REPORT_ROOT, "final_release", "_control",
                         "final_release_spec.json")

# Rows of the analysis matrix whose status this release closes.
FORMATTED_ROWS = ["A09", "A10", "A11"]
SSP_ROW = "A19"
SSP_STATUS = "OPTIONAL_DEFERRED"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec(path: str = SPEC_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def legacy_table(row: str) -> pd.DataFrame:
    """Read a stored faithful table.  Read-only; never recomputed."""
    if row not in LEGACY_TABLES:
        raise ValueError(f"unknown legacy row {row!r}")
    return pd.read_csv(os.path.join(LEGACY_DIR, LEGACY_TABLES[row]), sep="\t")


def legacy_source_hashes() -> Dict[str, str]:
    return {repo_relative(os.path.join(LEGACY_DIR, n)):
            sha256_file(os.path.join(LEGACY_DIR, n))
            for n in sorted(os.listdir(LEGACY_DIR))}


def copy_with_provenance(src: str, dst: str) -> dict:
    """Copy a selected artefact and record both hashes plus an equality verdict.

    Copying rather than moving is deliberate: the source reports stay exactly
    where the earlier sprints validated them.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_sha = sha256_file(src)
    shutil.copyfile(src, dst)
    dst_sha = sha256_file(dst)
    return {"source_path": repo_relative(src), "source_sha256": src_sha,
            "release_path": repo_relative(dst), "release_sha256": dst_sha,
            "equality": "IDENTICAL" if src_sha == dst_sha else "DIFFERS"}


def select_figures(spec: dict) -> List[dict]:
    """Flatten the frozen selection into deterministic index rows."""
    rows = []
    for cat, key in (("MAIN", "main_figures"),
                     ("SUPPLEMENTARY", "supplementary_figures")):
        for f in spec[key]:
            rows.append({
                "category": cat, "figure_number": f["number"],
                "analysis": f["analysis"], "regime_kind": f["regime_kind"],
                "dataset_regime": f["dataset"],
                "source_stem": f["source_stem"],
                "release_stem": f"{f['number']}_{os.path.basename(f['source_stem'])}",
                "purpose": f["purpose"], "main_finding": f["main_finding"],
                "primary_limitation": f["primary_limitation"],
                "reason_for_inclusion": (
                    "carries one of the six frozen main narrative points"
                    if cat == "MAIN" else
                    "scientifically useful but not required for the main "
                    "narrative"),
            })
    rows.sort(key=lambda r: (r["category"] != "MAIN",
                             int(r["figure_number"][1:]), r["source_stem"]))
    return rows


def excluded_figures() -> List[dict]:
    """Figures deliberately not selected, with the reason recorded."""
    return [{
        "category": "MECHANISM_HANDOFF_ONLY",
        "figure_number": "", "analysis": "A22",
        "regime_kind": "adapted", "dataset_regime": "LICHTHEIM_CLEAN",
        "source_stem": "reports/behavioral_wfe_fulllexicon_93a577f/"
                       "error_taxonomy/length_effect_mechanism_handoff.md",
        "release_stem": "", "purpose": "factual handoff to the separate "
                                       "mechanism project",
        "main_finding": "no finding; it poses factual questions only",
        "primary_limitation": "not a figure and carries no estimate",
        "reason_for_inclusion": "belongs to the separate mechanism project, "
                                "not to this release's figure set",
    }]


def status_summary(spec: dict) -> pd.DataFrame:
    """Analysis-status table.  Mirrors the matrix; asserts nothing new."""
    rows = [
        ("A01", "Script versioning and code promotion", "n/a", "ALREADY_VALIDATED", "Sprint 1"),
        ("A02", "Production manifest closure", "n/a", "ALREADY_VALIDATED", "Sprint 1"),
        ("A03", "Documentation and portable provenance", "n/a", "ALREADY_VALIDATED", "Sprint 1"),
        ("A04", "Clean length curves", "adapted", "ALREADY_VALIDATED", "Sprint 1"),
        ("A05", "Clean length slopes and LTM-WM contrast", "adapted", "ALREADY_VALIDATED", "Sprint 1"),
        ("A06", "Clean serial position", "faithful", "ALREADY_VALIDATED", "Sprint 1"),
        ("A07", "Gate and confidence, clean set", "adapted", "ALREADY_VALIDATED", "Sprint 1"),
        ("A08", "Gate and confidence by exposure status", "adapted", "ALREADY_VALIDATED", "Sprint 1"),
        ("A09", "Faithful Figure 2A", "faithful", "ALREADY_VALIDATED_FORMATTED", "final release"),
        ("A10", "Faithful Figure 2C", "faithful", "ALREADY_VALIDATED_FORMATTED", "final release"),
        ("A11", "Faithful feature importance", "faithful", "ALREADY_VALIDATED_FORMATTED", "final release"),
        ("A12", "Clean morphology x length", "adapted", "ALREADY_VALIDATED", "Sprint 2"),
        ("A13", "Faithful morphology x length", "faithful", "ALREADY_VALIDATED", "Sprint 2"),
        ("A14", "Trained-real frequency", "adapted", "ALREADY_VALIDATED", "Sprint 3"),
        ("A15", "Adapted feature importance", "adapted", "ALREADY_VALIDATED", "Sprint 5"),
        ("A16", "Faithful error taxonomy", "faithful", "ALREADY_VALIDATED", "Sprint 4"),
        ("A17", "Clean error taxonomy", "adapted", "ALREADY_VALIDATED", "Sprint 4"),
        ("A18", "Premature EOS", "adapted", "ALREADY_VALIDATED", "Sprint 4"),
        ("A19", "SSP / sonority", "faithful", SSP_STATUS, "not started"),
        ("A20", "Neural representations", "adapted", "OUT_OF_SCOPE", "separate project"),
        ("A21", "Route ablations", "adapted", "OUT_OF_SCOPE", "separate project"),
        ("A22", "Causal length-effect mechanism", "adapted", "OUT_OF_SCOPE", "separate project"),
    ]
    return pd.DataFrame(rows, columns=["analysis_id", "analysis_name",
                                       "faithful_or_adapted", "final_status",
                                       "closed_in"])


def dataset_regime_summary() -> pd.DataFrame:
    return pd.DataFrame([
        {"dataset_regime": "FAITHFUL_WFE_ALL", "n_items": 1200,
         "composition": "all original WFE items with their source labels",
         "kind": "faithful",
         "caveat": "122 source-real items were never trained; 9 source-pseudo "
                   "items collide with the training lexicon; source labels are "
                   "not training exposure"},
        {"dataset_regime": "LICHTHEIM_CLEAN", "n_items": 1062,
         "composition": "671 TRAINED_REAL_EXACT + 391 NOVEL_PSEUDOWORD",
         "kind": "adapted",
         "caveat": "lexicality and training exposure are perfectly confounded; "
                   "the factor is a lexicality/exposure contrast"},
        {"dataset_regime": "ALL_WITH_EXPOSURE_STRATA", "n_items": 1200,
         "composition": "six training-exposure categories",
         "kind": "adapted",
         "caveat": "three categories have n <= 7 and are descriptive only"},
        {"dataset_regime": "TRAINED_REAL_FREQUENCY_PRIMARY", "n_items": 671,
         "composition": "trained real words with a Zipf value", "kind": "adapted",
         "caveat": "frequency is undefined for pseudowords and is never imputed"},
        {"dataset_regime": "TRAINED_REAL_FREQUENCY_SENSITIVITY", "n_items": 678,
         "composition": "adds the 7 pronunciation-variant words", "kind": "adapted",
         "caveat": "sensitivity only; never the primary claim"},
    ])


def checkpoint_summary(prov: dict) -> pd.DataFrame:
    epochs = {"19": 155, "20": 130, "21": 145, "22": 140}
    rows = []
    for seed in ("19", "20", "21", "22"):
        rows.append({"seed": int(seed), "epoch": epochs[seed],
                     "checkpoint_path": prov["checkpoint_paths"][seed],
                     "checkpoint_sha256": prov["checkpoint_sha256"][seed],
                     "role": "primary" if seed == "21" else "primary",
                     "exact_zero_sensitivity_member": seed in ("19", "20", "22"),
                     "loaded_in_final_release": False})
    return pd.DataFrame(rows)
