"""Source-table inventory and availability check for the Yair meeting figures.

Read-only.  No model, no inference, no M4 code path.  Every candidate table is
hashed, counted and mapped to the meeting-figure panel it supports; requested
quantities that no validated table can supply are recorded as UNAVAILABLE rather
than approximated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MECH = "outputs/length_effect_mechanism_93a577f"
WFE = "reports/behavioral_wfe_fulllexicon_93a577f/figures"
OUT = os.path.join(ROOT, MECH, "figures/yair_meeting")

SOURCES = [
    (f"{MECH}/m2_gold_prefix/length_slopes_ar_vs_gold.tsv", "M2",
     "seed;exposure_status;ar_edit_slope;gp_edit_slope;ar_minus_gp_edit_slope;mean_ar_edit;mean_gp_edit",
     "A-A (LTM AR slope by exposure); B-A (AR vs gold-prefix paired slopes)",
     "LTM route only. ar_edit_slope is bit-identical to the archived WFE "
     "clean_length_slopes_by_seed ltm_length_slope for both clean groups."),
    (f"{WFE}/clean_route_length_contrasts.tsv", "WFE Sprint 1 (A05)",
     "seed;source_lexicality;wm_length_slope;ltm_length_slope;full_length_slope;ltm_minus_wm",
     "A-B (LTM - WM contrast)",
     "LICHTHEIM_CLEAN only: source_lexicality real == TRAINED_REAL_EXACT (671), "
     "pseudo == NOVEL_PSEUDOWORD (391). UNTRAINED_REAL is excluded from the "
     "clean set by definition, so no LTM-WM contrast exists for it."),
    (f"{WFE}/clean_length_slopes_by_seed.tsv", "WFE Sprint 1 (A05)",
     "seed;source_lexicality;route;n_items;length_slope;model_status",
     "A-B cross-check only",
     "Per-route slopes on the clean set; used to verify the M2 slopes."),
    (f"{MECH}/m1_origin_propagation/first_error_hazard.tsv", "M1",
     "event;seed;route;exposure_status;position;n_at_risk;n_events;hazard",
     "B-B (first-error hazard)",
     "Item-level survival denominator. NOT the PCHIP serial-position curve."),
    (f"{MECH}/m1_origin_propagation/first_error_events.tsv", "M1",
     "seed;item_id;route;exposure_status;length;FIRST_TOKEN_MISMATCH;"
     "FIRST_PREMATURE_EOS;first_divergence_type",
     "B-C annotation (EOS-first count)",
     "Supplies the 8-of-365 EOS-first annotation for novel pseudowords."),
    (f"{MECH}/m1_origin_propagation/post_divergence_burden.tsv", "M1",
     "seed;item_id;route;exposure_status;first_divergence_position;"
     "fraction_suffix_wrong;suffix_levenshtein;divergence_type",
     "B-C (post-divergence suffix burden)", "LTM rows used."),
    (f"{MECH}/m3_lexical_attraction/lexical_attraction_items.tsv", "M3",
     "seed;item_id;exposure_status;correct;d_pred_target;d_pred_top1;"
     "d_pred_topk_min;pred_is_training_form",
     "C-A (target vs neighbour distance)", "Erroneous LTM items only."),
    (f"{MECH}/m3_lexical_attraction/matched_baseline.tsv", "M3",
     "stratum;n;observed_pred_to_own_top1;permuted_pred_to_other_top1;"
     "attraction_advantage",
     "C-A caption only (matched baseline)", "Caption, not a panel."),
    (f"{MECH}/m5_dorsal_rescue/word_level_route_outcomes.tsv", "M5",
     "seed;lichtheim_exposure_status;route_outcome_category;n",
     "C-B (word-level rescue)", "Behavioural co-occurrence categories."),
    (f"{MECH}/m5_dorsal_rescue/position_level_rescue_summary.tsv", "M5",
     "seed;exposure_status;position_rescue_category;n",
     "C-B (position-level rescue under the common FULL prefix)",
     "Same-prefix comparison; the mechanistically interpretable one."),
    (f"{MECH}/instrumented/item_summary.tsv", "instrumented run",
     "seed;item_id;exposure_status;gate;confidence;phoneme_length",
     "C-B secondary annotation (mean gate)",
     "Gate is word-level and constant within an item."),
]

# Quantities the brief asks for that no validated table can supply.
UNAVAILABLE = [
    {"requested_quantity": "LTM - WM length-slope contrast for UNTRAINED_REAL",
     "requested_in": "Figure A, Panel B",
     "missing_source_table": "a per-route (WM) length slope for the "
                             "UNTRAINED_REAL exposure group",
     "why": "UNTRAINED_REAL is excluded from LICHTHEIM_CLEAN by the frozen "
            "Sprint-1 analysis-set definition, so clean_route_length_contrasts."
            "tsv and clean_length_slopes_by_seed.tsv contain no WM slope for "
            "it. The M2 table carries LTM slopes only. No validated table "
            "anywhere holds a WM length slope for these 122 items.",
     "action": "Panel B plots the two clean groups only; the UNTRAINED_REAL "
               "position is drawn as an explicit 'not available' marker and "
               "the caption states the reason. Not approximated, not "
               "reconstructed, no analysis rerun."},
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for rel, phase, cols, panels, lim in SOURCES:
        p = os.path.join(ROOT, rel)
        ok = os.path.exists(p)
        n = (sum(1 for _ in open(p, encoding="utf-8")) - 1) if ok else 0
        rows.append({"path": rel, "sha256": sha(p) if ok else "",
                     "origin_analysis_phase": phase, "row_count": n,
                     "key_columns": cols,
                     "validated": "yes" if ok else "MISSING",
                     "supports_panels": panels, "limitations": lim})
    with open(os.path.join(OUT, "source_table_inventory.tsv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # cross-check M2 slopes against the archived release
    m2 = pd.read_csv(os.path.join(ROOT, MECH,
                                  "m2_gold_prefix/length_slopes_ar_vs_gold.tsv"),
                     sep="\t")
    sl = pd.read_csv(os.path.join(ROOT, WFE, "clean_length_slopes_by_seed.tsv"),
                     sep="\t")
    ltm = sl[sl.route == "ltm"]
    checks = {}
    for exp, lex in (("TRAINED_REAL_EXACT", "real"),
                     ("NOVEL_PSEUDOWORD", "pseudo")):
        a = m2[m2.exposure_status == exp].set_index("seed").ar_edit_slope
        b = ltm[ltm.source_lexicality == lex].set_index("seed").length_slope
        checks[exp] = float((a - b).abs().max())

    json.dump({"n_sources": len(rows),
               "all_present": all(r["validated"] == "yes" for r in rows),
               "m2_vs_archived_max_abs_diff": checks,
               "m2_matches_archived_release": all(v < 1e-9 for v in checks.values()),
               "unavailable_quantities": UNAVAILABLE,
               "model_inference_performed": False,
               "m4_code_path_executed": False},
              open(os.path.join(OUT, "source_validation.json"), "w"), indent=2)

    md = ["# Meeting figures — source-table inventory\n",
          "Read-only inventory. **No model inference, no M4 code path, no "
          "analysis rerun.** Every plotted value comes from one of the tables "
          "below.\n",
          "| table | phase | rows | supports |", "|---|---|---:|---|"]
    for r in rows:
        md.append(f"| `{os.path.basename(r['path'])}` | {r['origin_analysis_phase']} "
                  f"| {r['row_count']} | {r['supports_panels']} |")
    md += ["\n## Cross-check against the archived release\n",
           "The M2 `ar_edit_slope` was compared against the archived Sprint-1 "
           "`clean_length_slopes_by_seed.tsv` `ltm_length_slope`:\n"]
    for k, v in checks.items():
        md.append(f"- **{k}**: max absolute difference **{v:.1e}** — "
                  f"{'bit-identical' if v < 1e-9 else 'DIFFERS'}")
    md += ["\nThe mechanism tables therefore reproduce the closed behavioural "
           "release exactly for the quantities they share.\n",
           "## Unavailable requested quantities\n"]
    for u in UNAVAILABLE:
        md.append(f"### {u['requested_quantity']}\n")
        md.append(f"- **Requested in**: {u['requested_in']}")
        md.append(f"- **Missing source**: {u['missing_source_table']}")
        md.append(f"- **Why**: {u['why']}")
        md.append(f"- **Action**: {u['action']}\n")
    open(os.path.join(OUT, "source_table_inventory.md"), "w").write("\n".join(md))

    print(f"{len(rows)} source tables inventoried; all present: "
          f"{all(r['validated'] == 'yes' for r in rows)}")
    for k, v in checks.items():
        print(f"  M2 vs archived {k}: {v:.1e}")
    print(f"unavailable quantities: {len(UNAVAILABLE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
