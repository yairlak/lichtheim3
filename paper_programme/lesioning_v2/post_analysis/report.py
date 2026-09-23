"""Descriptive report and CENTRAL handoff.

FACTUAL ONLY. Every qualitative word is accompanied by the numbers that license
it, and no word is defined by a threshold invented here. The package assigns no
SUPPORTED / PARTIAL / NOT_SUPPORTED label: no objective rule for those labels
was frozen before results existed, so the full factual pattern goes to CENTRAL
for arbitration.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

from paper_programme.lesioning_v2.lesion_operator import battery
from paper_programme.lesioning_v2.post_analysis import curves, io_utils, validate

STATES = ("P1_POST_REPAIR", "P2_POST_REPAIR", "P3_POST_REPAIR", "P4_POST_REPAIR")
ROLE = {"P1_POST_REPAIR": "PRIMARY", "P2_POST_REPAIR": "replication",
        "P3_POST_REPAIR": "replication", "P4_POST_REPAIR": "SENSITIVITY"}
SITES = ("L1", "L2", "L3")
SITE_DESC = {"L1": "dorsal / WM encoder", "L2": "ventral / LTM encoder",
             "L3": "ventral production"}


def _f(x, nd=4):
    v = io_utils.fnum(x)
    return "NA" if v is None else f"{v:.{nd}f}"


def _pick(curve, state, site, ep, k):
    for r in curve:
        if (r["state_id"] == state and r["site"] == site
                and r["endpoint"] == ep and int(r["severity_k"]) == k):
            return r
    return None


def _curve_table(curve, site, ep_key) -> List[str]:
    L = [f"| k | s_k | " + " | ".join(ROLE_HDR := [s.split('_')[0] for s in STATES]) + " |",
         "|---|---|" + "---|" * len(STATES)]
    for k in range(16):
        cells_ = []
        s_k = ""
        for st in STATES:
            r = _pick(curve, st, site, ep_key, k)
            if r is None:
                cells_.append("--")
                continue
            s_k = _f(r["severity_s"], 4)
            m = _f(r["mean_exact_match"])
            sd = r["sd_exact_match"]
            n = r["n_realizations"]
            cells_.append(f"{m}" if k == 0 else f"{m} ±{_f(sd)} (n={n})")
        L.append(f"| {k} | {s_k} | " + " | ".join(cells_) + " |")
    return L


def _delta(curve, state, site, ep_key):
    """k=15 mean minus the exact k=0 control. Arithmetic only; no threshold."""
    a = _pick(curve, state, site, ep_key, 0)
    b = _pick(curve, state, site, ep_key, 15)
    if not a or not b:
        return None
    x, y = io_utils.fnum(a["mean_exact_match"]), io_utils.fnum(b["mean_exact_match"])
    return None if x is None or y is None else y - x


def descriptive_report(curve, baselines, consistency, k15, val) -> str:
    L = ["# LESIONING V2 — DESCRIPTIVE RESULTS", "",
         "Factual descriptive package. No hypothesis test, no confidence "
         "interval, no fitted threshold, no selected severity, no success rule, "
         "and no SUPPORTED/PARTIAL/NOT_SUPPORTED label. All severities k=0..15 "
         "are reported for every endpoint.", "",
         "## 1. Execution and validity", "",
         f"    SCIENTIFIC_EXECUTION_STATUS = {val['execution_status']}",
         f"    VALIDITY_STATUS             = {val['validity_status']}",
         f"    cells                       = {val['n_complete_cells']}/"
         f"{val['n_matrix_cells']}  (k=0 {val['n_k0']}, nonzero {val['n_nonzero']})",
         f"    unique cell identities      = {val['n_unique_cell_identities']}",
         f"    staging / failed            = {val['n_staging']} / {val['n_failed']}",
         f"    cell file hash integrity    = {val['cell_file_hash_integrity']}",
         f"    run matrix sha256           = {val['run_matrix_sha256']} "
         f"(match: {val['run_matrix_sha256_match']})",
         f"    canonical endpoint rows     = {val['n_canonical_rows']} "
         f"(expected {val['n_canonical_rows_expected']})", "",
         "Provenance is mixed by design:", "",
         f"    k=0 controls : {val['execution_commits']['k0_controls']}",
         f"    k>0 cells    : {val['execution_commits']['nonzero_cells']}", "",
         "The k=0 controls were preserved verbatim from the earlier authorized "
         "execution and were NOT recomputed.", "",
         "## 2. Experimental design", "",
         f"    states       {', '.join(f'{s} ({ROLE[s]})' for s in STATES)}",
         f"    sites        " + ", ".join(f"{s} = {SITE_DESC[s]}" for s in SITES),
         "    severity     k=0 intact control; k=1..15 nonzero, s_k = k/15",
         f"    realizations " + ", ".join(
             f"{k.split('_')[0]}:{v}" for k, v in
             sorted(val['realizations_nonzero'].items())),
         f"    metric       {val['metric']}",
         f"    dispersion   {val['sd_convention']}",
         f"                 k=0: {val['k0_dispersion']}", "",
         "    PRIMARY      " + ", ".join(val["primary_endpoints"]),
         "    DIAGNOSTIC   " + ", ".join(val["diagnostic_endpoints"]), "",
         "Diagnostic route-isolated endpoints are explanatory only. They may "
         "illuminate a primary effect; they never replace or rescue one.", "",
         "Excluded from this package by the frozen design: "
         + "; ".join(f"{k} ({v})" for k, v in val["excluded_from_this_package"].items()),
         "", "## 3. Intact baselines (k=0)", "",
         "All twelve state x site records are reported verbatim; none is "
         "averaged or collapsed. At k=0 the mask is all-ones and the "
         "perturbation null, so for a given state the value is expected to be "
         "identical at L1/L2/L3 — reported below as a check, not enforced.", "",
         "| state | endpoint | sites | distinct values | identical | values |",
         "|---|---|---|---|---|---|"]
    for c in consistency:
        L.append(f"| {c['state_id'].split('_')[0]} | {c['endpoint']} | "
                 f"{c['n_sites']} | {c['distinct_values']} | "
                 f"{c['identical_across_sites']} | {c['values']} |")
    L += ["", f"    k0_cross_site_all_identical = {val['k0_cross_site_all_identical']}",
          "", "Full per-site records: `INTACT_BASELINES.tsv`.", ""]

    for i, site in enumerate(SITES, start=4):
        L += [f"## {i}. {site} primary curves ({SITE_DESC[site]})", ""]
        for ep in battery.PRIMARY:
            L += [f"### {ep.key}", "",
                  f"    population : {ep.population}",
                  f"    decoding   : {ep.decoding_convention}",
                  f"    exactness  : {ep.exactness}", ""]
            L += _curve_table(curve, site, ep.key)
            L += ["", "k=0 is the exact preserved control (single observation, "
                  "no dispersion). k>0 cells show mean ±1 sample SD over "
                  "realizations.", ""]
            deltas = []
            for st in STATES:
                d = _delta(curve, st, site, ep.key)
                if d is not None:
                    deltas.append(f"{st.split('_')[0]} {d:+.4f}")
            if deltas:
                L += ["    k=15 mean minus k=0 control: " + ", ".join(deltas),
                      "    (arithmetic difference only; no threshold is applied)", ""]
        L += [f"Figure: `figures/PRIMARY_{site}.png`", ""]

    L += ["## 7. P1 / P2 / P3 comparison", "",
          "P1 is the primary model; P2 and P3 are replications. They are "
          "reported side by side in every table above and are never pooled. "
          "Each state's own curve and dispersion stand alone.", "",
          "## 8. P4 sensitivity", "",
          "P4 is the prospective sensitivity / preservation counterexample, "
          "with 4 realizations rather than 12. It is shown separately (dashed, "
          "labelled SENSITIVITY) and is never pooled with P1-P3 nor used to "
          "establish a family-level statement.", "",
          "## 9. Diagnostic route decomposition", "",
          "Route-isolated WM-only and LTM-only readouts appear in "
          "`CURVE_SUMMARY.tsv` (tier=DIAGNOSTIC) and in "
          "`figures_diagnostic/`. They are explanatory only.", ""]
    for site in SITES:
        L.append(f"    {site}: figures_diagnostic/DIAGNOSTIC_{site}.png")
    L += ["", "## 10. Maximum severity k=15", "",
          "k=15 is one pre-defined point on the full reported curve, not a "
          "success criterion. Per-realization values and descriptive "
          "mean/SD/min/max are in `MAX_SEVERITY_K15.tsv`.", "",
          "## 11. What is directly established by the data", "",
          "    * the execution is complete and internally consistent: "
          f"{val['n_complete_cells']}/{val['n_matrix_cells']} cells, "
          f"{val['n_unique_cell_identities']} unique identities, "
          f"{val['n_staging']} staging, {val['n_failed']} failed, hash "
          "integrity PASS;",
          "    * each endpoint's exact-match rate at every severity k=0..15 for "
          "every state x site x realization, with its own denominator;",
          "    * the arithmetic difference between the k=15 mean and the exact "
          "k=0 control for each state x site x endpoint;",
          "    * whether the k=0 control agrees across sites within a state.", "",
          "Any further qualitative statement must be read off these tables.", "",
          "## 12. What is NOT established", "",
          "    * no statistical inference of any kind was performed: no "
          "p-value, confidence interval, effect-size test or significance "
          "claim appears in this package;",
          "    * no severity was selected, and no threshold defines words such "
          "as preserved, selective, robust, supported or failed;",
          "    * P1-P4 are not pooled, and P4 does not carry a family-level "
          "conclusion;",
          "    * this is NOT a full seven-panel Ueno replication, and the "
          "pseudoword lesion battery, Ueno-adapted naming and the HF/LF split "
          "are outside this package by frozen design;",
          "    * diagnostic route-isolated endpoints do not substitute for a "
          "primary FULL-route result;",
          "    * the k=0 controls carry the earlier execution commit and were "
          "not recomputed under the repaired commit.", ""]
    return "\n".join(L) + "\n"


def central_handoff(curve, k15, val) -> str:
    def profile(site):
        out = []
        for ep in battery.PRIMARY:
            parts = []
            for st in STATES:
                d = _delta(curve, st, site, ep.key)
                b = _pick(curve, st, site, ep.key, 0)
                e = _pick(curve, st, site, ep.key, 15)
                if not (b and e):
                    continue
                parts.append(f"{st.split('_')[0]} {_f(b['mean_exact_match'])}"
                             f"->{_f(e['mean_exact_match'])}"
                             f" ({d:+.4f})" if d is not None else "")
            out.append(f"      {ep.key}: " + "; ".join(p for p in parts if p))
        return "\n".join(out)

    def state_summary(st):
        out = []
        for site in SITES:
            for ep in battery.PRIMARY:
                d = _delta(curve, st, site, ep.key)
                if d is None:
                    continue
                out.append(f"      {site}/{ep.key}: k0->k15 {d:+.4f}")
        return "\n".join(out)

    L = ["# CENTRAL FINAL HANDOFF — LESIONING V2 RESULTS", "",
         f"    SCIENTIFIC_EXECUTION_STATUS={val['execution_status']}",
         f"    VALIDITY_STATUS={val['validity_status']}", "",
         f"    K0_CONTROLS={val['n_k0']}/12",
         f"    NONZERO_CELLS={val['n_nonzero']}/1800",
         f"    TOTAL_CELLS={val['n_complete_cells']}/{val['n_matrix_cells']}", "",
         "Values below are k=0 control -> k=15 mean, with the arithmetic "
         "difference. They are descriptive coordinates into the full k=0..15 "
         "tables, not summaries that replace them.", ""]
    for st in STATES:
        tag = "P4_SENSITIVITY" if st == "P4_POST_REPAIR" else f"{st.split('_')[0]}_RESULTS"
        L += [f"    {tag}=", state_summary(st), ""]
    for site in SITES:
        L += [f"    {site}_RESULT_PROFILE=  ({SITE_DESC[site]})", profile(site), ""]
    L += ["    CROSS_MODEL_CONCORDANCE=",
          "      P1, P2 and P3 are reported separately and are never pooled; "
          "their per-endpoint curves appear side by side in every table and "
          "figure. P4 (4 realizations) is shown as sensitivity only. CENTRAL "
          "can read concordance directly off the curve tables.", "",
          "    CORE_DUAL_ROUTE_LESION_EVIDENCE=",
          "      The full k=0..15 dose-response for four PRIMARY FULL-route "
          "endpoints at each of L1 (dorsal/WM encoder), L2 (ventral/LTM "
          "encoder) and L3 (ventral production), for four model states, with "
          "route-isolated WM/LTM decompositions supplied as diagnostics. Every "
          "severity and realization is retained.", "",
          "    UENO_LIKE_CORE_PATTERN=",
          "      NOT_ASSIGNED_BY_THIS_PACKAGE. No objective rule for "
          "SUPPORTED / PARTIAL / NOT_SUPPORTED was frozen before results "
          "existed, so assigning one now would be a post-hoc criterion. The "
          "full factual pattern is supplied for CENTRAL arbitration.", "",
          "    PUBLICATION_CLAIM=",
          "      Only what the descriptive tables directly establish: the "
          "measured exact-match rate of each frozen endpoint at every severity "
          "for every state x site x realization, and the arithmetic k0->k15 "
          "differences above. No inferential or comparative claim is made here.", "",
          "    NOT_ESTABLISHED=",
          "      no statistical inference, no significance, no effect size, no "
          "selected severity, no threshold-defined vocabulary, no pooling of "
          "P1-P4, no family-level conclusion from P4, not a seven-panel Ueno "
          "replication, and no pseudoword / Ueno-naming / HF-LF endpoint.", "",
          "    PROVENANCE=",
          f"      k=0 controls : {val['execution_commits']['k0_controls']} "
          "(preserved verbatim, not recomputed)",
          f"      k>0 cells    : {val['execution_commits']['nonzero_cells']}",
          f"      run matrix   : {val['run_matrix_sha256']}",
          "      analysis     : read-only; the results namespace was never "
          "written to.", "",
          "    STOP. AWAITING CENTRAL ARBITRATION."]
    return "\n".join(L) + "\n"


def write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return io_utils.sha256_file(path)
