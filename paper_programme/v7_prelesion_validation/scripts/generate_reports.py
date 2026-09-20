#!/usr/bin/env python
"""Generate the V7 pre-lesion reports FROM FROZEN RESULT FILES ONLY.

This program never imports torch, never loads a checkpoint, and never runs a
model. It reads `results/` and applies the frozen classification rules
mechanically (imported from the torch-free `rules` module, never redefined).

    python generate_reports.py --results <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import rules as pe                 # noqa: E402  frozen rules, imported not copied
#   NOTE: `rules` is torch-free by construction, so reporting is
#   structurally incapable of running a model.

SLOTS = ("P1", "P2", "P3", "P4")
ROUTES = ("full", "wm", "ltm")

# The EIGHT binary transition endpoints in SOURCE_POST_PAIRED.json. This is an
# explicit frozen schema, not a discovered one: reporting must fail closed if
# the schema drifts, never quietly skip an endpoint or mistake a scalar for a
# transition table.
PAIRED_TRANSITION_ENDPOINTS = (
    "FULL_canonical",
    "FULL_freear",
    "WM_canonical",
    "WM_freear",
    "LTM_canonical",
    "LTM_freear",
    "Naming",
    "C",
)
TRANSITION_KEYS = ("cc", "cw", "wc", "ww", "undefined")

# Present in the frozen file but NOT binary endpoints. Listed so their status
# is explicit rather than implied by absence.
PAIRED_CONTINUOUS_SUMMARIES = ("c_ltm_delta_summary", "g_delta_summary")
PAIRED_CONTINUOUS_SCALARS = ("c_ltm_n_changed", "g_n_changed")
PAIRED_METADATA = ("n_items_paired",)


class ReportSchemaError(RuntimeError):
    """The frozen paired-result schema is not what reporting requires."""


def load(results: str):
    summaries = json.load(open(os.path.join(results, "STATE_SUMMARY.json")))
    manifest = json.load(open(os.path.join(results, "RUN_MANIFEST.json")))
    return {s["state_id"]: s for s in summaries}, manifest


def _fmt(x, nd=6):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def realword_report(states, man) -> str:
    L = ["# V7 INTACT ROUTE VALIDATION — RESULTS", "",
         f"    runner commit   {man['runner_commit']}",
         f"    design anchor   {man['design_commit']}",
         f"    torch           {man['torch_version']}", "",
         "Generated from frozen result files only; no model was run.", "",
         "## Global and route endpoints", "",
         "| state | Rcan err | Rfree err | Naming err | C err | WMcan | WMfree | LTMcan | LTMfree |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sid, s in states.items():
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            sid, s["rep_canonical_full_errors"], s["rep_freear_full_errors"],
            s["naming_errors"], s["c_errors"],
            _fmt(s["rep_canonical_wm"], 4), _fmt(s["rep_freear_wm"], 4),
            _fmt(s["rep_canonical_ltm"], 4), _fmt(s["rep_freear_ltm"], 4)))

    L += ["", "## Frozen official POST battery invariant", "",
          "| slot | expected Rcan/Rfree/N/C | observed | match |", "|---|---|---|---|"]
    inv = man["frozen_official_post_battery_invariant"]
    for slot in SLOTS:
        s = states[f"{slot}_POST_REPAIR"]
        obs = (s["rep_canonical_full_errors"], s["rep_freear_full_errors"],
               s["naming_errors"], s["c_errors"])
        e = inv[slot]
        exp = (e["Rcan"], e["Rfree"], e["N"], e["C"])
        L.append(f"| {slot} | {'/'.join(map(str,exp))} | {'/'.join(map(str,obs))} | "
                 f"{'YES' if obs == exp else 'NO — STOP'} |")

    L += ["", "## Gating distributions (real-word R)", ""]
    for sid, s in states.items():
        for f in ("c_ltm", "g"):
            d = s.get(f"{f}_summary") or {}
            L.append(f"    {sid} {f}: n={d.get('n')} mean={_fmt(d.get('mean'))} "
                     f"sd={_fmt(d.get('sd'))} p05={_fmt(d.get('p05'))} "
                     f"p50={_fmt(d.get('p50'))} p95={_fmt(d.get('p95'))}")

    L += ["", "## Route pathology flags", ""]
    flags = pathology_flags(states)
    L += [f"    {sid}: {', '.join(f) if f else 'none'}" for sid, f in flags.items()]
    return "\n".join(L) + "\n"


def pathology_flags(states) -> dict:
    out = {}
    for sid, s in states.items():
        f = []
        for r in ("wm", "ltm"):
            if s.get(f"rep_canonical_{r}") == 0.0:
                f.append(f"ISOLATED_REALWORD_ROUTE_EXACT_ZERO:{r}")
            if s.get(f"pseudo_primary_{r}_no_eos") == s.get("pseudo_primary_n"):
                f.append(f"HUNDRED_PCT_NO_EOS:{r}")
        out[sid] = f
    return out


def pseudoword_report(states, man) -> str:
    L = ["# V7 PSEUDOWORD VALIDATION — RESULTS", "",
         "PRIMARY population: V7_COMMON_UNSEEN (N=378), identical across states.",
         "Decoder: genuine autonomous free-AR (no target-length termination).", "",
         "| state | WM acc | LTM acc | FULL acc | WM NED | LTM NED | delta_acc | delta_ned | ordering |",
         "|---|---|---|---|---|---|---|---|---|"]
    for sid, s in states.items():
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            sid, _fmt(s["pseudo_primary_wm_exact_acc"]),
            _fmt(s["pseudo_primary_ltm_exact_acc"]),
            _fmt(s["pseudo_primary_full_exact_acc"]),
            _fmt(s["pseudo_primary_wm_mean_ned"]),
            _fmt(s["pseudo_primary_ltm_mean_ned"]),
            _fmt(s["ordering_delta_acc"]), _fmt(s["ordering_delta_ned"]),
            s["ordering_ordering"]))
    L += ["", "## Arm-A preservation (P1/P2/P3 primary; P4 sensitivity)", ""]
    for slot in SLOTS:
        src = states[f"{slot}_SOURCE"]["ordering_ordering"]
        post = states[f"{slot}_POST_REPAIR"]["ordering_ordering"]
        tag = "" if slot != "P4" else "   (sensitivity only)"
        L.append(f"    {slot}: SOURCE={src} -> POST={post} : "
                 f"{pe.preservation(src, post)}{tag}")
    L += ["", "## Secondary sensitivity and descriptive strata", "",
          "    seed-specific unseen populations: see "
          "PSEUDOWORD_SEED_SENSITIVITY_ITEM_LEVEL.tsv (secondary only)",
          "    exposed-13 and trained-real-671: descriptive only; the 671-vs-378",
          "    comparison is a LEXICALITY/EXPOSURE contrast, never a pure causal",
          "    lexicality effect."]
    return "\n".join(L) + "\n"


def validate_paired_schema(paired: dict) -> None:
    """Fail closed unless every slot carries every expected transition endpoint.

    A missing endpoint, or an endpoint missing any of cc/cw/wc/ww/undefined, is
    an error -- never silently skipped.
    """
    for slot in SLOTS:
        if slot not in paired:
            raise ReportSchemaError(f"SOURCE_POST_PAIRED.json has no slot {slot}")
        d = paired[slot]
        for ep in PAIRED_TRANSITION_ENDPOINTS:
            if ep not in d:
                raise ReportSchemaError(
                    f"{slot}: expected transition endpoint {ep!r} is missing")
            t = d[ep]
            if not isinstance(t, dict):
                raise ReportSchemaError(
                    f"{slot}.{ep} is {type(t).__name__}, not a transition table")
            missing = [k for k in TRANSITION_KEYS if k not in t]
            if missing:
                raise ReportSchemaError(
                    f"{slot}.{ep} is missing transition keys {missing}")


def paired_report(results, states) -> str:
    path = os.path.join(results, "SOURCE_POST_PAIRED.json")
    if not os.path.exists(path):
        raise ReportSchemaError(f"missing frozen result file: {path}")
    paired = json.load(open(path))
    validate_paired_schema(paired)

    L = ["# V7 SOURCE -> POST_REPAIR PAIRED ANALYSIS", "",
         "Item-level transitions, read from the frozen SOURCE_POST_PAIRED.json.",
         "No model was run and no scientific result was recomputed.", ""]
    for slot in SLOTS:
        d = paired[slot]
        L += [f"## {slot}", "",
              f"    items paired: {d.get('n_items_paired')}", "",
              "| endpoint | c->c | c->w | w->c | w->w | undefined |",
              "|---|---|---|---|---|---|"]
        # iterate the FROZEN tuple, never the file's key order
        for ep in PAIRED_TRANSITION_ENDPOINTS:
            t = d[ep]
            L.append(f"| {ep} | {t['cc']} | {t['cw']} | {t['wc']} | {t['ww']} "
                     f"| {t['undefined']} |")
        changed = {ep: d[ep].get("changed_item_ids", [])
                   for ep in PAIRED_TRANSITION_ENDPOINTS}
        L += ["", "    changed items per endpoint:"]
        for ep, ids in changed.items():
            L.append(f"      {ep}: n={len(ids)}"
                     + (f"  ids={', '.join(ids)}" if 0 < len(ids) <= 40
                        else ("  (see SOURCE_POST_PAIRED.json)" if ids else "")))

        # continuous quantities: reported AS continuous, never as transitions
        L += ["", "    continuous endpoints (not binary transitions):"]
        for key in PAIRED_CONTINUOUS_SUMMARIES:
            summ = d.get(key)
            if isinstance(summ, dict):
                L.append(f"      {key}: n={summ.get('n')} "
                         f"mean={_fmt(summ.get('mean'))} sd={_fmt(summ.get('sd'))} "
                         f"min={_fmt(summ.get('min'))} p50={_fmt(summ.get('p50'))} "
                         f"max={_fmt(summ.get('max'))}")
        for key in PAIRED_CONTINUOUS_SCALARS:
            if key in d:
                L.append(f"      {key}: {d[key]}")
        L.append("")

    L += ["## Interpretation categories", "",
          "    WHAT_ARM_A_REPAIRS              endpoints with w->c > 0",
          "    WHAT_ARM_A_PRESERVES            endpoints with c->w == 0",
          "    WHAT_ARM_A_CHANGES_INCIDENTALLY non-C endpoints with any change",
          "",
          "Arm A is NOT assumed to affect only C.", ""]
    for slot in SLOTS:
        d = paired[slot]
        rep = [ep for ep in PAIRED_TRANSITION_ENDPOINTS if d[ep]["wc"] > 0]
        pres = [ep for ep in PAIRED_TRANSITION_ENDPOINTS if d[ep]["cw"] == 0]
        inc = [ep for ep in PAIRED_TRANSITION_ENDPOINTS
               if ep != "C" and (d[ep]["wc"] or d[ep]["cw"])]
        L += [f"### {slot}",
              f"    WHAT_ARM_A_REPAIRS              {', '.join(rep) or 'none'}",
              f"    WHAT_ARM_A_PRESERVES            {', '.join(pres) or 'none'}",
              f"    WHAT_ARM_A_CHANGES_INCIDENTALLY {', '.join(inc) or 'none'}", ""]
    return "\n".join(L) + "\n"


def classify(states) -> dict:
    """Mechanical application of the frozen classification rules."""
    core = [f"{s}_POST_REPAIR" for s in ("P1", "P2", "P3")]
    orderings = {s: states[s]["ordering_ordering"] for s in core}
    pres = {s: pe.preservation(states[f"{s}_SOURCE"]["ordering_ordering"],
                               states[f"{s}_POST_REPAIR"]["ordering_ordering"])
            for s in ("P1", "P2", "P3")}
    family = ("YES" if all(v in ("PRESERVED", "PRESERVED_FROM_NONDOMINANT_SOURCE")
                           for v in pres.values()) else "NO")
    flags = pathology_flags(states)
    all_wm = all(v == "WM_DOMINANT" for v in orderings.values())
    any_flag = any(flags[s] for s in core)
    if all_wm and family == "YES" and not any_flag:
        status = "V7_LESION_READY"
    elif any(v == "LTM_DOMINANT" for v in orderings.values()):
        status = "V7_NOT_LESION_READY"
    else:
        status = "V7_LESION_READY_WITH_QUALIFICATION"
    return {"PRELESION_VALIDATION_STATUS": status,
            "post_orderings": orderings, "preservation": pres,
            "ARM_A_PSEUDOWORD_PRESERVATION": family,
            "pathology_flags": flags}


def verify_frozen_results(results: str) -> dict:
    """Re-verify every file listed in FILE_SHA256SUMS. Reporting must never
    modify one; this is checked before AND after the reports are written."""
    import hashlib
    man = os.path.join(results, "FILE_SHA256SUMS")
    if not os.path.exists(man):
        raise ReportSchemaError(f"missing {man}")
    seen = {}
    for line in open(man):
        line = line.strip()
        if not line:
            continue
        want, rel = line.split(None, 1)
        p = os.path.join(results, rel.strip())
        if not os.path.exists(p):
            raise ReportSchemaError(f"frozen result missing: {rel}")
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != want:
            raise ReportSchemaError(
                f"FROZEN RESULT ALTERED: {rel}\n expected {want}\n got      {got}")
        seen[rel.strip()] = got
    return seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate reports from frozen results.")
    ap.add_argument("--results", default=os.path.join(PKG, "results"))
    a = ap.parse_args(argv)
    if not os.path.isdir(a.results):
        print(f"no results namespace: {a.results}", file=sys.stderr)
        return 2
    before = verify_frozen_results(a.results)
    states, man = load(a.results)
    out = os.path.join(a.results, "reports")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "V7_INTACT_ROUTE_VALIDATION_RESULTS.md"), "w").write(
        realword_report(states, man))
    open(os.path.join(out, "V7_PSEUDOWORD_VALIDATION_RESULTS.md"), "w").write(
        pseudoword_report(states, man))
    open(os.path.join(out, "V7_SOURCE_REPAIR_PAIRED_ANALYSIS.md"), "w").write(
        paired_report(a.results, states))
    cls = classify(states)
    json.dump(cls, open(os.path.join(out, "validation_classification.json"), "w"),
              indent=1)
    hand = ["# CENTRAL STEERING HANDOFF — V7 PRE-LESION VALIDATION", "",
            f"    PRELESION_VALIDATION_STATUS={cls['PRELESION_VALIDATION_STATUS']}", ""]
    for slot in SLOTS:
        hand.append(f"    {slot}_POST_REPAIR_ROUTE_PROFILE="
                    f"{states[f'{slot}_POST_REPAIR']['ordering_ordering']}")
    hand += ["",
             "    PSEUDOWORD_DORSAL_VENTRAL_ORDERING="
             + ("WM_DOMINANT_P1_P2_P3"
                if all(v == "WM_DOMINANT" for v in cls["post_orderings"].values())
                else "MIXED_SEE_REPORT"),
             f"    ARM_A_PSEUDOWORD_PRESERVATION={cls['ARM_A_PSEUDOWORD_PRESERVATION']}",
             "    GO_FOR_LESIONING_V2_IMPLEMENTATION=NO", "",
             "Derived mechanically from the frozen rules; no model was run during "
             "reporting.", "", "    STOP. RETURN TO CENTRAL."]
    open(os.path.join(out, "CENTRAL_STEERING_HANDOFF_V7_PRELESION_VALIDATION.md"),
         "w").write("\n".join(hand) + "\n")
    after = verify_frozen_results(a.results)
    if before != after:
        raise RuntimeError("REFUSED: a frozen result file changed during reporting")
    print(f"reports written to {out}")
    print(f"FROZEN_RESULTS_HASH_CHECK=PASS_ALL ({len(after)} files)")
    print(f"PRELESION_VALIDATION_STATUS={cls['PRELESION_VALIDATION_STATUS']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
