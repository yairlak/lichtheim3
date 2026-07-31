"""Post-hoc enrichment of behavioral WFE predictions — no model, no inference.

Part of the additive behavioral-evaluation patch specified in
outputs/behavioral_wfe_fulllexicon_93a577f/audits/
minimal_behavioral_evaluation_patch_FINAL.md

Takes an `item_level_predictions.tsv` written by scripts/external_eval.py and
joins it 1:1 with the frozen analysis manifest, then derives ONLY the fields
the frozen required-output matrix declares derivable:

    per route : insertions, deletions, substitutions, predicted_length,
                premature_eos
    per item  : frequency_class, lichtheim_exposure_status, the Dager
                membership fields, and analysis-set membership flags

Deliberately NOT derived here:
  * gate, lexical_confidence, eos_position — the frozen matrix marks these as
    requiring instrumentation at decode time.  They are read through from the
    prediction file if present and never reconstructed;
  * any serial-position / Error_Indices column.  Figure 2C
    (SCRIPT_FAITHFUL_SERIAL_POSITION) must be built from the stored sequences
    with the Dager zip-mismatch rule, NOT from edit-operation alignments;
    computing positions here would invite exactly that confusion.

Usage:
    python scripts/enrich_behavioral_predictions.py \\
        --pred     outputs/<run>/wfe_ar/item_level_predictions.tsv \\
        --manifest outputs/behavioral_wfe_fulllexicon_93a577f/audits/wfe_analysis_item_manifest.tsv \\
        --sets     outputs/behavioral_wfe_fulllexicon_93a577f/audits/wfe_analysis_set_membership.tsv \\
        --out      outputs/<run>/wfe_ar/item_level_predictions_enriched.tsv
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MANIFEST_DEFAULT = os.path.join(
    ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f", "audits",
    "wfe_analysis_item_manifest.tsv")
SETS_DEFAULT = os.path.join(
    ROOT, "outputs", "behavioral_wfe_fulllexicon_93a577f", "audits",
    "wfe_analysis_set_membership.tsv")

ROUTES = ("full", "wm", "ltm")

# Frozen frequency thresholds (Dager WFE binning, protocol freeze 2026-07-30).
ZIPF_LOW_MAX = 3.5
ZIPF_HIGH_MIN = 4.0

# Manifest columns copied through verbatim (never recomputed here).
MANIFEST_PASSTHROUGH = [
    "lichtheim_exposure_status",
    "lichtheim_training_orthography",
    "lichtheim_training_phonology",
    "lichtheim_same_word_same_pronunciation",
    "lichtheim_training_homophones",
    "dager_training_orthography",
    "dager_training_phonology",
    "dager_collision_status",
    "dager_discrepancy_category",
]

# ---------------------------------------------------------------------------
# Edit operations — backend policy
# ---------------------------------------------------------------------------
#
# The faithful publication path MUST use the same backend as Dager et al.
# Evidence from the original repository (danieldager/swp-model @ dc09eb3):
#     requirements.txt:13        levenshtein>=0.26.1
#     swp/utils/datasets.py:11   from Levenshtein import editops
#     scripts/jeanzay_setup.sh:18 pip install --upgrade --no-cache-dir levenshtein
# Note the distribution is `Levenshtein` (rapidfuzz-backed), NOT the legacy
# `python-Levenshtein` wrapper: they can resolve tied optimal alignments
# differently (Dager U11).
#
# SCOPE of that difference — narrower than it may appear:
#   * Figures 2A and 2B are NOT affected.  They use only the TOTAL raw edit
#     distance, which is identical across all optimal alignments.
#   * Figure 2C is NOT affected.  It uses the original zip-mismatch
#     Error_Indices method, which performs no alignment at all.
#   * The backend DOES matter for the insertion/deletion/substitution
#     taxonomy, for Figure 8-style error-type analyses, and for any separately
#     labelled aligned-error extension — i.e. exactly the quantities this
#     script derives.
#
# MODE_FAITHFUL  : requires Levenshtein.editops; fails loudly if unavailable.
# MODE_ADAPTED   : permits the internal deterministic traceback, and every
#                  output is explicitly labelled non-publication.

MODE_FAITHFUL = "faithful"
MODE_ADAPTED = "adapted"

FAITHFUL_BACKEND = "Levenshtein.editops"
ADAPTED_BACKEND = "internal_dp_traceback"
REQUIRED_BACKEND_SPEC = "Levenshtein>=0.26.1"

try:  # pragma: no cover - depends on the environment
    import Levenshtein as _lev  # type: ignore
except ImportError:  # pragma: no cover
    _lev = None


class FaithfulBackendUnavailable(RuntimeError):
    """Raised when the faithful path cannot use Dager's editops backend.

    Never downgraded to a warning: a silent fallback would change the
    insertion/deletion/substitution taxonomy — and any Figure 8-style
    error-type analysis built on it — without leaving a trace.  (Totals are
    alignment-invariant, so Figures 2A/2B/2C are not at risk here.)
    """


def _levenshtein_version() -> Optional[str]:
    if _lev is None:
        return None
    try:
        import importlib.metadata as _md
        return _md.version("Levenshtein")
    except Exception:  # pragma: no cover
        return getattr(_lev, "__version__", None)


def resolve_editops_backend(mode: str) -> dict:
    """Backend record for `mode`.  Raises in faithful mode if unavailable."""
    if mode not in (MODE_FAITHFUL, MODE_ADAPTED):
        raise ValueError(f"unknown mode {mode!r}; "
                         f"expected {MODE_FAITHFUL!r} or {MODE_ADAPTED!r}")

    if mode == MODE_FAITHFUL:
        if _lev is None or not hasattr(_lev, "editops"):
            raise FaithfulBackendUnavailable(
                f"faithful mode requires {REQUIRED_BACKEND_SPEC} "
                f"(`from Levenshtein import editops`), the backend used by "
                f"Dager et al. (swp-model requirements.txt:13). It is not "
                f"importable in this environment.\n"
                f"  install it:  python -m pip install '{REQUIRED_BACKEND_SPEC}'\n"
                f"  or run with --mode {MODE_ADAPTED} to use the internal "
                f"deterministic traceback — whose output is labelled "
                f"NON-PUBLICATION and must not be used for the "
                f"insertion/deletion/substitution taxonomy, Figure 8-style "
                f"error-type analyses, or any aligned-error extension.")
        return {
            "analysis_mode": MODE_FAITHFUL,
            "editops_backend": FAITHFUL_BACKEND,
            "editops_backend_package": "Levenshtein",
            "editops_backend_version": _levenshtein_version(),
            "editops_backend_required_spec": REQUIRED_BACKEND_SPEC,
            "publication_eligible": True,
            "editops_tie_breaking": (
                "library-internal traceback (Levenshtein/rapidfuzz); identical "
                "to the backend used by Dager et al."),
            "affects": (
                "insertion/deletion/substitution taxonomy; Figure 8-style "
                "error-type analyses; any separately labelled aligned-error "
                "extension"),
            "does_not_affect": (
                "Figures 2A/2B (total raw edit distance only, "
                "alignment-invariant) and Figure 2C (zip-mismatch "
                "Error_Indices, no alignment)"),
        }

    return {
        "analysis_mode": MODE_ADAPTED,
        "editops_backend": ADAPTED_BACKEND,
        "editops_backend_package": None,
        "editops_backend_version": None,
        "editops_backend_required_spec": REQUIRED_BACKEND_SPEC,
        "publication_eligible": False,
        "ADAPTED_NON_PUBLICATION": True,
        "editops_tie_breaking": (
            "internal deterministic traceback: match > replace > delete > "
            "insert; only the operation TOTAL is alignment-invariant "
            "(Dager U11). NOT the Dager backend."),
        "affects": (
            "insertion/deletion/substitution taxonomy; Figure 8-style "
            "error-type analyses; any separately labelled aligned-error "
            "extension"),
        "does_not_affect": (
            "Figures 2A/2B (total raw edit distance only, alignment-invariant) "
            "and Figure 2C (zip-mismatch Error_Indices, no alignment)"),
        "warning": (
            "ADAPTED MODE — operation counts were NOT produced with the Dager "
            "editops backend and must not be used for the "
            "insertion/deletion/substitution taxonomy, Figure 8-style "
            "error-type analyses, or any aligned-error extension. Totals are "
            "alignment-invariant, so Figures 2A/2B/2C are unaffected."),
    }


def _editops_internal(src: List[str], dst: List[str]) -> List[str]:
    """Edit-operation script turning `src` into `dst` (Dager's direction:
    src = gold target, dst = prediction).

    Returns a list of "replace" / "delete" / "insert" labels.

    Tie-breaking: several optimal alignments generally exist and the split
    among replace/delete/insert is NOT alignment-invariant (only their total,
    the edit distance, is).  python-Levenshtein resolves ties inside its C
    traceback and documents no rule (Dager audit, U11).  When that library is
    unavailable this deterministic traceback is used instead, preferring, at
    each step: diagonal match, then replace, then delete, then insert.  The
    backend actually used is recorded in the output as `editops_backend`.
    """
    n, m = len(src), len(dst)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if src[i - 1] == dst[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost,
                           dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1)

    ops: List[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (i > 0 and j > 0 and src[i - 1] == dst[j - 1]
                and dp[i][j] == dp[i - 1][j - 1]):
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append("replace")
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append("delete")
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append("insert")
            j -= 1
        else:  # pragma: no cover - unreachable for a consistent DP table
            raise AssertionError("inconsistent edit-distance table")
    ops.reverse()
    return ops


def edit_operations(target_syms: List[str], pred_syms: List[str],
                    mode: str = MODE_FAITHFUL) -> Dict[str, int]:
    """{'insertions','deletions','substitutions','edit_distance'} for one item.

    Operand convention matches Dager `enrich_for_plotting`: atomic ARPAbet
    tokens, source = gold target, destination = prediction.

    mode=MODE_FAITHFUL (default) uses Dager's `Levenshtein.editops` and raises
    FaithfulBackendUnavailable if it cannot be imported — it never falls back
    silently.  mode=MODE_ADAPTED uses the internal traceback and its results
    are labelled non-publication by the caller.
    """
    if mode == MODE_FAITHFUL:
        resolve_editops_backend(MODE_FAITHFUL)      # raises if unavailable
        ops = [op for op, _, _ in _lev.editops(target_syms, pred_syms)]
    elif mode == MODE_ADAPTED:
        ops = _editops_internal(target_syms, pred_syms)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    c = collections.Counter(ops)
    return {"insertions": c["insert"], "deletions": c["delete"],
            "substitutions": c["replace"], "edit_distance": len(ops)}


def frequency_class(lexicality: str, zipf: object) -> str:
    """Frozen WFE frequency binning: low <= 3.5, high >= 4.0."""
    if str(lexicality).strip() != "real":
        return "n/a_pseudoword"
    try:
        z = float(zipf)
    except (TypeError, ValueError):
        return "missing_zipf"
    if z != z:  # NaN
        return "missing_zipf"
    if z <= ZIPF_LOW_MAX:
        return "low"
    if z >= ZIPF_HIGH_MIN:
        return "high"
    return "ambiguous_3.5_4.0"


# ---------------------------------------------------------------------------

def enrich(pred_df: pd.DataFrame, manifest_df: pd.DataFrame,
           sets_df: Optional[pd.DataFrame],
           routes: Tuple[str, ...] = ROUTES,
           mode: str = MODE_FAITHFUL) -> Tuple[pd.DataFrame, dict]:
    """Return (enriched dataframe, report).  Raises on any join defect.

    mode=MODE_FAITHFUL (default, publication path) fails fast unless Dager's
    Levenshtein.editops backend is importable.
    """
    backend = resolve_editops_backend(mode)     # fail-fast before any work
    pred_ids = pred_df["item_id"].tolist()
    man_ids = manifest_df["item_id"].tolist()

    if len(set(pred_ids)) != len(pred_ids):
        dup = [k for k, v in collections.Counter(pred_ids).items() if v > 1]
        raise AssertionError(f"duplicate item_id in predictions: {dup[:10]}")
    if len(set(man_ids)) != len(man_ids):
        dup = [k for k, v in collections.Counter(man_ids).items() if v > 1]
        raise AssertionError(f"duplicate item_id in manifest: {dup[:10]}")
    missing = sorted(set(pred_ids) - set(man_ids))
    if missing:
        raise AssertionError(
            f"{len(missing)} predicted items absent from the manifest "
            f"(no item may be silently dropped): {missing[:10]}")

    out = pred_df.copy()
    man_idx = manifest_df.set_index("item_id")

    # --- manifest passthrough (copied exactly, never recomputed) ----------
    for col in MANIFEST_PASSTHROUGH:
        if col in man_idx.columns:
            out[col] = [man_idx.at[i, col] for i in pred_ids]

    # --- derived per-item -------------------------------------------------
    lex_col = "lexicality" if "lexicality" in out.columns else "source_lexicality"
    out["frequency_class"] = [
        frequency_class(l, z)
        for l, z in zip(out[lex_col], out.get("zipf_frequency", [None] * len(out)))
    ]

    # --- derived per route ------------------------------------------------
    for route in routes:
        pcol, tcol = f"{route}_predicted", f"{route}_target"
        if pcol not in out.columns or tcol not in out.columns:
            continue
        ins, dele, sub, plen, prem = [], [], [], [], []
        for pred_s, tgt_s in zip(out[pcol].fillna(""), out[tcol].fillna("")):
            p = str(pred_s).split()
            t = str(tgt_s).split()
            ops = edit_operations(t, p, mode=mode)
            ins.append(ops["insertions"])
            dele.append(ops["deletions"])
            sub.append(ops["substitutions"])
            plen.append(len(p))
            prem.append(len(p) < len(t))
        out[f"{route}_insertions"] = ins
        out[f"{route}_deletions"] = dele
        out[f"{route}_substitutions"] = sub
        out[f"{route}_predicted_length"] = plen
        out[f"{route}_premature_eos"] = prem

    # --- analysis-set membership flags ------------------------------------
    set_names: List[str] = []
    if sets_df is not None:
        set_names = sorted(sets_df["analysis_set"].unique())
        by_set = {s: set(sets_df.loc[sets_df["analysis_set"] == s, "item_id"])
                  for s in set_names}
        for s in set_names:
            out[f"in_{s}"] = [i in by_set[s] for i in pred_ids]

    # Per-row backend provenance: every row carries the mode it was built with,
    # so a faithful and an adapted table can never be silently concatenated.
    out["analysis_mode"] = backend["analysis_mode"]
    out["editops_backend"] = backend["editops_backend"]
    out["editops_backend_version"] = backend["editops_backend_version"] or ""

    report = {
        "n_prediction_rows": len(pred_df),
        "n_manifest_rows": len(manifest_df),
        "n_joined": len(out),
        "join_is_one_to_one": len(out) == len(pred_df),
        "n_unmatched_predictions": 0,
        "manifest_items_not_evaluated": len(set(man_ids) - set(pred_ids)),
        "routes_enriched": [r for r in routes
                            if f"{r}_predicted" in pred_df.columns],
        **backend,
        "dager_backend_evidence": {
            "repository": "danieldager/swp-model @ dc09eb3",
            "requirements.txt:13": "levenshtein>=0.26.1",
            "swp/utils/datasets.py:11": "from Levenshtein import editops",
            "scripts/jeanzay_setup.sh:18":
                "pip install --upgrade --no-cache-dir levenshtein",
        },
        "analysis_sets": set_names,
        "derived_fields_only_from_frozen_matrix": True,
        "serial_position_fields_derived": False,
        "serial_position_note": (
            "Figure 2C (SCRIPT_FAITHFUL_SERIAL_POSITION) uses the Dager "
            "zip-mismatch rule on the stored sequences and is NOT derived "
            "from these edit operations"),
    }

    # Sanity: operation counts must reconstruct the stored edit distance.
    for route in report["routes_enriched"]:
        lhs = (out[f"{route}_insertions"] + out[f"{route}_deletions"]
               + out[f"{route}_substitutions"])
        rhs = out[f"{route}_edit_dist"]
        bad = int((lhs != rhs).sum())
        if bad:
            raise AssertionError(
                f"{route}: {bad} items where insertions+deletions+"
                f"substitutions != stored edit distance")
    return out, report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--manifest", default=MANIFEST_DEFAULT)
    p.add_argument("--sets", default=SETS_DEFAULT)
    p.add_argument("--out", default=None)
    p.add_argument("--mode", default=MODE_FAITHFUL,
                   choices=[MODE_FAITHFUL, MODE_ADAPTED],
                   help=f"{MODE_FAITHFUL} (default, publication path): requires "
                        f"{REQUIRED_BACKEND_SPEC} and fails if absent. "
                        f"{MODE_ADAPTED}: internal traceback, output labelled "
                        f"NON-PUBLICATION.")
    args = p.parse_args()

    try:
        backend = resolve_editops_backend(args.mode)
    except FaithfulBackendUnavailable as exc:
        print(f"\nFATAL: {exc}\n", file=sys.stderr)
        return 2
    if args.mode == MODE_ADAPTED:
        print("=" * 72, file=sys.stderr)
        print("WARNING — ADAPTED MODE (NON-PUBLICATION)", file=sys.stderr)
        print(backend["warning"], file=sys.stderr)
        print("=" * 72, file=sys.stderr)

    pred_df = pd.read_csv(args.pred, sep="\t")
    manifest_df = pd.read_csv(args.manifest, sep="\t")
    sets_df = (pd.read_csv(args.sets, sep="\t")
               if args.sets and os.path.exists(args.sets) else None)

    out_df, report = enrich(pred_df, manifest_df, sets_df, mode=args.mode)

    suffix = "" if args.mode == MODE_FAITHFUL else "_ADAPTED_NON_PUBLICATION"
    out_path = args.out or args.pred.replace(
        ".tsv", f"_enriched{suffix}.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)
    report_path = out_path.replace(".tsv", "_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"-> {out_path} ({len(out_df)} rows, {len(out_df.columns)} columns)")
    print(f"-> {report_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
