"""Aggregate gridsearch evaluation results and rank by FULL route metrics only.

Phase 2D / Phase 4 tool.

Walks --root for metrics.json files produced by scripts/evaluate_train_*.py,
aggregates them into TSV and JSON, and ranks runs using FULL route AR metrics
exclusively.  WM/LTM isolated metrics are kept in the output but never used
for ranking (they are diagnostic only — the double-dissociation makes isolated
route performance non-comparable across TF settings).

Supported metrics.json formats
--------------------------------
Nested (produced by evaluate_train_lexicon_ceiling.py):
    {
      "results": {
        "train": {"full": {"n_errors": N, "exact_match": F, "edit_dist": F,
                            "norm_edit_dist": F}, "wm": {...}, "ltm": {...}},
        "val":   {"full": {...}, "wm": {...}, "ltm": {...}}
      },
      "checkpoint": "...",       # path used to enrich with H / tf_ratio / mode
      "cfg_max_words": ...,
      "cfg_seed": ...,
      ...
    }

Flat (legacy / hand-crafted):
    {"full_ar_errors_train": N, "full_ar_exact_val": F, ...}
    Passed through unchanged.

Ranking rule (deterministic lexicographic tuple, no opaque scalar score)
-------------------------------------------------------------------------
  1. full_ar_errors_train   ASC   (fewer training errors = better)
  2. full_ar_exact_val      DESC  (higher val exact match = better)
  3. full_ar_edit_val       ASC   (lower val edit distance = better)
  4. full_ar_ned_val        ASC   (lower val normalised edit distance = better)
  5. H                      ASC   (smaller hidden size = less capacity = more interpretable)
  6. n_params               ASC   (smaller param count, if present in metrics.json)
  7. arch_pref              ASC   (0 for unigru_last_hidden, 1 for bigru_masked_mean;
                                   unigru preferred: no padding artifact, symmetric with WM)
  8. tf_ratio               ASC   (only after genuine equality on all above;
                                   smaller = harder training regime)

Isolated route metrics (wm_*, ltm_*): written to output TSV for diagnostics
    but NEVER appear in the rank key.
WFE full / WFE pseudoword full: written to output TSV as behavioral safeguards
    but do NOT drive the main ranking in Phase 4.

FULL route = model.forward() with the gate active (not route-isolated WM or LTM).

Required metrics (minimum for a run to be included)
----------------------------------------------------
  full_ar_errors_train   (int or float)  — from results.train.full.n_errors
  full_ar_exact_val      (float, 0-1)    — from results.val.full.exact_match

Optional metadata extracted from the checkpoint (if checkpoint path is present):
  H                 int    hidden size (from cfg_wm.hidden)
  ltm_encoder_mode  str    architecture mode (from cfg_ltm.ltm_encoder_mode)
  tf_ratio          float  teacher forcing ratio (from cfg_train.teacher_forcing_ratio)
  premotor_dim      int    (from checkpoint top-level key)

Output files (written to --out_dir)
------------------------------------
  all_runs.tsv      : one row per run, all metrics, sorted by ckpt path
  ranked_runs.tsv   : same rows, sorted by the ranking tuple above
  aggregate_summary.json : top-10 runs + global stats + list of skipped runs

Usage
-----
    python scripts/aggregate_gridsearch.py \\
        --root outputs/gridsearch \\
        --out_dir outputs/gridsearch/summary
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# --------------------------------------------------------------------------- #
# Nested → flat normalisation
# --------------------------------------------------------------------------- #

def _nested_to_flat(raw: dict) -> dict:
    """Flatten evaluate_train_lexicon_ceiling.py nested metrics.json to flat schema.

    If ``raw`` has no ``"results"`` key it is already in the legacy flat format
    and is returned unchanged.

    Nested JSON paths → flat keys (both train and val splits):
        results.{split}.full.n_errors       → full_ar_errors_{split}
        results.{split}.full.exact_match    → full_ar_exact_{split}
        results.{split}.full.edit_dist      → full_ar_edit_{split}
        results.{split}.full.norm_edit_dist → full_ar_ned_{split}
        results.{split}.full.n_items        → full_ar_n_items_{split}
        results.{split}.wm.*                → wm_ar_*_{split}   (diagnostic)
        results.{split}.ltm.*               → ltm_ar_*_{split}  (diagnostic)

    Only keys that are actually present in the nested dict are emitted; absent
    optional sub-dicts (e.g. a run without --include_val) produce no flat keys,
    which lets validate_run correctly reject incomplete runs.
    """
    if "results" not in raw:
        return raw  # legacy flat format — pass through

    flat: dict = {}

    # --- top-level metadata ---
    _META_KEYS = (
        "checkpoint", "decode_mode", "wm_noise_enabled", "deterministic",
        "glove_present", "lexicon_source", "n_train", "n_val",
        "cfg_max_words", "cfg_epochs", "cfg_seed", "evaluation_note",
        "splits_evaluated",
    )
    for k in _META_KEYS:
        if k in raw:
            flat[k] = raw[k]

    results = raw.get("results", {})

    _KEY_MAP = {
        "n_errors":      "errors",
        "exact_match":   "exact",
        "edit_dist":     "edit",
        "norm_edit_dist": "ned",
        "n_items":       "n_items",
        "phoneme_acc":   "phoneme_acc",
    }

    def _extract(split: str, route: str, prefix: str) -> None:
        d = results.get(split, {}).get(route, {})
        if not d:
            return
        for src_key, dst_suffix in _KEY_MAP.items():
            if src_key in d:
                flat[f"{prefix}_{dst_suffix}_{split}"] = d[src_key]

    for split in ("train", "val"):
        _extract(split, "full", "full_ar")
        _extract(split, "wm",   "wm_ar")
        _extract(split, "ltm",  "ltm_ar")

    return flat


def _enrich_from_checkpoint(m: dict) -> dict:
    """Attempt to add H, ltm_encoder_mode, tf_ratio from the saved checkpoint.

    Silently no-ops if torch is unavailable, the checkpoint path is absent,
    the file has moved, or loading fails for any reason.  Uses setdefault so
    values already present in the flat dict (from a future eval-script update)
    are never overwritten.
    """
    ckpt_path = m.get("checkpoint", "")
    if not ckpt_path or not os.path.exists(ckpt_path):
        return m
    try:
        import torch  # optional dependency for aggregator
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_wm  = ckpt.get("cfg_wm",    {})
        cfg_ltm = ckpt.get("cfg_ltm",   {})
        cfg_tr  = ckpt.get("cfg_train", {})
        if "hidden" in cfg_wm:
            m.setdefault("H", int(cfg_wm["hidden"]))
        if "ltm_encoder_mode" in cfg_ltm:
            m.setdefault("ltm_encoder_mode", cfg_ltm["ltm_encoder_mode"])
        if "teacher_forcing_ratio" in cfg_tr:
            m.setdefault("tf_ratio", float(cfg_tr["teacher_forcing_ratio"]))
        if "premotor_dim" in ckpt:
            m.setdefault("premotor_dim", int(ckpt["premotor_dim"]))
    except Exception:
        pass
    return m


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def find_metrics_files(root: str) -> list[str]:
    """Recursively find all metrics.json files under root."""
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname == "metrics.json":
                hits.append(os.path.join(dirpath, fname))
    return sorted(hits)


def load_run(metrics_path: str) -> dict:
    """Load a metrics.json, normalise to flat schema, enrich with checkpoint metadata.

    Supports both the nested format produced by evaluate_train_lexicon_ceiling.py
    and the legacy flat format (pass-through).
    """
    with open(metrics_path) as f:
        raw = json.load(f)
    flat = _nested_to_flat(raw)
    flat.setdefault("ckpt_dir", os.path.dirname(metrics_path))
    flat = _enrich_from_checkpoint(flat)
    return flat


# --------------------------------------------------------------------------- #
# Metric extraction
# --------------------------------------------------------------------------- #

_REQUIRED_METRICS = ("full_ar_errors_train", "full_ar_exact_val")


def validate_run(m: dict, path: str) -> bool:
    """Return True iff the flat dict contains the minimum metrics for ranking.

    Required (must be present after normalisation):
        full_ar_errors_train   — from results.train.full.n_errors
        full_ar_exact_val      — from results.val.full.exact_match

    A run is skipped (not just downranked) if these are absent, because ranking
    without them is undefined.  This catches runs that evaluated only the train
    split (--include_val not passed) or that are from an unrelated tool.
    """
    missing = [k for k in _REQUIRED_METRICS if k not in m]
    if missing:
        print(
            f"  [skip] {path}: missing required keys {missing}  "
            f"(nested path: full_ar_errors_train=results.train.full.n_errors, "
            f"full_ar_exact_val=results.val.full.exact_match)",
            file=sys.stderr,
        )
        return False
    return True


_ARCH_PREF = {
    "unigru_last_hidden": 0,   # preferred: no padding artifact, symmetric with WM
    "bigru_masked_mean":  1,   # historical baseline
}


def rank_key(m: dict) -> tuple:
    """Return the deterministic lexicographic sort key for a run.  Lower = better.

    Tuple positions (see module docstring for the full ranking rule):
        (errors_train, -exact_val, edit_val, ned_val, H, n_params, arch_pref, tf_ratio)

    WM/LTM isolated metrics are intentionally absent from this key.
    """
    errors_train = float(m.get("full_ar_errors_train", float("inf")))
    exact_val    = float(m.get("full_ar_exact_val",    0.0))
    edit_val     = float(m.get("full_ar_edit_val",     float("inf")))
    ned_val      = float(m.get("full_ar_ned_val",      float("inf")))
    H            = int(m.get("H",                      999))
    n_params     = int(m.get("n_params",               999_999_999))
    mode         = m.get("ltm_encoder_mode",           "")
    arch_pref    = _ARCH_PREF.get(mode, 2)   # unknown architecture sorts last
    tf_ratio     = float(m.get("tf_ratio",             1.0))
    return (errors_train, -exact_val, edit_val, ned_val, H, n_params, arch_pref, tf_ratio)


# --------------------------------------------------------------------------- #
# TSV output helpers
# --------------------------------------------------------------------------- #

def _all_columns(runs: list[dict]) -> list[str]:
    """Stable column order: required metrics first, rest alphabetically."""
    priority = [
        "ckpt_dir",
        # FULL route — train
        "full_ar_errors_train", "full_ar_exact_train",
        "full_ar_edit_train",   "full_ar_ned_train",
        # FULL route — val
        "full_ar_errors_val",   "full_ar_exact_val",
        "full_ar_edit_val",     "full_ar_ned_val",
        # run metadata
        "H", "tf_ratio", "ltm_encoder_mode", "gate_threshold",
        "ventral_noise", "interference_noise",
        # WFE safeguards
        "full_ar_wfe_train", "full_ar_wfe_val",
        # isolated routes (diagnostic only)
        "wm_ar_errors_train",  "wm_ar_exact_train",
        "wm_ar_errors_val",    "wm_ar_exact_val",
        "ltm_ar_errors_train", "ltm_ar_exact_train",
        "ltm_ar_errors_val",   "ltm_ar_exact_val",
    ]
    all_keys = set()
    for m in runs:
        all_keys.update(m.keys())
    tail = sorted(all_keys - set(priority))
    return [c for c in priority if c in all_keys] + tail


def write_tsv(runs: list[dict], path: str) -> None:
    cols = _all_columns(runs)
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for m in runs:
            f.write("\t".join(str(m.get(c, "")) for c in cols) + "\n")
    print(f"[aggregate] wrote {len(runs)} rows -> {path}")


# --------------------------------------------------------------------------- #
# Summary stats
# --------------------------------------------------------------------------- #

def _numeric_stats(runs: list[dict], key: str) -> dict:
    vals = [float(m[key]) for m in runs if key in m and m[key] != ""]
    if not vals:
        return {}
    return {
        "min": min(vals), "max": max(vals),
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "n": len(vals),
    }


def build_summary(ranked: list[dict], skipped: list[dict] | None = None) -> dict:
    metric_keys = [k for k in _all_columns(ranked)
                   if k not in ("ckpt_dir",) and ranked
                   and isinstance(ranked[0].get(k, ""), (int, float))]
    return {
        "n_runs": len(ranked),
        "n_skipped": len(skipped) if skipped else 0,
        "skipped_runs": skipped or [],
        "top10": ranked[:10],
        "ranking_rule": {
            "tuple": "(errors_train ASC, -exact_val ASC, edit_val ASC, ned_val ASC, "
                     "H ASC, n_params ASC, arch_pref ASC, tf_ratio ASC)",
            "1_errors_train":  "full_ar_errors_train ASC (primary)",
            "2_exact_val":     "full_ar_exact_val DESC",
            "3_edit_val":      "full_ar_edit_val ASC",
            "4_ned_val":       "full_ar_ned_val ASC",
            "5_H":             "hidden_size ASC",
            "6_n_params":      "n_params ASC",
            "7_arch_pref":     "unigru_last_hidden=0 preferred over bigru_masked_mean=1",
            "8_tf_ratio":      "tf_ratio ASC (only after genuine equality above)",
            "isolation_note":  "WM/LTM isolated metrics: diagnostic only, never in rank key",
            "wfe_note":        "WFE full reported as behavioral safeguard, not in rank key (Phase 4)",
        },
        "json_paths": {
            "full_ar_errors_train": "results.train.full.n_errors",
            "full_ar_exact_train":  "results.train.full.exact_match",
            "full_ar_edit_train":   "results.train.full.edit_dist",
            "full_ar_ned_train":    "results.train.full.norm_edit_dist",
            "full_ar_errors_val":   "results.val.full.n_errors",
            "full_ar_exact_val":    "results.val.full.exact_match",
            "full_ar_edit_val":     "results.val.full.edit_dist",
            "full_ar_ned_val":      "results.val.full.norm_edit_dist",
        },
        "stats_per_metric": {k: _numeric_stats(ranked, k) for k in metric_keys},
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate gridsearch metrics.json files and rank by FULL-route metrics."
    )
    p.add_argument("--root", required=True,
                   help="Root directory to walk for metrics.json files.")
    p.add_argument("--out_dir", required=True,
                   help="Directory to write all_runs.tsv, ranked_runs.tsv, aggregate_summary.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.root):
        print(f"ERROR: --root not found: {args.root}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    paths = find_metrics_files(args.root)
    print(f"[aggregate] found {len(paths)} metrics.json files under {args.root}")

    runs = []
    skipped: list[dict] = []
    for p in paths:
        try:
            m = load_run(p)
        except (json.JSONDecodeError, OSError) as e:
            reason = f"load error: {e}"
            print(f"  [skip] {p}: {reason}", file=sys.stderr)
            skipped.append({"path": p, "reason": reason})
            continue
        if validate_run(m, p):
            runs.append(m)
        else:
            missing = [k for k in _REQUIRED_METRICS if k not in m]
            skipped.append({"path": p, "reason": f"missing required keys {missing}"})

    if skipped:
        print(f"[aggregate] {len(skipped)} run(s) skipped (see aggregate_summary.json "
              f"for details).", file=sys.stderr)

    if not runs:
        print("[aggregate] no valid runs found — nothing to write.", file=sys.stderr)
        summary_path = os.path.join(args.out_dir, "aggregate_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"n_runs": 0, "n_skipped": len(skipped),
                       "skipped_runs": skipped}, f, indent=2)
        sys.exit(0)

    all_sorted  = sorted(runs, key=lambda m: m.get("ckpt_dir", ""))
    ranked      = sorted(runs, key=rank_key)

    write_tsv(all_sorted, os.path.join(args.out_dir, "all_runs.tsv"))
    write_tsv(ranked,     os.path.join(args.out_dir, "ranked_runs.tsv"))

    summary = build_summary(ranked, skipped=skipped)
    summary_path = os.path.join(args.out_dir, "aggregate_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[aggregate] summary -> {summary_path}")

    # Print top-3 ranking to stdout
    print("\n=== TOP-3 RUNS (FULL-route ranking) ===")
    for i, m in enumerate(ranked[:3], 1):
        print(f"  #{i}: {m.get('ckpt_dir', '?')}")
        print(f"       full_ar_errors_train={m.get('full_ar_errors_train', '?')}"
              f"  full_ar_exact_val={m.get('full_ar_exact_val', '?')}")
        print(f"       H={m.get('H', '?')}  tf_ratio={m.get('tf_ratio', '?')}"
              f"  ltm_mode={m.get('ltm_encoder_mode', '?')}")
    print()


if __name__ == "__main__":
    main()
