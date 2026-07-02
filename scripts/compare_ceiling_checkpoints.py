"""Task 5: Compare error sets across two ceiling evaluation runs.

Compares train_errors.tsv files from two different checkpoints and
reports which errors are common, which were fixed, and which are new.

Inputs:
    outputs/train_lexicon_ceiling_e60/train_errors.tsv
    outputs/train_lexicon_ceiling_e90/train_errors.tsv

Outputs:
    outputs/train_lexicon_ceiling_comparison/comparison.json
    outputs/train_lexicon_ceiling_comparison/comparison.md
    outputs/train_lexicon_ceiling_comparison/error_overlap.tsv

Usage:
    python scripts/compare_ceiling_checkpoints.py \\
        --a outputs/train_lexicon_ceiling_e60/train_errors.tsv \\
        --b outputs/train_lexicon_ceiling_e90/train_errors.tsv \\
        --label_a e60 --label_b e90

    # Or using --a and --b as checkpoint names (will look up default error paths):
    python scripts/compare_ceiling_checkpoints.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_A = os.path.join(ROOT, "outputs", "train_lexicon_ceiling_e60", "train_errors.tsv")
DEFAULT_B = os.path.join(ROOT, "outputs", "train_lexicon_ceiling_e90", "train_errors.tsv")
OUT_DIR   = os.path.join(ROOT, "outputs", "train_lexicon_ceiling_comparison")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a",       default=DEFAULT_A,
                   help="path to first  checkpoint's train_errors.tsv")
    p.add_argument("--b",       default=DEFAULT_B,
                   help="path to second checkpoint's train_errors.tsv")
    p.add_argument("--label_a", default="ckpt_a",
                   help="label for first checkpoint (e.g. 'e60')")
    p.add_argument("--label_b", default="ckpt_b",
                   help="label for second checkpoint (e.g. 'e90')")
    p.add_argument("--out_dir", default=OUT_DIR)
    return p.parse_args()


def _load(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"  WARNING: {label} errors file not found: {path}")
        return None
    df = pd.read_csv(path, sep="\t")
    return df


def route_summary(df: pd.DataFrame | None, label: str) -> dict:
    if df is None or len(df) == 0:
        return {"n": 0}
    wm_ok  = int((df["wm_exact_match"]  == 1).sum()) if "wm_exact_match"  in df else None
    ltm_ok = int((df["ltm_exact_match"] == 1).sum()) if "ltm_exact_match" in df else None
    both_wrong = int(
        ((df["wm_exact_match"] == 0) & (df["ltm_exact_match"] == 0)).sum()
    ) if "wm_exact_match" in df else None
    return {
        "n":              len(df),
        "n_wm_correct":   wm_ok,
        "n_ltm_correct":  ltm_ok,
        "n_both_wrong":   both_wrong,
        "mean_length":    round(float(df["length"].mean()), 2) if "length" in df else None,
        "mean_rank":      round(float(df["rank"].mean()),   0)  if "rank"   in df else None,
    }


def compare(df_a: pd.DataFrame, df_b: pd.DataFrame,
            label_a: str, label_b: str) -> dict:
    words_a = set(df_a["word"].dropna()) if df_a is not None else set()
    words_b = set(df_b["word"].dropna()) if df_b is not None else set()

    common  = words_a & words_b
    fixed   = words_a - words_b   # in A but not B → fixed by B
    new     = words_b - words_a   # in B but not A → new error at B

    def sub(df, words):
        return df[df["word"].isin(words)] if df is not None else pd.DataFrame()

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_errors_a":    len(df_a) if df_a is not None else 0,
        "n_errors_b":    len(df_b) if df_b is not None else 0,
        "n_common":      len(common),
        "n_fixed":       len(fixed),
        "n_new":         len(new),
        "pct_common_of_a": round(100.0 * len(common) / max(len(words_a), 1), 1),
        "stability":     (
            "STABLE — same errors across checkpoints"
            if len(common) / max(len(words_a) + len(words_b), 1) > 0.8
            else (
                "MOVING — error sets differ substantially (likely non-determinism "
                "or warm-restart noise)"
            )
        ),
        "route_summary_a": route_summary(df_a, label_a),
        "route_summary_b": route_summary(df_b, label_b),
        "route_summary_common":  route_summary(sub(df_a, common),  f"common (from {label_a})"),
        "route_summary_fixed":   route_summary(sub(df_a, fixed),   f"fixed  (from {label_a})"),
        "route_summary_new":     route_summary(sub(df_b, new),     f"new    (from {label_b})"),
        "common_words":  sorted(common),
        "fixed_words":   sorted(fixed)[:50],
        "new_words":     sorted(new)[:50],
    }


def write_overlap_tsv(df_a, df_b, common, fixed, new,
                      label_a, label_b, path: str) -> None:
    rows = []
    for df, src_label, cat in [
        (df_a, label_a, "common"),
        (df_a, label_a, "fixed_in_b"),
        (df_b, label_b, "new_in_b"),
    ]:
        if df is None:
            continue
        if cat == "common":
            sub = df[df["word"].isin(common)]
        elif cat == "fixed_in_b":
            sub = df[df["word"].isin(fixed)]
        else:
            sub = df[df["word"].isin(new)]
        for _, row in sub.iterrows():
            r = row.to_dict()
            r["category"]   = cat
            r["from_ckpt"]  = src_label
            rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(path, sep="\t", index=False)
    print(f"  -> {path}")


def write_md(cmp: dict, path: str) -> None:
    a, b = cmp["label_a"], cmp["label_b"]
    lines = [
        "# Ceiling Checkpoint Comparison",
        "",
        f"Comparing **{a}** (source A) vs **{b}** (source B).",
        "",
        "## Error counts",
        "",
        f"| Checkpoint | Total errors |",
        f"|---|---|",
        f"| {a} | {cmp['n_errors_a']} |",
        f"| {b} | {cmp['n_errors_b']} |",
        "",
        "## Overlap analysis",
        "",
        f"| Category | Count | Notes |",
        f"|---|---|---|",
        f"| Common errors (both {a} and {b}) | {cmp['n_common']} | "
        f"stable residuals |",
        f"| Fixed by {b} (in {a}, not in {b}) | {cmp['n_fixed']} | "
        f"genuine improvement |",
        f"| New in {b} (not in {a}) | {cmp['n_new']} | "
        f"regression / noise |",
        "",
        f"**{cmp['pct_common_of_a']}% of {a} errors are also in {b}.**",
        "",
        f"**Stability verdict: {cmp['stability']}**",
        "",
        "## Route analysis per category",
        "",
        "| Category | n | WM correct | LTM correct | Both wrong |",
        "|---|---|---|---|---|",
    ]
    for key, label in [
        ("route_summary_a",       a),
        ("route_summary_b",       b),
        ("route_summary_common",  "common"),
        ("route_summary_fixed",   f"fixed by {b}"),
        ("route_summary_new",     f"new in {b}"),
    ]:
        d = cmp[key]
        lines.append(
            f"| {label} | {d['n']} "
            f"| {d.get('n_wm_correct', '—')} "
            f"| {d.get('n_ltm_correct', '—')} "
            f"| {d.get('n_both_wrong', '—')} |"
        )

    lines += [
        "",
        "## Common errors (stable residuals)",
        "",
        f"{cmp['n_common']} words fail in both checkpoints.  These are the true",
        "hard residual cases that need more training.",
        "",
        "| Word |",
        "|---|",
    ]
    for w in cmp["common_words"][:30]:
        lines.append(f"| {w} |")
    if len(cmp["common_words"]) > 30:
        lines.append(f"| *(+{len(cmp['common_words'])-30} more)* |")

    lines += [
        "",
        "## Interpretation",
        "",
        f"- Only {cmp['n_common']} errors are shared between {a} and {b}.",
        f"- {cmp['n_fixed']} errors present in {a} are absent from {b} — "
        f"but {cmp['n_new']} new errors appeared in {b}.",
        "- This pattern is consistent with **non-deterministic evaluation** "
        "(WM interference noise changing which border-case items pass each run),",
        "  **and/or** optimizer warm-restart at epoch 60 creating a different",
        "  gradient trajectory for epochs 61–90.",
        "",
        "**Recommended next steps:**",
        f"1. Re-evaluate {a} with the fixed ceiling script (WM noise disabled)",
        "   to establish a stable baseline error set.",
        f"2. Resume from {a} with proper optimizer-state restore and lower LR.",
        "3. Re-evaluate the resulting checkpoint and compare again.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[compare_ceiling] A ({args.label_a}): {args.a}")
    print(f"[compare_ceiling] B ({args.label_b}): {args.b}")

    df_a = _load(args.a, args.label_a)
    df_b = _load(args.b, args.label_b)

    if df_a is None and df_b is None:
        print("ERROR: both input files missing.  "
              "Run evaluate_train_lexicon_ceiling.py with per-checkpoint --out_dir.")
        sys.exit(1)

    if df_a is None:
        df_a = pd.DataFrame(columns=["word"])
    if df_b is None:
        df_b = pd.DataFrame(columns=["word"])

    words_a = set(df_a["word"].dropna())
    words_b = set(df_b["word"].dropna())
    common  = words_a & words_b
    fixed   = words_a - words_b
    new     = words_b - words_a

    cmp = compare(df_a, df_b, args.label_a, args.label_b)

    json_path = os.path.join(args.out_dir, "comparison.json")
    cmp_for_json = {k: v for k, v in cmp.items()
                    if k not in ("common_words", "fixed_words", "new_words")}
    cmp_for_json["common_words"] = cmp["common_words"]
    cmp_for_json["fixed_words"]  = cmp["fixed_words"]
    cmp_for_json["new_words"]    = cmp["new_words"]
    with open(json_path, "w") as f:
        json.dump(cmp_for_json, f, indent=2, default=str)
    print(f"  -> {json_path}")

    write_overlap_tsv(df_a, df_b, common, fixed, new,
                      args.label_a, args.label_b,
                      os.path.join(args.out_dir, "error_overlap.tsv"))

    write_md(cmp, os.path.join(args.out_dir, "comparison.md"))

    print(f"\n  === COMPARISON SUMMARY ===")
    print(f"  {args.label_a} errors : {len(df_a)}")
    print(f"  {args.label_b} errors : {len(df_b)}")
    print(f"  Common              : {len(common)}")
    print(f"  Fixed by {args.label_b:5s}     : {len(fixed)}")
    print(f"  New in  {args.label_b:5s}     : {len(new)}")
    print(f"  Stability           : {cmp['stability'][:50]}…")


if __name__ == "__main__":
    main()
