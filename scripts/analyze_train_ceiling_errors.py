"""Analyze remaining train-split errors after a ceiling evaluation.

Input:
    outputs/train_lexicon_ceiling/train_errors.tsv

Outputs:
    outputs/train_lexicon_ceiling/error_analysis.json
    outputs/train_lexicon_ceiling/error_analysis.md
    outputs/train_lexicon_ceiling/error_analysis_by_length.tsv
    outputs/train_lexicon_ceiling/error_analysis_by_rank_bin.tsv

Usage:
    python scripts/analyze_train_ceiling_errors.py
    python scripts/analyze_train_ceiling_errors.py \\
        --errors outputs/train_lexicon_ceiling/train_errors.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ERRORS_PATH = os.path.join(ROOT, "outputs", "train_lexicon_ceiling", "train_errors.tsv")
OUT_DIR     = os.path.join(ROOT, "outputs", "train_lexicon_ceiling")

RANK_BINS = [
    (1,      1_000,  "rank 1–1k (very high freq)"),
    (1_001,  5_000,  "rank 1k–5k (high freq)"),
    (5_001,  10_000, "rank 5k–10k (medium freq)"),
    (10_001, 20_000, "rank 10k–20k (low freq)"),
    (20_001, 30_000, "rank 20k–30k (very low freq / tail)"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--errors",  default=ERRORS_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    return p.parse_args()


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def analyze(df: pd.DataFrame) -> dict:
    n = len(df)

    # Route categorisation
    wm_correct   = df["wm_exact_match"]  == 1
    ltm_correct  = df["ltm_exact_match"] == 1
    full_only_wrong       = (~wm_correct) & (~ltm_correct)          # both routes also wrong
    wm_correct_full_wrong = wm_correct   & ~ltm_correct             # WM saves but LTM + gate fail
    ltm_correct_full_wrong= ltm_correct  & ~wm_correct              # LTM saves but WM + gate fail
    both_correct_full_wrong = wm_correct & ltm_correct              # gate mixing fails even when both routes are right
    wm_or_ltm_correct = wm_correct | ltm_correct                   # at least one route is right

    # Length distribution
    len_dist = df["length"].value_counts().sort_index().to_dict()

    # Rank-bin distribution
    rank_bins = []
    for lo, hi, label in RANK_BINS:
        mask = (df["rank"] >= lo) & (df["rank"] <= hi)
        rank_bins.append({
            "bin_label":   label,
            "rank_lo":     lo,
            "rank_hi":     hi,
            "n_errors":    int(mask.sum()),
            "mean_length": round(float(df.loc[mask, "length"].mean()), 2) if mask.sum() else float("nan"),
        })

    # Edit-distance distribution (all errors are exactly 1 by sorting order, check)
    ed_dist = df["full_edit_dist"].value_counts().sort_index().to_dict()

    # Examples: WM correct but full wrong (gate routing issue)
    ex_wm_saves = (
        df[wm_correct_full_wrong]
        [["word", "rank", "length", "target_phonemes",
          "full_predicted", "wm_predicted", "ltm_predicted",
          "full_edit_dist"]]
        .sort_values("rank")
        .head(15)
        .to_dict(orient="records")
    )

    # Examples: both routes wrong (hard items — neither route can do it)
    ex_both_wrong = (
        df[full_only_wrong]
        [["word", "rank", "length", "target_phonemes",
          "full_predicted", "wm_predicted", "ltm_predicted",
          "full_edit_dist"]]
        .sort_values("rank")
        .head(10)
        .to_dict(orient="records")
    )

    # Long low-frequency errors (length=9, rank>20k)
    ex_long_low = (
        df[(df["length"] == 9) & (df["rank"] > 20000)]
        [["word", "rank", "length", "target_phonemes",
          "full_predicted", "wm_predicted", "ltm_predicted"]]
        .sort_values("rank", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    # High-frequency surprises (rank < 5000)
    ex_highfreq = (
        df[df["rank"] < 5000]
        [["word", "rank", "length", "target_phonemes",
          "full_predicted", "wm_predicted"]]
        .sort_values("rank")
        .to_dict(orient="records")
    )

    # Most common single-phoneme error type (substitutions)
    substitutions = []
    for _, row in df.iterrows():
        tgt  = row["target_phonemes"].split()
        pred = row["full_predicted"].split()
        for i, (t, p) in enumerate(zip(tgt, pred)):
            if t != p:
                substitutions.append(f"{t}→{p}")
    from collections import Counter
    top_subs = Counter(substitutions).most_common(20)

    return {
        "n_errors":                    n,
        "n_wm_correct":                int(wm_correct.sum()),
        "n_ltm_correct":               int(ltm_correct.sum()),
        "n_both_routes_wrong":         int(full_only_wrong.sum()),
        "n_both_routes_correct_gate_fails": int(both_correct_full_wrong.sum()),
        "n_wm_correct_full_wrong":     int(wm_correct_full_wrong.sum()),
        "n_ltm_correct_full_wrong":    int(ltm_correct_full_wrong.sum()),
        "n_at_least_one_route_correct":int(wm_or_ltm_correct.sum()),
        "pct_gate_routing_issue":      round(100.0 * wm_or_ltm_correct.sum() / max(n, 1), 1),
        "pct_hard_neither_route_right":round(100.0 * full_only_wrong.sum()   / max(n, 1), 1),
        "length_distribution":         {str(k): int(v) for k, v in len_dist.items()},
        "mean_error_length":           round(float(df["length"].mean()), 2),
        "pct_length_9":                round(100.0 * (df["length"] == 9).sum() / max(n, 1), 1),
        "edit_distance_distribution":  {str(k): int(v) for k, v in ed_dist.items()},
        "mean_error_edit_dist":        round(float(df["full_edit_dist"].mean()), 3),
        "rank_bin_distribution":       rank_bins,
        "n_high_freq_errors_rank_lt5k":int((df["rank"] < 5000).sum()),
        "n_tail_errors_rank_gt20k":    int((df["rank"] > 20000).sum()),
        "top_substitutions":           [{"sub": s, "count": c} for s, c in top_subs],
        "examples_wm_correct_full_wrong":   ex_wm_saves,
        "examples_both_routes_wrong":        ex_both_wrong,
        "examples_long_low_freq":            ex_long_low,
        "examples_high_freq_surprises":      ex_highfreq,
        "interpretation": (
            "Most errors (pct_gate_routing_issue%) have at least one route correct — "
            "the gate is choosing the wrong mix.  The hard residual cases "
            "(pct_hard_neither_route_right%) need more training exposure.  "
            "Errors are concentrated in length=9, rank>20k words, but a few "
            "high-frequency words also fail (check examples_high_freq_surprises)."
        ).replace("pct_gate_routing_issue%", f"{round(100.0*wm_or_ltm_correct.sum()/max(n,1),1)}%")
         .replace("pct_hard_neither_route_right%", f"{round(100.0*full_only_wrong.sum()/max(n,1),1)}%"),
    }


def write_md(report: dict, path: str) -> None:
    n = report["n_errors"]
    lines = [
        "# Train Ceiling Error Analysis",
        "",
        "> Source: `outputs/train_lexicon_ceiling/train_errors.tsv`",
        "> Evaluation regime: teacher-forced (gold prefix at each decoder step).",
        "",
        f"**Total full-route errors: {n}**  "
        f"(full exact-match = {1 - n/25136:.4f} over 25 136 training words)",
        "",
        "---",
        "",
        "## Route breakdown",
        "",
        "| Category | n | % of errors |",
        "|---|---|---|",
        f"| WM correct, full wrong (gate routing issue) | "
        f"{report['n_wm_correct_full_wrong']} | "
        f"{round(100*report['n_wm_correct_full_wrong']/n,1)}% |",
        f"| LTM correct, full wrong (gate routing issue) | "
        f"{report['n_ltm_correct_full_wrong']} | "
        f"{round(100*report['n_ltm_correct_full_wrong']/n,1)}% |",
        f"| Both routes correct, gate mixing fails | "
        f"{report['n_both_routes_correct_gate_fails']} | "
        f"{round(100*report['n_both_routes_correct_gate_fails']/n,1)}% |",
        f"| Both routes wrong (neither can produce the word) | "
        f"{report['n_both_routes_wrong']} | "
        f"{round(100*report['n_both_routes_wrong']/n,1)}% |",
        "",
        f"**{report['pct_gate_routing_issue']}% of errors have at least one route correct** — "
        "mainly a gate-mixing issue, not a capacity issue.",
        f"**{report['pct_hard_neither_route_right']}% of errors are 'hard' — neither route produces the word** "
        "— these are the cases that need more training.",
        "",
        "---",
        "",
        "## Length distribution of errors",
        "",
        "| Length | n errors |",
        "|---|---|",
    ]
    for k, v in sorted(report["length_distribution"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"Mean error length: {report['mean_error_length']}  "
        f"({report['pct_length_9']}% are max-length words with 9 phonemes)",
        "",
        "---",
        "",
        "## Rank / frequency distribution of errors",
        "",
        "| Rank bin | n errors | Mean length |",
        "|---|---|---|",
    ]
    for r in report["rank_bin_distribution"]:
        lines.append(f"| {r['bin_label']} | {r['n_errors']} | {_fmt(r['mean_length'])} |")
    lines += [
        "",
        f"**{report['n_high_freq_errors_rank_lt5k']} errors** are rank < 5 000 (high-frequency words — "
        "surprising and should be resolved with more training).",
        "",
        "---",
        "",
        "## Edit-distance distribution",
        "",
        "| Edit distance | n errors |",
        "|---|---|",
    ]
    for k, v in sorted(report["edit_distance_distribution"].items(), key=lambda x: int(x[0])):
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"Mean edit distance: {report['mean_error_edit_dist']}.  "
        "Almost all errors are 1–2 phonemes off — very close to correct.",
        "",
        "---",
        "",
        "## Top substitution errors (target→predicted)",
        "",
        "| Substitution | Count |",
        "|---|---|",
    ]
    for item in report["top_substitutions"][:15]:
        lines.append(f"| {item['sub']} | {item['count']} |")

    lines += [
        "",
        "---",
        "",
        "## Examples: WM correct but full wrong (gate routing errors)",
        "",
        "These are cases where the dorsal buffer produces the right word, but the gate",
        "routes (partially) to LTM, degrading the output.  More training should sharpen",
        "the gate's confidence signal for these items.",
        "",
        "| Word | Rank | Len | Target | Full predicted | WM predicted |",
        "|---|---|---|---|---|---|",
    ]
    for e in report["examples_wm_correct_full_wrong"]:
        lines.append(
            f"| {e['word']} | {e['rank']} | {e['length']} "
            f"| {e['target_phonemes']} | {e['full_predicted']} | {e['wm_predicted']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Examples: high-frequency surprises (rank < 5 000)",
        "",
        "These are words the model should have memorised by now.  "
        "Single-phoneme confusion near a phonetically similar segment.",
        "",
        "| Word | Rank | Len | Target | Full predicted |",
        "|---|---|---|---|---|",
    ]
    for e in report["examples_high_freq_surprises"]:
        lines.append(
            f"| {e['word']} | {e['rank']} | {e['length']} "
            f"| {e['target_phonemes']} | {e['full_predicted']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Examples: both routes wrong (hard residual cases)",
        "",
        "Neither WM nor LTM produces the correct form.  "
        "These need more training gradient exposure.",
        "",
        "| Word | Rank | Len | Target | Full predicted | WM predicted | LTM predicted |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in report["examples_both_routes_wrong"]:
        lines.append(
            f"| {e['word']} | {e['rank']} | {e['length']} "
            f"| {e['target_phonemes']} | {e['full_predicted']} "
            f"| {e['wm_predicted']} | {e['ltm_predicted']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Interpretation and next steps",
        "",
        report["interpretation"],
        "",
        "**Recommended action:** run 60 total epochs (30 more from the current checkpoint)",
        "using `--resume_from`.  The gate-routing errors should resolve as the model",
        "converges further; the hard cases need continued gradient exposure.",
        "",
        "```bash",
        "python scripts/train_checkpoint.py \\",
        "    --lexicon_path data/lexicon_en_glove_covered.tsv \\",
        "    --max_words 30000 --epochs 60 --seed 0 \\",
        "    --resume_from checkpoints/lichtheim3_30k_glove.pt \\",
        "    --ckpt checkpoints/lichtheim3_30k_glove_e60.pt",
        "```",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


def main():
    args = parse_args()

    if not os.path.exists(args.errors):
        print(f"\nERROR: errors file not found: {args.errors}")
        print("Run first: python scripts/evaluate_train_lexicon_ceiling.py --ckpt checkpoints/...")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[analyze_errors] Loading {args.errors} …")
    df = pd.read_csv(args.errors, sep="\t")
    print(f"  {len(df)} errors loaded")

    report = analyze(df)

    # JSON
    json_path = os.path.join(args.out_dir, "error_analysis.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  -> {json_path}")

    # Markdown
    md_path = os.path.join(args.out_dir, "error_analysis.md")
    write_md(report, md_path)

    # By-length TSV
    length_rows = []
    for length_val, grp in df.groupby("length"):
        wm_ok  = int((grp["wm_exact_match"]  == 1).sum())
        ltm_ok = int((grp["ltm_exact_match"] == 1).sum())
        n_grp  = len(grp)
        both_wrong = int(((grp["wm_exact_match"] == 0) & (grp["ltm_exact_match"] == 0)).sum())
        length_rows.append({
            "length":            int(length_val),
            "n_errors":          n_grp,
            "mean_rank":         round(float(grp["rank"].mean()), 0),
            "mean_full_edit":    round(float(grp["full_edit_dist"].mean()), 3),
            "n_wm_correct":      wm_ok,
            "n_ltm_correct":     ltm_ok,
            "n_both_wrong":      both_wrong,
            "pct_wm_correct":    round(100.0 * wm_ok / n_grp, 1),
        })
    len_tsv = os.path.join(args.out_dir, "error_analysis_by_length.tsv")
    pd.DataFrame(length_rows).to_csv(len_tsv, sep="\t", index=False)
    print(f"  -> {len_tsv}")

    # By-rank-bin TSV
    bin_rows = []
    for lo, hi, label in RANK_BINS:
        grp = df[(df["rank"] >= lo) & (df["rank"] <= hi)]
        n_grp = len(grp)
        if n_grp == 0:
            bin_rows.append({"bin": label, "rank_lo": lo, "rank_hi": hi,
                             "n_errors": 0, "mean_length": float("nan"),
                             "n_wm_correct": 0, "n_ltm_correct": 0, "n_both_wrong": 0})
            continue
        wm_ok  = int((grp["wm_exact_match"]  == 1).sum())
        ltm_ok = int((grp["ltm_exact_match"] == 1).sum())
        bin_rows.append({
            "bin":            label,
            "rank_lo":        lo,
            "rank_hi":        hi,
            "n_errors":       n_grp,
            "mean_length":    round(float(grp["length"].mean()), 2),
            "mean_full_edit": round(float(grp["full_edit_dist"].mean()), 3),
            "n_wm_correct":   wm_ok,
            "n_ltm_correct":  ltm_ok,
            "n_both_wrong":   int(((grp["wm_exact_match"]==0) & (grp["ltm_exact_match"]==0)).sum()),
            "pct_wm_correct": round(100.0 * wm_ok / n_grp, 1),
        })
    bin_tsv = os.path.join(args.out_dir, "error_analysis_by_rank_bin.tsv")
    pd.DataFrame(bin_rows).to_csv(bin_tsv, sep="\t", index=False)
    print(f"  -> {bin_tsv}")

    # Console summary
    print(f"\n  === ERROR ANALYSIS SUMMARY ===")
    print(f"  Total errors       : {report['n_errors']}")
    print(f"  WM correct          : {report['n_wm_correct']} ({report['pct_gate_routing_issue']}% with ≥1 route right)")
    print(f"  LTM correct         : {report['n_ltm_correct']}")
    print(f"  Both routes wrong   : {report['n_both_routes_wrong']} ({report['pct_hard_neither_route_right']}%)")
    print(f"  High-freq (rank<5k) : {report['n_high_freq_errors_rank_lt5k']}")
    print(f"  Mean error length   : {report['mean_error_length']}")
    print(f"  Mean edit distance  : {report['mean_error_edit_dist']}")
    print()


if __name__ == "__main__":
    main()
