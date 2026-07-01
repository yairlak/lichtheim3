"""Step 2: Validate CSV compatibility with Yair-L3.

Read-only inspection of the three external CSV files.  Writes:
    outputs/external_eval/csv_inspection_report.json
    outputs/external_eval/csv_inspection_summary.md

Usage (from repo root):
    python scripts/inspect_csvs.py
    python scripts/inspect_csvs.py \\
        --wfe  data/raw-nwr_swp/wfe.csv \\
        --ssp  data/raw-nwr_swp/ssp.csv \\
        --phon data/raw-nwr_swp/phonemes.csv \\
        --out  outputs/external_eval
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.phonemes import build_vocab

DATA_DIR  = os.path.join(ROOT, "data", "raw-nwr_swp")
WFE_PATH  = os.path.join(DATA_DIR, "wfe.csv")
SSP_PATH  = os.path.join(DATA_DIR, "ssp.csv")
PHON_PATH = os.path.join(DATA_DIR, "phonemes.csv")
OUT_DIR   = os.path.join(ROOT, "outputs", "external_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wfe",  default=WFE_PATH)
    p.add_argument("--ssp",  default=SSP_PATH)
    p.add_argument("--phon", default=PHON_PATH)
    p.add_argument("--out",  default=OUT_DIR)
    return p.parse_args()


def _parse_phoneme_list(s: str) -> list[str] | None:
    """Parse a Python-list-repr string like \"['AH', 'T', 'EH']\" -> ['AH','T','EH']."""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        result = ast.literal_eval(s.strip())
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def inspect_phonemes_csv(path: str, vocab_set: set[str]) -> dict:
    df = pd.read_csv(path)
    phoneme_col = df["Phoneme"].tolist()
    in_vocab = [p for p in phoneme_col if p in vocab_set]
    missing  = [p for p in phoneme_col if p not in vocab_set]
    return {
        "path": path,
        "n_rows": len(df),
        "columns": list(df.columns),
        "phonemes": phoneme_col,
        "n_in_yair_l3_vocab": len(in_vocab),
        "n_not_in_yair_l3_vocab": len(missing),
        "missing_from_yair_l3": missing,
        "note": (
            "All 39 phonemes in phonemes.csv match Yair-L3 inventory"
            if not missing else
            f"WARNING: {missing} not in Yair-L3 vocabulary"
        ),
    }


def inspect_wfe(path: str, vocab_set: set[str]) -> dict:
    df = pd.read_csv(path)
    # Drop unnamed index column if present
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    report: dict = {
        "path": path,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
    }

    # --- Lexicality / condition / size / morphology ---
    for col in ("Lexicality", "Condition", "Size", "Morphology", "Part of Speech"):
        if col in df.columns:
            report[f"unique_{col.lower().replace(' ', '_')}"] = (
                df[col].value_counts().to_dict()
            )

    # --- Phoneme analysis on No_Stress column ---
    parsed_ok, parsed_fail = [], []
    all_phon_syms: set[str] = set()
    unknown_syms: set[str] = set()
    lengths: list[int] = []

    for _, row in df.iterrows():
        ns = row.get("No_Stress", "")
        parsed = _parse_phoneme_list(str(ns)) if pd.notna(ns) else None
        if parsed is None:
            parsed_fail.append(str(row.get("Word", row.name)))
            continue
        parsed_ok.append(parsed)
        lengths.append(len(parsed))
        all_phon_syms.update(parsed)
        unknown_syms.update(p for p in parsed if p not in vocab_set)

    report["no_stress_parsed_ok"]   = len(parsed_ok)
    report["no_stress_parse_fail"]  = len(parsed_fail)
    report["parse_fail_items"]      = parsed_fail[:20]
    report["unique_phonemes_in_no_stress"] = sorted(all_phon_syms)
    report["phonemes_NOT_in_yair_l3_vocab"] = sorted(unknown_syms)
    report["length_distribution"] = (
        pd.Series(lengths).value_counts().sort_index().to_dict()
    )
    report["length_min"]    = int(min(lengths)) if lengths else None
    report["length_max"]    = int(max(lengths)) if lengths else None
    report["length_mean"]   = round(float(pd.Series(lengths).mean()), 2) if lengths else None
    report["items_exceeding_train_max_phonemes_9"] = int(
        sum(1 for l in lengths if l > 9)
    )

    # --- Duplicate words ---
    if "Word" in df.columns:
        dup_words = df[df.duplicated("Word", keep=False)]["Word"].tolist()
        report["duplicate_words"] = len(dup_words)
        report["duplicate_word_examples"] = dup_words[:10]

    # --- Frequency check ---
    if "Zipf_Frequency" in df.columns:
        real = df[df["Lexicality"] == "real"]["Zipf_Frequency"]
        report["zipf_frequency_real_words"] = {
            "min": float(real.min()) if len(real) else None,
            "max": float(real.max()) if len(real) else None,
            "mean": float(real.mean()) if len(real) else None,
            "n_missing": int(real.isnull().sum()),
        }

    return report


def inspect_ssp(path: str, vocab_set: set[str]) -> dict:
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    report: dict = {
        "path": path,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
    }

    for col in ("Type", "Sonority", "Length"):
        if col in df.columns:
            vc = df[col].value_counts()
            report[f"unique_{col.lower()}"] = vc.to_dict()

    if "Sonority" in df.columns:
        son = df["Sonority"].dropna()
        report["sonority_range"] = {
            "min": float(son.min()), "max": float(son.max()),
            "n_unique": int(son.nunique()),
        }

    all_phon_syms: set[str] = set()
    unknown_syms: set[str]  = set()
    parsed_ok, parsed_fail = 0, 0
    lengths: list[int] = []

    for _, row in df.iterrows():
        ns = row.get("No_Stress", "")
        parsed = _parse_phoneme_list(str(ns)) if pd.notna(ns) else None
        if parsed is None:
            parsed_fail += 1
            continue
        parsed_ok += 1
        lengths.append(len(parsed))
        all_phon_syms.update(parsed)
        unknown_syms.update(p for p in parsed if p not in vocab_set)

    report["no_stress_parsed_ok"]   = parsed_ok
    report["no_stress_parse_fail"]  = parsed_fail
    report["unique_phonemes_in_no_stress"] = sorted(all_phon_syms)
    report["phonemes_NOT_in_yair_l3_vocab"] = sorted(unknown_syms)
    report["lengths_all_equal_3"] = all(l == 3 for l in lengths)
    report["length_distribution"] = (
        pd.Series(lengths).value_counts().sort_index().to_dict()
    )

    return report


def write_markdown(report: dict, out_path: str) -> None:
    lines = ["# Yair-L3 External CSV Inspection Report\n"]

    wfe = report.get("wfe", {})
    ssp = report.get("ssp", {})
    phon = report.get("phonemes", {})
    vocab_ok = not phon.get("missing_from_yair_l3")

    lines.append("## Vocabulary check (phonemes.csv vs Yair-L3)")
    lines.append(f"- phonemes.csv rows : {phon.get('n_rows')}")
    lines.append(f"- All in Yair-L3 vocab : {'YES' if vocab_ok else 'NO'}")
    if not vocab_ok:
        lines.append(f"- Missing : {phon.get('missing_from_yair_l3')}")
    lines.append("")

    lines.append("## WFE dataset")
    lines.append(f"- Path   : `{wfe.get('path')}`")
    lines.append(f"- Rows   : {wfe.get('n_rows')}")
    lines.append(f"- No_Stress parsed OK / FAIL : "
                 f"{wfe.get('no_stress_parsed_ok')} / {wfe.get('no_stress_parse_fail')}")
    lines.append(f"- Unknown phonemes : {wfe.get('phonemes_NOT_in_yair_l3_vocab') or 'none'}")
    lines.append(f"- Length range : {wfe.get('length_min')}–{wfe.get('length_max')}  "
                 f"(mean {wfe.get('length_mean')})")
    lines.append(f"- Items with length > 9 (out-of-training-distribution) : "
                 f"{wfe.get('items_exceeding_train_max_phonemes_9')}")
    if wfe.get("unique_lexicality"):
        lines.append(f"- Lexicality : {wfe['unique_lexicality']}")
    if wfe.get("unique_condition"):
        lines.append(f"- Conditions : {wfe['unique_condition']}")
    if wfe.get("unique_size"):
        lines.append(f"- Size : {wfe['unique_size']}")
    if wfe.get("unique_morphology"):
        lines.append(f"- Morphology : {wfe['unique_morphology']}")
    lines.append("")

    lines.append("## SSP dataset")
    lines.append(f"- Path  : `{ssp.get('path')}`")
    lines.append(f"- Rows  : {ssp.get('n_rows')}")
    lines.append(f"- No_Stress parsed OK / FAIL : "
                 f"{ssp.get('no_stress_parsed_ok')} / {ssp.get('no_stress_parse_fail')}")
    lines.append(f"- Unknown phonemes : {ssp.get('phonemes_NOT_in_yair_l3_vocab') or 'none'}")
    lines.append(f"- All length=3 : {ssp.get('lengths_all_equal_3')}")
    lines.append(f"- Types : {ssp.get('unique_type')}")
    son_r = ssp.get("sonority_range", {})
    lines.append(f"- Sonority range : {son_r.get('min')}–{son_r.get('max')}  "
                 f"({son_r.get('n_unique')} unique values)")
    lines.append("")

    lines.append("## Conclusion")
    all_ok = (
        vocab_ok
        and not wfe.get("phonemes_NOT_in_yair_l3_vocab")
        and not ssp.get("phonemes_NOT_in_yair_l3_vocab")
        and wfe.get("no_stress_parse_fail", 1) == 0
        and ssp.get("no_stress_parse_fail", 1) == 0
    )
    lines.append(
        "All phonemes and No_Stress sequences are compatible with Yair-L3."
        if all_ok else
        "⚠ Some items may need exclusion — see details above."
    )
    lines.append("")
    lines.append("**Note:** Evaluation will be teacher-forced (gold previous "
                 "phoneme fed at each decoder step). This is the same regime "
                 "used by all evaluate/*.py scripts in Yair-L3.")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {out_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    vocab = build_vocab()
    vocab_set = set(vocab.itos[3:])  # skip PAD/BOS/EOS

    print("[inspect_csvs] Reading phonemes.csv …")
    phon_report = inspect_phonemes_csv(args.phon, vocab_set)

    print("[inspect_csvs] Reading wfe.csv …")
    wfe_report  = inspect_wfe(args.wfe,  vocab_set)

    print("[inspect_csvs] Reading ssp.csv …")
    ssp_report  = inspect_ssp(args.ssp,  vocab_set)

    full_report = {
        "phonemes": phon_report,
        "wfe":      wfe_report,
        "ssp":      ssp_report,
    }

    json_path = os.path.join(args.out, "csv_inspection_report.json")
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"  -> {json_path}")

    md_path = os.path.join(args.out, "csv_inspection_summary.md")
    write_markdown(full_report, md_path)

    # --- Console summary ---
    print("\n===== COMPATIBILITY SUMMARY =====")
    print(f"  phonemes.csv : {phon_report['n_rows']} rows, "
          f"missing from Yair-L3: {phon_report['missing_from_yair_l3'] or 'none'}")
    print(f"  wfe.csv      : {wfe_report['n_rows']} rows, "
          f"unknown phonemes: {wfe_report['phonemes_NOT_in_yair_l3_vocab'] or 'none'}, "
          f"parse fails: {wfe_report['no_stress_parse_fail']}")
    print(f"  ssp.csv      : {ssp_report['n_rows']} rows, "
          f"unknown phonemes: {ssp_report['phonemes_NOT_in_yair_l3_vocab'] or 'none'}, "
          f"parse fails: {ssp_report['no_stress_parse_fail']}")


if __name__ == "__main__":
    main()
