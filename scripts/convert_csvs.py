"""Step 3: Convert raw CSV files to Yair-L3-compatible TSV format.

Reads the original CSVs without modifying them; writes clean TSV files with
explicit columns and safe formatting.

Outputs:
    data/eval_external/wfe_yair_l3_format.tsv
    data/eval_external/ssp_yair_l3_format.tsv

Usage (from repo root):
    python scripts/convert_csvs.py
    python scripts/convert_csvs.py \\
        --wfe  data/raw-nwr_swp/wfe.csv \\
        --ssp  data/raw-nwr_swp/ssp.csv \\
        --phon data/raw-nwr_swp/phonemes.csv \\
        --out  data/eval_external
"""
from __future__ import annotations

import argparse
import ast
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
OUT_DIR   = os.path.join(ROOT, "data", "eval_external")

# Yair-L3 training max sequence length; items longer than this are OOD
TRAIN_MAX_LEN = 9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wfe",  default=WFE_PATH)
    p.add_argument("--ssp",  default=SSP_PATH)
    p.add_argument("--phon", default=PHON_PATH)
    p.add_argument("--out",  default=OUT_DIR)
    return p.parse_args()


def _parse_phoneme_list(s) -> list[str] | None:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        result = ast.literal_eval(s.strip())
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def _phon_str(lst: list[str]) -> str:
    """Serialise a phoneme list as a space-separated string for the TSV."""
    return " ".join(lst)


def convert_wfe(wfe_path: str, vocab_set: set[str], out_dir: str) -> str:
    df = pd.read_csv(wfe_path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    rows = []
    n_excluded_parse   = 0
    n_excluded_vocab   = 0
    n_ood_length       = 0

    for idx, row in df.iterrows():
        ns   = row.get("No_Stress", "")
        phs  = _parse_phoneme_list(str(ns)) if pd.notna(ns) else None

        if phs is None:
            n_excluded_parse += 1
            continue

        unknown = [p for p in phs if p not in vocab_set]
        if unknown:
            n_excluded_vocab += 1
            note = f"EXCLUDED:unknown_phonemes:{','.join(unknown)}"
            # Still write to TSV so there's a record, but mark excluded
        else:
            note = ""

        ood = len(phs) > TRAIN_MAX_LEN
        if ood:
            n_ood_length += 1
            note += ("|" if note else "") + f"OOD_LENGTH:{len(phs)}"

        rows.append({
            "item_id":        f"wfe_{int(idx):04d}",
            "word":           str(row.get("Word", "")),
            "target_phonemes": _phon_str(phs),
            "lexicality":     str(row.get("Lexicality", "")),
            "condition":      str(row.get("Condition", "")),
            "size":           str(row.get("Size", "")),
            "morphology":     str(row.get("Morphology", "")),
            "frequency":      str(row.get("Frequency", "")),
            "zipf_frequency": str(row.get("Zipf_Frequency", "")),
            "length_phonemes": int(len(phs)),
            "part_of_speech": str(row.get("Part of Speech", "")),
            "vowel_count":    str(row.get("Vowel Count", "")),
            "consonant_count":str(row.get("Consonant Count", "")),
            "source_file":    os.path.basename(wfe_path),
            "notes":          note,
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "wfe_eval.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"[convert_wfe] {len(rows)} rows written -> {out_path}")
    print(f"  excluded (parse fail): {n_excluded_parse}")
    print(f"  excluded (vocab mismatch): {n_excluded_vocab}")
    print(f"  flagged OOD length>9: {n_ood_length}")
    return out_path


def convert_ssp(ssp_path: str, vocab_set: set[str], out_dir: str) -> str:
    df = pd.read_csv(ssp_path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    rows = []
    n_excluded_parse = 0
    n_excluded_vocab = 0

    for idx, row in df.iterrows():
        ns  = row.get("No_Stress", "")
        phs = _parse_phoneme_list(str(ns)) if pd.notna(ns) else None

        if phs is None:
            n_excluded_parse += 1
            continue

        unknown = [p for p in phs if p not in vocab_set]
        note = ""
        if unknown:
            n_excluded_vocab += 1
            note = f"EXCLUDED:unknown_phonemes:{','.join(unknown)}"

        rows.append({
            "item_id":        f"ssp_{int(idx):05d}",
            "word_or_id":     f"ssp_{int(idx):05d}",   # SSP has no orthographic form
            "target_phonemes": _phon_str(phs),
            "sonority":       str(row.get("Sonority", "")),
            "type":           str(row.get("Type", "")),
            "length_phonemes": int(len(phs)),
            "source_file":    os.path.basename(ssp_path),
            "notes":          note,
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "ssp_eval.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"[convert_ssp] {len(rows)} rows written -> {out_path}")
    print(f"  excluded (parse fail): {n_excluded_parse}")
    print(f"  excluded (vocab mismatch): {n_excluded_vocab}")
    return out_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    vocab     = build_vocab()
    vocab_set = set(vocab.itos[3:])

    print("[convert_csvs] Converting WFE …")
    convert_wfe(args.wfe, vocab_set, args.out)

    print("[convert_csvs] Converting SSP …")
    convert_ssp(args.ssp, vocab_set, args.out)

    print("\n[convert_csvs] Done.  Original CSV files are untouched.")
    print(f"  Converted files written to: {args.out}")


if __name__ == "__main__":
    main()
