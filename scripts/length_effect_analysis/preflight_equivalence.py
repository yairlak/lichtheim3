"""Preflight: the instrumented AR paths must reproduce the canonical tokens exactly.

Runs the instrumented LTM-only and FULL AR streams on a deterministic seed-22
subset and compares every generated token, the final prediction, the EOS
position and the edit distance against the frozen canonical evaluator output.

EXACT_TOKEN_EQUIVALENCE must be PASS with zero differences.  Any difference is a
hard stop: the full cohort must not be run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.external_eval import load_model_and_vocab               # noqa: E402
from scripts.length_effect_analysis import instrument as I           # noqa: E402

CK22 = ("archives/fulllexicon_93a577f/extracted/fulllexicon_final_bundle_93a577f/"
        "selected_checkpoints/seed_22_epoch_0140.pt")
CANON = ("outputs/behavioral_wfe_fulllexicon_93a577f/full_wfe_evaluation/seed22/"
         "wfe_ar/item_level_predictions_enriched.tsv")
WFE = "data/eval_external/wfe_eval.tsv"


def load_items(vocab):
    df = pd.read_csv(os.path.join(ROOT, WFE), sep="\t")
    df = df[~df["notes"].fillna("").str.contains("EXCLUDED", na=False)].copy()
    df = df.reset_index(drop=True)
    forms, keep = [], []
    for i, row in df.iterrows():
        syms = row["target_phonemes"].split()
        ids = [vocab.stoi[s] for s in syms if s in vocab.stoi]
        if len(ids) == len(syms):
            forms.append(ids)
            keep.append(i)
    return df.loc[keep].reset_index(drop=True), forms


def pick_subset(df, canon):
    """Deterministic subset: trained real + novel pseudo, lengths 3/5/7/9,
    correct items, LTM-error items, and premature-EOS items where available."""
    c = canon.set_index("item_id")
    sel = []
    for L in (3, 5, 7, 9):
        for status in ("TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD"):
            sub = c[(c["length_phonemes"] == L)
                    & (c["lichtheim_exposure_status"] == status)]
            if len(sub) == 0:
                continue
            ok = sub[sub["ltm_exact_match"] == 1]
            bad = sub[sub["ltm_exact_match"] == 0]
            eos = sub[pd.to_numeric(sub["ltm_eos_position"],
                                    errors="coerce").notna()]
            for part in (ok, bad, eos):
                sel += list(part.index[:2])
    return sorted(set(sel))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(
        ROOT, "outputs/length_effect_mechanism_93a577f/instrumented"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    model, vocab, meta = load_model_and_vocab(os.path.join(ROOT, CK22), "cpu")
    df, forms = load_items(vocab)
    canon = pd.read_csv(os.path.join(ROOT, CANON), sep="\t")
    ids = pick_subset(df, canon)
    idx = [i for i, r in df.iterrows() if r["item_id"] in set(ids)]
    sub_forms = [forms[i] for i in idx]
    sub_ids = [df.loc[i, "item_id"] for i in idx]
    print(f"[preflight] subset: {len(sub_ids)} items, "
          f"lengths {sorted({len(f) for f in sub_forms})}")

    cmap = canon.set_index("item_id")
    diffs, rows = [], []
    for start in range(0, len(sub_forms), I.BATCH_SIZE):
        bf = sub_forms[start:start + I.BATCH_SIZE]
        bi = sub_ids[start:start + I.BATCH_SIZE]
        ltm = I.ar_stream(model, vocab, bf, "cpu", route="ltm")
        full = I.ar_stream(model, vocab, bf, "cpu", route="full",
                           capture_all_routes=True)
        for j, item in enumerate(bi):
            c = cmap.loc[item]
            for route, res in (("ltm", ltm), ("full", full)):
                pred = " ".join(vocab.itos[t] for t in res["preds"][j])
                cpred = str(c[f"{route}_predicted"]) if not pd.isna(
                    c[f"{route}_predicted"]) else ""
                ceos = c[f"{route}_eos_position"]
                ceos = None if pd.isna(ceos) or ceos == "" else int(float(ceos))
                ok_tok = pred == cpred
                ok_eos = res["eos_position"][j] == ceos
                rows.append({"item_id": item, "route": route,
                             "instrumented_prediction": pred,
                             "canonical_prediction": cpred,
                             "prediction_match": ok_tok,
                             "instrumented_eos": res["eos_position"][j],
                             "canonical_eos": ceos, "eos_match": ok_eos})
                if not (ok_tok and ok_eos):
                    diffs.append((item, route, pred, cpred))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out_dir, "preflight_equivalence.tsv"),
               sep="\t", index=False)
    verdict = "PASS" if not diffs else "FAIL"
    summary = {"EXACT_TOKEN_EQUIVALENCE": verdict,
               "n_items": len(sub_ids), "n_comparisons": len(rows),
               "n_differences": len(diffs),
               "lengths_covered": sorted({len(f) for f in sub_forms}),
               "differences": diffs[:20]}
    with open(os.path.join(args.out_dir, "preflight_equivalence.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "differences"},
                     indent=2))
    if diffs:
        print("\nFAIL — differences found; do not run the full cohort.")
        for d in diffs[:10]:
            print("  ", d)
        return 1
    print("\nEXACT_TOKEN_EQUIVALENCE = PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
