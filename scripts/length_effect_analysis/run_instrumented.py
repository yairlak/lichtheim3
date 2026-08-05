"""Consolidated instrumented run over all four checkpoints and all 1,200 WFE items.

Streams A (gold prefix), B (LTM-only AR) and C (FULL-AR).  No WM-only AR stream
(M0 decision D10).  Equivalence with the frozen canonical predictions is checked
for every seed and item and written to `canonical_equivalence.tsv`; the run fails
closed if any token differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.external_eval import load_model_and_vocab               # noqa: E402
from scripts.length_effect_analysis import instrument as I           # noqa: E402
from scripts.length_effect_analysis.bank_audit import (              # noqa: E402
    COSINE_TIE_POLICY, PHON_TIE_POLICY, build_bank, phon_string)

CKPTS = {19: ("selected_checkpoints/seed_19_epoch_0155.pt", 155,
              "7d05f9c2ad5a53e705f7d55ccde2581754918938d8ca888da35c0a859666478e"),
         20: ("selected_checkpoints/seed_20_epoch_0130.pt", 130,
              "b44548b6916ea89c6f099402b78031063445e572932acee8dd7558a73dfc6cfb"),
         21: ("selected_checkpoints/seed_21_epoch_0145.pt", 145,
              "ab58092e7c2bfac42ab977352e6d5d6416ca605b71a3eacb777300060b30f5cf"),
         22: ("selected_checkpoints/seed_22_epoch_0140.pt", 140,
              "a15846cbf3c7df88ed289512bbb20cbefd2121d0deec1b39f363932a743da595")}
BUNDLE = "archives/fulllexicon_93a577f/extracted/fulllexicon_final_bundle_93a577f"
WFE = "data/eval_external/wfe_eval.tsv"
CANON_T = ("outputs/behavioral_wfe_fulllexicon_93a577f/full_wfe_evaluation/"
           "seed{seed}/wfe_ar/item_level_predictions_enriched.tsv")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def git(*a):
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=I.DEFAULT_TOP_K)
    ap.add_argument("--seeds", default="19,20,21,22")
    ap.add_argument("--out_dir", default=os.path.join(
        ROOT, "outputs/length_effect_mechanism_93a577f/instrumented"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()

    # bank metadata (shared across seeds; identical lexicon for all four)
    ck0, vocab0, entries, _ = build_bank(
        os.path.join(ROOT, BUNDLE, CKPTS[19][0]))
    bank_words = [e.word for e in entries]
    bank_phon = [phon_string(e, vocab0) for e in entries]

    item_rows, ts_rows, neigh_rows, equiv_rows = [], [], [], []
    reps = {}
    for seed in seeds:
        rel, epoch, exp_sha = CKPTS[seed]
        ck_path = os.path.join(ROOT, BUNDLE, rel)
        assert sha(ck_path) == exp_sha, f"checkpoint SHA mismatch for seed {seed}"
        model, vocab, meta = load_model_and_vocab(ck_path, "cpu")
        df, forms = load_items(vocab)
        canon = pd.read_csv(os.path.join(ROOT, CANON_T.format(seed=seed)),
                            sep="\t").set_index("item_id")
        print(f"[seed {seed}] {len(forms)} items")

        h_wm_all, s_hat_all, pooled_all = [], [], []
        gp_wm, gp_ltm, fp_wm, fp_ltm = [], [], [], []
        for start in range(0, len(forms), I.BATCH_SIZE):
            bf = forms[start:start + I.BATCH_SIZE]
            bidx = list(range(start, min(start + I.BATCH_SIZE, len(forms))))
            batch = I.make_batch(bf, vocab, "cpu")
            enc = I.encoder_quantities(model, batch, top_k=args.top_k)
            gold, gst = I.gold_prefix_pass(model, vocab, bf, "cpu")
            ltm = I.ar_stream(model, vocab, bf, "cpu", route="ltm")
            full = I.ar_stream(model, vocab, bf, "cpu", route="full",
                               capture_all_routes=True)

            # per-item records
            for j, gi in enumerate(bidx):
                r = df.loc[gi]
                item = r["item_id"]
                c = canon.loc[item]
                L = len(bf[j])
                ltm_pred = " ".join(vocab.itos[t] for t in ltm["preds"][j])
                full_pred = " ".join(vocab.itos[t] for t in full["preds"][j])
                for route, pred, eosp in (("ltm", ltm_pred, ltm["eos_position"][j]),
                                          ("full", full_pred, full["eos_position"][j])):
                    cp = "" if pd.isna(c[f"{route}_predicted"]) else str(c[f"{route}_predicted"])
                    ce = c[f"{route}_eos_position"]
                    ce = None if (pd.isna(ce) or ce == "") else int(float(ce))
                    equiv_rows.append({
                        "seed": seed, "item_id": item, "route": route,
                        "instrumented_prediction": pred, "canonical_prediction": cp,
                        "prediction_match": pred == cp,
                        "instrumented_eos": eosp, "canonical_eos": ce,
                        "eos_match": eosp == ce,
                        "canonical_edit_dist": float(c[f"{route}_edit_dist"])})
                ti = int(enc["topk_idx"][j, 0])
                item_rows.append({
                    "seed": seed, "epoch": epoch, "checkpoint_sha256": exp_sha,
                    "item_id": item, "source_label": r["lexicality"],
                    "exposure_status": c["lichtheim_exposure_status"],
                    "phoneme_length": L,
                    "target_tokens": " ".join(vocab.itos[t] for t in bf[j]),
                    "existing_ar_full_prediction": "" if pd.isna(c["full_predicted"]) else c["full_predicted"],
                    "existing_ar_wm_prediction": "" if pd.isna(c["wm_predicted"]) else c["wm_predicted"],
                    "existing_ar_ltm_prediction": "" if pd.isna(c["ltm_predicted"]) else c["ltm_predicted"],
                    "instrumented_ltm_prediction": ltm_pred,
                    "instrumented_full_prediction": full_pred,
                    "ltm_eos_position": ltm["eos_position"][j],
                    "full_eos_position": full["eos_position"][j],
                    "confidence": float(enc["confidence"][j]),
                    "gate": float(enc["gate"][j]),
                    "bank_margin": float(enc["margin"][j]),
                    "bank_density": float(enc["density"][j]),
                    "top1_neighbor_id": ti,
                    "top1_neighbor_word": bank_words[ti],
                    "top1_neighbor_phonology": bank_phon[ti],
                    "top1_similarity": float(enc["topk_sim"][j, 0]),
                    "top1_top2_margin": float(enc["topk_sim"][j, 0]
                                              - enc["topk_sim"][j, 1]),
                })
                for k in range(args.top_k):
                    bi = int(enc["topk_idx"][j, k])
                    neigh_rows.append({"seed": seed, "item_id": item, "rank": k + 1,
                                       "bank_row": bi, "word": bank_words[bi],
                                       "phonology": bank_phon[bi],
                                       "cosine": float(enc["topk_sim"][j, k])})
                # timestep rows
                for rec in gold[j]:
                    for route in ("wm", "ltm", "full"):
                        ts_rows.append(_ts(seed, exp_sha, item, route,
                                           "gold_prefix", "gold", rec, L, vocab))
                for rec in ltm["steps"][j]:
                    ts_rows.append(_ts(seed, exp_sha, item, "ltm", "autoregressive",
                                       "ltm_generated", rec, L, vocab, ar=True))
                for rec in full["steps"][j]:
                    for route in ("wm", "ltm", "full"):
                        ts_rows.append(_ts(seed, exp_sha, item, route,
                                           "autoregressive", "full_generated",
                                           rec, L, vocab, ar=(route == "full")))
            h_wm_all.append(enc["h_wm"].numpy())
            s_hat_all.append(enc["s_hat"].numpy())
            pooled_all.append(enc["ltm_h0"].numpy())
            gp_wm.append(_pad(gst["wm_premotor"].numpy()))
            gp_ltm.append(_pad(gst["ltm_premotor"].numpy()))
            if full["wm_premotor_t0"] is not None:
                fp_wm.append(_pad(full["wm_premotor_t0"].numpy()))
                fp_ltm.append(_pad(full["ltm_premotor_t0"].numpy()))
            if start % (I.BATCH_SIZE * 5) == 0:
                print(f"   {min(start+I.BATCH_SIZE,len(forms))}/{len(forms)}",
                      end="\r")
        reps[f"wm_encoder_hidden_seed{seed}"] = np.concatenate(h_wm_all)
        reps[f"s_hat_seed{seed}"] = np.concatenate(s_hat_all)
        reps[f"ltm_decoder_h0_seed{seed}"] = np.concatenate(pooled_all)
        reps[f"wm_premotor_gold_prefix_seed{seed}"] = np.concatenate(gp_wm)
        reps[f"ltm_premotor_gold_prefix_seed{seed}"] = np.concatenate(gp_ltm)
        print(f"   seed {seed} done  ({time.time()-t0:.0f}s)")

    # ---------------- equivalence gate
    eq = pd.DataFrame(equiv_rows)
    eq.to_csv(os.path.join(args.out_dir, "canonical_equivalence.tsv"),
              sep="\t", index=False)
    n_bad = int((~eq["prediction_match"]).sum() + (~eq["eos_match"]).sum())
    verdict = "PASS" if n_bad == 0 else "FAIL"
    print(f"\nCANONICAL_EQUIVALENCE = {verdict} "
          f"({len(eq)} comparisons, {n_bad} differences)")

    pd.DataFrame(item_rows).to_csv(
        os.path.join(args.out_dir, "item_summary.tsv"), sep="\t", index=False)
    pd.DataFrame(neigh_rows).to_csv(
        os.path.join(args.out_dir, "lexical_neighbors.tsv"), sep="\t", index=False)
    ts = pd.DataFrame(ts_rows)
    try:
        ts.to_parquet(os.path.join(args.out_dir, "timestep_metrics.parquet"),
                      index=False)
    except Exception:
        ts.to_csv(os.path.join(args.out_dir, "timestep_metrics.tsv"),
                  sep="\t", index=False)
    np.savez_compressed(os.path.join(args.out_dir, "representations.npz"), **reps)

    prov = {
        "phase": "M1/M2/M3/M5 consolidated instrumented inference",
        "checkpoint_training_commit": "93a577fd9822955fa272ee733fa7e2acf81f1333",
        "behavioral_evaluation_commit": "e876b755d0475ed11e5fbc0419a0bd8860dfd325",
        "mechanism_analysis_commit_or_dirty_state": {
            "repository_head": git("rev-parse", "HEAD"),
            "repository_dirty": bool(git("status", "--porcelain").strip()),
            "note": "mechanism analysis scripts are untracked at run time"},
        "checkpoints": {str(s): {"path": os.path.join(BUNDLE, CKPTS[s][0]),
                                 "sha256": CKPTS[s][2], "seed": s,
                                 "epoch": CKPTS[s][1]} for s in seeds},
        "dataset_hashes": {"wfe_tsv": sha(os.path.join(ROOT, WFE))},
        "lexicon_hashes": {
            "lexicon_file_sha256": ck0["lexicon_file_sha256"],
            "ordered_training_words_sha256": ck0["ordered_training_words_sha256"]},
        "vocabulary_hash": hashlib.sha256(
            "\n".join(vocab0.itos).encode()).hexdigest(),
        "architecture_configuration": {
            "ltm_encoder_mode": ck0["cfg_ltm"]["ltm_encoder_mode"],
            "wm_hidden": ck0["cfg_wm"]["hidden"],
            "premotor_dim": ck0["premotor_dim"],
            "gate_alpha": ck0["cfg_gating"]["alpha"],
            "gate_threshold": ck0["cfg_gating"]["gate_threshold"],
            "gate_usage_prior": ck0["cfg_gating"]["usage_prior"]},
        "decode_mode": "deterministic autoregressive + gold-prefix diagnostic",
        "prefix_modes": ["gold", "ltm_generated", "full_generated"],
        "top_k": args.top_k,
        "noise_settings": {"wm_interference_noise": 0.0, "ltm_ventral_noise": 0.0,
                           "apply_noise": False},
        "readout_mode": "forced-length (bounded by gold target length)",
        "instrumented_script_sha256": {
            p: sha(os.path.join(ROOT, "scripts/length_effect_analysis", p))
            for p in ("instrument.py", "run_instrumented.py", "bank_audit.py",
                      "preflight_equivalence.py")},
        "metric_conventions": {
            "entropy_log_base": I.ENTROPY_LOG_BASE,
            "entropy_vocabulary": I.ENTROPY_VOCAB,
            "target_rank_tie_policy": I.TARGET_RANK_TIE_POLICY,
            "special_token_treatment": I.SPECIAL_NOTE,
            "cosine_tie_policy": COSINE_TIE_POLICY,
            "phonological_distance_tie_policy": PHON_TIE_POLICY},
        "canonical_equivalence": verdict,
        "training_performed": False, "weights_modified": False,
        "architecture_changed": False,
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
        f.write("\n")
    with open(os.path.join(args.out_dir, "run_manifest.json"), "w") as f:
        json.dump({"seeds": seeds, "n_items": len(item_rows) // len(seeds),
                   "item_rows": len(item_rows), "timestep_rows": len(ts_rows),
                   "neighbor_rows": len(neigh_rows),
                   "equivalence_comparisons": len(eq),
                   "equivalence_verdict": verdict,
                   "streams": ["A_gold_prefix", "B_ltm_ar", "C_full_ar"],
                   "wm_only_ar_stream_run": False,
                   "elapsed_seconds": round(time.time() - t0, 1)}, f, indent=2)
        f.write("\n")
    print(f"rows: item={len(item_rows)} timestep={len(ts_rows)} "
          f"neighbors={len(neigh_rows)}  elapsed={time.time()-t0:.0f}s")
    return 0 if verdict == "PASS" else 1


def _pad(a, maxlen=10):
    out = np.full((a.shape[0], maxlen, a.shape[2]), np.nan, dtype=np.float32)
    out[:, :a.shape[1]] = a
    return out


def _ts(seed, cksha, item, route, decode_mode, prefix_source, rec, L, vocab,
        ar=False):
    tid = rec["target_token_id"]
    pid = rec.get("predicted_token_id") if ar else rec.get(f"{route}_pred_id")
    if pid is None:
        pid = rec.get(f"{route}_pred_id")
    t = rec["timestep"]
    return {"seed": seed, "checkpoint_sha256": cksha, "item_id": item,
            "route": route, "decode_mode": decode_mode,
            "prefix_source": prefix_source, "timestep": t,
            "absolute_position": t + 1,
            "relative_position": (t / (L - 1)) if L > 1 else 0.0,
            "target_token_id": tid, "target_token": vocab.itos[tid],
            "predicted_token_id": pid, "predicted_token": vocab.itos[pid],
            "is_correct": int(pid == tid),
            "is_eos_prediction": int(pid == vocab.eos_id),
            "target_rank": rec.get(f"{route}_target_rank"),
            "target_logit": rec.get(f"{route}_target_logit"),
            "best_non_target_logit": rec.get(f"{route}_best_non_target_logit"),
            "target_margin": rec.get(f"{route}_target_margin"),
            "entropy": rec.get(f"{route}_entropy"),
            "entropy_phoneme_only": rec.get(f"{route}_entropy_phon"),
            "gate": rec.get("gate")}


if __name__ == "__main__":
    sys.exit(main())
