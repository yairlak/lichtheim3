"""Phase 1B: cohort-level aggregation of the frozen naming/comprehension probe.

Reads the per-seed outputs produced by frozen_probe.py and emits one compact
machine-readable table (one row per seed) plus cohort mean/SD/min/max.

No scientific decision is made here: populations, metrics, homophone policy,
frequency bands and decoding convention are all inherited unchanged from
Phase 1A.  Frequency bands are frozen at the Phase 1A definition:
    1-1k, 1k-5k, 5k-15k, 15k-end   (on the lexicon frequency rank)

With n=4 checkpoints the cohort statistics are descriptive only (mean, sample
SD, min, max).  No inferential statistics are computed: 4 seeds characterise
robustness across checkpoints, not population significance.

Usage:
    python scripts/naming_comprehension/aggregate_cohort.py \
        --run-dir outputs/naming_comprehension_93a577f \
        --seeds 19:0155 20:0130 21:0145 22:0140
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Phase 1A frequency bands — frozen, do not redefine.
FREQ_BANDS = ((1, 1000), (1000, 5000), (5000, 15000), (15000, 10 ** 9))
BAND_LABELS = ("1-1k", "1k-5k", "5k-15k", "15k-end")


def _mean(xs: List[float]) -> float:
    return float(st.mean(xs)) if xs else float("nan")


def glove_coverage(ckpt_path: str) -> Dict[str, object]:
    """GloVe coverage/fallback status recorded in the checkpoint provenance."""
    import torch
    if not os.path.exists(ckpt_path):
        raise SystemExit(
            f"Checkpoint referenced by a per-seed summary is not available:\n"
            f"  {ckpt_path}\n"
            "The aggregator reads GloVe coverage from the checkpoint itself; "
            "re-run the per-seed probe or restore the archive before aggregating.")
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "ckpt_seed": c["cfg_train"]["seed"],
        "ckpt_total_epochs_trained": c["total_epochs_trained"],
        "glove_present": bool(c["glove_present"]),
        "n_glove_found": int(c["n_glove_found"]),
        "n_glove_fallback": int(c["n_glove_fallback"]),
    }


def load_seed(run_dir: str, seed: str, epoch: str) -> Dict[str, object]:
    d = os.path.join(run_dir, f"seed{seed}_e{epoch.lstrip('0')}")
    if not os.path.isdir(d):
        d = os.path.join(run_dir, f"seed{seed}_e{epoch}")
    summ = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
    with open(os.path.join(d, "per_item.tsv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    strict = [r for r in rows if r["is_homophone"] == "0"]
    homo = [r for r in rows if r["is_homophone"] == "1"]
    c = summ["comprehension"]
    row: Dict[str, object] = {
        "seed": int(seed),
        "epoch": int(epoch),
        # --- comprehension, unique-phonology population (primary) ---
        "comp_n": c["strict_unique_phonology"]["n"],
        "comp_target_cos_mean": c["strict_unique_phonology"]["comp_target_cos_mean"],
        "comp_target_rank_median": st.median(
            [int(r["comp_target_rank"]) for r in strict]),
        "comp_top1": c["strict_unique_phonology"]["comp_top1_mean"],
        "comp_top5": c["strict_unique_phonology"]["comp_top5_mean"],
        "comp_margin_mean": c["strict_unique_phonology"]["comp_margin_mean"],
        "comp_c_ltm_mean_aux": c["strict_unique_phonology"]["comp_c_ltm_mean"],
        # --- homophones, reported separately ---
        "homo_n": c["homophones_separate"]["n"],
        "homo_top1_strict": c["homophones_separate"]["comp_top1_mean"],
        "homo_top5": c["homophones_separate"]["comp_top5_mean"],
        "homo_class_aware_top1_aux":
            c["homophones_separate"]["comp_top1_same_phonology_aux_mean"],
        # --- shift diagnostics ---
        "shift_cos_mean": summ["distribution_shift"]["cos_shat_glove_mean"],
        "shift_cos_median": summ["distribution_shift"]["cos_shat_glove_median"],
        "shift_l2_mean": summ["distribution_shift"]["l2_mean"],
        "shift_mse_mean": summ["distribution_shift"]["mse_mean"],
        "norm_shat_mean": summ["distribution_shift"]["norm_shat_mean"],
        "norm_glove_mean": summ["distribution_shift"]["norm_glove_mean"],
        # --- provenance echo ---
        "ckpt_path": summ["provenance"]["checkpoint_path"],
        "ckpt_sha256": summ["provenance"]["checkpoint_sha256"],
        "ckpt_training_commit": summ["provenance"]["checkpoint_training_commit"],
        "eval_git_commit": summ["provenance"]["eval_git"]["commit"],
        "lexicon_file_sha256": summ["provenance"]["lexicon_file_sha256"],
        "ordered_bank_sha256": summ["provenance"]["ordered_training_words_sha256"],
        "decoding_cap": summ["naming_convention"]["global_cap_steps"],
        "ltm_encoder_mode": summ["provenance"]["ltm_encoder_mode"],
    }
    # GloVe coverage as recorded by the checkpoint at training time; the probe's
    # fail-fast guard already refused to run on any lexicon with more fallback
    # vectors than this, so a completed run certifies real-GloVe semantics.
    row.update(glove_coverage(summ["provenance"]["checkpoint_path"]))
    for cond in ("glove", "shat"):
        n = summ["naming"][cond]
        row[f"naming_{cond}_exact"] = n[f"naming_{cond}_exact_mean"]
        row[f"naming_{cond}_wer"] = n["whole_word_error_rate"]
        row[f"naming_{cond}_edit_mean"] = n[f"naming_{cond}_edit_mean"]
        row[f"naming_{cond}_eos_rate"] = n[f"naming_{cond}_eos_emitted_mean"]
    # frequency gradient on the strict population, frozen Phase 1A bands
    for (lo, hi), lab in zip(FREQ_BANDS, BAND_LABELS):
        band = [r for r in strict if lo <= int(r["freq_rank"]) < hi]
        row[f"comp_top1_rank_{lab}"] = _mean([int(r["comp_top1"]) for r in band])
        row[f"n_rank_{lab}"] = len(band)
    return row


# Fields that MUST be identical across every seed for a cohort table to mean
# anything: they define the data, the retrieval problem and the eval code.
COHORT_INVARIANTS: Tuple[str, ...] = (
    "lexicon_file_sha256", "ordered_bank_sha256", "ltm_encoder_mode",
    "decoding_cap", "eval_git_commit", "ckpt_training_commit",
)


def check_cohort_provenance(rows: List[Dict[str, object]]) -> Dict[str, object]:
    """Verify the seeds are actually aggregatable, BEFORE computing statistics.

    Raises on any mismatch rather than silently averaging incomparable runs.
    """
    if not rows:
        raise SystemExit("No per-seed rows to aggregate.")
    shared: Dict[str, object] = {}
    for field in COHORT_INVARIANTS:
        values = {r[field] for r in rows}
        if len(values) != 1:
            detail = ", ".join(f"seed{r['seed']}={r[field]!r}" for r in rows)
            raise SystemExit(
                f"Cohort provenance mismatch on {field!r}: seeds are not "
                f"comparable and must not be aggregated.\n  {detail}")
        shared[field] = values.pop()

    # Each seed must be a DISTINCT checkpoint (guards against aggregating the
    # same file several times through duplicated --seeds arguments).
    shas = [r["ckpt_sha256"] for r in rows]
    if len(set(shas)) != len(shas):
        raise SystemExit(
            f"Duplicate checkpoint SHA256 across seeds: {shas}. "
            "Each cohort row must be a different checkpoint.")
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"Duplicate seeds in cohort: {seeds}.")
    for r in rows:
        if r["seed"] != r["ckpt_seed"]:
            raise SystemExit(
                f"Seed label {r['seed']} does not match the seed recorded in "
                f"the checkpoint ({r['ckpt_seed']}).")
        if not r["glove_present"] or r["n_glove_fallback"] != 0:
            raise SystemExit(
                f"seed{r['seed']}: real-GloVe semantics not certified "
                f"(glove_present={r['glove_present']}, "
                f"n_glove_fallback={r['n_glove_fallback']}).")
    # Populations must match across seeds (same lexicon, same partition rule).
    for field in ("comp_n", "homo_n"):
        if len({r[field] for r in rows}) != 1:
            raise SystemExit(
                f"Population size {field!r} differs across seeds: "
                f"{[(r['seed'], r[field]) for r in rows]}.")
    shared["verified"] = True
    shared["n_seeds"] = len(rows)
    return shared


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seeds", nargs="+", required=True,
                    help="seed:epoch pairs, e.g. 19:0155")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or os.path.join(args.run_dir, "cohort")

    rows = [load_seed(args.run_dir, *s.split(":")) for s in args.seeds]
    rows.sort(key=lambda r: r["seed"])
    cohort_provenance = check_cohort_provenance(rows)   # fail fast, before stats

    principal = [
        "comp_target_cos_mean", "comp_target_rank_median", "comp_top1",
        "comp_top5", "comp_margin_mean", "comp_c_ltm_mean_aux",
        "homo_top1_strict", "homo_top5", "homo_class_aware_top1_aux",
        "naming_glove_exact", "naming_glove_wer", "naming_glove_edit_mean",
        "naming_glove_eos_rate", "naming_shat_exact", "naming_shat_wer",
        "naming_shat_edit_mean", "naming_shat_eos_rate",
        "shift_cos_mean", "shift_cos_median", "shift_l2_mean", "shift_mse_mean",
        "norm_shat_mean", "norm_glove_mean",
    ] + [f"comp_top1_rank_{lab}" for lab in BAND_LABELS]

    cohort = {}
    for k in principal:
        vals = [float(r[k]) for r in rows]
        cohort[k] = {
            "mean": _mean(vals),
            "sd": float(st.stdev(vals)) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    os.makedirs(out_dir, exist_ok=True)
    cols = list(rows[0].keys())
    tsv = os.path.join(out_dir, "cohort_by_seed.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    payload = {
        "cohort": "fulllexicon_93a577f_seeds19_22",
        "n_checkpoints": len(rows),
        "statistics_note": ("descriptive only (mean, sample SD, min, max); "
                            "no inferential statistics with n=4"),
        "frequency_bands": {lab: list(b) for lab, b in zip(BAND_LABELS, FREQ_BANDS)},
        "frequency_bands_note": "frozen at Phase 1A definition; not redefined",
        "cohort_provenance": cohort_provenance,
        "by_seed": rows,
        "cohort_stats": cohort,
    }
    js = os.path.join(out_dir, "cohort_summary.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[aggregate_cohort] {len(rows)} seeds -> {tsv}")
    print(f"[aggregate_cohort] summary -> {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
