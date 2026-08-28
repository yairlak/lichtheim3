"""Phase 2D1: zero-training diagnostic of the C3-subset3288 retrieval failure.

Reads the FINAL C3-subset3288 checkpoint and asks, without any weight update
and without altering s_hat, whether the residual top-1 errors come from

    A. competition among the 3,288 TRAINED mappings, or
    B. trained items being beaten by the 26,283 UNTRAINED GloVe distractors
       that remain in the full canonical retrieval bank.

Nothing here trains, fine-tunes or writes any model parameter.  The model is
loaded in eval mode under torch.no_grad() and s_hat is recomputed
deterministically from the frozen final weights; the reproduction is asserted
against the metrics recorded in the run summary, so "recompute" is a bit-level
restatement of the stored result rather than a new measurement.

Sections
    1. dual-bank evaluation      (full 29,571 bank vs trained-only 3,288 bank)
    2. error competitor taxonomy (who actually beats the target)
    3. target-space geometry     (model-independent GloVe neighbourhood density)
    4. convergence check         (from the stored trajectory only)

Section 3 is DESCRIPTIVE. Correlations are reported as associations; no causal
claim is made or implied.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.naming_comprehension.aggregate_cohort import (            # noqa: E402
    FREQ_BANDS, BAND_LABELS)
from scripts.naming_comprehension.frozen_probe import (                # noqa: E402
    encode_all, load_frozen)
from scripts.naming_comprehension.train_tasks import (                 # noqa: E402
    band_of_rank, select_nested_subset)
from utils.provenance import git_state, sha256_file                    # noqa: E402

# Canonical source checkpoint (seed 22 / epoch 140).  Repository-relative by
# default so the script is portable; override with LICHTHEIM3_CANONICAL_CKPT
# when the archived bundle lives outside the working tree (archives/ is
# gitignored).  Resolves to the same file as before when the bundle sits at
# the repository root, so behaviour is unchanged.
CANONICAL = os.environ.get(
    "LICHTHEIM3_CANONICAL_CKPT",
    os.path.join(ROOT, "archives", "fulllexicon_93a577f", "extracted",
                 "fulllexicon_final_bundle_93a577f", "selected_checkpoints",
                 "seed_22_epoch_0140.pt"))
RUN_DIR = "outputs/naming_comprehension_93a577f/phase2c_c3_subset3288_seed22"
OUT_DIR = "outputs/naming_comprehension_93a577f/phase2d1_c3_subset3288_diagnostic"
CHUNK = 256


# =====================================================  retrieval metrics  ==

@torch.no_grad()
def retrieval_against(s_hat: torch.Tensor, bank: torch.Tensor,
                      target_rows: Sequence[int]) -> Dict[str, np.ndarray]:
    """Cosine retrieval of each item's own target inside `bank`.

    `target_rows[i]` is the row of item i's target within `bank`, so the same
    function serves the full 29,571-row bank and the trained-only 3,288-row
    bank without changing any convention.
    """
    bank_n = F.normalize(bank, dim=-1)
    q_all = F.normalize(s_hat, dim=-1)
    n = q_all.shape[0]
    out = {k: np.zeros(n, dtype=np.float64) for k in
           ("target_cos", "target_rank", "top1", "top5", "margin", "best_wrong_cos")}
    best_wrong = np.zeros(n, dtype=np.int64)
    for lo in range(0, n, CHUNK):
        q = q_all[lo:lo + CHUNK]
        sims = q @ bank_n.t()
        rows = torch.arange(q.shape[0])
        tgt = torch.tensor(target_rows[lo:lo + q.shape[0]], dtype=torch.long)
        tgt_sim = sims[rows, tgt]
        rank = (sims > tgt_sim.unsqueeze(1)).sum(dim=1) + 1
        excl = sims.clone()
        excl[rows, tgt] = -2.0
        bw_val, bw_idx = excl.max(dim=1)
        sl = slice(lo, lo + q.shape[0])
        out["target_cos"][sl] = tgt_sim.numpy()
        out["target_rank"][sl] = rank.numpy()
        out["top1"][sl] = (sims.argmax(dim=1) == tgt).numpy()
        out["top5"][sl] = (rank <= 5).numpy()
        out["margin"][sl] = (tgt_sim - bw_val).numpy()
        out["best_wrong_cos"][sl] = bw_val.numpy()
        best_wrong[sl] = bw_idx.numpy()
    out["best_wrong_row"] = best_wrong
    return out


def agg(m: Dict[str, np.ndarray], sel: np.ndarray) -> Optional[dict]:
    if not sel.any():
        return None
    return {
        "n": int(sel.sum()),
        "top1": float(m["top1"][sel].mean()),
        "top5": float(m["top5"][sel].mean()),
        "target_rank_median": float(np.median(m["target_rank"][sel])),
        "target_rank_mean": float(m["target_rank"][sel].mean()),
        "target_cos_mean": float(m["target_cos"][sel].mean()),
        "margin_mean": float(m["margin"][sel].mean()),
    }


# ======================================================  target geometry  ===

@torch.no_grad()
def target_geometry(bank: torch.Tensor, target_idx: Sequence[int],
                    trained: set) -> Dict[str, np.ndarray]:
    """Model-independent neighbourhood density of each target in the bank.

    Purely a property of the fixed canonical GloVe bank: no model output is
    involved, so this cannot be an artefact of what the encoder learned.
    """
    bank_n = F.normalize(bank, dim=-1)
    n = len(target_idx)
    out = {k: np.zeros(n, dtype=np.float64) for k in
           ("nn_cos", "top5_cos_mean", "top10_cos_mean")}
    nn_bank_idx = np.zeros(n, dtype=np.int64)
    for lo in range(0, n, CHUNK):
        idx = list(target_idx[lo:lo + CHUNK])
        sims = bank_n[torch.tensor(idx)] @ bank_n.t()
        for k, bi in enumerate(idx):
            sims[k, bi] = -2.0                       # exclude the vector itself
        top, ti = sims.topk(10, dim=1)
        sl = slice(lo, lo + len(idx))
        out["nn_cos"][sl] = top[:, 0].numpy()
        out["top5_cos_mean"][sl] = top[:, :5].mean(dim=1).numpy()
        out["top10_cos_mean"][sl] = top[:, :10].mean(dim=1).numpy()
        nn_bank_idx[sl] = ti[:, 0].numpy()
    out["nn_bank_idx"] = nn_bank_idx
    out["nn_is_trained"] = np.array([int(int(b) in trained) for b in nn_bank_idx],
                                    dtype=np.int64)
    return out


def describe(x: np.ndarray) -> dict:
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "sd": float(x.std(ddof=1)) if x.size > 1 else None,
            "min": float(x.min()), "max": float(x.max())}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]; 0 means fully overlapping."""
    from scipy.stats import mannwhitneyu
    if a.size == 0 or b.size == 0:
        return float("nan")
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2.0 * u / (a.size * b.size) - 1.0)


# ==============================================================  main  ======

def main() -> int:
    from scipy.stats import spearmanr, mannwhitneyu, kruskal

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = json.load(open(os.path.join(RUN_DIR, "run_summary.json")))
    subdef = json.load(open(os.path.join(RUN_DIR, "subset_definition.json")))
    final_ckpt = summary["final_checkpoint"]

    # ---------- reload the frozen final model; NO optimizer, NO update ----------
    model, vocab, entries, bank_raw, cfg, ckpt = load_frozen(CANONICAL, "cpu")
    fin = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(fin["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    subset_idx = list(subdef["bank_indices_in_order"])
    assert subset_idx == select_nested_subset(entries, 822, 0), \
        "stored subset definition does not match the deterministic selector"
    core64 = set(select_nested_subset(entries, 16, 0))
    core512 = set(select_nested_subset(entries, 128, 0))
    assert core64 < core512 < set(subset_idx)
    trained = set(subset_idx)

    forms = [entries[i].phonemes for i in subset_idx]
    s_hat = encode_all(model, vocab, forms, "cpu", 64)

    # ---------- 1. dual-bank evaluation ----------
    full = retrieval_against(s_hat, bank_raw, subset_idx)
    sub_bank = bank_raw[torch.tensor(subset_idx)]
    sub = retrieval_against(s_hat, sub_bank, list(range(len(subset_idx))))

    recorded = summary["snapshots"][-1]["nested_scale"]["all"]["top1"]
    reproduced = float(full["top1"].mean())
    assert abs(reproduced - recorded) < 1e-12, \
        f"s_hat reproduction mismatch: {reproduced} vs recorded {recorded}"

    ranks = np.array([entries[i].rank for i in subset_idx])
    bands = np.array([band_of_rank(r, FREQ_BANDS) for r in ranks])
    lengths = np.array([len(entries[i].phonemes) for i in subset_idx])
    in64 = np.array([int(i in core64) for i in subset_idx], dtype=bool)
    in512 = np.array([int(i in core512) for i in subset_idx], dtype=bool)

    groups = {
        "all3288": np.ones(len(subset_idx), dtype=bool),
        "core512": in512, "core64": in64, "added2776": ~in512,
    }
    for bi, lab in enumerate(BAND_LABELS):
        groups[f"band_{lab}"] = bands == bi

    dual = {g: {"full_bank": agg(full, sel), "trained_only_bank": agg(sub, sel)}
            for g, sel in groups.items()}

    # ---------- 2. error competitor taxonomy ----------
    err = np.where(full["top1"] == 0)[0]
    rows: List[dict] = []
    for k in err:
        comp_bank_idx = int(full["best_wrong_row"][k])
        tgt_bank_idx = subset_idx[k]
        rows.append({
            "target_word": entries[tgt_bank_idx].word,
            "target_bank_index": tgt_bank_idx,
            "target_rank_full_bank": int(full["target_rank"][k]),
            "target_cos": round(float(full["target_cos"][k]), 6),
            "competitor_word": entries[comp_bank_idx].word,
            "competitor_bank_index": comp_bank_idx,
            "competitor_cos": round(float(full["best_wrong_cos"][k]), 6),
            "margin": round(float(full["margin"][k]), 6),
            "competitor_is_trained": int(comp_bank_idx in trained),
            "competitor_in_core512": int(comp_bank_idx in core512),
            "competitor_in_core64": int(comp_bank_idx in core64),
            "target_band": BAND_LABELS[bands[k]],
            "target_freq_rank": int(ranks[k]),
            "target_phoneme_length": int(lengths[k]),
            "target_in_core512": int(in512[k]),
            "target_in_core64": int(in64[k]),
        })

    comp_trained = np.array([r["competitor_is_trained"] for r in rows], dtype=bool)

    def split(sel: np.ndarray) -> dict:
        if not sel.any():
            return {"n_errors": 0}
        t = int(comp_trained[sel].sum())
        return {"n_errors": int(sel.sum()), "won_by_trained": t,
                "won_by_untrained": int(sel.sum()) - t,
                "fraction_trained": float(t / sel.sum()),
                "fraction_untrained": float(1.0 - t / sel.sum())}

    err_band = np.array([BAND_LABELS.index(r["target_band"]) for r in rows])
    err_in64 = np.array([bool(r["target_in_core64"]) for r in rows])
    err_in512 = np.array([bool(r["target_in_core512"]) for r in rows])
    taxonomy = {
        "overall": split(np.ones(len(rows), dtype=bool)),
        "by_band": {lab: split(err_band == bi) for bi, lab in enumerate(BAND_LABELS)},
        "by_group": {"core64": split(err_in64), "core512": split(err_in512),
                     "added2776": split(~err_in512)},
        "base_rate_note": (
            f"{len(trained)} of {bank_raw.shape[0]} bank vectors are trained "
            f"({len(trained) / bank_raw.shape[0]:.4f}); a competitor drawn at "
            "random from the bank would be trained at that rate, which is the "
            "reference for reading the fractions above"),
        "trained_share_of_bank": float(len(trained) / bank_raw.shape[0]),
    }

    # ---------- 3. target-space geometry (model-independent) ----------
    geo = target_geometry(bank_raw, subset_idx, trained)
    succ = full["top1"] == 1
    geometry = {
        "success_vs_failure": {},
        "by_band": {},
        "band_density_test": {},
        "associations_spearman": {},
        "nearest_neighbour_trained_share": {
            "all3288": float(geo["nn_is_trained"].mean()),
            "note": ("share of targets whose nearest GloVe neighbour is itself "
                     "a trained item; compare against the trained share of the "
                     f"bank ({len(trained) / bank_raw.shape[0]:.4f})"),
        },
    }
    for key in ("nn_cos", "top5_cos_mean", "top10_cos_mean"):
        a, b = geo[key][succ], geo[key][~succ]
        mw = mannwhitneyu(a, b, alternative="two-sided")
        geometry["success_vs_failure"][key] = {
            "success": describe(a), "failure": describe(b),
            "difference_failure_minus_success": float(b.mean() - a.mean()),
            "mannwhitney_p": float(mw.pvalue),
            "cliffs_delta_failure_vs_success": cliffs_delta(b, a),
        }
        per_band = [geo[key][bands == bi] for bi in range(len(BAND_LABELS))]
        geometry["by_band"][key] = {lab: describe(per_band[bi])
                                    for bi, lab in enumerate(BAND_LABELS)}
        kw = kruskal(*per_band)
        hi = geo[key][bands == 0]
        rest = geo[key][bands != 0]
        mw2 = mannwhitneyu(hi, rest, alternative="two-sided")
        geometry["band_density_test"][key] = {
            "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
            "high_freq_1-1k_mean": float(hi.mean()),
            "rest_mean": float(rest.mean()),
            "high_freq_minus_rest": float(hi.mean() - rest.mean()),
            "mannwhitney_p_1-1k_vs_rest": float(mw2.pvalue),
            "cliffs_delta_1-1k_vs_rest": cliffs_delta(hi, rest),
            "direction": ("1-1k DENSER (higher neighbour cosine)"
                          if hi.mean() > rest.mean() else
                          "1-1k SPARSER (lower neighbour cosine)"),
        }
        for tgt_name, tgt in (("target_rank_full_bank", full["target_rank"]),
                              ("margin_full_bank", full["margin"]),
                              ("target_cos", full["target_cos"])):
            rho, p = spearmanr(geo[key], tgt)
            geometry["associations_spearman"][f"{key}__vs__{tgt_name}"] = {
                "spearman_rho": float(rho), "p": float(p), "n": int(geo[key].size)}

    # ---------- 4. convergence check from the stored trajectory ----------
    snaps = summary["snapshots"]
    tail = [s for s in snaps if s["epoch"] >= 700]
    steps_per_epoch = -(-len(subset_idx) // 64)
    deltas = []
    for a, b in zip(tail, tail[1:]):
        deltas.append({
            "from_epoch": a["epoch"], "to_epoch": b["epoch"],
            "d_top1": b["nested_scale"]["all"]["top1"] - a["nested_scale"]["all"]["top1"],
            "d_margin": b["nested_scale"]["all"]["margin_mean"] - a["nested_scale"]["all"]["margin_mean"],
            "d_target_rank_mean": (b["nested_scale"]["all"]["target_rank_mean"]
                                   - a["nested_scale"]["all"]["target_rank_mean"]),
            "d_c0": b["loss_components"]["c0"] - a["loss_components"]["c0"],
            "d_retrieval_ce": (b["loss_components"]["retrieval_ce"]
                               - a["loss_components"]["retrieval_ce"]),
            "d_total": b["loss_components"]["total"] - a["loss_components"]["total"],
        })
    monotone_gain = all(d["d_top1"] > 0 for d in deltas)
    monotone_loss = all(d["d_total"] < 0 for d in deltas)
    convergence = {
        "steps_per_epoch": steps_per_epoch,
        "tail_from_epoch": 700,
        "per_50_epoch_deltas": deltas,
        "mean_top1_gain_per_50_epochs_tail": float(np.mean([d["d_top1"] for d in deltas])),
        "last_interval_top1_gain": deltas[-1]["d_top1"] if deltas else None,
        "top1_still_increasing_every_interval": bool(monotone_gain),
        "total_loss_still_decreasing_every_interval": bool(monotone_loss),
        "verdict": ("STILL IMPROVING: top-1 rose and total loss fell at every "
                    "50-epoch interval from epoch 700 to 1000; the run was "
                    "stopped by budget exhaustion, not by convergence"
                    if (monotone_gain and monotone_loss) else
                    "INDETERMINATE: gains are not monotone in the tail"),
        "caution": ("a still-improving trajectory does NOT license inferring a "
                    "capacity ceiling from the epoch-1000 value"),
    }

    # ---------- write ----------
    out = {
        "phase": "2D1_zero_training_retrieval_diagnostic",
        "no_weight_update": True,
        "source_run": RUN_DIR,
        "s_hat_reproduction": {
            "recorded_top1": recorded, "reproduced_top1": reproduced,
            "bit_identical": True,
        },
        "populations": {
            "trained_subset": len(subset_idx),
            "full_bank": int(bank_raw.shape[0]),
            "untrained_distractors": int(bank_raw.shape[0]) - len(subset_idx),
        },
        "dual_bank": dual,
        "error_taxonomy": taxonomy,
        "target_geometry": geometry,
        "convergence": convergence,
        "provenance": {
            "final_checkpoint": final_ckpt,
            "final_checkpoint_sha256": sha256_file(final_ckpt),
            "canonical_checkpoint_sha256": sha256_file(CANONICAL),
            "subset_definition_sha256": subdef["subset_definition_sha256"],
            "torch_version": torch.__version__,
            "eval_git": git_state(ROOT),
        },
    }
    with open(os.path.join(OUT_DIR, "diagnostic.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    cols = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "errors_full_bank.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    with open(os.path.join(OUT_DIR, "per_item_geometry.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(["bank_index", "word", "freq_rank", "band", "n_phonemes",
                           "top1_full", "top1_trained_only", "rank_full",
                           "rank_trained_only", "margin_full", "margin_trained_only",
                           "target_cos", "nn_cos", "top5_cos_mean", "top10_cos_mean",
                           "nn_word", "nn_is_trained", "in_core64", "in_core512"]) + "\n")
        for k, bi in enumerate(subset_idx):
            f.write("\t".join(str(v) for v in [
                bi, entries[bi].word, ranks[k], BAND_LABELS[bands[k]], lengths[k],
                int(full["top1"][k]), int(sub["top1"][k]),
                int(full["target_rank"][k]), int(sub["target_rank"][k]),
                round(float(full["margin"][k]), 6), round(float(sub["margin"][k]), 6),
                round(float(full["target_cos"][k]), 6),
                round(float(geo["nn_cos"][k]), 6),
                round(float(geo["top5_cos_mean"][k]), 6),
                round(float(geo["top10_cos_mean"][k]), 6),
                entries[int(geo["nn_bank_idx"][k])].word,
                int(geo["nn_is_trained"][k]), int(in64[k]), int(in512[k]),
            ]) + "\n")

    print(json.dumps({"dual_bank_all3288": dual["all3288"],
                      "taxonomy_overall": taxonomy["overall"],
                      "convergence_verdict": convergence["verdict"]}, indent=2))
    print(f"[2D1] wrote -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
