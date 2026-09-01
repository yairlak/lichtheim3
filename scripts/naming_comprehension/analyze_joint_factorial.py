#!/usr/bin/env python3
"""Phase 4A3d — synthesis of the seed22 2x2 objective factorial.

ANALYSIS ONLY. This script trains nothing, resumes nothing and never writes
into a scientific run directory. It reads the four completed cells, rebuilds
one canonical trajectory table, computes the descriptive factorial contrasts,
optionally audits the N-only WM endpoint anomaly, and draws the figure set.

The factorial:

    condition | retrieval | naming
    ----------+-----------+--------
    H0        |    OFF    |  OFF
    C-only    |    ON     |  OFF
    N-only    |    OFF    |  ON
    J0        |    ON     |  ON

H0 and J0 were run in two segments (e0->e160, then an exact-resume extension to
e440); their trajectories are stitched here, dropping the duplicated e160 row.
C-only and N-only are single e0->e440 runs.

Run (from the repository root):

    python scripts/naming_comprehension/analyze_joint_factorial.py \
        --seed 22 --output reports/joint_scratch_factorial_seed22

Add --wm-audit to re-run the canonical full-lexicon AR evaluator over the four
final checkpoints and write the per-item WM/LTM/FULL error tables. That pass is
read-only but costs a few minutes, so it is opt-in.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LEXICON_N = 29571
SUBSET_N = 3288
FINAL_EPOCH = 440
FINAL_STEP = 203_720

# Each condition maps to the run directories that supply its trajectory, in
# order. A second entry is a continuation whose rows are kept only where they
# advance past the first segment's last step.
CONDITIONS: Dict[str, dict] = {
    "H0": {
        "regime": "h0", "retrieval": False, "naming": False,
        "dirs": ["outputs/joint_scratch/phase4a2a_h0_seed22_e160",
                 "outputs/joint_scratch/phase4a2c_h0_seed22_e440"],
    },
    "C-only": {
        "regime": "c_only", "retrieval": True, "naming": False,
        "dirs": ["outputs/joint_scratch/phase4a3b_c_only_seed22_e440"],
    },
    "N-only": {
        "regime": "n_only", "retrieval": False, "naming": True,
        "dirs": ["outputs/joint_scratch/phase4a3c_n_only_seed22_e440"],
    },
    "J0": {
        "regime": "j0", "retrieval": True, "naming": True,
        "dirs": ["outputs/joint_scratch/phase4a2b_j0_seed22_e160",
                 "outputs/joint_scratch/phase4a2c_j0_seed22_e440"],
    },
}
ORDER = ("H0", "C-only", "N-only", "J0")

# One visual identity per condition, reused unchanged across every figure.
# Line styles are not decoration: several curves coincide exactly (H0 and
# C-only sit on the naming floor, N-only and J0 on the naming ceiling), so a
# thick solid line under a thin broken one is what keeps both visible without
# displacing any value.
COLOR = {"H0": "#8a8a8a", "C-only": "#1b6ca8", "N-only": "#e08b1e", "J0": "#c0392b"}
LINESTYLE = {"H0": "-", "C-only": ":", "N-only": "-", "J0": "--"}
LINEWIDTH = {"H0": 2.8, "C-only": 1.9, "N-only": 2.8, "J0": 1.9}

# metrics.tsv column -> canonical column in the derived table
COLMAP = {
    "rep_wm": "rep_wm", "rep_ltm": "rep_ltm", "rep_full": "rep_full",
    "comp_top1": "comp_top1", "comp_top5": "comp_top5",
    "comp_rank_median": "comp_rank_median", "comp_rank_mean": "comp_rank_mean",
    "comp_cos_mean": "comp_cosine", "comp_margin_mean": "comp_margin",
    "naming_exact": "naming_exact", "naming_wer": "naming_wer",
    "naming_mean_edit": "naming_edit", "naming_eos_rate": "naming_eos",
    "naming_pred_len_mean": "naming_pred_len",
    "naming_target_len_mean": "naming_target_len",
    "probe_rep_ltm": "probe_rep_ltm", "probe_rep_full": "probe_rep_full",
    "full_rep_ltm": "full_rep_ltm", "full_rep_wm": "full_rep_wm",
    "full_rep_full": "full_rep_full",
}
TABLE_COLS = (["condition", "retrieval_enabled", "naming_enabled", "seed",
               "rep_epoch", "step"] + list(dict.fromkeys(COLMAP.values())))


def die(msg: str) -> None:
    raise SystemExit(f"[analyze_joint_factorial] FATAL: {msg}")


def fnum(v: str) -> Optional[float]:
    """Missing stays missing: never interpolated, never forward-filled."""
    if v is None or v == "" or v == "nan":
        return None
    return float(v)


def errors(acc: Optional[float], n: int) -> Optional[int]:
    return None if acc is None else round((1.0 - acc) * n)


# =====================================================  loading  ==========

def load_condition(name: str, seed: int) -> List[dict]:
    """One trajectory per condition, at most one row per optimizer step.

    Two kinds of duplicate have to be collapsed, and they are collapsed
    differently:

    * Within a run file, the last step carries two rows -- the cadence
      evaluation (full_rep_* absent) followed by the endpoint evaluation
      (full_rep_* present). The later row is a strict superset, so it wins.
    * Across a stitched H0/J0 pair, the first segment's final step would repeat
      the continuation's starting point. The continuation never re-evaluates a
      step the first segment already produced, so ordering by step and keeping
      the last occurrence handles both cases without forward-filling anything.
    """
    spec = CONDITIONS[name]
    by_step: Dict[int, dict] = {}
    for d in spec["dirs"]:
        path = os.path.join(ROOT, d, "metrics.tsv")
        if not os.path.exists(path):
            die(f"{name}: required source missing: {d}/metrics.tsv")
        seg = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
        if not seg:
            die(f"{name}: empty metrics file {d}/metrics.tsv")
        for r in seg:
            step = int(r["step"])
            out = {"condition": name,
                   "retrieval_enabled": spec["retrieval"],
                   "naming_enabled": spec["naming"],
                   "seed": seed,
                   "rep_epoch": int(r["rep_epoch"]), "step": step}
            for src, dst in COLMAP.items():
                out[dst] = fnum(r.get(src))
            by_step[step] = out          # later row wins; see docstring
    if not by_step:
        die(f"{name}: no rows loaded")
    return [by_step[s] for s in sorted(by_step)]


def verify_provenance(name: str, seed: int, rows: Sequence[dict]) -> dict:
    """Check the final checkpoint's recorded identity against this cell's spec."""
    import torch
    spec = CONDITIONS[name]
    ckpt = os.path.join(ROOT, spec["dirs"][-1], "checkpoints",
                        f"step_{FINAL_STEP:08d}.pt")
    if not os.path.exists(ckpt):
        die(f"{name}: final checkpoint missing: {ckpt}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck["regime"] != spec["regime"]:
        die(f"{name}: checkpoint regime {ck['regime']!r} != {spec['regime']!r}")
    if int(ck["seed"]) != seed:
        die(f"{name}: checkpoint seed {ck['seed']} != {seed}")
    if int(ck["global_step"]) != FINAL_STEP or int(ck["rep_epoch"]) != FINAL_EPOCH:
        die(f"{name}: checkpoint at step {ck['global_step']} / epoch "
            f"{ck['rep_epoch']}, expected {FINAL_STEP} / {FINAL_EPOCH}")
    if ck.get("glove_fallback", 0):
        die(f"{name}: checkpoint reports {ck['glove_fallback']} GloVe fallbacks")
    # Presence flags exist only in checkpoints written at/after the factorial
    # commit; older H0/J0 checkpoints carry the regime, which implies them.
    s = ck.get("resolved_settings", {})
    ret = s.get("retrieval_enabled", spec["retrieval"])
    nam = s.get("naming_enabled", spec["naming"])
    if (bool(ret), bool(nam)) != (spec["retrieval"], spec["naming"]):
        die(f"{name}: presence flags {(ret, nam)} contradict the factorial spec")
    if rows[-1]["rep_epoch"] != FINAL_EPOCH:
        die(f"{name}: trajectory ends at epoch {rows[-1]['rep_epoch']}")
    return {
        "condition": name, "regime": ck["regime"], "seed": int(ck["seed"]),
        "final_epoch": int(ck["rep_epoch"]), "final_step": int(ck["global_step"]),
        "retrieval_enabled": bool(ret), "naming_enabled": bool(nam),
        "lambda_C": s.get("lambda_C"), "lambda_N": s.get("lambda_N"),
        "tau": s.get("tau"),
        "presence_source": ("explicit" if "retrieval_enabled" in s
                            else "derived from regime (pre-factorial checkpoint)"),
        "subset_definition_sha256": ck.get("subset_definition_sha256"),
        "glove_found": ck.get("glove_found"),
        "glove_fallback": ck.get("glove_fallback"),
        "git_commit": ck.get("git", {}).get("commit"),
        "git_tracked_dirty": ck.get("git", {}).get("tracked_dirty"),
        "source_dirs": list(spec["dirs"]),
        "n_snapshots": len(rows),
    }


def check_alignment(data: Dict[str, List[dict]]) -> dict:
    """All four cells must share the same developmental evaluation schedule."""
    sched = {c: [r["step"] for r in rs] for c, rs in data.items()}
    ref = sched["H0"]
    bad = {c: (len(s), len(ref)) for c, s in sched.items() if s != ref}
    if bad:
        detail = ", ".join(f"{c}: {a} steps vs H0 {b}" for c, (a, b) in bad.items())
        die(f"evaluation schedules differ across conditions ({detail})")
    dup = len(ref) != len(set(ref))
    if dup:
        die("duplicate steps within a stitched trajectory")
    return {"n_snapshots": len(ref), "first_step": ref[0], "last_step": ref[-1],
            "identical_across_conditions": True, "duplicate_steps": False}


# ==================================================  contrasts  ===========

def endpoint(data: Dict[str, List[dict]]) -> Dict[str, dict]:
    return {c: rows[-1] for c, rows in data.items()}


def contrasts(end: Dict[str, dict], metric: str) -> Optional[dict]:
    H, C, N, J = (end[c].get(metric) for c in ORDER)
    if None in (H, C, N, J):
        return None
    return {"metric": metric, "H0": H, "C_only": C, "N_only": N, "J0": J,
            "retrieval_effect_naming_off": C - H,
            "retrieval_effect_naming_on": J - N,
            "naming_effect_retrieval_off": N - H,
            "naming_effect_retrieval_on": J - C,
            "interaction": J - C - N + H}


def ltm_error_decomposition(end: Dict[str, dict]) -> dict:
    E = {c: errors(end[c]["full_rep_ltm"], LEXICON_N) for c in ORDER}
    if any(v is None for v in E.values()):
        die("full-lexicon LTM missing from an endpoint row")
    EH, EC, EN, EJ = (E[c] for c in ORDER)
    ret, nam = EC - EH, EN - EH
    return {"E_H0": EH, "E_C_only": EC, "E_N_only": EN, "E_J0": EJ,
            "retrieval_cost_naming_off": ret,
            "retrieval_cost_naming_on": EJ - EN,
            "naming_cost_retrieval_off": nam,
            "naming_cost_retrieval_on": EJ - EC,
            "additive_prediction": ret + nam,
            "observed_J0_cost": EJ - EH,
            "interaction": EJ - EC - EN + EH,
            "lexicon_size": LEXICON_N}


def crossings(rows: Sequence[dict], metric: str,
              thresholds: Sequence[float]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for t in thresholds:
        hit = [r for r in rows if r.get(metric) is not None
               and r[metric] >= t - 1e-12]
        sus = None
        for a, b in zip(rows, rows[1:]):
            if (a.get(metric) is not None and b.get(metric) is not None
                    and a[metric] >= t - 1e-12 and b[metric] >= t - 1e-12):
                sus = a["rep_epoch"]
                break
        out[f"{t:.2f}"] = {"first_epoch": hit[0]["rep_epoch"] if hit else None,
                           "sustained_epoch": sus}
    return out


# ====================================================  writing  ===========

def write_tsv(path: str, cols: Sequence[str], rows: Sequence[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), delimiter="\t",
                           extrasaction="ignore", restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return path


# =====================================================  figures  ==========

def style(ax, xlabel: str, ylabel: str, title: str, title_size: float = 12.0):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=title_size, pad=9)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, fig_dir: str, name: str) -> List[str]:
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _plot_metric(ax, data, metric, lo=None, hi=None, label=True, lw_scale=1.0):
    for c in ORDER:
        pts = [(r["rep_epoch"], r[metric] * 100) for r in data[c]
               if r.get(metric) is not None
               and (lo is None or r["rep_epoch"] >= lo)
               and (hi is None or r["rep_epoch"] <= hi)]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLOR[c], ls=LINESTYLE[c],
                lw=LINEWIDTH[c] * lw_scale, label=c if label else None, zorder=3)


def figure1(data, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))

    a = axes[0]
    _plot_metric(a, data, "comp_top1")
    a.axvline(100, color="0.55", lw=1.0, ls="--", zorder=1)
    a.text(108, 92, "LR 1e-3 → 1e-4", fontsize=8.5, color="0.35")
    style(a, "repetition epoch", "top-1 retrieval (%)",
          "A · Explicit comprehension\n(subset3288 probed against all 29,571 GloVe vectors)")
    a.set_ylim(-3, 100)
    a.legend(fontsize=10, frameon=False, loc=(0.60, 0.30))

    b = axes[1]
    _plot_metric(b, data, "naming_exact")
    b.axvline(100, color="0.55", lw=1.0, ls="--", zorder=1)
    style(b, "repetition epoch", "free-AR exact match (%)",
          "B · Naming from true GloVe\n(subset3288, greedy AR, no length leakage)")
    b.set_ylim(-4, 104)
    b.text(150, 92, "N-only and J0 coincide", fontsize=8.5, color="0.35")
    b.text(150, 6, "H0 and C-only coincide", fontsize=8.5, color="0.35")
    ins = b.inset_axes([0.44, 0.34, 0.52, 0.40])
    _plot_metric(ins, data, "naming_exact", hi=30, label=False, lw_scale=0.62)
    ins.set_title("e0–e30", fontsize=8.5, pad=3)
    ins.tick_params(labelsize=7.5)
    ins.grid(alpha=0.2, linewidth=0.5)

    c_ax = axes[2]
    _plot_metric(c_ax, data, "rep_ltm")
    c_ax.axvline(100, color="0.55", lw=1.0, ls="--", zorder=1)
    style(c_ax, "repetition epoch", "LTM exact match (%)",
          "C · Isolated ventral repetition\n(subset3288; FULL stays ≥99.9% in all cells)")
    c_ax.set_ylim(60, 101)
    # The late-training ordering H0 > N-only > C-only > J0 spans ~2 points and
    # is invisible on the full axis, so it gets its own zoom rather than a
    # truncated main axis that would hide the early development.
    ins2 = c_ax.inset_axes([0.40, 0.18, 0.56, 0.42])
    _plot_metric(ins2, data, "rep_ltm", lo=150, label=False, lw_scale=0.62)
    ins2.set_ylim(96, 100)
    ins2.set_title("e150–e440", fontsize=8.5, pad=3)
    ins2.tick_params(labelsize=7.5)
    ins2.grid(alpha=0.2, linewidth=0.5)

    fig.suptitle("Seed22 objective factorial, trained from random initialization to e440",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return save(fig, fig_dir, "fig1_developmental_trajectories")


def figure2(end, decomp, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    x = range(len(ORDER))

    a = axes[0]
    a.bar(x, [100 * end[c]["comp_top1"] for c in ORDER],
          color=[COLOR[c] for c in ORDER], width=0.62, zorder=3)
    for i, c in enumerate(ORDER):
        a.text(i, 100 * end[c]["comp_top1"] + 2.2,
               f"{100 * end[c]['comp_top1']:.1f}", ha="center", fontsize=10)
    a.set_xticks(list(x))
    a.set_xticklabels(ORDER, fontsize=10)
    a.set_ylim(0, 100)
    style(a, "", "top-1 retrieval (%)",
          "A · Explicit comprehension at e440")

    b = axes[1]
    E = [decomp[f"E_{c.replace('-', '_')}"] for c in ORDER]
    b.bar(x, E, color=[COLOR[c] for c in ORDER], width=0.62, zorder=3)
    for i, v in enumerate(E):
        b.text(i, v + 35, f"{v}", ha="center", fontsize=10)
    b.set_xticks(list(x))
    b.set_xticklabels(ORDER, fontsize=10)
    b.set_ylim(0, max(E) * 1.18)
    style(b, "", f"LTM errors / {LEXICON_N:,} words",
          "B · Isolated ventral repetition cost at e440")

    fig.suptitle("Seed22 factorial endpoint · single run per cell, no variance bars",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    return save(fig, fig_dir, "fig2_factorial_endpoint")


def figure3(decomp, fig_dir):
    ret = decomp["retrieval_cost_naming_off"]
    nam = decomp["naming_cost_retrieval_off"]
    pred, obs = decomp["additive_prediction"], decomp["observed_J0_cost"]
    labels = ["Retrieval only\n(C-only − H0)", "Naming only\n(N-only − H0)",
              "Additive\nprediction", "Observed\n(J0 − H0)"]
    vals = [ret, nam, pred, obs]
    cols = [COLOR["C-only"], COLOR["N-only"], "#b7b7b7", COLOR["J0"]]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(range(4), vals, color=cols, width=0.6, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 22, f"+{v}", ha="center", fontsize=10.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.2)
    style(ax, "", f"excess LTM errors vs H0 / {LEXICON_N:,} words",
          "Ventral repetition cost decomposes approximately additively\n"
          f"(seed22 descriptive decomposition; interaction = "
          f"{decomp['interaction']:+d} errors)")
    ax.annotate("", xy=(3, obs), xytext=(2, pred),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.2))
    fig.tight_layout()
    return save(fig, fig_dir, "fig3_ltm_cost_decomposition")


# ===================================================  WM audit  ===========

def wm_audit(out_dir: str, seed: int) -> dict:
    """Re-run the canonical full-lexicon AR evaluator on the final checkpoints."""
    import torch
    from scripts.naming_comprehension.train_joint_scratch import JointScratchTrainer
    from scripts.evaluate_train_lexicon_ceiling import evaluate_forms_ar

    kw = dict(device="cpu", max_words=30000,
              lexicon_path="data/lexicon_en_glove_covered.tsv",
              dorsal_pool_size=4000, batch_size=64, subset_mode="nested",
              subset_per_band=822, subset_size=3288, lr_boundary_steps=46300,
              allow_glove_fallback=False, require_subset_hash=True)
    summary: Dict[str, dict] = {}
    per_item_rows: List[dict] = []
    subset_words = probe_words = None

    for cond in ORDER:
        spec = CONDITIONS[cond]
        tr = JointScratchTrainer(regime=spec["regime"], seed=seed, **kw)
        if subset_words is None:
            subset_words = {tr.entries[i].word for i in tr.subset_idx}
            probe_words = {tr.entries[i].word for i in tr.probe_idx}
        ck = torch.load(os.path.join(ROOT, spec["dirs"][-1], "checkpoints",
                                     f"step_{FINAL_STEP:08d}.pt"),
                        map_location="cpu", weights_only=False)
        tr.model.load_state_dict(ck["model_state_dict"])
        tr.model.eval()
        with torch.no_grad():
            rows = evaluate_forms_ar(tr.model, tr.vocab, tr.entries, "cpu",
                                     routes=("full", "wm", "ltm"), wm_noise=False)
        counts = {r: sum(1 for x in rows if not x[f"{r}_exact_match"])
                  for r in ("full", "wm", "ltm")}
        summary[cond] = {"errors": counts}
        for x in rows:
            if x["wm_exact_match"] and x["full_exact_match"]:
                continue          # keep only rows that carry an error somewhere
            per_item_rows.append({
                "condition": cond, "word": x["word"], "rank": x["rank"],
                "length": x["length"], "target_phonemes": x["target_phonemes"],
                "wm_exact": x["wm_exact_match"], "wm_predicted": x["wm_predicted"],
                "wm_edit": x["wm_edit_dist"],
                "wm_pred_len": len(x["wm_predicted"].split()),
                "ltm_exact": x["ltm_exact_match"], "ltm_predicted": x["ltm_predicted"],
                "full_exact": x["full_exact_match"], "full_predicted": x["full_predicted"],
                "in_subset3288": int(x["word"] in subset_words),
                "in_probe": int(x["word"] in probe_words),
            })
        print(f"  [wm-audit] {cond:7s} FULL {counts['full']:5d}  "
              f"WM {counts['wm']:5d}  LTM {counts['ltm']:5d}", flush=True)

    n_rows = [r for r in per_item_rows if r["condition"] == "N-only" and not r["wm_exact"]]
    summary["N-only"]["wm_error_detail"] = {
        "n_wm_errors": len(n_rows),
        "rescued_by_full": sum(1 for r in n_rows if r["full_exact"]),
        "also_wrong_in_ltm": sum(1 for r in n_rows if not r["ltm_exact"]),
        "in_subset3288": sum(r["in_subset3288"] for r in n_rows),
        "in_probe": sum(r["in_probe"] for r in n_rows),
        "mean_target_length": (round(sum(r["length"] for r in n_rows) / max(len(n_rows), 1), 2)),
        "edit_distances": sorted(r["wm_edit"] for r in n_rows),
        "words": sorted(r["word"] for r in n_rows),
    }
    cols = ["condition", "word", "rank", "length", "target_phonemes",
            "wm_exact", "wm_predicted", "wm_edit", "wm_pred_len",
            "ltm_exact", "ltm_predicted", "full_exact", "full_predicted",
            "in_subset3288", "in_probe"]
    write_tsv(os.path.join(out_dir, "data", "wm_audit_per_item.tsv"),
              cols, per_item_rows)
    with open(os.path.join(out_dir, "data", "wm_audit_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ========================================================  main  ==========

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--output", default="reports/joint_scratch_factorial_seed22",
                   help="repo-relative output directory for tables and figures")
    p.add_argument("--wm-audit", action="store_true",
                   help="also re-run the full-lexicon AR evaluator (a few minutes)")
    args = p.parse_args(argv)

    out_dir = os.path.join(ROOT, args.output)
    data_dir, fig_dir = os.path.join(out_dir, "data"), os.path.join(out_dir, "figures")

    data = {c: load_condition(c, args.seed) for c in ORDER}
    prov = {c: verify_provenance(c, args.seed, data[c]) for c in ORDER}
    align = check_alignment(data)
    print(f"[analyze_joint_factorial] four cells verified, "
          f"{align['n_snapshots']} aligned snapshots each, seed {args.seed}")

    flat = [r for c in ORDER for r in data[c]]
    traj = write_tsv(os.path.join(data_dir, "factorial_trajectories_seed22.tsv"),
                     TABLE_COLS, flat)

    end = endpoint(data)
    decomp = ltm_error_decomposition(end)

    # endpoint table, accuracy + error counts
    end_rows = []
    for c in ORDER:
        e = end[c]
        end_rows.append({
            "condition": c, "retrieval_enabled": e["retrieval_enabled"],
            "naming_enabled": e["naming_enabled"], "rep_epoch": e["rep_epoch"],
            "step": e["step"],
            "comp_top1": e["comp_top1"], "comp_top5": e["comp_top5"],
            "comp_rank_median": e["comp_rank_median"],
            "comp_cosine": e["comp_cosine"], "comp_margin": e["comp_margin"],
            "naming_exact": e["naming_exact"], "naming_wer": e["naming_wer"],
            "rep_ltm_subset": e["rep_ltm"], "probe_rep_ltm": e["probe_rep_ltm"],
            "full_rep_full": e["full_rep_full"], "full_rep_wm": e["full_rep_wm"],
            "full_rep_ltm": e["full_rep_ltm"],
            "full_errors_full": errors(e["full_rep_full"], LEXICON_N),
            "full_errors_wm": errors(e["full_rep_wm"], LEXICON_N),
            "full_errors_ltm": errors(e["full_rep_ltm"], LEXICON_N),
        })
    endpoint_tsv = write_tsv(os.path.join(data_dir, "fig2_endpoint.tsv"),
                             list(end_rows[0].keys()), end_rows)

    # descriptive factorial contrasts
    metrics = ["comp_top1", "comp_top5", "naming_exact", "naming_wer",
               "rep_ltm", "probe_rep_ltm", "full_rep_ltm", "full_rep_full",
               "full_rep_wm"]
    con_rows = [c for c in (contrasts(end, m) for m in metrics) if c is not None]
    contrasts_tsv = write_tsv(
        os.path.join(data_dir, "factorial_contrasts_e440.tsv"),
        list(con_rows[0].keys()), con_rows)

    decomp_tsv = write_tsv(
        os.path.join(data_dir, "fig3_decomposition.tsv"),
        ["quantity", "errors"],
        [{"quantity": k, "errors": v} for k, v in decomp.items()])

    # acquisition crossings
    cross = {
        "comprehension_top1": {c: crossings(data[c], "comp_top1",
                                            [0.20, 0.50, 0.80, 0.90, 0.95])
                              for c in ORDER},
        "naming_exact": {c: crossings(data[c], "naming_exact",
                                      [0.20, 0.50, 0.80, 0.95, 0.99, 1.00])
                        for c in ORDER},
    }
    with open(os.path.join(data_dir, "acquisition_crossings.json"),
              "w", encoding="utf-8") as fh:
        json.dump(cross, fh, indent=2)

    fig1_src = write_tsv(os.path.join(data_dir, "fig1_developmental.tsv"),
                         ["condition", "rep_epoch", "step", "comp_top1",
                          "comp_top5", "naming_exact", "rep_ltm", "rep_full",
                          "rep_wm"], flat)

    f1 = figure1(data, fig_dir)
    f2 = figure2(end, decomp, fig_dir)
    f3 = figure3(decomp, fig_dir)

    audit = wm_audit(out_dir, args.seed) if args.wm_audit else None

    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": args.seed, "conditions": prov, "alignment": align,
                   "lexicon_size": LEXICON_N, "subset_size": SUBSET_N,
                   "final_epoch": FINAL_EPOCH, "final_step": FINAL_STEP,
                   "note": "analysis-only; no training, no scientific artifact "
                           "was written or modified"}, fh, indent=2, default=str)

    print(f"  trajectories : {os.path.relpath(traj, ROOT)} ({len(flat)} rows)")
    print(f"  endpoint     : {os.path.relpath(endpoint_tsv, ROOT)}")
    print(f"  contrasts    : {os.path.relpath(contrasts_tsv, ROOT)}")
    print(f"  decomposition: {os.path.relpath(decomp_tsv, ROOT)}")
    for group in (f1, f2, f3):
        for pth in group:
            print(f"  figure       : {os.path.relpath(pth, ROOT)}")
    if audit:
        print("  wm audit     : "
              f"{os.path.relpath(os.path.join(out_dir, 'data'), ROOT)}/wm_audit_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
