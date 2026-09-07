"""Read-only characterisation of residual comprehension errors.

Compares residual C error items against the FULL canonical C population
(27,981) on lexical and phonological features, so the question "are the
survivors systematically longer / rarer / morphologically confusable?" is
answered by distributions and effect sizes rather than anecdotes.

METADATA IS ONLY WHAT THE REPO ACTUALLY HAS.  `LexEntry` carries word,
phonemes, freq and rank -- there is no syllable or morphology field, so:
  * `syllable_proxy` is the ARPABET VOWEL COUNT, a derived proxy, labelled as
    such and never called a syllable count;
  * there is no target-morphology field.  The `relation` label in the error
    TSVs describes the TARGET-vs-COMPETITOR pair (homophone / calendar /
    numeric / morphological / semantic_neighbour / other) and is used for the
    "is morphological confusion enriched?" question, which is what it can
    actually answer.

STATISTICS.  Pooled residuals across seeds are NOT iid -- the same item can
appear in several seeds -- so no significance test is computed on them.  What
is reported instead: distributions, standardized mean differences against the
population, categorical enrichment ratios, and the SAME effect size computed
independently per seed so sign-consistency across the four seeds can be read
directly.

Sets analysed: pooled residuals, unique residual items, the four-way
persistent core, and the churn-only items (wrong in exactly one seed).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ARPABET_VOWELS = {"IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW",
                  "AH", "ER", "AY", "AW", "OY"}
CONTINUOUS = ["orth_length", "phon_length", "syllable_proxy", "log10_rank",
              "freq"]
FROM_TSV = ["target_rank", "target_cos", "margin_target_minus_top1", "in_top5"]


def num(v):
    if v in (None, "", "nan", "NaN"):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def read_tsv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: str, rows: List[dict]) -> None:
    if not rows:
        open(path, "w").write("")
        print(f"[resid] (0 rows) {path}")
        return
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[resid] wrote {path}  ({len(rows)} rows)")


def population_features(glove_path: str, lexicon_path: str) -> Dict[int, dict]:
    """Features of every canonical C target, from the real lexicon."""
    from config import default_config
    from data.lexicon import build_lexicon
    from data.phonemes import build_vocab
    from scripts.naming_comprehension.train_tasks import (
        canonical_phonology_indices)

    cfg = default_config()
    cfg.data.use_real = True
    cfg.data.lexicon_path = lexicon_path
    cfg.data.glove_path = glove_path
    cfg.data.max_words = 30000
    cfg.data.split_mode = "full_lexicon"
    cfg.data.val_fraction = 0.0
    vocab = build_vocab()
    lex = build_lexicon(cfg.data, vocab)
    entries = list(lex.entries)
    fb = int(getattr(lex.load_stats, "n_glove_fallback", 0))
    if fb:
        raise SystemExit(f"FATAL: {fb} GloVe fallbacks -- refusing to analyse")
    c_idx = canonical_phonology_indices(entries)
    itos = vocab.itos
    phon_groups: Dict[tuple, int] = Counter(
        tuple(e.phonemes) for e in entries)
    out = {}
    for i in c_idx:
        e = entries[i]
        syms = [itos[p] for p in e.phonemes]
        out[i] = {
            "bank_index": i, "word": e.word,
            "orth_length": len(e.word),
            "phon_length": len(e.phonemes),
            "phonemes": " ".join(syms),
            "syllable_proxy": sum(1 for s in syms if s in ARPABET_VOWELS),
            "freq_rank": e.rank,
            "log10_rank": math.log10(max(int(e.rank), 1)),
            "freq": float(e.freq),
            "homophone_group_size": phon_groups[tuple(e.phonemes)],
        }
    print(f"[resid] population: {len(out)} canonical C targets "
          f"(bank {len(entries)}, 0 fallback)")
    return out


def smd(sample: Sequence[float], pop: Sequence[float]) -> Optional[float]:
    """Standardized mean difference against the population SD."""
    s = [x for x in sample if x is not None]
    p = [x for x in pop if x is not None]
    if len(s) < 2 or len(p) < 2:
        return None
    sd = st.pstdev(p)
    return None if sd == 0 else round((st.mean(s) - st.mean(p)) / sd, 4)


def dist(vals: Sequence[float]) -> dict:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {"n": 0}
    q = st.quantiles(v, n=4) if len(v) > 3 else [v[0], st.median(v), v[-1]]
    return {"n": len(v), "mean": round(st.mean(v), 4),
            "median": round(st.median(v), 4),
            "q1": round(q[0], 4), "q3": round(q[2], 4),
            "min": round(v[0], 4), "max": round(v[-1], 4),
            "sd": round(st.pstdev(v), 4) if len(v) > 1 else 0.0}


def compare(label: str, items: Sequence[dict], pop: Dict[int, dict],
            n_seeds: Optional[int] = None) -> List[dict]:
    rows = []
    for feat in CONTINUOUS:
        pv = [p[feat] for p in pop.values()]
        sv = [it[feat] for it in items if it.get(feat) is not None]
        d_s, d_p = dist(sv), dist(pv)
        rows.append({"set": label, "feature": feat, "n_items": len(sv),
                     "n_population": len(pv),
                     **{f"resid_{k}": v for k, v in d_s.items() if k != "n"},
                     **{f"pop_{k}": v for k, v in d_p.items() if k != "n"},
                     "standardized_mean_diff": smd(sv, pv),
                     "n_seeds_pooled": n_seeds})
    return rows


def categorical(label: str, items: Sequence[dict], pop: Dict[int, dict],
                key: str, pop_key: Optional[str] = None) -> List[dict]:
    """Counts, proportions and enrichment vs the population base rate."""
    rows = []
    c = Counter(str(it.get(key, "")) for it in items if it.get(key) not in
                (None, ""))
    total = sum(c.values())
    if pop_key:
        pc = Counter(str(p.get(pop_key, "")) for p in pop.values())
        ptot = sum(pc.values())
    else:
        pc, ptot = None, 0
    for k, v in c.most_common():
        base = (pc[k] / ptot) if pc and ptot else None
        prop = v / total if total else None
        rows.append({"set": label, "variable": key, "value": k, "count": v,
                     "proportion": round(prop, 6) if prop else None,
                     "population_proportion": (round(base, 6) if base
                                               else None),
                     "enrichment_vs_population": (round(prop / base, 3)
                                                  if base else None)})
    return rows


def load_residuals(runs_root: str, run_template: str, u: int,
                   seeds: Sequence[int], pop: Dict[int, dict]):
    """Per-seed residual items, joined to population features."""
    per = {}
    for s in seeds:
        d = os.path.join(runs_root, run_template.format(seed=s),
                         f"error_audit_u{u}", "comp_errors.tsv")
        rows = read_tsv(d)
        if not rows:
            print(f"[resid] MISSING {d}")
            continue
        items = []
        for r in rows:
            idx = int(r["target_bank_index"])
            base = pop.get(idx)
            if base is None:                 # not a canonical C target
                continue
            it = dict(base)
            it["seed"] = s
            for k in FROM_TSV:
                it[k] = num(r.get(k))
            it["relation"] = r.get("relation")
            it["pred_word"] = r.get("pred_word")
            it["is_homophone_of_target"] = num(r.get("is_homophone_of_target"))
            items.append(it)
        per[s] = items
        print(f"[resid] seed {s}: {len(items)} residual C items")
    return per


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-template", required=True,
                    help="e.g. 'final_rep_rescue123_h512_s{seed}'")
    ap.add_argument("--u", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", default="19,20,21,22")
    ap.add_argument("--glove-path", required=True)
    ap.add_argument("--lexicon-path",
                    default="data/lexicon_en_glove_covered.tsv")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(a.out_dir, exist_ok=True)

    pop = population_features(a.glove_path, a.lexicon_path)
    per = load_residuals(a.runs_root, a.run_template, a.u, seeds, pop)
    if not per:
        raise SystemExit("FATAL: no residual audits found")

    pooled = [it for items in per.values() for it in items]
    counts = Counter(it["bank_index"] for it in pooled)
    unique_idx = sorted(counts)
    core_idx = sorted(i for i, n in counts.items() if n == len(per))
    churn_idx = sorted(i for i, n in counts.items() if n == 1)
    first = {}
    for it in pooled:
        first.setdefault(it["bank_index"], it)
    unique = [first[i] for i in unique_idx]
    core = [first[i] for i in core_idx]
    churn = [first[i] for i in churn_idx]
    print(f"[resid] pooled {len(pooled)} | unique {len(unique)} | "
          f"four-way core {len(core)} | churn-only {len(churn)}")

    write_tsv(os.path.join(a.out_dir, "residual_item_features.tsv"),
              sorted(pooled, key=lambda r: (r["seed"], r["bank_index"])))
    write_tsv(os.path.join(a.out_dir, "persistent_core.tsv"),
              [dict(r, n_seeds_wrong=counts[r["bank_index"]]) for r in core])

    summary: List[dict] = []
    summary += compare(f"{a.label}:pooled", pooled, pop, len(per))
    summary += compare(f"{a.label}:unique", unique, pop)
    summary += compare(f"{a.label}:four_way_core", core, pop)
    summary += compare(f"{a.label}:churn_only", churn, pop)
    for s, items in per.items():
        summary += compare(f"{a.label}:seed{s}", items, pop)
    write_tsv(os.path.join(a.out_dir, "residual_vs_population_summary.tsv"),
              summary)

    cat: List[dict] = []
    for lab, items in ((f"{a.label}:pooled", pooled),
                       (f"{a.label}:unique", unique),
                       (f"{a.label}:four_way_core", core),
                       (f"{a.label}:churn_only", churn)):
        cat += categorical(lab, items, pop, "relation")
        cat += categorical(lab, items, pop, "homophone_group_size",
                           "homophone_group_size")
    write_tsv(os.path.join(a.out_dir, "residual_categorical.tsv"), cat)

    per_seed = []
    for s, items in per.items():
        rec = {"label": a.label, "seed": s, "n_errors": len(items)}
        for feat in CONTINUOUS:
            rec[f"{feat}_smd"] = smd([it[feat] for it in items],
                                     [p[feat] for p in pop.values()])
            rec[f"{feat}_median"] = dist([it[feat] for it in items]).get("median")
        rel = Counter(it.get("relation") for it in items)
        for k in ("semantic_neighbour", "morphological", "homophone",
                  "numeric", "calendar", "other"):
            rec[f"rel_{k}"] = rel.get(k, 0)
        m = [it.get("margin_target_minus_top1") for it in items]
        rec["frac_margin_within_0.01"] = round(
            sum(1 for x in m if x is not None and x > -0.01)
            / max(len(items), 1), 6)
        rec["frac_in_top5"] = round(
            sum(1 for it in items if it.get("in_top5")) / max(len(items), 1), 6)
        per_seed.append(rec)
    write_tsv(os.path.join(a.out_dir, "per_seed_summary.tsv"), per_seed)

    # sign-consistency across seeds: the headline robustness check
    consistency = []
    for feat in CONTINUOUS:
        vals = [r[f"{feat}_smd"] for r in per_seed
                if r.get(f"{feat}_smd") is not None]
        if not vals:
            continue
        consistency.append({
            "feature": feat, "n_seeds": len(vals),
            "smd_mean": round(st.mean(vals), 4),
            "smd_min": round(min(vals), 4), "smd_max": round(max(vals), 4),
            "same_sign_all_seeds": int(all(v > 0 for v in vals)
                                       or all(v < 0 for v in vals))})
    write_tsv(os.path.join(a.out_dir, "seed_consistency.tsv"), consistency)

    json.dump({"label": a.label, "u": a.u, "seeds": seeds,
               "run_template": a.run_template,
               "n_pooled": len(pooled), "n_unique": len(unique),
               "n_four_way_core": len(core), "n_churn_only": len(churn),
               "population": len(pop),
               "note": "pooled residuals are NOT iid across seeds; effect "
                       "sizes are descriptive and seed-level consistency is "
                       "the robustness check",
               "syllable_proxy": "ARPABET vowel count (derived, not a "
                                 "lexicon field)"},
              open(os.path.join(a.out_dir, "residual_meta.json"), "w"),
              indent=1)

    if not a.no_plots:
        try:
            _plots(a.out_dir, a.label, pop, pooled, core)
        except Exception as exc:                       # pragma: no cover
            print(f"[resid] plots skipped: {exc}")
    print("[resid] done")
    return 0


def _plots(out_dir, label, pop, pooled, core):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specs = [("phon_length", "phoneme length"),
             ("orth_length", "orthographic length"),
             ("log10_rank", "log10 frequency rank"),
             ("syllable_proxy", "ARPABET vowel count (syllable proxy)")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for ax, (feat, name) in zip(axes.ravel(), specs):
        pv = [p[feat] for p in pop.values()]
        sv = [it[feat] for it in pooled]
        bins = 20
        ax.hist(pv, bins=bins, density=True, alpha=0.45, label="C population")
        ax.hist(sv, bins=bins, density=True, alpha=0.55, label="residuals")
        if core:
            ax.hist([it[feat] for it in core], bins=bins, density=True,
                    histtype="step", lw=1.6, label="4-seed core")
        ax.set_xlabel(name); ax.set_ylabel("density")
        ax.legend(fontsize=7, frameon=False)
    for ax, (key, name) in zip(axes.ravel()[4:],
                               [("margin_target_minus_top1", "target-winner margin"),
                                ("target_rank", "target rank")]):
        vals = [it.get(key) for it in pooled if it.get(key) is not None]
        if key == "target_rank":
            vals = [min(v, 50) for v in vals]
            ax.set_xlabel("target rank (clipped at 50)")
        else:
            ax.set_xlabel(name)
        ax.hist(vals, bins=30, alpha=0.7, color="tab:purple")
        ax.set_ylabel("residual items")
    fig.suptitle(f"Residual C errors vs the full canonical C population "
                 f"({label})", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"residual_features.{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[resid] wrote {p}")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
