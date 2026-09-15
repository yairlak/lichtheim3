"""Statistics for the gate / route-competence audit.

Every choice here is fixed by `EXPERIMENT_CONTRACT.md` §6.3, §6.4 and §7.1 and
was written before any result existed.  Nothing in this module selects a test
after seeing data, and nothing optimises a gate parameter.

Two deliberate design points:

* `AUROC` and Cliff's `delta` are reported together but are NOT independent
  evidence: `delta == 2*AUROC - 1` identically.  `auroc_delta_identity_ok`
  asserts it numerically so the recap cannot present them as corroborating.
* The power rule (§6.4) is applied here, not at reporting time: an
  underpowered cell yields `None` for AUROC/delta/CI rather than a number that
  would later have to be retracted.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .gate_probe import CompetenceCategory

MIN_N_PER_CELL = 30            # contract §6.4
N_BOOTSTRAP = 10_000           # contract §6.3
GATE_RANGE = (0.0323, 0.6457)  # attainable at alpha=2.0, tau=0.7 (audit CLAIM 6)


# ------------------------------------------------------------------ primitives

def describe(x: Sequence[float]) -> Dict[str, Optional[float]]:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None,
                "q25": None, "q75": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max())}


def auroc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """Rank-based AUROC with exact tie handling (ties contribute 0.5)."""
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    if p.size == 0 or n.size == 0:
        return None
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1, dtype=float)
    # average ranks within ties
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[:p.size].sum() - p.size * (p.size + 1) / 2) / (p.size * n.size))


def cliffs_delta(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    a = auroc(pos, neg)
    return None if a is None else 2.0 * a - 1.0


def auroc_delta_identity_ok(a: Optional[float], d: Optional[float]) -> bool:
    return a is None or d is None or abs(d - (2 * a - 1)) < 1e-12


def bootstrap_ci(pos: Sequence[float], neg: Sequence[float], seed: int,
                 n_boot: int = N_BOOTSTRAP) -> Optional[Dict[str, float]]:
    """Stratified bootstrap 95% CI for AUROC (and hence for delta)."""
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    if p.size < MIN_N_PER_CELL or n.size < MIN_N_PER_CELL:
        return None
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = auroc(rng.choice(p, p.size, replace=True),
                        rng.choice(n, n.size, replace=True))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc_lo": float(lo), "auroc_hi": float(hi),
            "delta_lo": float(2 * lo - 1), "delta_hi": float(2 * hi - 1),
            "n_boot": n_boot, "seed": seed}


def spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, Optional[float]]:
    from scipy import stats
    a, b = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return {"rho": None, "p": None, "n": int(ok.sum())}
    r = stats.spearmanr(a[ok], b[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


def mcnemar(a_correct: Sequence[int], b_correct: Sequence[int]) -> Dict[str, object]:
    """Exact paired McNemar.  `a` = native gate, `b` = fixed05 (contract §7.1)."""
    from scipy import stats
    a, b = np.asarray(a_correct, int), np.asarray(b_correct, int)
    unchanged_correct = int(((a == 1) & (b == 1)).sum())
    errors_gained = int(((a == 1) & (b == 0)).sum())     # native right, fixed wrong
    errors_recovered = int(((a == 0) & (b == 1)).sum())  # native wrong, fixed right
    unchanged_wrong = int(((a == 0) & (b == 0)).sum())
    disc = errors_gained + errors_recovered
    p = (float(stats.binomtest(errors_recovered, disc, 0.5).pvalue)
         if disc > 0 else None)
    return {"unchanged_correct": unchanged_correct,
            "errors_gained": errors_gained,
            "errors_recovered": errors_recovered,
            "unchanged_wrong": unchanged_wrong,
            "n_discordant": disc,
            "net_change_in_correct": errors_recovered - errors_gained,
            "acc_native": float((a == 1).mean()) if a.size else None,
            "acc_fixed05": float((b == 1).mean()) if b.size else None,
            "p_exact_mcnemar": p,
            "test": "exact binomial on discordant pairs (two-sided)"}


# --------------------------------------------------------------- per-state roll-up

def _cells(rows: List[dict], conv: str) -> Dict[str, List[dict]]:
    key = "competence_category" if conv == "canonical" else "competence_category_freear"
    out: Dict[str, List[dict]] = {c: [] for c in CompetenceCategory.ALL}
    for r in rows:
        if key in r:
            out[r[key]].append(r)
    return out


def summarize_state(rows: List[dict], seed: int) -> dict:
    """The full preregistered read-out for ONE manifest state."""
    g = [r["gate"] for r in rows]
    conventions = ["canonical"] + (["freear"] if "freear_full_exact" in rows[0] else [])

    out: dict = {
        "n_items": len(rows),
        "gate": describe(g),
        "gate_attainable_range": {"lo": GATE_RANGE[0], "hi": GATE_RANGE[1]},
        "gate_frac_above_half": float(np.mean(np.asarray(g) > 0.5)),
        "lexical_confidence": describe([r["lexical_confidence"] for r in rows]),
        "covariates_spearman_vs_gate": {
            "zipf_approx": spearman(g, [r["zipf_approx"] for r in rows]),
            "length": spearman(g, [r["length"] for r in rows]),
            "lexical_confidence": spearman(g, [r["lexical_confidence"] for r in rows]),
            "lexical_margin": spearman(g, [r["lexical_margin"] for r in rows]),
            "lexical_density": spearman(g, [r["lexical_density"] for r in rows]),
        },
        "competence": {},
        "experiment1": {},
        "experiment2": {},
        "route_accuracy": {},
    }

    for conv in conventions:
        out["route_accuracy"][conv] = {
            r: float(np.mean([row[f"{conv}_{r}_exact"] for row in rows]))
            for r in ("full", "wm", "ltm", "fixed05")}

    # ---- competence cells (primary = canonical convention, contract §6.1) ----
    cells = _cells(rows, "canonical")
    for name, cr in cells.items():
        out["competence"][name] = {"n": len(cr), "gate": describe([r["gate"] for r in cr])}

    # ---- EXPERIMENT 1: does g separate LTM_ONLY from WM_ONLY? (§6.2) ----
    for conv in conventions:
        c = _cells(rows, conv)
        pos = [r["gate"] for r in c[CompetenceCategory.LTM_ONLY]]   # expect HIGHER g
        neg = [r["gate"] for r in c[CompetenceCategory.WM_ONLY]]
        powered = len(pos) >= MIN_N_PER_CELL and len(neg) >= MIN_N_PER_CELL
        a = auroc(pos, neg) if powered else None
        d = cliffs_delta(pos, neg) if powered else None
        assert auroc_delta_identity_ok(a, d)
        med_pos = float(np.median(pos)) if pos else None
        med_neg = float(np.median(neg)) if neg else None
        out["experiment1"][conv] = {
            "status": ("STRUCTURALLY_EMPTY — n(LTM_ONLY_CORRECT) <= 2 because the "
                       "dorsal route is at ceiling; see contract AMENDMENT 1"
                       if len(pos) < MIN_N_PER_CELL else "TESTED"),
            "hypothesis": "median g(LTM_ONLY_CORRECT) > median g(WM_ONLY_CORRECT)",
            "n_ltm_only": len(pos), "n_wm_only": len(neg),
            "powered": powered,
            "power_rule": f"both cells must have n >= {MIN_N_PER_CELL} (contract 6.4)",
            "median_gate_ltm_only": med_pos,
            "median_gate_wm_only": med_neg,
            "median_difference": (None if med_pos is None or med_neg is None
                                  else med_pos - med_neg),
            "auroc": a, "cliffs_delta": d,
            "ci95": bootstrap_ci(pos, neg, seed) if powered else None,
            "note": ("UNDERPOWERED: distributions and counts only, no effect size "
                     "(contract 6.4)") if not powered else
                    ("g is a strictly monotone function of lexical_confidence, so this "
                     "AUROC is identically AUROC(c_LTM): it measures whether VENTRAL "
                     "CONFIDENCE predicts which route is right, not whether the gate "
                     "arbitrates (contract 6.5)"),
        }

    # ---- EXPERIMENT 1', reformulated primary: WM_ONLY vs BOTH (AMENDMENT 1) ----
    # The dorsal route is at ceiling, so LTM_ONLY_CORRECT is empty by construction
    # and the only competence contrast the data support is "items the ventral route
    # fails" vs "items it does not".  Predicted direction: LOWER g for WM_ONLY,
    # hence AUROC < 0.5 and delta < 0.
    out["experiment1_prime"] = {}
    for conv in conventions:
        c = _cells(rows, conv)
        pos = [r["gate"] for r in c[CompetenceCategory.WM_ONLY]]     # ventral FAILS
        neg = [r["gate"] for r in c[CompetenceCategory.BOTH]]        # ventral succeeds
        powered = len(pos) >= MIN_N_PER_CELL and len(neg) >= MIN_N_PER_CELL
        a = auroc(pos, neg) if powered else None
        d = cliffs_delta(pos, neg) if powered else None
        assert auroc_delta_identity_ok(a, d)
        med_pos = float(np.median(pos)) if pos else None
        med_neg = float(np.median(neg)) if neg else None
        out["experiment1_prime"][conv] = {
            "hypothesis": "median g(WM_ONLY_CORRECT) < median g(BOTH_CORRECT)",
            "predicted_direction": "auroc < 0.5 (delta < 0)",
            "positive_class": "WM_ONLY_CORRECT (ventral route fails)",
            "negative_class": "BOTH_CORRECT (ventral route succeeds)",
            "n_wm_only": len(pos), "n_both": len(neg),
            "powered": powered,
            "median_gate_wm_only": med_pos,
            "median_gate_both": med_neg,
            "median_difference": (None if med_pos is None or med_neg is None
                                  else med_pos - med_neg),
            "auroc": a, "cliffs_delta": d,
            "direction_as_predicted": (None if a is None else bool(a < 0.5)),
            "ci95": bootstrap_ci(pos, neg, seed) if powered else None,
            "note": ("g is rank-identical to lexical_confidence, so a confirmed H1' "
                     "means VENTRAL CONFIDENCE IS INFORMATIVE ABOUT VENTRAL FAILURE, "
                     "not that the gate arbitrates between routes (contract 6.5). "
                     "The gate is bit-invariant to every dorsal parameter "
                     "(CODE_AUDIT_GATE.md CLAIM 5)."),
        }

    # ---- EXPERIMENT 2: forced 0.5/0.5, paired item by item (§7.1) ----
    for conv in conventions:
        native = [r[f"{conv}_full_exact"] for r in rows]
        fixed = [r[f"{conv}_fixed05_exact"] for r in rows]
        res = {"overall": mcnemar(native, fixed), "by_competence_category": {}}
        c = _cells(rows, conv)
        for name, cr in c.items():
            if cr:
                res["by_competence_category"][name] = mcnemar(
                    [r[f"{conv}_full_exact"] for r in cr],
                    [r[f"{conv}_fixed05_exact"] for r in cr])
        # descriptive breakdowns, no p-values (contract §7.1 multiplicity)
        z = np.asarray([r["zipf_approx"] for r in rows], float)
        if np.unique(z).size >= 5:
            edges = np.percentile(z, [20, 40, 60, 80])
            qi = np.digitize(z, edges)
            res["by_frequency_quintile_descriptive"] = {
                f"Q{q+1}": mcnemar([native[i] for i in range(len(rows)) if qi[i] == q],
                                   [fixed[i] for i in range(len(rows)) if qi[i] == q])
                for q in range(5) if (qi == q).any()}
        lengths = sorted({r["length"] for r in rows})
        res["by_length_descriptive"] = {
            str(L): mcnemar([native[i] for i, r in enumerate(rows) if r["length"] == L],
                            [fixed[i] for i, r in enumerate(rows) if r["length"] == L])
            for L in lengths}
        res["intervention_note"] = (
            "INFERENCE INTERVENTION ONLY. This does NOT establish how a fixed-fusion "
            "model would learn from scratch (contract 7). Training included "
            "lambda_gate*(mean(g)-0.5)^2 with usage_prior=0.5, so the model was "
            "optimised under mild pressure toward the symmetry this intervention "
            "imposes (contract 7.2).")
        out["experiment2"][conv] = res

    return out
