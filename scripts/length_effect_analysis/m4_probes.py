"""M4 Phase C — stagewise linear-readout localisation of the LTM length effect.

Analysis-only.  No model is constructed, no checkpoint is loaded, no decoder is
executed, no token is generated.  Every input is an already-validated array or
table produced by the frozen instrumented run and by the validated stage-1
encoder-only extraction.

The protocol was frozen in `m4_representation/m4_probe_protocol.md` before any
probe was fitted; this module implements it and nothing else.

Vocabulary discipline: probe performance is **linearly accessible information**.
It is never called "information the model uses".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEEDS = [19, 20, 21, 22]
ALPHAS = (0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5
MAX_POSITION = 8                     # positions 0..8
BOOT_B = 10_000
BOOT_SEED = 20260730
CI = (2.5, 97.5)

CONFIRMATORY = ("TRAINED_REAL_EXACT", "NOVEL_PSEUDOWORD")
EXTENSION = ("UNTRAINED_REAL",)
ELIGIBLE = CONFIRMATORY + EXTENSION
EXCLUDED = ("TRAINED_REAL_PRON_VARIANT", "PSEUDO_TRAINING_WORD",
            "PSEUDO_TRAINING_HOMOPHONE")

LENGTH_NOTE = (
    "The slope estimates a linear trend across the observed WFE lengths. The "
    "absence of length 6 does not invalidate the fit, but the coefficient "
    "should not be interpreted as evidence that the relationship is exactly "
    "linear at every intermediate length.")

INSTR = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/instrumented")
STAGE1 = os.path.join(INSTR, "stage1_encoder_extraction")
M4 = os.path.join(ROOT, "outputs/length_effect_mechanism_93a577f/m4_representation")

# stage key -> (label, kind, is_positionwise)
STAGES = {
    "ltm_encoder_hidden": ("1 LTM encoder h", "probe", False),
    "s_hat": ("2 raw s_hat", "probe", False),
    "ltm_decoder_h0": ("3 LTM decoder h0", "probe", False),
    "ltm_premotor_gold_prefix": ("4 gold-prefix premotor", "probe", True),
    "ltm_actual_gold_prefix_output": ("5 actual gold-prefix output",
                                      "actual_model_output", True),
    "wm_encoder_hidden": ("WM encoder h", "probe", False),
    "wm_premotor_gold_prefix": ("WM gold-prefix premotor", "probe", True),
}
LTM_PROBE_STAGES = ["ltm_encoder_hidden", "s_hat", "ltm_decoder_h0",
                    "ltm_premotor_gold_prefix"]
WM_PROBE_STAGES = ["wm_encoder_hidden", "wm_premotor_gold_prefix"]
PROBE_STAGES = LTM_PROBE_STAGES + WM_PROBE_STAGES
ALL_STAGES = LTM_PROBE_STAGES + ["ltm_actual_gold_prefix_output"] + WM_PROBE_STAGES


def sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def git(*a) -> str:
    return subprocess.run(("git",) + a, cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


# ============================================================ items and folds

def load_item_table() -> pd.DataFrame:
    """Canonical eligible-item table, verified identical across all four seeds."""
    s = pd.read_csv(os.path.join(INSTR, "item_summary.tsv"), sep="\t")
    per_seed = {}
    for seed in SEEDS:
        sub = s[s["seed"] == seed].reset_index(drop=True)
        sub["row_index"] = np.arange(len(sub))
        per_seed[seed] = sub
    ref = per_seed[SEEDS[0]]
    for seed in SEEDS[1:]:
        o = per_seed[seed]
        assert o["item_id"].tolist() == ref["item_id"].tolist(), seed
        assert o["phoneme_length"].tolist() == ref["phoneme_length"].tolist(), seed
        assert o["exposure_status"].tolist() == ref["exposure_status"].tolist(), seed
        assert o["target_tokens"].tolist() == ref["target_tokens"].tolist(), seed
    items = ref[["row_index", "item_id", "exposure_status", "phoneme_length",
                 "target_tokens"]].copy()
    items = items[items["exposure_status"].isin(ELIGIBLE)].reset_index(drop=True)
    return items


def assign_folds(items: pd.DataFrame) -> pd.DataFrame:
    """Deterministic item-grouped folds, stratified by exposure x length.

    Within each stratum, `item_id` is sorted lexicographically and
    `fold = rank mod 5`.  No RNG; identical for every seed and every stage.
    """
    out = items.copy()
    out["fold"] = -1
    for (_exp, _len), grp in out.groupby(["exposure_status", "phoneme_length"],
                                         sort=True):
        order = grp.sort_values("item_id").index
        out.loc[order, "fold"] = np.arange(len(order)) % N_FOLDS
    assert (out["fold"] >= 0).all()
    return out


# =============================================== phoneme targets and features

def phoneme_targets(items: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """(n_items, 9) int matrix of phoneme class ids; -1 where no phoneme."""
    toks = [r.split() for r in items["target_tokens"]]
    vocab = sorted({t for row in toks for t in row})
    code = {t: i for i, t in enumerate(vocab)}
    Y = np.full((len(items), MAX_POSITION + 1), -1, dtype=np.int64)
    for i, row in enumerate(toks):
        assert len(row) == items["phoneme_length"].iloc[i]
        assert len(row) <= MAX_POSITION + 1
        for p, t in enumerate(row):
            Y[i, p] = code[t]
    return Y, vocab


class Features:
    """Lazy per-seed access to the validated representation arrays."""

    def __init__(self, rows: np.ndarray):
        self.rows = rows                       # canonical row indices of eligible items
        self._npz = np.load(os.path.join(INSTR, "representations.npz"))
        self._cache: Dict[Tuple[str, int], np.ndarray] = {}

    def get(self, stage: str, seed: int) -> np.ndarray:
        key = (stage, seed)
        if key in self._cache:
            return self._cache[key]
        if stage == "ltm_encoder_hidden":
            a = np.load(os.path.join(STAGE1, f"ltm_encoder_hidden_seed{seed}.npy"))
        else:
            a = self._npz[f"{stage}_seed{seed}"]
        a = np.asarray(a, dtype=np.float64)[self.rows]
        self._cache[key] = a
        return a


def build_position_block(x: np.ndarray, pos: np.ndarray, n_pos: int) -> np.ndarray:
    """Position-block design: row r has its features in block `pos[r]` only.

    Rows have disjoint support across positions, so per-position fitting with
    per-position intercepts and per-position standardisation is exactly the
    block-diagonal solution of this design.  Constructed and tested; the fitting
    path slices it per position.
    """
    n, d = x.shape
    X = np.zeros((n, n_pos * d), dtype=x.dtype)
    X[np.arange(n), :] = 0.0
    for p in range(n_pos):
        m = pos == p
        if m.any():
            X[np.ix_(np.flatnonzero(m), np.arange(p * d, (p + 1) * d))] = x[m]
    return X


# ============================================================ ridge estimator

class RidgeHead:
    """Deterministic multiclass ridge readout, one symmetric eigendecomposition.

    One-vs-rest targets in {-1, +1}; weighted centering and (optional) weighted
    standardisation fitted on the training rows only; every alpha in the grid is
    obtained from the same factorisation.  Prediction is argmax of the decision
    scores with ties resolved to the lowest class index.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                 classes: Optional[np.ndarray] = None, scale: bool = True):
        n = len(y)
        w = np.asarray(w, dtype=np.float64)
        assert w.min() >= 0 and w.sum() > 0
        wn = w / w.sum()                       # for means
        ws = w * (n / w.sum())                 # keeps alpha on the sklearn scale
        self.classes = np.unique(y) if classes is None else np.asarray(classes)
        Y = np.full((n, len(self.classes)), -1.0)
        Y[np.arange(n), np.searchsorted(self.classes, y)] = 1.0

        self.xm = wn @ X
        Xc = X - self.xm
        if scale:
            var = wn @ (Xc ** 2)
            self.xs = np.sqrt(np.maximum(var, 0.0))
            self.xs[self.xs < 1e-8] = 1e-8
        else:
            self.xs = np.ones(X.shape[1])
        Xc = Xc / self.xs
        self.ym = wn @ Y
        Yc = Y - self.ym
        sw = np.sqrt(ws)[:, None]
        Xw, Yw = Xc * sw, Yc * sw
        G = Xw.T @ Xw
        b = Xw.T @ Yw
        lam, Q = np.linalg.eigh(G)
        self.lam, self.Q, self.Qtb = lam, Q, Q.T @ b

    def coef(self, alpha: float) -> np.ndarray:
        return self.Q @ (self.Qtb / (self.lam + alpha)[:, None])

    def decision(self, X: np.ndarray, alpha: float) -> np.ndarray:
        return ((X - self.xm) / self.xs) @ self.coef(alpha) + self.ym

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        return self.classes[np.argmax(self.decision(X, alpha), axis=1)]


class RidgeMultiOutput:
    """Deterministic multi-output ridge regression (unordered count probe)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, scale: bool = True):
        n = len(X)
        self.xm = X.mean(0)
        Xc = X - self.xm
        if scale:
            self.xs = Xc.std(0)
            self.xs[self.xs < 1e-8] = 1e-8
        else:
            self.xs = np.ones(X.shape[1])
        Xc = Xc / self.xs
        self.ym = Y.mean(0)
        Yc = Y - self.ym
        G = Xc.T @ Xc
        b = Xc.T @ Yc
        lam, Q = np.linalg.eigh(G)
        self.lam, self.Q, self.Qtb = lam, Q, Q.T @ b

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        coef = self.Q @ (self.Qtb / (self.lam + alpha)[:, None])
        return ((X - self.xm) / self.xs) @ coef + self.ym


# ================================================================== weighting

def cell_weights(items: pd.DataFrame, train_mask: np.ndarray) -> Dict[Tuple, float]:
    """1 / n_items_in_cell, computed on TRAINING items only."""
    tr = items[train_mask]
    n = tr.groupby(["exposure_status", "phoneme_length"]).size()
    return {k: 1.0 / v for k, v in n.items()}


def row_weights(items: pd.DataFrame, idx: np.ndarray,
                cw: Dict[Tuple, float]) -> np.ndarray:
    exp = items["exposure_status"].to_numpy()
    ln = items["phoneme_length"].to_numpy()
    return np.array([cw.get((exp[i], ln[i]), 0.0) for i in idx])


# ========================================================== the ordered probe

def _rows_for(items: pd.DataFrame, Y: np.ndarray, item_idx: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand items into (item, position) rows with a valid phoneme target."""
    it, po, ta = [], [], []
    for i in item_idx:
        L = int(items["phoneme_length"].iloc[i])
        for p in range(L):
            it.append(i)
            po.append(p)
            ta.append(Y[i, p])
    return np.array(it), np.array(po), np.array(ta)


def _fit_positions(feat_rows, pos, tgt, w, classes, alphas, scale=True):
    heads = {}
    for p in np.unique(pos):
        m = pos == p
        heads[int(p)] = RidgeHead(feat_rows[m], tgt[m], w[m], classes=classes,
                                  scale=scale)
    return heads


def _score(heads, feat_rows, pos, tgt, alpha, w=None) -> float:
    ok, tot = 0.0, 0.0
    for p, h in heads.items():
        m = pos == p
        if not m.any():
            continue
        pred = h.predict(feat_rows[m], alpha)
        ww = np.ones(m.sum()) if w is None else w[m]
        ok += float((pred == tgt[m]) @ ww)
        tot += float(ww.sum())
    return ok / tot if tot > 0 else 0.0


def _stage_rows(feats: Features, stage: str, seed: int, item_idx: np.ndarray,
                pos: np.ndarray, pca: Optional[int] = None,
                normalise: bool = False) -> np.ndarray:
    """Feature matrix for the given (item, position) rows.  PCA is applied later
    (it must be fitted on training rows only), so this returns native features."""
    A = feats.get(stage, seed)
    if normalise:
        A = A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-12)
    if STAGES[stage][2]:                     # timestep-specific
        return A[item_idx, pos]
    return A[item_idx]


def run_ordered_probe(items, folds, Y, feats, seed, stage, *, weighted=True,
                      pca=None, normalise=False):
    """Five-fold OOF ordered phoneme-at-position readout for one seed and stage."""
    n_items = len(items)
    all_idx = np.arange(n_items)
    item_r, pos, tgt = _rows_for(items, Y, all_idx)
    feat = _stage_rows(feats, stage, seed, item_r, pos, normalise=normalise)
    assert np.isfinite(feat).all(), (stage, seed)
    fold_of_item = folds["fold"].to_numpy()
    row_fold = fold_of_item[item_r]
    classes = np.unique(Y[Y >= 0])

    oof_pred = np.full(len(tgt), -1, dtype=np.int64)
    oof_margin = np.full(len(tgt), np.nan)
    chosen_alpha = {}
    for k in range(N_FOLDS):
        te = row_fold == k
        tr = ~te
        tr_items = fold_of_item != k
        cw = cell_weights(items, tr_items)
        w_tr = row_weights(items, item_r[tr], cw) if weighted \
            else np.ones(int(tr.sum()))
        Xtr, Xte = feat[tr], feat[te]
        if pca is not None:
            mu = Xtr.mean(0)
            U, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
            V = Vt[:pca].T
            Xtr, Xte = (Xtr - mu) @ V, (Xte - mu) @ V

        # ---- nested alpha selection, inside the training folds only
        inner_folds = [f for f in range(N_FOLDS) if f != k]
        acc = {a: [] for a in ALPHAS}
        for j in inner_folds:
            ite = row_fold[tr] == j
            itr = ~ite
            icw = cell_weights(items, (fold_of_item != k) & (fold_of_item != j))
            iw = row_weights(items, item_r[tr][itr], icw) if weighted \
                else np.ones(int(itr.sum()))
            # scoring weights come from the inner TRAINING composition only
            sw = row_weights(items, item_r[tr][ite], icw)
            heads = _fit_positions(Xtr[itr], pos[tr][itr], tgt[tr][itr], iw,
                                   classes, ALPHAS)
            for a in ALPHAS:
                acc[a].append(_score(heads, Xtr[ite], pos[tr][ite],
                                     tgt[tr][ite], a, sw))
        means = {a: float(np.mean(v)) for a, v in acc.items()}
        best = max(ALPHAS, key=lambda a: (means[a], -a))
        chosen_alpha[k] = best

        heads = _fit_positions(Xtr, pos[tr], tgt[tr], w_tr, classes, ALPHAS)
        te_rows = np.flatnonzero(te)
        for p, h in heads.items():
            m = pos[te] == p
            if not m.any():
                continue
            dec = h.decision(Xte[m], best)
            pr = h.classes[np.argmax(dec, axis=1)]
            oof_pred[te_rows[m]] = pr
            t = tgt[te][m]
            true_col = np.searchsorted(h.classes, t)
            ts = dec[np.arange(len(t)), true_col]
            d2 = dec.copy()
            d2[np.arange(len(t)), true_col] = -np.inf
            oof_margin[te_rows[m]] = ts - d2.max(1)
    assert (oof_pred >= 0).all()
    return pd.DataFrame({
        "seed": seed, "stage": stage,
        "item_id": items["item_id"].to_numpy()[item_r],
        "exposure_status": items["exposure_status"].to_numpy()[item_r],
        "phoneme_length": items["phoneme_length"].to_numpy()[item_r],
        "fold": row_fold, "position": pos, "target_class": tgt,
        "predicted_class": oof_pred, "correct": (oof_pred == tgt).astype(int),
        "decision_margin": oof_margin,
    }), chosen_alpha


def actual_output_rows(items: pd.DataFrame) -> pd.DataFrame:
    """Stage 5: the trained model's actual gold-prefix tokenwise decisions.

    Not a probe and not fitted: read from the frozen `timestep_metrics.tsv`.
    """
    t = pd.read_csv(os.path.join(INSTR, "timestep_metrics.tsv"), sep="\t")
    t = t[(t["route"] == "ltm") & (t["decode_mode"] == "gold_prefix")
          & (t["prefix_source"] == "gold")]
    t = t[t["target_token"] != "<eos>"]
    keep = set(items["item_id"])
    t = t[t["item_id"].isin(keep)].copy()
    meta = items.set_index("item_id")[["exposure_status", "phoneme_length"]]
    t = t.join(meta, on="item_id")
    t["stage"] = "ltm_actual_gold_prefix_output"
    t = t.rename(columns={"timestep": "position", "is_correct": "correct"})
    return t[["seed", "stage", "item_id", "exposure_status", "phoneme_length",
              "position", "correct"]]


# ================================================== per-item summaries, slopes

def item_error(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["seed", "stage", "item_id", "exposure_status",
                    "phoneme_length"], as_index=False)["correct"].mean()
    g["token_error"] = 1.0 - g["correct"]
    return g.drop(columns=["correct"])


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vx = x.var()
    return float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 0 else np.nan


def macro_f1(tgt: np.ndarray, pred: np.ndarray) -> float:
    cls = np.unique(tgt)
    f1 = []
    for c in cls:
        tp = float(((pred == c) & (tgt == c)).sum())
        fp = float(((pred == c) & (tgt != c)).sum())
        fn = float(((pred != c) & (tgt == c)).sum())
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


# ================================================== hierarchical bootstrap

class Bootstrap:
    """Paired hierarchical bootstrap over the OOF item-level quantities.

    Seeds are resampled with replacement; then items with replacement within
    `exposure_status x phoneme_length` strata.  The same drawn indices are reused
    across stages *and* across exposure groups within a replicate, so every
    stage-to-stage and novel-minus-trained contrast is paired.  Probes are never
    refitted inside a replicate: intervals are conditional on the fitted OOF
    probes.
    """

    def __init__(self, err: np.ndarray, lengths: np.ndarray, strata: np.ndarray,
                 groups: np.ndarray, b: int = BOOT_B, rng_seed: int = BOOT_SEED,
                 chunk: int = 200):
        self.err = err                      # (n_stages, n_seeds, n_items)
        self.lengths = lengths.astype(float)
        self.groups = np.asarray(groups)
        self.b, self.chunk = b, chunk
        self.rng = np.random.default_rng(rng_seed)
        uniq = sorted(set(strata.tolist()))
        self._strata_idx = [np.flatnonzero(strata == s) for s in uniq]
        # column -> exposure group of the stratum that column is drawn from
        self._col_group = np.concatenate(
            [self.groups[idx] for idx in self._strata_idx])
        self.group_names = sorted(set(self.groups.tolist()))

    def _draw_items(self, n_draw: int) -> np.ndarray:
        out = np.empty((n_draw, self.err.shape[2]), dtype=np.int64)
        col = 0
        for idx in self._strata_idx:
            k = len(idx)
            out[:, col:col + k] = idx[self.rng.integers(0, k, size=(n_draw, k))]
            col += k
        assert col == self.err.shape[2]
        return out

    def run(self) -> Dict[str, Dict[str, np.ndarray]]:
        """{group: {"mean": (B, n_stages), "slope": (B, n_stages)}}."""
        S, Sd, _ = self.err.shape
        out = {g: {"mean": np.empty((self.b, S)), "slope": np.empty((self.b, S))}
               for g in self.group_names}
        col_mask = {g: self._col_group == g for g in self.group_names}
        done = 0
        while done < self.b:
            m = min(self.chunk, self.b - done)
            seed_idx = self.rng.integers(0, Sd, size=(m, Sd))
            item_idx = self._draw_items(m * Sd).reshape(m, Sd, -1)
            for g in self.group_names:
                cm = col_mask[g]
                gi = item_idx[:, :, cm]
                L = self.lengths[gi]
                Lc = L - L.mean(-1, keepdims=True)
                vx = (Lc ** 2).mean(-1)
                for s in range(S):
                    e = self.err[s][seed_idx[..., None], gi]     # (m, Sd, n_g)
                    out[g]["mean"][done:done + m, s] = e.mean(-1).mean(-1)
                    cov = (Lc * (e - e.mean(-1, keepdims=True))).mean(-1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        sl = np.where(vx > 0, cov / vx, np.nan)
                    out[g]["slope"][done:done + m, s] = np.nanmean(sl, axis=-1)
            done += m
        return out


def pct(a: np.ndarray) -> Tuple[float, float]:
    return (float(np.nanpercentile(a, CI[0])), float(np.nanpercentile(a, CI[1])))
