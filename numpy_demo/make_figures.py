"""Train the dual-route model once and write every figure into an organized
`figures/` tree:

    figures/
      train/      training_loss.png, logfreq_curriculum.png
      eval/       generalization.png, nonword_by_length.png, primacy_recency.png
      ablation/   ablation_severity.png, ablation_dissociation.png, ablation_length.png

    python -m numpy_demo.make_figures            # train fresh
    python -m numpy_demo.make_figures --reuse    # reuse cached model (fast)

The trained model is cached under outputs/numpy_demo/ so figures can be
regenerated without retraining.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_demo.common import make_data, train_model, build_model
from numpy_demo.dual_route_numpy import accuracy
from numpy_demo.ablation import _sweep, _plot_severity, _plot_dissociation, _plot_length

FIG = "figures"
CACHE = os.path.join("outputs", "numpy_demo", ".model_cache.npz")
N_CORE, STEPS, H_SEM, D_WM = 1800, 2200, 32, 96


def _path(*parts):
    p = os.path.join(FIG, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def get_or_train(data, reuse):
    if reuse and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        model = build_model(data, h_sem=int(z["h_sem"]), d_wm=int(z["d_wm"]))
        model.set_params({k: z["p_" + k] for k in model.PARAMS})
        hist = [{"step": int(s), "train": float(t), "unseen": float(u)}
                for s, t, u in zip(z["h_step"], z["h_train"], z["h_unseen"])]
        print("[figures] reusing cached model")
        return model, hist
    hist = []
    print("[figures] training the dual-route model ...")
    model = train_model(data, steps=STEPS, h_sem=H_SEM, d_wm=D_WM,
                        verbose=True, history=hist)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, h_sem=H_SEM, d_wm=D_WM,
             h_step=[h["step"] for h in hist], h_train=[h["train"] for h in hist],
             h_unseen=[h["unseen"] for h in hist],
             **{"p_" + k: v for k, v in model.get_params().items()})
    return model, hist


def _acc(model, X, T, M, route, rng=None):
    return accuracy(model, X, T, M, route=route,
                    rng=rng or np.random.default_rng(0))


# ----------------------------------------------------------------- eval figures
def _bars(report, routes, path):
    pools = list(report.keys())
    x = np.arange(len(pools)); w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, r in enumerate(routes):
        ax.bar(x + (i - 1) * w, [report[p][r] for p in pools], w, label=r)
    ax.set_xticks(x); ax.set_xticklabels(pools)
    ax.set_ylabel("whole-word repetition accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Repetition by item type and route")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _length_curve(model, data, path):
    X, T, M, lens = data["Xnw"], data["Tnw"], data["Mnw"], data["len_nw"]
    uniq = sorted(set(int(l) for l in lens))
    xs, series = [], {"ventral": [], "dorsal": [], "gated": []}
    for L_ in uniq:
        sel = lens == L_
        if sel.sum() < 4:
            continue
        xs.append(L_)
        for r in series:
            series[r].append(_acc(model, X[sel], T[sel], M[sel], r)[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    sty = {"ventral": "--o", "dorsal": "-s", "gated": ":^"}
    for r, ys in series.items():
        ax.plot(xs, ys, sty[r], label=r)
    ax.set(title="Nonword repetition vs length", xlabel="word length (phonemes)",
           ylabel="phoneme accuracy", ylim=(0, 1.05))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _logfreq_curriculum(data, path):
    ranks = np.array([e.rank for e in data["train"]])
    order = np.argsort(ranks)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks[order], data["present_prob"][order], lw=2)
    ax.set(title="Log-frequency presentation curriculum",
           xlabel="word frequency rank (1 = most frequent)",
           ylabel="presentation probability")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def primacy_recency(model, data, path, n_seq=300, n_trials=10, noise=2.0):
    """Serial-position curve of the dorsal route on fixed-length pronounceable
    sequences, with interference noise. Errors concentrate away from the edges."""
    vocab, L = data["vocab"], data["L"]
    cons = [vocab.stoi[s] for s in vocab.itos[3:] if vocab.sonority[vocab.stoi[s]] < 0.9]
    vow = [vocab.stoi[s] for s in vocab.itos[3:] if vocab.sonority[vocab.stoi[s]] >= 0.95]
    rng = np.random.default_rng(1)
    T = np.zeros((n_seq, L), np.int64)
    for i in range(n_seq):
        for t in range(L):
            T[i, t] = int(rng.choice(vow if t % 2 else cons))   # CV alternation
    X = np.zeros((n_seq, L, vocab.size), np.float32)
    X[np.arange(n_seq)[:, None], np.arange(L)[None, :], T] = 1.0
    M = np.ones((n_seq, L), np.float32)
    acc = np.zeros(L)
    for tr in range(n_trials):
        _, c = model.forward(X, T, M, lesion={"dorsal": (0.0, noise)},
                             rng=np.random.default_rng(50 + tr))
        acc += (c["pd"].argmax(-1) == T).mean(0)
    acc /= n_trials
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, L + 1), acc, marker="o")
    ax.set(title="Serial-position curve (dorsal recurrent route)",
           xlabel="serial position", ylabel="phoneme accuracy", ylim=(0, 1.05))
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return acc.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="reuse cached model")
    args = ap.parse_args()

    data = make_data(n_core=N_CORE, n_unseen=400, n_nonword=300, L=9, seed=0)
    model, hist = get_or_train(data, args.reuse)
    summary = {"lexicon": data["source"], "n_train": len(data["train"])}

    # ---- TRAIN ----
    st = [h["step"] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(st, [h["train"] for h in hist], label="train")
    plt.plot(st, [h["unseen"] for h in hist], label="held-out real words")
    plt.xlabel("training step"); plt.ylabel("cross-entropy loss")
    plt.title("Dual-route repetition: training loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(_path("train", "training_loss.png"), dpi=130); plt.close()
    _logfreq_curriculum(data, _path("train", "logfreq_curriculum.png"))

    # ---- EVAL ----
    pools = {"trained word": ("Xtr", "Ttr", "Mtr"),
             "unseen real word": ("Xun", "Tun", "Mun"),
             "nonword": ("Xnw", "Tnw", "Mnw")}
    routes = ["ventral", "dorsal", "gated"]
    report = {}
    for name, (kx, kt, km) in pools.items():
        report[name] = {r: float(_acc(model, data[kx], data[kt], data[km], r)[1])
                        for r in routes}
        print(f"[eval] {name:18s} " + "  ".join(
            f"{r}={report[name][r]:.3f}" for r in routes))
    _bars(report, routes, _path("eval", "generalization.png"))
    _length_curve(model, data, _path("eval", "nonword_by_length.png"))
    summary["serial_position"] = primacy_recency(
        model, data, _path("eval", "primacy_recency.png"))
    summary["accuracy"] = report

    # ---- ABLATION ----
    k = 300
    word = (data["Xtr"][:k], data["Ttr"][:k], data["Mtr"][:k])
    nonword = (data["Xnw"], data["Tnw"], data["Mnw"])
    sweeps = {"dorsal": _sweep(model, word, nonword, "dorsal"),
              "ventral": _sweep(model, word, nonword, "ventral")}
    _plot_severity(sweeps, _path("ablation", "ablation_severity.png"))
    _plot_dissociation(sweeps, 0.8, _path("ablation", "ablation_dissociation.png"))
    _plot_length(model, data, _path("ablation", "ablation_length.png"))
    for site in ("dorsal", "ventral"):
        s = sweeps[site][0.8]
        print(f"[ablation] {site:7s} @0.8  word={s['word'][0]:.3f}  "
              f"nonword={s['nonword'][0]:.3f}")

    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] figures under ./{FIG}/ (train, eval, ablation)")


if __name__ == "__main__":
    main()
