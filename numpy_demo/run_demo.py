"""Train the NumPy dual-route model on the realistic English lexicon with
log-frequency weighting, plot the loss, and evaluate repetition on trained
words, held-out real words, and novel nonwords (cf. arXiv:2506.13450).

    python -m numpy_demo.run_demo

Writes figures + a JSON report into outputs/numpy_demo/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numpy_demo.common import make_data, train_model
from numpy_demo.dual_route_numpy import accuracy

OUT = os.path.join("outputs", "numpy_demo")


def _acc(model, X, T, M, route, rng=None):
    return accuracy(model, X, T, M, route=route,
                    rng=rng or np.random.default_rng(0))


def main():
    os.makedirs(OUT, exist_ok=True)
    data = make_data(n_core=2500, n_unseen=400, n_nonword=400, L=9, seed=0)
    print(f"[data] lexicon={data['source']} train(words)={len(data['train'])} "
          f"unseen(real)={len(data['unseen'])} nonwords={len(data['nonword'])}")

    hist = []
    model = train_model(data, epochs=400, h_sem=256, verbose=True, history=hist)

    # ---- loss curve ----
    ep = [h["epoch"] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["train"] for h in hist], label="train (log-freq weighted)")
    plt.plot(ep, [h["unseen"] for h in hist], label="held-out real words")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy loss")
    plt.title("Dual-route repetition: training loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "training_loss.png"), dpi=130); plt.close()

    # ---- accuracy by item type and route ----
    pools = {"trained word": ("Xtr", "Ttr", "Mtr"),
             "unseen real word": ("Xun", "Tun", "Mun"),
             "nonword": ("Xnw", "Tnw", "Mnw")}
    routes = ["ventral", "dorsal", "gated"]
    report = {}
    for name, (kx, kt, km) in pools.items():
        report[name] = {}
        for r in routes:
            _, w = _acc(model, data[kx], data[kt], data[km], r)
            report[name][r] = float(w)
        print(f"[acc] {name:18s} " + "  ".join(
            f"{r}={report[name][r]:.3f}" for r in routes))
    _bars(report, routes, os.path.join(OUT, "generalization.png"))

    # ---- nonword repetition by length (dorsal length effect) ----
    _length_curve(model, data, os.path.join(OUT, "unseen_by_length.png"))

    # ---- the log-frequency training curriculum ----
    _logfreq_curriculum(data, os.path.join(OUT, "frequency_effect.png"))

    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump({"history": hist, "accuracy": report}, f, indent=2)
    print(f"\n[done] figures + report.json in {OUT}/")


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
    xs = []
    series = {"ventral": [], "dorsal": [], "gated": []}
    for L_ in uniq:
        sel = lens == L_
        if sel.sum() < 4:
            continue
        xs.append(L_)
        for r in series:
            pa, _ = _acc(model, X[sel], T[sel], M[sel], r)
            series[r].append(pa)
    fig, ax = plt.subplots(figsize=(6, 4))
    sty = {"ventral": "--o", "dorsal": "-s", "gated": ":^"}
    for r, ys in series.items():
        ax.plot(xs, ys, sty[r], label=r)
    ax.set(title="Nonword repetition vs length",
           xlabel="word length (phonemes)", ylabel="phoneme accuracy", ylim=(0, 1.05))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _logfreq_curriculum(data, path):
    """Show the training curriculum: word presentation weight vs frequency rank.
    Words are sampled/weighted by LOG frequency (Zipfian freq -> log compresses
    the high-frequency tail)."""
    ranks = np.sort(np.array([e.rank for e in data["train"]]))
    w = data["w_train"][np.argsort([e.rank for e in data["train"]])]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ranks, w, lw=2)
    ax.set(title="Log-frequency training curriculum",
           xlabel="word frequency rank (1 = most frequent)",
           ylabel="relative training weight")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
