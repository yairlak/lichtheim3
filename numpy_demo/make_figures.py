"""Run training + all runnable evaluations and save every figure into an
organized `figures/` tree:

    figures/
      train/      training_loss.png, logfreq_curriculum.png
      eval/       generalization.png, nonword_by_length.png, primacy_recency.png
      ablation/   ablation_severity.png, ablation_dissociation.png, ablation_length.png

Trains the dual-route model once and reuses it for every figure.

    python -m numpy_demo.make_figures
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
from numpy_demo.run_demo import _bars, _length_curve, _logfreq_curriculum, _acc
from numpy_demo.ablation import _sweep, _plot_severity, _plot_dissociation, _plot_length

FIG_ROOT = "figures"


def _path(*parts):
    p = os.path.join(FIG_ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def primacy_recency(model, data, path, length=None, n_seq=300, n_trials=12):
    """Serial-position curve of the dorsal WM buffer on random sequences.
    Capacity + primacy/recency trace strength -> elevated edges, sagging middle."""
    vocab, L = data["vocab"], data["L"]
    length = length or L
    phon = [vocab.stoi[s] for s in vocab.itos[3:]]
    rng = np.random.default_rng(1)
    X = np.zeros((n_seq, L, vocab.size), np.float32)
    T = np.full((n_seq, L), vocab.pad_id, np.int64)
    M = np.zeros((n_seq, L), np.float32)
    for i in range(n_seq):
        for t in range(length):
            pid = int(rng.choice(phon))
            X[i, t, pid] = 1.0
            T[i, t] = pid
            M[i, t] = 1.0
    acc = np.zeros(length)
    for tr in range(n_trials):
        r = np.random.default_rng(100 + tr)
        _, c = model.forward(X, M=M, noise_std=0.6, rng=r)
        pred = c["pd"].argmax(-1)
        acc += (pred[:, :length] == T[:, :length]).mean(0)
    acc /= n_trials
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, length + 1), acc, marker="o")
    plt.title("Serial-position curve (dorsal WM buffer)")
    plt.xlabel("serial position"); plt.ylabel("phoneme accuracy")
    plt.ylim(0, 1.05); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()
    return acc.tolist()


def main():
    print("[figures] training the dual-route model once ...")
    data = make_data(n_core=2500, n_unseen=400, n_nonword=400, L=9, seed=0)
    hist = []
    model = train_model(data, epochs=400, h_sem=256, verbose=True, history=hist)
    summary = {"lexicon": data["source"], "n_train": len(data["train"])}

    # ---------------- TRAIN ----------------
    ep = [h["epoch"] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["train"] for h in hist], label="train (log-freq weighted)")
    plt.plot(ep, [h["unseen"] for h in hist], label="held-out real words")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy loss")
    plt.title("Dual-route repetition: training loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(_path("train", "training_loss.png"), dpi=130); plt.close()
    _logfreq_curriculum(data, _path("train", "logfreq_curriculum.png"))

    # ---------------- EVAL ----------------
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

    # ---------------- ABLATION ----------------
    k = 600
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

    os.makedirs(FIG_ROOT, exist_ok=True)
    with open(os.path.join(FIG_ROOT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] figures written under ./{FIG_ROOT}/ (train, eval, ablation)")


if __name__ == "__main__":
    main()
