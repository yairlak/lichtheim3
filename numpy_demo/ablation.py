"""Lesion / ablation study, after Ueno, Saito, Rogers & Lambon Ralph (2011),
"Lichtheim 2", Neuron 72:385-396.

Ueno et al. simulate brain damage by removing a proportion of a layer's
incoming links AND adding noise over its output, titrating severity and
repeating over random "patients" (seeds), then plotting mean +/- SE. Damaging
the DORSAL (iSMG) pathway selectively impairs repetition of NONWORDS while
sparing real words (conduction-aphasia profile); damaging the VENTRAL (vATL)
pathway does the reverse (lexical/semantic loss, nonwords spared).

Here we reproduce that design on the dual-route repetition model:

  * "words"    = trained lexicon items (seen)
  * "nonwords" = novel held-out strings (unseen)
  * a "patient" = the gated full model with one pathway lesioned at severity s

Figures written to outputs/numpy_demo/:
  ablation_severity.png      severity curves, 2 panels (dorsal | ventral lesion)
  ablation_dissociation.png  double dissociation at a fixed severity
  ablation_length.png        length x dorsal-lesion interaction (nonwords)
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
SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_PATIENTS = 10
DORSAL_NOISE_MAX = 5.0      # logit-scale noise at severity 1 (beta=6)
VENTRAL_NOISE_MAX = 4.0


def _lesion_spec(site, s):
    noise = (DORSAL_NOISE_MAX if site == "dorsal" else VENTRAL_NOISE_MAX) * s
    return {site: (s, noise)}


def _sweep(model, word, nonword, site):
    """Return dict severity -> {word:(mean,se), nonword:(mean,se)} (whole-word acc).

    `word` and `nonword` are (X, T, M) tuples.
    """
    res = {}
    for s in SEVERITIES:
        wseen, wnon = [], []
        for patient in range(N_PATIENTS):
            rng = np.random.default_rng(2000 + patient)
            les = _lesion_spec(site, s) if s > 0 else None
            _, ws = _acc_pair(model, *word, rng, les)
            _, wn = _acc_pair(model, *nonword, rng, les)
            wseen.append(ws); wnon.append(wn)
            if s == 0.0:           # intact: identical every patient, one is enough
                break
        res[s] = {
            "word": (float(np.mean(wseen)), float(_se(wseen))),
            "nonword": (float(np.mean(wnon)), float(_se(wnon))),
        }
    return res


def _acc_pair(model, X, T, M, rng, les):
    # numpy accuracy() returns (phoneme_acc, word_acc)
    return accuracy(model, X, T, M, route="gated", rng=rng, lesion=les)


def _se(v):
    v = np.asarray(v, float)
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[ablation] training the healthy model on the real lexicon ...")
    data = make_data(n_core=2500, n_unseen=400, n_nonword=400, L=9, seed=0)
    model = train_model(data, epochs=400, verbose=True)

    # eval pools: real trained words vs novel nonwords (subset words for speed)
    k = 600
    word = (data["Xtr"][:k], data["Ttr"][:k], data["Mtr"][:k])
    nonword = (data["Xnw"], data["Tnw"], data["Mnw"])

    print("[ablation] lesioning dorsal (iSMG) and ventral (vATL) pathways ...")
    sweeps = {"dorsal": _sweep(model, word, nonword, "dorsal"),
              "ventral": _sweep(model, word, nonword, "ventral")}

    _plot_severity(sweeps, os.path.join(OUT, "ablation_severity.png"))
    _plot_dissociation(sweeps, 0.8, os.path.join(OUT, "ablation_dissociation.png"))
    _plot_length(model, data, os.path.join(OUT, "ablation_length.png"))

    with open(os.path.join(OUT, "ablation_report.json"), "w") as f:
        json.dump(sweeps, f, indent=2)
    for site in ("dorsal", "ventral"):
        s = 0.8
        w = sweeps[site][s]["word"][0]; nw = sweeps[site][s]["nonword"][0]
        print(f"[ablation] {site:7s} lesion @s=0.8  word={w:.3f}  nonword={nw:.3f}")
    print(f"\n[done] ablation figures in {OUT}/")


# --------------------------------------------------------------------- figures
def _plot_severity(sweeps, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    titles = {"dorsal": "Dorsal (iSMG) pathway lesion",
              "ventral": "Ventral (vATL) pathway lesion"}
    xs = [int(s * 100) for s in SEVERITIES]
    for ax, site in zip(axes, ("dorsal", "ventral")):
        r = sweeps[site]
        for cond, style in (("word", dict(marker="o", color="tab:blue")),
                            ("nonword", dict(marker="s", color="tab:red"))):
            ys = [r[s][cond][0] for s in SEVERITIES]
            es = [r[s][cond][1] for s in SEVERITIES]
            ax.errorbar(xs, ys, yerr=es, capsize=3, label=cond, **style)
        ax.set(title=titles[site], xlabel="lesion severity (% links removed + noise)",
               ylim=(0, 1.03))
        ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("repetition accuracy")
    fig.suptitle("Lesion severity curves (mean ± SE over patients)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_dissociation(sweeps, s, path):
    fig, ax = plt.subplots(figsize=(6.2, 4))
    sites = ["dorsal", "ventral"]
    x = np.arange(len(sites)); w = 0.38
    word = [sweeps[si][s]["word"][0] for si in sites]
    nonword = [sweeps[si][s]["nonword"][0] for si in sites]
    word_e = [sweeps[si][s]["word"][1] for si in sites]
    nonword_e = [sweeps[si][s]["nonword"][1] for si in sites]
    ax.bar(x - w/2, word, w, yerr=word_e, capsize=3, label="word (trained)",
           color="tab:blue")
    ax.bar(x + w/2, nonword, w, yerr=nonword_e, capsize=3, label="nonword (novel)",
           color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(["Dorsal lesion\n(conduction-aphasia)",
                        "Ventral lesion\n(lexical-semantic)"])
    ax.set_ylabel("repetition accuracy"); ax.set_ylim(0, 1.03)
    ax.set_title(f"Double dissociation under lesion (severity {int(s*100)}%)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def _plot_length(model, data, path):
    """Nonword repetition by length, intact vs dorsal lesion (length interaction)."""
    Xte, Tte, Mte, lens = data["Xnw"], data["Tnw"], data["Mnw"], data["len_nw"]
    uniq = sorted(set(int(l) for l in lens))
    xs, intact, lesioned = [], [], []
    for L_ in uniq:
        sel = lens == L_
        if sel.sum() < 3:
            continue
        xs.append(L_)
        rng = np.random.default_rng(7)
        intact.append(accuracy(model, Xte[sel], Tte[sel], Mte[sel],
                               route="gated", rng=rng)[0])
        accs = []
        for patient in range(N_PATIENTS):
            rng = np.random.default_rng(3000 + patient)
            accs.append(accuracy(model, Xte[sel], Tte[sel], Mte[sel], route="gated",
                                 rng=rng, lesion={"dorsal": (0.6, 3.0)})[0])
        lesioned.append(float(np.mean(accs)))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, intact, marker="o", label="intact")
    ax.plot(xs, lesioned, marker="s", label="dorsal lesion (60%)")
    ax.set(title="Nonword repetition by length: dorsal lesion hits long items hardest",
           xlabel="word length (phonemes)", ylabel="phoneme accuracy (nonwords)",
           ylim=(0, 1.03))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
