"""Tiny matplotlib helpers shared by the evaluation scripts."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def line(xs: Sequence[float], ys: Sequence[float], title: str, xlabel: str,
         ylabel: str, path: str, yerr: Sequence[float] | None = None) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    if yerr is not None:
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3)
    else:
        ax.plot(xs, ys, marker="o")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def multiline(xs: Sequence[float], series: Dict[str, Sequence[float]], title: str,
              xlabel: str, ylabel: str, path: str) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", label=name)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def heatmap(mat: np.ndarray, title: str, xlabel: str, ylabel: str, path: str,
            xticklabels: List[str] | None = None,
            yticklabels: List[str] | None = None) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=6)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=6)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def grouped_bars(groups: List[str], series: Dict[str, List[float]], title: str,
                 ylabel: str, path: str) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    n = len(series)
    width = 0.8 / max(n, 1)
    x = np.arange(len(groups))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + i * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels(groups)
    ax.set(title=title, ylabel=ylabel)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
