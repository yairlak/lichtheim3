"""One-page schematic of the Lichtheim3 final training lineage.

Makes explicit that every stage is a CONTINUATION of the same from-scratch
run -- not a separately pretrained model -- and annotates only the
intentional treatment changes.
"""
from __future__ import annotations
import argparse, os, sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, sub, fc, ec="0.25"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
                fontsize=8.6, fontweight="bold")
        ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="top",
                fontsize=7.1, linespacing=1.45)

    def arrow(x1, y1, x2, y2, label=None, style="-|>", colour="0.25"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, lw=1.1, color=colour,
                                     shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, max(y1, y2) + 1.3, label, ha="center",
                    fontsize=6.9, style="italic", color="0.2")

    box(1, 17, 17, 13, "FROM SCRATCH",
        "random init, seed s\nwm128 / enc E / dec E\nno pretraining", "#eef3fa")
    box(21, 17, 18, 13, "joint 1:2:3  u0 → u500",
        "one shared AdamW\nlr 1e-3 → 1e-4 at\nR-cursor 46,300 (u100)", "#eef3fa")
    box(42, 26, 17, 12, "H512  u500 → u750",
        "same recipe,\nlonger horizon", "#fde9e9")
    box(42, 4, 17, 12, "H256  (stopped u500)",
        "C ≈ 57.8%\nnot continued", "#e9eef7")
    box(62, 33, 16, 11, "u750 → u850\ncontrol 1e-4", "R/N/C = 1e-4", "#efefef")
    box(62, 19, 16, 11, "u750 → u850\nall 3e-5",
        "R/N/C = 3e-5\nN exact 1.0", "#e8f4ea")
    box(82, 26, 16, 11, "u850 → u1200\nall 3e-5", "R/N/C = 3e-5", "#e8f4ea")
    box(82, 10, 16, 11, "u850 → u1200\nC-high", "R/N = 3e-5\nC = 1e-4", "#f1e9f7")

    arrow(18, 23.5, 21, 23.5)
    arrow(39, 25, 42, 32, "capacity factor:\nenc/dec 256 vs 512")
    arrow(39, 22, 42, 12)
    arrow(59, 33, 62, 38, "branch: global LR")
    arrow(59, 31, 62, 26)
    arrow(78, 24.5, 82, 31, "branch: C-only LR")
    arrow(78, 23, 82, 18)
    ax.text(50, 43.5, "Lichtheim3 final joint lineage — every arrow is a "
            "CONTINUATION of the same run\n(model, shared AdamW moments, RNG, "
            "cursors and schedule phase all preserved; only the annotated "
            "factor changes)",
            ha="center", va="top", fontsize=9.2)
    ax.text(50, 1.5, "u = repetition exposures;  1 macro-cycle = 6 optimizer "
            "steps = 1 R + 2 N + 3 C;  steps(u) = 2778 u",
            ha="center", fontsize=7, color="0.35")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(a.out_dir, f"figF_lineage.{ext}")
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"[lineage] wrote {p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
