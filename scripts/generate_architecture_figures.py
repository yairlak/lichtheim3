#!/usr/bin/env python3
"""Generate Lichtheim3 architecture meeting figures.

Creates:
    outputs/architecture_meeting_figures/
        current_architecture_global.{svg,png}
        gate_premotor_blend.{svg,png}
        wm_noise_location.{svg,png}
        tf_vs_ar_decoding.{svg,png}
        current_vs_future_gate.{svg,png}
        lichtheim2_inspiration_vs_l3_current.{svg,png}

Run from repo root:
    python scripts/generate_architecture_figures.py
"""
from __future__ import annotations
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs', 'architecture_meeting_figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ── colour palette ────────────────────────────────────────────────────────────
WM   = dict(fc='#DBEAFE', ec='#1D4ED8', tc='#1E3A5F')   # blue   – dorsal
LTM  = dict(fc='#FFEDD5', ec='#C2410C', tc='#7C2D12')   # orange – ventral
GATE = dict(fc='#EDE9FE', ec='#6D28D9', tc='#3B0764')   # purple – gate
MTR  = dict(fc='#D1FAE5', ec='#047857', tc='#064E3B')   # green  – motor
NSE  = dict(fc='#FEE2E2', ec='#B91C1C', tc='#7F1D1D')   # red    – noise
IO   = dict(fc='#F1F5F9', ec='#475569', tc='#0F172A')   # slate  – io
BLD  = dict(fc='#F3E8FF', ec='#7C3AED', tc='#4C1D95')   # violet – blend
NTE  = dict(fc='#FAFAFA', ec='#D1D5DB', tc='#6B7280')   # gray   – note
PRP  = dict(fc='#FFF1F2', ec='#E11D48', tc='#9F1239')   # rose   – proposed
BGW  = dict(fc='#EFF6FF', ec='#93C5FD', tc='#1E3A5F')   # pale blue band
BGL  = dict(fc='#FFF7ED', ec='#FED7AA', tc='#7C2D12')   # pale orange band


# ── primitives ────────────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, txt, s=IO, fs=8.5, lw=2.0, zo=3):
    """Rounded rectangle centred at (cx, cy)."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.05',
        fc=s['fc'], ec=s['ec'], lw=lw, zorder=zo, clip_on=False))
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
            color=s['tc'], fontweight='bold', multialignment='center',
            linespacing=1.3, zorder=zo + 1, clip_on=False)


def notebox(ax, cx, cy, w, h, txt, s=NTE, fs=7.5, zo=3):
    """Light note box (not bold)."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.05',
        fc=s['fc'], ec=s['ec'], lw=1.2, zorder=zo, clip_on=False))
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
            color=s['tc'], multialignment='center',
            linespacing=1.3, zorder=zo + 1, clip_on=False)


def band(ax, x0, y0, x1, y1, s, alpha=0.5, zo=0):
    """Background colour band."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle='round,pad=0.1',
        fc=s['fc'], ec=s['ec'], lw=1.0, alpha=alpha, zorder=zo, clip_on=False))


def harr(ax, x0, y, x1, c='#475569', lw=1.8, lbl=None, yo=0.13, zo=2):
    """Horizontal arrow x0→x1 at height y."""
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=14),
                zorder=zo)
    if lbl:
        ax.text((x0 + x1) / 2, y + yo, lbl, ha='center', va='bottom',
                fontsize=7.5, color=c, fontstyle='italic', zorder=zo + 1)


def varr(ax, x, y0, y1, c='#475569', lw=1.8, lbl=None, xo=0.12, zo=2):
    """Vertical arrow y0→y1 at x."""
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=14),
                zorder=zo)
    if lbl:
        ax.text(x + xo, (y0 + y1) / 2, lbl, ha='left', va='center',
                fontsize=7.5, color=c, fontstyle='italic', zorder=zo + 1)


def elbow(ax, pts, c='#475569', lw=1.8, zo=2):
    """Polyline ending with an arrowhead at pts[-1]."""
    for i in range(len(pts) - 2):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        ax.plot([x0, x1], [y0, y1], color=c, lw=lw,
                solid_capstyle='round', solid_joinstyle='round', zorder=zo)
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=14),
                zorder=zo)


def save(fig, stem):
    for ext in ('svg', 'png'):
        p = os.path.join(OUT, f'{stem}.{ext}')
        fig.savefig(p, format=ext, bbox_inches='tight', dpi=150)
        print(f'  → {p}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: current_architecture_global
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_global():
    fig, ax = plt.subplots(figsize=(17, 8.5))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # ── Background route bands ───────────────────────────────────────────────
    band(ax, 3.0, 4.9, 10.0, 7.8, BGW, alpha=0.4, zo=0)
    band(ax, 3.0, 0.3, 10.0, 3.2, BGL, alpha=0.4, zo=0)
    ax.text(6.5, 7.65, 'WM / Dorsal Route', ha='center', va='bottom',
            fontsize=9, color=WM['ec'], fontweight='bold')
    ax.text(6.5, 0.45, 'LTM / Ventral Route', ha='center', va='bottom',
            fontsize=9, color=LTM['ec'], fontweight='bold')

    # ── Input ────────────────────────────────────────────────────────────────
    rbox(ax, 0.85, 4.0, 1.3, 0.9, 'x₁:T\nphoneme IDs', IO, fs=8)

    # ── Shared Embedding ─────────────────────────────────────────────────────
    rbox(ax, 2.45, 4.0, 1.6, 0.9, 'Shared\nEmbedding\nE_embed', IO, fs=8)
    harr(ax, 1.5, 4.0, 1.65, c=IO['ec'])

    # Embedding fans to WM (y=6.1) and LTM (y=1.9)
    # WM branch:
    elbow(ax, [(3.25, 4.0), (3.55, 4.0), (3.55, 6.1), (3.85, 6.1)], c=WM['ec'])
    # LTM branch:
    elbow(ax, [(3.25, 4.0), (3.55, 4.0), (3.55, 1.9), (3.85, 1.9)], c=LTM['ec'])

    # ── WM route (y = 6.1) ───────────────────────────────────────────────────
    yw = 6.1
    rbox(ax, 5.15, yw, 2.2, 0.85, 'WM GRU Encoder\n(pack_padded_sequence)', WM, fs=8)
    harr(ax, 3.85, yw, 4.05, c=WM['ec'])          # arrow to encoder left

    rbox(ax, 7.25, yw, 1.8, 0.85, '+ε ~ N(0,σ²I)\nh̃_WM', NSE, fs=8)
    harr(ax, 6.25, yw, 6.35, c=WM['ec'], lbl='h_WM', yo=0.14)

    rbox(ax, 9.0, yw, 1.7, 0.85, 'WM GRU\nDecoder', WM, fs=8)
    harr(ax, 8.15, yw, 8.15, c=WM['ec'])

    # z_WM exit → elbow to gate
    elbow(ax, [(9.85, yw), (10.3, yw), (10.3, 4.45)], c=WM['ec'])
    ax.text(10.05, yw + 0.14, 'z_WM,t', ha='center', va='bottom',
            fontsize=7.5, color=WM['ec'], fontstyle='italic')

    # ── LTM route (y = 1.9) ──────────────────────────────────────────────────
    yl = 1.9
    rbox(ax, 5.15, yl, 2.2, 0.85, 'biGRU Encoder\n(no pack_padded)', LTM, fs=8)
    harr(ax, 3.85, yl, 4.05, c=LTM['ec'])

    rbox(ax, 7.3, yl, 1.9, 0.85, 'MLP\ns_hat ∈ R³⁰⁰', LTM, fs=8)
    harr(ax, 6.25, yl, 6.35, c=LTM['ec'], lbl='masked mean\npool → pooled', yo=0.14)

    # Semantic bank below s_hat
    rbox(ax, 7.3, 0.65, 1.9, 0.72,
         'B_lex  (frozen GloVe)\nc_LTM = max cosine', LTM, fs=7.8)
    varr(ax, 7.3, yl - 0.425, 0.65 + 0.36, c=LTM['ec'], lbl='s_hat', xo=0.14)

    rbox(ax, 9.0, yl, 1.7, 0.85, 'LTM GRU\nDecoder', LTM, fs=8)
    harr(ax, 8.25, yl, 8.15, c=LTM['ec'])

    # z_LTM exit → elbow to gate
    elbow(ax, [(9.85, yl), (10.3, yl), (10.3, 3.55)], c=LTM['ec'])
    ax.text(10.05, yl + 0.14, 'z_LTM,t', ha='center', va='bottom',
            fontsize=7.5, color=LTM['ec'], fontstyle='italic')

    # c_LTM → gate (bottom route via lower path)
    elbow(ax, [(7.3, 0.29), (7.3, 0.1), (11.6, 0.1), (11.6, 3.55)], c=GATE['ec'])
    ax.text(9.45, 0.2, 'c_LTM  (LTM confidence only → gate input)', ha='center',
            va='bottom', fontsize=7.5, color=GATE['ec'], fontstyle='italic')

    # ── Gate ─────────────────────────────────────────────────────────────────
    rbox(ax, 11.6, 4.0, 2.3, 0.9,
         'Gate\ng = σ(4 · (c_LTM − 0.5))', GATE, fs=8.5)
    # arrows from WM/LTM elbow ends arrive at gate left
    harr(ax, 10.3, 4.0, 10.45, c='#6D28D9', lw=1.5)

    # ── Premotor Blend ────────────────────────────────────────────────────────
    rbox(ax, 13.7, 4.0, 2.2, 0.9,
         'z_full = g · z_LTM\n+ (1−g) · z_WM', BLD, fs=8.5)
    harr(ax, 12.75, 4.0, 12.6, c=GATE['ec'], lbl='g', yo=0.14)

    # ── Shared Motor Cortex ───────────────────────────────────────────────────
    rbox(ax, 15.75, 4.0, 1.7, 0.9,
         'Motor Cortex\nW_motor  (shared)', MTR, fs=8.5)
    harr(ax, 14.8, 4.0, 14.9, c=BLD['ec'])

    # Output
    ax.annotate('', xy=(16.85, 4.0), xytext=(16.6, 4.0),
                arrowprops=dict(arrowstyle='->', color=MTR['ec'], lw=1.8,
                                mutation_scale=14), zorder=2)
    ax.text(16.88, 4.0, 'softmax\nŷ₁:S', ha='left', va='center',
            fontsize=8.5, color=MTR['tc'], fontweight='bold')

    # Also note shared W_motor for all routes
    ax.text(15.75, 3.2, 'Same W_motor for WM, LTM, and full routes',
            ha='center', va='top', fontsize=7, color=MTR['tc'], fontstyle='italic')

    # ── Note at bottom right ──────────────────────────────────────────────────
    notebox(ax, 14.0, 1.0, 5.0, 1.2,
            'Current implemented checkpoint\nnot a proposed revision\n'
            '(checkpoint: lichtheim3_30k_glove_e60_to_e120_lowlr.pt)',
            fs=7)

    ax.set_title('Lichtheim3 — Current Implemented Architecture',
                 fontsize=11, fontweight='bold', pad=8, color='#1E293B')
    save(fig, 'current_architecture_global')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: gate_premotor_blend
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_gate():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # ── Inputs ───────────────────────────────────────────────────────────────
    rbox(ax, 1.2, 5.5, 1.8, 0.85, 'z_WM,t\n(WM premotor)', WM, fs=8.5)
    rbox(ax, 1.2, 2.5, 1.8, 0.85, 'z_LTM,t\n(LTM premotor)', LTM, fs=8.5)
    rbox(ax, 1.2, 1.0, 2.0, 0.8, 'c_LTM\n(max cosine sim)', LTM, fs=8.5)

    # ── Gate ─────────────────────────────────────────────────────────────────
    rbox(ax, 4.8, 1.0, 3.0, 1.0,
         'g = σ(4 · (c_LTM − 0.5))\n∈ (0, 1)   scalar per item', GATE, fs=9)
    harr(ax, 2.2, 1.0, 3.3, c=GATE['ec'], lbl='c_LTM only', yo=0.13)

    # Gate value flows up to blend
    elbow(ax, [(4.8, 1.5), (4.8, 4.0)], c=GATE['ec'], lw=1.8)
    ax.text(5.0, 2.7, 'g', ha='left', va='center',
            fontsize=10, color=GATE['ec'], fontstyle='italic', fontweight='bold')

    # WM and LTM premotor → blend
    harr(ax, 2.1, 5.5, 3.3, c=WM['ec'])
    elbow(ax, [(2.1, 5.5), (3.5, 5.5), (3.5, 4.45)], c=WM['ec'])
    harr(ax, 2.1, 2.5, 3.3, c=LTM['ec'])
    elbow(ax, [(2.1, 2.5), (3.5, 2.5), (3.5, 3.55)], c=LTM['ec'])

    # ── Premotor Blend ────────────────────────────────────────────────────────
    rbox(ax, 5.5, 4.0, 3.2, 1.0,
         'z_full,t = g · z_LTM,t\n+ (1−g) · z_WM,t', BLD, fs=9)

    # ── Motor Cortex ──────────────────────────────────────────────────────────
    rbox(ax, 9.2, 4.0, 2.4, 1.0,
         'W_motor  (shared)\nLogit projection', MTR, fs=9)
    harr(ax, 7.1, 4.0, 8.0, c=BLD['ec'], lbl='z_full,t', yo=0.13)

    # softmax output
    rbox(ax, 12.0, 4.0, 2.0, 0.9, 'softmax\np(yₜ | y<t, x)', IO, fs=9)
    harr(ax, 10.4, 4.0, 11.0, c=MTR['ec'], lbl='logits', yo=0.13)

    # ── Properties box ───────────────────────────────────────────────────────
    notebox(ax, 5.0, 6.4, 5.6, 1.8,
            'Current gate properties\n'
            '• scalar per item  (same g at every decoder step t)\n'
            '• depends only on LTM confidence  c_LTM\n'
            '• no WM reliability input\n'
            '• invariant to WM noise  σ\n'
            '• blends premotor states  z,  not logits\n'
            '• zero trainable parameters',
            fs=7.8)

    ax.set_title('Gate and Premotor Blend — Lichtheim3 Current Checkpoint',
                 fontsize=11, fontweight='bold', pad=8, color='#1E293B')
    save(fig, 'gate_premotor_blend')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: wm_noise_location
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_noise():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    y = 3.5   # main chain y-position

    # ── Main chain ────────────────────────────────────────────────────────────
    rbox(ax, 1.0, y, 1.5, 0.85, 'x₁:T\nphoneme IDs', IO, fs=8.5)
    rbox(ax, 3.0, y, 1.9, 0.85, 'Shared\nEmbedding\nE_embed', IO, fs=8)
    rbox(ax, 5.2, y, 2.0, 0.85, 'WM GRU Encoder\n(pack_padded)', WM, fs=8.5)
    rbox(ax, 7.8, y, 1.3, 0.85, 'h_WM', WM, fs=9)

    # Noise addition — visually prominent
    rbox(ax, 9.5, y, 1.4, 0.95, '+ ε', NSE, fs=14, lw=2.5)
    rbox(ax, 11.3, y, 1.5, 0.85, 'h̃_WM', WM, fs=9)

    ax.set_clip_on(False)
    # Noise source box
    rbox(ax, 9.5, 1.8, 2.4, 1.0,
         'ε ~ N(0, σ²I)\nσ: per-dim noise std\n(default σ = 0.10)', NSE, fs=8.5)
    varr(ax, 9.5, 2.3, y - 0.475, c=NSE['ec'], lw=2.2)

    # arrows in chain
    harr(ax, 1.75, y, 2.05, c=IO['ec'])
    harr(ax, 3.95, y, 4.2, c=WM['ec'])
    harr(ax, 6.2, y, 7.15, c=WM['ec'], lbl='h_WM', yo=0.14)
    harr(ax, 8.45, y, 8.8, c=WM['ec'])
    harr(ax, 10.2, y, 10.55, c=WM['ec'])

    # Arrow from h̃_WM to decoder (implied)
    ax.annotate('', xy=(12.4, y), xytext=(12.05, y),
                arrowprops=dict(arrowstyle='->', color=WM['ec'], lw=1.8,
                                mutation_scale=14), zorder=2)
    ax.text(12.45, y, 'WM\nDecoder', ha='left', va='center',
            fontsize=8.5, color=WM['tc'], fontweight='bold')

    # ── "Active when" label ───────────────────────────────────────────────────
    notebox(ax, 9.5, 4.8, 5.5, 0.7,
            'Noise active: model.training=True  OR  collect=True (WFE sweep)\n'
            'Noise off: ceiling eval, deterministic AR eval (collect=False)',
            fs=7.5)

    # ── "Not noised" box ─────────────────────────────────────────────────────
    notebox(ax, 3.8, 1.3, 6.0, 1.5,
            'Not noised:\n'
            '• phoneme IDs   • phoneme embeddings\n'
            '• LTM encoder, pooling, s_hat\n'
            '• semantic bank B_lex\n'
            '• gate g   • logits   • decoder steps',
            fs=7.8)

    ax.set_title('WM Noise Location — Applied to h_WM Only',
                 fontsize=11, fontweight='bold', pad=8, color='#1E293B')
    save(fig, 'wm_noise_location')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: tf_vs_ar_decoding
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_decoding():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Column divider
    ax.plot([7, 7], [0.4, 5.8], color='#CBD5E1', lw=1.5, ls='--', zorder=1)

    def step_row(ax, x_base, y, step_i, dec_in_txt, pred_txt, s_in, s_out, lbl_in, lbl_out):
        """Draw one decoder step row."""
        # decoder input box
        rbox(ax, x_base + 0.9, y, 1.5, 0.65, dec_in_txt, s_in, fs=8)
        # arrow → decoder
        harr(ax, x_base + 1.65, y, x_base + 2.05, c='#475569', lw=1.5)
        # decoder step box
        rbox(ax, x_base + 2.85, y, 1.6, 0.65, f'Decoder\nstep t={step_i}', IO, fs=8)
        # → output
        harr(ax, x_base + 3.65, y, x_base + 4.05, c='#475569', lw=1.5)
        rbox(ax, x_base + 4.85, y, 1.5, 0.65, pred_txt, s_out, fs=8)
        # labels
        ax.text(x_base + 0.9, y - 0.5, lbl_in, ha='center', va='top',
                fontsize=7, color=s_in['tc'], fontstyle='italic')

    # ── Left panel: Teacher-Forced ────────────────────────────────────────────
    ax.text(3.5, 5.6, 'Teacher-Forced Decoding', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(3.5, 5.1, 'ceiling / debug only', ha='center', va='center',
            fontsize=9, color='#64748B', fontstyle='italic')

    gold_steps = [
        (1, 'BOS', 'y₁ pred'),
        (2, 'gold y₁', 'y₂ pred'),
        (3, 'gold y₂', 'y₃ pred'),
    ]
    for i, (step, din, pred) in enumerate(gold_steps):
        yy = 4.2 - i * 1.05
        rbox(ax, 1.0, yy, 1.5, 0.65, din, MTR if din == 'BOS' else IO, fs=8)
        harr(ax, 1.75, yy, 1.95, c='#475569', lw=1.5)
        rbox(ax, 2.75, yy, 1.6, 0.65, f'Decoder  t={step}', IO, fs=8)
        harr(ax, 3.55, yy, 3.75, c='#475569', lw=1.5)
        rbox(ax, 4.65, yy, 1.5, 0.65, pred, MTR, fs=8)

        if din not in ('BOS',):
            ax.annotate('', xy=(1.0, yy + 0.325), xytext=(1.0, yy + 0.68),
                        arrowprops=dict(arrowstyle='->', color=MTR['ec'],
                                        lw=1.5, mutation_scale=12, linestyle='dashed'),
                        zorder=2)

    ax.text(1.0, 1.8,
            'Decoder input at each step t:\ngold token y_{t−1}  (not the model\'s output)',
            ha='center', va='top', fontsize=8, color='#334155',
            multialignment='center')
    ax.add_patch(FancyBboxPatch((0.1, 1.0), 6.4, 0.65,
                                boxstyle='round,pad=0.04',
                                fc='#F0FDF4', ec='#22C55E', lw=1.5, zorder=3))
    ax.text(3.3, 1.325, '✓  Errors cannot propagate   |   Used only as a ceiling/debug probe',
            ha='center', va='center', fontsize=8, color='#166534', fontweight='bold')

    # ── Right panel: Autoregressive ───────────────────────────────────────────
    ax.text(10.5, 5.6, 'Autoregressive Decoding', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#1E293B')
    ax.text(10.5, 5.1, 'behavioral evaluation — main regime',
            ha='center', va='center', fontsize=9, color='#C2410C',
            fontweight='bold', fontstyle='italic')

    ar_steps = [
        (1, 'BOS', 'ŷ₁'),
        (2, 'ŷ₁  ← model output', 'ŷ₂'),
        (3, 'ŷ₂  ← model output', 'ŷ₃'),
    ]
    for i, (step, din, pred) in enumerate(ar_steps):
        yy = 4.2 - i * 1.05
        rbox(ax, 8.0, yy, 1.5, 0.65, din, MTR if din == 'BOS' else LTM, fs=7.5)
        harr(ax, 8.75, yy, 8.95, c='#475569', lw=1.5)
        rbox(ax, 9.75, yy, 1.6, 0.65, f'Decoder  t={step}', IO, fs=8)
        harr(ax, 10.55, yy, 10.75, c='#475569', lw=1.5)
        rbox(ax, 11.65, yy, 1.5, 0.65, pred, MTR, fs=8)

        # feedback arrow (model output → next input)
        if i < 2:
            next_yy = yy - 1.05
            elbow(ax, [(11.65, yy - 0.325), (11.65, yy - 0.68),
                       (8.0, next_yy + 0.68), (8.0, next_yy + 0.325)],
                  c=NSE['ec'], lw=1.5)

    ax.text(10.0, 1.8,
            'Decoder input at each step t:\nmodel\'s own argmax prediction ŷ_{t−1}',
            ha='center', va='top', fontsize=8, color='#334155',
            multialignment='center')
    ax.add_patch(FancyBboxPatch((7.1, 1.0), 6.4, 0.65,
                                boxstyle='round,pad=0.04',
                                fc='#FFF7ED', ec='#F97316', lw=1.5, zorder=3))
    ax.text(10.3, 1.325,
            '⚠  Errors propagate step-to-step   |   Reports free-generation accuracy',
            ha='center', va='center', fontsize=8, color='#7C2D12', fontweight='bold')

    ax.set_title('Teacher-Forced vs Autoregressive Decoding',
                 fontsize=11, fontweight='bold', pad=8, color='#1E293B')
    save(fig, 'tf_vs_ar_decoding')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: current_vs_future_gate
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_gate_comparison():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Column divider
    ax.plot([6.5, 6.5], [0.3, 6.8], color='#CBD5E1', lw=1.5, ls='--', zorder=1)

    # ── Column headers ────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.2, 6.0), 5.9, 0.65,
                                boxstyle='round,pad=0.05',
                                fc=GATE['fc'], ec=GATE['ec'], lw=2.0, zorder=3))
    ax.text(3.15, 6.325, 'Current implemented gate', ha='center', va='center',
            fontsize=11, fontweight='bold', color=GATE['tc'])

    ax.add_patch(FancyBboxPatch((6.7, 6.0), 5.9, 0.65,
                                boxstyle='round,pad=0.05',
                                fc=PRP['fc'], ec=PRP['ec'], lw=2.0, zorder=3))
    ax.text(9.65, 6.325,
            'Proposed future gate  (not implemented)', ha='center', va='center',
            fontsize=10, fontweight='bold', color=PRP['tc'])

    # ── Current gate details ──────────────────────────────────────────────────
    items_cur = [
        (GATE, 'g = σ(4 · (c_LTM − 0.5))', ''),
        (GATE, 'Input: c_LTM  only', '(max cosine to semantic bank)'),
        (GATE, 'Scope: item-level scalar', 'same g at every decoder step t'),
        (NTE,  'No learnable parameters', ''),
        (NTE,  'No WM reliability input', ''),
        (NTE,  'Invariant to WM noise σ', ''),
        (NTE,  'In checkpoint — no retrain needed', ''),
    ]
    for i, (s, main, sub) in enumerate(items_cur):
        yy = 5.1 - i * 0.7
        ax.add_patch(FancyBboxPatch((0.3, yy - 0.22), 5.8, 0.44,
                                    boxstyle='round,pad=0.03',
                                    fc=s['fc'], ec=s['ec'], lw=1.2, zorder=3))
        txt = main + (f'\n{sub}' if sub else '')
        ax.text(3.2, yy, txt, ha='center', va='center',
                fontsize=8, color=s['tc'], multialignment='center',
                linespacing=1.25, fontweight='bold', zorder=4)

    # ── Proposed gate details ─────────────────────────────────────────────────
    items_prop = [
        (PRP, 'g_t = σ(β₀ + β_L·c_LTM − β_W·r_WM,t\n+ β_len·len(x) + β_σ·σ)', ''),
        (PRP, 'Input: c_LTM + WM reliability r_WM,t\n+ length + noise level', ''),
        (PRP, 'Scope: possibly step-level g_t', 'varies with t if r_WM is step-level'),
        (PRP, 'Trainable β parameters', '(scalars or small MLP)'),
        (PRP, 'WM noise awareness', 'β_σ term responds to σ'),
        (NTE, 'Proposed — not in current checkpoint', ''),
        (PRP, 'Requires retraining', ''),
    ]
    for i, (s, main, sub) in enumerate(items_prop):
        yy = 5.1 - i * 0.7
        ax.add_patch(FancyBboxPatch((6.8, yy - 0.22), 5.8, 0.44,
                                    boxstyle='round,pad=0.03',
                                    fc=s['fc'], ec=s['ec'], lw=1.2, zorder=3))
        txt = main + (f'\n{sub}' if sub else '')
        ax.text(9.7, yy, txt, ha='center', va='center',
                fontsize=8, color=s['tc'], multialignment='center',
                linespacing=1.25, fontweight='bold', zorder=4)

    # Bottom stamp
    notebox(ax, 9.65, 0.42, 5.8, 0.6,
            '⚠  Proposed — not implemented  |  requires retraining',
            s=PRP, fs=8)

    ax.set_title('Gate Design: Current Checkpoint vs Proposed Revision',
                 fontsize=11, fontweight='bold', pad=8, color='#1E293B')
    save(fig, 'current_vs_future_gate')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: lichtheim2_inspiration_vs_l3_current
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_l2_vs_l3():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Column divider
    ax.plot([6.5, 6.5], [0.3, 6.8], color='#CBD5E1', lw=1.5, ls='--', zorder=1)

    # ── Headers ───────────────────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.2, 6.1), 5.9, 0.65,
                                boxstyle='round,pad=0.05',
                                fc='#F0F9FF', ec='#7DD3FC', lw=2.0, zorder=3))
    ax.text(3.15, 6.425, 'Ueno / Lichtheim2  (inspiration)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#0C4A6E')

    ax.add_patch(FancyBboxPatch((6.7, 6.1), 5.9, 0.65,
                                boxstyle='round,pad=0.05',
                                fc=IO['fc'], ec=IO['ec'], lw=2.0, zorder=3))
    ax.text(9.65, 6.425, 'Lichtheim3  (current checkpoint)',
            ha='center', va='center', fontsize=10, fontweight='bold', color=IO['tc'])

    # ── Left: Ueno / L2 conceptual ───────────────────────────────────────────
    l2_items = [
        ('Dual-route architecture', '#0C4A6E'),
        ('Dorsal: sound → motor\n(phonological buffer)', WM['tc']),
        ('Ventral: sound → semantics\n(lexical/conceptual)', LTM['tc']),
        ('Tick-by-tick recurrence\n(copy-back from motor to buffer)', '#0C4A6E'),
        ('Semantic → phonological\nnaming pathway', LTM['tc']),
        ('Lesion analysis (route damage\n→ aphasic error pattern)', '#0C4A6E'),
    ]
    for i, (txt, col) in enumerate(l2_items):
        yy = 5.35 - i * 0.82
        ax.add_patch(FancyBboxPatch((0.3, yy - 0.27), 5.8, 0.54,
                                    boxstyle='round,pad=0.04',
                                    fc='#F0F9FF', ec='#7DD3FC', lw=1.2, zorder=3))
        ax.text(3.2, yy, txt, ha='center', va='center',
                fontsize=8.5, color=col, multialignment='center',
                linespacing=1.25, fontweight='bold', zorder=4)

    # ── Right: L3 current ─────────────────────────────────────────────────────
    l3_items = [
        ('Dual-route architecture', IO['tc']),
        ('WM / dorsal route\nGRU encoder-decoder (pack_padded)', WM['tc']),
        ('LTM / ventral route\nbiGRU + GloVe-aligned MLP + bank', LTM['tc']),
        ('No tick-by-tick copy-back\n(one-shot gate, not recurrent routing)', '#475569'),
        ('No separate naming pathway\n(LTM uses lexical anchoring, not naming)', '#475569'),
        ('Gate g from c_LTM only\n(passive LTM anchor, not full L2 gate)', GATE['tc']),
    ]
    for i, (txt, col) in enumerate(l3_items):
        yy = 5.35 - i * 0.82
        ax.add_patch(FancyBboxPatch((6.8, yy - 0.27), 5.8, 0.54,
                                    boxstyle='round,pad=0.04',
                                    fc=IO['fc'], ec=IO['ec'], lw=1.2, zorder=3))
        ax.text(9.7, yy, txt, ha='center', va='center',
                fontsize=8.5, color=col, multialignment='center',
                linespacing=1.25, fontweight='bold', zorder=4)

    # ── Bottom note ───────────────────────────────────────────────────────────
    notebox(ax, 6.5, 0.45, 12.5, 0.65,
            'The current checkpoint is not a faithful tick-by-tick Lichtheim2 '
            'implementation.\nIt is a modern dual-route prototype inspired by the '
            'dorsal/ventral division.',
            fs=7.8)

    ax.set_title('Ueno / Lichtheim2 Inspiration vs Lichtheim3 Current Checkpoint',
                 fontsize=11, fontweight='bold', pad=8, color='#1E293B')
    save(fig, 'lichtheim2_inspiration_vs_l3_current')


# ═══════════════════════════════════════════════════════════════════════════════
# README
# ═══════════════════════════════════════════════════════════════════════════════

README_CONTENT = """\
# Architecture Meeting Figures — Lichtheim3

Generated by `scripts/generate_architecture_figures.py`.
Source of truth: `docs/current_and_proposed_architecture_equations.md`.

---

## Figure index

### 1. `current_architecture_global.svg`

**Slide:** Opening architecture slide.
**Message:** The model has two routes — a WM/dorsal GRU route and an LTM/ventral
biGRU + GloVe-aligned route — that are combined by a parameter-free gate using
LTM confidence only, before a shared Motor Cortex projects to phoneme probabilities.
**Caveats:**
- This is the current implemented checkpoint only.
- No proposed revisions are shown here.
- The LTM encoder does not use `pack_padded_sequence`; the WM encoder does.

---

### 2. `gate_premotor_blend.svg`

**Slide:** Gate mechanism detail.
**Message:** The gate is a scalar computed from LTM confidence only. It blends
premotor states (z_WM and z_LTM) — not logits — before a single shared
W_motor projection. The gate has zero trainable parameters and is fully invariant
to WM noise.
**Caveats:**
- Current gate properties box must be verbally emphasised:
  no WM reliability signal, no step-level variation.

---

### 3. `wm_noise_location.svg`

**Slide:** Noise explanation.
**Message:** WM noise (ε ~ N(0,σ²I)) is applied to the WM encoder's final hidden
state h_WM only, producing h̃_WM. Nothing else is noised. Noise is active during
training (model.training=True) and during WFE sweeps (collect=True); it is off
for ceiling and deterministic AR eval (collect=False).
**Caveats:**
- The `--wm_noise` flag in the ceiling eval applies noise to the WM-isolated route
  only, not to WM inside the full/gated route.

---

### 4. `tf_vs_ar_decoding.svg`

**Slide:** Decoding regime.
**Message:** Teacher-forced (TF) decoding feeds gold tokens at each step —
errors cannot propagate. TF is used only as a ceiling/debug probe. Autoregressive
(AR) decoding feeds the model's own previous output — errors propagate. AR is the
behavioral evaluation regime for all figures.
**Caveats:**
- No numerical results are shown in this figure.
- Do not report TF exact-match as a behavioral result.

---

### 5. `current_vs_future_gate.svg`

**Slide:** Gate design discussion.
**Message:** The current gate is simple, parameter-free, and in the checkpoint —
no retrain needed. The proposed gate would accept WM reliability, length, and
noise level, and would require retraining. The proposed column is visually marked
as not implemented.
**Caveats:**
- Right column is conceptual only; signs of β terms are not validated.
- Do not present the proposed gate as the current model.

---

### 6. `lichtheim2_inspiration_vs_l3_current.svg`

**Slide:** Conceptual background / motivation.
**Message:** Lichtheim3 is inspired by the Ueno / Lichtheim2 dorsal/ventral
division, but is not a faithful tick-by-tick implementation. It lacks copy-back
recurrence and a separate naming pathway. The LTM route implements lexical
anchoring, not semantic comprehension.
**Caveats:**
- Do not equate current model results with Ueno/Lichtheim2 predictions.
- The note at the bottom of this figure should be read aloud on the slide.

---

## Global caveats

- All figures show the current implemented checkpoint
  (`lichtheim3_30k_glove_e60_to_e120_lowlr.pt`).
- Proposed/future items are visually distinguished (rose/pink border, "not
  implemented" stamp). Never show them as current results.
- Gate limitation: the current gate cannot respond to WM noise or item difficulty.
  Full/gated route noise robustness is structural (passive LTM anchor), not adaptive.
- TF vs AR: only AR figures belong in behavioral results. TF is ceiling only.
- Noise location: only h_WM is noised. LTM, gate, logits, and decoder steps are
  all noise-free.
"""


def write_readme():
    path = os.path.join(OUT, 'README.md')
    with open(path, 'w') as f:
        f.write(README_CONTENT)
    print(f'  → {path}')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating architecture meeting figures …')
    fig1_global()
    fig2_gate()
    fig3_noise()
    fig4_decoding()
    fig5_gate_comparison()
    fig6_l2_vs_l3()
    write_readme()
    print('Done.')
