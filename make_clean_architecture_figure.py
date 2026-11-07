# -*- coding:utf-8 -*-
"""
Flagship Publication-Grade Vector PDF Architecture & Hypersphere Figures (code_new)
===================================================================================
Designed in Nature Machine Intelligence / Briefings in Bioinformatics visual style:
- Minimal formula clutter: replaces text-heavy equation boxes with rich visual
  architectural primitives.
- 3D isometric tensor blocks (multi-channel feature cubes with top/side bevels).
- Iconic biomedical inputs: pharmacophore-highlighted 2D molecular graph + 3D-shaded
  secondary structure ribbons (alpha-helix, beta-sheet, coil) aligned with sequence tracks.
- Visual neural micro-diagrams inside Hybrid Encoder: receptive-field dilation grids,
  SE channel-attention intensity bars, bidirectional recurrent node chains, and
  multi-head self-attention connection webs.
- Rich Cross-Modal Interaction Bridge: Q-K-V cross-attention flow with a glowing
  Drug-Protein Substructural Affinity Matrix and visual Sigmoidal Gating (⊙).
- True 3D-shaded Unit Hypersphere (S^{d-1}) with latitude/longitude wireframe ellipses,
  compact geodesic caps, prototype vectors (W_1, W_0), and geodesic angular margin (+m).
"""

import os
import shutil
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
)

# TrueType vector fonts for lossless scaling in Overleaf / Adobe Illustrator
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["mathtext.fontset"] = "stixsans"

OUT_DIR = Path("sci_plots_comparison")
OUT_DIR.mkdir(exist_ok=True)
PAPER_NEW_DIR = Path(__file__).resolve().parents[1] / "paper_new"

PALETTE = {
    "ink": "#1E293B",
    "sub_ink": "#475569",
    "muted": "#64748B",
    "stage_edge": "#C8D2DF",
    # Stage 1 Header (Soft Slate Blue, matching reference image Stage 1)
    "s1_bg": "#FFFFFF",
    "s1_hdr_bg": "#EEF3FA",
    "s1_hdr_line": "#D5E0EE",
    "s1_hdr_text": "#284870",
    # Stage 2 Header (Soft Warm Cream/Amber, matching reference image Stage 2)
    "s2_bg": "#FFFFFF",
    "s2_hdr_bg": "#FDF4E3",
    "s2_hdr_line": "#E8D6B5",
    "s2_hdr_text": "#724614",
    # Stage 3 Header (Soft Lavender/Indigo, matching reference image Stage 3)
    "s3_bg": "#FFFFFF",
    "s3_hdr_bg": "#F0EEFA",
    "s3_hdr_line": "#DAD5F0",
    "s3_hdr_text": "#372C75",
    # Stage 4 Header (Soft Sage/Teal-Mint, harmonious 4th pastel stage)
    "s4_bg": "#FFFFFF",
    "s4_hdr_bg": "#EAF6F2",
    "s4_hdr_line": "#C7E2D8",
    "s4_hdr_text": "#1A5648",
    # Inner Stream / Card Palettes (Soft pastel borders & fills matching reference boxes)
    "drug_main": "#4A78B8",
    "drug_edge": "#94B4DA",
    "drug_dark": "#264773",
    "drug_light": "#E6EFFA",
    "drug_pale": "#F7FAFE",
    "prot_main": "#388E72",
    "prot_edge": "#88C4AE",
    "prot_dark": "#1D5643",
    "prot_light": "#E3F5EE",
    "prot_pale": "#F5FBF8",
    "amber_edge": "#E2B27E",
    "amber_pale": "#FFFBF5",
    "amber_light": "#FDF2E2",
    "amber_dark": "#754614",
    "attn_main": "#C87A38",
    "attn_light": "#FDF3E7",
    "gate_main": "#C48434",
    "gate_light": "#FDF6EA",
    "s4_edge": "#A89CD8",
    "s4_hdr": "#372C75",
    "s4_light": "#F2EFFB",
    "s4_pale": "#FAF9FE",
}


def draw_card(
    ax,
    x,
    y,
    w,
    h,
    fc="#FFFFFF",
    ec="#CBD5E1",
    lw=1.15,
    radius=0.10,
    ls="-",
    alpha=1.0,
    zorder=2,
    shadow=False,
):
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.022, y - 0.026),
            w,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            linewidth=0,
            facecolor="#64748B",
            alpha=0.10,
            zorder=zorder - 0.5,
        )
        ax.add_patch(sh)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=ls,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_header_card(
    ax,
    x,
    y,
    w,
    h,
    title,
    hdr_fc,
    hdr_text_color,
    divider_color,
    body_fc="#FFFFFF",
    ec="#C8D2DF",
    hdr_h=0.34,
    radius=0.10,
    lw=1.15,
    fontsize=8.6,
    zorder=2,
    shadow=False,
):
    """
    Draws a rounded container with a soft pastel top header strip, flat horizontal
    bottom divider line on the header, and clean rounded outer border — matching
    the exact box style of the user's reference image.
    """
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.025, y - 0.030),
            w,
            h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            linewidth=0,
            facecolor="#64748B",
            alpha=0.10,
            zorder=zorder - 0.5,
        )
        ax.add_patch(sh)

    # 1. Base white/tinted rounded body
    base = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=0,
        edgecolor="none",
        facecolor=body_fc,
        zorder=zorder,
    )
    ax.add_patch(base)

    # 2. Top rounded header fill + flat bottom filler so bottom of header is straight
    hdr_round = FancyBboxPatch(
        (x, y + h - hdr_h),
        w,
        hdr_h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=0,
        edgecolor="none",
        facecolor=hdr_fc,
        zorder=zorder + 0.1,
    )
    ax.add_patch(hdr_round)
    hdr_flat = Rectangle(
        (x, y + h - hdr_h),
        w,
        min(radius, hdr_h * 0.6),
        linewidth=0,
        edgecolor="none",
        facecolor=hdr_fc,
        zorder=zorder + 0.2,
    )
    ax.add_patch(hdr_flat)

    # 3. Thin horizontal divider line at bottom of header
    ax.plot(
        [x + 0.01, x + w - 0.01],
        [y + h - hdr_h, y + h - hdr_h],
        color=divider_color,
        lw=0.95,
        zorder=zorder + 0.3,
    )

    # 4. Crisp outer rounded border drawn on top
    border = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor="none",
        zorder=zorder + 0.4,
    )
    ax.add_patch(border)

    # 5. Centered single-line bold title
    ax.text(
        x + w / 2,
        y + h - hdr_h / 2,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=hdr_text_color,
        zorder=zorder + 0.5,
    )
    return border


def draw_stage_panel(ax, x, y, w, h, stage_num, title, subtitle, bg_color, hdr_bg, hdr_line, hdr_text_color):
    full_title = f"Stage {stage_num}  |  {title}"
    draw_header_card(
        ax,
        x,
        y,
        w,
        h,
        title=full_title,
        hdr_fc=hdr_bg,
        hdr_text_color=hdr_text_color,
        divider_color=hdr_line,
        body_fc=bg_color,
        ec=PALETTE["stage_edge"],
        hdr_h=0.54,
        radius=0.16,
        lw=1.25,
        fontsize=11.0,
        zorder=1,
        shadow=True,
    )


def draw_arrow(ax, p1, p2, color="#0F172A", lw=1.5, ms=12, rad=0.0, ls="-", zorder=6):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def draw_ortho_arrow(ax, points, color="#0F172A", lw=1.6, ms=12, ls="-", zorder=6):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(
            xs[:-1],
            ys[:-1],
            color=color,
            lw=lw,
            ls=ls,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )
    draw_arrow(ax, points[-2], points[-1], color=color, lw=lw, ms=ms, ls=ls, zorder=zorder)


def draw_op_node(ax, cx, cy, symbol="+", r=0.115, fc="#FFFFFF", ec="#0F172A", text_color="#0F172A", fontsize=9.5, zorder=7):
    c = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, lw=1.3, zorder=zorder)
    ax.add_patch(c)
    ax.text(cx, cy, symbol, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=text_color, zorder=zorder + 1)


def draw_3d_tensor_cube(ax, cx, cy, w=1.45, h=0.36, depth=0.13, n_slices=6, face_colors=None, top_color="#DBEAFE", side_color="#1E40AF", ec="#1E293B", zorder=4):
    """
    Draws an isometric 3D multi-channel tensor block (front slices + 3D top bevel + 3D right depth face).
    Makes latent representations look like true 3D neural tensors rather than flat 2D boxes.
    """
    x0 = cx - (w + depth) / 2
    y0 = cy - (h + depth * 0.7) / 2
    dx = depth
    dy = depth * 0.72

    if isinstance(face_colors, str):
        face_colors = [face_colors] * n_slices

    slice_w = w / n_slices
    # Front slices
    for i in range(n_slices):
        sx = x0 + i * slice_w
        fc = face_colors[i % len(face_colors)]
        rect = Rectangle((sx, y0), slice_w, h, facecolor=fc, edgecolor=ec, lw=0.75, zorder=zorder)
        ax.add_patch(rect)
        # Top bevel per slice
        top_poly = Polygon(
            [(sx, y0 + h), (sx + slice_w, y0 + h), (sx + slice_w + dx, y0 + h + dy), (sx + dx, y0 + h + dy)],
            closed=True,
            facecolor=top_color,
            edgecolor=ec,
            lw=0.65,
            zorder=zorder,
        )
        ax.add_patch(top_poly)

    # Right 3D depth face
    side_poly = Polygon(
        [(x0 + w, y0), (x0 + w + dx, y0 + dy), (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)],
        closed=True,
        facecolor=side_color,
        edgecolor=ec,
        lw=0.75,
        zorder=zorder,
    )
    ax.add_patch(side_poly)


def draw_molecule_pharmacophore(ax, cx, cy, scale=0.80):
    """Draws an iconic 2D drug molecular graph with glowing pharmacophore halos."""
    r = 0.27 * scale
    ang = np.deg2rad(np.arange(0, 360, 60) + 30)
    c1x, c1y = cx - 1.06 * scale, cy + 0.05 * scale

    # Soft pharmacophore highlight halos behind aromatic ring & amide group
    ax.add_patch(Circle((c1x, c1y), r * 1.38, facecolor="#93C5FD", alpha=0.32, edgecolor="none", zorder=3))

    ring1 = np.column_stack([c1x + r * np.cos(ang), c1y + r * np.sin(ang)])
    ax.add_patch(Polygon(ring1, closed=True, facecolor="#EFF6FF", edgecolor="#1E293B", lw=1.4, zorder=4))
    for i in (0, 2, 4):
        p_a = ring1[i] * 0.75 + np.array([c1x, c1y]) * 0.25
        p_b = ring1[(i + 1) % 6] * 0.75 + np.array([c1x, c1y]) * 0.25
        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color="#1D4ED8", lw=1.2, zorder=4)

    v_right = ring1[5]
    p1 = (v_right[0] + 0.24 * scale, v_right[1] - 0.10 * scale)
    p2 = (p1[0] + 0.24 * scale, p1[1] + 0.15 * scale)
    p3 = (p2[0] + 0.24 * scale, p2[1] - 0.15 * scale)
    p4 = (p3[0] + 0.24 * scale, p3[1] + 0.15 * scale)
    p5 = (p4[0] + 0.24 * scale, p4[1] - 0.12 * scale)

    # Pharmacophore halo on H-bond donor/acceptor amide region
    ax.add_patch(Ellipse(((p2[0] + p4[0]) / 2, p4[1] + 0.08 * scale), 0.72 * scale, 0.52 * scale, facecolor="#FDE68A", alpha=0.45, edgecolor="none", zorder=3))

    ax.plot(
        [v_right[0], p1[0], p2[0], p3[0], p4[0], p5[0]],
        [v_right[1], p1[1], p2[1], p3[1], p4[1], p5[1]],
        color="#1E293B",
        lw=1.4,
        zorder=4,
    )
    ax.plot([p2[0], p2[0]], [p2[1] + 0.08 * scale, p2[1] + 0.25 * scale], color="#1D4ED8", lw=1.4, zorder=4)
    ax.add_patch(Circle(p2, 0.085 * scale, facecolor="#2563EB", edgecolor="white", lw=0.8, zorder=5))
    ax.text(p2[0], p2[1], "N", ha="center", va="center", fontsize=6.5, fontweight="bold", color="white", zorder=6)

    ax.plot([p4[0] - 0.03 * scale, p4[0] - 0.03 * scale], [p4[1], p4[1] + 0.22 * scale], color="#DC2626", lw=1.3, zorder=4)
    ax.plot([p4[0] + 0.03 * scale, p4[0] + 0.03 * scale], [p4[1], p4[1] + 0.22 * scale], color="#DC2626", lw=1.3, zorder=4)
    ax.add_patch(Circle((p4[0], p4[1] + 0.28 * scale), 0.085 * scale, facecolor="#DC2626", edgecolor="white", lw=0.8, zorder=5))
    ax.text(p4[0], p4[1] + 0.28 * scale, "O", ha="center", va="center", fontsize=6.5, fontweight="bold", color="white", zorder=6)

    c2x = p5[0] + r * np.cos(np.deg2rad(30))
    c2y = p5[1] - r * np.sin(np.deg2rad(30))
    ax.add_patch(Circle((c2x, c2y), r * 1.35, facecolor="#A7F3D0", alpha=0.35, edgecolor="none", zorder=3))
    ring2 = np.column_stack([c2x + r * np.cos(ang), c2y + r * np.sin(ang)])
    ax.add_patch(Polygon(ring2, closed=True, facecolor="#ECFDF5", edgecolor="#1E293B", lw=1.4, zorder=4))
    for i in (1, 3, 5):
        p_a = ring2[i] * 0.75 + np.array([c2x, c2y]) * 0.25
        p_b = ring2[(i + 1) % 6] * 0.75 + np.array([c2x, c2y]) * 0.25
        ax.plot([p_a[0], p_b[0]], [p_a[1], p_b[1]], color="#059669", lw=1.2, zorder=4)

    v2_right = ring2[0]
    t1 = (v2_right[0] + 0.22 * scale, v2_right[1] + 0.11 * scale)
    t2 = (t1[0] + 0.22 * scale, t1[1] - 0.11 * scale)
    ax.plot([v2_right[0], t1[0], t2[0]], [v2_right[1], t1[1], t2[1]], color="#1E293B", lw=1.4, zorder=4)
    ax.add_patch(Circle(t2, 0.085 * scale, facecolor="#2563EB", edgecolor="white", lw=0.8, zorder=5))
    ax.text(t2[0], t2[1], "N", ha="center", va="center", fontsize=6.5, fontweight="bold", color="white", zorder=6)


def draw_ss_folding_pocket(ax, cx, cy):
    """Draws a vivid 3D-shaded protein secondary-structure pocket (Alpha-Helix + Beta-Sheet + Loop)."""
    # Soft pocket cavity halo
    ax.add_patch(Ellipse((cx, cy), 1.38, 0.56, facecolor="#D1FAE5", alpha=0.45, edgecolor="none", zorder=3))

    # 1. 3D Alpha-Helix ribbon on left (back-turns + front-turns for 3D depth)
    s = np.linspace(0, 1, 220)
    theta = 5.5 * np.pi * s
    hx = (cx - 0.56) + 0.36 * s + 0.08 * np.sin(theta) * 0.6
    hy = (cy + 0.17) - 0.32 * s + 0.09 * np.sin(theta) * 0.8
    ax.plot(hx, hy - 0.02, color="#065F46", lw=3.0, solid_capstyle="round", zorder=4)
    ax.plot(hx, hy, color="#10B981", lw=2.2, solid_capstyle="round", zorder=5)
    ax.plot(hx, hy + 0.015, color="#A7F3D0", lw=0.9, solid_capstyle="round", zorder=6)

    # 2. Dual Antiparallel Beta-Sheet Pleated Arrows in center
    for offset_x, offset_y, dx, dy in [(-0.08, -0.15, 0.22, 0.29), (0.08, 0.15, -0.22, -0.29)]:
        ax.annotate(
            "",
            xy=(cx + offset_x + dx, cy + offset_y + dy),
            xytext=(cx + offset_x, cy + offset_y),
            arrowprops=dict(
                arrowstyle="simple,head_width=0.58,head_length=0.50,tail_width=0.28",
                facecolor="#F59E0B",
                edgecolor="#92400E",
                lw=0.85,
            ),
            zorder=5,
        )

    # 3. Flexible Coil / Hinge Loop on right
    sc = np.linspace(0, 1, 120)
    cx_w = (cx + 0.30) + 0.25 * sc + 0.06 * np.sin(3.2 * np.pi * sc)
    cy_w = (cy + 0.16) - 0.30 * sc + 0.06 * np.cos(2.5 * np.pi * sc)
    ax.plot(cx_w, cy_w, color="#2563EB", lw=2.1, solid_capstyle="round", zorder=5)


def draw_hybrid_encoder_visual(ax, x, y, w, h, title, theme_main, theme_edge, theme_dark, theme_light, theme_pale):
    """
    Draws a visually rich Hybrid Encoder panel where each of the 4 stages has a custom
    neural architecture micro-schematic (dilation receptive field, SE channel bars,
    BiLSTM recurrent chain, and Multi-Head Self-Attention web) inside a soft pastel
    header card matching the reference style.
    """
    draw_header_card(
        ax,
        x,
        y,
        w,
        h,
        title=title,
        hdr_fc=theme_light,
        hdr_text_color=theme_dark,
        divider_color=theme_edge,
        body_fc=theme_pale,
        ec=theme_edge,
        hdr_h=0.35,
        radius=0.11,
        lw=1.15,
        fontsize=9.5,
        zorder=2,
        shadow=True,
    )

    bw = w - 0.66
    bx = x + 0.13
    rx = bx + bw + 0.27
    mid_x = bx + bw / 2

    b1_y, b1_h = y + 2.30, 0.72  # 1. Multi-Scale Dilated CNN
    b2_y, b2_h = y + 1.54, 0.54  # 2. SE Channel Recalibration
    b3_y, b3_h = y + 0.80, 0.52  # 3. BiLSTM Sequential Context
    b4_y, b4_h = y + 0.08, 0.50  # 4. Pre-Norm Transformer

    # --- Block 1: Multi-Scale Dilated CNN (with visual kernel grids) ---
    draw_card(ax, bx, b1_y, bw, b1_h, fc="#FFFFFF", ec=theme_edge, lw=1.05, radius=0.06, zorder=3)
    ax.text(mid_x, b1_y + 0.56, "Multi-Scale Dilated CNN", ha="center", va="center", fontsize=8.8, fontweight="bold", color=PALETTE["ink"], zorder=4)
    sub_w = (bw - 0.18) / 2
    # Left branch: k=3 dense kernel icon
    lx = bx + 0.05
    draw_card(ax, lx, b1_y + 0.07, sub_w, 0.36, fc=theme_light, ec=theme_edge, lw=0.8, radius=0.04, zorder=4)
    for k_i in range(3):
        ax.add_patch(Rectangle((lx + 0.06 + k_i * 0.10, b1_y + 0.14), 0.08, 0.18, facecolor=theme_main, edgecolor="white", lw=0.6, zorder=5))
    ax.text(lx + sub_w - 0.05, b1_y + 0.24, "$k=3$\nLocal", ha="right", va="center", fontsize=6.9, fontweight="bold", color=theme_dark, zorder=5)

    # Right branch: k=5, d=2 dilated kernel icon (alternating filled/hollow cells)
    rx_sub = bx + 0.13 + sub_w
    draw_card(ax, rx_sub, b1_y + 0.07, sub_w, 0.36, fc=theme_light, ec=theme_edge, lw=0.8, radius=0.04, zorder=4)
    for k_i in range(5):
        is_active = k_i % 2 == 0
        ax.add_patch(
            Rectangle(
                (rx_sub + 0.04 + k_i * 0.062, b1_y + 0.14),
                0.052,
                0.18,
                facecolor=theme_main if is_active else "#FFFFFF",
                edgecolor=theme_dark,
                lw=0.55,
                zorder=5,
            )
        )
    ax.text(rx_sub + sub_w - 0.04, b1_y + 0.24, "$k=5,d=2$\nDilated", ha="right", va="center", fontsize=6.7, fontweight="bold", color=theme_dark, zorder=5)

    # --- Block 2: SE Channel Recalibration (with channel weight bar visual) ---
    draw_card(ax, bx, b2_y, bw, b2_h, fc="#FFFFFF", ec=theme_edge, lw=1.05, radius=0.06, zorder=3)
    ax.text(bx + 0.10, b2_y + 0.36, "SE Channel Attention", ha="left", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["ink"], zorder=4)
    ax.text(bx + 0.10, b2_y + 0.14, "Dynamic channel weight", ha="left", va="center", fontsize=7.3, color=PALETTE["sub_ink"], zorder=4)
    # Visual channel attention bar graphic on right of Block 2
    bar_heights = [0.14, 0.32, 0.22, 0.38, 0.16, 0.34]
    bar_cols = [theme_light, theme_main, theme_light, "#E5A03A", theme_light, theme_dark]
    bar_x0 = bx + bw - 0.78
    for b_i, (bh, bc) in enumerate(zip(bar_heights, bar_cols)):
        ax.add_patch(Rectangle((bar_x0 + b_i * 0.11, b2_y + 0.08), 0.08, bh, facecolor=bc, edgecolor=theme_dark, lw=0.55, zorder=5))

    # --- Block 3: BiLSTM Sequential Context (with forward/backward state chain visual) ---
    draw_card(ax, bx, b3_y, bw, b3_h, fc="#FFFFFF", ec=theme_edge, lw=1.05, radius=0.06, zorder=3)
    ax.text(bx + 0.10, b3_y + 0.35, "BiLSTM Context", ha="left", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["ink"], zorder=4)
    ax.text(bx + 0.10, b3_y + 0.14, "Bidirectional sequence", ha="left", va="center", fontsize=7.3, color=PALETTE["sub_ink"], zorder=4)
    # Mini recurrent chain nodes on right of Block 3
    chain_xs = [bx + bw - 0.70, bx + bw - 0.44, bx + bw - 0.18]
    ax.plot([chain_xs[0], chain_xs[2]], [b3_y + 0.35, b3_y + 0.35], color=theme_main, lw=1.1, zorder=4)
    ax.plot([chain_xs[0], chain_xs[2]], [b3_y + 0.17, b3_y + 0.17], color=theme_dark, lw=1.1, ls="--", zorder=4)
    for cx_node in chain_xs:
        ax.add_patch(Circle((cx_node, b3_y + 0.35), 0.055, facecolor=theme_main, edgecolor="white", lw=0.6, zorder=5))
        ax.add_patch(Circle((cx_node, b3_y + 0.17), 0.055, facecolor=theme_light, edgecolor=theme_dark, lw=0.6, zorder=5))

    # --- Block 4: Pre-Norm Transformer (with Multi-Head Self-Attention web visual) ---
    draw_card(ax, bx, b4_y, bw, b4_h, fc="#FFFFFF", ec=theme_edge, lw=1.05, radius=0.06, zorder=3)
    ax.text(bx + 0.10, b4_y + 0.34, "Transformer Encoder", ha="left", va="center", fontsize=8.6, fontweight="bold", color=PALETTE["ink"], zorder=4)
    ax.text(bx + 0.10, b4_y + 0.13, "Multi-Head Self-Attn", ha="left", va="center", fontsize=7.3, color=PALETTE["sub_ink"], zorder=4)
    # Mini bipartite self-attention web on right of Block 4
    web_xs = [bx + bw - 0.68, bx + bw - 0.47, bx + bw - 0.26]
    for wx1 in web_xs:
        for wx2 in web_xs:
            ax.plot([wx1, wx2], [b4_y + 0.12, b4_y + 0.38], color=theme_main, lw=0.65, alpha=0.55, zorder=4)
    for wx in web_xs:
        ax.add_patch(Circle((wx, b4_y + 0.12), 0.04, facecolor=theme_dark, edgecolor="none", zorder=5))
        ax.add_patch(Circle((wx, b4_y + 0.38), 0.04, facecolor="#E5A03A", edgecolor="none", zorder=5))

    # Vertical sequential arrows
    draw_arrow(ax, (mid_x, b1_y), (mid_x, b2_y + b2_h), color=theme_dark, lw=1.2, ms=8.5)
    draw_arrow(ax, (mid_x, b2_y), (mid_x, b3_y + b3_h), color=theme_dark, lw=1.2, ms=8.5)
    draw_arrow(ax, (mid_x, b3_y), (mid_x, b4_y + b4_h), color=theme_dark, lw=1.2, ms=8.5)

    # Residual (+) skip loops on right
    for (src_y, dst_y) in [
        (b1_y + b1_h * 0.55, b2_y + b2_h * 0.55),
        (b2_y + b2_h * 0.24, b3_y + b3_h * 0.50),
    ]:
        plus_y = (src_y + dst_y) / 2
        ax.plot([bx + bw, rx, rx], [src_y, src_y, dst_y], color=theme_main, lw=1.1, ls="--", zorder=4)
        draw_arrow(ax, (rx, dst_y), (bx + bw, dst_y), color=theme_main, lw=1.1, ms=8, ls="--")
        draw_op_node(ax, rx, plus_y, symbol="+", r=0.095, fc="#FFFFFF", ec=theme_main, text_color=theme_dark, fontsize=8.8, zorder=6)

    return {
        "in_port": (bx, b1_y + b1_h * 0.50),
        "out_port": (bx + bw, b4_y + b4_h * 0.50),
    }


def draw_affinity_matrix_visual(ax, cx, cy, size=0.82, zorder=5):
    """Draws a glowing 6x6 Drug-Protein Substructural Affinity Heatmap with pocket hot-spots."""
    n = 6
    cell = size / n
    x0 = cx - size / 2
    y0 = cy - size / 2
    weights = np.array([
        [0.12, 0.25, 0.88, 0.30, 0.15, 0.10],
        [0.18, 0.78, 0.42, 0.15, 0.22, 0.18],
        [0.92, 0.35, 0.20, 0.84, 0.12, 0.25],
        [0.10, 0.22, 0.68, 0.95, 0.35, 0.15],
        [0.24, 0.15, 0.28, 0.38, 0.89, 0.42],
        [0.14, 0.62, 0.18, 0.22, 0.35, 0.82],
    ])
    cmap = plt.cm.YlOrRd
    for r in range(n):
        for c in range(n):
            val = weights[r, c]
            rect = Rectangle(
                (x0 + c * cell, y0 + (n - 1 - r) * cell),
                cell,
                cell,
                facecolor=cmap(0.12 + 0.82 * val),
                edgecolor="#FFFFFF",
                linewidth=0.7,
                zorder=zorder,
            )
            ax.add_patch(rect)
    border = Rectangle((x0, y0), size, size, fill=False, edgecolor="#9A3412", linewidth=1.2, zorder=zorder + 1)
    ax.add_patch(border)
    ax.text(cx, y0 + size + 0.10, "Attention Affinity", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#9A3412", zorder=zorder + 1)


def draw_3d_hypersphere(ax, cx, cy, r=1.18):
    """
    Draws a striking 3D-shaded Unit Hypersphere (S^{d-1}) with:
    - Layered radial shading & 3D equatorial/meridian wireframe ellipses
    - Compact Active (y=1) & Inactive (y=0) geodesic caps on the 3D surface
    - Prototype vectors W_1 and W_0 from the origin
    - Geodesic Angular Margin (+m) zone and Focal hard-example boundary points
    """
    # 1. Layered 3D sphere shading (concentric offset circles for specular highlight)
    ax.add_patch(Circle((cx + 0.03, cy - 0.04), r, facecolor="#94A3B8", alpha=0.22, edgecolor="none", zorder=3))
    ax.add_patch(Circle((cx, cy), r, facecolor="#F1F5F9", edgecolor="#0F172A", lw=1.6, zorder=4))
    ax.add_patch(Circle((cx - 0.12 * r, cy + 0.12 * r), r * 0.84, facecolor="#F8FAFC", edgecolor="none", alpha=0.75, zorder=4.1))
    ax.add_patch(Circle((cx - 0.24 * r, cy + 0.24 * r), r * 0.55, facecolor="#FFFFFF", edgecolor="none", alpha=0.85, zorder=4.2))

    # 2. 3D Wireframe Latitude (Equator) & Longitude (Meridian) Ellipses
    equator_back = Arc((cx, cy), 2 * r, 0.62 * r, angle=0, theta1=0, theta2=180, color="#94A3B8", lw=0.9, ls=":", zorder=4.3)
    equator_front = Arc((cx, cy), 2 * r, 0.62 * r, angle=0, theta1=180, theta2=360, color="#64748B", lw=1.1, ls="--", zorder=4.6)
    meridian = Ellipse((cx, cy), 0.68 * r, 2 * r, angle=18, fill=False, edgecolor="#94A3B8", lw=0.9, ls=":", zorder=4.3)
    ax.add_patch(equator_back)
    ax.add_patch(equator_front)
    ax.add_patch(meridian)

    # 3. Geodesic Angular Margin Zone (+m) between Active cluster edge (98 deg) and Decision Boundary (48 deg)
    w_margin = Wedge((cx, cy), r, 48, 98, facecolor="#FEF08A", alpha=0.72, edgecolor="#D97706", lw=1.1, ls="--", zorder=4.5)
    ax.add_patch(w_margin)

    # 4. Active Binding Cluster Geodesic Cap (100 deg to 164 deg)
    cap_active = Wedge((cx, cy), r * 1.03, 100, 164, width=0.18 * r, facecolor="#DDD6FE", edgecolor="#6D28D9", lw=1.25, zorder=5)
    ax.add_patch(cap_active)
    for deg, r_off in [(108, 0.94), (118, 0.96), (128, 0.93), (138, 0.96), (148, 0.94), (156, 0.95), (123, 0.88), (143, 0.88)]:
        ra = np.deg2rad(deg)
        px, py = cx + r * r_off * np.cos(ra), cy + r * r_off * np.sin(ra)
        ax.scatter([px], [py], c="#7C3AED", s=40, marker="o", edgecolors="#FFFFFF", lw=0.7, zorder=6.5)

    # Highlight 2 "Hard Positive" samples near margin boundary (Focal modulation focus)
    for deg in (102, 95):
        ra = np.deg2rad(deg)
        hx, hy = cx + r * 0.94 * np.cos(ra), cy + r * 0.94 * np.sin(ra)
        ax.scatter([hx], [hy], c="#DC2626", s=58, marker="*", edgecolors="#FFFFFF", lw=0.6, zorder=7)

    # 5. Inactive Cluster Geodesic Cap (-70 deg to -10 deg)
    cap_inactive = Wedge((cx, cy), r * 1.03, -70, -10, width=0.18 * r, facecolor="#A7F3D0", edgecolor="#047857", lw=1.25, zorder=5)
    ax.add_patch(cap_inactive)
    for deg, r_off in [(-62, 0.95), (-52, 0.93), (-42, 0.96), (-32, 0.94), (-22, 0.95), (-15, 0.93), (-47, 0.88), (-27, 0.88)]:
        ra = np.deg2rad(deg)
        nx, ny = cx + r * r_off * np.cos(ra), cy + r * r_off * np.sin(ra)
        ax.scatter([nx], [ny], c="#059669", s=40, marker="^", edgecolors="#FFFFFF", lw=0.7, zorder=6.5)

    # 6. Class Prototype Vectors W_1 and W_0 from sphere center
    for deg, col, lbl, off in [
        (132, "#5B21B6", r"$\hat{\mathbf{W}}_1$", (-0.25, 0.16)),
        (-40, "#065F46", r"$\hat{\mathbf{W}}_0$", (0.26, -0.14)),
    ]:
        ra = np.deg2rad(deg)
        wx, wy = cx + r * 1.02 * np.cos(ra), cy + r * 1.02 * np.sin(ra)
        draw_arrow(ax, (cx, cy), (wx, wy), color=col, lw=2.1, ms=11, zorder=7)
        ax.text(wx + off[0], wy + off[1], lbl, ha="center", va="center", fontsize=9.8, fontweight="bold", color=col, zorder=8)

    # 7. Decision Hyperplane Diameter (48 deg to -132 deg)
    for deg in (48, -132):
        ra = np.deg2rad(deg)
        ax.plot([cx, cx + r * np.cos(ra)], [cy, cy + r * np.sin(ra)], color="#334155", lw=1.25, ls="--", zorder=5.5)

    # Sphere center origin node
    ax.scatter([cx], [cy], c="#0F172A", s=28, zorder=8)

    # 8. Curved Geodesic Margin Arrow (+m) outside the rim
    arc_ang = np.deg2rad(np.linspace(50, 96, 40))
    ax.plot(cx + (r + 0.09) * np.cos(arc_ang), cy + (r + 0.09) * np.sin(arc_ang), color="#B45309", lw=2.2, zorder=7)
    ax.text(
        cx + (r + 0.23) * np.cos(np.deg2rad(73)),
        cy + (r + 0.23) * np.sin(np.deg2rad(73)),
        "$+m$",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#92400E",
        zorder=8,
    )
    ax.text(
        cx + 0.58 * r * np.cos(np.deg2rad(75)),
        cy + 0.58 * r * np.sin(np.deg2rad(75)),
        "Margin\nZone",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#92400E",
        zorder=8,
    )
    # Target angle arc theta_y
    theta_ang = np.deg2rad(np.linspace(98, 132, 25))
    ax.plot(cx + 0.42 * r * np.cos(theta_ang), cy + 0.42 * r * np.sin(theta_ang), color="#5B21B6", lw=1.4, zorder=7)
    ax.text(
        cx + 0.57 * r * np.cos(np.deg2rad(115)),
        cy + 0.57 * r * np.sin(np.deg2rad(115)),
        r"$\theta_y$",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#5B21B6",
        zorder=8,
    )


def generate_framework_figure():
    fig, ax = plt.subplots(figsize=(18.8, 8.6), dpi=300)
    ax.set_xlim(0.05, 18.75)
    ax.set_ylim(0.34, 8.88)
    ax.axis("off")

    # 4 Stage Columns (no top title banner, no bottom text line)
    sy, sh = 0.42, 8.38
    s1_x, s1_w = 0.20, 4.18
    s2_x, s2_w = 4.72, 3.92
    s3_x, s3_w = 8.98, 5.24
    s4_x, s4_w = 14.56, 4.04

    draw_stage_panel(
        ax, s1_x, sy, s1_w, sh, 1,
        "Biomolecular Input & Priors",
        "",
        PALETTE["s1_bg"], PALETTE["s1_hdr_bg"], PALETTE["s1_hdr_line"], PALETTE["s1_hdr_text"],
    )
    draw_stage_panel(
        ax, s2_x, sy, s2_w, sh, 2,
        "Multi-Scale Hybrid Backbone",
        "",
        PALETTE["s2_bg"], PALETTE["s2_hdr_bg"], PALETTE["s2_hdr_line"], PALETTE["s2_hdr_text"],
    )
    draw_stage_panel(
        ax, s3_x, sy, s3_w, sh, 3,
        "Cross-Attention & Gated Fusion",
        "",
        PALETTE["s3_bg"], PALETTE["s3_hdr_bg"], PALETTE["s3_hdr_line"], PALETTE["s3_hdr_text"],
    )
    draw_stage_panel(
        ax, s4_x, sy, s4_w, sh, 4,
        "Hyperspherical Margin Head",
        "",
        PALETTE["s4_bg"], PALETTE["s4_hdr_bg"], PALETTE["s4_hdr_line"], PALETTE["s4_hdr_text"],
    )

    # =========================================================================
    # STAGE 1: Biomolecular Input & 3D Embedding Tensors
    # =========================================================================
    d_x = s1_x + 0.22
    d_w = s1_w - 0.44
    d_cx = d_x + d_w / 2

    ax.text(d_x, 7.98, "(a) Drug Chemical Stream", ha="left", va="center", fontsize=10.2, fontweight="bold", color=PALETTE["drug_dark"], zorder=3)

    # 1. Drug Chemical Structure with Pharmacophore Halos
    draw_header_card(
        ax, d_x, 6.46, d_w, 1.34,
        title="Small-Molecule Ligand (SMILES)",
        hdr_fc=PALETTE["drug_light"], hdr_text_color=PALETTE["drug_dark"], divider_color=PALETTE["drug_edge"],
        body_fc="#FFFFFF", ec=PALETTE["drug_edge"], hdr_h=0.30, radius=0.09, lw=1.15, fontsize=9.2, shadow=True,
    )
    draw_molecule_pharmacophore(ax, d_cx - 0.06, 6.95, scale=0.78)

    # 2. Token Sequence Ribbon
    draw_card(ax, d_x, 5.46, d_w, 0.74, fc=PALETTE["drug_pale"], ec=PALETTE["drug_edge"], lw=1.1, radius=0.08)
    ax.text(d_cx, 6.01, "Chemical Substructure Tokens", ha="center", va="center", fontsize=9.0, fontweight="bold", color=PALETTE["drug_dark"], zorder=4)
    tokens = ["C", "1", "=", "N", "C", "C", "(=O)", "C"]
    tok_x = d_x + 0.16
    for tok in tokens:
        tw = 0.43 if len(tok) > 1 else 0.31
        draw_card(ax, tok_x, 5.55, tw, 0.27, fc="#FFFFFF", ec=PALETTE["drug_edge"], lw=0.85, radius=0.04, zorder=4)
        ax.text(tok_x + tw / 2, 5.685, tok, ha="center", va="center", fontsize=8.0, fontweight="bold", color=PALETTE["drug_dark"], zorder=5)
        tok_x += tw + 0.055
    ax.text(tok_x + 0.12, 5.685, "···", ha="center", va="center", fontsize=10.5, fontweight="bold", color=PALETTE["drug_dark"], zorder=5)

    # 3. 3D Drug Embedding Tensor Block
    d_emb_y, d_emb_h = 4.24, 0.96
    draw_card(ax, d_x, d_emb_y, d_w, d_emb_h, fc="#FFFFFF", ec=PALETTE["drug_edge"], lw=1.15, radius=0.09, shadow=True)
    ax.text(d_cx, d_emb_y + 0.76, "3D Drug Embedding Tensor", ha="center", va="center", fontsize=9.2, fontweight="bold", color=PALETTE["drug_dark"], zorder=4)
    draw_3d_tensor_cube(
        ax, d_cx - 0.45, d_emb_y + 0.35, w=1.68, h=0.32, depth=0.12, n_slices=7,
        face_colors=["#3B82F6", "#60A5FA", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#60A5FA"],
        top_color="#DBEAFE", side_color="#1E40AF", zorder=5,
    )
    ax.text(d_cx + 1.12, d_emb_y + 0.35, "$100 \\times 128$", ha="center", va="center", fontsize=8.8, fontweight="bold", color=PALETTE["drug_dark"], zorder=5)

    draw_arrow(ax, (d_cx, 6.46), (d_cx, 5.46 + 0.74), color=PALETTE["drug_main"], lw=1.35)
    draw_arrow(ax, (d_cx, 5.46), (d_cx, d_emb_y + d_emb_h), color=PALETTE["drug_main"], lw=1.35)

    # Divider
    ax.plot([s1_x + 0.16, s1_x + s1_w - 0.16], [4.02, 4.02], color="#CBD5E1", lw=1.0, ls="--", zorder=2)

    # (b) Structure-Informed Protein Stream
    ax.text(d_x, 3.78, "(b) Structure-Informed Protein Stream", ha="left", va="center", fontsize=10.2, fontweight="bold", color=PALETTE["prot_dark"], zorder=3)

    p_sub_w = (d_w - 0.14) / 2
    p_box1_x = d_x
    p_box2_x = d_x + p_sub_w + 0.14
    p_top_y, p_top_h = 2.22, 1.38

    # Left: Primary Sequence Track
    draw_header_card(
        ax, p_box1_x, p_top_y, p_sub_w, p_top_h,
        title="Primary Sequence",
        hdr_fc=PALETTE["prot_light"], hdr_text_color=PALETTE["prot_dark"], divider_color=PALETTE["prot_edge"],
        body_fc="#FFFFFF", ec=PALETTE["prot_edge"], hdr_h=0.29, radius=0.08, lw=1.1, fontsize=8.7, shadow=True,
    )
    ax.text(p_box1_x + p_sub_w / 2, p_top_y + 0.90, "Amino Acid Chain", ha="center", va="center", fontsize=7.6, color=PALETTE["sub_ink"], zorder=4)
    aa_lets = ["M", "A", "D", "K", "T", "G", "V"]
    aa_cols = ["#EF4444", "#10B981", "#F59E0B", "#8B5CF6", "#06B6D4", "#3B82F6", "#EC4899"]
    aa_x0 = p_box1_x + 0.12
    for idx_a, (al, ac) in enumerate(zip(aa_lets, aa_cols)):
        ax.add_patch(Rectangle((aa_x0 + idx_a * 0.22, p_top_y + 0.42), 0.19, 0.28, facecolor=ac, edgecolor="#1E293B", lw=0.65, zorder=5))
        ax.text(aa_x0 + idx_a * 0.22 + 0.095, p_top_y + 0.56, al, ha="center", va="center", fontsize=7.4, fontweight="bold", color="white", zorder=6)
    ax.text(p_box1_x + p_sub_w / 2, p_top_y + 0.17, "Residue Token Track", ha="center", va="center", fontsize=7.5, fontweight="bold", color=PALETTE["prot_dark"], zorder=4)

    # Right: NetSurfP-3.0 3-State Secondary Structure Folding Pocket
    draw_header_card(
        ax, p_box2_x, p_top_y, p_sub_w, p_top_h,
        title="Secondary Structure",
        hdr_fc=PALETTE["prot_light"], hdr_text_color=PALETTE["prot_dark"], divider_color=PALETTE["prot_edge"],
        body_fc="#FFFFFF", ec=PALETTE["prot_edge"], hdr_h=0.29, radius=0.08, lw=1.1, fontsize=8.7, shadow=True,
    )
    ax.text(p_box2_x + p_sub_w / 2, p_top_y + 0.90, "NetSurfP-3.0 Motifs", ha="center", va="center", fontsize=7.6, color=PALETTE["sub_ink"], zorder=4)
    draw_ss_folding_pocket(ax, p_box2_x + p_sub_w / 2, p_top_y + 0.52)
    ax.text(p_box2_x + p_sub_w / 2, p_top_y + 0.16, "Helix (H) · Sheet (E) · Coil (C)", ha="center", va="center", fontsize=7.1, fontweight="bold", color=PALETTE["prot_dark"], zorder=4)

    # Bottom: 3D Structure-Informed Protein Tensor Block
    p_emb_y, p_emb_h = 0.62, 1.02
    draw_card(ax, d_x, p_emb_y, d_w, p_emb_h, fc="#FFFFFF", ec=PALETTE["prot_edge"], lw=1.15, radius=0.09, shadow=True)
    ax.text(d_cx, p_emb_y + 0.80, "3D Structure-Informed Protein Tensor", ha="center", va="center", fontsize=9.1, fontweight="bold", color=PALETTE["prot_dark"], zorder=4)
    draw_3d_tensor_cube(
        ax, d_cx - 0.45, p_emb_y + 0.37, w=1.68, h=0.34, depth=0.12, n_slices=7,
        face_colors=["#10B981", "#34D399", "#6EE7B7", "#34D399", "#10B981", "#059669", "#34D399"],
        top_color="#D1FAE5", side_color="#065F46", zorder=5,
    )
    ax.text(d_cx + 1.12, p_emb_y + 0.37, "$1000 \\times 128$", ha="center", va="center", fontsize=8.8, fontweight="bold", color=PALETTE["prot_dark"], zorder=5)

    # Merge bracket with (+) node
    y_merge = 1.94
    ax.plot(
        [p_box1_x + p_sub_w / 2, p_box1_x + p_sub_w / 2, p_box2_x + p_sub_w / 2, p_box2_x + p_sub_w / 2],
        [p_top_y, y_merge, y_merge, p_top_y],
        color=PALETTE["prot_main"],
        lw=1.35,
        zorder=4,
    )
    draw_op_node(ax, d_cx, y_merge, symbol="+", r=0.115, fc=PALETTE["prot_pale"], ec=PALETTE["prot_main"], text_color=PALETTE["prot_dark"], fontsize=10.2, zorder=6)
    draw_arrow(ax, (d_cx, y_merge - 0.115), (d_cx, p_emb_y + p_emb_h), color=PALETTE["prot_main"], lw=1.35)

    # =========================================================================
    # STAGE 2: Multi-Scale Hybrid Backbone
    # =========================================================================
    enc_x = s2_x + 0.18
    enc_w = s2_w - 0.36
    drug_enc_ports = draw_hybrid_encoder_visual(
        ax, enc_x, 4.58, enc_w, 3.52,
        "Drug Multi-Scale Hybrid Encoder",
        theme_main=PALETTE["drug_main"],
        theme_edge=PALETTE["drug_edge"],
        theme_dark=PALETTE["drug_dark"],
        theme_light=PALETTE["drug_light"],
        theme_pale=PALETTE["drug_pale"],
    )
    prot_enc_ports = draw_hybrid_encoder_visual(
        ax, enc_x, 0.98, enc_w, 3.44,
        "Protein Multi-Scale Hybrid Encoder",
        theme_main=PALETTE["prot_main"],
        theme_edge=PALETTE["prot_edge"],
        theme_dark=PALETTE["prot_dark"],
        theme_light=PALETTE["prot_light"],
        theme_pale=PALETTE["prot_pale"],
    )

    # Bottom visual legend pill in Stage 2
    draw_card(ax, enc_x, 0.50, enc_w, 0.38, fc=PALETTE["amber_pale"], ec=PALETTE["amber_edge"], lw=1.0, radius=0.06, ls="--", zorder=3)
    ax.text(
        enc_x + enc_w / 2, 0.69,
        "Local Pharmacophore → Channel Recalibration → Global Context",
        ha="center", va="center", fontsize=7.4, fontweight="bold", color=PALETTE["amber_dark"], zorder=4,
    )

    # Stage 1 -> Stage 2 port-to-port routing
    ch12_d = (s1_x + s1_w + s2_x) / 2 - 0.04
    draw_ortho_arrow(
        ax,
        [(d_x + d_w, d_emb_y + d_emb_h / 2), (ch12_d, d_emb_y + d_emb_h / 2), (ch12_d, drug_enc_ports["in_port"][1]), drug_enc_ports["in_port"]],
        color=PALETTE["drug_main"], lw=1.55,
    )
    ch12_p = (s1_x + s1_w + s2_x) / 2 + 0.04
    draw_ortho_arrow(
        ax,
        [(d_x + d_w, p_emb_y + p_emb_h / 2), (ch12_p, p_emb_y + p_emb_h / 2), (ch12_p, prot_enc_ports["in_port"][1]), prot_enc_ports["in_port"]],
        color=PALETTE["prot_main"], lw=1.55,
    )

    # =========================================================================
    # STAGE 3: Bidirectional Cross-Attention & Dynamic Gated Fusion
    # =========================================================================
    s3_pad = 0.22
    col_w = (s3_w - 2 * s3_pad - 0.28) / 2
    left_x = s3_x + s3_pad
    right_x = left_x + col_w + 0.28
    left_cx = left_x + col_w / 2
    right_cx = right_x + col_w / 2
    s3_cx = s3_x + s3_w / 2

    # 1. Contextual 3D Feature Tensors (Top of Stage 3)
    top_y, top_h = 6.98, 0.90
    draw_card(ax, left_x, top_y, col_w, top_h, fc="#FFFFFF", ec=PALETTE["drug_edge"], lw=1.15, radius=0.08, shadow=True)
    ax.text(left_cx, top_y + 0.68, "Drug Contextual Tensor", ha="center", va="center", fontsize=8.9, fontweight="bold", color=PALETTE["drug_dark"], zorder=4)
    draw_3d_tensor_cube(
        ax, left_cx, top_y + 0.28, w=1.35, h=0.26, depth=0.10, n_slices=5,
        face_colors=["#3B82F6", "#60A5FA", "#93C5FD", "#60A5FA", "#2563EB"],
        top_color="#DBEAFE", side_color="#1E40AF", zorder=5,
    )

    draw_card(ax, right_x, top_y, col_w, top_h, fc="#FFFFFF", ec=PALETTE["prot_edge"], lw=1.15, radius=0.08, shadow=True)
    ax.text(right_cx, top_y + 0.68, "Protein Contextual Tensor", ha="center", va="center", fontsize=8.9, fontweight="bold", color=PALETTE["prot_dark"], zorder=4)
    draw_3d_tensor_cube(
        ax, right_cx, top_y + 0.28, w=1.35, h=0.26, depth=0.10, n_slices=5,
        face_colors=["#10B981", "#34D399", "#6EE7B7", "#34D399", "#059669"],
        top_color="#D1FAE5", side_color="#065F46", zorder=5,
    )

    # Stage 2 -> Stage 3 port-to-port routing
    ch23_d = (s2_x + s2_w + s3_x) / 2 - 0.05
    draw_ortho_arrow(
        ax,
        [drug_enc_ports["out_port"], (ch23_d, drug_enc_ports["out_port"][1]), (ch23_d, top_y + top_h * 0.50), (left_x, top_y + top_h * 0.50)],
        color=PALETTE["drug_main"], lw=1.55,
    )
    ch23_p = (s2_x + s2_w + s3_x) / 2 + 0.05
    draw_ortho_arrow(
        ax,
        [prot_enc_ports["out_port"], (ch23_p, prot_enc_ports["out_port"][1]), (ch23_p, 8.02), (right_cx, 8.02), (right_cx, top_y + top_h)],
        color=PALETTE["prot_main"], lw=1.55,
    )

    # 2. Bidirectional Cross-Attention Bridge (soft header card with Q-K-V badges & 6x6 Affinity Matrix)
    ca_y, ca_h = 5.00, 1.70
    draw_header_card(
        ax, left_x, ca_y, s3_w - 2 * s3_pad, ca_h,
        title="Bidirectional Multi-Head Cross-Attention Bridge",
        hdr_fc=PALETTE["amber_light"], hdr_text_color=PALETTE["amber_dark"], divider_color=PALETTE["amber_edge"],
        body_fc=PALETTE["amber_pale"], ec=PALETTE["amber_edge"], hdr_h=0.33, radius=0.10, lw=1.15, fontsize=9.6, shadow=True,
    )

    # Left Sub-card: Drug -> Protein Attention
    ca_sub_w = 1.64
    draw_card(ax, left_x + 0.12, ca_y + 0.13, ca_sub_w, 1.13, fc="#FFFFFF", ec=PALETTE["drug_edge"], lw=1.05, radius=0.07, zorder=4)
    ax.text(left_x + 0.12 + ca_sub_w / 2, ca_y + 1.07, "Drug → Protein", ha="center", va="center", fontsize=8.8, fontweight="bold", color=PALETTE["drug_dark"], zorder=5)
    ax.text(left_x + 0.12 + ca_sub_w / 2, ca_y + 0.84, "Substructure Query", ha="center", va="center", fontsize=7.4, color=PALETTE["sub_ink"], zorder=5)
    # Q_d, K_p, V_p pills
    for p_i, (plbl, pfc, pec, ptc) in enumerate([
        ("Q_d", PALETTE["drug_light"], PALETTE["drug_edge"], PALETTE["drug_dark"]),
        ("K_p", PALETTE["prot_light"], PALETTE["prot_edge"], PALETTE["prot_dark"]),
        ("V_p", PALETTE["prot_light"], PALETTE["prot_edge"], PALETTE["prot_dark"]),
    ]):
        px = left_x + 0.22 + p_i * 0.49
        draw_card(ax, px, ca_y + 0.24, 0.44, 0.37, fc=pfc, ec=pec, lw=0.85, radius=0.05, zorder=5)
        ax.text(px + 0.22, ca_y + 0.425, f"${plbl}$", ha="center", va="center", fontsize=8.3, fontweight="bold", color=ptc, zorder=6)

    # Center Substructural Affinity Matrix Heatmap
    draw_affinity_matrix_visual(ax, s3_cx, ca_y + 0.67, size=0.61, zorder=5)
    ax.text(s3_cx, ca_y + 0.20, "Pocket–Ligand Grid", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["amber_dark"], zorder=5)

    # Right Sub-card: Protein -> Drug Attention
    ca_r_x = right_x + col_w - 0.12 - ca_sub_w
    draw_card(ax, ca_r_x, ca_y + 0.13, ca_sub_w, 1.13, fc="#FFFFFF", ec=PALETTE["prot_edge"], lw=1.05, radius=0.07, zorder=4)
    ax.text(ca_r_x + ca_sub_w / 2, ca_y + 1.07, "Protein → Drug", ha="center", va="center", fontsize=8.8, fontweight="bold", color=PALETTE["prot_dark"], zorder=5)
    ax.text(ca_r_x + ca_sub_w / 2, ca_y + 0.84, "Pocket Residue Query", ha="center", va="center", fontsize=7.4, color=PALETTE["sub_ink"], zorder=5)
    for p_i, (plbl, pfc, pec, ptc) in enumerate([
        ("Q_p", PALETTE["prot_light"], PALETTE["prot_edge"], PALETTE["prot_dark"]),
        ("K_d", PALETTE["drug_light"], PALETTE["drug_edge"], PALETTE["drug_dark"]),
        ("V_d", PALETTE["drug_light"], PALETTE["drug_edge"], PALETTE["drug_dark"]),
    ]):
        px = ca_r_x + 0.10 + p_i * 0.49
        draw_card(ax, px, ca_y + 0.24, 0.44, 0.37, fc=pfc, ec=pec, lw=0.85, radius=0.05, zorder=5)
        ax.text(px + 0.22, ca_y + 0.425, f"${plbl}$", ha="center", va="center", fontsize=8.3, fontweight="bold", color=ptc, zorder=6)

    # Cross-modal exchange arrows
    draw_arrow(ax, (left_x + 0.12 + ca_sub_w, ca_y + 0.78), (s3_cx - 0.35, ca_y + 0.78), color=PALETTE["attn_main"], lw=1.25, ms=8.5)
    draw_arrow(ax, (s3_cx - 0.35, ca_y + 0.60), (left_x + 0.12 + ca_sub_w, ca_y + 0.60), color=PALETTE["attn_main"], lw=1.25, ms=8.5)
    draw_arrow(ax, (ca_r_x, ca_y + 0.78), (s3_cx + 0.35, ca_y + 0.78), color=PALETTE["attn_main"], lw=1.25, ms=8.5)
    draw_arrow(ax, (s3_cx + 0.35, ca_y + 0.60), (ca_r_x, ca_y + 0.60), color=PALETTE["attn_main"], lw=1.25, ms=8.5)

    draw_arrow(ax, (left_cx, top_y), (left_cx, ca_y + ca_h), color=PALETTE["drug_main"], lw=1.35)
    draw_arrow(ax, (right_cx, top_y), (right_cx, ca_y + ca_h), color=PALETTE["prot_main"], lw=1.35)

    # 3. Dual-Statistic (Mean || Max) Masked Pooling
    mp_y, mp_h = 3.72, 0.96
    draw_card(ax, left_x + 0.04, mp_y, col_w - 0.08, mp_h, fc="#FFFFFF", ec=PALETTE["drug_edge"], lw=1.1, radius=0.07, zorder=3)
    ax.text(left_cx, mp_y + 0.76, "Dual-Statistic Pooling", ha="center", va="center", fontsize=8.7, fontweight="bold", color=PALETTE["drug_dark"], zorder=4)
    draw_card(ax, left_x + 0.14, mp_y + 0.34, col_w - 0.28, 0.24, fc=PALETTE["drug_light"], ec=PALETTE["drug_edge"], lw=0.8, radius=0.04, zorder=4)
    ax.text(left_cx, mp_y + 0.46, "Mean Pool (Global Profile)", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["drug_dark"], zorder=5)
    draw_card(ax, left_x + 0.14, mp_y + 0.06, col_w - 0.28, 0.24, fc=PALETTE["drug_pale"], ec=PALETTE["drug_edge"], lw=0.8, radius=0.04, zorder=4)
    ax.text(left_cx, mp_y + 0.18, "Max Pool (Salient Motif)", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["drug_dark"], zorder=5)

    draw_card(ax, right_x + 0.04, mp_y, col_w - 0.08, mp_h, fc="#FFFFFF", ec=PALETTE["prot_edge"], lw=1.1, radius=0.07, zorder=3)
    ax.text(right_cx, mp_y + 0.76, "Masked Dual Pooling", ha="center", va="center", fontsize=8.7, fontweight="bold", color=PALETTE["prot_dark"], zorder=4)
    draw_card(ax, right_x + 0.14, mp_y + 0.34, col_w - 0.28, 0.24, fc=PALETTE["prot_light"], ec=PALETTE["prot_edge"], lw=0.8, radius=0.04, zorder=4)
    ax.text(right_cx, mp_y + 0.46, "Masked Mean (Domain)", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["prot_dark"], zorder=5)
    draw_card(ax, right_x + 0.14, mp_y + 0.06, col_w - 0.28, 0.24, fc=PALETTE["prot_pale"], ec=PALETTE["prot_edge"], lw=0.8, radius=0.04, zorder=4)
    ax.text(right_cx, mp_y + 0.18, "Masked Max (Pocket Peak)", ha="center", va="center", fontsize=7.3, fontweight="bold", color=PALETTE["prot_dark"], zorder=5)

    draw_arrow(ax, (left_cx, ca_y), (left_cx, mp_y + mp_h), color=PALETTE["drug_main"], lw=1.35)
    draw_arrow(ax, (right_cx, ca_y), (right_cx, mp_y + mp_h), color=PALETTE["prot_main"], lw=1.35)

    # 4. Joint Feature Concatenation (3D Multi-Modal Tensor)
    cat_w = 4.24
    cat_x = s3_cx - cat_w / 2
    cat_y, cat_h = 2.52, 0.84
    draw_card(ax, cat_x, cat_y, cat_w, cat_h, fc="#FFFFFF", ec=PALETTE["s4_edge"], lw=1.15, radius=0.08, shadow=True)
    ax.text(s3_cx, cat_y + 0.64, "Joint Multi-Modal Representation (512-D)", ha="center", va="center", fontsize=9.1, fontweight="bold", color=PALETTE["s3_hdr_text"], zorder=4)
    draw_3d_tensor_cube(
        ax, s3_cx, cat_y + 0.25, w=2.40, h=0.26, depth=0.10, n_slices=8,
        face_colors=["#3B82F6", "#60A5FA", "#93C5FD", "#60A5FA", "#34D399", "#10B981", "#6EE7B7", "#059669"],
        top_color="#EDE9FE", side_color="#4C1D95", zorder=5,
    )

    y_cat_merge = 3.52
    ax.plot([left_cx, left_cx, right_cx, right_cx], [mp_y, y_cat_merge, y_cat_merge, mp_y], color=PALETTE["ink"], lw=1.3, zorder=4)
    draw_arrow(ax, (s3_cx, y_cat_merge), (s3_cx, cat_y + cat_h), lw=1.3)

    # 5. Dynamic Sigmoidal Feature Gate (Soft pastel header card)
    gate_w = 4.44
    gate_x = s3_cx - gate_w / 2
    gate_y, gate_h = 0.52, 1.76
    draw_header_card(
        ax, gate_x, gate_y, gate_w, gate_h,
        title="Dynamic Sigmoidal Pocket Gate",
        hdr_fc=PALETTE["amber_light"], hdr_text_color=PALETTE["amber_dark"], divider_color=PALETTE["amber_edge"],
        body_fc=PALETTE["amber_pale"], ec=PALETTE["amber_edge"], hdr_h=0.31, radius=0.09, lw=1.15, fontsize=9.3, shadow=True,
    )

    draw_arrow(ax, (s3_cx, cat_y), (s3_cx, gate_y + gate_h), color=PALETTE["ink"], lw=1.3, ms=10)

    id_x = s3_cx - 1.15
    gt_x = s3_cx + 0.65
    split_y = gate_y + 1.34
    mul_y = gate_y + 0.74

    ax.plot([id_x, gt_x], [split_y, split_y], color=PALETTE["ink"], lw=1.25, zorder=4)
    ax.scatter([s3_cx], [split_y], c=PALETTE["ink"], s=15, zorder=5)

    # Sigmoid Gate Visual Card on Right Branch
    g_card_w, g_card_h = 2.16, 0.44
    draw_card(ax, gt_x - g_card_w / 2, gate_y + 0.83, g_card_w, g_card_h, fc="#FFFFFF", ec=PALETTE["amber_edge"], lw=1.0, radius=0.05, zorder=4)
    # Mini Sigmoid curve icon on left of gate card
    sig_t = np.linspace(-3, 3, 40)
    sig_x = (gt_x - 0.82) + 0.08 * sig_t
    sig_y = (gate_y + 1.05) + 0.12 * (1 / (1 + np.exp(-sig_t)) - 0.5) * 2
    ax.plot(sig_x, sig_y, color="#D97706", lw=1.8, zorder=5)
    ax.text(gt_x + 0.22, gate_y + 1.13, "Sigmoid Gate $\\sigma(\\cdot)$", ha="center", va="center", fontsize=8.2, fontweight="bold", color=PALETTE["amber_dark"], zorder=5)
    ax.text(gt_x + 0.22, gate_y + 0.93, "Noise suppression", ha="center", va="center", fontsize=7.2, color=PALETTE["sub_ink"], zorder=5)
    draw_arrow(ax, (gt_x, split_y), (gt_x, gate_y + 0.83 + g_card_h), lw=1.2, ms=8.5)

    ax.plot([id_x, id_x], [split_y, mul_y], color=PALETTE["ink"], lw=1.25, zorder=4)
    ax.text(id_x - 0.08, (split_y + mul_y) / 2, "Identity\nStream", ha="right", va="center", fontsize=7.6, fontweight="bold", color=PALETTE["sub_ink"], zorder=5)
    draw_arrow(ax, (id_x, mul_y), (s3_cx - 0.14, mul_y), lw=1.25, ms=9)

    ax.plot([gt_x, gt_x], [gate_y + 0.83, mul_y], color=PALETTE["gate_main"], lw=1.25, zorder=4)
    draw_arrow(ax, (gt_x, mul_y), (s3_cx + 0.14, mul_y), color=PALETTE["gate_main"], lw=1.25, ms=9)

    draw_op_node(ax, s3_cx, mul_y, symbol=r"$\odot$", r=0.125, fc="#FFFFFF", ec=PALETTE["gate_main"], text_color=PALETTE["amber_dark"], fontsize=11.0, zorder=6)
    draw_arrow(ax, (s3_cx, mul_y - 0.125), (s3_cx, gate_y + 0.42), lw=1.25, ms=9)

    draw_3d_tensor_cube(
        ax, s3_cx - 0.58, gate_y + 0.20, w=1.62, h=0.22, depth=0.09, n_slices=6,
        face_colors=["#F59E0B", "#FBBF24", "#D97706", "#F59E0B", "#FBBF24", "#B45309"],
        top_color="#FEF08A", side_color="#92400E", zorder=5,
    )
    ax.text(s3_cx + 1.06, gate_y + 0.20, "Gated Pocket\nTensor (512-D)", ha="center", va="center", fontsize=7.9, fontweight="bold", color=PALETTE["amber_dark"], zorder=5)

    # =========================================================================
    # STAGE 4: Hyperspherical Angular-Margin Learning (3D Sphere Visual)
    # =========================================================================
    s4_pad = 0.22
    s4_bx = s4_x + s4_pad
    s4_bw = s4_w - 2 * s4_pad
    s4_cx = s4_x + s4_w / 2

    # 1. Deep Projection & L2 Hyperspherical Normalization
    fe_y, fe_h = 6.26, 1.62
    draw_header_card(
        ax, s4_bx, fe_y, s4_bw, fe_h,
        title="Deep Projector & Unit Normalization",
        hdr_fc=PALETTE["s4_light"], hdr_text_color=PALETTE["s4_hdr"], divider_color=PALETTE["s4_edge"],
        body_fc="#FFFFFF", ec=PALETTE["s4_edge"], hdr_h=0.33, radius=0.09, lw=1.15, fontsize=9.2, shadow=True,
    )

    # Funnel projection graphic: 512-D -> 256-D Unit Vector
    draw_card(ax, s4_bx + 0.20, fe_y + 0.78, s4_bw - 0.40, 0.36, fc=PALETTE["s4_pale"], ec=PALETTE["s4_edge"], lw=0.9, radius=0.05, zorder=4)
    ax.text(s4_cx, fe_y + 0.96, "MLP Projector (512-D → 256-D)", ha="center", va="center", fontsize=8.4, fontweight="bold", color=PALETTE["ink"], zorder=5)
    draw_arrow(ax, (s4_cx, fe_y + 0.78), (s4_cx, fe_y + 0.54), color=PALETTE["s4_hdr"], lw=1.2, ms=8.5)

    draw_3d_tensor_cube(
        ax, s4_cx - 0.55, fe_y + 0.28, w=1.42, h=0.25, depth=0.09, n_slices=6,
        face_colors=["#7C3AED", "#8B5CF6", "#A78BFA", "#8B5CF6", "#7C3AED", "#6D28D9"],
        top_color="#EDE9FE", side_color="#4C1D95", zorder=5,
    )
    ax.text(s4_cx + 0.96, fe_y + 0.28, r"Unit $\hat{\mathbf{h}} \in \mathbb{S}^{255}$" + "\n" + r"($\|\hat{\mathbf{h}}\|_2 = 1$)", ha="center", va="center", fontsize=8.0, fontweight="bold", color=PALETTE["s4_hdr"], zorder=5)

    # Clean inter-stage bus from Stage 3 Gated Tensor -> Stage 4 Projector
    route_x = (s3_x + s3_w + s4_x) / 2
    draw_ortho_arrow(
        ax,
        [(gate_x + gate_w, gate_y + 0.22), (route_x, gate_y + 0.22), (route_x, fe_y + 0.96), (s4_bx, fe_y + 0.96)],
        color=PALETTE["s4_hdr"], lw=1.55,
    )

    # 2. 3D Unit Hypersphere (S^{255}) Angular-Margin Classifier Card
    af_y, af_h = 1.66, 4.30
    draw_header_card(
        ax, s4_bx, af_y, s4_bw, af_h,
        title=r"3D Unit Hypersphere ($\mathbb{S}^{255}$) Margin Space",
        hdr_fc=PALETTE["s4_light"], hdr_text_color=PALETTE["s4_hdr"], divider_color=PALETTE["s4_edge"],
        body_fc="#FFFFFF", ec=PALETTE["s4_edge"], hdr_h=0.34, radius=0.10, lw=1.15, fontsize=9.2, shadow=True,
    )
    ax.text(s4_cx, af_y + af_h - 0.48, "ArcFace Geodesic Penalty + Focal Hard-Example Mining", ha="center", va="center", fontsize=7.6, color=PALETTE["sub_ink"], zorder=4)

    # Draw 3D Unit Hypersphere
    draw_3d_hypersphere(ax, s4_cx, af_y + 2.18, r=1.14)

    # Visual legend & concise objective summary below 3D sphere
    ax.scatter([s4_bx + 0.26], [af_y + 0.69], c="#7C3AED", s=38, marker="o", zorder=6)
    ax.text(s4_bx + 0.40, af_y + 0.69, "Active Cap ($y=1$)", ha="left", va="center", fontsize=7.6, fontweight="bold", color="#5B21B6", zorder=6)
    ax.scatter([s4_bx + 1.88], [af_y + 0.69], c="#059669", s=40, marker="^", zorder=6)
    ax.text(s4_bx + 2.02, af_y + 0.69, "Inactive Cap ($y=0$)", ha="left", va="center", fontsize=7.6, fontweight="bold", color="#065F46", zorder=6)

    ax.scatter([s4_bx + 0.26], [af_y + 0.43], c="#DC2626", s=54, marker="*", zorder=6)
    ax.text(s4_bx + 0.40, af_y + 0.43, "Hard Boundary Pairs (Focal Weighted)", ha="left", va="center", fontsize=7.5, fontweight="bold", color="#B91C1C", zorder=6)

    draw_card(ax, s4_bx + 0.16, af_y + 0.08, s4_bw - 0.32, 0.25, fc=PALETTE["s4_light"], ec=PALETTE["s4_edge"], lw=0.8, radius=0.04, zorder=4)
    ax.text(s4_cx, af_y + 0.205, r"Margin Logit: $s \cdot \cos(\theta_y + m)$", ha="center", va="center", fontsize=8.1, fontweight="bold", color=PALETTE["s4_hdr"], zorder=5)

    draw_arrow(ax, (s4_cx, fe_y), (s4_cx, af_y + af_h), color=PALETTE["s4_hdr"], lw=1.35)

    # 3. Calibrated DTI Decision Output Card (with probability bar visual)
    out_y, out_h = 0.52, 0.86
    draw_header_card(
        ax, s4_bx, out_y, s4_bw, out_h,
        title="Calibrated DTI Interaction Output",
        hdr_fc=PALETTE["s4_hdr_bg"], hdr_text_color=PALETTE["s4_hdr_text"], divider_color=PALETTE["s4_hdr_line"],
        body_fc="#FFFFFF", ec=PALETTE["s4_hdr_line"], hdr_h=0.30, radius=0.08, lw=1.15, fontsize=9.0, shadow=True,
    )
    # Visual probability meter bar
    bar_w = s4_bw - 0.44
    bar_x = s4_bx + 0.22
    ax.add_patch(Rectangle((bar_x, out_y + 0.14), bar_w * 0.78, 0.25, facecolor="#388E72", edgecolor="#1D5643", lw=0.8, zorder=5))
    ax.add_patch(Rectangle((bar_x + bar_w * 0.78, out_y + 0.14), bar_w * 0.22, 0.25, facecolor="#E2E8F0", edgecolor="#64748B", lw=0.8, zorder=5))
    ax.text(bar_x + bar_w * 0.39, out_y + 0.265, "Active Binding (P = 0.94)", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#FFFFFF", zorder=6)

    draw_arrow(ax, (s4_cx, af_y), (s4_cx, out_y + out_h), color=PALETTE["s4_hdr"], lw=1.35)

    plt.tight_layout(pad=0.06)
    for stem in ("Figure1_ArcMDTI_Framework", "Figure1_ArcMDTI_Framework_clean"):
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved flagship Figure1_ArcMDTI_Framework(.pdf/.png) & Figure1_ArcMDTI_Framework_clean(.pdf/.png)")


def generate_hypersphere_geometry_figure():
    """Generates Figure2_Hypersphere_Margin_Geometry.(pdf/png) as crisp vector graphics."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=300)
    rng = np.random.default_rng(42)

    # Panel (a): Standard Linear Classifier + Cross-Entropy
    ax = axes[0]
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Conventional Linear Classifier + CE", fontsize=11.5, fontweight="bold", color=PALETTE["ink"], pad=10)

    circle = Circle((0, 0), 1.0, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.2, ls="--")
    ax.add_patch(circle)
    ax.plot([-0.95, 0.95], [-0.85, 0.85], color="#1E293B", lw=1.6, label="Hyperplane $W^\\top h + b = 0$")
    ax.fill_between([-0.95, 0.95], [-0.98, 0.72], [-0.72, 0.98], color="#FDBA74", alpha=0.25, label="Ambiguous Overlap Zone")

    pos_x = rng.normal(-0.32, 0.26, 28)
    pos_y = rng.normal(0.32, 0.26, 28)
    neg_x = rng.normal(0.32, 0.26, 36)
    neg_y = rng.normal(-0.32, 0.26, 36)
    ax.scatter(pos_x, pos_y, c="#DC2626", s=42, alpha=0.82, edgecolors="white", lw=0.5, label="Active Pairs ($y=1$)")
    ax.scatter(neg_x, neg_y, c="#2563EB", s=42, alpha=0.82, edgecolors="white", lw=0.5, label="Inactive Pairs ($y=0$)")
    ax.text(
        0,
        -1.22,
        "Radial norm $\\|h\\|_2$ conflated with angle;\nno explicit geodesic margin between classes",
        ha="center",
        va="center",
        fontsize=8.8,
        color=PALETTE["sub_ink"],
    )
    ax.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.9)

    # Panel (b): ArcMDTI Hyperspherical ArcFace + Focal
    ax = axes[1]
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) ArcMDTI: Hyperspherical ArcFace ($+m$) + Focal", fontsize=11.5, fontweight="bold", color=PALETTE["ink"], pad=10)

    sph = Circle((0, 0), 1.0, facecolor="#F8FAFC", edgecolor="#0F172A", lw=1.6)
    ax.add_patch(sph)

    w_pos = Wedge((0, 0), 1.0, 118, 168, width=0.12, facecolor="#FCA5A5", alpha=0.75, edgecolor="#B91C1C", lw=1.2)
    w_neg = Wedge((0, 0), 1.0, -58, -8, width=0.12, facecolor="#93C5FD", alpha=0.75, edgecolor="#1D4ED8", lw=1.2)
    w_margin = Wedge((0, 0), 1.0, 26, 68, facecolor="#FEF08A", alpha=0.55, edgecolor="#CA8A04", lw=1.1, ls="--")
    ax.add_patch(w_pos)
    ax.add_patch(w_neg)
    ax.add_patch(w_margin)

    ang_pos = np.deg2rad(np.linspace(123, 163, 18))
    ang_neg = np.deg2rad(np.linspace(-53, -13, 22))
    ax.scatter(0.94 * np.cos(ang_pos), 0.94 * np.sin(ang_pos), c="#B91C1C", s=38, edgecolors="white", lw=0.5, zorder=5)
    ax.scatter(0.94 * np.cos(ang_neg), 0.94 * np.sin(ang_neg), c="#1D4ED8", s=38, edgecolors="white", lw=0.5, zorder=5)

    for deg, col, lbl in [(143, "#991B1B", r"$\hat{W}_1$"), (-33, "#1E40AF", r"$\hat{W}_0$")]:
        rad = np.deg2rad(deg)
        ax.plot([0, 0.98 * np.cos(rad)], [0, 0.98 * np.sin(rad)], color=col, lw=2.0)
        ax.text(1.13 * np.cos(rad), 1.13 * np.sin(rad), lbl, ha="center", va="center", fontsize=10, fontweight="bold", color=col)

    ax.text(
        0.58 * np.cos(np.deg2rad(47)),
        0.58 * np.sin(np.deg2rad(47)),
        "Geodesic\nMargin $+m$",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#92400E",
    )
    ax.text(
        0,
        -1.22,
        "Logit $s\\cos(\\theta_y + m)$ compacts intra-class\ncaps and enforces wide margin on $\\mathbb{S}^{d-1}$",
        ha="center",
        va="center",
        fontsize=8.8,
        color=PALETTE["sub_ink"],
    )

    # Panel (c): Synergistic Gradient Dynamics
    ax = axes[2]
    p_y = np.linspace(0.01, 0.99, 200)
    grad_ce = 1.0 - p_y
    grad_focal_1 = ((1.0 - p_y) ** 1.0) * (1.0 - 1.0 * p_y * np.log(p_y)) * (1.0 - p_y)
    grad_focal_2 = ((1.0 - p_y) ** 2.0) * (1.0 - 2.0 * p_y * np.log(p_y)) * (1.0 - p_y)

    ax.plot(p_y, grad_ce, ls="--", lw=1.8, color="#64748B", label=r"ArcFace + CE ($\gamma=0$)")
    ax.plot(p_y, grad_focal_1, ls="-", lw=2.4, color="#B91C1C", label=r"ArcMDTI ($\gamma=1.0$, Ours)")
    ax.plot(p_y, grad_focal_2, ls="-.", lw=1.8, color="#1D4ED8", label=r"ArcFace + Focal ($\gamma=2.0$)")
    ax.fill_between(p_y[p_y >= 0.6], 0, grad_ce[p_y >= 0.6], color="#CBD5E1", alpha=0.35)
    ax.annotate(
        "Suppresses noisy/easy\nnegative gradients",
        xy=(0.80, 0.045),
        xytext=(0.56, 0.56),
        arrowprops=dict(arrowstyle="->", color="#0F172A", lw=1.1),
        fontsize=8.2,
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CBD5E1"),
    )

    ax.set_title("(c) Geodesic Gradient Modulation", fontsize=11.5, fontweight="bold", color=PALETTE["ink"], pad=10)
    ax.set_xlabel("Posterior Ground-Truth Probability $P_y$", fontsize=9.5, fontweight="bold", color=PALETTE["ink"])
    ax.set_ylabel("Effective Angular Gradient Weight", fontsize=9.5, fontweight="bold", color=PALETTE["ink"])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.28)
    ax.grid(True, ls="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8.0, frameon=True)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "Figure2_Hypersphere_Margin_Geometry.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "Figure2_Hypersphere_Margin_Geometry.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved Figure2_Hypersphere_Margin_Geometry(.pdf/.png)")


if __name__ == "__main__":
    generate_framework_figure()
    generate_hypersphere_geometry_figure()

    if PAPER_NEW_DIR.exists():
        fig_dir = PAPER_NEW_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fname in (
            "Figure1_ArcMDTI_Framework.pdf",
            "Figure1_ArcMDTI_Framework.png",
            "Figure1_ArcMDTI_Framework_clean.pdf",
            "Figure1_ArcMDTI_Framework_clean.png",
            "Figure2_Hypersphere_Margin_Geometry.pdf",
            "Figure2_Hypersphere_Margin_Geometry.png",
        ):
            src = OUT_DIR / fname
            if src.exists():
                shutil.copy2(src, fig_dir / fname)
        print(f"[OK] Synced vector PDFs & PNGs to {fig_dir}")
