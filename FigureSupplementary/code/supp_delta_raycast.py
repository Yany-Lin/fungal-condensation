#!/usr/bin/env python3
"""Supplementary Figure S6: δ (dry-zone width) raycast methodology.

Left column (3 rows): circular hydrogel schematic + dry-zone annulus +
  Voronoi raycast lines + δ annotation for agar.4, 1to1.1, 2to1.1.
Right column: bar chart of δ for all 35 trials.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    DELTA, CONDITIONS, HG_AGG_DIR, FG_AGG_DIR,
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    apply_style, clean_axes, save_fig,
)

OUT_DIR = OUTPUT_DIR / 'S7'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3 exemplar trials — agar (small δ), 1:1 (medium), 2:1 (large)
EXEMPLARS = [
    ('agar.4',  'Agar',      '#9E9E9E', 'A'),
    ('1to1.1',  '1:1 NaCl', '#5B8FC9', 'B'),
    ('2to1.1',  '2:1 NaCl', '#C0392B', 'C'),
]

N_RAYS   = 60    # number of Voronoi boundary samples
T_TARGET = 15.0  # minutes


def load_polygon(tid):
    for agg_dir in [HG_AGG_DIR, FG_AGG_DIR]:
        path = agg_dir / f'{tid}_boundary_polygon.csv'
        if path.exists():
            df = pd.read_csv(path)
            return df['x'].values, df['y'].values
    return None, None


def fit_circle(xs, ys):
    """Fit circle to polygon: centroid + mean radius (µm)."""
    cx = np.mean(xs)
    cy = np.mean(ys)
    r  = np.mean(np.hypot(xs - cx, ys - cy))
    return cx, cy, r


def sample_circle_uniform(cx, cy, r, n):
    """Return n points uniformly spaced on a circle (µm)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cx + r * np.cos(angles), cy + r * np.sin(angles)


def draw_raycast_panel(ax, tid, label, color, panel_letter):
    delta_um = DELTA[tid]

    # Load polygon → fit circle
    xs, ys = load_polygon(tid)
    if xs is None:
        ax.text(0.5, 0.5, 'no polygon', transform=ax.transAxes,
                ha='center', va='center', color='grey')
        return

    cx_c, cy_c, r_c = fit_circle(xs, ys)   # µm
    cx_c_mm  = cx_c / 1000
    cy_c_mm  = cy_c / 1000
    r_c_mm   = r_c  / 1000
    delta_mm = delta_um / 1000

    # Load droplets — frame nearest T_TARGET
    agg_dir = FG_AGG_DIR if tid.startswith(('Green', 'white', 'black')) else HG_AGG_DIR
    df = pd.read_csv(agg_dir / f'{tid}_edt_droplets.csv')
    times = df['time_min'].unique()
    best_t = times[np.argmin(np.abs(times - T_TARGET))]
    frame = df[df['time_min'] == best_t].copy()

    cx   = frame['cx'].values
    cy_d = frame['cy'].values
    radius = frame['radius_um'].values
    dist   = frame['distance_um'].values

    cx_mm  = cx   / 1000
    cy_mm  = cy_d / 1000

    dry_outer = Circle((cx_c_mm, cy_c_mm), r_c_mm + delta_mm,
                       facecolor='#FFF9C4', edgecolor='none',
                       alpha=0.85, zorder=2)
    ax.add_patch(dry_outer)

    hg_disk = Circle((cx_c_mm, cy_c_mm), r_c_mm,
                     facecolor='white', edgecolor=color,
                     linewidth=1.5, alpha=1.0, zorder=3)
    ax.add_patch(hg_disk)

    dry_edge = Circle((cx_c_mm, cy_c_mm), r_c_mm + delta_mm,
                      facecolor='none', edgecolor='#F9A825',
                      linewidth=0.9, linestyle='--', alpha=0.85, zorder=4)
    ax.add_patch(dry_edge)

    d_norm = np.clip(dist / 2500, 0, 1)
    drop_colors = plt.cm.plasma(d_norm)
    scale = 500
    ax.scatter(cx_mm, cy_mm, s=radius**2 * scale / 1e6,
               c=drop_colors, alpha=0.70, linewidths=0, zorder=5, rasterized=True)

    bx, by = sample_circle_uniform(cx_c, cy_c, r_c, N_RAYS)
    bx_mm  = bx / 1000
    by_mm  = by / 1000

    if len(cx_mm) > 0:
        dmat     = np.hypot(cx_mm[:, None] - bx_mm[None, :],
                            cy_mm[:, None] - by_mm[None, :])
        assigned = np.argmin(dmat, axis=1)

        out_dirs_x = bx_mm - cx_c_mm
        annot_ray  = int(np.argmin(out_dirs_x))   # most-leftward boundary point

        for ray_idx in range(len(bx_mm)):
            owned = np.where(assigned == ray_idx)[0]
            if len(owned) == 0:
                continue
            d_owned = dmat[owned, ray_idx]
            nearest = owned[np.argmin(d_owned)]

            bxi, byi = bx_mm[ray_idx], by_mm[ray_idx]
            dxi, dyi = cx_mm[nearest], cy_mm[nearest]

            ax.plot([bxi, dxi], [byi, dyi],
                    color='#555555', lw=0.45, alpha=0.40, zorder=6)
            ax.plot(bxi, byi, '.', color=color, ms=1.8, zorder=7)

            if ray_idx == annot_ray and len(owned) > 0:
                ray_len = np.hypot(dxi - bxi, dyi - byi)
                if ray_len > 0.02:   # skip degenerate rays
                    mscale = np.clip(delta_mm * 18, 3.5, 6.5)
                    ax.annotate('', xy=(dxi, dyi), xytext=(bxi, byi),
                                arrowprops=dict(arrowstyle='<->',
                                                color='#111111', lw=0.9,
                                                mutation_scale=mscale),
                                zorder=9)
                    ray_ux = (bxi - cx_c_mm) / (np.hypot(bxi - cx_c_mm,
                                                           byi - cy_c_mm) + 1e-9)
                    ray_uy = (byi - cy_c_mm) / (np.hypot(bxi - cx_c_mm,
                                                           byi - cy_c_mm) + 1e-9)
                    mid_x = (bxi + dxi) / 2 + ray_ux * delta_mm * 0.7
                    mid_y = (byi + dyi) / 2 + ray_uy * delta_mm * 0.7
                    ax.text(mid_x, mid_y, 'δ',
                            fontsize=TICK_SIZE + 2,
                            ha='center', va='center',
                            color='#111111', zorder=10)

    ax.set_aspect('equal')
    ax.axis('off')   # no ticks/spines — scale bar only

    ax.autoscale_view()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    pad_x = 0.04 * (xlim[1] - xlim[0])
    pad_y = 0.04 * (ylim[1] - ylim[0])
    sb_x  = xlim[0] + pad_x
    sb_y  = ylim[0] + pad_y
    ax.plot([sb_x, sb_x + 1.0], [sb_y, sb_y],
            color='k', lw=1.8, solid_capstyle='butt', zorder=11)
    ax.text(sb_x + 0.5, sb_y + 0.015 * (ylim[1] - ylim[0]), '1 mm',
            ha='center', va='bottom', fontsize=TICK_SIZE - 0.5, color='k')

    ax.set_title(f'{label}   δ = {delta_um:.0f} µm',
                 fontsize=TICK_SIZE + 0.5, color=color,
                 fontweight='bold', pad=5)

    ax.text(-0.04, 1.05, panel_letter, transform=ax.transAxes,
            fontsize=PANEL_LBL, va='top')


def main():
    apply_style()

    fig = plt.figure(figsize=(180 * MM, 175 * MM))

    left_w, right_w = 0.56, 0.29
    left_x, right_x = 0.05, 0.67
    panel_h = 0.25
    bottoms = [0.70, 0.40, 0.10]

    for ypos, (tid, label, color, letter) in zip(bottoms, EXEMPLARS):
        ax = fig.add_axes([left_x, ypos, left_w, panel_h])
        print(f'Drawing {tid}...', flush=True)
        draw_raycast_panel(ax, tid, label, color, letter)

    ax_bar = fig.add_axes([right_x, 0.10, right_w, 0.85])

    x_pos = 0
    xtick_positions, xtick_labels = [], []
    bar_w = 0.7

    for cond_key, trial_ids, cond_label, color in CONDITIONS:
        deltas = [DELTA[tid] for tid in trial_ids if tid in DELTA]
        if not deltas:
            continue
        for d in deltas:
            ax_bar.bar(x_pos, d / 1000, width=bar_w, color=color,
                       alpha=0.82, edgecolor='white', linewidth=0.3, zorder=2)
            x_pos += 1

        mean_d = np.mean(deltas) / 1000
        ax_bar.hlines(mean_d, x_pos - len(deltas) - 0.4, x_pos - 0.6,
                      color=color, lw=1.4, zorder=3)

        mid = x_pos - len(deltas) / 2.0 - 0.5
        xtick_positions.append(mid)
        short = (cond_label
                 .replace(' NaCl', '\nNaCl')
                 .replace('Aspergillus', 'Asper.')
                 .replace('Rhizopus', 'Rhiz.')
                 .replace('Mucor', 'Mucor'))
        xtick_labels.append(short)
        x_pos += 0.5

    clean_axes(ax_bar)
    ax_bar.set_xticks(xtick_positions)
    ax_bar.set_xticklabels(xtick_labels, fontsize=TICK_SIZE - 2, rotation=0)
    ax_bar.set_ylabel('δ (mm)', fontsize=LABEL_SIZE - 0.5)
    ax_bar.set_ylim(0, None)
    ax_bar.tick_params(labelsize=TICK_SIZE - 1, axis='y')
    ax_bar.set_title('δ by trial', fontsize=TICK_SIZE, pad=3)
    ax_bar.text(-0.24, 1.05, 'D', transform=ax_bar.transAxes,
                fontsize=PANEL_LBL, va='top')

    legend_patch = mpatches.Patch(facecolor='#FFF9C4', edgecolor='#F9A825',
                                  linestyle='--', linewidth=1.2,
                                  label='dry zone  (δ)')
    fig.legend(handles=[legend_patch],
               loc='lower left',
               bbox_to_anchor=(left_x, bottoms[2] - 0.08),
               fontsize=TICK_SIZE + 1,
               frameon=False,
               handlelength=1.8,
               handleheight=1.2)

    stem = str(OUT_DIR / 'FigureS7_delta_raycast')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'Saved: {stem}.pdf/.svg')


if __name__ == '__main__':
    main()
