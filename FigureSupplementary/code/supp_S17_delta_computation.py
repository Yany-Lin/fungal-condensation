#!/usr/bin/env python3
"""Supplementary Figure S17: Delta (Delta) computation methodology.

Three panels:
  A — Example boundary polygon (1to1.1) with Steiner offset at r = Delta shown
      as a dashed ring, illustrating how the parallel body formula is applied.
  B — Bar chart of Delta values for all 35 trials, colour-coded by condition.
      Shows the range (78-1005 um) and systematic increase with NaCl concentration.
  C — Scatter: Delta vs perimeter P_body for all trials with polygon data.
      Reveals that Delta scales with boundary complexity.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    DELTA, CONDITIONS, HG_AGG_DIR, FG_AGG_DIR,
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    apply_style, clean_axes, save_fig,
    polygon_area_perimeter,
)

OUT_DIR = OUTPUT_DIR / 'S17'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _agg_dir(tid):
    if tid.startswith(('Green', 'white', 'black')):
        return FG_AGG_DIR
    return HG_AGG_DIR


def load_polygon(tid):
    """Return (xs, ys) arrays for the boundary polygon of trial tid."""
    path = _agg_dir(tid) / f'{tid}_boundary_polygon.csv'
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    return df['x'].values, df['y'].values


def steiner_offset_polygon(xs, ys, r):
    """Approximate Steiner offset polygon by outward-offsetting vertices.

    For visualisation only — moves each vertex outward along the local
    outward normal by distance r.
    """
    n = len(xs)
    oxs, oys = [], []
    for i in range(n):
        # Tangent direction
        dx = xs[(i + 1) % n] - xs[(i - 1) % n]
        dy = ys[(i + 1) % n] - ys[(i - 1) % n]
        length = np.hypot(dx, dy)
        if length < 1e-9:
            oxs.append(xs[i])
            oys.append(ys[i])
            continue
        # Outward normal (assuming CCW polygon → left normal)
        nx = -dy / length
        ny =  dx / length
        oxs.append(xs[i] + r * nx)
        oys.append(ys[i] + r * ny)
    return np.array(oxs), np.array(oys)


def main():
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 78 * MM))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.18,
                        wspace=0.42)

    ax = axes[0]
    DEMO_TID = '1to1.1'
    xs, ys = load_polygon(DEMO_TID)

    if xs is not None:
        _, perim = polygon_area_perimeter(xs, ys)
        delta = DELTA[DEMO_TID]

        xc = np.append(xs, xs[0])
        yc = np.append(ys, ys[0])
        ax.fill(xc / 1000, yc / 1000, color='#B3D7FF', alpha=0.45, zorder=1)
        ax.plot(xc / 1000, yc / 1000, color='#2471A3', lw=1.0, zorder=2,
                label='Hyphae boundary')

        ox, oy = steiner_offset_polygon(xs, ys, delta)
        oxc = np.append(ox, ox[0])
        oyc = np.append(oy, oy[0])
        ax.plot(oxc / 1000, oyc / 1000, '--', color='#E74C3C', lw=0.9,
                zorder=3, label=f'r = \u0394 = {delta:.0f} um')

        cx_mean = np.mean(xs) / 1000
        cy_mean = np.mean(ys) / 1000
        ax.plot(cx_mean, cy_mean, 'k+', ms=5, mew=0.8, zorder=4)

    clean_axes(ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)', fontsize=LABEL_SIZE - 0.5)
    ax.set_ylabel('y (mm)', fontsize=LABEL_SIZE - 0.5)
    ax.tick_params(labelsize=TICK_SIZE - 1.5)
    ax.set_title(f'Example: {DEMO_TID}', fontsize=TICK_SIZE, pad=3)
    ax.legend(fontsize=TICK_SIZE - 2.5, frameon=False, loc='lower right',
              handlelength=1.2, labelspacing=0.3)
    ax.text(-0.18, 1.07, 'a', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    ax = axes[1]

    x_pos = 0
    xtick_positions = []
    xtick_labels = []
    bar_w = 0.7

    for cond_key, trial_ids, cond_label, color in CONDITIONS:
        deltas = [DELTA[tid] for tid in trial_ids if tid in DELTA]
        if not deltas:
            continue
        for j, (tid, d) in enumerate(zip(trial_ids, deltas)):
            ax.bar(x_pos, d / 1000, width=bar_w, color=color, alpha=0.80,
                   edgecolor='white', linewidth=0.3, zorder=2)
            x_pos += 1

        mean_d = np.mean(deltas) / 1000
        ax.hlines(mean_d, x_pos - len(deltas) - 0.4, x_pos - 0.6,
                  color=color, lw=1.2, zorder=3)

        mid = x_pos - len(deltas) / 2.0 - 0.5
        xtick_positions.append(mid)
        xtick_labels.append(cond_label.replace(' NaCl', '\nNaCl').replace('Aspergillus', 'Asper.'))
        x_pos += 0.6  # gap between conditions

    clean_axes(ax)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=TICK_SIZE - 2.5, rotation=0)
    ax.set_ylabel('\u0394 (mm)', fontsize=LABEL_SIZE - 0.5)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=TICK_SIZE - 1.5, axis='y')
    ax.set_title('Characteristic length \u0394 by trial', fontsize=TICK_SIZE, pad=3)
    ax.text(-0.22, 1.07, 'b', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    ax = axes[2]

    perims, deltas, colors = [], [], []
    for cond_key, trial_ids, cond_label, color in CONDITIONS:
        for tid in trial_ids:
            xs_t, ys_t = load_polygon(tid)
            if xs_t is None or tid not in DELTA:
                continue
            _, P = polygon_area_perimeter(xs_t, ys_t)
            perims.append(P / 1000)       # mm
            deltas.append(DELTA[tid] / 1000)  # mm
            colors.append(color)

    if perims:
        ax.scatter(perims, deltas, c=colors, s=18, alpha=0.85,
                   linewidths=0.3, edgecolors='white', zorder=3)

        slope, intercept, r, p, _ = stats.linregress(perims, deltas)
        x_fit = np.linspace(min(perims), max(perims), 100)
        ax.plot(x_fit, slope * x_fit + intercept, '--', color='#555555',
                lw=0.9, zorder=2,
                label=f'r = {r:.2f},  p < 0.001' if p < 0.001
                      else f'r = {r:.2f},  p = {p:.3f}')
        ax.legend(fontsize=TICK_SIZE - 2.5, frameon=False, loc='upper left',
                  handlelength=1.2)

    clean_axes(ax)
    ax.set_xlabel('Boundary perimeter P (mm)', fontsize=LABEL_SIZE - 0.5)
    ax.set_ylabel('\u0394 (mm)', fontsize=LABEL_SIZE - 0.5)
    ax.tick_params(labelsize=TICK_SIZE - 1.5)
    ax.set_title('\u0394 vs boundary perimeter', fontsize=TICK_SIZE, pad=3)
    ax.text(-0.22, 1.07, 'c', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    handles = [mpatches.Patch(color=color, label=label)
               for _, _, label, color in CONDITIONS]
    fig.legend(handles=handles, loc='lower center', ncol=7,
               fontsize=TICK_SIZE - 2.5, frameon=False,
               bbox_to_anchor=(0.52, -0.01))

    stem = str(OUT_DIR / 'FigureS17_delta_computation')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'Saved: {stem}.pdf/.svg')


if __name__ == '__main__':
    main()
