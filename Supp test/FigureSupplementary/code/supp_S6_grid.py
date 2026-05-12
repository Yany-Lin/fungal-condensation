#!/usr/bin/env python3
"""Supplementary Figure S6: 30-trial R(t) grid (6 conditions × 5 replicates).

Each subplot shows median R(t) colored by r' (distance from dry-zone edge),
matching the style of main-text Figure 1C.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import (
    CONDITIONS, DELTA, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    LW_SPINE, apply_style, clean_axes, load_droplets, save_fig,
)

BIN_WIDTH_UM  = 300
MIN_DROPS_BIN = 5
T_MIN         = 5.0
T_MAX         = 15.0
R_ANCHOR      = 20.0
T_ANCHOR      = 5.0
R_MAX         = 70
R_CBAR_MAX    = 2.0   # mm — colorbar range

OUT_DIR = OUTPUT_DIR / 'S6'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_one_cell(ax, df, delta_um):
    """Plot Panel-C–style R(t) colored by r' into a single axes."""
    data = df[(df['time_min'] >= T_MIN) & (df['time_min'] <= T_MAX)
              & (df['radius_um'] > 0)].copy()
    data['r_prime'] = data['distance_um'] - delta_um
    data = data[data['r_prime'] >= 0]

    if data.empty:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=TICK_SIZE - 1, color='grey')
        return

    max_r = data['r_prime'].max()
    edges = np.arange(0, max_r + BIN_WIDTH_UM, BIN_WIDTH_UM)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    cmap = plt.cm.plasma

    for i in range(len(edges) - 1):
        in_bin = data[(data['r_prime'] >= edges[i]) & (data['r_prime'] < edges[i + 1])]
        if in_bin.empty:
            continue

        ts = []
        for t, frame in in_bin.groupby('time_min'):
            R = frame['radius_um'].values
            if len(R) >= MIN_DROPS_BIN:
                ts.append((t, np.median(R)))

        if len(ts) < 3:
            continue

        t_arr = np.array([x[0] for x in ts])
        R_arr = np.array([x[1] for x in ts])
        color = cmap(bin_centers[i] / 1000 / R_CBAR_MAX)
        ax.plot(t_arr, R_arr, '-', color=color, lw=0.7, alpha=0.85)

    t_ref = np.linspace(T_MIN, T_MAX, 200)
    ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** (1/3),
            'k--', lw=0.5, alpha=0.5)
    ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** 1.0,
            color='#C0392B', ls='--', lw=0.5, alpha=0.5)

    ax.set_xlim(T_MIN + 0.5, T_MAX + 0.5)
    ax.set_ylim(0, R_MAX)


def main():
    apply_style()

    n_rows, n_cols = 7, 5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(180 * MM, 245 * MM),
        sharex=True, sharey=True,
    )
    fig.subplots_adjust(
        left=0.09, right=0.88, top=0.94, bottom=0.06,
        hspace=0.35, wspace=0.15,
    )

    for row_idx, (cond_key, trial_ids, cond_label, cond_color) in enumerate(CONDITIONS):
        for col_idx, tid in enumerate(trial_ids):
            ax = axes[row_idx, col_idx]
            delta = DELTA[tid]

            print(f'  [{row_idx},{col_idx}] {tid} (delta={delta:.0f} um) ... ', end='')
            df = load_droplets(tid)
            plot_one_cell(ax, df, delta)
            print(f'{len(df):,} droplets')

            clean_axes(ax)
            ax.tick_params(labelsize=TICK_SIZE - 1.5, pad=1.5)

            if row_idx == 0:
                ax.set_title(f'Rep {col_idx + 1}', fontsize=TICK_SIZE, pad=3)

        axes[row_idx, 0].text(
            -0.45, 0.5, cond_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=LABEL_SIZE, fontweight='bold', color=cond_color,
            ha='right', va='center', rotation=90,
        )

    fig.text(0.48, 0.015, 'Time (min)', ha='center', fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, 'Median R (µm)', va='center', rotation=90,
             fontsize=LABEL_SIZE)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                norm=plt.Normalize(0, R_CBAR_MAX))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("r' (mm)", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 1.5)

    stem = str(OUT_DIR / 'FigureS6_30trial_grid')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'\nSaved: {stem}.png/.pdf/.svg')


if __name__ == '__main__':
    main()
