#!/usr/bin/env python3
"""Supplementary Figure S13: All-15-fungal-trial R(d) profiles + tanh fits (3x5 grid)."""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import (
    DELTA, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    apply_style, clean_axes, save_fig,
)
from pathlib import Path

FUNGI_CONDITIONS = [
    ('Green', ['Green.1','Green.2','Green.3','Green.4','Green.5'],
     'Aspergillus', '#4CAF50'),
    ('black', ['black.1','black.2','black.3','black.4','black.5'],
     'Rhizopus', '#212121'),
    ('white', ['white.1','white.2','white.3','white.4','white.5'],
     'Mucor', '#757575'),
]

T_WINDOW = (14.5, 15.5)
BIN_WIDTH = 100
MIN_DROPS = 5
MAX_DIST  = 2500

BASE = Path(__file__).resolve().parents[2]
FG_AGG = BASE / 'FigureFungi' / 'raw_data' / 'aggregate_edt'
METRICS = BASE / 'FigureFungi' / 'output' / 'fungi_metrics.csv'
OUT_DIR = OUTPUT_DIR / 'S13'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tanh_model(d, y_near, y_far, alpha, r0):
    return (y_near + y_far) / 2 + (y_far - y_near) / 2 * np.tanh(alpha * (d - r0))


def main():
    apply_style()
    metrics = pd.read_csv(METRICS)
    params = {row['trial_id']: row for _, row in metrics.iterrows()}

    fig, axes = plt.subplots(3, 5, figsize=(180 * MM, 120 * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.08,
                         hspace=0.30, wspace=0.12)

    for row_idx, (cond_key, trial_ids, cond_label, cond_color) in enumerate(FUNGI_CONDITIONS):
        for col_idx, tid in enumerate(trial_ids):
            ax = axes[row_idx, col_idx]
            delta = DELTA[tid]

            path = FG_AGG / f'{tid}_edt_droplets.csv'
            df = pd.read_csv(path)
            tw = df[(df['time_min'] >= T_WINDOW[0]) &
                    (df['time_min'] <= T_WINDOW[1])].copy()

            print(f'  [{row_idx},{col_idx}] {tid}: {len(tw)} droplets')

            ax.scatter(tw['distance_um'] / 1000, tw['radius_um'],
                       s=0.3, alpha=0.15, color='#888888', rasterized=True,
                       edgecolors='none')

            bins = np.arange(0, MAX_DIST + BIN_WIDTH, BIN_WIDTH)
            tw['dbin'] = pd.cut(tw['distance_um'], bins=bins,
                                labels=bins[:-1] + BIN_WIDTH/2).astype(float)
            grp = tw.groupby('dbin')['radius_um']
            prof = grp.agg(mean='mean', n='count').reset_index()
            prof = prof[prof['n'] >= MIN_DROPS]
            ax.plot(prof['dbin'] / 1000, prof['mean'],
                    'o-', color=cond_color, ms=2, lw=1.0, zorder=3)

            p = params.get(tid)
            if p is not None and not pd.isna(p.get('alpha')):
                d_fit = np.linspace(0, MAX_DIST, 300)
                r0 = p['r0_um'] if not pd.isna(p.get('r0_um')) else np.median(prof['dbin'])
                R_fit = tanh_model(d_fit, p['y_near'], p['y_far'], p['alpha'], r0)
                ax.plot(d_fit / 1000, R_fit, '-', color=cond_color,
                        lw=1.2, alpha=0.7, zorder=2)

            ax.axvline(delta / 1000, color=cond_color, ls=':', lw=0.6, alpha=0.6)
            ax.set_xlim(0, MAX_DIST / 1000)
            ax.set_ylim(0, 70)
            clean_axes(ax)
            ax.tick_params(labelsize=TICK_SIZE - 1.5, pad=1.5)

            if row_idx == 0:
                ax.set_title(f'Rep {col_idx + 1}', fontsize=TICK_SIZE, pad=3)

        axes[row_idx, 0].text(
            -0.35, 0.5, cond_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=TICK_SIZE - 0.5, fontweight='bold', color=cond_color,
            ha='right', va='center', rotation=90, linespacing=1.3,
        )

    fig.text(0.53, 0.01, 'Distance from boundary (mm)', ha='center',
             fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, 'R (µm)', va='center', rotation=90,
             fontsize=LABEL_SIZE)

    stem = str(OUT_DIR / 'FigureS13_fungi_Rd_profiles')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'\nSaved: {stem}.png/.pdf/.svg')


if __name__ == '__main__':
    main()
