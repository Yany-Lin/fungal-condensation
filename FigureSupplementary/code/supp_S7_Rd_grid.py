#!/usr/bin/env python3
"""Supplementary Figure S7: All-15-trial R(d) profiles + tanh fits (3x5 grid).

Each subplot shows droplet scatter at t=14.5-15.5 min, binned mean overlay,
tanh fit curve, and delta marker.
"""

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

HG_CONDITIONS = [
    ('agar', ['agar.1','agar.2','agar.3','agar.4','agar.5'],
     'Agar (a_w=1.00)', '#9E9E9E'),
    ('0.5to1', ['0.5to1.2','0.5to1.3','0.5to1.4','0.5to1.5','0.5to1.7'],
     '0.5:1 NaCl (a_w=0.93)', '#E67E22'),
    ('1to1', ['1to1.1','1to1.2','1to1.3','1to1.4','1to1.5'],
     '1:1 NaCl (a_w=0.87)', '#5B8FC9'),
    ('2to1', ['2to1.1','2to1.2','2to1.3','2to1.4','2to1.5'],
     '2:1 NaCl (a_w=0.75)', '#C0392B'),
]

T_WINDOW = (14.5, 15.5)
BIN_WIDTH = 100  # um
MIN_DROPS = 5
MAX_DIST  = 2500  # um

BASE = Path(__file__).resolve().parents[2]
HG_AGG = BASE / 'FigureHGAggregate' / 'raw_data' / 'aggregate_edt'
OUT_DIR = OUTPUT_DIR / 'S5'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    apply_style()

    fig, axes = plt.subplots(4, 5, figsize=(200 * MM, 195 * MM),
                              sharex=True, sharey=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.08,
                         hspace=0.45, wspace=0.28)

    for row_idx, (cond_key, trial_ids, cond_label, cond_color) in enumerate(HG_CONDITIONS):
        for col_idx, tid in enumerate(trial_ids):
            ax = axes[row_idx, col_idx]
            delta = DELTA[tid]

            # Load droplets in time window
            path = HG_AGG / f'{tid}_edt_droplets.csv'
            df = pd.read_csv(path)
            tw = df[(df['time_min'] >= T_WINDOW[0]) &
                    (df['time_min'] <= T_WINDOW[1])].copy()

            print(f'  [{row_idx},{col_idx}] {tid}: {len(tw)} droplets')

            # Ghost scatter — very faint, just shows data density
            ax.scatter(tw['distance_um'] / 1000, tw['radius_um'],
                       s=0.2, alpha=0.06, color='#999999', rasterized=True,
                       edgecolors='none')

            # Binned mean ± SEM
            bins = np.arange(0, MAX_DIST + BIN_WIDTH, BIN_WIDTH)
            tw['dbin'] = pd.cut(tw['distance_um'], bins=bins,
                                labels=bins[:-1] + BIN_WIDTH/2).astype(float)
            grp = tw.groupby('dbin')['radius_um']
            prof = grp.agg(mean='mean',
                           sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)),
                           n='count').reset_index()
            prof = prof[prof['n'] >= MIN_DROPS]
            d_mm = prof['dbin'] / 1000

            ax.fill_between(d_mm, prof['mean'] - prof['sem'],
                            prof['mean'] + prof['sem'],
                            alpha=0.25, color=cond_color, zorder=2)
            ax.plot(d_mm, prof['mean'],
                    '-', color=cond_color, lw=2.0, zorder=3)

            # Delta marker — solid vertical line, more visible
            ax.axvline(delta / 1000, color=cond_color, ls='--', lw=0.8, alpha=0.7,
                       zorder=4)

            ax.set_xlim(0, MAX_DIST / 1000)
            ax.set_ylim(bottom=0)
            clean_axes(ax)
            ax.tick_params(labelsize=TICK_SIZE - 0.5, pad=2)

            # Trial ID in top-left corner of each panel (below panel border)
            ax.text(0.97, 0.97, tid, transform=ax.transAxes,
                    fontsize=TICK_SIZE - 1.5, ha='right', va='top',
                    color=cond_color, alpha=0.75)

        # Row label
        axes[row_idx, 0].text(
            -0.35, 0.5, cond_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=TICK_SIZE, fontweight='bold', color=cond_color,
            ha='right', va='center', rotation=90, linespacing=1.3,
        )

    fig.text(0.53, 0.01, 'Distance from boundary (mm)', ha='center',
             fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, 'R (um)', va='center', rotation=90,
             fontsize=LABEL_SIZE)

    stem = str(OUT_DIR / 'FigureS5_Rd_profiles_grid')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'\nSaved: {stem}.pdf/.svg')


if __name__ == '__main__':
    main()
