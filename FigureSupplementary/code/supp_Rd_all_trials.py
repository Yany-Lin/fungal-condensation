#!/usr/bin/env python3
"""Supplementary Figure S3: All 35 trials R(d) profiles — 7×5 grid.

One panel per trial (7 conditions × 5 replicates).
Each panel: ghost scatter + mean ± SEM band + Δ marker.
Replaces former separate hydrogel and fungi grids.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import (
    DELTA, CONDITIONS, HG_AGG_DIR, FG_AGG_DIR,
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    apply_style, clean_axes, save_fig,
)

T_WINDOW  = (14.5, 15.5)
BIN_WIDTH = 100    # µm
MIN_DROPS = 5
MAX_DIST  = 2500   # µm

OUT_DIR = OUTPUT_DIR / 'S3'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def agg_dir(tid):
    if tid.startswith(('Green', 'white', 'black')):
        return FG_AGG_DIR
    return HG_AGG_DIR


def main():
    apply_style()

    n_cond = len(CONDITIONS)   # 7
    fig, axes = plt.subplots(n_cond, 5,
                              figsize=(200 * MM, 255 * MM),
                              sharex=True, sharey=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.975, bottom=0.055,
                        hspace=0.48, wspace=0.28)

    for row_idx, (cond_key, trial_ids, cond_label, cond_color) in enumerate(CONDITIONS):
        for col_idx, tid in enumerate(trial_ids):
            ax = axes[row_idx, col_idx]
            delta = DELTA[tid]

            path = agg_dir(tid) / f'{tid}_edt_droplets.csv'
            df = pd.read_csv(path)
            tw = df[(df['time_min'] >= T_WINDOW[0]) &
                    (df['time_min'] <= T_WINDOW[1])].copy()

            print(f'  {tid}: {len(tw)} droplets', flush=True)

            ax.scatter(tw['distance_um'] / 1000, tw['radius_um'],
                       s=0.2, alpha=0.06, color='#999999',
                       rasterized=True, edgecolors='none')

            bins = np.arange(0, MAX_DIST + BIN_WIDTH, BIN_WIDTH)
            tw['dbin'] = pd.cut(tw['distance_um'], bins=bins,
                                labels=(bins[:-1] + BIN_WIDTH / 2)).astype(float)
            grp = tw.groupby('dbin')['radius_um']
            prof = grp.agg(
                mean='mean',
                sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)),
                n='count',
            ).reset_index()
            prof = prof[prof['n'] >= MIN_DROPS]
            d_mm = prof['dbin'] / 1000

            ax.fill_between(d_mm,
                            prof['mean'] - prof['sem'],
                            prof['mean'] + prof['sem'],
                            alpha=0.25, color=cond_color, zorder=2)
            ax.plot(d_mm, prof['mean'],
                    '-', color=cond_color, lw=2.0, zorder=3)

            ax.axvline(delta / 1000, color=cond_color,
                       ls='--', lw=0.9, alpha=0.75, zorder=4)

            ax.set_xlim(0, MAX_DIST / 1000)
            ax.set_ylim(bottom=0)
            clean_axes(ax)
            ax.tick_params(labelsize=TICK_SIZE - 0.5, pad=2)

            ax.text(0.97, 0.97, tid, transform=ax.transAxes,
                    fontsize=TICK_SIZE - 1.5, ha='right', va='top',
                    color=cond_color, alpha=0.80)

        axes[row_idx, 0].text(
            -0.36, 0.5, cond_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=TICK_SIZE, fontweight='bold', color=cond_color,
            ha='right', va='center', rotation=90, linespacing=1.3,
        )

    fig.text(0.53, 0.015, 'Distance from boundary (mm)',
             ha='center', fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, 'R (µm)',
             va='center', rotation=90, fontsize=LABEL_SIZE)

    stem = str(OUT_DIR / 'FigureS3_Rd_all_trials')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'\nSaved: {stem}.pdf/.svg')


if __name__ == '__main__':
    main()
