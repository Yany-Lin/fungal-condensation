#!/usr/bin/env python3
"""Supplementary Figure S17: Per-trial Rd profiles, fungi only (3x5 grid).

Larger panels than the all-35 grid (SuppFig2) — dedicated view of the
15 fungal trials so reviewers can read each panel without zooming.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    DELTA, CONDITIONS,
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    apply_style, clean_axes, save_fig,
)
FG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureFungi/raw_data/aggregate_edt')

T_WINDOW = (14.5, 15.5)
BIN_WIDTH = 100
MIN_DROPS = 5
MAX_DIST = 2500

OUT_DIR = OUTPUT_DIR / 'S17_fungi'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# fungal rows from CONDITIONS in canonical order (Asp, Rhz, Muc)
FUNGI_ROWS = [c for c in CONDITIONS if c[0] in ('Green', 'black', 'white')]


def main():
    apply_style()
    fig, axes = plt.subplots(3, 5, figsize=(200 * MM, 130 * MM),
                              sharex=True, sharey=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.10,
                        hspace=0.40, wspace=0.30)

    for row_idx, (key, trial_ids, label, color) in enumerate(FUNGI_ROWS):
        for col_idx, tid in enumerate(trial_ids):
            ax = axes[row_idx, col_idx]
            delta = DELTA[tid]
            df = pd.read_csv(FG_AGG_DIR / f'{tid}_edt_droplets.csv')
            tw = df[(df['time_min'] >= T_WINDOW[0]) &
                    (df['time_min'] <= T_WINDOW[1])].copy()

            ax.scatter(tw['distance_um'] / 1000, tw['radius_um'],
                       s=0.3, alpha=0.06, color='#999999',
                       rasterized=True, edgecolors='none')
            bins = np.arange(0, MAX_DIST + BIN_WIDTH, BIN_WIDTH)
            tw['dbin'] = pd.cut(tw['distance_um'], bins=bins,
                                labels=(bins[:-1] + BIN_WIDTH / 2)).astype(float)
            grp = tw.groupby('dbin')['radius_um']
            prof = grp.agg(mean='mean',
                           sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)),
                           n='count').reset_index()
            prof = prof[prof['n'] >= MIN_DROPS]
            d_mm = prof['dbin'] / 1000

            ax.fill_between(d_mm, prof['mean'] - prof['sem'],
                            prof['mean'] + prof['sem'],
                            alpha=0.25, color=color, zorder=2)
            ax.plot(d_mm, prof['mean'], '-', color=color, lw=2.0, zorder=3)
            ax.axvline(delta / 1000, color=color, ls='--', lw=0.9, alpha=0.75, zorder=4)

            ax.set_xlim(0, MAX_DIST / 1000)
            ax.set_ylim(bottom=0)
            clean_axes(ax)
            ax.tick_params(labelsize=TICK_SIZE - 0.5, pad=2)
            ax.text(0.97, 0.97, f'Rep {col_idx + 1}', transform=ax.transAxes,
                    fontsize=TICK_SIZE - 1, ha='right', va='top',
                    color=color, alpha=0.85)

        axes[row_idx, 0].text(-0.32, 0.5, label,
                              transform=axes[row_idx, 0].transAxes,
                              fontsize=TICK_SIZE + 1, fontweight='bold',
                              color=color, ha='right', va='center', rotation=90)

    fig.text(0.53, 0.025, 'Distance from boundary (mm)',
             ha='center', fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, r'$\bar{R}$ ($\mu$m)',
             va='center', rotation=90, fontsize=LABEL_SIZE)

    stem = str(OUT_DIR / 'FigureS17_fungi_Rd')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.pdf')


if __name__ == '__main__':
    main()
