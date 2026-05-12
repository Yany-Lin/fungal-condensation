#!/usr/bin/env python3
"""Supplementary Figure S5: Near-field surface coverage suppression.

Shows epsilon(t) for near-field bins across exemplar trials, demonstrating
that the hygroscopic sink suppresses coverage during condensation and
accelerates evaporation in the near field.

Panel A: epsilon(t) near-field (r' <= 300 um) for 6 exemplar trials
Panel B: epsilon(t) far-field  (r' = 1800-2100 um) for same trials
Panel C: droplet count N(t) near-field for same trials
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
    LW_SPINE, apply_style, clean_axes, save_fig,
)

EXEMPLARS = {
    'agar':   'agar.3',
    '0.5to1': '0.5to1.3',
    '1to1':   '1to1.2',
    '2to1':   '2to1.2',
    'Green':  'Green.1',
    'black':  'black.3',
    'white':  'white.1',
}

NEAR_R_MAX = 300
FAR_R_LO   = 1800   # r' = 1800-2100 um
FAR_R_HI   = 2100

OUT_DIR = OUTPUT_DIR / 'S5'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_zone_timeseries(ts_df, trial_id, r_lo, r_hi):
    """Aggregate epsilon, count, and R for a distance zone over time."""
    sub = ts_df[(ts_df['trial_id'] == trial_id) &
                (ts_df['r_prime_um'] >= r_lo) &
                (ts_df['r_prime_um'] <= r_hi)]
    if sub.empty:
        return pd.DataFrame()
    agg = sub.groupby('time_min').agg(
        epsilon=('epsilon', 'mean'),
        n_droplets=('n_droplets', 'sum'),
        R_median=('R_median', 'mean'),
    ).reset_index()
    return agg


def main():
    apply_style()

    ts = pd.read_csv(OUT_DIR.parent / 'S3' / 'beysens_all_timeseries.csv')

    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 70 * MM))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18,
                         wspace=0.35)

    for cond_key, trial_ids, cond_label, cond_color in CONDITIONS:
        tid = EXEMPLARS[cond_key]
        lw = 1.6 if cond_key in ('agar', '2to1') else 0.9
        alpha = 1.0 if cond_key in ('agar', '2to1') else 0.6

        # Near field
        near = load_zone_timeseries(ts, tid, 0, NEAR_R_MAX)
        if not near.empty:
            axes[0].plot(near['time_min'], near['epsilon'],
                         '-', color=cond_color, lw=lw, alpha=alpha,
                         label=cond_label)
            axes[2].plot(near['time_min'], near['n_droplets'],
                         '-', color=cond_color, lw=lw, alpha=alpha,
                         label=cond_label)

        # Far field
        far = load_zone_timeseries(ts, tid, FAR_R_LO, FAR_R_HI)
        if not far.empty:
            axes[1].plot(far['time_min'], far['epsilon'],
                         '-', color=cond_color, lw=lw, alpha=alpha,
                         label=cond_label)

    ax = axes[0]
    ax.set_xlabel('Time (min)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Surface coverage  ε', fontsize=LABEL_SIZE)
    ax.set_title("Near field (r' ≤ 0.3 mm)", fontsize=LABEL_SIZE, pad=4)
    ax.set_ylim(0, 0.20)
    ax.set_xlim(4, 20)
    ax.axvline(15.0, color='grey', ls=':', lw=0.6, alpha=0.5)
    ax.text(0.03, 0.95, 'A', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')
    clean_axes(ax)
    ax.tick_params(labelsize=TICK_SIZE)

    ax = axes[1]
    ax.set_xlabel('Time (min)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Surface coverage  ε', fontsize=LABEL_SIZE)
    ax.set_title("Far field (r' = 1.8–2.1 mm)", fontsize=LABEL_SIZE, pad=4)
    ax.set_ylim(0, 0.20)
    ax.set_xlim(4, 20)
    ax.axvline(15.0, color='grey', ls=':', lw=0.6, alpha=0.5)
    ax.text(0.03, 0.95, 'B', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')
    clean_axes(ax)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=TICK_SIZE - 1.5, loc='upper right', frameon=False,
              handlelength=1.5, labelspacing=0.3)

    ax = axes[2]
    ax.set_xlabel('Time (min)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Droplet count (near field)', fontsize=LABEL_SIZE)
    ax.set_title("Near-field nucleation", fontsize=LABEL_SIZE, pad=4)
    ax.set_xlim(4, 20)
    ax.axvline(15.0, color='grey', ls=':', lw=0.6, alpha=0.5)
    ax.text(0.03, 0.95, 'C', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')
    clean_axes(ax)
    ax.tick_params(labelsize=TICK_SIZE)

    stem = str(OUT_DIR / 'FigureS5_epsilon_near_far')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'Saved: {stem}.png/.pdf/.svg')


if __name__ == '__main__':
    main()
