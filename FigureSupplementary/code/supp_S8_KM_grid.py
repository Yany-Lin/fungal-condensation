#!/usr/bin/env python3
"""Supplementary Figure S8: All-20-trial Kaplan-Meier survival curves (4x5 grid).

Each subplot shows KM curves at 4 distance bands with tau50 markers.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import (
    DELTA, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    apply_style, clean_axes, save_fig,
)
from pathlib import Path

HG_CONDITIONS = [
    ('agar',   ['agar.1','agar.2','agar.3','agar.4','agar.5'],
     'Agar', '#9E9E9E'),
    ('0.5to1', ['0.5to1.2','0.5to1.3','0.5to1.4','0.5to1.5','0.5to1.7'],
     '0.5:1 NaCl', '#E67E22'),
    ('1to1',   ['1to1.1','1to1.2','1to1.3','1to1.4','1to1.5'],
     '1:1 NaCl', '#5B8FC9'),
    ('2to1',   ['2to1.1','2to1.2','2to1.3','2to1.4','2to1.5'],
     '2:1 NaCl', '#C0392B'),
]

T_SEED      = 900
MIN_FRAMES  = 3
DIST_BANDS  = [
    (600,  1200, '#E57373', '0.9 mm'),
    (1200, 1800, '#FFB74D', '1.5 mm'),
    (1800, 2400, '#81C784', '2.1 mm'),
    (2600, 3200, '#64B5F6', '2.9 mm'),
]
MIN_TRACKS_BIN = 10

BASE = Path(__file__).resolve().parents[2]
TRACK_DIR = BASE / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
OUT_DIR_S8 = OUTPUT_DIR / 'S8'
OUT_DIR_S8.mkdir(parents=True, exist_ok=True)


def plot_one_km(ax, tid):
    """Plot KM survival curves for one trial at 4 distance bands."""
    path = TRACK_DIR / f'{tid}_track_histories.csv'
    if not path.exists():
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=TICK_SIZE - 1, color='grey')
        return

    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd'] = (df['t_death_s'] - T_SEED) / 60.0  # minutes
    df = df[df['tau_fwd'] > 0]

    for d_lo, d_hi, color, label in DIST_BANDS:
        band = df[(df['distance_um'] >= d_lo) & (df['distance_um'] < d_hi)]
        if len(band) < MIN_TRACKS_BIN:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(band['tau_fwd'], event_observed=~band['censored'])

        ax.step(kmf.survival_function_.index,
                kmf.survival_function_.values.flatten(),
                where='post', color=color, lw=0.9, label=label)

        median = kmf.median_survival_time_
        if np.isfinite(median):
            ax.plot(median, 0.5, 'o', color=color, ms=3, zorder=5)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', ls=':', lw=0.4, alpha=0.5)


def main():
    apply_style()

    fig, axes = plt.subplots(4, 5, figsize=(180 * MM, 155 * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.07,
                         hspace=0.30, wspace=0.12)

    for row_idx, (cond_key, trial_ids, cond_label, cond_color) in enumerate(HG_CONDITIONS):
        for col_idx, tid in enumerate(trial_ids):
            ax = axes[row_idx, col_idx]
            print(f'  [{row_idx},{col_idx}] {tid} ... ', end='', flush=True)
            plot_one_km(ax, tid)
            print('done')

            clean_axes(ax)
            ax.tick_params(labelsize=TICK_SIZE - 1.5, pad=1.5)

            if row_idx == 0:
                ax.set_title(f'Rep {col_idx + 1}', fontsize=TICK_SIZE, pad=3)

        axes[row_idx, 0].text(
            -0.35, 0.5, cond_label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=TICK_SIZE - 0.5, fontweight='bold', color=cond_color,
            ha='right', va='center', rotation=90,
        )

    axes[0, 0].legend(fontsize=TICK_SIZE - 2.5, loc='upper right',
                       frameon=False, handlelength=1.0, labelspacing=0.2)

    fig.text(0.53, 0.01, 'Forward lifetime (min)', ha='center',
             fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, 'Survival S(t)', va='center', rotation=90,
             fontsize=LABEL_SIZE)

    stem = str(OUT_DIR_S8 / 'FigureS8_KM_survival')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'\nSaved: {stem}.png/.pdf/.svg')


if __name__ == '__main__':
    main()
