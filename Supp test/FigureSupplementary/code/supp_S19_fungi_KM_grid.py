#!/usr/bin/env python3
"""Supplementary Figure S19: Per-trial KM survival curves for the 15 fungal trials.

3x5 grid (Aspergillus / Rhizopus / Mucor x 5 reps each), mirroring SuppFig4
but for fungi. Same 4 distance bands and stratification logic as SuppFig4.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    apply_style, clean_axes, save_fig,
)

FUNGI_CONDITIONS = [
    ('Aspergillus', ['Green.1','Green.2','Green.3','Green.4','Green.5'], '#4CAF50'),
    ('Rhizopus',    ['black.1','black.2','black.3','black.4','black.5'], '#212121'),
    ('Mucor',       ['white.1','white.2','white.3','white.4','white.5'], '#757575'),
]

T_SEED = 900
MIN_FRAMES = 3
DIST_BANDS = [
    (600,  1200, '#E57373', '0.9 mm'),
    (1200, 1800, '#FFB74D', '1.5 mm'),
    (1800, 2400, '#81C784', '2.1 mm'),
    (2600, 3200, '#64B5F6', '2.9 mm'),
]
MIN_TRACKS_BIN = 10
TRACK_DIR = Path('/Volumes/T7/FINAL OSF/FigureHGAggregate/code/test_tracking/output')

OUT_DIR = OUTPUT_DIR / 'S19'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_one_km(ax, tid):
    path = TRACK_DIR / f'{tid}_track_histories.csv'
    if not path.exists():
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=TICK_SIZE - 1, color='gray')
        return
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd'] = (df['t_death_s'] - T_SEED) / 60.0
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
    fig, axes = plt.subplots(3, 5, figsize=(180 * MM, 120 * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.10,
                        hspace=0.30, wspace=0.12)
    for row_idx, (label, ids, color) in enumerate(FUNGI_CONDITIONS):
        for col_idx, tid in enumerate(ids):
            ax = axes[row_idx, col_idx]
            plot_one_km(ax, tid)
            clean_axes(ax)
            ax.tick_params(labelsize=TICK_SIZE - 1.5, pad=1.5)
            if row_idx == 0:
                ax.set_title(f'Rep {col_idx + 1}', fontsize=TICK_SIZE, pad=3)
        axes[row_idx, 0].text(-0.35, 0.5, label,
                              transform=axes[row_idx, 0].transAxes,
                              fontsize=TICK_SIZE, fontweight='bold', color=color,
                              ha='right', va='center', rotation=90)
    axes[0, 0].legend(fontsize=TICK_SIZE - 2.5, loc='upper right',
                      frameon=False, handlelength=1.0, labelspacing=0.2,
                      title='Distance', title_fontsize=TICK_SIZE - 2.5)
    fig.text(0.53, 0.02, 'Forward lifetime (min)', ha='center', fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, 'Survival S(t)', va='center', rotation=90, fontsize=LABEL_SIZE)
    stem = str(OUT_DIR / 'FigureS19_fungi_KM')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'Saved: {stem}.pdf')


if __name__ == '__main__':
    main()
