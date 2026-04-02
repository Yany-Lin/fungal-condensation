#!/usr/bin/env python3
"""Panel E (FigureRSR): KM survival curves at 5 distances, RSR2 exemplar.
95% CI from Greenwood's formula; single-frame blobs excluded."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import KaplanMeierFitter

THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR   = THIS_DIR.parent / 'raw_data'
TRACK_CSV = RAW_DIR / 'droplet_tracks' / 'RSR2_droplet_tracks.csv'

DIST_UM    = [200, 600, 1000, 1800, 2600]
DIST_BIN   = 400
MIN_DUR    = 0.5
MIN_N      = 10

_CMAP       = plt.colormaps['plasma']
BAND_COLORS = [_CMAP(i / (len(DIST_UM) - 1)) for i in range(len(DIST_UM))]

MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0;  LW = 0.6

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TS,
    'axes.linewidth': LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'svg.fonttype': 'none',
})


def style_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TS)


def main():
    df = pd.read_csv(TRACK_CSV)
    df = df[df['duration_min'] > MIN_DUR].copy()   # drop single-frame artifacts

    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.16)

    kmf = KaplanMeierFitter()
    legend_handles = []
    legend_labels  = []

    print('RSR2 KM τ₅₀ per distance band:')
    for d_um, color in zip(DIST_UM, BAND_COLORS):
        sub = df[np.abs(df['dist_um'] - d_um) <= DIST_BIN / 2].copy()
        if len(sub) < MIN_N:
            print(f'  {d_um/1000:.1f} mm: n={len(sub)} (skip)')
            continue

        kmf.fit(sub['duration_min'], event_observed=~sub['censored'])
        sf  = kmf.survival_function_
        ci  = kmf.confidence_interval_
        t   = sf.index.values
        s   = sf.iloc[:, 0].values
        lo  = ci.iloc[:, 0].values
        hi  = ci.iloc[:, 1].values

        line, = ax.plot(t, s, color=color, lw=1.4,
                        drawstyle='steps-post', zorder=3)
        ax.fill_between(t, lo, hi, color=color, alpha=0.12,
                        linewidth=0, step='post', zorder=2)
        legend_handles.append(line)
        legend_labels.append(f'{d_um / 1000:.1f} mm')

        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            ax.plot(t50, 0.5, 'o', color=color, ms=4.5,
                    markeredgecolor='white', markeredgewidth=0.4, zorder=5)
        print(f'  {d_um/1000:.1f} mm: n={len(sub)}  τ₅₀ = {t50:.2f} min')

    ax.axhline(0.5, color='gray', ls='--', lw=0.7, alpha=0.7, zorder=1)
    ax.text(0.04, 0.52, r'$\tau_{50}$', transform=ax.transAxes,
            color='gray', fontsize=TS + 1.5, fontstyle='italic',
            va='bottom', ha='left')

    ax.set_xlabel(r'$\tau$ (min)', fontsize=LS, labelpad=3)
    ax.set_ylabel('Fraction surviving', fontsize=LS, labelpad=3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    style_ax(ax)

    ax.legend(legend_handles, legend_labels,
              title='Distance from boundary', title_fontsize=TS - 0.5,
              fontsize=TS - 0.5, loc='upper right', framealpha=0.0,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.3)
    ax.text(-0.20, 1.05, 'E', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')

    for ext in ('.svg', '.pdf', '.png'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        kw['dpi'] = 600 if ext != '.png' else 300
        fig.savefig(OUTPUT_DIR / f'panel_E{ext}', **kw)
    plt.close(fig)
    print(f'Saved → {OUTPUT_DIR}/panel_E.*')


if __name__ == '__main__':
    main()
