#!/usr/bin/env python3
"""Supplementary Figure S31: Per-trial Kaplan-Meier survival curves for the
six Gymnosporangium yamadae field trials.

2x3 grid: top row = healthy (RSR1, RSR2, RSR7), bottom row = diseased
(RSRDiseased3, 5, 6). Each panel shows survival fraction vs lifetime,
stratified by distance bands from the rust boundary. Mirrors main paper
Fig 5E (which shows just one representative) for all 6 field trials.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, apply_style, save_fig

T7_RSR = Path('/Volumes/T7/Fungal Hygroscopy/RAW/RSR RAW')
RSR_FOLDERS = {
    'RSR1': T7_RSR / 'RSR 1', 'RSR2': T7_RSR / 'RSR 2',
    'RSR7': T7_RSR / 'RSR10',
    'RSRDiseased3': T7_RSR / 'RSR 3', 'RSRDiseased5': T7_RSR / 'RSR 5',
    'RSRDiseased6': T7_RSR / 'RSR 6',
}

ZONE_CENTERS = [0.3, 0.7, 1.1, 1.7, 2.5, 3.3]
BIN_W = 0.4
HEALTHY = ['RSR1', 'RSR2', 'RSR7']
DISEASED = ['RSRDiseased3', 'RSRDiseased5', 'RSRDiseased6']
LABELS = {'RSR1': 'Healthy 1', 'RSR2': 'Healthy 2', 'RSR7': 'Healthy 3',
          'RSRDiseased3': 'Diseased 1', 'RSRDiseased5': 'Diseased 2',
          'RSRDiseased6': 'Diseased 3'}

OUT = OUTPUT_DIR / 'S31'
OUT.mkdir(parents=True, exist_ok=True)


def plot_panel(ax, sample, show_legend=False, show_ylabel=False, show_xlabel=False):
    folder = RSR_FOLDERS[sample]
    tracks_csv = folder / 'Every 30s from Evap' / 'survival_analysis' / 'droplet_tracks.csv'
    if not tracks_csv.exists():
        ax.text(0.5, 0.5, f'no data\n({sample})', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        ax.axis('off')
        return

    tracks = pd.read_csv(tracks_csv)
    cmap = plt.colormaps['plasma']
    n_zones = len(ZONE_CENTERS)
    legend_handles = []

    for i, ctr in enumerate(ZONE_CENTERS):
        lo = ctr - BIN_W / 2
        hi = ctr + BIN_W / 2
        color = cmap(i / (n_zones - 1))
        sub = tracks[(tracks['dist_um'] >= lo * 1000) & (tracks['dist_um'] < hi * 1000)]
        if len(sub) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['duration_min'], event_observed=~sub['censored'])
        t = kmf.survival_function_.index.values
        s = kmf.survival_function_['KM_estimate'].values
        line, = ax.step(t, s, where='post', color=color, lw=1.4, alpha=0.9, zorder=3)
        if show_legend:
            legend_handles.append((line, f'{ctr:.1f} mm'))

    ax.axhline(0.5, color='gray', ls='--', lw=0.6, alpha=0.5, zorder=1)
    ax.set_ylim(0, 1.10)
    ax.set_xlim(0, 10)
    ax.tick_params(labelsize=TICK_SIZE - 1, pad=2)
    if show_ylabel:
        ax.set_ylabel('Fraction surviving', fontsize=LABEL_SIZE)
    if show_xlabel:
        ax.set_xlabel('Lifetime (min)', fontsize=LABEL_SIZE)
    ax.text(0.97, 0.97, LABELS[sample], transform=ax.transAxes,
            ha='right', va='top', fontsize=TICK_SIZE - 0.5, fontweight='bold')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    if show_legend and legend_handles:
        ax.legend(*zip(*legend_handles),
                  title='Distance from\nboundary',
                  fontsize=TICK_SIZE - 2, title_fontsize=TICK_SIZE - 2,
                  loc='lower left', frameon=False, handlelength=1.2)


def main():
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 110 * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10,
                        hspace=0.20, wspace=0.10)

    for col, sample in enumerate(HEALTHY):
        plot_panel(axes[0, col], sample,
                   show_legend=(col == 0),
                   show_ylabel=(col == 0))
    for col, sample in enumerate(DISEASED):
        plot_panel(axes[1, col], sample,
                   show_ylabel=(col == 0),
                   show_xlabel=True)

    fig.text(0.5, 0.985, 'Healthy', ha='center', va='top',
             fontsize=LABEL_SIZE + 1, fontweight='bold', color='#2E86AB')
    fig.text(0.5, 0.515, 'Diseased', ha='center', va='top',
             fontsize=LABEL_SIZE + 1, fontweight='bold', color='#A0522D')

    save_fig(fig, str(OUT / 'FigureS31_field_KM'))
    plt.close(fig)
    print(f'Saved S31 to {OUT}')


if __name__ == '__main__':
    main()
