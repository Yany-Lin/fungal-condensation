#!/usr/bin/env python3
"""Strip plot of dry-zone width delta for all source types."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT   = THIS_DIR.resolve().parent.parent
HG_METRICS  = REPO_ROOT / 'FigureHGAggregate' / 'output' / 'hydrogel_metrics.csv'
F_METRICS   = OUTPUT_DIR / 'fungi_metrics.csv'

MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0;  LW = 0.6

COLORS = {
    'Agar':         '#3A9E6F',
    'Aspergillus':  '#4CAF50',
    'Mucor':        '#9E9E9E',
    'Rhizopus':     '#212121',
    '0.5:1 NaCl':   '#E67E22',
}

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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=TS, pad=2)


def main():
    hg = pd.read_csv(HG_METRICS)
    fm = pd.read_csv(F_METRICS)

    groups = {
        'Agar':        hg[hg['hydrogel_type'] == 'agar']['delta_um'].dropna().tolist(),
        'Aspergillus': fm[fm['species'] == 'Green']['delta_um'].dropna().tolist(),
        'Mucor':       fm[fm['species'] == 'White']['delta_um'].dropna().tolist(),
        'Rhizopus':    fm[fm['species'] == 'Black']['delta_um'].dropna().tolist(),
        '0.5:1 NaCl':  hg[hg['hydrogel_type'] == '0.5:1']['delta_um'].dropna().tolist(),
    }

    order = ['Agar', 'Aspergillus', 'Mucor', 'Rhizopus', '0.5:1 NaCl']

    fig, ax = plt.subplots(figsize=(85 * MM, 80 * MM))
    fig.subplots_adjust(left=0.22, right=0.95, top=0.92, bottom=0.22)

    rng = np.random.default_rng(42)

    for xi, grp in enumerate(order):
        vals = np.array(groups[grp])
        color = COLORS[grp]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=18, alpha=0.7,
                   edgecolors='white', linewidths=0.3, zorder=3)

        mean = vals.mean()
        sd   = vals.std(ddof=1)
        cross_w = 0.22
        ax.plot([xi - cross_w, xi + cross_w], [mean, mean],
                color=color, lw=2.0, solid_capstyle='round', zorder=4)
        ax.plot([xi, xi], [mean - sd, mean + sd],
                color=color, lw=2.0, solid_capstyle='round', zorder=4)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha='right', fontsize=TS)
    ax.set_ylabel(r'$\delta$ (µm)', fontsize=LS, labelpad=3)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(bottom=0)
    style_ax(ax)

    ax.text(-0.28, 1.05, 'F', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')

    for ext in ('.svg', '.pdf', '.png'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'panel_F{ext}', **kw)
    plt.close(fig)
    print(f'Saved → {OUTPUT_DIR}/panel_F.*')

    print('\nGroup means ± SEM:')
    for grp in order:
        vals = np.array(groups[grp])
        print(f'  {grp:<14}: n={len(vals)}  '
              f'δ = {vals.mean():.1f} ± {vals.std(ddof=1)/np.sqrt(len(vals)):.1f} µm')


if __name__ == '__main__':
    main()
