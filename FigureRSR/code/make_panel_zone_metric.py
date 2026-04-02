#!/usr/bin/env python3
"""
Survival-gradient metric  (τ₅₀,far − τ₅₀,near) / τ₅₀,mid  vs  δ (µm)
for the four hydrogel groups, styled to match panel L.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'

METRICS_CSV = OUTPUT_DIR / 'rsr_and_lab_dstar_metrics.csv'

MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0;  LW = 0.6

COLORS      = {'Agar': '#3A9E6F', '0.5:1': '#E67E22', '1:1': '#5B8FC9', '2:1': '#C0392B'}
GROUP_ORDER = ['Agar', '0.5:1', '1:1', '2:1']

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
    df = pd.read_csv(METRICS_CSV)
    hg = df[(df['system'] == 'Hydrogel') & df['delta_um'].notna() & df['zone_metric'].notna()].copy()

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.16)

    x_all, y_all = [], []

    for grp in GROUP_ORDER:
        sub = hg[hg['group'] == grp]
        for _, row in sub.iterrows():
            x = row['delta_um']
            y = row['zone_metric']
            x_all.append(x)
            y_all.append(y)
            jitter = rng.uniform(-10, 10)
            ax.scatter(x + jitter, y, c=COLORS[grp], s=35, alpha=0.65,
                       edgecolors='white', linewidths=0.4, zorder=3, label=grp)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered = {g: by_label[g] for g in GROUP_ORDER if g in by_label}
    ax.legend(ordered.values(), ordered.keys(),
              fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3)

    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() >= 3:
        sl, ic, r, p, _ = stats.linregress(x_arr[valid], y_arr[valid])
        r2 = r ** 2
        xfit = np.linspace(0, 1100, 200)
        ax.plot(xfit, ic + sl * xfit, 'k--', lw=1.2, alpha=0.7, zorder=2)
        ax.text(0.95, 0.08, f'$R^2$ = {r2:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)
        print(f'  zone_metric vs δ: R²={r2:.3f}, slope={sl:.6f}, p={p:.2e}')

    ax.set_xlabel(r'$\delta$ (µm)', fontsize=LS, labelpad=3)
    ax.set_ylabel(
        r'$(\tau_{50,\mathrm{far}} - \tau_{50,\mathrm{near}})\,/\,\tau_{50,\mathrm{mid}}$',
        fontsize=LS, labelpad=3)
    ax.set_xlim(0, 1100)
    ax.set_ylim(bottom=0)
    style_ax(ax)
    ax.set_box_aspect(1)
    ax.text(-0.18, 1.05, 'A', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')

    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(OUTPUT_DIR / f'panel_zone_metric{ext}',
                    bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)
    print(f'  Saved → {OUTPUT_DIR}/panel_zone_metric.*')


if __name__ == '__main__':
    main()
