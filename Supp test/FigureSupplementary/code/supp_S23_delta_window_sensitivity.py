#!/usr/bin/env python3
"""Supplementary Figure S23: Sensitivity of dry-zone width δ to the
late-condensation averaging window.

Re-computes δ for each of the 35 lab trials at five time windows ranging
from (8-15) to (12-15) minutes, using the same 1st-percentile-of-distance
estimator that defines the canonical δ at (10-15). Two panels:
  A — δ_alt vs δ_canonical scatter for each alternate window.
  B — Per-trial δ across all 5 windows; confirms condition ordering is
      preserved across the sweep.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    DELTA, CONDITIONS,
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    apply_style, save_fig,
)

HG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureHGAggregate/raw_data/aggregate_edt')
FG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureFungi/raw_data/aggregate_edt')

WINDOWS = [(8, 15), (9, 15), (10, 15), (11, 15), (12, 15)]
CANONICAL_IDX = 2  # (10, 15)
PERCENTILE = 1.0   # 1st-percentile-of-distance proxy

OUT_DIR = OUTPUT_DIR / 'S23'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def trial_path(tid):
    if tid.startswith(('Green', 'white', 'black')):
        return FG_AGG_DIR / f'{tid}_edt_droplets.csv'
    return HG_AGG_DIR / f'{tid}_edt_droplets.csv'


def compute_delta(tid, t_lo, t_hi):
    p = trial_path(tid)
    if not p.exists(): return np.nan
    df = pd.read_csv(p)
    sub = df[(df['time_min'] >= t_lo) & (df['time_min'] <= t_hi) &
             (df['radius_um'] > 0)]
    if len(sub) < 10: return np.nan
    return float(np.percentile(sub['distance_um'].values, PERCENTILE))


def main():
    apply_style()

    # Compute delta at each window for each trial
    rows = []
    for key, ids, label, color in CONDITIONS:
        for tid in ids:
            for (t_lo, t_hi) in WINDOWS:
                d = compute_delta(tid, t_lo, t_hi)
                rows.append({'tid': tid, 'condition': label, 'color': color,
                             'window': f'{t_lo}-{t_hi}', 't_lo': t_lo,
                             'delta_um': d})
    df = pd.DataFrame(rows)
    canonical = df[df['t_lo'] == WINDOWS[CANONICAL_IDX][0]].set_index('tid')['delta_um']

    fig = plt.figure(figsize=(190 * MM, 95 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4],
                          left=0.08, right=0.97, top=0.90, bottom=0.18,
                          wspace=0.30)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # Panel A: alt vs canonical scatter
    win_markers = ['^', 's', 'o', 'D', 'v']  # 5 windows
    for w_idx, (t_lo, t_hi) in enumerate(WINDOWS):
        if w_idx == CANONICAL_IDX:
            continue
        sub = df[df['t_lo'] == t_lo].set_index('tid')
        x = canonical.values
        y = sub.loc[canonical.index, 'delta_um'].values
        for tid in canonical.index:
            color = sub.loc[tid, 'color']
            axA.scatter(canonical[tid], sub.loc[tid, 'delta_um'],
                        marker=win_markers[w_idx], s=18, c=color,
                        edgecolors='white', linewidths=0.3, alpha=0.75, zorder=3)

    lo = 0
    hi = max(canonical.max(), df['delta_um'].max()) * 1.05
    axA.plot([lo, hi], [lo, hi], '--', color='black', lw=0.8, alpha=0.6, zorder=1)
    axA.set_xlim(lo, hi); axA.set_ylim(lo, hi)
    axA.set_xlabel(r'$\delta$ at canonical window 10$-$15 min ($\mu$m)',
                   fontsize=LABEL_SIZE, labelpad=2)
    axA.set_ylabel(r'$\delta$ at alternate window ($\mu$m)',
                   fontsize=LABEL_SIZE, labelpad=2)
    axA.tick_params(labelsize=TICK_SIZE - 0.5)
    axA.text(-0.18, 1.05, 'A', transform=axA.transAxes,
             fontsize=11, fontweight='bold', va='top')
    for sp in ('top', 'right'): axA.spines[sp].set_visible(False)
    # window legend (markers only)
    win_handles = [plt.Line2D([0], [0], marker=win_markers[i], color='w',
                              markerfacecolor='gray', markeredgecolor='black',
                              markersize=5,
                              label=f'{w[0]}$-${w[1]} min')
                   for i, w in enumerate(WINDOWS) if i != CANONICAL_IDX]
    axA.legend(handles=win_handles, fontsize=TICK_SIZE - 1, loc='upper left',
               frameon=False, handletextpad=0.3, labelspacing=0.3,
               title='Alternate window', title_fontsize=TICK_SIZE - 1)

    # Panel B: per-trial delta across windows
    win_centers = np.array([np.mean(w) for w in WINDOWS])
    for tid in df['tid'].unique():
        sub = df[df['tid'] == tid].sort_values('t_lo')
        color = sub['color'].iloc[0]
        axB.plot(win_centers, sub['delta_um'].values, '-',
                 color=color, lw=0.6, alpha=0.55, zorder=2)
        axB.scatter(win_centers, sub['delta_um'].values, s=12, c=color,
                    edgecolors='white', linewidths=0.3, zorder=3)

    axB.axvline(np.mean(WINDOWS[CANONICAL_IDX]), color='black', ls='--', lw=0.8,
                alpha=0.5, zorder=1)
    axB.text(np.mean(WINDOWS[CANONICAL_IDX]) + 0.1,
             axB.get_ylim()[1] * 0.95, 'canonical', fontsize=TICK_SIZE - 1,
             ha='left', va='top', color='black', alpha=0.7)
    axB.set_xticks(win_centers)
    axB.set_xticklabels([f'{w[0]}$-${w[1]}' for w in WINDOWS])
    axB.set_xlabel('Time-window (min)', fontsize=LABEL_SIZE, labelpad=2)
    axB.set_ylabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=2)
    axB.tick_params(labelsize=TICK_SIZE - 0.5)
    axB.text(-0.13, 1.05, 'B', transform=axB.transAxes,
             fontsize=11, fontweight='bold', va='top')
    for sp in ('top', 'right'): axB.spines[sp].set_visible(False)

    # condition legend on B (same colors as everywhere)
    cond_handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=c[3], markeredgecolor='white',
                               markersize=5, label=c[2]) for c in CONDITIONS]
    axB.legend(handles=cond_handles, fontsize=TICK_SIZE - 1, loc='upper right',
               frameon=False, handletextpad=0.3, labelspacing=0.3, ncol=2)

    save_fig(fig, str(OUT_DIR / 'FigureS23_delta_window'))
    plt.close(fig)
    print('Saved S23')

    # Sanity: print canonical delta vs DELTA dict for a few trials
    print('\nCanonical (this run) vs DELTA dict:')
    for tid in ['agar.1', '2to1.1', 'Green.1', 'white.1', 'black.1']:
        canon = canonical.get(tid, np.nan)
        ref = DELTA.get(tid, np.nan)
        print(f'  {tid:<12}  this={canon:6.0f}  DELTA={ref:6.0f}')


if __name__ == '__main__':
    main()
