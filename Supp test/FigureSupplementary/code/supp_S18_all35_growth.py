#!/usr/bin/env python3
"""Supplementary Figure S18: All-35 growth curves with per-trial β fits.

7×5 grid: log–log R̄(t) vs t for every trial, distance-binned curves on
the plasma colormap. Each panel annotates the near-bin power-law slope
β (log–log fit on t = 5–15 min). Reference dashed lines mark β = 1/3
(diffusion-limited) and β = 1 (coalescence-dominated).
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    CONDITIONS, DELTA, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    apply_style, clean_axes, save_fig,
)

HG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureHGAggregate/raw_data/aggregate_edt')
FG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureFungi/raw_data/aggregate_edt')

BIN_WIDTH_UM = 300
MIN_DROPS = 5
T_MIN, T_MAX = 5.0, 15.0
DELTA_UM = 900   # boundary-layer assumption: only fit beyond near-source dry zone
NEAR_BIN_LIM = 1500  # near-bin = first 600 µm beyond DELTA_UM
R_CBAR_MAX = 2.0
R_ANCHOR = 20.0
T_ANCHOR = 5.0

OUT_DIR = OUTPUT_DIR / 'S18'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_near_beta(df_data):
    """Fit log–log slope (β) on the most-near distance bin."""
    near = df_data[(df_data['distance_um'] >= DELTA_UM) &
                   (df_data['distance_um'] < DELTA_UM + 600)]
    ts = []
    for t, frame in near.groupby('time_min'):
        R = frame['radius_um'].values
        if len(R) >= MIN_DROPS:
            ts.append((t, np.median(R)))
    if len(ts) < 5:
        return None
    t_arr = np.array([x[0] for x in ts])
    R_arr = np.array([x[1] for x in ts])
    if (R_arr > 0).all():
        sl, ic, r, p, se = stats.linregress(np.log(t_arr), np.log(R_arr))
        return float(sl)
    return None


def plot_panel(ax, tid, color):
    agg_dir = FG_AGG_DIR if tid.startswith(('Green', 'white', 'black')) else HG_AGG_DIR
    p = agg_dir / f'{tid}_edt_droplets.csv'
    if not p.exists():
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=TICK_SIZE - 1, color='gray')
        return None
    df = pd.read_csv(p)
    data = df[(df['time_min'] >= T_MIN) & (df['time_min'] <= T_MAX)
              & (df['radius_um'] > 0) & (df['distance_um'] >= DELTA_UM)].copy()
    if data.empty:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=TICK_SIZE - 1, color='gray')
        return None

    max_r = data['distance_um'].max()
    edges = np.arange(DELTA_UM, max_r + BIN_WIDTH_UM, BIN_WIDTH_UM)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    cmap = plt.cm.plasma
    for i in range(len(edges) - 1):
        in_bin = data[(data['distance_um'] >= edges[i]) &
                      (data['distance_um'] < edges[i + 1])]
        if in_bin.empty: continue
        ts = []
        for t, frame in in_bin.groupby('time_min'):
            R = frame['radius_um'].values
            if len(R) >= MIN_DROPS:
                ts.append((t, np.median(R)))
        if len(ts) < 3: continue
        t_arr = np.array([x[0] for x in ts])
        R_arr = np.array([x[1] for x in ts])
        c = cmap(bin_centers[i] / 1000 / R_CBAR_MAX)
        ax.plot(t_arr, R_arr, '-', color=c, lw=0.7, alpha=0.85)

    # reference slopes anchored at (5, 20)
    t_ref = np.linspace(T_MIN, T_MAX, 200)
    ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** (1 / 3),
            'k--', lw=0.5, alpha=0.55)
    ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** 1.0,
            color='#C0392B', ls='--', lw=0.5, alpha=0.55)

    beta = fit_near_beta(data)
    if beta is not None:
        ax.text(0.97, 0.06, f'$\\beta$={beta:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=TICK_SIZE - 1.5,
                color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.7))

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(5, 100)
    ax.set_xticks([5, 10, 15])
    ax.set_xticklabels(['5', '10', '15'])
    ax.set_yticks([10, 100])
    ax.set_yticklabels(['10', '100'])
    clean_axes(ax)
    ax.tick_params(labelsize=TICK_SIZE - 1.5, pad=1.5)
    ax.text(0.04, 0.96, tid, transform=ax.transAxes,
            ha='left', va='top', fontsize=TICK_SIZE - 1.5,
            color=color, alpha=0.85)
    return beta


def main():
    apply_style()
    fig, axes = plt.subplots(7, 5, figsize=(190 * MM, 240 * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.06,
                        hspace=0.30, wspace=0.10)
    betas = {}
    for r, (key, ids, label, color) in enumerate(CONDITIONS):
        for c, tid in enumerate(ids):
            beta = plot_panel(axes[r, c], tid, color)
            if beta is not None:
                betas.setdefault(label, []).append(beta)
        axes[r, 0].text(-0.30, 0.5, label, transform=axes[r, 0].transAxes,
                        fontsize=TICK_SIZE + 0.5, fontweight='bold',
                        color=color, ha='right', va='center', rotation=90)

    fig.text(0.53, 0.025, 'Time (min)', ha='center', fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, r'Mean $\bar{R}$ ($\mu$m)',
             va='center', rotation=90, fontsize=LABEL_SIZE)

    save_fig(fig, str(OUT_DIR / 'FigureS18_all35_growth'))
    plt.close(fig)
    print('Saved S18; per-condition near-bin β:')
    for k, v in betas.items():
        print(f'  {k:<14}  β = {np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}  (n={len(v)})')


if __name__ == '__main__':
    main()
