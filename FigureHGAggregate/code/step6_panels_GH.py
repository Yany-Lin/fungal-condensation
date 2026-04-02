#!/usr/bin/env python3
"""R(d) exemplar scatter with zone shading, and zone-based size gradient vs water activity."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR    = Path(__file__).parent
PROJECT_DIR = THIS_DIR.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HG_METRICS = OUTPUT_DIR / 'hydrogel_metrics.csv'
AGG_DIR    = PROJECT_DIR / 'raw_data' / 'aggregate_edt'

COLORS ={'Agar': '#3A9E6F', '0.5:1': '#E67E22', '1:1': '#5B8FC9', '2:1': '#C0392B'}

EXEMPLAR_AGAR  = 'agar.4'
EXEMPLAR_0p5   = '0.5to1.2'
EXEMPLAR_1to1  = '1to1.1'
EXEMPLAR_NACL  = '2to1.2'

T_WINDOW     = (14.5, 15.5)
BIN_WIDTH_UM = 100
MIN_DROPS    = 3
MAX_DIST_MM  = 2.5

N_NEAR_BINS  = 5
D_MID_LO     = 900;   D_MID_HI  = 1100
D_FAR_LO     = 1900;  D_FAR_HI  = 2100

MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0
LW = 0.6

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


def load_droplets(trial_id):
    path = AGG_DIR / f'{trial_id}_edt_droplets.csv'
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    tw = df[(df['time_min'] >= T_WINDOW[0]) &
            (df['time_min'] <= T_WINDOW[1]) &
            (df['radius_um'] > 0)].copy()
    return tw


def binned_profile(tw):
    bins = np.arange(0, tw['distance_um'].max() + BIN_WIDTH_UM, BIN_WIDTH_UM)
    tw = tw.copy()
    tw['bin'] = pd.cut(tw['distance_um'], bins=bins,
                       labels=bins[:-1] + BIN_WIDTH_UM / 2).astype(float)
    grp = tw.groupby('bin')['radius_um']
    prof = grp.agg(r='mean', sd='std', n='count').reset_index()
    prof = prof[prof['n'] >= MIN_DROPS].sort_values('bin')
    prof['sem'] = (prof['sd'] / np.sqrt(prof['n'])).fillna(0)
    return prof['bin'].values, prof['r'].values, prof['sem'].values


def compute_zone_metric(trial_id, delta_um):
    """(R_far - R_near) / R_mid. Empty near bins count as R=0 (dry zone suppression)."""
    tw = load_droplets(trial_id)
    if len(tw) < 30:
        return None

    max_d = tw['distance_um'].max()
    bins = np.arange(0, max_d + BIN_WIDTH_UM, BIN_WIDTH_UM)
    tw2 = tw.copy()
    tw2['bin'] = pd.cut(tw2['distance_um'], bins=bins,
                        labels=bins[:-1] + BIN_WIDTH_UM / 2).astype(float)
    grp = tw2.groupby('bin')['radius_um'].mean()

    near_centers = [BIN_WIDTH_UM / 2 + i * BIN_WIDTH_UM
                    for i in range(N_NEAR_BINS)]
    near_vals = [grp[c] if c in grp.index else 0.0 for c in near_centers]
    r_near = np.mean(near_vals)

    mid_vals = grp[(grp.index >= D_MID_LO) & (grp.index <= D_MID_HI)]
    if len(mid_vals) < 1:
        return None
    r_mid = mid_vals.mean()

    far_vals = grp[(grp.index >= D_FAR_LO) & (grp.index <= D_FAR_HI)]
    if len(far_vals) < 1:
        return None
    r_far = far_vals.mean()

    if r_mid <= 0:
        return None

    metric = (r_far - r_near) / r_mid
    return {'r_near': r_near, 'r_mid': r_mid, 'r_far': r_far,
            'metric': metric}



def plot_panel_G(ax, hg):
    for tid, color, label in [
            (EXEMPLAR_NACL,  COLORS['2:1'],   '2:1 NaCl'),
            (EXEMPLAR_1to1,  COLORS['1:1'],   '1:1 NaCl'),
            (EXEMPLAR_0p5,   COLORS['0.5:1'], '0.5:1 NaCl'),
            (EXEMPLAR_AGAR,  COLORS['Agar'],  'Agar')]:
        tw = load_droplets(tid)
        d_vals, r_vals, sem_vals = binned_profile(tw)
        xm = d_vals / 1000.0

        far_mask = (d_vals >= D_FAR_LO) & (d_vals <= D_FAR_HI)
        r_far = r_vals[far_mask].mean() if far_mask.any() else r_vals[-3:].mean()
        r_norm = r_vals / r_far
        sem_norm = sem_vals / r_far

        ax.errorbar(xm, r_norm, yerr=sem_norm,
                    fmt='-', color=color, lw=1.3,
                    capsize=1.5, elinewidth=0.5, capthick=0.5,
                    label=label, zorder=3)

        delta_rows = hg[hg['trial_id'] == tid]
        if len(delta_rows) > 0:
            delta_mm = delta_rows.iloc[0]['delta_um'] / 1000.0
            ax.axvline(delta_mm, color=color, ls=':', lw=1.2, alpha=0.85, zorder=2)

    ax.axvspan(0, N_NEAR_BINS * BIN_WIDTH_UM / 1000,
               color='#E57373', alpha=0.10, zorder=0)
    ax.axvspan(D_MID_LO / 1000, D_MID_HI / 1000,
               color='#FF9800', alpha=0.10, zorder=0)
    ax.axvspan(D_FAR_LO / 1000, D_FAR_HI / 1000,
               color='#2196F3', alpha=0.08, zorder=0)

    ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5, zorder=0)

    ax.text(N_NEAR_BINS * BIN_WIDTH_UM / 2000, 1.62, 'Near',
            ha='center', va='bottom', fontsize=TS, color='#C62828',
            fontstyle='italic')
    ax.text((D_MID_LO + D_MID_HI) / 2000, 1.62, 'Mid',
            ha='center', va='bottom', fontsize=TS, color='#E65100',
            fontstyle='italic')
    ax.text((D_FAR_LO + D_FAR_HI) / 2000, 1.62, 'Far',
            ha='center', va='bottom', fontsize=TS, color='#1565C0',
            fontstyle='italic')

    ax.set_xlabel('Distance from source (mm)', fontsize=LS, labelpad=3)
    ax.set_ylabel(r'$R\,/\,R_{\mathrm{far}}$', fontsize=LS, labelpad=3)
    ax.set_xlim(0, MAX_DIST_MM)
    ax.set_ylim(0, 1.7)
    ax.legend(fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3)
    style_ax(ax)
    ax.text(-0.18, 1.05, 'E', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')



def plot_panel_H(ax, hg):
    x_all, y_all = [], []

    for _, row in hg.iterrows():
        zm = compute_zone_metric(row['trial_id'], row['delta_um'])
        if zm is None:
            continue
        group = row['hydrogel_type']
        group_label = {'agar': 'Agar', '0.5:1': '0.5:1', '1:1': '1:1', '2:1': '2:1'}[group]
        x_all.append(row['delta_um'])
        y_all.append(zm['metric'])

        ax.scatter(row['delta_um'], zm['metric'],
                   c=COLORS[group_label], s=35, alpha=0.65,
                   edgecolors='white', linewidths=0.4, zorder=3,
                   label=group_label)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3)

    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    valid = ~np.isnan(y_arr)
    sl, ic, r, p, se = stats.linregress(x_arr[valid], y_arr[valid])
    r2 = r**2
    xfit = np.linspace(0, 1100, 200)
    ax.plot(xfit, ic + sl * xfit, 'k--', lw=1.2, alpha=0.7, zorder=2)
    ax.text(0.95, 0.08, f'$R^2$ = {r2:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)

    ax.set_xlabel(r'$\delta$ (µm)', fontsize=LS, labelpad=3)
    ax.set_ylabel(r'$(R_{\mathrm{far}} - R_{\mathrm{near}})\,/\,R_{\mathrm{mid}}$',
                  fontsize=LS, labelpad=3)
    ax.set_xlim(0, 1100)
    style_ax(ax)
    ax.text(-0.18, 1.05, 'H', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')

    print(f"  Panel H: R\u00b2={r2:.3f}, slope={sl:.3f}, p={p:.2e}")



def main():
    hg = pd.read_csv(HG_METRICS)
    hg = hg.reset_index(drop=True)
    print(f"Loaded {len(hg)} hydrogel trials")

    fig, (ax_g, ax_h) = plt.subplots(1, 2, figsize=(170 * MM, 75 * MM))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.17,
                        wspace=0.38)

    plot_panel_G(ax_g, hg)
    plot_panel_H(ax_h, hg)

    for ext in ('.png', '.pdf', '.svg'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(OUTPUT_DIR / f'panels_EH{ext}',
                    bbox_inches='tight', facecolor='white',
                    dpi=dpi if dpi else 'figure')
    plt.close(fig)
    print(f"\nSaved to {OUTPUT_DIR}/panels_EH.*")


if __name__ == '__main__':
    main()
