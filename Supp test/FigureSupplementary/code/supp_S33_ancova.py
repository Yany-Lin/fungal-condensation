#!/usr/bin/env python3
"""Supplementary Figure S33: ANCOVA test of universal scaling for the
zone-based size and survival gradients.

Formal model-comparison test for the universal-collapse claim in Fig. 3D-E.
Compares four candidate models for size_gradient ~ delta and dtau50_dr ~ delta:
  * Universal: y = a + b*delta  (single line, all 35 trials)
  * Parallel:  y = a_g + b*delta (separate intercepts, common slope, by system)
  * Two-line:  y = a_g + b_g*delta (separate intercepts and slopes, by system)
  * Six-line:  y = a_c + b_c*delta (free intercepts and slopes per condition)

For zone_metric, BIC strongly prefers Universal (lowest BIC, dBIC=0). For
dtau50_dr, the lower-evidence Two-line and Universal models are within
~1 BIC unit of each other (Akaike weights 0.57 vs 0.21), neither strongly
preferred. Combined: the size-gradient universality is statistically
strong; the survival-gradient universality is consistent with universality
but not formally distinguishable from a small genus offset at this n.

Data: additions/2_ANCOVA_universality/{model_comparison_*.csv, ancova_results.txt}
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
from supp_common import OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, apply_style, save_fig

ANCOVA_DIR = Path('/Volumes/T7/FINAL OSF/additions/2_ANCOVA_universality')
UNIV_CSV = Path('/Volumes/T7/FINAL OSF/FigureTable/output/universal_metrics.csv')
RSR_CSV = Path('/Volumes/T7/FINAL OSF/FigureRSR/output/rsr_and_lab_dstar_metrics.csv')

C_HYDRO = '#5B8FC9'
C_FUNGI = '#4CAF50'

OUT_DIR = OUTPUT_DIR / 'S33'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_separate(df, x_col, y_col):
    """Return (slope, intercept) for hydrogel and fungi subsets."""
    h = df[df['system'] == 'Hydrogel']
    f = df[df['system'] == 'Fungi']
    sl_h, ic_h, *_ = stats.linregress(h[x_col], h[y_col])
    sl_f, ic_f, *_ = stats.linregress(f[x_col], f[y_col])
    return (sl_h, ic_h), (sl_f, ic_f)


def fit_universal(df, x_col, y_col):
    sl, ic, *_ = stats.linregress(df[x_col], df[y_col])
    return sl, ic


def panel_regression(ax, df, x_col, y_col, ylabel, panel_letter, title,
                     model_table_path):
    h = df[df['system'] == 'Hydrogel']
    f = df[df['system'] == 'Fungi']
    ax.scatter(h[x_col], h[y_col], s=30, c=C_HYDRO, edgecolors='white',
               linewidths=0.4, label='Hydrogel', alpha=0.85, zorder=3)
    ax.scatter(f[x_col], f[y_col], s=30, c=C_FUNGI, edgecolors='white',
               linewidths=0.4, label='Fungi', alpha=0.85, zorder=3)

    (sl_h, ic_h), (sl_f, ic_f) = fit_separate(df, x_col, y_col)
    sl_u, ic_u = fit_universal(df, x_col, y_col)
    x_grid = np.linspace(df[x_col].min(), df[x_col].max(), 200)
    ax.plot(x_grid, sl_u * x_grid + ic_u, '-', color='black', lw=1.5,
            alpha=0.7, label='Universal fit', zorder=2)
    ax.plot(x_grid, sl_h * x_grid + ic_h, '--', color=C_HYDRO, lw=1.0,
            alpha=0.7, label='Hydrogel-only fit', zorder=2)
    ax.plot(x_grid, sl_f * x_grid + ic_f, '--', color=C_FUNGI, lw=1.0,
            alpha=0.7, label='Fungi-only fit', zorder=2)

    # Read the BIC table for inline annotation
    bic_df = pd.read_csv(model_table_path)
    best = bic_df.loc[bic_df['BIC'].idxmin()]
    txt = (f"BIC ranking (full table inset):\n"
           f"  Best: {best['Model']}  (R²={best['R2']:.3f})\n"
           f"  ΔBIC vs next best: {bic_df['BIC'].sort_values().diff().iloc[1]:.2f}")
    ax.text(0.04, 0.95, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=TICK_SIZE - 1.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    ax.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=2)
    ax.tick_params(labelsize=TICK_SIZE - 0.5)
    ax.legend(fontsize=TICK_SIZE - 1.5, loc='lower right',
              frameon=False, handletextpad=0.4)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=LABEL_SIZE - 0.5, pad=4)
    ax.text(-0.18, 1.05, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')


def main():
    apply_style()
    um = pd.read_csv(UNIV_CSV)
    rsr = pd.read_csv(RSR_CSV)
    lab = um[um['system'].isin(['Hydrogel', 'Fungi'])].copy()
    lab_dt = rsr[rsr['system'].isin(['Hydrogel', 'Fungi'])].copy()

    fig = plt.figure(figsize=(190 * MM, 90 * MM))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.97, top=0.88, bottom=0.16,
                          wspace=0.30)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # Panel A: zone gradient vs delta
    panel_regression(axA, lab, 'delta_um', 'zone_metric',
                     r'Size gradient $(\bar{R}_{\rm far}-\bar{R}_{\rm near})/\bar{R}_{\rm mid}$',
                     'A', 'Size gradient (Fig. 3E)',
                     ANCOVA_DIR / 'model_comparison_zone_metric.csv')

    # Panel B: dtau50/dr vs delta
    panel_regression(axB, lab_dt, 'delta_um', 'dtau50_dr',
                     r'$d\tau_{50}/dr$ (min/mm)',
                     'B', r'Survival gradient ($d\tau_{50}/dr$)',
                     ANCOVA_DIR / 'model_comparison_dtau50_dr.csv')

    save_fig(fig, str(OUT_DIR / 'FigureS33_ancova'))
    plt.close(fig)
    print('Saved S33')

    # Print summary
    for x_col, y_col, name in [('delta_um', 'zone_metric', 'zone_metric'),
                                ('delta_um', 'dtau50_dr', 'dtau50_dr')]:
        if y_col == 'zone_metric':
            d = lab
        else:
            d = lab_dt
        d2 = d.dropna(subset=[x_col, y_col])
        (sl_h, ic_h), (sl_f, ic_f) = fit_separate(d2, x_col, y_col)
        sl_u, ic_u = fit_universal(d2, x_col, y_col)
        print(f'\n{name}:')
        print(f'  Universal slope = {sl_u:.4f},   intercept = {ic_u:.3f}')
        print(f'  Hydrogel slope  = {sl_h:.4f},   intercept = {ic_h:.3f}')
        print(f'  Fungi slope     = {sl_f:.4f},   intercept = {ic_f:.3f}')


if __name__ == '__main__':
    main()
