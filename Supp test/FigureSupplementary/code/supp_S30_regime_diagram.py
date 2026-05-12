#!/usr/bin/env python3
"""Supplementary Figure S30: R_s/δ regime diagram.

Shows where each of the 35 laboratory trials sits on the R_s/δ axis,
mapping to the boundary-layer-dominated regime invoked in the Discussion.

R_s/δ < 1   point-sink regime (curvature confounds chemistry)
R_s/δ ~ 1   transition
R_s/δ > 1   boundary-layer regime (depletion set by sink thermodynamics)

R_s = 1500 µm for both hydrogel disks (radius 1.5 mm) and fungal
patches (3 mm dermal-punch diameter ⇒ radius 1.5 mm). Single source
radius across all 35 trials.
"""

import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL, apply_style, save_fig

OUT = OUTPUT_DIR / 'S30'
OUT.mkdir(parents=True, exist_ok=True)

UNIV_CSV = Path('/Volumes/T7/FINAL OSF/FigureTable/output/universal_metrics.csv')
R_S_UM = 1500.0  # source radius in µm

GROUP_COLOR = {
    'Agar':     '#9E9E9E',
    '0.5:1':    '#E67E22',
    '1:1':      '#5B8FC9',
    '2:1':      '#C0392B',
    'Green':    '#4CAF50',
    'White':    '#757575',
    'Black':    '#212121',
}
GROUP_LABEL = {
    'Agar': 'Agar', '0.5:1': '0.5:1 NaCl', '1:1': '1:1 NaCl', '2:1': '2:1 NaCl',
    'Green': 'Aspergillus', 'White': 'Mucor', 'Black': 'Rhizopus',
}
ORDER = ['Agar', '0.5:1', '1:1', '2:1', 'Green', 'White', 'Black']


def main():
    apply_style()
    um = pd.read_csv(UNIV_CSV)
    lab = um[um['system'].isin(['Hydrogel', 'Fungi'])].copy()
    lab['Rs_over_delta'] = R_S_UM / lab['delta_um']

    fig, ax = plt.subplots(figsize=(180 * MM, 90 * MM))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.32)

    # shaded regimes
    ax.axvspan(0.01, 1.0, color='#FFE0E0', alpha=0.5, zorder=0)
    ax.axvspan(1.0, 100.0, color='#E0FFE0', alpha=0.5, zorder=0)
    ax.axvline(1.0, color='black', lw=0.8, ls='--', zorder=1)

    ax.text(0.3, 0.92, 'point-sink regime\n(geometry $\\approx$ chemistry)',
            transform=ax.transAxes, fontsize=TICK_SIZE - 0.5, ha='center', va='top',
            color='#A02020')
    ax.text(0.78, 0.92, 'boundary-layer regime\n(thermodynamics-dominated)',
            transform=ax.transAxes, fontsize=TICK_SIZE - 0.5, ha='center', va='top',
            color='#205020')

    # plot each trial
    y_jit = np.random.RandomState(42).uniform(-0.18, 0.18, len(lab))
    for i, (idx, row) in enumerate(lab.iterrows()):
        c = GROUP_COLOR[row['group']]
        ax.scatter(row['Rs_over_delta'], y_jit[i], s=40, c=c,
                   edgecolors='white', linewidths=0.4, zorder=3, alpha=0.85)

    # vertical band for the operating range claimed in Discussion (1.7-14)
    rng = lab['Rs_over_delta'].dropna()
    rng_lo, rng_hi = rng.min(), rng.max()
    ax.errorbar([(rng_lo + rng_hi) / 2], [-0.6], xerr=[[(rng_hi - rng_lo) / 2]],
                fmt='none', ecolor='black', capsize=4, lw=1.2, zorder=2)
    ax.text((rng_lo + rng_hi) / 2, -0.85,
            f'observed range:  $R_s/\\delta = {rng_lo:.1f}$ to ${rng_hi:.1f}$',
            ha='center', va='top', fontsize=TICK_SIZE)

    ax.set_xscale('log')
    ax.set_xlim(0.5, 30)
    ax.set_ylim(-1.2, 1.0)
    ax.set_xlabel(r'$R_s\,/\,\delta$  (source radius / dry-zone width)',
                  fontsize=LABEL_SIZE, labelpad=20)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=GROUP_COLOR[g],
                          markeredgecolor='white', markersize=7, label=GROUP_LABEL[g])
               for g in ORDER]
    ax.legend(handles=handles, fontsize=TICK_SIZE - 0.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), ncol=7, frameon=False, handletextpad=0.4,
              columnspacing=0.8)

    save_fig(fig, str(OUT / 'FigureS30_regime_diagram'))
    plt.close(fig)
    print(f'Saved S30 to {OUT}')
    print(f'  Range: {rng_lo:.2f} to {rng_hi:.2f}')


if __name__ == '__main__':
    main()
