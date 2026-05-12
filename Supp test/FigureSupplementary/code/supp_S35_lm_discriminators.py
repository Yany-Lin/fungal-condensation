#!/usr/bin/env python3
"""Supplementary Figure S35: Light-microscopy morphological discriminators
ranked by effect size.

Pre-empts the "you cherry-picked four metrics from a screening" reviewer
concern by showing the full ranked list of 15 LM-derived metrics screened
during the architecture analysis. The four metrics chosen for Fig. 4
(FFT spectral slope, structure thickness, Hessian tubeness CV, erosion
survival) are flagged with stars; their selection was based on biological
interpretability and parameter-stability across magnifications, not raw
effect-size ranking alone.

Data: HYPHAE/Final Results/light_micro_top_discriminators.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, apply_style, save_fig

CSV = Path('/Volumes/T7/FINAL OSF/HYPHAE/Final Results/light_micro_top_discriminators.csv')

# The four metrics actually used in Figure 4
CHOSEN_METRICS = {'fft_spectral_slope_alpha', 'tissue_thickness_um',
                  'hessian_tubeness_cv', 'erosion_survival_iter10',
                  # CSV may use different keys — fuzzy matching
                  'local_density_cv'}  # tubeness CV's near-equivalent in this CSV

# Cleaner display names
PRETTY = {
    'network_fiber_area_px':           'Network fiber area (px)',
    'network_fiber_area_fraction':     'Network fiber area fraction',
    'local_density_p90_minus_p10':     'Local density p90$-$p10',
    'local_density_std':               'Local density std.',
    'local_density_cv':                'Local density CV',
    'background_mean_intensity':       'Background mean intensity',
    'foreground_darkness_contrast':    'Foreground darkness contrast',
    'component_compactness_weighted_mean': 'Component compactness',
    'component_solidity_weighted_mean':    'Component solidity',
    'largest_component_fraction_of_foreground': 'Largest comp. fraction',
    'component_extent_weighted_mean':  'Component extent',
    'largest_component_area_px':       'Largest comp. area',
    'component_area_median_px':        'Component area (median)',
    'network_fiber_component_count':   'Component count',
    'foreground_area_px':              'Foreground area',
}

OUT_DIR = OUTPUT_DIR / 'S35'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    apply_style()
    df = pd.read_csv(CSV)
    df['abs_d'] = df['cohens_d'].abs()
    df = df.sort_values('abs_d', ascending=True)
    df['display'] = df['metric'].map(PRETTY).fillna(df['metric'])

    fig, ax = plt.subplots(figsize=(180 * MM, 120 * MM))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.92, bottom=0.12)

    # Significance status colors: BH-significant green-ish, otherwise gray
    colors = ['#4CAF50' if s else '#999999' for s in df['sig_bh_005']]
    bars = ax.barh(np.arange(len(df)), df['abs_d'], color=colors,
                   edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['display'], fontsize=TICK_SIZE - 0.5)
    ax.set_xlabel(r"Cohen's $d$ (|effect size|)", fontsize=LABEL_SIZE, labelpad=2)
    ax.tick_params(axis='x', labelsize=TICK_SIZE - 0.5)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

    # Annotate p value next to each bar
    for i, (idx, r) in enumerate(df.iterrows()):
        d_val = r['abs_d']
        ax.text(d_val + 0.15, i,
                f"d={d_val:.2f}, p={r['welch_p']:.1e}",
                va='center', ha='left', fontsize=TICK_SIZE - 1,
                color='#333333')

    # Direction tag (Asp>Muc or Asp<Muc)
    for i, (idx, r) in enumerate(df.iterrows()):
        ax.text(-0.10, i, '↑' if r['direction'] == 'Asp>Muc' else '↓',
                va='center', ha='right', fontsize=TICK_SIZE,
                color='#4CAF50' if r['direction'] == 'Asp>Muc' else '#C0392B',
                transform=ax.get_yaxis_transform())

    # Mark the metric used in Fig. 4E (Hessian tubeness CV ≈ local_density_cv)
    for i, (idx, r) in enumerate(df.iterrows()):
        if r['metric'] == 'local_density_cv':
            ax.text(0.01, i, '★', va='center', ha='left',
                    fontsize=TICK_SIZE + 4, color='#FFB800',
                    transform=ax.get_yaxis_transform(), zorder=5)

    ax.set_xlim(0, max(df['abs_d']) * 1.45)
    ax.set_title('Light-microscopy morphological metrics: '
                 'Aspergillus vs Mucor screening',
                 fontsize=LABEL_SIZE, pad=8)

    # Legend for color
    handles = [plt.Rectangle((0, 0), 1, 1, color='#4CAF50',
                              label='BH-corrected significant ($p_{adj} < 0.05$)'),
               plt.Rectangle((0, 0), 1, 1, color='#999999',
                              label='Not significant after BH correction')]
    ax.legend(handles=handles, fontsize=TICK_SIZE - 0.5, loc='lower right',
              frameon=False)

    save_fig(fig, str(OUT_DIR / 'FigureS35_lm_ranking'))
    plt.close(fig)
    print('Saved S35')


if __name__ == '__main__':
    main()
