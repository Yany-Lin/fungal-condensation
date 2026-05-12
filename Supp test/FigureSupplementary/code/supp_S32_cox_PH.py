#!/usr/bin/env python3
"""Supplementary Figure S32: Cox proportional-hazards model formalizes the
distance effect on droplet survival.

Per-trial Cox PH model with covariates {distance from boundary, log(R0)}.
Trial-level distance hazard ratio (HR < 1 = increased survival further from
sink) is shown across all 32 successfully-fit trials. The HR varies
systematically with sink strength (delta), confirming the spatial pattern
seen in the Kaplan-Meier analysis.

Three panels:
  A — Forest plot: per-trial HR with 95% CI, ordered by condition.
  B — HR vs delta scatter — distance effect is stronger for stronger sinks.
  C — Schoenfeld residual test p-values, confirming PH assumption holds.

Data: additions/5_cox_PH_model/cox_per_trial_results.csv (32 trials × 17 cols)
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
    DELTA, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, apply_style, save_fig,
)

CSV = Path('/Volumes/T7/FINAL OSF/additions/5_cox_PH_model/cox_per_trial_results.csv')

GROUP_COLOR = {
    'Agar': '#9E9E9E', '0.5:1': '#E67E22', '1:1': '#5B8FC9', '2:1': '#C0392B',
    'Green': '#4CAF50', 'Black': '#212121', 'White': '#757575',
}
GROUP_ORDER = ['Agar', '0.5:1', '1:1', '2:1', 'Green', 'Black', 'White']
GROUP_LABEL = {
    'Agar': 'Agar', '0.5:1': '0.5:1 NaCl', '1:1': '1:1 NaCl', '2:1': '2:1 NaCl',
    'Green': 'Aspergillus', 'Black': 'Rhizopus', 'White': 'Mucor',
}

OUT_DIR = OUTPUT_DIR / 'S32'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    apply_style()
    df = pd.read_csv(CSV)
    df['delta_um'] = df['trial_id'].map(DELTA)
    df['color'] = df['group'].map(GROUP_COLOR)
    df = df.dropna(subset=['delta_um'])

    # Sort by group order then by HR
    df['_grp_idx'] = df['group'].apply(lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99)
    df = df.sort_values(['_grp_idx', 'HR_distance']).reset_index(drop=True)

    fig = plt.figure(figsize=(190 * MM, 165 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0],
                          left=0.10, right=0.97, top=0.97, bottom=0.06,
                          hspace=0.30, wspace=0.30)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    # ── Panel A: forest plot ──
    y_pos = np.arange(len(df))
    err_lo = df['HR_distance'] - df['HR_dist_ci_lo']
    err_hi = df['HR_dist_ci_hi'] - df['HR_distance']
    axA.errorbar(df['HR_distance'], y_pos,
                 xerr=[err_lo, err_hi], fmt='none', ecolor='gray',
                 elinewidth=0.7, capsize=2, zorder=2)
    for i, row in df.iterrows():
        axA.scatter(row['HR_distance'], i, s=22, c=row['color'],
                    edgecolors='white', linewidths=0.4, zorder=3)
    axA.axvline(1.0, color='black', ls='--', lw=0.8, alpha=0.6, zorder=1)
    axA.text(1.02, len(df) - 0.5, 'no effect', fontsize=TICK_SIZE - 1,
             ha='left', va='top', color='black', alpha=0.7)

    axA.set_yticks(y_pos)
    axA.set_yticklabels(df['trial_id'], fontsize=TICK_SIZE - 1)
    axA.set_xlabel('Distance hazard ratio (per mm)',
                   fontsize=LABEL_SIZE, labelpad=2)
    axA.set_xlim(0.2, 1.5)
    axA.tick_params(axis='x', labelsize=TICK_SIZE - 0.5)
    for sp in ('top', 'right'): axA.spines[sp].set_visible(False)
    axA.set_title(f'Per-trial Cox PH distance hazard ratio (n = {len(df)} trials)',
                  fontsize=LABEL_SIZE, pad=4)
    axA.text(-0.16, 1.02, 'A', transform=axA.transAxes,
             fontsize=11, fontweight='bold', va='top')

    # condition legend (right side)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=GROUP_COLOR[g],
                          markeredgecolor='white', markersize=6,
                          label=GROUP_LABEL[g])
               for g in GROUP_ORDER]
    axA.legend(handles=handles, fontsize=TICK_SIZE - 1, loc='upper right',
               frameon=False, handletextpad=0.4, labelspacing=0.3)

    # ── Panel B: HR vs δ ──
    axB.scatter(df['delta_um'], df['HR_distance'], s=28, c=df['color'],
                edgecolors='white', linewidths=0.4, alpha=0.85, zorder=3)
    rho, p_rho = stats.spearmanr(df['delta_um'], df['HR_distance'])
    axB.text(0.05, 0.05, f"Spearman $r$ = {rho:.2f},  $p$ = {p_rho:.1e}",
             transform=axB.transAxes, ha='left', va='bottom',
             fontsize=TICK_SIZE - 0.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='none', alpha=0.8))
    axB.axhline(1.0, color='black', ls='--', lw=0.6, alpha=0.5, zorder=1)
    axB.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=2)
    axB.set_ylabel('Distance hazard ratio',
                   fontsize=LABEL_SIZE, labelpad=2)
    axB.tick_params(labelsize=TICK_SIZE - 0.5)
    for sp in ('top', 'right'): axB.spines[sp].set_visible(False)
    axB.text(-0.18, 1.05, 'B', transform=axB.transAxes,
             fontsize=11, fontweight='bold', va='top')

    # ── Panel C: Schoenfeld residual p-values ──
    axC.axhspan(0, 0.05, color='#FFE0E0', alpha=0.5, zorder=0)
    axC.scatter(df.index, df['PH_p_distance'], s=24, c=df['color'],
                edgecolors='white', linewidths=0.4, alpha=0.85, zorder=3,
                label='distance')
    axC.scatter(df.index, df['PH_p_logR0'], s=20, c=df['color'],
                edgecolors='black', linewidths=0.4, alpha=0.85, zorder=3,
                marker='^', label=r'log $R_0$')
    axC.axhline(0.05, color='red', ls='--', lw=0.6, alpha=0.6, zorder=1)
    axC.set_xlabel('Trial index', fontsize=LABEL_SIZE, labelpad=2)
    axC.set_ylabel('Schoenfeld $p$-value\n(PH assumption)',
                   fontsize=LABEL_SIZE, labelpad=2)
    axC.set_yscale('log')
    axC.set_ylim(1e-3, 1.0)
    axC.legend(fontsize=TICK_SIZE - 1, loc='lower right', frameon=False,
               handletextpad=0.4)
    axC.tick_params(labelsize=TICK_SIZE - 0.5)
    for sp in ('top', 'right'): axC.spines[sp].set_visible(False)
    axC.text(-0.18, 1.05, 'C', transform=axC.transAxes,
             fontsize=11, fontweight='bold', va='top')

    save_fig(fig, str(OUT_DIR / 'FigureS32_cox_PH'))
    plt.close(fig)
    print('Saved S32')

    # Summary stats
    print(f'\nN trials: {len(df)}')
    print(f'Distance HR range: {df["HR_distance"].min():.2f}–{df["HR_distance"].max():.2f}')
    print(f'All p_distance < 1e-3: {(df["p_distance"] < 1e-3).sum()}/{len(df)}')
    print(f'PH violations (p<0.05): distance={((df["PH_p_distance"]<0.05).sum())}, '
          f'logR0={((df["PH_p_logR0"]<0.05).sum())}')


if __name__ == '__main__':
    main()
