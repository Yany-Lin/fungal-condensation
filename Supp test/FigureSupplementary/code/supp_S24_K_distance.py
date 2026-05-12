#!/usr/bin/env python3
"""Supplementary Figure S24: Per-droplet evaporation coefficient K vs
distance from source — physical mechanism behind dry-zone formation.

The dry-zone width δ is an emergent observable; the underlying physical
quantity is the local evaporation rate (or its surrogate, the d² law's
slope K = dr²/dt). If vapor flux is depleted near hygroscopic sinks,
droplets there should evaporate slower (smaller K). This figure shows
that prediction holds across all 35 trials.

Four panels:
  A — Per-droplet K vs distance for 3 exemplar trials (agar, 2:1 NaCl,
      Aspergillus). Binned medians ± SEM overlaid; dK/dr slope annotated.
  B — Per-trial dK/dr scatter vs δ across all 35 trials. Stronger sinks
      → larger K gradient.
  C — K_ratio (K_near / K_far) as a function of group, showing
      systematic depression of near-source K.
  D — K_near and K_far per condition (paired bar plot).

Data: additions/4_K_distance_evaporation/K_summary_all_trials.csv
      additions/4_K_distance_evaporation/K_fits_{agar.4, 2to1.2, Green.1}.csv
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

K_DIR = Path('/Volumes/T7/FINAL OSF/additions/4_K_distance_evaporation')

GROUP_COLOR = {
    'Agar': '#9E9E9E', '0.5:1': '#E67E22', '1:1': '#5B8FC9', '2:1': '#C0392B',
    'Green': '#4CAF50', 'Black': '#212121', 'White': '#757575',
}
GROUP_ORDER = ['Agar', '0.5:1', '1:1', '2:1', 'Green', 'Black', 'White']
GROUP_LABEL = {
    'Agar': 'Agar', '0.5:1': '0.5:1 NaCl', '1:1': '1:1 NaCl', '2:1': '2:1 NaCl',
    'Green': 'Aspergillus', 'Black': 'Rhizopus', 'White': 'Mucor',
}

OUT_DIR = OUTPUT_DIR / 'S24'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bin_K_by_distance(df, n_bins=8):
    """Return bin centers and median K per bin (filtered for R2 > 0.5)."""
    df = df[df['R2_gof'] > 0.5].copy()
    if len(df) < 20: return None, None, None
    edges = np.linspace(df['distance_um'].min(), df['distance_um'].max(), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    medians = []; sems = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = df[(df['distance_um'] >= lo) & (df['distance_um'] < hi)]
        if len(sub) < 5:
            medians.append(np.nan); sems.append(np.nan)
        else:
            medians.append(np.median(sub['K_fit']))
            sems.append(sub['K_fit'].std() / np.sqrt(len(sub)))
    return centers / 1000, np.array(medians), np.array(sems)


def main():
    apply_style()
    summary = pd.read_csv(K_DIR / 'K_summary_all_trials.csv')
    summary['color'] = summary['group'].map(GROUP_COLOR)

    fig = plt.figure(figsize=(190 * MM, 160 * MM))
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.97, top=0.94, bottom=0.08,
                          hspace=0.45, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # ── Panel A: K(distance) for 3 exemplars ──
    EXEMPLARS = [
        ('agar.4', 'Agar (no sink)', '#9E9E9E'),
        ('Green.1', 'Aspergillus', '#4CAF50'),
        ('2to1.2', '2:1 NaCl', '#C0392B'),
    ]
    for tid, label, color in EXEMPLARS:
        path = K_DIR / f'K_fits_{tid}.csv'
        if not path.exists(): continue
        df = pd.read_csv(path)
        df = df[df['R2_gof'] > 0.5]
        # ghost scatter
        axA.scatter(df['distance_um'] / 1000, df['K_fit'], s=2, c=color,
                    alpha=0.10, rasterized=True, zorder=1)
        # binned
        c, m, se = bin_K_by_distance(df)
        if c is not None:
            axA.errorbar(c, m, yerr=se, fmt='o-', color=color, ms=5,
                         mfc=color, mec='white', mew=0.4, lw=1.4, capsize=2,
                         label=label, zorder=3)
            # slope annotation
            mask = np.isfinite(m)
            if mask.sum() >= 3:
                sl, ic, *_ = stats.linregress(c[mask], m[mask])
                axA.text(0.97, 0.04 + 0.10 * EXEMPLARS.index((tid, label, color)),
                         f'  $dK/dr$ = {sl:.2f} (m{label[:3]})',
                         transform=axA.transAxes, ha='right', va='bottom',
                         fontsize=TICK_SIZE - 1.5, color=color)
    axA.set_xlim(0, 4); axA.set_ylim(bottom=0)
    axA.set_xlabel(r'Distance from boundary (mm)', fontsize=LABEL_SIZE, labelpad=2)
    axA.set_ylabel(r'$K$ ($\mu$m$^2$/s)  per-droplet evap. coefficient',
                   fontsize=LABEL_SIZE - 0.5, labelpad=2)
    axA.legend(fontsize=TICK_SIZE - 0.5, loc='upper left', frameon=False,
               handletextpad=0.4)
    axA.tick_params(labelsize=TICK_SIZE - 0.5)
    for sp in ('top', 'right'): axA.spines[sp].set_visible(False)
    axA.text(-0.20, 1.05, 'A', transform=axA.transAxes,
             fontsize=11, fontweight='bold', va='top')

    # ── Panel B: dK/dr vs δ for all 35 trials ──
    for grp in GROUP_ORDER:
        sub = summary[summary['group'] == grp]
        if len(sub) == 0: continue
        axB.scatter(sub['delta_um'], sub['dK_dr'], s=28,
                    c=GROUP_COLOR[grp], edgecolors='white',
                    linewidths=0.4, alpha=0.85, zorder=3,
                    label=GROUP_LABEL[grp])
    rho, p_rho = stats.spearmanr(summary['delta_um'], summary['dK_dr'])
    r_p, p_p = stats.pearsonr(summary['delta_um'], summary['dK_dr'])
    axB.text(0.05, 0.95,
             f'Spearman $r$ = {rho:.2f}, $p$ = {p_rho:.1e}\n'
             f'Pearson $r$ = {r_p:.2f}, $p$ = {p_p:.1e}',
             transform=axB.transAxes, ha='left', va='top',
             fontsize=TICK_SIZE - 1.5,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='none', alpha=0.85))
    axB.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=2)
    axB.set_ylabel(r'$dK/dr$ ($\mu$m$^2$/s/mm)',
                   fontsize=LABEL_SIZE, labelpad=2)
    axB.tick_params(labelsize=TICK_SIZE - 0.5)
    axB.legend(fontsize=TICK_SIZE - 1.5, loc='lower right',
               frameon=False, handletextpad=0.3, labelspacing=0.2,
               ncol=2)
    for sp in ('top', 'right'): axB.spines[sp].set_visible(False)
    axB.text(-0.18, 1.05, 'B', transform=axB.transAxes,
             fontsize=11, fontweight='bold', va='top')

    # ── Panel C: K_ratio per group (boxplot) ──
    box_data = []
    box_labels = []
    box_colors = []
    for grp in GROUP_ORDER:
        sub = summary[summary['group'] == grp]
        if len(sub) == 0: continue
        box_data.append(sub['K_ratio'].dropna().values)
        box_labels.append(GROUP_LABEL[grp])
        box_colors.append(GROUP_COLOR[grp])

    bp = axC.boxplot(box_data, positions=range(1, len(box_data) + 1),
                     widths=0.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color='white', lw=1.4),
                     whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    for box, c in zip(bp['boxes'], box_colors):
        box.set_facecolor(c); box.set_alpha(0.55)
    rng = np.random.RandomState(7)
    for i, (vals, c) in enumerate(zip(box_data, box_colors), 1):
        axC.scatter(i + rng.uniform(-0.10, 0.10, len(vals)), vals,
                    s=14, c=c, edgecolors='white', linewidths=0.3,
                    alpha=0.85, zorder=3)
    axC.axhline(1.0, color='gray', ls=':', lw=0.5, alpha=0.5, zorder=1)
    axC.set_xticks(range(1, len(box_labels) + 1))
    axC.set_xticklabels(box_labels, fontsize=TICK_SIZE - 1, rotation=20, ha='right')
    axC.set_ylabel(r'$K_{\rm near} / K_{\rm far}$',
                   fontsize=LABEL_SIZE, labelpad=2)
    axC.tick_params(axis='y', labelsize=TICK_SIZE - 0.5)
    for sp in ('top', 'right'): axC.spines[sp].set_visible(False)
    axC.text(-0.18, 1.05, 'C', transform=axC.transAxes,
             fontsize=11, fontweight='bold', va='top')

    # ── Panel D: paired K_near vs K_far ──
    width = 0.35
    grp_means_near = [summary[summary['group'] == g]['K_near'].mean() for g in GROUP_ORDER]
    grp_means_far = [summary[summary['group'] == g]['K_far'].mean() for g in GROUP_ORDER]
    grp_sem_near = [summary[summary['group'] == g]['K_near'].sem() for g in GROUP_ORDER]
    grp_sem_far = [summary[summary['group'] == g]['K_far'].sem() for g in GROUP_ORDER]
    x = np.arange(len(GROUP_ORDER))
    axD.bar(x - width / 2, grp_means_near, width, yerr=grp_sem_near,
            color=[GROUP_COLOR[g] for g in GROUP_ORDER], alpha=0.55,
            edgecolor='white', linewidth=0.5, label=r'$K_{\rm near}$',
            error_kw=dict(elinewidth=0.6, capsize=2))
    axD.bar(x + width / 2, grp_means_far, width, yerr=grp_sem_far,
            color=[GROUP_COLOR[g] for g in GROUP_ORDER], alpha=0.95,
            edgecolor='white', linewidth=0.5, label=r'$K_{\rm far}$',
            error_kw=dict(elinewidth=0.6, capsize=2))
    axD.set_xticks(x)
    axD.set_xticklabels([GROUP_LABEL[g] for g in GROUP_ORDER],
                        fontsize=TICK_SIZE - 1, rotation=20, ha='right')
    axD.set_ylabel(r'$K$ ($\mu$m$^2$/s)', fontsize=LABEL_SIZE, labelpad=2)
    axD.tick_params(axis='y', labelsize=TICK_SIZE - 0.5)
    axD.legend(fontsize=TICK_SIZE - 0.5, loc='upper right', frameon=False,
               handletextpad=0.4)
    for sp in ('top', 'right'): axD.spines[sp].set_visible(False)
    axD.text(-0.18, 1.05, 'D', transform=axD.transAxes,
             fontsize=11, fontweight='bold', va='top')

    save_fig(fig, str(OUT_DIR / 'FigureS24_K_distance'))
    plt.close(fig)
    print('Saved S24')

    print(f'\nK_ratio summary by group:')
    for g in GROUP_ORDER:
        sub = summary[summary['group'] == g]['K_ratio']
        if len(sub) > 0:
            print(f'  {GROUP_LABEL[g]:<14}  K_near/K_far = {sub.mean():.2f} ± {sub.std(ddof=1):.2f}  (n={len(sub)})')


if __name__ == '__main__':
    main()
