#!/usr/bin/env python3
"""Statistical testing of all Light Microscopy metrics from analyze_fungi.py.

Reads the pre-computed fungi_metrics.csv (9 images × 56 metrics),
runs Welch's t, Mann-Whitney U, Cohen's d on every numeric column,
applies Benjamini-Hochberg FDR correction, classifies metrics as
prep-independent vs prep-dependent, and checks magnification confound.

Usage:
    python light_microscopy_statistics.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── paths ──
METRICS_CSV = Path('/Users/yany/Desktop/Leyun microscopy/analysis_outputs/fungi_metrics.csv')
OUT_DIR = Path(__file__).resolve().parent / 'results'
OUT_DIR.mkdir(exist_ok=True)

MM = 1 / 25.4
C_ASP = '#4CAF50'
C_MUC = '#757575'

# ── prep-independent metrics (ratios, shape, CV — not absolute amounts) ──
PREP_INDEPENDENT = {
    'area_fraction', 'network_fiber_area_fraction',
    'foreground_darkness_contrast',
    'largest_component_fraction_of_foreground',
    'component_extent_weighted_mean', 'component_compactness_weighted_mean',
    'component_solidity_weighted_mean',
    'local_density_mean', 'local_density_std', 'local_density_cv',
    'local_density_p90_minus_p10',
    'fft_anisotropy',
    'branchpoint_to_endpoint_ratio',  # derived below
}

SKIP_COLS = {
    'magnification', 'microns_per_pixel', 'pixels_per_micron',
    'skeleton_analysis_scale', 'fft_analysis_step_px',
    'usable_field_area_px', 'usable_field_area_um2', 'usable_field_fraction',
}


def benjamini_hochberg(pvals):
    """BH-FDR correction. Returns adjusted p-values."""
    n = len(pvals)
    ranked = np.argsort(pvals)
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        if i == n - 1:
            adjusted[ranked[i]] = pvals[ranked[i]]
        else:
            adjusted[ranked[i]] = min(
                adjusted[ranked[i + 1]],
                pvals[ranked[i]] * n / rank
            )
    return np.clip(adjusted, 0, 1)


def main():
    df = pd.read_csv(METRICS_CSV)
    print(f'Loaded {len(df)} images × {len(df.columns)} columns')
    print(f'Labels: {df["label"].unique()}')

    # Derive branchpoint-to-endpoint ratio
    if 'skeleton_branchpoints' in df.columns and 'skeleton_endpoints' in df.columns:
        df['branchpoint_to_endpoint_ratio'] = (
            df['skeleton_branchpoints'] / df['skeleton_endpoints'].replace(0, np.nan)
        )

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    test_cols = [c for c in numeric_cols if c not in SKIP_COLS]

    asp = df[df['label'] == 'Green']
    muc = df[df['label'] == 'White']
    print(f'Asp (Green): {len(asp)}, Muc (White): {len(muc)}')

    # ── Run all tests ──
    results = []
    for col in test_cols:
        a = asp[col].dropna().values
        m = muc[col].dropna().values
        if len(a) < 2 or len(m) < 2:
            continue

        t_stat, t_p = stats.ttest_ind(a, m, equal_var=False)
        try:
            u_stat, u_p = stats.mannwhitneyu(a, m, alternative='two-sided')
        except ValueError:
            u_stat, u_p = np.nan, np.nan

        n1, n2 = len(a), len(m)
        pooled = np.sqrt(((n1 - 1) * a.std(ddof=1)**2 + (n2 - 1) * m.std(ddof=1)**2) / (n1 + n2 - 2))
        d = (a.mean() - m.mean()) / pooled if pooled > 0 else 0

        prep_indep = col in PREP_INDEPENDENT
        results.append({
            'metric': col,
            'asp_mean': round(a.mean(), 4),
            'asp_std': round(a.std(ddof=1), 4),
            'muc_mean': round(m.mean(), 4),
            'muc_std': round(m.std(ddof=1), 4),
            'n_asp': n1,
            'n_muc': n2,
            'welch_t': round(t_stat, 4),
            'welch_p': t_p,
            'mw_U': u_stat,
            'mw_p': u_p,
            'cohens_d': round(d, 4),
            'direction': 'Asp>Muc' if a.mean() > m.mean() else 'Asp<Muc',
            'prep_independent': prep_indep,
        })

    res = pd.DataFrame(results)

    # BH-FDR correction
    res['bh_welch_p'] = benjamini_hochberg(res['welch_p'].values)
    res['bh_mw_p'] = benjamini_hochberg(res['mw_p'].fillna(1).values)
    res['sig_bh_005'] = res['bh_welch_p'] < 0.05

    # Sort by absolute effect size
    res['abs_d'] = res['cohens_d'].abs()
    res = res.sort_values('abs_d', ascending=False)

    # ── Save full table ──
    csv_path = OUT_DIR / 'light_micro_all_statistics.csv'
    res.drop(columns='abs_d').to_csv(csv_path, index=False)
    print(f'\nFull table: {csv_path}')

    # ── Top discriminators ──
    top = res.head(15).copy()
    top_path = OUT_DIR / 'light_micro_top_discriminators.csv'
    top.drop(columns='abs_d').to_csv(top_path, index=False)
    print(f'Top 15: {top_path}')

    # ── Print summary ──
    print(f'\n{"="*80}')
    print('TOP 15 METRICS BY EFFECT SIZE')
    print(f'{"="*80}')
    print(f'{"Metric":<45} {"d":>6} {"p_raw":>8} {"p_BH":>8} {"Prep":>5} {"Dir":>8}')
    print('-' * 80)
    for _, r in top.iterrows():
        prep = '  Y' if r['prep_independent'] else '  N'
        print(f'{r["metric"]:<45} {r["cohens_d"]:>6.2f} {r["welch_p"]:>8.4f} '
              f'{r["bh_welch_p"]:>8.4f} {prep:>5} {r["direction"]:>8}')

    sig_count = res['sig_bh_005'].sum()
    sig_prep = res[res['prep_independent'] & res['sig_bh_005']]
    print(f'\nSignificant after BH correction: {sig_count}/{len(res)}')
    print(f'Significant AND prep-independent: {len(sig_prep)}')

    # ── Magnification confound check ──
    print(f'\n{"="*80}')
    print('MAGNIFICATION CONFOUND CHECK (top 6 metrics)')
    print(f'{"="*80}')

    top6 = top.head(6)['metric'].values
    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 100 * MM))
    axes = axes.ravel()

    for i, col in enumerate(top6):
        ax = axes[i]
        for genus, color, label in [('Green', C_ASP, 'Asp'), ('White', C_MUC, 'Muc')]:
            sub = df[df['label'] == genus]
            ax.scatter(sub['magnification'], sub[col], c=color, s=30,
                       alpha=0.8, label=label, zorder=3)
        ax.set_xlabel('Magnification', fontsize=7)
        ax.set_ylabel(col.replace('_', ' ')[:25], fontsize=6)
        ax.set_xticks([10, 20, 40])
        ax.legend(fontsize=6)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        # Print per-mag values
        print(f'\n  {col}:')
        for mag in [10, 20, 40]:
            a_vals = df[(df['label'] == 'Green') & (df['magnification'] == mag)][col].values
            m_vals = df[(df['label'] == 'White') & (df['magnification'] == mag)][col].values
            a_str = f'{a_vals.mean():.3f}' if len(a_vals) else 'N/A'
            m_str = f'{m_vals.mean():.3f}' if len(m_vals) else 'N/A'
            print(f'    {mag}X: Asp={a_str}, Muc={m_str}')

    fig.suptitle('Magnification confound check — top 6 metrics', fontsize=9, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    mag_path = OUT_DIR / 'light_micro_mag_confound.png'
    fig.savefig(mag_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nMagnification plot: {mag_path}')

    # ── Effect size bar chart (top 15, colored by prep independence) ──
    fig, ax = plt.subplots(figsize=(120 * MM, 100 * MM))
    fig.subplots_adjust(left=0.45, right=0.95, top=0.92, bottom=0.10)

    y_pos = np.arange(len(top))[::-1]
    colors = [C_ASP if r['prep_independent'] else '#999' for _, r in top.iterrows()]
    bars = ax.barh(y_pos, top['cohens_d'].values, color=colors, alpha=0.7, height=0.7)

    # Significance markers
    for j, (_, r) in enumerate(top.iterrows()):
        yp = y_pos[j]
        d = r['cohens_d']
        star = ''
        if r['bh_welch_p'] < 0.001:
            star = '***'
        elif r['bh_welch_p'] < 0.01:
            star = '**'
        elif r['bh_welch_p'] < 0.05:
            star = '*'
        if star:
            ax.text(d + 0.15 * np.sign(d), yp, star, ha='left' if d > 0 else 'right',
                    va='center', fontsize=8, fontweight='bold')

    labels = [r['metric'].replace('_', ' ')[:35] for _, r in top.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Cohen's d", fontsize=8)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_title('Light Microscopy: Top 15 discriminating metrics', fontsize=9, fontweight='bold')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    for ext in ('.png', '.pdf'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUT_DIR / f'light_micro_effect_sizes{ext}', **kw)
    plt.close(fig)

    print(f'\nEffect size plot saved.')
    print(f'\nDone. All outputs in {OUT_DIR}/')


if __name__ == '__main__':
    main()
