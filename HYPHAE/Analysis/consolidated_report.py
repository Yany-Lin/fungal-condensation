#!/usr/bin/env python3
"""Consolidated statistical report + effect-size forest plot.

Merges results from all three imaging scales into one coherent report.
Generates a forest plot showing Cohen's d with 95% CI for every
significant metric, grouped by scale and category.

Usage:
    python consolidated_report.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path(__file__).resolve().parent / 'results'
MM = 1 / 25.4
C_ASP = '#4CAF50'
C_MUC = '#757575'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
})


def cohens_d_ci(d, n1, n2, alpha=0.05):
    """Approximate 95% CI for Cohen's d (Hedges & Olkin)."""
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(1 - alpha / 2)
    return d - z * se, d + z * se


def main():
    report = []
    forest_data = []  # (label, d, ci_lo, ci_hi, p, scale, category)

    report.append('CONSOLIDATED MORPHOLOGICAL ANALYSIS')
    report.append('Nature Communications — Fungal Hyphae as Hygroscopic Vapor Sinks')
    report.append('=' * 70)

    # ════════════════════════════════════════════
    # SCALE 1: Colony surface FFT (from run_all.py)
    # ════════════════════════════════════════════
    report.append('\n\n── SCALE 1: Colony surface (Spacing 2/Green + W2) ──')

    fft_csv = OUT_DIR / 'fft_spectral_slopes.csv'
    if fft_csv.exists():
        fft = pd.read_csv(fft_csv)
        sa = fft.loc[fft['genus'] == 'Aspergillus', 'alpha'].values
        sm = fft.loc[fft['genus'] == 'Mucor', 'alpha'].values
        t, p = stats.ttest_ind(sa, sm, equal_var=False)
        u, up = stats.mannwhitneyu(sa, sm, alternative='two-sided')
        n1, n2 = len(sa), len(sm)
        pooled = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2)/(n1+n2-2))
        d = (sa.mean() - sm.mean()) / pooled
        ci = cohens_d_ci(d, n1, n2)
        report.append(f'  FFT spectral slope (α):')
        report.append(f'    Asp: {sa.mean():.4f} ± {sa.std(ddof=1):.4f} (n={n1})')
        report.append(f'    Muc: {sm.mean():.4f} ± {sm.std(ddof=1):.4f} (n={n2})')
        report.append(f'    Welch t={t:.3f}, p={p:.4f}')
        report.append(f'    Mann-Whitney U={u:.0f}, p={up:.4f}')
        report.append(f"    Cohen's d={d:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]")
        forest_data.append(('FFT α (colony)', d, ci[0], ci[1], p,
                           'Colony surface', 'Spectral'))

    # ════════════════════════════════════════════
    # SCALE 2: 3D macro (from roi_session.json + fragmentation)
    # ════════════════════════════════════════════
    report.append('\n\n── SCALE 2: 3D macro colony (ROI-selected) ──')

    import json
    session_path = OUT_DIR / '3d_overlays' / 'roi_session.json'
    if session_path.exists():
        with open(session_path) as f:
            session = json.load(f)
        saved = {k: v for k, v in session.items()
                 if not k.startswith('_') and v.get('status') != 'deleted'}
        asp_a = np.array([v['alpha'] for v in saved.values()
                         if v.get('genus') == 'Aspergillus' and v.get('alpha') is not None])
        muc_a = np.array([v['alpha'] for v in saved.values()
                         if v.get('genus') == 'Mucor' and v.get('alpha') is not None])
        if len(asp_a) >= 2 and len(muc_a) >= 2:
            t, p = stats.ttest_ind(asp_a, muc_a, equal_var=False)
            n1, n2 = len(asp_a), len(muc_a)
            pooled = np.sqrt(((n1-1)*asp_a.std(ddof=1)**2 + (n2-1)*muc_a.std(ddof=1)**2)/(n1+n2-2))
            d = (asp_a.mean() - muc_a.mean()) / pooled
            ci = cohens_d_ci(d, n1, n2)
            report.append(f'  FFT spectral slope (α):')
            report.append(f'    Asp: {asp_a.mean():.4f} ± {asp_a.std(ddof=1):.4f} (n={n1})')
            report.append(f'    Muc: {muc_a.mean():.4f} ± {muc_a.std(ddof=1):.4f} (n={n2})')
            report.append(f'    Welch t={t:.3f}, p={p:.4f}')
            report.append(f"    Cohen's d={d:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]")
            forest_data.append(('FFT α (3D macro)', d, ci[0], ci[1], p,
                               '3D macro', 'Spectral'))

    # Fragmentation metrics
    frag_csv = OUT_DIR / 'fragmentation_results.csv'
    if frag_csv.exists():
        frag = pd.read_csv(frag_csv)
        frag_metrics = [
            ('n_components', 'N components', 'Morphology'),
            ('tissue_frac', 'Tissue fraction', 'Morphology'),
            ('mean_gap_um', 'Mean gap width', 'Pore/Gap'),
            ('mean_area_um2', 'Mean component area', 'Morphology'),
            ('lacunarity_mean', 'Mean lacunarity', 'Spatial'),
            ('lacunarity_max_scale', 'Lacunarity (max scale)', 'Spatial'),
            ('fractal_dim', 'Fractal dimension', 'Spatial'),
        ]
        report.append('\n  Surface fragmentation:')
        for col, label, cat in frag_metrics:
            sa = frag.loc[frag['genus'] == 'Aspergillus', col].dropna().values
            sm = frag.loc[frag['genus'] == 'Mucor', col].dropna().values
            if len(sa) < 2 or len(sm) < 2:
                continue
            t, p = stats.ttest_ind(sa, sm, equal_var=False)
            n1, n2 = len(sa), len(sm)
            pooled = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2)/(n1+n2-2))
            d_val = (sa.mean() - sm.mean()) / pooled if pooled > 0 else 0
            ci = cohens_d_ci(d_val, n1, n2)
            report.append(f'    {label}: Asp {sa.mean():.2f}±{sa.std(ddof=1):.2f} vs '
                         f'Muc {sm.mean():.2f}±{sm.std(ddof=1):.2f}, '
                         f'd={d_val:.2f}, p={p:.4f}')
            if p < 0.1:
                forest_data.append((f'{label} (3D)', d_val, ci[0], ci[1], p,
                                   '3D macro', cat))

    # ════════════════════════════════════════════
    # SCALE 3: Light Microscopy (from analyze_fungi.py + Hessian)
    # ════════════════════════════════════════════
    report.append('\n\n── SCALE 3: Light Microscopy (disaggregated tissue) ──')

    # Hessian
    hess_csv = OUT_DIR / 'hessian_foreground.csv'
    if hess_csv.exists():
        hess = pd.read_csv(hess_csv)
        sa = hess.loc[hess['genus'] == 'Aspergillus', 'fg_frac'].values
        sm = hess.loc[hess['genus'] == 'Mucor', 'fg_frac'].values
        t, p = stats.ttest_ind(sa, sm, equal_var=False)
        n1, n2 = len(sa), len(sm)
        pooled = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2)/(n1+n2-2))
        d_val = (sa.mean() - sm.mean()) / pooled
        ci = cohens_d_ci(d_val, n1, n2)
        report.append(f'  Hessian tubeness foreground fraction:')
        report.append(f'    Asp: {sa.mean():.4f} ± {sa.std(ddof=1):.4f} (n={n1})')
        report.append(f'    Muc: {sm.mean():.4f} ± {sm.std(ddof=1):.4f} (n={n2})')
        report.append(f'    Welch t={t:.3f}, p={p:.4f}')
        report.append(f"    Cohen's d={d_val:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]")
        forest_data.append(('Hessian fg fraction', d_val, ci[0], ci[1], p,
                           'Light Microscopy', 'Network'))

    # Top metrics from analyze_fungi.py
    lm_csv = OUT_DIR / 'light_micro_top_discriminators.csv'
    if lm_csv.exists():
        lm = pd.read_csv(lm_csv)
        # Only prep-independent metrics that survived BH
        sig_prep = lm[(lm['prep_independent'] == True) & (lm['sig_bh_005'] == True)]
        report.append(f'\n  Prep-independent metrics (BH-corrected p<0.05):')
        for _, r in sig_prep.iterrows():
            ci = cohens_d_ci(r['cohens_d'], r['n_asp'], r['n_muc'])
            report.append(f'    {r["metric"]}: d={r["cohens_d"]:.2f}, '
                         f'p_raw={r["welch_p"]:.4f}, p_BH={r["bh_welch_p"]:.4f}')
            forest_data.append((r['metric'].replace('_', ' ')[:30] + ' (LM)',
                               r['cohens_d'], ci[0], ci[1], r['welch_p'],
                               'Light Microscopy',
                               'Network' if 'fiber' in r['metric'] else 'Spatial'))

    # ════════════════════════════════════════════
    # CONVERGENT EVIDENCE TABLE
    # ════════════════════════════════════════════
    report.append('\n\n── CONVERGENT EVIDENCE ──')
    report.append(f'{"Metric":<35} {"Scale":<18} {"d":>6} {"p":>8}')
    report.append('-' * 70)
    for label, d, ci_lo, ci_hi, p, scale, cat in sorted(forest_data, key=lambda x: -abs(x[1])):
        p_str = f'{p:.4f}' if p >= 0.001 else f'{p:.1e}'
        report.append(f'  {label:<33} {scale:<18} {d:>6.2f} {p_str:>8}')

    report.append(f'\nTotal significant metrics: {len(forest_data)}')
    report.append(f'Across {len(set(s[5] for s in forest_data))} imaging scales')

    txt = '\n'.join(report)
    print(txt)
    with open(OUT_DIR / 'consolidated_stats.txt', 'w') as f:
        f.write(txt)

    # Save convergent evidence CSV
    ce_df = pd.DataFrame(forest_data,
                         columns=['metric', 'cohens_d', 'ci_lo', 'ci_hi',
                                  'p_value', 'scale', 'category'])
    ce_df.to_csv(OUT_DIR / 'convergent_evidence_table.csv', index=False)

    # ════════════════════════════════════════════
    # FOREST PLOT
    # ════════════════════════════════════════════
    if not forest_data:
        print('No data for forest plot.')
        return

    # Sort by scale then effect size
    scale_order = {'Colony surface': 0, '3D macro': 1, 'Light Microscopy': 2}
    forest_data.sort(key=lambda x: (scale_order.get(x[5], 9), -abs(x[1])))

    n_items = len(forest_data)
    fig_h = max(80, 18 * n_items + 40)

    fig, ax = plt.subplots(figsize=(140 * MM, fig_h * MM))
    fig.subplots_adjust(left=0.45, right=0.88, top=0.94, bottom=0.08)

    y_pos = np.arange(n_items)[::-1]
    current_scale = None
    scale_colors = {'Colony surface': '#1976D2', '3D macro': '#FF9800',
                    'Light Microscopy': '#4CAF50'}

    for i, (label, d, ci_lo, ci_hi, p, scale, cat) in enumerate(forest_data):
        yp = y_pos[i]
        color = scale_colors.get(scale, '#666')

        # Draw CI line + point
        ax.plot([ci_lo, ci_hi], [yp, yp], color=color, lw=1.5, solid_capstyle='round')
        ax.plot(d, yp, 'o', color=color, markersize=6, zorder=5)

        # Significance stars
        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        else:
            star = ''
        if star:
            ax.text(ci_hi + 0.1, yp, star, va='center', fontsize=8, fontweight='bold', color=color)

        # Scale separator
        if scale != current_scale:
            if current_scale is not None:
                ax.axhline(yp + 0.5, color='#ddd', lw=0.5, ls='--')
            current_scale = scale

    # Labels
    labels = [f[0] for f in forest_data]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.5)

    ax.axvline(0, color='k', lw=0.8, ls='-')
    ax.set_xlabel("Cohen's d  (Asp vs Muc)", fontsize=9)
    ax.set_title('Effect sizes across imaging scales', fontsize=10, fontweight='bold')

    # Scale legend
    for scale, color in scale_colors.items():
        ax.plot([], [], 'o-', color=color, label=scale, markersize=5, lw=1.5)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    for ext in ('.png', '.pdf'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUT_DIR / f'effect_size_forest_plot{ext}', **kw)
    plt.close(fig)

    print(f'\nForest plot saved.')
    print(f'All outputs in {OUT_DIR}/')


if __name__ == '__main__':
    main()
