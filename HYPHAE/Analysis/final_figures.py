#!/usr/bin/env python3
"""Generate final publication figures.

1. final_figure: 4 boxplots (FFT colony, FFT 3D, local density CV, lacunarity)
2. effect_size_forest_plot: 4-row forest plot of same metrics
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE    = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / 'Final Results'

MM = 1 / 25.4
C_ASP = '#4CAF50'
C_MUC = '#757575'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'svg.fonttype': 'none',
})


def add_bracket(ax, sa, sm, y_offset_frac=0.08):
    ymax = max(np.max(sa), np.max(sm))
    ymin = min(np.min(sa), np.min(sm))
    rng = ymax - ymin
    y = ymax + rng * y_offset_frac
    ax.plot([1, 1, 2, 2], [y, y + rng * 0.03, y + rng * 0.03, y], 'k-', lw=0.6)
    t, p = stats.ttest_ind(sa, sm, equal_var=False)
    if p < 0.001:
        p_str = f'p = {p:.1e}'
    else:
        p_str = f'p = {p:.3f}'
    ax.text(1.5, y + rng * 0.05, p_str, ha='center', fontsize=6.5)


def boxstrip(ax, sa, sm, ylabel):
    bp = ax.boxplot([sa, sm], positions=[1, 2], widths=0.45,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', lw=1.2),
                    whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8))
    bp['boxes'][0].set_facecolor(C_ASP)
    bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor(C_MUC)
    bp['boxes'][1].set_alpha(0.55)
    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.uniform(-0.10, 0.10, len(sa)), sa,
               s=14, c=C_ASP, alpha=0.8, edgecolors='none', zorder=3)
    ax.scatter(2 + rng.uniform(-0.10, 0.10, len(sm)), sm,
               s=14, c=C_MUC, alpha=0.8, edgecolors='none', zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Aspergillus', 'Mucor'], style='italic', fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    add_bracket(ax, sa, sm)


def cohens_d_ci(d, n1, n2):
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    z = 1.96
    return d - z * se, d + z * se


def load_data():
    """Load the 4 datasets and return (label, asp_values, muc_values, scale_label)."""
    datasets = []

    # 1. FFT colony surface
    fft = pd.read_csv(OUT_DIR / 'fft_spectral_slopes.csv')
    datasets.append((
        r'Spectral slope $\alpha$' + '\n(colony surface)',
        fft.loc[fft['genus'] == 'Aspergillus', 'alpha'].values,
        fft.loc[fft['genus'] == 'Mucor', 'alpha'].values,
        'Colony surface',
    ))

    # 2. FFT 3D macro
    session_path = BASE / 'Analysis' / 'results' / '3d_overlays' / 'roi_session.json'
    with open(session_path) as f:
        session = json.load(f)
    saved = {k: v for k, v in session.items()
             if not k.startswith('_') and v.get('status') != 'deleted'}
    asp_3d = np.array([v['alpha'] for v in saved.values()
                       if v.get('genus') == 'Aspergillus' and v.get('alpha') is not None])
    muc_3d = np.array([v['alpha'] for v in saved.values()
                       if v.get('genus') == 'Mucor' and v.get('alpha') is not None])
    datasets.append((
        r'Spectral slope $\alpha$' + '\n(3D macro)',
        asp_3d, muc_3d,
        '3D macro',
    ))

    # 3. Local density CV
    fungi = pd.read_csv('/Users/yany/Desktop/Leyun microscopy/analysis_outputs/fungi_metrics.csv')
    datasets.append((
        'Local density CV\n(light microscopy)',
        fungi.loc[fungi['label'] == 'Green', 'local_density_cv'].values,
        fungi.loc[fungi['label'] == 'White', 'local_density_cv'].values,
        'Light Microscopy',
    ))

    # 4. Lacunarity
    frag = pd.read_csv(OUT_DIR / 'fragmentation_results.csv')
    datasets.append((
        'Lacunarity\n(3D macro)',
        frag.loc[frag['genus'] == 'Aspergillus', 'lacunarity_max_scale'].values,
        frag.loc[frag['genus'] == 'Mucor', 'lacunarity_max_scale'].values,
        '3D macro',
    ))

    return datasets


def make_final_figure(datasets):
    fig, axes = plt.subplots(1, 4, figsize=(180 * MM, 65 * MM))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18, wspace=0.50)

    for ax, (ylabel, sa, sm, _) in zip(axes, datasets):
        boxstrip(ax, sa, sm, ylabel)

    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUT_DIR / f'final_figure{ext}', **kw)
    plt.close(fig)
    print(f'Saved: final_figure.{{png,pdf,svg}}')


def make_forest_plot(datasets):
    scale_colors = {
        'Colony surface': '#1976D2',
        '3D macro': '#FF9800',
        'Light Microscopy': '#4CAF50',
    }

    # Compute effect sizes
    items = []
    for label, sa, sm, scale in datasets:
        clean_label = label.replace('\n', ' ')
        n1, n2 = len(sa), len(sm)
        pooled = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2) / (n1+n2-2))
        d = (sa.mean() - sm.mean()) / pooled if pooled > 0 else 0
        ci_lo, ci_hi = cohens_d_ci(d, n1, n2)
        t, p = stats.ttest_ind(sa, sm, equal_var=False)
        items.append((clean_label, d, ci_lo, ci_hi, p, scale))

    # Sort by scale then |d|
    scale_order = {'Colony surface': 0, '3D macro': 1, 'Light Microscopy': 2}
    items.sort(key=lambda x: (scale_order.get(x[5], 9), -abs(x[1])))

    n = len(items)
    fig, ax = plt.subplots(figsize=(120 * MM, 55 * MM))
    fig.subplots_adjust(left=0.42, right=0.92, top=0.92, bottom=0.12)

    y_pos = np.arange(n)[::-1]

    for i, (label, d, ci_lo, ci_hi, p, scale) in enumerate(items):
        yp = y_pos[i]
        color = scale_colors.get(scale, '#666')
        ax.plot([ci_lo, ci_hi], [yp, yp], color=color, lw=2.5, solid_capstyle='round')
        ax.plot(d, yp, 'o', color=color, markersize=8, zorder=5)

        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        else:
            star = ''
        x_text = max(ci_hi, abs(ci_lo)) + 0.15
        ax.text(ci_hi + 0.12, yp, star, va='center', fontsize=9,
                fontweight='bold', color=color)

    labels = [item[0] for item in items]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel("Cohen's d  (Asp vs Muc)", fontsize=9)
    ax.set_title('Effect sizes across imaging scales', fontsize=10, fontweight='bold')

    for scale, color in scale_colors.items():
        ax.plot([], [], 'o-', color=color, label=scale, markersize=6, lw=2)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    for ext in ('.png', '.pdf'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUT_DIR / f'effect_size_forest_plot{ext}', **kw)
    plt.close(fig)
    print(f'Saved: effect_size_forest_plot.{{png,pdf}}')


def main():
    datasets = load_data()
    make_final_figure(datasets)
    make_forest_plot(datasets)
    print(f'\nAll saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
