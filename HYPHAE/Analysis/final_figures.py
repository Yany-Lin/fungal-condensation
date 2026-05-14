#!/usr/bin/env python3
"""Generate final publication figures for Figure 4.

4 boxplots:
  C: FFT spectral slope alpha (3D manual ROIs)
  D: Local density CV (3D manual ROIs)
  E: Local density CV (light microscopy, analyze_fungi.py segmentation)
  F: Local density p90-p10 range (light microscopy)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import laplace

BASE    = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / 'Final Results'
ROI_DIR = BASE / 'Analysis' / 'results' / '3d_overlays'

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


def otsu_uint8(img8):
    hist = np.bincount(img8.ravel(), minlength=256).astype(float)
    total = hist.sum()
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    sum_bg = np.cumsum(hist * np.arange(256))
    mean_bg = sum_bg / np.maximum(w_bg, 1)
    mean_fg = (sum_bg[-1] - sum_bg) / np.maximum(w_fg, 1)
    var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(var))


def load_gray(path):
    from PIL import Image
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def load_data():
    """Load the 4 datasets."""
    datasets = []

    # ── Panel C: FFT alpha on 3D ROIs ──
    session_path = ROI_DIR / 'roi_session.json'
    with open(session_path) as f:
        session = json.load(f)
    saved = {k: v for k, v in session.items()
             if not k.startswith('_') and v.get('status') != 'deleted'}
    asp_fft = np.array([v['alpha'] for v in saved.values()
                        if v.get('genus') == 'Aspergillus' and v.get('alpha') is not None])
    muc_fft = np.array([v['alpha'] for v in saved.values()
                        if v.get('genus') == 'Mucor' and v.get('alpha') is not None])
    datasets.append((
        r'Spectral slope $\alpha$' + '\n(3D colony surface)',
        asp_fft, muc_fft,
        '3D',
    ))

    # ── Panel D: Local density CV on 3D ROIs ──
    # No segmentation — analyze the raw crop directly.
    # The manual ROI IS the region of interest. Dark pixels = tissue, light = gaps.
    # Measure mean intensity per patch; CV of patch means = spatial heterogeneity.
    cal = session['_calibration']['um_per_px']
    patch_px = int(round(256.0 / cal))  # 256 um patches

    cv_asp, cv_muc = [], []
    for key, val in saved.items():
        genus = val.get('genus')
        roi_path = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
        if not roi_path.exists():
            continue
        img = load_gray(roi_path)
        # Normalize to [0,1] so patches are comparable across images
        lo, hi = np.percentile(img, [0.5, 99.5])
        if hi <= lo: hi = lo + 1
        img_norm = np.clip((img - lo) / (hi - lo), 0, 1)
        h, w = img_norm.shape
        patch_means = []
        for y0 in range(0, h - patch_px + 1, patch_px):
            for x0 in range(0, w - patch_px + 1, patch_px):
                patch_means.append(img_norm[y0:y0+patch_px, x0:x0+patch_px].mean())
        if len(patch_means) < 4:
            continue
        patch_means = np.array(patch_means)
        cv = patch_means.std(ddof=1) / patch_means.mean() if patch_means.mean() > 0 else 0
        if genus == 'Aspergillus':
            cv_asp.append(cv)
        else:
            cv_muc.append(cv)

    datasets.append((
        'Local density CV\n(3D colony surface)',
        np.array(cv_asp), np.array(cv_muc),
        '3D',
    ))

    # ── Panels E & F: from light microscopy (already computed) ──
    # Legacy Leyun-side CSV; fall back to repo-local copy under HYPHAE/Analysis/results/.
    _legacy = Path('/Users/yany/Desktop/Leyun microscopy/analysis_outputs/fungi_metrics.csv')
    _local  = Path(__file__).resolve().parent / 'results' / 'fungi_metrics.csv'
    fungi = pd.read_csv(_legacy if _legacy.exists() else _local)

    datasets.append((
        'Local density CV\n(light microscopy)',
        fungi.loc[fungi['label'] == 'Green', 'local_density_cv'].values,
        fungi.loc[fungi['label'] == 'White', 'local_density_cv'].values,
        'Light Microscopy',
    ))

    datasets.append((
        'Local density\np90 \u2013 p10\n(light microscopy)',
        fungi.loc[fungi['label'] == 'Green', 'local_density_p90_minus_p10'].values,
        fungi.loc[fungi['label'] == 'White', 'local_density_p90_minus_p10'].values,
        'Light Microscopy',
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


def main():
    datasets = load_data()

    # Print stats
    labels = ['Panel C (FFT 3D)', 'Panel D (CV 3D)',
              'Panel E (CV Light Micro)', 'Panel F (p90-p10 Light Micro)']
    for label, (ylabel, sa, sm, _) in zip(labels, datasets):
        t, p = stats.ttest_ind(sa, sm, equal_var=False)
        n1, n2 = len(sa), len(sm)
        pooled = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2) / (n1+n2-2))
        d = (sa.mean() - sm.mean()) / pooled if pooled > 0 else 0
        print(f'{label}:')
        print(f'  Asp: {sa.mean():.4f} +/- {sa.std(ddof=1):.4f} (n={n1})')
        print(f'  Muc: {sm.mean():.4f} +/- {sm.std(ddof=1):.4f} (n={n2})')
        print(f'  Welch t={t:.3f}, p={p:.6f}, d={d:.3f}')
        print()

    make_final_figure(datasets)
    print(f'All saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
