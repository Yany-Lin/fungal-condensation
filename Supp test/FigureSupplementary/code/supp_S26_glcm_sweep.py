#!/usr/bin/env python3
"""Supplementary Figure S26: GLCM parameter robustness.

The FFT-cross-validation panel (SuppFig9) used GLCM contrast at distance=1
and 4 orientations averaged. This figure re-computes GLCM contrast across
a parameter grid and shows that the Asp/Mucor genus ordering and the
Spearman correlation with FFT alpha are preserved across reasonable
parameter choices.

Sweep:
  distance d in {1, 2, 4, 8} px
  gray-level quantization L in {32, 64, 128, 256}

Two panels:
  A — Asp/Mucor median GLCM contrast across the grid (heatmap of values).
  B — Spearman r (alpha vs GLCM) across the grid; values ~ -0.6 throughout.
"""
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, apply_style, save_fig

ROI_DIR = Path('/Volumes/T7/FINAL OSF/HYPHAE/Analysis/results/3d_overlays')
SESSION_JSON = ROI_DIR / 'roi_session.json'
FFT_CSV = Path('/Volumes/T7/FINAL OSF/FigureHyphae/output/fft_per_roi.csv')

DISTANCES = [1, 2, 4, 8]
GRAY_LEVELS = [32, 64, 128, 256]

C_ASP = '#4CAF50'
C_MUC = '#757575'

OUT_DIR = OUTPUT_DIR / 'S26'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def glcm_contrast(img, d, L):
    """Quantize image to L gray levels, then compute average GLCM contrast
    over 4 cardinal directions at offset d."""
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo: hi = lo + 1
    q = np.clip((img - lo) / (hi - lo) * (L - 1), 0, L - 1).astype(np.int32)
    contrasts = []
    # horizontal
    a = q[:, d:].astype(float); b = q[:, :-d].astype(float)
    contrasts.append(((a - b) ** 2).mean())
    # vertical
    a = q[d:, :].astype(float); b = q[:-d, :].astype(float)
    contrasts.append(((a - b) ** 2).mean())
    # diagonal
    a = q[d:, d:].astype(float); b = q[:-d, :-d].astype(float)
    contrasts.append(((a - b) ** 2).mean())
    # anti-diagonal
    a = q[d:, :-d].astype(float); b = q[:-d, d:].astype(float)
    contrasts.append(((a - b) ** 2).mean())
    return float(np.mean(contrasts))


def collect_rois():
    with open(SESSION_JSON) as f:
        session = json.load(f)
    saved = {k: v for k, v in session.items()
             if not k.startswith('_') and v.get('status') != 'deleted'}
    rows = []
    for key, val in saved.items():
        genus = val.get('genus')
        if genus not in ('Aspergillus', 'Mucor'):
            continue
        roi_path = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
        if roi_path.exists():
            rows.append((genus, roi_path, key))
    return rows


def main():
    apply_style()

    # Load alpha values from existing CSV
    import pandas as pd
    fft_df = pd.read_csv(FFT_CSV).set_index('file')

    rois = collect_rois()
    n_d, n_L = len(DISTANCES), len(GRAY_LEVELS)

    asp_med = np.zeros((n_d, n_L))
    muc_med = np.zeros((n_d, n_L))
    rho = np.zeros((n_d, n_L))

    print(f'Computing GLCM at {n_d}x{n_L} = {n_d*n_L} parameter combos for {len(rois)} ROIs...')
    for di, d in enumerate(DISTANCES):
        for Li, L in enumerate(GRAY_LEVELS):
            asp_vals, muc_vals, alpha_for_corr, glcm_for_corr = [], [], [], []
            for genus, roi_path, key in rois:
                img = load_gray(roi_path)
                g = glcm_contrast(img, d, L)
                if genus == 'Aspergillus': asp_vals.append(g)
                else: muc_vals.append(g)
                # match key to fft_df row by stem name
                stem = Path(key).stem.replace('_roi', '') + '.JPG'
                if stem in fft_df.index:
                    alpha_for_corr.append(fft_df.loc[stem, 'alpha'])
                    glcm_for_corr.append(g)
            asp_med[di, Li] = np.median(asp_vals)
            muc_med[di, Li] = np.median(muc_vals)
            if len(alpha_for_corr) >= 5:
                rho[di, Li], _ = stats.spearmanr(alpha_for_corr, glcm_for_corr)
            else:
                rho[di, Li] = np.nan
        print(f'  d={d} done')

    # Save figure
    fig = plt.figure(figsize=(190 * MM, 100 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          left=0.08, right=0.97, top=0.88, bottom=0.18, wspace=0.30)

    # Panel A: Asp/Muc ratio of median GLCM contrast
    axA = fig.add_subplot(gs[0])
    ratio = asp_med / muc_med
    # Asp has lower texture so ratio < 1; flip to consistent (Muc/Asp > 1)
    ratio = muc_med / asp_med
    im = axA.imshow(ratio, cmap='viridis', aspect='auto', origin='lower')
    axA.set_xticks(range(n_L)); axA.set_xticklabels(GRAY_LEVELS)
    axA.set_yticks(range(n_d)); axA.set_yticklabels(DISTANCES)
    axA.set_xlabel('Gray-level quantization $L$', fontsize=LABEL_SIZE, labelpad=2)
    axA.set_ylabel('Pixel offset $d$', fontsize=LABEL_SIZE, labelpad=2)
    axA.set_title(r'Mucor/Aspergillus GLCM contrast ratio',
                  fontsize=LABEL_SIZE, pad=6)
    cb = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=TICK_SIZE - 1)
    # annotate cells
    for di in range(n_d):
        for Li in range(n_L):
            axA.text(Li, di, f'{ratio[di, Li]:.1f}',
                     ha='center', va='center', fontsize=TICK_SIZE - 1,
                     color='white' if ratio[di, Li] < ratio.mean() else 'black')
    axA.text(-0.18, 1.10, 'A', transform=axA.transAxes,
             fontsize=11, fontweight='bold', va='top')
    axA.tick_params(labelsize=TICK_SIZE - 0.5)

    # Panel B: Spearman correlation alpha vs GLCM
    axB = fig.add_subplot(gs[1])
    im = axB.imshow(rho, cmap='RdBu_r', aspect='auto', origin='lower',
                    vmin=-0.8, vmax=0.8)
    axB.set_xticks(range(n_L)); axB.set_xticklabels(GRAY_LEVELS)
    axB.set_yticks(range(n_d)); axB.set_yticklabels(DISTANCES)
    axB.set_xlabel('Gray-level quantization $L$', fontsize=LABEL_SIZE, labelpad=2)
    axB.set_ylabel('Pixel offset $d$', fontsize=LABEL_SIZE, labelpad=2)
    axB.set_title(r'Spearman $r$: $\alpha$ vs GLCM contrast',
                  fontsize=LABEL_SIZE, pad=6)
    cb = plt.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=TICK_SIZE - 1)
    for di in range(n_d):
        for Li in range(n_L):
            axB.text(Li, di, f'{rho[di, Li]:.2f}',
                     ha='center', va='center', fontsize=TICK_SIZE - 1,
                     color='white' if abs(rho[di, Li]) > 0.5 else 'black')
    axB.text(-0.18, 1.10, 'B', transform=axB.transAxes,
             fontsize=11, fontweight='bold', va='top')
    axB.tick_params(labelsize=TICK_SIZE - 0.5)

    save_fig(fig, str(OUT_DIR / 'FigureS26_glcm_sweep'))
    plt.close(fig)
    print(f'\nSaved S26')
    print(f'Mucor/Asp ratio range: {ratio.min():.2f}–{ratio.max():.2f}')
    print(f'Spearman r range:      {np.nanmin(rho):.2f}–{np.nanmax(rho):.2f}')


if __name__ == '__main__':
    main()
