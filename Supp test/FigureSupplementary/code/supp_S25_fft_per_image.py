#!/usr/bin/env python3
"""Supplementary Figure S25: Per-image FFT power spectra with linear fits.

24 panels (13 Aspergillus + 11 Mucor) showing log-log radial PSD with the
[0.02, 0.40] cycles/px fitting band shaded and the fitted line overlaid.
Per-panel α slope annotated.
"""
import json
import sys
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
TILE = 256
STRIDE = 128
FREQ_LO, FREQ_HI = 0.02, 0.40
C_ASP = '#4CAF50'
C_MUC = '#757575'

OUT_DIR = OUTPUT_DIR / 'S25'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gray(p):
    with Image.open(p) as im:
        arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def tile_radial(tile):
    n = tile.shape[0]
    c = tile - tile.mean()
    win = np.outer(np.hanning(n), np.hanning(n))
    fft = np.fft.fftshift(np.fft.fft2(c * win))
    p2d = np.abs(fft) ** 2
    cy, cx = n // 2, n // 2
    y, x = np.arange(n) - cy, np.arange(n) - cx
    yy, xx = np.meshgrid(y, x, indexing='ij')
    r = np.sqrt(xx ** 2 + yy ** 2).astype(int)
    rm = n // 2
    rf = np.clip(r.ravel(), 0, rm - 1)
    rs = np.bincount(rf, weights=p2d.ravel(), minlength=rm)
    rc = np.bincount(rf, minlength=rm).astype(float)
    rc[rc == 0] = 1
    freq = np.arange(rm) / n
    return freq[1:], (rs / rc)[1:]


def fft_image_spectrum(img):
    h, w = img.shape
    all_powers = []
    for y0 in range(0, h - TILE + 1, STRIDE):
        for x0 in range(0, w - TILE + 1, STRIDE):
            t = img[y0:y0 + TILE, x0:x0 + TILE]
            if t.std() < 3.0: continue
            f, p = tile_radial(t)
            all_powers.append(p)
    if not all_powers:
        return None, None, None
    mp = np.mean(all_powers, axis=0)
    f = np.arange(1, TILE // 2) / TILE
    valid = (f >= FREQ_LO) & (f <= FREQ_HI) & (mp > 0)
    if valid.sum() < 10:
        return f, mp, None
    sl, ic, r, p, se = stats.linregress(np.log10(f[valid]), np.log10(mp[valid]))
    return f, mp, (sl, ic)


def collect_rois():
    """Return list of (genus, roi_path, key) tuples."""
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
    rows.sort(key=lambda r: (r[0] != 'Aspergillus', r[2]))
    return rows


def main():
    apply_style()
    rois = collect_rois()
    n = len(rois)
    print(f'Found {n} ROIs')
    n_cols, n_rows = 6, int(np.ceil(n / 6))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(190 * MM, 30 * n_rows * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.07,
                        hspace=0.40, wspace=0.10)
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

    f_band = None
    for i, (genus, roi_path, key) in enumerate(rois):
        ax = axes_flat[i]
        img = load_gray(roi_path)
        f, mp, fit = fft_image_spectrum(img)
        color = C_ASP if genus == 'Aspergillus' else C_MUC
        if f is None:
            ax.text(0.5, 0.5, 'no PSD', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            continue
        if f_band is None:
            f_band = (f >= FREQ_LO) & (f <= FREQ_HI)

        # PSD line
        ax.loglog(f, mp, '-', color=color, lw=0.8, alpha=0.85)
        # Shaded fitting band
        ax.axvspan(FREQ_LO, FREQ_HI, color='gray', alpha=0.10, zorder=1)
        if fit is not None:
            sl, ic = fit
            f_fit = f[f_band]
            mp_fit = 10 ** (sl * np.log10(f_fit) + ic)
            ax.loglog(f_fit, mp_fit, '-', color='red', lw=1.0, alpha=0.85, zorder=3)
            ax.text(0.05, 0.05, f'$\\alpha$={sl:.2f}', transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=TICK_SIZE - 1.5,
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.7))

        ax.set_xlim(0.01, 0.5)
        ax.tick_params(labelsize=TICK_SIZE - 2, pad=1.5)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        # short label: filename stem only
        short = Path(key).stem.split('_')[-1] if '_' in Path(key).stem else Path(key).stem[-6:]
        ax.text(0.95, 0.95, f'{genus[:3]} {short}', transform=ax.transAxes,
                ha='right', va='top', fontsize=TICK_SIZE - 2,
                color=color, alpha=0.85)

    # blank out unused axes
    for j in range(len(rois), n_rows * n_cols):
        axes_flat[j].axis('off')

    fig.text(0.53, 0.020, 'Spatial frequency (cycles/px)',
             ha='center', fontsize=LABEL_SIZE)
    fig.text(0.018, 0.5, 'Power spectral density (a.u.)',
             va='center', rotation=90, fontsize=LABEL_SIZE)

    save_fig(fig, str(OUT_DIR / 'FigureS25_fft_per_image'))
    plt.close(fig)
    print(f'Saved S25 with {n} per-image PSDs')


if __name__ == '__main__':
    main()
