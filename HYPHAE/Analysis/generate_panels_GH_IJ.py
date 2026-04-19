#!/usr/bin/env python3
"""Generate Figure 3 panels G, H, I, J.

G, H: Leyun microscopy images (Aspergillus vs Mucor, 40X)
I:    Spectral slope boxplot (from existing 1D transect data)
J:    Spectral slope vs delta scatter
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from scipy import stats

# ---------- paths ----------
THIS_DIR   = Path(__file__).resolve().parent
MICRO_DIR  = THIS_DIR.parent / 'Leyun microscopy'
OUT_DIR    = THIS_DIR.parent / 'test'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- style (match existing Figure 3) ----------
MM = 1 / 25.4
TS = 7.0       # tick label
LS = 8.5       # axis label
PL = 12.0      # panel label
LW = 0.6       # axis linewidth

plt.rcParams.update({
    'font.family':     'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':        TS,
    'axes.linewidth':   LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size':  3.5, 'ytick.major.size':  3.5,
    'xtick.direction':  'out', 'ytick.direction':  'out',
    'svg.fonttype':     'none',
})

C_ASP  = '#4CAF50'
C_MUC  = '#757575'

# Delta values from condensation assay (um, n=5 per genus)
DELTA_ASP = np.array([279.5, 316.0, 297.8, 285.6, 311.7])
DELTA_MUC = np.array([198.5, 126.2, 120.0, 131.6, 123.0])

# Microscopy images for panels G, H
IMG_G = MICRO_DIR / 'Pink_40X_2.TIF'   # Aspergillus 40X (better of the two)
IMG_H = MICRO_DIR / 'White_40X_1.TIF'  # Mucor 40X

# Calibration at 40X: 0.1887 um/px (from analysis_summary.md)
UM_PER_PX_40X = 0.1887


def add_scalebar(ax, um_per_px, bar_um=50, color='white', loc='lower right'):
    """Add a scale bar to an image axis."""
    bar_px = bar_um / um_per_px
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w = xlim[1] - xlim[0]
    h = ylim[0] - ylim[1]  # inverted y for images

    if 'right' in loc:
        x0 = xlim[1] - 0.05 * w - bar_px
    else:
        x0 = xlim[0] + 0.05 * w
    if 'lower' in loc:
        y0 = ylim[0] - 0.05 * h
    else:
        y0 = ylim[1] + 0.05 * h + h * 0.02

    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, lw=2.5, solid_capstyle='butt')
    ax.text(x0 + bar_px / 2, y0 - h * 0.03, f'{bar_um} um',
            color=color, fontsize=TS, ha='center', va='top', fontweight='bold')


def panel_GH(fig, gs_g, gs_h):
    """Panels G, H: microscopy images."""
    for gs, img_path, label, panel_letter in [
        (gs_g, IMG_G, 'Aspergillus', 'G'),
        (gs_h, IMG_H, 'Mucor', 'H'),
    ]:
        ax = fig.add_subplot(gs)
        img = np.array(Image.open(img_path))
        # Center-crop to square
        h, w = img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = img[y0:y0+side, x0:x0+side]
        ax.imshow(crop, cmap='gray' if crop.ndim == 2 else None)
        ax.axis('off')
        add_scalebar(ax, UM_PER_PX_40X, bar_um=50)
        # Genus label
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                fontsize=TS, fontstyle='italic', color='white',
                va='top', ha='left', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5, lw=0))
        # Panel letter
        ax.text(-0.05, 1.05, panel_letter, transform=ax.transAxes,
                fontsize=PL, fontweight='bold', va='top', ha='right')


def panel_I(ax):
    """Panel I: spectral slope boxplot from automated 2D FFT data."""
    df = pd.read_csv(THIS_DIR / 'spectral_2d_results.csv')
    sa = df[df.genus == 'Aspergillus']['alpha'].astype(float).values
    sm = df[df.genus == 'Mucor']['alpha'].astype(float).values

    t_stat, t_p = stats.ttest_ind(sa, sm, equal_var=False)

    bp = ax.boxplot([sa, sm], positions=[1, 2], widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', lw=1.2),
                    whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8))
    bp['boxes'][0].set_facecolor(C_ASP)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(C_MUC)
    bp['boxes'][1].set_alpha(0.6)

    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.uniform(-0.15, 0.15, len(sa)), sa,
               s=12, c=C_ASP, alpha=0.7, edgecolors='none', zorder=3)
    ax.scatter(2 + rng.uniform(-0.15, 0.15, len(sm)), sm,
               s=12, c=C_MUC, alpha=0.7, edgecolors='none', zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Aspergillus', 'Mucor'], fontsize=TS, style='italic')
    ax.set_ylabel(r'Spectral slope $\alpha$', fontsize=LS)

    # Significance bracket
    y_max = max(sa.max(), sm.max()) + 0.03
    ax.plot([1, 1, 2, 2], [y_max, y_max + 0.02, y_max + 0.02, y_max],
            color='black', lw=0.8)
    ax.text(1.5, y_max + 0.025, f'p = {t_p:.1e}',
            ha='center', va='bottom', fontsize=TS - 1)

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TS)
    ax.text(-0.18, 1.08, 'I', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')


def panel_J(ax):
    """Panel J: spectral slope vs delta scatter."""
    df = pd.read_csv(THIS_DIR / 'spectral_2d_results.csv')
    sa = df[df.genus == 'Aspergillus']['alpha'].astype(float).values
    sm = df[df.genus == 'Mucor']['alpha'].astype(float).values

    slope_means = [sa.mean(), sm.mean()]
    slope_sds   = [sa.std(ddof=1), sm.std(ddof=1)]
    delta_means = [DELTA_ASP.mean(), DELTA_MUC.mean()]
    delta_sds   = [DELTA_ASP.std(ddof=1), DELTA_MUC.std(ddof=1)]

    ax.errorbar(slope_means[0], delta_means[0],
                xerr=slope_sds[0], yerr=delta_sds[0],
                fmt='o', color=C_ASP, markersize=7,
                markeredgecolor='white', markeredgewidth=0.5,
                capsize=3, capthick=0.8, elinewidth=0.8,
                label='Aspergillus', zorder=3)
    ax.errorbar(slope_means[1], delta_means[1],
                xerr=slope_sds[1], yerr=delta_sds[1],
                fmt='o', color=C_MUC, markersize=7,
                markeredgecolor='white', markeredgewidth=0.5,
                capsize=3, capthick=0.8, elinewidth=0.8,
                label='Mucor', zorder=3)
    ax.plot(slope_means, delta_means, color='gray', ls='--', lw=0.7, alpha=0.5, zorder=1)

    ax.set_xlabel(r'Spectral slope $\alpha$', fontsize=LS)
    ax.set_ylabel(r'$\bar{\delta}$ ($\mu$m)', fontsize=LS)
    ax.legend(fontsize=TS - 1, framealpha=0.9, loc='best',
              handletextpad=0.3, borderpad=0.3, prop={'style': 'italic'})

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TS)
    ax.text(-0.22, 1.08, 'J', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')


def main():
    fig = plt.figure(figsize=(180 * MM, 90 * MM))
    gs = fig.add_gridspec(1, 4, wspace=0.45,
                          left=0.02, right=0.97, top=0.88, bottom=0.15,
                          width_ratios=[1, 1, 1, 1])

    panel_GH(fig, gs[0], gs[1])

    ax_i = fig.add_subplot(gs[2])
    panel_I(ax_i)

    ax_j = fig.add_subplot(gs[3])
    panel_J(ax_j)

    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUT_DIR / f'panels_GHIJ{ext}', **kw)
    plt.close(fig)
    print(f'Saved panels_GHIJ.* to {OUT_DIR}')


if __name__ == '__main__':
    main()
