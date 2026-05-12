#!/usr/bin/env python3
"""Supplementary Figure S27: Erosion-survival retention vs iteration count
for all 9 light-microscopy images.

Single-panel plot of retention fraction vs erosion iteration (1-20).
Aspergillus traces in shades of green; Mucor traces in shades of gray.
Mean +/- SEM band per genus. Vertical line at iteration 10 (canonical).
Demonstrates the genus separation persists across the full erosion sweep.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, apply_style, save_fig

# Reuse from FigureHyphae/code
sys.path.insert(0, '/Volumes/T7/FINAL OSF/FigureHyphae/code')
from step_final_rebuild_all import (  # noqa: E402
    load_gray, seg_micro, MICRO_IMAGES, MICRO_DIR
)

C_ASP = '#4CAF50'
C_MUC = '#757575'
N_ITER_MAX = 20

OUT_DIR = OUTPUT_DIR / 'S27'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def erosion_retention_curve(mask, n_iter):
    """Return retention fraction at iter 0..n_iter."""
    orig = mask.sum()
    out = [1.0]
    cur = mask.copy()
    for i in range(1, n_iter + 1):
        cur = ndi.binary_erosion(cur)
        out.append(cur.sum() / orig if orig > 0 else 0.0)
    return np.array(out)


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(160 * MM, 95 * MM))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.18)

    asp_curves, muc_curves = [], []
    for fname, genus, label, mag in MICRO_IMAGES:
        img = load_gray(MICRO_DIR / fname)
        lo, hi = np.percentile(img, [0.5, 99.5])
        img_n = np.clip((img - lo) / max(hi - lo, 1), 0, 1)
        mask = seg_micro(img, label)
        curve = erosion_retention_curve(mask, N_ITER_MAX)
        if genus == 'Aspergillus':
            asp_curves.append(curve)
            color = C_ASP
        else:
            muc_curves.append(curve)
            color = C_MUC
        # individual trace
        ax.plot(range(N_ITER_MAX + 1), curve, '-', color=color,
                lw=0.7, alpha=0.45, zorder=2)
        print(f'  {fname:<22} ({genus[:3]} {mag}X) iter10={curve[10]:.3f}')

    # mean +/- SEM bands
    iters = np.arange(N_ITER_MAX + 1)
    for curves, color, name in [(np.array(asp_curves), C_ASP, 'Aspergillus'),
                                 (np.array(muc_curves), C_MUC, 'Mucor')]:
        m = curves.mean(0)
        s = curves.std(0, ddof=1) / np.sqrt(curves.shape[0])
        ax.plot(iters, m, '-', color=color, lw=2.0, label=name, zorder=4)
        ax.fill_between(iters, m - s, m + s, color=color, alpha=0.20, zorder=3)

    # Canonical line
    ax.axvline(10, color='black', ls='--', lw=0.7, alpha=0.5, zorder=1)
    ax.text(10.3, 1.02, 'canonical', fontsize=TICK_SIZE - 0.5,
            color='black', alpha=0.7)

    ax.set_xlabel('Erosion iteration (1-px disk per step)',
                  fontsize=LABEL_SIZE, labelpad=2)
    ax.set_ylabel('Tissue retention fraction', fontsize=LABEL_SIZE, labelpad=2)
    ax.set_xlim(0, N_ITER_MAX)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=TICK_SIZE - 0.5)
    ax.legend(fontsize=TICK_SIZE - 0.5, loc='upper right', frameon=False)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)

    # Annotation at iter 10
    asp_arr = np.array(asp_curves)[:, 10]
    muc_arr = np.array(muc_curves)[:, 10]
    txt = (f'At iteration 10:\n'
           f'  \\textit{{Aspergillus}}: {asp_arr.mean():.2f} ± {asp_arr.std(ddof=1):.2f}\n'
           f'  \\textit{{Mucor}}: {muc_arr.mean():.2f} ± {muc_arr.std(ddof=1):.2f}')
    # Use plain text since LaTeX rendering may not be available
    txt = (f'At iteration 10:\n'
           f'  Aspergillus: {asp_arr.mean():.2f} ± {asp_arr.std(ddof=1):.2f}\n'
           f'  Mucor: {muc_arr.mean():.2f} ± {muc_arr.std(ddof=1):.2f}')
    ax.text(0.40, 0.92, txt, transform=ax.transAxes,
            fontsize=TICK_SIZE - 0.5, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    save_fig(fig, str(OUT_DIR / 'FigureS27_erosion_curves'))
    plt.close(fig)
    print('Saved S27')


if __name__ == '__main__':
    main()
