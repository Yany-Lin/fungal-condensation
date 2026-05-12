#!/usr/bin/env python3
"""Supplementary Figure S28: Within-tissue tubeness response histograms
for all 9 light-microscopy images.

3x3 grid: rows = magnification (10x, 20x, 40x); cols = Asp rep1, Asp rep2,
Mucor (only one Mucor image per magnification). Each panel shows the
within-tissue Hessian tubeness response distribution and annotated CV.
Demonstrates the bimodal-Asp / unimodal-Mucor pattern is consistent
across replicates and magnifications, not just one representative image.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, apply_style, save_fig

sys.path.insert(0, '/Volumes/T7/FINAL OSF/FigureHyphae/code')
from step_final_rebuild_all import (  # noqa: E402
    load_gray, seg_micro, multiscale_tubeness, MICRO_DIR
)

C_ASP = '#4CAF50'
C_MUC = '#757575'

# Group images by magnification, with explicit panel positions
LAYOUT = [
    # row 0: 10x
    ('Pink_10X_1.TIF', 'Asp', C_ASP, 0, 0, 'Aspergillus rep 1'),
    ('Pink_10X_2.TIF', 'Asp', C_ASP, 0, 1, 'Aspergillus rep 2'),
    ('White_10X_1.TIF', 'Muc', C_MUC, 0, 2, 'Mucor rep 1'),
    # row 1: 20x
    ('Pink_20X_1.TIF', 'Asp', C_ASP, 1, 0, 'Aspergillus rep 1'),
    ('Pink_20X_2.TIF', 'Asp', C_ASP, 1, 1, 'Aspergillus rep 2'),
    ('White_20X_1.TIF', 'Muc', C_MUC, 1, 2, 'Mucor rep 1'),
    # row 2: 40x
    ('Pink_40X_1.TIF', 'Asp', C_ASP, 2, 0, 'Aspergillus rep 1'),
    ('Pink_40X_2.TIF', 'Asp', C_ASP, 2, 1, 'Aspergillus rep 2'),
    ('White_40X_1.TIF', 'Muc', C_MUC, 2, 2, 'Mucor rep 1'),
]

OUT_DIR = OUTPUT_DIR / 'S28'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(190 * MM, 130 * MM),
                              sharex=True, sharey='row')
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.10,
                        hspace=0.35, wspace=0.18)

    rows_done = set()
    for (fname, gen_short, color, row, col, panel_label) in LAYOUT:
        ax = axes[row, col]
        img = load_gray(MICRO_DIR / fname)
        lo, hi = np.percentile(img, [0.5, 99.5])
        img_n = np.clip((img - lo) / max(hi - lo, 1), 0, 1)
        mask = seg_micro(img, 'Pink' if gen_short == 'Asp' else 'White')
        tub = multiscale_tubeness(img_n)
        vals = tub[mask]
        if len(vals) < 100:
            ax.text(0.5, 0.5, 'no tissue', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            continue
        cv = vals.std() / vals.mean() if vals.mean() > 0 else np.nan

        clipped = vals[vals < np.percentile(vals, 99)]
        ax.hist(clipped, bins=80, density=True, color=color, alpha=0.55,
                edgecolor='none')
        ax.tick_params(labelsize=TICK_SIZE - 1, pad=2)
        ax.text(0.97, 0.95,
                f'{panel_label}\nCV = {cv:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=TICK_SIZE - 0.5, color=color, fontweight='bold')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        # row label
        if col == 0 and row not in rows_done:
            mag = ['10×', '20×', '40×'][row]
            ax.text(-0.30, 0.5, mag, transform=ax.transAxes,
                    fontsize=TICK_SIZE + 1, fontweight='bold',
                    ha='right', va='center', rotation=90, color='black')
            rows_done.add(row)
        print(f'  {fname:<22} ({gen_short} {row}{col})  CV = {cv:.3f}')

    fig.text(0.53, 0.02, 'Hessian tubeness response (within tissue mask)',
             ha='center', fontsize=LABEL_SIZE)
    fig.text(0.015, 0.5, 'Density', va='center', rotation=90,
             fontsize=LABEL_SIZE)
    save_fig(fig, str(OUT_DIR / 'FigureS28_tubeness_hist_all9'))
    plt.close(fig)
    print('Saved S28')


if __name__ == '__main__':
    main()
