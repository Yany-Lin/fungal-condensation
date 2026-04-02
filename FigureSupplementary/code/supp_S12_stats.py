#!/usr/bin/env python3
"""Supplementary Figure S12: Statistical diagnostics (2x2 assembly)."""

import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, PANEL_LBL, apply_style, save_fig

BS = Path(__file__).resolve().parents[2] / 'additions' / '3_bootstrap_CIs'
COX = Path(__file__).resolve().parents[2] / 'additions' / '5_cox_PH_model'
OUT = OUTPUT_DIR / 'S12'

PANELS = [
    (BS / 'linearity_tests.png', 'a'),
    (BS / 'influence_diagnostics.png', 'b'),
    (COX / 'cox_forest_plot.png', 'c'),
    (COX / 'cox_schoenfeld.png', 'd'),
]

def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 160 * MM))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    for ax, (path, label) in zip(axes.flat, PANELS):
        img = mpimg.imread(str(path))
        ax.imshow(img)
        ax.axis('off')
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=PANEL_LBL, fontweight='bold', va='top')

    save_fig(fig, str(OUT / 'FigureS12_stats_diagnostics'))
    plt.close(fig)
    print(f'Saved: {OUT}/FigureS12_stats_diagnostics.png/.pdf/.svg')

if __name__ == '__main__':
    main()
