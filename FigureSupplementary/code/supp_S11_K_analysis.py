#!/usr/bin/env python3
"""Supplementary Figure S11: K evaporation rate analysis (2x2 assembly)."""

import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, PANEL_LBL, apply_style, save_fig

SRC = Path(__file__).resolve().parents[2] / 'additions' / '4_K_distance_evaporation'
OUT = OUTPUT_DIR / 'S11'

PANELS = [
    (SRC / 'K_hexbin.png', 'a'),
    (SRC / 'size_matched_KM.png', 'b'),
    (SRC / 'd2_collapse.png', 'c'),
    (SRC / 'K_universal_vs_delta.png', 'd'),
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

    save_fig(fig, str(OUT / 'FigureS11_K_analysis'))
    plt.close(fig)
    print(f'Saved: {OUT}/FigureS11_K_analysis.png/.pdf/.svg')

if __name__ == '__main__':
    main()
