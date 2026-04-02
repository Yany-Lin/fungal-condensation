#!/usr/bin/env python3
"""Supplementary Figure S9: Bootstrap CIs on Figure 2 regressions (2x2 assembly)."""

import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, LABEL_SIZE, PANEL_LBL, apply_style, save_fig

SRC = Path(__file__).resolve().parents[2] / 'additions' / '3_bootstrap_CIs'
OUT = OUTPUT_DIR / 'S9'

PANELS = [
    (SRC / 'panel_2F_bootstrap.png', 'A'),
    (SRC / 'panel_2H_bootstrap.png', 'B'),
    (SRC / 'panel_2K_bootstrap.png', 'C'),
    (SRC / 'panel_2L_bootstrap.png', 'D'),
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
                fontsize=PANEL_LBL, va='top')

    save_fig(fig, str(OUT / 'FigureS9_bootstrap_CIs'))
    plt.close(fig)
    print(f'Saved: {OUT}/FigureS9_bootstrap_CIs.png/.pdf/.svg')

if __name__ == '__main__':
    main()
