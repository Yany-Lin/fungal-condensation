#!/usr/bin/env python3
"""Supplementary Figure S14: Bootstrap CIs + ANCOVA for panels 3D & 3E (2x2 assembly)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, PANEL_LBL, apply_style, save_fig
from pathlib import Path

BS = Path(__file__).resolve().parents[2] / 'additions' / '3_bootstrap_CIs'
AN = Path(__file__).resolve().parents[2] / 'additions' / '2_ANCOVA_universality'
OUT = OUTPUT_DIR / 'S14'

PANELS = [
    (BS / 'panel_3D_bootstrap.png', 'a'),
    (BS / 'panel_3E_bootstrap.png', 'b'),
    (AN / 'ancova_panel3D.png', 'c'),
    (AN / 'ancova_panel3E.png', 'd'),
]

def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 160 * MM))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, (path, label) in zip(axes.flat, PANELS):
        ax.imshow(mpimg.imread(str(path))); ax.axis('off')
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=PANEL_LBL, fontweight='bold', va='top')
    save_fig(fig, str(OUT / 'FigureS14_universality_stats'))
    plt.close(fig)
    print(f'Saved: {OUT}/FigureS14_universality_stats.png/.pdf/.svg')

if __name__ == '__main__': main()
