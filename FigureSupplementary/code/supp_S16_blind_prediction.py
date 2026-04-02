#!/usr/bin/env python3
"""Supplementary Figure S16: Blind prediction + a_w inversion (2x2 assembly)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, PANEL_LBL, apply_style, save_fig
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'additions' / '1_blind_simulation_prediction'
OUT = OUTPUT_DIR / 'S16'

PANELS = [
    (SRC / 'fig_blind_aw_sweep.png', 'a'),
    (SRC / 'fig_blind_parity.png', 'b'),
    (SRC / 'inversion_aw_by_genus.png', 'c'),
    (SRC / 'aw_sweep.png', 'd'),
]

def main():
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 160 * MM))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for ax, (path, label) in zip(axes.flat, PANELS):
        ax.imshow(mpimg.imread(str(path))); ax.axis('off')
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=PANEL_LBL, fontweight='bold', va='top')
    save_fig(fig, str(OUT / 'FigureS16_blind_prediction'))
    plt.close(fig)
    print(f'Saved: {OUT}/FigureS16_blind_prediction.png/.pdf/.svg')

if __name__ == '__main__': main()
