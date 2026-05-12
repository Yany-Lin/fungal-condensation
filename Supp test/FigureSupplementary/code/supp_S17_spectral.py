#!/usr/bin/env python3
"""Supplementary Figure S17: FFT spectral slope analysis (assembly)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, apply_style, save_fig
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'Hyphal Analysis' / 'FigureSpectralSlope.png'
OUT = OUTPUT_DIR / 'S17'

def main():
    apply_style()
    img = mpimg.imread(str(SRC))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(180 * MM, 180 * MM * h / w))
    ax.imshow(img); ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_fig(fig, str(OUT / 'FigureS17_spectral_slope'))
    plt.close(fig)
    print(f'Saved: {OUT}/FigureS17_spectral_slope.png/.pdf/.svg')

if __name__ == '__main__': main()
