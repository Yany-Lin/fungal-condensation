#!/usr/bin/env python3
"""Supplementary Figure S15: Bayesian universality composite (assembly)."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, apply_style, save_fig
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'additions' / '2_ANCOVA_universality' / 'universality_composite.png'
OUT = OUTPUT_DIR / 'S15'

def main():
    apply_style()
    img = mpimg.imread(str(SRC))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(180 * MM, 180 * MM * h / w))
    ax.imshow(img); ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save_fig(fig, str(OUT / 'FigureS15_bayesian_universality'))
    plt.close(fig)
    print(f'Saved: {OUT}/FigureS15_bayesian_universality.png/.pdf/.svg')

if __name__ == '__main__': main()
