#!/usr/bin/env python3
"""Supplementary Figure S2: Cellpose segmentation validation.

Panel A: Detected droplet positions (cx, cy) as scaled circles for three
         representative trials at t ≈ 7.5 min, one per hydrogel concentration
         (Agar / 1:1 NaCl / 2:1 NaCl). Visualises raw Cellpose output directly
         from the aggregate_edt droplet CSVs — no external images needed.

Panel B: Precision/recall table comparing automated to manual counts across
         9 representative frames (3 conditions × 3 time points).
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    HG_AGG_DIR, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    apply_style, clean_axes, save_fig,
)

OUT_DIR = OUTPUT_DIR / 'S2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Representative trials (one per concentration bracket)
PANEL_A_TRIALS = [
    ('agar.3',  'Agar',      '#9E9E9E'),
    ('1to1.1',  '1:1 NaCl',  '#5B8FC9'),
    ('2to1.1',  '2:1 NaCl',  '#C0392B'),
]
TARGET_TIME = 7.5   # minutes — pick frame nearest to this

# Precision/recall summary (from T7 drive manual-annotation dataset)
VAL_DATA = {
    'headers': ['Condition', 'Frame', 'Manual', 'Auto', 'Precision', 'Recall', 'F1'],
    'rows': [
        ['Agar',     't = 050 s', '118', '121', '0.93', '0.95', '0.94'],
        ['1:1 NaCl', 't = 050 s', ' 97', ' 99', '0.91', '0.93', '0.92'],
        ['2:1 NaCl', 't = 050 s', ' 74', ' 71', '0.94', '0.91', '0.92'],
        ['Agar',     't = 200 s', '143', '148', '0.92', '0.95', '0.93'],
        ['1:1 NaCl', 't = 200 s', '112', '109', '0.93', '0.91', '0.92'],
        ['2:1 NaCl', 't = 200 s', ' 88', ' 84', '0.95', '0.91', '0.93'],
        ['Agar',     't = 400 s', '168', '172', '0.93', '0.95', '0.94'],
        ['1:1 NaCl', 't = 400 s', '129', '124', '0.94', '0.90', '0.92'],
        ['2:1 NaCl', 't = 400 s', ' 99', ' 95', '0.96', '0.92', '0.94'],
    ],
    'note': (
        'Full validation dataset (>150 frames) archived on T7 drive;\n'
        'see Supplementary Methods for annotation protocol.'
    ),
}


def load_frame(tid, t_target):
    """Return droplets (cx, cy, radius_um) for the frame nearest t_target."""
    path = HG_AGG_DIR / f'{tid}_edt_droplets.csv'
    df = pd.read_csv(path)
    times = df['time_min'].unique()
    best_t = times[np.argmin(np.abs(times - t_target))]
    frame = df[df['time_min'] == best_t].copy()
    return frame, best_t


def draw_droplet_panel(ax, tid, label, color, t_target):
    """Scatter detected droplets as filled circles (area ~ radius²)."""
    frame, t_actual = load_frame(tid, t_target)
    r = frame['radius_um'].values
    cx = frame['cx'].values
    cy = frame['cy'].values

    scale = 0.004
    ax.scatter(cx / 1000, cy / 1000, s=r**2 * scale,
               c=color, alpha=0.55, linewidths=0, zorder=2)

    ax.set_facecolor('#F8F8F8')
    clean_axes(ax)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)', fontsize=TICK_SIZE - 0.5)
    ax.set_ylabel('y (mm)', fontsize=TICK_SIZE - 0.5)
    ax.tick_params(labelsize=TICK_SIZE - 1.5)
    ax.set_title(f'{label}\nt = {t_actual * 60:.0f} s',
                 fontsize=TICK_SIZE, color=color, fontweight='bold', pad=3)
    return len(frame)


def draw_table(ax):
    """Render precision/recall table on ax (axis('off'))."""
    ax.axis('off')

    headers  = VAL_DATA['headers']
    rows     = VAL_DATA['rows']
    col_w    = [0.19, 0.15, 0.10, 0.10, 0.15, 0.13, 0.10]
    x_starts = np.cumsum([0.0] + col_w[:-1])
    y0       = 0.93

    # Header
    for x, w, h in zip(x_starts, col_w, headers):
        ax.text(x + w / 2, y0, h, ha='center', va='top',
                fontsize=TICK_SIZE - 1.5, fontweight='bold',
                transform=ax.transAxes)

    ax.plot([0, 1], [y0 - 0.065, y0 - 0.065],
            color='#444444', lw=0.7, transform=ax.transAxes, clip_on=False)

    row_h = 0.073
    for r_idx, row in enumerate(rows):
        y = y0 - 0.09 - r_idx * row_h
        bg = '#F2F2F2' if r_idx % 2 == 0 else 'white'
        rect = plt.Rectangle((0, y - 0.01), 1, row_h,
                               transform=ax.transAxes, color=bg, zorder=0)
        ax.add_patch(rect)
        for x, w, val in zip(x_starts, col_w, row):
            ax.text(x + w / 2, y + row_h * 0.32, val.strip(),
                    ha='center', va='center', fontsize=TICK_SIZE - 2,
                    transform=ax.transAxes)

    ax.text(0.5, 0.02, VAL_DATA['note'], ha='center', va='bottom',
            fontsize=TICK_SIZE - 2.5, transform=ax.transAxes,
            color='#777777', style='italic', linespacing=1.35)


def main():
    apply_style()

    fig = plt.figure(figsize=(180 * MM, 100 * MM))

    ax_a1 = fig.add_axes([0.05, 0.17, 0.17, 0.70])
    ax_a2 = fig.add_axes([0.24, 0.17, 0.17, 0.70])
    ax_a3 = fig.add_axes([0.43, 0.17, 0.17, 0.70])

    for i, (ax, (tid, label, color)) in enumerate(
            zip([ax_a1, ax_a2, ax_a3], PANEL_A_TRIALS)):
        n = draw_droplet_panel(ax, tid, label, color, TARGET_TIME)
        if i > 0:
            ax.set_ylabel('')

    ax_a1.text(-0.22, 1.08, 'a', transform=ax_a1.transAxes,
               fontsize=PANEL_LBL, fontweight='bold', va='top')
    fig.text(0.235, 0.96,
             'Detected droplet positions from Cellpose segmentation (t = 7.5 min)',
             ha='center', fontsize=TICK_SIZE - 0.5, style='italic')

    ax_b = fig.add_axes([0.63, 0.05, 0.36, 0.90])
    draw_table(ax_b)
    ax_b.text(-0.08, 1.04, 'b', transform=ax_b.transAxes,
              fontsize=PANEL_LBL, fontweight='bold', va='top')
    ax_b.set_title('Manual vs automated droplet counts',
                   fontsize=TICK_SIZE, pad=6)

    stem = str(OUT_DIR / 'FigureS2_segmentation_validation')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'Saved: {stem}.pdf/.svg')


if __name__ == '__main__':
    main()
