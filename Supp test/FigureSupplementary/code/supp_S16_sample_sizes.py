#!/usr/bin/env python3
"""Supplementary Figure S16: Per-trial droplet counts and tracked-track
sample sizes.

Three panels:
  A — Total droplet detections per trial during the late-condensation
      window (t = 14.5-15.5 min).
  B — Total tracked droplets per trial entering the d* / KM analysis
      (n_frames >= 3, surviving past evaporation onset).
  C — Droplet count vs distance from boundary, pooled within condition;
      shows that all conditions have comparable spatial sampling density.

Demonstrates that aggregate metrics (delta, R-bar profiles, KM survival)
are not biased by per-trial sample-size variability.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    CONDITIONS, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    apply_style, save_fig,
)

HG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureHGAggregate/raw_data/aggregate_edt')
FG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureFungi/raw_data/aggregate_edt')
TRACK_DIR = Path('/Volumes/T7/FINAL OSF/FigureHGAggregate/code/test_tracking/output')

OUT_DIR = OUTPUT_DIR / 'S16'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def trial_path(tid):
    if tid.startswith(('Green', 'white', 'black')):
        return FG_AGG_DIR / f'{tid}_edt_droplets.csv'
    return HG_AGG_DIR / f'{tid}_edt_droplets.csv'


def main():
    apply_style()
    fig = plt.figure(figsize=(190 * MM, 200 * MM))
    gs = fig.add_gridspec(3, 1, left=0.10, right=0.97, top=0.97, bottom=0.06,
                          hspace=0.45, height_ratios=[1, 1, 1.1])
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2])

    # ── Panel A: detection count at t ≈ 15 min ──
    bar_x = []
    bar_y = []
    bar_color = []
    bar_label = []
    pos = 0
    cond_centers = []
    for cond_key, ids, label, color in CONDITIONS:
        start = pos
        for tid in ids:
            df = pd.read_csv(trial_path(tid))
            sub = df[(df['time_min'] >= 14.5) & (df['time_min'] <= 15.5)
                     & (df['radius_um'] > 0)]
            bar_x.append(pos); bar_y.append(len(sub)); bar_color.append(color)
            bar_label.append(tid)
            pos += 1
        cond_centers.append((start + pos - 1) / 2)
        pos += 1
    axA.bar(bar_x, bar_y, color=bar_color, edgecolor='white', linewidth=0.5)
    axA.set_ylabel('Detections at t ≈ 15 min',
                   fontsize=LABEL_SIZE, labelpad=2)
    axA.set_xticks(cond_centers)
    axA.set_xticklabels([c[2] for c in CONDITIONS],
                        fontsize=TICK_SIZE - 0.5, rotation=20, ha='right')
    axA.tick_params(labelsize=TICK_SIZE - 0.5)
    axA.text(-0.07, 1.05, 'A', transform=axA.transAxes, fontsize=11,
             fontweight='bold', va='top')
    for sp in ('top', 'right'): axA.spines[sp].set_visible(False)
    print(f'Panel A: {len(bar_y)} trials, count range {min(bar_y)}-{max(bar_y)}')

    # ── Panel B: tracked droplets per trial ──
    bar2_x, bar2_y, bar2_color = [], [], []
    pos = 0
    cond_centers2 = []
    for cond_key, ids, label, color in CONDITIONS:
        start = pos
        for tid in ids:
            tp = TRACK_DIR / f'{tid}_track_histories.csv'
            if not tp.exists():
                bar2_x.append(pos); bar2_y.append(0)
                bar2_color.append('#dddddd')
            else:
                df = pd.read_csv(tp)
                df = df[df['n_frames'] >= 3]
                bar2_x.append(pos); bar2_y.append(len(df))
                bar2_color.append(color)
            pos += 1
        cond_centers2.append((start + pos - 1) / 2)
        pos += 1
    axB.bar(bar2_x, bar2_y, color=bar2_color, edgecolor='white', linewidth=0.5)
    axB.set_ylabel('Tracked droplets ($n_{frames} \\geq 3$)',
                   fontsize=LABEL_SIZE, labelpad=2)
    axB.set_xticks(cond_centers2)
    axB.set_xticklabels([c[2] for c in CONDITIONS],
                        fontsize=TICK_SIZE - 0.5, rotation=20, ha='right')
    axB.tick_params(labelsize=TICK_SIZE - 0.5)
    axB.text(-0.07, 1.05, 'B', transform=axB.transAxes, fontsize=11,
             fontweight='bold', va='top')
    for sp in ('top', 'right'): axB.spines[sp].set_visible(False)
    print(f'Panel B: total tracked = {sum(bar2_y):,}')

    # ── Panel C: droplet count vs distance, pooled within condition ──
    BIN_W = 100
    DIST_MAX = 3000
    edges = np.arange(0, DIST_MAX + BIN_W, BIN_W)
    centers = (edges[:-1] + edges[1:]) / 2

    for cond_key, ids, label, color in CONDITIONS:
        all_dist = []
        for tid in ids:
            df = pd.read_csv(trial_path(tid))
            sub = df[(df['time_min'] >= 14.5) & (df['time_min'] <= 15.5)
                     & (df['radius_um'] > 0)]
            all_dist.append(sub['distance_um'].values)
        if not all_dist: continue
        d = np.concatenate(all_dist)
        h, _ = np.histogram(d, bins=edges)
        h = h / len(ids)   # average per trial per bin
        axC.plot(centers / 1000, h, '-', color=color, lw=1.2, label=label, alpha=0.85)

    axC.set_xlabel('Distance from source boundary (mm)',
                   fontsize=LABEL_SIZE, labelpad=2)
    axC.set_ylabel('Mean detections per 100 µm bin per trial',
                   fontsize=LABEL_SIZE, labelpad=2)
    axC.set_xlim(0, 3.0)
    axC.tick_params(labelsize=TICK_SIZE - 0.5)
    axC.legend(fontsize=TICK_SIZE - 1, loc='upper right',
               frameon=False, ncol=2, labelspacing=0.3)
    axC.text(-0.07, 1.05, 'C', transform=axC.transAxes, fontsize=11,
             fontweight='bold', va='top')
    for sp in ('top', 'right'): axC.spines[sp].set_visible(False)

    save_fig(fig, str(OUT_DIR / 'FigureS16_sample_sizes'))
    plt.close(fig)
    print('Saved S16')


if __name__ == '__main__':
    main()
