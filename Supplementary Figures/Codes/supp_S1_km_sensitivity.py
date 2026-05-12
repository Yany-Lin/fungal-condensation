#!/usr/bin/env python3
"""Supplementary Figure S1: KM sensitivity analysis.

3×3 grid showing τ₅₀(d) profiles under 9 parameter combinations:
  rows  → bin width  [100, 200, 400 µm]
  cols  → min tracks [5,  15,  30]
Canonical parameters (bin_width=200 µm, min_tracks=10) highlighted.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lifelines import KaplanMeierFitter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    DELTA, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    apply_style, clean_axes, save_fig,
)

TRACK_DIR = Path(__file__).resolve().parents[2] / \
            'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'

HG_CONDITIONS = [
    ('agar',   ['agar.1','agar.2','agar.3','agar.4','agar.5'],
     'Agar',        '#9E9E9E'),
    ('0.5to1', ['0.5to1.2','0.5to1.3','0.5to1.4','0.5to1.5','0.5to1.7'],
     '0.5:1 NaCl',  '#E67E22'),
    ('1to1',   ['1to1.1','1to1.2','1to1.3','1to1.4','1to1.5'],
     '1:1 NaCl',    '#5B8FC9'),
    ('2to1',   ['2to1.1','2to1.2','2to1.3','2to1.4','2to1.5'],
     '2:1 NaCl',    '#C0392B'),
]

T_SEED     = 900    # seconds
MIN_FRAMES = 3
D_MAX      = 4000   # µm — max distance to consider

# Canonical parameters (match main-paper analysis)
CANONICAL_BW  = 200   # µm
CANONICAL_MT  = 10    # min tracks per bin

BIN_WIDTHS  = [100, 200, 400]   # rows
MIN_TRACKS  = [5,  10,  30]    # cols

OUT_DIR = OUTPUT_DIR / 'S1'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_condition_tracks(cond_key, trial_ids):
    """Load & pool track histories for one condition."""
    frames = []
    for tid in trial_ids:
        path = TRACK_DIR / f'{tid}_track_histories.csv'
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df['n_frames'] >= MIN_FRAMES].copy()
        df['tau_fwd'] = (df['t_death_s'] - T_SEED) / 60.0
        df = df[df['tau_fwd'] > 0]
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def compute_tau50_profile(df, bin_width, min_tracks):
    """Return arrays (bin_centers, tau50) for a pooled DataFrame."""
    bin_edges = np.arange(0, D_MAX + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    tau50s = []
    for lo, center in zip(bin_edges[:-1], bin_centers):
        hi = lo + bin_width
        sub = df[(df['distance_um'] >= lo) & (df['distance_um'] < hi)]
        if len(sub) < min_tracks:
            tau50s.append(np.nan)
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['tau_fwd'], event_observed=~sub['censored'])
        tau50s.append(kmf.median_survival_time_)
    return bin_centers / 1000.0, np.array(tau50s, dtype=float)  # dist in mm


def main():
    apply_style()

    cond_data = {}
    for cond_key, trial_ids, label, color in HG_CONDITIONS:
        print(f'Loading {cond_key}...', flush=True)
        cond_data[cond_key] = (load_condition_tracks(cond_key, trial_ids), label, color)

    fig, axes = plt.subplots(3, 3, figsize=(180 * MM, 178 * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.91, bottom=0.10,
                        hspace=0.42, wspace=0.18)

    for row, bw in enumerate(BIN_WIDTHS):
        for col, mt in enumerate(MIN_TRACKS):
            ax = axes[row, col]
            is_canonical = (bw == CANONICAL_BW) and (mt == CANONICAL_MT)

            for cond_key, (df, label, color) in cond_data.items():
                if df is None:
                    continue
                d_mm, tau50 = compute_tau50_profile(df, bw, mt)
                mask = np.isfinite(tau50)
                ax.plot(d_mm[mask], tau50[mask], 'o-',
                        color=color, lw=0.8, ms=2.5, label=label)

            clean_axes(ax)
            ax.tick_params(labelsize=TICK_SIZE, pad=2)
            ax.set_xlim(0, D_MAX / 1000)
            ax.set_ylim(0, None)

            panel_letter = chr(ord('A') + row * 3 + col)
            ax.text(-0.12, 1.10, panel_letter, transform=ax.transAxes,
                    fontsize=PANEL_LBL, va='top')

            if is_canonical:
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(1.8)
                    sp.set_color('#2196F3')
                ax.set_title(f'[canonical] {bw} um / {mt}', fontsize=TICK_SIZE,
                             pad=3, color='#2196F3', fontweight='bold')
            else:
                ax.set_title(f'{bw} um / {mt}', fontsize=TICK_SIZE, pad=3)

    for col, mt in enumerate(MIN_TRACKS):
        axes[0, col].annotate(f'min tracks = {mt}', xy=(0.5, 1.0),
                              xycoords='axes fraction', ha='center',
                              fontsize=TICK_SIZE, va='bottom',
                              xytext=(0, 18), textcoords='offset points')

    for row, bw in enumerate(BIN_WIDTHS):
        axes[row, 0].set_ylabel('tau50 (min)', fontsize=LABEL_SIZE - 0.5)
        axes[row, 0].text(-0.38, 0.5, f'{bw} um bins',
                          transform=axes[row, 0].transAxes,
                          fontsize=TICK_SIZE, ha='right', va='center',
                          rotation=90)

    fig.text(0.55, 0.02, 'Distance from hyphae (mm)', ha='center',
             fontsize=LABEL_SIZE)

    handles = [mpatches.Patch(color=color, label=label)
               for _, (_, label, color) in cond_data.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=TICK_SIZE - 1.5, frameon=False,
               bbox_to_anchor=(0.55, -0.01))

    stem = str(OUT_DIR / 'FigureS1_km_sensitivity')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'\nSaved: {stem}.png/.pdf/.svg')


if __name__ == '__main__':
    main()
