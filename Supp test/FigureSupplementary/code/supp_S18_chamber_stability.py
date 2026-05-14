#!/usr/bin/env python3
"""Supplementary Figure S18: Chamber temperature/humidity stability across 13 trials.

Two panels:
  A — Mean trajectory (line) +/- 1 SD shaded band, for temperature and humidity,
      across 13 standardized condensation trials.
  B — Stability table summarizing trial-to-trial variability:
      time-pooled SD (root-mean-square of the across-trial SD time series),
      mean MAE-style values, and per-channel min/max trajectory range.

Source CSV: Methods Figure /ChamberModel/NATURE_model_13trials.csv
The 13-trial CSV provides the across-trial mean and SD at each timepoint;
time-pooled SD is the relevant scalar for assessing trial-to-trial spread.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    apply_style, clean_axes, save_fig,
)

BASE_DIR  = Path(__file__).resolve().parent.parent.parent.parent
# Canonical (Yany-local) location of the chamber-model time series.
# The same file is mirrored to the Zenodo deposit as chamber_stability_13trials.csv;
# a user reproducing from the deposit can place that file at the repo root and the
# fallback below will pick it up.
_canonical = BASE_DIR / 'Methods Figure ' / 'ChamberModel' / 'NATURE_model_13trials.csv'
_fallback  = BASE_DIR / 'chamber_stability_13trials.csv'
CSV_PATH  = _canonical if _canonical.exists() else _fallback
OUT_DIR   = OUTPUT_DIR / 'S18'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    apply_style()

    df = pd.read_csv(CSV_PATH)
    n_trials = int(df['Temp_n'].iloc[0])

    t      = df['Time_min'].values
    Tm     = df['Temp_mean'].values
    Tsd    = df['Temp_std'].values
    Hm     = df['Hum_mean'].values
    Hsd    = df['Hum_std'].values

    # Time-pooled SD = sqrt(mean(SD_t^2)) — the scalar trial-to-trial spread
    T_pooled = float(np.sqrt(np.mean(Tsd ** 2)))
    H_pooled = float(np.sqrt(np.mean(Hsd ** 2)))
    T_meanSD = float(np.mean(Tsd))
    H_meanSD = float(np.mean(Hsd))

    # ---------------- Figure ----------------
    fig = plt.figure(figsize=(180 * MM, 95 * MM))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0],
                           left=0.08, right=0.97, top=0.92, bottom=0.18, wspace=0.35)

    # ---- Panel A: trajectories ----
    axA = fig.add_subplot(gs[0, 0])
    color_T = '#C0392B'
    color_H = '#2E86AB'

    axA.plot(t, Tm, color=color_T, lw=1.2, label='Temperature')
    axA.fill_between(t, Tm - Tsd, Tm + Tsd, color=color_T, alpha=0.20, lw=0)
    axA.set_xlabel('Time (min)', fontsize=LABEL_SIZE, labelpad=2)
    axA.set_ylabel('Temperature ($^\\circ$C)', fontsize=LABEL_SIZE, labelpad=2,
                   color=color_T)
    axA.tick_params(axis='y', labelcolor=color_T, labelsize=TICK_SIZE)
    axA.tick_params(axis='x', labelsize=TICK_SIZE)
    axA.set_xlim(t.min(), t.max())

    axB_twin = axA.twinx()
    axB_twin.plot(t, Hm, color=color_H, lw=1.2, label='Humidity')
    axB_twin.fill_between(t, Hm - Hsd, Hm + Hsd, color=color_H, alpha=0.20, lw=0)
    axB_twin.set_ylabel('Relative humidity (%)', fontsize=LABEL_SIZE, labelpad=2,
                        color=color_H)
    axB_twin.tick_params(axis='y', labelcolor=color_H, labelsize=TICK_SIZE)
    axB_twin.spines['top'].set_visible(False)
    axA.spines['top'].set_visible(False)

    axA.text(-0.13, 1.05, 'A', transform=axA.transAxes,
             fontsize=PANEL_LBL, fontweight='bold', va='top')

    # ---- Panel B: stats table ----
    axT = fig.add_subplot(gs[0, 1])
    axT.axis('off')

    rows = [
        ['Metric', 'Temperature', 'Humidity'],
        ['n trials', f'{n_trials}', f'{n_trials}'],
        ['Trajectory range',
         f'{Tm.min():.1f}$-${Tm.max():.1f} $^\\circ$C',
         f'{Hm.min():.1f}$-${Hm.max():.1f} %'],
        ['Mean across-trial SD',
         f'{T_meanSD:.2f} $^\\circ$C',
         f'{H_meanSD:.2f} %'],
        ['Time-pooled SD (RMS)',
         f'{T_pooled:.2f} $^\\circ$C',
         f'{H_pooled:.2f} %'],
    ]

    table = axT.table(cellText=rows, loc='center', cellLoc='center',
                      colWidths=[0.55, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(TICK_SIZE)
    table.scale(1, 1.6)

    # Header row styling
    for j in range(3):
        cell = table[0, j]
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#E8E8E8')
    # Highlight the headline RMS row
    for j in range(3):
        cell = table[4, j]
        cell.set_text_props(weight='bold')

    axT.text(-0.05, 1.05, 'B', transform=axT.transAxes,
             fontsize=PANEL_LBL, fontweight='bold', va='top')

    # ---- Save ----
    save_fig(fig, str(OUT_DIR / 'FigureS18_chamber_stability'))
    plt.close(fig)

    print(f'Saved S18 to {OUT_DIR}')
    print(f'  Temp: pooled SD = {T_pooled:.3f} C   mean SD = {T_meanSD:.3f} C')
    print(f'  Hum:  pooled SD = {H_pooled:.3f} %RH mean SD = {H_meanSD:.3f} %RH')


if __name__ == '__main__':
    main()
