#!/usr/bin/env python3
"""Supplementary Figure S10: Tanh fit parameters + diagnostics (2x3 grid).

Row 1: (a) y_near vs (1-a_w), (b) y_far vs (1-a_w), (c) alpha vs (1-a_w)
Row 2: (d) Residual plot, (e) Observed vs predicted parity, (f) Sensitivity heatmap
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import (
    DELTA, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
    apply_style, clean_axes, save_fig,
)
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
METRICS = BASE / 'FigureHGAggregate' / 'output' / 'hydrogel_metrics.csv'
SENSITIVITY = BASE / 'FigureTable' / 'output' / 'sensitivity_size_gradient.csv'
HG_AGG = BASE / 'FigureHGAggregate' / 'raw_data' / 'aggregate_edt'

T_WINDOW = (14.5, 15.5)
BIN_WIDTH = 100
MIN_DROPS = 5

# Colors by condition
COND_COLORS = {'agar': '#9E9E9E', '0.5:1': '#E67E22', '1:1': '#5B8FC9', '2:1': '#C0392B'}

OUT_DIR = OUTPUT_DIR / 'S8'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tanh_model(d, y_near, y_far, alpha, r0):
    return (y_near + y_far) / 2 + (y_far - y_near) / 2 * np.tanh(alpha * (d - r0))


def main():
    apply_style()
    metrics = pd.read_csv(METRICS)

    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 158 * MM))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.10,
                         hspace=0.55, wspace=0.40)

    param_panels = [
        ('y_near', 'y_near (µm)', 'a'),
        ('y_far', 'y_far (µm)', 'b'),
        ('alpha', 'α (µm⁻¹)', 'c'),
    ]

    for col_idx, (col, ylabel, label) in enumerate(param_panels):
        ax = axes[0, col_idx]
        for htype, color in COND_COLORS.items():
            sub = metrics[metrics['hydrogel_type'] == htype]
            ax.scatter(sub['one_minus_aw'], sub[col],
                       c=color, s=25, zorder=3, edgecolors='white',
                       linewidths=0.3, label=htype)

        x = metrics['one_minus_aw'].values
        y = metrics[col].values
        valid = ~np.isnan(y)
        if valid.sum() >= 5:
            slope, intercept, r, p, se = stats.linregress(x[valid], y[valid])
            x_fit = np.linspace(0, 0.28, 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'k--', lw=0.7, alpha=0.6)
            ax.text(0.95, 0.05, f'R²={r**2:.2f}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=TICK_SIZE - 1)

        ax.set_xlabel('1 − a_w', fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.text(0.03, 0.95, label, transform=ax.transAxes,
                fontsize=PANEL_LBL, fontweight='bold', va='top')
        clean_axes(ax)
        ax.tick_params(labelsize=TICK_SIZE)

        if col_idx == 0:
            ax.legend(fontsize=TICK_SIZE - 2, frameon=False, loc='upper left')

    ax_resid = axes[1, 0]
    all_resid_d, all_resid_v = [], []

    for _, row in metrics.iterrows():
        tid = row['trial_id']
        if pd.isna(row.get('alpha')):
            continue

        path = HG_AGG / f'{tid}_edt_droplets.csv'
        df = pd.read_csv(path)
        tw = df[(df['time_min'] >= T_WINDOW[0]) &
                (df['time_min'] <= T_WINDOW[1])].copy()

        bins = np.arange(0, tw['distance_um'].max() + BIN_WIDTH, BIN_WIDTH)
        tw['dbin'] = pd.cut(tw['distance_um'], bins=bins,
                            labels=bins[:-1] + BIN_WIDTH/2).astype(float)
        prof = tw.groupby('dbin')['radius_um'].agg(['mean', 'count']).reset_index()
        prof = prof[prof['count'] >= MIN_DROPS]

        r0 = row['r0_um'] if not pd.isna(row.get('r0_um')) else np.median(prof['dbin'])
        predicted = tanh_model(prof['dbin'].values, row['y_near'], row['y_far'],
                                row['alpha'], r0)
        residual = prof['mean'].values - predicted

        color = COND_COLORS[row['hydrogel_type']]
        ax_resid.scatter(prof['dbin'] / 1000, residual,
                         s=3, alpha=0.4, color=color, edgecolors='none',
                         rasterized=True)
        all_resid_d.extend(prof['dbin'].values)
        all_resid_v.extend(residual)

    ax_resid.axhline(0, color='k', ls='-', lw=0.5, alpha=0.5)
    ax_resid.set_xlabel('Distance (mm)', fontsize=LABEL_SIZE)
    ax_resid.set_ylabel('Residual (µm)', fontsize=LABEL_SIZE)
    ax_resid.text(0.03, 0.95, 'd', transform=ax_resid.transAxes,
                  fontsize=PANEL_LBL, fontweight='bold', va='top')
    clean_axes(ax_resid)
    ax_resid.tick_params(labelsize=TICK_SIZE)

    ax_parity = axes[1, 1]
    all_obs, all_pred = [], []

    for _, row in metrics.iterrows():
        tid = row['trial_id']
        if pd.isna(row.get('alpha')):
            continue

        path = HG_AGG / f'{tid}_edt_droplets.csv'
        df = pd.read_csv(path)
        tw = df[(df['time_min'] >= T_WINDOW[0]) &
                (df['time_min'] <= T_WINDOW[1])].copy()

        bins = np.arange(0, tw['distance_um'].max() + BIN_WIDTH, BIN_WIDTH)
        tw['dbin'] = pd.cut(tw['distance_um'], bins=bins,
                            labels=bins[:-1] + BIN_WIDTH/2).astype(float)
        prof = tw.groupby('dbin')['radius_um'].agg(['mean', 'count']).reset_index()
        prof = prof[prof['count'] >= MIN_DROPS]

        r0 = row['r0_um'] if not pd.isna(row.get('r0_um')) else np.median(prof['dbin'])
        predicted = tanh_model(prof['dbin'].values, row['y_near'], row['y_far'],
                                row['alpha'], r0)

        color = COND_COLORS[row['hydrogel_type']]
        ax_parity.scatter(predicted, prof['mean'].values,
                          s=3, alpha=0.4, color=color, edgecolors='none',
                          rasterized=True)
        all_obs.extend(prof['mean'].values)
        all_pred.extend(predicted)

    lims = [0, 70]
    ax_parity.plot(lims, lims, 'k--', lw=0.7, alpha=0.5)
    ax_parity.set_xlim(lims)
    ax_parity.set_ylim(lims)
    ax_parity.set_xlabel('Predicted R (µm)', fontsize=LABEL_SIZE)
    ax_parity.set_ylabel('Observed R (µm)', fontsize=LABEL_SIZE)
    ax_parity.set_aspect('equal')
    ax_parity.text(0.03, 0.95, 'e', transform=ax_parity.transAxes,
                   fontsize=PANEL_LBL, fontweight='bold', va='top')

    all_obs, all_pred = np.array(all_obs), np.array(all_pred)
    ss_res = np.sum((all_obs - all_pred)**2)
    ss_tot = np.sum((all_obs - all_obs.mean())**2)
    r2 = 1 - ss_res / ss_tot
    ax_parity.text(0.95, 0.05, f'R²={r2:.3f}', transform=ax_parity.transAxes,
                   ha='right', va='bottom', fontsize=TICK_SIZE - 1)
    clean_axes(ax_parity)
    ax_parity.tick_params(labelsize=TICK_SIZE)

    ax_sens = axes[1, 2]
    sens = pd.read_csv(SENSITIVITY)

    sens['label'] = (sens.iloc[:, 0].astype(str) + ' | ' +
                     sens.iloc[:, 1].astype(str) + ' | ' +
                     sens.iloc[:, 2].astype(str))
    r2_vals = sens.iloc[:, 4].values  # R² column

    order = np.argsort(r2_vals)[::-1]
    colors = plt.cm.viridis((r2_vals[order] - 0.85) / 0.10)
    ax_sens.barh(range(len(r2_vals)), r2_vals[order],
                 color=colors, height=0.8, edgecolor='none')
    ax_sens.set_xlim(0.85, 0.95)
    ax_sens.set_yticks([])
    ax_sens.set_xlabel('R²', fontsize=LABEL_SIZE)
    ax_sens.set_ylabel('Zone definitions (38 variants)', fontsize=LABEL_SIZE)
    ax_sens.text(0.03, 0.95, 'f', transform=ax_sens.transAxes,
                 fontsize=PANEL_LBL, fontweight='bold', va='top')
    ax_sens.axvline(0.869, color='red', ls=':', lw=0.6, alpha=0.7)
    ax_sens.text(0.875, len(r2_vals) * 0.95, 'min', fontsize=TICK_SIZE - 2,
                 color='red', va='top')
    clean_axes(ax_sens)
    ax_sens.tick_params(labelsize=TICK_SIZE - 1)

    stem = str(OUT_DIR / 'FigureS8_tanh_diagnostics')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'Saved: {stem}.pdf/.svg')


if __name__ == '__main__':
    main()
