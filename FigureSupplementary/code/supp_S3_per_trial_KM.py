#!/usr/bin/env python3
"""Supplementary Figure S2 (file: S3): Per-condition KM survival curves.

2x2 grid — one panel per hydrogel condition.
Each panel:
  - 5 thin step curves, one per replicate (shows trial-to-trial variability)
  - 1 bold pooled KM curve (all 5 replicates combined)
  - 95% CI band on the pooled curve only
  - Vertical dashed line at pooled tau50
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, PANEL_LBL,
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

T_SEED     = 900   # seconds
MIN_FRAMES = 3
CI_ALPHA   = 0.95
X_MAX      = 12    # minutes

OUT_DIR = OUTPUT_DIR / 'S3'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_trial(tid):
    path = TRACK_DIR / f'{tid}_track_histories.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[df['tau_fwd'] > 0]
    return df if len(df) >= 5 else None


def plot_condition(ax, trial_ids, label, color):
    frames = []
    for tid in trial_ids:
        df = load_trial(tid)
        if df is None:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(df['tau_fwd'], event_observed=~df['censored'],
                alpha=1 - CI_ALPHA)
        t = kmf.survival_function_.index.values
        s = kmf.survival_function_.values.flatten()
        ax.step(t, s, where='post', color=color, lw=0.7, alpha=0.30)

        frames.append(df)

    if not frames:
        return

    pooled = pd.concat(frames, ignore_index=True)
    kmf_pool = KaplanMeierFitter()
    kmf_pool.fit(pooled['tau_fwd'], event_observed=~pooled['censored'],
                 alpha=1 - CI_ALPHA)

    t_p = kmf_pool.survival_function_.index.values
    s_p = kmf_pool.survival_function_.values.flatten()
    ci  = kmf_pool.confidence_interval_survival_function_
    s_lo = ci.iloc[:, 0].values
    s_hi = ci.iloc[:, 1].values

    ax.fill_between(t_p, s_lo, s_hi, step='post', alpha=0.18, color=color)
    ax.step(t_p, s_p, where='post', color=color, lw=2.0, label='Pooled')

    median = kmf_pool.median_survival_time_
    if np.isfinite(median):
        ax.axvline(median, color=color, lw=1.0, ls='--', alpha=0.8)
        ax.plot(median, 0.5, 'o', color=color, ms=5, zorder=6,
                label=f'$\\tau_{{50}}$ = {median:.1f} min')

    ax.axhline(0.5, color='#AAAAAA', ls=':', lw=0.5)

    ax.set_title(label, fontsize=LABEL_SIZE, color=color,
                 fontweight='bold', pad=4)
    ax.legend(fontsize=TICK_SIZE - 1.5, frameon=False,
              loc='upper right', handlelength=1.2, labelspacing=0.25)


def main():
    apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(160 * MM, 140 * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.10,
                        hspace=0.32, wspace=0.15)

    panel_labels = ['a', 'b', 'c', 'd']
    for idx, (ax, (cond_key, trial_ids, cond_label, color)) in enumerate(
            zip(axes.flat, HG_CONDITIONS)):
        plot_condition(ax, trial_ids, cond_label, color)
        clean_axes(ax)
        ax.set_xlim(0, X_MAX)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=TICK_SIZE - 0.5, pad=2)

        ax.text(-0.08, 1.06, panel_labels[idx], transform=ax.transAxes,
                fontsize=PANEL_LBL, fontweight='bold', va='top')

    for ax in axes[1, :]:
        ax.set_xlabel('Forward lifetime (min)', fontsize=LABEL_SIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel('Survival  S(\u03c4)', fontsize=LABEL_SIZE)

    fig.text(0.53, 0.97,
             'Thin lines = individual replicates  \u2502  Bold = pooled KM  \u2502  Band = 95% CI',
             ha='center', fontsize=TICK_SIZE - 0.5, style='italic', color='#555555')

    stem = str(OUT_DIR / 'FigureS3_per_trial_KM')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'Saved: {stem}.pdf/.svg')


if __name__ == '__main__':
    main()
