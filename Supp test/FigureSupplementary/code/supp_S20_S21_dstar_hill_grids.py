#!/usr/bin/env python3
"""Supplementary Figures S20 + S21: Per-trial d* Hill fits.

S20 (4x5 grid): tau50(r) profile + Hill fit + d* marker, all 20 hydrogel trials.
S21 (3x5 grid): same for the 15 fungal trials. Trials assigned d* = 0
  (flat profile) are annotated as such.

Reuses _get_tau50_profile + _fit_hill from FigureRSR/code/step2_rsr_metrics_and_universal_plots.py.
Edit MIN_BINS / FLAT_RANGE constants at top to match the current definition.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, apply_style, clean_axes, save_fig

# ---- Parameters identical to step2_rsr_metrics_and_universal_plots.py ----
HG_TRACK = Path('/Volumes/T7/FINAL OSF/FigureHGAggregate/code/test_tracking/output')
T_SEED_S = 900
MIN_FRAMES = 3
DIST_BIN_UM = 200
MIN_PER_BIN = 15
MIN_BINS = 3
LAB_CAP_MM = 4.0
FLAT_RANGE = 1.0


def _hill(d, T0, A, K, n):
    dn = np.power(np.maximum(d, 0), n)
    Kn = np.power(K, n)
    return T0 + A * dn / (Kn + dn)


def get_profile(tid):
    p = HG_TRACK / f'{tid}_track_histories.csv'
    if not p.exists(): return None, None
    df = pd.read_csv(p)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd'] = (df['t_death_s'] - T_SEED_S) / 60.0
    df = df[(df['tau_fwd'] > 0) & (df['distance_um'] <= LAB_CAP_MM * 1000)].copy()
    if len(df) < 30: return None, None
    bins = np.arange(0, df['distance_um'].max() + DIST_BIN_UM, DIST_BIN_UM)
    df['db'] = pd.cut(df['distance_um'], bins=bins, labels=False)
    kmf = KaplanMeierFitter()
    d_vals, t_vals = [], []
    for b in sorted(df['db'].dropna().unique()):
        sub = df[df['db'] == b]
        if len(sub) < MIN_PER_BIN: continue
        center = (bins[int(b)] + DIST_BIN_UM / 2) / 1000.0
        kmf.fit(sub['tau_fwd'], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            d_vals.append(center); t_vals.append(t50)
    return (np.array(d_vals), np.array(t_vals)) if len(d_vals) >= MIN_BINS else (None, None)


def fit_hill(d, t):
    if len(d) < MIN_BINS: return None
    if float(t.max() - t.min()) < FLAT_RANGE: return {'flat': True}
    T0_g = float(t[0]); A_g = max(float(t.max() - t.min()), 0.1)
    try:
        popt, _ = curve_fit(_hill, d, t, p0=[T0_g, A_g, np.median(d), 2.0],
                            bounds=([0, 0, 0.01, 0.3],
                                    [t.max(), t.max() * 3, d.max() * 10, 10.0]),
                            max_nfev=50000, method='trf')
    except Exception:
        return None
    return {'T0': popt[0], 'A': popt[1], 'K': popt[2], 'n': popt[3], 'flat': False}


def panel(ax, tid, color):
    d, t = get_profile(tid)
    if d is None:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', va='center',
                fontsize=TICK_SIZE - 1, color='gray'); ax.axis('off'); return
    ax.plot(d, t, 'o', ms=4, color=color, alpha=0.85, zorder=3)
    fit = fit_hill(d, t)
    if fit and not fit.get('flat'):
        d_fine = np.linspace(d.min(), d.max(), 200)
        t_fine = _hill(d_fine, fit['T0'], fit['A'], fit['K'], fit['n'])
        ax.plot(d_fine, t_fine, '-', color=color, lw=1.4, alpha=0.85, zorder=2)
        ax.axvline(fit['K'], ls='--', lw=0.8, color=color, alpha=0.7)
        ax.text(0.97, 0.05, f'$d^*={fit["K"]:.2f}$ mm', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=TICK_SIZE - 1.5)
    else:
        ax.text(0.50, 0.06, '$d^* = 0$ (flat)', transform=ax.transAxes,
                ha='center', va='bottom', fontsize=TICK_SIZE - 1, color='gray')
    ax.set_xlim(0, LAB_CAP_MM); ax.set_ylim(bottom=0)
    clean_axes(ax)
    ax.tick_params(labelsize=TICK_SIZE - 1, pad=2)
    ax.text(0.97, 0.97, tid, transform=ax.transAxes, ha='right', va='top',
            fontsize=TICK_SIZE - 1.5, color=color, alpha=0.8)


HYDROGEL = [
    ('Agar',      ['agar.1','agar.2','agar.3','agar.4','agar.5'],     '#9E9E9E'),
    ('0.5:1 NaCl',['0.5to1.2','0.5to1.3','0.5to1.4','0.5to1.5','0.5to1.7'],'#E67E22'),
    ('1:1 NaCl',  ['1to1.1','1to1.2','1to1.3','1to1.4','1to1.5'],     '#5B8FC9'),
    ('2:1 NaCl',  ['2to1.1','2to1.2','2to1.3','2to1.4','2to1.5'],     '#C0392B'),
]
FUNGI = [
    ('Aspergillus', ['Green.1','Green.2','Green.3','Green.4','Green.5'], '#4CAF50'),
    ('Rhizopus',    ['black.1','black.2','black.3','black.4','black.5'], '#212121'),
    ('Mucor',       ['white.1','white.2','white.3','white.4','white.5'], '#757575'),
]


def render_grid(rows, n_rows, name, height_mm):
    apply_style()
    fig, axes = plt.subplots(n_rows, 5, figsize=(200 * MM, height_mm * MM),
                              sharex=True, sharey=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.10,
                        hspace=0.40, wspace=0.30)
    for r, (label, ids, color) in enumerate(rows):
        for c, tid in enumerate(ids):
            panel(axes[r, c], tid, color)
        axes[r, 0].text(-0.32, 0.5, label, transform=axes[r, 0].transAxes,
                        fontsize=TICK_SIZE + 1, fontweight='bold',
                        color=color, ha='right', va='center', rotation=90)
    fig.text(0.53, 0.025, 'Distance from boundary (mm)', ha='center', fontsize=LABEL_SIZE)
    fig.text(0.02, 0.5, r'$\tau_{50}$ (min)', va='center', rotation=90, fontsize=LABEL_SIZE)
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    save_fig(fig, str(out / f'Figure{name}'))
    plt.close(fig)


def main():
    render_grid(HYDROGEL, 4, 'S20', 200)
    print('Saved S20 (hydrogels)')
    render_grid(FUNGI, 3, 'S21', 150)
    print('Saved S21 (fungi)')


if __name__ == '__main__':
    main()
