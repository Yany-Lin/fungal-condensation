#!/usr/bin/env python3
"""Bootstrap 95% CIs for Figure 2 calibration regressions (panels A-D)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'code' / 'test_tracking'))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from make_manuscript_panels import (
    compute_tau_zone_metric, fit_hill_dstar, HG_TRIALS
)

# Paths
METRICS = Path(__file__).resolve().parents[2] / 'FigureTable' / 'output' / 'universal_metrics.csv'
OUT_DIR = Path(__file__).resolve().parents[2] / 'additions' / '3_bootstrap_CIs'
SUPP_DIR = Path(__file__).resolve().parents[2] / 'supplementary_figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
MM = 1 / 25.4
TS = 7.0
LS = 8.5
PL = 12.0
LW = 0.6
COLORS = {'Agar': '#3A9E6F', '0.5:1': '#E67E22', '1:1': '#5B8FC9', '2:1': '#C0392B'}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TS,
    'axes.linewidth': LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'svg.fonttype': 'none',
})

N_BOOT = 10_000
RNG = np.random.default_rng(2024)


def style_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TS)


def bootstrap_regression(x, y, n_boot=N_BOOT, xfit=None):
    if xfit is None:
        xfit = np.linspace(x.min() - 0.02 * (x.max() - x.min()),
                           x.max() + 0.02 * (x.max() - x.min()), 200)
    n = len(x)
    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    r2s = np.empty(n_boot)
    yhat_boot = np.empty((n_boot, len(xfit)))

    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        sl, ic, r, p, se = stats.linregress(xb, yb)
        slopes[i] = sl
        intercepts[i] = ic
        r2s[i] = r ** 2
        yhat_boot[i] = ic + sl * xfit

    sl0, ic0, r0, p0, se0 = stats.linregress(x, y)
    ci_lo = np.percentile(yhat_boot, 2.5, axis=0)
    ci_hi = np.percentile(yhat_boot, 97.5, axis=0)

    resid = y - (ic0 + sl0 * x)
    sigma_resid = np.std(resid, ddof=2)
    pi_lo_arr = yhat_boot - 1.96 * sigma_resid
    pi_hi_arr = yhat_boot + 1.96 * sigma_resid
    pi_lo = np.percentile(pi_lo_arr, 2.5, axis=0)
    pi_hi = np.percentile(pi_hi_arr, 97.5, axis=0)

    def ci(arr, est):
        return {'est': est, 'lo': np.percentile(arr, 2.5),
                'hi': np.percentile(arr, 97.5)}

    info = {
        'slope': ci(slopes, sl0),
        'intercept': ci(intercepts, ic0),
        'r2': ci(r2s, r0 ** 2),
        'p_value': p0,
        'n': n,
    }
    bands = {
        'xfit': xfit, 'yfit': ic0 + sl0 * xfit,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'pi_lo': pi_lo, 'pi_hi': pi_hi,
    }
    return info, bands


def plot_panel(ax, x, y, groups, xlabel, ylabel, panel_label, xfit=None):
    valid = ~np.isnan(y) & ~np.isnan(x)
    xv, yv, gv = x[valid], y[valid], groups[valid]

    info, bands = bootstrap_regression(xv, yv, xfit=xfit)

    ax.fill_between(bands['xfit'], bands['pi_lo'], bands['pi_hi'],
                    color='#888888', alpha=0.08, zorder=1, label='95% prediction')
    ax.fill_between(bands['xfit'], bands['ci_lo'], bands['ci_hi'],
                    color='#444444', alpha=0.18, zorder=1, label='95% CI')
    ax.plot(bands['xfit'], bands['yfit'], 'k--', lw=1.2, alpha=0.7, zorder=2)

    for group, color in COLORS.items():
        mask = gv == group
        if mask.any():
            ax.scatter(xv[mask], yv[mask], c=color, s=35, alpha=0.65,
                       zorder=3, edgecolors='white', linewidths=0.4, label=group)

    r2 = info['r2']
    ax.text(0.95, 0.08,
            f'$R^2$ = {r2["est"]:.3f}\n[{r2["lo"]:.3f}, {r2["hi"]:.3f}]',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)

    ax.set_xlabel(xlabel, fontsize=LS, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=LS, labelpad=3)
    style_ax(ax)
    ax.text(-0.18, 1.05, panel_label, transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')
    ax.legend(fontsize=TS - 1, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.3)

    return info


def main():
    um = pd.read_csv(METRICS)
    hg = um[um['system'] == 'Hydrogel'].copy()
    hg['group_label'] = hg['group']
    hg['one_minus_aw'] = 1.0 - hg['a_w'].values
    print(f'Loaded {len(hg)} hydrogel trials')

    # Compute survival zone metric and d* for each trial
    print('\nComputing survival zone metric and d* from tracked data...')
    tau_zone_vals = []
    dstar_vals = []
    for tid in hg['trial_id']:
        tzm = compute_tau_zone_metric(tid)
        tau_zone_vals.append(tzm if tzm is not None else np.nan)

        K, popt = fit_hill_dstar(tid)
        if K is not None and K > 0:
            dstar_vals.append(K)
        elif K == 0:
            dstar_vals.append(0.0)
        else:
            dstar_vals.append(np.nan)

    hg['tau_zone_metric'] = tau_zone_vals
    hg['d_star'] = dstar_vals

    print(f'  tau_zone_metric: {sum(~np.isnan(hg["tau_zone_metric"]))} valid')
    print(f'  d_star: {sum(~np.isnan(hg["d_star"]))} valid')

    groups = hg['group_label'].values

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(170 * MM, 170 * MM))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.95, bottom=0.08,
                        wspace=0.40, hspace=0.35)

    # Panel A (2F): delta vs (1 - a_w)
    print('\n-- Panel A (2F): delta vs (1-a_w) --')
    xfit_aw = np.linspace(-0.02, 0.30, 200)
    info_A = plot_panel(axes[0, 0],
                        hg['one_minus_aw'].values,
                        hg['delta_um'].values,
                        groups,
                        r'$(1 - a_w)$',
                        r'$\delta$ ($\mu$m)',
                        'A', xfit=xfit_aw)
    axes[0, 0].set_xlim(-0.03, 0.30)
    axes[0, 0].set_xticks([0.0, 0.1, 0.2, 0.3])
    print(f'  R^2 = {info_A["r2"]["est"]:.3f} [{info_A["r2"]["lo"]:.3f}, {info_A["r2"]["hi"]:.3f}]')

    # Panel B (2H): zone_metric vs delta
    print('\n-- Panel B (2H): size gradient vs delta --')
    xfit_d = np.linspace(-20, 1050, 200)
    info_B = plot_panel(axes[0, 1],
                        hg['delta_um'].values,
                        hg['zone_metric'].values,
                        groups,
                        r'$\delta$ ($\mu$m)',
                        r'$(R_{\mathrm{far}} - R_{\mathrm{near}})\,/\,R_{\mathrm{mid}}$',
                        'B', xfit=xfit_d)
    axes[0, 1].set_xlim(0, 1100)
    print(f'  R^2 = {info_B["r2"]["est"]:.3f} [{info_B["r2"]["lo"]:.3f}, {info_B["r2"]["hi"]:.3f}]')

    # Panel C (2K): survival zone metric vs delta
    print('\n-- Panel C (2K): survival gradient vs delta --')
    info_C = plot_panel(axes[1, 0],
                        hg['delta_um'].values,
                        hg['tau_zone_metric'].values,
                        groups,
                        r'$\delta$ ($\mu$m)',
                        r'$(\tau_{50,\mathrm{far}} - \tau_{50,\mathrm{near}})\,/\,\tau_{50,\mathrm{mid}}$',
                        'C', xfit=xfit_d)
    axes[1, 0].set_xlim(0, 1100)
    print(f'  R^2 = {info_C["r2"]["est"]:.3f} [{info_C["r2"]["lo"]:.3f}, {info_C["r2"]["hi"]:.3f}]')

    # Panel D (2L): d* vs delta
    print('\n-- Panel D (2L): d* vs delta --')
    info_D = plot_panel(axes[1, 1],
                        hg['delta_um'].values,
                        hg['d_star'].values,
                        groups,
                        r'$\delta$ ($\mu$m)',
                        r'$d^*$ (mm)',
                        'D', xfit=xfit_d)
    axes[1, 1].set_xlim(0, 1100)
    print(f'  R^2 = {info_D["r2"]["est"]:.3f} [{info_D["r2"]["lo"]:.3f}, {info_D["r2"]["hi"]:.3f}]')

    # Save
    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(OUT_DIR / f'bootstrap_CIs_corrected{ext}',
                    bbox_inches='tight', facecolor='white',
                    dpi=dpi if dpi else 'figure')
    # Also save directly as SuppFig6
    fig.savefig(SUPP_DIR / 'SuppFig6.pdf',
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'\nSaved to {OUT_DIR}/bootstrap_CIs_corrected.* and {SUPP_DIR}/SuppFig6.pdf')


if __name__ == '__main__':
    main()
