#!/usr/bin/env python3
"""Bootstrap 95% CIs for Figure 3 & 4 regressions (panels 3D, 3E, 4B)."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUT_DIR = Path(__file__).resolve().parents[2] / 'additions' / '3_bootstrap_CIs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_REPO = Path(__file__).resolve().parents[2]
UNIVERSAL_CSV = _REPO / 'FigureTable' / 'output' / 'universal_metrics.csv'
RSR_CSV = _REPO / 'FigureRSR' / 'raw_data' / 'droplets_calibrated_mm.csv'

MM         = 1 / 25.4
TICK_SIZE  = 7.0
LABEL_SIZE = 8.5
PANEL_LBL  = 12.0
LW         = 0.6
LW_DATA    = 0.8

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TICK_SIZE,
    'axes.linewidth': LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': LW_DATA,
    'svg.fonttype': 'none',
})

COLORS = {
    'Agar': '#3A9E6F', '0.5:1': '#E67E22', '1:1': '#3A6FBF', '2:1': '#C0392B',
    'Green': '#4CAF50', 'White': '#9E9E9E', 'Black': '#212121',
}
EDGE = {
    'Agar': '#2E7D32', '0.5:1': '#B7600A', '1:1': '#2C5F9F', '2:1': '#922B21',
    'Green': '#2E7D32', 'White': '#616161', 'Black': '#000000',
}
MARKER = {
    'Agar': 'o', '0.5:1': 'o', '1:1': 'o', '2:1': 'o',
    'Green': 'D', 'White': 'D', 'Black': 'D',
}
GROUP_ORDER = ['Agar', '0.5:1', '1:1', '2:1', 'Green', 'White', 'Black']
HG_GROUPS   = {'Agar', '0.5:1', '1:1', '2:1'}

N_BOOT = 10_000
RNG = np.random.default_rng(42)


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=TICK_SIZE, pad=2)


def _save(fig, stem):
    for ext in ('.svg', '.pdf', '.png'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'{stem}{ext}', **kw)
    plt.close(fig)
    print(f'  Saved -> {OUTPUT_DIR}/{stem}.*')


def bootstrap_regression(x, y, n_boot=N_BOOT, rng=RNG):
    """Bootstrap OLS: return arrays of slopes, intercepts, R² values."""
    n = len(x)
    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    r2s = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        res = stats.linregress(xb, yb)
        slopes[i] = res.slope
        intercepts[i] = res.intercept
        r2s[i] = res.rvalue ** 2
    return slopes, intercepts, r2s


def bootstrap_stratified(x, y, group_labels, n_boot=N_BOOT, rng=RNG):
    """Stratified bootstrap: resample within each group."""
    unique_groups = np.unique(group_labels)
    n = len(x)
    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    r2s = np.empty(n_boot)
    # Pre-compute group indices
    group_idx = {g: np.where(group_labels == g)[0] for g in unique_groups}
    for i in range(n_boot):
        boot_idx = []
        for g in unique_groups:
            gi = group_idx[g]
            boot_idx.append(rng.choice(gi, size=len(gi), replace=True))
        idx = np.concatenate(boot_idx)
        xb, yb = x[idx], y[idx]
        res = stats.linregress(xb, yb)
        slopes[i] = res.slope
        intercepts[i] = res.intercept
        r2s[i] = res.rvalue ** 2
    return slopes, intercepts, r2s


def ci95(arr):
    return np.percentile(arr, [2.5, 97.5])


def theil_sen(x, y):
    """Theil-Sen robust regression."""
    res = stats.theilslopes(y, x, alpha=0.95)
    return res  # slope, intercept, low_slope, high_slope


def load_universal():
    df = pd.read_csv(UNIVERSAL_CSV)
    # Exclude leaf trials
    df = df[~df['group'].isin(['Healthy', 'Diseased'])].copy()
    return df


def panel_3D(summary_rows):
    print('\nPanel 3D: dτ₅₀/dr vs δ')
    df = load_universal()

    # Overall regression
    mask = df['dtau50_dr'].notna() & df['delta_um'].notna()
    d = df.loc[mask].copy()
    x_all = d['delta_um'].values
    y_all = d['dtau50_dr'].values
    g_all = d['group'].values

    res_ols = stats.linregress(x_all, y_all)
    print(f'  OLS overall: slope={res_ols.slope:.5f}, '
          f'R²={res_ols.rvalue**2:.3f}')

    # Bootstrap — overall (case resampling)
    sl_b, ic_b, r2_b = bootstrap_regression(x_all, y_all)
    print(f'  Bootstrap slope: {np.median(sl_b):.5f} '
          f'[{ci95(sl_b)[0]:.5f}, {ci95(sl_b)[1]:.5f}]')
    print(f'  Bootstrap R²:    {np.median(r2_b):.3f} '
          f'[{ci95(r2_b)[0]:.3f}, {ci95(r2_b)[1]:.3f}]')

    summary_rows.append({
        'panel': '3D', 'regression': 'overall', 'method': 'OLS',
        'slope': res_ols.slope, 'slope_CI_lo': ci95(sl_b)[0],
        'slope_CI_hi': ci95(sl_b)[1],
        'R2': res_ols.rvalue**2, 'R2_CI_lo': ci95(r2_b)[0],
        'R2_CI_hi': ci95(r2_b)[1],
        'intercept': res_ols.intercept,
        'p_value': res_ols.pvalue, 'n': len(x_all),
    })

    # Stratified bootstrap — overall
    sl_s, ic_s, r2_s = bootstrap_stratified(x_all, y_all, g_all)
    print(f'  Stratified slope: {np.median(sl_s):.5f} '
          f'[{ci95(sl_s)[0]:.5f}, {ci95(sl_s)[1]:.5f}]')
    print(f'  Stratified R²:    {np.median(r2_s):.3f} '
          f'[{ci95(r2_s)[0]:.3f}, {ci95(r2_s)[1]:.3f}]')

    summary_rows.append({
        'panel': '3D', 'regression': 'overall_stratified', 'method': 'OLS',
        'slope': res_ols.slope, 'slope_CI_lo': ci95(sl_s)[0],
        'slope_CI_hi': ci95(sl_s)[1],
        'R2': res_ols.rvalue**2, 'R2_CI_lo': ci95(r2_s)[0],
        'R2_CI_hi': ci95(r2_s)[1],
        'intercept': res_ols.intercept,
        'p_value': res_ols.pvalue, 'n': len(x_all),
    })

    # Theil-Sen overall
    ts = theil_sen(x_all, y_all)
    print(f'  Theil-Sen overall: slope={ts.slope:.5f} '
          f'[{ts.low_slope:.5f}, {ts.high_slope:.5f}]')
    summary_rows.append({
        'panel': '3D', 'regression': 'overall', 'method': 'Theil-Sen',
        'slope': ts.slope, 'slope_CI_lo': ts.low_slope,
        'slope_CI_hi': ts.high_slope,
        'R2': '', 'R2_CI_lo': '', 'R2_CI_hi': '',
        'intercept': ts.intercept,
        'p_value': '', 'n': len(x_all),
    })

    # Size-matched regression
    mask_sm = d['dtau50_dr_sizematched'].notna()
    d_sm = d.loc[mask_sm]
    x_sm = d_sm['delta_um'].values
    y_sm = d_sm['dtau50_dr_sizematched'].values
    g_sm = d_sm['group'].values

    res_sm = stats.linregress(x_sm, y_sm)
    print(f'\n  OLS size-matched: slope={res_sm.slope:.5f}, '
          f'R²={res_sm.rvalue**2:.3f}')

    sl_sm, ic_sm, r2_sm = bootstrap_regression(x_sm, y_sm)
    print(f'  Bootstrap slope: {np.median(sl_sm):.5f} '
          f'[{ci95(sl_sm)[0]:.5f}, {ci95(sl_sm)[1]:.5f}]')
    print(f'  Bootstrap R²:    {np.median(r2_sm):.3f} '
          f'[{ci95(r2_sm)[0]:.3f}, {ci95(r2_sm)[1]:.3f}]')

    summary_rows.append({
        'panel': '3D', 'regression': 'size-matched', 'method': 'OLS',
        'slope': res_sm.slope, 'slope_CI_lo': ci95(sl_sm)[0],
        'slope_CI_hi': ci95(sl_sm)[1],
        'R2': res_sm.rvalue**2, 'R2_CI_lo': ci95(r2_sm)[0],
        'R2_CI_hi': ci95(r2_sm)[1],
        'intercept': res_sm.intercept,
        'p_value': res_sm.pvalue, 'n': len(x_sm),
    })

    # Stratified — size-matched
    sl_ss, ic_ss, r2_ss = bootstrap_stratified(x_sm, y_sm, g_sm)
    print(f'  Stratified slope: {np.median(sl_ss):.5f} '
          f'[{ci95(sl_ss)[0]:.5f}, {ci95(sl_ss)[1]:.5f}]')
    print(f'  Stratified R²:    {np.median(r2_ss):.3f} '
          f'[{ci95(r2_ss)[0]:.3f}, {ci95(r2_ss)[1]:.3f}]')

    summary_rows.append({
        'panel': '3D', 'regression': 'size-matched_stratified', 'method': 'OLS',
        'slope': res_sm.slope, 'slope_CI_lo': ci95(sl_ss)[0],
        'slope_CI_hi': ci95(sl_ss)[1],
        'R2': res_sm.rvalue**2, 'R2_CI_lo': ci95(r2_ss)[0],
        'R2_CI_hi': ci95(r2_ss)[1],
        'intercept': res_sm.intercept,
        'p_value': res_sm.pvalue, 'n': len(x_sm),
    })

    # Theil-Sen size-matched
    ts_sm = theil_sen(x_sm, y_sm)
    print(f'  Theil-Sen size-matched: slope={ts_sm.slope:.5f} '
          f'[{ts_sm.low_slope:.5f}, {ts_sm.high_slope:.5f}]')
    summary_rows.append({
        'panel': '3D', 'regression': 'size-matched', 'method': 'Theil-Sen',
        'slope': ts_sm.slope, 'slope_CI_lo': ts_sm.low_slope,
        'slope_CI_hi': ts_sm.high_slope,
        'R2': '', 'R2_CI_lo': '', 'R2_CI_hi': '',
        'intercept': ts_sm.intercept,
        'p_value': '', 'n': len(x_sm),
    })

    pct_diff = abs(ts.slope - res_ols.slope) / abs(res_ols.slope) * 100
    pct_diff_sm = abs(ts_sm.slope - res_sm.slope) / abs(res_sm.slope) * 100
    print(f'\n  OLS vs Theil-Sen difference (overall):      {pct_diff:.1f}%')
    print(f'  OLS vs Theil-Sen difference (size-matched): {pct_diff_sm:.1f}%')

    fig, ax = plt.subplots(figsize=(85 * MM, 75 * MM))
    fig.subplots_adjust(left=0.17, right=0.95, top=0.93, bottom=0.18)

    for grp in GROUP_ORDER:
        gd = d[d['group'] == grp]
        if len(gd) == 0:
            continue
        mk = MARKER[grp]
        ms = 6.0 if grp in HG_GROUPS else 5.0
        mec = 'white' if grp in HG_GROUPS else EDGE[grp]
        mew = 0.4 if grp in HG_GROUPS else 0.5
        ax.scatter(gd['delta_um'], gd['dtau50_dr'],
                   marker=mk, s=ms**2, color=COLORS[grp],
                   edgecolors=mec, linewidths=mew,
                   label=grp, zorder=5)

    xfit = np.linspace(0, 1100, 200)

    # CI band — overall (from bootstrap intercepts and slopes)
    y_boot = ic_b[:, None] + sl_b[:, None] * xfit[None, :]
    y_lo, y_hi = np.percentile(y_boot, [2.5, 97.5], axis=0)
    ax.fill_between(xfit, y_lo, y_hi, color='#333333', alpha=0.10, zorder=1)
    ax.plot(xfit, res_ols.intercept + res_ols.slope * xfit,
            '-', color='#333333', lw=LW_DATA, alpha=0.8, zorder=3)

    # CI band — size-matched
    y_boot_sm = ic_sm[:, None] + sl_sm[:, None] * xfit[None, :]
    y_lo_sm, y_hi_sm = np.percentile(y_boot_sm, [2.5, 97.5], axis=0)
    ax.fill_between(xfit, y_lo_sm, y_hi_sm, color='#C0392B', alpha=0.08, zorder=1)
    ax.plot(xfit, res_sm.intercept + res_sm.slope * xfit,
            '-', color='#C0392B', lw=LW_DATA, alpha=0.8, zorder=3)

    # Annotations
    r2_ov = res_ols.rvalue**2
    r2_sm_val = res_sm.rvalue**2
    ax.text(0.95, 0.22,
            f'Overall $R^2$ = {r2_ov:.2f} [{ci95(r2_b)[0]:.2f}, {ci95(r2_b)[1]:.2f}]',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=TICK_SIZE - 1.0, color='#333333')
    ax.text(0.95, 0.10,
            f'Size-matched $R^2$ = {r2_sm_val:.2f} [{ci95(r2_sm)[0]:.2f}, {ci95(r2_sm)[1]:.2f}]',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=TICK_SIZE - 1.0, color='#C0392B')

    ax.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_ylabel(r'd$\tau_{50}$/d$r$ (min mm$^{-1}$)',
                  fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(0, 1100)
    style_ax(ax)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ord_h = [by_label[g] for g in GROUP_ORDER if g in by_label]
    ord_l = [g for g in GROUP_ORDER if g in by_label]
    # Add regression line legend entries
    ord_h.append(Line2D([0], [0], color='#333333', lw=LW_DATA))
    ord_l.append('Overall OLS')
    ord_h.append(Line2D([0], [0], color='#C0392B', lw=LW_DATA))
    ord_l.append('Size-matched OLS')
    ax.legend(ord_h, ord_l, fontsize=TICK_SIZE - 1.0, loc='upper left',
              frameon=False, labelspacing=0.3, handlelength=1.2,
              handletextpad=0.4, ncol=2, columnspacing=0.8)

    ax.text(-0.14, 1.04, 'D', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    _save(fig, 'panel_3D_bootstrap')


def panel_3E(summary_rows):
    print('\nPanel 3E: zone metric vs δ')
    df = load_universal()

    mask = df['zone_metric'].notna() & df['delta_um'].notna()
    d = df.loc[mask].copy()
    x = d['delta_um'].values
    y = d['zone_metric'].values
    g = d['group'].values

    res = stats.linregress(x, y)
    print(f'  OLS: slope={res.slope:.6f}, R²={res.rvalue**2:.3f}')

    # Bootstrap
    sl_b, ic_b, r2_b = bootstrap_regression(x, y)
    print(f'  Bootstrap slope: {np.median(sl_b):.6f} '
          f'[{ci95(sl_b)[0]:.6f}, {ci95(sl_b)[1]:.6f}]')
    print(f'  Bootstrap R²:    {np.median(r2_b):.3f} '
          f'[{ci95(r2_b)[0]:.3f}, {ci95(r2_b)[1]:.3f}]')

    summary_rows.append({
        'panel': '3E', 'regression': 'overall', 'method': 'OLS',
        'slope': res.slope, 'slope_CI_lo': ci95(sl_b)[0],
        'slope_CI_hi': ci95(sl_b)[1],
        'R2': res.rvalue**2, 'R2_CI_lo': ci95(r2_b)[0],
        'R2_CI_hi': ci95(r2_b)[1],
        'intercept': res.intercept,
        'p_value': res.pvalue, 'n': len(x),
    })

    # Stratified bootstrap
    sl_s, ic_s, r2_s = bootstrap_stratified(x, y, g)
    print(f'  Stratified slope: {np.median(sl_s):.6f} '
          f'[{ci95(sl_s)[0]:.6f}, {ci95(sl_s)[1]:.6f}]')
    print(f'  Stratified R²:    {np.median(r2_s):.3f} '
          f'[{ci95(r2_s)[0]:.3f}, {ci95(r2_s)[1]:.3f}]')

    summary_rows.append({
        'panel': '3E', 'regression': 'overall_stratified', 'method': 'OLS',
        'slope': res.slope, 'slope_CI_lo': ci95(sl_s)[0],
        'slope_CI_hi': ci95(sl_s)[1],
        'R2': res.rvalue**2, 'R2_CI_lo': ci95(r2_s)[0],
        'R2_CI_hi': ci95(r2_s)[1],
        'intercept': res.intercept,
        'p_value': res.pvalue, 'n': len(x),
    })

    # Theil-Sen
    ts = theil_sen(x, y)
    print(f'  Theil-Sen: slope={ts.slope:.6f} '
          f'[{ts.low_slope:.6f}, {ts.high_slope:.6f}]')
    pct_diff = abs(ts.slope - res.slope) / abs(res.slope) * 100
    print(f'  OLS vs Theil-Sen difference: {pct_diff:.1f}%')

    summary_rows.append({
        'panel': '3E', 'regression': 'overall', 'method': 'Theil-Sen',
        'slope': ts.slope, 'slope_CI_lo': ts.low_slope,
        'slope_CI_hi': ts.high_slope,
        'R2': '', 'R2_CI_lo': '', 'R2_CI_hi': '',
        'intercept': ts.intercept,
        'p_value': '', 'n': len(x),
    })

    fig, ax = plt.subplots(figsize=(85 * MM, 75 * MM))
    fig.subplots_adjust(left=0.17, right=0.95, top=0.93, bottom=0.18)

    for grp in GROUP_ORDER:
        gd = d[d['group'] == grp]
        if len(gd) == 0:
            continue
        mk = MARKER[grp]
        ms = 6.0 if grp in HG_GROUPS else 5.0
        mec = 'white' if grp in HG_GROUPS else EDGE[grp]
        mew = 0.4 if grp in HG_GROUPS else 0.5
        ax.scatter(gd['delta_um'], gd['zone_metric'],
                   marker=mk, s=ms**2, color=COLORS[grp],
                   edgecolors=mec, linewidths=mew,
                   label=grp, zorder=5)

    xfit = np.linspace(0, 1100, 200)

    # CI band
    y_boot = ic_b[:, None] + sl_b[:, None] * xfit[None, :]
    y_lo, y_hi = np.percentile(y_boot, [2.5, 97.5], axis=0)
    ax.fill_between(xfit, y_lo, y_hi, color='#333333', alpha=0.10, zorder=1)
    ax.plot(xfit, res.intercept + res.slope * xfit,
            '-', color='#333333', lw=LW_DATA, alpha=0.8, zorder=3)

    r2 = res.rvalue**2
    ax.text(0.95, 0.08,
            f'$R^2$ = {r2:.2f} [{ci95(r2_b)[0]:.2f}, {ci95(r2_b)[1]:.2f}]',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=TICK_SIZE - 1.0)

    ax.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_ylabel(r'$(R_{\rm far} - R_{\rm near})\,/\,R_{\rm mid}$',
                  fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(0, 1100)
    style_ax(ax)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ord_h = [by_label[g] for g in GROUP_ORDER if g in by_label]
    ord_l = [g for g in GROUP_ORDER if g in by_label]
    ax.legend(ord_h, ord_l, fontsize=TICK_SIZE - 1.0, loc='upper left',
              frameon=False, labelspacing=0.3, handlelength=1.2,
              handletextpad=0.4, ncol=2, columnspacing=0.8)

    ax.text(-0.14, 1.04, 'E', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    _save(fig, 'panel_3E_bootstrap')


def panel_4B(summary_rows):
    print('\nPanel 4B: RSR leaf regressions')
    df = pd.read_csv(RSR_CSV)
    um = load_universal()

    TRIALS = ['RSR1', 'RSR2', 'RSR7',
              'RSRDiseased3', 'RSRDiseased5', 'RSRDiseased6']
    LABELS = {
        'RSR1': 'Healthy 1', 'RSR2': 'Healthy 2', 'RSR7': 'Healthy 3',
        'RSRDiseased3': 'Diseased 1', 'RSRDiseased5': 'Diseased 2',
        'RSRDiseased6': 'Diseased 3',
    }

    # Per-trial slopes and correlations from raw data
    trial_slopes = []
    trial_rs = []
    trial_slope_cis = []

    for tid in TRIALS:
        sub = df[df['sample'] == tid].copy()
        x = sub['dist_mm'].values
        y = sub['r_eq_mm'].values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        res = stats.linregress(x, y)
        # Convert slope to µm/mm for consistency with universal_metrics
        slope_um_mm = res.slope * 1000  # mm/mm -> µm/mm
        r_val = res.rvalue

        trial_slopes.append(slope_um_mm)
        trial_rs.append(r_val)

        # Per-trial bootstrap CI on slope
        sl_t, _, _ = bootstrap_regression(x, y)
        sl_t_um = sl_t * 1000
        ci_lo, ci_hi = ci95(sl_t_um)
        trial_slope_cis.append((ci_lo, ci_hi))

        print(f'  {tid}: slope={slope_um_mm:.2f} µm/mm '
              f'[{ci_lo:.2f}, {ci_hi:.2f}], r={r_val:.4f}')

        summary_rows.append({
            'panel': '4B', 'regression': f'{tid}_per_trial',
            'method': 'OLS',
            'slope': slope_um_mm,
            'slope_CI_lo': ci_lo, 'slope_CI_hi': ci_hi,
            'R2': r_val**2, 'R2_CI_lo': '', 'R2_CI_hi': '',
            'intercept': res.intercept * 1000,
            'p_value': res.pvalue, 'n': len(x),
        })

    # Mean dR/dr across 6 trials — bootstrap CI on the mean
    trial_slopes_arr = np.array(trial_slopes)
    trial_rs_arr = np.array(trial_rs)

    mean_slope = np.mean(trial_slopes_arr)
    mean_r = np.mean(trial_rs_arr)

    # Bootstrap on the 6 trial-level statistics
    boot_mean_slopes = np.empty(N_BOOT)
    boot_mean_rs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, len(trial_slopes_arr), size=len(trial_slopes_arr))
        boot_mean_slopes[i] = np.mean(trial_slopes_arr[idx])
        boot_mean_rs[i] = np.mean(trial_rs_arr[idx])

    print(f'\n  Mean dR/dr across 6 trials: {mean_slope:.2f} µm/mm '
          f'[{ci95(boot_mean_slopes)[0]:.2f}, {ci95(boot_mean_slopes)[1]:.2f}]')
    print(f'  Mean Pearson r across 6 trials: {mean_r:.4f} '
          f'[{ci95(boot_mean_rs)[0]:.4f}, {ci95(boot_mean_rs)[1]:.4f}]')

    summary_rows.append({
        'panel': '4B', 'regression': 'mean_across_6_trials',
        'method': 'Bootstrap mean',
        'slope': mean_slope,
        'slope_CI_lo': ci95(boot_mean_slopes)[0],
        'slope_CI_hi': ci95(boot_mean_slopes)[1],
        'R2': mean_r**2,
        'R2_CI_lo': ci95(boot_mean_rs)[0]**2,
        'R2_CI_hi': ci95(boot_mean_rs)[1]**2,
        'intercept': '',
        'p_value': '', 'n': 6,
    })
    summary_rows.append({
        'panel': '4B', 'regression': 'mean_Pearson_r_across_6_trials',
        'method': 'Bootstrap mean',
        'slope': '',
        'slope_CI_lo': '', 'slope_CI_hi': '',
        'R2': mean_r,
        'R2_CI_lo': ci95(boot_mean_rs)[0],
        'R2_CI_hi': ci95(boot_mean_rs)[1],
        'intercept': '',
        'p_value': '', 'n': 6,
    })

    fig, axes = plt.subplots(3, 2, figsize=(110 * MM, 130 * MM),
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.30, wspace=0.20,
                        left=0.14, right=0.96, top=0.92, bottom=0.10)

    axes[0, 0].set_title('Healthy', fontsize=LABEL_SIZE, fontweight='bold', pad=6)
    axes[0, 1].set_title('Diseased', fontsize=LABEL_SIZE, fontweight='bold', pad=6)

    healthy = ['RSR1', 'RSR2', 'RSR7']
    diseased = ['RSRDiseased3', 'RSRDiseased5', 'RSRDiseased6']

    for col, trial_list in enumerate([healthy, diseased]):
        for row, tid in enumerate(trial_list):
            ax = axes[row, col]
            sub = df[df['sample'] == tid].copy()
            x = sub['dist_mm'].values
            y = sub['r_eq_mm'].values
            valid = np.isfinite(x) & np.isfinite(y)
            xv, yv = x[valid], y[valid]

            ax.scatter(xv, yv, s=3, c='#888888', alpha=0.4,
                       edgecolors='none', rasterized=True, zorder=1)

            res = stats.linregress(xv, yv)
            xfit = np.linspace(xv.min(), xv.max(), 200)

            # Bootstrap CI band
            sl_t, ic_t, _ = bootstrap_regression(xv, yv)
            y_boot = ic_t[:, None] + sl_t[:, None] * xfit[None, :]
            y_lo, y_hi = np.percentile(y_boot, [2.5, 97.5], axis=0)
            ax.fill_between(xfit, y_lo, y_hi, color='#C0392B',
                            alpha=0.12, zorder=2)
            ax.plot(xfit, res.intercept + res.slope * xfit,
                    color='#C0392B', lw=1.2, zorder=3)

            # Slope CI text
            idx_t = TRIALS.index(tid)
            ci_lo, ci_hi = trial_slope_cis[idx_t]
            ax.text(0.95, 0.08,
                    f'$r$ = {trial_rs[idx_t]:.2f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=TICK_SIZE - 1.0)

            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)

    fig.text(0.55, 0.02, 'Distance from boundary (mm)',
             ha='center', fontsize=LABEL_SIZE)
    fig.text(0.02, 0.52, r'$r_{\mathrm{eq}}$ (mm)',
             ha='center', va='center', rotation=90, fontsize=LABEL_SIZE)

    axes[0, 0].text(-0.35, 1.15, 'B', transform=axes[0, 0].transAxes,
                    fontsize=PANEL_LBL, fontweight='bold', va='top')

    _save(fig, 'panel_4B_bootstrap')


def main():
    print('=' * 70)
    print('Bootstrap CIs for Figure 3 & 4 regressions')
    print(f'N_BOOT = {N_BOOT}')
    print('=' * 70)

    summary_rows = []

    panel_3D(summary_rows)
    panel_3E(summary_rows)
    panel_4B(summary_rows)

    # Save summary
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / 'bootstrap_summary.csv', index=False)
    print(f'\n  Summary saved -> {OUTPUT_DIR}/bootstrap_summary.csv')

    # Print formatted table
    print('\n' + '=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)
    for _, r in summary.iterrows():
        sl_str = ''
        if r['slope'] != '':
            sl_str = f"slope={r['slope']:.5f}"
            if r['slope_CI_lo'] != '':
                sl_str += f" [{r['slope_CI_lo']:.5f}, {r['slope_CI_hi']:.5f}]"
        r2_str = ''
        if r['R2'] != '':
            r2_str = f"R²={r['R2']:.3f}" if isinstance(r['R2'], float) else f"R²={r['R2']}"
            if r['R2_CI_lo'] != '':
                r2_str += f" [{r['R2_CI_lo']:.3f}, {r['R2_CI_hi']:.3f}]"
        print(f"  {r['panel']} | {r['regression']:30s} | {r['method']:14s} | "
              f"{sl_str:45s} | {r2_str}")

    print('\nDone.')


if __name__ == '__main__':
    main()
