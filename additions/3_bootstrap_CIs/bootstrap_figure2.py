#!/usr/bin/env python3
"""Bootstrap 95% CIs for Figure 2 regressions (panels 2F, 2H, 2K, 2L)."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

METRICS = Path(__file__).resolve().parents[2] / 'FigureTable' / 'output' / 'universal_metrics.csv'
OUT_DIR = Path(__file__).resolve().parents[2] / 'additions' / '3_bootstrap_CIs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
TS = 7.0          # tick / annotation font size
LS = 8.5          # axis-label font size
PL = 12.0         # panel-label font size
LW = 0.6          # spine / tick width
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
    """Return bootstrap statistics and band arrays."""
    if xfit is None:
        xfit = np.linspace(x.min() - 0.02, x.max() + 0.02, 200)

    n = len(x)
    slopes = np.empty(n_boot)
    intercepts = np.empty(n_boot)
    r2s = np.empty(n_boot)
    rhos = np.empty(n_boot)
    yhat_boot = np.empty((n_boot, len(xfit)))

    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        sl, ic, r, p, se = stats.linregress(xb, yb)
        slopes[i] = sl
        intercepts[i] = ic
        r2s[i] = r ** 2
        rhos[i] = stats.spearmanr(xb, yb).statistic
        yhat_boot[i] = ic + sl * xfit

    # Point estimates on original data
    sl0, ic0, r0, p0, se0 = stats.linregress(x, y)
    rho0 = stats.spearmanr(x, y).statistic

    # Confidence band (envelope of regression lines)
    ci_lo = np.percentile(yhat_boot, 2.5, axis=0)
    ci_hi = np.percentile(yhat_boot, 97.5, axis=0)

    # Prediction band = CI band + residual noise
    resid = y - (ic0 + sl0 * x)
    sigma_resid = np.std(resid, ddof=2)
    # For each bootstrap line, add +/- noise term
    pi_lo_arr = np.empty((n_boot, len(xfit)))
    pi_hi_arr = np.empty((n_boot, len(xfit)))
    for i in range(n_boot):
        pi_lo_arr[i] = yhat_boot[i] - 1.96 * sigma_resid
        pi_hi_arr[i] = yhat_boot[i] + 1.96 * sigma_resid
    pi_lo = np.percentile(pi_lo_arr, 2.5, axis=0)
    pi_hi = np.percentile(pi_hi_arr, 97.5, axis=0)

    def ci(arr, est):
        return {'est': est, 'lo': np.percentile(arr, 2.5),
                'hi': np.percentile(arr, 97.5)}

    info = {
        'slope': ci(slopes, sl0),
        'intercept': ci(intercepts, ic0),
        'r2': ci(r2s, r0 ** 2),
        'spearman_rho': ci(rhos, rho0),
        'p_value': p0,
        'n': n,
    }
    bands = {
        'xfit': xfit,
        'yfit': ic0 + sl0 * xfit,
        'ci_lo': ci_lo, 'ci_hi': ci_hi,
        'pi_lo': pi_lo, 'pi_hi': pi_hi,
    }
    return info, bands


def plot_regression_panel(ax, x, y, groups, ylabel, panel_label,
                          xfit=None, show_legend=True, square=False):
    """Scatter + bootstrap CI/PI bands for a single regression panel."""
    if xfit is None:
        xfit = np.linspace(-0.02, 0.30, 200)

    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    gv = groups[valid]

    info, bands = bootstrap_regression(xv, yv, xfit=xfit)

    # Prediction band (lighter)
    ax.fill_between(bands['xfit'], bands['pi_lo'], bands['pi_hi'],
                    color='#888888', alpha=0.08, zorder=1,
                    label='95% prediction')

    # Confidence band (darker)
    ax.fill_between(bands['xfit'], bands['ci_lo'], bands['ci_hi'],
                    color='#444444', alpha=0.18, zorder=1,
                    label='95% CI')

    # Best-fit line
    ax.plot(bands['xfit'], bands['yfit'], 'k--', lw=1.2, alpha=0.7, zorder=2)

    # Data points (colored by group)
    for group, color in COLORS.items():
        mask = gv == group
        if mask.any():
            ax.scatter(xv[mask], yv[mask], c=color, s=35, alpha=0.65,
                       zorder=3, edgecolors='white', linewidths=0.4,
                       label=group)

    # R^2 annotation
    r2 = info['r2']
    ax.text(0.95, 0.08,
            f'$R^2$ = {r2["est"]:.3f}\n[{r2["lo"]:.3f}, {r2["hi"]:.3f}]',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)

    ax.set_xlabel(r'$(1 - a_w)$', fontsize=LS, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=LS, labelpad=3)
    ax.set_xlim(-0.03, 0.30)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    style_ax(ax)

    if panel_label:
        ax.text(-0.18, 1.05, panel_label, transform=ax.transAxes,
                fontsize=PL, fontweight='bold', va='top')

    if show_legend:
        ax.legend(fontsize=TS - 1, loc='upper left', framealpha=0.9,
                  handletextpad=0.3, borderpad=0.3, labelspacing=0.3)

    if square:
        ax.set_box_aspect(1)

    return info


def save_panel(fig, name):
    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(OUT_DIR / f'{name}{ext}',
                    bbox_inches='tight', facecolor='white',
                    dpi=dpi if dpi else 'figure')
    plt.close(fig)


def main():
    um = pd.read_csv(METRICS)

    hg = um[um['system'] == 'Hydrogel'].copy()
    hg['group_label'] = hg['group']  # already 'Agar', '1:1', '2:1'
    hg['one_minus_aw'] = 1.0 - hg['a_w'].values
    print(f'Loaded {len(hg)} hydrogel trials')
    print(hg[['trial_id', 'group', 'a_w', 'one_minus_aw']].to_string(index=False))

    x_aw = hg['one_minus_aw'].values
    groups = hg['group_label'].values

    xfit = np.linspace(-0.02, 0.30, 200)
    summary_rows = []

    def record(panel, info):
        row = {'panel': panel, 'n': info['n'], 'p_value': info['p_value']}
        for key in ['slope', 'intercept', 'r2', 'spearman_rho']:
            d = info[key]
            row[f'{key}_est'] = d['est']
            row[f'{key}_lo'] = d['lo']
            row[f'{key}_hi'] = d['hi']
        summary_rows.append(row)

    print('\n── Panel 2F ──')
    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.16)
    info_F = plot_regression_panel(
        ax, x_aw, hg['delta_um'].values, groups,
        r'$\delta$ ($\mu$m)', 'F', xfit=xfit)
    save_panel(fig, 'panel_2F_bootstrap')
    record('2F (delta vs 1-aw)', info_F)
    print(f"  slope = {info_F['slope']['est']:.1f} "
          f"[{info_F['slope']['lo']:.1f}, {info_F['slope']['hi']:.1f}]")
    print(f"  R^2   = {info_F['r2']['est']:.3f} "
          f"[{info_F['r2']['lo']:.3f}, {info_F['r2']['hi']:.3f}]")
    print(f"  rho   = {info_F['spearman_rho']['est']:.3f} "
          f"[{info_F['spearman_rho']['lo']:.3f}, {info_F['spearman_rho']['hi']:.3f}]")

    print('\n── Panel 2H ──')
    y_zone = hg['zone_metric'].values
    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.16)
    info_H = plot_regression_panel(
        ax, x_aw, y_zone, groups,
        r'$(R_{\mathrm{far}} - R_{\mathrm{near}})\,/\,R_{\mathrm{mid}}$',
        'H', xfit=xfit)
    save_panel(fig, 'panel_2H_bootstrap')
    record('2H (zone metric vs 1-aw)', info_H)
    print(f"  slope = {info_H['slope']['est']:.4f} "
          f"[{info_H['slope']['lo']:.4f}, {info_H['slope']['hi']:.4f}]")
    print(f"  R^2   = {info_H['r2']['est']:.3f} "
          f"[{info_H['r2']['lo']:.3f}, {info_H['r2']['hi']:.3f}]")
    print(f"  rho   = {info_H['spearman_rho']['est']:.3f} "
          f"[{info_H['spearman_rho']['lo']:.3f}, {info_H['spearman_rho']['hi']:.3f}]")

    print('\n── Panel 2K ──')
    y_dtau = hg['dtau50_dr'].values
    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.92, bottom=0.16)
    info_K = plot_regression_panel(
        ax, x_aw, y_dtau, groups,
        r'd$\tau_{50}$/d$r$ (min mm$^{-1}$)', 'K',
        xfit=xfit, square=True)
    save_panel(fig, 'panel_2K_bootstrap')
    record('2K (dtau/dr vs 1-aw)', info_K)
    print(f"  slope = {info_K['slope']['est']:.3f} "
          f"[{info_K['slope']['lo']:.3f}, {info_K['slope']['hi']:.3f}]")
    print(f"  R^2   = {info_K['r2']['est']:.3f} "
          f"[{info_K['r2']['lo']:.3f}, {info_K['r2']['hi']:.3f}]")
    print(f"  rho   = {info_K['spearman_rho']['est']:.3f} "
          f"[{info_K['spearman_rho']['lo']:.3f}, {info_K['spearman_rho']['hi']:.3f}]")

    print('\n── Panel 2L ──')
    y_total = hg['dtau50_dr'].values
    y_size  = hg['dtau50_dr_sizematched'].values

    # Drop NaN for each channel independently
    valid_total = ~np.isnan(y_total)
    valid_size  = ~np.isnan(y_size)

    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.22, right=0.92, top=0.92, bottom=0.16)

    xt = x_aw[valid_total]
    yt = y_total[valid_total]
    gt = groups[valid_total]
    info_L_total, bands_total = bootstrap_regression(xt, yt, xfit=xfit)

    xs = x_aw[valid_size]
    ys = y_size[valid_size]
    gs = groups[valid_size]
    info_L_size, bands_size = bootstrap_regression(xs, ys, xfit=xfit)

    # Prediction band — Total (lightest)
    ax.fill_between(xfit, bands_total['pi_lo'], bands_total['pi_hi'],
                    color='#333333', alpha=0.06, zorder=0)

    # CI band — Total
    ax.fill_between(xfit, bands_total['ci_lo'], bands_total['ci_hi'],
                    color='#333333', alpha=0.15, zorder=1)

    # CI band — Size channel
    ax.fill_between(xfit, bands_size['ci_lo'], bands_size['ci_hi'],
                    color='#3A9E6F', alpha=0.15, zorder=1)

    # Rate channel shading (gap between the two best-fit lines)
    ax.fill_between(xfit, bands_size['yfit'], bands_total['yfit'],
                    color='#C0392B', alpha=0.12, zorder=1,
                    label='Rate channel')

    # Best-fit lines
    r2_t = info_L_total['r2']['est']
    r2_s = info_L_size['r2']['est']
    ax.plot(xfit, bands_total['yfit'], 'k-', lw=1.2, alpha=0.7, zorder=2,
            label=f'Total ($R^2$={r2_t:.2f})')
    ax.plot(xfit, bands_size['yfit'], 'k--', lw=1.2, alpha=0.7, zorder=2,
            label=f'Size channel ($R^2$={r2_s:.2f})')

    # Data points
    for group, color in COLORS.items():
        mask_t = gt == group
        mask_s = gs == group
        if mask_t.any():
            ax.scatter(xt[mask_t], yt[mask_t], c=color, s=35, alpha=0.55,
                       edgecolors='white', linewidths=0.4, marker='o', zorder=4)
        if mask_s.any():
            ax.scatter(xs[mask_s], ys[mask_s], c=color, s=35, alpha=0.55,
                       edgecolors='white', linewidths=0.4, marker='^', zorder=3)

    ax.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.legend(fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.4, borderpad=0.3)
    ax.set_xlabel(r'$(1 - a_w)$', fontsize=LS + 1, labelpad=4)
    ax.set_ylabel(r'd$\tau_{50}$/d$r$  (min mm$^{-1}$)',
                  fontsize=LS, labelpad=4)
    ax.set_xlim(-0.03, 0.30)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    style_ax(ax)

    # R^2 CIs as inset text
    ax.text(0.95, 0.08,
            f'Total: $R^2$={r2_t:.3f} [{info_L_total["r2"]["lo"]:.3f}, '
            f'{info_L_total["r2"]["hi"]:.3f}]\n'
            f'Size:  $R^2$={r2_s:.3f} [{info_L_size["r2"]["lo"]:.3f}, '
            f'{info_L_size["r2"]["hi"]:.3f}]',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=TS - 0.5)

    save_panel(fig, 'panel_2L_bootstrap')

    record('2L Total (dtau/dr vs 1-aw)', info_L_total)
    record('2L Size channel', info_L_size)

    print(f"  Total slope = {info_L_total['slope']['est']:.3f} "
          f"[{info_L_total['slope']['lo']:.3f}, {info_L_total['slope']['hi']:.3f}]")
    print(f"  Total R^2   = {info_L_total['r2']['est']:.3f} "
          f"[{info_L_total['r2']['lo']:.3f}, {info_L_total['r2']['hi']:.3f}]")
    print(f"  Size slope  = {info_L_size['slope']['est']:.3f} "
          f"[{info_L_size['slope']['lo']:.3f}, {info_L_size['slope']['hi']:.3f}]")
    print(f"  Size R^2    = {info_L_size['r2']['est']:.3f} "
          f"[{info_L_size['r2']['lo']:.3f}, {info_L_size['r2']['hi']:.3f}]")

    summary = pd.DataFrame(summary_rows)

    # Formatted columns for manuscript insertion
    def fmt_ci(est, lo, hi, prec=3):
        return f'{est:.{prec}f} [{lo:.{prec}f}, {hi:.{prec}f}]'

    summary['slope_formatted'] = summary.apply(
        lambda r: fmt_ci(r['slope_est'], r['slope_lo'], r['slope_hi'],
                         prec=3 if abs(r['slope_est']) < 10 else 1), axis=1)
    summary['intercept_formatted'] = summary.apply(
        lambda r: fmt_ci(r['intercept_est'], r['intercept_lo'],
                         r['intercept_hi'], prec=3), axis=1)
    summary['r2_formatted'] = summary.apply(
        lambda r: fmt_ci(r['r2_est'], r['r2_lo'], r['r2_hi']), axis=1)
    summary['spearman_formatted'] = summary.apply(
        lambda r: fmt_ci(r['spearman_rho_est'], r['spearman_rho_lo'],
                         r['spearman_rho_hi']), axis=1)

    csv_path = OUT_DIR / 'bootstrap_summary.csv'
    summary.to_csv(csv_path, index=False)
    print(f'\n── Summary saved to {csv_path} ──')
    print(summary[['panel', 'n', 'slope_formatted', 'r2_formatted',
                    'spearman_formatted', 'p_value']].to_string(index=False))

    print(f'\nAll panels saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
