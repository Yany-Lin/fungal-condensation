#!/usr/bin/env python3
"""
Visual panels for the log-lifetime survival gradient analysis.

Outputs:
  panel_D_examples.{svg,pdf,png}  — 4 representative trials: log(tau) vs distance
  supp_all_trials.{svg,pdf,png}   — all 35 trials in a 5x7 grid
  supp_km_zones.{svg,pdf,png}     — KM survival curves near vs far zone
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from lifelines import KaplanMeierFitter

BASE      = Path(__file__).resolve().parents[2]
TRACK_DIR = BASE / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
UNIVERSAL = BASE / 'FigureTable' / 'output' / 'universal_metrics.csv'
OUT       = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

T_SEED     = 900.0
MIN_FRAMES = 3

ALL_TRIALS = {
    'agar.1': 'Agar',   'agar.2': 'Agar',   'agar.3': 'Agar',
    'agar.4': 'Agar',   'agar.5': 'Agar',
    '0.5to1.2': '0.5:1', '0.5to1.3': '0.5:1', '0.5to1.4': '0.5:1',
    '0.5to1.5': '0.5:1',
    '1to1.1': '1:1',   '1to1.2': '1:1',   '1to1.3': '1:1',
    '1to1.4': '1:1',   '1to1.5': '1:1',
    '2to1.1': '2:1',   '2to1.2': '2:1',   '2to1.3': '2:1',
    '2to1.4': '2:1',   '2to1.5': '2:1',
    'Green.1': 'Green', 'Green.2': 'Green', 'Green.3': 'Green',
    'Green.4': 'Green', 'Green.5': 'Green',
    'white.1': 'White', 'white.2': 'White', 'white.3': 'White',
    'white.4': 'White', 'white.5': 'White',
    'black.1': 'Black', 'black.2': 'Black', 'black.3': 'Black',
    'black.4': 'Black', 'black.5': 'Black',
}
GROUP_ORDER = ['Agar', '0.5:1', '1:1', '2:1', 'Green', 'White', 'Black']

COLORS = {
    'Agar': '#3A9E6F', '0.5:1': '#E67E22', '1:1': '#3A6FBF', '2:1': '#C0392B',
    'Green': '#4CAF50', 'White': '#9E9E9E', 'Black': '#212121',
}
EDGE = {
    'Agar': '#2E7D32', '0.5:1': '#B7600A', '1:1': '#2C5F9F', '2:1': '#922B21',
    'Green': '#2E7D32', 'White': '#616161', 'Black': '#000000',
}

MM         = 1 / 25.4
TICK_SIZE  = 7.0
LABEL_SIZE = 8.5
LW         = 0.6

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TICK_SIZE,
    'axes.linewidth': LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'svg.fonttype': 'none',
})

uni       = pd.read_csv(UNIVERSAL)
delta_map = dict(zip(uni['trial_id'], uni['delta_um']))


def load_trial(trial_id):
    fp = TRACK_DIR / f'{trial_id}_track_histories.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[df['tau_fwd_min'] > 0].copy()
    df['d_mm'] = df['distance_um'] / 1000.0
    return df


def km_tau50_per_bin(df, bin_mm=0.2, min_per_bin=15):
    """KM tau50 per distance bin. Returns (d_centers, tau50s). Display only."""
    kmf   = KaplanMeierFitter()
    d_max = df['d_mm'].max()
    bins  = np.arange(0, d_max + bin_mm, bin_mm)
    dfc   = df.copy()
    dfc['bin'] = pd.cut(dfc['d_mm'], bins=bins, labels=False)
    bx, by = [], []
    for b in sorted(dfc['bin'].dropna().unique()):
        sub = dfc[dfc['bin'] == b]
        if len(sub) < min_per_bin:
            continue
        center = bins[int(b)] + bin_mm / 2
        kmf.fit(sub['tau_fwd_min'], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        if np.isfinite(t50) and center > 0:
            bx.append(center)
            by.append(t50)
    return np.array(bx), np.array(by)


def compute_G(df, min_bins=4):
    """G = log(τ50(d90) / τ50(d10)) read off PCHIP spline through KM dots.
    Directly tied to what is plotted. No model extrapolation.
    Returns (G, d10, d90, spl) or (None, None, None, None)."""
    bx, by = km_tau50_per_bin(df)
    if len(bx) < min_bins:
        return None, None, None, None
    spl  = PchipInterpolator(bx, by)
    d10  = float(np.clip(np.percentile(df['d_mm'], 10), bx[0], bx[-1]))
    d90  = float(np.clip(np.percentile(df['d_mm'], 90), bx[0], bx[-1]))
    t_near = float(spl(d10))
    t_far  = float(spl(d90))
    if t_near <= 0 or t_far <= 0:
        return None, None, None, None
    G = float(np.log(t_far / t_near))
    return G, d10, d90, spl


def _hill(d, T0, A, K, n):
    """Hill / Michaelis-Menten saturation:
       T(d) = T0 + A * d^n / (K^n + d^n)
    T0: near-source baseline, A: amplitude (T∞ - T0), K: half-sat distance, n: cooperativity.
    Fits the steep early rise + sharp plateau that a simple sat-exp cannot capture.
    AIC advantage over sat-exp: ~36 units for 1:1 NaCl, ~40 units for 2:1 NaCl.
    """
    dn = np.power(np.maximum(d, 0), n)
    Kn = np.power(K, n)
    return T0 + A * dn / (Kn + dn)


def km_loglog_beta(df, min_bins=4):
    """Power-law slope β from log-log regression on KM tau50 per bin.
    β > 0 means farther droplets live longer.
    Measured over the observed data range — no extrapolation to d=0.
    Returns β or None."""
    bx, by = km_tau50_per_bin(df)
    mask = (bx > 0) & (by > 0)
    if mask.sum() < min_bins:
        return None
    return float(stats.linregress(np.log(bx[mask]), np.log(by[mask])).slope)


def fit_hill(df, min_bins=5, min_r2=0.80):
    """Fit Hill function to KM tau50 per bin (the open circles).
    Fits to binned medians so the curve actually traces the trend.
    Quality-gates on R²_KM before returning.

    Returns (T0, A, K, n, r2_km) or None.
    Metric for Panel D: A = T∞ − T0 (lifetime gradient amplitude, minutes).
    """
    bx, by = km_tau50_per_bin(df)
    if len(bx) < min_bins:
        return None

    T0_g = float(by[0])
    A_g  = float(by[-1] - by[0])
    if A_g <= 0:
        A_g = 0.1
    K_g  = float(np.median(bx))
    n_g  = 2.0

    lo = [0,    0,    0.01, 0.5]
    hi = [by.max(), by.max() * 3, bx.max() * 3, 8.0]

    try:
        popt, _ = curve_fit(_hill, bx, by,
                            p0=[T0_g, A_g, K_g, n_g],
                            bounds=(lo, hi), max_nfev=50000, method='trf')
    except Exception:
        return None

    T0_f, A_f, K_f, n_f = popt
    by_pred = _hill(bx, *popt)
    ss_res  = np.sum((by - by_pred) ** 2)
    ss_tot  = np.sum((by - by.mean()) ** 2)
    r2_km   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if r2_km < min_r2:
        return None

    return T0_f, A_f, K_f, n_f, r2_km


def compute_dstar(df, min_bins=3, flat_range=1.0):
    """d* — half-attenuation distance (mm), completely model-free.

    Defined as the source-boundary distance at which KM tau50 reaches the
    midpoint between the near-source minimum (first bin) and far-field
    plateau (max observed bin).  Estimated by linear interpolation between
    adjacent KM bins — no functional form assumed.

    Flat trials (KM tau50 range < flat_range min) are assigned d* = 0:
    no detectable gradient → characteristic reach is zero.

    Returns d* ≥ 0 or None if insufficient bins.
    """
    bx, by = km_tau50_per_bin(df)
    if len(bx) < min_bins:
        return None
    tau_range = float(by.max() - by.min())
    if tau_range < flat_range:
        return 0.0
    T0     = float(by[0])
    T_inf  = float(by.max())
    mid    = (T0 + T_inf) / 2.0
    if mid <= by[0]:
        return float(bx[0])
    if mid >= by[-1]:
        return float(bx[-1])
    for i in range(len(by) - 1):
        if by[i] <= mid <= by[i + 1]:
            span = by[i + 1] - by[i]
            frac = (mid - by[i]) / span if span != 0 else 0.0
            return float(bx[i] + frac * (bx[i + 1] - bx[i]))
    return float(bx[-1])


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=TICK_SIZE, pad=2)


def _save(fig, stem):
    for ext in ('.svg', '.pdf', '.png'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        # PDF/SVG: rasterized scatter layers embedded at 600 dpi;
        # vector elements (axes, lines, text) remain crisp.
        # PNG: 300 dpi is sufficient for screen/web use.
        kw['dpi'] = 300 if ext == '.png' else 600
        fig.savefig(OUT / f'{stem}{ext}', **kw)
    plt.close(fig)
    print(f'Saved → {OUT}/{stem}.*')


EXAMPLES = [
    ('agar.1',  'Agar',  'Agar\n(control)'),
    ('1to1.3',  '1:1',   '1:1 NaCl\n(moderate)'),
    ('2to1.5',  '2:1',   '2:1 NaCl\n(strong)'),
    ('Green.5', 'Green', 'Green mold\n(fungal)'),
]
EXAMPLE_IDS = {tid for tid, _, _ in EXAMPLES}

BIN_MM  = 0.2
MIN_BIN = 15

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(4 * 58 * MM, 50 * MM + 68 * MM))
gs  = gridspec.GridSpec(2, 4, figure=fig,
                        height_ratios=[1, 1.25],
                        hspace=0.60, wspace=0.50,
                        left=0.08, right=0.98, top=0.95, bottom=0.08)

ax_top  = [fig.add_subplot(gs[0, i]) for i in range(4)]
ax_agg  = fig.add_subplot(gs[1, :])

example_betas = {}

for ax, (tid, grp, label) in zip(ax_top, EXAMPLES):
    df    = load_trial(tid)
    c     = COLORS[grp]
    ec    = EDGE[grp]
    delta = delta_map.get(tid, float('nan'))

    # Raw scatter — log-log requires d > 0 and tau > 0
    mask = (df['d_mm'] > 0) & (df['tau_fwd_min'] > 0)
    ax.scatter(df.loc[mask, 'd_mm'], df.loc[mask, 'tau_fwd_min'], s=2.5, color=c,
               alpha=0.15, edgecolors='none', rasterized=True, zorder=2)

    # KM tau50 per bin — open circles
    bx, by = km_tau50_per_bin(df)
    if len(bx) >= 2:
        ax.plot(bx, by, 'o', color=ec, ms=3.5,
                markerfacecolor='white', markeredgewidth=0.8, zorder=4)

    # Hill curve — visualization only (smooth interpolant through KM dots)
    fit   = fit_hill(df)
    dstar = compute_dstar(df)
    if fit is not None:
        T0_f, A_f, K_f, n_f, r2_km = fit
        xfit = np.exp(np.linspace(np.log(bx[0]), np.log(bx[-1]), 300))
        ax.plot(xfit, _hill(xfit, T0_f, A_f, K_f, n_f),
                color=ec, lw=1.3, zorder=5)
    if dstar is not None and dstar > 0:
        example_betas[tid] = dstar
        ax.text(0.97, 0.94, f'$d^*$ = {dstar:.2f} mm',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=TICK_SIZE - 0.5, color=ec, fontweight='bold')
    else:
        ax.text(0.97, 0.94, 'no gradient',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=TICK_SIZE - 0.5, color='#888888')

    ax.text(0.97, 0.06, f'$\\delta$ = {delta:.0f} µm',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=TICK_SIZE - 0.5, color='#555555')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(label, fontsize=LABEL_SIZE, color=c, pad=3, linespacing=1.3)
    ax.set_xlabel('Distance (mm)', fontsize=LABEL_SIZE, labelpad=2)
    if ax is ax_top[0]:
        ax.set_ylabel('Lifetime (min)', fontsize=LABEL_SIZE, labelpad=2)
    style_ax(ax)

GROUP_ORDER_AGG = ['Agar', '0.5:1', '1:1', '2:1', 'Green', 'White', 'Black']
HG_GROUPS = {'Agar', '0.5:1', '1:1', '2:1'}
MARKER = {'Agar':'o','0.5:1':'o','1:1':'o','2:1':'o',
          'Green':'D','White':'D','Black':'D'}

groups = {g: {'x': [], 'y': []} for g in GROUP_ORDER_AGG}

for tid, grp in ALL_TRIALS.items():
    delta = delta_map.get(tid)
    if delta is None or not np.isfinite(delta):
        continue
    df = load_trial(tid)
    if df is None or len(df) < 50:
        continue
    dstar = compute_dstar(df)
    if dstar is not None:
        groups[grp]['x'].append(delta)
        groups[grp]['y'].append(dstar)

# Background individual trial dots
for grp in GROUP_ORDER_AGG:
    gd = groups[grp]
    if not gd['x']:
        continue
    ax_agg.scatter(gd['x'], gd['y'], marker=MARKER[grp], s=14,
                   color=COLORS[grp], alpha=0.25, edgecolors='none', zorder=3)

# Group means ± SEM
for grp in GROUP_ORDER_AGG:
    gd = groups[grp]
    if len(gd['x']) < 2:
        continue
    from scipy.stats import sem as _sem
    xm, ym = np.mean(gd['x']), np.mean(gd['y'])
    xe, ye = _sem(gd['x']),    _sem(gd['y'])
    fmt = 'o' if grp in HG_GROUPS else 'D'
    ms  = 5.5 if grp in HG_GROUPS else 4.5
    mec = 'white' if grp in HG_GROUPS else EDGE[grp]
    mew = 0.4 if grp in HG_GROUPS else 0.5
    ax_agg.errorbar(xm, ym, xerr=xe, yerr=ye,
                    fmt=fmt, color=COLORS[grp], markersize=ms,
                    markeredgecolor=mec, markeredgewidth=mew,
                    capsize=2.5, capthick=LW, elinewidth=LW,
                    ecolor=EDGE[grp], label=grp, zorder=5)

# Regression line — linear fit (K=0 trials included)
xa = np.array([v for g in GROUP_ORDER_AGG for v in groups[g]['x']])
ya = np.array([v for g in GROUP_ORDER_AGG for v in groups[g]['y']])
valid = np.isfinite(ya)
if valid.sum() >= 3:
    res = stats.linregress(xa[valid], ya[valid])
    r2  = res.rvalue ** 2
    xfit = np.linspace(0, 1100, 300)
    ax_agg.plot(xfit, res.intercept + res.slope * xfit,
                '-', color='#333333', lw=0.9, alpha=0.7, zorder=2)
    ax_agg.text(0.97, 0.08, f'$R^2$ = {r2:.2f}',
                transform=ax_agg.transAxes, ha='right', va='bottom',
                fontsize=TICK_SIZE - 0.5, color='#333333')

# Highlight the 4 example trials with larger outlined markers + leader labels
EXAMPLE_LABEL_OFFSET = {
    'agar.1':  (-60,  0.05),
    '1to1.3':  (-60,  0.08),
    '2to1.5':  (-80,  0.10),
    'Green.5': (-60, -0.10),
}
for tid, grp, label in EXAMPLES:
    delta = delta_map.get(tid)
    beta  = example_betas.get(tid)
    if delta is None or beta is None:
        continue
    c  = COLORS[grp]
    ec = EDGE[grp]
    mk = MARKER[grp]
    ax_agg.scatter([delta], [beta], marker=mk, s=60, color=c,
                   edgecolors=ec, linewidths=1.0, zorder=7)
    short = label.replace('\n', ' ')
    dx, dy = EXAMPLE_LABEL_OFFSET.get(tid, (-50, 0.04))
    ax_agg.annotate(short,
                    xy=(delta, beta),
                    xytext=(delta + dx, beta + dy),
                    fontsize=TICK_SIZE - 1.5, color=ec,
                    arrowprops=dict(arrowstyle='-', color=ec,
                                   lw=0.5, shrinkB=3),
                    ha='right')

ax_agg.axhline(0, color='#aaaaaa', lw=0.4, ls=':', zorder=0)
ax_agg.set_xlim(0, 1100)
ax_agg.set_xlabel(r'$\delta$ (µm)', fontsize=LABEL_SIZE, labelpad=3)
ax_agg.set_ylabel(r'$d^*$ (mm)', fontsize=LABEL_SIZE, labelpad=3)

handles, labels_l = ax_agg.get_legend_handles_labels()
by_label = dict(zip(labels_l, handles))
ordered  = [(by_label[l], l) for l in GROUP_ORDER_AGG if l in by_label]
if ordered:
    ax_agg.legend(*zip(*ordered), fontsize=TICK_SIZE - 0.5,
                  loc='upper left', frameon=False,
                  labelspacing=0.3, handlelength=1.2,
                  handletextpad=0.4, ncol=2, columnspacing=0.8)
style_ax(ax_agg)

_save(fig, 'panel_D_examples')


N_ROWS, N_COLS = 5, 7
fig, axes = plt.subplots(N_ROWS, N_COLS,
                         figsize=(N_COLS * 36 * MM, N_ROWS * 34 * MM))
fig.subplots_adjust(hspace=0.65, wspace=0.50,
                    left=0.06, right=0.98, top=0.94, bottom=0.08)

for col_idx, grp in enumerate(GROUP_ORDER):
    grp_trials = [(tid, g) for tid, g in ALL_TRIALS.items() if g == grp]
    c  = COLORS[grp]
    ec = EDGE[grp]

    # Column header
    fig.text((col_idx + 0.5) / N_COLS, 0.97, grp,
             ha='center', va='top', fontsize=TICK_SIZE, color=c,
             fontweight='bold',
             transform=fig.transFigure)

    for row_idx, (tid, _) in enumerate(grp_trials):
        ax    = axes[row_idx][col_idx]
        df    = load_trial(tid)
        delta = delta_map.get(tid, float('nan'))

        if df is not None and len(df) >= 10:
            mask = (df['d_mm'] > 0) & (df['tau_fwd_min'] > 0)
            ax.scatter(df.loc[mask, 'd_mm'], df.loc[mask, 'tau_fwd_min'],
                       s=1.0, color=c,
                       alpha=0.15, edgecolors='none', rasterized=True, zorder=2)
            bxs, bys = km_tau50_per_bin(df, min_per_bin=10)
            if len(bxs) >= 2:
                ax.plot(bxs, bys, 'o', color=ec, ms=2.0,
                        markerfacecolor='white', markeredgewidth=0.5, zorder=4)
            fit_s = fit_hill(df, min_bins=4)
            dstar_s = compute_dstar(df)
            if fit_s is not None and len(bxs) >= 2:
                T0_s, A_s, K_s, n_s, _ = fit_s
                xfit_s = np.exp(np.linspace(np.log(bxs[0]), np.log(bxs[-1]), 100))
                ax.plot(xfit_s, _hill(xfit_s, T0_s, A_s, K_s, n_s),
                        color=ec, lw=0.9, zorder=3)
            if dstar_s is not None and dstar_s > 0:
                ax.text(0.97, 0.04, f'$d^*$={dstar_s:.2f}',
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=5.0, color=ec)
            ax.set_xscale('log')
            ax.set_yscale('log')

        trial_num = tid.split('.')[-1]
        ax.set_title(f'#{trial_num}  δ={delta:.0f}', fontsize=5.5,
                     color='#444444', pad=2)
        ax.tick_params(labelsize=5.0, pad=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if row_idx == N_ROWS - 1:
            ax.set_xlabel('d (mm)', fontsize=5.5, labelpad=1)
        if col_idx == 0:
            ax.set_ylabel('τ (min)', fontsize=5.5, labelpad=1)

_save(fig, 'supp_all_trials')


KM_EXAMPLES = [
    ('agar.1',  'Agar',  'Agar (control)'),
    ('1to1.3',  '1:1',   '1:1 NaCl'),
    ('2to1.3',  '2:1',   '2:1 NaCl'),
    ('Green.5', 'Green', 'Green mold'),
]

NEAR_MAX = 0.5   # mm  (< 0.5 mm = near zone)
FAR_MIN  = 1.0   # mm  (> 1.0 mm = far zone)

fig, axes = plt.subplots(1, 4, figsize=(4 * 58 * MM, 52 * MM))
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.22, wspace=0.50)

kmf = KaplanMeierFitter()

for ax, (tid, grp, label) in zip(axes, KM_EXAMPLES):
    df  = load_trial(tid)
    c   = COLORS[grp]
    ec  = EDGE[grp]

    near = df[df['d_mm'] <  NEAR_MAX].copy()
    far  = df[df['d_mm'] >= FAR_MIN].copy()

    for zone_df, zone_label, ls, alpha in [
        (near, f'Near  (<{NEAR_MAX} mm)', '--', 0.65),
        (far,  f'Far  (>{FAR_MIN} mm)',   '-',  1.00),
    ]:
        if len(zone_df) < 10:
            continue
        kmf.fit(zone_df['tau_fwd_min'],
                event_observed=~zone_df['censored'],
                label=zone_label)
        t = kmf.survival_function_.index.values
        s = kmf.survival_function_.iloc[:, 0].values
        ax.step(t, s, where='post', color=c, lw=1.2, ls=ls, alpha=alpha,
                label=zone_label)
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            ax.plot([t50, t50], [0, 0.5], color=c, lw=0.5, ls=ls, alpha=0.45)
            ax.plot([0, t50],   [0.5, 0.5], color=c, lw=0.5, ls=':', alpha=0.35)

    ax.axhline(0.5, color='#cccccc', lw=0.4, ls=':', zorder=0)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(label, fontsize=LABEL_SIZE, color=c, pad=3)
    ax.set_xlabel('Forward lifetime (min)', fontsize=LABEL_SIZE, labelpad=2)
    if ax is axes[0]:
        ax.set_ylabel('Survival probability', fontsize=LABEL_SIZE, labelpad=2)
    ax.legend(fontsize=TICK_SIZE - 1.5, frameon=False, loc='upper right',
              handlelength=1.5, labelspacing=0.25)
    style_ax(ax)

_save(fig, 'supp_km_zones')

print('Done.')
