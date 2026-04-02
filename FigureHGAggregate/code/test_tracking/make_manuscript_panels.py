#!/usr/bin/env python3
"""Generate manuscript Figure 2 panels I, J, K, L from tracked droplet data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import KaplanMeierFitter
from scipy import stats
from scipy.optimize import curve_fit

THIS_DIR    = Path(__file__).parent
TRACK_OUT   = THIS_DIR / 'output'                          # tracked CSVs
FIG_OUT     = THIS_DIR.parent.parent / 'output'             # figure output
HG_METRICS  = THIS_DIR.parent.parent / 'output' / 'hydrogel_metrics.csv'

HG_TRIALS = {
    'agar.1': 'Agar', 'agar.2': 'Agar', 'agar.3': 'Agar',
    'agar.4': 'Agar', 'agar.5': 'Agar',
    '0.5to1.2': '0.5:1', '0.5to1.3': '0.5:1', '0.5to1.4': '0.5:1',
    '0.5to1.5': '0.5:1', '0.5to1.7': '0.5:1',
    '1to1.1': '1:1', '1to1.2': '1:1', '1to1.3': '1:1',
    '1to1.4': '1:1', '1to1.5': '1:1',
    '2to1.1': '2:1', '2to1.2': '2:1', '2to1.3': '2:1',
    '2to1.4': '2:1', '2to1.5': '2:1',
}
GROUP_ORDER = ['Agar', '0.5:1', '1:1', '2:1']

MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0
LW = 0.6
COLORS = {'Agar': '#3A9E6F', '0.5:1': '#E67E22', '1:1': '#5B8FC9', '2:1': '#C0392B'}
SURV_DIST_UM  = [900, 1500, 2100, 2900]
SURV_COLORS   = ['#1a237e', '#7b1fa2', '#e65100', '#f0a500']

MIN_FRAMES = 3
DIST_BIN = 200  # µm

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


def style_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TS)


def load_trial(trial_id):
    path = TRACK_OUT / f'{trial_id}_track_histories.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['lifetime_min'] = df['lifetime_s'] / 60.0
    return df



def _tau50_bins(df, time_col):
    d_max = df['distance_um'].max()
    bins  = np.arange(0, d_max + DIST_BIN, DIST_BIN)
    df    = df.copy()
    df['db'] = pd.cut(df['distance_um'], bins=bins, labels=False)
    kmf = KaplanMeierFitter()
    d_vals, tau_vals = [], []
    for b in sorted(df['db'].dropna().unique()):
        sub = df[df['db'] == b]
        if len(sub) < 15:
            continue
        center = bins[int(b)] + DIST_BIN / 2
        kmf.fit(sub[time_col], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            d_vals.append(center / 1000.0)
            tau_vals.append(t50)
    return np.array(d_vals), np.array(tau_vals)


def tracked_tau50_profile(trial_id):
    return tracked_tau50_profile_fwd(trial_id)


def tracked_tau50_profile_fwd(trial_id):
    df = load_trial(trial_id)
    if df is None or len(df) < 15:
        return None, None
    df = df.copy()
    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[df['tau_fwd_min'] > 0]
    if len(df) < 15:
        return None, None
    return _tau50_bins(df, 'tau_fwd_min')


T_SEED = 900  # seed frame ≈ 15 min (start of evaporation phase)


def _hill(d, T0, A, K, n):
    dn = np.power(np.maximum(d, 0), n)
    Kn = np.power(K, n)
    return T0 + A * dn / (Kn + dn)


def _fit_hill_on_profile(bx, by, min_bins=5, flat_range=1.0):
    if len(bx) < min_bins:
        return None, None
    if float(by.max() - by.min()) < flat_range:
        return 0.0, None
    T0_g = float(by[0])
    A_g  = max(float(by[-1] - by[0]), 0.1)
    lo   = [0, 0, 0.01, 0.5]
    hi   = [by.max(), by.max() * 3, bx.max() * 3, 8.0]
    try:
        popt, _ = curve_fit(_hill, bx, by, p0=[T0_g, A_g, np.median(bx), 2.0],
                            bounds=(lo, hi), max_nfev=50000, method='trf')
    except Exception:
        return None, None
    ss_res = np.sum((by - _hill(bx, *popt)) ** 2)
    ss_tot = np.sum((by - by.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if r2 < 0.80:
        return None, None
    return float(popt[2]), popt


def fit_hill_dstar(trial_id, min_bins=5, min_r2=0.80, flat_range=1.0):
    d_mm, tau_min = tracked_tau50_profile_fwd(trial_id)
    if d_mm is None or len(d_mm) < min_bins:
        return None, None
    bx, by = np.array(d_mm), np.array(tau_min)
    if float(by.max() - by.min()) < flat_range:
        return 0.0, None
    T0_g = float(by[0])
    A_g  = max(float(by[-1] - by[0]), 0.1)
    lo   = [0, 0, 0.01, 0.5]
    hi   = [by.max(), by.max() * 3, bx.max() * 3, 8.0]
    try:
        popt, _ = curve_fit(_hill, bx, by, p0=[T0_g, A_g, np.median(bx), 2.0],
                            bounds=(lo, hi), max_nfev=50000, method='trf')
    except Exception:
        return None, None
    by_pred = _hill(bx, *popt)
    ss_res  = np.sum((by - by_pred) ** 2)
    ss_tot  = np.sum((by - by.mean()) ** 2)
    r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if r2 < min_r2:
        return None, None
    return float(popt[2]), popt


SURV_TRIALS_2to1 = ['2to1.1', '2to1.2', '2to1.3', '2to1.4', '2to1.5']


def make_panel_I():
    frames = [load_trial(t) for t in SURV_TRIALS_2to1]
    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    if df.empty:
        print('Panel I: no data')
        return

    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.16)

    kmf = KaplanMeierFitter()
    legend_handles = []
    legend_labels  = []

    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[df['tau_fwd_min'] > 0].copy()

    for d_um, color in zip(SURV_DIST_UM, SURV_COLORS):
        sub = df[np.abs(df['distance_um'] - d_um) <= DIST_BIN / 2].copy()
        if len(sub) < 10:
            continue
        kmf.fit(sub['tau_fwd_min'], event_observed=~sub['censored'])
        sf  = kmf.survival_function_
        ci  = kmf.confidence_interval_
        t   = sf.index.values
        s   = sf.iloc[:, 0].values
        lo  = ci.iloc[:, 0].values
        hi  = ci.iloc[:, 1].values

        line, = ax.plot(t, s, color=color, lw=1.4, drawstyle='steps-post', zorder=3)
        ax.fill_between(t, lo, hi, color=color, alpha=0.08, linewidth=0,
                        step='post', zorder=2)
        legend_handles.append(line)
        legend_labels.append(f'{d_um / 1000:.1f} mm')

        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            ax.plot(t50, 0.5, 'o', color=color, ms=4.5,
                    markeredgecolor='white', markeredgewidth=0.4, zorder=5)

    ax.axhline(0.5, color='gray', ls='--', lw=0.7, alpha=0.7, zorder=1)
    ax.text(0.04, 0.52, r'$\tau_{50}$', transform=ax.transAxes,
            color='gray', fontsize=TS + 1.5, fontstyle='italic', va='bottom', ha='left')

    ax.set_xlabel(r'$\tau$ (min)', fontsize=LS, labelpad=3)
    ax.set_ylabel('Fraction surviving', fontsize=LS, labelpad=3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    style_ax(ax)

    ax.legend(legend_handles, legend_labels,
              title='Distance from source', title_fontsize=TS - 0.5,
              fontsize=TS - 0.5, loc='upper right', framealpha=0.0,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.3)
    ax.text(-0.20, 1.05, 'I', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')

    for ext in ('.svg', '.pdf', '.png'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        kw['dpi'] = 600 if ext != '.png' else 300
        fig.savefig(FIG_OUT / f'panel_I{ext}', **kw)
    plt.close(fig)
    print(f'Panel I → {FIG_OUT}/panel_I.*')


J_EXEMPLARS = [
    ('agar.3',   'Agar',  'Agar'),
    ('0.5to1.3', '0.5:1', '0.5:1 NaCl'),
    ('1to1.3',   '1:1',   '1:1 NaCl'),
    ('2to1.3',   '2:1',   '2:1 NaCl'),
]
J_EDGE = {'Agar': '#2E7D32', '0.5:1': '#B7560A', '1:1': '#2C5F9F', '2:1': '#922B21'}


def plot_panel_J(ax):
    for tid, grp, label in J_EXEMPLARS:
        color = COLORS[grp]
        edge  = J_EDGE[grp]

        df = load_trial(tid)
        if df is not None:
            df = df.copy()
            df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
            df = df[df['tau_fwd_min'] > 0]
            mask = df['distance_um'] > 0
            ax.scatter(df.loc[mask, 'distance_um'].values / 1000.0,
                       df.loc[mask, 'tau_fwd_min'].values,
                       s=1.5, color=color, alpha=0.10,
                       edgecolors='none', rasterized=True, zorder=1)

        d_mm, tau_min = tracked_tau50_profile(tid)
        if d_mm is None or len(d_mm) < 3:
            continue
        ax.plot(d_mm, tau_min, 'o', ms=3.5,
                markerfacecolor='white', markeredgecolor=edge,
                markeredgewidth=0.9, zorder=4, label=label)

        K, popt = fit_hill_dstar(tid)
        if popt is not None:
            xfit = np.linspace(d_mm[0], d_mm[-1], 300)
            ax.plot(xfit, _hill(xfit, *popt), '-', color=edge, lw=1.4, zorder=5)
        if K is not None and K > 0:
            ax.axvline(K, color=edge, ls='--', lw=0.7, alpha=0.6, zorder=3)
            ax.text(K, -0.08, r'$d^*$', transform=ax.get_xaxis_transform(),
                    color=edge, fontsize=TS, fontstyle='italic',
                    ha='center', va='top')

        if grp == '2:1':
            for d_um, sc in zip(SURV_DIST_UM, SURV_COLORS):
                d_pt = d_um / 1000.0
                if d_pt < d_mm[0] or d_pt > d_mm[-1]:
                    continue
                t_val = tau_min[np.argmin(np.abs(d_mm - d_pt))]
                ax.plot(d_pt, t_val, 'o', color=sc, ms=5.5,
                        markeredgecolor='white', markeredgewidth=0.5,
                        zorder=7, label=f'{d_pt:.1f} mm')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=TS - 0.5, loc='upper left',
              framealpha=0.0, handletextpad=0.3, borderpad=0.3,
              labelspacing=0.3)
    ax.set_xlabel('Distance from source (mm)', fontsize=LS, labelpad=3)
    ax.set_ylabel(r'$\tau_{50}$ (min)', fontsize=LS, labelpad=3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    style_ax(ax)
    ax.set_box_aspect(1)
    ax.text(-0.18, 1.05, 'J', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')


K_NEAR_MAX_MM = 0.5   # 0-500 µm (canonical, matches size gradient)
K_MID_LO_MM   = 0.9;  K_MID_HI_MM = 1.1   # 900-1100 µm
K_FAR_LO_MM   = 1.9;  K_FAR_HI_MM = 2.1   # 1900-2100 µm


def compute_tau_zone_metric(trial_id):
    """(tau50_far - tau50_near) / tau50_mid. Empty near zone => tau_near = 0."""
    d_mm, tau_min = tracked_tau50_profile_fwd(trial_id)
    if d_mm is None or len(d_mm) < 5:
        return None
    near_vals = tau_min[d_mm <= K_NEAR_MAX_MM]
    mid_vals  = tau_min[(d_mm >= K_MID_LO_MM) & (d_mm <= K_MID_HI_MM)]
    far_vals  = tau_min[(d_mm >= K_FAR_LO_MM) & (d_mm <= K_FAR_HI_MM)]
    if len(mid_vals) < 1 or len(far_vals) < 1:
        return None
    tau_near = near_vals.mean() if len(near_vals) >= 1 else 0.0
    tau_mid  = mid_vals.mean()
    if tau_mid <= 0:
        return None
    return (far_vals.mean() - tau_near) / tau_mid


def plot_panel_K(ax):
    hg = pd.read_csv(HG_METRICS)
    delta_map = dict(zip(hg['trial_id'], hg['delta_um']))
    rng = np.random.default_rng(42)

    x_all, y_all = [], []

    for tid, grp in HG_TRIALS.items():
        if tid not in delta_map:
            continue
        zm = compute_tau_zone_metric(tid)
        if zm is None:
            continue
        delta = delta_map[tid]
        x_all.append(delta)
        y_all.append(zm)
        jitter = rng.uniform(-10, 10)
        ax.scatter(delta + jitter, zm, c=COLORS[grp], s=35, alpha=0.65,
                   edgecolors='white', linewidths=0.4, zorder=3, label=grp)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3)

    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() >= 3:
        sl, ic, r, p, se = stats.linregress(x_arr[valid], y_arr[valid])
        r2 = r ** 2
        xfit = np.linspace(0, 1100, 200)
        ax.plot(xfit, ic + sl * xfit, 'k--', lw=1.2, alpha=0.7, zorder=2)
        ax.text(0.95, 0.08, f'$R^2$ = {r2:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)
        print(f'  Panel K (zone metric): R²={r2:.3f}, slope={sl:.6f}, p={p:.2e}')

    ax.set_xlabel(r'$\delta$ (µm)', fontsize=LS, labelpad=3)
    ax.set_ylabel(
        r'$(\tau_{50,\mathrm{far}} - \tau_{50,\mathrm{near}})\,/\,\tau_{50,\mathrm{mid}}$',
        fontsize=LS, labelpad=3)
    ax.set_xlim(0, 1100)
    ax.set_ylim(bottom=0)
    style_ax(ax)
    ax.set_box_aspect(1)
    ax.text(-0.18, 1.05, 'K', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')


def make_panels_JK():
    fig, (ax_j, ax_k) = plt.subplots(1, 2, figsize=(170 * MM, 75 * MM))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.17,
                        wspace=0.38)
    plot_panel_J(ax_j)
    plot_panel_K(ax_k)

    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(FIG_OUT / f'panels_JK{ext}',
                    bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)
    print(f'Panels J+K → {FIG_OUT}/panels_JK.*')



def make_panel_L():
    hg = pd.read_csv(HG_METRICS)
    delta_map = dict(zip(hg['trial_id'], hg['delta_um']))
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.16)

    x_all, y_all = [], []

    for tid, grp in HG_TRIALS.items():
        if tid not in delta_map:
            continue
        dstar, _ = fit_hill_dstar(tid)
        if dstar is None:
            continue
        delta = delta_map[tid]
        x_all.append(delta)
        y_all.append(dstar)
        jitter = rng.uniform(-10, 10)
        ax.scatter(delta + jitter, dstar, c=COLORS[grp], s=35, alpha=0.65,
                   edgecolors='white', linewidths=0.4, zorder=3, label=grp)
        print(f'  L  {tid:>10s}  {grp:>5s}  δ={delta:.0f} µm  d*={dstar:.3f} mm')

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3)

    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    valid = ~np.isnan(y_arr)
    if valid.sum() >= 3:
        sl, ic, r, p, se = stats.linregress(x_arr[valid], y_arr[valid])
        r2 = r ** 2
        xfit = np.linspace(0, 1100, 100)
        ax.plot(xfit, ic + sl * xfit, 'k--', lw=1.2, alpha=0.7, zorder=2)
        ax.text(0.95, 0.08, f'$R^2$ = {r2:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)
        print(f'  Panel L (d* vs δ): R²={r2:.3f}, slope={sl:.6f}, p={p:.2e}')

    ax.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.set_xlabel(r'$\delta$ (µm)', fontsize=LS, labelpad=3)
    ax.set_ylabel(r'$d^*$ (mm)', fontsize=LS, labelpad=3)
    ax.set_xlim(0, 1100)
    style_ax(ax)
    ax.set_box_aspect(1)
    ax.text(-0.18, 1.05, 'L', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')

    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(FIG_OUT / f'panel_L{ext}',
                    bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)
    print(f'  Panel L → {FIG_OUT}/panel_L.*')



def make_panel_J():
    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.16)
    plot_panel_J(ax)
    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(FIG_OUT / f'panel_J{ext}',
                    bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)
    print(f'Panel J → {FIG_OUT}/panel_J.*')


def main():
    print('=== Generating manuscript panels (tracked) ===\n')
    make_panel_I()
    make_panel_J()
    make_panels_JK()
    make_panel_L()
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
