#!/usr/bin/env python3
"""Universal mechanism panels: survival gradient vs drying power, size gradient, d2-law."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
from lifelines import KaplanMeierFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS_DIR    = Path(__file__).parent
OUTPUT_DIR  = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT   = THIS_DIR.resolve().parent.parent
TRACK_DIR   = REPO_ROOT / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
HG_METRICS  = REPO_ROOT / 'FigureHGAggregate' / 'output' / 'hydrogel_metrics.csv'
F_METRICS   = OUTPUT_DIR / 'fungi_metrics.csv'

ALL_TRIALS = {
    'agar.1': 'Agar', 'agar.2': 'Agar', 'agar.3': 'Agar',
    'agar.4': 'Agar', 'agar.5': 'Agar',
    '0.5to1.2': '0.5:1', '0.5to1.3': '0.5:1', '0.5to1.4': '0.5:1',
    '0.5to1.5': '0.5:1', '0.5to1.7': '0.5:1',
    '1to1.1': '1:1', '1to1.2': '1:1', '1to1.3': '1:1',
    '1to1.4': '1:1', '1to1.5': '1:1',
    '2to1.1': '2:1', '2to1.2': '2:1', '2to1.3': '2:1',
    '2to1.4': '2:1', '2to1.5': '2:1',
    'Green.1': 'Green', 'Green.2': 'Green', 'Green.3': 'Green',
    'Green.4': 'Green', 'Green.5': 'Green',
    'white.1': 'White', 'white.2': 'White', 'white.3': 'White',
    'white.4': 'White', 'white.5': 'White',
    'black.1': 'Black', 'black.2': 'Black', 'black.3': 'Black',
    'black.4': 'Black', 'black.5': 'Black',
}

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

MM         = 1 / 25.4
TICK_SIZE  = 7.0
LABEL_SIZE = 8.5
PANEL_LBL  = 12.0
LW         = 0.6
LW_DATA    = 0.8

MIN_FRAMES = 3
DIST_BIN   = 200       # µm
T_SEED     = 900       # s (15 min)

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


def load_trial(trial_id):
    path = TRACK_DIR / f'{trial_id}_track_histories.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[df['tau_fwd_min'] > 0].copy()
    return df


def tau50_profile(trial_id, size_range=None, min_per_bin=15):
    df = load_trial(trial_id)
    if df is None:
        return None, None
    if size_range is not None:
        df = df.dropna(subset=['R_eq_seed']).copy()
        df = df[(df['R_eq_seed'] >= size_range[0]) &
                (df['R_eq_seed'] <= size_range[1])].copy()
        if len(df) < 30:
            return None, None
    d_max = df['distance_um'].max()
    bins = np.arange(0, d_max + DIST_BIN, DIST_BIN)
    df['db'] = pd.cut(df['distance_um'], bins=bins, labels=False)
    kmf = KaplanMeierFitter()
    d_vals, tau_vals = [], []
    for b in sorted(df['db'].dropna().unique()):
        sub = df[df['db'] == b]
        if len(sub) < min_per_bin:
            continue
        center = bins[int(b)] + DIST_BIN / 2
        kmf.fit(sub['tau_fwd_min'], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            d_vals.append(center / 1000.0)      # mm
            tau_vals.append(t50)
    if not d_vals:
        return None, None
    return np.array(d_vals), np.array(tau_vals)


def get_iqr_band(trial_id):
    df = load_trial(trial_id)
    if df is None:
        return None
    r = df['R_eq_seed'].dropna()
    r = r[r > 0]
    if len(r) < 30:
        return None
    return (r.quantile(0.25), r.quantile(0.75))


def get_delta_map():
    delta = {}
    hg = pd.read_csv(HG_METRICS)
    delta.update(dict(zip(hg['trial_id'], hg['delta_um'])))
    fm = pd.read_csv(F_METRICS)
    delta.update(dict(zip(fm['trial_id'], fm['delta_um'])))
    return delta


def _dedup_legend(ax, loc='upper left', ncol=2, extra_handles=None):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_h, ordered_l = [], []
    for l in GROUP_ORDER:
        if l in by_label:
            ordered_h.append(by_label[l])
            ordered_l.append(l)
    if extra_handles:
        for h, l in extra_handles:
            ordered_h.append(h)
            ordered_l.append(l)
    if not ordered_l:
        return
    ax.legend(ordered_h, ordered_l,
              fontsize=TICK_SIZE - 0.5, loc=loc, frameon=False,
              labelspacing=0.3, handlelength=1.2, handletextpad=0.4,
              ncol=ncol, columnspacing=0.8)


def _km_tau50_per_bin(df, bin_mm=0.2, min_per_bin=15):
    kmf  = KaplanMeierFitter()
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


def _hill(d, T0, A, K, n):
    dn = np.power(np.maximum(d, 0), n)
    Kn = np.power(K, n)
    return T0 + A * dn / (Kn + dn)


def fit_dstar(trial_id, min_bins=5, min_r2=0.80, flat_range=1.0):
    """Fit Hill function to KM tau50 per bin; flat trials -> d*=0."""
    df = load_trial(trial_id)
    if df is None:
        return None
    df = df.copy()
    df['d_mm'] = df['distance_um'] / 1000.0

    bx, by = _km_tau50_per_bin(df)
    if len(bx) < 2:
        return None
    if float(by.max() - by.min()) < flat_range:
        return 0.0
    if len(bx) < min_bins:
        return None

    T0_g = float(by[0])
    A_g  = max(float(by[-1] - by[0]), 0.1)
    K_g  = float(np.median(bx))
    lo   = [0, 0, 0.01, 0.5]
    hi   = [by.max(), by.max() * 3, bx.max() * 3, 8.0]
    try:
        popt, _ = curve_fit(_hill, bx, by, p0=[T0_g, A_g, K_g, 2.0],
                            bounds=(lo, hi), max_nfev=50000, method='trf')
    except Exception:
        return None

    by_pred = _hill(bx, *popt)
    ss_res  = np.sum((by - by_pred) ** 2)
    ss_tot  = np.sum((by - by.mean()) ** 2)
    r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    if r2 < min_r2:
        return None

    return float(popt[2])


def make_panel_G():
    delta_map = get_delta_map()

    fig, ax = plt.subplots(figsize=(75 * MM, 68 * MM))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.20)

    groups = {g: {'x': [], 'y': []} for g in GROUP_ORDER}

    for tid, grp in ALL_TRIALS.items():
        if tid not in delta_map:
            continue
        delta = delta_map[tid]
        dstar = fit_dstar(tid)
        if dstar is None:
            print(f'  D  {tid:12s}: skip')
            continue
        groups[grp]['x'].append(delta)
        groups[grp]['y'].append(dstar)
        print(f'  D  {tid:12s}: delta={delta:.0f} um, d*={dstar:.3f} mm')

    for grp in GROUP_ORDER:
        gd = groups[grp]
        if not gd['x']:
            continue
        ax.scatter(gd['x'], gd['y'], marker=MARKER[grp], s=18,
                   color=COLORS[grp], alpha=0.25, edgecolors='none', zorder=3)

    HG_GROUPS = {'Agar', '0.5:1', '1:1', '2:1'}
    for grp in GROUP_ORDER:
        gd = groups[grp]
        if len(gd['x']) < 2:
            continue
        xm, ym = np.mean(gd['x']), np.mean(gd['y'])
        xe, ye = np.std(gd['x'], ddof=1), np.std(gd['y'], ddof=1)
        fmt = 'o' if grp in HG_GROUPS else 'D'
        ms  = 6.0 if grp in HG_GROUPS else 5.0
        mec = 'white' if grp in HG_GROUPS else EDGE[grp]
        mew = 0.4 if grp in HG_GROUPS else 0.5
        ax.errorbar(xm, ym, xerr=xe, yerr=ye,
                    fmt=fmt, color=COLORS[grp], markersize=ms,
                    markeredgecolor=mec, markeredgewidth=mew,
                    capsize=2.5, capthick=LW, elinewidth=LW,
                    ecolor=EDGE[grp], label=grp, zorder=5)

    _dedup_legend(ax, loc='upper left')

    xa = np.concatenate([groups[g]['x'] for g in GROUP_ORDER])
    ya = np.concatenate([groups[g]['y'] for g in GROUP_ORDER])
    valid = np.isfinite(ya)
    xfit = np.linspace(0, 1100, 100)
    if valid.sum() >= 3:
        res = stats.linregress(xa[valid], ya[valid])
        r2  = res.rvalue ** 2
        ax.plot(xfit, res.intercept + res.slope * xfit,
                '-', color='#333333', lw=LW_DATA, alpha=0.7, zorder=2)
        ax.text(0.95, 0.08, f'$R^2$ = {r2:.2f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=TICK_SIZE - 0.5)
        print(f'\n  Panel D: R2={r2:.3f}, slope={res.slope:.5f}, n={valid.sum()}')

    ax.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_ylabel(r'$d^*$ (mm)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(0, 1100)
    ax.axhline(0, color='#aaaaaa', lw=0.4, ls=':', zorder=0)
    style_ax(ax)
    ax.text(-0.14, 1.04, 'D', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    _save(fig, 'panel_D')


H_BIN_UM   = 100
H_MIN_DROP = 3
H_T_WINDOW = (14.5, 15.5)
N_NEAR_BINS = 5           # first 5 bins (0-500 um); empty bins count as R=0
D_MID_LO    = 900;  D_MID_HI  = 1100
D_FAR_LO    = 1900; D_FAR_HI  = 2100

HG_AGG_DIR  = REPO_ROOT / 'FigureHGAggregate' / 'raw_data' / 'aggregate_edt'
F_AGG_DIR   = REPO_ROOT / 'FigureFungi' / 'raw_data' / 'aggregate_edt'


def _load_edt_droplets(trial_id):
    for d in (HG_AGG_DIR, F_AGG_DIR):
        p = d / f'{trial_id}_edt_droplets.csv'
        if p.exists():
            df = pd.read_csv(p)
            tw = df[(df['time_min'] >= H_T_WINDOW[0]) &
                    (df['time_min'] <= H_T_WINDOW[1]) &
                    (df['radius_um'] > 0)].copy()
            return tw
    return None


def _size_gradient(trial_id):
    """(R_far - R_near) / R_mid from binned means at t=15 min."""
    tw = _load_edt_droplets(trial_id)
    if tw is None or len(tw) < 30:
        return None

    max_d = tw['distance_um'].max()
    bins = np.arange(0, max_d + H_BIN_UM, H_BIN_UM)
    tw = tw.copy()
    tw['bin'] = pd.cut(tw['distance_um'], bins=bins,
                       labels=bins[:-1] + H_BIN_UM / 2).astype(float)
    grp = tw.groupby('bin')['radius_um'].mean()

    near_centers = [H_BIN_UM / 2 + i * H_BIN_UM for i in range(N_NEAR_BINS)]
    near_vals = [grp[c] if c in grp.index else 0.0 for c in near_centers]
    R_near = np.mean(near_vals)

    mid_vals = grp[(grp.index >= D_MID_LO) & (grp.index <= D_MID_HI)]
    if len(mid_vals) < 1:
        return None
    R_mid = mid_vals.mean()

    far_vals = grp[(grp.index >= D_FAR_LO) & (grp.index <= D_FAR_HI)]
    if len(far_vals) < 1:
        return None
    R_far = far_vals.mean()

    if R_mid <= 0:
        return None

    return (R_far - R_near) / R_mid


def make_panel_H():
    delta_map = get_delta_map()

    fig, ax = plt.subplots(figsize=(75 * MM, 68 * MM))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.20)

    groups = {g: {'x': [], 'y': []} for g in GROUP_ORDER}

    for tid, grp in ALL_TRIALS.items():
        if tid not in delta_map:
            continue
        result = _size_gradient(tid)
        if result is None:
            continue
        delta = delta_map[tid]
        groups[grp]['x'].append(delta)
        groups[grp]['y'].append(result)
        print(f'  H  {tid:12s}: delta={delta:.0f} um, '
              f'size_grad={result:+.3f}')

    for grp in GROUP_ORDER:
        gd = groups[grp]
        if not gd['x']:
            continue
        ax.scatter(gd['x'], gd['y'], marker=MARKER[grp], s=18,
                   color=COLORS[grp], alpha=0.25, edgecolors='none', zorder=3)

    HG_GROUPS = {'Agar', '0.5:1', '1:1', '2:1'}
    for grp in GROUP_ORDER:
        gd = groups[grp]
        if len(gd['x']) < 2:
            continue
        xm, ym = np.mean(gd['x']), np.mean(gd['y'])
        xe, ye = np.std(gd['x'], ddof=1), np.std(gd['y'], ddof=1)
        fmt = 'o' if grp in HG_GROUPS else 'D'
        ms  = 6.0 if grp in HG_GROUPS else 5.0
        mec = 'white' if grp in HG_GROUPS else EDGE[grp]
        mew = 0.4 if grp in HG_GROUPS else 0.5
        ax.errorbar(xm, ym, xerr=xe, yerr=ye,
                    fmt=fmt, color=COLORS[grp], markersize=ms,
                    markeredgecolor=mec, markeredgewidth=mew,
                    capsize=2.5, capthick=LW, elinewidth=LW,
                    ecolor=EDGE[grp], label=grp, zorder=5)

    _dedup_legend(ax, loc='upper left')

    xa = np.concatenate([groups[g]['x'] for g in GROUP_ORDER])
    ya = np.concatenate([groups[g]['y'] for g in GROUP_ORDER])
    valid = ~np.isnan(ya)
    if valid.sum() >= 3:
        res = stats.linregress(xa[valid], ya[valid])
        r2 = res.rvalue ** 2
        xfit = np.linspace(0, 1100, 100)
        ax.plot(xfit, res.intercept + res.slope * xfit,
                '-', color='#333333', lw=LW_DATA, alpha=0.7, zorder=2)
        ax.text(0.95, 0.08,
                f'$R^2$ = {r2:.2f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=TICK_SIZE - 0.5)
        print(f'\n  Panel H: R2={r2:.3f}, slope={res.slope:.6f}, '
              f'n={valid.sum()}')

    ax.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_ylabel(r'$(R_{\mathrm{far}} - R_{\mathrm{near}})\,/\,R_{\mathrm{mid}}$',
                  fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(0, 1100)
    style_ax(ax)
    ax.text(-0.14, 1.04, 'E', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    _save(fig, 'panel_E')


def main():
    print('=== Generating panels D, E (universal mechanism) ===\n')

    make_panel_G()
    print()
    make_panel_H()

    print('\nDone.')


if __name__ == '__main__':
    main()
