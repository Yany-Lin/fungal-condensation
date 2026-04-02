#!/usr/bin/env python3
"""
Per-trial Weibull AFT (Accelerated Failure Time) model: Panel D equivalent.

Model (per trial):
    log(T_fwd) = alpha + beta_d * d_mm  +  [beta_R * log(R0_um)]  +  sigma*eps
    eps ~ Gumbel  (Weibull AFT parameterization via lifelines lambda_ submodel)

beta_d > 0  =>  farther droplets have LONGER forward lifetime (survival gradient).
Mechanistic grounding: d2-law gives T = R0^2 / K(d), so
    log(T) = 2*log(R0) - log(K(d))
AFT structure exactly matches this; Cox PH does not.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from lifelines import WeibullAFTFitter

warnings.filterwarnings('ignore')

BASE       = Path(__file__).resolve().parents[2]
TRACK_DIR  = BASE / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
CALIB_DIRS = [
    BASE / 'FigureHGAggregate' / 'raw_data',
    BASE / 'FigureFungi'       / 'raw_data',
]
UNIVERSAL  = BASE / 'FigureTable' / 'output' / 'universal_metrics.csv'
OUT        = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

T_SEED     = 900.0   # s  (seed frame at 15 min)
MIN_FRAMES = 3
MIN_N      = 40      # min droplets for a stable AFT fit
MIN_EVENTS = 10      # min evaporation events (not censored)

ALL_TRIALS = {
    'agar.1':'Agar','agar.2':'Agar','agar.3':'Agar','agar.4':'Agar','agar.5':'Agar',
    '0.5to1.2':'0.5:1','0.5to1.3':'0.5:1','0.5to1.4':'0.5:1',
    '0.5to1.5':'0.5:1',
    '1to1.1':'1:1','1to1.2':'1:1','1to1.3':'1:1','1to1.4':'1:1','1to1.5':'1:1',
    '2to1.1':'2:1','2to1.2':'2:1','2to1.3':'2:1','2to1.4':'2:1','2to1.5':'2:1',
    'Green.1':'Green','Green.2':'Green','Green.3':'Green',
    'Green.4':'Green','Green.5':'Green',
    'white.1':'White','white.2':'White','white.3':'White',
    'white.4':'White','white.5':'White',
    'black.1':'Black','black.2':'Black','black.3':'Black',
    'black.4':'Black','black.5':'Black',
}

COLORS = {
    'Agar':'#3A9E6F','0.5:1':'#E67E22','1:1':'#3A6FBF','2:1':'#C0392B',
    'Green':'#4CAF50','White':'#9E9E9E','Black':'#212121',
}
EDGE = {
    'Agar':'#2E7D32','0.5:1':'#B7600A','1:1':'#2C5F9F','2:1':'#922B21',
    'Green':'#2E7D32','White':'#616161','Black':'#000000',
}
MARKER     = {'Agar':'o','0.5:1':'o','1:1':'o','2:1':'o','Green':'D','White':'D','Black':'D'}
GROUP_ORDER = ['Agar','0.5:1','1:1','2:1','Green','White','Black']
HG_GROUPS   = {'Agar','0.5:1','1:1','2:1'}

MM         = 1 / 25.4
TICK_SIZE  = 7.0
LABEL_SIZE = 8.5
PANEL_LBL  = 12.0
LW         = 0.6


def load_calib(trial_id):
    for d in CALIB_DIRS:
        p = d / trial_id / 'calibration.json'
        if p.exists():
            with open(p) as f:
                return json.load(f)['scale']['pixel_size_um']
    raise FileNotFoundError(f'No calibration: {trial_id}')


def load_trial(trial_id):
    fp = TRACK_DIR / f'{trial_id}_track_histories.csv'
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[(df['tau_fwd_min'] > 0) & (df['R_eq_seed'] > 0)].copy()
    df['d_mm'] = df['distance_um'] / 1000.0
    pix = load_calib(trial_id)
    df['R0_um']     = df['R_eq_seed'] * pix
    df['log_R0_um'] = np.log(df['R0_um'])
    df['event']     = (~df['censored']).astype(int)
    return df


def aft_betas(trial_id):
    """
    Fit Weibull AFT per trial.
    Returns (beta_total, beta_adj):
      beta_total : beta_d from distance-only AFT
      beta_adj   : beta_d from distance + log(R0_um) AFT
    beta_d > 0  =>  farther droplets live longer.
    """
    df = load_trial(trial_id)
    if df is None or len(df) < MIN_N:
        return None, None
    if df['event'].sum() < MIN_EVENTS:
        return None, None

    # Total: distance only
    fit1 = df[['tau_fwd_min', 'event', 'd_mm']].dropna()
    try:
        waft1 = WeibullAFTFitter()
        waft1.fit(fit1, duration_col='tau_fwd_min', event_col='event')
        beta_tot = float(waft1.params_.loc[('lambda_', 'd_mm')])
    except Exception as e:
        print(f'  {trial_id}: AFT total failed — {e}')
        return None, None

    # Adjusted: distance + log(R0)
    fit2 = df[['tau_fwd_min', 'event', 'd_mm', 'log_R0_um']].dropna()
    try:
        waft2 = WeibullAFTFitter()
        waft2.fit(fit2, duration_col='tau_fwd_min', event_col='event')
        beta_adj = float(waft2.params_.loc[('lambda_', 'd_mm')])
    except Exception as e:
        print(f'  {trial_id}: AFT adjusted failed — {e}')
        return beta_tot, None

    return beta_tot, beta_adj


uni = pd.read_csv(UNIVERSAL)
delta_map = dict(zip(uni['trial_id'], uni['delta_um']))

print('Per-trial Weibull AFT fits')
print(f'{"Trial":15s}  {"Grp":6s}  {"n":>5s}  {"evts":>5s}  {"beta_tot":>9s}  {"beta_adj":>9s}')
print('-' * 68)

rows       = []
groups     = {g: {'x': [], 'y': []} for g in GROUP_ORDER}
groups_adj = {g: {'x': [], 'y': []} for g in GROUP_ORDER}

for tid, grp in ALL_TRIALS.items():
    delta = delta_map.get(tid)
    if delta is None or not np.isfinite(delta):
        print(f'  {tid:15s}: no delta — skip')
        continue

    df = load_trial(tid)
    n    = len(df) if df is not None else 0
    n_ev = int(df['event'].sum()) if df is not None else 0

    beta_tot, beta_adj = aft_betas(tid)

    bt_str = f'{beta_tot:+9.4f}' if beta_tot is not None else '     skip'
    ba_str = f'{beta_adj:+9.4f}' if beta_adj is not None else '     skip'
    print(f'  {tid:15s}  {grp:6s}  {n:5d}  {n_ev:5d}  {bt_str}  {ba_str}')

    if beta_tot is not None:
        groups[grp]['x'].append(delta)
        groups[grp]['y'].append(beta_tot)

    if beta_adj is not None:
        groups_adj[grp]['x'].append(delta)
        groups_adj[grp]['y'].append(beta_adj)

    rows.append({
        'trial_id': tid, 'group': grp, 'delta_um': delta,
        'beta_total': beta_tot, 'beta_adj': beta_adj,
        'n': n, 'n_events': n_ev,
    })

res_df = pd.DataFrame(rows)
res_df.to_csv(OUT / 'aft_betas.csv', index=False)

xa    = np.array([v for g in GROUP_ORDER for v in groups[g]['x']])
ya    = np.array([v for g in GROUP_ORDER for v in groups[g]['y']])
xa2   = np.array([v for g in GROUP_ORDER for v in groups_adj[g]['x']])
ya2   = np.array([v for g in GROUP_ORDER for v in groups_adj[g]['y']])

valid  = np.isfinite(ya)
valid2 = np.isfinite(ya2)

res    = stats.linregress(xa[valid],   ya[valid])
r2     = res.rvalue ** 2

print(f'\nTotal regression:    R²={r2:.3f}, slope={res.slope:.6f}, '
      f'p={res.pvalue:.2e}, n={valid.sum()}')

if valid2.sum() >= 3:
    res2   = stats.linregress(xa2[valid2], ya2[valid2])
    r2_adj = res2.rvalue ** 2
    print(f'Adjusted regression: R²={r2_adj:.3f}, slope={res2.slope:.6f}, '
          f'p={res2.pvalue:.2e}, n={valid2.sum()}')
else:
    res2   = None
    r2_adj = float('nan')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TICK_SIZE,
    'axes.linewidth': LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size': 3.5,  'ytick.major.size': 3.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 0.8,
    'svg.fonttype': 'none',
})

fig, ax = plt.subplots(figsize=(75*MM, 68*MM))
fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.20)

xfit = np.linspace(0, 1100, 300)

# Individual trial dots (semi-transparent)
for grp in GROUP_ORDER:
    gd = groups[grp]
    if not gd['x']:
        continue
    ax.scatter(gd['x'], gd['y'], marker=MARKER[grp], s=18,
               color=COLORS[grp], alpha=0.25, edgecolors='none', zorder=3)

# Group means ± SEM
for grp in GROUP_ORDER:
    gd = groups[grp]
    if len(gd['x']) < 2:
        continue
    xm = np.mean(gd['x']);  ym = np.mean(gd['y'])
    xe = stats.sem(gd['x']); ye = stats.sem(gd['y'])
    fmt = 'o' if grp in HG_GROUPS else 'D'
    ms  = 6.0 if grp in HG_GROUPS else 5.0
    mec = 'white' if grp in HG_GROUPS else EDGE[grp]
    mew = 0.4    if grp in HG_GROUPS else 0.5
    ax.errorbar(xm, ym, xerr=xe, yerr=ye,
                fmt=fmt, color=COLORS[grp], markersize=ms,
                markeredgecolor=mec, markeredgewidth=mew,
                capsize=2.5, capthick=LW, elinewidth=LW,
                ecolor=EDGE[grp], label=grp, zorder=5)

# Total regression line (black solid)
ax.plot(xfit, res.intercept + res.slope * xfit,
        '-', color='#333333', lw=0.8, alpha=0.7, zorder=2)

# Size-corrected regression line (red dashed)
if res2 is not None:
    ax.plot(xfit, res2.intercept + res2.slope * xfit,
            '--', color='#C0392B', lw=0.8, alpha=0.7, zorder=2)

# Zero reference
ax.axhline(0, color='#aaaaaa', lw=0.4, ls=':', zorder=0)

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered  = [(by_label[l], l) for l in GROUP_ORDER if l in by_label]
if ordered:
    ax.legend(*zip(*ordered), fontsize=TICK_SIZE - 0.5, loc='upper left',
              frameon=False, labelspacing=0.3, handlelength=1.2,
              handletextpad=0.4, ncol=2, columnspacing=0.8)

ax.text(0.95, 0.18, f'Total $R^2$ = {r2:.2f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=TICK_SIZE - 0.5, color='#333333')
if np.isfinite(r2_adj):
    ax.text(0.95, 0.08, f'Size-corrected $R^2$ = {r2_adj:.2f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=TICK_SIZE - 0.5, color='#C0392B')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=TICK_SIZE, pad=2)
ax.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=3)
ax.set_ylabel(r'Weibull AFT $\beta_d$ (log-min mm$^{-1}$)',
              fontsize=LABEL_SIZE, labelpad=3)
ax.set_xlim(0, 1100)
ax.text(-0.14, 1.04, 'D', transform=ax.transAxes,
        fontsize=PANEL_LBL, fontweight='bold', va='top')

for ext in ('.svg', '.pdf', '.png'):
    kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
    if ext == '.png':
        kw['dpi'] = 300
    fig.savefig(OUT / f'panel_D_AFT{ext}', **kw)
plt.close(fig)
print(f'\nSaved → {OUT}/panel_D_AFT.*')
