#!/usr/bin/env python
"""
Cox Proportional Hazards analysis of droplet lifetimes near vapor sinks.

Replaces the binned Kaplan-Meier -> tau50 -> regression pipeline with a
single semi-parametric model that uses every individual droplet, properly
handles censoring, and controls for initial droplet size.

Model:  h(t | x) = h0(t) * exp(beta1 * distance_mm + beta2 * log_R0_um)

Output folder: additions/5_cox_PH_model/
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*newton.*')
warnings.filterwarnings('ignore', message='.*convergence.*')

TRACK_DIR = Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
CALIB_DIRS = [
    Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'raw_data',
    Path(__file__).resolve().parents[2] / 'FigureFungi' / 'raw_data',
]
UNIVERSAL = Path(__file__).resolve().parents[2] / 'FigureTable' / 'output' / 'universal_metrics.csv'
OUT = Path(__file__).resolve().parents[2] / 'additions' / '5_cox_PH_model'
OUT.mkdir(parents=True, exist_ok=True)

TRIAL_GROUP = {
    'agar.1': 'Agar', 'agar.2': 'Agar', 'agar.3': 'Agar',
    'agar.4': 'Agar', 'agar.5': 'Agar',
    '0.5to1.2': '0.5:1', '0.5to1.3': '0.5:1', '0.5to1.4': '0.5:1',
    '0.5to1.5': '0.5:1', '0.5to1.7': '0.5:1',
    '1to1.1': '1:1', '1to1.2': '1:1', '1to1.3': '1:1',
    '1to1.4': '1:1', '1to1.5': '1:1',
    '2to1.1': '2:1', '2to1.2': '2:1', '2to1.3': '2:1',
    '2to1.4': '2:1', '2to1.5': '2:1',
    'Green.1': 'Green', 'Green.2': 'Green', 'Green.3': 'Green',
    'Green.3_new': 'Green', 'Green.4': 'Green', 'Green.5': 'Green',
    'white.1': 'White', 'white.3': 'White', 'white.4': 'White',
    'white.5': 'White', 'white.6': 'White',
    'Black.2': 'Black', 'black.3': 'Black', 'black.4': 'Black',
    'black.new': 'Black', 'black.new2': 'Black',
}

TRIAL_SYSTEM = {t: ('Hydrogel' if g in ('Agar', '1:1', '2:1') else 'Fungi')
                for t, g in TRIAL_GROUP.items()}

EXEMPLARS = ['2to1.2', 'agar.4', 'Green.1']

# Group colours
GROUP_COLORS = {
    'Agar': '#4daf4a', '0.5:1': '#E67E22', '1:1': '#377eb8', '2:1': '#e41a1c',
    'Green': '#228B22', 'White': '#999999', 'Black': '#333333',
}

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 0.8,
})


def load_calibration(trial_id):
    """Return pixel_size_um for a given trial."""
    for d in CALIB_DIRS:
        p = d / trial_id / 'calibration.json'
        if p.exists():
            with open(p) as f:
                return json.load(f)['scale']['pixel_size_um']
    raise FileNotFoundError(f'No calibration for {trial_id}')


def load_trial(trial_id, min_frames=3):
    """Load and pre-process one trial's track histories."""
    fp = TRACK_DIR / f'{trial_id}_track_histories.csv'
    df = pd.read_csv(fp)
    pix = load_calibration(trial_id)

    # Convert R_eq_seed from pixels to µm
    df['R0_um'] = df['R_eq_seed'] * pix

    # distance already in µm; convert to mm for interpretability
    df['distance_mm'] = df['distance_um'] / 1000.0

    # Filter
    df = df[(df['n_frames'] >= min_frames) &
            (df['lifetime_s'] > 0) &
            df['R0_um'].notna() &
            (df['R0_um'] > 0)].copy()

    df['log_R0_um'] = np.log(df['R0_um'])

    # Event indicator: 1 = fully observed (uncensored), 0 = censored
    df['event'] = (~df['censored']).astype(int)

    return df


def fit_cox(df, formula_cols, duration_col='lifetime_s', event_col='event',
            penalizer=0.01):
    """Fit a Cox PH model and return the fitter."""
    cph = CoxPHFitter(penalizer=penalizer)
    model_df = df[formula_cols + [duration_col, event_col]].dropna()
    cph.fit(model_df, duration_col=duration_col, event_col=event_col)
    return cph


print('=' * 70)
print('SECTION 1: Per-trial Cox PH fits')
print('=' * 70)

results = []

for trial_id in sorted(TRIAL_GROUP.keys()):
    try:
        df = load_trial(trial_id)
    except FileNotFoundError:
        print(f'  {trial_id}: skipped (no calibration)')
        continue

    if len(df) < 30:
        print(f'  {trial_id}: skipped (n={len(df)} < 30)')
        continue

    covars = ['distance_mm', 'log_R0_um']
    cph = fit_cox(df, covars)
    s = cph.summary

    hr_dist = np.exp(s.loc['distance_mm', 'coef'])
    coef_dist = s.loc['distance_mm', 'coef']
    p_dist = s.loc['distance_mm', 'p']
    ci_lo = np.exp(s.loc['distance_mm', 'coef lower 95%'])
    ci_hi = np.exp(s.loc['distance_mm', 'coef upper 95%'])
    concordance = cph.concordance_index_

    hr_size = np.exp(s.loc['log_R0_um', 'coef'])
    p_size = s.loc['log_R0_um', 'p']

    df['dist_x_logR0'] = df['distance_mm'] * df['log_R0_um']
    covars_int = ['distance_mm', 'log_R0_um', 'dist_x_logR0']
    try:
        cph_int = fit_cox(df, covars_int)
        s_int = cph_int.summary
        p_interaction = s_int.loc['dist_x_logR0', 'p']
        hr_interaction = np.exp(s_int.loc['dist_x_logR0', 'coef'])
    except Exception:
        p_interaction = np.nan
        hr_interaction = np.nan

    try:
        model_df = df[covars + ['lifetime_s', 'event']].dropna()
        cph_test = CoxPHFitter(penalizer=0.01)
        cph_test.fit(model_df, duration_col='lifetime_s', event_col='event')
        ph_test = proportional_hazard_test(cph_test, model_df,
                                           time_transform='rank')
        ph_p_dist = ph_test.summary.loc['distance_mm', 'p']
        ph_p_size = ph_test.summary.loc['log_R0_um', 'p']
    except Exception:
        ph_p_dist = np.nan
        ph_p_size = np.nan

    row = {
        'trial_id': trial_id,
        'group': TRIAL_GROUP[trial_id],
        'system': TRIAL_SYSTEM[trial_id],
        'n_droplets': len(df),
        'n_events': df['event'].sum(),
        'pct_censored': 100 * (1 - df['event'].mean()),
        'HR_distance': hr_dist,
        'coef_distance': coef_dist,
        'p_distance': p_dist,
        'HR_dist_ci_lo': ci_lo,
        'HR_dist_ci_hi': ci_hi,
        'HR_logR0': hr_size,
        'p_logR0': p_size,
        'concordance': concordance,
        'p_interaction': p_interaction,
        'HR_interaction': hr_interaction,
        'PH_p_distance': ph_p_dist,
        'PH_p_logR0': ph_p_size,
    }
    results.append(row)

    sig = '***' if p_dist < 0.001 else ('**' if p_dist < 0.01 else
           ('*' if p_dist < 0.05 else ''))
    print(f'  {trial_id:15s}  n={len(df):5d}  HR_dist={hr_dist:.3f}'
          f'  [{ci_lo:.3f}, {ci_hi:.3f}]  p={p_dist:.1e}{sig}'
          f'  C={concordance:.3f}')

res_df = pd.DataFrame(results)
res_df.to_csv(OUT / 'cox_per_trial_results.csv', index=False)
print(f'\nSaved per-trial results: {OUT / "cox_per_trial_results.csv"}')
print(f'  Trials fitted: {len(res_df)}')
print(f'  HR_distance < 1 (protective = farther is safer): '
      f'{(res_df["HR_distance"] < 1).sum()} / {len(res_df)}')
print(f'  Significant at p<0.05: {(res_df["p_distance"] < 0.05).sum()} / '
      f'{len(res_df)}')


print('\n' + '=' * 70)
print('SECTION 2: Forest plot')
print('=' * 70)

# Sort by group then by HR
group_order = ['2:1', '1:1', 'Agar', 'Green', 'White', 'Black']
res_df['_gord'] = res_df['group'].map({g: i for i, g in enumerate(group_order)})
res_df = res_df.sort_values(['_gord', 'HR_distance']).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(3.5, 5.5))

y_positions = []
y = 0
prev_group = None
for _, row in res_df.iterrows():
    if prev_group is not None and row['group'] != prev_group:
        y += 0.6  # gap between groups
    y_positions.append(y)
    prev_group = row['group']
    y += 1

y_positions = np.array(y_positions)

for _, row in res_df.iterrows():
    yp = y_positions[_]
    color = GROUP_COLORS.get(row['group'], '#666')
    # CI bar
    ax.plot([row['HR_dist_ci_lo'], row['HR_dist_ci_hi']], [yp, yp],
            color=color, linewidth=1.2, solid_capstyle='round')
    # Point
    marker = 'o' if row['p_distance'] < 0.05 else 'o'
    facecolor = color if row['p_distance'] < 0.05 else 'white'
    ax.plot(row['HR_distance'], yp, marker='o', markersize=4,
            markerfacecolor=facecolor, markeredgecolor=color,
            markeredgewidth=0.8, zorder=5)

# Null line
ax.axvline(1, color='#888', linewidth=0.5, linestyle='--', zorder=0)

# Labels
ax.set_yticks(y_positions)
labels = []
for _, row in res_df.iterrows():
    sig = '*' if row['p_distance'] < 0.05 else ''
    labels.append(f"{row['trial_id']}{sig}")
ax.set_yticklabels(labels, fontsize=6.5)
ax.invert_yaxis()

ax.set_xlabel('Hazard ratio per mm distance')
ax.set_title('Cox PH: distance effect on evaporation hazard')

# Group brackets on right
used_groups = []
for grp in group_order:
    mask = res_df['group'] == grp
    if mask.sum() == 0:
        continue
    idx = res_df.index[mask]
    ymin_g = y_positions[idx.min()]
    ymax_g = y_positions[idx.max()]
    ymid = (ymin_g + ymax_g) / 2
    ax.text(ax.get_xlim()[1] * 1.02, ymid, grp, fontsize=6.5,
            va='center', ha='left', color=GROUP_COLORS.get(grp, '#666'),
            fontweight='bold', clip_on=False)

ax.set_xscale('log')
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# Add note: filled = p<0.05
ax.text(0.02, 0.02, 'Filled = p < 0.05', transform=ax.transAxes,
        fontsize=6, va='bottom', ha='left', color='#555')

plt.tight_layout()
for ext in ['svg', 'png', 'pdf']:
    fig.savefig(OUT / f'cox_forest_plot.{ext}')
plt.close()
print(f'Saved forest plot.')


print('\n' + '=' * 70)
print('SECTION 3: Exemplar partial-effect plots')
print('=' * 70)

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)

for i, trial_id in enumerate(EXEMPLARS):
    ax = axes[i]
    df = load_trial(trial_id)

    covars = ['distance_mm', 'log_R0_um']
    cph = fit_cox(df, covars)

    # Partial effect of distance: plot predicted survival for 3 distances
    # at median log_R0_um
    median_logR0 = df['log_R0_um'].median()
    dists = np.quantile(df['distance_mm'], [0.1, 0.5, 0.9])
    dist_labels = ['Near', 'Mid', 'Far']
    colors_ex = ['#c62828', '#f57c00', '#1565c0']

    for j, (d, lab, col) in enumerate(zip(dists, dist_labels, colors_ex)):
        test_row = pd.DataFrame({'distance_mm': [d], 'log_R0_um': [median_logR0]})
        sf = cph.predict_survival_function(test_row)
        ax.plot(sf.index / 60, sf.values[:, 0], color=col, label=lab,
                linewidth=1.0)

    ax.set_xlabel('Time (min)')
    if i == 0:
        ax.set_ylabel('Survival probability')
    ax.set_title(trial_id, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc='lower left')

    # Annotate HR
    row_res = res_df[res_df['trial_id'] == trial_id].iloc[0]
    hr_text = (f"HR = {row_res['HR_distance']:.2f}\n"
               f"p = {row_res['p_distance']:.1e}")
    ax.text(0.97, 0.97, hr_text, transform=ax.transAxes, fontsize=6,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc',
                      alpha=0.8))

plt.tight_layout()
for ext in ['svg', 'png', 'pdf']:
    fig.savefig(OUT / f'cox_exemplar.{ext}')
plt.close()
print('Saved exemplar plots.')


print('\n' + '=' * 70)
print('SECTION 4: HR_distance vs delta, comparison with KM dτ₅₀/dr')
print('=' * 70)

uni = pd.read_csv(UNIVERSAL)
merged = res_df.merge(uni[['trial_id', 'delta_um', 'dtau50_dr',
                           'dtau50_dr_sizematched']],
                      on='trial_id', how='left')
merged = merged.dropna(subset=['delta_um'])

fig, ax = plt.subplots(figsize=(3.5, 3))

for grp in group_order:
    mask = merged['group'] == grp
    if mask.sum() == 0:
        continue
    sub = merged[mask]
    ax.scatter(sub['delta_um'], sub['HR_distance'],
               color=GROUP_COLORS.get(grp, '#666'), s=20, label=grp,
               edgecolors='white', linewidths=0.3, zorder=5)

# Error bars
for _, row in merged.iterrows():
    ax.plot([row['delta_um'], row['delta_um']],
            [row['HR_dist_ci_lo'], row['HR_dist_ci_hi']],
            color=GROUP_COLORS.get(row['group'], '#666'),
            linewidth=0.5, alpha=0.4)

# Regression line
slope, intercept, r, p, se = stats.linregress(merged['delta_um'],
                                                merged['HR_distance'])
x_fit = np.linspace(merged['delta_um'].min(), merged['delta_um'].max(), 100)
ax.plot(x_fit, intercept + slope * x_fit, 'k--', linewidth=0.8)

r2 = r ** 2
ax.text(0.03, 0.97, f'$R^2$ = {r2:.2f}\np = {p:.1e}',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.8))

ax.set_xlabel('$\\delta$ ($\\mu$m)')
ax.set_ylabel('Hazard ratio per mm distance')
ax.axhline(1, color='#888', linewidth=0.4, linestyle=':')
ax.legend(frameon=False, fontsize=6, loc='lower right')
ax.set_title('Cox distance HR vs. vapor sink strength')

plt.tight_layout()
for ext in ['svg', 'png', 'pdf']:
    fig.savefig(OUT / f'HR_vs_delta.{ext}')
plt.close()
print(f'  HR vs delta: R² = {r2:.3f}, p = {p:.2e}, slope = {slope:.4f}')

valid = merged.dropna(subset=['dtau50_dr'])
if len(valid) > 5:
    # log(HR) < 0 means farther droplets die slower → should correlate
    # with dτ₅₀/dr > 0 (farther droplets live longer)
    # But note: HR < 1 means LOWER hazard at greater distance
    # So -log(HR) should correlate positively with dτ₅₀/dr
    log_hr = np.log(valid['HR_distance'].values)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    for grp in group_order:
        mask = valid['group'] == grp
        if mask.sum() == 0:
            continue
        sub = valid[mask]
        ax.scatter(-np.log(sub['HR_distance']), sub['dtau50_dr'],
                   color=GROUP_COLORS.get(grp, '#666'), s=20, label=grp,
                   edgecolors='white', linewidths=0.3, zorder=5)

    neg_log_hr = -np.log(valid['HR_distance'].values)
    dtau = valid['dtau50_dr'].values
    r_corr, p_corr = stats.pearsonr(neg_log_hr, dtau)
    slope_c, inter_c, _, _, _ = stats.linregress(neg_log_hr, dtau)
    x_fit = np.linspace(neg_log_hr.min(), neg_log_hr.max(), 100)
    ax.plot(x_fit, inter_c + slope_c * x_fit, 'k--', linewidth=0.8)

    ax.text(0.03, 0.97, f'r = {r_corr:.2f}\np = {p_corr:.1e}',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc',
                      alpha=0.8))

    ax.set_xlabel('$-\\ln(\\mathrm{HR}_{\\mathrm{distance}})$')
    ax.set_ylabel('$d\\tau_{50}/dr$ (s/mm, KM-derived)')
    ax.set_title('Cox vs. Kaplan-Meier distance effects')
    ax.legend(frameon=False, fontsize=6, loc='lower right')
    plt.tight_layout()
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(OUT / f'cox_vs_KM.{ext}')
    plt.close()
    print(f'  Cox vs KM:   r = {r_corr:.3f}, p = {p_corr:.2e}')


print('\n' + '=' * 70)
print('SECTION 5: Pooled Cox PH analysis')
print('=' * 70)

pools = {
    '2:1 NaCl (all)': ['2to1.1', '2to1.2', '2to1.3', '2to1.4', '2to1.6'],
    '1:1 NaCl (all)': ['1to1.1', '1to1.2', '1to1.3', '1to1.4', '1to1.5'],
    'Agar (all)': ['agar.2', 'agar.3', 'agar.4', 'agar.5', 'agar.6'],
}

for pool_name, trial_list in pools.items():
    frames = []
    for tid in trial_list:
        try:
            d = load_trial(tid)
            d['trial'] = tid
            frames.append(d)
        except Exception:
            pass
    if not frames:
        continue
    pooled = pd.concat(frames, ignore_index=True)

    # Stratified Cox: fit with trial as stratum (each trial has its own
    # baseline hazard, but shared covariate effects)
    covars = ['distance_mm', 'log_R0_um']
    model_df = pooled[covars + ['lifetime_s', 'event', 'trial']].dropna()
    cph_pool = CoxPHFitter(penalizer=0.01)
    cph_pool.fit(model_df, duration_col='lifetime_s', event_col='event',
                 strata=['trial'])
    print(f'\n  {pool_name} (stratified by trial):')
    print(f'    n = {len(model_df)}, events = {model_df["event"].sum()}')
    for cov in covars:
        s = cph_pool.summary.loc[cov]
        print(f'    {cov:15s}  HR={np.exp(s["coef"]):.3f}'
              f'  [{np.exp(s["coef lower 95%"]):.3f},'
              f' {np.exp(s["coef upper 95%"]):.3f}]'
              f'  p={s["p"]:.1e}')

print('\n  Hydrogel vs Fungi pooled comparison')
all_frames = []
for tid in TRIAL_GROUP:
    try:
        d = load_trial(tid)
        d['trial'] = tid
        d['system'] = TRIAL_SYSTEM[tid]
        d['group'] = TRIAL_GROUP[tid]
        all_frames.append(d)
    except Exception:
        pass

all_pooled = pd.concat(all_frames, ignore_index=True)
all_pooled['is_fungi'] = (all_pooled['system'] == 'Fungi').astype(int)

covars_sys = ['distance_mm', 'log_R0_um', 'is_fungi']
model_df = all_pooled[covars_sys + ['lifetime_s', 'event', 'trial']].dropna()
cph_sys = CoxPHFitter(penalizer=0.01)
cph_sys.fit(model_df, duration_col='lifetime_s', event_col='event',
            strata=['trial'])

print(f'    n = {len(model_df)}, events = {model_df["event"].sum()}')
for cov in covars_sys:
    s = cph_sys.summary.loc[cov]
    print(f'    {cov:15s}  HR={np.exp(s["coef"]):.3f}'
          f'  [{np.exp(s["coef lower 95%"]):.3f},'
          f' {np.exp(s["coef upper 95%"]):.3f}]'
          f'  p={s["p"]:.1e}')


print('\n' + '=' * 70)
print('SECTION 6: Model diagnostics')
print('=' * 70)

fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

for i, trial_id in enumerate(EXEMPLARS):
    df = load_trial(trial_id)
    covars = ['distance_mm', 'log_R0_um']
    model_df = df[covars + ['lifetime_s', 'event']].dropna()
    cph = fit_cox(df, covars)

    # Schoenfeld residuals
    try:
        schoen = cph.compute_residuals(model_df, 'schoenfeld')

        for j, cov in enumerate(covars):
            ax = axes[j, i]
            times = schoen.index
            resids = schoen[cov].values

            ax.scatter(times / 60, resids, s=1, alpha=0.3, color='#333',
                       rasterized=True)

            # Lowess smoothing
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(resids, times / 60, frac=0.3)
                ax.plot(smoothed[:, 0], smoothed[:, 1], color='#e41a1c',
                        linewidth=1.2)
            except ImportError:
                pass

            ax.axhline(0, color='#888', linewidth=0.4, linestyle='--')
            ax.set_xlabel('Time (min)')
            cov_label = 'distance (mm)' if cov == 'distance_mm' else \
                        'ln($R_0$) ($\\mu$m)'
            if i == 0:
                ax.set_ylabel(f'Schoenfeld resid.\n{cov_label}')
            ax.set_title(trial_id if j == 0 else '', fontweight='bold')
    except Exception as e:
        print(f'  Schoenfeld residuals failed for {trial_id}: {e}')

plt.tight_layout()
for ext in ['svg', 'png', 'pdf']:
    fig.savefig(OUT / f'cox_schoenfeld.{ext}')
plt.close()
print('Saved Schoenfeld residual plots.')

fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=True)

for i, trial_id in enumerate(EXEMPLARS):
    ax = axes[i]
    df = load_trial(trial_id)

    # Bin distance into tertiles
    df['dist_bin'] = pd.qcut(df['distance_mm'], 3,
                             labels=['Near', 'Mid', 'Far'])
    colors_ll = ['#c62828', '#f57c00', '#1565c0']

    for j, (bin_label, col) in enumerate(zip(['Near', 'Mid', 'Far'],
                                              colors_ll)):
        sub = df[df['dist_bin'] == bin_label]
        kmf = KaplanMeierFitter()
        kmf.fit(sub['lifetime_s'], event_observed=sub['event'])

        # log(-log(S(t))) vs log(t)
        surv = kmf.survival_function_.iloc[1:]  # skip t=0
        st = surv.values.flatten()
        t = surv.index.values
        valid = st > 0
        if valid.sum() < 2:
            continue
        log_t = np.log(t[valid])
        log_neg_log_s = np.log(-np.log(st[valid]))

        ax.plot(log_t, log_neg_log_s, color=col, label=bin_label,
                linewidth=0.8)

    ax.set_xlabel('ln(time)')
    if i == 0:
        ax.set_ylabel('ln(-ln(S(t)))')
    ax.set_title(trial_id, fontweight='bold')
    ax.legend(frameon=False, fontsize=6)

plt.tight_layout()
for ext in ['svg', 'png', 'pdf']:
    fig.savefig(OUT / f'cox_loglog.{ext}')
plt.close()
print('Saved log-log survival plots.')

print('\n  PH assumption test (Schoenfeld, p-values):')
print(f'  {"Trial":15s} {"p(distance)":>12s} {"p(logR0)":>12s}  {"PH OK?":>6s}')
for _, row in res_df.iterrows():
    ok = 'Yes' if (row['PH_p_distance'] > 0.05 and
                   row['PH_p_logR0'] > 0.05) else 'No'
    print(f'  {row["trial_id"]:15s} {row["PH_p_distance"]:12.3e}'
          f' {row["PH_p_logR0"]:12.3e}  {ok:>6s}')


print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)

# Key statistics
n_sig = (res_df['p_distance'] < 0.05).sum()
n_hr_lt1 = (res_df['HR_distance'] < 1).sum()
median_hr = res_df['HR_distance'].median()
median_c = res_df['concordance'].median()

print(f'  Trials analysed:          {len(res_df)}')
print(f'  HR_distance < 1:          {n_hr_lt1}/{len(res_df)}'
      f' (farther = lower hazard)')
print(f'  Significant (p<0.05):     {n_sig}/{len(res_df)}')
print(f'  Median HR_distance:       {median_hr:.3f}')
print(f'  Median concordance index: {median_c:.3f}')
print(f'  HR vs delta R²:           {r2:.3f}')

# By system
for sys in ['Hydrogel', 'Fungi']:
    sub = res_df[res_df['system'] == sys]
    print(f'\n  {sys}:')
    print(f'    Median HR: {sub["HR_distance"].median():.3f}')
    print(f'    Significant: {(sub["p_distance"] < 0.05).sum()}/{len(sub)}')

# PH assumption
n_ph_ok = ((res_df['PH_p_distance'] > 0.05) &
           (res_df['PH_p_logR0'] > 0.05)).sum()
print(f'\n  PH assumption satisfied:  {n_ph_ok}/{len(res_df)} trials')

print(f'\nAll outputs saved to: {OUT}')
print('Files:')
for f in sorted(OUT.glob('*')):
    print(f'  {f.name}')
