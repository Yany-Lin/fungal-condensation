#!/usr/bin/env python3
"""
Rate-channel visualization for Nature Communications.
Demonstrates that evaporation rate K is genuinely higher near the hygroscopic source,
independent of initial droplet size.

Figure 1 (size_matched_KM):  Size-matched Kaplan-Meier survival curves
Figure 2 (K_hexbin):         Size-controlled K vs distance + marginal histograms
Figure 3 (d2_collapse):      d²-law collapse (normalized) colored by distance
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from scipy.spatial import cKDTree
from lifelines import KaplanMeierFitter
from pathlib import Path

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

DATA_DIR = Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
OUT_DIR  = Path(__file__).resolve().parents[2] / 'additions' / '4_K_distance_evaporation'

# Colors
C_NEAR = '#D32F2F'   # warm red – near source
C_FAR  = '#1565C0'   # cool blue – far from source

NEAR_THRESH = 500    # um
FAR_THRESH  = 1500   # um


def load_trial(name):
    return pd.read_csv(DATA_DIR / f'{name}_track_histories.csv')


def pool_trials(names):
    """Load and concatenate multiple trials, keeping only valid rows."""
    frames = []
    for trial in names:
        df = load_trial(trial)
        df['trial'] = trial
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    return pool


HYGRO_TRIALS = ['1to1.1', '1to1.2', '1to1.3', '1to1.4', '1to1.5']
AGAR_TRIALS  = ['agar.2', 'agar.3', 'agar.4', 'agar.5', 'agar.6']
FUNGI_TRIALS = ['Green.1', 'Green.2', 'Green.3', 'Green.4', 'Green.5']


def size_match(near_df, far_df, tolerance_um=5.0):
    """For each near droplet, find closest-sized far droplet (without replacement)."""
    near = near_df.dropna(subset=['R_eq_seed']).sort_values('R_eq_seed').reset_index(drop=True)
    far = far_df.dropna(subset=['R_eq_seed']).copy().reset_index(drop=True)

    if len(near) == 0 or len(far) == 0:
        return pd.DataFrame(), pd.DataFrame()

    matched_near_idx = []
    matched_far_idx = []
    used_far = set()

    far_tree = cKDTree(far[['R_eq_seed']].values)

    for i, row in near.iterrows():
        k = min(50, len(far))
        dists, idxs = far_tree.query([row['R_eq_seed']], k=k)
        dists = np.atleast_1d(dists.squeeze())
        idxs = np.atleast_1d(idxs.squeeze())
        for d, j in zip(dists, idxs):
            if j not in used_far and d < tolerance_um:
                matched_near_idx.append(i)
                matched_far_idx.append(j)
                used_far.add(j)
                break

    mn = near.iloc[matched_near_idx].reset_index(drop=True)
    mf = far.iloc[matched_far_idx].reset_index(drop=True)
    return mn, mf


def plot_km_panel(ax, near_df, far_df, title, show_ylabel=True, show_legend=True):
    """Plot paired KM curves for size-matched near vs far droplets."""
    mn, mf = size_match(near_df, far_df, tolerance_um=5.0)

    if len(mn) == 0:
        ax.text(0.5, 0.5, 'No matched pairs', transform=ax.transAxes, ha='center')
        ax.set_title(title, fontweight='bold', pad=6)
        return 0.0

    kmf_near = KaplanMeierFitter()
    kmf_far  = KaplanMeierFitter()

    kmf_near.fit(mn['lifetime_s'], event_observed=~mn['censored'],
                 label=f'Near (< {NEAR_THRESH} \u00b5m)')
    kmf_far.fit(mf['lifetime_s'], event_observed=~mf['censored'],
                label=f'Far (> {FAR_THRESH} \u00b5m)')

    # Plot -- suppress lifelines auto-legend, we draw our own
    kmf_far.plot_survival_function(ax=ax, color=C_FAR, ci_show=True, ci_alpha=0.12)
    kmf_near.plot_survival_function(ax=ax, color=C_NEAR, ci_show=True, ci_alpha=0.12)

    # Shade the gap
    t_common = np.sort(np.unique(np.concatenate([
        kmf_near.survival_function_.index.values,
        kmf_far.survival_function_.index.values
    ])))
    s_near = np.interp(t_common, kmf_near.survival_function_.index,
                       kmf_near.survival_function_.values.flatten())
    s_far  = np.interp(t_common, kmf_far.survival_function_.index,
                       kmf_far.survival_function_.values.flatten())
    ax.fill_between(t_common, s_near, s_far, color='#FFD54F', alpha=0.22, zorder=0)

    ax.set_title(title, fontweight='bold', pad=6)
    ax.set_xlabel('Time (s)')
    if show_ylabel:
        ax.set_ylabel('Survival probability')
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(left=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Remove auto-legend, add custom if requested
    ax.get_legend().remove()
    if show_legend:
        ax.legend(loc='upper right', frameon=True, framealpha=0.92,
                  edgecolor='none', handlelength=1.5)

    # Annotate matched sizes
    size_str = f'Matched $R_0$: {mn["R_eq_seed"].median():.0f} \u00b5m'
    ax.text(0.97, 0.42, size_str, transform=ax.transAxes, ha='right', va='top',
            fontsize=6.5, fontstyle='italic', color='#555')

    # Compute max gap for return and annotation
    gap = s_far - s_near
    max_gap = gap.max()

    if max_gap > 0.08:
        idx = np.argmax(gap)
        t_gap = t_common[idx]
        s_mid = (s_near[idx] + s_far[idx]) / 2
        # Place annotation in a clear area
        ax.annotate(f'$\\Delta S$ = {max_gap:.0%}',
                    xy=(t_gap, s_mid), fontsize=7,
                    ha='center', va='center', color='#E65100', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='#FFF8E1', ec='none', alpha=0.85))

    return max_gap


def make_figure1():
    """Size-matched KM survival: pooled hygroscopic, pooled fungal, pooled agar."""
    configs = [
        (HYGRO_TRIALS, 'Hygroscopic salt (1:1)'),
        (FUNGI_TRIALS, 'Fungal source'),
        (AGAR_TRIALS,  'Agar control'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)

    for i, (trials, label) in enumerate(configs):
        pool = pool_trials(trials)
        near = pool[pool['distance_um'] < NEAR_THRESH]
        far  = pool[pool['distance_um'] > FAR_THRESH]
        gap = plot_km_panel(axes[i], near, far, label,
                            show_ylabel=(i == 0), show_legend=(i == 0))

    for i, letter in enumerate('abc'):
        axes[i].text(-0.12, 1.08, letter, transform=axes[i].transAxes,
                     fontsize=12, fontweight='bold', va='top')

    plt.tight_layout(w_pad=0.8)

    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(OUT_DIR / f'size_matched_KM.{ext}')
    plt.close(fig)
    print('  >> size_matched_KM saved')


def make_figure2():
    """Size-controlled K vs distance.

    Left:   Hexbin of K vs distance (raw, shows size confound)
    Right:  Size-binned K vs distance (controls for size; the real signal)
    """
    pool = pool_trials(HYGRO_TRIALS)
    unc = pool[~pool['censored']].dropna(subset=['R_eq_seed']).copy()
    unc = unc[unc['lifetime_s'] > 0]
    unc['K'] = unc['R_eq_seed']**2 / unc['lifetime_s']
    unc = unc[unc['K'] < unc['K'].quantile(0.99)]
    unc = unc[unc['K'] > 0]

    fig = plt.figure(figsize=(7.0, 3.5), layout='constrained')
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.15], wspace=0.35, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    hb = ax1.hexbin(unc['distance_um'], unc['K'], gridsize=30, cmap='inferno',
                    mincnt=1, linewidths=0.15, edgecolors='face')

    # Running median
    bins = np.linspace(0, unc['distance_um'].max(), 20)
    bc = (bins[:-1] + bins[1:]) / 2
    meds = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (unc['distance_um'] >= lo) & (unc['distance_um'] < hi)
        vals = unc.loc[mask, 'K']
        meds.append(vals.median() if len(vals) > 5 else np.nan)
    meds = np.array(meds)
    v = ~np.isnan(meds)
    ax1.plot(bc[v], meds[v], 'w-', lw=2.5, zorder=5)
    ax1.plot(bc[v], meds[v], color='#00E676', lw=1.2, zorder=6, label='Median')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.85, edgecolor='none', fontsize=7)

    ax1.set_xlabel('Distance from source (\u00b5m)')
    ax1.set_ylabel('$K = R_0^2 / \\tau$ (\u00b5m\u00b2/s)')
    ax1.set_title('Raw $K$ (size-confounded)', fontweight='bold', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    cb = fig.colorbar(hb, ax=ax1, fraction=0.045, pad=0.02, label='Count')
    cb.ax.tick_params(labelsize=6)

    ax2 = fig.add_subplot(gs[1])

    # Define size bins with good coverage
    size_bins = [(12, 22), (22, 32), (32, 45)]
    colors_size = ['#7B1FA2', '#1976D2', '#388E3C']

    for (slo, shi), col in zip(size_bins, colors_size):
        sub = unc[(unc['R_eq_seed'] >= slo) & (unc['R_eq_seed'] < shi)]
        if len(sub) < 20:
            continue

        # Binned medians
        dist_bins = np.linspace(0, sub['distance_um'].max(), 12)
        dbc = (dist_bins[:-1] + dist_bins[1:]) / 2
        K_meds = []
        K_q25 = []
        K_q75 = []
        for lo, hi in zip(dist_bins[:-1], dist_bins[1:]):
            mask = (sub['distance_um'] >= lo) & (sub['distance_um'] < hi)
            vals = sub.loc[mask, 'K']
            if len(vals) >= 3:
                K_meds.append(vals.median())
                K_q25.append(vals.quantile(0.25))
                K_q75.append(vals.quantile(0.75))
            else:
                K_meds.append(np.nan)
                K_q25.append(np.nan)
                K_q75.append(np.nan)

        K_meds = np.array(K_meds)
        K_q25 = np.array(K_q25)
        K_q75 = np.array(K_q75)
        v = ~np.isnan(K_meds)

        label = f'$R_0$ = {slo}\u2013{shi} \u00b5m'
        ax2.plot(dbc[v], K_meds[v], 'o-', color=col, ms=3.5, lw=1.3, label=label, zorder=5)
        ax2.fill_between(dbc[v], K_q25[v], K_q75[v], color=col, alpha=0.1, zorder=3)

    ax2.set_xlabel('Distance from source (\u00b5m)')
    ax2.set_ylabel('$K = R_0^2 / \\tau$ (\u00b5m\u00b2/s)')
    ax2.set_title('Size-controlled $K$', fontweight='bold', fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='none',
               fontsize=6.5, handlelength=1.5, title='Size bin', title_fontsize=7)

    # Panel labels
    ax1.text(-0.18, 1.08, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax2.text(-0.15, 1.08, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold')

    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(OUT_DIR / f'K_hexbin.{ext}')
    plt.close(fig)
    print('  >> K_hexbin saved')


def make_figure3():
    """d²-law normalized collapse and paired near-vs-far K within size bins.

    Left:   R²(t)/R²(0) vs t/tau — all curves should collapse to a 1->~0 line
    Right:  Near vs far K within matched size bins (paired comparison)
    """
    pool = pool_trials(HYGRO_TRIALS)
    unc = pool[~pool['censored']].dropna(subset=['R_eq_seed', 'R_eq_death']).copy()
    unc = unc[unc['lifetime_s'] > 0]
    unc['K'] = unc['R_eq_seed']**2 / unc['lifetime_s']
    unc = unc[unc['K'] < unc['K'].quantile(0.99)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    cmap = LinearSegmentedColormap.from_list('near_far', [C_NEAR, '#FFB74D', C_FAR])
    d_max = unc['distance_um'].max()

    # Subsample
    sub = unc.sample(n=min(500, len(unc)), random_state=42)

    for _, row in sub.iterrows():
        R0sq = row['R_eq_seed']**2
        Rfsq = row['R_eq_death']**2
        tau = row['lifetime_s']
        d_norm = row['distance_um'] / d_max
        # Normalized: x = t/tau (0 to 1), y = R²/R₀² (1 to Rf²/R0²)
        ax1.plot([0, 1], [1, Rfsq / R0sq],
                 color=cmap(d_norm), alpha=0.18, lw=0.4)

    # Theoretical d²-law: R²/R₀² = 1 - t/τ → line from (0,1) to (1,0)
    ax1.plot([0, 1], [1, 0], 'k--', lw=1.8, zorder=10,
             label='$d^2$-law: $R^2/R_0^2 = 1 - t/\\tau$')

    ax1.set_xlabel('Normalized time  $t \\,/\\, \\tau$')
    ax1.set_ylabel('$R^2(t) \\;/\\; R_0^2$')
    ax1.set_xlim(-0.02, 1.05)
    ax1.set_ylim(-0.05, 1.12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper right', frameon=True, framealpha=0.92, edgecolor='none', fontsize=6.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, d_max))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, fraction=0.04, pad=0.02, label='Distance (\u00b5m)')
    cb.ax.tick_params(labelsize=6)

    ax1.text(0.03, 0.07, 'Each line: one droplet\n(seed \u2192 death)',
             transform=ax1.transAxes, fontsize=6.5, fontstyle='italic', color='#555')

    size_bins = [(12, 20), (20, 28), (28, 36), (36, 50)]
    bin_labels = []
    near_meds = []
    far_meds = []
    near_iqr = []
    far_iqr = []

    for slo, shi in size_bins:
        sub = unc[(unc['R_eq_seed'] >= slo) & (unc['R_eq_seed'] < shi)]
        near = sub[sub['distance_um'] < NEAR_THRESH]
        far  = sub[sub['distance_um'] > FAR_THRESH]
        if len(near) >= 5 and len(far) >= 5:
            bin_labels.append(f'{slo}\u2013{shi}')
            near_meds.append(near['K'].median())
            far_meds.append(far['K'].median())
            near_iqr.append((near['K'].quantile(0.25), near['K'].quantile(0.75)))
            far_iqr.append((far['K'].quantile(0.25), far['K'].quantile(0.75)))

    x = np.arange(len(bin_labels))
    w = 0.32

    near_err = np.array([[m - q[0], q[1] - m] for m, q in zip(near_meds, near_iqr)]).T
    far_err  = np.array([[m - q[0], q[1] - m] for m, q in zip(far_meds, far_iqr)]).T

    ax2.bar(x - w/2, near_meds, w, color=C_NEAR, alpha=0.85, label=f'Near (< {NEAR_THRESH} \u00b5m)',
            yerr=near_err, capsize=3, error_kw=dict(lw=0.8, capthick=0.8))
    ax2.bar(x + w/2, far_meds, w, color=C_FAR, alpha=0.85, label=f'Far (> {FAR_THRESH} \u00b5m)',
            yerr=far_err, capsize=3, error_kw=dict(lw=0.8, capthick=0.8))

    # Connect paired bars with lines
    for xi, nm, fm in zip(x, near_meds, far_meds):
        ax2.plot([xi - w/2, xi + w/2], [nm, fm], 'k-', lw=0.5, alpha=0.4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels)
    ax2.set_xlabel('Initial radius bin $R_0$ (\u00b5m)')
    ax2.set_ylabel('Median $K$ (\u00b5m\u00b2/s)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='upper left', frameon=True, framealpha=0.92, edgecolor='none', fontsize=7)

    # Percent increase labels
    for xi, nm, fm in zip(x, near_meds, far_meds):
        pct = (nm - fm) / fm * 100
        ymax = max(nm, fm) * 1.15
        if pct > 0:
            ax2.text(xi, ymax, f'+{pct:.0f}%', ha='center', va='bottom',
                     fontsize=6.5, fontweight='bold', color='#C62828')

    # Panel labels
    ax1.text(-0.18, 1.08, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax2.text(-0.15, 1.08, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()

    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(OUT_DIR / f'd2_collapse.{ext}')
    plt.close(fig)
    print('  >> d2_collapse saved')


if __name__ == '__main__':
    print('Generating rate-channel figures ...')
    make_figure1()
    make_figure2()
    make_figure3()
    print('Done.')
