#!/usr/bin/env python3
"""Supplementary Figure S4: Beysens growth profiles across all 30 trials.

Log-log R(t) colored by r', analysis window 10-15 min.
Phase 1: Batch-compute beta(r') and epsilon(r') profiles for all 30 trials.
Phase 2: Auto-select one exemplar per condition (or use overrides).
Phase 3: Composite figure (6 rows × 1 col) + individual QC PNGs.
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from supp_common import (
    CONDITIONS, DELTA, ALL_TRIALS, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    PANEL_LBL, LW_SPINE, apply_style, clean_axes, load_droplets,
    load_boundary, steiner_bin_area, save_fig,
)

BIN_WIDTH_UM     = 300
MIN_DROPLETS_BIN = 5
MIN_FRAMES_FIT   = 5
T_DATA_MIN       = 10.0   # narrower window: 10-15 min
T_DATA_MAX       = 15.0
R_MAX            = 65
R_ANCHOR         = 15.0
T_ANCHOR         = 10.0   # anchor at start of window

# Manual exemplar overrides (condition_key -> trial_id).
# Manually chosen: best far-field β≈1/3 + strongest near-field suppression
EXEMPLAR_OVERRIDE = {
    'agar':   'agar.4',     # far_β=0.77, modest gradient
    '0.5to1': '0.5to1.5',  # far_β=0.67, gradient=0.68 (best 0.5:1)
    '1to1':   '1to1.3',    # near bins 0.15+0.33 (suppressed+1/3), r2=0.96
    '2to1':   '2to1.3',    # far_β=0.55, only 2to1 with positive gradient
    'Green':  'Green.3',   # far_β=0.28 ≈ 1/3, gradient=0.70 (best overall)
    'black':  'black.5',   # far_β=0.67, r2=0.91
    'white':  'white.2',   # far_β=1.07, r2=0.98 (white all bad — best available)
}

OUT_DIR = OUTPUT_DIR / 'S3'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_profile(droplets, delta_um, P_body):
    """Compute beta(r') and epsilon(r') profiles via OLS log-log fit."""
    droplets = droplets.copy()
    droplets['r_prime_um'] = droplets['distance_um'] - delta_um

    half = BIN_WIDTH_UM / 2
    bin_centers = np.arange(0, 3600, BIN_WIDTH_UM)
    bin_edges   = bin_centers - half

    ts_rows, profile_rows = [], []

    for j, bc in enumerate(bin_centers):
        lo, hi = bin_edges[j], bin_edges[j] + BIN_WIDTH_UM
        in_bin = droplets[(droplets['r_prime_um'] >= lo) &
                          (droplets['r_prime_um'] < hi)]
        if in_bin.empty:
            continue

        A_bin = steiner_bin_area(P_body, delta_um, lo, hi) if P_body else np.nan

        bin_ts = []
        for t, frame in in_bin.groupby('time_min'):
            R = frame['radius_um'].values
            R = R[R > 0]
            if len(R) < MIN_DROPLETS_BIN:
                continue
            R_med = np.median(R)
            eps = (np.sum(np.pi * R**2) / A_bin
                   if P_body and not np.isnan(A_bin) and A_bin > 0
                   else np.nan)
            ts_rows.append({'r_prime_um': bc, 'time_min': t,
                            'R_median': R_med, 'n_droplets': len(R),
                            'epsilon': eps})
            bin_ts.append((t, R_med))

        if len(bin_ts) < MIN_FRAMES_FIT:
            continue

        # Filter to analysis window
        bin_ts_win = [(t, R) for t, R in bin_ts
                      if T_DATA_MIN <= t <= T_DATA_MAX]
        if len(bin_ts_win) < MIN_FRAMES_FIT:
            continue

        t_arr = np.array([x[0] for x in bin_ts_win])
        R_arr = np.array([x[1] for x in bin_ts_win])

        # Trim onset if V-shape detected
        idx_min = np.argmin(R_arr)
        if idx_min > 0:
            t_arr, R_arr = t_arr[idx_min:], R_arr[idx_min:]
        if len(t_arr) < MIN_FRAMES_FIT:
            continue

        # MAD outlier filter
        med_R = np.median(R_arr)
        mad_R = np.median(np.abs(R_arr - med_R))
        keep = np.abs(R_arr - med_R) <= 3 * mad_R
        t_arr, R_arr = t_arr[keep], R_arr[keep]
        if len(t_arr) < MIN_FRAMES_FIT:
            continue

        # OLS log-log fit: log R = beta * log t + log A
        lt, lR = np.log(t_arr), np.log(R_arr)
        beta, logA, r_val, p_val, se = stats.linregress(lt, lR)
        A_fit = np.exp(logA)

        eps_final = np.nanmedian([row['epsilon'] for row in ts_rows
                                   if row['r_prime_um'] == bc and
                                   T_DATA_MIN <= row['time_min'] <= T_DATA_MAX])

        profile_rows.append({
            'r_prime_um': bc, 'beta': beta, 'A': A_fit,
            'beta_se': se, 'r2': r_val**2, 'n_frames': len(t_arr),
            'epsilon_final': eps_final,
        })

    return pd.DataFrame(profile_rows), pd.DataFrame(ts_rows)


def plot_qc(profile, ts, tid, delta_um):
    """Generate log-log R(t) QC figure for one trial."""
    T_MIN, T_MAX = T_DATA_MIN, T_DATA_MAX
    t_ref = np.linspace(T_MIN, T_MAX, 200)

    r_bins = sorted(profile['r_prime_um'].unique())
    if len(r_bins) < 3:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"{tid}  |  delta={delta_um:.0f} um  |  {len(r_bins)} bins",
                 fontsize=11, y=0.98)

    cmap = plt.cm.plasma
    r_sorted = sorted(ts['r_prime_um'].unique())
    for i, bc in enumerate(r_sorted):
        sub = ts[(ts['r_prime_um'] == bc) &
                 (ts['time_min'] >= T_MIN) & (ts['time_min'] <= T_MAX)
                 ].sort_values('time_min')
        if len(sub) < 3:
            continue
        color = cmap(i / max(len(r_sorted)-1, 1))
        ax.plot(sub['time_min'], sub['R_median'], 'o-', color=color,
                lw=1.2, ms=3, alpha=0.8)

        # Show fitted line for this bin
        row = profile[profile['r_prime_um'] == bc]
        if not row.empty:
            beta_v = float(row['beta'].iloc[0])
            A_v = float(row['A'].iloc[0])
            ax.plot(t_ref, A_v * t_ref**beta_v, '-', color=color,
                    lw=1.8, alpha=0.35)

    # Reference lines
    ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** (1/3),
            'k--', lw=1.5, alpha=0.6, label=r'$\beta$=1/3')
    ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** 1.0,
            color='#C0392B', ls='--', lw=1.5, alpha=0.6, label=r'$\beta$=1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(T_MIN * 0.95, T_MAX * 1.05)
    ax.set_ylim(5, R_MAX)
    ax.set_xlabel('Time (min)', fontsize=10)
    ax.set_ylabel('Median R (um)', fontsize=10)
    ax.legend(fontsize=8, loc='upper left', frameon=False)
    clean_axes(ax)

    if r_sorted:
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                    norm=plt.Normalize(0, r_sorted[-1]/1000))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("r' (mm)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    beta_min = profile['beta'].min()
    beta_max = profile['beta'].max()
    r2_med = profile['r2'].median()
    ax.text(0.97, 0.05,
            f'beta: [{beta_min:.2f}, {beta_max:.2f}]\n'
            f'R2 median: {r2_med:.2f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT_DIR / f'beysens_{tid}.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def select_exemplars(all_profiles):
    """Pick one best trial per condition based on usable bins and r2."""
    exemplars = {}
    for cond_key, trial_ids, cond_label, _ in CONDITIONS:
        if cond_key in EXEMPLAR_OVERRIDE:
            exemplars[cond_key] = EXEMPLAR_OVERRIDE[cond_key]
            continue

        best_tid, best_score = None, -1
        for tid in trial_ids:
            sub = all_profiles[all_profiles['trial_id'] == tid]
            if sub.empty:
                continue
            n_bins = len(sub)
            med_r2 = sub['r2'].median()
            beta_range = sub['beta'].max() - sub['beta'].min()
            score = n_bins * med_r2 * (1 + beta_range)
            if score > best_score:
                best_score = score
                best_tid = tid
        if best_tid:
            exemplars[cond_key] = best_tid
    return exemplars


def plot_composite(exemplars, all_profiles, all_ts):
    """7-condition 2-column composite: log-log R(t) colored by r'."""
    apply_style()
    # 4 rows × 2 cols — 8 slots, last one empty (G centred manually)
    fig, axes = plt.subplots(4, 2, figsize=(180 * MM, 230 * MM),
                              sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.84, top=0.97, bottom=0.06,
                         hspace=0.52, wspace=0.28)

    T_MIN, T_MAX = T_DATA_MIN, T_DATA_MAX
    t_ref = np.linspace(T_MIN, T_MAX, 200)
    panel_letters = list('ABCDEFG')
    cmap = plt.cm.plasma

    axes_flat = axes.flatten()
    axes_flat[-1].set_visible(False)   # 8th slot unused

    for idx, (cond_key, _, cond_label, cond_color) in enumerate(CONDITIONS):
        tid = exemplars.get(cond_key)
        if tid is None:
            continue

        ax = axes_flat[idx]
        profile  = all_profiles[all_profiles['trial_id'] == tid]
        delta_um = DELTA[tid]
        R_MAX_MM = 3.5   # colorbar upper bound in r (mm)
        VIS_BW   = 150   # µm — finer bins for display, starting at r'=0

        # Recompute binned R(t) with 150µm bins anchored at r'=0
        raw = load_droplets(tid)
        raw['r_prime_um'] = raw['distance_um'] - delta_um
        raw = raw[raw['r_prime_um'] >= 0]   # exclude inside dry zone
        vis_centers = np.arange(VIS_BW / 2, 3000, VIS_BW)

        for bc in vis_centers:
            lo, hi = bc - VIS_BW / 2, bc + VIS_BW / 2
            inbin = raw[(raw['r_prime_um'] >= lo) & (raw['r_prime_um'] < hi) &
                        (raw['time_min'] >= T_MIN) & (raw['time_min'] <= T_MAX)]
            ts_bin = (inbin.groupby('time_min')['radius_um']
                           .agg(R_median='median', n='count')
                           .reset_index())
            ts_bin = ts_bin[ts_bin['n'] >= MIN_DROPLETS_BIN].sort_values('time_min')
            if len(ts_bin) < 3:
                continue

            r_mm = (bc + delta_um) / 1000
            c = cmap(np.clip(r_mm / R_MAX_MM, 0, 1))
            ax.plot(ts_bin['time_min'], ts_bin['R_median'], 'o-',
                    color=c, lw=0.9, ms=2.0, alpha=0.85)

        # Fitted power-law overlays from cached profile (300µm bins)
        for _, row in profile.iterrows():
            if np.isnan(row.get('beta', np.nan)):
                continue
            r_mm = (row['r_prime_um'] + delta_um) / 1000
            c = cmap(np.clip(r_mm / R_MAX_MM, 0, 1))
            ax.plot(t_ref, row['A'] * t_ref**row['beta'], '-', color=c,
                    lw=1.2, alpha=0.28)

        # Reference lines
        ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** (1/3),
                'k--', lw=0.8, alpha=0.55)
        ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** 1.0,
                color='#C0392B', ls='--', lw=0.8, alpha=0.55)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(T_MIN * 0.992, T_MAX * 1.008)
        ax.set_ylim(10, 65)
        clean_axes(ax)
        ax.tick_params(labelsize=TICK_SIZE, pad=2)
        ax.set_xlabel('Time (min)', fontsize=LABEL_SIZE - 0.5)
        ax.set_ylabel('Median R (µm)', fontsize=LABEL_SIZE - 0.5)

        # Panel letter — tight to top-left corner
        ax.text(-0.10, 1.10, panel_letters[idx], transform=ax.transAxes,
                fontsize=PANEL_LBL, va='top')

        # Condition + trial as coloured title
        ax.set_title(f'{cond_label}  ({tid})', fontsize=TICK_SIZE,
                     color=cond_color, fontweight='bold', pad=3)

        if idx == 0:
            ax.legend([r'$\beta$=1/3', r'$\beta$=1'],
                      fontsize=TICK_SIZE - 1, loc='upper left',
                      frameon=False, handlelength=1.2)

    # Centre panel G horizontally between the two column positions
    fig.canvas.draw()
    pos_A = axes_flat[0].get_position()
    pos_B = axes_flat[1].get_position()
    centre_x = (pos_A.x0 + pos_B.x1) / 2
    pos_G = axes_flat[6].get_position()
    axes_flat[6].set_position([centre_x - pos_G.width / 2,
                                pos_G.y0, pos_G.width, pos_G.height])

    # Colorbar — r (mm), 0 to 3.5
    cbar_ax = fig.add_axes([0.86, 0.10, 0.018, 0.74])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 3.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('r (mm)', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 0.5)

    stem = str(OUT_DIR / 'FigureS3_beysens_profiles')
    save_fig(fig, stem)
    plt.close(fig)
    print(f'\nComposite saved: {stem}.pdf/.svg')


def main():
    apply_style()
    all_profiles_list, all_ts_list = [], []

    for tid in ALL_TRIALS:
        delta = DELTA[tid]
        print(f'  {tid} (delta={delta:.0f}) ... ', end='', flush=True)

        df = load_droplets(tid)
        P_body = load_boundary(tid)
        if P_body is None:
            P_body = 2 * np.pi * delta
            print('[circ approx] ', end='')

        profile, ts = compute_profile(df, delta, P_body)
        if profile.empty:
            print('SKIP (no bins)')
            continue

        profile['trial_id'] = tid
        ts['trial_id'] = tid
        all_profiles_list.append(profile)
        all_ts_list.append(ts)

        print(f'{len(profile)} bins, '
              f'beta=[{profile["beta"].min():.2f}, {profile["beta"].max():.2f}]')

        # QC plot
        plot_qc(profile, ts, tid, delta)

    all_profiles = pd.concat(all_profiles_list, ignore_index=True)
    all_ts = pd.concat(all_ts_list, ignore_index=True)

    # Save CSVs
    all_profiles.to_csv(OUT_DIR / 'beysens_all_profiles.csv', index=False)
    all_ts.to_csv(OUT_DIR / 'beysens_all_timeseries.csv', index=False)
    print(f'\nProfiles: {len(all_profiles)} rows -> beysens_all_profiles.csv')
    print(f'Timeseries: {len(all_ts)} rows -> beysens_all_timeseries.csv')

    # Select exemplars
    exemplars = select_exemplars(all_profiles)
    print(f'\nExemplars: {exemplars}')

    # Composite figure
    plot_composite(exemplars, all_profiles, all_ts)

    print('\nDone.')


if __name__ == '__main__':
    main()
