#!/usr/bin/env python3
"""Re-track exemplar trials to extract per-frame R_eq(t) histories,
then fit the d²-law  R²(t) = R₀² − K·t  to each droplet trajectory.

This script imports the tracking pipeline from track_droplets.py and
saves the full per-frame history for every droplet.  It then:
  • Fits a linear model to R²(t) for each droplet to get K and R² (GoF)
  • Generates spaghetti plots of R²(t) for near vs far droplets
  • Generates a histogram of d²-law R² goodness-of-fit
  • Generates K(distance) from direct fits
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic, linregress

OUT_DIR = Path(__file__).resolve().parents[2] / 'additions' / '4_K_distance_evaporation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add the tracking code to the path
TRACK_CODE = Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'code' / 'test_tracking'
sys.path.insert(0, str(TRACK_CODE))

import track_droplets as td

# Raw data roots
RAW_HG   = Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'raw_data'
RAW_FUNG = Path(__file__).resolve().parents[2] / 'FigureFungi' / 'raw_data'

# Exemplar trials to retrack
EXEMPLARS = {
    '2to1.2': {'raw': RAW_HG,   'polygon': False,
               'label': '2:1 hydrogel', 'color': '#d62728'},
    'agar.4': {'raw': RAW_HG,   'polygon': False,
               'label': 'Agar control', 'color': '#1f77b4'},
    'Green.1': {'raw': RAW_FUNG, 'polygon': True,
                'label': 'Fungus – Green', 'color': '#2ca02c'},
}

BIN_WIDTH = 200       # µm for distance binning
MIN_FRAMES_FIT = 5    # minimum frames for linear fit
MIN_LIFETIME = 60     # s
NEAR_THRESH = 500     # µm — "near" boundary
FAR_THRESH  = 1500    # µm — "far" boundary


plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 0.8,
    'savefig.dpi': 300,
    'figure.dpi': 150,
})


def retrack_trial(trial_id, raw_root, prefer_polygon):
    """Re-run tracking, return (tracks_df, all_tracks_dict, pixel_size)."""
    print(f"\n{'='*60}")
    print(f"Re-tracking {trial_id}")
    print(f"{'='*60}")
    df, coal_f, coal_b, all_tracks = td.run_tracking(
        trial_id, raw_data_root=raw_root, prefer_polygon=prefer_polygon)

    # Get pixel size for conversions
    trial_dir = raw_root / trial_id
    cal = json.loads((trial_dir / 'calibration.json').read_text())
    px = cal['scale']['pixel_size_um']

    return df, all_tracks, px


def extract_trajectories(all_tracks, px, summary_df):
    """Extract per-frame R_eq(t) trajectories for each droplet.

    IMPORTANT: We only fit the *evaporation phase* — from the frame of
    maximum R onward.  The backward-tracked condensation phase (R
    increasing) would contaminate the d^2-law fit.

    Returns a list of dicts, one per droplet, with:
      track_id, distance_um (at seed), times_s[], R_um[], R2_um2[]
    Also includes the d^2-law fit results.
    """
    results = []
    for tid, trk in all_tracks.items():
        hist = trk.get('history', [])
        if len(hist) < MIN_FRAMES_FIT:
            continue

        hist_sorted = sorted(hist, key=lambda h: h['time_s'])
        times_full = np.array([h['time_s'] for h in hist_sorted])
        R_px_full  = np.array([h['R_eq']   for h in hist_sorted])
        R_um_full  = R_px_full * px

        # Get distance from summary df
        row = summary_df.loc[summary_df['track_id'] == tid]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        dist_um = row['distance_um']
        death_cause = row['death_cause']
        lifetime = row['lifetime_s']

        # Only fit evaporated droplets with enough data
        if death_cause != 'lost' or lifetime < MIN_LIFETIME:
            continue

        peak_idx = np.argmax(R_um_full)
        times = times_full[peak_idx:]
        R_um  = R_um_full[peak_idx:]
        R2    = R_um ** 2

        if len(times) < MIN_FRAMES_FIT:
            continue

        # Shift time so t=0 at peak
        t0 = times[0]
        t_shifted = times - t0

        # Fit R^2(t) = R0^2 - K*t  ->  slope = -K
        slope, intercept, r_value, p_value, se = linregress(t_shifted, R2)
        K_fit = -slope           # K > 0 means evaporation
        R2_gof = r_value ** 2    # R^2 goodness of fit
        R0_fit = np.sqrt(max(intercept, 0))

        if K_fit <= 0:
            continue  # non-physical: droplet grew overall

        results.append({
            'track_id': tid,
            'distance_um': dist_um,
            'death_cause': death_cause,
            'lifetime_s': lifetime,
            'n_frames_evap': len(times),
            'n_frames_total': len(times_full),
            'K_fit': K_fit,
            'R2_gof': R2_gof,
            'R0_fit_um': R0_fit,
            'K_fit_se': abs(se),
            'times_s': t_shifted,        # evaporation phase only
            'R_um': R_um,
            'R2_um2': R2,
            'times_full_s': times_full - times_full[0],  # full trajectory
            'R_um_full': R_um_full,
            'R2_um2_full': R_um_full**2,
        })

    return results


def plot_spaghetti(results, trial_id, info):
    """Plot R^2(t) trajectories for nearest vs farthest tercile."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)

    dists = np.array([r['distance_um'] for r in results])
    if len(dists) < 6:
        plt.close(fig)
        print(f"  (skipped spaghetti for {trial_id}: too few droplets)")
        return

    t_lo = np.percentile(dists, 33)
    t_hi = np.percentile(dists, 67)

    near = [r for r in results if r['distance_um'] <= t_lo]
    far  = [r for r in results if r['distance_um'] >= t_hi]

    for ax, subset, label_fmt, col in [
        (axes[0], near, 'Nearest third (d < %.1f mm)', info['color']),
        (axes[1], far,  'Farthest third (d > %.1f mm)', '0.45'),
    ]:
        thresh = t_lo if 'Nearest' in label_fmt else t_hi
        label = label_fmt % (thresh / 1000)

        if not subset:
            ax.set_title(label, fontsize=7)
            ax.text(0.5, 0.5, 'No droplets', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7, color='0.5')
            continue
        for r in subset:
            ax.plot(r['times_s'] / 60, r['R2_um2'],
                    color=col, alpha=0.2, lw=0.4, zorder=1)
        # Median trajectory via interpolation
        tmax = max(r['times_s'][-1] for r in subset)
        t_grid = np.arange(0, tmax + 1, 30)
        R2_interp = []
        for r in subset:
            R2_i = np.interp(t_grid, r['times_s'], r['R2_um2'],
                             left=np.nan, right=np.nan)
            R2_interp.append(R2_i)
        R2_arr = np.array(R2_interp)
        med = np.nanmedian(R2_arr, axis=0)
        valid = np.sum(~np.isnan(R2_arr), axis=0) >= 3
        ax.plot(t_grid[valid] / 60, med[valid], color='k', lw=1.5,
                zorder=5)

        med_K = np.median([r['K_fit'] for r in subset])
        ax.set_title(label + '\nmedian $K$ = %.2f $\\mu$m$^2$ s$^{-1}$'
                     % med_K, fontsize=7)

    for ax in axes:
        ax.set_xlabel('Time since peak (min)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(r'$R^2$ ($\mu$m$^2$)')

    fig.suptitle(info['label'], fontsize=9, y=1.02)
    fig.tight_layout()
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(OUT_DIR / f'R2_trajectories_{trial_id}.{ext}',
                    bbox_inches='tight')
    plt.close(fig)
    print(f"  → R2_trajectories_{trial_id}")


def plot_gof_histogram(all_results, trial_labels):
    """Histogram of R² goodness-of-fit across all exemplar trials."""
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bins_h = np.linspace(0, 1, 30)

    for tid, results in all_results.items():
        gof = [r['R2_gof'] for r in results]
        ax.hist(gof, bins=bins_h, alpha=0.5,
                color=trial_labels[tid]['color'],
                label=trial_labels[tid]['label'],
                density=True, edgecolor='white', linewidth=0.3)

    ax.set_xlabel(r'$R^{2}$ of d$^{2}$-law fit')
    ax.set_ylabel('Density')
    ax.legend(fontsize=6.5, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(OUT_DIR / f'd2_law_gof.{ext}')
    plt.close(fig)
    print(f"  → d2_law_gof")


def plot_K_fitted(all_results, trial_labels):
    """K(distance) from direct linear fits — with error bars."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for tid, results in all_results.items():
        if len(results) < 10:
            continue
        dist = np.array([r['distance_um'] for r in results])
        K    = np.array([r['K_fit'] for r in results])
        bins = np.arange(0, dist.max() + BIN_WIDTH, BIN_WIDTH)
        if len(bins) < 2:
            continue
        med, edges, _ = binned_statistic(dist, K, statistic='median', bins=bins)
        q25, _, _ = binned_statistic(dist, K,
                      statistic=lambda x: np.percentile(x, 25), bins=bins)
        q75, _, _ = binned_statistic(dist, K,
                      statistic=lambda x: np.percentile(x, 75), bins=bins)
        counts, _, _ = binned_statistic(dist, K, statistic='count', bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ok = counts >= 3

        info = trial_labels[tid]
        ax.plot(centres[ok] / 1000, med[ok], 'o-',
                color=info['color'], markersize=3.5,
                label=info['label'], zorder=3)
        ax.fill_between(centres[ok] / 1000, q25[ok], q75[ok],
                        color=info['color'], alpha=0.15, lw=0)

    ax.set_xlabel('Distance from source (mm)')
    ax.set_ylabel(r'Evaporation rate $K$ ($\mu$m$^2$ s$^{-1}$)')
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(OUT_DIR / f'K_vs_distance_fitted.{ext}')
    plt.close(fig)
    print(f"  → K_vs_distance_fitted")


def main():
    all_results = {}

    for trial_id, info in EXEMPLARS.items():
        # Re-track
        df, all_tracks, px = retrack_trial(
            trial_id, info['raw'], info['polygon'])

        # Extract trajectories and fit d²-law
        results = extract_trajectories(all_tracks, px, df)
        all_results[trial_id] = results

        print(f"  {trial_id}: {len(results)} fitted droplets")
        if results:
            gof_arr = [r['R2_gof'] for r in results]
            print(f"    R² GoF:  median={np.median(gof_arr):.3f}, "
                  f"mean={np.mean(gof_arr):.3f}")
            K_arr = [r['K_fit'] for r in results]
            print(f"    K (fit): median={np.median(K_arr):.3f} µm²/s")

        # Save per-droplet fit results
        fit_rows = [{k: v for k, v in r.items()
                     if k not in ('times_s', 'R_um', 'R2_um2',
                                  'times_full_s', 'R_um_full', 'R2_um2_full')}
                    for r in results]
        pd.DataFrame(fit_rows).to_csv(
            OUT_DIR / f'K_fits_{trial_id}.csv', index=False)

        # Spaghetti plot for this trial
        plot_spaghetti(results, trial_id, info)

    # Cross-trial figures
    plot_gof_histogram(all_results, EXEMPLARS)
    plot_K_fitted(all_results, EXEMPLARS)

    print("\nDone.")


if __name__ == '__main__':
    main()
