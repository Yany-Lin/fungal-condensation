#!/usr/bin/env python3
"""Extract evaporation rate K from existing track-history summaries.

Physics: the d²-law gives R²(t) = R₀² − K·t, so at disappearance
(R ≈ R_death) after lifetime τ:  K = (R_birth² − R_death²) / τ.

We use R_eq_birth (earliest tracked size) and R_eq_death (last tracked
size) in µm, converting from the pixel values stored in the CSVs via
each trial's calibration.  For "lost" droplets the tracker loses them
near the detection floor, so R_death is close to the minimum
resolvable size — the approximation K ≈ R_birth² / τ would overcount,
but using (R_birth² − R_death²)/τ is more accurate.

Filters:
  • death_cause == 'lost'  (fully evaporated, not censored or merged)
  • lifetime > 60 s
  • n_frames >= 5
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

TRACK_DIR = Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
OUT_DIR   = Path(__file__).resolve().parents[2] / 'additions' / '4_K_distance_evaporation'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Calibration roots (hydrogels vs fungi have different raw_data trees)
CAL_HG   = Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'raw_data'
CAL_FUNG = Path(__file__).resolve().parents[2] / 'FigureFungi' / 'raw_data'

EXEMPLAR_TRIALS = {
    '2to1.2': {'label': '2:1 hydrogel (trial 2)', 'color': '#d62728',
               'cal_root': CAL_HG},
    'agar.4': {'label': 'Agar control (trial 4)', 'color': '#1f77b4',
               'cal_root': CAL_HG},
    'Green.1': {'label': 'Fungus – Green (trial 1)', 'color': '#2ca02c',
                'cal_root': CAL_FUNG},
}

ALL_HG_TRIALS = [
    '2to1.1', '2to1.2', '2to1.3', '2to1.4', '2to1.6',
    '1to1.1', '1to1.2', '1to1.3', '1to1.4', '1to1.5',
    'agar.2', 'agar.3', 'agar.4', 'agar.5', 'agar.6',
]

FUNGI_TRIALS = ['Green.1', 'Green.2', 'Green.3', 'Green.4', 'Green.5',
                'Black.2']

BIN_WIDTH = 200  # µm


def get_pixel_size(trial_id, cal_root):
    """Read pixel_size_um from calibration.json."""
    cal_path = cal_root / trial_id / 'calibration.json'
    cal = json.loads(cal_path.read_text())
    return cal['scale']['pixel_size_um']


def load_and_filter(trial_id, cal_root, min_lifetime=60, min_frames=5):
    """Load track CSV, convert R to µm, compute K, and filter."""
    csv_path = TRACK_DIR / f'{trial_id}_track_histories.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    px = get_pixel_size(trial_id, cal_root)

    # Convert R from pixels to µm
    df['R_birth_um'] = df['R_eq_birth'] * px
    df['R_death_um'] = df['R_eq_death'] * px
    df['R_seed_um']  = df['R_eq_seed']  * px

    # K = (R_birth² − R_death²) / τ   [µm²/s]
    df['K'] = (df['R_birth_um']**2 - df['R_death_um']**2) / df['lifetime_s']

    # Filter
    mask = (
        (df['death_cause'] == 'lost') &
        (df['lifetime_s'] > min_lifetime) &
        (df['n_frames'] >= min_frames) &
        (df['K'] > 0) &
        np.isfinite(df['K'])
    )
    return df.loc[mask].copy()


def bin_K(df, bin_width=BIN_WIDTH):
    """Bin K by distance_um.  Returns bin centres, medians, IQR."""
    d = df['distance_um'].values
    K = df['K'].values
    bins = np.arange(0, d.max() + bin_width, bin_width)
    if len(bins) < 2:
        return None, None, None, None, None

    median, edges, _ = binned_statistic(d, K, statistic='median', bins=bins)
    q25, _, _ = binned_statistic(d, K, statistic=lambda x: np.percentile(x, 25), bins=bins)
    q75, _, _ = binned_statistic(d, K, statistic=lambda x: np.percentile(x, 75), bins=bins)
    counts, _, _ = binned_statistic(d, K, statistic='count', bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Drop bins with < 3 droplets
    ok = counts >= 3
    return centres[ok], median[ok], q25[ok], q75[ok], counts[ok]


plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
    'savefig.dpi': 300,
    'figure.dpi': 150,
})


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for tid, info in EXEMPLAR_TRIALS.items():
        df = load_and_filter(tid, info['cal_root'])
        if df is None or len(df) < 10:
            print(f"  {tid}: skipped (too few droplets)")
            continue
        centres, med, q25, q75, counts = bin_K(df)
        if centres is None:
            continue
        ax.plot(centres / 1000, med, 'o-', color=info['color'],
                markersize=3.5, label=info['label'], zorder=3)
        ax.fill_between(centres / 1000, q25, q75,
                        color=info['color'], alpha=0.15, lw=0)
        print(f"  {tid}: {len(df)} droplets, "
              f"K range {med.min():.2f}–{med.max():.2f} µm²/s")

    ax.set_xlabel('Distance from source (mm)')
    ax.set_ylabel(r'Evaporation rate $K$ ($\mu$m$^{2}$ s$^{-1}$)')
    ax.legend(fontsize=6.5, frameon=False, loc='best')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    for ext in ('svg', 'png', 'pdf'):
        fig.savefig(OUT_DIR / f'K_vs_distance.{ext}')
    plt.close(fig)
    print(f"  → Saved K_vs_distance.svg/png/pdf")

    fig2, ax2 = plt.subplots(figsize=(3.5, 2.6))

    # Pool all 2:1 trials
    group_map = {
        '2to1': {'trials': ['2to1.1', '2to1.2', '2to1.3', '2to1.4', '2to1.6'],
                 'color': '#d62728', 'label': '2:1 hydrogel'},
        '1to1': {'trials': ['1to1.1', '1to1.2', '1to1.3', '1to1.4', '1to1.5'],
                 'color': '#ff7f0e', 'label': '1:1 hydrogel'},
        'agar': {'trials': ['agar.2', 'agar.3', 'agar.4', 'agar.5', 'agar.6'],
                 'color': '#1f77b4', 'label': 'Agar control'},
    }

    for gname, ginfo in group_map.items():
        frames = []
        for tid in ginfo['trials']:
            d = load_and_filter(tid, CAL_HG)
            if d is not None and len(d) > 5:
                frames.append(d)
        if not frames:
            continue
        pooled = pd.concat(frames, ignore_index=True)
        centres, med, q25, q75, _ = bin_K(pooled)
        if centres is None or len(med) < 2:
            continue

        # Normalise by far-field median (last 3 bins)
        K_far = np.nanmedian(med[-3:])
        ratio = med / K_far
        r_lo  = q25 / K_far
        r_hi  = q75 / K_far

        ax2.plot(centres / 1000, ratio, 'o-', color=ginfo['color'],
                 markersize=3.5, label=ginfo['label'], zorder=3)
        ax2.fill_between(centres / 1000, r_lo, r_hi,
                         color=ginfo['color'], alpha=0.12, lw=0)

    ax2.axhline(1, color='0.5', ls='--', lw=0.6, zorder=1)
    ax2.set_xlabel('Distance from source (mm)')
    ax2.set_ylabel(r'$K / K_{\rm far}$')
    ax2.legend(fontsize=6.5, frameon=False, loc='best')
    ax2.set_xlim(left=0)
    ax2.spines[['top', 'right']].set_visible(False)
    fig2.tight_layout()
    for ext in ('svg', 'png', 'pdf'):
        fig2.savefig(OUT_DIR / f'K_ratio_vs_distance.{ext}')
    plt.close(fig2)
    print(f"  → Saved K_ratio_vs_distance.svg/png/pdf")

    rows = []
    for tid in ALL_HG_TRIALS:
        d = load_and_filter(tid, CAL_HG)
        if d is not None:
            rows.append({'trial': tid, 'group': tid.split('.')[0],
                         'n_droplets': len(d),
                         'K_median': d['K'].median(),
                         'K_q25': d['K'].quantile(0.25),
                         'K_q75': d['K'].quantile(0.75)})
    for tid in FUNGI_TRIALS:
        csv_path = TRACK_DIR / f'{tid}_track_histories.csv'
        if csv_path.exists():
            d = load_and_filter(tid, CAL_FUNG)
            if d is not None:
                rows.append({'trial': tid, 'group': 'fungi',
                             'n_droplets': len(d),
                             'K_median': d['K'].median(),
                             'K_q25': d['K'].quantile(0.25),
                             'K_q75': d['K'].quantile(0.75)})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / 'K_summary_all_trials.csv', index=False)
    print(f"\n  Summary ({len(summary)} trials):")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
