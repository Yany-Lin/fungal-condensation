#!/usr/bin/env python3
"""Regenerate sensitivity_size_gradient.csv with all 35 lab trials.

Sweeps 36 zone-boundary combinations (4 near × 3 mid × 3 far) and computes
(R_far - R_near) / R_mid for each, then correlates against delta.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
REPO_ROOT  = THIS_DIR.resolve().parent.parent

HG_METRICS = REPO_ROOT / 'FigureHGAggregate' / 'output' / 'hydrogel_metrics.csv'
F_METRICS  = REPO_ROOT / 'FigureFungi' / 'output' / 'fungi_metrics.csv'
HG_AGG_DIR = REPO_ROOT / 'FigureHGAggregate' / 'raw_data' / 'aggregate_edt'
F_AGG_DIR  = REPO_ROOT / 'FigureFungi' / 'raw_data' / 'aggregate_edt'

H_BIN_UM    = 100
H_T_WINDOW  = (14.5, 15.5)

NEAR_RANGES = [(0, 300), (0, 500), (0, 700), (0, 1000)]
MID_RANGES  = [(750, 1250), (800, 1200), (900, 1100)]
FAR_RANGES  = [(1500, 2500), (1700, 2300), (1900, 2100)]


def load_delta_map():
    hg = pd.read_csv(HG_METRICS)
    fm = pd.read_csv(F_METRICS)
    delta = {}
    for _, row in hg.iterrows():
        delta[row['trial_id']] = row['delta_um']
    for _, row in fm.iterrows():
        delta[row['trial_id']] = row['delta_um']
    return delta


def load_edt_droplets(trial_id):
    for d in (HG_AGG_DIR, F_AGG_DIR):
        p = d / f'{trial_id}_edt_droplets.csv'
        if p.exists():
            df = pd.read_csv(p)
            tw = df[(df['time_min'] >= H_T_WINDOW[0]) &
                    (df['time_min'] <= H_T_WINDOW[1]) &
                    (df['radius_um'] > 0)].copy()
            return tw
    return None


def size_gradient(tw, near_range, mid_range, far_range):
    if tw is None or len(tw) < 30:
        return np.nan

    max_d = tw['distance_um'].max()
    bins = np.arange(0, max_d + H_BIN_UM, H_BIN_UM)
    tw2 = tw.copy()
    tw2['bin'] = pd.cut(tw2['distance_um'], bins=bins,
                        labels=bins[:-1] + H_BIN_UM / 2).astype(float)
    grp = tw2.groupby('bin')['radius_um'].mean()

    n_near_bins = int(near_range[1] / H_BIN_UM)
    near_centers = [H_BIN_UM / 2 + i * H_BIN_UM for i in range(n_near_bins)]
    near_vals = [grp[c] if c in grp.index else 0.0 for c in near_centers]
    R_near = np.mean(near_vals) if near_vals else np.nan

    mid_vals = grp[(grp.index >= mid_range[0]) & (grp.index <= mid_range[1])]
    if len(mid_vals) < 1:
        return np.nan
    R_mid = mid_vals.mean()

    far_vals = grp[(grp.index >= far_range[0]) & (grp.index <= far_range[1])]
    if len(far_vals) < 1:
        return np.nan
    R_far = far_vals.mean()

    if R_mid <= 0:
        return np.nan

    return (R_far - R_near) / R_mid


def main():
    delta_map = load_delta_map()
    trial_ids = sorted(delta_map.keys())
    print(f"Total lab trials: {len(trial_ids)}")

    # Pre-load all EDT data
    edt_data = {}
    for tid in trial_ids:
        edt_data[tid] = load_edt_droplets(tid)
        status = "OK" if edt_data[tid] is not None else "MISSING"
        print(f"  {tid}: {status}")

    rows = []
    for near in NEAR_RANGES:
        for mid in MID_RANGES:
            for far in FAR_RANGES:
                zg_vals = []
                delta_vals = []
                for tid in trial_ids:
                    zg = size_gradient(edt_data[tid], near, mid, far)
                    if np.isfinite(zg) and np.isfinite(delta_map[tid]):
                        zg_vals.append(zg)
                        delta_vals.append(delta_map[tid])

                n = len(zg_vals)
                if n >= 5:
                    r2 = stats.linregress(delta_vals, zg_vals).rvalue ** 2
                    rho, p = stats.spearmanr(delta_vals, zg_vals)
                else:
                    r2, rho, p = np.nan, np.nan, np.nan

                rows.append({
                    'Near (µm)': f'{near[0]}–{near[1]}',
                    'Mid (µm)': f'{mid[0]}–{mid[1]}',
                    'Far (µm)': f'{far[0]}–{far[1]}',
                    'n': n,
                    'R²': round(r2, 3),
                    'ρ': round(rho, 3),
                    'p': round(p, 3),
                })

    df = pd.DataFrame(rows).sort_values('R²', ascending=False)
    out_path = OUTPUT_DIR / 'sensitivity_size_gradient.csv'
    df.to_csv(out_path, index=False)

    print(f"\nSaved {out_path}")
    print(f"R² range: {df['R²'].min():.3f} – {df['R²'].max():.3f}")
    print(f"n range: {df['n'].min()} – {df['n'].max()}")

    canonical = df[(df['Near (µm)'] == '0–500') &
                   (df['Mid (µm)'] == '900–1100') &
                   (df['Far (µm)'] == '1900–2100')]
    if len(canonical):
        c = canonical.iloc[0]
        print(f"\nCanonical (0–500, 900–1100, 1900–2100): R²={c['R²']}, n={c['n']}")


if __name__ == '__main__':
    main()
