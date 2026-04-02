#!/usr/bin/env python3
"""Compute d*, delta, zone_metric for all 41 trials and generate scatter plots."""

import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from lifelines import KaplanMeierFitter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'FigureHGAggregate' / 'code' / 'test_tracking'))
from make_manuscript_panels import compute_tau_zone_metric as _fig2k_tau_zone

THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR    = THIS_DIR.parent / 'raw_data'

OSF_ROOT   = THIS_DIR.resolve().parent.parent
UNIV_CSV   = OSF_ROOT / 'FigureTable' / 'output' / 'universal_metrics.csv'
HG_TRACK   = OSF_ROOT / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
HG_AGG_DIR = OSF_ROOT / 'FigureHGAggregate' / 'raw_data' / 'aggregate_edt'
F_AGG_DIR  = OSF_ROOT / 'FigureFungi'        / 'raw_data' / 'aggregate_edt'

_T7_RAW        = Path('/Volumes/T7/Fungal Hygroscopy/RAW/RSR RAW')
T7_RSR         = _T7_RAW if _T7_RAW.exists() else RAW_DIR / 'rsr_raw'
RSR_DZV_CSV    = RAW_DIR / 'rsr_dry_zone_summary.csv'
T7_RSR_FOLDERS = {
    'RSR1':         T7_RSR / 'RSR 1',
    'RSR2':         T7_RSR / 'RSR 2',
    'RSR7':         T7_RSR / 'RSR10',
    'RSRDiseased3': T7_RSR / 'RSR 3',
    'RSRDiseased5': T7_RSR / 'RSR 5',
    'RSRDiseased6': T7_RSR / 'RSR 6',
}

T_SEED_S    = 900
MIN_FRAMES  = 3
DIST_BIN_UM = 200
MIN_PER_BIN = 15
MIN_BINS    = 3
FLAT_RANGE  = 1.0
LAB_CAP_MM  = 4.0
RSR_CAP_MM       = 4.0
RSR_SIZE_BIN_MM  = 0.4
RSR_MIN_SIZE_BIN = 10
RSR_FLAT_RANGE_UM = 5.0

DELTA_T_WINDOW = (14.5, 15.5)

NEAR_MAX_MM = 0.5
MID_LO_MM   = 0.9;  MID_HI_MM  = 1.1
FAR_LO_MM   = 1.9;  FAR_HI_MM  = 2.1

TAU_MID_LO_MM    = 0.8;  TAU_MID_HI_MM    = 1.2   # wider for 400µm RSR bins
TAU_FAR_LO_MM    = 1.6;  TAU_FAR_HI_MM    = 2.4
TAU_SLOPE_CAP_MM = 2.0

RSR_FOLDERS = {
    'RSR1':         T7_RSR / 'RSR 1',
    'RSR2':         T7_RSR / 'RSR 2',
    'RSR7':         T7_RSR / 'RSR10',
    'RSRDiseased3': T7_RSR / 'RSR 3',
    'RSRDiseased5': T7_RSR / 'RSR 5',
    'RSRDiseased6': T7_RSR / 'RSR 6',
}

RSR_GROUP = {
    'RSR1': 'Healthy', 'RSR2': 'Healthy', 'RSR7': 'Healthy',
    'RSRDiseased3': 'Diseased', 'RSRDiseased5': 'Diseased', 'RSRDiseased6': 'Diseased',
}

RSR_GROUPS = {'Healthy', 'Diseased'}
RSR_COLOR  = '#E74C3C'

GROUP_COLOR = {
    'Agar':     '#BBBBBB',
    '0.5:1':    '#BBBBBB',
    '1:1':      '#BBBBBB',
    '2:1':      '#BBBBBB',
    'Green':    '#BBBBBB',
    'White':    '#BBBBBB',
    'Black':    '#BBBBBB',
    'Healthy':  RSR_COLOR,
    'Diseased': RSR_COLOR,
}
GROUP_MARKER = {
    'Agar': 'o', '0.5:1': 'o', '1:1': 'o', '2:1': 'o',
    'Green': 'o', 'White': 'o', 'Black': 'o',
    'Healthy': 'o', 'Diseased': 'o',
}

def _hill(d, T0, A, K, n):
    dn = np.power(np.maximum(d, 0), n)
    Kn = np.power(K, n)
    return T0 + A * dn / (Kn + dn)


def _fit_hill_constrained(bx, by, label=''):
    """Hill fit with T0 and A fixed to observed near/far values, fitting only K and n."""
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    ok = np.isfinite(bx) & np.isfinite(by)
    bx, by = bx[ok], by[ok]
    if len(bx) < MIN_BINS:
        return np.nan, np.nan
    T0_fix = float(by[0])
    A_fix  = float(by.max() - by.min())
    if A_fix < FLAT_RANGE:
        return 0.0, 1.0
    def _hill_fixed(d, K, n):
        dn = np.power(np.maximum(d, 0), n)
        Kn = np.power(K, n)
        return T0_fix + A_fix * dn / (Kn + dn)
    try:
        popt, _ = curve_fit(_hill_fixed, bx, by,
                            p0=[np.median(bx), 2.0],
                            bounds=([0.01, 0.3], [bx.max() * 3, 10.0]),
                            max_nfev=50000, method='trf')
    except Exception:
        return np.nan, np.nan
    by_pred = _hill_fixed(bx, *popt)
    ss_res  = np.sum((by - by_pred) ** 2)
    ss_tot  = np.sum((by - by.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    K  = float(popt[0])
    print(f'    [{label}] d*={K:.3f} mm  R²={r2:.3f}  '
          f'(T0={T0_fix:.1f}, A={A_fix:.1f}µm, constrained)')
    return K, r2


def _fit_hill(bx, by, label=''):
    """Four-parameter Hill fit. Returns (K, r2, A, n)."""
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    ok = np.isfinite(bx) & np.isfinite(by)
    bx, by = bx[ok], by[ok]
    if len(bx) < MIN_BINS:
        return np.nan, np.nan, np.nan, np.nan
    if float(by.max() - by.min()) < FLAT_RANGE:
        return 0.0, 1.0, 0.0, 1.0   # flat profile → d*=0
    T0_g = float(by[0])
    A_g  = max(float(by.max() - by.min()), 0.1)
    lo   = [0, 0, 0.01, 0.3]
    hi   = [by.max(), by.max() * 3, bx.max() * 10, 10.0]
    try:
        popt, _ = curve_fit(_hill, bx, by,
                            p0=[T0_g, A_g, np.median(bx), 2.0],
                            bounds=(lo, hi), max_nfev=50000, method='trf')
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    by_pred = _hill(bx, *popt)
    ss_res  = np.sum((by - by_pred) ** 2)
    ss_tot  = np.sum((by - by.mean()) ** 2)
    r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    K = float(popt[2]);  A = float(popt[1]);  n = float(popt[3])
    print(f'    [{label}] d*={K:.3f} mm  R²={r2:.3f}  A={A:.2f}  n={n:.2f}')
    return K, r2, A, n


def compute_lab_dRdr(trial_id):
    for d in (HG_AGG_DIR, F_AGG_DIR):
        p = d / f'{trial_id}_edt_droplets.csv'
        if p.exists():
            df = pd.read_csv(p)
            tw = df[(df['time_min'] >= DELTA_T_WINDOW[0]) &
                    (df['time_min'] <= DELTA_T_WINDOW[1]) &
                    (df['radius_um'] > 0)].copy()
            if len(tw) < 10:
                return np.nan
            dist_mm = tw['distance_um'].values / 1000.0
            radius  = tw['radius_um'].values
            sl, _, _, _, _ = stats.linregress(dist_mm, radius)
            return float(sl)   # µm / mm
    return np.nan


def compute_lab_delta_p1(trial_id):
    for d in (HG_AGG_DIR, F_AGG_DIR):
        p = d / f'{trial_id}_edt_droplets.csv'
        if p.exists():
            df = pd.read_csv(p)
            tw = df[(df['time_min'] >= DELTA_T_WINDOW[0]) &
                    (df['time_min'] <= DELTA_T_WINDOW[1]) &
                    (df['distance_um'] > 0)].copy()
            if len(tw) < 10:
                return np.nan
            return float(np.percentile(tw['distance_um'], 1))
    return np.nan


def _get_tau50_profile(trial_id):
    path = HG_TRACK / f'{trial_id}_track_histories.csv'
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED_S) / 60.0
    df = df[(df['tau_fwd_min'] > 0) & (df['distance_um'] <= LAB_CAP_MM * 1000)].copy()
    if len(df) < 30:
        return None, None
    bins = np.arange(0, df['distance_um'].max() + DIST_BIN_UM, DIST_BIN_UM)
    df['db'] = pd.cut(df['distance_um'], bins=bins, labels=False)
    kmf = KaplanMeierFitter()
    d_vals, tau_vals = [], []
    for b in sorted(df['db'].dropna().unique()):
        sub = df[df['db'] == b]
        if len(sub) < MIN_PER_BIN:
            continue
        center = (bins[int(b)] + DIST_BIN_UM / 2) / 1000.0
        kmf.fit(sub['tau_fwd_min'], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            d_vals.append(center)
            tau_vals.append(t50)
    if len(d_vals) < MIN_BINS:
        return None, None
    return np.array(d_vals), np.array(tau_vals)


def compute_lab_dstar(trial_id):
    d_vals, tau_vals = _get_tau50_profile(trial_id)
    if d_vals is None:
        return np.nan
    K, r2, A, n = _fit_hill(d_vals, tau_vals, label=trial_id)
    tag = f'd*={K:.3f} mm' if (np.isfinite(K) and K > 0) else \
          ('d*=0.0 (flat)' if K == 0.0 else 'd*=NaN')
    print(f'  {trial_id:<12}: {tag}  n={len(d_vals)}')
    return K


def compute_lab_tau50_metrics(trial_id):
    d_vals, tau_vals = _get_tau50_profile(trial_id)
    if d_vals is None:
        return np.nan, np.nan, np.nan
    tau_range    = float(tau_vals.max() - tau_vals.min())
    tau_integral = float(np.trapezoid(tau_vals, d_vals))
    tau_fold     = float(tau_vals.max() / tau_vals.min()) if tau_vals.min() > 0 else np.nan
    return tau_range, tau_integral, tau_fold


def compute_lab_delta_R(trial_id):
    for d in (HG_AGG_DIR, F_AGG_DIR):
        p = d / f'{trial_id}_edt_droplets.csv'
        if p.exists():
            df = pd.read_csv(p)
            tw = df[(df['time_min'] >= DELTA_T_WINDOW[0]) &
                    (df['time_min'] <= DELTA_T_WINDOW[1]) &
                    (df['radius_um'] > 0)].copy()
            if len(tw) < 10:
                return np.nan
            dist_mm = tw['distance_um'] / 1000.0
            near = tw[dist_mm < NEAR_MAX_MM]['radius_um']
            far  = tw[(dist_mm >= FAR_LO_MM) & (dist_mm <= FAR_HI_MM)]['radius_um']
            R_near = near.mean() if len(near) >= 1 else 0.0
            if len(far) < 1:
                return np.nan
            return float(far.mean() - R_near)
    return np.nan


TAU_MIN_THRESH_MIN = 1.0
TAU_CAP_MM         = 4.0

def _load_rsr_tau50(sample):
    folder = T7_RSR_FOLDERS.get(sample)
    if folder is None:
        return None, None
    p = folder / 'Every 30s from Evap' / 'survival_analysis' / 'tau50_by_distance.csv'
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    valid = (df['tau50_min'].notna() &
             (df['tau50_min'] > TAU_MIN_THRESH_MIN) &
             (df['distance_mm'] <= TAU_CAP_MM))
    df = df[valid].copy()
    if len(df) < 3:
        return None, None
    return df['distance_mm'].values, df['tau50_min'].values


def compute_rsr_tau50_metrics(sample):
    d, t = _load_rsr_tau50(sample)
    if d is None:
        return np.nan, np.nan, np.nan, np.nan
    sl, ic, r, p, _ = stats.linregress(d, t)
    tau_range = float(t.max() - t.min())
    tau_fold  = float(t.max() / t.min()) if t.min() > 0 else np.nan
    # Also try Hill fit for d*
    K, r2, A, n = _fit_hill(d, t, label=f'{sample}_tau50')
    dstar = K if np.isfinite(K) else np.nan
    print(f'  {sample:<16}: dtau50/dr={sl:.2f} min/mm  R²={r**2:.3f}  '
          f'range={tau_range:.1f} min  fold={tau_fold:.2f}x  d*={dstar:.3f}mm' if np.isfinite(dstar)
          else f'  {sample:<16}: dtau50/dr={sl:.2f} min/mm  R²={r**2:.3f}  '
               f'range={tau_range:.1f} min  fold={tau_fold:.2f}x  d*=NaN')
    return float(sl), tau_range, tau_fold, dstar


def _tau50_zone_from_profile(d_arr, t_arr):
    near_mask = d_arr < NEAR_MAX_MM
    mid_mask  = (d_arr >= TAU_MID_LO_MM) & (d_arr <= TAU_MID_HI_MM)
    far_mask  = (d_arr >= TAU_FAR_LO_MM) & (d_arr <= TAU_FAR_HI_MM)

    tau_near = float(t_arr[near_mask].mean()) if near_mask.sum() > 0 else 0.0
    if mid_mask.sum() == 0 or far_mask.sum() == 0:
        return np.nan, tau_near, np.nan, np.nan
    tau_mid = float(t_arr[mid_mask].mean())
    tau_far = float(t_arr[far_mask].mean())
    if tau_mid <= 0:
        return np.nan, tau_near, tau_mid, tau_far
    return float((tau_far - tau_near) / tau_mid), tau_near, tau_mid, tau_far


def compute_lab_tau50_zone(trial_id):
    d, t = _get_tau50_profile(trial_id)
    if d is None:
        return np.nan, np.nan, np.nan, np.nan
    return _tau50_zone_from_profile(d, t)


def compute_rsr_tau50_zone(sample):
    d, t = _load_rsr_tau50(sample)
    if d is None:
        return np.nan, np.nan, np.nan, np.nan
    return _tau50_zone_from_profile(d, t)


def _tau50_zone_firstbin_from_profile(d_arr, t_arr):
    """Zone metric using nearest data bin as near anchor (no imputation)."""
    mid_mask = (d_arr >= TAU_MID_LO_MM) & (d_arr <= TAU_MID_HI_MM)
    far_mask = (d_arr >= TAU_FAR_LO_MM) & (d_arr <= TAU_FAR_HI_MM)
    if mid_mask.sum() == 0 or far_mask.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    near_idx = int(np.argmin(d_arr))
    tau_near = float(t_arr[near_idx])
    d_near   = float(d_arr[near_idx])
    tau_mid  = float(t_arr[mid_mask].mean())
    tau_far  = float(t_arr[far_mask].mean())
    if tau_mid <= 0:
        return np.nan, tau_near, tau_mid, tau_far, d_near
    return float((tau_far - tau_near) / tau_mid), tau_near, tau_mid, tau_far, d_near


def _tau50_zone_extrap_from_profile(d_arr, t_arr):
    """Zone metric with near anchor extrapolated to d=0 via OLS."""
    mid_mask = (d_arr >= TAU_MID_LO_MM) & (d_arr <= TAU_MID_HI_MM)
    far_mask = (d_arr >= TAU_FAR_LO_MM) & (d_arr <= TAU_FAR_HI_MM)
    if len(d_arr) < 2 or mid_mask.sum() == 0 or far_mask.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    sl, ic, _, _, _ = stats.linregress(d_arr, t_arr)
    tau_near = max(float(ic), 0.0)          # intercept at d=0, clipped to ≥0
    tau_mid  = float(t_arr[mid_mask].mean())
    tau_far  = float(t_arr[far_mask].mean())
    if tau_mid <= 0:
        return np.nan, tau_near, tau_mid, tau_far
    return float((tau_far - tau_near) / tau_mid), tau_near, tau_mid, tau_far


def compute_lab_tau50_zone_extrap(trial_id):
    d, t = _get_tau50_profile(trial_id)
    if d is None:
        return np.nan, np.nan, np.nan, np.nan
    return _tau50_zone_extrap_from_profile(d, t)


def compute_rsr_tau50_zone_extrap(sample):
    d, t = _load_rsr_tau50(sample)
    if d is None:
        return np.nan, np.nan, np.nan, np.nan
    return _tau50_zone_extrap_from_profile(d, t)


def compute_lab_tau50_zone_fb(trial_id):
    d, t = _get_tau50_profile(trial_id)
    if d is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return _tau50_zone_firstbin_from_profile(d, t)


def compute_rsr_tau50_zone_fb(sample):
    d, t = _load_rsr_tau50(sample)
    if d is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return _tau50_zone_firstbin_from_profile(d, t)


def compute_lab_tau50_slope(trial_id, d_cap=None):
    d, t = _get_tau50_profile(trial_id)
    if d is None or len(d) < 3:
        return np.nan
    if d_cap is not None:
        mask = d <= d_cap
        d, t = d[mask], t[mask]
    if len(d) < 3:
        return np.nan
    sl, _, _, _, _ = stats.linregress(d, t)
    return float(sl)


def compute_rsr_dstar(drops_df, sample):
    """Constrained Hill fit to spatial r_eq profile; dry-zone bins anchored at 0."""
    sub = drops_df[(drops_df['sample'] == sample) &
                   (drops_df['r_eq_mm'] > 0) &
                   (drops_df['dist_mm'] <= RSR_CAP_MM)].copy()
    if len(sub) < 10:
        print(f'  {sample}: too few droplets')
        return np.nan

    d_max = sub['dist_mm'].max()
    bins  = np.arange(0, d_max + RSR_SIZE_BIN_MM, RSR_SIZE_BIN_MM)
    sub['bin'] = pd.cut(sub['dist_mm'], bins=bins, labels=False)

    d_vals, r_vals = [], []
    found_populated = False
    for b in range(len(bins) - 1):
        bsub   = sub[sub['bin'] == b]
        center = float(bins[b] + RSR_SIZE_BIN_MM / 2)
        if len(bsub) >= RSR_MIN_SIZE_BIN:
            found_populated = True
            d_vals.append(center)
            r_vals.append(float(bsub['r_eq_mm'].mean() * 1000))
        elif not found_populated:
            d_vals.append(center)
            r_vals.append(0.0)

    r_arr = np.array(r_vals)
    if float(r_arr.max() - r_arr.min()) < RSR_FLAT_RANGE_UM:
        print(f'  {sample}: flat r_eq → d*=0.0')
        return 0.0

    K, r2 = _fit_hill_constrained(d_vals, r_vals, label=sample)
    tag = f'd*={K:.3f} mm  R²={r2:.3f}' if np.isfinite(K) and K > 0 else \
          ('d*=0.0 (flat)' if K == 0.0 else 'd*=NaN')
    print(f'  {sample:<16}: {tag}  n={len(d_vals)}')
    return K


def compute_rsr_delta(drops_df, sample):
    sub   = drops_df[drops_df['sample'] == sample].copy()
    dists = sub['dist_mm'].dropna().values * 1000  # → µm
    dists = dists[dists > 0]
    if len(dists) < 10:
        return np.nan
    return float(np.percentile(dists, 1))


def compute_rsr_zone_metric(drops_df, sample):
    sub = drops_df[drops_df['sample'] == sample].copy()
    sub = sub[sub['r_eq_mm'] > 0].copy()
    if len(sub) < 10:
        return np.nan
    near = sub[sub['dist_mm'] <  NEAR_MAX_MM]['r_eq_mm']
    mid  = sub[(sub['dist_mm'] >= MID_LO_MM) & (sub['dist_mm'] <= MID_HI_MM)]['r_eq_mm']
    far  = sub[(sub['dist_mm'] >= FAR_LO_MM) & (sub['dist_mm'] <= FAR_HI_MM)]['r_eq_mm']
    R_near = near.mean() if len(near) >= 1 else 0.0
    if len(mid) < 1 or len(far) < 1:
        return np.nan
    R_mid = mid.mean()
    R_far = far.mean()
    if R_mid <= 0:
        return np.nan
    return float((R_far - R_near) / R_mid)


DZV_MIN_BINS  = 4
DZV_MIN_ALIVE = 30
DZV_DROP_FRAC = 0.40

def compute_lab_dzv(trial_id):
    path = HG_TRACK / f'{trial_id}_track_histories.csv'
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= 3].copy()

    non_cens = df[~df['censored']]
    if len(non_cens) < DZV_MIN_ALIVE:
        return np.nan

    t_start = max(float(T_SEED_S), float(non_cens['t_death_s'].min()))
    t_end   = float(non_cens['t_death_s'].max())
    if (t_end - t_start) < 30.0 * DZV_MIN_BINS:
        return np.nan

    time_edges  = np.arange(t_start, t_end + 30.0, 30.0)
    delta_vals, time_mids, n_alive_vals = [], [], []

    for i in range(len(time_edges) - 1):
        t     = time_edges[i]
        alive = df[(df['censored']) | (df['t_death_s'] > t)]
        if len(alive) < DZV_MIN_ALIVE:
            break
        delta_vals.append(float(np.percentile(alive['distance_um'], 5)))
        time_mids.append(float((time_edges[i] + time_edges[i + 1]) / 2.0))
        n_alive_vals.append(len(alive))

    if len(delta_vals) < DZV_MIN_BINS:
        return np.nan

    valid_end = len(delta_vals)
    for j in range(1, len(n_alive_vals)):
        if (n_alive_vals[j - 1] - n_alive_vals[j]) / max(n_alive_vals[j - 1], 1) > DZV_DROP_FRAC:
            valid_end = j
            break

    dv = np.array(delta_vals[:valid_end])
    if len(dv) >= 4:
        diffs = np.diff(dv)
        iqr   = float(np.percentile(diffs, 75) - np.percentile(diffs, 25))
        for j, d in enumerate(diffs):
            if abs(d) > 3.0 * iqr + 1.0:
                valid_end = min(valid_end, j + 1)
                break

    if valid_end < DZV_MIN_BINS:
        return np.nan

    t_arr = np.array(time_mids[:valid_end])
    d_arr = np.array(delta_vals[:valid_end])
    slope, _, _, _, _ = stats.linregress(t_arr, d_arr)
    return float(slope * 60.0)


RSR_DZV_MAP = {
    'RSR1': 'RSR1', 'RSR2': 'RSR2', 'RSR10': 'RSR7',
    'RSR3': 'RSRDiseased3', 'RSR5': 'RSRDiseased5', 'RSR6': 'RSRDiseased6',
}
RSR_DZV_R2_MIN = 0.30

def load_rsr_dzv():
    if not RSR_DZV_CSV.exists():
        return {}
    df  = pd.read_csv(RSR_DZV_CSV)
    out = {}
    for _, row in df.iterrows():
        tid = RSR_DZV_MAP.get(row['sample'])
        if tid is None:
            continue
        vel = float(row['front_vel_um_min'])
        r2  = float(row['front_vel_r2']) if pd.notna(row['front_vel_r2']) else 0.0
        out[tid] = vel if (vel > 0 and r2 >= RSR_DZV_R2_MIN) else np.nan
    return out


MM = 1 / 25.4
TS = 6.5; LS = 7.5; PL = 10.0; LW = 0.6

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          TS,
    'axes.linewidth':     LW,
    'xtick.major.width':  LW, 'ytick.major.width': LW,
    'xtick.major.size':   3.0, 'ytick.major.size': 3.0,
    'xtick.direction':    'out', 'ytick.direction': 'out',
    'svg.fonttype':       'none',
})


def _scatter_ax(ax, df, xcol, ycol, xlbl, ylbl):
    all_x, all_y = [], []
    for pass_rsr in (False, True):
        for grp, gdf in df.groupby('group'):
            is_rsr = grp in RSR_GROUPS
            if is_rsr != pass_rsr:
                continue
            both = gdf[[xcol, ycol]].dropna()
            if len(both) == 0:
                continue
            if is_rsr:
                ax.scatter(both[xcol], both[ycol], s=22, c=RSR_COLOR,
                           marker='o', edgecolors='white', linewidths=0.4,
                           zorder=4, label='Red star rust')
            else:
                ax.scatter(both[xcol], both[ycol], s=10, c='#BBBBBB',
                           marker='o', edgecolors='none', alpha=0.6,
                           zorder=2, label='Lab' if grp == 'Agar' else '_nolegend_')
            all_x.extend(both[xcol].tolist())
            all_y.extend(both[ycol].tolist())

    ax.set_xlabel(xlbl, fontsize=LS)
    ax.set_ylabel(ylbl, fontsize=LS)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    xa, ya = np.array(all_x), np.array(all_y)
    if len(xa) >= 3:
        sl, ic, r, p, _ = stats.linregress(xa, ya)
        xfit = np.linspace(xa.min(), xa.max(), 100)
        ax.plot(xfit, ic + sl * xfit, color='#444444', lw=0.9, ls='--', zorder=2)
        pstr = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
        ax.text(0.97, 0.05, f'$r^2$={r**2:.2f}, {pstr}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=TS - 0.5, color='#444444')
    return xa, ya


def main():
    print('=== Step 2: d*, delta (consistent), zone_metric, norm_slope ===\n')

    drops    = pd.read_csv(RAW_DIR / 'droplets_calibrated_mm.csv')
    rsr_meta = pd.read_csv(RAW_DIR / 'rsr_finalized_metrics.csv')
    print(f'RSR droplets: {len(drops)} rows, {drops["sample"].nunique()} samples')

    print('\n--- RSR metrics ---')
    rsr_rows = []
    for samp in RSR_FOLDERS:
        delta_um = compute_rsr_delta(drops, samp)
        zone_m   = compute_rsr_zone_metric(drops, samp)
        row_meta = rsr_meta[rsr_meta['sample'] == samp]
        if len(row_meta) == 1:
            dRdr   = float(row_meta['dR_dr_um_per_mm'].values[0])
            deltaR = float(row_meta['delta_R_um'].values[0])
        else:
            dRdr = deltaR = np.nan
        dtau_dr, tau_range, tau_fold, dstar_tau = compute_rsr_tau50_metrics(samp)
        zone_val, _, _, tau_far_rsr = compute_rsr_tau50_zone(samp)
        zone_fb_val, _, _, _, d_near_rsr = compute_rsr_tau50_zone_fb(samp)
        zone_ex_val, _, _, _ = compute_rsr_tau50_zone_extrap(samp)
        d_rsr_tmp, t_rsr_tmp = _load_rsr_tau50(samp)
        if d_rsr_tmp is not None:
            mask_2 = d_rsr_tmp <= TAU_SLOPE_CAP_MM
            sl_2mm_rsr = float(stats.linregress(d_rsr_tmp[mask_2], t_rsr_tmp[mask_2])[0]) \
                         if mask_2.sum() >= 3 else np.nan
        else:
            sl_2mm_rsr = np.nan
        dtau_dr_norm_rsr = float(dtau_dr / tau_far_rsr) \
                           if (np.isfinite(dtau_dr) and np.isfinite(tau_far_rsr) and tau_far_rsr > 0) \
                           else np.nan
        # RSR uses extrapolated zone metric (wider zones fit 400µm pixel-survival bins)
        rsr_zone_plotted = zone_ex_val
        print(f'  {samp:<16}: delta={delta_um:.0f}µm  dR_dr={dRdr:.2f}µm/mm  ΔR={deltaR:.1f}µm  '
              f'τ₅₀_zone={rsr_zone_plotted:.3f}' if np.isfinite(rsr_zone_plotted) else
              f'  {samp:<16}: delta={delta_um:.0f}µm  dR_dr={dRdr:.2f}µm/mm  ΔR={deltaR:.1f}µm  τ₅₀_zone=NaN')
        rsr_rows.append({
            'trial_id':         samp,
            'group':            RSR_GROUP[samp],
            'system':           'Leaf',
            'dstar_mm':         np.nan,
            'dstar_tau50_mm':   dstar_tau,
            'delta_um':         delta_um,
            'zone_metric':      zone_m,
            'dRdr_um_per_mm':   dRdr,
            'deltaR_um':        deltaR,
            'tau50_range_min':  tau_range,
            'tau50_integral':   np.nan,
            'tau50_fold':       tau_fold,
            'dtau50_dr':        dtau_dr,
            'tau50_zone':       rsr_zone_plotted,
            'tau50_zone_fb':    zone_fb_val,
            'tau50_zone_ex':    zone_ex_val,
            'dtau50_dr_2mm':    sl_2mm_rsr,
            'dtau50_dr_norm':   dtau_dr_norm_rsr,
            'dzv_um_per_min':   np.nan,
        })
    rsr_df = pd.DataFrame(rsr_rows)

    univ = pd.read_csv(UNIV_CSV)
    lab  = univ[univ['system'] != 'Leaf'].copy()
    print(f'\n--- Lab trials: {len(lab)} rows ---')

    print('\n--- Lab delta (1st percentile of distance_um at 15 min) ---')
    lab = lab.copy()
    lab_delta_p1 = {}
    for _, row in lab.iterrows():
        tid = row['trial_id']
        d   = compute_lab_delta_p1(tid)
        lab_delta_p1[tid] = d
        print(f'  {tid:<12}: p1={d:.0f}µm  (raycast={row["delta_um"]:.0f}µm)')
    lab['delta_um_p1'] = lab['trial_id'].map(lab_delta_p1)

    print('\n--- Lab d* (KM τ₅₀ Hill fit) ---')
    dstar_map = {}
    for _, row in lab.iterrows():
        tid = row['trial_id']
        dstar_map[tid] = compute_lab_dstar(tid)
    lab['dstar_mm'] = lab['trial_id'].map(dstar_map)

    print('\n--- Lab τ₅₀ range / integral / fold / d*_tau50 (n=30 tracked) ---')
    tau_range_map = {}; tau_integ_map = {}; tau_fold_map = {}; dstar_tau_map = {}
    for _, row in lab.iterrows():
        tid = row['trial_id']
        rng, integ, fold = compute_lab_tau50_metrics(tid)
        d_vals, tau_vals = _get_tau50_profile(tid)
        if d_vals is not None:
            K, r2, A, n = _fit_hill(d_vals, tau_vals, label=f'{tid}_tau50')
            dstar_tau_map[tid] = K if np.isfinite(K) else np.nan
        else:
            dstar_tau_map[tid] = np.nan
        tau_range_map[tid] = rng
        tau_integ_map[tid] = integ
        tau_fold_map[tid]  = fold
        if np.isfinite(rng):
            print(f'  {tid:<12}: range={rng:.1f} min  fold={fold:.2f}x  d*_τ={dstar_tau_map[tid]:.3f}mm' if np.isfinite(dstar_tau_map[tid]) else f'  {tid:<12}: range={rng:.1f} min  fold={fold:.2f}x  d*_τ=NaN')
    lab['tau50_range_min']  = lab['trial_id'].map(tau_range_map)
    lab['tau50_integral']   = lab['trial_id'].map(tau_integ_map)
    lab['tau50_fold']       = lab['trial_id'].map(tau_fold_map)
    lab['dstar_tau50_mm']   = lab['trial_id'].map(dstar_tau_map)

    lab['dtau50_dr'] = lab['trial_id'].map(
        dict(zip(pd.read_csv(UNIV_CSV)['trial_id'],
                 pd.read_csv(UNIV_CSV)['dtau50_dr'])))

    print('\n--- Lab τ₅₀ zone / restricted slope / normalized slope ---')
    tau50_zone_map = {}; tau50_zone_fb_map = {}; tau50_zone_ex_map = {}
    dtau50_dr_2mm_map = {}; dtau50_dr_norm_map = {}
    for _, row in lab.iterrows():
        tid = row['trial_id']
        # Use Figure 2K's function for tau50_zone (guarantees identical values)
        fig2k_val = _fig2k_tau_zone(tid)
        tau50_zone_map[tid] = fig2k_val if fig2k_val is not None else np.nan
        # Keep step2's own variants for diagnostics
        zone_val, _, _, tau_far_lab = compute_lab_tau50_zone(tid)
        zone_fb_val, _, _, _, d_near_lab = compute_lab_tau50_zone_fb(tid)
        zone_ex_val, _, _, _ = compute_lab_tau50_zone_extrap(tid)
        sl_2mm = compute_lab_tau50_slope(tid, d_cap=TAU_SLOPE_CAP_MM)
        sl_full = compute_lab_tau50_slope(tid)
        dtau_dr_norm = float(sl_full / tau_far_lab) \
                       if (np.isfinite(sl_full) and np.isfinite(tau_far_lab) and tau_far_lab > 0) \
                       else np.nan
        tau50_zone_fb_map[tid]  = zone_fb_val
        tau50_zone_ex_map[tid]  = zone_ex_val
        dtau50_dr_2mm_map[tid]  = sl_2mm
        dtau50_dr_norm_map[tid] = dtau_dr_norm
        d_near_str = f'{d_near_lab:.2f}mm' if np.isfinite(d_near_lab) else 'NaN'
        zv = tau50_zone_map[tid]
        if np.isfinite(zv):
            print(f'  {tid:<12}: τ₅₀_zone={zv:.3f}  '
                  f'τ₅₀_zone_fb={zone_fb_val:.3f} (near@{d_near_str})  '
                  f'τ₅₀_zone_ex={zone_ex_val:.3f}  '
                  f'slope_2mm={sl_2mm:.3f}  norm={dtau_dr_norm:.4f}')
        else:
            print(f'  {tid:<12}: τ₅₀_zone=NaN  τ₅₀_zone_fb=NaN  τ₅₀_zone_ex=NaN')
    lab['tau50_zone']     = lab['trial_id'].map(tau50_zone_map)
    lab['tau50_zone_fb']  = lab['trial_id'].map(tau50_zone_fb_map)
    lab['tau50_zone_ex']  = lab['trial_id'].map(tau50_zone_ex_map)
    lab['dtau50_dr_2mm']  = lab['trial_id'].map(dtau50_dr_2mm_map)
    lab['dtau50_dr_norm'] = lab['trial_id'].map(dtau50_dr_norm_map)

    print('\n--- Lab dR/dr and ΔR (size profile at 15 min) ---')
    dRdr_map = {}; deltaR_map = {}
    for _, row in lab.iterrows():
        tid = row['trial_id']
        sl  = compute_lab_dRdr(tid)
        dr  = compute_lab_delta_R(tid)
        dRdr_map[tid]   = sl
        deltaR_map[tid] = dr
        print(f'  {tid:<12}: dR/dr={sl:.2f} µm/mm  ΔR={dr:.1f} µm')
    lab['dRdr_um_per_mm'] = lab['trial_id'].map(dRdr_map)
    lab['deltaR_um']      = lab['trial_id'].map(deltaR_map)

    # Keep raycast delta from universal_metrics.csv (matches Figures 2 & 3)
    # lab['delta_um_p1'] is available as a diagnostic column but not used for plotting

    print('\n--- Lab dry zone velocity (dδ/dt from track_histories) ---')
    dzv_map = {}
    for _, row in lab.iterrows():
        tid = row['trial_id']
        v   = compute_lab_dzv(tid)
        dzv_map[tid] = v
        if np.isfinite(v):
            print(f'  {tid:<12}: DZV={v:.1f} µm/min')
        else:
            print(f'  {tid:<12}: DZV=NaN')
    lab['dzv_um_per_min'] = lab['trial_id'].map(dzv_map)

    print('\n--- RSR dry zone velocity (from rsr_dry_zone_summary.csv) ---')
    rsr_dzv = load_rsr_dzv()
    for tid, v in rsr_dzv.items():
        if np.isfinite(v):
            print(f'  {tid:<16}: DZV={v:.1f} µm/min')
        else:
            print(f'  {tid:<16}: DZV=NaN (filtered)')
    rsr_df['dzv_um_per_min'] = rsr_df['trial_id'].map(rsr_dzv)

    keep = ['trial_id', 'group', 'system',
            'dstar_mm', 'dstar_tau50_mm', 'delta_um', 'zone_metric',
            'dRdr_um_per_mm', 'deltaR_um',
            'tau50_range_min', 'tau50_integral', 'tau50_fold', 'dtau50_dr',
            'tau50_zone', 'tau50_zone_fb', 'tau50_zone_ex', 'dtau50_dr_2mm', 'dtau50_dr_norm',
            'dzv_um_per_min']
    all_df = pd.concat([lab[keep], rsr_df[keep]], ignore_index=True)

    out_csv = OUTPUT_DIR / 'rsr_and_lab_dstar_metrics.csv'
    all_df.to_csv(out_csv, index=False)
    print(f'\n-> Saved {out_csv}  ({len(all_df)} rows)')

    print('\n=== Correlation statistics ===')
    pairs = [
        ('dstar_mm',       'delta_um',    'd* (mm)',            'δ (µm)',       35),
        ('zone_metric',    'delta_um',    'zone_metric',        'δ (µm)',       41),
        ('dstar_mm',       'zone_metric', 'd* (mm)',            'zone_metric',  35),
        ('dRdr_um_per_mm', 'delta_um',    'dR/dr (µm/mm)',      'δ (µm)',       41),
        ('dRdr_um_per_mm', 'zone_metric', 'dR/dr (µm/mm)',      'zone_metric',  41),
        ('deltaR_um',      'delta_um',    'ΔR (µm)',            'δ (µm)',       41),
        ('deltaR_um',      'zone_metric', 'ΔR (µm)',            'zone_metric',  41),
        ('tau50_range_min', 'delta_um',    'τ₅₀ range (min)',   'δ (µm)',      36),
        ('tau50_range_min', 'zone_metric', 'τ₅₀ range (min)',   'zone_metric', 36),
        ('tau50_range_min', 'dstar_mm',    'τ₅₀ range (min)',   'd* (mm)',     30),
        ('tau50_fold',      'delta_um',    'τ₅₀ fold',          'δ (µm)',      36),
        ('tau50_fold',      'dstar_mm',    'τ₅₀ fold',          'd* (mm)',     30),
        ('dtau50_dr',       'delta_um',    'dτ₅₀/dr (min/mm)',  'δ (µm)',      36),
        ('dtau50_dr',       'zone_metric', 'dτ₅₀/dr (min/mm)',  'zone_metric', 36),
        ('dtau50_dr',       'dstar_mm',    'dτ₅₀/dr (min/mm)',  'd* (mm)',     30),
        ('dstar_tau50_mm',  'delta_um',    'd*_τ (mm)',         'δ (µm)',      36),
        ('dstar_tau50_mm',  'zone_metric', 'd*_τ (mm)',         'zone_metric', 36),
        ('dstar_tau50_mm',  'dstar_mm',    'd*_τ (mm)',         'd*_spatial',  30),
        ('tau50_zone',      'delta_um',    'τ₅₀_zone',          'δ (µm)',      41),
        ('tau50_zone',      'zone_metric', 'τ₅₀_zone',          'zone_metric', 41),
        ('tau50_zone',      'dstar_mm',    'τ₅₀_zone',          'd* (mm)',     35),
        ('tau50_zone_fb',   'delta_um',    'τ₅₀_zone_fb',       'δ (µm)',      41),
        ('tau50_zone_fb',   'zone_metric', 'τ₅₀_zone_fb',       'zone_metric', 41),
        ('tau50_zone_fb',   'dstar_mm',    'τ₅₀_zone_fb',       'd* (mm)',     35),
        ('tau50_zone_ex',   'delta_um',    'τ₅₀_zone_ex',       'δ (µm)',      41),
        ('tau50_zone_ex',   'zone_metric', 'τ₅₀_zone_ex',       'zone_metric', 41),
        ('tau50_zone_ex',   'dstar_mm',    'τ₅₀_zone_ex',       'd* (mm)',     35),
        ('dtau50_dr_2mm',   'delta_um',    'dτ₅₀/dr ≤2mm',      'δ (µm)',      41),
        ('dtau50_dr_2mm',   'zone_metric', 'dτ₅₀/dr ≤2mm',      'zone_metric', 41),
        ('dtau50_dr_2mm',   'dstar_mm',    'dτ₅₀/dr ≤2mm',      'd* (mm)',     35),
        ('dtau50_dr_norm',  'delta_um',    'dτ₅₀/dr norm',      'δ (µm)',      41),
        ('dtau50_dr_norm',  'zone_metric', 'dτ₅₀/dr norm',      'zone_metric', 41),
        ('dtau50_dr_norm',  'dstar_mm',    'dτ₅₀/dr norm',      'd* (mm)',     35),
        ('dzv_um_per_min',  'delta_um',    'DZV (µm/min)',      'δ (µm)',      37),
        ('dzv_um_per_min',  'zone_metric', 'DZV (µm/min)',      'zone_metric', 37),
        ('dzv_um_per_min',  'dstar_mm',    'DZV (µm/min)',      'd* (mm)',     32),
    ]
    for xc, yc, xl, yl, expected_n in pairs:
        sub = all_df[[xc, yc]].dropna()
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub[xc], sub[yc])
            pstr = 'p<0.001' if p < 0.001 else f'p={p:.3f}'
            print(f'  {xl:<22} vs {yl:<16}: r²={r**2:.3f}  {pstr}  n={len(sub)}'
                  f'  (expected {expected_n})')

    TAU_LBL  = (r'$(\tau_{50,\rm far}-\tau_{50,\rm near})'
                r'/\tau_{50,\rm mid}$')
    ZONE_LBL = r'$(R_{\rm far}-R_{\rm near})/R_{\rm mid}$'
    DELTA_LBL = r'$\delta$ (µm)'

    fig, (axA, axB, axD) = plt.subplots(1, 3, figsize=(195 * MM, 62 * MM))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.18, wspace=0.50)

    TAU_ZONE_CAP = 5.0
    dfA = all_df[all_df['tau50_zone'].isna() | (all_df['tau50_zone'] <= TAU_ZONE_CAP)]
    _scatter_ax(axA, dfA, 'delta_um', 'tau50_zone', DELTA_LBL, TAU_LBL)
    axA.margins(x=0.06, y=0.06)
    axA.text(-0.28, 1.06, 'A', transform=axA.transAxes,
             fontsize=PL, fontweight='bold', va='top')

    _scatter_ax(axB, all_df, 'tau50_zone', 'zone_metric', TAU_LBL, ZONE_LBL)
    axB.text(-0.28, 1.06, 'B', transform=axB.transAxes,
             fontsize=PL, fontweight='bold', va='top')

    _scatter_ax(axD, all_df, 'delta_um', 'zone_metric', DELTA_LBL, ZONE_LBL)
    axD.text(-0.28, 1.06, 'D', transform=axD.transAxes,
             fontsize=PL, fontweight='bold', va='top')

    handles, labels_leg = [], []
    for ax in (axA, axB, axD):
        for hi, li in zip(*ax.get_legend_handles_labels()):
            if li not in labels_leg:
                handles.append(hi); labels_leg.append(li)
    fig.legend(handles, labels_leg, loc='lower center', ncol=2,
               fontsize=TS - 0.5, frameon=False,
               bbox_to_anchor=(0.54, -0.03))
    fig.subplots_adjust(bottom=0.28)

    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'panels_ABD{ext}', **kw)
    plt.close(fig)
    print(f'\n-> Saved {OUTPUT_DIR}/panels_ABD.*')

    print('\n=== Delta consistency check: p1 vs raycast ===')
    from scipy.stats import pearsonr as _pr
    lab_both = lab[['trial_id', 'delta_um', 'delta_um_p1']].dropna()
    # delta_um here is already p1; recover raycast from univ
    univ2 = pd.read_csv(UNIV_CSV)
    univ2 = univ2[univ2['system'] != 'Leaf'][['trial_id', 'delta_um']].rename(
        columns={'delta_um': 'delta_raycast'})
    cmp = lab_both.merge(univ2, on='trial_id')
    r_cmp, p_cmp = _pr(cmp['delta_um'], cmp['delta_raycast'])
    print(f'  p1 vs raycast: r²={r_cmp**2:.3f}  p={p_cmp:.2e}  n={len(cmp)}')


if __name__ == '__main__':
    main()
