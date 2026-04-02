#!/usr/bin/env python3
"""Scatter subplots of r_eq vs distance for 6 leaf condensation trials.

Each panel also overlays the binned mean profile for the Healthy group
(red) and the Diseased group (dark blue), averaged across the 3 samples
in each group.  Error bands show ±1 SEM across the 3 samples.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR    = THIS_DIR.parent / 'raw_data'

MM = 1 / 25.4
TS = 6.5;  LS = 7.5;  PL = 10.0
LW = 0.6

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

HEALTHY  = ['RSR1', 'RSR2', 'RSR7']
DISEASED = ['RSRDiseased3', 'RSRDiseased5', 'RSRDiseased6']

GROUP_COLOR = {'Healthy': '#E74C3C', 'Diseased': '#2E4057'}

NEAR_MAX = 0.5
MID_LO, MID_HI = 0.9, 1.1
FAR_LO,  FAR_HI = 1.9, 2.1

ZONE_COLOR = {'near': '#E74C3C', 'mid': '#3498DB', 'far': '#27AE60'}
ZONE_ALPHA = 0.10

_T7_RAW = Path('/Volumes/T7/Fungal Hygroscopy/RAW/RSR RAW')
T7_RSR = _T7_RAW if _T7_RAW.exists() else RAW_DIR / 'rsr_raw'

BIN_MM      = 0.2   # bin width for group mean profiles (finer → data closer to 0)
CAP_MM      = 4.0   # max distance
MIN_BIN     = 3     # min droplets per bin to include in mean
MIN_SAMPLES = 1     # min samples per group per bin to plot (no SEM if only 1)


def _group_mean_profile(df, samples):
    """Per-sample binned means averaged across samples → (centers, mean, sem)."""
    bins    = np.arange(0, CAP_MM + BIN_MM, BIN_MM)
    centers = (bins[:-1] + bins[1:]) / 2
    sample_means = []
    for samp in samples:
        sub = df[(df['sample'] == samp) & (df['r_eq_mm'] > 0) &
                 (df['dist_mm'] <= CAP_MM)].copy()
        row = []
        for i in range(len(bins) - 1):
            bsub = sub[(sub['dist_mm'] >= bins[i]) & (sub['dist_mm'] < bins[i + 1])]
            row.append(bsub['r_eq_mm'].mean() if len(bsub) >= MIN_BIN else np.nan)
        sample_means.append(row)
    arr    = np.array(sample_means, dtype=float)
    n_samp = (~np.isnan(arr)).sum(axis=0)
    mean   = np.nanmean(arr, axis=0)
    sem    = np.where(n_samp >= 2,
                      np.nanstd(arr, axis=0, ddof=0) / np.sqrt(n_samp.clip(1)),
                      np.nan)
    mean[n_samp < MIN_SAMPLES] = np.nan
    return centers, mean, sem


def main():
    df = pd.read_csv(RAW_DIR / 'droplets_calibrated_mm.csv')
    print(f'Loaded {len(df)} droplets from {df["sample"].nunique()} samples')

    group_profiles = {
        'Healthy':  _group_mean_profile(df, HEALTHY),
        'Diseased': _group_mean_profile(df, DISEASED),
    }

    fig, axes = plt.subplots(3, 2, figsize=(100 * MM, 120 * MM),
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.20,
                        left=0.14, right=0.96, top=0.92, bottom=0.10)

    axes[0, 0].set_title('Healthy', fontsize=LS, fontweight='bold', pad=6)
    axes[0, 1].set_title('Diseased', fontsize=LS, fontweight='bold', pad=6)

    for col, trial_list in enumerate([HEALTHY, DISEASED]):
        for row, sample in enumerate(trial_list):
            ax = axes[row, col]
            sub = df[df['sample'] == sample].copy()

            x = sub['dist_mm'].values
            y = sub['r_eq_mm'].values

            ax.scatter(x, y, s=3, c='#CCCCCC', alpha=0.35,
                       edgecolors='none', rasterized=True, zorder=1)

            bins    = np.arange(0, CAP_MM + BIN_MM, BIN_MM)
            centers = (bins[:-1] + bins[1:]) / 2
            bx_vals, by_vals = [], []
            for i in range(len(bins) - 1):
                mask = (x >= bins[i]) & (x < bins[i + 1]) & np.isfinite(y)
                if mask.sum() >= MIN_BIN:
                    bx_vals.append(centers[i])
                    by_vals.append(float(y[mask].mean()))
            if len(bx_vals) >= 2:
                ax.plot(bx_vals, by_vals, color='#555555', lw=1.1,
                        zorder=2, marker='o', ms=2.0, markeredgewidth=0)

            for grp, (cx, mn, sem) in group_profiles.items():
                c   = GROUP_COLOR[grp]
                ok  = np.isfinite(mn)
                ax.plot(cx[ok], mn[ok], color=c, lw=1.4, zorder=4,
                        label=grp if (row == 0 and col == 0) else '_nolegend_')
                ax.fill_between(cx[ok], (mn - sem)[ok], (mn + sem)[ok],
                                color=c, alpha=0.15, zorder=3, linewidth=0)

            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)

    fig.text(0.55, 0.02, 'Distance from boundary (mm)',
             ha='center', fontsize=LS)
    fig.text(0.02, 0.52, r'$R$ (mm)',
             ha='center', va='center', rotation=90, fontsize=LS)

    axes[0, 0].text(-0.35, 1.15, 'B', transform=axes[0, 0].transAxes,
                    fontsize=PL, fontweight='bold', va='top')
    axes[0, 0].legend(fontsize=TS - 1, frameon=False, loc='upper left',
                      handlelength=1.2, borderpad=0.3)

    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'panel_B{ext}', **kw)
    plt.close(fig)
    print(f'\nSaved to {OUTPUT_DIR}/panel_B.*')


def _group_normalized_profile(df, samples):
    """Normalize each sample's binned r_eq by its far-field mean (1.6-2.4 mm)."""
    FAR_LO, FAR_HI = 1.6, 2.4
    bins    = np.arange(0, CAP_MM + BIN_MM, BIN_MM)
    centers = (bins[:-1] + bins[1:]) / 2
    normed_rows = []
    for samp in samples:
        sub = df[(df['sample'] == samp) & (df['r_eq_mm'] > 0) &
                 (df['dist_mm'] <= CAP_MM)].copy()
        far = sub[(sub['dist_mm'] >= FAR_LO) & (sub['dist_mm'] <= FAR_HI)]['r_eq_mm']
        if len(far) < MIN_BIN:
            continue
        ref = far.mean()
        row = []
        for i in range(len(bins) - 1):
            bsub = sub[(sub['dist_mm'] >= bins[i]) & (sub['dist_mm'] < bins[i + 1])]
            row.append(bsub['r_eq_mm'].mean() / ref if len(bsub) >= MIN_BIN else np.nan)
        normed_rows.append(row)
    arr      = np.array(normed_rows, dtype=float)
    n_samp   = (~np.isnan(arr)).sum(axis=0)
    mean     = np.nanmean(arr, axis=0)
    sem      = np.where(n_samp >= 2,
                        np.nanstd(arr, axis=0, ddof=0) / np.sqrt(n_samp.clip(1)),
                        np.nan)
    mean[n_samp < MIN_SAMPLES] = np.nan
    return centers, mean, sem


def _add_zone_shading(ax, orientation='vertical'):
    zones = [
        ('near', 0,       NEAR_MAX),
        ('mid',  MID_LO,  MID_HI),
        ('far',  FAR_LO,  FAR_HI),
    ]
    for name, lo, hi in zones:
        ax.axvspan(lo, hi, color=ZONE_COLOR[name], alpha=ZONE_ALPHA,
                   zorder=0, linewidth=0)
        ax.text((lo + hi) / 2, ax.get_ylim()[1],
                name, ha='center', va='bottom',
                fontsize=TS - 1.5, color=ZONE_COLOR[name])


def plot_grouped(df):
    fig, ax = plt.subplots(figsize=(65 * MM, 60 * MM))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.88, bottom=0.18)

    for grp, samples in [('Healthy', HEALTHY), ('Diseased', DISEASED)]:
        cx, mn, sem = _group_normalized_profile(df, samples)
        c  = GROUP_COLOR[grp]
        ok = np.isfinite(mn)
        ax.plot(cx[ok], mn[ok], color=c, lw=1.4, label=grp, zorder=3)
        ax.fill_between(cx[ok], (mn - sem)[ok], (mn + sem)[ok],
                        color=c, alpha=0.18, zorder=2, linewidth=0)

    ax.axhline(1.0, color='#AAAAAA', lw=0.7, ls=':', zorder=1)
    ax.autoscale(enable=True, axis='y')
    ax.set_ylim(bottom=0)
    _add_zone_shading(ax)

    ax.set_xlabel('Distance from boundary (mm)', fontsize=LS)
    ax.set_ylabel(r'$R\,/\,R_{\rm far}$', fontsize=LS)
    ax.legend(fontsize=TS - 0.5, frameon=False, loc='lower right',
              handlelength=1.2, borderpad=0.3)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'panel_B_grouped{ext}', **kw)
    plt.close(fig)
    print(f'Saved to {OUTPUT_DIR}/panel_B_grouped.*')


_CAL_DIR = RAW_DIR / 'calibrations'
_CAL_NAME = {
    'RSR1': 'rsr1', 'RSR2': 'rsr2', 'RSR7': 'rsr10',
    'RSRDiseased3': 'rsr3', 'RSRDiseased5': 'rsr5', 'RSRDiseased6': 'rsr6',
}


def _compute_droplet_tracks(folder, cal_file, cache_csv=None):
    import os, glob, json as _json, h5py
    from scipy.ndimage import (binary_opening, label as _label,
                               center_of_mass, distance_transform_edt, sum as _ndisum,
                               generate_binary_structure, iterate_structure)
    from PIL import Image, ImageDraw

    if cache_csv and Path(cache_csv).exists():
        return pd.read_csv(cache_csv)

    DT            = 30
    MIN_BLOB_PX   = 50
    MAX_LINK_PX   = 40.0

    with open(cal_file) as f:
        bdata = _json.load(f)
    SCALE = bdata['calibration']['scale_um_per_px']
    bpts  = np.array(bdata['boundary_points_px'])
    IMG_W, IMG_H = bdata['image_size']

    bnd_img = Image.new('L', (IMG_W, IMG_H), 0)
    ImageDraw.Draw(bnd_img).line([tuple(p) for p in bpts.tolist()], fill=255, width=5)
    edt_um = distance_transform_edt(~(np.array(bnd_img) > 0)) * SCALE

    fungal = np.zeros((IMG_H, IMG_W), dtype=bool)
    mean_bx = float(np.mean(bpts[:, 0]))
    fside   = 'right' if mean_bx > IMG_W / 2 else 'left'
    for i in range(len(bpts) - 1):
        y1, y2 = int(bpts[i][1]), int(bpts[i+1][1])
        x1, x2 = int(bpts[i][0]), int(bpts[i+1][0])
        if y1 == y2:
            row  = min(y1, IMG_H - 1)
            xb   = min(x1, x2) if fside == 'right' else max(x1, x2)
            if fside == 'right': fungal[row, xb:] = True
            else:                fungal[row, :xb] = True
            continue
        for row in range(min(y1, y2), min(max(y1, y2) + 1, IMG_H)):
            frac = (row - y1) / (y2 - y1) if y2 != y1 else 0.0
            xb   = max(0, min(int(x1 + frac * (x2 - x1)), IMG_W - 1))
            if fside == 'right': fungal[row, xb:] = True
            else:                fungal[row, :xb] = True
    miny  = int(min(p[1] for p in bpts)); maxy = int(max(p[1] for p in bpts))
    first_x = int(bpts[0][0]);            last_x = int(bpts[-1][0])
    if fside == 'right':
        fungal[:miny, first_x:] = True; fungal[maxy:, last_x:] = True
    else:
        fungal[:miny, :first_x] = True; fungal[maxy:, :last_x] = True
    edt_um[fungal] = np.nan

    margin = int(IMG_H * 0.175)
    vert   = np.ones((IMG_H, IMG_W), dtype=bool)
    vert[:margin, :] = False; vert[IMG_H - margin:, :] = False
    edt_um[~vert] = np.nan

    evap_dir   = folder / 'Every 30s from Evap'
    jpgs       = sorted(glob.glob(str(evap_dir / '*.jpg')))
    struct     = generate_binary_structure(2, 1)
    struct_sm  = iterate_structure(struct, 2)
    raw_probs, valid_jpgs = [], []
    for jp in jpgs:
        h5p = evap_dir / (Path(jp).stem + '_Probabilities.h5')
        if not h5p.exists(): continue
        try:
            with h5py.File(h5p, 'r') as f:
                raw_probs.append(f['exported_data'][:, :, 0].copy())
            valid_jpgs.append(jp)
        except Exception:
            pass

    if not raw_probs:
        print('  No H5 files found — cannot track blobs')
        return pd.DataFrame(columns=['birth_s', 'death_s', 'duration_min', 'dist_um', 'censored'])

    valid_px    = ~fungal & vert
    norm_factor = max(float(np.percentile(np.concatenate([p[valid_px] for p in raw_probs]), 99)), 0.01)

    masks, times = [], []
    for idx, (jp, prob) in enumerate(zip(valid_jpgs, raw_probs)):
        m  = binary_opening(np.clip(prob / norm_factor, 0, 1) > 0.5, structure=struct_sm)
        lbl, nf = _label(m)
        if nf > 0:
            szs = np.array(_ndisum(m, lbl, range(1, nf + 1)))
            m[np.isin(lbl, np.where(szs < MIN_BLOB_PX)[0] + 1)] = False
        m &= ~fungal & vert
        masks.append(m)
        times.append(idx * DT)

    if masks:
        pigment = masks[-1].copy()
        masks   = [m & ~pigment for m in masks]

    active  = {}
    next_id = 0
    records = []

    for mask, t in zip(masks, times):
        lbl, nf = _label(mask)
        if nf == 0:
            for tid, info in list(active.items()):
                records.append({'birth_s': info['birth_s'], 'death_s': t,
                                'dist_um': info['dist_um'], 'censored': False})
            active.clear()
            continue

        cents = center_of_mass(mask, lbl, range(1, nf + 1))
        b_dists = [float(np.nanmean(edt_um[lbl == (i + 1)])) for i in range(nf)]

        matched_active, matched_blob = set(), set()
        for i, (cen, dist) in enumerate(zip(cents, b_dists)):
            if np.isnan(dist): continue
            best_tid, best_d = None, MAX_LINK_PX
            for tid, info in active.items():
                if tid in matched_active: continue
                dy = cen[0] - info['centroid'][0]
                dx = cen[1] - info['centroid'][1]
                dd = (dy * dy + dx * dx) ** 0.5
                if dd < best_d:
                    best_d, best_tid = dd, tid
            if best_tid is not None:
                active[best_tid]['centroid'] = cen
                matched_active.add(best_tid)
                matched_blob.add(i)

        for tid in [k for k in list(active) if k not in matched_active]:
            info = active.pop(tid)
            records.append({'birth_s': info['birth_s'], 'death_s': t,
                            'dist_um': info['dist_um'], 'censored': False})

        for i, (cen, dist) in enumerate(zip(cents, b_dists)):
            if i not in matched_blob and not np.isnan(dist):
                active[next_id] = {'centroid': cen, 'dist_um': dist, 'birth_s': t}
                next_id += 1

    last_t = times[-1] if times else 0.0
    for tid, info in active.items():
        records.append({'birth_s': info['birth_s'], 'death_s': last_t,
                        'dist_um': info['dist_um'], 'censored': True})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df['duration_min'] = (df['death_s'] - df['birth_s']) / 60.0
    df = df[df['duration_min'] >= 0].copy()
    if cache_csv:
        df.to_csv(cache_csv, index=False)
    print(f'  Tracked {len(df)} blobs  ({df["censored"].sum()} censored)')
    return df


def plot_rsr_km_example(sample='RSRDiseased5'):
    rsr_map = {
        'RSR1': T7_RSR / 'RSR 1', 'RSR2': T7_RSR / 'RSR 2',
        'RSR7': T7_RSR / 'RSR10',
        'RSRDiseased3': T7_RSR / 'RSR 3', 'RSRDiseased5': T7_RSR / 'RSR 5',
        'RSRDiseased6': T7_RSR / 'RSR 6',
    }
    folder   = rsr_map[sample]
    dens_csv = folder / 'Every 30s from Evap' / 'survival_analysis' / 'distance_band_density.csv'
    tau_csv  = folder / 'Every 30s from Evap' / 'survival_analysis' / 'tau50_by_distance.csv'

    if not dens_csv.exists() or not tau_csv.exists():
        print(f'  Missing files for {sample}, skipping KM example')
        return

    dens = pd.read_csv(dens_csv)
    tau  = pd.read_csv(tau_csv)

    band_cols = [c for c in dens.columns if c.startswith('band_')]
    band_mm   = np.array([int(c.replace('band_', '').replace('um', '')) / 1000
                          for c in band_cols])

    cal_file  = _CAL_DIR / f'{_CAL_NAME[sample]}.json'
    cache_csv = folder / 'Every 30s from Evap' / 'survival_analysis' / 'droplet_tracks.csv'
    tracks    = _compute_droplet_tracks(folder, cal_file, cache_csv=cache_csv)

    _ZONE_CENTERS = [0.3, 0.7, 1.1, 1.7, 2.5, 3.3]
    BIN_W = 0.4
    _cmap = plt.colormaps['plasma']
    _KM_ZONES = []
    for i, ctr in enumerate(_ZONE_CENTERS):
        lo = ctr - BIN_W / 2;  hi = ctr + BIN_W / 2
        color = _cmap(i / (len(_ZONE_CENTERS) - 1))
        _KM_ZONES.append((lo, hi, color, f'{ctr:.1f} mm'))

    fig, (ax_km, ax_tau) = plt.subplots(1, 2, figsize=(130 * MM, 58 * MM))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.18, wspace=0.45)

    from lifelines import KaplanMeierFitter as _KMF
    t_max_km = 0.0
    for lo, hi, color, lbl in _KM_ZONES:
        sub = tracks[(tracks['dist_um'] >= lo * 1000) & (tracks['dist_um'] < hi * 1000)]
        if len(sub) < 5:
            continue
        kmf = _KMF()
        kmf.fit(sub['duration_min'], event_observed=~sub['censored'])
        t_vals = kmf.survival_function_.index.values
        s_vals = kmf.survival_function_['KM_estimate'].values
        ci_lo  = kmf.confidence_interval_['KM_estimate_lower_0.95'].values
        ci_hi  = kmf.confidence_interval_['KM_estimate_upper_0.95'].values
        ax_km.step(t_vals, s_vals, where='post', color=color, lw=1.6,
                   label=lbl, zorder=3)
        t_max_km = max(t_max_km, t_vals[-1])
    ax_km.axhline(0.5, color='gray', ls='--', lw=0.7, alpha=0.5, zorder=1)
    ax_km.set_xlabel('Time since birth (min)', fontsize=LS)
    ax_km.set_ylabel('Fraction surviving', fontsize=LS)
    ax_km.set_ylim(0, 1.15)
    ax_km.set_xlim(0, 10)
    ax_km.legend(title='Distance from\nboundary (mm)', fontsize=TS - 1.5,
                 title_fontsize=TS - 1.5, frameon=False,
                 loc='upper right', handlelength=1.2)
    ax_km.text(-0.28, 1.06, 'A', transform=ax_km.transAxes,
               fontsize=PL, fontweight='bold', va='top')
    for sp in ['top', 'right']:
        ax_km.spines[sp].set_visible(False)

    valid_tau = (tau['tau50_min'].notna() &
                 (tau['tau50_min'] > 1.0) &
                 (tau['distance_mm'] <= CAP_MM))
    td = tau.loc[valid_tau, 'distance_mm'].values
    tt = tau.loc[valid_tau, 'tau50_min'].values

    ax_tau.scatter(td, tt, s=14, c='#555555', zorder=3, edgecolors='none')
    ax_tau.plot(td, tt, color='#555555', lw=0.8, zorder=2)

    if len(td) >= 4 and (tt.max() - tt.min()) > 1.0:
        from scipy.optimize import curve_fit as _curve_fit
        def _hill_local(d, T0, A, K, n):
            dn = np.power(np.maximum(d, 0.0), n)
            Kn = np.power(np.maximum(K, 1e-6), n)
            return T0 + A * dn / (Kn + dn)
        try:
            p0 = [tt.min(), tt.max() - tt.min(), float(np.median(td)), 2.0]
            lb = [0,        0,                   0.05,                  0.5]
            ub = [tt.max(), tt.max() * 5,         CAP_MM,               5.0]
            popt, _ = _curve_fit(_hill_local, td, tt, p0=p0,
                                 bounds=(lb, ub), maxfev=4000)
            tt_pred = _hill_local(td, *popt)
            ss_res  = np.sum((tt - tt_pred) ** 2)
            ss_tot  = np.sum((tt - tt.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            K_fit = popt[2]
            if r2 >= 0.60 and 0.05 < K_fit < CAP_MM * 0.95:
                d_fit = np.linspace(td.min(), td.max(), 200)
                ax_tau.plot(d_fit, _hill_local(d_fit, *popt),
                            'k-', lw=1.2, alpha=0.7, zorder=4)
                ax_tau.axvline(K_fit, color='#E74C3C', ls='--',
                               lw=1.0, alpha=0.8, zorder=5)
                ax_tau.text(K_fit + 0.08, tt.max() * 0.92,
                            f'd*={K_fit:.2f} mm\n$R^2$={r2:.2f}',
                            fontsize=TS - 1, color='#E74C3C', va='top')
                print(f'  {sample}: d*={K_fit:.3f} mm  R²={r2:.3f}')
            else:
                print(f'  {sample}: Hill fit poor — R²={r2:.3f}, K={K_fit:.3f} mm')
        except Exception as e:
            print(f'  {sample}: Hill fit failed — {e}')
    else:
        print(f'  {sample}: too few bins ({len(td)}) or flat profile for Hill fit')

    ax_tau.set_xlabel('Distance from boundary (mm)', fontsize=LS)
    ax_tau.set_ylabel(r'$\tau_{50}$ (min)', fontsize=LS)
    ax_tau.set_ylim(bottom=0)
    _add_zone_shading(ax_tau)
    ax_tau.text(-0.28, 1.06, 'B', transform=ax_tau.transAxes,
                fontsize=PL, fontweight='bold', va='top')
    for sp in ['top', 'right']:
        ax_tau.spines[sp].set_visible(False)

    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'rsr_km_example_{sample}{ext}', **kw)
    plt.close(fig)
    print(f'Saved to {OUTPUT_DIR}/rsr_km_example_{sample}.*')


if __name__ == '__main__':
    main()
    import pandas as _pd
    _df = _pd.read_csv(
        Path(__file__).parent.parent / 'raw_data' / 'droplets_calibrated_mm.csv')
    plot_grouped(_df)
    plot_rsr_km_example('RSRDiseased5')
