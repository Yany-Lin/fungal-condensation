#!/usr/bin/env python3
"""Calibrate adaptive segmentation parameters for 3D ROIs.

Target: match previous session's validated values:
  Asp DT median ≈ 7.2 µm, Muc DT median ≈ 4.6 µm

Sweep threshold parameters and find the combination that produces
these values. Then use that segmentation for ALL downstream analysis.
"""

import json
import numpy as np
from scipy import ndimage as ndi
from pathlib import Path
from PIL import Image

_T7 = Path('/Volumes/T7/FINAL OSF')
_REPO = Path(__file__).resolve().parent.parent.parent
BASE = _T7 if _T7.exists() else _REPO
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
CAL = 0.94  # µm/px

def load_gray(path):
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr

with open(SESSION) as f:
    session = json.load(f)
saved = {k: v for k, v in session.items()
         if not k.startswith('_') and v.get('status') != 'deleted'}

# Pick 2 representative images (1 Asp, 1 Muc) for fast sweeping
asp_key = [k for k, v in saved.items() if v.get('genus') == 'Aspergillus'][0]
muc_key = [k for k, v in saved.items() if v.get('genus') == 'Mucor'][0]

asp_img = load_gray(ROI_DIR / 'Aspergillus' / f'{Path(asp_key).stem}_roi.jpg')
muc_img = load_gray(ROI_DIR / 'Mucor' / f'{Path(muc_key).stem}_roi.jpg')

print(f'Test images: {asp_key} (Asp), {muc_key} (Muc)')
print(f'Target: Asp DT ≈ 7.2 µm, Muc DT ≈ 4.6 µm')
print()

def seg_and_dt(img, sigma_smooth, sigma_local, thresh_mult, direction='local_minus_smooth'):
    """Try a segmentation and return median DT in µm."""
    smooth = ndi.gaussian_filter(img, sigma=sigma_smooth)
    local = ndi.gaussian_filter(img, sigma=sigma_local)
    if direction == 'local_minus_smooth':
        dr = local - smooth
    else:
        dr = smooth - local
    t = dr.mean() + thresh_mult * dr.std()
    mask = dr > t
    mask = ndi.binary_closing(mask, iterations=1)
    mask = ndi.binary_opening(mask, iterations=1)
    if mask.sum() < 100:
        return 0, 0, mask
    dt = ndi.distance_transform_edt(mask) * CAL
    return np.median(dt[mask]), mask.mean(), mask

# ── Sweep 1: Match analyze_fungi.py style (sigma=1 smooth, sigma=32 local) ──
print('=' * 70)
print('SWEEP 1: analyze_fungi.py style (smooth=1, local=32, dark_response=local-smooth)')
print('=' * 70)
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]:
    dt_a, f_a, _ = seg_and_dt(asp_img, 1.0, 32.0, thresh, 'local_minus_smooth')
    dt_m, f_m, _ = seg_and_dt(muc_img, 1.0, 32.0, thresh, 'local_minus_smooth')
    marker = ' <-- CLOSE' if abs(dt_a - 7.2) < 1.0 and abs(dt_m - 4.6) < 1.0 else ''
    print(f'  thresh={thresh:.1f}  Asp: DT={dt_a:.1f}µm f={f_a:.3f}  '
          f'Muc: DT={dt_m:.1f}µm f={f_m:.3f}  ratio={dt_a/dt_m:.2f}{marker}')

# ── Sweep 2: Inverted (smooth - local), various sigmas ──
print()
print('=' * 70)
print('SWEEP 2: Inverted (smooth=64, local=16, dark_response=smooth-local)')
print('=' * 70)
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]:
    dt_a, f_a, _ = seg_and_dt(asp_img, 64.0, 16.0, thresh, 'smooth_minus_local')
    dt_m, f_m, _ = seg_and_dt(muc_img, 64.0, 16.0, thresh, 'smooth_minus_local')
    marker = ' <-- CLOSE' if abs(dt_a - 7.2) < 1.0 and abs(dt_m - 4.6) < 1.0 else ''
    print(f'  thresh={thresh:.1f}  Asp: DT={dt_a:.1f}µm f={f_a:.3f}  '
          f'Muc: DT={dt_m:.1f}µm f={f_m:.3f}  ratio={dt_a/dt_m:.2f}{marker}')

# ── Sweep 3: analyze_fungi style but with Otsu on dark_response ──
print()
print('=' * 70)
print('SWEEP 3: Percentile-based threshold on dark_response')
print('=' * 70)
for pctile in [60, 65, 70, 72, 75, 78, 80, 82, 85, 88, 90]:
    def seg_pctile(img, p):
        smooth = ndi.gaussian_filter(img, sigma=1.0)
        local = ndi.gaussian_filter(smooth, sigma=32.0)
        dr = local - smooth
        t = np.percentile(dr, p)
        mask = dr > t
        mask = ndi.binary_closing(mask, iterations=1)
        mask = ndi.binary_opening(mask, iterations=1)
        if mask.sum() < 100:
            return 0, 0
        dt = ndi.distance_transform_edt(mask) * CAL
        return np.median(dt[mask]), mask.mean()
    dt_a, f_a = seg_pctile(asp_img, pctile)
    dt_m, f_m = seg_pctile(muc_img, pctile)
    marker = ' <-- CLOSE' if abs(dt_a - 7.2) < 1.0 and abs(dt_m - 4.6) < 1.0 else ''
    print(f'  pctile={pctile}  Asp: DT={dt_a:.1f}µm f={f_a:.3f}  '
          f'Muc: DT={dt_m:.1f}µm f={f_m:.3f}  ratio={dt_a/dt_m:.2f}{marker}')

# ── Sweep 4: Raw Otsu on grayscale (dark = tissue) ──
print()
print('=' * 70)
print('SWEEP 4: Otsu on raw grayscale (dark < threshold = tissue)')
print('=' * 70)
for scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
    def seg_otsu(img, s):
        lo, hi = np.percentile(img, [0.5, 99.5])
        if hi <= lo: hi = lo + 1
        img8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        hist = np.bincount(img8.ravel(), minlength=256).astype(float)
        total = hist.sum()
        w_bg = np.cumsum(hist)
        w_fg = total - w_bg
        sum_bg = np.cumsum(hist * np.arange(256))
        mean_bg = sum_bg / np.maximum(w_bg, 1)
        mean_fg = (sum_bg[-1] - sum_bg) / np.maximum(w_fg, 1)
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        thresh = int(np.argmax(var) * s)
        mask = img8 < thresh
        mask = ndi.binary_closing(mask, iterations=1)
        mask = ndi.binary_opening(mask, iterations=1)
        if mask.sum() < 100:
            return 0, 0
        dt = ndi.distance_transform_edt(mask) * CAL
        return np.median(dt[mask]), mask.mean()
    dt_a, f_a = seg_otsu(asp_img, scale)
    dt_m, f_m = seg_otsu(muc_img, scale)
    marker = ' <-- CLOSE' if abs(dt_a - 7.2) < 1.0 and abs(dt_m - 4.6) < 1.0 else ''
    print(f'  otsu_scale={scale:.1f}  Asp: DT={dt_a:.1f}µm f={f_a:.3f}  '
          f'Muc: DT={dt_m:.1f}µm f={f_m:.3f}  ratio={dt_a/dt_m:.2f}{marker}')

# ── Sweep 5: sigma sweep with fixed threshold ──
print()
print('=' * 70)
print('SWEEP 5: Sigma sweep (local-smooth, thresh=0.5*std)')
print('=' * 70)
for sig_s, sig_l in [(1, 16), (1, 32), (1, 48), (1, 64),
                      (2, 32), (2, 64), (4, 32), (4, 64),
                      (8, 32), (8, 64), (16, 64)]:
    dt_a, f_a, _ = seg_and_dt(asp_img, float(sig_s), float(sig_l), 0.5, 'local_minus_smooth')
    dt_m, f_m, _ = seg_and_dt(muc_img, float(sig_s), float(sig_l), 0.5, 'local_minus_smooth')
    marker = ' <-- CLOSE' if abs(dt_a - 7.2) < 1.5 and abs(dt_m - 4.6) < 1.5 else ''
    print(f'  smooth={sig_s:2d}, local={sig_l:2d}  Asp: DT={dt_a:.1f}µm f={f_a:.3f}  '
          f'Muc: DT={dt_m:.1f}µm f={f_m:.3f}  ratio={dt_a/dt_m:.2f}{marker}')
