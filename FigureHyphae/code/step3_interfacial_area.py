#!/usr/bin/env python3
"""Interfacial area framework: colony sink strength scales with
accessible hygroscopic material per unit footprint.

Computes from binary masks:
  1. Tissue fraction (f_tissue) — how much of the footprint is covered
  2. Perimeter density (P/A) — tissue-air boundary per unit area
  3. Structure thickness (DT median) — how thick each structure is
  4. Absorbing capacity proxy — f_tissue × thickness
"""

import json
import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from pathlib import Path
from PIL import Image
import time

ROI_DIR = Path('/Volumes/T7/FINAL OSF/HYPHAE/Analysis/results/3d_overlays')
SESSION = ROI_DIR / 'roi_session.json'
MICRO_DIR = Path('/Volumes/T7/FINAL OSF/HYPHAE/Light Microscopy')
OUT = Path('/Users/yany/Downloads')

CAL_3D = 0.94  # µm/px for 3D ROIs


def load_gray(path):
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def adaptive_seg_3d(img):
    smooth = ndi.gaussian_filter(img, sigma=1.0)
    local_mean = ndi.gaussian_filter(smooth, sigma=32.0)
    dark_response = local_mean - smooth
    threshold = dark_response.mean() + 0.5 * dark_response.std()
    mask = dark_response > threshold
    mask = ndi.binary_closing(mask, iterations=1)
    mask = ndi.binary_opening(mask, iterations=1)
    return mask


def otsu_threshold(values):
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    w_bg = np.cumsum(hist).astype(float)
    w_fg = total - w_bg
    sum_bg = np.cumsum(hist * centers)
    mean_bg = sum_bg / np.maximum(w_bg, 1)
    mean_fg = (sum_bg[-1] - sum_bg) / np.maximum(w_fg, 1)
    var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    return centers[np.argmax(var)]


def segment_fungi_micro(img, label):
    smooth = ndi.gaussian_filter(img, sigma=1.0)
    local_mean = ndi.gaussian_filter(smooth, sigma=32.0)
    dark_response = local_mean - smooth
    valid = np.ones(img.shape, dtype=bool)
    margin = max(8, int(round(min(img.shape) * 0.006)))
    valid[:margin, :] = False
    valid[-margin:, :] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False

    values = smooth[valid]
    global_t = otsu_threshold(values)
    dr_values = dark_response[valid]
    dr_median = float(np.median(dr_values))
    dr_mad = float(np.median(np.abs(dr_values - dr_median)) + 1e-9)

    if label.lower() in ('pink', 'green'):
        intensity_cap = float(np.percentile(values, 62))
        local_t = max(float(np.percentile(dr_values, 72)), dr_median + 0.80 * dr_mad)
        broad_mask = smooth < min(global_t, intensity_cap)
        fiber_mask = valid & (dark_response > local_t)
        mask = valid & (broad_mask | fiber_mask)
        min_size, max_hole = 28, 1500
    else:
        intensity_cap = float(np.percentile(values, 32))
        local_t = max(float(np.percentile(dr_values, 78)), dr_median + 1.15 * dr_mad)
        fiber_mask = valid & (dark_response > local_t)
        strong_dark = (smooth < intensity_cap) & (dark_response > np.percentile(dr_values, 48))
        mask = valid & (fiber_mask | strong_dark)
        min_size, max_hole = 10, 80

    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)
    mask &= valid
    labels_arr, nlab = ndi.label(mask)
    if nlab > 0:
        sizes = np.bincount(labels_arr.ravel())
        keep = sizes >= min_size
        keep[0] = False
        mask = keep[labels_arr]
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    hole_labels, nhole = ndi.label(holes)
    if nhole > 0:
        hole_sizes = np.bincount(hole_labels.ravel())
        small = hole_sizes <= max_hole
        small[0] = False
        mask = mask | small[hole_labels]
    mask &= valid
    return mask.astype(bool)


def compute_interfacial_metrics(mask, cal_um_per_px=1.0):
    """Compute interfacial area metrics from binary mask."""
    h, w = mask.shape
    total_area = h * w  # pixels

    # Tissue fraction
    f_tissue = mask.sum() / total_area

    # Perimeter: tissue pixels with at least one pore neighbor
    # Use morphological gradient: dilated - eroded
    dilated = ndi.binary_dilation(mask, iterations=1)
    eroded = ndi.binary_erosion(mask, iterations=1)
    boundary = dilated & ~eroded  # boundary region
    perimeter_px = boundary.sum()
    # Perimeter density = boundary pixels / total area (1/px units)
    perimeter_density = perimeter_px / total_area  # 1/px
    perimeter_density_um = perimeter_density / cal_um_per_px  # 1/µm

    # Distance transform: median thickness of tissue structures
    if mask.sum() > 10:
        dt = ndi.distance_transform_edt(mask) * cal_um_per_px
        dt_median = np.median(dt[mask])
    else:
        dt_median = 0.0

    # Absorbing capacity: tissue fraction × thickness
    absorbing_capacity = f_tissue * dt_median

    # Specific surface area: perimeter / tissue_area (1/px → 1/µm)
    tissue_area = mask.sum()
    if tissue_area > 0:
        specific_surface = perimeter_px / tissue_area / cal_um_per_px  # 1/µm
    else:
        specific_surface = 0.0

    return {
        'f_tissue': f_tissue,
        'perimeter_density': perimeter_density_um,
        'dt_median_um': dt_median,
        'absorbing_capacity': absorbing_capacity,
        'specific_surface': specific_surface,
    }


# ═══════════════════════════════════════════════════════════════
print('=' * 70)
print('INTERFACIAL AREA FRAMEWORK')
print('=' * 70)

# ── 3D Colony ROIs ──
print('\n── 3D Colony Surface ROIs ──')
with open(SESSION) as f:
    session = json.load(f)
saved = {k: v for k, v in session.items()
         if not k.startswith('_') and v.get('status') != 'deleted'}

results_3d = []
for key, val in saved.items():
    genus = val.get('genus')
    roi_path = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
    if not roi_path.exists():
        continue
    img = load_gray(roi_path)
    mask = adaptive_seg_3d(img)
    m = compute_interfacial_metrics(mask, cal_um_per_px=CAL_3D)
    results_3d.append({'file': key, 'genus': genus, **m})
    print(f'  {key} ({genus}): f={m["f_tissue"]:.3f}, '
          f'P/A={m["perimeter_density"]:.4f}/µm, '
          f'DT={m["dt_median_um"]:.1f}µm, '
          f'cap={m["absorbing_capacity"]:.3f}, '
          f'SSA={m["specific_surface"]:.3f}/µm')

# ── Light Microscopy ──
print('\n── Light Microscopy ──')
# Calibrations (µm/px) for each magnification
cal_micro = {10: 1.0, 20: 0.5, 40: 0.25}  # approximate for standard microscope
micro_images = [
    ('Pink_10X_1.TIF', 'Aspergillus', 'Pink', 10),
    ('Pink_10X_2.TIF', 'Aspergillus', 'Pink', 10),
    ('Pink_20X_1.TIF', 'Aspergillus', 'Pink', 20),
    ('Pink_20X_2.TIF', 'Aspergillus', 'Pink', 20),
    ('Pink_40X_1.TIF', 'Aspergillus', 'Pink', 40),
    ('Pink_40X_2.TIF', 'Aspergillus', 'Pink', 40),
    ('White_10X_1.TIF', 'Mucor', 'White', 10),
    ('White_20X_1.TIF', 'Mucor', 'White', 20),
    ('White_40X_1.TIF', 'Mucor', 'White', 40),
]

results_lm = []
for fname, genus, label, mag in micro_images:
    path = MICRO_DIR / fname
    img = load_gray(path)
    mask = segment_fungi_micro(img, label)
    cal = cal_micro[mag]
    m = compute_interfacial_metrics(mask, cal_um_per_px=cal)
    results_lm.append({'file': fname, 'genus': genus, 'mag': mag, **m})
    print(f'  {fname} ({genus} {mag}X): f={m["f_tissue"]:.3f}, '
          f'P/A={m["perimeter_density"]:.4f}/µm, '
          f'DT={m["dt_median_um"]:.1f}µm, '
          f'cap={m["absorbing_capacity"]:.3f}, '
          f'SSA={m["specific_surface"]:.3f}/µm')


# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('STATISTICS')
print('=' * 70)

for label, results in [('3D Colony ROIs', results_3d), ('Light Microscopy', results_lm)]:
    print(f'\n── {label} ──')
    asp = [r for r in results if r['genus'] == 'Aspergillus']
    muc = [r for r in results if r['genus'] == 'Mucor']

    for metric, mname in [('f_tissue', 'Tissue fraction'),
                           ('perimeter_density', 'Perimeter density'),
                           ('dt_median_um', 'DT median (µm)'),
                           ('absorbing_capacity', 'Absorb. capacity'),
                           ('specific_surface', 'Specific surface')]:
        va = np.array([r[metric] for r in asp])
        vm = np.array([r[metric] for r in muc])
        if len(va) < 2 or len(vm) < 2:
            continue
        t, p = stats.ttest_ind(va, vm, equal_var=False)
        ratio = va.mean() / vm.mean() if vm.mean() != 0 else np.inf
        n1, n2 = len(va), len(vm)
        pooled = np.sqrt(((n1-1)*va.std(ddof=1)**2 + (n2-1)*vm.std(ddof=1)**2) / (n1+n2-2))
        d = (va.mean() - vm.mean()) / pooled if pooled > 0 else 0
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f'  {mname:20s}  Asp={va.mean():.4f}±{va.std(ddof=1):.4f}  '
              f'Muc={vm.mean():.4f}±{vm.std(ddof=1):.4f}  '
              f'ratio={ratio:.3f}  d={d:.2f}  p={p:.4g} {sig}')


# ═══════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('PREDICTION CHECK')
print('=' * 70)

# Key question: does absorbing_capacity ratio (f_tissue × DT) predict δ ratio?
asp_cap = np.array([r['absorbing_capacity'] for r in results_3d if r['genus'] == 'Aspergillus'])
muc_cap = np.array([r['absorbing_capacity'] for r in results_3d if r['genus'] == 'Mucor'])

cap_ratio = asp_cap.mean() / muc_cap.mean()
measured_delta_ratio = 2.13  # viable Aspergillus vs viable Mucor only
print(f'  Absorbing capacity ratio (3D): {cap_ratio:.3f}')
print(f'  Measured δ ratio: {measured_delta_ratio:.2f}')
print(f'  Match: {"GOOD" if abs(cap_ratio - measured_delta_ratio) / measured_delta_ratio < 0.2 else "PARTIAL" if abs(cap_ratio - measured_delta_ratio) / measured_delta_ratio < 0.4 else "POOR"}')

# Also check tissue fraction × thickness separately
asp_f = np.array([r['f_tissue'] for r in results_3d if r['genus'] == 'Aspergillus'])
muc_f = np.array([r['f_tissue'] for r in results_3d if r['genus'] == 'Mucor'])
asp_dt = np.array([r['dt_median_um'] for r in results_3d if r['genus'] == 'Aspergillus'])
muc_dt = np.array([r['dt_median_um'] for r in results_3d if r['genus'] == 'Mucor'])

print(f'\n  Components:')
print(f'    Tissue fraction ratio: {asp_f.mean()/muc_f.mean():.3f}')
print(f'    DT thickness ratio:    {asp_dt.mean()/muc_dt.mean():.3f}')
print(f'    Product:               {asp_f.mean()/muc_f.mean() * asp_dt.mean()/muc_dt.mean():.3f}')

# Perimeter density
asp_p = np.array([r['perimeter_density'] for r in results_3d if r['genus'] == 'Aspergillus'])
muc_p = np.array([r['perimeter_density'] for r in results_3d if r['genus'] == 'Mucor'])
print(f'    Perimeter density ratio: {asp_p.mean()/muc_p.mean():.3f}')

# Specific surface area (perimeter / tissue area)
asp_s = np.array([r['specific_surface'] for r in results_3d if r['genus'] == 'Aspergillus'])
muc_s = np.array([r['specific_surface'] for r in results_3d if r['genus'] == 'Mucor'])
print(f'    Specific surface ratio: {asp_s.mean()/muc_s.mean():.3f}')
print(f'      (Mucor has HIGHER SSA — thin filaments have more surface per volume)')
print(f'      (But Asp has more TOTAL tissue per footprint → net stronger sink)')

# Save
import csv
for results, outname in [(results_3d, 'interfacial_metrics_3d.csv'),
                          (results_lm, 'interfacial_metrics_lm.csv')]:
    outpath = OUT / outname
    keys = ['file', 'genus'] + (['mag'] if 'mag' in results[0] else []) + \
           ['f_tissue', 'perimeter_density', 'dt_median_um', 'absorbing_capacity', 'specific_surface']
    with open(outpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f'\nSaved: {outpath}')
