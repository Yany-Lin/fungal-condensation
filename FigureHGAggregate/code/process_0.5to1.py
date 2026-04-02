#!/usr/bin/env python3
"""Batch-process the 0.5to1 hydrogel trials."""

import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from scipy.ndimage import distance_transform_edt, find_objects

THIS_DIR = Path(__file__).parent
RAW_DATA = THIS_DIR.parent / 'raw_data'
AGG_DIR  = RAW_DATA / 'aggregate_edt'
AGG_DIR.mkdir(parents=True, exist_ok=True)

TRIAL_CONFIG = {
    '0.5to1.1': {'folder': '0.5to1.1', 'npy_subdir': 'Results', 'source_type': 'ellipse'},
    '0.5to1.2': {'folder': '0.5to1.2', 'npy_subdir': 'Results', 'source_type': 'ellipse'},
    '0.5to1.3': {'folder': '0.5to1.3', 'npy_subdir': 'Results', 'source_type': 'ellipse'},
    '0.5to1.4': {'folder': '0.5to1.4', 'npy_subdir': 'Results', 'source_type': 'ellipse'},
    '0.5to1.5': {'folder': '0.5to1.5', 'npy_subdir': 'Results', 'source_type': 'ellipse'},
    '0.5to1.6': {'folder': '0.5to1.6', 'npy_subdir': 'Results', 'source_type': 'ellipse'},
}

BIN_WIDTH_UM  = 200
MIN_DROPLETS  = 3


def parse_timestamp_seconds(filename: str) -> float | None:
    m = re.search(r'(\d+)m(\d+)s', Path(filename).stem)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def load_calibration(cal_path: Path) -> dict:
    with open(cal_path) as f:
        return json.load(f)


def build_edt_map_ellipse(shape, cal: dict):
    """Build EDT distance map from calibration ellipse (for hydrogels)."""
    ell = cal['source_ellipse']
    cx, cy = ell['center_px']
    a = ell['width_px'] / 2
    b = ell['height_px'] / 2
    angle = np.radians(ell.get('angle_deg', 0))
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    dx, dy = x - cx, y - cy
    ca, sa = np.cos(angle), np.sin(angle)
    xr = dx * ca + dy * sa
    yr = -dx * sa + dy * ca
    mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1
    edt = distance_transform_edt(~mask).astype(np.float32)
    edt[mask] = 0
    return edt, mask


def process_frame(npy_path: Path, time_sec: float, pixel_size: float,
                  edt_map: np.ndarray) -> list[dict]:
    masks = np.load(npy_path)
    time_min = time_sec / 60.0
    rows = []
    n_labels = masks.max()
    if n_labels == 0:
        return rows
    slices = find_objects(masks)
    for lbl_idx, slc in enumerate(slices):
        if slc is None:
            continue
        lbl = lbl_idx + 1
        sub = masks[slc]
        obj = sub == lbl
        area_px = obj.sum()
        if area_px < 4:
            continue
        radius_um = np.sqrt(area_px / np.pi) * pixel_size
        ys_local, xs_local = np.where(obj)
        cx = (xs_local.mean() + slc[1].start)
        cy = (ys_local.mean() + slc[0].start)
        dist_px = edt_map[int(cy), int(cx)]
        dist_um = dist_px * pixel_size
        rows.append({'time_min': time_min, 'radius_um': radius_um,
                     'distance_um': dist_um, 'cx': cx, 'cy': cy})
    return rows


def compute_binned_statistics(droplet_df: pd.DataFrame) -> pd.DataFrame:
    max_dist = droplet_df['distance_um'].max()
    bins = np.arange(0, max_dist + BIN_WIDTH_UM, BIN_WIDTH_UM)
    droplet_df = droplet_df.copy()
    droplet_df['distance_bin_um'] = (
        pd.cut(droplet_df['distance_um'], bins=bins,
               labels=bins[:-1] + BIN_WIDTH_UM / 2).astype(float))
    grouped = droplet_df.groupby(['time_min', 'distance_bin_um'])['radius_um']
    binned = grouped.agg(['mean', 'std', 'count', 'sem']).reset_index()
    binned.columns = ['time_min', 'distance_bin_um',
                      'mean_radius_um', 'std_radius_um', 'n_droplets', 'sem_radius_um']
    return binned[binned['n_droplets'] >= MIN_DROPLETS]



def process_trial(trial_id: str, cfg: dict):
    trial_dir = RAW_DATA / cfg['folder']
    cal_path  = trial_dir / 'calibration.json'
    npy_dir   = trial_dir / cfg['npy_subdir']

    if not cal_path.exists():
        print(f'  [SKIP] {trial_id}: calibration.json not found at {cal_path}')
        return
    if not npy_dir.exists():
        print(f'  [SKIP] {trial_id}: NPY folder not found at {npy_dir}')
        return

    cal = load_calibration(cal_path)
    pixel_size = cal['scale']['pixel_size_um']

    npy_files = sorted(f for f in npy_dir.glob('*_masks.npy')
                       if not f.name.startswith('._'))
    if not npy_files:
        print(f'  [SKIP] {trial_id}: no *_masks.npy found in {npy_dir}')
        return

    first_mask = np.load(npy_files[0])
    shape = first_mask.shape

    edt_map, source_mask = build_edt_map_ellipse(shape, cal)

    print(f'  {trial_id}: {len(npy_files)} frames, shape={shape}, '
          f'max_dist={edt_map.max()*pixel_size:.0f} um')

    all_droplets = []
    for f in npy_files:
        t = parse_timestamp_seconds(f.name)
        if t is None:
            continue
        all_droplets.extend(process_frame(f, t, pixel_size, edt_map))

    if not all_droplets:
        print(f'  [WARN] {trial_id}: no droplets extracted')
        return

    df = pd.DataFrame(all_droplets)

    out_droplets = AGG_DIR / f'{trial_id}_edt_droplets.csv'
    df.to_csv(out_droplets, index=False)

    binned = compute_binned_statistics(df)
    out_binned = AGG_DIR / f'{trial_id}_edt_binned_statistics.csv'
    binned.to_csv(out_binned, index=False)

    src = cal.get('source_boundary', {})
    if src.get('polygon_x'):
        poly = pd.DataFrame({'x': np.array(src['polygon_x']) * pixel_size,
                              'y': np.array(src['polygon_y']) * pixel_size})
        poly.to_csv(AGG_DIR / f'{trial_id}_boundary_polygon.csv', index=False)

    print(f'         {len(df):,} droplets')
    print(f'         distance range: {df["distance_um"].min():.1f} - {df["distance_um"].max():.1f} um')
    print(f'         time range:     {df["time_min"].min():.1f} - {df["time_min"].max():.1f} min')
    print(f'  -> saved {out_droplets.name}, {out_binned.name}')

    return df



if __name__ == '__main__':
    print(f'Processing {len(TRIAL_CONFIG)} 0.5to1 trial(s) -> {AGG_DIR}\n')

    summary = {}
    for tid, cfg in TRIAL_CONFIG.items():
        print(f'--- {tid} ---')
        df = process_trial(tid, cfg)
        if df is not None:
            summary[tid] = {
                'n_droplets': len(df),
                'dist_min': df['distance_um'].min(),
                'dist_max': df['distance_um'].max(),
                'time_min': df['time_min'].min(),
                'time_max': df['time_min'].max(),
            }
        print()

    print('=' * 70)
    print(f'{"Trial":<12} {"Droplets":>10} {"Dist min (um)":>14} {"Dist max (um)":>14} {"t min":>8} {"t max":>8}')
    print('-' * 70)
    for tid, s in summary.items():
        print(f'{tid:<12} {s["n_droplets"]:>10,} {s["dist_min"]:>14.1f} {s["dist_max"]:>14.1f} '
              f'{s["time_min"]:>8.1f} {s["time_max"]:>8.1f}')
    print('=' * 70)
    print('Done.')
