#!/usr/bin/env python3
"""Bidirectional droplet tracking with Hungarian matching and physics priors."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

MIN_AREA = 100        # px²
MAX_AREA = 100_000    # px²
SIGMA_SPATIAL = 10.0  # px
SIGMA_SIZE_FRAC = 0.3
COST_MAX = 9.0
GROWTH_EPS = 0.05
N_LOST_MAX = 2
T_SEED_S = 900

_HG_MASK_SUB = {
    'agar.1': 'Results', 'agar.2': 'Result', 'agar.3': 'Result',
    'agar.4': 'Results', 'agar.5': 'Results',
    '0.5to1.2': 'Results', '0.5to1.3': 'Results', '0.5to1.4': 'Results',
    '0.5to1.5': 'Results', '0.5to1.7': 'Results',
    '1to1.1': 'Results', '1to1.2': 'Results', '1to1.3': 'Result',
    '1to1.4': 'Result',  '1to1.5': 'Results',
    '2to1.1': 'Results', '2to1.2': 'Result',  '2to1.3': 'Results',
    '2to1.4': 'Results', '2to1.5': 'Results',
}

RAW_DATA = Path(__file__).resolve().parent.parent.parent / 'raw_data'



class EllipseBoundary:
    def __init__(self, center, width, height, angle_deg=0):
        self.cx, self.cy = center
        self.a = width / 2
        self.b = height / 2
        self.angle = np.radians(angle_deg)
        self.cos_a = np.cos(self.angle)
        self.sin_a = np.sin(self.angle)

    def distance(self, x, y):
        x = np.atleast_1d(x).astype(float)
        y = np.atleast_1d(y).astype(float)
        dx, dy = x - self.cx, y - self.cy
        x_e =  dx * self.cos_a + dy * self.sin_a
        y_e = -dx * self.sin_a + dy * self.cos_a
        r_norm = np.sqrt((x_e / self.a)**2 + (y_e / self.b)**2)
        inside = r_norm <= 1
        theta = np.arctan2(y_e, x_e)
        xb = self.a * np.cos(theta)
        yb = self.b * np.sin(theta)
        dist = np.sqrt((x_e - xb)**2 + (y_e - yb)**2)
        dist[inside] = 0
        return dist


class PolygonBoundary:
    def __init__(self, polygon_x, polygon_y, image_shape):
        from matplotlib.path import Path as MplPath
        verts = np.column_stack([polygon_x, polygon_y])
        path = MplPath(verts)
        rows, cols = image_shape
        yg, xg = np.mgrid[:rows, :cols]
        pts = np.column_stack([xg.ravel(), yg.ravel()])
        mask = path.contains_points(pts).reshape(image_shape)
        self._edt = distance_transform_edt(~mask).astype(np.float32)
        self._edt[mask] = 0
        self._shape = image_shape

    def distance(self, x, y):
        x = np.atleast_1d(x).astype(float)
        y = np.atleast_1d(y).astype(float)
        r = np.clip(np.round(y).astype(int), 0, self._shape[0] - 1)
        c = np.clip(np.round(x).astype(int), 0, self._shape[1] - 1)
        return self._edt[r, c].astype(float)



def parse_timestamp(filename):
    m = re.search(r'(\d+)m(\d+)s', Path(filename).stem)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def load_mask(filepath):
    data = np.load(filepath, allow_pickle=True)
    if data.ndim == 2:
        return data
    if data.ndim == 0:
        d = data.item()
        if isinstance(d, dict):
            return d.get('masks', d.get('seg'))
    raise ValueError(f"Unrecognized mask format: {filepath}")


def load_calibration(trial_dir, mask_shape=None, prefer_polygon=False):
    cal = json.loads((trial_dir / 'calibration.json').read_text())
    pixel_size = cal['scale']['pixel_size_um']

    has_polygon = 'source_boundary' in cal and cal['source_boundary'].get('polygon_x')
    has_ellipse = 'source_ellipse' in cal

    if prefer_polygon and has_polygon and mask_shape is not None:
        src = cal['source_boundary']
        boundary = PolygonBoundary(src['polygon_x'], src['polygon_y'], mask_shape)
    elif has_ellipse and not prefer_polygon:
        se = cal['source_ellipse']
        boundary = EllipseBoundary(
            se['center_px'], se['width_px'], se['height_px'],
            se.get('angle_deg', 0),
        )
    elif has_polygon and mask_shape is not None:
        src = cal['source_boundary']
        boundary = PolygonBoundary(src['polygon_x'], src['polygon_y'], mask_shape)
    elif has_ellipse:
        se = cal['source_ellipse']
        boundary = EllipseBoundary(
            se['center_px'], se['width_px'], se['height_px'],
            se.get('angle_deg', 0),
        )
    else:
        raise ValueError(f"No usable source geometry in {trial_dir / 'calibration.json'}")

    return pixel_size, boundary


def discover_frames(trial_id, raw_data_root=None):
    raw = raw_data_root or RAW_DATA
    trial_dir = raw / trial_id
    sub = _HG_MASK_SUB.get(trial_id, 'Results')
    mask_dir = trial_dir / sub
    if not mask_dir.is_dir():
        alt = 'Result' if sub == 'Results' else 'Results'
        mask_dir = trial_dir / alt
    frames = []
    for p in sorted(mask_dir.glob('*_masks.npy')):
        if p.name.startswith('._'):
            continue
        t = parse_timestamp(p.name)
        if t is not None:
            frames.append((t, p))
    frames.sort(key=lambda x: x[0])
    return frames



def extract_detections(mask_path, pixel_size, boundary):
    mask = load_mask(mask_path)
    props = regionprops(mask)
    rows = []
    for p in props:
        area = p.area
        if area < MIN_AREA or area > MAX_AREA:
            continue
        cy, cx = p.centroid
        R_eq = np.sqrt(area / np.pi)
        dist_px = boundary.distance(np.array([cx]), np.array([cy]))[0]
        rows.append({
            'label': p.label,
            'cx': cx, 'cy': cy,
            'area_px': area,
            'R_eq': R_eq,
            'distance_um': dist_px * pixel_size,
        })
    return pd.DataFrame(rows)



def compute_cost_matrix(tracks_state, detections, direction='forward'):
    n_t = len(tracks_state)
    n_d = len(detections)
    if n_t == 0 or n_d == 0:
        return np.empty((n_t, n_d))

    t_xy = np.array([[s['cx'], s['cy']] for s in tracks_state])
    d_xy = detections[['cx', 'cy']].values
    t_R = np.array([s['R_eq'] for s in tracks_state])
    d_R = detections['R_eq'].values

    dx = t_xy[:, 0:1] - d_xy[:, 0:1].T  # (n_t, n_d)
    dy = t_xy[:, 1:2] - d_xy[:, 1:2].T
    spatial = (dx**2 + dy**2) / SIGMA_SPATIAL**2

    sigma_R = np.maximum(SIGMA_SIZE_FRAC * t_R, 1.0)  # (n_t,)
    dR = d_R[np.newaxis, :] - t_R[:, np.newaxis]       # (n_t, n_d)
    size = dR**2 / sigma_R[:, np.newaxis]**2

    cost = spatial + size

    # Forward penalizes growth, backward penalizes shrinkage
    eps = GROWTH_EPS * t_R[:, np.newaxis]  # (n_t, n_d)
    if direction == 'forward':
        growth_mask = dR > eps
        cost[growth_mask] += 4.0
    else:
        shrink_mask = dR < -eps
        cost[shrink_mask] += 4.0

    cost[cost > COST_MAX] = COST_MAX + 1.0
    return cost



def assign_hungarian(cost_matrix, cost_max=COST_MAX):
    n_t, n_d = cost_matrix.shape
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched, unmatched_tracks, unmatched_dets = [], [], []
    matched_rows, matched_cols = set(), set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= cost_max:
            matched.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)
        else:
            unmatched_tracks.append(r)

    for i in range(n_t):
        if i not in matched_rows and i not in set(unmatched_tracks):
            unmatched_tracks.append(i)
    for j in range(n_d):
        if j not in matched_cols:
            unmatched_dets.append(j)

    return matched, unmatched_tracks, unmatched_dets



def detect_coalescence(tracks_state, detections, matched, mask_path):
    """Detect mergers: 2+ track centroids landing inside one mask label."""
    if len(tracks_state) < 2:
        return []

    mask = load_mask(mask_path)

    label_to_tracks = {}
    for t_idx, s in enumerate(tracks_state):
        r = int(np.clip(round(s['cy']), 0, mask.shape[0] - 1))
        c = int(np.clip(round(s['cx']), 0, mask.shape[1] - 1))
        lbl = mask[r, c]
        if lbl > 0:
            label_to_tracks.setdefault(lbl, []).append(t_idx)

    events = []
    for lbl, t_idxs in label_to_tracks.items():
        if len(t_idxs) < 2:
            continue
        d_idx = None
        for di in range(len(detections)):
            if detections.iloc[di]['label'] == lbl:
                d_idx = di
                break
        events.append({
            'child_det_idx': d_idx,
            'parent_track_idxs': t_idxs,
            'det_label': lbl,
        })

    return events



def track_direction(frames, seed_tracks, direction='forward', n_lost_max=N_LOST_MAX):
    tracks = {tid: dict(t) for tid, t in seed_tracks.items()}
    for tid in tracks:
        tracks[tid]['history'] = list(tracks[tid]['history'])

    lost_count = {tid: 0 for tid in tracks}
    active = set(tracks.keys())
    coalescence_events = []
    next_id = max(tracks.keys()) + 1 if tracks else 0

    for time_s, dets, mask_path in frames:
        if not active:
            break

        active_list = sorted(active)
        tracks_state = []
        for tid in active_list:
            t = tracks[tid]
            tracks_state.append({
                'track_id': tid,
                'cx': t['cx'], 'cy': t['cy'], 'R_eq': t['R_eq'],
            })

        if len(dets) == 0:
            for tid in active_list:
                lost_count[tid] = lost_count.get(tid, 0) + 1
            to_term = [tid for tid in active_list if lost_count[tid] >= n_lost_max]
            for tid in to_term:
                active.discard(tid)
                tracks[tid]['t_death'] = time_s
                tracks[tid]['death_cause'] = 'lost'
            continue

        cost = compute_cost_matrix(tracks_state, dets, direction)
        matched, unmatched_t, unmatched_d = assign_hungarian(cost)

        coal_evts = detect_coalescence(tracks_state, dets, matched, mask_path)

        coalesced_track_idxs = set()
        for evt in coal_evts:
            parent_idxs = evt['parent_track_idxs']
            child_d_idx = evt['child_det_idx']

            parent_tids = [active_list[i] for i in parent_idxs if i < len(active_list)]
            if len(parent_tids) < 2:
                continue

            surviving = max(parent_tids, key=lambda tid: tracks[tid]['R_eq'])
            for tid in parent_tids:
                if tid != surviving:
                    coalesced_track_idxs.add(tid)
                    tracks[tid]['t_death'] = time_s
                    tracks[tid]['death_cause'] = 'coalescence'
                    tracks[tid]['R_eq_at_death'] = tracks[tid]['R_eq']
                    active.discard(tid)
                    lost_count.pop(tid, None)

            coalescence_events.append({
                'time_s': time_s,
                'parent_tids': parent_tids,
                'surviving_tid': surviving,
                'det_label': evt['det_label'],
            })

        for t_idx, d_idx in matched:
            tid = active_list[t_idx]
            if tid not in active:
                continue
            row = dets.iloc[d_idx]
            tracks[tid]['cx'] = row['cx']
            tracks[tid]['cy'] = row['cy']
            tracks[tid]['R_eq'] = row['R_eq']
            tracks[tid]['area_px'] = row['area_px']
            tracks[tid]['distance_um'] = row['distance_um']
            tracks[tid]['history'].append({
                'time_s': time_s,
                'cx': row['cx'], 'cy': row['cy'],
                'R_eq': row['R_eq'],
            })
            lost_count[tid] = 0

        for t_idx in unmatched_t:
            tid = active_list[t_idx]
            if tid not in active:
                continue
            lost_count[tid] = lost_count.get(tid, 0) + 1
            if lost_count[tid] >= n_lost_max:
                active.discard(tid)
                tracks[tid]['t_death'] = time_s
                tracks[tid]['death_cause'] = 'lost'
                tracks[tid]['R_eq_at_death'] = tracks[tid]['R_eq']

        if direction == 'backward':
            for d_idx in unmatched_d:
                row = dets.iloc[d_idx]
                tracks[next_id] = {
                    'track_id': next_id,
                    'cx': row['cx'], 'cy': row['cy'],
                    'R_eq': row['R_eq'],
                    'area_px': row['area_px'],
                    'distance_um': row['distance_um'],
                    't_birth': time_s,
                    'birth_cause': 'before_window',
                    't_death': None, 'death_cause': None,
                    'R_eq_at_death': None,
                    'history': [{'time_s': time_s, 'cx': row['cx'],
                                 'cy': row['cy'], 'R_eq': row['R_eq']}],
                }
                active.add(next_id)
                lost_count[next_id] = 0
                next_id += 1

    return tracks, coalescence_events



def run_tracking(trial_id, raw_data_root=None, prefer_polygon=False, verbose=True):
    raw = Path(raw_data_root) if raw_data_root else RAW_DATA
    trial_dir = raw / trial_id

    frames_raw = discover_frames(trial_id, raw_data_root=raw)
    if verbose:
        print(f"[{trial_id}] Found {len(frames_raw)} frames")

    first_mask = load_mask(frames_raw[0][1])
    mask_shape = first_mask.shape

    pixel_size, boundary = load_calibration(
        trial_dir, mask_shape=mask_shape, prefer_polygon=prefer_polygon)

    frame_data = []  # (time_s, detections_df, mask_path)
    for time_s, mask_path in frames_raw:
        dets = extract_detections(mask_path, pixel_size, boundary)
        frame_data.append((time_s, dets, mask_path))

    seed_idx = min(range(len(frame_data)),
                   key=lambda i: abs(frame_data[i][0] - T_SEED_S))
    seed_time, seed_dets, seed_mask = frame_data[seed_idx]
    if verbose:
        print(f"  Seed frame: t={seed_time}s ({seed_time/60:.1f} min), "
              f"{len(seed_dets)} droplets")

    seed_tracks = {}
    for i, (_, row) in enumerate(seed_dets.iterrows()):
        seed_tracks[i] = {
            'track_id': i,
            'cx': row['cx'], 'cy': row['cy'],
            'R_eq': row['R_eq'],
            'area_px': row['area_px'],
            'distance_um': row['distance_um'],
            't_birth': seed_time,
            'birth_cause': 'seed',
            't_death': None, 'death_cause': None,
            'R_eq_at_death': None,
            'history': [{'time_s': seed_time, 'cx': row['cx'],
                         'cy': row['cy'], 'R_eq': row['R_eq']}],
        }

    fwd_frames = frame_data[seed_idx + 1:]
    if verbose:
        print(f"  Forward: {len(fwd_frames)} frames")
    tracks_fwd, coal_fwd = track_direction(fwd_frames, seed_tracks, 'forward')

    bwd_frames = list(reversed(frame_data[:seed_idx]))
    if verbose:
        print(f"  Backward: {len(bwd_frames)} frames")
    tracks_bwd, coal_bwd = track_direction(bwd_frames, seed_tracks, 'backward')

    all_tracks = {}
    for tid, t in tracks_fwd.items():
        all_tracks[tid] = dict(t)
        all_tracks[tid]['history'] = list(t['history'])
        all_tracks[tid]['_fwd_death'] = t.get('t_death')
        all_tracks[tid]['_fwd_death_cause'] = t.get('death_cause')
        all_tracks[tid]['_fwd_R_death'] = t.get('R_eq_at_death')

    for tid, t in tracks_bwd.items():
        bwd_hist = sorted(t['history'], key=lambda h: h['time_s'])
        if tid in all_tracks:
            bwd_only = [h for h in bwd_hist if h['time_s'] != seed_time]
            all_tracks[tid]['history'] = bwd_only + all_tracks[tid]['history']
            bwd_cause = t.get('death_cause')
            if bwd_cause == 'coalescence':
                all_tracks[tid]['birth_cause'] = 'coalescence'
        else:
            all_tracks[tid] = dict(t)
            all_tracks[tid]['history'] = list(bwd_hist)
            all_tracks[tid]['_fwd_death'] = None
            all_tracks[tid]['_fwd_death_cause'] = None
            all_tracks[tid]['_fwd_R_death'] = None
            bwd_cause = t.get('death_cause')
            if bwd_cause == 'coalescence':
                all_tracks[tid]['birth_cause'] = 'coalescence'

    rows = []
    last_time = frame_data[-1][0]
    for tid, t in all_tracks.items():
        hist = t['history']
        if not hist:
            continue
        hist_sorted = sorted(hist, key=lambda h: h['time_s'])
        t_first = hist_sorted[0]['time_s']
        t_last_obs = hist_sorted[-1]['time_s']
        R_first = hist_sorted[0]['R_eq']
        R_last = hist_sorted[-1]['R_eq']

        seed_obs = [h for h in hist_sorted if h['time_s'] == seed_time]
        if seed_obs:
            R_seed = seed_obs[0]['R_eq']
            cx_s, cy_s = seed_obs[0]['cx'], seed_obs[0]['cy']
        else:
            R_seed = np.nan
            cx_s, cy_s = hist_sorted[0]['cx'], hist_sorted[0]['cy']

        cx_birth, cy_birth = hist_sorted[0]['cx'], hist_sorted[0]['cy']
        dist_birth_um = float(boundary.distance(
            np.array([cx_birth]), np.array([cy_birth]))[0]) * pixel_size
        dist_seed_um = float(boundary.distance(
            np.array([cx_s]), np.array([cy_s]))[0]) * pixel_size

        fwd_death = t.get('_fwd_death')
        fwd_cause = t.get('_fwd_death_cause')
        fwd_R = t.get('_fwd_R_death')

        if fwd_death is not None and fwd_cause is not None:
            t_death = fwd_death
            death_cause = fwd_cause
            R_death = fwd_R if fwd_R else R_last
            censored = (death_cause == 'coalescence')
        else:
            t_death = t_last_obs
            death_cause = 'censored'
            R_death = R_last
            censored = True

        birth_cause = t.get('birth_cause', 'seed')
        lifetime = t_death - t_first

        rows.append({
            'track_id': tid,
            'n_frames': len(hist_sorted),
            't_birth_s': t_first,
            't_death_s': t_death,
            'lifetime_s': lifetime,
            'birth_cause': birth_cause,
            'death_cause': death_cause,
            'censored': censored,
            'R_eq_birth': R_first,
            'R_eq_seed': R_seed,
            'R_eq_death': R_death,
            'distance_um': dist_seed_um,
            'distance_birth_um': dist_birth_um,
            'cx_seed': cx_s,
            'cy_seed': cy_s,
        })

    tracks_df = pd.DataFrame(rows)
    if verbose:
        n_coal = len([r for r in rows if r['death_cause'] == 'coalescence'])
        n_lost = len([r for r in rows if r['death_cause'] == 'lost'])
        n_cens = len([r for r in rows if r['censored']])
        print(f"  Total tracks: {len(rows)}")
        print(f"  Coalescence: {n_coal}, Lost: {n_lost}, Censored: {n_cens}")
        print(f"  Coalescence events (fwd): {len(coal_fwd)}, (bwd): {len(coal_bwd)}")

    return tracks_df, coal_fwd, coal_bwd, all_tracks


if __name__ == '__main__':
    import sys
    trial = sys.argv[1] if len(sys.argv) > 1 else '2to1.1'
    df, cf, cb, trk = run_tracking(trial)
    outdir = Path(__file__).parent / 'output'
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / f'{trial}_track_histories.csv', index=False)
    print(f"Saved {len(df)} tracks to {outdir / f'{trial}_track_histories.csv'}")
