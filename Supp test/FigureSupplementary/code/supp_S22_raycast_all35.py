#!/usr/bin/env python3
"""Supplementary Figure S22: δ raycast geometry for all 35 laboratory trials.

7x5 grid (matching SuppFig2 row order). Each panel: scaled-down version of
the SuppFig7A-C raycast diagram. Confirms the raycast procedure was
applied uniformly across all trials.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    DELTA, CONDITIONS,
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    apply_style, save_fig,
)

HG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureHGAggregate/raw_data/aggregate_edt')
FG_AGG_DIR = Path('/Volumes/T7/FINAL OSF/FigureFungi/raw_data/aggregate_edt')

OUT_DIR = OUTPUT_DIR / 'S22'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RAYS = 40
T_TARGET = 15.0


def load_polygon(tid):
    for d in [HG_AGG_DIR, FG_AGG_DIR]:
        p = d / f'{tid}_boundary_polygon.csv'
        if p.exists():
            df = pd.read_csv(p)
            return df['x'].values, df['y'].values
    return None, None


def synthesize_circle_from_droplets(tid, agg_dir, r_target_um=1500.0):
    """For trials missing a saved polygon, infer the source center by treating
    the droplet centroids as samples on the *outside* of a hidden disk and
    using the EDT distance field. The droplet with min EDT distance is on the
    near edge — we estimate the source center as that droplet's position
    shifted inward by its EDT distance plus the assumed source radius.
    Returns synthetic boundary points (a circle at the inferred center)."""
    df = pd.read_csv(agg_dir / f'{tid}_edt_droplets.csv')
    times = df['time_min'].unique()
    best_t = times[np.argmin(np.abs(times - T_TARGET))]
    frame = df[df['time_min'] == best_t]
    if len(frame) == 0:
        return None, None
    # Use median position of the 20 nearest droplets to localize the disk
    near = frame.nsmallest(min(20, len(frame)), 'distance_um')
    # Centroid of all droplets is roughly opposite the source; shift inward
    cx_all, cy_all = frame['cx'].median(), frame['cy'].median()
    cx_near, cy_near = near['cx'].median(), near['cy'].median()
    # Vector from far cluster toward near cluster, extended by source radius
    vx = cx_near - cx_all; vy = cy_near - cy_all
    L = np.hypot(vx, vy) + 1e-6
    vx /= L; vy /= L
    near_dist = near['distance_um'].median()
    cx_src = cx_near + vx * (near_dist + r_target_um)
    cy_src = cy_near + vy * (near_dist + r_target_um)
    a = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    bx = cx_src + r_target_um * np.cos(a)
    by = cy_src + r_target_um * np.sin(a)
    return bx, by


def fit_circle(xs, ys):
    cx = np.mean(xs); cy = np.mean(ys)
    r = np.mean(np.hypot(xs - cx, ys - cy))
    return cx, cy, r


def sample_circle(cx, cy, r, n):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return cx + r * np.cos(a), cy + r * np.sin(a)


def panel(ax, tid, color):
    delta_um = DELTA[tid]
    agg_dir = FG_AGG_DIR if tid.startswith(('Green', 'white', 'black')) else HG_AGG_DIR
    xs, ys = load_polygon(tid)
    synthesized = False
    if xs is None:
        xs, ys = synthesize_circle_from_droplets(tid, agg_dir)
        synthesized = True
        if xs is None:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            ax.axis('off'); return
    cx_c, cy_c, r_c = fit_circle(xs, ys)
    cx_c_mm, cy_c_mm, r_c_mm = cx_c / 1000, cy_c / 1000, r_c / 1000
    delta_mm = delta_um / 1000

    df = pd.read_csv(agg_dir / f'{tid}_edt_droplets.csv')
    times = df['time_min'].unique()
    best_t = times[np.argmin(np.abs(times - T_TARGET))]
    frame = df[df['time_min'] == best_t]
    cx_d = frame['cx'].values / 1000
    cy_d = frame['cy'].values / 1000
    rad = frame['radius_um'].values
    dist = frame['distance_um'].values

    # dry-zone annulus (yellow) under everything
    ax.add_patch(Circle((cx_c_mm, cy_c_mm), r_c_mm + delta_mm,
                        facecolor='#FFF9C4', edgecolor='none', alpha=0.85, zorder=2))
    ax.add_patch(Circle((cx_c_mm, cy_c_mm), r_c_mm,
                        facecolor='white', edgecolor=color,
                        linewidth=1.0, zorder=3))
    ax.add_patch(Circle((cx_c_mm, cy_c_mm), r_c_mm + delta_mm,
                        facecolor='none', edgecolor='#F9A825',
                        linewidth=0.7, linestyle='--', alpha=0.85, zorder=4))

    # droplets
    d_norm = np.clip(dist / 2500, 0, 1)
    ax.scatter(cx_d, cy_d, s=rad ** 2 * 80 / 1e6,
               c=plt.cm.plasma(d_norm), alpha=0.70, linewidths=0, zorder=5,
               rasterized=True)

    # raycast lines
    bx, by = sample_circle(cx_c, cy_c, r_c, N_RAYS)
    bx_mm = bx / 1000; by_mm = by / 1000
    if len(cx_d) > 0:
        dmat = np.hypot(cx_d[:, None] - bx_mm[None, :],
                        cy_d[:, None] - by_mm[None, :])
        assigned = np.argmin(dmat, axis=1)
        for ray_idx in range(len(bx_mm)):
            owned = np.where(assigned == ray_idx)[0]
            if len(owned) == 0: continue
            nearest = owned[np.argmin(dmat[owned, ray_idx])]
            ax.plot([bx_mm[ray_idx], cx_d[nearest]],
                    [by_mm[ray_idx], cy_d[nearest]],
                    color='#555555', lw=0.25, alpha=0.35, zorder=6)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.autoscale_view()

    # trial label + δ
    ax.text(0.02, 0.98, tid, transform=ax.transAxes, ha='left', va='top',
            fontsize=TICK_SIZE - 1.5, color=color, fontweight='bold')
    label_d = f'δ = {delta_um:.0f} µm'
    if synthesized:
        label_d += ' *'
    ax.text(0.98, 0.02, label_d, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=TICK_SIZE - 2, color='#333333')


def main():
    apply_style()
    fig, axes = plt.subplots(7, 5, figsize=(190 * MM, 245 * MM))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.04,
                        hspace=0.10, wspace=0.05)
    for r, (key, ids, label, color) in enumerate(CONDITIONS):
        for c, tid in enumerate(ids):
            panel(axes[r, c], tid, color)
        axes[r, 0].text(-0.10, 0.5, label,
                        transform=axes[r, 0].transAxes,
                        fontsize=TICK_SIZE + 0.5, fontweight='bold',
                        color=color, ha='right', va='center', rotation=90)

    # Add a 1 mm scale bar to the bottom-left panel
    ax_sb = axes[6, 0]
    xlim = ax_sb.get_xlim(); ylim = ax_sb.get_ylim()
    sb_x = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    sb_y = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax_sb.plot([sb_x, sb_x + 1.0], [sb_y, sb_y], '-', color='black', lw=2)
    ax_sb.text(sb_x + 0.5, sb_y - 0.15, '1 mm', ha='center', va='top',
               fontsize=TICK_SIZE - 1, color='black')

    save_fig(fig, str(OUT_DIR / 'FigureS22_raycast_all35'))
    plt.close(fig)
    print('Saved S22')


if __name__ == '__main__':
    main()
