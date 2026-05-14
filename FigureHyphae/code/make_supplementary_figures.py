#!/usr/bin/env python3
"""Supplementary figures — final publication quality.
S18: 3D segmentation + DT
S19: Hessian tubeness overlays (light micro)
S20: Progressive erosion demonstration
S21: Local density CV derivation
S22: Absorbing capacity decomposition + SSA
"""

import json, numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from pathlib import Path
from PIL import Image
from skimage.segmentation import find_boundaries
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_T7 = Path('/Volumes/T7/FINAL OSF')
_REPO = Path(__file__).resolve().parent.parent.parent
BASE = _T7 if _T7.exists() else _REPO
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
HESSIAN_LM = BASE / 'HYPHAE' / 'Analysis' / 'results' / 'hessian_overlays'
SESSION = ROI_DIR / 'roi_session.json'
MICRO_DIR = BASE / 'HYPHAE' / 'Light Microscopy'
OUT = BASE / 'FigureHyphae' / 'figures'
CAL_3D = 0.94; MM = 1/25.4
C_ASP = '#4CAF50'; C_MUC = '#757575'

# Best aspect-ratio-matched ROIs
ASP_ROI_KEY = '20251214_222552.JPG'  # 1.5:1
MUC_ROI_KEY = '20251210_155926.JPG'  # 1.9:1

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'mathtext.default': 'regular',
})


def savefig(fig, name):
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.03}
        if ext == '.png':
            kw['dpi'] = 600
        fig.savefig(OUT / f'{name}{ext}', **kw)
    plt.close(fig)
    print(f'  Saved: {name}')


def load_gray(p):
    with Image.open(p) as im:
        arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def seg_3d(img):
    s = ndi.gaussian_filter(img, sigma=1.0)
    l = ndi.gaussian_filter(s, sigma=32.0)
    dr = l - s
    t = dr.mean() + 0.5 * dr.std()
    m = dr > t
    return ndi.binary_opening(ndi.binary_closing(m, iterations=1), iterations=1)


def otsu_t(v):
    h, e = np.histogram(v, bins=256)
    c = (e[:-1] + e[1:]) / 2
    t = h.sum()
    wb = np.cumsum(h).astype(float)
    wf = t - wb
    sb = np.cumsum(h * c)
    mb = sb / np.maximum(wb, 1)
    mf = (sb[-1] - sb) / np.maximum(wf, 1)
    return c[np.argmax(wb * wf * (mb - mf) ** 2)]


def seg_micro(img, label):
    s = ndi.gaussian_filter(img, sigma=1.0)
    l = ndi.gaussian_filter(s, sigma=32.0)
    dr = l - s
    v = np.ones(img.shape, dtype=bool)
    m = max(8, int(round(min(img.shape) * 0.006)))
    v[:m, :] = False; v[-m:, :] = False; v[:, :m] = False; v[:, -m:] = False
    vals = s[v]; gt = otsu_t(vals); drv = dr[v]
    dm = float(np.median(drv))
    dd = float(np.median(np.abs(drv - dm)) + 1e-9)
    if label.lower() in ('pink', 'green'):
        ic = float(np.percentile(vals, 62))
        lt = max(float(np.percentile(drv, 72)), dm + 0.80 * dd)
        bm = s < min(gt, ic); fm = v & (dr > lt); mask = v & (bm | fm)
        ms, mh = 28, 1500
    else:
        ic = float(np.percentile(vals, 32))
        lt = max(float(np.percentile(drv, 78)), dm + 1.15 * dd)
        fm = v & (dr > lt); sd = (s < ic) & (dr > np.percentile(drv, 48))
        mask = v & (fm | sd); ms, mh = 10, 80
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)
    mask &= v
    la, nl = ndi.label(mask)
    if nl > 0:
        sz = np.bincount(la.ravel()); k = sz >= ms; k[0] = False; mask = k[la]
    fl = ndi.binary_fill_holes(mask); ho = fl & ~mask; hl, nhl = ndi.label(ho)
    if nhl > 0:
        hs = np.bincount(hl.ravel()); sm = hs <= mh; sm[0] = False; mask = mask | sm[hl]
    mask &= v
    return mask.astype(bool)


def norm_display(img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) / (hi - lo), 0, 1)


def add_scalebar(ax, img_shape, cal_um, bar_um=100, color='white', loc='lower-left'):
    h, w = img_shape
    bar_px = int(bar_um / cal_um)
    if loc == 'lower-left':
        bx, by = int(w * 0.04), int(h * 0.93)
        tx, ty = bx + bar_px // 2, by - int(h * 0.035)
    else:
        bx, by = int(w * 0.72), int(h * 0.93)
        tx, ty = bx + bar_px // 2, by - int(h * 0.035)
    ax.plot([bx, bx + bar_px], [by, by], color=color, lw=2.5, solid_capstyle='butt')
    ax.text(tx, ty, f'{bar_um} \u00b5m', color=color, ha='center', fontsize=6,
            fontweight='bold')


def add_panel_label(ax, label, color='black', bg=None):
    props = dict(fontsize=13, fontweight='bold', va='top', color=color)
    if bg:
        props['bbox'] = dict(boxstyle='round,pad=0.15', facecolor=bg, alpha=0.7, edgecolor='none')
    ax.text(-0.06, 1.06, label, transform=ax.transAxes, **props)


def stripbox(ax, sa, sm, yl, show_p=True):
    bp = ax.boxplot([sa, sm], positions=[1, 2], widths=0.4, patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', lw=1.4),
                    whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                    boxprops=dict(linewidth=0.6))
    bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.55)
    bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.55)
    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.uniform(-0.09, 0.09, len(sa)), sa,
               s=16, c=C_ASP, alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
    ax.scatter(2 + rng.uniform(-0.09, 0.09, len(sm)), sm,
               s=16, c=C_MUC, alpha=0.85, edgecolors='white', linewidths=0.3, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([r'$\it{Aspergillus}$', r'$\it{Mucor}$'], fontsize=7)
    ax.set_ylabel(yl, fontsize=7.5)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    if show_p:
        t, p = stats.ttest_ind(sa, sm, equal_var=False)
        ym = max(sa.max(), sm.max())
        r = ym - min(sa.min(), sm.min())
        y = ym + r * 0.08
        ax.plot([1, 1, 2, 2], [y, y + r * 0.03, y + r * 0.03, y], 'k-', lw=0.6)
        p_str = f'p = {p:.1e}' if p < 0.001 else f'p = {p:.4f}'
        ax.text(1.5, y + r * 0.06, p_str, ha='center', fontsize=6)


# ── Load session ──
with open(SESSION) as f:
    session = json.load(f)
saved = {k: v for k, v in session.items()
         if not k.startswith('_') and v.get('status') != 'deleted'}

asp_img = load_gray(ROI_DIR / 'Aspergillus' / f'{Path(ASP_ROI_KEY).stem}_roi.jpg')
muc_img = load_gray(ROI_DIR / 'Mucor' / f'{Path(MUC_ROI_KEY).stem}_roi.jpg')
asp_mask = seg_3d(asp_img)
muc_mask = seg_3d(muc_img)
asp_dt = ndi.distance_transform_edt(asp_mask) * CAL_3D
muc_dt = ndi.distance_transform_edt(muc_mask) * CAL_3D


# ═══════════════════════════════════════════════════════════════
print('S18: 3D Colony — Segmentation + Distance Transform')
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 115 * MM))
fig.subplots_adjust(hspace=0.20, wspace=0.08)

for row, (img, mask, dt, genus, color) in enumerate([
    (asp_img, asp_mask, asp_dt, 'Aspergillus', C_ASP),
    (muc_img, muc_mask, muc_dt, 'Mucor', C_MUC)]):

    nd = norm_display(img)

    # Raw
    axes[row, 0].imshow(nd, cmap='gray')
    axes[row, 0].axis('off')
    add_scalebar(axes[row, 0], img.shape, CAL_3D, 100)
    axes[row, 0].set_title(f'{genus}', fontsize=9, style='italic',
                            color=color, fontweight='bold', pad=4)

    # Boundary
    disp = np.stack([nd] * 3, axis=-1)
    boundary = find_boundaries(mask, mode='thick')
    boundary = ndi.binary_dilation(boundary, iterations=2)
    disp[boundary] = [0.95, 0.15, 0.15]
    axes[row, 1].imshow(disp)
    axes[row, 1].axis('off')
    axes[row, 1].set_title(f'Tissue mask ($f$ = {mask.mean():.3f})', fontsize=7.5, pad=4)

    # DT
    dt_d = dt.copy(); dt_d[~mask] = 0
    im = axes[row, 2].imshow(dt_d, cmap='inferno', vmin=0, vmax=12)
    axes[row, 2].axis('off')
    med = np.median(dt[mask])
    axes[row, 2].set_title(f'Thickness (median = {med:.1f} \u00b5m)', fontsize=7.5, pad=4)

cbar = fig.colorbar(im, ax=axes[:, 2], shrink=0.65, pad=0.03, aspect=20)
cbar.set_label('Distance transform (\u00b5m)', fontsize=7)
cbar.ax.tick_params(labelsize=6)

add_panel_label(axes[0, 0], 'A')
add_panel_label(axes[0, 1], 'B')
add_panel_label(axes[0, 2], 'C')
savefig(fig, 'figS18_seg_3d')


# ═══════════════════════════════════════════════════════════════
print('S19: Light Micro — Hessian Tubeness')
# ═══════════════════════════════════════════════════════════════

asp_hess = plt.imread(HESSIAN_LM / 'Pink_10X_1_overlay.png')
muc_hess = plt.imread(HESSIAN_LM / 'White_10X_1_overlay.png')

fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 90 * MM))
fig.subplots_adjust(wspace=0.04)

axes[0].imshow(asp_hess)
axes[0].axis('off')
axes[0].text(0.02, 0.98, 'A', transform=axes[0].transAxes, fontsize=14,
             fontweight='bold', va='top', color='white',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'))

axes[1].imshow(muc_hess)
axes[1].axis('off')
axes[1].text(0.02, 0.98, 'B', transform=axes[1].transAxes, fontsize=14,
             fontweight='bold', va='top', color='white',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6, edgecolor='none'))

savefig(fig, 'figS19_hessian_lm')


# ═══════════════════════════════════════════════════════════════
print('S20: Erosion Demonstration')
# ═══════════════════════════════════════════════════════════════

asp_lm = load_gray(MICRO_DIR / 'Pink_10X_1.TIF')
muc_lm = load_gray(MICRO_DIR / 'White_10X_1.TIF')
asp_lm_mask = seg_micro(asp_lm, 'Pink')
muc_lm_mask = seg_micro(muc_lm, 'White')

stages = [0, 3, 5, 10, 20]

fig, axes = plt.subplots(2, len(stages), figsize=(183 * MM, 78 * MM))
fig.subplots_adjust(hspace=0.10, wspace=0.04)

for row, (img, mask, genus, color) in enumerate([
    (asp_lm, asp_lm_mask, 'Aspergillus', C_ASP),
    (muc_lm, muc_lm_mask, 'Mucor', C_MUC)]):

    nd = norm_display(img)
    orig_area = mask.sum()

    for col, n_iter in enumerate(stages):
        current = ndi.binary_erosion(mask, iterations=n_iter) if n_iter > 0 else mask.copy()
        survival = current.sum() / orig_area if orig_area > 0 else 0

        vis = np.stack([nd] * 3, axis=-1).copy()
        eroded_away = mask & ~current
        vis[eroded_away] = [0.35, 0.45, 0.85]
        vis[current] = [0.88, 0.18, 0.18]

        axes[row, col].imshow(vis)
        axes[row, col].axis('off')

        if row == 0:
            if col == 0:
                axes[row, col].set_title('Original', fontsize=6.5, fontweight='bold', pad=3)
            else:
                axes[row, col].set_title(f'Erosion = {n_iter}\n({survival:.0%} survived)',
                                          fontsize=6, pad=3)
        else:
            if col == 0:
                pass
            else:
                axes[row, col].set_title(f'{survival:.0%}', fontsize=6, pad=3)

    # Row label
    axes[row, 0].text(-0.08, 0.5, genus, transform=axes[row, 0].transAxes,
                       fontsize=8, style='italic', color=color, fontweight='bold',
                       rotation=90, va='center', ha='right')

legend_elements = [Patch(facecolor=[0.88, 0.18, 0.18], edgecolor='none', label='Survived'),
                   Patch(facecolor=[0.35, 0.45, 0.85], edgecolor='none', label='Eroded away')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=7,
           framealpha=0.9, edgecolor='none', bbox_to_anchor=(0.5, -0.01))

add_panel_label(axes[0, 0], 'A')
add_panel_label(axes[1, 0], 'B')
savefig(fig, 'figS20_erosion_demo')


# ═══════════════════════════════════════════════════════════════
print('S21: Local Density CV')
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(170 * MM, 130 * MM))
fig.subplots_adjust(hspace=0.22, wspace=0.22)

cal_px = 1.0
patch_um = 256
patch_px = int(round(patch_um / cal_px))

for row, (img, mask, genus, color) in enumerate([
    (asp_lm, asp_lm_mask, 'Aspergillus', C_ASP),
    (muc_lm, muc_lm_mask, 'Mucor', C_MUC)]):

    h, w = mask.shape
    nd = norm_display(img)

    # Mask + grid
    disp = np.stack([nd] * 3, axis=-1)
    overlay = disp.copy()
    overlay[mask] = [0.22, 0.65, 0.22]
    grid_img = (0.55 * disp + 0.45 * overlay).copy()
    for y0 in range(0, h - patch_px + 1, patch_px):
        grid_img[max(0, y0 - 1):y0 + 2, :, :] = [1, 0.82, 0]
    for x0 in range(0, w - patch_px + 1, patch_px):
        grid_img[:, max(0, x0 - 1):x0 + 2, :] = [1, 0.82, 0]

    axes[row, 0].imshow(grid_img)
    axes[row, 0].axis('off')
    axes[row, 0].set_title(f'{genus} + {patch_um} \u00b5m grid', fontsize=7.5,
                            style='italic', color=color, fontweight='bold', pad=4)

    # Patch density heatmap
    ny = (h - patch_px) // patch_px + 1
    nx = (w - patch_px) // patch_px + 1
    pm = np.full((ny, nx), np.nan)
    fracs = []
    for iy in range(ny):
        for ix in range(nx):
            f = mask[iy * patch_px:(iy + 1) * patch_px,
                     ix * patch_px:(ix + 1) * patch_px].mean()
            pm[iy, ix] = f
            fracs.append(f)
    fracs = np.array(fracs)
    cv = fracs.std(ddof=1) / fracs.mean() if fracs.mean() > 0 else 0

    im = axes[row, 1].imshow(pm, cmap='RdYlGn_r', aspect='equal',
                              vmin=0, vmax=0.8, interpolation='nearest')
    axes[row, 1].axis('off')
    axes[row, 1].set_title(f'Patch tissue fraction (CV = {cv:.3f})',
                            fontsize=8, fontweight='bold', pad=4)

    for iy in range(ny):
        for ix in range(nx):
            if not np.isnan(pm[iy, ix]):
                axes[row, 1].text(ix, iy, f'{pm[iy, ix]:.2f}', ha='center', va='center',
                                   fontsize=5.5, fontweight='bold',
                                   color='white' if pm[iy, ix] > 0.4 else 'black')

cbar = fig.colorbar(im, ax=axes[:, 1], shrink=0.6, pad=0.03, aspect=20)
cbar.set_label('Tissue fraction', fontsize=7)
cbar.ax.tick_params(labelsize=6)

add_panel_label(axes[0, 0], 'A')
add_panel_label(axes[0, 1], 'B')
savefig(fig, 'figS21_density_cv_legacy')


# ═══════════════════════════════════════════════════════════════
print('S22: Absorbing Capacity + SSA')
# ═══════════════════════════════════════════════════════════════

ft_a, ft_m, dt_a, dt_m, cap_a, cap_m = [], [], [], [], [], []
for k, v in saved.items():
    g = v.get('genus')
    rp = ROI_DIR / g / f'{Path(k).stem}_roi.jpg'
    if not rp.exists():
        continue
    img = load_gray(rp)
    mask = seg_3d(img)
    dt_arr = ndi.distance_transform_edt(mask) * CAL_3D
    ft = mask.mean()
    dt = np.median(dt_arr[mask]) if mask.sum() > 100 else 0
    if g == 'Aspergillus':
        ft_a.append(ft); dt_a.append(dt); cap_a.append(ft * dt)
    else:
        ft_m.append(ft); dt_m.append(dt); cap_m.append(ft * dt)

ft_a, ft_m = np.array(ft_a), np.array(ft_m)
dt_a, dt_m = np.array(dt_a), np.array(dt_m)
cap_a, cap_m = np.array(cap_a), np.array(cap_m)

iface = pd.read_csv(BASE / 'FigureHyphae' / 'output' / 'interfacial_metrics_3d.csv')
ssa_a = iface.loc[iface['genus'] == 'Aspergillus', 'specific_surface'].values
ssa_m = iface.loc[iface['genus'] == 'Mucor', 'specific_surface'].values

fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 55 * MM))
fig.subplots_adjust(wspace=0.55)

stripbox(axes[0], ft_a, ft_m, 'Tissue fraction')
stripbox(axes[1], dt_a, dt_m, 'Thickness (\u00b5m)')
stripbox(axes[2], cap_a, cap_m, 'Absorbing capacity\n($f$ \u00d7 thickness)')
stripbox(axes[3], ssa_a, ssa_m, 'Specific surface area\n(1/\u00b5m)')

# Highlight direction reversal on SSA
axes[3].set_title(r'$\it{Mucor}$ > $\it{Aspergillus}$', fontsize=6.5, color=C_MUC, pad=3)

for ax, l in zip(axes, 'ABCD'):
    add_panel_label(ax, l)

savefig(fig, 'figS22_absorbing_capacity')


print('\nAll 5 supplementary figures saved at 600 DPI.')
print(f'Output: {OUT}/')
