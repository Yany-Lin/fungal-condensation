#!/usr/bin/env python3
"""Supplementary schematic: f_tissue as a pixel-counting fraction.

Per professor's framing: f_tissue answers "how much of the colony footprint is
filled with hyphae versus empty space?" — literally white pixels / total pixels
after binarization. This figure isolates that idea: raw -> binary mask -> zoom
showing per-pixel classification + N_tissue/N_total.

Inputs: macro colony-surface ROIs at 0.94 µm/px (same as Fig 4 thickness panel).
Outputs figS_ftissue.{pdf,png,svg} in FigureHyphae/figures/.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Paths ──
_T7 = Path('/Volumes/T7/FINAL OSF')
_WIN = Path(r'D:\FINAL OSF')
_REPO = Path(__file__).resolve().parent.parent.parent
BASE = next((p for p in (_T7, _WIN, _REPO) if p.exists()), _REPO)
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
OUT = BASE / 'FigureHyphae' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

CAL_3D = 0.94
MM = 1 / 25.4
C_ASP = '#4CAF50'
C_MUC = '#757575'
C_BOUNDARY = '#FFD54F'   # warm yellow on dark mask
C_ZOOMBOX = '#FFD54F'

ASP_ROI_KEY = '20251214_222552.JPG'
MUC_ROI_KEY = '20251210_155926.JPG'

ZOOM_PX = 80

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'mathtext.default': 'regular',
})


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


def norm_display(img):
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    return np.clip((img - lo) / (hi - lo), 0, 1)


def pick_zoom_center(mask, zoom_px):
    """Center the zoom on a boundary pixel near the image center,
    such that the window contains a balanced mix of tissue/background."""
    boundary = find_boundaries(mask, mode='inner')
    h, w = mask.shape
    yy, xx = np.where(boundary)
    if len(yy) == 0:
        return h // 2, w // 2
    cy_lo, cy_hi = int(h * 0.25), int(h * 0.75)
    cx_lo, cx_hi = int(w * 0.25), int(w * 0.75)
    keep = (yy >= cy_lo) & (yy < cy_hi) & (xx >= cx_lo) & (xx < cx_hi)
    yy_c, xx_c = yy[keep], xx[keep]
    if len(yy_c) == 0:
        yy_c, xx_c = yy, xx
    half = zoom_px // 2
    best_score = -1.0
    best_yx = (yy_c[0], xx_c[0])
    rng = np.random.default_rng(7)
    idx = rng.choice(len(yy_c), size=min(300, len(yy_c)), replace=False)
    for k in idx:
        cy, cx = int(yy_c[k]), int(xx_c[k])
        if cy - half < 0 or cx - half < 0 or cy + half > h or cx + half > w:
            continue
        win = mask[cy - half:cy + half, cx - half:cx + half]
        f = win.mean()
        score = 1.0 - abs(f - 0.5) * 2
        if score > best_score:
            best_score = score
            best_yx = (cy, cx)
    return best_yx


def add_scalebar(ax, img_shape, cal_um, bar_um=100, color='white'):
    h, w = img_shape
    bar_px = bar_um / cal_um
    bx, by = w * 0.04, h * 0.94
    ax.plot([bx, bx + bar_px], [by, by], color=color, lw=2.5, solid_capstyle='butt')
    ax.text(bx + bar_px / 2, by - h * 0.04, f'{bar_um} µm', color=color,
            ha='center', fontsize=6.5, fontweight='bold')


def fmt_count(n):
    if n >= 1_000_000:
        return f'{n / 1e6:.2f}M'
    if n >= 1_000:
        return f'{n / 1e3:.1f}k'
    return f'{n}'


def savefig(fig, name):
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white', 'pad_inches': 0.04}
        if ext == '.png':
            kw['dpi'] = 600
        fig.savefig(OUT / f'{name}{ext}', **kw)
    plt.close(fig)
    print(f'  Saved: {name}')


# ── Compute ──
print('Loading ROIs...')
asp_img = load_gray(ROI_DIR / 'Aspergillus' / f'{Path(ASP_ROI_KEY).stem}_roi.jpg')
muc_img = load_gray(ROI_DIR / 'Mucor' / f'{Path(MUC_ROI_KEY).stem}_roi.jpg')
asp_mask = seg_3d(asp_img)
muc_mask = seg_3d(muc_img)

asp_f, muc_f = float(asp_mask.mean()), float(muc_mask.mean())
asp_n_t, asp_n_tot = int(asp_mask.sum()), int(asp_mask.size)
muc_n_t, muc_n_tot = int(muc_mask.sum()), int(muc_mask.size)
print(f'  Aspergillus: f={asp_f:.4f}  ({asp_n_t:,} / {asp_n_tot:,})')
print(f'  Mucor:       f={muc_f:.4f}  ({muc_n_t:,} / {muc_n_tot:,})')

asp_cy, asp_cx = pick_zoom_center(asp_mask, ZOOM_PX)
muc_cy, muc_cx = pick_zoom_center(muc_mask, ZOOM_PX)
half = ZOOM_PX // 2


# ── Figure ──
fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 110 * MM),
                          gridspec_kw={'width_ratios': [1.0, 1.0, 1.05]})
fig.subplots_adjust(hspace=0.18, wspace=0.10, left=0.04, right=0.98,
                     top=0.95, bottom=0.04)

rows = [
    (asp_img, asp_mask, asp_cy, asp_cx, 'Aspergillus', C_ASP,
     asp_f, asp_n_t, asp_n_tot),
    (muc_img, muc_mask, muc_cy, muc_cx, 'Mucor', C_MUC,
     muc_f, muc_n_t, muc_n_tot),
]

panel_letters = [['A', 'B', 'C'], ['D', 'E', 'F']]

for row, (img, mask, cy, cx, genus, color, f_val, n_t, n_tot) in enumerate(rows):
    nd = norm_display(img)

    # ── Col 1: raw ROI ──
    ax = axes[row, 0]
    ax.imshow(nd, cmap='gray', interpolation='nearest')
    ax.axis('off')
    add_scalebar(ax, img.shape, CAL_3D, 100)
    ax.text(0.03, 0.97, panel_letters[row][0], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='black',
                      alpha=0.55, edgecolor='none'))
    ax.text(0.98, 0.03, genus, transform=ax.transAxes, fontsize=8.5,
            ha='right', va='bottom', style='italic', color='white',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.22', facecolor=color,
                      alpha=0.85, edgecolor='none'))
    if row == 0:
        ax.set_title('Raw colony-surface ROI', fontsize=8.5, pad=4)

    # ── Col 2: binarized mask (tissue = white, void = black) ──
    ax = axes[row, 1]
    ax.imshow(mask, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')
    rect = Rectangle((cx - half, cy - half), ZOOM_PX, ZOOM_PX,
                      fill=False, ec=C_ZOOMBOX, lw=1.6)
    ax.add_patch(rect)
    ax.text(0.03, 0.97, panel_letters[row][1], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', color='white',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='black',
                      alpha=0.55, edgecolor='none'))
    ax.text(0.98, 0.03,
            f'N$_\\mathregular{{tissue}}$ = {fmt_count(n_t)}    '
            f'N$_\\mathregular{{total}}$ = {fmt_count(n_tot)}',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='white',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='black',
                      alpha=0.55, edgecolor='none'))
    if row == 0:
        ax.set_title('Binarized mask  (white = tissue,  black = void)',
                      fontsize=8.5, pad=4)

    # ── Col 3: zoom + equation ──
    ax = axes[row, 2]
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Place the zoom in the upper portion of the column
    zoom_ax = ax.inset_axes([0.06, 0.32, 0.62, 0.68])
    win_mask = mask[cy - half:cy + half, cx - half:cx + half]
    zoom_ax.imshow(win_mask, cmap='gray', interpolation='nearest',
                    vmin=0, vmax=1, extent=(0, ZOOM_PX, ZOOM_PX, 0))
    # Subtle pixel grid
    for i in range(0, ZOOM_PX + 1, 5):
        zoom_ax.axhline(i, color=C_ZOOMBOX, lw=0.18, alpha=0.55)
        zoom_ax.axvline(i, color=C_ZOOMBOX, lw=0.18, alpha=0.55)
    zoom_ax.set_xticks([])
    zoom_ax.set_yticks([])
    for sp in zoom_ax.spines.values():
        sp.set_edgecolor(C_ZOOMBOX)
        sp.set_linewidth(1.6)
    zoom_ax.set_title(f'{ZOOM_PX}×{ZOOM_PX} px  '
                       f'({ZOOM_PX*CAL_3D:.0f}×{ZOOM_PX*CAL_3D:.0f} µm)',
                       fontsize=6.8, pad=2)

    # Equation block in upper-right corner of column
    ax.text(0.74, 0.95,
            r'$f_{\mathrm{tissue}} = \dfrac{N_{\mathrm{tissue}}}{N_{\mathrm{total}}}$',
            fontsize=12, va='top', ha='left')
    ax.text(0.74, 0.55,
            f'= {f_val:.3f}',
            fontsize=13, fontweight='bold', va='top', ha='left', color=color)

    # Stacked horizontal bar
    bar_y, bar_h = 0.14, 0.10
    bar_x0, bar_x1 = 0.06, 0.94
    bar_w = bar_x1 - bar_x0
    ax.add_patch(Rectangle((bar_x0, bar_y), bar_w * f_val, bar_h,
                            facecolor='white', edgecolor='black', lw=0.6))
    ax.add_patch(Rectangle((bar_x0 + bar_w * f_val, bar_y),
                            bar_w * (1 - f_val), bar_h,
                            facecolor='black', edgecolor='black', lw=0.6))
    # Labels under the bar
    ax.plot([bar_x0, bar_x0 + bar_w * f_val],
            [bar_y - 0.015, bar_y - 0.015], color=color, lw=1.4,
            solid_capstyle='butt')
    ax.text(bar_x0 + bar_w * f_val / 2, bar_y - 0.045,
            f'tissue ({f_val:.0%})',
            ha='center', va='top', fontsize=7, color=color, fontweight='bold')
    ax.text(bar_x0 + bar_w * f_val + bar_w * (1 - f_val) / 2, bar_y - 0.045,
            f'void ({1 - f_val:.0%})',
            ha='center', va='top', fontsize=7, color='dimgray')

    # Panel letter
    ax.text(-0.02, 1.02, panel_letters[row][2], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

    if row == 0:
        # Header above col 3
        ax.text(0.5, 1.10, r'$f_{\mathrm{tissue}}$ = white-pixel fraction',
                transform=ax.transAxes, fontsize=8.5, ha='center',
                fontweight='bold')

savefig(fig, 'figS_ftissue')
print('Done.')
