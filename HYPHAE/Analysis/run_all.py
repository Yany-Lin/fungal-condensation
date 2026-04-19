#!/usr/bin/env python3
"""Master analysis script for HYPHAE folder.

Runs all three analyses and compiles results:
  1. 2D FFT spectral slope (Spacing 2/Green Asp + Spacing/W2 Muc)
  2. Hessian tubeness foreground fraction (Leyun microscopy)
  3. Laplacian variance (Spacing 2 root Asp + Spacing/W2 Muc)

All outputs go to HYPHAE/Hyphal Analysis/results/

Usage:
    python run_all.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from scipy.ndimage import gaussian_filter, laplace, distance_transform_edt
from scipy.ndimage import binary_opening, binary_closing

BASE     = Path(__file__).resolve().parents[1]  # HYPHAE/
THIS_DIR = Path(__file__).resolve().parent       # Hyphal Analysis/
OUT_DIR  = THIS_DIR / 'results'
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# DATA SOURCES
# ============================================================

# 2D FFT: original 14 Lab/Green Aspergillus + 7 quality Mucor
FFT_ASP_DIR = BASE / 'Spacing 2' / 'Green'
FFT_ASP_INCLUDE = {
    '20251214_221838', '20251214_221853', '20251214_221920',
    '20251214_221942', '20251214_222017', '20251214_222031',
    '20251214_222052', '20251214_222121', '20251214_222146',
    '20251214_222205', '20251214_222313', '20251214_222452',
    '20251214_222530', '20251214_222552',
}
FFT_MUC_DIR = BASE / 'Spacing' / 'VMS_JPG' / 'W2'
FFT_MUC_INCLUDE = {
    '20251210_155950', '20251210_155519', '20251210_155530',
    '20251210_155614', '20251210_155713', '20251210_155748',
    '20251210_155819',
}

# Hessian tubeness: Leyun microscopy
MICRO_DIR = BASE / 'Leyun microscopy'
MICRO_IMAGES = [
    ('Pink_10X_1.TIF', 'Aspergillus', 10),
    ('Pink_10X_2.TIF', 'Aspergillus', 10),
    ('Pink_20X_1.TIF', 'Aspergillus', 20),
    ('Pink_20X_2.TIF', 'Aspergillus', 20),
    ('Pink_40X_1.TIF', 'Aspergillus', 40),
    ('Pink_40X_2.TIF', 'Aspergillus', 40),
    ('White_10X_1.TIF', 'Mucor', 10),
    ('White_20X_1.TIF', 'Mucor', 20),
    ('White_40X_1.TIF', 'Mucor', 40),
]

# Laplacian: quality-filtered Spacing images
LAP_ASP_DIR = BASE / 'Spacing 2'
LAP_MUC_DIR = BASE / 'Spacing' / 'VMS_JPG' / 'W2'

DELTA = {
    'Aspergillus': np.array([279.5, 316.0, 297.8, 285.6, 311.7]),
    'Mucor': np.array([198.5, 126.2, 120.0, 131.6, 123.0]),
}

# ============================================================
# SHARED UTILITIES
# ============================================================

TILE_SIZE = 512
TILE_STRIDE = 256
FREQ_LO, FREQ_HI = 0.01, 0.45
FOCUS_PCTILE = 10
MIN_TEXTURE = 5.0
HESSIAN_SCALES = [1, 2, 4, 8, 16]

MM = 1 / 25.4
C_ASP = '#4CAF50'
C_MUC = '#757575'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'svg.fonttype': 'none',
})


def load_gray(path):
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def load_normalized(path):
    arr = load_gray(path)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo: hi = lo + 1
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def otsu_uint8(img8):
    hist = np.bincount(img8.ravel(), minlength=256).astype(float)
    total = hist.sum()
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    sum_bg = np.cumsum(hist * np.arange(256))
    mean_bg = sum_bg / np.maximum(w_bg, 1)
    mean_fg = (sum_bg[-1] - sum_bg) / np.maximum(w_fg, 1)
    var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(var))


def otsu_float(img_f):
    img8 = np.clip(img_f / max(img_f.max(), 1e-9) * 255, 0, 255).astype(np.uint8)
    return img_f.max() * otsu_uint8(img8) / 255


def list_jpgs(folder, include=None):
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.upper() == '.JPG' and not p.name.startswith('.')
        and p.stem.lower() != 'ruler'
        and (include is None or p.stem in include)
    )


def compare(sa, sm, label):
    lines = [f'\n{label}:']
    lines.append(f'  Asp: {sa.mean():.4f} +/- {sa.std(ddof=1):.4f} (n={len(sa)})')
    lines.append(f'  Muc: {sm.mean():.4f} +/- {sm.std(ddof=1):.4f} (n={len(sm)})')
    t, tp = stats.ttest_ind(sa, sm, equal_var=False)
    u, up = stats.mannwhitneyu(sa, sm, alternative='two-sided')
    n1, n2 = len(sa), len(sm)
    pooled = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2)/(n1+n2-2))
    d = (sa.mean() - sm.mean()) / pooled if pooled > 0 else 0
    lines.append(f"  Welch's t={t:.3f}, p={tp:.4f}")
    lines.append(f'  Mann-Whitney U={u:.0f}, p={up:.4f}')
    lines.append(f"  Cohen's d={d:.3f}")
    try:
        ks = stats.ks_2samp(sa, sm)
        lines.append(f'  KS D={ks.statistic:.3f}, p={ks.pvalue:.4f}')
    except: pass
    return lines, tp, d


# ============================================================
# ANALYSIS 1: 2D FFT SPECTRAL SLOPE
# ============================================================

def tile_radial(tile):
    n = tile.shape[0]
    c = tile - tile.mean()
    h = np.outer(np.hanning(n), np.hanning(n))
    fft = np.fft.fftshift(np.fft.fft2(c * h))
    p2d = np.abs(fft)**2
    cy, cx = n//2, n//2
    yi, xi = np.arange(n)-cy, np.arange(n)-cx
    yy, xx = np.meshgrid(yi, xi, indexing='ij')
    r = np.sqrt(xx**2+yy**2).astype(np.int32)
    rm = n//2
    rf = np.clip(r.ravel(), 0, rm-1)
    rs = np.bincount(rf, weights=p2d.ravel(), minlength=rm)
    rc = np.bincount(rf, minlength=rm).astype(float)
    rc[rc==0] = 1
    f = np.arange(rm)/n
    return f[1:], (rs/rc)[1:]


def fft_analyze_image(img):
    h, w = img.shape
    tiles = []
    for y0 in range(0, h-TILE_SIZE+1, TILE_STRIDE):
        for x0 in range(0, w-TILE_SIZE+1, TILE_STRIDE):
            t = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
            tiles.append((y0, x0, laplace(t).var(), t.std()))
    laps = np.array([t[2] for t in tiles])
    cutoff = np.percentile(laps, FOCUS_PCTILE)
    passed = [(y, x) for y, x, l, s in tiles if l >= cutoff and s >= MIN_TEXTURE]
    powers = []
    for y0, x0 in passed:
        t = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
        f, p = tile_radial(t)
        powers.append(p)
    if not powers:
        return np.nan, np.nan, 0, len(tiles)
    mp = np.mean(powers, axis=0)
    v = (f >= FREQ_LO) & (f <= FREQ_HI) & (mp > 0)
    if v.sum() < 10:
        return np.nan, np.nan, len(passed), len(tiles)
    sl, ic, r, p, se = stats.linregress(np.log10(f[v]), np.log10(mp[v]))
    return sl, r**2, len(passed), len(tiles)


def run_fft():
    print('\n' + '='*60)
    print('ANALYSIS 1: 2D FFT SPECTRAL SLOPE')
    print('='*60)

    rows = []
    for genus, folder, include in [
        ('Aspergillus', FFT_ASP_DIR, FFT_ASP_INCLUDE),
        ('Mucor', FFT_MUC_DIR, FFT_MUC_INCLUDE),
    ]:
        images = list_jpgs(folder, include)
        print(f'\n{genus}: {len(images)} images')
        for p in images:
            img = load_gray(p)
            sl, r2, nt, tot = fft_analyze_image(img)
            rows.append({'image': p.name, 'genus': genus, 'alpha': sl,
                         'r2': r2, 'tiles_pass': nt, 'tiles_total': tot,
                         'retention_pct': round(100*nt/tot, 1)})
            print(f'  {p.name}: alpha={sl:.3f}, R2={r2:.3f}, '
                  f'tiles={nt}/{tot} ({100*nt/tot:.0f}%)')

    # Save
    csv_path = OUT_DIR / 'fft_spectral_slopes.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sa = np.array([r['alpha'] for r in rows if r['genus'] == 'Aspergillus'])
    sm = np.array([r['alpha'] for r in rows if r['genus'] == 'Mucor'])
    lines, p_val, d_val = compare(sa, sm, '2D FFT Spectral Slope')
    for l in lines: print(l)
    return rows, sa, sm


# ============================================================
# ANALYSIS 2: HESSIAN TUBENESS (LEYUN MICROSCOPY)
# ============================================================

def hessian_tubeness(img, sigma):
    s2 = sigma * sigma
    Ixx = gaussian_filter(img, sigma=sigma, order=[0, 2]) * s2
    Iyy = gaussian_filter(img, sigma=sigma, order=[2, 0]) * s2
    Ixy = gaussian_filter(img, sigma=sigma, order=[1, 1]) * s2
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    disc = np.sqrt(np.clip((trace/2)**2 - det, 0, None))
    return np.clip(trace/2 + disc, 0, None)


def multiscale_tubeness(img):
    resp = np.zeros_like(img)
    for s in HESSIAN_SCALES:
        resp = np.maximum(resp, hessian_tubeness(img, s))
    return resp


def compute_porosity(mask):
    """Distance-weighted porosity within the mask."""
    clean = binary_opening(mask, structure=np.ones((3,3)), iterations=1)
    closed = binary_closing(clean, structure=np.ones((7,7)), iterations=2)
    pores = closed & ~clean
    if closed.sum() < 100:
        return 0.0, 0.0, closed, pores
    edt = distance_transform_edt(closed)
    sigma_w = max(edt.max() * 0.25, 3)
    weights = 1.0 / (1.0 + np.exp(-(edt - sigma_w) / max(sigma_w/4, 1)))
    weights[~closed] = 0
    w_sum = weights[closed].sum()
    w_por = weights[pores].sum() / w_sum if w_sum > 0 else 0
    uw_por = pores[closed].mean()
    return float(w_por), float(uw_por), closed, pores


def make_hessian_overlay(img, tubeness, mask, pores, fname, metrics, out_path):
    """4-panel QC overlay for Hessian analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(tubeness, cmap='inferno')
    axes[0, 1].set_title('Multi-scale Hessian tubeness', fontsize=10)
    axes[0, 1].axis('off')

    rgb = np.stack([img, img, img], axis=-1).copy()
    rgb[mask, 0] = np.clip(rgb[mask, 0] + 0.4, 0, 1)
    rgb[mask, 1] *= 0.4
    rgb[mask, 2] *= 0.4
    axes[1, 0].imshow(rgb)
    axes[1, 0].set_title(f'Detected structures (fg={metrics["fg_frac"]:.1%})', fontsize=10)
    axes[1, 0].axis('off')

    class_img = np.full((*img.shape, 3), 0.12)
    solid = mask & ~pores
    class_img[solid] = [0.2, 0.4, 0.9]
    class_img[pores] = [1.0, 0.9, 0.2]
    axes[1, 1].imshow(class_img)
    axes[1, 1].set_title(
        f'Internal: solid (blue) + pores (yellow) | '
        f'porosity={metrics["w_porosity"]:.3f}', fontsize=10)
    axes[1, 1].axis('off')

    fig.suptitle(f'{fname}  ({metrics["genus"]} {metrics["mag"]}X)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run_hessian():
    print('\n' + '='*60)
    print('ANALYSIS 2: HESSIAN TUBENESS (Leyun microscopy)')
    print('='*60)

    overlay_dir = OUT_DIR / 'hessian_overlays'
    overlay_dir.mkdir(exist_ok=True)

    rows = []
    for fname, genus, mag in MICRO_IMAGES:
        path = MICRO_DIR / fname
        img = load_normalized(path)
        tub = multiscale_tubeness(img)
        thresh = otsu_float(tub)
        mask = tub >= thresh
        fg = mask.mean()

        # Porosity
        w_por, uw_por, closed_mask, pores = compute_porosity(mask)

        metrics = {
            'file': fname, 'genus': genus, 'mag': mag,
            'fg_frac': round(fg, 4),
            'w_porosity': round(w_por, 4),
            'uw_porosity': round(uw_por, 4),
            'pore_px': int(pores.sum()),
            'mask_px': int(closed_mask.sum()),
        }
        rows.append(metrics)

        # QC overlay
        overlay_path = overlay_dir / f'{Path(fname).stem}_overlay.png'
        make_hessian_overlay(img, tub, mask, pores, fname, metrics, overlay_path)

        print(f'  {fname} ({genus} {mag}X): fg={fg:.1%}, '
              f'w_por={w_por:.4f}, uw_por={uw_por:.4f}')

    csv_path = OUT_DIR / 'hessian_foreground.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Summary figure: foreground fraction + porosity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(130*MM, 60*MM))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.85, bottom=0.20, wspace=0.50)
    for ax, key, ylabel in [
        (ax1, 'fg_frac', 'Foreground fraction'),
        (ax2, 'w_porosity', 'Internal porosity\n(distance-weighted)'),
    ]:
        asp_v = [r[key] for r in rows if r['genus'] == 'Aspergillus']
        muc_v = [r[key] for r in rows if r['genus'] == 'Mucor']
        bp = ax.boxplot([asp_v, muc_v], positions=[1, 2], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='white', lw=1.2))
        bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.6)
        rng = np.random.default_rng(42)
        ax.scatter(1+rng.uniform(-0.12, 0.12, len(asp_v)), asp_v,
                   s=18, c=C_ASP, alpha=0.7, edgecolors='none', zorder=3)
        ax.scatter(2+rng.uniform(-0.12, 0.12, len(muc_v)), muc_v,
                   s=18, c=C_MUC, alpha=0.7, edgecolors='none', zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Asp.', 'Muc.'], style='italic', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    fig.suptitle('Hessian tubeness detection', fontsize=9, fontweight='bold')
    for ext in ('.png', '.pdf'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png': kw['dpi'] = 200
        fig.savefig(OUT_DIR / f'hessian_summary{ext}', **kw)
    plt.close(fig)

    sa = np.array([r['fg_frac'] for r in rows if r['genus'] == 'Aspergillus'])
    sm = np.array([r['fg_frac'] for r in rows if r['genus'] == 'Mucor'])
    lines, p_val, d_val = compare(sa, sm, 'Hessian Foreground Fraction')
    for l in lines: print(l)
    return rows, sa, sm


# ============================================================
# ANALYSIS 3: LAPLACIAN VARIANCE (SPACING IMAGES)
# ============================================================

def lap_var(path):
    img = load_gray(path)
    lap = laplace(img[::4, ::4])
    return float(lap.var())


def run_laplacian():
    print('\n' + '='*60)
    print('ANALYSIS 3: LAPLACIAN VARIANCE (Spacing images)')
    print('='*60)

    rows = []

    # All Spacing 2 root images (Aspergillus)
    asp_files = list_jpgs(LAP_ASP_DIR)
    print(f'\nAspergillus: {len(asp_files)} images (Spacing 2 root)')
    for p in asp_files:
        v = lap_var(p)
        rows.append({'image': p.name, 'genus': 'Aspergillus', 'lap_var': round(v, 2)})
        print(f'  {p.name}: {v:.2f}')

    # Quality-filtered W2 (retention >= 65% from batch analysis)
    muc_files = list_jpgs(LAP_MUC_DIR)
    print(f'\nMucor: {len(muc_files)} images (W2)')
    for p in muc_files:
        v = lap_var(p)
        rows.append({'image': p.name, 'genus': 'Mucor', 'lap_var': round(v, 2)})
        print(f'  {p.name}: {v:.2f}')

    csv_path = OUT_DIR / 'laplacian_variance.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sa = np.array([r['lap_var'] for r in rows if r['genus'] == 'Aspergillus'])
    sm = np.array([r['lap_var'] for r in rows if r['genus'] == 'Mucor'])
    lines, p_val, d_val = compare(sa, sm, 'Laplacian Variance')
    for l in lines: print(l)
    return rows, sa, sm


# ============================================================
# SPACING OVERLAYS (Hessian tubeness on colony-surface photos)
# ============================================================

SPACING_MAX_PX = 2048  # downsample large JPGs for Hessian speed
SPACING_SCALES = [2, 4, 8, 16, 32]  # larger scales for macro features


def make_spacing_overlay(img, tubeness, mask, fname, genus, fg, out_path):
    """4-panel QC overlay for spacing images."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(tubeness, cmap='inferno')
    axes[0, 1].set_title('Multi-scale Hessian tubeness', fontsize=10)
    axes[0, 1].axis('off')

    rgb = np.stack([img, img, img], axis=-1).copy()
    rgb[mask, 0] = np.clip(rgb[mask, 0] + 0.4, 0, 1)
    rgb[mask, 1] *= 0.4
    rgb[mask, 2] *= 0.4
    axes[1, 0].imshow(rgb)
    axes[1, 0].set_title(f'Detected structures (fg={fg:.1%})', fontsize=10)
    axes[1, 0].axis('off')

    # Binary mask
    axes[1, 1].imshow(mask, cmap='gray')
    axes[1, 1].set_title('Binary mask', fontsize=10)
    axes[1, 1].axis('off')

    fig.suptitle(f'{fname}  ({genus})', fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run_spacing_overlays():
    print('\n' + '='*60)
    print('SPACING OVERLAYS: Hessian tubeness on colony-surface photos')
    print('='*60)

    asp_overlay_dir = OUT_DIR / 'spacing_overlays' / 'Aspergillus'
    muc_overlay_dir = OUT_DIR / 'spacing_overlays' / 'Mucor'
    asp_overlay_dir.mkdir(parents=True, exist_ok=True)
    muc_overlay_dir.mkdir(parents=True, exist_ok=True)

    for genus, folder, overlay_dir in [
        ('Aspergillus', LAP_ASP_DIR, asp_overlay_dir),
        ('Mucor', LAP_MUC_DIR, muc_overlay_dir),
    ]:
        images = list_jpgs(folder)
        print(f'\n{genus}: {len(images)} images')
        for p in images:
            img_full = load_normalized(p)
            # Downsample for speed
            h, w = img_full.shape
            scale = max(1, max(h, w) / SPACING_MAX_PX)
            if scale > 1:
                s = int(scale)
                img = img_full[::s, ::s]
            else:
                img = img_full

            # Hessian tubeness with macro-appropriate scales
            resp = np.zeros_like(img)
            for sigma in SPACING_SCALES:
                resp = np.maximum(resp, hessian_tubeness(img, sigma))
            thresh = otsu_float(resp)
            mask = resp >= thresh
            fg = mask.mean()

            out_path = overlay_dir / f'{p.stem}_overlay.png'
            make_spacing_overlay(img, resp, mask, p.name, genus, fg, out_path)
            print(f'  {p.name}: fg={fg:.1%}')

    print(f'\nSpacing overlays saved to: {OUT_DIR}/spacing_overlays/')


# ============================================================
# SUMMARY FIGURE
# ============================================================

def make_summary(fft_sa, fft_sm, hess_sa, hess_sm, lap_sa, lap_sm):
    fig, axes = plt.subplots(1, 3, figsize=(180*MM, 60*MM))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.22, wspace=0.45)

    datasets = [
        (axes[0], fft_sa, fft_sm, r'Spectral slope $\alpha$', '2D FFT\n(colony surface)'),
        (axes[1], hess_sa, hess_sm, 'Foreground fraction', 'Hessian tubeness\n(microscopy)'),
        (axes[2], lap_sa, lap_sm, 'Laplacian variance', 'Laplacian\n(colony surface)'),
    ]

    for ax, sa, sm, ylabel, subtitle in datasets:
        bp = ax.boxplot([sa, sm], positions=[1, 2], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='white', lw=1.2))
        bp['boxes'][0].set_facecolor(C_ASP); bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(C_MUC); bp['boxes'][1].set_alpha(0.6)
        rng = np.random.default_rng(42)
        ax.scatter(1+rng.uniform(-0.12, 0.12, len(sa)), sa,
                   s=12, c=C_ASP, alpha=0.7, edgecolors='none', zorder=3)
        ax.scatter(2+rng.uniform(-0.12, 0.12, len(sm)), sm,
                   s=12, c=C_MUC, alpha=0.7, edgecolors='none', zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Asp.', 'Muc.'], style='italic', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(subtitle, fontsize=7, color='#666')

        t, p = stats.ttest_ind(sa, sm, equal_var=False)
        ymax = max(sa.max(), sm.max())
        rng_y = ymax - min(sa.min(), sm.min())
        y = ymax + rng_y * 0.08
        ax.plot([1, 1, 2, 2], [y, y+rng_y*0.03, y+rng_y*0.03, y], 'k-', lw=0.6)
        ax.text(1.5, y+rng_y*0.04,
                f'p={p:.3f}' if p >= 0.001 else f'p={p:.1e}',
                ha='center', fontsize=6)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png': kw['dpi'] = 300
        fig.savefig(OUT_DIR / f'summary_three_methods{ext}', **kw)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print('HYPHAE MASTER ANALYSIS')
    print('=' * 60)

    fft_rows, fft_sa, fft_sm = run_fft()
    hess_rows, hess_sa, hess_sm = run_hessian()
    lap_rows, lap_sa, lap_sm = run_laplacian()
    run_spacing_overlays()

    # Summary figure
    make_summary(fft_sa, fft_sm, hess_sa, hess_sm, lap_sa, lap_sm)

    # Combined stats report
    report = []
    report.append('HYPHAE ANALYSIS — COMBINED REPORT')
    report.append('=' * 60)

    for label, sa, sm in [
        ('2D FFT spectral slope', fft_sa, fft_sm),
        ('Hessian foreground fraction', hess_sa, hess_sm),
        ('Laplacian variance', lap_sa, lap_sm),
    ]:
        lines, p, d = compare(sa, sm, label)
        report.extend(lines)

    report.append('\n\nDelta values:')
    for g in ['Aspergillus', 'Mucor']:
        d = DELTA[g]
        report.append(f'  {g}: delta = {d.mean():.1f} +/- {d.std(ddof=1):.1f} um (n={len(d)})')

    txt = '\n'.join(report)
    print(f'\n{txt}')
    with open(OUT_DIR / 'combined_stats.txt', 'w') as f:
        f.write(txt)

    print(f'\n{"="*60}')
    print(f'All results saved to: {OUT_DIR}/')
    print(f'  fft_spectral_slopes.csv')
    print(f'  hessian_foreground.csv')
    print(f'  hessian_overlays/*.png  (9 QC overlays)')
    print(f'  hessian_summary.{{png,pdf}}')
    print(f'  spacing_overlays/Aspergillus/*.png')
    print(f'  spacing_overlays/Mucor/*.png')
    print(f'  laplacian_variance.csv')
    print(f'  combined_stats.txt')
    print(f'  summary_three_methods.{{png,pdf,svg}}')


if __name__ == '__main__':
    main()
