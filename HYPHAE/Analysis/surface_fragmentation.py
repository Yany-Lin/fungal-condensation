#!/usr/bin/env python3
"""Surface fragmentation analysis on 3D colony ROI crops.

Applies identical Otsu → connected components pipeline to both genera.
Computes perimeter density, gap width, lacunarity, fractal dimension.

Usage:
    python surface_fragmentation.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from scipy.ndimage import (label, binary_opening, binary_closing,
                           binary_erosion, distance_transform_edt)

BASE    = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / 'results'
ROI_DIR = OUT_DIR / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'

MM = 1 / 25.4
C_ASP = '#4CAF50'
C_MUC = '#757575'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 8, 'axes.linewidth': 0.6,
})


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


def load_roi_gray(path):
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def segment(img):
    """Otsu on raw grayscale. Dark = tissue."""
    lo, hi = np.percentile(img, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    norm8 = np.clip((img - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    thresh = otsu_uint8(norm8)
    mask = norm8 < thresh
    struct3 = np.ones((3, 3))
    struct5 = np.ones((5, 5))
    clean = binary_opening(mask, structure=struct3, iterations=2)
    clean = binary_closing(clean, structure=struct5, iterations=2)
    return clean


def component_metrics(mask, um_per_px):
    """Connected component analysis."""
    labeled, n_raw = label(mask)
    if n_raw == 0:
        return {}, labeled, np.array([])

    sizes = np.bincount(labeled.ravel())[1:]  # skip bg
    min_area = max(50, int(100 / um_per_px**2))  # ~100 µm² minimum
    big = np.where(sizes >= min_area)[0] + 1

    if len(big) == 0:
        return {}, labeled, np.array([])

    areas = np.array([sizes[lbl - 1] for lbl in big])
    struct3 = np.ones((3, 3))
    perimeters = []
    for lbl in big:
        cmask = labeled == lbl
        interior = binary_erosion(cmask, structure=struct3)
        perimeters.append(cmask.sum() - interior.sum())
    perimeters = np.array(perimeters)

    roi_area = mask.shape[0] * mask.shape[1]
    gap_mask = ~mask
    gap_edt = distance_transform_edt(gap_mask)

    # Largest component fraction
    largest_frac = areas.max() / areas.sum() if areas.sum() > 0 else 0

    m = {
        'n_components': len(big),
        'tissue_frac': round(float(mask.mean()), 4),
        'gap_frac': round(float(gap_mask.mean()), 4),
        'total_perim_um': round(float(perimeters.sum() * um_per_px), 1),
        'perim_density_um_per_1000um2': round(
            float(perimeters.sum() / roi_area / um_per_px * 1000), 4),
        'mean_gap_um': round(
            float(gap_edt[gap_mask].mean() * 2 * um_per_px) if gap_mask.any() else 0, 1),
        'median_gap_um': round(
            float(np.median(gap_edt[gap_mask]) * 2 * um_per_px) if gap_mask.any() else 0, 1),
        'mean_area_um2': round(float(areas.mean() * um_per_px**2), 1),
        'median_area_um2': round(float(np.median(areas) * um_per_px**2), 1),
        'largest_component_frac': round(float(largest_frac), 4),
    }
    return m, labeled, areas


def lacunarity(mask, box_sizes=None):
    """Gliding-box lacunarity. Returns (box_sizes, lambda_values)."""
    h, w = mask.shape
    if box_sizes is None:
        max_r = min(h, w) // 4
        box_sizes = [r for r in [4, 8, 16, 32, 64, 128, 256] if r <= max_r]
    if not box_sizes:
        return np.array([]), np.array([])

    mass = mask.astype(np.float32)
    # Integral image for fast box sums
    integ = np.cumsum(np.cumsum(mass, axis=0), axis=1)

    def box_sum(y0, x0, r):
        y1, x1 = y0 + r - 1, x0 + r - 1
        s = integ[y1, x1]
        if y0 > 0:
            s -= integ[y0 - 1, x1]
        if x0 > 0:
            s -= integ[y1, x0 - 1]
        if y0 > 0 and x0 > 0:
            s += integ[y0 - 1, x0 - 1]
        return s

    lambdas = []
    for r in box_sizes:
        sums = []
        step = max(1, r // 2)
        for y0 in range(0, h - r + 1, step):
            for x0 in range(0, w - r + 1, step):
                sums.append(box_sum(y0, x0, r))
        sums = np.array(sums)
        mu = sums.mean()
        if mu > 0:
            lam = sums.var() / (mu * mu) + 1
        else:
            lam = 1.0
        lambdas.append(lam)

    return np.array(box_sizes), np.array(lambdas)


def box_counting_dimension(mask):
    """Fractal dimension via box-counting."""
    h, w = mask.shape
    max_r = min(h, w) // 2
    sizes = [r for r in [2, 4, 8, 16, 32, 64, 128, 256, 512] if r <= max_r]
    if len(sizes) < 3:
        return np.nan

    counts = []
    for r in sizes:
        n = 0
        for y0 in range(0, h, r):
            for x0 in range(0, w, r):
                block = mask[y0:y0 + r, x0:x0 + r]
                if block.any():
                    n += 1
        counts.append(n)

    log_s = np.log(1.0 / np.array(sizes))
    log_n = np.log(np.array(counts).astype(float))
    valid = np.isfinite(log_n) & (log_n > 0)
    if valid.sum() < 3:
        return np.nan
    sl, _, r2, _, _ = stats.linregress(log_s[valid], log_n[valid])
    return sl


def compare(sa, sm, label):
    """Statistical comparison."""
    if len(sa) < 2 or len(sm) < 2:
        return f'\n{label}: insufficient data'
    t, tp = stats.ttest_ind(sa, sm, equal_var=False)
    u, up = stats.mannwhitneyu(sa, sm, alternative='two-sided')
    n1, n2 = len(sa), len(sm)
    pooled = np.sqrt(((n1 - 1) * sa.std(ddof=1)**2 + (n2 - 1) * sm.std(ddof=1)**2) / (n1 + n2 - 2))
    d = (sa.mean() - sm.mean()) / pooled if pooled > 0 else 0
    lines = [f'\n{label}:',
             f'  Asp: {sa.mean():.4f} ± {sa.std(ddof=1):.4f} (n={n1})',
             f'  Muc: {sm.mean():.4f} ± {sm.std(ddof=1):.4f} (n={n2})',
             f"  Welch t={t:.3f}, p={tp:.4f}",
             f'  Mann-Whitney U={u:.0f}, p={up:.4f}',
             f"  Cohen's d={d:.3f}"]
    return '\n'.join(lines)


def main():
    # Load session
    with open(SESSION) as f:
        session = json.load(f)

    cal = session.get('_calibration', {})
    um_per_px = cal.get('um_per_px', 1.0)
    print(f'Calibration: {um_per_px:.4f} µm/px')

    # Find all saved ROIs
    all_rows = []

    for genus_dir in [ROI_DIR / 'Aspergillus', ROI_DIR / 'Mucor']:
        if not genus_dir.exists():
            continue
        genus = genus_dir.name
        roi_files = sorted(genus_dir.glob('*_roi.jpg'))
        print(f'\n{genus}: {len(roi_files)} ROIs')

        for roi_path in roi_files:
            fname = roi_path.stem.replace('_roi', '') + '.JPG'

            # Check if deleted
            entry = session.get(fname, {})
            if entry.get('status') == 'deleted':
                continue

            img = load_roi_gray(roi_path)
            mask = segment(img)

            # Component metrics
            metrics, labeled, areas = component_metrics(mask, um_per_px)
            if not metrics:
                print(f'  {fname}: no components found, skipping')
                continue

            # Lacunarity
            bs, lam = lacunarity(mask)
            if len(lam) >= 2:
                # Mean lacunarity across scales
                metrics['lacunarity_mean'] = round(float(lam.mean()), 4)
                # Lacunarity at largest scale
                metrics['lacunarity_max_scale'] = round(float(lam[-1]), 4)
            else:
                metrics['lacunarity_mean'] = np.nan
                metrics['lacunarity_max_scale'] = np.nan

            # Fractal dimension
            D = box_counting_dimension(mask)
            metrics['fractal_dim'] = round(float(D), 4) if np.isfinite(D) else np.nan

            metrics['file'] = fname
            metrics['genus'] = genus
            all_rows.append(metrics)

            print(f'  {fname}: n={metrics["n_components"]}, '
                  f'perim_dens={metrics["perim_density_um_per_1000um2"]:.2f}, '
                  f'gap={metrics["mean_gap_um"]:.0f}µm, '
                  f'lac={metrics["lacunarity_mean"]:.3f}, '
                  f'D={metrics["fractal_dim"]:.3f}')

    if not all_rows:
        print('No data. Exiting.')
        return

    # Save CSV
    csv_path = OUT_DIR / 'fragmentation_results.csv'
    fields = ['file', 'genus'] + [k for k in all_rows[0] if k not in ('file', 'genus')]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nCSV: {csv_path}')

    # Statistics
    report = ['SURFACE FRAGMENTATION ANALYSIS', '=' * 60]
    report.append(f'Calibration: {um_per_px:.4f} µm/px')

    test_keys = [
        ('n_components', 'N components'),
        ('tissue_frac', 'Tissue fraction'),
        ('gap_frac', 'Gap fraction'),
        ('perim_density_um_per_1000um2', 'Perimeter density (µm/1000µm²)'),
        ('mean_gap_um', 'Mean gap width (µm)'),
        ('median_gap_um', 'Median gap width (µm)'),
        ('mean_area_um2', 'Mean component area (µm²)'),
        ('largest_component_frac', 'Largest component fraction'),
        ('lacunarity_mean', 'Mean lacunarity'),
        ('lacunarity_max_scale', 'Lacunarity (largest scale)'),
        ('fractal_dim', 'Fractal dimension'),
    ]

    for key, label_txt in test_keys:
        sa = np.array([r[key] for r in all_rows if r['genus'] == 'Aspergillus'
                       and np.isfinite(r.get(key, np.nan))])
        sm = np.array([r[key] for r in all_rows if r['genus'] == 'Mucor'
                       and np.isfinite(r.get(key, np.nan))])
        report.append(compare(sa, sm, label_txt))

    txt = '\n'.join(report)
    print(f'\n{txt}')
    stats_path = OUT_DIR / 'fragmentation_stats.txt'
    with open(stats_path, 'w') as f:
        f.write(txt)

    # Summary figure: boxplots for key metrics
    plot_keys = [
        ('perim_density_um_per_1000um2', 'Perimeter density\n(µm/1000µm²)'),
        ('mean_gap_um', 'Mean gap width\n(µm)'),
        ('lacunarity_mean', 'Mean lacunarity'),
        ('fractal_dim', 'Fractal dimension'),
    ]

    fig, axes = plt.subplots(1, len(plot_keys), figsize=(180 * MM, 55 * MM))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.25, wspace=0.45)

    for ax, (key, ylabel) in zip(axes, plot_keys):
        sa = [r[key] for r in all_rows if r['genus'] == 'Aspergillus'
              and np.isfinite(r.get(key, np.nan))]
        sm = [r[key] for r in all_rows if r['genus'] == 'Mucor'
              and np.isfinite(r.get(key, np.nan))]
        if not sa or not sm:
            ax.set_visible(False)
            continue

        bp = ax.boxplot([sa, sm], positions=[1, 2], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='white', lw=1.2))
        bp['boxes'][0].set_facecolor(C_ASP)
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(C_MUC)
        bp['boxes'][1].set_alpha(0.6)
        rng = np.random.default_rng(42)
        ax.scatter(1 + rng.uniform(-0.12, 0.12, len(sa)), sa,
                   s=10, c=C_ASP, alpha=0.7, edgecolors='none', zorder=3)
        ax.scatter(2 + rng.uniform(-0.12, 0.12, len(sm)), sm,
                   s=10, c=C_MUC, alpha=0.7, edgecolors='none', zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Asp.', 'Muc.'], style='italic', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

        # p-value bracket
        sa_arr, sm_arr = np.array(sa), np.array(sm)
        if len(sa_arr) >= 2 and len(sm_arr) >= 2:
            t, p = stats.ttest_ind(sa_arr, sm_arr, equal_var=False)
            ymax = max(sa_arr.max(), sm_arr.max())
            rng_y = ymax - min(sa_arr.min(), sm_arr.min())
            y = ymax + rng_y * 0.08
            ax.plot([1, 1, 2, 2], [y, y + rng_y * 0.03, y + rng_y * 0.03, y], 'k-', lw=0.6)
            p_str = f'p={p:.3f}' if p >= 0.001 else f'p={p:.1e}'
            ax.text(1.5, y + rng_y * 0.04, p_str, ha='center', fontsize=6)

    fig.suptitle('Surface fragmentation: 3D colony ROIs', fontsize=9, fontweight='bold')
    for ext in ('.png', '.pdf'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUT_DIR / f'fragmentation_summary{ext}', **kw)
    plt.close(fig)

    print(f'\nAll saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
