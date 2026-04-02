#!/usr/bin/env python3
"""H5 mask overlays for all 6 RSR samples. Droplets coloured by distance
from boundary (parula). Requires T7 drive for raw images."""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw
import h5py, json
from pathlib import Path
from scipy.ndimage import distance_transform_edt
_PARULA_RGB = [
    [0.2081,0.1663,0.5292],[0.2116,0.1898,0.5777],[0.2123,0.2138,0.6270],
    [0.2081,0.2386,0.6771],[0.1959,0.2645,0.7279],[0.1707,0.2919,0.7792],
    [0.1253,0.3210,0.8303],[0.0591,0.3516,0.8683],[0.0117,0.3833,0.8820],
    [0.0060,0.4130,0.8828],[0.0082,0.4327,0.8805],[0.0219,0.4521,0.8800],
    [0.0416,0.4703,0.8790],[0.0609,0.4878,0.8770],[0.0795,0.5045,0.8751],
    [0.0961,0.5206,0.8729],[0.1133,0.5354,0.8699],[0.1306,0.5498,0.8664],
    [0.1494,0.5636,0.8616],[0.1687,0.5773,0.8554],[0.1865,0.5906,0.8477],
    [0.2026,0.6034,0.8380],[0.2150,0.6156,0.8270],[0.2269,0.6272,0.8148],
    [0.2379,0.6384,0.8011],[0.2487,0.6492,0.7862],[0.2595,0.6598,0.7707],
    [0.2702,0.6700,0.7548],[0.2821,0.6800,0.7388],[0.2952,0.6898,0.7226],
    [0.3087,0.6994,0.7063],[0.3219,0.7089,0.6905],[0.3349,0.7183,0.6749],
    [0.3483,0.7276,0.6592],[0.3630,0.7366,0.6426],[0.3781,0.7455,0.6247],
    [0.3929,0.7541,0.6062],[0.4076,0.7625,0.5869],[0.4232,0.7707,0.5667],
    [0.4394,0.7786,0.5455],[0.4559,0.7860,0.5237],[0.4723,0.7931,0.5015],
    [0.4888,0.7999,0.4788],[0.5063,0.8060,0.4554],[0.5253,0.8112,0.4304],
    [0.5455,0.8154,0.4040],[0.5661,0.8187,0.3765],[0.5864,0.8210,0.3480],
    [0.6062,0.8224,0.3188],[0.6256,0.8228,0.2889],[0.6449,0.8224,0.2582],
    [0.6640,0.8210,0.2274],[0.6828,0.8189,0.1965],[0.7010,0.8161,0.1650],
    [0.7190,0.8124,0.1330],[0.7367,0.8079,0.1004],[0.7545,0.8026,0.0700],
    [0.7724,0.7966,0.0454],[0.7905,0.7898,0.0248],[0.8085,0.7824,0.0136],
    [0.8269,0.7740,0.0066],[0.8638,0.7555,0.0100],[0.9349,0.9349,0.0549],
]
parula = LinearSegmentedColormap.from_list('parula', _PARULA_RGB, N=256)

MATLAB_ORANGE = (0.8500, 0.3250, 0.0980)

THIS_DIR = Path(__file__).parent
RAW_DIR  = THIS_DIR.parent / 'raw_data'
CAL_BASE = RAW_DIR / 'calibrations'
_T7_RAW = Path('/Volumes/T7/Fungal Hygroscopy/RAW/RSR RAW')
RAW_BASE = _T7_RAW if _T7_RAW.exists() else RAW_DIR / 'rsr_raw'

SAMPLES = [
    ('RSR1',         RAW_BASE / 'RSR 1',  CAL_BASE / 'rsr1.json'),
    ('RSR2',         RAW_BASE / 'RSR 2',  CAL_BASE / 'rsr2.json'),
    ('RSR7',         RAW_BASE / 'RSR10',  CAL_BASE / 'rsr10.json'),
    ('RSRDiseased3', RAW_BASE / 'RSR 3',  CAL_BASE / 'rsr3.json'),
    ('RSRDiseased5', RAW_BASE / 'RSR 5',  CAL_BASE / 'rsr5.json'),
    ('RSRDiseased6', RAW_BASE / 'RSR 6',  CAL_BASE / 'rsr6.json'),
]

OUT_DIR = Path(__file__).parent.parent / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_IDX   = 0       # first frame of evaporation
SHOW_MM     = 3.2     # mm to show on non-fungal side
FUNGAL_MM   = 1.2     # mm to show on fungal side
DIST_MAX_MM = 3.2     # parula range: 0 mm = blue, 3.2 mm = yellow
DPI         = 300


def render_sample(name, raw_dir, cal_json, frame_idx=FRAME_IDX):
    evap_dir = raw_dir / 'Every 30s from Evap'
    if not evap_dir.exists():
        print(f'  SKIP {name}: evap dir not found')
        return

    with open(cal_json) as f:
        cal = json.load(f)
    scale    = cal['calibration']['scale_um_per_px']
    bnd_pts  = np.array(cal['boundary_points_px'])
    img_w, img_h = cal['image_size']
    mm_to_px = 1000.0 / scale

    jpgs      = sorted(evap_dir.glob('*.jpg'))
    h5s       = sorted(evap_dir.glob('*_Probabilities.h5'))
    jpg_stems = {p.stem.replace(' EVAP', ''): p for p in jpgs}
    h5_stems  = {p.stem.replace('_Probabilities', ''): p for p in h5s}
    common    = sorted(set(jpg_stems) & set(h5_stems))
    if not common:
        print(f'  SKIP {name}: no matching jpg+h5 pairs')
        return
    idx      = int(0.75 * (len(common) - 1)) if frame_idx is None else min(frame_idx, len(common) - 1)
    stem     = common[idx]
    jpg_path = jpg_stems[stem]
    h5_path  = h5_stems[stem]
    print(f'  {name}: {jpg_path.name}')

    raw_img = np.array(Image.open(jpg_path).convert('RGB'))
    with h5py.File(h5_path, 'r') as f:
        data = f['exported_data'][:]
    ch0_mean = data[:, :, 0].mean()
    ch1_mean = data[:, :, 1].mean()
    droplet_ch = 0 if ch0_mean < ch1_mean else 1
    prob      = data[:, :, droplet_ch].astype(np.float32)
    p99       = max(float(np.percentile(prob, 99)), 0.01)
    prob_norm = np.clip(prob / p99, 0.0, 1.0)
    mask      = prob_norm > 0.5

    bnd_img = Image.new('L', (img_w, img_h), 0)
    ImageDraw.Draw(bnd_img).line([tuple(p) for p in bnd_pts.tolist()],
                                 fill=255, width=5)
    bnd_arr = np.array(bnd_img) > 0
    dist_mm = distance_transform_edt(~bnd_arr) * scale / 1000.0

    mean_bx     = bnd_pts[:, 0].mean()
    fungal_left = mean_bx <= img_w / 2
    fungal_mask = np.zeros((img_h, img_w), dtype=bool)
    for i in range(len(bnd_pts) - 1):
        y1b, y2b = int(bnd_pts[i][1]), int(bnd_pts[i+1][1])
        xb = int(bnd_pts[i][0])
        if y1b == y2b:
            continue
        y_lo, y_hi = min(y1b, y2b), max(y1b, y2b)
        for y in range(y_lo, min(y_hi + 1, img_h)):
            if fungal_left:
                fungal_mask[y, :xb] = True
            else:
                fungal_mask[y, xb:] = True

    bnd_x_max = int(bnd_pts[:, 0].max())
    bnd_x_min = int(bnd_pts[:, 0].min())
    bnd_y_min = int(bnd_pts[:, 1].min())
    bnd_y_max = int(bnd_pts[:, 1].max())
    if fungal_left:
        x0 = max(0, bnd_x_max - int(FUNGAL_MM * mm_to_px))
        x1 = min(img_w, bnd_x_max + int(SHOW_MM * mm_to_px))
    else:
        x0 = max(0, bnd_x_min - int(SHOW_MM * mm_to_px))
        x1 = min(img_w, bnd_x_min + int(FUNGAL_MM * mm_to_px))
    y0 = max(0, bnd_y_min + 40)
    y1 = min(img_h, bnd_y_max - 40)

    raw_crop    = raw_img[y0:y1, x0:x1]
    mask_crop   = mask[y0:y1, x0:x1]
    fungal_crop = fungal_mask[y0:y1, x0:x1]
    dist_crop   = dist_mm[y0:y1, x0:x1]
    show_mask   = mask_crop & ~fungal_crop

    bnd_shifted = bnd_pts.copy().astype(float)
    bnd_shifted[:, 0] -= x0
    bnd_shifted[:, 1] -= y0

    H, W  = raw_crop.shape[:2]
    rgba  = np.zeros((H, W, 4), dtype=np.float32)
    d_norm = np.clip(dist_crop / DIST_MAX_MM, 0.0, 1.0)
    colors = parula(d_norm)
    rgba[show_mask, :3] = colors[show_mask, :3]
    rgba[show_mask, 3]  = 0.88

    fig_w_px = x1 - x0
    fig_h_px = y1 - y0
    fig, ax  = plt.subplots(figsize=(fig_w_px / DPI, fig_h_px / DPI),
                             subplot_kw={'aspect': 'equal'})
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    gray   = np.array(Image.fromarray(raw_crop).convert('L'))
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    gray_s = np.clip((gray.astype(float) - lo) / (hi - lo), 0, 1)
    ax.imshow(gray_s, cmap='gray', vmin=0, vmax=1,
              interpolation='bicubic', origin='upper',
              extent=[0, fig_w_px, fig_h_px, 0])
    ax.imshow(rgba, origin='upper', extent=[0, fig_w_px, fig_h_px, 0],
              interpolation='nearest')

    bx, by = bnd_shifted[:, 0], bnd_shifted[:, 1]
    ax.plot(bx, by, color=MATLAB_ORANGE, lw=1.2, zorder=5, alpha=0.85,
            linestyle=(0, (5, 4)), solid_capstyle='butt', dash_capstyle='butt')

    bar_px = mm_to_px
    margin = 55
    bar_y  = fig_h_px - margin
    bar_x1 = fig_w_px - margin
    bar_x0 = bar_x1 - bar_px
    bar_h  = 10
    ax.fill_between([bar_x0, bar_x1], [bar_y - bar_h/2]*2, [bar_y + bar_h/2]*2,
                    color='white', zorder=6)
    ax.text((bar_x0 + bar_x1) / 2, bar_y - bar_h/2 - 7, '1 mm',
            color='white', fontsize=8, fontweight='bold',
            ha='center', va='bottom', zorder=6, fontfamily='Arial')

    ax.set_xlim(0, fig_w_px)
    ax.set_ylim(fig_h_px, 0)
    ax.axis('off')

    stem_out = f'mask_overlay_{name.lower()}'
    for ext in ('.png', '.pdf'):
        kw = {'bbox_inches': 'tight', 'pad_inches': 0, 'facecolor': '#0a0a0a'}
        if ext == '.png':
            kw['dpi'] = DPI
        fig.savefig(OUT_DIR / f'{stem_out}{ext}', **kw)
    plt.close(fig)
    print(f'    -> {OUT_DIR}/{stem_out}.png  ({fig_w_px}x{fig_h_px} px)')


def make_parula_colorbar():
    import matplotlib.colorbar as mcbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cal_rsr2 = CAL_BASE / 'rsr2.json'
    with open(cal_rsr2) as f:
        cal = json.load(f)
    scale = cal['calibration']['scale_um_per_px']   # µm/px (unused for bar, but confirms units)

    fig, ax = plt.subplots(figsize=(3.2, 0.45))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.72, bottom=0.38)

    norm = Normalize(vmin=0, vmax=DIST_MAX_MM)
    sm   = ScalarMappable(cmap=parula, norm=norm)
    sm.set_array([])

    cb = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cb.set_ticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, DIST_MAX_MM])
    cb.set_ticklabels(['0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', f'{DIST_MAX_MM:.1f}'])
    cb.ax.tick_params(labelsize=7, colors='white', length=3, width=0.6)
    cb.outline.set_edgecolor('white')
    cb.outline.set_linewidth(0.6)
    cb.set_label('Distance from boundary (mm)', fontsize=7, color='white',
                 labelpad=4, fontfamily='Arial')

    fig.patch.set_facecolor('#0a0a0a')
    ax.xaxis.label.set_color('white')

    out = OUT_DIR / 'parula_distance_colorbar.svg'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.05, facecolor='#0a0a0a',
                format='svg')
    plt.close(fig)
    print(f'  colorbar -> {out}')

make_parula_colorbar()

for name, raw_dir, cal_json in SAMPLES:
    render_sample(name, raw_dir, cal_json)

for _name in ('RSRDiseased6', 'RSR2'):
    _s = next(s for s in SAMPLES if s[0] == _name)
    render_sample(f'{_name}_75pct', _s[1], _s[2], frame_idx=None)
