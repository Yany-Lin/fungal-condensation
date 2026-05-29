#!/usr/bin/env python3
"""Render full-resolution side-by-side raw + bright-mask previews.

Settings (committed): adaptive bright-response, σ_smooth=1, σ_local=32, k=0.5,
morph 1 iter, polarity = bright (white pixel = hyphae).

Outputs:
  D:/FINAL OSF/FigureHyphae/figures/preview_fullres/
    asp_canonical_raw_vs_mask.png       (canonical Asp)
    muc_canonical_raw_vs_mask.png       (canonical Muc)
    all24/*_raw_vs_mask.png             (one per ROI)
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

BASE = Path(r'D:\FINAL OSF')
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
OUT_DIR = BASE / 'FigureHyphae' / 'figures' / 'preview_fullres'
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALL_DIR = OUT_DIR / 'all24'
ALL_DIR.mkdir(parents=True, exist_ok=True)

ASP_KEY = '20251214_222552.JPG'
MUC_KEY = '20251210_155926.JPG'

CAL_3D = 0.94
SIGMA_SMOOTH = 1.0
SIGMA_LOCAL = 32.0
K = 0.5
MORPH_ITERS = 1

DIVIDER_PX = 8         # gold divider width between raw and mask
GUTTER_COLOR = (255, 200, 50)   # warm yellow
LABEL_BG = (0, 0, 0, 160)
LABEL_FG = (255, 255, 255, 255)
ASP_BADGE = (76, 175, 80)
MUC_BADGE = (117, 117, 117)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')


def gauss(img, sigma):
    if sigma <= 0: return img
    r = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
    a = img.unsqueeze(0).unsqueeze(0)
    a = F.pad(a, (r, r, 0, 0), mode='reflect')
    a = F.conv2d(a, k.view(1, 1, 1, -1))
    a = F.pad(a, (0, 0, r, r), mode='reflect')
    a = F.conv2d(a, k.view(1, 1, -1, 1))
    return a.squeeze(0).squeeze(0)


def morph(mask, iters=1):
    m = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(iters): m = F.max_pool2d(m, 3, stride=1, padding=1)
    for _ in range(iters): m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters): m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters): m = F.max_pool2d(m, 3, stride=1, padding=1)
    return m.squeeze(0).squeeze(0) > 0.5


def bright_response(img_gpu):
    s = gauss(img_gpu, SIGMA_SMOOTH)
    l = gauss(s, SIGMA_LOCAL)
    br = s - l
    t = br.mean() + K * br.std()
    return morph(br > t, MORPH_ITERS)


def load_gray(p):
    with Image.open(p) as im:
        a = np.asarray(im).astype(np.float32)
    if a.ndim == 3: a = a[..., :3].mean(axis=2)
    return a


def display_raw(img_np):
    lo, hi = np.percentile(img_np, [0.5, 99.5])
    return np.clip((img_np - lo) / max(hi - lo, 1), 0, 1)


def compose_side_by_side(img_np, mask_np, genus_label=None, f_val=None,
                          badge_color=None):
    """Return a side-by-side PIL.Image at NATIVE resolution.
    Layout: [raw] [gold divider] [mask]
    Annotations: genus badge bottom-left of raw, f value bottom-right of mask.
    """
    h, w = img_np.shape
    # Raw → 3-channel uint8
    raw_disp = (display_raw(img_np) * 255).astype(np.uint8)
    raw_rgb = np.stack([raw_disp] * 3, axis=-1)
    # Mask → 3-channel uint8 (white = tissue, black = background)
    mask_rgb = np.stack([(mask_np * 255).astype(np.uint8)] * 3, axis=-1)
    # Divider
    div = np.tile(np.array(GUTTER_COLOR, dtype=np.uint8),
                   (h, DIVIDER_PX, 1))
    combined = np.concatenate([raw_rgb, div, mask_rgb], axis=1)
    im = Image.fromarray(combined, mode='RGB').convert('RGBA')

    # Annotations
    draw = ImageDraw.Draw(im, mode='RGBA')
    # Choose label size relative to image height (~3% of height)
    font_px = max(18, int(h * 0.03))
    try:
        font = ImageFont.truetype('arial.ttf', font_px)
    except Exception:
        try:
            font = ImageFont.truetype('DejaVuSans-Bold.ttf', font_px)
        except Exception:
            font = ImageFont.load_default()

    pad = max(8, int(font_px * 0.4))
    # Genus badge (bottom-left of raw panel)
    if genus_label and badge_color:
        bbox = draw.textbbox((0, 0), genus_label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bx, by = pad * 2, h - th - pad * 3
        draw.rounded_rectangle(
            [bx - pad, by - pad // 2, bx + tw + pad, by + th + pad // 2],
            radius=pad // 2,
            fill=badge_color + (220,))
        draw.text((bx, by), genus_label, font=font, fill=LABEL_FG)
    # f_tissue value (bottom-right of mask panel)
    if f_val is not None:
        text = f'f = {f_val:.3f}'
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bx = combined.shape[1] - tw - pad * 2
        by = h - th - pad * 3
        draw.rounded_rectangle(
            [bx - pad, by - pad // 2, bx + tw + pad, by + th + pad // 2],
            radius=pad // 2, fill=(0, 0, 0, 170))
        draw.text((bx, by), text, font=font, fill=LABEL_FG)

    # Column titles at the very top
    title_font_px = max(20, int(h * 0.035))
    try: tfont = ImageFont.truetype('arial.ttf', title_font_px)
    except Exception:
        try: tfont = ImageFont.truetype('DejaVuSans-Bold.ttf', title_font_px)
        except Exception: tfont = font
    title_pad = title_font_px // 2
    # Build a header strip and stack it above the image
    header_h = title_font_px + title_pad * 2
    header = Image.new('RGBA', (im.size[0], header_h),
                        (255, 255, 255, 255))
    hdraw = ImageDraw.Draw(header)
    # 'Raw ROI' centered over the raw panel
    raw_center_x = w // 2
    bbox_r = hdraw.textbbox((0, 0), 'Raw ROI', font=tfont)
    hdraw.text((raw_center_x - (bbox_r[2] - bbox_r[0]) // 2, title_pad),
                'Raw ROI', font=tfont, fill=(20, 20, 20, 255))
    # 'Mask (white = hyphae)' centered over the mask panel
    mask_center_x = w + DIVIDER_PX + w // 2
    txt = 'Mask  (white = hyphae)'
    bbox_m = hdraw.textbbox((0, 0), txt, font=tfont)
    hdraw.text((mask_center_x - (bbox_m[2] - bbox_m[0]) // 2, title_pad),
                txt, font=tfont, fill=(20, 20, 20, 255))
    # Stack header on top of image
    final = Image.new('RGBA', (im.size[0], im.size[1] + header_h),
                       (255, 255, 255, 255))
    final.paste(header, (0, 0))
    final.paste(im, (0, header_h), im)
    return final.convert('RGB')


# ── Run ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}

records = []
for key, val in saved.items():
    g = val.get('genus')
    rp = ROI_DIR / g / f'{Path(key).stem}_roi.jpg'
    if rp.exists():
        records.append({'key': key, 'genus': g, 'path': rp})

print(f'Loaded {len(records)} ROIs')
print(f'Settings: σ_smooth={SIGMA_SMOOTH}, σ_local={SIGMA_LOCAL}, k={K}, '
       f'morph={MORPH_ITERS}, polarity=BRIGHT')

n_done = 0
for r in records:
    img_np = load_gray(r['path'])
    img_gpu = torch.from_numpy(img_np).to(DEVICE)
    mask = bright_response(img_gpu)
    mask_np = mask.cpu().numpy()
    f_val = float(mask_np.mean())

    badge = ASP_BADGE if r['genus'] == 'Aspergillus' else MUC_BADGE
    label = r['genus']
    pil = compose_side_by_side(img_np, mask_np,
                                 genus_label=label, f_val=f_val,
                                 badge_color=badge)
    # All-24 gallery (named by genus + key stem)
    stem = Path(r['key']).stem
    out_all = ALL_DIR / f'{r["genus"][:3]}_{stem}_raw_vs_mask.png'
    pil.save(out_all, format='PNG', optimize=False, compress_level=1)
    # Canonical featured
    if r['key'] == ASP_KEY:
        pil.save(OUT_DIR / 'asp_canonical_raw_vs_mask.png',
                  format='PNG', optimize=False, compress_level=1)
        print(f'  Asp canonical: {OUT_DIR / "asp_canonical_raw_vs_mask.png"}  '
              f'({pil.size[0]}×{pil.size[1]} px, f={f_val:.3f})')
    if r['key'] == MUC_KEY:
        pil.save(OUT_DIR / 'muc_canonical_raw_vs_mask.png',
                  format='PNG', optimize=False, compress_level=1)
        print(f'  Muc canonical: {OUT_DIR / "muc_canonical_raw_vs_mask.png"}  '
              f'({pil.size[0]}×{pil.size[1]} px, f={f_val:.3f})')
    n_done += 1

print(f'\nWrote {n_done} ROIs full-resolution side-by-side images:')
print(f'  Featured (canonical): {OUT_DIR}')
print(f'  All 24:               {ALL_DIR}')
print('Done.')
