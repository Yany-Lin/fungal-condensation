#!/usr/bin/env python3
"""Interactive threshold explorer for colony-surface ROIs.

Launches a local Gradio web app at http://127.0.0.1:7860 with:
  - ROI selector (all 24 Asp + Muc ROIs)
  - Method picker (adaptive dark-response, Sauvola, Otsu + offset)
  - Live sliders for each method's parameters
  - Side-by-side raw / mask / overlay
  - Live f_tissue readout
  - "Save current params" button (writes JSON next to outputs)

GPU-accelerated (RTX 5070 Ti). Each slider move re-segments in <200 ms.
Closes with Ctrl-C in terminal.
"""

import json, time
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import gradio as gr

# ── Paths ──
BASE = Path(r'D:\FINAL OSF')
ROI_DIR = BASE / 'HYPHAE' / 'Analysis' / 'results' / '3d_overlays'
SESSION = ROI_DIR / 'roi_session.json'
OUT_JSON = BASE / 'FigureHyphae' / 'output' / 'threshold_explorer_params.json'

CAL_3D = 0.94
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0) if DEVICE == "cuda" else "n/a"}')


# ── Load ROIs once into memory (CPU); push to GPU on demand ──
with open(SESSION) as f:
    saved = {k: v for k, v in json.load(f).items()
             if not k.startswith('_') and v.get('status') != 'deleted'}

roi_cache = {}        # label -> (img_np, img_gpu)
roi_labels = []       # display strings, ordered Asp first, then Muc
for genus in ('Aspergillus', 'Mucor'):
    for key, val in saved.items():
        if val.get('genus') != genus: continue
        rp = ROI_DIR / genus / f'{Path(key).stem}_roi.jpg'
        if not rp.exists(): continue
        with Image.open(rp) as im:
            a = np.asarray(im).astype(np.float32)
        if a.ndim == 3: a = a[..., :3].mean(axis=2)
        label = f'{genus[:3]} | {key}'
        roi_cache[label] = (a, torch.from_numpy(a).to(DEVICE))
        roi_labels.append(label)
print(f'Loaded {len(roi_labels)} ROIs')


# ── GPU primitives ──
def _gauss(img, sigma):
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


def _morph(mask, iters=1):
    m = mask.float().unsqueeze(0).unsqueeze(0)
    for _ in range(iters): m = F.max_pool2d(m, 3, stride=1, padding=1)
    for _ in range(iters): m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters): m = -F.max_pool2d(-m, 3, stride=1, padding=1)
    for _ in range(iters): m = F.max_pool2d(m, 3, stride=1, padding=1)
    return (m.squeeze(0).squeeze(0) > 0.5)


def _otsu(values_1d, nbins=512):
    vmin, vmax = float(values_1d.min()), float(values_1d.max())
    if vmax <= vmin: return vmin
    hist = torch.histc(values_1d, bins=nbins, min=vmin, max=vmax)
    centers = torch.linspace(vmin, vmax, nbins, device=values_1d.device)
    wb = torch.cumsum(hist, 0); wf = hist.sum() - wb
    sum_b = torch.cumsum(hist * centers, 0)
    sum_t = (hist * centers).sum()
    mb = sum_b / wb.clamp(min=1)
    mf = (sum_t - sum_b) / wf.clamp(min=1)
    var = wb * wf * (mb - mf) ** 2
    var = torch.where((wb > 0) & (wf > 0), var, torch.zeros_like(var))
    return float(centers[int(torch.argmax(var).item())])


# ── Segmentation methods ──
def seg_adaptive(img_gpu, sigma_smooth, sigma_local, k_thresh, morph_iters):
    s = _gauss(img_gpu, float(sigma_smooth))
    l = _gauss(s, float(sigma_local))
    dr = l - s
    t = dr.mean() + float(k_thresh) * dr.std()
    mask = dr > t
    return _morph(mask, int(morph_iters)) if morph_iters > 0 else mask


def seg_sauvola(img_gpu, window, k, R, morph_iters):
    lo, hi = img_gpu.min(), img_gpu.max()
    n = (img_gpu - lo) / max(float(hi - lo), 1e-6)
    w = int(window)
    x = n.unsqueeze(0).unsqueeze(0)
    mean = F.avg_pool2d(x, w, stride=1, padding=w // 2, count_include_pad=False)
    msq = F.avg_pool2d(x ** 2, w, stride=1, padding=w // 2, count_include_pad=False)
    mean = mean[..., :n.shape[0], :n.shape[1]]
    msq = msq[..., :n.shape[0], :n.shape[1]]
    std = (msq - mean ** 2).clamp(min=0).sqrt()
    T = mean * (1 + float(k) * (std / float(R) - 1))
    mask = (x < T).squeeze(0).squeeze(0)
    return _morph(mask, int(morph_iters)) if morph_iters > 0 else mask


def seg_otsu_offset(img_gpu, sigma_smooth, offset, morph_iters):
    s = _gauss(img_gpu, float(sigma_smooth))
    t = _otsu(s.ravel()) + float(offset) * float(s.std())
    mask = s < t
    return _morph(mask, int(morph_iters)) if morph_iters > 0 else mask


# ── Rendering ──
def make_overlay(img_np, mask_np, alpha=0.45, color=(0.95, 0.15, 0.15)):
    """RGB overlay: grayscale image with red highlight where mask=True."""
    lo, hi = np.percentile(img_np, [0.5, 99.5])
    nd = np.clip((img_np - lo) / max(hi - lo, 1), 0, 1)
    rgb = np.stack([nd, nd, nd], axis=-1)
    overlay = rgb.copy()
    overlay[mask_np] = (1 - alpha) * rgb[mask_np] + alpha * np.array(color)
    return (overlay * 255).astype(np.uint8)


def make_mask_image(mask_np):
    return (mask_np.astype(np.uint8) * 255)


def make_raw_image(img_np):
    lo, hi = np.percentile(img_np, [0.5, 99.5])
    nd = np.clip((img_np - lo) / max(hi - lo, 1), 0, 1)
    return (nd * 255).astype(np.uint8)


# ── Gradio callback ──
def update(roi_label, method, polarity,
            adap_smooth, adap_local, adap_k, adap_morph,
            sau_window, sau_k, sau_R, sau_morph,
            ots_smooth, ots_offset, ots_morph):
    if roi_label not in roi_cache:
        return None, None, None, '—'
    img_np, img_gpu = roi_cache[roi_label]
    t0 = time.time()
    if method == 'Adaptive (dark-response)':
        mask = seg_adaptive(img_gpu, adap_smooth, adap_local, adap_k, adap_morph)
    elif method == 'Sauvola (local)':
        mask = seg_sauvola(img_gpu, sau_window, sau_k, sau_R, sau_morph)
    elif method == 'Otsu + offset':
        mask = seg_otsu_offset(img_gpu, ots_smooth, ots_offset, ots_morph)
    else:
        return None, None, None, 'unknown method'
    # Polarity: by default the methods are written for tissue = darker than
    # background. If the user says hyphae are actually the brighter pixels
    # (e.g. white Mucor cotton, raised conidial heads catching light), flip.
    if polarity == 'tissue = brighter':
        mask = ~mask
    mask_np = mask.cpu().numpy()
    elapsed = (time.time() - t0) * 1000
    raw_img = make_raw_image(img_np)
    mask_img = make_mask_image(mask_np)
    overlay_img = make_overlay(img_np, mask_np)
    f_val = float(mask_np.mean())
    n_t = int(mask_np.sum()); n_tot = int(mask_np.size)
    summary = (f'**f_tissue = {f_val:.4f}**  '
                f'({n_t:,} / {n_tot:,} px)   '
                f'compute: {elapsed:.0f} ms')
    return raw_img, mask_img, overlay_img, summary


def save_params(roi_label, method, polarity,
                 adap_smooth, adap_local, adap_k, adap_morph,
                 sau_window, sau_k, sau_R, sau_morph,
                 ots_smooth, ots_offset, ots_morph):
    params = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'roi': roi_label, 'method': method, 'polarity': polarity,
        'adaptive': {'sigma_smooth': adap_smooth, 'sigma_local': adap_local,
                      'k_threshold': adap_k, 'morph_iters': adap_morph},
        'sauvola':  {'window': sau_window, 'k': sau_k, 'R': sau_R,
                      'morph_iters': sau_morph},
        'otsu_offset': {'sigma_smooth': ots_smooth, 'offset': ots_offset,
                         'morph_iters': ots_morph},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if OUT_JSON.exists():
        try:
            history = json.loads(OUT_JSON.read_text())
            if not isinstance(history, list): history = [history]
        except Exception: history = []
    history.append(params)
    OUT_JSON.write_text(json.dumps(history, indent=2))
    return f'Saved {len(history)} param sets to {OUT_JSON}'


# ── UI ──
with gr.Blocks(title='Hyphae Threshold Explorer',
                theme=gr.themes.Soft()) as demo:
    gr.Markdown(f'# Hyphae Threshold Explorer\n'
                 f'24 ROIs (13 Asp + 11 Muc), GPU = `{DEVICE}`. '
                 f'Pixel calibration: {CAL_3D} µm/px.')
    with gr.Row():
        roi_dd = gr.Dropdown(roi_labels, value=roi_labels[0],
                              label='ROI', scale=2)
        method_dd = gr.Dropdown(
            ['Adaptive (dark-response)', 'Sauvola (local)', 'Otsu + offset'],
            value='Adaptive (dark-response)', label='Method', scale=1)
        polarity_dd = gr.Radio(
            ['tissue = darker', 'tissue = brighter'],
            value='tissue = brighter',
            label='Polarity (which pixels are hyphae?)', scale=2)
        save_btn = gr.Button('💾 Save params', scale=1)
    save_msg = gr.Markdown('')

    with gr.Tabs():
        with gr.Tab('Adaptive (dark-response)'):
            with gr.Row():
                adap_smooth = gr.Slider(0, 4, value=1.0, step=0.1,
                                          label='σ_smooth (px)')
                adap_local = gr.Slider(4, 96, value=32, step=2,
                                         label='σ_local (px)')
                adap_k = gr.Slider(-2.0, 3.0, value=0.5, step=0.05,
                                     label='threshold k (T = μ + k·σ)')
                adap_morph = gr.Slider(0, 3, value=1, step=1,
                                         label='morph iters')
        with gr.Tab('Sauvola (local)'):
            with gr.Row():
                sau_window = gr.Slider(8, 256, value=64, step=8,
                                         label='window (px)')
                sau_k = gr.Slider(0.01, 0.50, value=0.20, step=0.01,
                                    label='k')
                sau_R = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label='R')
                sau_morph = gr.Slider(0, 3, value=1, step=1,
                                        label='morph iters')
        with gr.Tab('Otsu + offset'):
            with gr.Row():
                ots_smooth = gr.Slider(0, 4, value=1.0, step=0.1,
                                         label='σ_smooth (px)')
                ots_offset = gr.Slider(-2.0, 2.0, value=0.0, step=0.05,
                                         label='offset (× std)')
                ots_morph = gr.Slider(0, 3, value=1, step=1,
                                        label='morph iters')

    summary = gr.Markdown('Adjust sliders to see live updates.')
    with gr.Row():
        raw_out = gr.Image(label='Raw ROI', type='numpy', height=380)
        mask_out = gr.Image(label='Binary mask', type='numpy', height=380)
        ov_out = gr.Image(label='Overlay (red = mask)', type='numpy', height=380)

    inputs = [roi_dd, method_dd, polarity_dd,
              adap_smooth, adap_local, adap_k, adap_morph,
              sau_window, sau_k, sau_R, sau_morph,
              ots_smooth, ots_offset, ots_morph]
    outputs = [raw_out, mask_out, ov_out, summary]

    # Live update on any change
    for w in inputs:
        w.change(update, inputs=inputs, outputs=outputs)
    demo.load(update, inputs=inputs, outputs=outputs)

    save_btn.click(save_params, inputs=inputs, outputs=[save_msg])

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, inbrowser=True,
                 show_error=True)
