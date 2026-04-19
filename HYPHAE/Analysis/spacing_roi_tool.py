#!/usr/bin/env python3
"""ROI selector + analyzer for 3D colony-surface images.

Step 1: Calibrate — click two points on ruler, type known distance
Step 2: Select ROI on each image → analyzes + generates overlay

Controls:
  Drag        — draw ROI rectangle
  s           — save ROI, run analysis, generate overlay, next
  d           — mark for deletion, next
  Right/Left  — navigate
  q           — quit
"""

import csv
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, laplace
from scipy import stats

BASE    = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / 'results' / '3d_overlays'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPACING_DIRS = [
    ('Aspergillus', BASE / '3D' / 'Aspergillus 3D'),
    ('Mucor',       BASE / '3D' / 'Mucor 3D'),
]

DISP_MAX = 800
TILE_SIZE = 256  # smaller tiles for smaller ROIs
TILE_STRIDE = 128
FREQ_LO, FREQ_HI = 0.01, 0.45


def load_gray(path, max_side=None):
    with Image.open(path) as im:
        arr = np.asarray(im).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    if max_side:
        s = max(1, max(arr.shape) // max_side)
        arr = arr[::s, ::s]
    return arr


def normalize01(arr):
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo: hi = lo + 1
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def load_roi_fullres(path, y0, y1, x0, x1):
    with Image.open(path) as im:
        crop = im.crop((x0, y0, x1, y1))
        arr = np.asarray(crop).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr  # raw values, not normalized


# ── FFT ──

def fft_on_roi(img):
    """FFT on raw pixel values. Returns (alpha, R2, freqs, power, n_pass, n_total)."""
    h, w = img.shape
    ts = TILE_SIZE
    if h < ts or w < ts:
        return np.nan, np.nan, None, None, 0, 0
    tiles = []
    for y0 in range(0, h - ts + 1, TILE_STRIDE):
        for x0 in range(0, w - ts + 1, TILE_STRIDE):
            t = img[y0:y0 + ts, x0:x0 + ts]
            tiles.append((y0, x0, laplace(t).var(), t.std()))
    if not tiles:
        return np.nan, np.nan, None, None, 0, 0
    laps = np.array([t[2] for t in tiles])
    cutoff = np.percentile(laps, 10)
    passed = [(y, x) for y, x, l, s in tiles if l >= cutoff and s >= 5.0]
    if not passed:
        return np.nan, np.nan, None, None, 0, len(tiles)
    n = ts
    f = (np.arange(n // 2) / n)[1:]
    h_win = np.outer(np.hanning(n), np.hanning(n))
    yi, xi = np.arange(n) - n // 2, np.arange(n) - n // 2
    yy, xx = np.meshgrid(yi, xi, indexing='ij')
    r = np.sqrt(xx ** 2 + yy ** 2).astype(np.int32)
    rm = n // 2
    rf = np.clip(r.ravel(), 0, rm - 1)
    powers = []
    for y0, x0 in passed:
        t = img[y0:y0 + n, x0:x0 + n]
        fft = np.fft.fftshift(np.fft.fft2((t - t.mean()) * h_win))
        p2d = np.abs(fft) ** 2
        rs = np.bincount(rf, weights=p2d.ravel(), minlength=rm)
        rc = np.bincount(rf, minlength=rm).astype(float)
        rc[rc == 0] = 1
        powers.append((rs / rc)[1:])
    mp = np.mean(powers, axis=0)
    v = (f >= FREQ_LO) & (f <= FREQ_HI) & (mp > 0)
    if v.sum() < 5:
        return np.nan, np.nan, f, mp, len(passed), len(tiles)
    sl, ic, r2, p, se = stats.linregress(np.log10(f[v]), np.log10(mp[v]))
    return sl, r2 ** 2, f, mp, len(passed), len(tiles)


# ── Save overlay via subprocess (avoids segfault) ──

OVERLAY_SCRIPT = '''
import sys, json, numpy as np
from PIL import Image
from scipy.ndimage import laplace
from scipy import stats

TILE_SIZE = 256
TILE_STRIDE = 128

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

args = json.loads(sys.argv[1])
path = args['path']
y0, y1, x0, x1 = args['roi']
out = args['out']
fname = args['fname']
genus = args['genus']
full_h, full_w = args['full_size']
alpha = args['alpha']
r2 = args['r2']
lv = args['lap_var']
tiles_pass = args['tiles_pass']
tiles_total = args['tiles_total']

with Image.open(path) as im:
    crop = im.crop((x0, y0, x1, y1))
    arr = np.asarray(crop).astype(np.float32)
if arr.ndim == 3:
    arr = arr[..., :3].mean(axis=2)
roi_h, roi_w = arr.shape

# Display version
lo, hi = np.percentile(arr, [0.5, 99.5])
if hi <= lo: hi = lo + 1
disp = np.clip((arr - lo) / (hi - lo), 0, 1)
ds = max(1, max(disp.shape) // 1500)
disp = disp[::ds, ::ds]

# Recompute FFT power spectrum for plotting
ts = TILE_SIZE
h, w = arr.shape
freqs, power = None, None
if h >= ts and w >= ts:
    tiles = []
    for ty in range(0, h - ts + 1, TILE_STRIDE):
        for tx in range(0, w - ts + 1, TILE_STRIDE):
            t = arr[ty:ty+ts, tx:tx+ts]
            tiles.append((ty, tx, laplace(t).var(), t.std()))
    if tiles:
        laps = np.array([t[2] for t in tiles])
        cutoff = np.percentile(laps, 10)
        passed = [(ty,tx) for ty,tx,l,s in tiles if l >= cutoff and s >= 5.0]
        if passed:
            n = ts
            freqs = (np.arange(n//2)/n)[1:]
            h_win = np.outer(np.hanning(n), np.hanning(n))
            yi, xi = np.arange(n)-n//2, np.arange(n)-n//2
            yy, xx = np.meshgrid(yi, xi, indexing='ij')
            r = np.sqrt(xx**2+yy**2).astype(np.int32)
            rm = n//2
            rf = np.clip(r.ravel(), 0, rm-1)
            pows = []
            for ty,tx in passed:
                t = arr[ty:ty+n, tx:tx+n]
                fft = np.fft.fftshift(np.fft.fft2((t-t.mean())*h_win))
                p2d = np.abs(fft)**2
                rs = np.bincount(rf, weights=p2d.ravel(), minlength=rm)
                rc = np.bincount(rf, minlength=rm).astype(float)
                rc[rc==0] = 1
                pows.append((rs/rc)[1:])
            power = np.mean(pows, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: ROI
axes[0].imshow(disp, cmap='gray')
axes[0].set_title(f'ROI  ({roi_w}x{roi_h} px)', fontsize=10)
axes[0].axis('off')

# Panel 2: Power spectrum + fit
if freqs is not None and power is not None:
    v = (freqs >= 0.01) & (freqs <= 0.45) & (power > 0)
    axes[1].loglog(freqs, power, 'k-', lw=0.8, alpha=0.6)
    if alpha is not None and v.sum() >= 5:
        sl, ic, _, _, _ = stats.linregress(np.log10(freqs[v]), np.log10(power[v]))
        fit = 10**(ic + sl * np.log10(freqs[v]))
        axes[1].loglog(freqs[v], fit, 'r-', lw=2,
                        label=f'alpha = {sl:.3f}')
        axes[1].legend(fontsize=10)
    axes[1].axvspan(0.01, 0.45, alpha=0.08, color='blue')
    axes[1].set_xlabel('Spatial frequency (cyc/px)', fontsize=9)
    axes[1].set_ylabel('Power', fontsize=9)
    axes[1].set_title(f'Radial power spectrum  (tiles: {tiles_pass}/{tiles_total})',
                       fontsize=10)
else:
    axes[1].text(0.5, 0.5, 'ROI too small\\nfor FFT', ha='center', va='center',
                 fontsize=14, transform=axes[1].transAxes)
    axes[1].set_title('Power spectrum', fontsize=10)

# Panel 3: Metrics
axes[2].axis('off')
a_s = f'{alpha:.3f}' if alpha is not None else 'N/A'
r2_s = f'{r2:.3f}' if r2 is not None else 'N/A'
txt = (f'FFT spectral slope (alpha)\\n'
       f'  alpha = {a_s}\\n'
       f'  R2    = {r2_s}\\n'
       f'  tiles = {tiles_pass}/{tiles_total}\\n\\n'
       f'Laplacian variance\\n'
       f'  {lv:.2f}\\n\\n'
       f'ROI: {roi_w} x {roi_h} px\\n'
       f'Frame: {full_w} x {full_h} px')
axes[2].text(0.05, 0.85, txt, fontsize=13, fontfamily='monospace',
             va='top', transform=axes[2].transAxes)
axes[2].set_title('Metrics', fontsize=10)

fig.suptitle(f'{fname}  ({genus})', fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
'''


def save_overlay_subprocess(path, roi, out_path, fname, genus, full_size, metrics):
    args = json.dumps({
        'path': str(path), 'roi': list(roi), 'out': str(out_path),
        'fname': fname, 'genus': genus, 'full_size': list(full_size),
        'alpha': metrics.get('alpha'),
        'r2': metrics.get('r2'),
        'lap_var': metrics['lap_var'],
        'tiles_pass': metrics.get('tiles_pass', 0),
        'tiles_total': metrics.get('tiles_total', 0),
    })
    import sys
    subprocess.run([sys.executable, '-c', OVERLAY_SCRIPT, args],
                   capture_output=True, timeout=120)


# ══════════════════════════════════════════════════
# STEP 1: CALIBRATION
# ══════════════════════════════════════════════════

ASP_RULER = BASE / 'EXTRA' / 'Spacing 2' / '20251214_201341.JPG'
MUC_RULER = BASE / '3D' / 'Mucor 3D' / 'Ruler.JPG'


def calibrate():
    # Show available rulers
    rulers = []
    if ASP_RULER.exists():
        rulers.append(('Aspergillus (200µm div, 4mm grid)', ASP_RULER))
    if MUC_RULER.exists():
        rulers.append(('Mucor', MUC_RULER))

    if not rulers:
        print('No ruler images found.')
        return None

    print('\nCalibration rulers:')
    for i, (desc, p) in enumerate(rulers):
        print(f'  [{i}] {desc}  —  {p.name}')
    print(f'  [s] Skip')

    choice = input('\nPick ruler: ').strip().lower()
    if choice == 's':
        return None
    try:
        ruler_path = rulers[int(choice)][1]
    except (ValueError, IndexError):
        print('Invalid. Skipping.')
        return None

    print(f'Using: {ruler_path.name}')

    with Image.open(ruler_path) as im:
        full_w, full_h = im.size
    raw = load_gray(ruler_path, max_side=1200)
    img = normalize01(raw)
    scale = full_h / img.shape[0]

    points = []
    result = {'um_per_px': None}

    fig, ax = plt.subplots(figsize=(14, 8), facecolor='black')
    ax.imshow(img, cmap='gray')
    ax.set_title('Click two points on a known distance',
                 fontsize=12, color='white', fontweight='bold')
    ax.axis('off')
    fig.set_facecolor('black')

    info = fig.text(0.5, 0.02, 'Click point 1...', ha='center',
                    fontsize=10, color='#aaa')

    ax_box = fig.add_axes([0.3, 0.06, 0.25, 0.04])
    ax_box.set_visible(False)
    tbox = TextBox(ax_box, 'Known µm: ', initial='')

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'r+', markersize=20, mew=3)
        if len(points) == 1:
            info.set_text('Click point 2...')
        elif len(points) == 2:
            p1, p2 = points[0], points[1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=2)
            dx = (p2[0] - p1[0]) * scale
            dy = (p2[1] - p1[1]) * scale
            result['dist_px'] = np.sqrt(dx ** 2 + dy ** 2)
            info.set_text(f'{result["dist_px"]:.0f} px — type distance in µm below, press Enter')
            ax_box.set_visible(True)
        fig.canvas.draw_idle()

    def on_submit(text):
        try:
            known_um = float(text)
            result['um_per_px'] = known_um / result['dist_px']
            info.set_text(f'{result["um_per_px"]:.4f} µm/px — close window to continue')
            fig.canvas.draw_idle()
        except ValueError:
            info.set_text('Invalid number, try again')
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    tbox.on_submit(on_submit)
    plt.show()

    if result.get('um_per_px'):
        print(f'Calibration: {result["um_per_px"]:.4f} µm/px')
    return result


# ══════════════════════════════════════════════════
# STEP 2: ROI SELECTION
# ══════════════════════════════════════════════════

class ROISelector:
    def __init__(self, calibration):
        self.calibration = calibration or {}
        self.images = []
        for genus, folder in SPACING_DIRS:
            for p in sorted(folder.iterdir()):
                if p.suffix.upper() == '.JPG' and not p.name.startswith('.') \
                        and p.stem.lower() != 'ruler':
                    self.images.append((p, genus))

        self.idx = 0
        self.roi = None

        self.session_path = OUT_DIR / 'roi_session.json'
        self.results = {}
        if self.session_path.exists():
            with open(self.session_path) as f:
                self.results = json.load(f)

        if self.calibration.get('um_per_px'):
            self.results['_calibration'] = self.calibration
            self._save_session()

        for i, (p, g) in enumerate(self.images):
            if p.name not in self.results:
                self.idx = i
                break

        self.fig, self.ax = plt.subplots(figsize=(13, 8), facecolor='black')
        self.fig.subplots_adjust(bottom=0.08, top=0.93)
        self.title = self.fig.text(0.5, 0.97, '', ha='center', fontsize=11,
                                   color='white', fontweight='bold')
        self.info = self.fig.text(0.5, 0.01, '', ha='center', fontsize=9,
                                  color='#aaa', fontfamily='monospace')

        self._drag_start = None
        self._rect = None
        self.fig.canvas.mpl_connect('button_press_event', self._mouse_down)
        self.fig.canvas.mpl_connect('motion_notify_event', self._mouse_move)
        self.fig.canvas.mpl_connect('button_release_event', self._mouse_up)
        self.fig.canvas.mpl_connect('key_press_event', self._key)

        self._load()

        n_done = sum(1 for p, g in self.images if p.name in self.results)
        print(f'{len(self.images)} images ({n_done} done)')
        print('Drag ROI → s=save+analyze  |  d=delete  |  arrows=nav  |  q=quit')
        plt.show()
        self._finish()

    def _save_session(self):
        with open(self.session_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def _load(self):
        path, genus = self.images[self.idx]
        self._path = path
        self._genus = genus

        raw = load_gray(path, max_side=DISP_MAX)
        self._disp = normalize01(raw)
        with Image.open(path) as im:
            self._full_w, self._full_h = im.size
        self._scale_x = self._full_w / self._disp.shape[1]
        self._scale_y = self._full_h / self._disp.shape[0]

        self.roi = None
        self._drag_start = None
        self.ax.clear()
        self.ax.imshow(self._disp, cmap='gray')
        self.ax.axis('off')
        self._rect = plt.Rectangle((0, 0), 0, 0, fill=False,
                                    edgecolor='red', linewidth=2.5)
        self._rect.set_visible(False)
        self.ax.add_patch(self._rect)

        entry = self.results.get(path.name, {})
        if entry.get('status') == 'deleted':
            tag = ' [DELETED]'
        elif path.name in self.results:
            tag = ' [SAVED]'
        else:
            tag = ''
        self.title.set_text(
            f'{path.name}  ({genus})  [{self.idx + 1}/{len(self.images)}]{tag}')
        self.info.set_text(
            'Drag ROI → s=save+analyze  |  d=delete  |  arrows=nav  |  q=quit')
        self.fig.canvas.draw_idle()

    def _mouse_down(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        self._drag_start = (event.xdata, event.ydata)

    def _mouse_move(self, event):
        if self._drag_start is None or event.inaxes != self.ax:
            return
        sx, sy = self._drag_start
        x0, x1 = sorted([sx, event.xdata])
        y0, y1 = sorted([sy, event.ydata])
        self._rect.set_xy((x0, y0))
        self._rect.set_width(x1 - x0)
        self._rect.set_height(y1 - y0)
        self._rect.set_visible(True)
        self.fig.canvas.draw_idle()

    def _mouse_up(self, event):
        if self._drag_start is None or event.inaxes != self.ax or event.button != 1:
            return
        sx, sy = self._drag_start
        self._drag_start = None
        dx0, dx1 = sorted([sx, event.xdata])
        dy0, dy1 = sorted([sy, event.ydata])
        x0 = max(0, min(int(dx0 * self._scale_x), self._full_w))
        x1 = max(0, min(int(dx1 * self._scale_x), self._full_w))
        y0 = max(0, min(int(dy0 * self._scale_y), self._full_h))
        y1 = max(0, min(int(dy1 * self._scale_y), self._full_h))
        self.roi = (y0, y1, x0, x1)
        self.info.set_text(f'ROI: {x1 - x0}x{y1 - y0} px  |  s=save+analyze')
        self.fig.canvas.draw_idle()

    def _save(self):
        if self.roi is None:
            self.info.set_text('No ROI! Drag a rectangle first.')
            self.fig.canvas.draw_idle()
            return
        y0, y1, x0, x1 = self.roi
        if (y1 - y0) < 50 or (x1 - x0) < 50:
            self.info.set_text('ROI too small')
            self.fig.canvas.draw_idle()
            return

        self.info.set_text('Analyzing...')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Load ROI at full res (raw pixel values)
        roi_raw = load_roi_fullres(self._path, y0, y1, x0, x1)
        roi_h, roi_w = roi_raw.shape

        # 1. FFT spectral slope (on raw values)
        alpha, r2, _, _, n_pass, n_total = fft_on_roi(roi_raw)

        # 2. Laplacian variance (on raw values)
        lap_ds = roi_raw[::4, ::4] if max(roi_h, roi_w) > 2000 else roi_raw
        lv = float(laplace(lap_ds).var())

        metrics = {
            'alpha': round(float(alpha), 4) if np.isfinite(alpha) else None,
            'r2': round(float(r2), 4) if np.isfinite(r2) else None,
            'lap_var': round(lv, 2),
            'tiles_pass': n_pass,
            'tiles_total': n_total,
        }

        # Save ROI crop
        genus_dir = OUT_DIR / self._genus
        genus_dir.mkdir(exist_ok=True)
        with Image.open(self._path) as im:
            crop = im.crop((x0, y0, x1, y1))
            crop.save(genus_dir / f'{self._path.stem}_roi.jpg', quality=95)

        # Save overlay in subprocess (prevents segfault)
        overlay_path = genus_dir / f'{self._path.stem}_overlay.png'
        save_overlay_subprocess(
            self._path, self.roi, overlay_path,
            self._path.name, self._genus,
            (self._full_h, self._full_w), metrics)

        # Session
        self.results[self._path.name] = {
            'genus': self._genus,
            'roi_px': [y0, y1, x0, x1],
            'roi_size_px': [roi_h, roi_w],
            'full_size_px': [self._full_h, self._full_w],
            **metrics,
        }
        self._save_session()

        a_str = f'{alpha:.3f}' if np.isfinite(alpha) else 'N/A'
        print(f'  {self._path.name}: {roi_w}x{roi_h}, '
              f'α={a_str}, lap={lv:.1f}, tiles={n_pass}/{n_total}')
        self._advance()

    def _mark_delete(self):
        self.results[self._path.name] = {
            'genus': self._genus,
            'status': 'deleted',
        }
        self._save_session()
        for ext in ('_overlay.png', '_roi.jpg'):
            f = OUT_DIR / self._genus / f'{self._path.stem}{ext}'
            if f.exists():
                f.unlink()
        print(f'  {self._path.name}: DELETED')
        self._advance()

    def _advance(self):
        if self.idx < len(self.images) - 1:
            self.idx += 1
            self._load()
        else:
            print('All done.')
            plt.close(self.fig)

    def _key(self, e):
        if e.key == 's':
            self._save()
        elif e.key == 'd':
            self._mark_delete()
        elif e.key == 'right':
            self.idx = (self.idx + 1) % len(self.images)
            self._load()
        elif e.key == 'left':
            self.idx = (self.idx - 1) % len(self.images)
            self._load()
        elif e.key in ('q', 'escape'):
            plt.close(self.fig)

    def _finish(self):
        saved = {f: d for f, d in self.results.items()
                 if not f.startswith('_') and d.get('status') != 'deleted'}
        deleted = {f: d for f, d in self.results.items()
                   if d.get('status') == 'deleted'}
        if not saved:
            return

        um = self.results.get('_calibration', {}).get('um_per_px')
        print(f'\n{"=" * 60}')
        print(f'{len(saved)} analyzed, {len(deleted)} deleted')
        if um:
            print(f'Calibration: {um:.4f} µm/px')
        print(f'{"=" * 60}')

        for f, d in sorted(saved.items()):
            rh, rw = d['roi_size_px']
            a = d.get('alpha')
            a_str = f'{a:.3f}' if a is not None else 'N/A'
            size = f'{rw}x{rh}'
            if um:
                size += f' ({rw * um:.0f}x{rh * um:.0f} µm)'
            print(f'  {f}: {size}, α={a_str}, lap={d["lap_var"]:.1f}')

        asp = [d for d in saved.values() if d['genus'] == 'Aspergillus']
        muc = [d for d in saved.values() if d['genus'] == 'Mucor']
        if len(asp) >= 2 and len(muc) >= 2:
            print(f'\n--- Group comparison ---')
            for key, label in [('alpha', 'FFT α'),
                               ('lap_var', 'Laplacian')]:
                av = [d[key] for d in asp if d.get(key) is not None]
                mv = [d[key] for d in muc if d.get(key) is not None]
                if len(av) >= 2 and len(mv) >= 2:
                    sa, sm = np.array(av), np.array(mv)
                    t, p = stats.ttest_ind(sa, sm, equal_var=False)
                    print(f'  {label}: Asp {sa.mean():.4f}±{sa.std(ddof=1):.4f} vs '
                          f'Muc {sm.mean():.4f}±{sm.std(ddof=1):.4f}, p={p:.4f}')

        csv_path = OUT_DIR / '3d_roi_results.csv'
        fields = ['file', 'genus', 'roi_size', 'alpha', 'r2', 'lap_var',
                  'tiles_pass', 'tiles_total']
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for f, d in sorted(saved.items()):
                w.writerow({
                    'file': f, 'genus': d['genus'],
                    'roi_size': f'{d["roi_size_px"][1]}x{d["roi_size_px"][0]}',
                    'alpha': d.get('alpha', ''), 'r2': d.get('r2', ''),
                    'lap_var': d['lap_var'],
                    'tiles_pass': d.get('tiles_pass', ''),
                    'tiles_total': d.get('tiles_total', ''),
                })
        print(f'\nCSV: {csv_path}')
        print(f'Overlays: {OUT_DIR}/')


if __name__ == '__main__':
    print('Step 1: Calibration')
    cal = calibrate()
    print('\nStep 2: ROI selection + analysis')
    ROISelector(cal)
