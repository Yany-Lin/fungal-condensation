#!/usr/bin/env python3
"""Shared utilities for supplementary figure generation."""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
HG_AGG_DIR = BASE_DIR / 'FigureHGAggregate' / 'raw_data' / 'aggregate_edt'
FG_AGG_DIR = BASE_DIR / 'FigureFungi'       / 'raw_data' / 'aggregate_edt'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'

DELTA = {
    'agar.1':   77.8, 'agar.2':   92.4, 'agar.3':  163.5,
    'agar.4':  100.1, 'agar.5':  101.6,
    '0.5to1.2': 337.3, '0.5to1.3': 232.3, '0.5to1.4': 236.9,
    '0.5to1.5': 247.3, '0.5to1.7': 317.9,
    '1to1.1':  420.5, '1to1.2':  286.6, '1to1.3':  420.1,
    '1to1.4':  363.7, '1to1.5':  462.7,
    '2to1.1':  681.1, '2to1.2':  803.2, '2to1.3':  933.4,
    '2to1.4':  872.9, '2to1.5': 1005.4,
    'Green.1': 279.5, 'Green.2': 316.0, 'Green.3': 297.8,
    'Green.4': 285.6, 'Green.5': 311.7,
    'white.1': 198.5, 'white.2': 126.2, 'white.3': 120.0,
    'white.4': 131.6, 'white.5': 123.0,
    'black.1': 125.4, 'black.2':  98.0, 'black.3': 111.7,
    'black.4': 135.7, 'black.5':  78.3,
}

CONDITIONS = [
    ('agar',   ['agar.1','agar.2','agar.3','agar.4','agar.5'],
     'Agar',       '#9E9E9E'),
    ('0.5to1', ['0.5to1.2','0.5to1.3','0.5to1.4','0.5to1.5','0.5to1.7'],
     '0.5:1 NaCl', '#E67E22'),
    ('1to1',   ['1to1.1','1to1.2','1to1.3','1to1.4','1to1.5'],
     '1:1 NaCl',   '#5B8FC9'),
    ('2to1',   ['2to1.1','2to1.2','2to1.3','2to1.4','2to1.5'],
     '2:1 NaCl',   '#C0392B'),
    ('Green', ['Green.1','Green.2','Green.3','Green.4','Green.5'],
     'Aspergillus', '#4CAF50'),
    ('black', ['black.1','black.2','black.3','black.4','black.5'],
     'Rhizopus',    '#212121'),
    ('white', ['white.1','white.2','white.3','white.4','white.5'],
     'Mucor',       '#757575'),
]

ALL_TRIALS = [tid for _, tids, _, _ in CONDITIONS for tid in tids]

MM         = 1 / 25.4
TICK_SIZE  = 7.0
LABEL_SIZE = 8.0
PANEL_LBL  = 10.5
LW_SPINE   = 0.6


def apply_style():
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          TICK_SIZE,
        'axes.linewidth':     LW_SPINE,
        'xtick.major.width':  LW_SPINE,
        'ytick.major.width':  LW_SPINE,
        'xtick.major.size':   3.0,
        'ytick.major.size':   3.0,
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'svg.fonttype':       'none',
    })


def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _agg_dir(tid):
    if tid.startswith(('Green', 'white', 'black')):
        return FG_AGG_DIR
    return HG_AGG_DIR


def load_droplets(tid):
    path = _agg_dir(tid) / f'{tid}_edt_droplets.csv'
    if not path.exists():
        raise FileNotFoundError(f'No droplet CSV: {path}')
    return pd.read_csv(path)


def load_boundary(tid):
    path = _agg_dir(tid) / f'{tid}_boundary_polygon.csv'
    if not path.exists():
        return None
    poly = pd.read_csv(path)
    _, perim = polygon_area_perimeter(poly['x'].values, poly['y'].values)
    return perim


def polygon_area_perimeter(xs, ys):
    n = len(xs)
    area = 0.5 * abs(sum(xs[i]*ys[(i+1)%n] - xs[(i+1)%n]*ys[i] for i in range(n)))
    perim = sum(np.hypot(xs[(i+1)%n]-xs[i], ys[(i+1)%n]-ys[i]) for i in range(n))
    return area, perim


def steiner_bin_area(P_body, delta_um, bin_lo, bin_hi):
    """Area of annular strip via Steiner parallel-body formula."""
    r_lo = delta_um + bin_lo
    r_hi = delta_um + bin_hi
    return P_body * (r_hi - r_lo) + np.pi * (r_hi**2 - r_lo**2)


def save_fig(fig, stem, dpi=300):
    for ext in ['.pdf', '.svg']:
        fig.savefig(f'{stem}{ext}', dpi=dpi, bbox_inches='tight', facecolor='white')
