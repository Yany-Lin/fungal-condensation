#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR   = Path(__file__).parent
AGG_DIR    = THIS_DIR.parent / 'raw_data' / 'aggregate_edt'
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRIAL_ID = '2to1.4'
DELTA_UM = 900

DIST_ZONES = [
    (800,  1200, '#C0392B', '800-1200 µm'),
    (1200, 1600, '#E67E22', '1200-1600 µm'),
    (1600, 2200, '#27AE60', '1600-2200 µm'),
    (2200, 3000, '#2980B9', '2200-3000 µm'),
]
EVAP_ONSET_MIN = 15.0
T_MARKERS = {'$t_1$': 5.5, '$t_2$': 13.0, '$t_3$': 17.5, '$t_4$': 20.0}

BIN_WIDTH_UM   = 300
MIN_DROPS_BIN  = 5
T_C_MIN        = 5.0
T_C_MAX        = 15.0
R_ANCHOR       = 20.0
T_ANCHOR       = 5.0
R_MAX          = 70

MM         = 1 / 25.4
TICK_SIZE  = 7.0
LABEL_SIZE = 8.0
PANEL_LBL  = 12.0
LW_SPINE   = 0.6


def _style():
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
    })


def plot_panel_B(df, out_stem):
    _style()
    fig, ax = plt.subplots(figsize=(87 * MM, 71 * MM))
    fig.subplots_adjust(left=0.17, right=0.95, top=0.95, bottom=0.20)

    for lo, hi, color, label in DIST_ZONES:
        zone = df[(df['distance_um'] >= lo) & (df['distance_um'] < hi)]
        counts = zone.groupby('time_min').size()
        if counts.max() == 0:
            continue
        norm = counts / counts.max()
        ax.plot(norm.index, norm.values, 'o-', color=color, ms=4.5,
                lw=1.2, label=label, markeredgewidth=0.3, zorder=3)

    for lbl, t_val in T_MARKERS.items():
        ax.text(t_val, -0.08, lbl, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=TICK_SIZE,
                bbox=dict(boxstyle='square,pad=0.15', facecolor='white',
                          edgecolor='black', linewidth=0.5))

    ax.set_xlabel('Time (min)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_ylabel('Normalized Count', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(4.5, 20.5)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(range(4, 22, 2))
    ax.set_xticks(np.arange(4, 21, 1), minor=True)
    ax.tick_params(which='major', labelsize=TICK_SIZE, pad=2)
    ax.tick_params(which='minor', length=2, width=LW_SPINE)

    ax.legend(fontsize=TICK_SIZE - 1, loc='upper right', frameon=True,
              framealpha=0.9, edgecolor='none', labelspacing=0.3,
              handlelength=1.5, handletextpad=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(-0.14, 1.02, 'B', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    for ext in ['.png', '.pdf', '.svg']:
        fig.savefig(f'{out_stem}{ext}', dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print(f'  Panel B saved')


def plot_panel_C(df, out_stem):
    _style()
    fig, ax = plt.subplots(figsize=(90 * MM, 70 * MM))
    fig.subplots_adjust(left=0.14, right=0.82, top=0.88, bottom=0.18)

    data = df[(df['time_min'] >= T_C_MIN) & (df['time_min'] <= T_C_MAX)
              & (df['radius_um'] > 0)].copy()
    data = data[data['distance_um'] >= DELTA_UM]

    r_min_um = DELTA_UM
    max_r = data['distance_um'].max()
    edges = np.arange(r_min_um, max_r + BIN_WIDTH_UM, BIN_WIDTH_UM)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    cmap = plt.cm.plasma
    r_cbar_min = r_min_um / 1000
    r_cbar_max = 3.0

    for i in range(len(edges) - 1):
        in_bin = data[(data['distance_um'] >= edges[i]) &
                      (data['distance_um'] < edges[i + 1])]
        if in_bin.empty:
            continue

        ts = []
        for t, frame in in_bin.groupby('time_min'):
            R = frame['radius_um'].values
            if len(R) >= MIN_DROPS_BIN:
                ts.append((t, np.median(R)))
        if len(ts) < 3:
            continue

        t_arr = np.array([x[0] for x in ts])
        R_arr = np.array([x[1] for x in ts])
        r_mm = bin_centers[i] / 1000
        color = cmap((r_mm - r_cbar_min) / (r_cbar_max - r_cbar_min))
        ax.plot(t_arr, R_arr, '-', color=color, lw=1.0, alpha=0.85)

    t_ref = np.linspace(T_C_MIN, T_C_MAX, 200)
    ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** (1 / 3),
            'k--', lw=1.0, alpha=0.7, label='β=⅓')
    ax.plot(t_ref, R_ANCHOR * (t_ref / T_ANCHOR) ** 1.0,
            color='#C0392B', ls='--', lw=1.0, alpha=0.7, label='β=1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time (min)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_ylabel('Mean R (µm)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(T_C_MIN, T_C_MAX)
    ax.set_ylim(4, 200)
    ax.set_xticks([5, 6, 7, 8, 9, 10, 12, 15])
    ax.set_yticks([5, 10, 20, 50, 100, 200])
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(labelsize=TICK_SIZE, pad=2, which='both')
    ax.tick_params(which='minor', length=2)

    ax.legend(fontsize=TICK_SIZE - 1, loc='upper left', frameon=False,
              handlelength=1.5, handletextpad=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(r_cbar_min, r_cbar_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label(r'$r$ (mm)', fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE - 1)

    ax.text(-0.14, 1.10, 'C', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    for ext in ['.png', '.pdf', '.svg']:
        fig.savefig(f'{out_stem}{ext}', dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print(f'  Panel C saved')


def main():
    print(f'Loading {TRIAL_ID} data...')
    path = AGG_DIR / f'{TRIAL_ID}_edt_droplets.csv'
    df = pd.read_csv(path)
    print(f'  {len(df):,} droplets')

    plot_panel_B(df, str(OUTPUT_DIR / 'panel_B'))
    plot_panel_C(df, str(OUTPUT_DIR / 'panel_C'))
    print('Done.')


if __name__ == '__main__':
    main()
