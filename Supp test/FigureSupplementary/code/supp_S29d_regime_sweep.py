#!/usr/bin/env python3
"""Supplementary Figure S29d: R_s/δ regime transition.

Sweep absorber radius R_s at fixed a_w and trace the dry-zone width δ.
Two regimes are expected:
  point-sink limit (R_s/δ ≪ 1): φ ~ 1/r far field, δ ∝ R_s.
  boundary-layer limit (R_s/δ ≫ 1): δ set by thermodynamic BC,
  φ collapses near a sharp depletion edge.

Discussion claims trials operate at R_s/δ ≈ 1.7-14, spanning the
transition. This figure demonstrates that transition by direct PDE solve.

Panels:
  A — φ(r, z=z_probe) profiles for representative R_s values, x-axis
      normalized by R_s. Point-sink and boundary-layer limits visible.
  B — δ/R_s vs R_s/δ on log-log axes, with the 35 measured trials
      overlaid as scatter.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE, DELTA, CONDITIONS,
    apply_style, save_fig,
)

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


def _build_axisymmetric_grid(Rmax, H, dr, dz):
    nr = int(np.ceil(Rmax / dr))
    nz = int(np.ceil(H / dz))
    r = (np.arange(nr) + 0.5) * dr
    z = (np.arange(nz) + 0.5) * dz
    return r, z


def solve_raised_cylinder_dirichlet_fd(*, Rmax, H, Rf, h_cyl, dr, dz, D,
                                       c_eq, c_inf,
                                       outer_bc='clamp', bottom_bc='noflux'):
    """Sparse direct solve of axisymmetric Laplace with raised cylinder Dirichlet sink.

    Verbatim port of vapor_sink_raised_cylinder.solve_raised_cylinder_dirichlet_fd
    (see /Volumes/T7/Downloads/vapor_sink_workspace/), inlined here to avoid the
    workspace module's fipy dependency.
    """
    r, z = _build_axisymmetric_grid(Rmax, H, dr, dz)
    nr, nz = len(r), len(z)
    c = np.full((nz, nr), float(c_inf), dtype=float)
    source = (r[None, :] <= Rf) & (z[:, None] <= h_cyl)
    if not np.any(source):
        raise ValueError('No source cells. Increase h_cyl or refine mesh.')
    c[source] = float(c_eq)

    fixed = source.copy()
    fixed[-1, :] = True; c[-1, :] = float(c_inf)
    if outer_bc.lower() == 'clamp':
        fixed[:, -1] = True; c[:, -1] = float(c_inf)
    bottom_mode = bottom_bc.lower()

    unknown = ~fixed
    n_unk = int(np.count_nonzero(unknown))
    if n_unk == 0:
        return r, z, c, source
    unk_idx = -np.ones((nz, nr), dtype=int)
    unk_idx[unknown] = np.arange(n_unk)

    A = lil_matrix((n_unk, n_unk), dtype=float)
    b = np.zeros(n_unk, dtype=float)
    inv_dz2 = 1.0 / (dz * dz)
    outer_mode = outer_bc.lower()

    for i in range(nz):
        for j in range(nr):
            if not unknown[i, j]:
                continue
            row = unk_idx[i, j]
            rj = r[j]
            r_plus = rj + 0.5 * dr
            r_minus = max(rj - 0.5 * dr, 0.0)
            aE = r_plus / (rj * dr * dr)
            aW = r_minus / (rj * dr * dr)
            aN = inv_dz2; aS = inv_dz2
            if i == 0 and bottom_mode == 'noflux':
                aS = 0.0
            if outer_mode == 'noflux' and j == nr - 1:
                aE = 0.0
            aP = 0.0
            if aE > 0 and j + 1 < nr:
                aP += aE
                if unknown[i, j + 1]: A[row, unk_idx[i, j + 1]] = -aE
                else: b[row] += aE * c[i, j + 1]
            if aW > 0 and j - 1 >= 0:
                aP += aW
                if unknown[i, j - 1]: A[row, unk_idx[i, j - 1]] = -aW
                else: b[row] += aW * c[i, j - 1]
            if aN > 0 and i + 1 < nz:
                aP += aN
                if unknown[i + 1, j]: A[row, unk_idx[i + 1, j]] = -aN
                else: b[row] += aN * c[i + 1, j]
            if aS > 0 and i - 1 >= 0:
                aP += aS
                if unknown[i - 1, j]: A[row, unk_idx[i - 1, j]] = -aS
                else: b[row] += aS * c[i - 1, j]
            A[row, row] = aP

    c_unk = spsolve(csr_matrix(A), b)
    c[unknown] = c_unk
    return r, z, c, source

OUT_DIR = OUTPUT_DIR / 'S29d'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / 'regime_sweep_cache.csv'
PROFILE_CACHE = OUT_DIR / 'regime_profiles_cache.npz'

# Fixed physics (validated workspace values)
A_W = 0.97            # absorber Dirichlet value (matches 1:1 NaCl validated)
C_INF = 1.03          # far-field
S_NUC = 0.02          # nucleation threshold
D = 2.5e-5            # m^2/s
H_CYL = 2.0e-3        # raised cylinder height
Z_PROBE = 25e-6

# Sweep
R_S_LIST_MM = [0.1, 0.2, 0.3, 0.5, 0.8, 1.5, 2.5, 4.0, 6.0]


def solve_one(Rf_m: float):
    """Run one Dirichlet PDE solve. Returns (r, c_at_zprobe, delta_m)."""
    # Domain must comfortably contain depletion zone. Scale with Rf.
    Rmax = max(8e-3, 4 * Rf_m + 4e-3)
    H = max(3e-3, H_CYL + 1.5e-3)
    # Coarser mesh for big domains to keep solve time reasonable.
    if Rf_m <= 0.5e-3:
        dr = 10e-6; dz = 10e-6
    elif Rf_m <= 2.0e-3:
        dr = 20e-6; dz = 20e-6
    else:
        dr = 30e-6; dz = 30e-6

    r, z, c, source = solve_raised_cylinder_dirichlet_fd(
        Rmax=Rmax, H=H, Rf=Rf_m, h_cyl=H_CYL,
        dr=dr, dz=dz, D=D, c_eq=A_W, c_inf=C_INF,
        outer_bc='clamp', bottom_bc='noflux',
    )
    # extract c(r) at z_probe above the surface (z = h_cyl + z_probe)
    z_target = H_CYL + Z_PROBE
    iz = int(np.argmin(np.abs(z - z_target)))
    c_line = c[iz, :]
    S_line = c_line - 1.0  # supersaturation
    # delta: distance from edge of cylinder where S first crosses S_nuc
    mask = r >= Rf_m
    rr = r[mask]; SS = S_line[mask]
    ge = np.where(SS >= S_NUC)[0]
    if ge.size == 0:
        delta = float('nan')
    else:
        i_hi = int(ge[0])
        if i_hi == 0:
            delta = float(rr[0] - Rf_m)
        else:
            i_lo = i_hi - 1
            S_lo, S_hi = SS[i_lo], SS[i_hi]
            r_lo, r_hi = rr[i_lo], rr[i_hi]
            frac = (S_NUC - S_lo) / (S_hi - S_lo) if S_hi != S_lo else 0.0
            delta = float(r_lo + frac * (r_hi - r_lo) - Rf_m)
    return r, c_line, S_line, delta


def run_sweep():
    rows = []
    profiles = {}  # Rf_mm -> dict(r, S)
    for Rf_mm in R_S_LIST_MM:
        Rf_m = Rf_mm * 1e-3
        print(f'  solving R_s = {Rf_mm:.2f} mm ...', flush=True)
        r, c_line, S_line, delta = solve_one(Rf_m)
        rows.append(dict(Rf_mm=Rf_mm, Rf_m=Rf_m, delta_m=delta,
                         R_over_delta=Rf_m / delta if delta > 0 else np.nan,
                         delta_over_R=delta / Rf_m if delta > 0 else np.nan))
        profiles[f'r_{Rf_mm}'] = r
        profiles[f'S_{Rf_mm}'] = S_line
        print(f'    delta = {delta*1e6:.1f} um, R_s/delta = {Rf_m/delta if delta>0 else float("nan"):.2f}')
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    np.savez(PROFILE_CACHE, **profiles)
    return df, profiles


def load_or_run():
    if CACHE.exists() and PROFILE_CACHE.exists():
        print('  loading cached sweep')
        df = pd.read_csv(CACHE)
        profiles = dict(np.load(PROFILE_CACHE))
        return df, profiles
    return run_sweep()


def main():
    apply_style()
    print('PDE regime sweep:')
    df, profiles = load_or_run()

    fig = plt.figure(figsize=(180 * MM, 80 * MM))
    gs = fig.add_gridspec(1, 2, left=0.09, right=0.97, top=0.91, bottom=0.18,
                          wspace=0.32)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # ── Panel A: φ profiles, x normalized by R_s ──
    repr_list = [0.1, 0.5, 1.5, 4.0]
    cmap = plt.cm.viridis
    for k, Rf_mm in enumerate(repr_list):
        r = profiles[f'r_{Rf_mm}']
        S = profiles[f'S_{Rf_mm}']
        Rf_m = Rf_mm * 1e-3
        # phi = (c_inf - c) / (c_inf - a_w), normalized depletion 1->0
        phi = (C_INF - (S + 1.0)) / (C_INF - A_W)
        # plot vs r/R_s
        x = r / Rf_m
        c_color = cmap(k / max(1, len(repr_list) - 1))
        axA.plot(x, phi, '-', color=c_color, lw=1.4,
                 label=f'$R_s$ = {Rf_mm:.1f} mm')

    # mark cylinder edge at r/R_s = 1
    axA.axvline(1.0, color='black', lw=0.5, ls=':', alpha=0.5)
    axA.text(1.02, 0.02, 'edge', fontsize=TICK_SIZE - 1.5,
             ha='left', va='bottom', color='black', alpha=0.6,
             transform=axA.get_xaxis_transform())
    axA.set_xlim(0, 6)
    axA.set_ylim(-0.05, 1.05)
    axA.set_xlabel(r'$r / R_s$', fontsize=LABEL_SIZE, labelpad=2)
    axA.set_ylabel(r'$\varphi = (c_\infty - c)/(c_\infty - c_{\rm eq})$',
                   fontsize=LABEL_SIZE, labelpad=2)
    axA.tick_params(labelsize=TICK_SIZE - 0.5)
    axA.legend(fontsize=TICK_SIZE - 1, loc='upper right', frameon=False,
               handletextpad=0.4, labelspacing=0.25)
    for sp in ('top', 'right'): axA.spines[sp].set_visible(False)
    axA.text(-0.18, 1.05, 'A', transform=axA.transAxes,
             fontsize=11, fontweight='bold', va='top')
    axA.set_title(f'Profiles at fixed $a_w = {A_W}$',
                  fontsize=LABEL_SIZE - 0.5, pad=4)

    # ── Panel B: δ/R_s vs R_s/δ ──
    ok = df.dropna(subset=['delta_m'])
    axB.plot(ok['R_over_delta'], ok['delta_over_R'], 'o-', color='#222',
             ms=5, mfc='#5B8FC9', mec='white', mew=0.5, lw=1.0, zorder=4,
             label='PDE sweep')

    # Overlay 35 measured trials
    rs_data = []  # (R_s/delta, delta/R_s, color)
    R_F_TRIAL = 1.5e-3  # measured trials all have R_s ≈ 1.5 mm
    color_map = {tid: c for _, tids, _, c in CONDITIONS for tid in tids}
    trial_pts = []
    for tid, dlt_um in DELTA.items():
        if not np.isfinite(dlt_um) or dlt_um <= 0: continue
        R_over_d = R_F_TRIAL / (dlt_um * 1e-6)
        d_over_R = (dlt_um * 1e-6) / R_F_TRIAL
        c_pt = color_map.get(tid, '#999')
        trial_pts.append((R_over_d, d_over_R, c_pt))
    if trial_pts:
        xs, ys, cs = zip(*trial_pts)
        axB.scatter(xs, ys, s=22, c=cs, edgecolors='white', linewidths=0.4,
                    alpha=0.9, zorder=3, label='35 trials (measured)')

    # asymptote annotations
    x_grid = np.array([min(ok['R_over_delta'].min(), 0.05),
                       max(ok['R_over_delta'].max(), 30.0)])
    # point-sink: δ ~ R_s (constant ratio)
    # This figure shows the actual scaling; annotate after fit.

    axB.set_xscale('log')
    axB.set_yscale('log')
    axB.set_xlabel(r'$R_s / \delta$', fontsize=LABEL_SIZE, labelpad=2)
    axB.set_ylabel(r'$\delta / R_s$', fontsize=LABEL_SIZE, labelpad=2)
    axB.tick_params(labelsize=TICK_SIZE - 0.5)
    axB.legend(fontsize=TICK_SIZE - 1, loc='upper right', frameon=False,
               handletextpad=0.4)
    for sp in ('top', 'right'): axB.spines[sp].set_visible(False)
    axB.text(-0.18, 1.05, 'B', transform=axB.transAxes,
             fontsize=11, fontweight='bold', va='top')
    axB.set_title(r'Regime transition: $\delta/R_s$ vs $R_s/\delta$',
                  fontsize=LABEL_SIZE - 0.5, pad=4)

    save_fig(fig, str(OUT_DIR / 'FigureS29d_regime_sweep'))
    plt.close(fig)
    print('Saved S29d')

    print('\nSweep summary:')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
