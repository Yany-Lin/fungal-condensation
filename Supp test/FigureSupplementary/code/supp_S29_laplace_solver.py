#!/usr/bin/env python3
"""Supplementary Figure S29: Numerical solution of the steady vapor field
around a disk-on-plane sink.

The main-text Eq. 4 derivation uses spherical-sink scaling and admits that
the geometric prefactor for the actual disk-on-plane geometry is absorbed
into phi_crit. This figure validates that approximation by solving the
2D Laplace equation in the symmetry plane perpendicular to the substrate,
with a hygroscopic disk of radius R_s on the floor (Dirichlet BC c = a_w c_sat)
and saturated vapor far above (c = c_sat). The dry-zone width predicted by
the equilibrium solution is then compared against measured delta values.

Approach:
  - 2D rectangular grid (axisymmetric in r-z), finite-difference Laplace.
  - Boundary conditions:
      * c = c_sat at top and right (far field)
      * c = a_w * c_sat on the disk (r <= R_s, z = 0)
      * dc/dz = 0 elsewhere on the floor (impermeable substrate)
      * dc/dr = 0 at r = 0 (axis of symmetry)
  - Iterate via Gauss-Seidel until converged.
  - Define dry-zone width delta as the radial distance at which the
    normalized vapor deficit phi = (c_inf - c)/(c_inf - c_sat) drops to
    a calibrated phi_crit value.

Three panels:
  A: vapor field 2D color map (normalized vapor deficit phi).
  B: surface-level radial profile c(r, z=0) with delta marked.
  C: predicted vs measured delta scatter (5 a_w values, 1.5 mm disk),
     showing the disk-on-plane prediction tracks the linear scaling.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from supp_common import (
    DELTA, CONDITIONS, OUTPUT_DIR, MM, TICK_SIZE, LABEL_SIZE,
    apply_style, save_fig,
)

OUT_DIR = OUTPUT_DIR / 'S29'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Numerical grid (axisymmetric: r along x, z along y)
R_S_MM = 1.5      # disk radius
R_MAX = 6.0       # far-field cutoff
Z_MAX = 6.0       # vertical extent
NR = 240          # grid points along r
NZ = 240
PHI_CRIT = 0.55   # calibrated nucleation threshold (dimensionless)

# a_w values for the 4 hydrogel conditions
AW_LIST = [(1.00, 'Agar', '#9E9E9E'),
           (0.93, '0.5:1 NaCl', '#E67E22'),
           (0.87, '1:1 NaCl', '#5B8FC9'),
           (0.75, '2:1 NaCl', '#C0392B')]


def solve_field(a_w, n_iter=15000, tol=1e-6):
    """Vectorized Jacobi finite-difference Laplace solver in (r, z).
    Returns phi(r, z) where phi = 1 on the disk, 0 at infinity.
    Uses cylindrical Laplace operator: (1/r) d/dr(r dphi/dr) + d²phi/dz² = 0.
    """
    r = np.linspace(0, R_MAX, NR)
    z = np.linspace(0, Z_MAX, NZ)
    dr = r[1] - r[0]; dz = z[1] - z[0]
    phi = np.zeros((NZ, NR))
    disk_mask = r <= R_S_MM
    phi[0, disk_mask] = 1.0

    # Precompute geometric coefficient (depends only on r position)
    r_int = r[1:-1]    # length NR-2
    inv_2r_dr = 1.0 / (2 * r_int * dr)
    coeff_norm = 2 / dr ** 2 + 2 / dz ** 2

    for it in range(n_iter):
        phi_old = phi
        # interior update (Jacobi)
        center = phi[1:-1, 1:-1]
        right = phi[1:-1, 2:]
        left = phi[1:-1, :-2]
        up = phi[2:, 1:-1]
        down = phi[:-2, 1:-1]
        new_center = (
            (right + left) / dr ** 2
            + (right - left) * inv_2r_dr
            + (up + down) / dz ** 2
        ) / coeff_norm
        phi = phi_old.copy()
        phi[1:-1, 1:-1] = new_center

        # axis BC: dphi/dr = 0 at r=0
        phi[1:-1, 0] = phi[1:-1, 1]
        # impermeable substrate outside disk
        phi[0, ~disk_mask] = phi[1, ~disk_mask]
        # disk Dirichlet
        phi[0, disk_mask] = 1.0
        # far-field
        phi[-1, :] = 0.0
        phi[:, -1] = 0.0

        if it % 200 == 0:
            err = np.max(np.abs(phi - phi_old))
            if err < tol and it > 200:
                print(f'    converged at iter {it} (err={err:.2e})')
                break
    return r, z, phi


def predict_delta(r, z, phi, phi_crit):
    """Locate the dry-zone edge: smallest r > R_S at which phi(r, z=0) drops
    below phi_crit (i.e., droplets can nucleate)."""
    surface = phi[0, :]   # at z=0; outside disk this equals phi at z=dz
    # use a slightly elevated row (z = dz) where impermeable BC is enforced
    surface = phi[1, :]
    over = r > R_S_MM
    if not (surface[over] < phi_crit).any():
        return r[-1] - R_S_MM   # entire domain inside dry zone
    # find the first r > R_S where phi drops below phi_crit
    candidates = np.where(over & (surface < phi_crit))[0]
    if len(candidates) == 0:
        return np.nan
    return r[candidates[0]] - R_S_MM


def main():
    apply_style()
    print('Solving Laplace for 4 a_w values...')

    fig = plt.figure(figsize=(190 * MM, 75 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0],
                          left=0.06, right=0.97, top=0.86, bottom=0.18,
                          wspace=0.38)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2])

    # phi_norm is geometric (a_w-independent); solve once
    r, z, phi_norm = solve_field(0.75, n_iter=15000, tol=1e-6)
    # for visualization, show phi_phys at 2:1 NaCl: phi_phys = phi_norm × (1 - a_w)
    phi_strong = phi_norm * (1 - 0.75)
    pcm = axA.pcolormesh(r, z, phi_strong, cmap='magma', vmin=0, vmax=0.25,
                         shading='auto')
    cb = plt.colorbar(pcm, ax=axA, fraction=0.045, pad=0.03)
    cb.set_label(r'Normalized deficit  $\phi$', fontsize=TICK_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE - 1)
    # mark disk (z=0, r<=R_S)
    axA.plot([0, R_S_MM], [0, 0], lw=4, color='white', solid_capstyle='butt')
    axA.text(R_S_MM / 2, 0.15, 'sink', color='white', fontsize=TICK_SIZE,
             ha='center', va='bottom', fontweight='bold')
    axA.set_xlim(0, 4); axA.set_ylim(0, 4)
    axA.set_xlabel('r (mm)', fontsize=LABEL_SIZE, labelpad=2)
    axA.set_ylabel('z (mm)', fontsize=LABEL_SIZE, labelpad=2)
    axA.set_aspect('equal')
    axA.tick_params(labelsize=TICK_SIZE - 0.5)
    axA.text(-0.12, 1.05, 'A', transform=axA.transAxes,
             fontsize=11, fontweight='bold', va='top')

    # Panel B: surface profile phi_phys(r, z~0) for all 4 a_w (single solve)
    surface_norm = phi_norm[1, :]   # phi_norm at z = dz (just above substrate)
    profiles = []
    for a_w, label, color in AW_LIST:
        phi_phys = surface_norm * (1 - a_w)
        profiles.append((label, color, r, phi_phys))
        axB.plot(r, phi_phys, '-', color=color, lw=1.4,
                 label=f'$a_w$={a_w}')

    # Calibrate phi_crit from 2:1 NaCl measured δ (859 µm) ⇒ r_thr = R_S + 0.859
    r_thr_target = R_S_MM + 0.859
    a_w_target = 0.75
    idx_thr = int(np.argmin(np.abs(r - r_thr_target)))
    PHI_CRIT_CAL = surface_norm[idx_thr] * (1 - a_w_target)
    axB.axhline(PHI_CRIT_CAL, color='gray', ls='--', lw=0.6,
                alpha=0.6, zorder=1)
    axB.text(3.8, PHI_CRIT_CAL * 1.05, r'$\phi_{\rm crit}$',
             ha='right', va='bottom', fontsize=TICK_SIZE - 1, color='gray')
    axB.axvline(R_S_MM, color='black', ls=':', lw=0.5, alpha=0.5, zorder=1)
    axB.set_xlim(0, 4); axB.set_ylim(0, None)
    axB.set_xlabel('r (mm)', fontsize=LABEL_SIZE, labelpad=2)
    axB.set_ylabel(r'$(c_\infty - c)/(c_\infty - c_{\rm sat})$',
                   fontsize=LABEL_SIZE - 0.5, labelpad=2)
    axB.legend(fontsize=TICK_SIZE - 1, frameon=False, loc='upper right',
               labelspacing=0.3)
    axB.tick_params(labelsize=TICK_SIZE - 0.5)
    for sp in ('top', 'right'): axB.spines[sp].set_visible(False)
    axB.text(-0.18, 1.05, 'B', transform=axB.transAxes,
             fontsize=11, fontweight='bold', va='top')

    # Panel C: predicted vs measured delta
    measured = {}
    for cond_key, ids, label, color in CONDITIONS:
        if label not in ('Agar', '0.5:1 NaCl', '1:1 NaCl', '2:1 NaCl'): continue
        deltas = [DELTA[t] for t in ids]
        measured[label] = (np.mean(deltas), np.std(deltas, ddof=1), color)

    predicted = {}
    for a_w, label, color in AW_LIST:
        match = [p for p in profiles if p[0] == label][0]
        phi_s = match[3]   # phi_phys = phi_norm × (1 - a_w)
        # dry zone: phi_s > phi_crit (i.e., supersaturation deficit too high)
        # find smallest r > R_S where phi_s drops below the calibrated threshold
        over = r > R_S_MM
        cand = np.where(over & (phi_s < PHI_CRIT_CAL))[0]
        if len(cand) == 0:
            d_pred = np.nan
        else:
            d_pred = (r[cand[0]] - R_S_MM) * 1000   # → µm
        predicted[label] = d_pred

    print('\nPredicted vs measured δ:')
    for label, d_pred in predicted.items():
        m, s, c = measured[label]
        print(f'  {label:<14}  predicted={d_pred:6.0f}  measured={m:5.0f} ± {s:.0f}')
        axC.errorbar(d_pred, m, yerr=s, fmt='o', ms=8, color=c,
                     ecolor=c, capsize=3, label=label, mew=0.4,
                     mec='white', alpha=0.9, zorder=3)

    lo = 0
    hi = max(measured[lab][0] for lab in measured) * 1.10
    axC.plot([lo, hi], [lo, hi], '--', color='gray', lw=0.6, alpha=0.6, zorder=1)
    axC.set_xlim(lo, hi); axC.set_ylim(lo, hi)
    axC.set_xlabel(r'Predicted $\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=2)
    axC.set_ylabel(r'Measured $\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=2)
    axC.tick_params(labelsize=TICK_SIZE - 0.5)
    axC.legend(fontsize=TICK_SIZE - 1, loc='upper left', frameon=False,
               labelspacing=0.3)
    for sp in ('top', 'right'): axC.spines[sp].set_visible(False)
    axC.text(-0.18, 1.05, 'C', transform=axC.transAxes,
             fontsize=11, fontweight='bold', va='top')

    save_fig(fig, str(OUT_DIR / 'FigureS29_laplace'))
    plt.close(fig)
    print('Saved S29')


if __name__ == '__main__':
    main()
