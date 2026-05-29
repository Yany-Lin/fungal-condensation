#!/usr/bin/env python3
"""Master reproducibility script for the f_tissue investigation.

Runs every analysis in dependency order, prints per-stage timing, and verifies
that all expected outputs land on disk. Idempotent — re-running overwrites prior
outputs deterministically (random seeds fixed).

Target end-to-end runtime on RTX 5070 Ti: ~60 s.

Usage:
    python code/run_all_ftissue.py
"""

import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT_CSV = BASE / 'output'
OUT_FIG = BASE / 'figures'
PREV = OUT_FIG / 'preview_fullres'

PY = sys.executable  # whichever venv invoked us

STAGES = [
    # (script, expected output(s), description)
    ('reproduce_all_final.py', [
        OUT_CSV / 'ftissue_all_methods_final.csv',
        OUT_FIG / 'figS_ftissue_convergence.pdf',
    ], '4 binary methods + earlier convergence draft'),
    ('continuous_bright_density.py', [
        OUT_CSV / 'ftissue_continuous_density.csv',
        OUT_FIG / 'figS_continuous_density.pdf',
        PREV / 'density_asp_heatmap.png',
        PREV / 'density_muc_heatmap.png',
    ], 'Continuous brightness-only metric (sensitivity ablation)'),
    ('photometric_density_reconstruction.py', [
        OUT_CSV / 'ftissue_photometric.csv',
        OUT_FIG / 'figS_photometric_reconstruction.pdf',
        PREV / 'reconstruction_asp_heatmap.png',
        PREV / 'reconstruction_muc_heatmap.png',
        PREV / 'reconstruction_asp_3d.png',
        PREV / 'reconstruction_muc_3d.png',
    ], 'PRIMARY: photometric reconstruction (brightness + gradient + structure tensor)'),
    ('ftissue_bright_final.py', [
        OUT_CSV / 'ftissue_bright_final.csv',
        OUT_FIG / 'figS_ftissue_bright.pdf',
    ], 'Bright vs dark adaptive paired comparison'),
    ('fullres_previews.py', [
        PREV / 'asp_canonical_raw_vs_mask.png',
        PREV / 'muc_canonical_raw_vs_mask.png',
        PREV / 'all24',
    ], 'Native-resolution raw + mask previews for all 24 ROIs'),
    ('final_convergence_figure.py', [
        OUT_FIG / 'figS_ftissue_final_convergence.pdf',
    ], '6-method convergence supplement (reads all CSVs above)'),
]

OPTIONAL_STAGES = [
    ('sauvola_supplement_and_validation.py', [
        OUT_CSV / 'ftissue_sauvola_final.csv',
        OUT_FIG / 'figS_ftissue_sauvola_supplement.pdf',
        OUT_FIG / 'figS_ftissue_sauvola_validation.pdf',
    ], 'Sauvola parameter sweep (6×5 = 30 configs × 24 ROIs; ~14 s)'),
    ('compare_segmentation_methods.py', [
        OUT_FIG / 'figS_ftissue_segcompare.pdf',
    ], '4-method binary comparison (current + Otsu + Frangi + Sauvola)'),
]


def run(script, expected_outputs, desc):
    path = HERE / script
    print(f'\n{"─" * 72}')
    print(f'[{script}]  {desc}')
    print(f'{"─" * 72}')
    if not path.exists():
        print(f'  SKIP — script not found at {path}')
        return None
    t0 = time.time()
    result = subprocess.run([PY, str(path)],
                              capture_output=True, text=True,
                              env={**__import__('os').environ,
                                   'PYTHONIOENCODING': 'utf-8'})
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'  FAILED (exit {result.returncode}) in {elapsed:.1f} s')
        print('  stderr (tail):')
        for line in result.stderr.splitlines()[-15:]:
            print(f'    {line}')
        return False
    print(f'  OK in {elapsed:.1f} s')
    # Verify outputs
    missing = []
    for out in expected_outputs:
        if not out.exists():
            missing.append(out)
    if missing:
        print(f'  WARN — expected outputs not found:')
        for m in missing:
            print(f'    {m}')
    return True


def main():
    print('═' * 72)
    print('f_tissue investigation — master reproducibility run')
    print('═' * 72)
    print(f'Base folder:  {BASE}')
    print(f'Python:       {PY}')
    print(f'Code dir:     {HERE}')
    print(f'Output dir:   {OUT_CSV}')
    print(f'Figure dir:   {OUT_FIG}')

    t_start = time.time()
    results = {}
    for stage in STAGES:
        ok = run(*stage)
        results[stage[0]] = ok

    print(f'\n{"═" * 72}')
    print('Required stages summary')
    print(f'{"═" * 72}')
    for s, ok in results.items():
        status = 'OK' if ok else ('SKIPPED' if ok is None else 'FAILED')
        print(f'  {status:>8}  {s}')

    # Optional / slower stages
    print(f'\n{"─" * 72}')
    print('Optional slower stages (Sauvola sweep + side-by-side comparison).')
    print(f'{"─" * 72}')
    answer = input('Run optional stages? [y/N]: ').strip().lower()
    if answer == 'y':
        for stage in OPTIONAL_STAGES:
            run(*stage)

    elapsed = time.time() - t_start
    print(f'\n{"═" * 72}')
    print(f'Total elapsed:  {elapsed:.1f} s')
    print(f'{"═" * 72}')

    # Final headline numbers
    csv_paths = {
        'photometric (primary)': OUT_CSV / 'ftissue_photometric.csv',
        'continuous (ablation)': OUT_CSV / 'ftissue_continuous_density.csv',
        'binary (4 methods)':    OUT_CSV / 'ftissue_all_methods_final.csv',
    }
    print('\nFinal CSV outputs:')
    for label, p in csv_paths.items():
        if p.exists():
            print(f'  {p}  ({p.stat().st_size:,} bytes)  — {label}')

    print('\nKey figures:')
    for p in [OUT_FIG / 'figS_photometric_reconstruction.pdf',
                OUT_FIG / 'figS_continuous_density.pdf',
                OUT_FIG / 'figS_ftissue_final_convergence.pdf']:
        if p.exists():
            print(f'  {p}  ({p.stat().st_size:,} bytes)')

    print('\nNext steps:')
    print('  - Read FTISSUE_METHODOLOGY_REPORT.md for the full write-up')
    print('  - View figures/figS_ftissue_final_convergence.png for the SI summary')
    print('  - View figures/preview_fullres/reconstruction_asp_3d.png for 3D topography')
    print('\nDone.')


if __name__ == '__main__':
    main()
