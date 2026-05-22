"""Convergence / divergence diagnostics for the RHEA finite-volume solver.

Reads the output_data_<iter>.csv snapshots and reports, per snapshot:
  - a global residual  ||q_n - q_{n-1}|| / (||q_{n-1}|| * dt_iters)  for rho, rho*u, rho*E
    (rate of change of the conserved state; -> 0 means converged, growth means diverging)
  - physical-sanity extrema: min/max of T, P, rho, sos and the minimum internal
    energy  e = E - 0.5|V|^2  (the quantity fed to the (rho,e)->(P,T) inversion;
    if it goes <=0 the temperature inversion has no physical root -> blow-up)
  - the (i,j) location and x of the worst cell, so divergence can be localized.

Usage:  pixi run python convergence_check.py [glob]
        default glob = output_data_*.csv  (interior cells only; ghosts dropped)
"""
import glob, os, re, sys
import numpy as np
import pandas as pd

pat = sys.argv[1] if len(sys.argv) > 1 else 'output_data_*.csv'
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1   # print every Nth snapshot
T_FLOOR = 150.0   # K: flag cells colder than this (near/below CO2 two-phase region)
files = sorted(glob.glob(pat), key=lambda p: int(re.search(r'_(\d+)\.csv$', p).group(1)))
if not files:
    raise SystemExit(f'no files match {pat!r}')

def load(fn):
    df = pd.read_csv(fn, skiprows=3, header=None, encoding='latin1')
    xr, zr = df.iloc[:, 0].to_numpy(), df.iloc[:, 2].to_numpy()
    nx, nz = len(np.unique(xr)), len(np.unique(zr))
    ny = len(xr) // (nx * nz)
    sh = (nx, ny, nz)
    g = lambda c: df.iloc[:, c].to_numpy().reshape(sh)
    return sh, g(0), g(3), g(4), g(5), g(6), g(7), g(9), g(10), g(11)  # x,rho,u,v,w,E,P,T,sos

prev = None
print(f'{"iter":>7} {"res_rho":>10} {"res_rhoE":>10} {"T_min":>9} {"T_max":>10} '
      f'{"P_min[bar]":>10} {"rho_min":>8} {"e_min":>11}  worst-cell(min e)')
print('-' * 100)
first_bad = None
for fn in files:
    it = int(re.search(r'_(\d+)\.csv$', fn).group(1))
    sh, x, rho, u, v, w, E, P, T, sos = load(fn)
    inj = (slice(1, sh[0]-1), slice(1, sh[1]-1), slice(1, sh[2]-1))  # interior
    e = E - 0.5*(u*u + v*v + w*w)              # specific internal energy
    eI = e[inj]
    Tmin, Tmax = np.nanmin(T[inj]), np.nanmax(T[inj])
    Pmin = np.nanmin(P[inj]); rmin = np.nanmin(rho[inj]); emin = np.nanmin(eI)
    bad = np.unravel_index(np.nanargmin(eI), eI.shape)
    bi, bj = bad[0]+1, bad[1]+1
    xb = x[bi, 1, 1]
    # residual vs previous snapshot (rate of change of conserved state)
    if prev is not None:
        rrho  = np.linalg.norm((rho - prev[0]))      / (np.linalg.norm(prev[0]) + 1e-30)
        rrhoE = np.linalg.norm((rho*E - prev[1]))    / (np.linalg.norm(prev[1]) + 1e-30)
        rs = f'{rrho:10.3e} {rrhoE:10.3e}'
    else:
        rs = f'{"-":>10} {"-":>10}'
    flag = ''
    if (not np.all(np.isfinite(T[inj]))) or Tmin <= 0 or rmin <= 0:
        flag = '  <-- NON-PHYSICAL'
        if first_bad is None:
            first_bad = it
    elif Tmin < T_FLOOR:
        flag = f'  <-- T<{T_FLOOR:.0f}K (two-phase risk)'
    if (it // 500) % stride == 0 or flag:
        print(f'{it:7d} {rs} {Tmin:9.2f} {Tmax:10.3e} {Pmin/1e5:10.2f} {rmin:8.2f} {emin:11.3e}  '
              f'(i={bi:2d},j={bj:2d}) x={xb*1e3:6.2f}mm{flag}')
    prev = (rho.copy(), (rho*E).copy())

print('-' * 100)
if first_bad is not None:
    print(f'FIRST non-physical snapshot: iter {first_bad}  '
          f'(min internal energy e=E-ke crossed 0 or T<=0 -> (rho,e)->T inversion fails)')
else:
    print('No non-physical snapshot detected in this set.')
