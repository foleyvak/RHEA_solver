"""Visualize the latest snapshot from the legacy curvilinear-transform solver
(rhea_flow_solver_marc.py).

IMPORTANT - coordinate system:
That solver writes the COMPUTATIONAL grid to the CSV: the y column is eta = y/L_y(x),
normalized to [0, 1] (ghost cells slightly outside). This script rebuilds the physical
y by multiplying each column by the local wall radius L_y(x) = nozzle_top_contour(x),
so the fields land on the real nozzle cross-section. It also auto-detects whether a CSV
is already physical (finite-volume solver output) and skips the un-normalization in that case.

Run from this folder:  pixi run python visualize_results.py
"""
import glob
import math
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import nozzlegeometry as nz_cont


# ----- Geometry (keep in sync with the solver that produced the data) -------------
rt       = 1.0e-3
rc       = 2.5e-3
R1_rt    = 5.0
R2_R1    = 0.5
Rexp_rt  = 2.0
theta    = math.radians(15.0)
alpha    = math.radians(20.0)
L_N      = 30.0e-3


# ----- Find the latest snapshot in this directory --------------------------------
here = os.path.dirname(os.path.abspath(__file__))
candidates = glob.glob(os.path.join(here, './output_data_*.csv'))
if not candidates:
    raise SystemExit('No output_data_*.csv files found next to this script.')

def _iter_num(path):
    m = re.search(r'output_data_(\d+)\.csv$', os.path.basename(path))
    return int(m.group(1)) if m else -1

filename = max(candidates, key=_iter_num)
output_iter = _iter_num(filename)
print(f'! Latest snapshot: {os.path.basename(filename)} (iteration {output_iter})')


# ----- Load CSV ----------------------------------------------------------------
# Columns: x, y, z, rho, u, v, w, E, s, P, T, sos
df = pd.read_csv(filename, skiprows=3, header=None, encoding='latin1')

# Auto-detect grid size: x depends only on i, z only on k.
_x_raw = df.iloc[:, 0].to_numpy()
_z_raw = df.iloc[:, 2].to_numpy()
n_total = len(_x_raw)
n_x_plus_2 = len(np.unique(_x_raw))
n_z_plus_2 = len(np.unique(_z_raw))
n_y_plus_2 = n_total // (n_x_plus_2 * n_z_plus_2)
num_grid_x = n_x_plus_2 - 2
num_grid_y = n_y_plus_2 - 2
num_grid_z = n_z_plus_2 - 2
print(f'  detected grid: Nx={num_grid_x}, Ny={num_grid_y}, Nz={num_grid_z}')
assert n_x_plus_2 * n_y_plus_2 * n_z_plus_2 == n_total, \
    f'CSV row count {n_total} != (Nx+2)(Ny+2)(Nz+2)'

shape3 = (n_x_plus_2, n_y_plus_2, n_z_plus_2)
def _col(idx):
    return df.iloc[:, idx].to_numpy().reshape(shape3)

x   = _col(0);  y   = _col(1)
rho = _col(3)
u   = _col(4);  v   = _col(5);  w   = _col(6)
P   = _col(9);  T   = _col(10); sos = _col(11)
Ma  = np.sqrt(u * u + v * v + w * w) / sos


# ----- Slice middle plane, drop ghosts, and recover physical y -------------------
isel = slice(1, num_grid_x + 1)
jsel = slice(1, num_grid_y + 1)

x_2d   = x [isel, jsel, 1]          # physical x (constant along j)
y_col  = y [isel, jsel, 1]          # eta (legacy) OR physical y (finite-volume)
P_2d   = P [isel, jsel, 1]
T_2d   = T [isel, jsel, 1]
Ma_2d  = Ma[isel, jsel, 1]

# Wall radius at each i-column from the analytic contour.
L_y_col = np.array([
    nz_cont.nozzle_top_contour(xi, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)
    for xi in x_2d[:, 0]
])

# Decide whether the y column is normalized (legacy, ~[0,1]) or already physical.
is_normalized = float(np.nanmax(np.abs(y_col))) > 5.0 * float(np.nanmax(L_y_col))
if is_normalized:
    print('  y column looks normalized (computational grid) - rebuilding physical y = eta * L_y(x)')
    y_2d = y_col * L_y_col[:, None]
else:
    print('  y column looks physical - using as-is')
    y_2d = y_col


# ----- Analytic wall contour (smooth) ------------------------------------------
x_wall = np.linspace(x_2d[:, 0].min(), x_2d[:, 0].max(), 600)
y_wall = np.array([
    nz_cont.nozzle_top_contour(xi, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)
    for xi in x_wall
])


# ----- Plot --------------------------------------------------------------------
print('! Generating subplots')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 7.5))
fig.suptitle(f'Legacy FD solver - iteration {output_iter}', fontsize=14, fontweight='bold')


def _field_panel(ax, field, cbar_label, fmt='%.4g', cmap='viridis', norm=None, ticks=None):
    pcm = ax.pcolormesh(x_2d, y_2d, field, cmap=cmap, shading='gouraud', norm=norm)
    ax.plot(x_wall, y_wall, color='black', linewidth=1.8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='datalim')
    cbar = fig.colorbar(pcm, ax=ax, format=fmt, ticks=ticks)
    cbar.set_label(cbar_label)
    return pcm


# (1) Pressure
_field_panel(ax1, P_2d / 1.0e5, 'Pressure [bar]', fmt='%.1f')

# (2) Temperature
_field_panel(ax2, T_2d, 'Temperature [K]', fmt='%.1f')

# (3) Mach number, with the sonic line Ma = 1 highlighted.
ma_min = float(np.nanmin(Ma_2d))
ma_max = float(np.nanmax(Ma_2d))
if ma_min < 1.0 < ma_max:
    ma_norm = mcolors.TwoSlopeNorm(vmin=ma_min, vcenter=1.0, vmax=ma_max)
    _field_panel(ax3, Ma_2d, 'Mach number (white = sonic)', fmt='%.2f',
                 cmap='coolwarm', norm=ma_norm, ticks=[ma_min, 1.0, ma_max])
    cs = ax3.contour(x_2d, y_2d, Ma_2d, levels=[1.0],
                     colors='k', linewidths=1.6, linestyles='--')
    ax3.clabel(cs, fmt={1.0: 'Ma=1'}, inline=True, fontsize=8)
else:
    _field_panel(ax3, Ma_2d, 'Mach number', fmt='%.2f')
    print(f'  note: Mach range [{ma_min:.3f}, {ma_max:.3f}] - no Ma=1 line in this snapshot')

# (4) Pressure along axis and wall side
ax4.plot(x_2d[:, 0], P_2d[:, 0] / 1.0e5, linewidth=1.6, label='axis (j=1)')
ax4.plot(x_2d[:, -1], P_2d[:, -1] / 1.0e5, linewidth=1.0, alpha=0.6,
         label=f'wall side (j={num_grid_y})')
ax4.axvline(0.0, color='k', ls=':', lw=0.8, label='throat')
ax4.set_xlabel('x [m]')
ax4.set_ylabel('Pressure [bar]')
ax4.set_ylim(0, None)
ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax4.legend(fontsize=8, loc='best')
ax4.grid(True, alpha=0.25)

plt.tight_layout()
png_filename = os.path.join(here, f'results_iter_{output_iter}.png')
plt.savefig(png_filename, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'! Saved {png_filename}')
