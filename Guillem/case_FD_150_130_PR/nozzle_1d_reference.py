"""Quasi-1-D real-gas isentropic nozzle reference (CoolProp), for validating the
finite-volume solver's converged centerline.

Given chamber (stagnation) conditions and the nozzle contour, marches the isentropic
flow from the chamber through the sonic throat into the supersonic diverging section,
using CoolProp for the real-gas equation of state (ground truth). For each axial
station x with area ratio A(x)/A* = L_y(x)/rt (planar nozzle), it solves mass
conservation rho(P)*V(P) = rho* V* (A*/A) on the subsonic branch (x<0) or the
supersonic branch (x>0).

Optionally overlays a solver CSV centerline (pass the path as argv[1]).

NOTE: this is the *isentropic* (shock-free, fully-expanding) reference. With a finite
back pressure (P_exit) and a large area ratio the real nozzle is over-expanded and
carries a normal shock in the diverging section -- downstream of the shock the solver
will depart from this curve. The reference is the right target for the converging
section, the throat, and the pre-shock expansion.

Run:  pixi run python nozzle_1d_reference.py [solver_output.csv]
"""
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import CoolProp.CoolProp as CP

import nozzlegeometry as nz_cont


# ----- Fluid & chamber (stagnation) conditions -----------------------------------
# Match rhea_flow_solver.py: P_ref/T_ref are the reservoir TOTAL conditions, P_exit the
# back pressure imposed at a subsonic exit.
FLUID = 'CO2'
P0 = 150.0e5     # chamber total pressure [Pa]   (= P_ref)
T0 = 400.0       # chamber total temperature [K] (= T_ref)
P_exit = 130.0e5  # back pressure [Pa]            (= P_exit)

# ----- Geometry (match rhea_flow_solver.py) --------------------------------------
rt      = 1.0e-3
rc      = 2.5e-3
R1_rt   = 5.0
R2_R1   = 0.5
Rexp_rt = 2.0
theta   = math.radians(15.0)
alpha   = math.radians(20.0)
L_N     = 30.0e-3


def Ly(x):
    return nz_cont.nozzle_top_contour(x, rt, rc, R1_rt, R2_R1, theta, Rexp_rt, alpha)


# ----- Isentrope helpers (state along s = s0) ------------------------------------
s0 = CP.PropsSI('S', 'P', P0, 'T', T0, FLUID)
h0 = CP.PropsSI('H', 'P', P0, 'T', T0, FLUID)


def state_at_P(P):
    """Return (rho, V, a, T, M) on the isentrope at pressure P, or None if two-phase."""
    try:
        h = CP.PropsSI('H', 'P', P, 'S', s0, FLUID)
        rho = CP.PropsSI('D', 'P', P, 'S', s0, FLUID)
        a = CP.PropsSI('A', 'P', P, 'S', s0, FLUID)
        T = CP.PropsSI('T', 'P', P, 'S', s0, FLUID)
    except ValueError:
        return None
    V = math.sqrt(max(2.0 * (h0 - h), 0.0))
    return rho, V, a, T, V / a


def mass_flux(P):
    st = state_at_P(P)
    return None if st is None else st[0] * st[1]


# Lowest single-phase pressure on the isentrope (below it, CoolProp -> two-phase).
P_floor = P0
for Ptest in np.linspace(P0, 0.01 * P0, 600):
    if state_at_P(Ptest) is None:
        break
    P_floor = Ptest
G_floor = mass_flux(P_floor)

# ----- Sonic / throat state: V(P*) = a(P*).  Throat is single-phase, bracket safely.
P_star = brentq(lambda P: state_at_P(P)[1] - state_at_P(P)[2], 0.5 * P0, 0.99 * P0, xtol=1.0)
rho_star, V_star, a_star, T_star, _ = state_at_P(P_star)
G_star = rho_star * V_star          # max mass flux density (at the throat)
print(f'Chamber: {P0/1e5:.1f} bar / {T0:.1f} K   ({FLUID})')
print(f'Throat (sonic): P* = {P_star/1e5:.2f} bar, T* = {T_star:.2f} K, '
      f'V* = a* = {V_star:.2f} m/s, rho* = {rho_star:.2f} kg/m3')
print(f'Isentrope single-phase floor: P >= {P_floor/1e5:.2f} bar (condensation below)')
print(f'Area ratio at exit A_exit/A* = {Ly(L_N)/rt:.2f}\n')


def solve_station(x):
    """Solve P(x) on the correct branch; return None if the station is in the two-phase region."""
    A_ratio = Ly(x) / rt                 # A(x)/A*  (>= 1)
    if A_ratio <= 1.0 + 1e-9:
        return P_star
    G_target = G_star / A_ratio          # rho*V at this station
    if x <= 0.0:
        return brentq(lambda P: mass_flux(P) - G_target, P_star * (1.0 + 1e-6), P0, xtol=1.0)
    # supersonic branch: root in [P_floor, P*]; below P_floor it would be two-phase
    if G_target <= G_floor:
        return None
    return brentq(lambda P: mass_flux(P) - G_target, P_floor * (1.0 + 1e-6), P_star * (1.0 - 1e-6), xtol=1.0)


# ----- March the nozzle ----------------------------------------------------------
x_left = -6.0e-3
xs = np.linspace(x_left, L_N, 240)
P_arr = np.full_like(xs, np.nan); T_arr = np.full_like(xs, np.nan)
M_arr = np.full_like(xs, np.nan); V_arr = np.full_like(xs, np.nan); rho_arr = np.full_like(xs, np.nan)
x_twophase = None
for i, x in enumerate(xs):
    P = solve_station(x)
    if P is None:
        if x_twophase is None:
            x_twophase = x
        continue
    rho, V, a, T, M = state_at_P(P)
    P_arr[i] = P; T_arr[i] = T; M_arr[i] = M; V_arr[i] = V; rho_arr[i] = rho
if x_twophase is not None:
    print(f'Isentrope enters two-phase at x = {x_twophase*1e3:.2f} mm '
          f'(A/A* = {Ly(x_twophase)/rt:.2f}); reference truncated there.')


# ===== Normal-shock + back-pressure analysis =====================================
# An over-expanded CD nozzle carries a steady normal shock in the diverging section.
# Question: for the given P_exit, is the steady shock UPSTREAM of condensation onset
# (whole steady solution single-phase) or not (flow must enter the two-phase dome)?

def state_on_isentrope(P, s_iso, h_stag):
    try:
        h   = CP.PropsSI('H', 'P', P, 'S', s_iso, FLUID)
        rho = CP.PropsSI('D', 'P', P, 'S', s_iso, FLUID)
    except ValueError:
        return None
    V = math.sqrt(max(2.0 * (h_stag - h), 0.0))
    return rho, V

def normal_shock(P1, rho1, V1):
    # Real-gas Rankine-Hugoniot across the shock: conserve mass G=rho*V, momentum
    # P + G^2/rho, and energy h + V^2/2 = h0. Solve for the downstream specific volume
    # v2 (the compression root v2 < v1; v2 = v1 is the trivial no-shock root).
    G = rho1 * V1
    v1 = 1.0 / rho1
    def resid(v2):
        P2 = P1 + G * G * (v1 - v2)
        if P2 <= 0.0:
            return 1.0e30
        h2 = h0 - 0.5 * (G * v2) ** 2
        try:
            return CP.PropsSI('D', 'P', P2, 'H', h2, FLUID) - 1.0 / v2
        except ValueError:
            return 1.0e30
    vs = np.linspace(v1 / 8.0, v1 * 0.999, 80)
    r_prev, v_prev = resid(vs[0]), vs[0]
    for v in vs[1:]:
        r = resid(v)
        if np.isfinite(r) and np.isfinite(r_prev) and r * r_prev < 0.0:
            v2 = brentq(resid, v_prev, v, xtol=1e-12)
            P2 = P1 + G * G * (v1 - v2); rho2 = 1.0 / v2
            return P2, rho2, G * v2, CP.PropsSI('S', 'P', P2, 'D', rho2, FLUID)
        r_prev, v_prev = r, v
    return None

Ar_exit = Ly(L_N) / rt
Ar_cond = G_star / G_floor                 # supersonic area ratio at condensation onset

def exit_pressure_for_shock(Ar_s):
    # Pre-shock supersonic state at area ratio Ar_s on the chamber isentrope s0.
    G_target = G_star / Ar_s
    if G_target <= G_floor:
        return None                        # shock at/after condensation -> not single-phase
    P1 = brentq(lambda P: mass_flux(P) - G_target, P_floor*(1.0+1e-6), P_star*(1.0-1e-6), xtol=1.0)
    rho1, V1, a1, T1, M1 = state_at_P(P1)
    sh = normal_shock(P1, rho1, V1)
    if sh is None:
        return None
    P2, rho2, V2, s2 = sh
    # March post-shock subsonic flow (isentrope s2, stagnation enthalpy h0) to the exit.
    G_exit = G_star / Ar_exit
    def f(P):
        st = state_on_isentrope(P, s2, h0)
        return (st[0]*st[1] if st is not None else 1.0e30) - G_exit
    try:
        P_ex = brentq(f, P2*(1.0+1e-6), P0, xtol=1.0)
    except ValueError:
        return None
    return P_ex, M1

def x_of_area_ratio(Ar):                   # invert Ly(x)=Ar*rt on the diverging branch
    try:
        return brentq(lambda x: Ly(x)/rt - Ar, 1.0e-6, L_N, xtol=1e-6)
    except ValueError:
        return float('nan')

print('\n--- Normal-shock / back-pressure analysis (P_exit = %.1f bar) ---' % (P_exit/1e5))
print('Condensation onset: A/A* = %.3f (x = %.2f mm)'
      % (Ar_cond, (x_twophase*1e3 if x_twophase is not None else float('nan'))))
Ar_scan = np.linspace(1.02, Ar_cond * 0.999, 40)
rows = [(Ar_s, *res) for Ar_s in Ar_scan if (res := exit_pressure_for_shock(Ar_s)) is not None]
if rows:
    P_lo = rows[-1][1]; P_hi = rows[0][1]   # exit P: shock at condensation onset -> near throat
    print('Single-phase normal shock gives exit P in [%.1f, %.1f] bar (shock at onset -> at throat).'
          % (P_lo/1e5, P_hi/1e5))
    if P_exit < P_lo:
        print('VERDICT: P_exit=%.1f bar is BELOW the single-phase minimum %.1f bar -> the steady'
              % (P_exit/1e5, P_lo/1e5))
        print('         shock must sit downstream of condensation onset: NO single-phase steady')
        print('         state exists; the flow necessarily over-expands into the two-phase dome.')
        print('         Fix (single-phase): raise P_exit above ~%.0f bar and/or raise T0.' % (P_lo/1e5))
    elif P_exit > P_hi:
        print('VERDICT: P_exit=%.1f bar exceeds the shock-at-throat exit pressure %.1f bar -> the'
              % (P_exit/1e5, P_hi/1e5))
        print('         flow likely stays subsonic throughout (no choke / no supersonic pocket).')
    else:
        Ars = np.array([r[0] for r in rows]); Pex = np.array([r[1] for r in rows])
        Ar_match = float(np.interp(P_exit, Pex[::-1], Ars[::-1]))
        print('VERDICT: single-phase steady state EXISTS -> normal shock at A/A* ~ %.2f (x ~ %.2f mm).'
              % (Ar_match, x_of_area_ratio(Ar_match)*1e3))


# ----- Optional: overlay a solver CSV centerline ---------------------------------
solver = None
if len(sys.argv) > 1:
    import pandas as pd
    df = pd.read_csv(sys.argv[1], skiprows=3, header=None, encoding='latin1')
    xr = df.iloc[:, 0].to_numpy(); zr = df.iloc[:, 2].to_numpy()
    nx2 = len(np.unique(xr)); nz2 = len(np.unique(zr)); ny2 = len(xr)//(nx2*nz2)
    shp = (nx2, ny2, nz2)
    X = xr.reshape(shp); P = df.iloc[:, 9].to_numpy().reshape(shp)
    T = df.iloc[:, 10].to_numpy().reshape(shp)
    u = df.iloc[:, 4].to_numpy().reshape(shp); v = df.iloc[:, 5].to_numpy().reshape(shp)
    w = df.iloc[:, 6].to_numpy().reshape(shp); sos = df.iloc[:, 11].to_numpy().reshape(shp)
    Ma = np.sqrt(u*u+v*v+w*w)/sos
    j = 1  # centerline (first interior row)
    solver = dict(x=X[1:nx2-1, j, 1], P=P[1:nx2-1, j, 1], T=T[1:nx2-1, j, 1], M=Ma[1:nx2-1, j, 1])
    print(f'Overlaying solver centerline from {sys.argv[1]}')


# ----- Plot ----------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
ax[0].plot(xs*1e3, P_arr/1e5, 'b-', label='1-D isentropic (CoolProp)')
ax[1].plot(xs*1e3, T_arr, 'b-')
ax[2].plot(xs*1e3, M_arr, 'b-')
if solver is not None:
    ax[0].plot(solver['x']*1e3, solver['P']/1e5, 'r.', ms=3, label='solver centerline')
    ax[1].plot(solver['x']*1e3, solver['T'], 'r.', ms=3)
    ax[2].plot(solver['x']*1e3, solver['M'], 'r.', ms=3)
for a_ in ax:
    a_.axvline(0.0, color='k', ls=':', lw=0.8)
    a_.grid(True, alpha=0.25)
ax[2].axhline(1.0, color='gray', ls='--', lw=0.8)
ax[0].set_ylabel('P [bar]'); ax[1].set_ylabel('T [K]'); ax[2].set_ylabel('Mach')
ax[2].set_xlabel('x [mm]')
ax[0].set_title(f'{FLUID} nozzle: 1-D real-gas isentropic reference')
ax[0].legend(fontsize=9)
fig.tight_layout()
fig.savefig('nozzle_1d_reference.png', dpi=140)
print('Saved nozzle_1d_reference.png')
