# Task #13 — West Subsonic Inflow (Aitken isentropic solver)

**Goal:** port the nozzle **West (inlet)** subsonic-inflow boundary from the Python
reference into `myRHEA::updateBoundaries()`. Follow-up to Task #10.

Reference: `Python/rhea_flow_solver.py:369-448` (West block of `update_boundaries`).

---

## 1. Scheme

The inlet fixes the **reference (stagnation-like) state** `(P_ref, T_ref)` and lets the
velocity come from the interior (Neumann on `u`), then solves the ghost thermodynamic state
from an **isentropic-expansion + stagnation-enthalpy** balance, accelerated by **Aitken's
Δ² process** for robust convergence with a real-gas thermodynamics interface.

Per West ghost cell `(0, j, k)` (inner `i=1`):
```
rho_ref, e_ref = ρ,e(P_ref, T_ref) ;  h_ref = e_ref + P_ref/rho_ref
u_g = u_in                                   # Neumann
v_g = (0 − w_in·v_in)/w_g ;  w_g = (0 − w_in·w_in)/w_g   # Dirichlet 0
P_g, T_g, rho_g ← current ghost (initial guess)
repeat (Aitken, up to max_iter_inflow):
   # two sub-steps produce x1, x2 ; Δ²-accelerate to rho_g
   P_g   = P_ref − u_g² / (1/rho_ref + 1/rho_g)          # isentropic expansion
   e_g   = e(P_g,T_g,rho_g) ;  h_g = h_ref − ½u_g²
   rho_g = P_g / (h_g − e_g) ;  T_g = T(P_g,rho_g)  → x1
   e_g   = e(P_g,T_g,rho_g)
   P_g   = P_ref − u_g²/(1/rho_ref+1/rho_g) ; rho_g = P_g/(h_g−e_g) ; T_g = T(P_g,rho_g) → x2
   denom = x2 − 2x1 + x0
   rho_g = x2 − (x2−x1)² / (denom + ε) ;  T_g = T(P_g,rho_g)
   if |(rho_g − x2)/rho_g| < rel_tol_inflow:  P_g = P_ref − u_g²/(1/rho_ref+1/rho_g); break
   x0 = rho_g
finalize ghost from (u_g,v_g,w_g,P_g,T_g)
```

Thermodynamic calls (present in the C++ interface): `calculateDensityInternalEnergyFrom
PressureTemperature`, `calculateInternalEnergyFromPressureTemperatureDensity`,
`calculateTemperatureFromPressureDensity`.

`s_ref` (Python line 388) is computed but unused → **not ported** (dead, like `drc_dx`).

## 2. C++ mapping

- `P_ref = bocos_P[_WEST_]`, `T_ref = bocos_T[_WEST_]` (the configured inlet reference;
  Task #15 sets these). `max_iter_inflow`, `rel_tol_inflow`, `ε` as constants matching the
  reference (`1e-5`, small `ε`).
- Implemented as a **West block inside `myRHEA::updateBoundaries()`** that overwrites the
  West ghost (the base generic pass runs first and is superseded here), reusing the
  `setGhostState` lambda from Task #10 for the finalize step.
- Weights `w_g,w_in` from the physical-x grid (Python 373-374), `x_0` = domain origin.
- Host loop (GPU deferred to Task #11).

## 3. Acceptance criteria

- [x] West block overwrites the inflow ghost for `NOZZLE_CD` using the Aitken solver.
- [x] Standalone (ideal-gas thermo): converged `(P_g, T_g, rho_g)` matches the actual
      Python West algorithm on 4 `(P_in,T_in,u_in)` inner states ⇒ `max rel diff = 0.0`.
- [x] Non-nozzle cases unaffected (West block only runs inside the `NOZZLE_CD` override).
