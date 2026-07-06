# Task #10 — Curvilinear North-Wall + South-Symmetry Boundary Conditions

**Goal:** apply the nozzle **North (no-slip wall)** and **South (symmetry axis)** boundaries
on the body-fitted grid. The wall pressure/temperature use the **curvilinear Neumann
cross-terms** (folded in from Task #9). West inflow, East outflow, and the `myRHEA` case
setup are split into follow-up Tasks #13/#14/#15.

Reference: `Python/rhea_flow_solver.py` `update_boundaries` — South 543-572, North 574-609.

---

## 1. Why these two need custom code

RHEA's generic BC framework (`FlowSolverRHEA::updateBoundaries`) applies **one** BC type per
boundary (Dirichlet/Neumann/…) to all of `u,v,w,P,T`. The nozzle South and North are
**mixed per-variable** and (for the wall) **metric-dependent**, so neither fits the generic
single-type model:

- **South (symmetry axis, y=y₀):** `u,w,P,T` Neumann (ghost = inner), `v` Dirichlet 0.
- **North (no-slip wall):** `u,v,w` Dirichlet 0, and `P,T` via a **curvilinear Neumann**
  extrapolation using the cell metrics.

## 2. North-wall curvilinear Neumann (the key part) — Python 586-593

At the wall ghost cell `(i, jN, k)` (inner row `jN−1`), with cell metric vectors
`∇ξ, ∇η, ∇ζ` (from `xi/eta/zeta_field`) evaluated at the ghost cell:
```
P_ξ  = (P[i+1,jN-1] − P[i-1,jN-1]) / (x[i+1,jN-1] − x[i-1,jN-1])
P_ζ  = (P[i,jN-1,k+1] − P[i,jN-1,k-1]) / (z[k+1] − z[k-1])
P_rhs = −[ (∇ξ·∇η) P_ξ + (∇ζ·∇η) P_ζ ] / (∇η·∇η)
P_g   = P_in + P_rhs · (η_ghost − η_inner)          # η = y / L_y(x)
```
and the analogous `T_ξ, T_ζ, T_rhs, T_g`. This enforces a wall-normal (η-direction) Neumann
condition on the curved wall: the `∇η·∇η` denominator and the `∇ξ·∇η`, `∇ζ·∇η` cross-dot
products are what make it curvilinear (on a Cartesian grid `∇ξ·∇η = 0` and it reduces to the
plain `P_g = P_in`). `η_ghost − η_inner = (y_field[jN] − y_field[jN-1]) / L_y(x)`.

Velocities are no-slip Dirichlet 0: `u_g = (0 − w_in·u_in)/w_g`, etc. (weights below).

## 3. South symmetry — Python 543-572

`u_g = u_in`, `w_g = w_in`, `P_g = P_in`, `T_g = T_in` (Neumann); `v_g = (0 − w_in·v_in)/w_g`
(Dirichlet 0 → odd reflection of the wall-normal velocity).

## 4. Weights

Following Python (normalized `η = y/L_y`): both boundaries lie midway between ghost and
inner on the symmetric body-fitted grid, so `w_g = w_in = 0.5` (North uses the explicit
midpoint `L_north = ½(η_g+η_in)`; South is symmetric about `y₀=0`). The code computes them
from the η-coordinates rather than hard-coding 0.5, matching the reference.

## 5. Integration & execution model

- New override `myRHEA::updateBoundaries()`:
  1. calls `FlowSolverRHEA::updateBoundaries()` (generic pass → handles West/East for now,
     until Tasks #13/#14, and a placeholder N/S that we overwrite);
  2. if `geometry_type == "NOZZLE_CD"`, **overwrites** the South and North ghost cells with
     the logic above, finalizing each ghost with the same sequence the base uses
     (`calculateDensityInternalEnergyFromPressureTemperature` → conserved + primitives +
     `s/sos/c_v/c_p`).
- **Host loop + `update host/device`** (reads neighbor stencils + metrics; mirrors the
  metric routines). GPU kernelization/`present()` finalized in Task #11.

## 6. Acceptance criteria

- [x] `myRHEA::updateBoundaries()` overwrites S/N for `NOZZLE_CD`; W/E still via base.
- [x] Standalone: the North `P_rhs/T_rhs → P_g/T_g` formula reproduces the literal Python
      transcription on a 32×32×1 nozzle grid with a smooth P/T field ⇒ `max|ΔP_g|=ΔT_g=0.0`.
- [x] On a Cartesian grid the wall Neumann reduces to `P_g = P_in`
      (`max|∇ξ·∇η| = 0.0 ⇒ max|P_g − P_in| = 0.0`).
- [x] Non-nozzle cases unaffected (override returns after base call).
