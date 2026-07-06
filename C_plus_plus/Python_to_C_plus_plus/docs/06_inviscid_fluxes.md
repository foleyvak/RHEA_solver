# Task #6 — Curvilinear Inviscid Fluxes (rotated-frame HLLC)

**Goal:** compute the inviscid (Euler) flux divergence on the body-fitted grid, using the
face metrics/Jacobian from Tasks #4/#5. This is the first task that changes a solver
operator.

Reference: `Python/rhea_flow_solver.py:1724-2068` (`inviscid_fluxes`), plus `HLLC_flux`
(1436-1506) and `waves_speed` (1417-1431).

**Design decision (user-approved):** faithfully port the Python scheme — a **first-order,
2-point, rotated-frame HLLC** — gated behind `geometry_type == "NOZZLE_CD"`. The existing
Cartesian high-order path (`calculateInviscidFluxes`, 6-point pluggable Riemann solver) is
left **untouched**.

---

## 1. Why the nozzle path is a different scheme

The C++ Cartesian inviscid flux does per-direction high-order (6-point) reconstruction and
a pluggable Riemann solver, with `flux = (F₊−F₋)/Δ_phys` and no metric terms. That cannot
represent fluxes through the slanted faces of a body-fitted cell.

The Python nozzle scheme instead, at each of the 6 faces:
1. builds the **face-normal orthonormal frame** `(n, t₁, t₂)` from the face metric vector,
2. rotates the L/R velocities into that frame (`V_n, V_t1, V_t2`),
3. solves a **1-D HLLC** Riemann problem along `n` (2-point, first order),
4. rotates the resulting momentum flux back to `(x,y,z)` and scales by the face-metric
   magnitude `|∇(·)|`,
5. sums the six face contributions with the Jacobian weighting.

This is the standard finite-volume curvilinear Euler discretization and is what we port.

---

## 2. Per-face algorithm (transcription)

For inner cell `(i,j,k)`, computational spacings (Python 1740-1742, `grid` = computational
coords, so `ξ=x`, `η=y/L_y`, `ζ=z`):
```
Δξ = ½ ( x_field[i+1] − x_field[i−1] )
Δη = ½ ( y_field[i,j+1] − y_field[i,j−1] ) / nozzleRadius(x_i)
Δζ = ½ ( z_field[k+1] − z_field[k−1] )
```

For each face `f = 0..5` (`ξ+,ξ−,η+,η−,ζ+,ζ−`), with `comp = f/2` (`ξ/η/ζ`):

```
mv = face_metric_field[f][comp][ :] / face_J_field[f]     # metric vector ∇(comp) at face
grad = |mv| ;  n = mv / grad
build (t1, t2) from n  (see §2.1) ; normalize t1, t2
project: V_n = u·n + v·n… (L and R) ; likewise V_t1, V_t2
# 1-D HLLC along n (2-point L/R states across the face):
rho_F   = HLLC(var0, …, P)                     # mass
rho_n_F = HLLC(var1, …, P−P_thermo)            # normal momentum (+pressure)
rho_t1F = HLLC(var2, …, P−P_thermo)            # t1 momentum (advected)
rho_t2F = HLLC(var3, …, P−P_thermo)            # t2 momentum (advected)
rhoE_F  = HLLC(var4, …, P)                      # energy
# rotate momentum flux back to Cartesian, scale all by grad:
rho_F  *= grad
rhou_F  = (rho_n_F·n[0] + rho_t1F·t1[0] + rho_t2F·t2[0])·grad
rhov_F  = (…·n[1] + …·t1[1] + …·t2[1])·grad
rhow_F  = (…·n[2] + …·t1[2] + …·t2[2])·grad
rhoE_F *= grad
```

L/R cells per face: `ξ+`:(i,i+1) `ξ−`:(i−1,i) `η+`:(j,j+1) `η−`:(j−1,j) `ζ+`:(k,k+1) `ζ−`:(k−1,k).

### 2.1 Tangent construction (per direction, avoids degeneracy)

- **ξ and η faces** (normal lies in the x–y plane for the nozzle):
  `t1 = (−n_y, n_x, 0)`, `t2 = (−n_x n_z, −n_y n_z, n_x²+n_y²)` (Python 1761-1764 / 1863-1866).
- **ζ faces** (normal ≈ z): `t1 = (n_z, 0, −n_x)`,
  `t2 = (−n_x n_y, n_x²+n_z², −n_y n_z)` (Python 1964-1967).

The two families are chosen so `t1` is non-degenerate exactly where each is used
(`n` in-plane for ξ/η; `n≈ẑ` for ζ) — important for the 2-D nozzle (`num_grid_z=1`,
`n_ζ=(0,0,1)`), where the ξ/η formula would divide by zero but the ζ formula gives
`t1=(1,0,0)`.

### 2.2 Divergence assembly (Python 2050-2068)

```
rho_inv  = (detJ/Δξ)(rho_F^{ξ+} − rho_F^{ξ−})
         + (detJ/Δη)(rho_F^{η+} − rho_F^{η−})
         + (detJ/Δζ)(rho_F^{ζ+} − rho_F^{ζ−})
```
with `detJ = det_Jacobian_field[i][j][k]`; likewise for `rhou/rhov/rhow/rhoE_inv`.

---

## 3. HLLC & wave speeds (ported verbatim)

`waves_speed` (Einfeldt/Roe averages) and `HLLC_flux` (Toro) are transcribed exactly
(`rhea_flow_solver.py:1417-1506`). In the rotated call the HLLC "u,v,w" are `V_n,V_t1,V_t2`;
`var_type` 0=mass, 1=normal-mom (+P), 2/3=tangential-mom (advected), 4=energy. To match the
Python reference bit-for-bit we port these rather than reuse the C++ `HllcApproximateRiemannSolver`
(whose 6-point interface and wave-speed estimates differ).

---

## 4. C++ structure

- Two device-callable member helpers (`#pragma acc routine`): `wavesSpeed(...)` and
  `hllcFlux(...)`, plus a `curvilinearFaceFlux(...)` helper that does one face (frame +
  projection + 5 HLLC calls + rotate-back), returning the 5 Cartesian face fluxes.
- `calculateInviscidFluxes()` gets a **host-side** guard at the top:
  `if (geometry_type == "NOZZLE_CD") { <curvilinear kernel>; return; }` — the existing
  Cartesian body is unchanged below it.
- The curvilinear kernel is a `#pragma acc parallel loop collapse(3)` over inner points,
  `present(...)` including `x/y/z_field`, the primitive fields, `det_Jacobian_field`,
  `face_metric_field[...][...][...]`, `face_J_field[...]`, and the `*_inv_flux` outputs.
  (Finalizing the exhaustive `present()` list / GPU data is Task #11; CPU build works now.)

---

## 5. Acceptance criteria

- [x] `calculateInviscidFluxes()` produces the curvilinear divergence for `NOZZLE_CD`
      (guard added); Cartesian path unchanged below the guard.
- [x] Standalone check: the generalized face-loop matches a **literal** Python-native
      per-direction assembly on a 32×32×1 nozzle grid with a smooth field ⇒ `max diff = 0.0`.
- [x] `hllcFlux`/`wavesSpeed` reproduce the actual Python functions on random scalar
      states ⇒ `max rel diff = 0.0` (bit-identical).
- [x] Uniform free-stream: `max|inviscid div| = 5.7e-4` — small but **not** exactly zero.
      This scheme does not discretely satisfy the geometric conservation law (the cell and
      face metrics are built by different stencils, so uniform-flow fluxes don't telescope);
      the residual is a property of the **Python reference scheme**, reproduced faithfully,
      not a porting error.
