# Task #8 — Jacobian Scaling in the Time-Step (CFL)

**Goal:** make the explicit CFL time-step limit consistent with the body-fitted grid by
using the curvilinear wall-normal spacing. Gated for `NOZZLE_CD`; Cartesian unchanged.

Reference: `Python/rhea_flow_solver.py:850-896` (`time_step`), specifically line 876.

---

## 1. Why

`calculateTimeStep()` limits `Δt` by acoustic (`CFL·Δ/S`), viscous (`CFL·ρΔ²/μ`) and
thermal (`CFL·ρc_pΔ²/κ`) scales in each direction, using local spacings `Δx, Δy, Δz`. On
the Cartesian grid these are physical half-differences. On the body-fitted nozzle grid the
wall-normal (`y`) spacing must reflect the coordinate transformation, otherwise the CFL
limit is mis-estimated where the grid is compressed (near the throat/wall).

The Python reference keeps `Δx`, `Δz` as physical half-differences but replaces the
wall-normal one:
```
# delta_y = 0.5*( eta[j+1] - eta[j-1] )          # "Problematic with CFL"
delta_y   = det_Jacobian * 0.5*( eta[j+1] - eta[j-1] )     # eta = y / L_y(x)   (line 876)
```

## 2. Port

Only `delta_y` changes; the rest of `calculateTimeStep()` is untouched.

```cpp
bool nozzle_geom = ( geometry_type == "NOZZLE_CD" );          // host-side
...
delta_x = 0.5*( x_field[i+1] - x_field[i-1] );                 // physical
delta_y = 0.5*( y_field[j+1] - y_field[j-1] );                 // physical (Cartesian)
delta_z = 0.5*( z_field[k+1] - z_field[k-1] );                 // physical
if( nozzle_geom )
    delta_y = det_Jacobian_field · 0.5·( y_field[j+1] − y_field[j-1] ) / nozzleRadius(x);
```

`(y_field[j+1] − y_field[j-1]) / nozzleRadius(x)` is the normalized-η difference
`eta[j+1] − eta[j-1]` (same column ⇒ same `L_y`), so this equals Python's
`det_Jacobian · 0.5·(eta[j+1] − eta[j-1])`. `det_Jacobian_field` is added to the kernel
`present()` clause; `nozzle_geom` is a scalar (firstprivate by default).

GPU note: like the flux kernels, this calls `nozzleRadius(...)` inside the loop — fine on
the CPU build; device-routine finalization is part of Task #11.

## 3. Acceptance criteria

- [x] `NOZZLE_CD`: `delta_y` uses the Jacobian-scaled normalized-η spacing; `delta_x/z`
      remain physical; rest of the CFL logic unchanged.
- [x] C++ `delta_y` vs. Python `det_J·0.5·(η[j+1]−η[j-1])` on a 32×32×1 nozzle grid:
      `max diff = 5.0e-14` (round-off).
- [x] `CARTESIAN`: `nozzle_geom == false` ⇒ byte-identical to the previous behavior.
