# Task #4 — Cell-Centered Metrics and Jacobian

**Goal:** fill the cell-centered metric fields allocated in Task #3 —
`det_Jacobian_field`, `xi_field[3]`, `eta_field[3]`, `zeta_field[3]` — from the
body-fitted grid built in Task #2. These are the `∂(ξ,η,ζ)/∂(x,y,z)` gradients and the
Jacobian that the transformed pressure/temperature RHS (Task #9), inviscid fluxes
(Task #6) and time step (Task #8) consume.

Reference: `Python/rhea_flow_solver.py:1033-1096` (inside `spatial_discretization()`).

---

## 1. The computational-coordinate convention (read this first)

This is the subtle part. Python builds two grids in `spatial_discretization()`:

- `physical_grid[i][j][k]` = the **physical** coordinates `(x, y, z)` (body-fitted).
- `grid[i][j][k]` = the **computational** coordinates, which are:
  - `grid[...][0] = x` (physical x, **not** normalized) → so `ξ ≡ x`,
  - `grid[...][1] = y / L_y(x_i)` (physical y **normalized** by the local radius) → `η ∈ [0,1]`,
  - `grid[...][2] = z` (physical z, not normalized) → `ζ ≡ z`.

(The normalization happens at `rhea_flow_solver.py:1014`, `grid[...][1] /= L_y[i]`, with
`y_0 = 0` for the nozzle.)

The metric tensor entries are `∂(physical)/∂(computational)`, computed by finite
differences: **numerators use `physical_grid`, denominators use `grid`**. Because we store
only the physical coordinates in `x/y/z_field` (Task #2), we reconstruct the computational
coordinate where a denominator needs it:

```
comp_x(i,j,k) = x_field                       (ξ = x)
comp_y(i,j,k) = y_field / nozzleRadius(x_i)    (η = y / L_y(x))
comp_z(i,j,k) = z_field                       (ζ = z)
```

Only the **η (j-direction) denominators** actually need the division; ξ and ζ denominators
are plain physical differences. `nozzleRadius(x_i) ≥ r_t > 0`, so the division is safe.

### 1.1 What the metric tensor looks like for the nozzle

With `ξ=x`, `η=y/L_y(x)`, `ζ=z`, the physical-derivative matrix
`M[phys][comp] = ∂(x,y,z)/∂(ξ,η,ζ)` is (per Python's `x/y/z_metric`):

```
            ∂/∂ξ        ∂/∂η       ∂/∂ζ
 x:         1           0          0
 y:         ∂y/∂x       L_y(x)     0
 z:         0           0          1
```

- `∂x/∂ξ = 1` because `ξ ≡ x`.
- `∂y/∂ξ = ∂y/∂x ≠ 0`: the wall slope — this is the whole point of a body-fitted grid.
- `∂y/∂η = L_y(x)`: the local height scaling.

So `det(M) = L_y(x)` (the first `det_Jacobian`, Python line 1082), and the inverse gives
`ξ_x=1, η_y=1/L_y, η_x=-(∂y/∂x)/L_y`, etc.

---

## 2. Algorithm (transcription of Python 1043-1094)

For every cell `(i,j,k)` in the `_ALL_` range:

1. **Metric tensor** `xm[c], ym[c], zm[c]` for `c = ξ(0), η(1), ζ(2)` — physical
   differences over `i`/`j`/`k` divided by the corresponding computational-coordinate
   difference, using **forward** diff at the low boundary, **backward** at the high
   boundary, **central** in between (exactly Python's `if i==0 / elif i==num+1 / else`).
2. **`det_Jacobian` (v1)** = `det(M)` via the 3×3 rule (Python 1082).
3. **Inverse metrics** `xi[d], eta[d], zeta[d]` for `d = x(0), y(1), z(2)` = cofactors of
   `M` divided by `det` (Python 1084-1092). These are the stored `ξ_x … ζ_z`.
4. **`det_Jacobian` (v2, overwrite)** = `det` of the *inverse* matrix
   `[ξ; η; ζ]` (Python 1094). **This overwrite is intentional** — the flux/time-step code
   uses this second value (for the nozzle it equals `1/L_y`). We replicate it exactly.

Storage mapping:
- `det_Jacobian_field[I1D] = det_Jacobian` (v2)
- `xi_field[d][I1D]  = ξ_d`, `eta_field[d][I1D] = η_d`, `zeta_field[d][I1D] = ζ_d`

`drc_dx` (Python 1037-1041) is computed in the reference but **never read** anywhere else,
so it is **not** ported (a dead diagnostic).

---

## 3. C++ placement & execution model

- New method `void calculateCellMetrics()` (declared in `FlowSolverRHEA.hpp`, defined in
  `FlowSolverRHEA.cpp`), guarded to run only for `geometry_type == "NOZZLE_CD"`.
- **Host compute + `update device`**, mirroring `fillMeshCoordinatesSizesFields()`: fill the
  metric `DistributedArray`s on the CPU, then `#pragma acc update device(...)` for them.
  (Metrics are geometry constants computed once — no need for a GPU kernel.)
- **Not wired into `execute()` yet** — that (and the ordering after the grid build) is
  Task #11. Task #4 delivers and unit-validates the routine.

### 3.1 Boundary / MPI caveat

Boundary cells use one-sided differences keyed on the **local `_ALL_` range**
(`topo->iter_common[_ALL_][_INIX_]/[_ENDX_]`, etc.). In **serial** (the validation target)
this is identical to Python's `i==0` / `i==num+1`. Under **multi-rank MPI** a rank's local
`_ALL_` boundary may be an interior interface, where a one-sided difference would be wrong;
correct metrics there require exchanging the coordinate fields' halos. That is deferred to
**Task #11** (which already owns the `x/y/z_field` halo question) and is called out here so
the limitation is explicit.

---

## 4. Acceptance criteria

- [ ] `calculateCellMetrics()` fills `det_Jacobian_field` and `xi/eta/zeta_field[0..2]`.
- [ ] Standalone reimplementation reproduces Python `det_Jacobian`, `xi`, `eta`, `zeta`
      on a 32×32×1 nozzle grid within round-off.
- [ ] For the nozzle: `ξ_x ≈ 1`, `η_y ≈ 1/L_y`, `ζ_z ≈ 1`, and `det_Jacobian ≈ 1/L_y`.
- [ ] Runs only for `NOZZLE_CD`; `CARTESIAN` untouched (fields stay zero).
- [ ] `drc_dx` intentionally omitted; documented above.
