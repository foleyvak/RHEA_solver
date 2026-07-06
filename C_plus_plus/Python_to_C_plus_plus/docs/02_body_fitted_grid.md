# Task #2 — Body-Fitted Physical Grid with `L_y(x)` Nozzle Contour

**Goal:** make the physical grid follow the nozzle wall, i.e. let the domain half-height
vary with axial position `x` via the contour `L_y(x)`, instead of being a constant-height
box. This produces the `x_field / y_field / z_field` **physical** coordinates that all the
metric/flux tasks (#4–#8) will differentiate.

Prerequisite: Task #1 (geometry params + derived break-points) — done.

---

## 1. Why this task

The nozzle wall is a curved contour. On a Cartesian box the wall can only be a straight
edge. A **body-fitted** grid places the top grid line (`j = num_grid_y`) exactly on the
wall `y = L_y(x)` and compresses the interior lines smoothly beneath it, so:

- wall boundary conditions are applied on real grid faces (no stair-steps / IBM);
- the throat and boundary layer are well resolved;
- the later coordinate transformation (metrics/Jacobian) has a smooth mapping to work on.

---

## 2. How the grid is built today (previous version)

The C++ mesh is **separable** (tensor product). Pipeline:

1. `ComputationalDomain::calculateGlobalGrid()` (`ComputationalDomain.cpp:36`) fills 1-D
   global arrays with stretching:
   ```
   globx[i] = x_0 + L_x*g(eta_x)          eta_x = (i-0.5)/gNx
   globy[j] = y_0 + L_y*g(eta_y)          eta_y = (j-0.5)/gNy
   globz[k] = z_0 + L_z*g(eta_z)          eta_z = (k-0.5)/gNz
   g(eta) = eta + A*(0.5 - eta)*(1 - eta)*eta      (stretching profile)
   ```
   with symmetric boundary-cell adjustments at `j=0` and `j=gNy+1`.
2. `ParallelTopology` (`ParallelTopology.cpp:253-271`) copies each rank's slice into the
   **local** arrays `mesh->x[i], mesh->y[j], mesh->z[k]`.
3. `FlowSolverRHEA::fillMeshCoordinatesSizesFields()` (~`FlowSolverRHEA.cpp:784`) fills the
   3-D fields:
   ```cpp
   x_field[I1D(i,j,k)] = mesh->x[i];
   y_field[I1D(i,j,k)] = mesh->y[j];   // same height for every column i  ← the limitation
   z_field[I1D(i,j,k)] = mesh->z[k];
   ```

`mesh->y[j]` has no `i`-dependence, so **the box height is constant**. That is exactly
what Task #2 changes.

---

## 3. What the Python version does (reference)

`spatial_discretization()` (`Python/rhea_flow_solver.py:900-1015`):

```python
eta_y = (j - 0.5)/num_grid_y
L_y[i] = contour(x_i)                       # piecewise nozzle radius, lines 963-979
grid[i][j][k][1] = y_0 + L_y[i]*eta_y + A_y*(0.5*L_y[i] - L_y[i]*eta_y)*(1-eta_y)*eta_y
```

i.e. **the same stretching profile `g(eta_y)`, but scaled by the local radius `L_y[i]`
instead of a constant `L_y`**:

```
y_phys(i,j) = y_0 + L_y(x_i) * g(eta_y_j)
```

The contour (Python 963-979):

| Axial range | Segment | Radius `L_y(x)` |
|---|---|---|
| `x ≤ x_c` | chamber | `r_c` |
| `x_c < x ≤ x2` | chamber→convergent arc (center `x_c`) | `r_c - R2*(1 - sqrt(1 - ((x-x_c)/R2)^2))` |
| `x2 < x ≤ x1` | straight convergent | `r1 - (x - x1)*tan(theta)` |
| `x1 < x ≤ x_t` | throat arc (center `x_t`) | `r_t + R1*(1 - sqrt(1 - ((x-x_t)/R1)^2))` |
| `x_t < x ≤ x_exp` | expansion arc (center `x_t`) | `r_t + Rexp*(1 - sqrt(1 - ((x-x_t)/Rexp)^2))` |
| `x > x_exp` | straight divergent | `r_exp + (x - x_exp)*tan(alpha)` |

(Python also normalizes `grid[...][1] /= L_y[i]` afterwards to store the *computational*
`eta`. We do **not** store that here — see §5.)

---

## 4. C++ approach — reuse the existing stretching via a ratio

Key observation: the base separable mesh is already built with the config `L_y`, so

```
(mesh->y[j] - y_0) / L_y  ==  g(eta_y_j)      (exactly, incl. stretching & boundary cells)
```

Therefore the body-fitted physical `y` is simply the base profile rescaled by the local
radius:

```cpp
y_field[I1D(i,j,k)] = y_0 + ( nozzleRadius(x_field[i]) / L_y ) * ( mesh->y[j] - y_0 );
```

**Why this is the clean choice:**
- It reuses the existing, tested stretching machinery (including the symmetric
  boundary-cell adjustment at `j=0` / `j=gNy+1`) — no re-deriving `eta_y` or recovering
  the global `j` index per rank.
- It is provably identical to Python: substituting `mesh->y[j] = y_0 + L_y*g(eta)` gives
  `y_0 + nozzleRadius(x_i)*g(eta)`, matching Python line 991 with `L_y[i] = nozzleRadius`.
- `x_field` (axial) and `z_field` (span) are unchanged from the separable mesh — the
  nozzle only reshapes `y`. `x` already spans the derived `L_x` from Task #1.

Requirement: `L_y` (config, `problem_parameters.L_y`) is the reference height used to
build the mesh; it must be non-zero (always true). Its exact value is irrelevant because
it cancels in the ratio — but it must match what the `ComputationalDomain` was built with
(it does: both read the same config `L_y`).

### 4.1 Contour helper

Add `double nozzleRadius(double x)` — a host member function transcribing Python 963-979,
using the Task #1 derived members (`nozzle_x_c, nozzle_x2, nozzle_x1, nozzle_x_t,
nozzle_x_exp`, radii, angles). Runs on the CPU inside `fillMeshCoordinatesSizesFields`
(that routine computes on host, then does `#pragma acc update device`).

### 4.2 Branch in `fillMeshCoordinatesSizesFields()`

```cpp
if( geometry_type == "NOZZLE_CD" ) {
    x = mesh->x[i];
    x_field = x;
    y_field = y_0 + ( nozzleRadius(x)/L_y )*( mesh->y[j] - y_0 );
    z_field = mesh->z[k];
} else {                       // CARTESIAN (unchanged)
    x_field = mesh->x[i]; y_field = mesh->y[j]; z_field = mesh->z[k];
}
```

---

## 5. Scope boundary / notes for later tasks

- **We store PHYSICAL coordinates only** in `x/y/z_field`. The *computational* coordinates
  the metrics need are: `xi ~ x` (physical, unnormalized), `eta ~ y/L_y(x)` (normalized),
  `zeta ~ z`. As implemented in Task #4, `eta` is reconstructed on the fly as
  `y_field / nozzleRadius(x_field)` — matching Python's `grid[...][1] /= L_y[i]` exactly
  (Python divides the full physical `y`; the nozzle uses `y_0 = 0`). No extra field is
  stored here. This is called out so Task #4 uses the **normalized** `eta` spacing in the
  metric denominators (not the raw physical `y`). See `docs/04_cell_metrics.md`.
- **Halo caveat:** `x/y/z_field` halos are intentionally not exchanged
  (the "do not activate" comment in `fillMeshCoordinatesSizesFields()`, ~`FlowSolverRHEA.cpp:808`).
  Since `x_field[i]` for a column is well-defined per rank
  and `nozzleRadius` is a pure function of `x`, the body-fitted `y` is correct in each
  rank's owned + halo range as long as `mesh->y[j]` and `mesh->x[i]` are populated there
  (they are, via `ParallelTopology`). Task #11 revisits coordinate halos for metric
  stencils at rank boundaries.
- **GPU:** no kernel change here (host fill + existing `update device`); the contour math
  is host-side.

---

## 6. Acceptance criteria

- [ ] `CARTESIAN` path unchanged (byte-identical grid to pre-change).
- [ ] `NOZZLE_CD`: `y_field` top line lies on `L_y(x)`; `y_field` at mid-throat equals the
      throat radius profile; `x_field` spans `[x_0, x_0 + L_x_derived]`.
- [ ] Generated `(x_field, y_field)` matches Python `physical_grid[:,:,k,0:2]` on the same
      grid (e.g. 32×32×1) within round-off.
- [ ] Contour is continuous across all six segment junctions.
