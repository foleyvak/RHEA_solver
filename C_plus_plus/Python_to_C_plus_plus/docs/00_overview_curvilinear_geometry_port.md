# Porting the Modified (Curvilinear Nozzle) Geometry to C++ RHEA — Overview

**Status:** in progress · **Scope owner:** STELLAR / Python→C++ port
**Audience:** anyone touching `C_pp/flowsolverrhea` who needs to understand *why* the
geometry handling is changing.

---

## 1. What this port is about

The Python prototype (`Python/rhea_flow_solver.py`) was extended to run a
**convergent–divergent (C–D) nozzle** on a **body-fitted, generalized curvilinear
grid**. The C++ solver (`C_pp/flowsolverrhea`) is the production, MPI + OpenACC/GPU
implementation, but it still assumes a **separable, orthogonal Cartesian mesh**.

This effort ports the "modified geometry" capability from Python into C++ so the
production solver can simulate the nozzle (and, more generally, any smoothly
varying body-fitted 2D/3D geometry) while keeping its parallel performance.

> **One-line summary:** teach C++ RHEA to solve the compressible Navier–Stokes
> equations in generalized curvilinear coordinates `(ξ, η, ζ)` instead of only on
> a rectangular Cartesian box.

---

## 2. Why we need it (motivation)

- **Physics target.** The STELLAR case of interest is a high-pressure C–D nozzle
  (throat radius `r_t = 0.8 mm`, chamber radius `r_c = 2 mm`, inlet `150 bar / 600 K`,
  outlet `10 bar`). The flow accelerates through a throat whose *wall follows a curved
  contour* — this cannot be represented on a rectangular box.
- **A Cartesian box can't fit the wall.** With the current mesh, the nozzle wall would
  have to be approximated by stair-steps or an immersed boundary. A **body-fitted grid**
  places grid lines *along* the wall, giving clean wall boundary conditions and far
  better resolution of the boundary layer and throat.
- **Already proven in Python.** The transformation, metrics, and flux formulation are
  validated in the Python version. We are porting a *known-good* algorithm, not inventing
  one — which is why the docs constantly cross-reference exact Python line numbers.

---

## 3. Previous version vs. new version (high level)

| Aspect | Previous (Cartesian) | New (Curvilinear nozzle) |
|---|---|---|
| Mesh type | Separable, orthogonal: `x[i], y[j], z[k]` | Body-fitted: `x(i,j,k), y(i,j,k), z(i,j,k)` |
| Wall shape | Straight box edges | Arbitrary contour `L_y(x)` (nozzle arcs) |
| Coordinates | Physical `(x, y, z)` used directly | Physical mapped to computational `(ξ, η, ζ)` |
| Spatial derivatives | `Δx = ½(x[i+1]−x[i−1])`, etc. | Chain rule via **metric tensor** + **Jacobian** |
| Extra stored fields | none | `det(J)`, cell metrics `ξ,η,ζ` gradients, face metrics, face Jacobians |
| Governing eqns | Cartesian conservative form | Transformed conservative form (Jacobian-weighted) |
| Config | fixed box (`L_x, L_y, L_z`) | box **or** nozzle contour params (`r_t, r_c, …`) |

### 3.1 The core mathematical change

In the Cartesian solver, a flux divergence is simply, e.g.

```
d(F)/dx ≈ ( F_{i+1/2} − F_{i−1/2} ) / Δx
```

In curvilinear coordinates we solve on a *uniform computational grid* `(ξ, η, ζ)` and
map back to physical space using the **coordinate transformation**. Derivatives become

```
d/dx = ξ_x ∂/∂ξ + η_x ∂/∂η + ζ_x ∂/∂ζ     (and similarly for d/dy, d/dz)
```

The conservative form is scaled by the transformation Jacobian `J = det(∂(x,y,z)/∂(ξ,η,ζ))`.
The quantities `ξ_x, ξ_y, … , ζ_z` are the **metric terms**; they are what the new
fields store, and they are `1/Δ` (trivial) on a Cartesian grid — which is exactly why
the Cartesian path is a special case and must reproduce the old results.

---

## 4. Where this lives in the two codebases

**Python (reference):** `Python/rhea_flow_solver.py`
- Geometry parameters: lines ~106–150
- Contour `L_y(x)` + grid build: `spatial_discretization()` ~900–1015
- Cell metrics + Jacobian: ~1033–1096
- Face metrics + face Jacobians: ~1098–1274
- Inviscid fluxes (metric-aware): `inviscid_fluxes()` ~1724–2068
- Viscous fluxes (metric-aware): `viscous_fluxes()` ~2079+
- Time step (Jacobian-scaled `Δy`): `time_step()` line 876
- P/T RHS cross terms: lines 588, 592

**C++ (target):** `C_pp/flowsolverrhea/src/FlowSolverRHEA.cpp` + `ComputationalDomain.*`
(C++ line numbers below are approximate — they drift as the port adds code; the
**function name is the authoritative anchor**. Values current as of Task #3.)
- Mesh generation: `ComputationalDomain::calculateGlobalGrid()` (separable today, `ComputationalDomain.cpp:36`)
- Coordinate fields: `FlowSolverRHEA::fillMeshCoordinatesSizesFields()` ~:784
- Inviscid fluxes: `calculateInviscidFluxes()` ~:2860
- Viscous fluxes: `calculateViscousFluxes()` ~:3217
- Time step: `calculateTimeStep()` ~:2590
- Pressure transport: `timeAdvancePressure()` ~:3460
- Boundaries: `updateBoundaries()` ~:1145
- Case setup: `myRHEA.cpp`

---

## 5. Key architectural constraints (why the C++ port is more than a copy-paste)

1. **MPI domain decomposition.** Fields are `DistributedArray`s split across ranks with
   halo layers (`ParallelTopology`). Any metric that reads a neighbor cell must respect
   halos — and note that `x/y/z_field` halos are **not** currently exchanged (see the
   "do not activate" comment in `fillMeshCoordinatesSizesFields()`, ~`FlowSolverRHEA.cpp:808`).
   Metric stencils at rank boundaries need attention.
2. **OpenACC / GPU.** Compute kernels use `#pragma acc parallel loop … present(...)`.
   Every new metric field must be added to `present()` clauses and to the
   `enter data` / `update device` / `exit data` lifecycle. (`DistributedArray::setTopology`
   already issues `enter data` per field, and its destructor issues `exit data`.)
3. **Performance & memory.** The face-metric array is large (`[i][j][k][6][3][3]` in
   Python → 54 components). In C++ it is stored as multiple `DistributedArray`s. As
   implemented in Task #3 they are allocated **unconditionally** (consistent with the
   solver's existing unconditional field allocation); gating allocation behind the
   curvilinear path is a noted future optimization (see `docs/03_metric_field_storage.md`).
4. **Backward compatibility.** All existing Cartesian cases must keep working
   **bit-for-bit**. We gate the new behavior behind a geometry selector and make the
   metrics reduce to the Cartesian formulas when that selector is `CARTESIAN`.

---

## 6. The plan (task map)

The work is tracked as tasks #1–#12. Grouped:

- **Setup (this doc set, task #1):** geometry parameters + config parsing.
- **Grid & metrics (#2–#5):** build body-fitted grid, allocate metric fields, compute
  cell metrics/Jacobian and face metrics/face Jacobians.
- **Transformed operators (#6–#9):** inviscid fluxes, viscous fluxes, time step, and
  P/T transport RHS.
- **Integration & correctness (#10–#12):** boundary conditions, wiring metrics into
  `execute()` with MPI/OpenACC, and validation against Python (plus a Cartesian
  regression).

See `01_geometry_parameters_and_config.md` for the Task #1 details.

---

## 7. Guiding principles for the whole port

- **Match Python numerically first, optimize later.** Keep index/stencil mappings
  identical to the Python reference so we can diff results.
- **Cartesian is a special case, not a separate path where avoidable.** Fewer code
  paths → fewer bugs. Where a separate branch is unavoidable, guard it clearly.
- **Document the "why" at each step.** Every task gets a short rationale so future
  readers understand the transformation, not just the diff.
