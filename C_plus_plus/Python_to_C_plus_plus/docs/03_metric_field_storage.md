# Task #3 — Storage for Curvilinear Metric Fields (cell + face)

**Goal:** allocate the data structures that hold the coordinate-transformation metrics —
cell-centered inverse metrics + Jacobian, and face-centered metrics + face Jacobians — as
`DistributedArray`s, so the compute (Tasks #4/#5) and the transformed operators
(Tasks #6–#8) have somewhere to read/write. No values are computed in this task.

Prerequisites: Task #1 (params), Task #2 (body-fitted grid) — done.

---

## 1. Why this task (and why these exact fields)

The transformed governing equations need, at every cell and every cell face, the
derivatives of the computational coordinates `(ξ, η, ζ)` with respect to the physical
coordinates `(x, y, z)`, plus the Jacobian of the mapping. In the Python reference these
live in `det_Jacobian`, `xi/eta/zeta`, `facemetrics`, and `face_J`
(`Python/rhea_flow_solver.py:215-228`). They are **geometry constants** — computed once
after the grid is built and then reused every RK stage / time step, so precomputing and
storing them (rather than recomputing) is the right call.

A `grep` of the Python solver confirms the full set is genuinely consumed by the flux
routines — nothing here is speculative:

| Quantity | Components used | Python shape |
|---|---|---|
| `det_Jacobian` | 1 | `[i,j,k]` |
| cell `xi`, `eta`, `zeta` gradients | 9 (each 3) | `[i,j,k,3]` |
| `facemetrics` | 54 (6 faces × 3 comp × 3 dir) | `[i,j,k,6,3,3]` |
| `face_J` | 6 | `[i,j,k,6]` |

Total: **70 scalar 3-D fields**.

---

## 2. How storage works in this codebase (previous version)

`DistributedArray` (`DistributedArray.hpp/.cpp`) is a **scalar-per-cell** 3-D field with:
- host allocation via `calloc` and the flattened `vector` accessed by the `I1D(i,j,k)` macro;
- `setTopology(topo, "name")` → allocates **and** issues `#pragma acc enter data
  copyin(vector[0:size])` (GPU);
- `~DistributedArray()` → `#pragma acc exit data delete(...)` + `free()` (auto cleanup);
- `update()` → MPI halo exchange (+ GPU-aware variants).

All existing fields are declared as members of `FlowSolverRHEA` and **unconditionally**
`setTopology`-d in the constructor (e.g. `x_field`, and the ~50 `avg_*/rmsf_*/favre_*`
fields — none are guarded by a runtime flag). GPU lifecycle and MPI halos come "for free"
from `DistributedArray`.

---

## 3. Design decision — layout

`DistributedArray` is scalar-per-cell and has a default constructor, so it can be used in
plain C-style arrays. Naming ~70 individual members would be unwieldy and error-prone;
instead we group them into arrays and `setTopology` them in loops:

```cpp
/// Curvilinear metric fields — cell-centered
DistributedArray det_Jacobian_field;        /// Jacobian determinant J
DistributedArray xi_field[3];               /// grad(xi)   = [ ξ_x, ξ_y, ξ_z ]
DistributedArray eta_field[3];              /// grad(eta)  = [ η_x, η_y, η_z ]
DistributedArray zeta_field[3];             /// grad(zeta) = [ ζ_x, ζ_y, ζ_z ]

/// Curvilinear metric fields — face-centered
DistributedArray face_metric_field[6][3][3];/// [face][comp-coord][phys-dir]
DistributedArray face_J_field[6];           /// face Jacobians
```

### 3.1 Index conventions (must match Python, lines 1265-1274)

**Face index (0..5):**

| idx | face | location |
|---|---|---|
| 0 | `ξ+` | `i+1/2` |
| 1 | `ξ−` | `i−1/2` |
| 2 | `η+` | `j+1/2` |
| 3 | `η−` | `j−1/2` |
| 4 | `ζ+` | `k+1/2` |
| 5 | `ζ−` | `k−1/2` |

**Computational-coordinate index (2nd):** `0=ξ, 1=η, 2=ζ`
**Physical-direction index (3rd):** `0=x, 1=y, 2=z`

So `face_metric_field[f][c][d][I1D(i,j,k)]` is the `d`-physical-derivative of computational
coordinate `c` at face `f` — the direct analogue of Python `facemetrics[i][j][k][f][c][d]`.

These conventions are mirrored as `#define`s (see §4) so the metric-compute and flux code
read symbolically instead of with magic numbers.

### 3.2 Allocation policy — unconditional (with a documented future optimization)

All 70 metric fields are `setTopology`-d **unconditionally** in the constructor, matching
the established codebase norm (the ~50 averaging fields are allocated unconditionally too,
regardless of `time_averaging_active`). Rationale:

- **Consistency & low risk:** no change to shared `DistributedArray` infrastructure.
- **Safety:** `DistributedArray`'s default constructor leaves `vector`/`size`
  uninitialized; a member that is never `setTopology`-d would crash in `~DistributedArray`
  (`free`/`acc exit data` on garbage). Allocating unconditionally sidesteps this entirely.

> **Future optimization (out of scope here):** to allocate metric fields only for
> `NOZZLE_CD` and save memory on large Cartesian runs, first make `DistributedArray`
> default-safe (`double* vector = nullptr; int size = 0;` via in-class initializers), then
> guard the metric `setTopology` calls with `if( geometry_type == "NOZZLE_CD" )`. Deferred
> to keep Task #3 minimal and avoid touching shared infra.

Memory note: 70 fields ≈ the same order as the existing averaging fields; negligible for
the 2-D nozzle case of interest, acceptable elsewhere.

---

## 4. Planned code changes

**`src/MacroParameters.hpp`** — add readable index constants (namespaced with `_..._` to
match existing macros like `_WEST_`, `_INNER_`):
```cpp
// Curvilinear face indices
#define _XI_P_ 0
#define _XI_M_ 1
#define _ETA_P_ 2
#define _ETA_M_ 3
#define _ZETA_P_ 4
#define _ZETA_M_ 5
// Computational-coordinate indices
#define _XI_ 0
#define _ETA_ 1
#define _ZETA_ 2
// Physical-direction indices
#define _XDIR_ 0
#define _YDIR_ 1
#define _ZDIR_ 2
```

**`src/FlowSolverRHEA.hpp`** — declare the arrays from §3 in a new "Curvilinear metric
fields" group alongside the other `DistributedArray` members.

**`src/FlowSolverRHEA.cpp`** (constructor, after the existing `setTopology` block) — loop
and `setTopology` each metric field with a short generated name (`fieldName` is 30 chars):
```cpp
det_Jacobian_field.setTopology(topo,"detJ");
for(int c=0;c<3;c++){ xi_field[c].setTopology(topo, ...); eta_field[c]...; zeta_field[c]...; }
for(int f=0;f<6;f++){ face_J_field[f].setTopology(topo, ...);
    for(int c=0;c<3;c++) for(int d=0;d<3;d++) face_metric_field[f][c][d].setTopology(topo, ...); }
```
Names like `"fm_5_2_2"`, `"fJ_5"`, `"xi_2"` all fit within 30 chars.

No compute, no kernel, no flux change in this task.

---

## 5. Acceptance criteria

- [ ] 70 metric `DistributedArray`s declared and `setTopology`-d; project still builds.
- [ ] Field names unique and < 30 chars.
- [ ] Index conventions (`#define`s) match Python's `facemetrics`/`face_J` ordering.
- [ ] `CARTESIAN` runs unaffected (fields allocated but unused/zero).
- [ ] Values remain zero (initialized by `calloc`) until Tasks #4/#5 fill them.
