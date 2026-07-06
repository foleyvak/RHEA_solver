# Changelog — Curvilinear Nozzle Geometry Port

Chronological record of code changes made while porting the modified (curvilinear
convergent–divergent nozzle) geometry from `Python/rhea_flow_solver.py` into the C++
solver `C_pp/flowsolverrhea`. One section per task.

Conventions: paths are relative to `C_pp/flowsolverrhea/`. Line numbers are indicative
(they shift as edits accumulate) — the anchor is the function/symbol name.

---

## Task #1 — Geometry parameters & configuration parsing  ✅ completed

**Intent:** add the nozzle / curvilinear geometry inputs and derive the nozzle contour
break-points, without changing any numerics (Cartesian cases stay byte-identical).
See `docs/01_geometry_parameters_and_config.md` for the rationale.

### `configuration_file.yaml`
- Added a new top-level `geometry:` block between `problem_parameters` and
  `computational_parameters`:
  - `geometry_type: 'CARTESIAN'` — selector (`CARTESIAN` | `NOZZLE_CD`); default keeps
    legacy Cartesian behavior.
  - `nozzle:` sub-block with 9 parameters: `r_t, r_c, R1_rt, R2_R1, Rexp_rt, theta,
    alpha, L_N, L_c` (used only when `geometry_type == NOZZLE_CD`).

### `src/FlowSolverRHEA.hpp`
- New **"Geometry parameters"** member section (after `configuration_file`):
  - `std::string geometry_type = "CARTESIAN";`
  - Primary nozzle members: `nozzle_r_t, nozzle_r_c, nozzle_R1_rt, nozzle_R2_R1,
    nozzle_Rexp_rt, nozzle_theta, nozzle_alpha, nozzle_L_N, nozzle_L_c`.
  - Derived nozzle members: `nozzle_R1, nozzle_R2, nozzle_Rexp, nozzle_theta_rad,
    nozzle_alpha_rad, nozzle_x_c, nozzle_r1, nozzle_r2, nozzle_x1, nozzle_x2,
    nozzle_x_t, nozzle_x_exp, nozzle_r_exp`.
- New method declaration (next to `readConfigurationFile`):
  - `virtual void calculateDerivedNozzleGeometry();`

### `src/FlowSolverRHEA.cpp`
- `readConfigurationFile()`: after the `computational_parameters` block, added a
  defensive parse of the optional `geometry` node — a missing node ⇒ `CARTESIAN`.
  When `geometry_type == "NOZZLE_CD"`, reads the 9 nozzle params and calls
  `calculateDerivedNozzleGeometry()`.
- New `calculateDerivedNozzleGeometry()` (placed after `readConfigurationFile()`):
  transcription of Python `rhea_flow_solver.py:127–147` — computes arc radii
  (`R1, R2, Rexp`), angles in rad, break-points (`x_c, r2, r1, x2, x1, x_t, x_exp,
  r_exp`), and **overwrites `L_x = x_t + L_N + L_c`**. Logs the derived geometry on
  rank 0.

### Validation
- Standalone `g++` reimplementation of the derived-geometry formulas reproduces the
  Python reference **exactly** for the reference inputs (`r_t=0.8e-3, r_c=2e-3,
  theta=10°, alpha=3°, L_N=50e-3, L_c=3e-3`, ratios 10/3/30):
  `x_t = 0.0126051754 m`, `L_x = 0.0656051754 m`, etc.
- Full `make` NOT run here: the bundled CoolProp submodule needs Boost math headers
  (`boost/math/tools/toms748_solve.hpp`) that are absent in this environment — a
  pre-existing dependency gap, not related to these edits. Full build/link must be
  confirmed in the complete toolchain.

### Behavioral impact
- None for existing cases: with no `geometry` block (or `CARTESIAN`), nothing changes.
- For `NOZZLE_CD`: only inputs + derived scalars are populated and `L_x` is overwritten;
  grid/flux behavior still unchanged until Task #2+.

---

## Task #2 — Body-fitted physical grid with `L_y(x)` contour  ✅ completed

**Intent:** make the physical grid follow the nozzle wall (height varies with `x`),
producing the body-fitted `x/y/z_field` physical coordinates. Cartesian path unchanged.
See `docs/02_body_fitted_grid.md` for the rationale.

### `src/FlowSolverRHEA.hpp`
- New method declaration (next to `calculateDerivedNozzleGeometry`):
  - `virtual double nozzleRadius(const double &x);`

### `src/FlowSolverRHEA.cpp`
- New `nozzleRadius(const double &x)` (placed after `calculateDerivedNozzleGeometry()`):
  transcription of Python `rhea_flow_solver.py:963-979` — piecewise C-D contour
  (chamber `r_c` → chamber-convergent arc `R2` → straight convergent `theta` → throat
  arc `R1` → expansion arc `Rexp` → straight divergent `alpha`).
- `fillMeshCoordinatesSizesFields()`: added a `geometry_type` branch.
  - `NOZZLE_CD`: `x_field = mesh->x[i]`, `z_field = mesh->z[k]`, and
    `y_field = y_0 + ( nozzleRadius(mesh->x[i])/L_y )*( mesh->y[j] - y_0 )`.
    The ratio `( mesh->y[j] - y_0 )/L_y` equals the base stretching profile `g(eta_y)`
    (incl. boundary-cell adjustment), so this equals Python `y_0 + L_y(x_i)*g(eta_y)`.
  - Else `CARTESIAN`: unchanged legacy fill.

### Validation
- Standalone `g++` reimplementation (base separable mesh + ratio) vs. Python-direct
  body-fitted formula on a 32×32 grid: `max|Δy| = 8.7e-19` (round-off).
- Contour continuous at all 6 junctions; `nozzleRadius(x_t) = 8.0000e-04 = r_t`,
  chamber `= r_c = 2.0e-3`, divergent exit expands as expected.
- Full `make` still not run here (same CoolProp/Boost dependency gap as Task #1).

### Behavioral impact
- `CARTESIAN`: none. `NOZZLE_CD`: `y_field` is now body-fitted to the wall; downstream
  metrics/fluxes still Cartesian until Tasks #3-#8.
- Note for Task #4: computational `eta` = `(y_field - y_0)/nozzleRadius(x_field)`
  (physical `y` is stored; normalization is reconstructed where metric denominators
  need it).

---

## Task #3 — Storage for curvilinear metric fields (cell + face)  ✅ completed

**Intent:** allocate the 70 metric `DistributedArray`s (cell metrics + Jacobian, face
metrics + face Jacobians) so Tasks #4/#5 can fill them. No values computed.
See `docs/03_metric_field_storage.md`.

### `src/MacroParameters.hpp`
- Added readable index `#define`s:
  - Face indices: `_XI_P_ 0, _XI_M_ 1, _ETA_P_ 2, _ETA_M_ 3, _ZETA_P_ 4, _ZETA_M_ 5`.
  - Computational-coordinate indices: `_XI_ 0, _ETA_ 1, _ZETA_ 2`.
  - Physical-direction indices: `_XDIR_ 0, _YDIR_ 1, _ZDIR_ 2`.
- Ordering matches Python `facemetrics`/`face_J` (`rhea_flow_solver.py:1265-1274`).

### `src/FlowSolverRHEA.hpp`
- New "Curvilinear metric fields" member group (after `x/y/z_field`):
  - `DistributedArray det_Jacobian_field;`
  - `DistributedArray xi_field[3], eta_field[3], zeta_field[3];` (cell grad components)
  - `DistributedArray face_metric_field[6][3][3];` (`[face][comp-coord][phys-dir]`)
  - `DistributedArray face_J_field[6];`

### `src/FlowSolverRHEA.cpp`
- Constructor (after `z_field.setTopology`): unconditionally `setTopology` all 70 metric
  fields via loops, generating short unique names (`detJ`, `xi_0..2`, `eta_0..2`,
  `zeta_0..2`, `fJ_0..5`, `fm_f_c_d`), consistent with the existing field-allocation norm.
  `DistributedArray::setTopology` also performs the OpenACC `enter data`; the destructor
  auto-frees. Names ≤ 30 chars.

### Validation
- Fields are `calloc`-initialized to zero, so `CARTESIAN` runs are unaffected (allocated
  but unused). No numerics touched.
- Full `make` still blocked by the CoolProp/Boost dependency gap (same as Tasks #1/#2);
  changes are limited to declarations + allocation loops following the proven pattern.

### Behavioral impact
- None until Tasks #4/#5 populate the fields. Memory: +70 3-D fields (same order as the
  existing unconditional averaging fields). Gating allocation behind `NOZZLE_CD` is noted
  as a future optimization (needs `DistributedArray` default-ctor null-safety first).

---

## Task #4 — Cell-centered metrics and Jacobian  ✅ completed

**Intent:** fill `det_Jacobian_field` and `xi/eta/zeta_field[3]` from the body-fitted grid.
See `docs/04_cell_metrics.md`.

### `src/FlowSolverRHEA.hpp`
- New method declaration: `virtual void calculateCellMetrics();`

### `src/FlowSolverRHEA.cpp`
- New `calculateCellMetrics()` (after `fillMeshCoordinatesSizesFields()`): transcription of
  Python `rhea_flow_solver.py:1043-1094`. Early-returns unless `geometry_type=="NOZZLE_CD"`.
  - Computational-coordinate convention: `xi = x` (physical), `eta = y/nozzleRadius(x)`
    (normalized), `zeta = z` (physical). Numerators use physical `x/y/z_field`; the η
    denominators divide by the local radius; ξ/ζ denominators are plain physical diffs.
  - Forward/backward/central FD keyed on the local `_ALL_` range (serial-correct;
    multi-rank needs coordinate halos — deferred to Task #11).
  - Builds metric tensor `xm/ym/zm`, `detJ` (line 1082), inverse metrics
    (`xi/eta/zeta`, lines 1084-1092), then **overwrites** `det_Jacobian` with the inverse
    matrix determinant (line 1094) — replicated exactly. Stores via `_XDIR_/_YDIR_/_ZDIR_`.
  - Host compute + `#pragma acc update device` for the 10 filled fields.
- `drc_dx` (Python 1037-1041) intentionally **not** ported (dead diagnostic; never read).
- Not yet called from `execute()` — wiring/order is Task #11.

### Validation
- Standalone program running the exact `calculateCellMetrics` logic vs. a faithful
  Python-native transcription on a 32×32×1 nozzle grid: `max diff detJ=7.3e-12, xi=0,
  eta=7.3e-12, zeta=0` (round-off; ξ/ζ bit-identical).
- Throat-column checks: `xi_x=1`, `eta_y=1/L_y`, `zeta_z=1`, `detJ=1/L_y` — as expected.
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- `NOZZLE_CD`: metric fields now populated (still unused by the solver until Tasks #6-#9).
  `CARTESIAN`: `calculateCellMetrics()` early-returns; fields remain zero.

---

## Task #5 — Face-centered metrics and face Jacobians  ✅ completed

**Intent:** fill `face_metric_field[6][3][3]` and `face_J_field[6]` (the intercell-face
metrics used by the flux routines). See `docs/05_face_metrics.md`.

### `src/FlowSolverRHEA.hpp`
- New method declaration: `virtual void calculateFaceMetrics();`

### `src/FlowSolverRHEA.cpp`
- New `calculateFaceMetrics()` (after `calculateCellMetrics()`): transcription of Python
  `rhea_flow_solver.py:1117-1274`, over **inner** cells; early-returns unless `NOZZLE_CD`.
  - Recognized that all 6 faces share one structure: the **normal** metric component is a
    one-sided difference across the face; the **transverse** components are the average of
    the two adjacent cells' central differences. Implemented as a single `f = 0..5` loop
    (`nd = f/2`, `s = ±1`) using three host lambdas: `compCoord` (ξ=x, η=y/L_y, ζ=z),
    `cellCentral`, `faceDeriv`.
  - The invert + double-determinant step is identical to the cell metrics; stores
    `face_metric_field[f][_XI_/_ETA_/_ZETA_][_XDIR_/_YDIR_/_ZDIR_]` and `face_J_field[f]`,
    matching Python `facemetrics[i][j][k][f][c][d]` / `face_J[i][j][k][f]`.
  - Host compute + `#pragma acc update device` (looped over the 6 faces).
- Not yet called from `execute()` — wiring/order is Task #11.

### Validation
- Standalone test computing face metrics **two independent ways** — the generalized loop
  vs. a **literal** transcription of the Python per-face stencils — on a 32×32×1 nozzle
  grid: `max diff facemetrics = 0.0, face_J = 0.0` (bit-identical across all 54+6
  components).
- Adjacency check: `ξ+` face at `(i,j,k)` equals `ξ−` face at `(i+1,j,k)` to `0.0`
  (physical face consistency).
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- `NOZZLE_CD`: face metric fields now populated (unused until Tasks #6/#7).
  `CARTESIAN`: early-returns; fields remain zero.

---

## Task #6 — Curvilinear inviscid fluxes (rotated-frame HLLC)  ✅ completed

**Intent:** compute the Euler flux divergence on the body-fitted grid.
**Design decision (user-approved):** faithfully port Python's first-order 2-point
rotated-frame HLLC, gated for `NOZZLE_CD`; Cartesian high-order path untouched.
See `docs/06_inviscid_fluxes.md`.

### `src/FlowSolverRHEA.hpp`
- Declared 3 device-callable helpers (`#pragma acc routine`): `wavesSpeed(...)`,
  `hllcFlux(...)`, `curvilinearFaceFlux(...)` (all `static`).

### `src/FlowSolverRHEA.cpp`
- Implemented the 3 helpers (verbatim transcription of `rhea_flow_solver.py:1417-1506`
  and the per-face frame/projection at `1757-1793`): `wavesSpeed` (Einfeldt), `hllcFlux`
  (Toro; var_type 0-4), `curvilinearFaceFlux` (build orthonormal frame `(n,t1,t2)` — ξ/η
  use one tangent construction, ζ another to avoid degeneracy — project L/R velocities,
  5 HLLC calls, rotate momentum back, scale by `|grad|`).
- `calculateInviscidFluxes()`: added a host-side `if (geometry_type=="NOZZLE_CD")` guard
  at the top with a new `#pragma acc parallel loop` over inner cells: per cell, a 6-face
  loop (`comp=f/2`, `s=±1`, L/R from `s`) calls `curvilinearFaceFlux` and assembles the
  divergence `Σ (detJ/Δ_comp)(F₊−F₋)` (Python `2050-2068`). The existing Cartesian body is
  unchanged, below the guard.
- OpenACC `present()` lists the core fields + `det_Jacobian_field`; exhaustive
  `face_metric_field`/`face_J_field` clauses deferred to Task #11 (CPU build unaffected).

### Validation
- `hllcFlux`/`wavesSpeed` vs. the actual Python functions on random states:
  `max rel diff = 0.0` (bit-identical).
- Full inviscid divergence: generalized face-loop vs. **literal** Python-native
  per-direction assembly on a 32×32×1 nozzle grid with a smooth field: `max diff = 0.0`.
- Free-stream (uniform) `max|div| = 5.7e-4` — the Python scheme's discrete-GCL residual,
  reproduced faithfully (documented, not a bug).
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- `NOZZLE_CD`: inviscid flux now uses the curvilinear rotated-frame HLLC (1st-order).
  `CARTESIAN`: byte-identical to before (guard not taken).

---

## Task #7 — Curvilinear viscous fluxes  ✅ completed

**Intent:** compute the viscous stress + heat flux divergence on the body-fitted grid.
Gated for `NOZZLE_CD`; Cartesian viscous path untouched. See `docs/07_viscous_fluxes.md`.

### `src/FlowSolverRHEA.cpp`
- `calculateViscousFluxes()`: added a host-side `if (geometry_type=="NOZZLE_CD")` guard at
  the top; existing Cartesian body unchanged below it. The nozzle path (transcription of
  `rhea_flow_solver.py:2079-2319`):
  - Reuses the Task-#5 lambdas (`compCoord`, `cellCentral`, `faceDeriv`) plus a `faceGrad`
    lambda that forms `∂F/∂x_d = Σ_c face_metric_field[f][c][d]·dcomp[c]` (facemetrics used
    directly, **not** `/face_J`), with `dcomp` = one-sided normal / averaged-transverse.
  - Per inner cell: for each axis (ξ/η/ζ) and side (±), builds the stress-tensor **row**
    (`Tp/Tm[axis][0..2]`) and energy flux (`reP/reM[axis]`) via face-averaged μ,κ,u,v,w.
  - Assembles `rho{u,v,w,E}_vis_flux` = `Σ_dir (detJ/Δ_dir)(mp·C − mm·C)` with
    `mp/mm[axis] = face_metric_field/face_J` and stress "columns" `C` (Python 2251-2309).
  - `work_vis_rhoe_flux = 1.0` (matches the reference's disabled placeholder; only used by
    the pressure-transport scheme, Task #9).
- **Execution model:** written as a **host loop + `#pragma acc update device`** (the wide
  face-gradient stencil isn't cleanly device-offloadable via scalar helpers); GPU
  kernelization is folded into Task #11 (documented in the code and doc 07).

### Validation
- Generalized assembly vs. a **literal** Python-native transcription (stress rows +
  `2260-2309` assembly) on a 32×32×1 nozzle grid, smooth field: `max diff = 5.1e-9`
  (round-off from operation ordering).
- Constant velocity + constant T ⇒ `max|viscous div| = 0.0` (no spurious stress/heat).
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- `NOZZLE_CD`: viscous flux now curvilinear. `CARTESIAN`: unchanged (guard not taken).

---

## Task #8 — Jacobian scaling in the time-step (CFL)  ✅ completed

**Intent:** apply the curvilinear wall-normal spacing in `calculateTimeStep()` so the CFL
limit is consistent on the body-fitted grid. See `docs/08_time_step.md`.

### `src/FlowSolverRHEA.cpp`
- `calculateTimeStep()`: added `bool nozzle_geom = (geometry_type=="NOZZLE_CD")` and
  `det_Jacobian_field` to the kernel `present()` clause. Inside the loop, for `nozzle_geom`
  the wall-normal spacing is overridden:
  `delta_y = det_Jacobian_field · 0.5·(y_field[j+1]−y_field[j-1]) / nozzleRadius(x)`
  (Jacobian-scaled normalized-η spacing, Python `rhea_flow_solver.py:876`). `delta_x`,
  `delta_z` stay physical. Cartesian path uses the original physical `delta_y`.
- The rest of the CFL logic (acoustic/viscous/thermal `min` reductions, particle loop) is
  unchanged.

### Validation
- C++ `delta_y` formula vs. Python's `det_J·0.5·(η[j+1]−η[j-1])` (η=y/L_y) on a 32×32×1
  nozzle grid: `max diff = 5.0e-14` (round-off).
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- `NOZZLE_CD`: time step now uses the Jacobian-scaled wall-normal spacing.
  `CARTESIAN`: byte-identical to before (`nozzle_geom == false`).

---

## Task #9 — Pressure/temperature transport RHS (scoping correction + guard)  ✅ completed

**Finding:** the Python reference has **no curvilinear pressure-transport RHS** to port, and
the cross-terms originally cited (lines 588/592) are actually **wall boundary conditions**.
See `docs/09_pressure_transport_rhs.md`.
- Python 588/592 (`P_rhs`/`T_rhs`) are the North-wall curvilinear **Neumann BC** inside
  `update_boundaries` → reassigned to **Task #10** (its description updated).
- The active pressure-transport RHS (`sum_fluxes_source_terms`, Python 2340-2364; used only
  when `transport_pressure_scheme=True`) is **non-curvilinear**; the `det_Jacobian` variant
  is commented out. The nozzle runs total energy (`transport_pressure_scheme=False`).

### `src/FlowSolverRHEA.cpp`
- `readConfigurationFile()`: added a guard rejecting `NOZZLE_CD + transport_pressure_scheme`
  (rank-0 message + `MPI_Abort`). Prevents silent wrong results, since the curvilinear
  inviscid kernel (Task #6) does not fill `P_inv_flux`. No numeric operator changed.

### Validation
- Logic-only (a config-time guard); total-energy nozzle path unaffected.
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- `NOZZLE_CD + transport_pressure_scheme` now aborts with a clear message; all other cases
  unchanged. Wall-BC cross-terms tracked under Task #10.

---

## Task #10 — Curvilinear North-wall + South-symmetry BCs  ✅ completed

**Intent:** apply the nozzle North (no-slip wall, curvilinear P/T Neumann) and South
(symmetry axis) boundaries. West inflow / East outflow / case setup split into Tasks
#13/#14/#15. See `docs/10_wall_symmetry_bc.md`.

### `myRHEA.hpp` / `myRHEA.cpp`
- Declared and implemented `myRHEA::updateBoundaries()` override:
  1. calls `FlowSolverRHEA::updateBoundaries()` (generic pass → West/East until #13/#14, and
     a placeholder N/S that is overwritten);
  2. for `NOZZLE_CD`, overwrites South and North ghost cells (host loop):
     - **South (symmetry, y=y₀):** `u,w,P,T` Neumann; `v` Dirichlet 0 (Python 543-572).
     - **North (no-slip wall):** `u,v,w` Dirichlet 0; `P,T` via curvilinear Neumann
       `P_rhs = −[(∇ξ·∇η)P_ξ + (∇ζ·∇η)P_ζ]/(∇η·∇η)`, `P_g = P_in + P_rhs·(η_g−η_in)`
       (Python 586-593), using the cell metrics `xi/eta/zeta_field` at the ghost.
  - Each ghost finalized via a `setGhostState` lambda mirroring the base
    (`calculateDensityInternalEnergyFromPressureTemperature` → conserved + primitives +
    `s/sos/c_v/c_p`). Host `update host/device`; GPU kernelization in Task #11.

### Validation
- North wall `P_g/T_g` formula vs. literal Python transcription on a 32×32×1 nozzle grid:
  `max|ΔP_g| = 0.0`, `max|ΔT_g| = 0.0` (bit-identical; confirms `η_g−η_in` and the metric
  dot-products).
- Cartesian-grid reduction: `max|∇ξ·∇η| = 0.0` ⇒ `max|P_g − P_in| = 0.0` (wall Neumann
  correctly degenerates to `P_g = P_in`).
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- `NOZZLE_CD`: South/North now use the nozzle curvilinear BCs; West/East still generic
  (Tasks #13/#14). Non-nozzle cases unaffected (override returns after the base call).

---

## Task #13 — West subsonic inflow (Aitken isentropic solver)  ✅ completed

**Intent:** port the nozzle inlet BC. See `docs/13_west_inflow.md`.

### `myRHEA.cpp`
- Added a West block in `myRHEA::updateBoundaries()` (transcription of
  `rhea_flow_solver.py:369-448`): reference state from `bocos_P/T[_WEST_]`; `u` Neumann,
  `v,w` Dirichlet 0; Aitken-Δ² iteration of the isentropic-expansion + stagnation-enthalpy
  balance (`calculateInternalEnergyFromPressureTemperatureDensity`,
  `calculateTemperatureFromPressureDensity`), converged on `rho_g` to `rel_tol_inflow=1e-5`;
  finalize via the shared `setGhostState` lambda. `s_ref` (Python 388) dropped (unused).

### Validation
- Standalone (ideal-gas thermo) vs. the actual Python West algorithm on 4 inner states:
  converged `(P_g, T_g, rho_g)` `max rel diff = 0.0` (bit-identical).
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- `NOZZLE_CD`: West ghost now set by the Aitken inflow (overrides the generic pass).
  Non-nozzle cases unaffected.

---

## Task #14 — East outflow (NSCBC subsonic + supersonic switch)  ✅ completed

**Intent:** port the nozzle outlet BC. See `docs/14_east_outflow.md`.

### `myRHEA.cpp`
- Added an East block in `myRHEA::updateBoundaries()` (transcription of
  `rhea_flow_solver.py:450-541`): a global `Ma_exit` switch (`< 1.2` → subsonic NSCBC
  characteristic outflow with `L1..L5` wave amplitudes relaxing to `P_outlet=bocos_P[_EAST_]`
  and `dQ_i` reconstruction to the ghost; else supersonic zeroth-order copy). `T_g` via
  `calculateTemperatureFromPressureDensityWithInitialGuess`; finalize via `setGhostState`.
- Added `rho_field`/`sos_field` to the nozzle-section `update host` list (read by NSCBC).

### Validation
- Standalone (ideal-gas thermo) vs. the actual Python East NSCBC on 4 inner stencils:
  ghost `(u,v,w,P,T)_g` `max rel diff = 0.0` (bit-identical). Supersonic branch trivial.
- Full `make` still blocked by the CoolProp/Boost dependency gap.
- MPI note: `Ma_exit` single-point switch; multi-rank reduction deferred to Task #11.

### Behavioral impact
- `NOZZLE_CD`: East ghost now set by NSCBC/supersonic outflow. All four nozzle boundaries
  (W/E/N/S) are now curvilinear-aware. Non-nozzle cases unaffected.

---

## Task #15 — myRHEA nozzle case setup (IC + source terms + config)  ✅ completed

**Intent:** define the C-D nozzle case, replacing the default 2-D Riemann problem.
See `docs/15_nozzle_case_setup.md`.

### `myRHEA.cpp`
- `setInitialConditions()`: replaced the 2-D-Riemann quadrant setup with the uniform nozzle
  IC (Python `initialize_uvwPT`): `u=100 m/s`, `v=w=0`, `P=150e5 Pa`, `T=600 K` on all cells.
- `calculateSourceTerms()`: unchanged — already zeroes `f_rhou/rhov/rhow/rhoE` (matches
  Python `source_terms`).

### `configuration_file.yaml`
- `geometry_type: NOZZLE_CD`.
- Fluid: `IDEAL_GAS`, `R_specific: 296.8`, `gamma: 1.4`.
- Transport: `CONSTANT`, `mu: 2.0e-5`, `kappa: 2.0e-2`.
- Problem: `L_y: 1.0e-2` (body-fit reference), `L_z: 1.0e-4`, `final_time: 10.0` (`L_x` derived).
- Computational: `num_grid 32/32/1`, `A_y: -1.0`, `CFL: 0.1`.
- BCs: west `SUBSONIC_INFLOW` (P=150e5, T=600 → inflow reference), east `SUBSONIC_OUTFLOW`
  (P=10e5 = `P_outlet`), south/north `NEUMANN` (overwritten by `myRHEA::updateBoundaries`),
  back/front `PERIODIC`. `transport_pressure_scheme: FALSE` (Task #9 guard not triggered).
- Output name → `nozzle_cd`.

### Validation
- Config values cross-checked against the Python reference (76-77, 152-162, 179-190).
- End-to-end run deferred to Task #12 (after Task #11 wires the metric routines).
- Full `make` still blocked by the CoolProp/Boost dependency gap.

### Behavioral impact
- The default build case is now the C-D nozzle (uniform IC + curvilinear BCs). To restore a
  Cartesian case, set `geometry_type: CARTESIAN` and the corresponding BCs.

---

## Task #11 — Wire metrics into the solver; MPI + OpenACC data  ✅ completed

**Intent:** call the metric routines before the time loop and settle the MPI/GPU data story.
See `docs/11_wiring.md`.

### `src/FlowSolverRHEA.cpp`
- `fillMeshCoordinatesSizesFields()`: appended `this->calculateCellMetrics();` and
  `this->calculateFaceMetrics();` after the `x/y/z_field` fill. Runs wherever the mesh is
  built (constructor + `initializeFromRestart`), once, before the time loop; both
  early-return for non-nozzle.
- `calculateInviscidFluxes()` (nozzle branch): converted from a `#pragma acc parallel loop`
  (with an incomplete `present()` for the 54 face-metric components) to a **host loop** —
  `update host` of inputs → host compute → `update device` of the `*_inv_flux` outputs —
  making all `NOZZLE_CD` operator paths (inviscid/viscous/BC/metrics) uniformly host-based.

### MPI / OpenACC
- **MPI coordinate halos:** no exchange added — `fillMesh` fills `x/y/z_field` over `_ALL_`
  from the per-rank `mesh->x/y/z` (which carry global neighbour coords), the same mechanism
  the base Cartesian flux relies on; halo-cell metric values are never read.
- **Deferred (documented) GPU/MPI items:** device-kernelize the nozzle host loops with full
  `present()` clauses + `nozzleRadius` as `acc routine`; add `MPI_Allreduce` for the East
  `Ma_exit` switch. Require a GPU/multi-rank build to validate; out of scope here.

### Validation
- Brace balance verified in `FlowSolverRHEA.cpp` and `myRHEA.cpp`; wiring/host-loop edits
  confirmed. Full `make` still blocked by the CoolProp/Boost dependency gap (Task #12).

### Behavioral impact
- `NOZZLE_CD`: metrics are now computed and consumed end-to-end (the solver is fully wired).
  `CARTESIAN`: unchanged (metric routines early-return; inviscid uses the original kernel).
