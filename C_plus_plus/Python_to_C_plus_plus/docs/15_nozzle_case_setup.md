# Task #15 — myRHEA Nozzle Case Setup (IC + source terms + config)

**Goal:** define the C-D nozzle case (replacing the default 2-D Riemann problem): uniform
initial conditions, zero source terms, and the YAML configuration that drives the
curvilinear geometry and the BCs from Tasks #10/#13/#14.

Reference: `Python/rhea_flow_solver.py` — `initialize_uvwPT` (288-298), `source_terms`
(786-799), problem/fluid parameters (76-77, 152-162, 179-190).

---

## 1. Initial conditions (`myRHEA::setInitialConditions`)

Python initializes a **uniform** field (line 288-298): `u = U_ref`, `v = w = 0`,
`P = P_ref`, `T = T_ref`. Port replaces the 2-D-Riemann quadrant setup with:
```
u = 100.0   [m/s]      (U_inlet)
v = 0, w = 0
P = 150.0e5 [Pa]       (P_inlet)
T = 600.0   [K]        (T_inlet)
```
applied to all (inner+boundary+halo) cells, then `u/v/w/P/T .update()` halos (as the base
does).

## 2. Source terms (`myRHEA::calculateSourceTerms`)

Python `source_terms` sets `f_rhou = f_rhov = f_rhow = f_rhoE = 0` (no body forces). The
existing `myRHEA::calculateSourceTerms` already zeroes these — **no change needed**.

## 3. Configuration (`configuration_file.yaml`)

| Block | Setting | Value (Python) |
|---|---|---|
| geometry | `geometry_type` | `NOZZLE_CD` (+ nozzle params already present) |
| fluid | `IDEAL_GAS`, `R_specific`, `gamma` | 296.8, 1.4 |
| transport | `CONSTANT`, `mu`, `kappa` | 2.0e-5, 2.0e-2 |
| problem | `x_0,y_0,z_0` / `L_y` / `L_z` | 0,0,0 / 1.0e-2 / 1.0e-4 (`L_x` derived) |
| computational | `num_grid_x/y/z` | 32 / 32 / 1 |
|  | `A_x/A_y/A_z` | 0 / −1.0 / 0 |
|  | `CFL` | 0.1 |
|  | `riemann_solver_scheme` | HLLC (nozzle path uses its own rotated HLLC) |
|  | `transport_pressure_scheme` | FALSE (total energy) |
| BCs | west | `SUBSONIC_INFLOW`, P=150e5, T=600, u=v=w=0 (P/T = inlet reference used by the Aitken solver) |
|  | east | `SUBSONIC_OUTFLOW`, P=10e5 (= `P_outlet`) |
|  | south / north | `NEUMANN` (placeholder; `myRHEA::updateBoundaries` overwrites with symmetry / wall) |
|  | back / front | `PERIODIC` (z homogeneous) |

Notes:
- `L_x` in YAML is a placeholder — `calculateDerivedNozzleGeometry()` overwrites it.
- `bocos_P/T[_WEST_]` serve as the inflow reference `(P_ref,T_ref)` (Task #13);
  `bocos_P[_EAST_]` is `P_outlet` (Task #14).
- South/North YAML types only affect the base generic pass, which the override discards for
  those two boundaries — set to a valid type (`NEUMANN`) to keep the base happy.

## 4. Acceptance criteria

- [x] `setInitialConditions` sets the uniform nozzle IC; `calculateSourceTerms` zero
      (unchanged — already zeroed).
- [x] YAML drives `NOZZLE_CD`, ideal-gas 296.8/1.4, constant μ/κ, and W/E/S/N/back/front BCs.
- [x] Config values match the Python reference; `transport_pressure_scheme` FALSE (so the
      Task #9 guard is not triggered).
- [ ] End-to-end run is exercised in Task #12 (after Task #11 wires the metrics).
