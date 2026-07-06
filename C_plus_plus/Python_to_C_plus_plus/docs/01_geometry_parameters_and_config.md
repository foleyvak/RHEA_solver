# Task #1 — Geometry Parameters & Configuration

**Goal:** introduce the nozzle / curvilinear-geometry parameters from the Python
version into the C++ RHEA configuration, *without* breaking existing Cartesian cases.

This document explains **what** parameters exist, **why** each is needed, and **how**
the current C++ config differs from the target. It is the reference for the actual
code change (YAML + `readConfigurationFile()` + member variables).

---

## 1. Why this task comes first

Everything downstream (grid build, metrics, transformed fluxes) needs to know *which*
geometry we are running and *with what dimensions*. Before touching any numerics we:

1. Define a **geometry selector** so Cartesian and nozzle cases can coexist.
2. Add the nozzle shape parameters and parse them.
3. Compute the **derived segment geometry** once, at construction, so the grid builder
   (Task #2) can consume clean values.

No flow physics changes in Task #1 — it is purely inputs and bookkeeping. That keeps
this step low-risk and reviewable on its own.

---

## 2. How geometry is configured *today* (previous version)

The domain is a rectangular box, fully described by origin + size + stretching.

`configuration_file.yaml`:
```yaml
problem_parameters:
   x_0: 0.0        # Domain origin x [m]
   y_0: 0.0        # Domain origin y [m]
   z_0: 0.0        # Domain origin z [m]
   L_x: 1.0        # Domain size x [m]
   L_y: 1.0        # Domain size y [m]
   L_z: 0.01       # Domain size z [m]

computational_parameters:
   num_grid_x: 100
   num_grid_y: 100
   num_grid_z: 1
   A_x: 0.0        # Stretching factor x
   A_y: 0.0        # Stretching factor y
   A_z: 0.0        # Stretching factor z
   external_mesh: 'FALSE'
   external_mesh_file: 'external_mesh_file.txt'
```

Parsed in `FlowSolverRHEA::readConfigurationFile()`
(`readConfigurationFile()`, `problem_parameters` block ~`src/FlowSolverRHEA.cpp:412`)
into members `x_0, y_0, z_0, L_x, L_y, L_z,
A_x, A_y, A_z, external_mesh, external_mesh_file`.

`L_y` is a single scalar — the box is the same height everywhere. **This is exactly the
limitation the port removes:** the nozzle needs the height to vary with `x`.

---

## 3. Parameters the nozzle geometry needs (new version)

From `Python/rhea_flow_solver.py` (lines ~117–150). These define the C–D nozzle
contour `L_y(x)` (local half-height / radius as a function of axial position).

### 3.1 Primary (user-provided) parameters

| Python name | Meaning | Example value | Units |
|---|---|---|---|
| `r_t` | Nozzle **throat** radius | `0.8e-3` | m |
| `r_c` | Nozzle **chamber** radius | `2.0e-3` | m |
| `R1_rt` | Convergent→throat arc ratio `R1/r_t` | `10.0` | – |
| `R2_R1` | Chamber→convergent arc ratio `R2/R1` | `3.0` | – |
| `Rexp_rt` | Expansion arc ratio `Rexp/r_t` | `30.0` | – |
| `theta` | Convergent segment inclination angle | `10.0` | deg |
| `alpha` | Conical nozzle half-angle | `3.0` | deg |
| `L_N` | Conical (divergent) section length | `50.0e-3` | m |
| `L_c` | Chamber section length | `3.0e-3` | m |

**Why these?** The C–D contour is built from circular arcs joined by straight
segments. Radii are given as *ratios* (`R1_rt`, `R2_R1`, `Rexp_rt`) because that is how
nozzle designs are parameterized — it keeps the shape self-similar when `r_t` changes.
`theta`/`alpha` set the convergent/divergent cone angles; `L_c`/`L_N` set the axial
lengths of chamber and divergent cone.

### 3.2 Derived parameters (computed once, not user input)

From `Python/rhea_flow_solver.py:127–147`:

```
R1    = r_t * R1_rt                         # convergent–throat arc radius
R2    = R1  * R2_R1                          # chamber–convergent arc radius
Rexp  = r_t * Rexp_rt                        # expansion arc radius
theta_rad = theta * pi/180
alpha_rad = alpha * pi/180

x_c   = L_c                                  # chamber end (axial)
r2    = r_c - R2*(1 - cos(theta_rad))
r1    = r_t + R1*(1 - cos(theta_rad))
x2    = x_c + R2*sin(theta_rad)
x1    = x2 + (r2 - r1)/tan(theta_rad)        # accounts for radius change r2→r1
x_t   = x1 + R1*sin(theta_rad)               # throat axial location
x_exp = x_t + Rexp*sin(alpha_rad)
r_exp = r_t + Rexp*(1 - cos(alpha_rad))

L_x   = x_t + L_N + L_c                      # TOTAL domain length is DERIVED
```

**Critical difference:** in the nozzle case, **`L_x` is computed from the geometry**,
not read from YAML (Python line 147 overwrites it). Likewise the effective `L_y` is not
a constant — it is the contour `L_y(x)` evaluated per column (built in Task #2).

### 3.3 The contour segments (for context; implemented in Task #2)

`L_y(x)` is piecewise (`Python:958–979`): chamber (const `r_c`) → chamber–convergent
arc → straight convergent → throat arc → expansion arc → straight divergent. Task #1
only needs to make the parameters and derived break-points (`x_c, x2, x1, x_t, x_exp`,
radii) available; Task #2 evaluates the piecewise function.

---

## 4. Proposed configuration change

Add a `geometry` block and a selector. Keep `problem_parameters` for the Cartesian box.

```yaml
##### GEOMETRY #####
geometry:
   # Geometry type: CARTESIAN (box; use problem_parameters L_x/L_y/L_z)
   #                NOZZLE_CD (convergent-divergent nozzle; L_x and contour derived)
   geometry_type: 'CARTESIAN'      # default → existing behavior unchanged
   nozzle:
      r_t: 0.8e-3                  # Throat radius [m]
      r_c: 2.0e-3                  # Chamber radius [m]
      R1_rt: 10.0                  # Convergent-throat arc ratio R1/r_t [-]
      R2_R1: 3.0                   # Chamber-convergent arc ratio R2/R1 [-]
      Rexp_rt: 30.0                # Expansion arc ratio Rexp/r_t [-]
      theta: 10.0                  # Convergent inclination angle [deg]
      alpha: 3.0                   # Conical half-angle [deg]
      L_N: 50.0e-3                 # Conical (divergent) section length [m]
      L_c: 3.0e-3                  # Chamber section length [m]
```

> **Design choices & rationale**
> - **Default `CARTESIAN`.** Existing YAML files that lack the `geometry` block, or set
>   it to `CARTESIAN`, behave exactly as before. This protects every current case.
> - **Nested `nozzle` sub-block.** Keeps nozzle-only knobs grouped and makes room for
>   future geometry types without polluting the top level.
> - **`L_x` stays in `problem_parameters` but is overridden** when `geometry_type ==
>   NOZZLE_CD` (mirrors Python line 147). We will log the derived `L_x` so the user
>   sees the effective domain length.

---

## 5. Code change — as built

> Implemented in Task #1. See `docs/CHANGELOG.md` for the exact file/symbol record.
> All nozzle members use a `nozzle_` prefix to avoid clashing with generic names
> (`r_c`, `theta`, …) elsewhere in the solver.

**`FlowSolverRHEA.hpp`** — members added:
```cpp
std::string geometry_type = "CARTESIAN";            // "CARTESIAN" | "NOZZLE_CD"
// nozzle primary
double nozzle_r_t, nozzle_r_c, nozzle_R1_rt, nozzle_R2_R1, nozzle_Rexp_rt,
       nozzle_theta, nozzle_alpha, nozzle_L_N, nozzle_L_c;
// nozzle derived
double nozzle_R1, nozzle_R2, nozzle_Rexp, nozzle_theta_rad, nozzle_alpha_rad;
double nozzle_x_c, nozzle_r1, nozzle_r2, nozzle_x1, nozzle_x2,
       nozzle_x_t, nozzle_x_exp, nozzle_r_exp;
```
plus the method declaration `virtual void calculateDerivedNozzleGeometry();`.

**`FlowSolverRHEA.cpp` — `readConfigurationFile()`** (after the `computational_parameters`
block): reads the optional `geometry` node defensively — a missing node defaults to
`CARTESIAN` via an explicit `if( geometry ) … else geometry_type = "CARTESIAN";`. When
`geometry_type == "NOZZLE_CD"`, the 9 `nozzle_*` params are read and
`calculateDerivedNozzleGeometry()` is called.

**New helper `calculateDerivedNozzleGeometry()`** — a direct transcription of Python
127–147, computing the arc radii/angles/break-points, overwriting `L_x`, and logging the
derived geometry on rank 0.

> **Note on backward-compatible parsing.** Reading `geometry` defensively (treat a
> missing node as `CARTESIAN`) means the change is additive: no existing YAML file must
> be edited to keep working.

---

## 6. Acceptance criteria for Task #1

- [ ] `geometry` block parses; missing block ⇒ `CARTESIAN` (existing cases unaffected).
- [ ] For `NOZZLE_CD`, all 9 primary params are read and the ~13 derived quantities are
      computed and match Python for the reference values (`r_t=0.8e-3`, `r_c=2e-3`,
      `theta=10°`, `alpha=3°`, `L_N=50e-3`, `L_c=3e-3`, ratios `10/3/30`).
- [ ] Derived `L_x` is logged and equals `x_t + L_N + L_c`.
- [ ] No numerics touched yet; a Cartesian run is byte-identical to pre-change.

---

## 7. What Task #1 deliberately does NOT do

- It does **not** build the `L_y(x)` contour or the grid → **Task #2**.
- It does **not** allocate or compute any metric fields → **Tasks #3–#5**.
- It does **not** change fluxes, time step, or BCs → **Tasks #6–#10**.

Keeping the boundary tight makes this a clean, self-contained, reviewable first step.
