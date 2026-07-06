# Task #7 — Curvilinear Viscous Fluxes

**Goal:** compute the viscous (stress + heat) flux divergence on the body-fitted grid,
using the face metrics from Task #5. Gated for `geometry_type == "NOZZLE_CD"`; the Cartesian
viscous path is left untouched.

Reference: `Python/rhea_flow_solver.py:2079-2319` (`viscous_fluxes`).

---

## 1. Structure (three reused building blocks)

At each of the 6 faces the algorithm needs the **physical gradients** of `u,v,w,T`. These
are built with the *same* pieces as the metric computation (Task #5), so the port reuses
them:

- `compCoord(c, cell)` — computational coordinate (`ξ=x`, `η=y/L_y`, `ζ=z`).
- `cellCentral(F, cell, c)` — cell-centered central derivative `∂F/∂(comp c)`.
- `faceDeriv(F, cell, nd, s)` — one-sided derivative across the face in the normal dir.

**Face gradient (chain rule).** For a face `f` with normal computational dir `nd` and
neighbor cell `nb = cell + s·e_nd`:
```
dcomp[c] = faceDeriv(F, cell, nd, s)                    if c == nd   (normal, one-sided)
         = ½( cellCentral(F,cell,c) + cellCentral(F,nb,c) ) if c ≠ nd  (transverse, averaged)
∂F/∂x_d  = Σ_c  face_metric_field[f][c][d] · dcomp[c]     (d = x,y,z)
```
Note: the viscous fluxes use `face_metric_field` **directly** (∇ of the computational
coordinate), *not* divided by `face_J` (Python 2097 vs. the inviscid's `/face_J`). Verified
against Python's ξ/η/ζ-face gradient blocks (2097-2225), including the `0.5` transverse
factors (absorbed into `dcomp`).

## 2. Per-face stress and heat flux

Face-averaged coefficients/velocities: `μ_a = ½(μ_cell+μ_nb)`, likewise `κ_a, u_a, v_a, w_a`.
With `div = ∂u/∂x + ∂v/∂y + ∂w/∂z`, each face contributes the **row of the stress tensor**
matching its axis (`nd`), plus the energy flux (`q = −κ_a ∇T`):

| face axis | row `T[0],T[1],T[2]` | energy `re` |
|---|---|---|
| ξ (x) | `2μ(u_x−div/3), μ(u_y+v_x), μ(u_z+w_x)` | `u_a·T0+v_a·T1+w_a·T2 − (−κ_a T_x)` |
| η (y) | `μ(v_x+u_y), 2μ(v_y−div/3), μ(v_z+w_y)` | `… − (−κ_a T_y)` |
| ζ (z) | `μ(w_x+u_z), μ(w_y+v_z), 2μ(w_z−div/3)` | `… − (−κ_a T_z)` |

(τ is symmetric, but each cross term is evaluated at a *different* face — e.g. `τ_xy` at ξ
vs. `τ_yx` at η — so they are kept distinct, exactly as Python does.)

The `+` faces (ξ+,η+,ζ+ = f 0,2,4) give `Tp[axis][·]`, `reP[axis]`; the `−` faces
(f 1,3,5) give `Tm`, `reM`.

## 3. Divergence assembly (Python 2251-2309)

Face metric vectors (as in the inviscid flux): `mp[axis] = face_metric_field[f+][axis]/face_J[f+]`,
`mm[axis] = face_metric_field[f−][axis]/face_J[f−]` for the three axes. Then for each
momentum component `mc ∈ {x,y,z}` (and energy):
```
Cp = ( Tp[ξ][mc], Tp[η][mc], Tp[ζ][mc] )     Cm = ( Tm[ξ][mc], Tm[η][mc], Tm[ζ][mc] )
flux = (detJ/Δξ)( mp[ξ]·Cp − mm[ξ]·Cm )
     + (detJ/Δη)( mp[η]·Cp − mm[η]·Cm )
     + (detJ/Δζ)( mp[ζ]·Cp − mm[ζ]·Cm )
```
→ `rhou/rhov/rhow_vis_flux`; energy uses `Cp=reP, Cm=reM` → `rhoE_vis_flux`.
`Δξ/Δη/Δζ` are the computational spacings (same as the inviscid task).

`work_vis_rhoe_flux` is set to `1.0` — this matches the Python reference, where the real
expression is commented out (a disabled placeholder, only relevant to the
pressure-transport scheme of Task #9; unused for the total-energy nozzle case).

---

## 4. C++ structure & a scoping note

- `calculateViscousFluxes()` gets a host-side `if (geometry_type=="NOZZLE_CD"){…return;}`
  guard; the existing Cartesian body is unchanged below it.
- **Execution model:** unlike the inviscid kernel (which passes scalar L/R states to a
  device helper), the viscous face gradient needs a *wide* stencil (normal + transverse
  neighbors of two cells). Expressing that cleanly uses local lambdas that index the
  fields — which compile and run correctly on the **CPU build** (current Makefile), but are
  not GPU-offloadable as-is. So the nozzle viscous path is written as a **host loop +
  `#pragma acc update device`** for now (mirroring the metric routines). Converting it to a
  device kernel (inlined stencil or device helper) is folded into **Task #11** (GPU/MPI),
  consistent with how OpenACC data has been deferred throughout.

---

## 5. Acceptance criteria

- [x] `calculateViscousFluxes()` fills `rho{u,v,w,E}_vis_flux` for `NOZZLE_CD` (guard
      added); Cartesian path unchanged below the guard.
- [x] Generalized assembly vs. a **literal** Python-native transcription (stress rows +
      divergence assembly) on a 32×32×1 nozzle grid with a smooth (u,v,w,T,μ,κ) field:
      `max diff = 5.1e-9` (round-off from operation ordering).
- [x] Constant velocity + constant T ⇒ `max|viscous div| = 0.0` (no spurious stress/heat).
