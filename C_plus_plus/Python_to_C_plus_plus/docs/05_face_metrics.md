# Task #5 — Face-Centered Metrics and Face Jacobians

**Goal:** fill `face_metric_field[6][3][3]` and `face_J_field[6]` (Task #3 storage) with the
coordinate-transformation metrics evaluated at the **six cell faces** (ξ±, η±, ζ±). These
are what the inviscid (Task #6) and viscous (Task #7) flux routines evaluate at each
intercell face.

Reference: `Python/rhea_flow_solver.py:1117-1274` (the face-metric block of
`spatial_discretization()`), which loops over **inner** cells only.

This is the largest block in the port. The key to doing it safely is recognizing the
repeated structure below, rather than transcribing 150 lines of near-duplicated algebra.

---

## 1. The unifying structure (why this is not 150 lines)

At a face normal to computational direction `nd` (with side `s = +` or `−`), the physical
derivative matrix `M[phys][comp] = ∂(x,y,z)/∂(ξ,η,ζ)` is built from:

- **Normal derivative** (component `M[·][nd]`): a compact **one-sided difference across the
  face**, between the two cells the face separates.
- **Transverse derivatives** (`M[·][c]`, `c ≠ nd`): the **average of the two adjacent
  cells' central differences** in direction `c`.

Once `M` is assembled, the face metrics and face Jacobian are obtained with the **exact
same invert-and-determinant routine as the cell metrics** (Task #4):

```
detJ  = det(M)
grad(ξ), grad(η), grad(ζ) = cofactors(M) / detJ     (the stored face metrics)
J_face = det( [grad(ξ); grad(η); grad(ζ)] )         (the stored face Jacobian)
```

I verified component-by-component that Python's ξ-face (lines 1121-1166), η-face
(1168-1214) and ζ-face (1216-1262) blocks are all this same pattern with `nd = ξ/η/ζ`
respectively, and that the cofactor/determinant formulas are identical to the cell-metric
ones (`rhea_flow_solver.py:1082-1094`).

### 1.1 Building blocks (three small host helpers)

Using the same computational-coordinate convention as Task #4
(`ξ=x`, `η=y/L_y(x)`, `ζ=z`):

```
compCoord(c, cell)         = x_field                        if c=ξ
                           = y_field / nozzleRadius(x_field) if c=η
                           = z_field                        if c=ζ

cellCentral(F, cell, c)    = ( F[cell+e_c] − F[cell−e_c] )
                             / ( compCoord(c, cell+e_c) − compCoord(c, cell−e_c) )

faceDeriv(F, cell, nd, s)  = ( F[hi] − F[lo] ) / ( compCoord(nd,hi) − compCoord(nd,lo) )
   with (hi,lo) = (cell+e_nd, cell) for s=+  ;  (cell, cell−e_nd) for s=−
```

`F` is one of the physical fields `x_field / y_field / z_field`. `nozzleRadius(x) ≥ r_t > 0`
so the η normalization is safe. (For η-normal / η-transverse terms, the two cells share the
same `i`, hence the same `L_y`, so the normalization is consistent — matching Python.)

### 1.2 Face assembly (per inner cell, per face)

```
faces: f=0 (ξ+), 1 (ξ−), 2 (η+), 3 (η−), 4 (ζ+), 5 (ζ−)
nd = f/2 ;  s = (f even ? +1 : −1) ;  neighbor = cell + s·e_nd

for c in {ξ,η,ζ}:
   if c == nd:  xm[c],ym[c],zm[c] = faceDeriv(x/y/z, cell, nd, s)
   else:        xm[c],ym[c],zm[c] = ½( cellCentral(·, cell, c) + cellCentral(·, neighbor, c) )

detJ, grad(ξ/η/ζ), J_face   ← same invert+det as Task #4
face_metric_field[f][_XI_][d]   = ξ_d
face_metric_field[f][_ETA_][d]  = η_d
face_metric_field[f][_ZETA_][d] = ζ_d
face_J_field[f]                 = J_face
```

Storage layout matches Python `facemetrics[i][j][k][f][c][d]` (`c`=ξ/η/ζ, `d`=x/y/z) and
`face_J[i][j][k][f]` (`rhea_flow_solver.py:1265-1274`).

---

## 2. Loop range and boundaries

Like Python, the loop is over **inner** cells (`topo->iter_common[_INNER_]`). Every stencil
access (`±1` in each direction, plus the face-neighbor cell) then lands within the `_ALL_`
range, so **no one-sided boundary handling is needed** here (this is simpler than the cell
metrics). Face metrics on the domain-boundary cells remain zero — consistent with Python,
and the flux routines only read inner-cell face metrics.

**MPI note:** the same coordinate-halo caveat as Task #4 applies at multi-rank interfaces
(the transverse averages read neighbor columns); correct in serial, deferred to Task #11.

---

## 3. C++ placement & execution model

- New method `void calculateFaceMetrics()` (declared in `.hpp`, defined in `.cpp`),
  early-returning unless `geometry_type == "NOZZLE_CD"`.
- **Host compute + `#pragma acc update device`** for the 60 filled face fields (54 metric
  components + 6 face Jacobians), mirroring `calculateCellMetrics()`.
- Uses local lambdas (`compCoord`, `cellCentral`, `faceDeriv`) capturing `this`; all
  host-side (geometry constants computed once). Not wired into `execute()` yet (Task #11).

---

## 4. Acceptance criteria

- [x] `calculateFaceMetrics()` fills `face_metric_field[0..5][ξ/η/ζ][x/y/z]` and
      `face_J_field[0..5]` on inner cells.
- [x] Standalone check reproduces the **literal** Python per-face stencils on a 32×32×1
      nozzle grid: **bit-identical** (`max diff = 0.0`) across all 54+6 components. (The
      literal transcription serves as the independent reference for the generalized loop.)
- [x] Consistency check: opposite faces of adjacent cells agree exactly,
      `face(ξ+) @ (i,j,k) == face(ξ−) @ (i+1,j,k)` → `0.0`.
- [x] Runs only for `NOZZLE_CD`; `CARTESIAN` untouched (fields stay zero).
