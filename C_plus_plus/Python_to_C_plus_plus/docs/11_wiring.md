# Task #11 — Wire Metrics into the Solver; MPI + OpenACC Data

**Goal:** make the curvilinear machinery actually run — compute the metrics before the time
loop — and settle the MPI-halo and OpenACC-data story for the nozzle paths.

---

## 1. Wiring the metric computation

`fillMeshCoordinatesSizesFields()` fills the (static) body-fitted `x/y/z_field`. It is
called from the constructor and from `initializeFromRestart()`. Since the metrics are a pure
function of that grid, `calculateCellMetrics()` and `calculateFaceMetrics()` are now called
**at the end of `fillMeshCoordinatesSizesFields()`** — so they run wherever the mesh is
(re)built, exactly once, before the time loop. Both early-return for non-nozzle geometries,
so this is a no-op for `CARTESIAN`.

## 2. MPI: coordinate halos need no exchange

The metric stencils read neighbour coordinates (`i±1, j±1, k±1`, incl. diagonal). Those are
available because `fillMeshCoordinatesSizesFields()` fills `x/y/z_field` over the **`_ALL_`
range (interior + boundary + halo)** from the per-rank local arrays `mesh->x/y/z`, which
already carry the global neighbour coordinates for each rank's halo cells (populated by
`ParallelTopology` from the global grid). This is the same mechanism the **base Cartesian
flux/time-step already rely on** (`delta_x = ½(x_field[i+1]−x_field[i−1])` at inner cells),
so the coordinate halos are correct without an explicit `x/y/z_field.update()`. Halo-cell
metric values themselves are never read (fluxes/BCs use interior + physical-boundary cells),
so one-sided differences at a rank's local `_ALL_` edge are harmless.

**Remaining MPI item (documented, deferred):** the East `Ma_exit` outflow switch (Task #14)
is computed from a single representative cell (as in Python); a multi-rank run needs an
`MPI_Allreduce` to make the switch globally consistent. Flagged here and in `docs/14`.

## 3. OpenACC / GPU status

The build ships with the **CPU** Makefile flags (GPU flags commented). To keep the port
correct and consistent under that reality:

- **Metric fields** get their device allocation from `DistributedArray::setTopology`
  (`enter data`) and are computed on host + `update device` (they are static constants).
- **The `NOZZLE_CD` operator paths run as host loops** — inviscid (converted here from a
  device kernel to a host loop for uniformity, avoiding a 54-entry `present()` clause for the
  face metrics), viscous, and the boundary override — each doing `update host` of inputs →
  host compute → `update device` of outputs. On the CPU build the `acc` pragmas are no-ops;
  on a GPU build these are correct but not yet offloaded.
- **Shared kernels** (`calculateTimeStep`) keep their `#pragma acc parallel loop`; the
  nozzle branch calls `nozzleRadius()` inside, same situation as the existing
  `trilinearInterpolation` (routine annotation deferred).

**Future GPU work (out of scope; needs a GPU build to validate):** convert the nozzle host
loops to device kernels with complete `present()` clauses, mark `nozzleRadius` as
`#pragma acc routine seq`, and add the East-switch `MPI_Allreduce`.

## 4. Acceptance criteria

- [x] `calculateCellMetrics()`/`calculateFaceMetrics()` invoked once after the mesh is built
      (constructor + restart), gated to `NOZZLE_CD`.
- [x] Metric stencils use correct coordinate halos with no added exchange (justified above).
- [x] Nozzle operator paths are host-consistent (inviscid now a host loop like viscous/BC).
- [ ] End-to-end build+run is Task #12 (requires a toolchain with the Boost/CoolProp deps).
