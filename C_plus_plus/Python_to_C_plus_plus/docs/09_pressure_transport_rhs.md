# Task #9 — Pressure/Temperature Transport RHS (scoping correction + guard)

**Outcome:** there are **no curvilinear cross-terms to port** into the pressure-transport
RHS, because the Python reference does not implement a curvilinear pressure-transport path.
This task therefore (a) records the finding, (b) reassigns the cross-terms I had originally
cited to Task #10 where they belong, and (c) adds a safety guard.

---

## 1. What the original task assumed vs. what the code shows

The original Task #9 said: *"port the metric cross-derivative terms in the P and T equation
RHS from Python (lines 588 and 592)."* On close reading of the reference:

### 1a. Lines 588/592 are **wall boundary conditions**, not a transport RHS
They live inside `update_boundaries()` (`Python/rhea_flow_solver.py:349`), in the **North
(wall)** block (`j = num_grid_y+1`, lines 574-609):
```
P_rhs = −[ (∇ξ·∇η) P_ξ + (∇ζ·∇η) P_ζ ] / (∇η·∇η)
P_g   = P_in + P_rhs·(η_ghost − η_in)          # curvilinear Neumann extrapolation of P
```
(and the analogous `T_rhs`/`T_g`). This sets the wall-ghost pressure/temperature so the
**wall-normal** derivative is consistent on the curved wall. It is a **boundary condition**
→ moved to **Task #10** (see that task's updated description).

### 1b. The actual pressure-transport RHS is **not curvilinear**
The pressure equation RHS (`sum_fluxes_source_terms`, `Python:2340-2364`, active only when
`transport_pressure_scheme=True`) is:
```
d_P_x = (P[i+1]−P[i-1])/(2 Δx);  d_P_y = …/(2 Δy);  d_P_z = …/(2 Δz)
P_inv_flux = u d_P_x + v d_P_y + w d_P_z + ρ a² (d_u_x + d_v_y + d_w_z)
```
with **no** `∇ξ/∇η/∇ζ` cross-metric terms. A curvilinear variant with `det_Jacobian`
scaling exists but is **commented out** (`Python:2415-2427`). So the reference's
pressure-transport path is effectively the plain (non-curvilinear) form.

Also, the **nozzle case runs `transport_pressure_scheme = False`** (total-energy), so this
path is never exercised by the target simulation.

---

## 2. Decision: guard `NOZZLE_CD + transport_pressure_scheme`

Because:
- the reference has no curvilinear pressure-transport RHS to faithfully port, and
- the curvilinear inviscid kernel (Task #6) intentionally does not fill `P_inv_flux`,

running `NOZZLE_CD` **with** `transport_pressure_scheme=True` would silently produce wrong
results. The safe, honest choice is to **reject that combination** with a clear error at
configuration time rather than ship a disabled/mismatched path.

**Implementation:** in `readConfigurationFile()`, after both are parsed:
```cpp
if( geometry_type == "NOZZLE_CD" && transport_pressure_scheme ) {
    cout << "NOZZLE_CD does not support transport_pressure_scheme (use total energy)." << endl;
    MPI_Abort( MPI_COMM_WORLD, 1 );
}
```

If a curvilinear pressure-transport scheme is needed later, it would first have to be
derived/enabled in the Python reference (the commented-out `det_Jacobian` variant) and then
ported — out of scope for this faithful port.

---

## 3. What this task changed

- **Code:** added the guard above; no numeric operator changed.
- **Plan:** the wall-BC cross-terms (588/592) are now part of **Task #10**.
- **No effect** on the total-energy nozzle case (the guarded path is unused there).

## 4. Acceptance criteria

- [x] `NOZZLE_CD + transport_pressure_scheme` is rejected at config time with a message.
- [x] Total-energy nozzle runs are unaffected (guard not triggered).
- [x] Wall-BC cross-terms explicitly reassigned to Task #10 (traceability preserved).
