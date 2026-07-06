# Task #14 — East Outflow (NSCBC subsonic + supersonic switch)

**Goal:** port the nozzle **East (outlet)** boundary into `myRHEA::updateBoundaries()`.
Follow-up to Task #10.

Reference: `Python/rhea_flow_solver.py:450-541` (East block of `update_boundaries`).

---

## 1. Scheme

A global **Mach switch** selects the outflow treatment (Python 453-464): from a
representative east-boundary cell, `Ma_exit = |V|/a`.
- `Ma_exit < 1.2` → **subsonic NSCBC** characteristic outflow (relax toward `P_outlet`).
- otherwise → **supersonic** zeroth-order extrapolation (`ghost = inner`).

### Subsonic NSCBC (Python 465-504)
Using inner cells `i-1` (a) and `i-2` (b), physical-x spacings `Δ_g = x_{i-1}-x_i`,
`Δ_in = x_{i-2}-x_{i-1}`, and `K = 0.9 a (1-Ma²)/L_x`:
```
L1 = K (P_a − P_outlet)/(u_a − a)         # incoming, relaxed to P_outlet
L2 = a² ρ_x − P_x ;  L3 = v_x ;  L4 = w_x ;  L5 = P_x + ρ a u_x
dQ1 = (L2 + ½(L5+L1))/a²
dQ2 = (L5 − L1)/(2 ρ a) ;  dQ3 = L3 ;  dQ4 = L4 ;  dQ5 = ½(L5+L1)
ρ_g = ρ_a − Δ_g dQ1 ; u_g = u_a − Δ_g dQ2 ; v_g = v_a − Δ_g dQ3 ;
w_g = w_a − Δ_g dQ4 ; P_g = P_a − Δ_g dQ5
T_g = T(P_g, ρ_g)   (calculateTemperatureFromPressureDensityWithInitialGuess)
```
`x_a` (subscript `_x`) are backward differences over `Δ_in`. `ρ_g` feeds `T_g`; the ghost
is then finalized from `(u_g,v_g,w_g,P_g,T_g)` via the shared `setGhostState`.

## 2. C++ mapping

- `P_outlet = bocos_P[_EAST_]` (Task #15 sets it). `L_x` from the derived nozzle geometry.
- Implemented as an **East block inside `myRHEA::updateBoundaries()`**, overwriting the East
  ghost after the base generic pass. Host loop (GPU deferred to Task #11).
- **MPI note:** `Ma_exit` uses one representative cell (as in Python); a multi-rank
  reduction to make the switch globally consistent is deferred to Task #11.

## 3. Acceptance criteria

- [x] East block overwrites the outflow ghost for `NOZZLE_CD` (NSCBC / supersonic switch).
- [x] Standalone (ideal-gas thermo): subsonic NSCBC ghost `(u,v,w,P,T)_g` matches the actual
      Python East algorithm on 4 inner stencils ⇒ `max rel diff = 0.0` (bit-identical).
- [x] Supersonic branch = zeroth-order copy (trivial); non-nozzle cases unaffected.
