# SIM-04 — FD · CO₂ · 150 → 130 bar · Peng-Robinson · Chung high-P transport

Finite-difference (curvilinear η=y/L_y) solver with real-gas EoS + high-pressure transport.
Fresh start (use_restart=False), 50×30×1, CFL=0.1. Headless; writes `output_data_<iter>.csv` and
`grid_*.png` here.

Run:  `pixi run python rhea_flow_solver_FD.py > run.log 2>&1 &`
Check convergence post-hoc (FD has no in-loop monitor):  `pixi run python convergence_check.py output_data_*.csv`

⚠ **The FD inlet imposes a fixed velocity (u_inlet=10 m/s), so the nozzle will NOT choke** — this is a
low-speed flow, not a choked nozzle, and not directly comparable to the choked FV runs. For a matched
comparison the FD inlet must be converted to a stagnation reservoir. See `../SIMULATION_PLAN_150_130.md`.
