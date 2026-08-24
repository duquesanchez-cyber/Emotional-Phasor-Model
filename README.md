# Emotional Phasor Model — Simulation Code

Code accompanying "Simulating Emotional Influence and Group Fragmentation in
Organized Crime: Insights from a Phase-Based Emotional Model" (SIMULATION).

## Contents
- `model.py` — core model (Eqs. 20–23), Table-4 parameters, all coupling
  schedules (Examples 1–4), the projected integrator used for all results in
  the paper (`integrate_projected`) and, for auditability, the naive variant
  (`integrate_flawed`, clip applied to the RHS evaluation only) whose failure
  mode is documented in the implementation note of the paper (Sec. 4,
  Example 2).
- `run_scenarios.py` — regenerates the four scenario figures
  (Ejemplo1–4.png). Usage: `python run_scenarios.py [ex1|ex2|ex3|ex4|all]`
- `run_analysis.py` — sensitivity sweeps (damping, K/b, initial conditions),
  first-order Kuramoto baseline comparison, solver convergence check, and the
  cohesion index C(t) evaluation. Usage:
  `python run_analysis.py [sweeps|baseline|conv|cohesion|all]`

## Reproducibility
All randomness (initial-condition robustness study) uses a fixed seed
(`numpy.random.default_rng(42)`). Integrator: scipy `solve_ivp`, RK45,
rtol=atol=1e-8; Radau used in the convergence check. The r∈[0,1] constraint is
enforced by projected integration: after every sampling step the radial state
is clipped and outward boundary velocity zeroed.

Note on external forcing: the exogenous terms F_i(t) and tau_i(t) of Eqs.
(20)-(23) of the paper are identically ZERO in all simulations (they do not
appear in the RHS in `model.py`); all exogenous influence enters through the
time-dependent coupling schedules. They are retained in the paper's formulation
as the entry point for modeling individual-level external events in future work.

To reproduce the failure mode of the naive clipping scheme documented in the
paper (Sec. 4, Example 2), call `model.integrate_flawed` with the chain
schedule (`sched_chain(500.0)`) and b=0.5: r reaches [-1083, +1434], whereas
`integrate_projected` keeps r in [0,1] identically.

## License
MIT (or as chosen by the authors before deposit).
