# Emotional Phasor Model — Simulation Code (v1.1.0)

Code accompanying "Simulating Emotional Influence and Group Fragmentation in
Organized Crime: Insights from a Phase-Based Emotional Model" (SIMULATION).

Version 1.1.0 accompanies the first revision of the manuscript. The main
change with respect to v1.0.x is the boundary treatment of the radial state:
the discrete per-interval resets of the previous version are replaced by a
continuous, event-located inelastic contact law (Eq. (14) of the revised
paper), and the convergence of both treatments is documented (Appendix B).

## Contents
- `model.py` — core model (Eqs. (9)–(12) of the revised paper), Table-4
  parameters and emotional states, all coupling schedules (Examples 1–4), and
  three radial boundary treatments:
  - `integrate_contact` — **the integrator used for every figure and metric
    in the revised paper.** Continuous inelastic ("sticking") contact law at
    r = 0 and r = 1, applied at event-located contact times via terminal
    events of `solve_ivp`: an agent reaching a bound with outward velocity is
    stopped there, held while the unconstrained radial acceleration points
    outward, and released (with continuous position and velocity) as soon as
    it points inward. Independent of the output sampling.
  - `integrate_projected` — the discrete-reset scheme of the previous version
    (v1.0.x), retained for auditability: the unconstrained system is
    integrated between consecutive output times, after which the radial state
    is clipped to [0,1] and outward boundary velocity is zeroed. Its result
    depends on the reset interval (Appendix B of the paper).
  - `integrate_flawed` — the naive variant (clip applied to the RHS
    evaluation only), whose failure mode (spurious radial excursions of
    order ±10³ in the chain scenario) is documented in Sec. 4, Example 2.
- `run_scenarios.py` — regenerates the four scenario figures
  (`Ejemplo1–4.png`) and the coherence-index figure (`cohesion_index.png`),
  and writes the scenario metrics to `figs/scenario_results.json`.
  Usage: `python run_scenarios.py --boundary {contact,reset,rhsclip}
  [--npts N]` (default: `contact`, 1600 output samples).
- `run_sensitivity.py` — damping sweep, coupling-magnitude (K/b) sweep and
  initial-condition robustness study of Sec. 3.2 (`sensitivity_analysis.png`,
  `figs/sensitivity_results.json`). Seed 42, 1600 output samples.
- `run_baseline.py` — matched first-order Kuramoto baseline comparison of
  Sec. 2.6 (`baseline_kuramoto.png`, `figs/baseline_results.json`): inertial
  model vs. matched first-order model (b·dθ/dt = f) vs. unit-rate first-order
  model (dθ/dt = f); unwrapped-phase overshoot and late-time amplitudes.
- `run_convergence.py` — Appendix B study (`convergence_appendix.png`,
  `figs/table7.json`): unconstrained-chain amplification, interval
  sensitivity of the discrete-reset scheme at 1600/3199/6397 output points
  (the grids of the referee's independent check), deviation of each reset
  solution from the contact-law solution, tolerance convergence of the
  contact law (1e-8 vs 1e-10), and solver cross-checks (RK45 vs Radau, with
  the contact law active, on sub-example 1.3 and Example 4 at b = 0.1).
- `regen_all.sh` — runs all of the above in sequence.

## Environment
Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2 — pinned in
`requirements.txt` and `environment.yml`. Integrator: scipy `solve_ivp`,
RK45, rtol = atol = 1e-8, with terminal events for the contact law; Radau and
RK45 at 1e-10 are used in the cross-checks only. No BDF integrator is used.

## Reproducibility
All randomness (initial-condition robustness study) uses a fixed seed
(`numpy.random.default_rng(42)`). Every figure and every number quoted in the
revised paper regenerates from these scripts; the mapping from scripts to
figures and tables is listed in Appendix C of the paper.

Note on external forcing: the exogenous terms F_i(t) and tau_i(t) of the
paper's Eqs. (9)–(12) are identically ZERO in all simulations (they do not
appear in the RHS in `model.py`); all exogenous influence enters through the
time-dependent coupling schedules. They are retained in the paper's
formulation as the natural entry point for modeling non-social external
events in future work.
