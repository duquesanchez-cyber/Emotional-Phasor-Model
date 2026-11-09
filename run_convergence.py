"""Boundary-treatment study of Appendix B (Table 7 and Figure 15).

Chain-of-command scenario of Example 2 (Table 5 initial conditions, N = 20,
K = 500, rtol = atol = 1e-8, t in [0, 80]). Reproduces, from the archived
code itself:

  1. the geometric amplification of the unconstrained radial chain;
  2. the interval sensitivity of the discrete-reset scheme at 1600, 3199 and
     6397 output points (successive halvings of the archived sampling grid --
     the grids used in Referee 3's independent check);
  3. the deviation of each reset solution from the event-located contact-law
     solution of Eq. (14), which halves with each halving of the reset
     interval (first-order convergence);
  4. the tolerance convergence of the contact law (rtol = atol tightened from
     1e-8 to 1e-10).

Outputs: figs/table7.json and figs/convergence_appendix.png.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m
import json, os, time

OUT = "figs"
os.makedirs(OUT, exist_ok=True)

GRIDS = (1600, 3199, 6397)
BS = (0.5, 5.0, 50.0)

def chain_args():
    r0, th0, rs, ts = m.states_ex2()
    return m.sched_chain(500.0), r0, th0, rs, ts

def unconstrained(b):
    from scipy.integrate import solve_ivp
    Kf, r0, th0, rs, ts = chain_args()
    f = m.rhs_factory(Kf, np.full(m.N, b), np.full(m.N, b), rs, ts)
    y0 = np.concatenate([r0, np.zeros(m.N), th0, np.zeros(m.N)])
    sol = solve_ivp(f, (0, 80), y0, t_eval=m.T_EVAL, method="RK45",
                    rtol=1e-8, atol=1e-8)
    return np.abs(sol.y[:m.N]).max(axis=1)

if __name__ == "__main__":
    res = {}

    print("1) unconstrained amplification", flush=True)
    for b in (0.5, 5.0):
        p = unconstrained(b)
        fac = p[2:] / p[1:-1]
        res[f"unconstrained_b{b}"] = dict(
            peaks_first6=[float(x) for x in p[:6]],
            last=float(p[-1]), median_factor=float(np.median(fac)))
        print(f"   b={b}: last follower peak |r| = {p[-1]:.3e}, "
              f"median factor/link = {np.median(fac):.1f}", flush=True)

    print("2-3) reset halvings and deviation from the contact law", flush=True)
    fig15a = {}; fig15b = {}
    for b in BS:
        Kf, r0, th0, rs, ts = chain_args()
        t0 = time.time()
        _, rC, thC, nev = m.integrate_contact(Kf, b=b, r0=r0, th0=th0,
                                              r_star=rs, th_star=ts,
                                              return_events=True)
        print(f"   b={b}: contact events = {nev} [{time.time()-t0:.1f}s]",
              flush=True)
        resets = {}
        for npts in GRIDS:
            te = np.linspace(0.0, 80.0, npts)
            _, rr, _ = m.integrate_projected(Kf, b=b, r0=r0, th0=th0,
                                             r_star=rs, th_star=ts, t_eval=te)
            resets[npts] = rr
        step = {1600: 1, 3199: 2, 6397: 4}
        row = dict(contact_events=int(nev))
        d12 = float(np.abs(resets[1600] - resets[3199][:, ::2]).max())
        d23 = float(np.abs(resets[3199] - resets[6397][:, ::2]).max())
        row["halving_1600_vs_3199"] = d12
        row["halving_3199_vs_6397"] = d23
        for npts in GRIDS:
            rr = resets[npts][:, ::step[npts]]
            row[f"reset{npts}_vs_contact_max"] = float(np.abs(rr - rC).max())
            row[f"reset{npts}_vs_contact_rms"] = \
                float(np.sqrt(((rr - rC) ** 2).mean()))
        res[f"b{b}"] = row
        fig15a[b] = [row[f"reset{n}_vs_contact_max"] for n in GRIDS]
        fig15b[b] = [d12, d23]
        print(f"   b={b}: halvings {d12:.3f}/{d23:.3f}; "
              f"reset-vs-contact {fig15a[b][0]:.3g}/{fig15a[b][1]:.3g}/"
              f"{fig15a[b][2]:.3g}", flush=True)
        if b == 0.5:
            np.save(f"{OUT}/chain_contact_b0.5.npy", rC)

    print("4) contact-law tolerance convergence", flush=True)
    for b in (0.5, 5.0):
        Kf, r0, th0, rs, ts = chain_args()
        _, r8, _, _ = m.integrate_contact(Kf, b=b, r0=r0, th0=th0,
                                          r_star=rs, th_star=ts,
                                          return_events=True)
        t0 = time.time()
        _, r10, _, nev10 = m.integrate_contact(Kf, b=b, r0=r0, th0=th0,
                                               r_star=rs, th_star=ts,
                                               rtol=1e-10, atol=1e-10,
                                               return_events=True)
        d = float(np.abs(r8 - r10).max())
        rms = float(np.sqrt(((r8 - r10) ** 2).mean()))
        res[f"tol_b{b}"] = dict(max=d, rms=rms, events_1e10=int(nev10))
        print(f"   b={b}: tol 1e-8 vs 1e-10: max = {d:.4f}, rms = {rms:.2e} "
              f"[{time.time()-t0:.1f}s]", flush=True)

    print("5) solver cross-check on the two most rapidly switched scenarios",
          flush=True)
    for tag, sched, bb, states in [
            ("ex1.3", m.sched_ex1("1.3"), 0.5, m.states_ex1("1.3")),
            ("ex4_b0.1", m.sched_ex4(), 0.1, m.states_ex4())]:
        r0, th0, rs, ts = states
        runs = {}
        for name, method, tol in [("RK45_1e-8", "RK45", 1e-8),
                                  ("Radau_1e-8", "Radau", 1e-8),
                                  ("RK45_1e-10", "RK45", 1e-10)]:
            t0 = time.time()
            _, r, th, _ = m.integrate_contact(sched, b=bb, r0=r0, th0=th0,
                                              r_star=rs, th_star=ts,
                                              method=method, rtol=tol,
                                              atol=tol, return_events=True)
            runs[name] = (r, th)
            print(f"   {tag} {name} [{time.time()-t0:.1f}s]", flush=True)
        rA, thA = runs["RK45_1e-8"]
        for other in ("Radau_1e-8", "RK45_1e-10"):
            rB, thB = runs[other]
            res[f"solver_{tag}_vs_{other}"] = dict(
                dr=float(np.abs(rA - rB).max()),
                dth=float(np.abs(m.wrap(thA - thB)).max()))
            print(f"   {tag} vs {other}: dr={res[f'solver_{tag}_vs_{other}']['dr']:.2e} "
                  f"dth={res[f'solver_{tag}_vs_{other}']['dth']:.2e}", flush=True)

    # ---- Figure 15 ----
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    marks = {0.5: "o-", 5.0: "s-", 50.0: "^-"}
    for b in BS:
        axs[0].loglog(GRIDS, fig15a[b], marks[b], label=f"b = {b}")
        axs[1].loglog(GRIDS[:2], fig15b[b], marks[b], label=f"b = {b}")
    ref = fig15a[0.5][0] * GRIDS[0] / np.array(GRIDS)
    axs[0].loglog(GRIDS, ref, "k:", label="first order")
    axs[0].set_title("(a) Discrete-reset scheme vs. continuous contact law")
    axs[0].set_xlabel("number of reset intervals on [0, 80]")
    axs[0].set_ylabel(r"$\max_{i,t}\,|r_i^{\mathrm{reset}}-r_i^{\mathrm{contact}}|$")
    axs[1].set_title("(b) Self-convergence of the reset scheme")
    axs[1].set_xlabel("coarser of the two compared resolutions")
    axs[1].set_ylabel("max radial difference, successive halvings")
    for ax in axs:
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT}/convergence_appendix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    json.dump(res, open(f"{OUT}/table7.json", "w"), indent=1)
    print("saved", f"{OUT}/convergence_appendix.png", flush=True)
    print("DONE", flush=True)
