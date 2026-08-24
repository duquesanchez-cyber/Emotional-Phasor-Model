"""Sensitivity sweeps, first-order Kuramoto baseline, convergence check,
and cohesion index C(t) evaluation."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m
import json, time, os, sys

OUT = "figs"  # output directory (created if absent)
os.makedirs(OUT, exist_ok=True)
T_SW = np.linspace(0, 80, 800)          # coarser eval grid for sweeps (faster)
TH_L = m.EMO["ANGRY"][1]                # leader angular target 1.71042
R_L  = m.EMO["ANGRY"][0]

def wrapd(a): return np.abs(m.wrap(a))

res = {}

# ---------------------------------------------------------------- 1) damping sweep
def damping_sweep():
    print("damping sweep...", flush=True)
    bs = np.logspace(-2, 2, 17)
    settle, overshoot = [], []
    r0, th0, rs, ts = m.states_ex1("1.3")   # abrupt coupling at t=20
    for b in bs:
        t0=time.time()
        t, r, th = m.integrate_projected(m.sched_ex1("1.3"), b=b, r0=r0, th0=th0,
                                         r_star=rs, th_star=ts, t_eval=T_SW)
        err = wrapd(th - TH_L).max(axis=0)          # max angular error over agents
        # settling: first time after step such that error stays < 0.1 rad
        idx = np.where(t >= 20)[0]
        st = np.nan
        for k in idx:
            if np.all(err[k:] < 0.1):
                st = t[k] - 20.0
                break
        settle.append(st)
        overshoot.append(float(np.max(np.abs(r - R_L)[:, t >= 20])))
        print(f"  b={b:8.3f} settle={st} over={overshoot[-1]:.3f} ({time.time()-t0:.1f}s)", flush=True)
    return bs, np.array(settle), np.array(overshoot)

# ---------------------------------------------------------------- 2) K/b sweep
def K_sweep():
    print("K/b sweep...", flush=True)
    Ks = np.logspace(-1, np.log10(500), 15)
    b = 0.5
    align_L, align_B = [], []
    r0, th0, rs, ts = m.states_ex1("1.3")
    for K in Ks:
        def sched(t, K=K):
            M = np.zeros((m.N, m.N))
            if t >= 20: M[:, 7] = K
            M[7,7]=0.0
            return M, M.copy()
        t0=time.time()
        t, r, th = m.integrate_projected(sched, b=b, r0=r0, th0=th0,
                                         r_star=rs, th_star=ts, t_eval=T_SW)
        dL = wrapd(th[:, -1] - TH_L).mean()
        dB = wrapd(th[:, -1] - ts).mean()
        align_L.append(1 - dL/np.pi); align_B.append(1 - dB/np.pi)
        print(f"  K={K:8.2f} K/b={K/b:8.1f} A_leader={align_L[-1]:.3f} A_base={align_B[-1]:.3f} ({time.time()-t0:.1f}s)", flush=True)
    return Ks, b, np.array(align_L), np.array(align_B)

# ---------------------------------------------------------------- 3) IC robustness
def ic_robustness():
    print("IC robustness...", flush=True)
    rng = np.random.default_rng(42)
    r0b, th0b, rs, ts = m.states_ex1("1.2")
    finals_r, finals_th = [], []
    for k in range(20):
        r0 = np.clip(r0b + rng.uniform(-0.05, 0.05, m.N), 0, 1)
        th0 = th0b + rng.uniform(-0.1, 0.1, m.N)
        t, r, th = m.integrate_projected(m.sched_ex1("1.2"), b=0.5, r0=r0, th0=th0,
                                         r_star=rs, th_star=ts, t_eval=T_SW)
        finals_r.append(r[:, -1]); finals_th.append(th[:, -1])
        print(f"  run {k+1}/20", flush=True)
    fr = np.array(finals_r); fth = np.array(finals_th)
    return fr.std(axis=0).max(), fth.std(axis=0).max(), fr, fth

# ---------------------------------------------------------------- 4) baseline Kuramoto
def baseline():
    print("baseline 1st vs 2nd order...", flush=True)
    r0, th0, rs, ts = m.states_ex1("1.1")
    t2, r2, th2 = m.integrate_projected(m.sched_ex1("1.1"), b=0.01, r0=r0, th0=th0,
                                        r_star=rs, th_star=ts)
    t1, r1, th1 = m.integrate_projected(m.sched_ex1("1.1"), b=0.01, r0=r0, th0=th0,
                                        r_star=rs, th_star=ts, first_order_theta=True)
    fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for i in range(m.N):
        ls = '-' if i < m.N//2 else '--'
        axs[0].plot(t2, th2[i], ls, lw=1.2)
        axs[1].plot(t1, th1[i], ls, lw=1.2)
    axs[0].axhline(TH_L, color='k', lw=0.8, alpha=0.5)
    axs[1].axhline(TH_L, color='k', lw=0.8, alpha=0.5)
    axs[0].set_title("Second-order (inertial) angular model — Eq. (23)", fontweight="bold")
    axs[1].set_title("First-order Kuramoto baseline (no inertia)", fontweight="bold")
    for a in axs:
        a.set_xlabel("Time (t)"); a.grid(True)
    axs[0].set_ylabel("θ(t)")
    plt.tight_layout(); plt.savefig(f"{OUT}/baseline_kuramoto.png", dpi=120, bbox_inches='tight')
    plt.close(fig)
    # ringing metric: number of sign changes of dθ/dt after coupling onset (agent-avg)
    def ring(th, t):
        seg = th[:, t > 25]
        d = np.diff(seg, axis=1)
        return float(np.mean(np.sum(np.abs(np.diff(np.sign(d), axis=1)) > 0, axis=1)))
    return ring(th2, t2), ring(th1, t1)

# ---------------------------------------------------------------- 5) convergence
def convergence():
    print("convergence check...", flush=True)
    out = {}
    for tag, sched, b, states in [
        ("ex1.3", m.sched_ex1("1.3"), 0.5, m.states_ex1("1.3")),
        ("ex4_b0.1", m.sched_ex4(), 0.1, m.states_ex4()),
    ]:
        r0, th0, rs, ts = states
        runs = {}
        for name, method, tol in [("RK45_1e-8","RK45",1e-8),
                                   ("Radau_1e-8","Radau",1e-8),
                                   ("RK45_1e-10","RK45",1e-10)]:
            t0=time.time()
            t, r, th = m.integrate_projected(sched, b=b, r0=r0, th0=th0, r_star=rs,
                                             th_star=ts, method=method, rtol=tol, atol=tol,
                                             t_eval=T_SW)
            runs[name] = (r, th)
            print(f"  {tag} {name}: {time.time()-t0:.1f}s", flush=True)
        rA, thA = runs["RK45_1e-8"]
        for other in ["Radau_1e-8", "RK45_1e-10"]:
            rB, thB = runs[other]
            out[f"{tag}_vs_{other}_dr"] = float(np.max(np.abs(rA - rB)))
            out[f"{tag}_vs_{other}_dth"] = float(np.max(np.abs(m.wrap(thA - thB))))
    return out

# ---------------------------------------------------------------- 6) cohesion C(t)
def cohesion_fig():
    print("cohesion C(t)...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    Cs = {}
    labels = {"3.1":"Sub-ex. 3.1 (succession)", "3.2":"Sub-ex. 3.2 (bifurcation)",
              "3.3":"Sub-ex. 3.3 (leaderless collapse)"}
    for sub in ["3.1","3.2","3.3"]:
        d = np.load(f"{OUT}/ex{sub}.npz")
        C = m.cohesion(d["r"], d["th"])
        Cs[sub] = (d["t"], C)
        ax.plot(d["t"], C, lw=2, label=labels[sub])
    # threshold: midpoint between cohesive steady C (3.1, t in [70,80]) and
    # fractured steady C (min of 3.2/3.3 over t in [70,80])
    c_coh = float(np.mean(Cs["3.1"][1][Cs["3.1"][0] >= 70]))
    c_fr = float(min(np.mean(Cs["3.2"][1][Cs["3.2"][0] >= 70]),
                     np.mean(Cs["3.3"][1][Cs["3.3"][0] >= 70])))
    c_th = 0.5*(c_coh + c_fr)
    ax.axhline(c_th, color='r', ls='--', lw=1.5, label=f"illustrative $C_{{frac}}$ = {c_th:.2f}")
    ax.set_xlabel("Time (t)"); ax.set_ylabel("C(t)")
    ax.set_title("Cohesion index C(t) — Example 3 scenarios", fontweight="bold")
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig(f"{OUT}/cohesion_index.png", dpi=120, bbox_inches='tight')
    plt.close(fig)
    return c_coh, c_fr, c_th

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all","sweeps"):
        bs, settle, over = damping_sweep()
        Ks, b, aL, aB = K_sweep()
        sd_r, sd_th, _, _ = ic_robustness()
        res["ic_max_std_r"] = float(sd_r); res["ic_max_std_th"] = float(sd_th)
        # figure with three panels
        fig, axs = plt.subplots(1, 3, figsize=(19, 4.8))
        ok = ~np.isnan(settle)
        axs[0].loglog(bs[ok], settle[ok], 'o-', lw=2)
        if (~ok).any():
            axs[0].plot(bs[~ok], np.full((~ok).sum(), np.nanmax(settle)*1.1), 'rx', ms=8,
                        label="not settled by t=80")
            axs[0].legend()
        axs[0].set_xlabel("$b_r=b_θ$"); axs[0].set_ylabel("settling time (t)")
        axs[0].set_title("(a) Damping sweep: settling time", fontweight="bold")
        ax0b = axs[0].twinx()
        ax0b.semilogx(bs, over, 's--', color='tab:orange', lw=1.5)
        ax0b.set_ylabel("peak radial excursion $\\max|r-r_L|$", color='tab:orange')

        axs[1].semilogx(Ks/b, aL, 'o-', lw=2, label="alignment with leader $A_L$")
        axs[1].semilogx(Ks/b, aB, 's--', lw=2, label="alignment with own baseline $A_B$")
        axs[1].axhline(0.9, color='gray', ls=':', lw=1)
        axs[1].set_xlabel("$K/b$"); axs[1].set_ylabel("alignment index")
        axs[1].set_title("(b) Coupling-magnitude sweep ($b$=0.5)", fontweight="bold")
        axs[1].legend(); axs[1].grid(True, which='both', alpha=0.3)

        axs[2].bar(["max std $r_i(80)$", "max std $θ_i(80)$ [rad]"], [res["ic_max_std_r"], res["ic_max_std_th"]],
                   color=['tab:blue','tab:green'])
        axs[2].set_title("(c) Initial-condition robustness (20 runs)", fontweight="bold")
        axs[2].grid(True, axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{OUT}/sensitivity_analysis.png", dpi=120, bbox_inches='tight')
        plt.close(fig)
        # store threshold: smallest K/b with A_L >= 0.9
        kb = Ks/b
        above = kb[aL >= 0.9]
        res["kb_threshold_A90"] = float(above.min()) if len(above) else None
        res["damping_settle"] = {f"{b_:.4g}": (None if np.isnan(s) else float(s))
                                  for b_, s in zip(bs, settle)}
        res["damping_overshoot"] = {f"{b_:.4g}": float(o) for b_, o in zip(bs, over)}

    if which in ("all","baseline"):
        ring2, ring1 = baseline()
        res["ring_2nd_order"] = ring2; res["ring_1st_order"] = ring1

    if which in ("all","conv"):
        res.update(convergence())

    if which in ("all","cohesion"):
        c_coh, c_fr, c_th = cohesion_fig()
        res["C_cohesive"] = c_coh; res["C_fractured"] = c_fr; res["C_frac_threshold"] = c_th

    # merge with existing
    p = f"{OUT}/analysis_results.json"
    old = {}
    if os.path.exists(p):
        old = json.load(open(p))
    old.update(res)
    json.dump(old, open(p, "w"), indent=1)
    print("DONE", flush=True)
