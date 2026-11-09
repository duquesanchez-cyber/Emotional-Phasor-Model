"""Sensitivity and robustness analysis (Figure 9 of the revised paper).

Three analyses, all integrated with the continuous contact law of Eq. (14)
at the paper's numerical settings (1600 output samples, rtol = atol = 1e-8):

  (a) damping sweep under the abrupt-onset schedule of sub-example 1.3
      (K = 500 switched on at t = 20): settling time of the angular
      alignment, measured from the coupling onset, and peak radial excursion;
  (b) coupling-magnitude sweep at fixed b = 0.5: final alignment with the
      leader's target (A_L) versus with each agent's own baseline (A_B), as
      functions of K/b;
  (c) initial-condition robustness (sub-example 1.2 parameters): 20 runs with
      the Table-4 seeds perturbed by uniform noise of amplitude +-0.05 in r
      and +-0.1 rad in theta (fixed NumPy seed 42), contrasting perturbation
      amplitude with the dispersion of the final state.

Outputs: figs/sensitivity_analysis.png and figs/sensitivity_results.json.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m
import json, os, time

OUT = "figs"
os.makedirs(OUT, exist_ok=True)
TH_L = m.EMO["ANGRY"][1]
R_L = m.EMO["ANGRY"][0]
SEED = 42
HORIZON = 60.0                      # max observable settling time after onset

def wrapd(a):
    return np.abs(m.wrap(a))

def contact(sched, b, states, **kw):
    r0, th0, rs, ts = states
    t, r, th, _ = m.integrate_contact(sched, b=b, r0=r0, th0=th0, r_star=rs,
                                      th_star=ts, return_events=True, **kw)
    return t, r, th

# ------------------------------------------------------------------ sweeps
def damping_sweep():
    print("damping sweep...", flush=True)
    bs = np.logspace(-2, 2, 17)
    settle, exc = [], []
    for b in bs:
        t0 = time.time()
        t, r, th = contact(m.sched_ex1("1.3"), b, m.states_ex1("1.3"))
        err = wrapd(th - TH_L).max(axis=0)
        idx = np.where(t >= 20)[0]
        st = np.nan
        for k in idx:
            if np.all(err[k:] < 0.1):
                st = t[k] - 20.0
                break
        settle.append(st)
        exc.append(float(np.max(np.abs(r - R_L)[:, t >= 20])))
        print(f"  b={b:8.3f} settle={st} exc={exc[-1]:.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return bs, np.array(settle), np.array(exc)

def K_sweep():
    print("K/b sweep (b = 0.5 fixed)...", flush=True)
    Ks = np.logspace(-1, np.log10(500), 15)
    b = 0.5
    aL, aB = [], []
    states = m.states_ex1("1.3")
    for K in Ks:
        def sched(t, K=K):
            M = np.zeros((m.N, m.N))
            if t >= 20:
                M[:, 7] = K
            M[7, 7] = 0.0
            return M, M.copy()
        t, r, th = contact(sched, b, states)
        ts = states[3]
        aL.append(1 - wrapd(th[:, -1] - TH_L).mean() / np.pi)
        aB.append(1 - wrapd(th[:, -1] - ts).mean() / np.pi)
        print(f"  K={K:8.2f} K/b={K/b:8.1f} A_L={aL[-1]:.3f} A_B={aB[-1]:.3f}",
              flush=True)
    return Ks, b, np.array(aL), np.array(aB)

def ic_robustness():
    print("IC robustness (seed 42, 20 runs)...", flush=True)
    rng = np.random.default_rng(SEED)
    r0b, th0b, rs, ts = m.states_ex1("1.2")
    fr, fth = [], []
    for k in range(20):
        r0 = np.clip(r0b + rng.uniform(-0.05, 0.05, m.N), 0, 1)
        th0 = th0b + rng.uniform(-0.1, 0.1, m.N)
        t, r, th = contact(m.sched_ex1("1.2"), 0.5, (r0, th0, rs, ts))
        fr.append(r[:, -1]); fth.append(th[:, -1])
        print(f"  run {k+1}/20", flush=True)
    fr = np.array(fr); fth = np.array(fth)
    return float(fr.std(axis=0).max()), float(fth.std(axis=0).max())

# ------------------------------------------------------------------ figure
if __name__ == "__main__":
    res = {}
    bs, settle, exc = damping_sweep()
    Ks, b, aL, aB = K_sweep()
    sd_r, sd_th = ic_robustness()

    ok = ~np.isnan(settle)
    b_first = float(bs[ok].min()) if ok.any() else None     # regime boundary
    b_min = float(bs[ok][np.argmin(settle[ok])]) if ok.any() else None
    s_min = float(np.nanmin(settle)) if ok.any() else None
    res["damping"] = dict(
        first_settling_b=b_first, settle_min=s_min, settle_argmin_b=b_min,
        settle_at_first=float(settle[ok][0]) if ok.any() else None,
        excursion_range=[float(exc.min()), float(exc.max())],
        table={f"{b_:.4g}": (None if np.isnan(s) else float(s))
               for b_, s in zip(bs, settle)})

    kb = Ks / b
    cross = None
    for i in range(1, len(kb)):
        if (aL[i] - aB[i]) > 0 >= (aL[i-1] - aB[i-1]):
            f = (0 - (aL[i-1]-aB[i-1])) / ((aL[i]-aB[i]) - (aL[i-1]-aB[i-1]))
            cross = float(np.exp(np.log(kb[i-1]) + f*(np.log(kb[i])-np.log(kb[i-1]))))
            break
    def first_kb(level):
        sel = kb[aL >= level]
        return float(sel.min()) if len(sel) else None
    res["coupling"] = dict(crossover_Kb=cross, Kb_A90=first_kb(0.9),
                           Kb_A95=first_kb(0.95), Kb_A99=first_kb(0.99),
                           A_L_final=float(aL[-1]))
    res["ic"] = dict(seed=SEED, amp_r=0.05, amp_th=0.1,
                     max_std_r=sd_r, max_std_th=sd_th)

    fig, axs = plt.subplots(1, 3, figsize=(19, 5.0))

    # (a) damping sweep with regime bands
    ax = axs[0]
    if b_first is not None:
        ax.axvspan(bs.min()*0.8, b_first*0.82, color="red", alpha=0.08)
        ax.axvspan(b_first*0.82, 30, color="green", alpha=0.07)
    ax.axvspan(30, bs.max()*1.2, color="orange", alpha=0.10)
    ax.axhline(HORIZON, ls="--", color="gray", lw=1)
    ax.loglog(bs[ok], settle[ok], "o-", lw=2, color="tab:blue",
              label="settling time")
    if (~ok).any():
        ax.plot(bs[~ok], np.full((~ok).sum(), HORIZON), "rx", ms=9, mew=2,
                label="not settled within horizon")
    ax.text(0.02, HORIZON*1.12, "simulation horizon: 60 time units after "
            "onset (t = 80)", fontsize=7.5, color="gray")
    ax.set_xlabel(r"$b_r=b_\theta$")
    ax.set_ylabel("settling time after coupling onset", color="tab:blue")
    ax.set_title("(a) Damping sweep (abrupt coupling, K=500)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax2 = ax.twinx()
    ax2.semilogx(bs, exc, "s--", color="tab:orange", lw=1.5)
    ax2.set_ylabel(r"peak radial excursion $\max|r_i-r_L|$",
                   color="tab:orange")

    # (b) coupling sweep with crossover
    ax = axs[1]
    ax.semilogx(kb, aL, "o-", lw=2, label=r"alignment with leader $A_L$")
    ax.semilogx(kb, aB, "s--", lw=2, color="tab:orange",
                label=r"alignment with own baseline $A_B$")
    if cross:
        ax.axvline(cross, ls=":", color="k", lw=1.2)
        ax.annotate(f"crossover\nK/b ≈ {cross:.1f}", xy=(cross, 0.72),
                    fontsize=8, ha="left")
    ax.set_xlabel(r"$K/b$ (fixed $b=0.5$)")
    ax.set_ylabel("alignment index")
    ax.set_title("(b) Coupling-magnitude sweep", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    # (c) IC robustness: input amplitude vs output dispersion
    ax = axs[2]
    x = np.arange(2); w = 0.38
    ax.bar(x - w/2, [0.05, 0.1], w, color="#9db8d9",
           label="input perturbation amplitude")
    ax.bar(x + w/2, [sd_r, sd_th], w, color="#1f4e79",
           label="max std of final state (20 runs)")
    ax.set_yscale("log")
    ax.set_xticks(x, [r"radial $r_i$", r"angular $\theta_i$ [rad]"])
    ax.set_title("(c) Initial-condition robustness", fontweight="bold",
                 fontsize=11)
    orders = np.log10(0.05/sd_r) if sd_r > 0 else np.inf
    ax.annotate(f"contraction of ≈{orders:.0f} orders\nof magnitude; all runs"
                "\ncollapse onto the same attractor",
                xy=(0.5, 0.45), xycoords="axes fraction", fontsize=8,
                ha="center")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT}/sensitivity_analysis.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    json.dump(res, open(f"{OUT}/sensitivity_results.json", "w"), indent=1)
    print("saved", f"{OUT}/sensitivity_analysis.png", flush=True)
    print("DONE", flush=True)
