"""Regenerate the four scenario figures (Ejemplo1-4.png), the decomposed
cohesion figure (cohesion_index.png: C, mean intensity, mean resultant length)
and the scenario metrics, under a selectable radial boundary treatment:

    python run_scenarios.py --boundary contact        # paper results (Eq. 14)
    python run_scenarios.py --boundary reset --npts 1600   # previous version
    python run_scenarios.py --boundary rhsclip        # naive clip (spurious)

The contact law is the integrator used for every scenario figure in the
revised paper; the other two are retained for auditability (Appendix B)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m
import json, time, os, argparse

OUT = "figs"  # output directory (created if absent)
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("which", nargs="?", default="all",
                choices=["all", "ex1", "ex2", "ex3", "ex4", "fig13"])
ap.add_argument("--boundary", default="contact",
                choices=["contact", "reset", "rhsclip"])
ap.add_argument("--npts", type=int, default=1600,
                help="output samples; for --boundary reset this is also the "
                     "number of reset intervals (plus one)")
ARGS, _ = ap.parse_known_args()

def integrate(sched, b, r0, th0, rs, ts, t_eval=None):
    """Dispatch on the selected boundary treatment."""
    te = m.T_EVAL if t_eval is None else t_eval
    if ARGS.npts != 1600 and t_eval is None:
        te = np.linspace(0.0, 80.0, ARGS.npts)
    if ARGS.boundary == "contact":
        t, r, th, nev = m.integrate_contact(sched, b=b, r0=r0, th0=th0,
                                            r_star=rs, th_star=ts, t_eval=te,
                                            return_events=True)
        return t, r, th, nev
    if ARGS.boundary == "reset":
        t, r, th = m.integrate_projected(sched, b=b, r0=r0, th0=th0,
                                         r_star=rs, th_star=ts, t_eval=te)
        return t, r, th, None
    t, r, th = m.integrate_flawed(sched, b=b, r0=r0, th0=th0,
                                  r_star=rs, th_star=ts, t_eval=te)
    return t, r, th, None

def plot_rows(rows, fname, final_marker_each=None):
    """rows: list of dicts {t, r, th, label}; 3 cols per row, paper style."""
    n = len(rows)
    fig, axs = plt.subplots(n, 3, figsize=(20, 5*n))
    if n == 1: axs = axs[None, :]
    for k, row in enumerate(rows):
        t, r, th, lab = row["t"], row["r"], row["th"], row["label"]
        for i in range(m.N):
            ls = '-' if i < m.N//2 else '--'
            lw = 2.0 if i < m.N//2 else 0.9
            axs[k,0].plot(t, r[i], ls, lw=lw)
            axs[k,1].plot(t, th[i], ls, lw=lw)
            x = r[i]*np.cos(th[i]); y = r[i]*np.sin(th[i])
            axs[k,2].plot(x, y, ls, lw=lw)
            axs[k,2].plot(x[0], y[0], 'kx', ms=9, mew=2)
            axs[k,2].plot(x[-1], y[-1], 'ks', ms=8)
        axs[k,0].set_title(f"Radius r(t) - {lab}", fontweight="bold", fontsize=13)
        axs[k,0].set_xlabel("Time (t)"); axs[k,0].set_ylabel("r(t)"); axs[k,0].grid(True)
        axs[k,1].set_title(f"Angle θ(t) - {lab}", fontweight="bold", fontsize=13)
        axs[k,1].set_xlabel("Time (t)"); axs[k,1].set_ylabel("θ(t)"); axs[k,1].grid(True)
        axs[k,2].set_title(f"Trajectories in Complex Plane - {lab}", fontweight="bold", fontsize=13)
        axs[k,2].set_xlabel("Re(z)"); axs[k,2].set_ylabel("Im(z)"); axs[k,2].grid(True)
        axs[k,2].set_xlim(-1.1,1.1); axs[k,2].set_ylim(-1.1,1.1); axs[k,2].set_aspect('equal')
        # row letter
        axs[k,0].annotate(f"{chr(97+k)})", xy=(-0.18, -0.12), xycoords='axes fraction',
                          fontsize=22, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print("saved", fname, flush=True)

results = {}

def run(sched, b, states, tag):
    r0, th0, rs, ts = states
    t0 = time.time()
    t, r, th, nev = integrate(sched, b, r0, th0, rs, ts)
    dt = time.time()-t0
    ev = f" events={nev}" if nev is not None else ""
    print(f"  {tag}: {dt:.1f}s r=[{r.min():.2f},{r.max():.2f}]{ev}", flush=True)
    if nev is not None:
        results[f"{tag}_contact_events"] = int(nev)
    return {"t": t, "r": r, "th": th}

if __name__ == "__main__":
    import sys
    which = ARGS.which
    print(f"boundary = {ARGS.boundary}", flush=True)

    if which in ("all", "ex1"):
        print("Example 1 (5 sub-examples)...", flush=True)
        rows = []
        for sub, b in [("1.1",0.01),("1.2",0.5),("1.3",0.5),("1.4",0.01),("1.5",0.5)]:
            d = run(m.sched_ex1(sub), b, m.states_ex1(sub), f"ex{sub}")
            d["label"] = f"$b_r$={b}, $b_θ$={b}"
            rows.append(d)
        plot_rows(rows, f"{OUT}/Ejemplo1.png")

    if which in ("all", "ex2"):
        print("Example 2 (chain, corrected)...", flush=True)
        rows = []
        for b in [0.5, 5.0, 50.0]:
            d = run(m.sched_chain(500.0), b, m.states_ex2(), f"chain b={b}")
            d["label"] = f"$b_r$={b}, $b_θ$={b}"
            rows.append(d)
            results[f"ex2_b{b}_rmin"] = float(d["r"].min())
            results[f"ex2_b{b}_rmax"] = float(d["r"].max())
            results[f"ex2_b{b}_final_r_span"] = [float(d["r"][:, -1].min()),
                                                 float(d["r"][:, -1].max())]
            # finite-time oscillation amplitude A_i = max_t r_i - min_t r_i
            A = d["r"].max(axis=1) - d["r"].min(axis=1)
            results[f"ex2_b{b}_amp_max"] = float(A.max())
            # settling: first time after which every angular error < 0.1 rad
            errA = np.abs(m.wrap(d["th"] - m.EMO["ANGRY"][1])).max(axis=0)
            st = None
            for k in range(len(d["t"])):
                if np.all(errA[k:] < 0.1):
                    st = float(d["t"][k]); break
            results[f"ex2_b{b}_settle_t"] = st
        plot_rows(rows, f"{OUT}/Ejemplo2.png")

    if which in ("all", "ex3"):
        print("Example 3 (3 sub-scenarios, b=1.0)...", flush=True)
        rows = []
        for sub in ["3.1","3.2","3.3"]:
            d = run(m.sched_ex3(sub), 1.0, m.states_ex3(sub), f"ex{sub}")
            d["label"] = f"$b_r$=1.0, $b_θ$=1.0"
            rows.append(d)
            np.savez(f"{OUT}/ex{sub}.npz", t=d["t"], r=d["r"], th=d["th"])
        plot_rows(rows, f"{OUT}/Ejemplo3.png")

    if which in ("all", "ex4"):
        print("Example 4 (convergence to conflict)...", flush=True)
        rows = []
        for b in [0.1, 1.0, 10.0]:
            d = run(m.sched_ex4(), b, m.states_ex4(), f"ex4 b={b}")
            d["label"] = f"$b_r$={b}, $b_θ$={b}"
            rows.append(d)
        plot_rows(rows, f"{OUT}/Ejemplo4.png")

    if which in ("all", "ex3", "fig13"):
        print("Figure 13 (C, mean intensity, mean resultant length)...", flush=True)
        fig, axs = plt.subplots(1, 3, figsize=(18, 4.4))
        cols = {"3.1": "tab:blue", "3.2": "tab:orange", "3.3": "tab:green"}
        for sub in ["3.1", "3.2", "3.3"]:
            z = np.load(f"{OUT}/ex{sub}.npz")
            t, r, th = z["t"], z["r"], z["th"]
            rbar = r.mean(axis=0)
            R = np.abs(np.exp(1j*th).mean(axis=0))
            C = rbar * np.exp(R - 1.0)
            axs[0].plot(t, C, color=cols[sub], label=f"sub-example {sub}")
            axs[1].plot(t, rbar, color=cols[sub])
            axs[2].plot(t, R, color=cols[sub])
            results[f"fig13_{sub}_C80"] = float(C[-1])
            results[f"fig13_{sub}_rbar80"] = float(rbar[-1])
            results[f"fig13_{sub}_R80"] = float(R[-1])
        axs[0].axhline(0.68, ls="--", color="k", lw=1)
        axs[0].set_title(r"(a) $C(t)=\bar r\,e^{R-1}$"); axs[0].legend(fontsize=9)
        axs[1].set_title(r"(b) mean intensity $\bar r(t)$")
        axs[2].set_title(r"(c) mean resultant length $R(t)$")
        for ax in axs:
            ax.set_xlabel("Time (t)"); ax.set_ylim(-0.02, 1.05); ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUT}/cohesion_index.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("saved", f"{OUT}/cohesion_index.png", flush=True)

    with open(f"{OUT}/scenario_results.json","w") as f:
        json.dump(results, f, indent=1)
    print("DONE", flush=True)
