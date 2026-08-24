"""Regenerate the four scenario figures (Ejemplo1-4.png) with the corrected
projected integrator, in the original paper style."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m
import json, time, os

OUT = "figs"  # output directory (created if absent)
os.makedirs(OUT, exist_ok=True)

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
    t, r, th = m.integrate_projected(sched, b=b, r0=r0, th0=th0, r_star=rs, th_star=ts)
    dt = time.time()-t0
    print(f"  {tag}: {dt:.1f}s r=[{r.min():.2f},{r.max():.2f}]", flush=True)
    return {"t": t, "r": r, "th": th}

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

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

    with open(f"{OUT}/scenario_results.json","w") as f:
        json.dump(results, f, indent=1)
    print("DONE", flush=True)
