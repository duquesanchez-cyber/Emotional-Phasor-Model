"""Matched first-order baseline comparison (Figure 6 of the revised paper).

Regenerates the three-panel figure under the sub-example 1.1 leadership
schedule and parameters (b_theta = 0.01):

  (a) second-order (inertial) angular model of Eq. (11);
  (b) matched first-order model b_theta * dtheta/dt = f  (inertia removed,
      damping retained -- the appropriate baseline);
  (c) unit-rate first-order model dtheta/dt = f (previous version's
      comparison, retained for completeness).

Prints the maximum unwrapped-phase overshoot past the leader's target and the
late-time (t in [60, 80]) oscillation amplitudes, and stores them in
figs/baseline_results.json.

Only the angular subsystem matters for this comparison; the radial subsystem
is integrated alongside with the contact law, as in the paper.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import model as m
import json, os, time

OUT = "figs"
os.makedirs(OUT, exist_ok=True)

B = 0.01
TH_L = m.EMO["ANGRY"][1]
LEADER = 7

def unwrap_overshoot(t, th):
    """Max signed excursion of unwrap(th_i) - TH_L past the target, taken
    along each follower's direction of approach after its first crossing."""
    over = 0.0
    for i in range(m.N):
        if i == LEADER:
            continue
        u = np.unwrap(th[i])
        # direction of approach: sign of (target - initial), in unwrapped terms
        tgt = u[0] + m.wrap(TH_L - th[i, 0])
        d = np.sign(tgt - u[0])
        if d == 0:
            continue
        exc = d * (u - tgt)          # >0 once past the target along approach
        k = np.argmax(exc > 0) if np.any(exc > 0) else None
        if k is not None:
            over = max(over, float(exc[k:].max()))
    return over

def late_amplitude(t, th):
    sel = t >= 60.0
    u = np.array([np.unwrap(th[i]) for i in range(m.N)])
    return float((u[:, sel].max(axis=1) - u[:, sel].min(axis=1)).max())

def run(mode, label):
    r0, th0, rs, ts = m.states_ex1("1.1")
    t0 = time.time()
    t, r, th, nev = m.integrate_contact(m.sched_ex1("1.1"), b=B, r0=r0, th0=th0,
                                        r_star=rs, th_star=ts,
                                        first_order_theta=mode,
                                        return_events=True)
    ov = unwrap_overshoot(t, th)
    amp = late_amplitude(t, th)
    print(f"  {label}: overshoot={ov:.3f} rad, late amplitude={amp:.2e} rad, "
          f"contact events={nev} [{time.time()-t0:.1f}s]", flush=True)
    return t, th, ov, amp

if __name__ == "__main__":
    res = {}
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.6), sharey=True)
    for ax, (mode, label, title) in zip(axs, [
            (False, "inertial", r"(a) inertial: $\ddot\theta+b_\theta\dot\theta=f$"),
            ("matched", "matched first order", r"(b) matched first order: $b_\theta\dot\theta=f$"),
            ("unit", "unit-rate first order", r"(c) unit-rate first order: $\dot\theta=f$")]):
        t, th, ov, amp = run(mode, label)
        for i in range(m.N):
            ax.plot(t, th[i], lw=0.8)
        ax.axhline(TH_L, color="k", lw=1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time (t)")
        ax.grid(True)
        res[label] = dict(overshoot_rad=ov, late_amplitude_rad=amp)
    axs[0].set_ylabel(r"$\theta_i(t)$ (wrapped)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/baseline_kuramoto.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    json.dump(res, open(f"{OUT}/baseline_results.json", "w"), indent=1)
    print("saved", f"{OUT}/baseline_kuramoto.png", flush=True)
    print("DONE", flush=True)
