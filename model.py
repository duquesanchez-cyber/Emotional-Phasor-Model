"""
Core simulation module for the emotional phasor model (Eqs. 20-23 of the paper).

Three integration modes:
  - integrate_flawed():   naive implementation in which the clip of r to [0,1]
                          is applied to the RHS evaluation only; retained for
                          auditability. Under strong coupling it produces
                          unbounded radial excursions (r in [-1083, +1434] for
                          the chain scenario at b=0.5), as documented in the
                          implementation note of the paper (Sec. 4, Example 2).
  - integrate_projected(): corrected implementation. State r is projected onto
                          [0,1] at every step; outward radial velocity at the
                          boundary is zeroed. No clip inside the RHS.

All parameters follow Table 4 of the manuscript.
"""
import numpy as np
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------ constants
N = 20
T_SPAN = (0.0, 80.0)
N_EVAL = 1600
T_EVAL = np.linspace(*T_SPAN, N_EVAL)
RTOL = ATOL = 1e-8

# Emotion coordinates (Table 3), in the attractor-list order of Table 4:
# happy, delighted, excited, astonished, aroused, tense, alarmed, angry,
# annoyed, afraid, distressed, frustrated, miserable, sad, gloomy, depressed,
# bored, droopy, tired, sleepy
R0 = np.array([0.95, 0.945, 1.0, 0.98, 1.0, 0.85, 0.89, 0.79, 0.78, 0.885,
               0.890, 0.7, 0.925, 0.90, 0.99, 0.93, 0.855, 0.99, 1.0, 1.0])
TH0 = np.array([0.15708, 0.36652, 0.78540, 1.13446, 1.20428, 1.58825, 1.65806,
                1.71042, 2.14675, 2.02458, 2.47837, 2.54818, -2.98451, -2.68781,
                -2.63545, -2.61799, -1.96350, -1.90241, -1.57952, -1.56032])

# named emotions used to override leader states in Examples 1.4/1.5, 3, 4
EMO = {
    "HAPPY": (0.95, 0.15708), "ANGRY": (0.79, 1.71042), "CALM": (1.0, -0.72431),
    "SERENE": (0.97, -0.54105), "TIRED": (1.0, -1.57952),
    "FRUSTRATED": (0.70, 2.54818),
}

# ------------------------------------------------------------------ dynamics
def rhs_factory(K_of_t, b_r, b_theta, r_star, th_star, clip_in_rhs=False,
                first_order_theta=False):
    """Vectorized RHS. K_of_t(t) -> (K_r, K_theta), both fresh arrays.

    first_order_theta:
      False      -> second-order (inertial) angular subsystem (Eq. 11)
      "matched"  -> matched first-order baseline b_theta*dtheta/dt = f
                    (inertia removed, damping retained; Fig. 6b)
      True/"unit"-> unit-rate first-order baseline dtheta/dt = f
                    (previous version's comparison; Fig. 6c)
    """
    def rhs(t, y):
        r = y[:N]; v = y[N:2*N]; th = y[2*N:3*N]; om = y[3*N:]
        if clip_in_rhs:                      # original flawed behaviour
            r = np.clip(r, 0.0, 1.0)
        K_r, K_th = K_of_t(t)
        # radial diffusive coupling: sum_j K[i,j]*(r_j - r_i)
        crow = K_r.sum(axis=1)
        coup_r = K_r @ r - crow * r
        # angular sinusoidal coupling: sum_j K[i,j]*sin(th_j - th_i)
        S = np.sin(th[None, :] - th[:, None])
        coup_th = (K_th * S).sum(axis=1)
        drdt = v
        dvdt = -(r - r_star) - b_r * v + coup_r
        if first_order_theta:
            f_th = -np.sin(th - th_star) + coup_th
            if first_order_theta == "matched":
                dthdt = f_th / b_theta
            else:                              # True / "unit": previous baseline
                dthdt = f_th
            domdt = np.zeros(N)
        else:
            dthdt = om
            domdt = -np.sin(th - th_star) - b_theta * om + coup_th
        return np.concatenate([drdt, dvdt, dthdt, domdt])
    return rhs

def integrate_flawed(K_of_t, b, r0=None, th0=None, r_star=None, th_star=None,
                     method="RK45", rtol=RTOL, atol=ATOL, t_eval=T_EVAL):
    """Single solve_ivp call, clip inside RHS only (original implementation)."""
    r0 = R0 if r0 is None else r0; th0 = TH0 if th0 is None else th0
    r_star = r0.copy() if r_star is None else r_star
    th_star = th0.copy() if th_star is None else th_star
    b_r = np.full(N, b); b_th = np.full(N, b)
    y0 = np.concatenate([r0, np.zeros(N), th0, np.zeros(N)])
    f = rhs_factory(K_of_t, b_r, b_th, r_star, th_star, clip_in_rhs=True)
    sol = solve_ivp(f, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                    method=method, rtol=rtol, atol=atol)
    return sol.t, sol.y[:N], wrap(sol.y[2*N:3*N])

def integrate_projected(K_of_t, b, r0=None, th0=None, r_star=None, th_star=None,
                        method="RK45", rtol=RTOL, atol=ATOL, t_eval=T_EVAL,
                        first_order_theta=False):
    """Corrected: segment-wise integration; after each segment the radial state
    is projected onto [0,1] and outward boundary velocity is zeroed."""
    r0 = R0 if r0 is None else r0; th0 = TH0 if th0 is None else th0
    r_star = r0.copy() if r_star is None else r_star
    th_star = th0.copy() if th_star is None else th_star
    b_r = np.full(N, b); b_th = np.full(N, b)
    f = rhs_factory(K_of_t, b_r, b_th, r_star, th_star, clip_in_rhs=False,
                    first_order_theta=first_order_theta)
    y = np.concatenate([r0, np.zeros(N), th0, np.zeros(N)])
    out = np.empty((4*N, len(t_eval))); out[:, 0] = y
    for k in range(len(t_eval) - 1):
        sol = solve_ivp(f, (t_eval[k], t_eval[k+1]), y, method=method,
                        rtol=rtol, atol=atol, dense_output=False)
        y = sol.y[:, -1]
        # projection: clip r, zero outward velocity at the active boundary
        r = y[:N]; v = y[N:2*N]
        lo = r < 0.0; hi = r > 1.0
        r[lo] = 0.0; r[hi] = 1.0
        v[lo & (v < 0)] = 0.0; v[hi & (v > 0)] = 0.0
        out[:, k+1] = y
    return t_eval, out[:N], wrap(out[2*N:3*N])

def integrate_contact(K_of_t, b, r0=None, th0=None, r_star=None, th_star=None,
                      method="RK45", rtol=RTOL, atol=ATOL, t_eval=T_EVAL,
                      first_order_theta=False, max_events=200000,
                      return_events=False):
    """Continuous inelastic ("sticking") contact law for r in [0,1] (Eq. 14).

    Hybrid, event-located integration:
      * each agent carries a contact state in {0: free, +1: stuck at r=1,
        -1: stuck at r=0}; stuck agents have dr/dt = dv/dt = 0 (v held at 0);
      * a free agent whose radius crosses a bound with outward velocity sticks
        there at the event-located contact time, with its radial velocity set
        to zero at the instant of contact;
      * a stuck agent is released, with continuous position and velocity, as
        soon as its unconstrained radial acceleration points inward; releases
        are processed instantly (before integrating) and the radius is nudged
        1e-12 inside the bound so the contact event cannot re-fire spuriously
        at the restart point;
      * agents that start on a bound with outward acceleration are stuck at
        t = 0.
    The angular subsystem is untouched.
    """
    r0 = R0 if r0 is None else r0; th0 = TH0 if th0 is None else th0
    r_star = r0.copy() if r_star is None else r_star
    th_star = th0.copy() if th_star is None else th_star
    b_r = np.full(N, b); b_th = np.full(N, b)
    base = rhs_factory(K_of_t, b_r, b_th, r_star, th_star, clip_in_rhs=False,
                       first_order_theta=first_order_theta)
    EPS = 1e-12
    stuck = np.zeros(N, dtype=int)

    def accel(t, y):
        yy = y.copy()
        yy[:N][stuck == 1] = 1.0
        yy[:N][stuck == -1] = 0.0
        yy[N:2*N][stuck != 0] = 0.0
        return base(t, yy)[N:2*N]

    REL = 5e-14      # release threshold: just past the event offset 1e-13,
                     # so a stuck agent whose (serialized) twin event already
                     # fired is released here; well above root-finding noise.

    def settle(t, y):
        """Reconcile the contact states at time t.

        Simultaneous identical events are serialized by the solver (only the
        first terminal root is returned), so after each event batch this
        sweep (i) releases every stuck agent whose unconstrained acceleration
        already points inward past the event threshold, and (ii) sticks every
        free agent left on or beyond a bound with an outward push -- the
        mirror case for serialized simultaneous hits. Repeats until stable.
        """
        changed = True
        while changed:
            changed = False
            a = accel(t, y)
            for i in range(N):
                if stuck[i] == 1 and a[i] < -REL:
                    stuck[i] = 0; y[i] = 1.0 - EPS; y[N+i] = 0.0; changed = True
                elif stuck[i] == -1 and a[i] > REL:
                    stuck[i] = 0; y[i] = EPS; y[N+i] = 0.0; changed = True
                elif stuck[i] == 0 and y[i] >= 1.0:
                    y[i] = 1.0
                    if y[N+i] > 0.0 or (y[N+i] == 0.0 and a[i] >= 0.0):
                        stuck[i] = 1; y[N+i] = 0.0; changed = True
                    else:
                        # nudging a free boundary agent inward changes the
                        # unconstrained acceleration of its stuck neighbours
                        # (the release-event function is evaluated at the
                        # nudged state), so the sweep must run again with
                        # the acceleration recomputed; otherwise a release
                        # event can be born already past its root and never
                        # fire (the solver only detects sign CHANGES).
                        y[i] = 1.0 - EPS; changed = True
                elif stuck[i] == 0 and y[i] <= 0.0:
                    y[i] = 0.0
                    if y[N+i] < 0.0 or (y[N+i] == 0.0 and a[i] <= 0.0):
                        stuck[i] = -1; y[N+i] = 0.0; changed = True
                    else:
                        y[i] = EPS; changed = True  # see upper-bound note
        return y

    def rhs(t, y):
        d = base(t, y)
        m = stuck != 0
        if m.any():
            d[:N][m] = 0.0
            d[N:2*N][m] = 0.0
        return d

    def make_events():
        evs, owners = [], []
        for i in range(N):
            if stuck[i] == 0:
                def hit_hi(t, y, i=i): return y[i] - 1.0
                hit_hi.terminal = True; hit_hi.direction = +1.0
                def hit_lo(t, y, i=i): return y[i]
                hit_lo.terminal = True; hit_lo.direction = -1.0
                evs += [hit_hi, hit_lo]; owners += [(i, +1), (i, -1)]
            else:
                sgn = float(stuck[i])
                # +1e-13 offset: a frozen agent whose unconstrained
                # acceleration is identically zero (grazing equilibrium)
                # keeps g > 0 and cannot trigger spurious roots; a genuine
                # inward crossing fires at accel = -1e-13, a negligible
                # shift of the release time.
                def release(t, y, i=i, sgn=sgn):
                    return sgn * accel(t, y)[i] + 1e-13
                release.terminal = True; release.direction = -1.0
                evs.append(release); owners.append((i, 0))
        return evs, owners

    t0f, tf = float(t_eval[0]), float(t_eval[-1])
    y = np.concatenate([r0, np.zeros(N), th0, np.zeros(N)])
    a0 = base(t0f, y)[N:2*N]
    for i in range(N):
        # >= / <= 0 inclusive: an agent resting on a bound exactly at its
        # attractor (zero acceleration) is held there, which freezes its
        # state exactly and prevents spurious event roots from floating-
        # point noise in g = r - 1 (degenerate grazing equilibrium).
        if y[i] >= 1.0 - 1e-15 and a0[i] >= 0.0:
            stuck[i] = 1; y[i] = 1.0; y[N+i] = 0.0
        elif y[i] <= 1e-15 and a0[i] <= 0.0:
            stuck[i] = -1; y[i] = 0.0; y[N+i] = 0.0
    y = settle(t0f, y)

    t_now = t0f
    out = np.empty((4*N, len(t_eval)))
    out[:, 0] = y
    filled = 1
    n_events = 0
    while t_now < tf - 1e-10:
        evs, owners = make_events()
        want = t_eval[t_eval > t_now + 1e-12]
        sol = solve_ivp(rhs, (t_now, tf), y, method=method,
                        rtol=rtol, atol=atol, events=evs,
                        t_eval=want if len(want) else None)
        k = sol.y.shape[1] if len(sol.t) else 0
        if k:
            out[:, filled:filled + k] = sol.y
            filled += k
        if sol.status == 1:
            times = [(te[0], j) for j, te in enumerate(sol.t_events) if len(te)]
            t_ev = min(t for t, _ in times)
            j0 = [j for t, j in times if t <= t_ev + 1e-12]
            y = sol.y_events[j0[0]][0].copy()
            for j in j0:
                i, kind = owners[j]
                if kind == +1 and stuck[i] == 0:
                    stuck[i] = 1; y[i] = 1.0; y[N+i] = 0.0; n_events += 1
                elif kind == -1 and stuck[i] == 0:
                    stuck[i] = -1; y[i] = 0.0; y[N+i] = 0.0; n_events += 1
                elif kind == 0 and stuck[i] != 0:
                    # the event root certifies the inward crossing: release
                    # here directly. At a coupling discontinuity the dense-
                    # output root can land a hair BEFORE the jump, where the
                    # true acceleration is still zero, so the settle() sweep
                    # alone would refuse the release and the same event
                    # would re-fire forever.
                    sgn = stuck[i]
                    stuck[i] = 0
                    y[i] = (1.0 - EPS) if sgn == 1 else EPS
                    y[N+i] = 0.0
            y = settle(t_ev, y)
            t_now = t_ev
            if n_events > max_events:
                raise RuntimeError("contact-event budget exceeded")
        else:
            t_now = tf
    if return_events:
        return t_eval, out[:N], wrap(out[2*N:3*N]), n_events
    return t_eval, out[:N], wrap(out[2*N:3*N])

def wrap(th):
    return (th + np.pi) % (2*np.pi) - np.pi

# ------------------------------------------------------------------ schedules
def sched_const(Kr):
    Kr = np.asarray(Kr, float)
    def f(t): return Kr, Kr
    return f

def sched_ex1(kind):
    """Example 1 sub-examples. Leader column = index 7 (8th individual)."""
    def f(t):
        K = np.zeros((N, N))
        if kind in ("1.1", "1.2"):          # gradual ramp
            if 20 <= t < 60: K[:, 7] = 12.5*(t - 20)
            elif t >= 60:    K[:, 7] = 500.0
        else:                                # 1.3/1.4/1.5: abrupt step
            if t >= 20:      K[:, 7] = 500.0
        K[7, 7] = 0.0
        return K, K.copy()
    return f

def sched_chain(K=500.0):
    """Example 2: chain of command, K[i+1,i]=K, leader index 0."""
    Kr = np.zeros((N, N))
    for i in range(N - 1):
        Kr[i+1, i] = K
    def f(t): return Kr, Kr
    return f

def sched_ex3(sub):
    """Example 3 sub-scenarios (Appendix A listings)."""
    def f(t):
        K = np.zeros((N, N))
        if sub == "3.1":
            if 0 <= t < 20:    K[:, 0] = 25*t
            elif 20 <= t < 40: K[:, 0] = 500 - 25*(t - 20)
            if 40 <= t < 60:   K[:, 7] = 25*(t - 40)
            elif 60 <= t < 80: K[:, 7] = 500 - 25*(t - 60)
        elif sub == "3.2":
            if 0 <= t < 20:    K[:, 0] = 25*t
            elif 20 <= t < 40: K[:, 0] = 500 - 25*(t - 20)
            if t >= 40:
                K[:, 0] = 0.0
                for i in range(10):
                    if 40 <= t < 60:   K[i, 7] = 25*(t - 40)
                    elif 60 <= t < 80: K[i, 7] = 500
                for i in range(10, 20):
                    if 40 <= t < 60:   K[i, 18] = 25*(t - 40)
                    elif 60 <= t < 80: K[i, 18] = 500
        elif sub == "3.3":
            if 0 <= t < 20:    K[:, 0] = 25*t
            elif 20 <= t < 40: K[:, 0] = 500 - 25*(t - 20)
        np.fill_diagonal(K, 0.0)
        return K, K.copy()
    return f

def sched_ex4():
    """Example 4: two groups converge onto Leader 11 (FRUSTRATED)."""
    def f(t):
        K = np.zeros((N, N))
        for i in range(10):
            if 0 <= t < 20:    K[i, 7] = 25*t
            elif 20 <= t < 40: K[i, 7] = 500
            elif 40 <= t < 60: K[i, 7] = 500 - 25*(t - 40)
        for i in range(10, 20):
            if 0 <= t < 20:    K[i, 18] = 25*t
            elif 20 <= t < 40: K[i, 18] = 500
            elif 40 <= t < 60: K[i, 18] = 500 - 25*(t - 40)
        if 40 <= t < 60:  K[:, 11] = 25*(t - 40)
        elif t >= 60:     K[:, 11] = 500
        np.fill_diagonal(K, 0.0)
        return K, K.copy()
    return f

# ------------------------------------------------------------------ state helpers
def states_ex1(sub):
    """Initial conditions and attractors for Example 1 sub-examples."""
    r0, th0 = R0.copy(), TH0.copy()
    rs, ts = R0.copy(), TH0.copy()          # attractors = Table 4 baselines
    if sub in ("1.4", "1.5"):                # leader starts CALM, attractor ANGRY
        r0[7], th0[7] = EMO["CALM"]
        rs[7], ts[7] = EMO["ANGRY"]
    return r0, th0, rs, ts

def states_ex2():
    """Example 2: leader index 0 fixed at ANGRY (initial and attractor)."""
    r0, th0 = R0.copy(), TH0.copy()
    r0[0], th0[0] = EMO["ANGRY"]
    rs, ts = r0.copy(), th0.copy()
    return r0, th0, rs, ts

def states_ex3(sub):
    r0, th0 = R0.copy(), TH0.copy()
    rs, ts = R0.copy(), TH0.copy()
    if sub == "3.1":
        r0[0], th0[0] = EMO["SERENE"]; rs[0], ts[0] = EMO["SERENE"]
        r0[7], th0[7] = EMO["CALM"];   rs[7], ts[7] = EMO["CALM"]
    elif sub == "3.2":
        r0[0], th0[0] = EMO["HAPPY"]; rs[0], ts[0] = EMO["HAPPY"]
        rs[7], ts[7] = EMO["ANGRY"]           # angry faction leader
        rs[18], ts[18] = EMO["TIRED"]         # tired faction leader
    elif sub == "3.3":
        r0[0], th0[0] = EMO["HAPPY"]; rs[0], ts[0] = EMO["HAPPY"]
    return r0, th0, rs, ts

def states_ex4():
    r0, th0 = R0.copy(), TH0.copy()
    rs, ts = R0.copy(), TH0.copy()
    rs[7], ts[7] = EMO["ANGRY"]
    rs[18], ts[18] = EMO["TIRED"]
    r0[11], th0[11] = EMO["FRUSTRATED"]
    rs[11], ts[11] = EMO["FRUSTRATED"]
    return r0, th0, rs, ts

# ------------------------------------------------------------------ metrics
def cohesion(r_sol, th_sol):
    """C(t) = mean intensity * exp(-circular variance)  (Eq. 24)."""
    mean_r = r_sol.mean(axis=0)
    R = np.abs(np.exp(1j * th_sol).mean(axis=0))   # mean resultant length
    circ_var = 1.0 - R
    return mean_r * np.exp(-circ_var)
