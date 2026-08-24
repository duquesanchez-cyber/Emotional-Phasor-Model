"""
Core simulation module for the emotional phasor model (Eqs. 20-23 of the paper).

Two integration modes:
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
    """Vectorized RHS. K_of_t(t) -> (K_r, K_theta), both fresh arrays."""
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
            dthdt = -np.sin(th - th_star) + coup_th
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
