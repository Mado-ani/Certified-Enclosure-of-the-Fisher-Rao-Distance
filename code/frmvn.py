"""
frmvn.py -- Fisher-Rao distance between multivariate normals:
certified lower/upper bounds, curve-length estimators, Calvo-Oller tools,
and a reference geodesic-shooting solver.

Conventions: a normal is a pair (mu, S) with mu shape (d,), S shape (d,d) SPD.
Fisher metric: ds^2 = dmu' S^{-1} dmu + 1/2 tr((S^{-1} dS)^2).
"""
import numpy as np
from scipy.linalg import expm, eigh
from scipy.optimize import least_squares, minimize

# ---------------------------------------------------------------- linear algebra
def sym(A):
    return 0.5 * (A + A.T)

def spd_fun(S, fun):
    w, U = eigh(sym(S))
    return (U * fun(w)) @ U.T

def sqrtm_spd(S):
    return spd_fun(S, np.sqrt)

def invsqrtm_spd(S):
    return spd_fun(S, lambda w: 1.0 / np.sqrt(w))

def powm_spd(S, t):
    return spd_fun(S, lambda w: w ** t)

def logm_spd(S):
    return spd_fun(S, np.log)

# ---------------------------------------------------------------- basic quantities
def fisher_norm2(mu, S, dmu, dS):
    Si = np.linalg.inv(S)
    A = Si @ dS
    return float(dmu @ Si @ dmu + 0.5 * np.trace(A @ A))

def kl(N1, N2):
    (m1, S1), (m2, S2) = N1, N2
    d = len(m1)
    S2i = np.linalg.inv(S2)
    dm = m2 - m1
    _, ld1 = np.linalg.slogdet(S1)
    _, ld2 = np.linalg.slogdet(S2)
    return 0.5 * (np.trace(S2i @ S1) + dm @ S2i @ dm - d + ld2 - ld1)

def jeffreys(N1, N2):
    (m1, S1), (m2, S2) = N1, N2
    d = len(m1)
    S1i, S2i = np.linalg.inv(S1), np.linalg.inv(S2)
    dm = m2 - m1
    return float(0.5 * np.trace(S2i @ S1 + S1i @ S2) - d + 0.5 * dm @ (S1i + S2i) @ dm)

def rho_univariate(m1, s1sq, m2, s2sq):
    """Exact Fisher-Rao distance between N(m1,s1sq) and N(m2,s2sq)."""
    s1, s2 = np.sqrt(s1sq), np.sqrt(s2sq)
    arg = 1.0 + (0.5 * (m1 - m2) ** 2 + (s1 - s2) ** 2) / (2.0 * s1 * s2)
    return np.sqrt(2.0) * np.arccosh(np.maximum(arg, 1.0))

def rho_spd(P1, P2):
    """Riemannian distance for ds^2 = 1/2 tr((P^{-1}dP)^2)."""
    lam = eigh(sym(P2), sym(P1), eigvals_only=True)
    return float(np.sqrt(0.5 * np.sum(np.log(lam) ** 2)))

def rho_common_cov(m1, m2, S):
    """Exact Fisher-Rao distance when Sigma1 = Sigma2 = S (Nielsen, Prop. 1)."""
    dm = m2 - m1
    mah = np.sqrt(dm @ np.linalg.solve(S, dm))
    return np.sqrt(2.0) * np.arccosh(1.0 + mah ** 2 / 4.0)

# ---------------------------------------------------------------- normalisation
def whiten(N1, N2):
    """Affine map sending N1 to N(0,I); returns image of N2 (m, S)."""
    (m1, S1), (m2, S2) = N1, N2
    W = invsqrtm_spd(S1)
    return W @ (m2 - m1), sym(W @ S2 @ W)

# ---------------------------------------------------------------- Calvo-Oller
def co_embed(mu, S, beta=1.0):
    d = len(mu)
    P = np.empty((d + 1, d + 1))
    P[:d, :d] = S + beta * np.outer(mu, mu)
    P[:d, d] = beta * mu
    P[d, :d] = beta * mu
    P[d, d] = beta
    return P

def co_unembed(P):
    """Inverse of f_beta: returns (mu, S, beta).  Projection = (mu, S) with beta->1."""
    d = P.shape[0] - 1
    beta = P[d, d]
    mu = P[:d, d] / beta
    S = P[:d, :d] - beta * np.outer(mu, mu)
    return mu, sym(S), beta

def rho_co(N1, N2):
    return rho_spd(co_embed(*N1), co_embed(*N2))

def co_beta_profile(N1, N2):
    """Weights w_i and eigenvalues lam_i with beta_t = sum_i w_i lam_i^t
    (beta_t = (d+1,d+1) entry of the SPD geodesic between f(N1), f(N2))."""
    m, S = whiten(N1, N2)
    P2 = co_embed(m, S)  # f(N1) = I after whitening
    lam, U = eigh(P2)
    w = U[-1, :] ** 2
    return w, lam

def beta_t(w, lam, t):
    t = np.atleast_1d(t)
    return (w[None, :] * lam[None, :] ** t[:, None]).sum(1)

def dbeta_t(w, lam, t):
    t = np.atleast_1d(t)
    L = np.log(lam)
    return (w[None, :] * L[None, :] * lam[None, :] ** t[:, None]).sum(1)

def beta_bounds(N1, N2, n=4001):
    """New C&O-curve certificate (Theorem 5.x):
       rho_N <= L(c_CO) <= int_0^1 beta_t^{-1/2} sqrt(rho_CO^2 - (beta'/beta)^2/2) dt
                         <= rho_CO * int_0^1 beta_t^{-1/2} dt.
    Returns (rho_co, sharp, simple, min_beta)."""
    w, lam = co_beta_profile(N1, N2)
    r = np.sqrt(0.5 * np.sum(np.log(lam) ** 2))
    t = np.linspace(0, 1, n)
    b = beta_t(w, lam, t)
    db = dbeta_t(w, lam, t)
    fac = np.maximum(1.0, b ** -0.5)
    inner = np.maximum(r ** 2 - 0.5 * (db / b) ** 2, 0.0)
    sharp = np.trapezoid(fac * np.sqrt(inner), t)
    simple = r * np.trapezoid(fac, t)
    return r, sharp, simple, b.min()

def co_curve(N1, N2, t):
    """Projected Calvo-Oller curve evaluated at array t (list of (mu,S))."""
    P1, P2 = co_embed(*N1), co_embed(*N2)
    R = sqrtm_spd(P1); Ri = np.linalg.inv(R)
    w, U = eigh(sym(Ri @ P2 @ Ri))
    out = []
    for tt in np.atleast_1d(t):
        G = R @ ((U * w ** tt) @ U.T) @ R
        mu, S, _ = co_unembed(sym(G))
        out.append((mu, S))
    return out

# ---------------------------------------------------------------- other curves
def curve_points(N1, N2, kind, T):
    t = np.linspace(0.0, 1.0, T + 1)
    (m1, S1), (m2, S2) = N1, N2
    pts = []
    if kind == "co":
        return co_curve(N1, N2, t)
    S1i, S2i = np.linalg.inv(S1), np.linalg.inv(S2)
    for a in t:
        if kind == "lin":
            pts.append(((1 - a) * m1 + a * m2, (1 - a) * S1 + a * S2))
        elif kind == "m":
            m = (1 - a) * m1 + a * m2
            S = (1 - a) * S1 + a * S2 + (1 - a) * np.outer(m1, m1) + a * np.outer(m2, m2) - np.outer(m, m)
            pts.append((m, sym(S)))
        elif kind == "e":
            S = np.linalg.inv((1 - a) * S1i + a * S2i)
            pts.append((S @ ((1 - a) * S1i @ m1 + a * S2i @ m2), sym(S)))
        elif kind == "em":
            Se = np.linalg.inv((1 - a) * S1i + a * S2i)
            me = Se @ ((1 - a) * S1i @ m1 + a * S2i @ m2)
            mm = (1 - a) * m1 + a * m2
            Sm = (1 - a) * S1 + a * S2 + (1 - a) * np.outer(m1, m1) + a * np.outer(m2, m2) - np.outer(mm, mm)
            pts.append((0.5 * (me + mm), sym(0.5 * (Se + Sm))))
        else:
            raise ValueError(kind)
    return pts

def curve_estimate(N1, N2, kind="co", T=1000, buggy=False, pts=None):
    """Corrected curve-length estimator  sum_{i=0}^{T-1} sqrt(D_J(c_i, c_{i+1})).
    Always a rigorous upper bound on rho_N (Prop. 3.1).
    buggy=True reproduces the indexing of Eq. (22) in arXiv:2302.08175 (sum from i=1)."""
    if pts is None:
        pts = curve_points(N1, N2, kind, T)
    s = [np.sqrt(max(jeffreys(pts[i], pts[i + 1]), 0.0)) for i in range(len(pts) - 1)]
    return float(np.sum(s[1:] if buggy else s))

def richardson(N1, N2, kind="co", T=200):
    a = curve_estimate(N1, N2, kind, T)
    b = curve_estimate(N1, N2, kind, 2 * T)
    return (4 * b - a) / 3.0, a, b

# ---------------------------------------------------------------- classical bounds
def ub_sqrtJ(N1, N2):
    return np.sqrt(jeffreys(N1, N2))

def ub_spc(N1, N2):
    """Strapasson-Porto-Costa upper bound (Eq. 24 of arXiv:2302.08175)."""
    m, S = whiten(N1, N2)
    D, O = eigh(S)
    mt = O.T @ m
    s = np.sqrt(D)
    # sum over i of univariate distances between N(0,1) and N(mt_i, D_i)
    return float(np.sqrt(np.sum(rho_univariate(0.0, 1.0, mt, D) ** 2)))

def ub_triangle(N1, N2):
    """Eq. (11): rho_P(S1,S2) + min Mahalanobis."""
    (m1, S1), (m2, S2) = N1, N2
    dm = m2 - m1
    mah = min(np.sqrt(dm @ np.linalg.solve(S1, dm)), np.sqrt(dm @ np.linalg.solve(S2, dm)))
    return rho_spd(S1, S2) + mah

# ---------------------------------------------------------------- new lower bound
def lb_marginal(N1, N2, n_starts=8, rng=None, return_dir=False):
    """sup over unit a of the univariate Fisher-Rao distance between the
    marginals a'X (Theorem 4.1).  Any a gives a valid lower bound."""
    m, S = whiten(N1, N2)
    d = len(m)
    rng = np.random.default_rng(0) if rng is None else rng

    def neg(a):
        a = a / np.linalg.norm(a)
        return -rho_univariate(0.0, 1.0, a @ m, a @ S @ a)

    starts = [m.copy() if np.linalg.norm(m) > 0 else np.eye(d)[0]]
    D, O = eigh(S)
    starts += [O[:, 0], O[:, -1]]
    starts += [rng.standard_normal(d) for _ in range(n_starts)]
    best, besta = 0.0, starts[0]
    for a0 in starts:
        if np.linalg.norm(a0) == 0:
            continue
        r = minimize(neg, a0, method="BFGS")
        if -r.fun > best:
            best, besta = -r.fun, r.x / np.linalg.norm(r.x)
    return (best, besta) if return_dir else best

def lb_cov(N1, N2):
    return rho_spd(N1[1], N2[1])

def lb_best(N1, N2):
    return max(rho_co(N1, N2), lb_marginal(N1, N2))

# ---------------------------------------------------------------- geodesics
def geodesic_ode(N0, v0, t1=1.0, rtol=1e-11, atol=1e-12):
    """Integrate mu''=S'S^{-1}mu', S''=S'S^{-1}S'-mu'mu'^T."""
    from scipy.integrate import solve_ivp
    mu0, S0 = N0
    dmu0, dS0 = v0
    d = len(mu0)

    def pack(mu, S, dmu, dS):
        return np.concatenate([mu, S.ravel(), dmu, dS.ravel()])

    def unpack(y):
        mu = y[:d]; S = y[d:d + d * d].reshape(d, d)
        dmu = y[d + d * d:2 * d + d * d]; dS = y[2 * d + d * d:].reshape(d, d)
        return mu, S, dmu, dS

    def f(t, y):
        mu, S, dmu, dS = unpack(y)
        Si = np.linalg.inv(S)
        return pack(dmu, dS, dS @ Si @ dmu, sym(dS @ Si @ dS - np.outer(dmu, dmu)))

    sol = solve_ivp(f, [0, t1], pack(mu0, S0, dmu0, dS0), method="DOP853", rtol=rtol, atol=atol)
    mu, S, _, _ = unpack(sol.y[:, -1])
    return mu, sym(S)

def exp_map_std(x, B, t=1.0):
    """Closed-form geodesic from N(0,I) with initial velocity (x, B)
    (Calvo & Oller 1991 / Eriksen 1986).  Returns (mu(t), S(t))."""
    d = len(x)
    A = np.zeros((2 * d + 1, 2 * d + 1))
    A[:d, :d] = -B
    A[:d, d] = x
    A[d, :d] = x
    A[d, d + 1:] = -x
    A[d + 1:, d] = -x
    A[d + 1:, d + 1:] = B
    E = expm(t * A)
    Delta = sym(E[:d, :d])
    delta = E[:d, d]
    S = np.linalg.inv(Delta)
    return S @ delta, sym(S)

def vech_to_sym(v, d):
    M = np.zeros((d, d))
    iu = np.triu_indices(d)
    M[iu] = v
    return M + M.T - np.diag(np.diag(M))

def sym_to_vech(M):
    return M[np.triu_indices(M.shape[0])]

def shoot(N1, N2, x0=None, B0=None, tol=1e-13, max_nfev=2000):
    """Reference Fisher-Rao distance by geodesic shooting with the closed-form
    exponential map.  Returns (distance, residual, (x,B))."""
    m, S = whiten(N1, N2)
    d = len(m)
    if x0 is None:
        # initial guess: tangent of the C&O curve at t=0 (finite difference), scaled
        h = 1e-4
        (mu_h, S_h), = co_curve((np.zeros(d), np.eye(d)), (m, S), [h])
        x0 = mu_h / h
        B0 = (S_h - np.eye(d)) / h
    target = np.concatenate([m, sym_to_vech(logm_spd(S))])

    def res(p):
        x = p[:d]; B = vech_to_sym(p[d:], d)
        mu1, S1 = exp_map_std(x, B)
        # residual in whitened, log-coordinates (well scaled)
        return np.concatenate([mu1, sym_to_vech(logm_spd(S1))]) - target

    p0 = np.concatenate([x0, sym_to_vech(B0)])
    sol = least_squares(res, p0, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev, method="lm")
    x = sol.x[:d]; B = vech_to_sym(sol.x[d:], d)
    dist = np.sqrt(x @ x + 0.5 * np.trace(B @ B))
    return dist, float(np.max(np.abs(sol.fun))), (x, B)

def shoot_robust(N1, N2, steps=(1.0,), **kw):
    """Shooting with homotopy in the target if the direct solve fails."""
    m, S = whiten(N1, N2)
    d = len(m)
    best = shoot(N1, N2, **kw)
    if best[1] < 1e-9:
        return best
    # homotopy: N(s m, S^s), s from small to 1
    x = B = None
    for s in np.linspace(0.1, 1.0, 10):
        Ns = (s * m, powm_spd(S, s))
        r = shoot((np.zeros(d), np.eye(d)), Ns, x0=x, B0=B, **kw)
        x, B = r[2]
    return r

def random_normal(d, rng, mean_scale=1.0):
    L = np.tril(rng.uniform(0, 1, (d, d)))
    return rng.uniform(0, mean_scale, d), L @ L.T + 1e-12 * np.eye(d)

# ---------------------------------------------------------------- multiple shooting
def _exp_std_with_vel(x, B):
    """exp map at N(0,I) at t=1, returning (mu, S, dmu, dS) (position and velocity)."""
    d = len(x)
    A = np.zeros((2 * d + 1, 2 * d + 1))
    A[:d, :d] = -B; A[:d, d] = x; A[d, :d] = x
    A[d, d + 1:] = -x; A[d + 1:, d] = -x; A[d + 1:, d + 1:] = B
    E = expm(A); GE = A @ E
    Delta = sym(E[:d, :d]); delta = E[:d, d]
    dDelta = sym(GE[:d, :d]); ddelta = GE[:d, d]
    S = np.linalg.inv(Delta); mu = S @ delta
    dS = -S @ dDelta @ S; dmu = dS @ delta + S @ ddelta
    return mu, sym(S), dmu, sym(dS)

def _frame(mu, S):
    """Affine frame sending N(mu,S) to N(0,I): x -> W (x - mu)."""
    return invsqrtm_spd(S), mu

def multishoot(N1, N2, K=8, init="co", tol=1e-12, max_nfev=400, z_init=None, return_z=False, frame="first"):
    """Fisher-Rao distance by multiple shooting with K segments.
    Unknowns: interior nodes (mu_k, log S_k) in the frame of N1, and the initial
    velocity of each segment in that node's whitened frame.  Residuals: position
    and velocity continuity at every node, expressed in whitened frames.
    Returns (distance, max residual, segment lengths)."""
    m, S = whiten(N1, N2)
    d = len(m); p = d + d * (d + 1) // 2
    I = (np.zeros(d), np.eye(d))
    if frame == "mid":
        # re-centre so that the midpoint of the C&O curve is N(0,I)
        (cm, CS), = co_curve(I, (m, S), [0.5])
        W = invsqrtm_spd(CS)
        I = (W @ (-cm), sym(W @ W))
        m, S = W @ (m - cm), sym(W @ S @ W)
    tt = np.linspace(0, 1, K + 1)
    if init == "co":
        nodes = co_curve(I, (m, S), tt)
    else:
        nodes = curve_points(I, (m, S), init, K)
    nodes[0] = I; nodes[-1] = (m, S)

    def node_vec(N):
        return np.concatenate([N[0], sym_to_vech(logm_spd(N[1]))])

    def vec_node(v):
        return v[:d], expm(vech_to_sym(v[d:], d))

    nint = (K - 1) * p
    if z_init is not None:
        z0 = np.asarray(z_init, float).copy()
    else:
        # initial velocities: single shooting between consecutive (close) nodes
        vel0 = []
        for k in range(K):
            r = shoot(nodes[k], nodes[k + 1])
            x, B = r[2]
            vel0.append(np.concatenate([x, sym_to_vech(B)]))
        z0 = np.concatenate([node_vec(nodes[k]) for k in range(1, K)] + vel0)

    def unpack(z):
        nd = [I] + [vec_node(z[k * p:(k + 1) * p]) for k in range(K - 1)] + [(m, S)]
        vel = [z[nint + k * p: nint + (k + 1) * p] for k in range(K)]
        return nd, vel

    def res(z):
        nd, vel = unpack(z)
        out = []
        for k in range(K):
            Wk, ck = _frame(*nd[k]); Wki = np.linalg.inv(Wk)
            x = vel[k][:d]; B = vech_to_sym(vel[k][d:], d)
            mu1, S1, dmu1, dS1 = _exp_std_with_vel(x, B)
            # back to global frame
            mug = Wki @ mu1 + ck; Sg = Wki @ S1 @ Wki.T
            dmug = Wki @ dmu1; dSg = Wki @ dS1 @ Wki.T
            # into frame of node k+1
            W1, c1 = _frame(*nd[k + 1])
            mu_, S_ = W1 @ (mug - c1), sym(W1 @ Sg @ W1.T)
            out.append(np.concatenate([mu_, sym_to_vech(logm_spd(S_))]))
            if k < K - 1:
                v_ = np.concatenate([W1 @ dmug, sym_to_vech(sym(W1 @ dSg @ W1.T))])
                out.append(v_ - vel[k + 1])
        return np.concatenate(out)

    sol = least_squares(res, z0, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev, method="trf", x_scale="jac")
    if np.max(np.abs(sol.fun)) > 1e-9:   # polish with Levenberg-Marquardt
        sol2 = least_squares(res, sol.x, xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev, method="lm")
        if np.max(np.abs(sol2.fun)) < np.max(np.abs(sol.fun)):
            sol = sol2
    _, vel = unpack(sol.x)
    seg = [np.sqrt(v[:d] @ v[:d] + 0.5 * np.trace(vech_to_sym(v[d:], d) @ vech_to_sym(v[d:], d))) for v in vel]
    out = (float(np.sum(seg)), float(np.max(np.abs(sol.fun))), seg)
    return out + (sol.x,) if return_z else out

def multishoot_continuation(N1, N2, K=8, J=8, max_nfev=1000):
    """Multiple shooting with continuation in the target: solve for targets
    c_CO(tau_j), tau_j = j/J, along the projected Calvo-Oller curve, each
    warm-started from the previous solution."""
    m, S = whiten(N1, N2)
    d = len(m)
    I = (np.zeros(d), np.eye(d))
    targets = co_curve(I, (m, S), np.linspace(0, 1, J + 1)[1:])
    targets[-1] = (m, S)
    z = None
    for j, Nt in enumerate(targets):
        v, rr, seg, z = multishoot(I, Nt, K=K, init="co", max_nfev=max_nfev, z_init=z, return_z=True)
    return v, rr, seg

def reference_distance(N1, N2, K=(8, 16), inits=("co", "e", "em")):
    """Reference value of rho_N.
    1. single shooting (with homotopy) if it converges to residual < 1e-8;
    2. otherwise multiple shooting in the frame centred at the C&O midpoint,
       then in the frame of N1, then with continuation, from several initial
       curves, until two independent starts agree to 1e-7.
    Returns (value, residual, method, spread between agreeing starts);
    spread == 0 means only one start converged."""
    r, res, _ = shoot_robust(N1, N2)
    if res < 1e-8:
        return r, res, "single", 0.0
    vals = []
    def agree():
        return len(vals) >= 2 and max(v[0] for v in vals) - min(v[0] for v in vals) < 1e-7
    for frame in ("mid", "first"):
        for k in K:
            for ini in inits:
                try:
                    v, rr, _ = multishoot(N1, N2, K=k, init=ini, max_nfev=1000, frame=frame)
                except Exception:
                    continue
                if rr < 1e-8:
                    vals.append((v, rr, f"multi {frame} K={k} {ini}"))
                if agree():
                    break
            if agree():
                break
        if agree():
            break
    if not agree():
        for J in (8, 16):
            try:
                v, rr, _ = multishoot_continuation(N1, N2, K=8, J=J)
            except Exception:
                continue
            if rr < 1e-8:
                vals.append((v, rr, f"continuation J={J}"))
                if agree():
                    break
    if not vals:
        return np.nan, np.inf, "failed", np.nan
    best = min(vals)
    spread = max(v[0] for v in vals) - best[0]
    return best[0], best[1], best[2], spread
