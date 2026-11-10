"""
verified.py -- floating-point-rigorous versions of the certified bounds,
using Arb ball arithmetic (python-flint).

Principle: approximate quantities (eigenvectors, optimal directions, curve
points) are computed in ordinary floating point; validity never depends on
their accuracy.  Only the final evaluations are done in ball arithmetic, from
the exact binary64 input data.

  verified_lb_co    : rigorous lower bound on rho_CO   (hence on rho_N)
  verified_lb_marg  : rigorous lower bound via one marginal direction
  verified_ub_chain : rigorous upper bound from any chain of SPD points
"""
import numpy as np
from scipy.linalg import eigh
from flint import arb, arb_mat, ctx

from frmvn import co_embed, curve_points, lb_marginal, invsqrtm_spd

ctx.prec = 128  # working precision of the balls (bits)


# ------------------------------------------------------------------ helpers
def A(M):
    """Exact conversion of a float matrix/vector to arb_mat."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    return arb_mat([[arb(float(x)) for x in row] for row in M])

def lo(x):
    return float(x.lower())

def hi(x):
    return float(x.upper())

def fro_upper(E):
    s = arb(0)
    for i in range(E.nrows()):
        for j in range(E.ncols()):
            s += E[i, j] * E[i, j]
    return s.sqrt().upper()          # arb upper endpoint

def embed_arb(mu, S):
    """f(mu,S) in ball arithmetic, from exact float inputs."""
    d = len(mu)
    m = [arb(float(x)) for x in mu]
    P = arb_mat(d + 1, d + 1)
    for i in range(d):
        for j in range(d):
            P[i, j] = arb(float(S[i, j])) + m[i] * m[j]
        P[i, d] = m[i]; P[d, i] = m[i]
    P[d, d] = arb(1)
    return P

def is_spd_certified(S_arb, S_float):
    """Certify that S is SPD.  With X the float eigenvectors of S, check in ball
    arithmetic that X is invertible (||X'X - I||_2 < 1) and that A = X'SX has
    all eigenvalues positive (Weyl about its diagonal).  By Sylvester's law of
    inertia, S is then SPD."""
    w, X = eigh(0.5 * (S_float + S_float.T))
    n = len(w)
    Xa = A(X)
    I = arb_mat(n, n); Dg = arb_mat(n, n)
    Aa = Xa.transpose() * S_arb * Xa
    for i in range(n):
        I[i, i] = arb(1); Dg[i, i] = arb(float(Aa[i, i].mid()))
    if not (fro_upper(Xa.transpose() * Xa - I) < 1):
        return False
    e = fro_upper(Aa - Dg)
    return bool(min(float(Aa[i, i].mid()) for i in range(n)) - e > 0)

# ------------------------------------------------------------------ lower bounds
def verified_lb_co(N1, N2):
    """Rigorous lower bound on rho_CO(N1,N2) = rho_P(f(N1), f(N2)).

    rho_P is congruence invariant, so for any invertible X the generalized
    eigenvalues of (X'P2X, X'P1X) equal those of (P2, P1).  With X the float
    generalized eigenvectors, A1 = X'P1X ~ I and A2 = X'P2X ~ diag.  Then
      lambda_i(A1)  in [1 - e1, 1 + e1],        e1 >= ||A1 - I||_2,
      lambda_(i)(A2) in [delta_(i) - e2, delta_(i) + e2]   (Weyl),
      mu_i in [lambda_i(A2)/lambda_max(A1), lambda_i(A2)/lambda_min(A1)]
    (Courant-Fischer), all evaluated in ball arithmetic."""
    P1f, P2f = co_embed(*N1), co_embed(*N2)
    _, X = eigh(P2f, P1f)
    n = X.shape[0]
    Xa = A(X)
    A1 = Xa.transpose() * embed_arb(*N1) * Xa
    A2 = Xa.transpose() * embed_arb(*N2) * Xa
    delta = [float(A2[i, i].mid()) for i in range(n)]
    I = arb_mat(n, n)
    Dg = arb_mat(n, n)
    for i in range(n):
        I[i, i] = arb(1); Dg[i, i] = arb(delta[i])
    e1 = fro_upper(A1 - I)
    e2 = fro_upper(A2 - Dg)
    if not (e1 < 1):
        return 0.0
    lmin_A1, lmax_A1 = 1 - e1, 1 + e1
    ssq = arb(0)
    for dl in sorted(delta):
        a_lo = (arb(dl) - e2).lower()
        a_hi = (arb(dl) + e2).upper()
        if not (a_lo > 0):
            return 0.0
        mu_lo = (a_lo / lmax_A1).lower()
        mu_hi = (a_hi / lmin_A1).upper()
        if mu_lo > 1:
            m = mu_lo.log().lower()
        elif mu_hi < 1:
            m = (-(mu_hi.log())).lower()
        else:
            m = arb(0)
        ssq += m * m
    return lo((ssq / 2).sqrt())

def verified_lb_marg(N1, N2, a=None):
    """Rigorous lower bound rho_1(a'X under N1, a'X under N2) for one direction
    a (in original coordinates).  The univariate distance is invariant under
    rescaling a, so a need not be normalised."""
    if a is None:
        _, b = lb_marginal(N1, N2, return_dir=True)
        a = invsqrtm_spd(N1[1]) @ b
    (m1, S1), (m2, S2) = N1, N2
    av = A(a.reshape(-1, 1))
    mu1 = (av.transpose() * A(m1.reshape(-1, 1)))[0, 0]
    mu2 = (av.transpose() * A(m2.reshape(-1, 1)))[0, 0]
    v1 = (av.transpose() * A(S1) * av)[0, 0]
    v2 = (av.transpose() * A(S2) * av)[0, 0]
    if not (v1 > 0 and v2 > 0):
        return 0.0
    s1, s2 = v1.sqrt(), v2.sqrt()
    dm = mu1 - mu2; ds = s1 - s2
    z = 1 + (dm * dm / 2 + ds * ds) / (2 * s1 * s2)
    z_lo = z.lower()
    if not (z_lo > 1):
        return 0.0
    return lo(arb(2).sqrt() * z_lo.acosh())


# ------------------------------------------------------------------ upper bound
def jeffreys_arb(Pa, Pb, Pai, Pbi, dmu):
    d = Pa.nrows()
    t = (Pbi * Pa).trace() + (Pai * Pb).trace()
    q = (dmu.transpose() * (Pai + Pbi) * dmu)[0, 0]
    return (t + q) / 2 - d

def verified_ub_chain(pts, N1, N2):
    """Rigorous upper bound sum_i sqrt(D_J(c_i, c_{i+1})) for a chain of float
    points.  Endpoints are replaced by the exact inputs; every interior point
    is certified SPD.  Returns +inf if certification fails."""
    pts = list(pts)
    pts[0] = N1; pts[-1] = N2
    S_arb, Si_arb, m_arb = [], [], []
    for (mu, S) in pts:
        Sf = 0.5 * (S + S.T)
        Sa = A(Sf)
        if not is_spd_certified(Sa, Sf):
            return np.inf
        S_arb.append(Sa); Si_arb.append(Sa.inv()); m_arb.append(A(mu.reshape(-1, 1)))
    tot = arb(0)
    for i in range(len(pts) - 1):
        DJ = jeffreys_arb(S_arb[i], S_arb[i + 1], Si_arb[i], Si_arb[i + 1], m_arb[i + 1] - m_arb[i])
        u = DJ.upper()
        tot += u.sqrt() if u > 0 else arb(0)
    return hi(tot)

def verified_bounds(N1, N2, kind="co", T=128):
    """Verified enclosure [LB, UB] of rho_N."""
    lb = max(verified_lb_co(N1, N2), verified_lb_marg(N1, N2))
    ub = verified_ub_chain(curve_points(N1, N2, kind, T), N1, N2)
    return lb, ub
