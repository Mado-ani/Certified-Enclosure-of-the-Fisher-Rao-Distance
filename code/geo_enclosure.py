"""
geo_enclosure.py -- verified two-sided enclosure along a computed geodesic.

Given the shooting solution (x, B) in the frame of N1 (whitening W = Sigma1^{-1/2}),
the curve  gamma(t) = A_L( exp-map_{N(0,I)}(t (x,B)) ),  A_L: N(m,S) -> N(mu1 + L m, L S L'),
with L any invertible matrix, is an EXACT Fisher-Rao geodesic (affine maps are isometries),
from  N1~ = N(mu1, L L')  to  N2~ = gamma(1).  We evaluate gamma(t_i) in ball arithmetic.

  UB  (unconditional):  sqrt(D_J(N1,N1~)) + sum_i sqrt(D_J(g_i,g_{i+1})) + sqrt(D_J(N2~,N2))
  LBc (conditional on gamma being minimising between N1~ and N2~):
        sum_i LB_CO(g_i, g_{i+1}) - sqrt(D_J(N1,N1~)) - sqrt(D_J(N2~,N2))
"""
import numpy as np
from scipy.linalg import eigh
from flint import arb, arb_mat

from frmvn import co_embed, sqrtm_spd, shoot_robust, whiten
from verified import A, fro_upper, jeffreys_arb, lo, hi


def _gen_matrix(x, B):
    d = len(x)
    G = np.zeros((2 * d + 1, 2 * d + 1))
    G[:d, :d] = -B; G[:d, d] = x; G[d, :d] = x
    G[d, d + 1:] = -x; G[d + 1:, d] = -x; G[d + 1:, d + 1:] = B
    return G

def _sub(M, r0, r1, c0, c1):
    return arb_mat([[M[i, j] for j in range(c0, c1)] for i in range(r0, r1)])

def geodesic_points_arb(N1, x, B, ts):
    """Ball enclosures (mu, S) of the exact geodesic points gamma(t), plus the
    exact start point N1~ = (mu1, L L') with L = float sqrt(Sigma1)."""
    mu1, S1 = N1
    d = len(mu1)
    L = A(sqrtm_spd(S1)); mu1a = A(mu1.reshape(-1, 1))
    Ga = A(_gen_matrix(x, B))
    out = []
    for t in ts:
        E = (Ga * arb(float(t))).exp()
        Delta = _sub(E, 0, d, 0, d); delta = _sub(E, 0, d, d, d + 1)
        Sstd = Delta.inv(); mstd = Sstd * delta
        out.append((mu1a + L * mstd, L * Sstd * L.transpose()))
    return out

def lb_co_arb(Na, Nb, float_mid):
    """Rigorous lower bound on rho_CO between two ball-valued normals (all
    points in the balls).  float_mid = ((mu,S),(mu,S)) midpoints for X."""
    P1f, P2f = co_embed(*float_mid[0]), co_embed(*float_mid[1])
    _, X = eigh(P2f, P1f)
    n = X.shape[0]; Xa = A(X)

    def emb(N):
        mu, S = N; d = S.nrows()
        P = arb_mat(d + 1, d + 1)
        for i in range(d):
            for j in range(d):
                P[i, j] = S[i, j] + mu[i, 0] * mu[j, 0]
            P[i, d] = mu[i, 0]; P[d, i] = mu[i, 0]
        P[d, d] = arb(1)
        return P

    A1 = Xa.transpose() * emb(Na) * Xa
    A2 = Xa.transpose() * emb(Nb) * Xa
    delta = [float(A2[i, i].mid()) for i in range(n)]
    I = arb_mat(n, n); Dg = arb_mat(n, n)
    for i in range(n):
        I[i, i] = arb(1); Dg[i, i] = arb(delta[i])
    e1 = fro_upper(A1 - I); e2 = fro_upper(A2 - Dg)
    if not (e1 < 1):
        return arb(0)
    ssq = arb(0)
    for dl in sorted(delta):
        a_lo = (arb(dl) - e2).lower(); a_hi = (arb(dl) + e2).upper()
        if not (a_lo > 0):
            return arb(0)
        mu_lo = (a_lo / (1 + e1)).lower(); mu_hi = (a_hi / (1 - e1)).upper()
        if mu_lo > 1:
            m = mu_lo.log().lower()
        elif mu_hi < 1:
            m = (-(mu_hi.log())).lower()
        else:
            m = arb(0)
        ssq += m * m
    return (ssq / 2).sqrt().lower()

def sqrtJ_upper(Na, Nb):
    """Upper bound on sqrt(D_J) for ball-valued normals."""
    Sa, Sb = Na[1], Nb[1]
    DJ = jeffreys_arb(Sa, Sb, Sa.inv(), Sb.inv(), Nb[0] - Na[0])
    u = DJ.upper()
    return u.sqrt().upper() if u > 0 else arb(0)

def mid_float(N):
    mu, S = N
    m = np.array([float(mu[i, 0].mid()) for i in range(mu.nrows())])
    s = np.array([[float(S[i, j].mid()) for j in range(S.ncols())] for i in range(S.nrows())])
    return m, 0.5 * (s + s.T)

def geodesic_enclosure(N1, N2, T=32, sol=None):
    """Returns dict with conditional LB, unconditional UB, and shooting length."""
    if sol is None:
        rho, res, (x, B) = shoot_robust(N1, N2)
    else:
        rho, res, (x, B) = sol
    ts = np.linspace(0, 1, T + 1)
    pts = geodesic_points_arb(N1, x, B, ts)
    N1a = (A(N1[0].reshape(-1, 1)), A(N1[1])); N2a = (A(N2[0].reshape(-1, 1)), A(N2[1]))
    gap0 = sqrtJ_upper(N1a, pts[0]); gap1 = sqrtJ_upper(pts[-1], N2a)
    mids = [mid_float(p) for p in pts]
    ub = gap0 + gap1; lbc = -(gap0 + gap1)
    for i in range(T):
        ub += sqrtJ_upper(pts[i], pts[i + 1])
        lbc += lb_co_arb(pts[i], pts[i + 1], (mids[i], mids[i + 1]))
    return dict(lb_cond=lo(lbc), ub=hi(ub), shoot=float(rho), res=float(res),
                gap=hi(gap0 + gap1))
