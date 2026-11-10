"""Reproduction of the construction in Nielsen (2023b, arXiv:2307.10644, Alg. 1-2)."""
import numpy as np
from frmvn import sym, sqrtm_spd, invsqrtm_spd, powm_spd, jeffreys, rho_co

def lift(mu, S):
    d = len(mu)
    Si = np.linalg.inv(S)
    M = np.zeros((2 * d + 1, 2 * d + 1))
    M[:d, :d] = np.eye(d); M[d, :d] = mu; M[d, d] = 1
    M[d + 1:, d] = -mu; M[d + 1:, d + 1:] = np.eye(d)
    D = np.zeros_like(M); D[:d, :d] = Si; D[d, d] = 1; D[d + 1:, d + 1:] = S
    return sym(M @ D @ M.T)

def unlift(G, d):
    S = np.linalg.inv(sym(G[:d, :d]))
    return S @ G[:d, d], sym(S)

def kob_curve(N0, N1, t):
    d = len(N0[0])
    G0, G1 = lift(*N0), lift(*N1)
    R = sqrtm_spd(G0); Ri = np.linalg.inv(R)
    C = sym(Ri @ G1 @ Ri)
    return [unlift(sym(R @ powm_spd(C, tt) @ R), d) for tt in np.atleast_1d(t)]

def kob_estimate(N0, N1, T):
    pts = kob_curve(N0, N1, np.linspace(0, 1, T + 1))
    pts[0] = N0; pts[-1] = N1
    return sum(np.sqrt(max(jeffreys(pts[i], pts[i + 1]), 0)) for i in range(T))

def approx_rao_mvn(N0, N1, eps, depth=0, maxdepth=30):
    """Algorithm 2 of Nielsen (2023b): recursive midpoint subdivision."""
    l = rho_co(N0, N1); u = np.sqrt(max(jeffreys(N0, N1), 0))
    if u / l > 1 + eps and depth < maxdepth:
        (N,) = kob_curve(N0, N1, [0.5])
        return approx_rao_mvn(N0, N, eps, depth + 1) + approx_rao_mvn(N, N1, eps, depth + 1)
    return u
