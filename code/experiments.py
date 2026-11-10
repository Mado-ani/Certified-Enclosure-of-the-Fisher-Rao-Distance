"""Experiments for Paper 1.  Usage: python experiments.py [part ...]
Parts: examples fig17 conv commoncov mcA mcB   (default: all)
Outputs JSON to ../results and PDF figures to ../figs."""
import sys, json, time, os
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from frmvn import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); FIG = os.path.join(HERE, "..", "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
C = dict(co="#2a78d6", m="#eb6834", e="#1baf7a", em="#c98500", lin="#e87ba4", other="#008300", truth="#222222")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "lines.linewidth": 1.6, "legend.frameon": False, "font.family": "serif"})
CURVES = ["lin", "m", "e", "em", "co"]

def save(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)

def all_bounds(N1, N2, T=64, truth=None):
    r = dict()
    r["co"] = rho_co(N1, N2)
    r["marg"] = lb_marginal(N1, N2)
    r["lb"] = max(r["co"], r["marg"])
    r["sqrtJ"] = ub_sqrtJ(N1, N2); r["spc"] = ub_spc(N1, N2); r["tri"] = ub_triangle(N1, N2)
    _, r["beta_sharp"], r["beta_simple"], r["beta_min"] = beta_bounds(N1, N2)
    for k in CURVES:
        a = curve_estimate(N1, N2, k, T); b = curve_estimate(N1, N2, k, 2 * T)
        r["curve_" + k] = b                      # rigorous upper bound
        r["rich_" + k] = (4 * b - a) / 3.0       # O(T^-4) point estimate of curve length
    r["ub"] = min([r["sqrtJ"], r["spc"], r["tri"], r["beta_sharp"]] + [r["curve_" + k] for k in CURVES])
    r["point"] = min(r["rich_" + k] for k in CURVES)
    if truth is not None:
        r["truth"] = truth
    return r

# ------------------------------------------------------------------ examples
def examples():
    S = np.array([[1.1, .9], [.9, 1.1]])
    ex = {
        "Ex1 (Pinele et al.)": ((np.array([-1., 0]), S), (np.array([6., 3]), S)),
        "Han-Park": ((np.zeros(2), np.diag([1, .1])), (np.ones(2), np.diag([.1, 1]))),
        "Ex2b": ((np.zeros(2), np.eye(2)), (np.array([1., 0]), np.array([[1, -1], [-1, 2.]]))),
        "Ex2c": ((np.zeros(2), np.eye(2)), (np.array([5., 0]), np.array([[1, -1], [-1, 2.]]))),
    }
    out = {}
    for name, (N1, N2) in ex.items():
        rho, res, _ = shoot(N1, N2)
        r = all_bounds(N1, N2, T=500, truth=rho)
        r["shoot_res"] = res
        for k in CURVES:
            r["T1000_" + k] = curve_estimate(N1, N2, k, 1000)
            r["T1000buggy_" + k] = curve_estimate(N1, N2, k, 1000, buggy=True)
        out[name] = r
        print(name, {k: round(v, 5) for k, v in r.items() if isinstance(v, float)})
    save("examples", out)

# ------------------------------------------------------------------ fig 17
def fig17():
    N1 = (np.zeros(2), np.diag([1, .1])); N2 = (np.ones(2), np.diag([.1, 1]))
    rho = shoot(N1, N2)[0]; lco = rho_co(N1, N2)
    Ts = np.arange(3, 101)
    bug = [curve_estimate(N1, N2, "co", T, buggy=True) for T in Ts]
    cor = [curve_estimate(N1, N2, "co", T) for T in Ts]
    rich = [richardson(N1, N2, "co", T)[0] for T in Ts]
    save("fig17", dict(T=Ts.tolist(), buggy=bug, correct=cor, rich=rich, rho=rho, rho_co=lco))
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    ax.plot(Ts, bug, "-", color="#999999", lw=1.0, marker="+", ms=4, label="Eq. (22) as printed (sum from $i=1$)")
    ax.plot(Ts, cor, "-", color=C["co"], label="corrected estimator (sum from $i=0$)")
    ax.axhline(rho, color=C["truth"], ls="--", lw=1.0, label=r"$\rho_\mathcal{N}$ (shooting) $=3.13293$")
    ax.axhline(lco, color=C["m"], ls=":", lw=1.2, label=r"Calvo–Oller lower bound $=3.04702$")
    ax.set_xlabel("$T$"); ax.set_ylabel("approximation of $\\rho_\\mathcal{N}$")
    ax.set_ylim(2.4, 3.35); ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig17.pdf")); plt.close(fig)

# ------------------------------------------------------------------ convergence
def conv():
    N1 = (np.zeros(2), np.diag([1, .1])); N2 = (np.ones(2), np.diag([.1, 1]))
    Ts = np.array([4, 8, 16, 32, 64, 128, 256])
    out = {}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for k in CURVES:
        L = richardson(N1, N2, k, 4000)[0]
        e = [curve_estimate(N1, N2, k, T) - L for T in Ts]
        rr = [abs(richardson(N1, N2, k, T)[0] - L) for T in Ts]
        out[k] = dict(L=L, err=e, rich=rr)
        ax.loglog(Ts, e, "-o", ms=3, color=C[k], label=k if k != "co" else "C&O")
        ax.loglog(Ts, rr, ":", color=C[k], lw=1.0)
    ax.loglog(Ts, 2.0 / Ts ** 2.0, "k--", lw=0.8); ax.text(70, 2.0 / 70 ** 2 * 1.6, "$T^{-2}$", fontsize=8)
    ax.loglog(Ts, 0.5 / Ts ** 4.0, "k--", lw=0.8); ax.text(30, 0.5 / 30 ** 4 * 2.2, "$T^{-4}$", fontsize=8)
    ax.set_xlabel("$T$"); ax.set_ylabel(r"$|\tilde\rho_T(c)-L(c)|$")
    ax.legend(ncol=5, fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "conv.pdf")); plt.close(fig)
    save("conv", dict(T=Ts.tolist(), **out))

# ------------------------------------------------------------------ common covariance
def commoncov():
    D = np.linspace(0.05, 10, 60)
    rows = []
    for u in D:
        N1 = (np.zeros(2), np.eye(2)); N2 = (np.array([u, 0.]), np.eye(2))
        exact = np.sqrt(2) * np.arccosh(1 + u * u / 4)
        r = dict(Delta=u, exact=exact, co=rho_co(N1, N2), marg=lb_marginal(N1, N2),
                 sqrtJ=ub_sqrtJ(N1, N2), curve_co=richardson(N1, N2, "co", 256)[0])
        _, r["beta_sharp"], r["beta_simple"], r["beta_min"] = beta_bounds(N1, N2)
        rows.append(r)
    save("commoncov", rows)
    g = lambda k: np.array([r[k] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax[0].plot(D, g("exact"), color=C["truth"], label=r"$\rho_\mathcal{N}$ (exact) = marginal bound")
    ax[0].plot(D, g("co"), color=C["m"], ls=":", label=r"$\rho_{\mathrm{CO}}$")
    ax[0].plot(D, g("curve_co"), color=C["co"], label="length of C&O curve")
    ax[0].plot(D, g("beta_sharp"), color=C["e"], ls="--", label=r"$\beta$-certificate (Thm. 5)")
    ax[0].plot(D, g("sqrtJ"), color="#999999", ls="-.", lw=1, label=r"$\sqrt{D_J}=\Delta$")
    ax[0].set_ylim(0, 10); ax[0].set_xlabel(r"Mahalanobis distance $\Delta$"); ax[0].set_ylabel("distance")
    ax[0].legend(fontsize=6.8, loc="upper left")
    ax[1].plot(D, g("co") / g("exact") - 1, color=C["m"], ls=":", label=r"$\rho_{\mathrm{CO}}/\rho_\mathcal{N}-1$")
    ax[1].plot(D, g("curve_co") / g("exact") - 1, color=C["co"], label=r"$L(c_{\mathrm{CO}})/\rho_\mathcal{N}-1$")
    ax[1].plot(D, g("beta_sharp") / g("exact") - 1, color=C["e"], ls="--", label=r"$\beta$-certificate$/\rho_\mathcal{N}-1$")
    ax[1].set_xlabel(r"$\Delta$"); ax[1].set_ylabel("relative error"); ax[1].legend(fontsize=6.8)
    ax[1].set_ylim(-0.25, 0.6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "commoncov.pdf")); plt.close(fig)

# ------------------------------------------------------------------ Monte Carlo A (Nielsen scheme 1)
def mcA(reps=200):
    rng = np.random.default_rng(2023)
    out = {}
    for d in [1, 2, 3, 4, 5]:
        rows = []
        while len(rows) < reps:
            N1 = random_normal(d, rng); N2 = random_normal(d, rng)
            if min(np.linalg.eigvalsh(N1[1]).min(), np.linalg.eigvalsh(N2[1]).min()) < 1e-6:
                continue
            rho, res, meth, spread = reference_distance(N1, N2)
            if not np.isfinite(rho):
                raise RuntimeError("reference solver failed")  # never discard silently
            r = all_bounds(N1, N2, truth=rho)
            r.update(ref_res=res, ref_method=meth, ref_spread=spread)
            if meth != "single":
                print("  multishoot", d, meth, f"res={res:.1e} spread={spread:.1e}", flush=True)
            rows.append(r)
        out[d] = rows
        print("mcA d", d, "done", flush=True)
    save("mcA", out)

# ------------------------------------------------------------------ Monte Carlo B (separation)
def pairB(d, s, rng):
    u = rng.standard_normal(d); u /= np.linalg.norm(u)
    G = sym(rng.standard_normal((d, d))); G /= np.linalg.norm(G)
    return (np.zeros(d), np.eye(d)), (s * u, sym(expm(s * G)))

def mcB(reps=40):
    rng = np.random.default_rng(11)
    out = {}
    for d in [2, 5, 10]:
        for s in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]:
            rows = []; fails = 0
            while len(rows) < reps:
                N1, N2 = pairB(d, s, rng)
                rho, res, _ = shoot_robust(N1, N2)
                if res > 1e-8:
                    fails += 1
                    if fails > 3 * reps: break
                    continue
                rows.append(all_bounds(N1, N2, truth=rho))
            out[f"{d}_{s}"] = dict(d=d, s=s, rows=rows, fails=fails)
            print("mcB", d, s, len(rows), "fails", fails, flush=True)
    save("mcB", out)


# ------------------------------------------------------------------ verified bounds on all MC pairs
def replay_pairs():
    """Regenerate exactly the Monte Carlo pairs of mcA and mcB (no pair is
    discarded after sampling, so the random streams replay deterministically)."""
    rng = np.random.default_rng(2023); A = {}
    for d in [1, 2, 3, 4, 5]:
        L = []
        while len(L) < 200:
            N1 = random_normal(d, rng); N2 = random_normal(d, rng)
            if min(np.linalg.eigvalsh(N1[1]).min(), np.linalg.eigvalsh(N2[1]).min()) < 1e-6:
                continue
            L.append((N1, N2))
        A[str(d)] = L
    rng = np.random.default_rng(11); B = {}
    for d in [2, 5, 10]:
        for s in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]:
            B[f"{d}_{s}"] = [pairB(d, s, rng) for _ in range(40)]
    return A, B

def verify_mc():
    from verified import verified_lb_co, verified_lb_marg, verified_ub_chain
    A, B = replay_pairs()
    RA, RB = json.load(open(os.path.join(RES, "mcA.json"))), json.load(open(os.path.join(RES, "mcB.json")))
    out = []
    jobs = [(A[k][i], RA[k][i]) for k in A for i in range(200)] + \
           [(B[k][i], RB[k]["rows"][i]) for k in B for i in range(40)]
    for (N1, N2), r in jobs:
        assert abs(rho_co(N1, N2) - r["co"]) < 1e-9 * max(1, r["co"])   # replay check
        best = min(CURVES, key=lambda c: r["curve_" + c])
        vlb = max(verified_lb_co(N1, N2), verified_lb_marg(N1, N2))
        vub = verified_ub_chain(curve_points(N1, N2, best, 128), N1, N2)
        out.append(dict(d=len(N1[0]), truth=r["truth"], flb=r["lb"], fub_curve=r["curve_" + best],
                        vlb=vlb, vub=vub))
    save("verified_mc", out)
    v = lambda f: sum(f(o) for o in out)
    print("n", len(out), "vlb>truth", v(lambda o: o["vlb"] > o["truth"]), "vub<truth", v(lambda o: o["vub"] < o["truth"]),
          "max |vlb-flb|/rho", max(abs(o["vlb"] - o["flb"]) / o["truth"] for o in out),
          "max |vub-fub|/rho", max(abs(o["vub"] - o["fub_curve"]) / o["truth"] for o in out),
          "inf", v(lambda o: not np.isfinite(o["vub"])))

# ------------------------------------------------------------------ high dimension
def highd(reps=10, n_verify=10):
    """Certified intervals without reference values for d = 20, 50, 100."""
    from verified import verified_lb_co, verified_lb_marg, verified_ub_chain
    rng = np.random.default_rng(31)
    out = {}
    for d in [20, 50, 100]:
        for s in [0.5, 1.0, 2.0, 4.0]:
            rows = []
            for k in range(reps):
                N1, N2 = pairB(d, s, rng)
                t0 = time.time(); r = all_bounds(N1, N2, T=64); r["time"] = time.time() - t0
                r["width"] = (r["ub"] - r["lb"]) / r["lb"]
                if k < n_verify:
                    t0 = time.time()
                    r["vlb"] = max(verified_lb_co(N1, N2), verified_lb_marg(N1, N2))
                    r["vub"] = verified_ub_chain(curve_points(N1, N2, "co", 128), N1, N2)
                    r["vtime"] = time.time() - t0
                rows.append(r)
            out[f"{d}_{s}"] = dict(d=d, s=s, rows=rows)
            print("highd", d, s, "median width", np.median([r["width"] for r in rows]),
                  "time", np.median([r["time"] for r in rows]), flush=True)
    save("highd", out)

# ------------------------------------------------------------------ regime map
def regime():
    B = json.load(open(os.path.join(RES, "mcB.json")))
    series = [("co", r"$\rho_{\mathrm{CO}}$", C["m"], "--", "v"),
              ("marg", r"$\rho_{\mathrm{marg}}$", C["e"], "--", "v"),
              ("spc", r"$U_{\mathrm{SPC}}$", C["lin"], "-", "s"),
              ("beta_sharp", r"$U_\beta$", C["em"], "-", "^"),
              ("curve_em", r"$\tilde\rho_{128}(c_{em})$", "#8a5cc2", "-", "D"),
              ("curve_co", r"$\tilde\rho_{128}(c_{\mathrm{CO}})$", C["co"], "-", "o")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, d in zip(axs, [2, 5, 10]):
        ss = sorted({v["s"] for v in B.values() if v["d"] == d})
        for key, lab, col, ls, mk in series:
            y = [np.median([100 * (r[key] / r["truth"] - 1) for r in B[f"{d}_{s}"]["rows"]]) for s in ss]
            ax.plot(ss, y, ls=ls, marker=mk, ms=3.5, color=col, label=lab, lw=1.4)
        ax.axhline(0, color="#222222", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.1); ax.set_ylim(-80, 80)
        ax.set_title(f"$d={d}$", fontsize=9); ax.set_xlabel("separation $s$")
    axs[0].set_ylabel(r"median $100(\mathrm{bound}/\rho_\mathcal{N}-1)$")
    for ax in axs:
        ax.text(0.97, 0.53, "upper bounds", fontsize=7, color="#555555", transform=ax.transAxes, va="bottom", ha="right")
        ax.text(0.97, 0.47, "lower bounds", fontsize=7, color="#555555", transform=ax.transAxes, va="top", ha="right")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "regime.pdf")); plt.close(fig)

# ------------------------------------------------------------------ Nielsen (2023b) check and geodesic enclosure
def literature_check():
    from nielsen23b import kob_estimate, approx_rao_mvn
    from geo_enclosure import geodesic_enclosure
    A, B = replay_pairs()
    RA, RB = json.load(open(os.path.join(RES, "mcA.json"))), json.load(open(os.path.join(RES, "mcB.json")))
    jobs = [("A", k, A[k][i], RA[k][i]) for k in A for i in range(200)] + \
           [("B", k, B[k][i], RB[k]["rows"][i]) for k in B for i in range(40)]
    out = []
    for sch, key, (N1, N2), r in jobs:
        o = dict(scheme=sch, key=key, d=len(N1[0]), truth=r["truth"], co=r["co"], lb=r["lb"])
        for name, fn in (("kob128", lambda: kob_estimate(N1, N2, 128)), ("alg2", lambda: approx_rao_mvn(N1, N2, 1e-3))):
            try:
                with np.errstate(all="ignore"):
                    v = float(fn())
                o[name] = v if np.isfinite(v) else None
            except Exception:
                o[name] = None   # numerical breakdown of the method
        if r.get("ref_method", "single") == "single":
            sol = shoot_robust(N1, N2)
            if sol[1] < 1e-8:
                for T in (16, 32):
                    g = geodesic_enclosure(N1, N2, T=T, sol=sol)
                    o[f"geo{T}_lb"], o[f"geo{T}_ub"], o["gap"] = g["lb_cond"], g["ub"], g["gap"]
        out.append(o)
    save("literature_check", out)
    k = [o["kob128"] / o["truth"] - 1 for o in out if o["kob128"] is not None]
    a = [o["alg2"] / o["truth"] - 1 for o in out if o["alg2"] is not None]
    print("failures kob", sum(o["kob128"] is None for o in out), "alg2", sum(o["alg2"] is None for o in out))
    g = [o for o in out if "geo32_lb" in o]
    print("n", len(out), "kob rel err median", np.median(k), "max", max(k),
          "alg2 rel err median", np.median(a), "max", max(a), "frac alg2 > 1e-3", np.mean([x > 1e-3 for x in a]))
    print("geo n", len(g), "viol", sum(o["geo32_lb"] > o["truth"] * (1 + 1e-12) or o["geo32_ub"] < o["truth"] for o in g),
          "width32 median", np.median([(o["geo32_ub"] - o["geo32_lb"]) / o["truth"] for o in g]),
          "max", max((o["geo32_ub"] - o["geo32_lb"]) / o["truth"] for o in g))

if __name__ == "__main__":
    parts = sys.argv[1:] or ["examples", "fig17", "conv", "commoncov", "mcA", "mcB"]
    for p in parts:
        t0 = time.time(); globals()[p](); print(p, "took", round(time.time() - t0, 1), "s", flush=True)
