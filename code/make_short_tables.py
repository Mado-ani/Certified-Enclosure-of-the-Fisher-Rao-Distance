"""Compact tables for the short (main-text) version, generated from results/*.json."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); OUT = os.path.join(HERE, "..", "tables", "main")
os.makedirs(OUT, exist_ok=True)
load = lambda n: json.load(open(os.path.join(RES, n + ".json")))
A, B, O = load("mcA"), load("mcB"), load("literature_check")
rowsA = [r for v in A.values() for r in v]; rowsB = [r for v in B.values() for r in v["rows"]]
LA = [o for o in O if o["scheme"] == "A"]; LB = [o for o in O if o["scheme"] == "B"]
V = load("verified_mc")
pc = lambda x: f"{100*x:.2f}"
def block(rows, L):
    w = [(r["ub"] - r["lb"]) / r["truth"] for r in rows]
    ub = [r["ub"] / r["truth"] - 1 for r in rows]
    lb = [1 - r["lb"] / r["truth"] for r in rows]
    pt = [abs(max(r["lb"], min(r["ub"], r["point"])) / r["truth"] - 1) for r in rows]
    g = [o for o in L if "geo32_lb" in o]
    gw = [(o["geo32_ub"] - o["geo32_lb"]) / o["truth"] for o in g]
    cont = sum(r["lb"] <= r["truth"] * (1 + 1e-9) and r["ub"] >= r["truth"] * (1 - 1e-9) for r in rows)
    return dict(n=len(rows), cont=cont, lb=np.median(lb), ub=np.median(ub), w=np.median(w), w95=np.quantile(w, .95),
                pt=np.median(pt), ng=len(g), gw=np.median(gw), gw95=np.quantile(gw, .95),
                gcont=sum(o["geo32_lb"] <= o["truth"] * (1 + 1e-12) and o["geo32_ub"] >= o["truth"] * (1 - 1e-12) for o in g))
a, b = block(rowsA, LA), block(rowsB, LB)
T = [r"\begin{tabular}{lcc}", r"\toprule", r"Performance measure & Scheme A & Scheme B\\", r"\midrule",
     f"Number of pairs & {a['n']} & {b['n']}\\\\",
     f"Coverage of $[\\mathrm{{LB}},\\mathrm{{UB}}]$ & {a['cont']}/{a['n']} & {b['cont']}/{b['n']}\\\\",
     f"Median lower-bound slack, $1-\\mathrm{{LB}}/\\rho_\\mathcal{{N}}$ (\\%) & {pc(a['lb'])} & {pc(b['lb'])}\\\\",
     f"Median upper-bound excess, $\\mathrm{{UB}}/\\rho_\\mathcal{{N}}-1$ (\\%) & {pc(a['ub'])} & {pc(b['ub'])}\\\\",
     f"Certified width $(\\mathrm{{UB}}-\\mathrm{{LB}})/\\rho_\\mathcal{{N}}$ (\\%): median / 95th pct. & {pc(a['w'])} / {pc(a['w95'])} & {pc(b['w'])} / {pc(b['w95'])}\\\\",
     f"Point estimate $\\hat\\rho$: median relative error (\\%) & {pc(a['pt'])} & {pc(b['pt'])}\\\\",
     r"\midrule",
     f"Geodesic enclosure ($T=32$): pairs evaluated & {a['ng']} & {b['ng']}\\\\",
     f"Geodesic enclosure: coverage & {a['gcont']}/{a['ng']} & {b['gcont']}/{b['ng']}\\\\",
     f"Geodesic enclosure: width (\\%), median / 95th pct. & {pc(a['gw'])} / {pc(a['gw95'])} & {pc(b['gw'])} / {pc(b['gw95'])}\\\\",
     r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "summary.tex"), "w").write("\n".join(T))
# lifted vs our curves, by scheme
RA = [r["curve_co"] for k in A for r in A[k]] + [r["curve_co"] for k in B for r in B[k]["rows"]]
def lift(L, co):
    k = [100 * (o["kob128"] / o["truth"] - 1) for o in L if o["kob128"] is not None]
    al = [100 * (o["alg2"] / o["truth"] - 1) for o in L if o["alg2"] is not None]
    c = [100 * (x / o["truth"] - 1) for o, x in zip(L, co)]
    return np.median(c), np.median(k), max(k), np.median(al), max(al), 100 * np.mean(np.array(al) > 0.1)
coA = RA[:len(LA)]; coB = RA[len(LA):]
la, lb_ = lift(LA, coA), lift(LB, coB)
T2 = [r"\begin{tabular}{lcc}", r"\toprule", r"Excess over $\rho_\mathcal{N}$ (\%) & Scheme A & Scheme B\\", r"\midrule",
      f"$c_{{\\mathrm{{CO}}}}$ curve, median & {la[0]:.2f} & {lb_[0]:.2f}\\\\",
      f"Lifted curve, median / max & {la[1]:.2f} / {la[2]:.1f} & {lb_[1]:.2f} / {lb_[2]:.1f}\\\\",
      f"Recursive scheme, median / max & {la[3]:.2f} / {la[4]:.1f} & {lb_[3]:.2f} / {lb_[4]:.1f}\\\\",
      f"Guarantee fails (\\% of pairs) & {la[5]:.0f} & {lb_[5]:.0f}\\\\",
      r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "lifted.tex"), "w").write("\n".join(T2))
# high-d compact
H = load("highd")
ss = [0.5, 1.0, 2.0, 4.0]
T3 = [r"\begin{tabular}{c cccc cc}", r"\toprule",
      r" & \multicolumn{4}{c}{width (\%) at $s=$} & \multicolumn{2}{c}{time (s)}\\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
      r"$d$ & $0.5$ & $1$ & $2$ & $4$ & float & verified\\", r"\midrule"]
for d in [20, 50, 100]:
    w = [np.median([r["width"] for r in H[f"{d}_{s}"]["rows"]]) * 100 for s in ss]
    t = np.median([r["time"] for s in ss for r in H[f"{d}_{s}"]["rows"]])
    vt = np.median([r["vtime"] for s in ss for r in H[f"{d}_{s}"]["rows"]])
    T3.append(f"{d} & " + " & ".join(f"{x:.2f}" for x in w) + f" & {t:.1f} & {vt:.1f}\\\\")
T3 += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "highd.tex"), "w").write("\n".join(T3))
print(a); print(b); print(la, lb_)
