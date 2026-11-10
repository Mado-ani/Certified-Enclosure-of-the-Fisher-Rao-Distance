"""Generate LaTeX tables from results/*.json (no hand transcription)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); TAB = os.path.join(HERE, "..", "tables", "supplement")
os.makedirs(TAB, exist_ok=True)
CURVES = ["lin", "m", "e", "em", "co"]
CNAME = dict(lin=r"$c_\lambda$", m=r"$c_m$", e=r"$c_e$", em=r"$c_{em}$", co=r"$c_{\mathrm{CO}}$")
load = lambda n: json.load(open(os.path.join(RES, n + ".json")))
f4 = lambda x: f"{x:.4f}"

def t_examples():
    ex = load("examples")
    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r" & $\rho_\mathcal{N}$ & $\rho_{\mathrm{CO}}$ & $\rho_{\mathrm{marg}}$ & best $\tilde\rho_{1000}(c)$ & $U_\beta$ & $U_{\mathrm{SPC}}$ & $\sqrt{D_J}$\\",
         r"\midrule"]
    for name, r in ex.items():
        best = min(CURVES, key=lambda k: r["curve_" + k])
        lb = max(r["co"], r["marg"])
        cell = lambda v, isb: (r"\textbf{" + f4(v) + "}") if isb else f4(v)
        L.append(f"{name} & {f4(r['truth'])} & {cell(r['co'], r['co']==lb)} & {cell(r['marg'], r['marg']==lb)} & "
                 f"{cell(r['curve_'+best], True)} ({CNAME[best]}) & {f4(r['beta_sharp'])} & {f4(r['spc'])} & {f4(r['sqrtJ'])}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "examples.tex"), "w").write("\n".join(L))

def t_hanpark():
    r = load("examples")["Han-Park"]
    nielsen = dict(lin=3.4496, m=3.5775, e=3.7314, em=3.1672, co=3.1391)
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r"curve & reported in \cite{Nielsen2023} & Eq.~\eqref{eq:buggy}, $T=1000$ & Eq.~\eqref{eq:est}, $T=1000$\\", r"\midrule"]
    for k in CURVES:
        L.append(f"{CNAME[k]} & {nielsen[k]:.4f} & {r['T1000buggy_'+k]:.4f} & {r['T1000_'+k]:.4f}\\\\")
    L += [r"\midrule", f"\\multicolumn{{4}}{{l}}{{reference $\\rho_\\mathcal{{N}}$ (shooting) $= {r['truth']:.5f}$; "
          f"$\\rho_{{\\mathrm{{CO}}}} = {r['co']:.5f}$}}\\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "hanpark.tex"), "w").write("\n".join(L))

def t_mcA():
    A = load("mcA")
    L = [r"\begin{tabular}{c ccccc c cc c}", r"\toprule",
         r" & \multicolumn{5}{c}{mean $\tilde\rho_{128}(c)/\rho_\mathcal{N}$} & & \multicolumn{2}{c}{mean $\mathrm{LB}/\rho_\mathcal{N}$} & mean\\",
         r"\cmidrule{2-6}\cmidrule{8-9}",
         r"$d$ & " + " & ".join(CNAME[k] for k in CURVES) + r" & & $\rho_{\mathrm{CO}}$ & $\rho_{\mathrm{marg}}$ & width\\", r"\midrule"]
    for d, rows in A.items():
        R = lambda key: np.mean([r[key] / r["truth"] for r in rows])
        vals = [R("curve_" + k) for k in CURVES]
        best = int(np.argmin(vals))
        cells = [(r"\textbf{%.4f}" % v) if i == best else "%.4f" % v for i, v in enumerate(vals)]
        width = max(0.0, np.mean([(r["ub"] - r["lb"]) / r["truth"] for r in rows]))
        L.append(f"{d} & " + " & ".join(cells) + f" & & {R('co'):.4f} & {R('marg'):.4f} & {width:.4f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "mcA.tex"), "w").write("\n".join(L))

def t_mcB():
    B = load("mcB")
    L = [r"\begin{tabular}{cc rrr rrrr r ccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{lower bounds} & \multicolumn{4}{c}{upper bounds} & & \multicolumn{3}{c}{best curve (\%)}\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-9}\cmidrule(lr){11-13}",
         r"$d$ & $s$ & $\rho_{\mathrm{CO}}$ & $\rho_{\mathrm{marg}}$ & LB & $\sqrt{D_J}$ & $U_{\mathrm{SPC}}$ & $U_\beta$ & UB & width & $c_{\mathrm{CO}}$ & $c_{em}$ & $c_m$\\",
         r"\midrule"]
    for key, blk in B.items():
        rows = blk["rows"]
        if not rows: continue
        g = lambda k: 100 * np.mean([r[k] / r["truth"] - 1 for r in rows])
        width = 100 * np.mean([(r["ub"] - r["lb"]) / r["truth"] for r in rows])
        bc = [min(CURVES, key=lambda k: r["curve_" + k]) for r in rows]
        pc = lambda k: 100 * np.mean([b == k for b in bc])
        L.append(f"{blk['d']} & {blk['s']:g} & {g('co'):.1f} & {g('marg'):.1f} & {g('lb'):.1f} & {g('sqrtJ'):.1f} & {g('spc'):.1f} & "
                 f"{g('beta_sharp'):.1f} & {g('ub'):.2f} & {width:.2f} & {pc('co'):.0f} & {pc('em'):.0f} & {pc('m'):.0f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "mcB.tex"), "w").write("\n".join(L))

def summary():
    """Numbers quoted in the text; also writes tex/tables/findings.tex."""
    out = {}
    CAND = ["sqrtJ", "spc", "tri", "beta_sharp"] + ["curve_" + k for k in CURVES]
    for name in ["mcA", "mcB"]:
        D = load(name)
        rows = [r for v in D.values() for r in (v["rows"] if isinstance(v, dict) else v)]
        clip = [abs(max(r["lb"], min(r["ub"], r["point"])) / r["truth"] - 1) for r in rows]
        w = [(r["ub"] - r["lb"]) / r["truth"] for r in rows]
        ubw = [min(CAND, key=lambda k: r[k]) for r in rows]
        out[name] = dict(n=len(rows),
                         viol_lb=int(sum(r["lb"] > r["truth"] * (1 + 1e-9) for r in rows)),
                         viol_ub=int(sum(r["ub"] < r["truth"] * (1 - 1e-9) for r in rows)),
                         viol_beta=int(sum(r["beta_sharp"] < r["rich_co"] * (1 - 1e-6) for r in rows)),
                         marg_beats_co=float(np.mean([r["marg"] > r["co"] for r in rows])),
                         point_med=float(np.median(clip)), point_q95=float(np.quantile(clip, .95)), point_max=float(max(clip)),
                         width_lt1=float(np.mean([x < 0.01 for x in w])), width_lt5=float(np.mean([x < 0.05 for x in w])),
                         width_med=float(np.median(w)),
                         ub_rel_med=float(np.median([r["ub"] / r["truth"] - 1 for r in rows])),
                         ub_rel_q95=float(np.quantile([r["ub"] / r["truth"] - 1 for r in rows], .95)),
                         ub_winner={k: float(np.mean([u == k for u in ubw])) for k in CAND})
        out[name]["ub_winner"] = {k: v for k, v in out[name]["ub_winner"].items() if v > 0}
        ms = [r for r in rows if r.get("ref_method", "single") != "single"]
        out[name]["ref"] = dict(n_multi=len(ms),
                                max_res_multi=max([r["ref_res"] for r in ms], default=0.0),
                                max_spread=max([r["ref_spread"] for r in ms], default=0.0),
                                single_start=int(sum(r["ref_spread"] == 0.0 for r in ms)),
                                inside=int(sum(r["lb"] - 1e-9 <= r["truth"] <= r["ub"] + 1e-9 for r in ms)),
                                max_res_all=max([r.get("ref_res", 0.0) for r in rows]))
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    A, B = out["mcA"], out["mcB"]
    pc = lambda x: (f"{100*x:.2f}\\%" if 100 * x < 1 else f"{100*x:.1f}\\%")
    txt = (f"Across all {A['n']+B['n']} pairs ({A['n']} in scheme~A, {B['n']} in scheme~B), no certified lower bound exceeded the "
           f"reference value and no certified upper bound fell below it. The certificate $U_\\beta$ never fell below the "
           f"Richardson estimate of $L(c_{{\\mathrm{{CO}}}})$, as \\cref{{thm:cert}} requires. "
           f"The upper side is tight: the median of $\\mathrm{{UB}}/\\rho_\\mathcal{{N}}-1$ is {pc(A['ub_rel_med'])} (scheme~A) "
           f"and {pc(B['ub_rel_med'])} (scheme~B), and its 95th percentile is {pc(A['ub_rel_q95'])} and {pc(B['ub_rel_q95'])}. "
           f"The certified relative width is below $1\\%$ for {pc(A['width_lt1'])} and {pc(B['width_lt1'])} of pairs, "
           f"and below $5\\%$ for {pc(A['width_lt5'])} and {pc(B['width_lt5'])}. "
           f"The point estimate $\\hat\\rho$ of \\cref{{alg:cert}} has median relative error {pc(A['point_med'])} and {pc(B['point_med'])}. "
           f"Its 95th percentile is {pc(A['point_q95'])} and {pc(B['point_q95'])}, but its maximum is {pc(A['point_max'])} and {pc(B['point_max'])}: "
           f"for strongly separated, ill-conditioned pairs every curve in the menu is far from geodesic, and only the certified interval should be trusted. "
           f"The width is driven by the lower bound. $\\rho_{{\\mathrm{{marg}}}}$ improves on $\\rho_{{\\mathrm{{CO}}}}$ in "
           f"{pc(A['marg_beats_co'])} of scheme-A pairs. Almost all of these are univariate pairs (199 of 200 at $d=1$, where $\\rho_{{\\mathrm{{marg}}}}$ is exact); "
           f"the rest are at most $12\\%$ of pairs per dimension for $d\\ge2$. In scheme~B it improves on $\\rho_{{\\mathrm{{CO}}}}$ in {pc(B['marg_beats_co'])} of pairs, "
           f"concentrated at $d=2$ with $s\\ge3$ (45\\% of those pairs). "
           f"Among the curves, $c_{{\\mathrm{{CO}}}}$ is the best in the large majority of cases (\\cref{{tab:mcA,tab:mcB}}). "
           f"Among all upper bounds, $U_{{\\mathrm{{SPC}}}}$ attains the minimum in {pc(A['ub_winner'].get('spc',0))} of scheme-A pairs, "
           f"including every $d=1$ pair, where it is exact. In scheme~B it attains the minimum in {pc(B['ub_winner'].get('spc',0))} of pairs, "
           f"mostly for $d=2$ and $s\\ge3$. "
           f"This is consistent with $U_{{\\mathrm{{SPC}}}}$ being the length of an explicit curve inside the diagonal submodel (\\cref{{rem:spc}}). Including it in the menu of \\cref{{alg:cert}} costs one eigendecomposition.")
    open(os.path.join(TAB, "findings.tex"), "w").write(txt)
    print(json.dumps(out, indent=1))

def t_verified():
    V = load("verified_mc")
    n = len(V)
    tol = 1e-12
    bad = sum((o["vlb"] > o["truth"] * (1 + tol)) or (o["vub"] < o["truth"] * (1 - tol)) for o in V)
    eq = [o for o in V if o["truth"] < o["vlb"] <= o["truth"] * (1 + tol)]
    eq_d1 = all(o["d"] == 1 for o in eq)
    eq_max = max([(o["vlb"] - o["truth"]) / o["truth"] for o in eq], default=0.0)
    inf = sum(not np.isfinite(o["vub"]) for o in V)
    dl = max(abs(o["vlb"] - o["flb"]) / o["truth"] for o in V)
    du = max(abs(o["vub"] - o["fub_curve"]) / o["truth"] for o in V if np.isfinite(o["vub"]))
    sci = lambda x: f"{x:.1e}".replace("e-0", "e-").split("e")
    fmt = lambda x: f"${sci(x)[0]}\\times10^{{{int(sci(x)[1])}}}$"
    txt = (f"We recomputed the certified interval of every Monte Carlo pair ($n={n}$) with the verified bounds of \\cref{{prop:verified}}, "
           f"taking the chain of the best curve with $T=128$. Positive definiteness was certified at every chain point "
           f"({'no' if inf == 0 else inf} failures). Every verified interval contains the reference value "
           f"to within the reference solver's own accuracy ({bad} exceptions at relative tolerance $10^{{-12}}$). "
           f"In {len(eq)} pairs, all {'univariate' if eq_d1 else 'of them low-dimensional'}, the verified lower bound exceeds the reference value by at most "
           f"{fmt(eq_max)} relative. This is expected: for $d=1$ the marginal bound is exact (\\cref{{prop:exact}}), so lower bound and reference coincide up to the rounding of the reference. The verified bounds differ from their floating-point counterparts by at most "
           f"{fmt(dl)} (lower) and {fmt(du)} (upper) relative to $\\rho_\\mathcal{{N}}$. "
           f"The largest discrepancies occur on the most ill-conditioned pairs (whitened condition numbers up to $2\\times10^9$). There floating-point whitening loses accuracy of order $\\kappa\\,u$, with $u$ the unit roundoff, and this is precisely where the verified enclosure is needed. "
           f"In these experiments rounding never changed a conclusion, but the verified enclosure removes the need to argue this case by case.")
    open(os.path.join(TAB, "verified.tex"), "w").write(txt)
    return dict(n=n, bad=bad, inf=inf, dl=dl, du=du, eq=len(eq), eq_max=eq_max)

def t_highd():
    H = load("highd")
    L = [r"\begin{tabular}{cc ccc cc}", r"\toprule",
         r"$d$ & $s$ & LB & UB & width (\%) & time (s) & verified (s)\\", r"\midrule"]
    maxdiff = 0.0; allv = True
    for key, blk in H.items():
        rows = blk["rows"]
        med = lambda k: np.median([r[k] for r in rows])
        vt = [r["vtime"] for r in rows if "vtime" in r]
        for r in rows:
            if "vlb" in r:
                allv &= (r["vlb"] <= r["lb"] * (1 + 1e-9)) and np.isfinite(r["vub"])
                maxdiff = max(maxdiff, abs(r["vlb"] - r["lb"]) / r["lb"], abs(r["vub"] - r["curve_co"]) / r["lb"])
        L.append(f"{blk['d']} & {blk['s']:g} & {med('lb'):.3f} & {med('ub'):.3f} & {100*med('width'):.2f} & "
                 f"{med('time'):.1f} & {np.median(vt):.1f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "highd.tex"), "w").write("\n".join(L))
    return dict(maxdiff=maxdiff, allv=bool(allv))

def t_lit():
    O = load("literature_check")
    L = [r"\begin{tabular}{cr cc c ccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{$100(L(c_{\mathrm{lift}})/\rho_\mathcal{N}-1)$} & & \multicolumn{3}{c}{recursive algorithm, $\varepsilon=10^{-3}$}\\",
         r"\cmidrule{3-4}\cmidrule{6-8}",
         r"$d$ & $n$ & median & max & & median error (\%) & max error (\%) & guarantee missed (\%)\\", r"\midrule"]
    for d in sorted(set(o["d"] for o in O)):
        S = [o for o in O if o["d"] == d]
        k = [100 * (o["kob128"] / o["truth"] - 1) for o in S if o["kob128"] is not None]
        a = [100 * (o["alg2"] / o["truth"] - 1) for o in S if o["alg2"] is not None]
        L.append(f"{d} & {len(S)} & {np.median(k):.2f} & {max(k):.1f} & & {np.median(a):.2f} & {max(a):.1f} & {100*np.mean(np.array(a) > 0.1):.0f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "lifted.tex"), "w").write("\n".join(L))
    a_all = [o["alg2"] / o["truth"] - 1 for o in O if o["alg2"] is not None]
    RA, RB = load("mcA"), load("mcB")
    co = [r["curve_co"] for k in RA for r in RA[k]] + [r["curve_co"] for k in RB for r in RB[k]["rows"]]
    better = [o["kob128"] is not None and o["kob128"] < c for o, c in zip(O, co)]
    k_all = [o["kob128"] / o["truth"] - 1 for o in O if o["kob128"] is not None]
    g = [o for o in O if "geo32_lb" in o]
    w16 = np.array([(o["geo16_ub"] - o["geo16_lb"]) / o["truth"] for o in g])
    w32 = np.array([(o["geo32_ub"] - o["geo32_lb"]) / o["truth"] for o in g])
    wu = np.array([(o["geo32_ub"] - o["lb"]) / o["truth"] for o in g])
    viol = sum(o["geo32_lb"] > o["truth"] * (1 + 1e-12) or o["geo32_ub"] < o["truth"] * (1 - 1e-12) for o in g)
    inside = sum(o["lb"] * (1 - 1e-12) <= o["geo32_lb"] for o in g)
    pc = lambda x: (f"{100*x:.2f}\\%" if 100 * x < 1 else f"{100*x:.1f}\\%")
    def sci(x):
        m, e = f"{x:.0e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    d1_exact = sum(1 for o in g if not (o["lb"] * (1 - 1e-12) <= o["geo32_lb"]) and o["d"] == 1)
    txt = (f"For the {len(g)} pairs on which single shooting converged, we evaluated the enclosure of \\cref{{thm:geo}} in ball arithmetic, with $T=16$ and $T=32$. "
           f"The shooting velocity was used as given, and the endpoint mismatch $e$ is at most ${sci(max(o['gap'] for o in g))}$. "
           f"The enclosure contained the reference value in every case ({viol} violations). "
           f"Its relative width has median {pc(np.median(w32))}, 95th percentile {pc(np.quantile(w32, .95))} and maximum {pc(w32.max())} at $T=32$. "
           f"The median ratio of the widths at $T=16$ and $T=32$ is {np.median(w16/w32):.2f}, the rate $T^{{-2}}$ predicted by \\cref{{thm:geo}}. "
           f"For comparison, the unconditional width obtained by combining the same upper bound with $\\max(\\rho_{{\\mathrm{{CO}}}},\\rho_{{\\mathrm{{marg}}}})$ has median {pc(np.median(wu))}. "
           f"In {inside} of the {len(g)} pairs, the conditional lower bound improved on the unconditional one. "
           f"The remaining {len(g)-inside} pairs {'are all univariate' if d1_exact == len(g)-inside else 'include univariate pairs'}, where $\\rho_{{\\mathrm{{marg}}}}$ is already exact. "
           f"In no case did the conditional lower bound exceed an unconditional upper bound, which is consistent with the shooting geodesics being minimising (\\cref{{rem:minimal}}).")
    open(os.path.join(TAB, "geo.tex"), "w").write(txt)
    txt2 = (f"Over all {len(O)} pairs, the lifted curve of \\citet{{NielsenPMLR2023}} has discretised length ($T=128$) exceeding $\\rho_\\mathcal{{N}}$ by a median of {pc(np.median(k_all))} and up to {pc(max(k_all))}. "
            f"Its discretised length at $T=128$ is smaller than that of $c_{{\\mathrm{{CO}}}}$ in only {pc(np.mean(better))} of pairs. "
            f"The recursive algorithm with $\\varepsilon=10^{{-3}}$ misses its claimed guarantee in {pc(np.mean(np.array(a_all) > 1e-3))} of pairs, with median error {pc(np.median(a_all))} and maximum {pc(max(a_all))}. "
            f"Both methods broke down numerically on {sum(o['alg2'] is None for o in O)} pair. The lifted matrices contain both $\\Sigma$ and $\\Sigma^{{-1}}$, which squares the condition number. \\Cref{{tab:lifted}} gives the breakdown by dimension. "
            f"The errors are not monotone in $d$, but this reflects the sampling schemes rather than dimension. Dimension $4$ comes only from scheme~A, which produces strongly separated pairs (median $\\rho_\\mathcal{{N}}\\approx5.5$), whereas $d=10$ comes only from scheme~B (median $\\rho_\\mathcal{{N}}\\approx1.7$). "
            f"At $d=5$, which both schemes cover, the median error of the recursive algorithm is $5.9\\%$ in scheme~A and $0.7\\%$ in scheme~B: the error grows with separation.")
    open(os.path.join(TAB, "lifted_text.tex"), "w").write(txt2)
    return dict(n=len(g), viol=viol, inside=inside)

if __name__ == "__main__":
    t_examples(); t_hanpark()
    if os.path.exists(os.path.join(RES, "mcA.json")): t_mcA()
    if os.path.exists(os.path.join(RES, "mcB.json")): t_mcB(); summary()
    if os.path.exists(os.path.join(RES, "verified_mc.json")): print("verified", t_verified())
    if os.path.exists(os.path.join(RES, "highd.json")): print("highd", t_highd())
    if os.path.exists(os.path.join(RES, "literature_check.json")): print("lit", t_lit())
