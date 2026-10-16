"""趨勢任一成立（早期上車）的折衷：MA200 之下、靠 CUSUM / TSMOM63 上車的部分只開 2 倍；MA200 之上維持 3 倍。
歷史 + 區塊蒙地卡羅。輸出附加到 results_ideas.txt。"""
import numpy as np
import pandas as pd
import indicators as ind
import engine as E
from run_tests import q, OK0, Z, ABOVE, c, r, w_rule, w_cont, evaluate
import run_ideas_mc as MC

out = []
P = lambda s: (print(s, flush=True), out.append(s))
w0 = w_rule(OK0)
b, broll, bcal = evaluate(w0); b, broll, bcal = evaluate(w0, broll, bcal)


def target(d, z, above, early, half):
    t = pd.Series(0.0, index=d.index)
    t[early & (z < 0.5)] = 0.5 if half else 1.0
    t[above & (z < 0.5)] = 1.0
    return t


P("\n早期上車只開 2 倍（歷史）")
alts = {f"CUSUM h{h:.0f}": ind.cusum_state(r, 0.25, h) > 0 for h in [4.0, 6.0, 8.0]}
alts["TSMOM63"] = (ind.tsmom(c, 63) > 0).fillna(False)
for name, early in alts.items():
    for half in [False, True]:
        o, _, _ = evaluate(w_cont(target(q, Z, ABOVE, early, half)), broll, bcal)
        P(f"  {name:10s} {'2x' if half else '3x'} irr99 {o['irr99']:.3f} irr10 {o['irr10']:.3f} win10 {o['win10']:2d}/17 "
          f"dIRR10y {o['d_irr10y_med']:+.3f}(min {o['d_irr10y_min']:+.3f}) worst {o['worst']:.2f} g10min {o['g10_min']:.2f} "
          f"worstyr {o['worst_year']:.2f} sw {o['sw']:.1f}")

res = []
for k in range(300):
    d = MC.path(); cc = d.close; z = E.atr_z(d); above = cc > cc.rolling(200).mean()
    s0 = d.index[MC.WARM]
    wcont = lambda t: (lambda tt: [None])(0)
    def wc(t):
        t = t.shift(1).fillna(0).values; ws = d.week_start.values; w = np.zeros(len(t)); cur = 0.0
        for i in range(len(t)):
            cur = t[i] if (t[i] < cur or ws[i]) else cur; w[i] = cur
        return w
    row = {}
    W = {"BASE": wc(target(d, z, above, above & False, False))}
    for h in [4.0, 6.0, 8.0]:
        e = ind.cusum_state(d.r, 0.25, h) > 0
        W[f"CUSUM h{h:.0f} 2x"] = wc(target(d, z, above, e, True))
    W["TSMOM63 2x"] = wc(target(d, z, above, (ind.tsmom(cc, 63) > 0).fillna(False), True))
    for name, w in W.items():
        st = E.stats(E.run(d, w, start=s0, z=z, rebalance_band=0.1)); row[name] = st["irr"]; row[name + "|w"] = st["worst"]
    res.append(row)
R = pd.DataFrame(res)
P("區塊拼貼蒙地卡羅（早期上車只開 2 倍）")
for k in [c for c in R.columns if "|" not in c and c != "BASE"]:
    dd = R[k] - R.BASE; dw = R[k + "|w"] - R["BASE|w"]
    P(f"  {k:14s} IRR 差 中位 {dd.median():+.4f} 平均 {dd.mean():+.4f} 勝率 {(dd > 0).mean():4.0%} 5% {dd.quantile(.05):+.4f} | 最慘 差 中位 {dw.median():+.3f} 5% {dw.quantile(.05):+.3f}")
open("results_ideas.txt", "a").write("\n".join(out) + "\n")
