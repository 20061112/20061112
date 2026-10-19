"""第四十三部分（續）：方向改成 SMA30 + 10 日斜率後，(a) 回頭檢查第一階段的時段 / K 線中心是否還在高原中間；(b) peff、volr 門檻掃描；(c) 套在鄰近設定上是否一致改善。研究期 only。"""
import numpy as np, pandas as pd
from piv_core import run, summ, direction, period, RESEARCH, D

DIR = direction(30, 10)
pd.set_option("display.width", 250)
rows = []
for k in (5, 10, 15):
    for h0 in (1, 2):
        for ln in (2, 3, 4):
            s = summ(run(k=k, hours=(h0, h0 + ln), dirs=DIR))
            rows.append(dict(k=k, hours=f"{h0:02d}-{h0 + ln:02d}", n=s["n"], mR=s["mR"], dt=s["dt"], hpos=s["hpos"], L=s["L"], S=s["S"]))
print("(a) SMA30/10 方向下的時段 × K 線\n" + pd.DataFrame(rows).round(3).to_string(index=False))

X = run(k=10, hours=(1, 4), dirs=DIR).join(D[["peff", "volr"]], on="day")
R0 = period(X, *RESEARCH)
rows = [dict(var="P30（SMA30/10）", **summ(X))]
for f in ("peff", "volr"):
    for q in (0.5, 0.67, 0.75, 0.8, 0.9):
        thr = R0[f].quantile(q)
        rows.append(dict(var=f"{f} ≤ q{q} ({thr:.2f})", **summ(X[X[f] <= thr])))
print("\n(b) 濾網門檻\n" + pd.DataFrame(rows).drop(columns=["nL", "nS"]).round(3).to_string(index=False))

out = []
for n in (20, 30, 50):
    for sl in (5, 10, 20):
        for k in (5, 10, 15):
            Y = run(k=k, hours=(1, 4), dirs=direction(n, sl)).join(D[["peff", "volr"]], on="day")
            a, b, c = summ(Y), summ(Y[Y.peff <= 0.85]), summ(Y[Y.volr <= 1.29])
            out.append((n, sl, k, round(a["dt"], 2), a["hpos"], round(b["dt"], 2), b["hpos"], round(c["dt"], 2), c["hpos"]))
O = pd.DataFrame(out, columns="sma slope k dt hpos dt_peff hpos_peff dt_volr hpos_volr".split())
print("\n(c) 鄰近設定（SMA × 斜率 × K 線）\n" + O.to_string(index=False))
print(f"peff ≤ 0.85 改善比例 {(O.dt_peff > O.dt).mean():.0%}（中位 {(O.dt_peff - O.dt).median():+.2f}）；"
      f"volr ≤ 1.29 改善比例 {(O.dt_volr > O.dt).mean():.0%}（中位 {(O.dt_volr - O.dt).median():+.2f}）")
