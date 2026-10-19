"""第四十部分（補）：第一階段的「時段地圖」在保留期還在不在？（保留期已打開，這裡只做診斷、不再挑參數）→ results_part40b.txt
每個 (起點, IB 長度, 用法) 格子：研究期 vs 保留期 每筆 ATR 的等級相關；以及 A、B 兩塊高原在保留期的平均。"""
import numpy as np, pandas as pd
from nas_core import run, period, RESEARCH, HOLDOUT, hm

rows = []
for mode in (1, -1):
    for ib in (15, 30, 45, 60, 90, 120):
        for s0 in range(60, 22 * 60 + 1, 30):
            if s0 + ib + 30 > 23 * 60 + 50:
                continue
            X = run(s0, ib=ib, mode=mode)
            X["A"] = X.pnl / X.atr
            r, h = period(X, *RESEARCH), period(X, *HOLDOUT)
            rows.append(dict(mode=mode, ib=ib, s0=s0, res=r.A.mean(), hold=h.A.mean(), n_hold=len(h)))
G = pd.DataFrame(rows)
G.to_csv("part40b_map.csv", index=False)
out = []
for mode in (1, -1):
    g = G[G["mode"] == mode]
    out.append(f"用法 {mode:+d}：研究期 vs 保留期 每格每筆 ATR 等級相關 = {g.res.corr(g.hold, method='spearman'):+.2f}（{len(g)} 格）")
inA = (G["mode"] == 1) & G.s0.between(630, 960) & G.ib.isin([30, 45, 60])
inB = (G["mode"] == -1) & G.s0.between(390, 510) & G.ib.isin([60, 90])
inX = (G["mode"] == 1) & G.s0.between(120, 630) & (G.ib == 60)
for lab, m in (("A 高原（順勢 10:30~16:00 × IB30~60）", inA), ("B 高原（反向 06:30~08:30 × IB60~90）", inB), ("XAU 時段（順勢 02:00~10:30 × IB60）", inX)):
    g = G[m]
    out.append(f"{lab}：研究期平均 {g.res.mean() * 100:+.2f}%ATR、正的格子 {(g.res > 0).mean():.0%}｜保留期平均 {g.hold.mean() * 100:+.2f}%ATR、正的格子 {(g.hold > 0).mean():.0%}")
P = G[G["mode"] == 1].pivot(index="s0", columns="ib", values="hold").mul(100).round(1)
P.index = [hm(s) for s in P.index]
out.append("\n保留期 順勢 每筆 %ATR（列 = IB 起點，欄 = IB 長度）\n" + P.to_string())
txt = "\n".join(out)
open("results_part40b.txt", "w").write(txt + "\n")
print(txt)
