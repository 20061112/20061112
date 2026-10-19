"""第四十一部分 第一階段：時框 —— K 線週期 × 進場小時 × 方向規則。研究期 only。→ part41_piv_stage1.csv
其餘照 XAU M15 核心（日線 SMA50 + 斜率定方向、前一根高低觸價、停損 2×ATR_k、RR 3、當天收盤平倉、無 B 條件、無上限 / 權益濾網）。
每格同時跑「對照」：同一小時每根 K 棒開盤直接順方向進場（同停損 / 停利）—— 突破訊號要比對照好才算有東西，不然只是日線方向的 beta。
"""
import numpy as np, pandas as pd
from piv_core import run, summ, direction

DIRS = direction()
rows = []
for k in (5, 10, 15, 30, 60):
    for h in range(1, 23):
        for mode in (1, 2):
            X = run(k=k, mode=mode, hours=(h, h + 1), dirs=DIRS)
            s = summ(X)
            if s["n"] < 30:
                continue
            row = dict(k=k, hour=h, mode=mode, **s)
            if mode == 1:
                c = summ(run(k=k, mode=1, hours=(h, h + 1), dirs=DIRS, ctrl=True))
                row.update(ctrl_mR=c["mR"], ctrl_dt=c["dt"])
            rows.append(row)
G = pd.DataFrame(rows)
G["excess"] = G.mR - G.ctrl_mR
G.to_csv("part41_piv_stage1.csv", index=False)
pd.set_option("display.width", 250)
for mode, lab in ((1, "順日線方向"), (2, "不分方向")):
    g = G[G["mode"] == mode]
    print(f"\n=== {lab}：每筆 R（列 = 進場小時，欄 = K 線分鐘）| 四個半年為正個數 ===")
    print(pd.concat([g.pivot(index="hour", columns="k", values="mR").round(3),
                     g.pivot(index="hour", columns="k", values="hpos").add_prefix("h")], axis=1).to_string())
g = G[G["mode"] == 1]
print("\n=== 順日線方向：突破 − 對照（每筆 R）===")
print(g.pivot(index="hour", columns="k", values="excess").round(3).to_string())
