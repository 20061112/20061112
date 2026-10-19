"""第四十五部分 第一階段：時框 —— K 線週期 × 型態 K 的小時（3 小時一格，滑動）× 訊號源（型態+z / 純 z 回到帶內）。研究期 only。→ part45_z_stage1.csv
其餘照 XAU 第四部分 D：z20 門檻 2、極值 10 根、突破進場 3 根內、停損極值外 0.1 ATR、2R、最多 16 根。"""
import numpy as np, pandas as pd
from z_core import run, summ

KS = [5, 10, 15, 20, 30, 60]
rows = []
for src in ("pattern", "zcross"):
    for k in KS:
        for h0 in range(1, 23):
            s = summ(run(k=k, src=src, hours=(h0, h0 + 3)))
            if s["n"] < 20:
                continue
            rows.append(dict(src=src, k=k, h0=h0, **s))
G = pd.DataFrame(rows)
G.to_csv("part45_z_stage1.csv", index=False)
pd.set_option("display.width", 250)
for src in ("pattern", "zcross"):
    g = G[G.src == src]
    print(f"\n=== {src}：每筆 R（列 = 型態 K 起始小時 h0，涵蓋 h0~h0+3；欄 = K 線分鐘）| 四個半年為正個數 ===")
    print(pd.concat([g.pivot(index="h0", columns="k", values="mR").round(3), g.pivot(index="h0", columns="k", values="hpos").add_prefix("h"),
                     g.pivot(index="h0", columns="k", values="n").add_prefix("n")], axis=1).to_string())
    print("全天：" + "  ".join(f"M{k}: {summ(run(k=k, src=src))['mR']:+.3f}R / t {summ(run(k=k, src=src))['dt']:+.2f}" for k in KS))
