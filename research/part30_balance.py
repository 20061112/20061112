"""第三十部分補充：哪個版本多空最平衡？每個版本逐年列出 做多 / 做空 的 PF（以 R 計）與每筆 R。"""
import numpy as np
import pandas as pd
from part29_dir_filter import BASES, A, L1, walk, dedup, pf

L0 = pd.read_pickle("part26_signals_L0.pkl")
L0["day"] = pd.to_datetime(L0.day)
V = dict(BASES)
V["L1 核心（不篩）"] = dedup(L1)
V["L0 觸價核心（不篩）"] = dedup(L0)
V["L0 + 反向尾巴（前推）"] = walk(L0, 0.9, 1.0, 0.0)
V["L1 + 尾巴 + 反向極值（前推）"] = walk(L1, 0.9, 0.9, 0.0)

rows = []
out = open("results_part30_balance.txt", "w")
out.write("多空平衡：每格 = 做多 PF(R) 每筆R / 做空 PF(R) 每筆R；平衡分數 = 8 個（年 × 方向）中 PF > 1 的個數\n")
for name, T in V.items():
    cells, good, worst = [], 0, np.inf
    for y, g in T.groupby(T.day.dt.year):
        lo, sh = g[g.side == 1].R, g[g.side == -1].R
        good += int(pf(lo) > 1) + int(pf(sh) > 1)
        worst = min(worst, pf(lo), pf(sh))
        cells.append(f"{y} 多 {pf(lo):.2f}({lo.mean():+.2f}) 空 {pf(sh):.2f}({sh.mean():+.2f})")
    lo, sh = T[T.side == 1].R, T[T.side == -1].R
    rows.append(dict(name=name, n=len(T), pf_all=pf(T.R), pf_long=pf(lo), pf_short=pf(sh), good=good, worst=worst))
    line = f"{name:24s} {len(T):5d}筆 PF(R) {pf(T.R):.2f} | 多 {pf(lo):.2f} 空 {pf(sh):.2f} | 年×方向 PF>1：{good}/8 最差 {worst:.2f}\n    " + " | ".join(cells)
    print(line)
    out.write(line + "\n")
out.close()
