"""第六十八部分：IB 突破核心（原版，規則一個都不改）在 2023/6 ~ 2024/10 的真正樣本外 → results_part68.txt、part68_trades.csv

資料：data/XAUUSD_M1_2023jun_2024sep.csv（2023/6/1 ~ 2024/9/30）+ 原本 2024/9 ~ 2026/10 M1；重疊的 2024/9 完全相同、時區一致。
規則同第六十七部分（IB 突破 + M5 收盤確認 + 前一天效率 > 0.633 不做，0.633 沿用研究期的數字）。
2023/6 ~ 2024/10 這 17 個月從未用來開發或挑選任何東西。
"""
import os
import pandas as pd
from load import load

COMB = "data/pseudo/XAUUSD_M1_2023_2026.csv"
if not os.path.exists(COMB):
    a = load("data/XAUUSD_M1_2023jun_2024sep.csv")
    b = load("data/XAUUSD_M1_2024_2026feb.csv")
    c = load("data/XAUUSD_M1_2026.csv")
    m = pd.concat([a, b[~b.index.isin(a.index)]])
    m = pd.concat([m, c[~c.index.isin(m.index)]]).sort_index()
    os.makedirs("data/pseudo", exist_ok=True)
    pd.DataFrame({"<DATE>": m.index.strftime("%Y.%m.%d"), "<TIME>": m.index.strftime("%H:%M:%S"), "<OPEN>": m.open,
                  "<HIGH>": m.high, "<LOW>": m.low, "<CLOSE>": m.close, "<TICKVOL>": m.tickvol.astype(int), "<VOL>": 0,
                  "<SPREAD>": m.spread.astype(int)}).to_csv(COMB, sep="\t", index=False)
os.environ["CORE_IB_DATA"] = COMB

import numpy as np                                       # noqa: E402
from core_ib import events, simulate, PEFF_MAX, DAYS    # noqa: E402
from part67_core_stats import enrich, block             # noqa: E402

PER = [("★ 樣本外 2023/6~2024/10", "2023-06-15", "2024-11-01"), ("  2023 下半年", "2023-06-15", "2024-01-01"),
       ("  2024/1~6", "2024-01-01", "2024-07-01"), ("  2024/7~10", "2024-07-01", "2024-11-01"),
       ("研究期 2024/11~2025/12", "2024-11-01", "2026-01-01"), ("2026/1~10", "2026-01-01", "2026-11-01"),
       ("全部 2023/6~2026/10", "2023-06-15", "2026-11-01")]

X = simulate(events())
T = enrich(X[(X.peff <= PEFF_MAX) & (X.day >= "2023-06-15")])
out = ["第六十八部分：IB 突破核心（原版）2023/6 ~ 2026/10 —— 每筆 1 盎司（0.01 手），美元，已扣點差",
       "（2023/6/1~14 為 ATR10 暖身，從 6/15 起算）"]
for lab, a, b in PER:
    S = T[(T.day >= a) & (T.day < b)]
    out.append(f"\n===== {lab} =====")
    out.append(block(S, [d for d in DAYS if a <= str(d.date()) < b]))

U = X[(X.day >= "2023-06-15") & (X.day < "2024-11-01")]
p_all, p_f = U.pnl, U.pnl[U.peff <= PEFF_MAX]
pf = lambda p: p[p > 0].sum() / -p[p <= 0].sum()
out.append(f"\n樣本外期間濾網效果：不加濾網 {len(p_all)} 筆 PF {pf(p_all):.2f} 每筆 {p_all.mean():+.2f}；"
           f"加濾網 {len(p_f)} 筆 PF {pf(p_f):.2f} 每筆 {p_f.mean():+.2f}；被濾掉的 {len(p_all) - len(p_f)} 筆 PF {pf(U.pnl[U.peff > PEFF_MAX]):.2f}")
S = T[T.day < "2024-11-01"]
out.append("樣本外 多 / 空：" + "、".join(f"{'多' if s == 1 else '空'} {len(g)} 筆 PF {pf(g.pnl):.2f} 每筆 {g.pnl.mean():+.2f}"
                                       for s, g in S.groupby("side")))
mo = T.groupby(T.day.dt.to_period("M")).agg(筆數=("pnl", "size"), PF=("pnl", pf), 總=("pnl", "sum"), 每筆R=("R", "mean"))
mo["累積"] = mo.總.cumsum()
out.append("\n===== 逐月 =====\n" + mo.round(2).to_string())
yr = T.groupby(T.day.dt.year).agg(筆數=("pnl", "size"), PF=("pnl", pf), 每筆=("pnl", "mean"), 每筆R=("R", "mean"), 總=("pnl", "sum"))
out.append("\n===== 逐年 =====\n" + yr.round(2).to_string())
T[T.day < "2024-11-01"].to_csv("part68_trades.csv", index=False)
txt = "\n".join(out)
open("results_part68.txt", "w").write(txt + "\n")
print(txt)
