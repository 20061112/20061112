"""第二十三部分 補充：代表參數（H1 BB20、2σ 進場、停損 2σ、收盤跌回中軌出場）的逐年多空、集中度與回撤。"""
import numpy as np
import pandas as pd

from load import load
from part23_bb_breakout import Data, simulate, HOLD

m1 = load("data/XAUUSD_M1_2023_2026.csv")
d = Data(m1, "H1", n=20)
tr = simulate(d, 2, 2, "trail_mid", HOLD["H1"])
tr.to_csv("part23_trades.csv", index=False)
tr["year"] = tr.time.dt.year
print("逐年 × 多空（平均 σ / 筆數）")
print(tr.pivot_table(index="year", columns="side", values="pnl_sd", aggfunc=["mean", "count"]).round(3))
print("\n2026（金價橫盤：4330 → 4118）逐月平均 σ")
t26 = tr[tr.year == 2026]
print(t26.groupby(t26.time.dt.month).pnl_sd.agg(["mean", "count"]).round(2).T)
q = tr.pnl_sd.sort_values(ascending=False).to_numpy()
print(f"\n筆數 {len(q)}，勝率 {(q > 0).mean():.1%}，中位數 {np.median(q):.3f}σ，平均 {q.mean():.3f}σ")
for k in (5, 10, 20):
    print(f"拿掉最好的 {k} 筆後平均：{q[k:].mean():.3f}σ")
eq = tr.pnl_r.cumsum().to_numpy()
dd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
lose = (tr.pnl_r < 0).astype(int).to_numpy()
runs = max((len(s) for s in "".join(map(str, lose)).split("0")), default=0)
print(f"以 R 計：總 {eq[-1]:.1f}R，平均 {tr.pnl_r.mean():.3f}R，最大回撤 {dd:.1f}R，最長連虧 {runs} 筆")
print("持倉根數（H1）中位數", tr.bars.median(), "出場原因", tr.why.value_counts(normalize=True).round(2).to_dict())
