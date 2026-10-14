"""第二十五部分 (1b)：定案版 A + 日線盤面過濾（事後套在 part25_versionA_trades.csv 上，過濾規則事先固定，不調參）。
D 同向  ：只做和日線狀態同方向的單（日線盤整 → 不做）
D 不逆向：日線多頭不做空、日線空頭不做多，盤整兩邊都做
輸出：python3 part25_versionA_regime.py > results_part25_versionA_regime.txt
"""
import numpy as np
import pandas as pd
from load import load
from part24_common import daily_regime

T = pd.read_csv("part25_versionA_trades.csv", parse_dates=["day", "t_in"])
m15 = load("data/XAUUSD_M15_2023_2026.csv")
reg = pd.Series(daily_regime(m15), index=m15.index)
T["reg"] = reg.reindex(T.t_in, method="ffill").to_numpy()
T["R"] = T.pnl / T.risk


def line(g):
    r = g.R
    if len(r) < 5:
        return f"n={len(r)}"
    day = g.groupby("day").R.sum()
    return (f"n={len(r):5d} 每筆 {r.mean():+.3f}R (t={r.mean() / r.std() * np.sqrt(len(r)):+.2f}) "
            f"總 {r.sum():+7.1f}R 日Sharpe {day.mean() / day.std() * np.sqrt(252):+.2f}")


oos = T.day < "2026-01-01"
for name, m in (("不過濾", np.ones(len(T), bool)), ("D 同向", T.side == T.reg),
                ("D 不逆向", (T.side == T.reg) | (T.reg == 0))):
    g = T[m]
    print(f"\n### {name}")
    print(f"  樣本外 2023/8~2025/12: {line(g[oos[m]])}")
    print(f"  研究期 2026          : {line(g[~oos[m]])}")
    for y, gy in g.groupby(g.day.dt.year):
        print(f"    {y}: {line(gy)}   多 {gy[gy.side == 1].R.mean():+.3f} / 空 {gy[gy.side == -1].R.mean():+.3f}（空 {(gy.side == -1).sum()} 筆）")
print("\n### 方向 × 日線狀態（樣本外）")
for s in (1, -1):
    for r in (1, 0, -1):
        print(f"  {'多' if s == 1 else '空'} 日線{ {1: '多頭', 0: '盤整', -1: '空頭'}[r] }: {line(T[oos & (T.side == s) & (T.reg == r)])}")
