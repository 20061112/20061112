"""第二十五部分 (1)：定案版 A 的真正樣本外測試（2023/6 ~ 2025/12 的 M1，版本 A 研究時從未看過）。
規則、參數、滾動前推門檻的算法完全沿用 final_A.py，不做任何修改；只把資料換成 data/XAUUSD_M1_2023_2026.csv。
輸出：python3 part25_versionA_oos.py > results_part25_versionA.txt（逐筆 part25_versionA_trades.csv）
"""
import tpo
tpo.load_m1.__defaults__ = ("data/XAUUSD_M1_2023_2026.csv",)   # 在其他模組 import 之前換資料
import numpy as np
import pandas as pd
from part17_balance import gen
from final_A import CFG, walk_forward


def line(T):
    r, R = T.pnl, T.pnl / T.risk
    eq = r.cumsum()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    day = T.groupby(T.day).pnl.sum()
    return (f"n={len(T):5d} 勝率 {(r > 0).mean():.2f} 每筆 {R.mean():+.3f}R (t={R.mean() / R.std() * np.sqrt(len(R)):+.2f}) "
            f"PF={gp / gl:.2f} 總 {r.sum():+8.0f} 回撤 {(eq.cummax() - eq).max():6.0f} 日Sharpe {day.mean() / day.std() * np.sqrt(252):+.2f}")


if __name__ == "__main__":
    X = gen(**CFG)
    T, used = walk_forward(X)
    T.to_csv("part25_versionA_trades.csv", index=False)
    print("定案版 A，參數凍結，資料 2023/6~2026/10（門檻每月只用過去資料重算）")
    print(f"\n全期      : {line(T)}")
    print(f"樣本外 2023/8~2025/12（研究時沒看過）: {line(T[T.day < '2026-01-01'])}")
    print(f"研究期 2026          : {line(T[T.day >= '2026-01-01'])}")
    print("\n逐年：")
    for y, g in T.groupby(T.day.dt.year):
        print(f"  {y}: {line(g)}")
    print("\n逐半年（每筆 R）：")
    for p, g in T.groupby(T.day.dt.to_period("Q")):
        print(f"  {p}: n={len(g):4d} {(g.pnl / g.risk).mean():+.3f}R  總 {g.pnl.sum():+7.0f}")
    print("\n多 / 空：")
    for s, g in T.groupby("side"):
        print(f"  {'多' if s == 1 else '空'}: 樣本外 {line(g[g.day < '2026-01-01'])}")
    print("\n門檻（每月）：")
    print(used.round(3).to_string(index=False))
