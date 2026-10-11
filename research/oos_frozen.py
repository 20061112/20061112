"""第十部分：凍結參數的真正樣本外測試。
資料：data/XAUUSD_M15_2023_2026.csv（2023/1/3~2026/10/9）、data/XAUUSD_M1_2023_2026.csv（2023/6/1~2026/10/8），
取自本 repo 的 claude/zen-pasteur-xz0bdu 分支。研究時只用過 2026/2/2 之後的資料，
所以 2026/2/2 以前全部是「從未看過」的樣本外。所有策略參數一律沿用先前檔案，不做任何修改。
"""
import numpy as np
import pandas as pd
from load import load
import lonbrk
from h1_strategy import run_h1
from mr_features import candidate_table
from orb_fade import fade
from backtest import stats

OOS_END = pd.Timestamp("2026-02-02")
M15 = load("data/XAUUSD_M15_2023_2026.csv")


def by_year(T, col, tcol):
    y = T[tcol].dt.year
    g = T.groupby(y)[col]
    return "  ".join(f"{k}: {v.mean():+.2f} ({len(v)})" for k, v in g)


def line(T, col="pnl_R", tcol="entry_time"):
    r = T[col]
    eq = r.cumsum()
    m = T.groupby(T[tcol].dt.to_period("M"))[col].sum()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return (f"n={len(r):4d} 期望 {r.mean():+.3f}R t={r.mean() / r.std() * np.sqrt(len(r)):+.2f} 勝率 {(r > 0).mean():.2f} "
            f"PF={gp / gl:.2f} 累計 {r.sum():+6.1f}R 回撤 {(eq.cummax() - eq).max():.1f}R 正月 {(m > 0).sum()}/{len(m)}")


def report(name, T, col="pnl_R", tcol="entry_time"):
    oos, ins = T[T[tcol] < OOS_END], T[T[tcol] >= OOS_END]
    print(f"\n### {name}")
    print(f"  樣本外 2023~2026/1 : {line(oos, col, tcol)}")
    print(f"  研究期 2026/2~10   : {line(ins, col, tcol)}")
    print(f"  逐年（平均R, 筆數）: {by_year(T, col, tcol)}")


if __name__ == "__main__":
    # 1. LONBRK（M15 參考實作，參數凍結）
    bars = M15.copy(); bars["sp"] = bars.spread * 0.01
    L = lonbrk.trades(bars[["open", "high", "low", "close", "sp"]])
    report("LONBRK 倫敦開盤突破亞洲區間（M15）", L)
    L.to_csv("oos_lonbrk_trades.csv", index=False)

    # 2. LONBRK 用 M1 逐根（2023/6 起）
    m1 = load("data/XAUUSD_M1_2023_2026.csv"); m1["sp"] = m1.spread * 0.01
    L1 = lonbrk.trades(m1[["open", "high", "low", "close", "sp"]])
    report("LONBRK（M1 逐根，2023/6 起）", L1)

    # 3. 第二級：M15 型態 + 布林外軌收盤 / 三選二（第四部分規則，訊號可重疊）
    C = candidate_table(M15)
    report("M15 型態 + 極值收在布林外軌外（D）", C[C.close_outside == 1])
    three = (C.er5_20 >= 2).astype(int) + (C.bbw_pct <= .5).astype(int) + (C.close_outside == 1).astype(int)
    report("M15 型態 三選二（ER比/帶寬/外軌）", C[three >= 2])
    report("M15 全部型態（基準）", C)

    # 4. H1 衝刺竭盡反轉（第三部分）
    report("H1 衝刺竭盡反轉（M15）", run_h1(M15))

    # 5. 紐約 15:30 OR15 假突破觸價反向（M15，目標 = 區間另一側 / 時間出場）
    m15 = M15.copy(); m15["sp"] = m15.spread * 0.01
    for tg in (True, False):
        F = fade(m15, 15, (15, 30), 15, mode="touch", target=tg)
        report(f"紐約 OR15 假突破反向（{'目標=另一側' if tg else '持有到21:00'}）", F, tcol="day")
