"""第二十三部分：版本 A 在 2025 失效的解法 —— 行情狀態開關。

診斷（part22_trades.csv）：2025 的虧損集中在 6、9、11、12 月，時段 / 方向 / IB 大小 / 星期都沒有穩定的結構差異，
屬於「行情狀態」問題，所以只比較狀態類的事前開關。所有候選先列出來，三段（2023/6~2024、2025、2026）同時看，
不在 2025 上單獨挑最佳值。

候選（都只用「今天以前」的資料，以交易日為單位決定今天做不做）：
  EQ_n   影子權益開關：版本 A 照常在影子帳本跑，最近 n 個交易日影子總 R > 0 才下實單（借 FadeEA 的影子帳本概念）
  TR_n   趨勢日比例：最近 n 天日效率 |收−開|/(高−低) 的平均 >= 全部過去天數的中位數才做（A 是吃趨勢日的策略）
  PE25   第二十部分：前一天效率在過去 25% 最高 → 不做
另外每一組都可以加「同時持有最多 3 筆」（第二十二部分）。
"""
import numpy as np
import pandas as pd
from load import load

T = pd.read_csv("part22_trades.csv", parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in").reset_index(drop=True)
PERIODS = [("2023/6~2024", "2023-06-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01"),
           ("2026/1~10", "2026-01-01", "2026-12-31")]

m1 = load("data/XAUUSD_M1_2023_2026.csv")
m1["day"] = (m1.index - pd.Timedelta(hours=1)).normalize()
Dd = m1.groupby("day").agg(o=("open", "first"), h=("high", "max"), l=("low", "min"), c=("close", "last"), n=("close", "size"))
Dd = Dd[Dd.n >= 1000]
Dd["eff"] = (Dd.c - Dd.o).abs() / (Dd.h - Dd.l)
days = Dd.index

daily_R = T.groupby("day").R.sum().reindex(days, fill_value=0.0)


def cap(X, k=3):
    keep, opened = [], []
    for r in X.itertuples():
        opened = [e for e in opened if e > r.t_in]
        if len(opened) < k:
            keep.append(r.Index); opened.append(r.exit_time)
    return X.loc[keep]


def rules():
    R = {"基準（A 原樣）": pd.Series(True, index=days)}
    for n in (10, 20, 40, 60):
        R[f"EQ_{n} 影子 {n} 天 R>0"] = daily_R.rolling(n).sum().shift(1).fillna(1) > 0
    for n in (10, 20):
        tr = Dd.eff.rolling(n).mean().shift(1)
        R[f"TR_{n} 趨勢日比例>=中位"] = (tr >= tr.expanding(60).median()).fillna(True)
    pe = Dd.eff.shift(1)
    R["PE25 前日效率不在前25%"] = (pe < pe.expanding(60).quantile(0.75)).fillna(True)
    return R


def stats(X):
    out = []
    for name, a, b in PERIODS:
        g = X[(X.day >= a) & (X.day < b)]
        out.append(f"{g.R.mean():+.3f} n{len(g):<5} Σ{g.R.sum():+5.0f}")
    X = X.sort_values("exit_time"); eq = X.R.cumsum(); dd = (eq - eq.cummax()).min()
    return " | ".join(out) + f" | Σ{X.R.sum():+5.0f} DD{dd:5.0f} Σ/DD {X.R.sum() / -dd:4.1f}"


lines = ["規則" + " " * 22 + " | ".join(f"{p:<20}" for p, _, _ in PERIODS) + " | 全期"]
for cp in (False, True):
    lines.append(f"\n## {'加上同時最多 3 筆' if cp else '不限同時持有'}")
    for name, ok in rules().items():
        X = T[T.day.map(ok).fillna(True).astype(bool)].reset_index(drop=True)
        if cp:
            X = cap(X)
        lines.append(f"{name:<24} {stats(X)}")
for key in ("EQ_20 影子 20 天 R>0", "EQ_60 影子 60 天 R>0"):
    ok = rules()[key]
    X = cap(T[T.day.map(ok).astype(bool)].reset_index(drop=True))
    lines.append(f"\n## {key} + 最多 3 筆，逐年")
    for y, g in X.groupby(X.day.dt.year):
        m = g.groupby(g.day.dt.to_period('M')).R.sum()
        lines.append(f"  {y}: n={len(g)} 每筆 {g.R.mean():+.3f}R  Σ{g.R.sum():+.0f}R  正月 {(m > 0).sum()}/{len(m)}")
    lines.append("  實單關閉的交易日比例：" + "  ".join(f"{y}: {(~ok)[ok.index.year == y].mean():.0%}" for y in sorted(set(days.year))))
txt = "\n".join(lines)
print(txt)
open("results_part23.txt", "w").write(txt + "\n")
