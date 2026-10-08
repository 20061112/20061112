"""重現策略回測與驗證：python3 run_strategy.py（約 1 分鐘）"""
import numpy as np
import pandas as pd
from load import load
from backtest import Cfg, resample, find_setups, backtest_m1, stats
from strategy import run

IS = pd.Timestamp("2026-09-01")
m1 = load("data/XAUUSD_M1_full.csv")


def line(s):
    return (f"n={s['n']:4d} 勝率={s['win']:.2f} 期望={s['expR']:+.3f}R t={s['t']:+.2f} "
            f"PF={s['pf']:.2f} 累計={s['totR']:+.1f}R 最大回撤={s['maxddR']:.1f}R")


def rep(name, tr):
    print(f"{name}\n  樣本內 6/29~8/31 : {line(stats(tr[tr.entry_time < IS]))}"
          f"\n  樣本外 9/1~10/8  : {line(stats(tr[tr.entry_time >= IS]))}")


print("== A. M15 規則逐步加入 ==")
rep("M15 基本（型態+趨勢段+突破進場）", run(m1, trend_max=1e9, er_max=9))
rep("+ 前段幅度上限 7 ATR", run(m1, er_max=9))
tr = run(m1)
rep("+ 前段 ER(20) <= 0.35（最終版）", tr)
tr.to_csv("final_trades.csv", index=False)

print("\n== B. 最終版逐月 ==")
print(tr.groupby(tr.entry_time.dt.to_period("M")).pnl_R.agg(n="size", 平均R="mean", 合計R="sum").round(2).to_string())

r = tr.pnl_R.to_numpy()
bs = np.array([np.random.default_rng(i).choice(r, len(r)).mean() for i in range(3000)])
print(f"\n== C. 全期間 bootstrap ==\n  期望 {r.mean():+.3f}R，95% 信賴區間 [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
print(f"  中位 R = ${tr.R_usd.median():.2f}；出場：{tr.exit.value_counts().to_dict()}")
for extra in (0.1, 0.3, 0.5):
    print(f"  每筆額外滑價 ${extra}: 期望 {((tr.pnl_usd - extra) / tr.R_usd).mean():+.3f}R")

print("\n== D. 同一套規則放到 M1 / M5 ==")
for tf, mins, hold in (("1min", 1, 60), ("5min", 5, 48)):
    d = resample(m1, tf)
    cfg = Cfg(pattern="both", swing=20, trend_len=40, trend_atr=3, entry="break", tp=2, buf=0.1, hold=hold)
    c = d.close
    er = ((c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum()).to_numpy()
    st = [s for s in find_setups(d, cfg) if s[5] <= 7 and er[s[0]] <= .35]
    rep(tf, backtest_m1(d, m1, cfg, mins, st))
