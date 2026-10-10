"""第四十三部分：突破窗（IB 結束後幾小時內可以進場，現行 3h）與 IB 長度（現行 60 分）掃描。

其他規則同最終版（OW、2R 保本、收盤出場、OW 篩選後去重、0.01 手最多 5 張）。
"""
import numpy as np
import pandas as pd
from part27_ow import data, run, pf, PER

out = open("results_part43.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M1": ("data/XAUUSD_M1_2026.csv", 1), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M15": ("data/XAUUSD_M15_2023_2026.csv", 15)}


def manage(r, ARR, bm):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; be = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            return (min(O[k], st) if s == 1 else max(O[k], st)), t[k] + pd.Timedelta(minutes=bm)
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if not be and s * (best - e) >= 2 * R:
            be = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=bm)


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def one(tf, path, bm, **kw):
    days, D, ARR = data(path, bm)
    X, _ = run(path, bm, **kw)
    T = X[X.ow & (X.day >= "2023-03-01")].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
    ex = [manage(r, ARR, bm) for r in T.itertuples()]
    T["exit_time"] = [b for a, b in ex]; T["pnl"] = T.side * (np.array([a for a, b in ex]) - T.lvl) - T.spread
    T["R"] = T.pnl / T.risk; T["u"] = T.pnl / T.atr
    alldays = [d for d in days if d >= T.day.min()]
    C5 = cap(T).sort_values("exit_time"); eq = C5.pnl.cumsum()
    dl = C5.groupby("day").pnl.sum().reindex(alldays, fill_value=0)
    late = (T.t_in - T.day - pd.to_timedelta(T.h + kw.get("ib_min", 60) / 60, unit="h")).dt.total_seconds() / 3600
    cells = ("  " + "  ".join(f"{n} {x.R.mean():+.2f}" for n, a, b in PER for x in [T[(T.day >= a) & (T.day <= b)]] if len(x))) if tf == "M15" else ""
    return (f"{len(T):5d}筆 勝{(T.R > 0.05).mean() * 100:3.0f}% 每筆{T.R.mean():+.3f}R PF{pf(T.R):.2f} 每筆÷ATR{T.u.mean() * 100:+5.1f}% "
            f"IB結束後平均{late.mean():.1f}h進場 | 5張 {C5.pnl.sum():+6,.0f}美元 回撤{(eq - eq.cummax()).min():+6,.0f} Sharpe{dl.mean() / dl.std() * np.sqrt(250):.2f}" + cells)


for tf, (path, bm) in SRC.items():
    P("\n" + "=" * 150 + f"\n[{tf}]" + ("  右側各期每筆 R" if tf == "M15" else ""))
    P(" A. 突破窗（IB 60 分）")
    for w in [0.5, 1, 2, 3, 4, 5, 6, 22]:
        P(f"  窗 {('到收盤' if w == 22 else f'{w}h'):6s} {'（現行）' if w == 3 else '      '} " + one(tf, path, bm, win=w))
    P(" B. IB 長度（窗 3h）")
    for ib in [30, 60, 90, 120, 180]:
        P(f"  IB {ib:3d} 分 {'（現行）' if ib == 60 else '      '} " + one(tf, path, bm, ib_min=ib))
out.close()
