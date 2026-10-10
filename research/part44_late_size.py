"""第四十四部分：依「突破多快」分倉 —— IB 結束後 1 小時內突破下大一點、1~3 小時下小一點。

交易 = 最終規則（OW、窗 3h、2R 保本、收盤出場、OW 篩選後去重）；最多同時 5 張（以筆數計）。
A. 依突破時間分組（0~0.5、0.5~1、1~2、2~3h）的每筆品質；M15 另列各期
B. 分倉方案：每筆手數 = 0.01 × 權重；比較總美元、平倉回撤、報酬÷回撤、日 Sharpe（後兩者與規模無關）
"""
import numpy as np
import pandas as pd
from part27_ow import data, run, pf, PER

out = open("results_part44.txt", "w")
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


SCHEMES = [("全部 0.01（現行）", lambda d: np.ones(len(d))),
           ("≤1h 0.02、1~3h 0.01", lambda d: np.where(d <= 1, 2.0, 1.0)),
           ("≤1h 0.03、1~3h 0.01", lambda d: np.where(d <= 1, 3.0, 1.0)),
           ("≤1h 0.02、1~3h 不做（= 1h 窗）", lambda d: np.where(d <= 1, 2.0, 0.0)),
           ("≤0.5h 0.03、0.5~1h 0.02、1~3h 0.01", lambda d: np.where(d <= 0.5, 3.0, np.where(d <= 1, 2.0, 1.0))),
           ("≤1h 0.02、1~2h 0.01、2~3h 不做", lambda d: np.where(d <= 1, 2.0, np.where(d <= 2, 1.0, 0.0)))]

for tf, (path, bm) in SRC.items():
    days, D, ARR = data(path, bm)
    X, _ = run(path, bm)
    T = X[X.ow & (X.day >= "2023-03-01")].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
    ex = [manage(r, ARR, bm) for r in T.itertuples()]
    T["exit_time"] = [b for a, b in ex]; T["pnl"] = T.side * (np.array([a for a, b in ex]) - T.lvl) - T.spread
    T["R"] = T.pnl / T.risk; T["u"] = T.pnl / T.atr
    T["delay"] = (T.t_in - T.day - pd.to_timedelta(T.h + 1, unit="h")).dt.total_seconds() / 3600
    alldays = [d for d in days if d >= T.day.min()]
    per = [p for p in PER if len(T[(T.day >= p[1]) & (T.day <= p[2])])]
    P("\n" + "=" * 140 + f"\n[{tf}]  A. 依 IB 結束到突破進場的時間分組" + ("（右側各期每筆 R）" if tf == "M15" else ""))
    for lab, a, b in [("0~0.5h", -1, 0.5), ("0.5~1h", 0.5, 1), ("1~2h", 1, 2), ("2~3h", 2, 3.1)]:
        g = T[(T.delay > a) & (T.delay <= b)]
        cells = ("  " + "  ".join(f"{n} {x.R.mean():+.2f}" for n, x0, x1 in per for x in [g[(g.day >= x0) & (g.day <= x1)]] if len(x))) if tf == "M15" else ""
        P(f"  {lab:7s} {len(g):5d}筆 勝{(g.R > 0.05).mean() * 100:3.0f}% 每筆{g.R.mean():+.3f}R PF{pf(g.R):.2f} 每筆÷ATR{g.u.mean() * 100:+5.1f}% 總{g.pnl.sum():+6,.0f}美元" + cells)
    C5 = cap(T)
    P(f"  B. 分倉（同一組最多 5 張的 {len(C5)} 筆）")
    for name, f in SCHEMES:
        w = f(C5.delay.values)
        Y = C5.assign(u=C5.pnl * w).sort_values("exit_time"); eq = Y.u.cumsum(); dd = (eq - eq.cummax()).min()
        dl = Y.groupby("day").u.sum().reindex(alldays, fill_value=0)
        yr = ("  逐年 " + "  ".join(f"{y} {v:+,.0f}" for y, v in Y.groupby(Y.day.dt.year).u.sum().items())) if tf == "M15" else ""
        P(f"  {name:34s} 平均手數 {w.mean() * 0.01:.3f}  總 {Y.u.sum():+6,.0f}美元  回撤 {dd:+6,.0f}  報酬÷回撤 {Y.u.sum() / -dd:4.1f}  "
          f"Sharpe {dl.mean() / dl.std() * np.sqrt(250):.2f}  最差單日 {dl.min():+5,.0f}" + yr)
out.close()
