"""第四十六部分（二）：在最終版完整流程下重測 RR。
第二十九部分的 RR 表是「A+B、同方向一次 1 筆、無斜率、無影子、只扣點差」，RR 4~5 最好；
最終版是「A+B+斜率、上限 8、影子」，這裡用最終版流程 + 點差 + 0.3 滑價重測：停損 2 / 3 ATR15 × RR 2 / 3 / 4 / 5 / 收盤。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G
from part30_max_positions import cap_total

W, IDX = G.W, G.IDX
C, SP, dayv = W.C, W.SP, W.dayv
SLIP = 0.3
dl = W.B.daily
sma50 = dl.C.rolling(50).mean()
atrd = (dl.H - dl.L).rolling(10).mean()
ds = pd.DataFrame({"pc": dl.C.shift(1), "ma": sma50.shift(1), "ma10": sma50.shift(11), "atr": atrd.shift(1), "o": dl.O})
ev = []
for i, L, side, a in G.ev:
    s = ds.loc[dayv[i]]
    if not (np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr)):
        continue
    if (s.pc - s.ma) * side > 0 and (L - s.o) * side / s.atr > -0.17 and (s.ma - s.ma10) * side > 0 and IDX[i] >= pd.Timestamp("2023-08-24"):
        ev.append((i, L, side, a))


def run(sm, t):
    rows = []
    for i, L, side, a in ev:
        risk = sm * a
        r = G.sim_detail(i, L, side, L - side * risk, L + side * (t if t else 1e6) * risk)
        if r is None:
            continue
        j, px, why = r
        net = (px - L) * side - SP[i] - SLIP
        rows.append((IDX[i], IDX[j], i, j, side, L, net, net / risk, why))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "why"])
    x = cap_total(x.sort_values("進場時間").reset_index(drop=True), 8).reset_index(drop=True)
    ex = x.sort_values("出場時間"); cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(50, min_periods=50).mean().values; et = ex.出場時間.values
    keep = []
    for tt in x.進場時間.values:
        k = np.searchsorted(et, tt, side="right") - 1
        keep.append(not (k >= 50 and cum[k] < ma[k]))
    return x[np.array(keep)]


def minute_mdd(df):
    n = len(IDX)
    D, E, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for r in df.itertuples():
        D[r.i] += r.dir * 2; D[r.j] -= r.dir * 2
        E[r.i] += r.dir * r.entry * 2; E[r.j] -= r.dir * r.entry * 2
        R[r.j] += r.net * 2
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    eq = R + C * D - E
    return (np.maximum.accumulate(eq) - eq).max()


lines = ["第四十六部分（二）：最終版流程下的 RR（0.02 口，點差 + 0.3 滑價，2023/8 ~ 2026/10）",
         f"{'停損':>4s} {'RR':>4s} {'筆/天':>5s} {'勝率':>5s} {'每筆R':>7s} {'總損益':>8s} {'MDD':>6s} {'最差月':>7s} {'年獲利/MDD':>9s} | {'訓練':>4s} {'驗證':>4s} | 逐年"]
for sm in (2.0, 3.0):
    for t in (2, 3, 4, 5, None):
        x = run(sm, t)
        pnl = x.net * 2
        yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
        m = minute_mdd(x)
        mon = pnl.groupby(x.進場時間.dt.to_period("M")).sum()
        tr = (x.進場時間 < "2025-01-01").values

        def rat(z, pz):
            d = pz.groupby(z.進場時間.dt.date).sum(); e = d.cumsum(); mm = (e.cummax() - e).max()
            return pz.sum() / mm
        yr = " / ".join(f"{pnl[(x.進場時間.dt.year == y).values].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
        lines.append(f"{sm:4g} {(str(t) if t else '收'):>4s} {len(x) / x.進場時間.dt.date.nunique():5.1f} {(x.R > 0).mean():5.0%} {x.R.mean():+7.3f} "
                     f"{pnl.sum():+8,.0f} {m:6,.0f} {mon.min():+7,.0f} {pnl.sum() / yrs / m:9.2f} | {rat(x[tr], pnl[tr]):4.1f} {rat(x[~tr], pnl[~tr]):4.1f} | {yr}")
        print(lines[-1], flush=True)
open("results_part46_rr.txt", "w").write("\n".join(lines) + "\n")
