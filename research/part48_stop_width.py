"""第四十八部分：停損放近 / 放遠（最終版流程：A+B+斜率 → 上限 8 → 影子；點差 + 0.3 滑價）。
停損 s = 1 / 1.5 / 2 / 2.5 / 3 / 4 × ATR15；RR 3 / 4。
兩種看法：(1) 固定 0.02 口（停損越遠每筆虧越多美元）；(2) 每筆固定風險 1R（公平比較，比值 = 年 R / R 計 MDD）。"""
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
EV = []
for i, L, side, a in G.ev:
    s = ds.loc[dayv[i]]
    if np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr) and (s.pc - s.ma) * side > 0 \
            and (L - s.o) * side / s.atr > -0.17 and (s.ma - s.ma10) * side > 0 and IDX[i] >= pd.Timestamp("2023-08-24"):
        EV.append((i, L, side, a))


def run(sm, t):
    rows = []
    for i, L, side, a in EV:
        risk = sm * a
        r = G.sim_detail(i, L, side, L - side * risk, L + side * t * risk)
        if r is None:
            continue
        j, px, why = r
        net = (px - L) * side - SP[i] - SLIP
        rows.append((IDX[i], IDX[j], i, j, side, L, net, net / risk, risk))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "risk"])
    x = cap_total(x.sort_values("進場時間").reset_index(drop=True), 8).reset_index(drop=True)
    ex = x.sort_values("出場時間"); cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(50, min_periods=50).mean().values; et = ex.出場時間.values
    keep = [not (k >= 50 and cum[k] < ma[k]) for k in (np.searchsorted(et, tt, side="right") - 1 for tt in x.進場時間.values)]
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


def rat(z, v):
    d = v.groupby(z.進場時間.dt.date).sum(); e = d.cumsum(); m = (e.cummax() - e).max()
    return v.sum() / m, m


lines = ["第四十八部分：停損放近 / 放遠（最終版流程，2023/8 ~ 2026/10，點差 + 0.3 滑價）",
         f"{'停損':>4s} {'RR':>3s} {'筆/天':>5s} {'勝率':>5s} {'停損$中位':>8s} | {'0.02口 總損益':>12s} {'MDD':>6s} {'年獲利/MDD':>9s} {'訓練':>4s} {'驗證':>4s} | "
         f"{'固定1R 每筆R':>11s} {'年R/MDD':>7s} {'訓練':>4s} {'驗證':>4s} | 逐年（0.02 口）"]
for t in (3, 4):
    for sm in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        x = run(sm, t)
        pnl = x.net * 2
        yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
        m = minute_mdd(x)
        tr = (x.進場時間 < "2025-01-01").values
        rr_all, mR = rat(x, x.R)
        yr = " / ".join(f"{pnl[(x.進場時間.dt.year == y).values].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
        lines.append(f"{sm:4g} {t:3d} {len(x) / x.進場時間.dt.date.nunique():5.1f} {(x.R > 0).mean():5.0%} {x.risk.median():8.2f} | "
                     f"{pnl.sum():+12,.0f} {m:6,.0f} {pnl.sum() / yrs / m:9.2f} {rat(x[tr], pnl[tr])[0]:4.1f} {rat(x[~tr], pnl[~tr])[0]:4.1f} | "
                     f"{x.R.mean():+11.3f} {x.R.sum() / yrs / mR:7.2f} {rat(x[tr], x.R[tr])[0]:4.1f} {rat(x[~tr], x.R[~tr])[0]:4.1f} | {yr}")
        print(lines[-1], flush=True)
    lines.append("")
open("results_part48.txt", "w").write("\n".join(lines) + "\n")
