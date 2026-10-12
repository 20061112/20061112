"""第四十二部分：同一套最終版規則換到 M5 / M15 / M30 / H1。

規則完全照 M15 最終版（參數不重調）：前一根 P 週期 K 棒高/低點突破、A + B + 50MA 斜率、停損 2 ATR_P、RR 3、
上限 8、權益濾網（50 筆）；另列「+ 漲過頭減半」。ATR_P = 前 20 根 P 週期 K 棒波幅平均。
0.02 口；成本：點差 + 每筆 0.3 美元/盎司滑價（另列只扣點差）。2023/8 ~ 2026/10；訓練 ~2024、驗證 2025~。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G
from part30_max_positions import cap_total

W, IDX = G.W, G.IDX
C, SP, dayv = W.C, W.SP, W.dayv
dl = W.B.daily
sma50 = dl.C.rolling(50).mean()
atrd = (dl.H - dl.L).rolling(10).mean()
ds = pd.DataFrame({"pc": dl.C.shift(1), "ma": sma50.shift(1), "ma10": sma50.shift(11), "atr": atrd.shift(1), "o": dl.O,
                   "dist50": ((dl.C - sma50) / atrd).shift(1)})


def build_all(P):
    rows = []
    for i, L, side, a in W.events(P):
        s = ds.loc[dayv[i]]
        if not (np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr)):
            continue
        if not ((s.pc - s.ma) * side > 0 and (L - s.o) * side / s.atr > -0.17 and (s.ma - s.ma10) * side > 0):
            continue
        risk = 2.0 * a
        r = G.sim_detail(i, L, side, L - side * risk, L + side * 3 * risk)
        if r is None:
            continue
        j, px, why = r
        rows.append(dict(進場時間=IDX[i], 出場時間=IDX[j], i=i, j=j, dir=side, entry=L, net=(px - L) * side - SP[i],
                         risk=risk, sp=SP[i], ext=s.dist50 * side))
    x = pd.DataFrame(rows).sort_values("進場時間").reset_index(drop=True)
    return x[x.進場時間 >= "2023-08-24"].reset_index(drop=True)


def eq_filter(df, slip, look=50):
    ex = df.sort_values("出場時間")
    cum = ((ex.net - slip) / ex.risk).cumsum().values
    ma = pd.Series(cum).rolling(look, min_periods=look).mean().values
    et = ex.出場時間.values
    keep = []
    for t in df.進場時間.values:
        k = np.searchsorted(et, t, side="right") - 1
        keep.append(not (k >= look and cum[k] < ma[k]))
    return df[np.array(keep)].copy()


def minute_mdd(df, slip):
    n = len(IDX)
    D, E, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for r in df.itertuples():
        q = r.lot * 100
        D[r.i] += r.dir * q; D[r.j] -= r.dir * q
        E[r.i] += r.dir * r.entry * q; E[r.j] -= r.dir * r.entry * q
        R[r.j] += (r.net - slip) * q
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    eq = R + C * D - E
    return (np.maximum.accumulate(eq) - eq).max()


def dratio(x, pnl):
    d = pnl.groupby(x.進場時間.dt.date).sum(); e = d.cumsum(); m = (e.cummax() - e).max()
    return pnl.sum() / m if m > 0 else np.nan


lines = []


def p(s=""):
    print(s)
    lines.append(s)


p("第四十二部分：最終版規則換週期（0.02 口，2023/8 ~ 2026/10）")
HDR = (f"{'週期 / 設定':22s} {'筆/天':>5s} {'停損中位':>7s} {'點差/R':>6s} {'每筆R':>7s} {'總損益':>8s} {'每年':>7s} {'MDD':>6s} {'最差月':>7s} "
       f"{'年獲利/MDD':>9s} | {'訓練':>4s} {'驗證':>4s} | 逐年")
res = {}
for P, nm in ((5, "M5"), (15, "M15"), (30, "M30"), (60, "H1")):
    res[nm] = cap_total(build_all(P), 8).reset_index(drop=True)
for slip, cl in ((0.3, "點差 + 每筆 0.3 美元滑價"), (0.0, "只扣點差")):
    p(f"\n=== {cl} ===")
    p(HDR)
    for nm, base in res.items():
        x0 = eq_filter(base, slip).reset_index(drop=True)
        for lab, ext in (("", False), (" + 漲過頭減半", True)):
            x = x0.copy()
            x["lot"] = np.where(ext & (x.ext > 3).values, 0.01, 0.02)
            pnl = (x.net - slip) * x.lot * 100
            R = (x.net - slip) / x.risk
            mdd = minute_mdd(x, slip)
            yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
            mon = pnl.groupby(x.進場時間.dt.to_period("M")).sum()
            tr = (x.進場時間 < "2025-01-01").values
            yr = " / ".join(f"{pnl[(x.進場時間.dt.year == y).values].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
            p(f"{nm + lab:22s} {len(x) / x.進場時間.dt.date.nunique():5.1f} {x.risk.median():7.2f} {(x.sp / x.risk).mean():6.3f} {R.mean():+7.3f} "
              f"{pnl.sum():+8,.0f} {pnl.sum() / yrs:+7,.0f} {mdd:6,.0f} {mon.min():+7,.0f} {pnl.sum() / yrs / mdd:9.2f} | "
              f"{dratio(x[tr], pnl[tr]):4.1f} {dratio(x[~tr], pnl[~tr]):4.1f} | {yr}")
open("results_part42.txt", "w").write("\n".join(lines) + "\n")
