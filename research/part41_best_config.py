"""第四十一部分：整體最佳運行方式 —— 把有效的元件組合起來，同一套成本比較。

元件：
  A+B      順 50 日均線、今天沒逆向走超過 0.17 日 ATR（第二十八部分）
  斜率     50 日均線 10 日斜率同向（第三十一部分）
  影子     權益濾網：影子累積 R < 50 筆均線時不下單（第三十一部分）
  漲過頭   前一日收盤離 50 日均線 > 3 個日 ATR（乘方向）→ 新單 0.01 口（第三十九部分）
  擁擠     已持倉 ≥ 4 筆 → 新單 0.01 口（第三十七部分）
基準口數 0.02；停損 2 ATR15、RR 3、上限 8。成本：點差（+ 每筆 0.3 美元/盎司滑價）。
訓練 2023/8~2024/12、驗證 2025/1~2026/10。MDD = 逐分鐘含浮動損益。
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

rows = []
for i, L, side, a in G.ev:
    s = ds.loc[dayv[i]]
    if not (np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr)):
        continue
    if not ((s.pc - s.ma) * side > 0 and (L - s.o) * side / s.atr > -0.17):
        continue
    risk = 2.0 * a
    r = G.sim_detail(i, L, side, L - side * risk, L + side * 3 * risk)
    if r is None:
        continue
    j, px, why = r
    net = (px - L) * side - SP[i]
    rows.append(dict(進場時間=IDX[i], 出場時間=IDX[j], i=i, j=j, dir=side, entry=L, net=net, risk=risk,
                     slope=(s.ma - s.ma10) * side > 0, ext=s.dist50 * side))
ALL = pd.DataFrame(rows).sort_values("進場時間").reset_index(drop=True)
ALL = ALL[ALL.進場時間 >= "2023-08-24"].reset_index(drop=True)


def eq_filter(df, look=50):
    ex = df.sort_values("出場時間")
    cum = (ex.net / ex.risk).cumsum().values
    ma = pd.Series(cum).rolling(look, min_periods=look).mean().values
    et = ex.出場時間.values
    keep = []
    for t in df.進場時間.values:
        k = np.searchsorted(et, t, side="right") - 1
        keep.append(not (k >= look and cum[k] < ma[k]))
    return df[np.array(keep)].copy()


def crowd(df, lim=4):
    out, open_ = [], []
    for r in df.itertuples():
        open_ = [x for x in open_ if x > r.進場時間]
        out.append(len(open_) >= lim)
        open_.append(r.出場時間)
    return np.array(out)


def build(slope, shadow, ext, crowded):
    x = ALL[ALL.slope] if slope else ALL
    x = cap_total(x.reset_index(drop=True), 8).reset_index(drop=True)
    if shadow:
        x = eq_filter(x).reset_index(drop=True)
    lot = np.full(len(x), 0.02)
    if ext:
        lot[(x.ext > 3).values] = 0.01
    if crowded:
        lot[crowd(x)] = 0.01
    x["lot"] = lot
    return x


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


def daily_ratio(x, pnl):
    d = pnl.groupby(x.進場時間.dt.date).sum(); e = d.cumsum(); m = (e.cummax() - e).max()
    return pnl.sum() / m if m > 0 else np.nan


CFG = [("① V0：A+B，不防回撤", dict(slope=False, shadow=False, ext=False, crowded=False)),
       ("② V0 + 影子", dict(slope=False, shadow=True, ext=False, crowded=False)),
       ("③ 最終版：+ 斜率 + 影子", dict(slope=True, shadow=True, ext=False, crowded=False)),
       ("④ ② + 漲過頭減半", dict(slope=False, shadow=True, ext=True, crowded=False)),
       ("⑤ ③ + 漲過頭減半", dict(slope=True, shadow=True, ext=True, crowded=False)),
       ("⑥ ⑤ + 擁擠減半", dict(slope=True, shadow=True, ext=True, crowded=True)),
       ("⑦ ① + 漲過頭減半", dict(slope=False, shadow=False, ext=True, crowded=False))]
lines = []


def p(s=""):
    print(s)
    lines.append(s)


yrs = (ALL.進場時間.max() - ALL.進場時間.min()).days / 365
p("第四十一部分：組合比較（基準 0.02 口，2023/8 ~ 2026/10）")
summary = []
for slip, cl in ((0.0, "只扣點差"), (0.3, "點差 + 每筆 0.3 美元滑價")):
    p(f"\n=== {cl} ===")
    p(f"{'設定':22s} {'筆/天':>5s} {'總損益':>8s} {'每年':>7s} {'MDD':>6s} {'最差月':>7s} {'正月':>6s} {'年獲利/MDD':>9s} | {'訓練比值':>6s} {'驗證比值':>6s} | 逐年")
    for lab, kw in CFG:
        x = build(**kw)
        pnl = (x.net - slip) * x.lot * 100
        mdd = minute_mdd(x, slip)
        mon = pnl.groupby(x.進場時間.dt.to_period("M")).sum()
        tr = (x.進場時間 < "2025-01-01").values
        yr = " / ".join(f"{pnl[(x.進場時間.dt.year == y).values].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
        p(f"{lab:22s} {len(x) / x.進場時間.dt.date.nunique():5.1f} {pnl.sum():+8,.0f} {pnl.sum() / yrs:+7,.0f} {mdd:6,.0f} {mon.min():+7,.0f} "
          f"{(mon > 0).sum():>3d}/{len(mon):<2d} {pnl.sum() / yrs / mdd:9.2f} | {daily_ratio(x[tr], pnl[tr]):6.1f} {daily_ratio(x[~tr], pnl[~tr]):6.1f} | {yr}")
        summary.append((cl, lab, pnl.sum(), mdd))
open("results_part41.txt", "w").write("\n".join(lines) + "\n")
