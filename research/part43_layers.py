"""第四十三部分：M15 前一根高低點突破 —— 從「完全沒篩選」開始，一層層套上去。

出場全部相同：停損 2 ATR15、RR 3、當天收盤平倉。0.02 口。2023/8 ~ 2026/10。
L0 全部突破（多空都做，不限持倉）
L1 + A 順 50 日均線（多頭只做多、空頭只做空）
L2 + B 今天沒逆向走超過 0.17 日 ATR
L3 + 同時最多 8 筆
L4 + 50MA 斜率同向
L5 + 權益濾網（影子，50 筆）               ← 最終版
L6 + 漲過頭減半（前一日離 50MA > 3 ATR → 0.01 口）
成本：點差 + 每筆 0.3 美元/盎司滑價；另列只扣點差。MDD = 逐分鐘含浮動。
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
    risk = 2.0 * a
    r = G.sim_detail(i, L, side, L - side * risk, L + side * 3 * risk)
    if r is None:
        continue
    j, px, why = r
    rows.append(dict(進場時間=IDX[i], 出場時間=IDX[j], i=i, j=j, dir=side, entry=L, net=(px - L) * side - SP[i], risk=risk,
                     A=(s.pc - s.ma) * side > 0, B=(L - s.o) * side / s.atr > -0.17, slope=(s.ma - s.ma10) * side > 0,
                     ext=s.dist50 * side))
ALL = pd.DataFrame(rows).sort_values("進場時間").reset_index(drop=True)
ALL = ALL[ALL.進場時間 >= "2023-08-24"].reset_index(drop=True)


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
    return (np.maximum.accumulate(eq) - eq).max(), np.abs(D).max() / 100


def dratio(x, pnl):
    d = pnl.groupby(x.進場時間.dt.date).sum(); e = d.cumsum(); m = (e.cummax() - e).max()
    return pnl.sum() / m if m > 0 else np.nan


def layers(slip):
    out = []
    x = ALL.copy(); out.append(("L0 全部突破（多空，不限持倉）", x))
    x = x[x.A]; out.append(("L1 + 順 50 日均線", x))
    x = x[x.B]; out.append(("L2 + 今天沒逆向走太多", x))
    x = cap_total(x.reset_index(drop=True), 8); out.append(("L3 + 最多 8 筆", x))
    x = cap_total(ALL[ALL.A & ALL.B & ALL.slope].reset_index(drop=True), 8); out.append(("L4 + 50MA 斜率同向", x))
    x = eq_filter(x.reset_index(drop=True), slip); out.append(("L5 + 權益濾網（最終版）", x))
    out.append(("L6 + 漲過頭減半", x.assign(half=True)))
    return out


lines = []


def p(s=""):
    print(s)
    lines.append(s)


yrs = (ALL.進場時間.max() - ALL.進場時間.min()).days / 365
p("第四十三部分：M15 突破 —— 從沒有篩選開始一層層套上去（0.02 口，2023/8 ~ 2026/10）")
for slip, cl in ((0.3, "點差 + 每筆 0.3 美元滑價"), (0.0, "只扣點差")):
    p(f"\n=== {cl} ===")
    p(f"{'層':26s} {'筆/天':>5s} {'勝率':>5s} {'每筆R':>7s} {'總損益':>9s} {'每年':>8s} {'MDD':>7s} {'最多口數':>6s} {'最差月':>8s} {'年獲利/MDD':>9s} | {'訓練':>5s} {'驗證':>5s} | 逐年")
    for lab, x in layers(slip):
        x = x.copy()
        x["lot"] = np.where(("half" in x.columns) & (x.ext > 3).values, 0.01, 0.02) if "half" in x.columns else 0.02
        pnl = (x.net - slip) * x.lot * 100
        R = (x.net - slip) / x.risk
        mdd, maxlot = minute_mdd(x, slip)
        mon = pnl.groupby(x.進場時間.dt.to_period("M")).sum()
        tr = (x.進場時間 < "2025-01-01").values
        yr = " / ".join(f"{pnl[(x.進場時間.dt.year == y).values].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
        p(f"{lab:26s} {len(x) / x.進場時間.dt.date.nunique():5.1f} {(R > 0).mean():5.0%} {R.mean():+7.3f} {pnl.sum():+9,.0f} {pnl.sum() / yrs:+8,.0f} "
          f"{mdd:7,.0f} {maxlot:6.2f} {mon.min():+8,.0f} {pnl.sum() / yrs / mdd:9.2f} | {dratio(x[tr], pnl[tr]):5.1f} {dratio(x[~tr], pnl[~tr]):5.1f} | {yr}")
open("results_part43.txt", "w").write("\n".join(lines) + "\n")
