"""第三十七部分：以 0.02 口為單位，比較防回撤措施前後（美元，每筆另扣 0.3 美元/盎司滑價）。

V0  A + B，上限 8（未加防回撤）
V1  V0 + 50 日均線斜率同向
V2  V1 + 權益曲線濾網（50 筆均線）＝ 最終版
V3  V2 + 已持倉 ≥ 4 筆時新單只下 0.01 口
V4  V2 + 第三十五部分的加權：權益高於均線較多 → 0.03 口、已持倉 ≥ 4 → 0.01 口、兩者同時 → 0.02 口
全部每筆 0.02 口（V3 部分 0.01 口）。MDD 列日結與逐分鐘含浮動。
"""
import numpy as np
import pandas as pd
import part29_rr_grid as G
from part30_max_positions import cap_total

W, IDX = G.W, G.IDX
C, SP, dayv = W.C, W.SP, W.dayv
import sys
GROSS = "gross" in sys.argv  # python3 part37_lot002.py gross → 不扣點差、不扣滑價
SLIP = 0.3
dl = W.B.daily
sma50 = dl.C.rolling(50).mean()
ATRd = (dl.H - dl.L).rolling(10).mean().shift(1)
ds = pd.DataFrame({"pc": dl.C.shift(1), "ma": sma50.shift(1), "ma10": sma50.shift(11), "atr": ATRd, "o": dl.O})

rows = []
for i, L, side, a in G.ev:
    s = ds.loc[dayv[i]]
    if not (np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr)):
        continue
    A = (s.pc - s.ma) * side > 0
    Bc = (L - s.o) * side / s.atr > -0.17
    if not (A and Bc):
        continue
    risk = 2.0 * a
    r = G.sim_detail(i, L, side, L - side * risk, L + side * 3 * risk)
    if r is None:
        continue
    j, px, why = r
    net = (px - L) * side - SP[i]
    rows.append(dict(進場時間=IDX[i], 出場時間=IDX[j], i=i, j=j, dir=side, entry=L, usd=(net + SP[i]) if GROSS else net - SLIP, R=((net + SP[i]) if GROSS else net - SLIP) / risk,
                     slope=(s.ma - s.ma10) * side > 0))
ALL = pd.DataFrame(rows).sort_values("進場時間").reset_index(drop=True)


def eq_filter(df, look=50):
    ex = df.sort_values("出場時間")
    cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(look, min_periods=look).mean().values
    et = ex.出場時間.values
    keep, gap = [], []
    for t in df.進場時間.values:
        k = np.searchsorted(et, t, side="right") - 1
        keep.append(not (k >= look and cum[k] < ma[k]))
        gap.append(cum[k] - ma[k] if k >= look else np.nan)
    out = df.copy()
    out["eq_gap"] = gap  # 影子累積 R − 50 筆均線
    return out[np.array(keep)]


def half_when_crowded(df, lim=4):
    lots, open_ = [], []
    for r in df.itertuples():
        open_ = [x for x in open_ if x > r.進場時間]
        lots.append(0.01 if len(open_) >= lim else 0.02)
        open_.append(r.出場時間)
    out = df.copy()
    out["lot"] = lots
    return out


V0 = cap_total(ALL, 8).copy()
V1 = cap_total(ALL[ALL.slope], 8).copy()
V2 = eq_filter(V1.reset_index(drop=True)).copy()
V3 = half_when_crowded(V2)
# V4：第三十五部分的加權（基準 0.02；權益高於均線較多 ×1.5 → 0.03；已持倉 ≥ 4 ×0.5 → 0.01；兩者同時 → 0.02）
cut = np.nanmedian(V2.eq_gap[V2.進場時間 < "2025-01-01"])  # 訓練期中位數
V4 = half_when_crowded(V2)
crowd = V4.lot < 0.02
V4["lot"] = np.select([(V4.eq_gap > cut) & ~crowd, (V4.eq_gap > cut) & crowd, crowd], [0.03, 0.02, 0.01], 0.02)
for v in (V0, V1, V2):
    v["lot"] = 0.02


def minute_mdd(df):
    n = len(IDX)
    D, E, R = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)
    for r in df.itertuples():
        q = r.lot * 100
        D[r.i] += r.dir * q; D[r.j] -= r.dir * q
        E[r.i] += r.dir * r.entry * q; E[r.j] -= r.dir * r.entry * q
        R[r.j] += r.usd * q
    D, E, R = np.cumsum(D)[:n], np.cumsum(E)[:n], np.cumsum(R)[:n]
    eq = R + C * D - E
    return (np.maximum.accumulate(eq) - eq).max()


lines = []


def p(s=""):
    print(s)
    lines.append(s)


p("第三十七部分：0.02 口，防回撤措施前後（2023/8 ~ 2026/10，美元，" + ("不扣點差、不扣滑價" if GROSS else "已扣點差 + 每筆 0.3 美元/盎司滑價") + "）\n")
summ = []
for lab, v in (("V0 A+B，上限 8（未防回撤）", V0), ("V1 + 50MA 斜率同向", V1), ("V2 + 權益濾網（最終版）", V2), ("V3 + 持倉 ≥4 新單 0.01 口", V3), ("V4 加權 0.01/0.02/0.03（順時 0.03、持倉≥4 0.01）", V4)):
    v = v[v.進場時間 >= "2023-08-24"]  # 與最終版同一起點（SMA50 暖機後）
    pnl = v.usd * v.lot * 100
    d = pnl.groupby(v.進場時間.dt.date).sum(); e = d.cumsum()
    mdd_d = (e.cummax() - e).max()
    mdd_m = minute_mdd(v)
    yrs = (v.進場時間.max() - v.進場時間.min()).days / 365
    mon = pnl.groupby(v.進場時間.dt.to_period("M")).sum()
    ev = pd.concat([pd.Series(-v.lot.values, v.出場時間.values), pd.Series(v.lot.values, v.進場時間.values)]).sort_index(kind="stable")
    yr = " / ".join(f"{pnl[v.進場時間.dt.year == y].sum():+,.0f}" for y in (2023, 2024, 2025, 2026))
    yr_dd = []
    for y in (2023, 2024, 2025, 2026):
        m = v.進場時間.dt.year == y
        dy = pnl[m].groupby(v.進場時間[m].dt.date).sum(); ey = dy.cumsum(); yr_dd.append(f"{(ey.cummax() - ey).max():,.0f}")
    p(f"=== {lab} ===")
    p(f"  筆數 {len(v)}（每天 {len(v) / v.進場時間.dt.date.nunique():.1f}），總損益 {pnl.sum():+,.0f}，每年 {pnl.sum() / yrs:+,.0f}，每月平均 {mon.mean():+,.0f}（正報酬 {(mon > 0).sum()}/{len(mon)} 月）")
    p(f"  MDD 日結 {mdd_d:,.0f} / 逐分鐘含浮動 {mdd_m:,.0f}；最差一天 {d.min():+,.0f}；最差一月 {mon.min():+,.0f}（{mon.idxmin()}）；同時最多 {ev.cumsum().max():.2f} 口")
    p(f"  逐年損益 2023/2024/2025/2026：{yr}；逐年 MDD：{' / '.join(yr_dd)}")
    p(f"  年獲利 / 逐分鐘 MDD = {pnl.sum() / yrs / mdd_m:.2f}；MDD ≤ 20% 需本金 ≥ {mdd_m / 0.2:,.0f}（對應年報酬約 {pnl.sum() / yrs / (mdd_m / 0.2):.0%}，單利）\n")
    summ.append((lab, len(v), pnl.sum(), pnl.sum() / yrs, mdd_m, d.min()))
open("results_part37_gross.txt" if GROSS else "results_part37.txt", "w").write("\n".join(lines) + "\n")
