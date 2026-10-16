"""第二十七部分：CUSUM H1 定案版（不加停利）的防護措施 + 固定 1 盎司的美元損益。
輸出：results_part27.txt
"""
import math, warnings, itertools
import numpy as np, pandas as pd
warnings.simplefilter("ignore", RuntimeWarning)
from load import load
from cusum_lab import bars, zscore, backtest, m1_backtest, summ, line, detrend_R, M1_PATH
from changepoint import cusum_signals, daily_pnl_R

out = open("results_part27.txt", "w")
def P(s=""):
    print(s); out.write(s + "\n"); out.flush()

hb = bars("H1"); z = zscore(hb); sig = cusum_signals(z, 0.25, 6)
T = backtest(hb, sig, 2.0)                       # H1 模擬 2023/1~
m1 = load(M1_PATH)
TM = m1_backtest(hb, sig, m1, 2.0)               # M1 逐根 2023/6~
yrs = (hb.index[-1] - hb.index[0]).days / 365.25

def ev(TT, lab):
    r = TT.pnl_R.to_numpy(); eq = np.cumsum(r); dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    yr = TT.groupby(TT.entry_time.dt.year).pnl_R.sum()
    P(f"  {lab:34s} {len(r):4d} 筆 每筆 {r.mean():+.3f}R t={r.mean() / r.std() * math.sqrt(len(r)):+.2f} 累計 {r.sum():+6.1f}R 回撤 {dd:5.1f}R "
      f"總R/回撤 {r.sum() / dd:4.1f} | 逐年 " + " ".join(f"{y % 100}:{v:+.1f}" for y, v in yr.items()))

# ====================================================================== 1. 防護措施
P("1) 防護措施（H1 模擬 2023/1 ~ 2026/10；每筆 1R）")
ev(T, "基準")

# 1a 週末前平倉
def weekend(b, sig, reenter):
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr, sp = b.atr.to_numpy(float), b.sp.to_numpy(float); idx = b.index; n = len(b)
    gap = np.r_[(idx[1:] - idx[:-1]) > pd.Timedelta(hours=12), False]   # 這根之後是週末/假日
    outl = []; pos = 0; carry = 0
    for i in range(1, n):
        s = sig[i - 1]
        if pos and s == -pos:
            outl.append(((o[i] - ent) * pos - sp[ei] - 0.0002 * ent * (idx[i].normalize() - idx[ei].normalize()).days, R, idx[ei], pos)); pos = 0
        if not pos and s == 0 and carry and reenter:
            s = carry
        carry = 0
        if not pos and s and atr[i - 1] > 0:
            pos, ent, ei, R = s, o[i], i, 2 * atr[i - 1]; stop = ent - pos * R
        if pos and ((pos > 0 and l[i] <= stop) or (pos < 0 and h[i] >= stop)):
            px = min(o[i], stop) if pos > 0 else max(o[i], stop)
            outl.append(((px - ent) * pos - sp[ei] - 0.0002 * ent * (idx[i].normalize() - idx[ei].normalize()).days, R, idx[ei], pos)); pos = 0
        if pos and gap[i]:
            outl.append(((c[i] - ent) * pos - sp[ei] - 0.0002 * ent * (idx[i].normalize() - idx[ei].normalize()).days, R, idx[ei], pos))
            carry = pos; pos = 0
    return pd.DataFrame([dict(pnl_R=p / R, entry_time=t, side=sd) for p, R, t, sd in outl])
ev(weekend(hb, sig, False), "週末前平倉（不再進場）")
ev(weekend(hb, sig, True), "週末前平倉，週一開盤同方向重進")

# 1b 長期趨勢濾網：多單只在 價格 > EMA(N 日)，空單只在 價格 < EMA
for days in (50, 100, 200):
    ema = hb.close.ewm(span=days * 23).mean().to_numpy()
    c = hb.close.to_numpy()
    f = sig.copy(); f[(sig > 0) & (c < ema)] = 0; f[(sig < 0) & (c > ema)] = 0
    # 反向訊號即使被濾掉，仍要平倉 → 用 side 參數不行，改成：濾掉的訊號只用來出場
    o, h, l = hb.open.to_numpy(), hb.high.to_numpy(), hb.low.to_numpy()
    TT = backtest(hb, sig, 2.0)
    keep = []
    for r in TT.itertuples():
        i = hb.index.get_loc(r.entry_time) - 1
        keep.append(f[i] != 0)
    ev(TT[np.array(keep)], f"只順 EMA{days}日 方向開倉（反向訊號仍平倉）")

# 1c 權益曲線開關：策略本身（影子帳）回撤 ≥ X R 時暫停，回到 < X/2 才恢復
def eq_switch(TT, X):
    r = TT.pnl_R.to_numpy(); eq = 0; pk = 0; on = True; take = []
    for x in r:
        take.append(on)
        eq += x; pk = max(pk, eq); dd = pk - eq
        if dd >= X: on = False
        elif dd < X / 2: on = True
    return TT[np.array(take)]
for X in (4, 6, 8):
    ev(eq_switch(T, X), f"影子帳回撤 ≥ {X}R 暫停")
# 1d 連虧暫停
def streak_pause(TT, N, M):
    r = TT.pnl_R.to_numpy(); take = []; run = 0; skip = 0
    for x in r:
        if skip > 0:
            take.append(False); skip -= 1
        else:
            take.append(True)
        run = run + 1 if x < 0 else 0
        if run >= N and skip == 0: skip = M; run = 0
    return TT[np.array(take)]
for N, M in ((4, 2), (5, 3), (6, 3)):
    ev(streak_pause(T, N, M), f"連虧 {N} 筆後跳過 {M} 筆")
# 1e 單筆停損上限（ATR 太大時縮手：R > 過去 1 年 R 的 90 百分位 → 不做）
Rq = T.R.rolling(46, min_periods=20).quantile(.9).shift()
ev(T[~(T.R > Rq)], "R 異常大（>過去一年 90%）不做")

# ====================================================================== 2. 監控門檻（自助法）
P("\n2) 上線後的監控門檻（以回測逐筆 R 做 20,000 次自助重抽，看「正常情況下」的範圍）")
rng = np.random.default_rng(0); r = T.pnl_R.to_numpy()
for N in (20, 46, 92):
    S = rng.choice(r, (20000, N)).sum(axis=1)
    P(f"  連續 {N:3d} 筆（約 {N / 46:.1f} 年）累計 R：5% 分位 {np.percentile(S, 5):+.1f}、1% 分位 {np.percentile(S, 1):+.1f}、虧損機率 {(S < 0).mean():.0%}")
DD = []
for _ in range(5000):
    e = np.cumsum(rng.choice(r, 46 * 3)); DD.append((np.maximum.accumulate(np.r_[0, e])[1:] - e).max())
DD = np.array(DD)
P(f"  3 年（138 筆）內最大回撤：中位 {np.median(DD):.1f}R、90% {np.percentile(DD, 90):.1f}R、95% {np.percentile(DD, 95):.1f}R、99% {np.percentile(DD, 99):.1f}R")
runs = []
for _ in range(5000):
    x = rng.choice(r, 46 * 3) < 0; m = c_ = 0
    for v in x:
        c_ = c_ + 1 if v else 0; m = max(m, c_)
    runs.append(m)
P(f"  3 年內最長連虧：中位 {np.median(runs):.0f} 筆、95% {np.percentile(runs, 95):.0f} 筆、99% {np.percentile(runs, 99):.0f} 筆")

# ====================================================================== 3. 固定 1 盎司
P("\n3) 固定 1 盎司（= 0.01 手；美元，已扣點差與融資 0.02%/日）")
for lab, TT, b0 in (("H1 模擬 2023/1~2026/10", T, hb), ("M1 逐根 2023/6~2026/10", TM, None)):
    p = TT.pnl.to_numpy(); eq = np.cumsum(p); dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq)
    P(f"  [{lab}] {len(p)} 筆 合計 {p.sum():+,.0f} 美元、每筆 {p.mean():+.1f}、勝 {p[p > 0].mean():+.1f} / 敗 {p[p <= 0].mean():+.1f}、"
      f"最大單筆 {p.max():+.0f} / {p.min():+.0f}、最大回撤 {dd.max():,.0f}（{TT.exit_time.iloc[dd.argmax()].date()}）")
TT = TM.copy()
dp = daily_pnl_R(hb, TT.assign(R=1.0, cost=TT.cost))      # R=1 → 美元盯市
dp = dp[dp.index >= TT.entry_time.min().normalize()]
eqd = dp.cumsum(); P(f"  M1 逐日盯市：最大回撤 {(eqd.cummax() - eqd).max():,.0f} 美元、最差單日 {dp.min():+.0f}、最好單日 {dp.max():+.0f}")
P("\n  逐年（M1 執行，進場年份）")
P(f"  {'年':6s} {'筆數':>4s} {'合計$':>8s} {'每筆$':>7s} {'R中位$':>7s} {'最大回撤$':>9s} {'金價':>10s}")
for y, g in TT.groupby(TT.entry_time.dt.year):
    e = g.pnl.cumsum().to_numpy(); d = (np.maximum.accumulate(np.r_[0, e])[1:] - e).max()
    px = hb.close[hb.index.year == y]
    P(f"  {y:<6d} {len(g):4d} {g.pnl.sum():+8.0f} {g.pnl.mean():+7.1f} {g.R.median():7.1f} {d:9.0f} {px.min():.0f}~{px.max():.0f}")
mon = TT.groupby(TT.exit_time.dt.to_period("M")).pnl.sum()
P(f"  每月（美元）：平均 {mon.mean():+.0f}、中位 {mon.median():+.0f}、最差 {mon.min():+.0f}、最好 {mon.max():+.0f}；虧損月 {(mon < 0).mean():.0%}")
P(f"  最大浮虧（單筆持倉中，M1）：{(TT.R * 0).max():.0f}" if False else "")
mae = []
mt = m1.index; mh, ml = m1.high.to_numpy(), m1.low.to_numpy()
for r_ in TT.itertuples():
    a, b_ = mt.searchsorted(r_.entry_time), mt.searchsorted(r_.exit_time)
    mae.append((r_.entry - ml[a:b_ + 1].min()) if r_.side > 0 else (mh[a:b_ + 1].max() - r_.entry))
mae = np.array(mae)
P(f"  單筆持倉最大浮虧：中位 {np.median(mae):.0f}、90% {np.percentile(mae, 90):.0f}、最大 {mae.max():.0f} 美元")
last = TT[TT.entry_time >= "2026-01-01"]
P(f"\n  以 2026 年的波動估計：R 中位 {last.R.median():.0f} 美元/盎司 → 每筆期望（+0.5R）約 {0.5 * last.R.median():+.0f} 美元、"
  f"每年 46 筆約 {0.5 * last.R.median() * 46:+,.0f} 美元；回撤 10R 約 {10 * last.R.median():,.0f} 美元、連虧 8 筆約 {8 * last.R.median():,.0f} 美元")
P(f"  名目價值約 {hb.close.iloc[-1]:,.0f} 美元/盎司；槓桿 1:100 時保證金約 {hb.close.iloc[-1] / 100:,.0f} 美元/盎司")
