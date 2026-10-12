"""第四十九部分：(1) 不含浮虧的 MDD (2) 額外防護措施（套在最終版實際下單之上；影子帳本不受影響）。
最終版流程（A+B+斜率 → 上限 8 → 影子），停損 2 ATR15，RR 3 / 4，0.02 口，點差 + 0.3 滑價。
MDD 三種：逐分鐘含浮動 / 逐筆（依出場時間累加已實現）/ 日結。
防護（只用進場當下已知的資訊）：
  單日虧損上限 K R：當天已實現（含已出場）虧損達 −K R（以 0.02 口的 R 美元計）後，不再開新單
  波動暴增暫停：前一日 ATR10 / 前 60 日 ATR10 平均 > k → 當天不做
  點差異常：進場點差 > 過去 20 個交易日同小時點差中位數 × k → 不進
  非農時段：每月第一個星期五 broker 15:00~16:59 不開新單（非農 = 紐約 8:30 = broker 15:30）
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
ds = pd.DataFrame({"pc": dl.C.shift(1), "ma": sma50.shift(1), "ma10": sma50.shift(11), "atr": atrd.shift(1), "o": dl.O,
                   "volrel": (atrd / atrd.rolling(60).mean()).shift(1)})
EV = []
for i, L, side, a in G.ev:
    s = ds.loc[dayv[i]]
    if np.isfinite(s.ma) and np.isfinite(s.ma10) and np.isfinite(s.atr) and (s.pc - s.ma) * side > 0 \
            and (L - s.o) * side / s.atr > -0.17 and (s.ma - s.ma10) * side > 0 and IDX[i] >= pd.Timestamp("2023-08-24"):
        EV.append((i, L, side, a, s.volrel))
# 點差基準：過去 20 個交易日同小時的中位數
sp = pd.Series(SP, index=IDX)
hr = IDX.hour
sp_hour_day = sp.groupby([W.dayv, hr]).median().unstack()
sp_base = sp_hour_day.rolling(20, min_periods=5).median().shift(1)


def run(t):
    rows = []
    for i, L, side, a, vr in EV:
        risk = 2.0 * a
        r = G.sim_detail(i, L, side, L - side * risk, L + side * t * risk)
        if r is None:
            continue
        j, px, why = r
        net = (px - L) * side - SP[i] - SLIP
        base = sp_base.loc[W.dayv[i], IDX[i].hour] if W.dayv[i] in sp_base.index else np.nan
        rows.append((IDX[i], IDX[j], i, j, side, L, net, net / risk, risk, vr, SP[i] / base if base and base > 0 else np.nan))
    x = pd.DataFrame(rows, columns=["進場時間", "出場時間", "i", "j", "dir", "entry", "net", "R", "risk", "volrel", "sprel"])
    x = cap_total(x.sort_values("進場時間").reset_index(drop=True), 8).reset_index(drop=True)
    ex = x.sort_values("出場時間"); cum = ex.R.cumsum().values
    ma = pd.Series(cum).rolling(50, min_periods=50).mean().values; et = ex.出場時間.values
    keep = [not (k >= 50 and cum[k] < ma[k]) for k in (np.searchsorted(et, tt, side="right") - 1 for tt in x.進場時間.values)]
    return x[np.array(keep)].reset_index(drop=True)


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


def daily_stop(x, K):
    keep, real = [], []
    for r in x.itertuples():
        day = r.進場時間.normalize()
        lost = sum(v for (te, v, dd) in real if dd == day and te <= r.進場時間)
        if lost <= -K:
            keep.append(False); continue
        keep.append(True); real.append((r.出場時間, r.R, day))
    return x[np.array(keep)]


def first_friday(ts):
    return (ts.dt.dayofweek == 4) & (ts.dt.day <= 7)


lines = []


def p(s=""):
    print(s, flush=True)
    lines.append(s)


def report(x, lab):
    pnl = x.net * 2
    yrs = (x.進場時間.max() - x.進場時間.min()).days / 365
    m_min = minute_mdd(x)
    e_tr = pnl.groupby(x.出場時間).sum().sort_index().cumsum(); m_tr = (e_tr.cummax() - e_tr).max()
    d = pnl.groupby(x.進場時間.dt.date).sum(); e_d = d.cumsum(); m_d = (e_d.cummax() - e_d).max()
    tr = (x.進場時間 < "2025-01-01").values

    def rat(z, pz):
        dz = pz.groupby(z.進場時間.dt.date).sum(); ez = dz.cumsum(); mz = (ez.cummax() - ez).max()
        return pz.sum() / mz
    p(f"  {lab:30s} 筆 {len(x):5d} 總 {pnl.sum():+8,.0f} | MDD 含浮動 {m_min:5,.0f} / 逐筆 {m_tr:5,.0f} / 日結 {m_d:5,.0f} | "
      f"年獲利/MDD(含浮動) {pnl.sum() / yrs / m_min:4.2f} (日結) {pnl.sum() / yrs / m_d:4.2f} | 最差日 {d.min():+6,.0f} | 訓練 {rat(x[tr], pnl[tr]):3.1f} 驗證 {rat(x[~tr], pnl[~tr]):3.1f}")


p("第四十九部分：不含浮虧的 MDD + 額外防護（0.02 口，點差 + 0.3 滑價，2023/8 ~ 2026/10）")
for t in (3, 4):
    x = run(t)
    d = (x.R).groupby(x.進場時間.dt.date).sum()
    p(f"\n=== RR {t} ===  單日 R 分布：最差 {d.min():+.1f}R、1% 分位 {d.quantile(.01):+.1f}R、5% 分位 {d.quantile(.05):+.1f}R")
    report(x, "最終版（無額外防護）")
    for K in (6, 10, 15):
        report(daily_stop(x, K), f"單日虧損上限 −{K}R")
    for k in (1.5, 1.8, 2.2):
        report(x[~(x.volrel > k).values], f"波動暴增暫停（ATR比 > {k}）")
    for k in (2.0, 3.0):
        report(x[~(x.sprel > k).values], f"點差 > 平常 {k} 倍不進")
    nfp = first_friday(x.進場時間) & x.進場時間.dt.hour.between(15, 16)
    report(x[~nfp.values], "非農時段不開新單")
    allg = daily_stop(x[~((x.volrel > 2.2) | (x.sprel > 3.0) | nfp).values], 15)
    report(allg, "組合：−15R + ATR>2.2 + 點差3倍 + 非農")
open("results_part49.txt", "w").write("\n".join(lines) + "\n")
