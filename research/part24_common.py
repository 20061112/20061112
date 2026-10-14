"""第二十四部分共用：統一的固定 ATR 出場、日線趨勢狀態、統計。

方向慣例：所有偵測都在「頂部空間」Y 進行（Y = 原始資料偵測頂部，Y = mirror(X) 偵測底部）。
  在 Y 中做空 = 原始空間中 side=-1（Y=X）或 side=+1（Y=mirror）。
統一出場（讓不同背離算法可以公平比較）：
  訊號棒下一根開盤進場，停損 1.5 ATR、停利 3 ATR（2R）、最長 32 根；同根同時碰到算停損；扣點差。
"""
import numpy as np
import pandas as pd
from part22_divergence import get_frames, indicators, mirror, nonoverlap, SPLIT

STOP_ATR = 1.5
TGT_ATR = 3.0
HOLD = 32


def sim(Y, t, d):
    """Y 空間中，t 收盤後下一根開盤進場；d=+1 做多、-1 做空。回傳 (ei, xi, pnl_R)。"""
    o, h, l, c, atr, spr = Y["o"], Y["h"], Y["l"], Y["c"], Y["atr"], Y["spr"]
    n = len(c)
    if t + 1 >= n or not atr[t] > 0:
        return None
    ei, ep, A = t + 1, o[t + 1], atr[t]
    R = STOP_ATR * A
    stop, tgt = ep - d * R, ep + d * TGT_ATR * A
    for j in range(ei, min(ei + HOLD + 1, n)):
        hit_s = l[j] <= stop if d == 1 else h[j] >= stop
        hit_t = h[j] >= tgt if d == 1 else l[j] <= tgt
        if hit_s:
            xp = stop if j == ei else (min(stop, o[j]) if d == 1 else max(stop, o[j]))
            return ei, j, (d * (xp - ep) - spr[ei]) / R
        if hit_t:
            return ei, j, (d * (tgt - ep) - spr[ei]) / R
    j = min(ei + HOLD, n - 1)
    return ei, j, (d * (c[j] - ep) - spr[ei]) / R


def fwd(Y, t, hz, d):
    c, atr, spr = Y["c"], Y["atr"], Y["spr"]
    if t + hz >= len(c):
        return np.nan
    return (d * (c[t + hz] - c[t]) - spr[t]) / atr[t]


def daily_regime(df):
    """日線狀態（只用前一天以前的資料）：
    多頭 = 日收 > SMA50 且 SMA50 比 20 天前高；空頭 = 日收 < SMA50 且 SMA50 比 20 天前低；其餘 = 盤整。"""
    d = df.close.resample("1D").last().dropna()
    sma = d.rolling(50).mean()
    up = (d > sma) & (sma > sma.shift(20))
    dn = (d < sma) & (sma < sma.shift(20))
    reg = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=d.index).shift(1)
    day = df.index.normalize()
    return reg.reindex(day).ffill().to_numpy()


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else np.nan


def fmt(r):
    r = pd.Series(r).dropna()
    return f"n={len(r):5d} {r.mean():+.3f}R t={tstat(r):+5.2f}" if len(r) >= 5 else f"n={len(r):5d}       -        "


def per_year(T):
    return " ".join(f"{y}:{v.mean():+.2f}" for y, v in T.groupby(T.entry_time.dt.year).pnl_R)


def split_line(name, T, width=24):
    if len(T) == 0:
        print(f"  {name:{width}s} (無)")
        return
    x = nonoverlap(T)
    a, b = x[x.entry_time < SPLIT], x[x.entry_time >= SPLIT]
    print(f"  {name:{width}s} {fmt(a.pnl_R):26s} {fmt(b.pnl_R):26s} {per_year(x)}")


def both_spaces(X):
    """回傳 [(Y, side_of_short_in_Y)]：Y=X 時在 Y 做空 = 原始做空(-1)；Y=mirror 時在 Y 做空 = 原始做多(+1)。"""
    return [(X, -1), (mirror(X), 1)]
