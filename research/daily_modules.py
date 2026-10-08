"""第八部分：與 FADE 互補的兩個日級模組（M15，2/2~10/8，broker 時間 = 紐約 + 7）。

DAYREV  ：在 broker entry_h:00 進場，方向 = 與前一個交易日 (close-open) 相反；
          在 exit_h:00 平倉（不跨日，配合 FADE 的 23:55 收盤線）；可選停損 = stop_k × 日 ATR(10)。
          星期一的「前一天」= 上週五（FADE 規格把星期一關掉，這裡兩種都測）。
LONBRK  ：broker 10:00~12:59 第一次突破亞洲盤（01:00~09:59）高/低點，以突破價進場，順突破方向；
          停損 = 亞洲區間另一側；在 exit_h:00 平倉。
成本：進場與出場各以當下 K 棒點差的一半計（合計約一個點差）。同根 K 棒內觸停損以停損價計。
"""
import numpy as np
import pandas as pd
from load import load

d = load("data/XAUUSD_M15_full.csv")
d["sp"] = d.spread * 0.01
d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
G = {k: g for k, g in d.groupby("day") if len(g) >= 60}
DAYS = sorted(G)
D = pd.DataFrame({k: dict(open=g.open.iloc[0], close=g.close.iloc[-1], high=g.high.max(), low=g.low.min()) for k, g in G.items()}).T
D["ret"] = D.close - D.open
D["tr"] = np.maximum(D.high - D.low, np.maximum((D.high - D.close.shift()).abs(), (D.low - D.close.shift()).abs()))
D["atr10"] = D.tr.rolling(10).mean().shift(1)            # 只用前幾天
D["prev_ret"] = D.ret.shift(1)
D["prev_dir"] = np.sign(D.prev_ret)
D["prev_size"] = D.prev_ret.abs() / D.atr10


def at(g, h):
    """broker h:00 開始的那根 K 棒的位置（沒有就回傳 None）"""
    idx = np.where(g.index.hour * 60 + g.index.minute >= h * 60)[0]
    return idx[0] if len(idx) else None


def dayrev(entry_h=1, exit_h=23, stop_k=None, monday=True, min_prev=0.0):
    out = []
    for k in DAYS:
        r = D.loc[k]
        if not np.isfinite(r.atr10) or r.prev_dir == 0 or not np.isfinite(r.prev_dir):
            continue
        if not monday and k.dayofweek == 0:
            continue
        if r.prev_size < min_prev:
            continue
        g = G[k]
        i0, i1 = at(g, entry_h), at(g, exit_h)
        if i0 is None or i1 is None or i1 <= i0:
            continue
        s = -r.prev_dir
        ep = g.open.iloc[i0]
        xp = g.open.iloc[i1]
        if stop_k:
            stop = ep - s * stop_k * r.atr10
            seg = g.iloc[i0:i1]
            hit = np.where((seg.low <= stop) if s == 1 else (seg.high >= stop))[0]
            if len(hit):
                xp = stop if s == 1 and seg.open.iloc[hit[0]] > stop or s == -1 and seg.open.iloc[hit[0]] < stop else seg.open.iloc[hit[0]]
        cost = (g.sp.iloc[i0] + g.sp.iloc[i1 - 1]) / 2
        out.append(dict(day=k, side=s, pnl=s * (xp - ep) - cost, atr=r.atr10, prev_size=r.prev_size, dow=k.dayofweek))
    return pd.DataFrame(out)


def lonbrk(exit_h=19, min_range=0.0, max_range=99.0):
    out = []
    for k in DAYS:
        g = G[k]
        asia = g[(g.index.hour >= 1) & (g.index.hour < 10)]
        lon = g[(g.index.hour >= 10) & (g.index.hour < 13)]
        r = D.loc[k]
        if len(asia) < 30 or len(lon) < 8 or not np.isfinite(r.atr10):
            continue
        hi, lo = asia.high.max(), asia.low.min()
        rng = (hi - lo) / r.atr10
        if not (min_range <= rng <= max_range):
            continue
        for t, row in lon.iterrows():
            s = 1 if row.high > hi else -1 if row.low < lo else 0
            if not s:
                continue
            ep = max(hi, row.open) if s == 1 else min(lo, row.open)
            stop = lo if s == 1 else hi
            after = g[(g.index >= t) & (g.index.hour < exit_h)]
            xp = after.close.iloc[-1]
            for j, (tt, b) in enumerate(after.iterrows()):
                if j == 0:
                    # 進場那根：若同時觸及另一側（極少見），保守算停損
                    if (b.low <= stop) if s == 1 else (b.high >= stop):
                        xp = stop; break
                    continue
                if (b.low <= stop) if s == 1 else (b.high >= stop):
                    xp = min(stop, b.open) if s == 1 else max(stop, b.open); break
            cost = row.sp
            out.append(dict(day=k, side=s, pnl=s * (xp - ep) - cost, R=abs(ep - stop), atr=r.atr10, rng=rng,
                            with_prev=s * (r.prev_dir if np.isfinite(r.prev_dir) else 0), dow=k.dayofweek))
            break
    T = pd.DataFrame(out)
    T["pnl_R"] = T.pnl / T.R
    return T


def summ(T, col="pnl"):
    x = T[col]
    h = T.day < pd.Timestamp("2026-06-01")
    eq = x.cumsum()
    dd = (eq.cummax() - eq).max()
    m = T.groupby(T.day.dt.to_period("M"))[col].sum()
    return (f"{len(T):3d} 天 平均 {x.mean():+7.2f}  t={x.mean() / x.std() * np.sqrt(len(x)):+.2f}  勝率 {(x > 0).mean():.2f}  "
            f"前 {x[h].mean():+6.2f} 後 {x[~h].mean():+6.2f}  累計 {x.sum():+7.1f}  最大回撤 {dd:.1f}  正月 {(m > 0).sum()}/{len(m)}")


def lonbrk_general(G_, D_, asia=(1, 10), win=(10, 13), exit_h=19, stop_mode="range"):
    """LONBRK 一般化版本：可改亞洲區間、突破時段、停損方式（range = 區間另一側 / mid = 區間中點）。"""
    out = []
    for k, g in G_.items():
        a = g[(g.index.hour >= asia[0]) & (g.index.hour < asia[1])]
        w = g[(g.index.hour >= win[0]) & (g.index.hour < win[1])]
        if len(a) < 4 * (asia[1] - asia[0]) * 0.8 / (15 / (g.index[1] - g.index[0]).seconds * 60) if False else len(a) < 8 or len(w) < 2:
            continue
        hi, lo = a.high.max(), a.low.min()
        for t, row in w.iterrows():
            s = 1 if row.high > hi else -1 if row.low < lo else 0
            if not s:
                continue
            ep = max(hi, row.open) if s == 1 else min(lo, row.open)
            stop = (lo if s == 1 else hi) if stop_mode == "range" else (hi + lo) / 2
            after = g[(g.index >= t) & (g.index.hour < exit_h)]
            if len(after) == 0:
                break
            xp = after.close.iloc[-1]
            for j, (tt, b) in enumerate(after.iterrows()):
                if (b.low <= stop) if s == 1 else (b.high >= stop):
                    xp = stop if j == 0 else (min(stop, b.open) if s == 1 else max(stop, b.open)); break
            R = abs(ep - stop)
            if R <= 0:
                break
            out.append(dict(day=k, side=s, pnl=s * (xp - ep) - row.sp, R=R))
            break
    T = pd.DataFrame(out)
    T["pnl_R"] = T.pnl / T.R
    return T
