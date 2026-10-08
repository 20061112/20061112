"""第九部分：紐約開盤區間「假突破」反向（ORB fade），M1/M3/M5（6/29~10/8）與 M15（2/2~10/8）。

開盤區間 OR = [open, open + or_min) 的高低。之後 win_min 分鐘內：
  mode = "touch" : 第一次觸及 OR 高（低）→ 立即反向做空（多），停損 = 觸價 + width_k × OR 寬度
  mode = "fail"  : 先突破 OR，之後某根 TF K 棒「收盤回到 OR 內」→ 下一根開盤反向；停損 = 突破後的極值
出場：目標 = OR 另一側（可關閉），否則 exit_h:00 平倉。路徑用同一週期 K 棒，同根同時觸停損/目標算停損。
"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample


def bars(tf):
    if tf == 15:
        d = load("data/XAUUSD_M15_full.csv")
    else:
        m1 = load("data/XAUUSD_M1_full.csv")
        d = m1 if tf == 1 else resample(m1, f"{tf}min")
    d = d.copy()
    d["sp"] = d.spread * 0.01
    return d


def fade(d, tf, open_hm=(15, 30), or_min=15, win_min=120, mode="fail", width_k=1.0, target=True, exit_h=21):
    o0 = open_hm[0] * 60 + open_hm[1]
    day = (d.index - pd.Timedelta(hours=1)).normalize()
    out = []
    for k, g in d.groupby(day):
        h = g.index.hour * 60 + g.index.minute
        rng = g[(h >= o0) & (h < o0 + or_min)]
        if len(rng) < max(1, or_min // tf * 0.8):
            continue
        hi, lo = rng.high.max(), rng.low.min()
        W = hi - lo
        if W <= 0:
            continue
        after = g[(h >= o0 + or_min) & (h < exit_h * 60)]
        win_end = o0 + or_min + win_min
        s, ep, stop, ei = 0, None, None, None
        broke, ext = 0, None
        for j, (t, b) in enumerate(after.iterrows()):
            if t.hour * 60 + t.minute >= win_end:
                break
            if mode == "touch":
                if b.high > hi or b.low < lo:
                    s = -1 if b.high > hi else 1
                    ep = max(hi, b.open) if s == -1 else min(lo, b.open)
                    stop = ep - s * width_k * W
                    ei = j
                    break
            else:
                if broke == 0:
                    if b.close > hi:
                        broke, ext = 1, b.high
                    elif b.close < lo:
                        broke, ext = -1, b.low
                    continue
                ext = max(ext, b.high) if broke == 1 else min(ext, b.low)
                if (broke == 1 and b.close < hi) or (broke == -1 and b.close > lo):
                    if j + 1 >= len(after):
                        break
                    s = -broke
                    ei = j + 1
                    ep = after.open.iloc[ei]
                    stop = ext
                    break
        if not s or s * (ep - stop) <= 0:
            continue
        R = s * (ep - stop)
        tgt = (lo if s == -1 else hi) if target else None
        if tgt is not None and s * (tgt - ep) <= 0:
            tgt = None
        xp = after.close.iloc[-1]
        for jj in range(ei, len(after)):
            b = after.iloc[jj]
            if (b.high >= stop) if s == -1 else (b.low <= stop):
                xp = stop if jj == ei else (max(stop, b.open) if s == -1 else min(stop, b.open)); break
            if tgt is not None and jj > ei and ((b.low <= tgt) if s == -1 else (b.high >= tgt)):
                xp = tgt; break
        pnl = s * (xp - ep) - after.sp.iloc[ei]
        out.append(dict(day=k, side=s, R=R, W=W, pnl=pnl, pnl_R=pnl / R))
    return pd.DataFrame(out)


def summ(T, split="2026-08-15"):
    if len(T) < 5:
        return f"{len(T)} 筆"
    r = T.pnl_R
    h = T.day < pd.Timestamp(split)
    eq = r.cumsum()
    m = T.groupby(T.day.dt.to_period("M")).pnl_R.sum()
    return (f"{len(T):3d} 筆 期望 {r.mean():+.3f}R t={r.mean() / r.std() * np.sqrt(len(r)):+.2f} 勝率 {(r > 0).mean():.2f} "
            f"中位R {T.R.median():5.2f} 前 {r[h].mean():+.2f} 後 {r[~h].mean():+.2f} 累計 {r.sum():+5.1f}R 回撤 {(eq.cummax() - eq).max():.1f}R 正月 {(m > 0).sum()}/{len(m)}")
