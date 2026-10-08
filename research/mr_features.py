"""第四部分：型態 + 均值回歸 / 布林帶 / z-score / 多長度 ER 比值。

候選：所有晨星/夜星/吞噬（極值為 swing 根內最高/低，不要求前段幅度），
每個候選用兩種出場各算一次結果（候選可重疊，用來做特徵研究）：
  pnl_R    : 停損極值外 0.1 ATR，停利 2R
  pnl_mid  : 停損同上，停利 = 進場當下的布林中軌（均值回歸出場）；中軌距離 < 0.5R 時改用 0.5R
進場一律為「型態後 wait 根內突破型態 K 棒高/低點」。
"""
import numpy as np
import pandas as pd
from backtest import Cfg, find_setups, _arrays

CAND = Cfg(pattern="both", swing=10, trend_len=20, trend_atr=0.0, entry="break", wait=3, tp=2.0, buf=0.1, hold=16)


def efficiency(c, n):
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


def indicators(d):
    c, h, l = d.close, d.high, d.low
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    X = pd.DataFrame(index=d.index)
    X["atr"] = tr.rolling(14).mean()
    for n in (20, 50, 100):
        ma, sd = c.rolling(n).mean(), c.rolling(n).std()
        X[f"ma{n}"] = ma
        X[f"z{n}"] = (c - ma) / sd
    sd20 = c.rolling(20).std()
    X["bb_up"], X["bb_dn"] = X.ma20 + 2 * sd20, X.ma20 - 2 * sd20
    X["bbw"] = 4 * sd20 / X.ma20
    X["bbw_pct"] = X.bbw.rolling(500).rank(pct=True)
    X["bbw_chg5"] = X.bbw / X.bbw.shift(5) - 1
    for n in (5, 10, 20, 40, 80):
        X[f"er{n}"] = efficiency(c, n)
    X["er5_20"] = X.er5 / X.er20
    X["er10_40"] = X.er10 / X.er40
    X["er20_80"] = X.er20 / X.er80
    X["er_short_long"] = X.er10 - X.er80        # 短期很直、長期很亂 → 區間內的急衝
    return X


def simulate(o, h, l, c, spr, t, s, stop, trig, tgt_fn, wait, hold):
    n = len(c)
    ei = None
    for j in range(t + 1, min(t + 1 + wait, n)):
        if (h[j] > trig) if s == 1 else (l[j] < trig):
            ei, ep = j, (max(trig, o[j]) if s == 1 else min(trig, o[j]))
            break
        if (l[j] <= stop) if s == 1 else (h[j] >= stop):
            return None
    if ei is None:
        return None
    R = s * (ep - stop)
    if R <= 0:
        return None
    tgt = tgt_fn(ep, R)
    for j in range(ei, min(ei + 1 + hold, n)):
        if (l[j] <= stop) if s == 1 else (h[j] >= stop):
            xp = stop if j == ei else (min(stop, o[j]) if s == 1 else max(stop, o[j]))
            return ei, j, ep, R, (s * (xp - ep) - spr[ei]) / R
        if j > ei and ((h[j] >= tgt) if s == 1 else (l[j] <= tgt)):
            return ei, j, ep, R, (s * (tgt - ep) - spr[ei]) / R
    j = min(ei + hold, n - 1)
    return ei, j, ep, R, (s * (c[j] - ep) - spr[ei]) / R


def candidate_table(d, cfg=CAND):
    X = indicators(d)
    o, h, l, c, atr, spr = _arrays(d)
    v = d.tickvol.to_numpy(float)
    rows = []
    for t, s, kind, ext, A, trend in find_setups(d, cfg):
        if t < 600:
            continue
        stop = ext - s * cfg.buf * A
        trig = h[t] if s == 1 else l[t]
        mid = X.ma20.iat[t]
        a = simulate(o, h, l, c, spr, t, s, stop, trig, lambda ep, R: ep + s * 2.0 * R, cfg.wait, cfg.hold)
        if a is None:
            continue
        b = simulate(o, h, l, c, spr, t, s, stop, trig,
                     lambda ep, R: mid if s * (mid - ep) >= 0.5 * R else ep + s * 0.5 * R, cfg.wait, cfg.hold)
        k = 3 if kind == "star" else 2
        seg = range(t - k + 1, t + 1)
        e = min(seg, key=lambda i: l[i]) if s == 1 else max(seg, key=lambda i: h[i])
        r = dict(setup=d.index[t], t=t, entry_time=d.index[a[0]], exit_time=d.index[a[1]], side=s, kind=kind,
                 R_usd=a[3], pnl_R=a[4], exit_mid=d.index[b[1]], pnl_mid=b[4])
        # 特徵（統一方向：正值 = 往「與反轉相反」的方向延伸，例如多單時價格在均線下方 → z 為正）
        for n in (20, 50, 100):
            r[f"z{n}"] = -s * X[f"z{n}"].iat[e]
        r["bbw_pct"] = X.bbw_pct.iat[t]
        r["bbw_chg5"] = X.bbw_chg5.iat[t]
        r["close_outside"] = int((c[e] < X.bb_dn.iat[e]) if s == 1 else (c[e] > X.bb_up.iat[e]))
        for col in ("er5", "er10", "er20", "er40", "er80", "er5_20", "er10_40", "er20_80", "er_short_long"):
            r[col] = X[col].iat[e]
        r["trend_atr"] = trend
        r["mid_dist_R"] = s * (mid - a[2]) / a[3]                  # 進場到中軌的距離（R 倍數）
        r["tv_e"] = v[e] / np.median(v[max(e - 96, 0):e])
        r["hour"] = d.index[t].hour
        rows.append(r)
    return pd.DataFrame(rows)
