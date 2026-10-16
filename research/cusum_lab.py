"""第二十三部分的共用工具：CUSUM 波段的通用回測（更多出場方式）與 M1 逐根執行。"""
import math
import numpy as np
import pandas as pd
from load import load
from changepoint import M15_PATH, cusum_signals

M1_PATH = "data/XAUUSD_M1_2023_2026.csv"


def bars(tf, src=None):
    """由 M15（或傳入的 M1）重取樣：M15 / M30 / H1 / H2 / H3 / H4 / D1。"""
    m = load(M15_PATH) if src is None else src
    m = m.copy(); m["sp"] = m.spread * 0.01
    rule = {"M15": "15min", "M30": "30min", "H1": "1h", "H2": "2h", "H3": "3h", "H4": "4h", "D1": "1D"}[tf]
    b = m.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "sp": "median"}).dropna()
    pc = b.close.shift()
    tr = np.maximum(b.high - b.low, np.maximum((b.high - pc).abs(), (b.low - pc).abs()))
    b["atr"] = tr.rolling(14).mean()
    b["r"] = np.log(b.close).diff()
    b["day"] = b.index.normalize()
    return b


def zscore(b, span=20, clip=4.0, seasonal=True):
    """報酬 / 事前波動。seasonal=True：同一時段過去 span 日 |r| 的 EWMA（去掉日內季節性）；False：一般 EWMA 標準差（span 根）。"""
    r = b.r
    if seasonal:
        slot = b.index.hour * 60 + b.index.minute
        a = r.abs().groupby(slot).transform(lambda x: x.ewm(span=span, min_periods=10).mean().shift())
        sig = a * math.sqrt(math.pi / 2)
    else:
        sig = np.sqrt((r ** 2).ewm(span=span * 24, min_periods=50).mean().shift())
    return (r / sig).clip(-clip, clip).fillna(0.0).to_numpy()


def backtest(b, sig, stop_atr=2.0, side="both", max_bars=None, trail_atr=None, be_R=None, delay=0,
             fin=0.0002, slip=0.0, risk_atr=2.0):
    """訊號於 K 棒收盤 → delay 根後的開盤進場；反向訊號反手（side 只限制「開新倉」的方向，反向訊號一律會平倉）。
    stop_atr=None → 不設停損（R 仍定義為 risk_atr × ATR，方便比較）。
    max_bars：持有超過 N 根就在開盤平倉。trail_atr：收盤後把停損移到「最佳收盤 − trail × ATR」。
    be_R：收盤浮盈 ≥ be_R × R 後停損移到進場價。slip：每邊滑價（美元）。"""
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr, sp = b.atr.to_numpy(float), b.sp.to_numpy(float)
    idx = b.index
    n = len(b)
    s = np.zeros(n, int)
    if delay:
        s[delay:] = sig[:-delay]
    else:
        s[:] = sig
    out = []
    pos = 0; ent = stop = R = A = best = 0.0; ei = -1
    def close(i, px, why):
        nonlocal pos
        days = max((idx[i].normalize() - idx[ei].normalize()).days, 0)
        gross = (px - ent) * pos
        cost = sp[ei] + 2 * slip + fin * ent * days
        out.append(dict(entry_time=idx[ei], exit_time=idx[i], side=pos, entry=ent, exit=px, R=R,
                        gross=gross, cost=cost, pnl=gross - cost, pnl_R=(gross - cost) / R, gross_R=gross / R,
                        bars=i - ei, days=days, why=why))
        pos = 0
    for i in range(1, n):
        sp_ = s[i - 1]
        if pos != 0:
            if sp_ == -pos:
                close(i, o[i], "flip")
            elif max_bars is not None and i - ei >= max_bars:
                close(i, o[i], "time")
        if pos == 0 and sp_ != 0 and (side == "both" or (side == "long") == (sp_ > 0)) \
                and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
            pos, ent, ei, A = sp_, o[i], i, atr[i - 1]
            R = (stop_atr or risk_atr) * A
            stop = ent - pos * stop_atr * A if stop_atr else (-np.inf if pos > 0 else np.inf)
            best = ent
        if pos == 1 and l[i] <= stop:
            close(i, min(o[i], stop), "stop")
        elif pos == -1 and h[i] >= stop:
            close(i, max(o[i], stop), "stop")
        if pos != 0:
            best = max(best, c[i]) if pos > 0 else min(best, c[i])
            if trail_atr:
                ns = best - pos * trail_atr * A
                stop = max(stop, ns) if pos > 0 else min(stop, ns)
            if be_R and (c[i] - ent) * pos >= be_R * R:
                stop = max(stop, ent) if pos > 0 else min(stop, ent)
    if pos != 0:
        close(n - 1, c[-1], "end")
    return pd.DataFrame(out)


def m1_backtest(hb, sig, m1, stop_atr=2.0, fin=0.0002, slip=0.0, tp_R=None):
    """訊號在 H1（hb）收盤產生；進出場改用 M1 逐根：
    進場 = 下一根 H1 開始時間後第一根 M1 的開盤；停損逐分鐘檢查（跳空穿越用該分鐘開盤價成交）；
    反手 = 反向訊號後下一根 H1 開始的第一根 M1 開盤。點差用實際進場與出場那一分鐘的點差各算一半。"""
    mt = m1.index.to_numpy()
    mo, mh, ml, mc = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    msp = m1.spread.to_numpy(float) * 0.01
    ht = hb.index.to_numpy()
    start = np.searchsorted(mt, ht)            # 每根 H1 對應的第一根 M1
    atr = hb.atr.to_numpy(float)
    n = len(hb)
    out = []
    pos = 0; ent = stop = R = 0.0; ej = -1
    def close(j, px, why):
        nonlocal pos
        t0, t1 = pd.Timestamp(mt[ej]), pd.Timestamp(mt[j])
        days = max((t1.normalize() - t0.normalize()).days, 0)
        gross = (px - ent) * pos
        cost = (msp[ej] + msp[j]) / 2 + 2 * slip + fin * ent * days
        out.append(dict(entry_time=t0, exit_time=t1, side=pos, entry=ent, exit=px, R=R, gross=gross, cost=cost,
                        pnl=gross - cost, pnl_R=(gross - cost) / R, gross_R=gross / R, days=days, why=why))
        pos = 0
    first = np.searchsorted(ht, mt[0]) + 1
    for i in range(max(first, 1), n):
        a, z = start[i], (start[i + 1] if i + 1 < n else len(mt))
        if a >= z:
            continue
        s = sig[i - 1]
        if pos != 0 and s == -pos:
            close(a, mo[a], "flip")
        if pos == 0 and s != 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
            pos, ent, ej = s, mo[a], a
            R = stop_atr * atr[i - 1]
            stop = ent - pos * R
        if pos != 0:
            seg = slice(max(a, ej), z)
            hit = np.where(ml[seg] <= stop)[0] if pos > 0 else np.where(mh[seg] >= stop)[0]
            js = seg.start + hit[0] if len(hit) else None
            jt = None
            if tp_R:
                tgt = ent + pos * tp_R * R
                th = np.where(mh[seg] >= tgt)[0] if pos > 0 else np.where(ml[seg] <= tgt)[0]
                jt = seg.start + th[0] if len(th) else None
            if js is not None and (jt is None or js <= jt):          # 同一分鐘都碰到 → 算停損
                px = min(mo[js], stop) if pos > 0 else max(mo[js], stop)
                close(js, px, "stop")
            elif jt is not None:
                px = max(mo[jt], tgt) if pos > 0 else min(mo[jt], tgt)
                close(jt, px, "tp")
    if pos != 0:
        close(len(mt) - 1, mc[-1], "end")
    return pd.DataFrame(out)


def detrend_R(b, T):
    """扣漂移：pnl_R − 方向 × （該期間的平均每小時漲幅 × 持有小時數）÷ R。用時間（而不是根數）計算，讓不同週期可以比較。"""
    if len(T) == 0:
        return pd.Series(dtype=float)
    hours = (b.index[-1] - b.index[0]).total_seconds() / 3600
    drift_h = (b.close.iloc[-1] - b.close.iloc[0]) / hours
    held = (T.exit_time - T.entry_time).dt.total_seconds() / 3600
    return T.pnl_R - T.side * drift_h * held / T.R


def summ(T, b=None):
    if len(T) < 2:
        return dict(n=len(T))
    r = T.pnl_R.to_numpy(); eq = np.cumsum(r)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    d = dict(n=len(r), mean=r.mean(), t=r.mean() / r.std(ddof=1) * math.sqrt(len(r)), pf=gp / gl if gl > 0 else np.inf,
             win=(r > 0).mean(), tot=r.sum(), dd=(np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max(),
             long=T[T.side > 0].pnl_R.mean(), short=T[T.side < 0].pnl_R.mean(),
             hold=(T.exit_time - T.entry_time).dt.total_seconds().mean() / 86400)
    yr = T.groupby(T.entry_time.dt.year).pnl_R.mean()
    d["minyr"] = yr.min(); d["years"] = yr
    if b is not None:
        dt = detrend_R(b, T)
        d["adj"] = dt.mean(); d["t_adj"] = dt.mean() / dt.std(ddof=1) * math.sqrt(len(dt))
        d["minyr_adj"] = dt.groupby(T.entry_time.dt.year).mean().min()
    return d


def line(d, adj=True):
    if d["n"] < 2:
        return f"n={d['n']}"
    s = (f"n={d['n']:4d} 每筆 {d['mean']:+.3f}R t={d['t']:+.2f} PF {d['pf']:.2f} 勝率 {d['win']:.2f} 累計 {d['tot']:+6.1f}R "
         f"回撤 {d['dd']:.1f}R 多 {d['long']:+.2f} 空 {d['short']:+.2f} 持有 {d['hold']:.1f}天 最差年 {d['minyr']:+.2f}")
    if adj and "adj" in d:
        s += f" | 去漂移 {d['adj']:+.3f}R t={d['t_adj']:+.2f}"
    return s


def backtest_pyr(b, sig, stop_atr=2.0, max_units=3, fin=0.0002):
    """加碼版：持有中再出現「同方向」訊號就再開一單（各自停損 2 ATR），最多 max_units 單；
    反向訊號 → 全部平倉並反手開 1 單。每單各自記成一筆交易。"""
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr, sp = b.atr.to_numpy(float), b.sp.to_numpy(float)
    idx = b.index; n = len(b)
    out, units = [], []                     # unit = [side, ent, stop, R, ei]
    def close(u, i, px, why):
        side, ent, stop, R, ei = u
        days = max((idx[i].normalize() - idx[ei].normalize()).days, 0)
        gross = (px - ent) * side; cost = sp[ei] + fin * ent * days
        out.append(dict(entry_time=idx[ei], exit_time=idx[i], side=side, entry=ent, exit=px, R=R, gross=gross, cost=cost,
                        pnl=gross - cost, pnl_R=(gross - cost) / R, gross_R=gross / R, bars=i - ei, days=days, why=why,
                        unit=len(units)))
    for i in range(1, n):
        s = sig[i - 1]
        if units and s == -units[0][0]:
            for u in units:
                close(u, i, o[i], "flip")
            units = []
        if s != 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0 and (not units or (units[0][0] == s and len(units) < max_units)):
            R = stop_atr * atr[i - 1]
            units.append([s, o[i], o[i] - s * R, R, i])
        keep = []
        for u in units:
            if u[0] == 1 and l[i] <= u[2]:
                close(u, i, min(o[i], u[2]), "stop")
            elif u[0] == -1 and h[i] >= u[2]:
                close(u, i, max(o[i], u[2]), "stop")
            else:
                keep.append(u)
        units = keep
    for u in units:
        close(u, n - 1, c[-1], "end")
    return pd.DataFrame(out).sort_values("entry_time").reset_index(drop=True)


def with_atr(b, n):
    """換 ATR 長度（n 根簡單平均 TR）。"""
    b = b.copy()
    pc = b.close.shift()
    tr = np.maximum(b.high - b.low, np.maximum((b.high - pc).abs(), (b.low - pc).abs()))
    b["atr"] = tr.rolling(n).mean()
    return b


def backtest_pm(b, sig, stop_atr=2.0, tp_R=None, max_bars=None, partial=None, be_after_partial=False,
                fin=0.0002, risk_atr=2.0):
    """持倉管理版（反向訊號一律反手）：
    tp_R：固定停利（R 倍數）。max_bars：持有上限（根）。partial=(R 倍數, 比例)：到價先平一部分，其餘照原規則；
    be_after_partial：部分停利後剩下的停損移到進場價。同一根 K 棒同時碰到停損與停利 → 算停損（保守）。
    pnl_R 是整筆（含部分平倉）的加權結果。"""
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr, sp = b.atr.to_numpy(float), b.sp.to_numpy(float)
    idx = b.index; n = len(b)
    out = []
    pos = 0; ent = stop = R = 0.0; ei = -1; done = 0.0; realized = 0.0
    def close(i, px, why):
        nonlocal pos
        days = max((idx[i].normalize() - idx[ei].normalize()).days, 0)
        rest = 1 - done
        gross = realized + rest * (px - ent) * pos
        cost = sp[ei] + fin * ent * days
        out.append(dict(entry_time=idx[ei], exit_time=idx[i], side=pos, entry=ent, exit=px, R=R, gross=gross, cost=cost,
                        pnl=gross - cost, pnl_R=(gross - cost) / R, gross_R=gross / R, bars=i - ei, days=days, why=why))
        pos = 0
    for i in range(1, n):
        s = sig[i - 1]
        if pos != 0:
            if s == -pos:
                close(i, o[i], "flip")
            elif max_bars is not None and i - ei >= max_bars:
                close(i, o[i], "time")
        if pos == 0 and s != 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
            pos, ent, ei = s, o[i], i
            R = (stop_atr or risk_atr) * atr[i - 1]
            stop = ent - pos * stop_atr * atr[i - 1] if stop_atr else (-np.inf if pos > 0 else np.inf)
            done = 0.0; realized = 0.0
        if pos == 0:
            continue
        adv = (l[i] - ent) * pos if pos > 0 else (ent - h[i])           # 最不利
        fav = (h[i] - ent) if pos > 0 else (ent - l[i])                 # 最有利
        if (pos > 0 and l[i] <= stop) or (pos < 0 and h[i] >= stop):
            close(i, min(o[i], stop) if pos > 0 else max(o[i], stop), "stop"); continue
        if partial and done == 0 and fav >= partial[0] * R:
            px = ent + pos * partial[0] * R
            if (o[i] - ent) * pos > partial[0] * R:
                px = o[i]
            realized += partial[1] * (px - ent) * pos; done = partial[1]
            if be_after_partial:
                stop = max(stop, ent) if pos > 0 else min(stop, ent)
        if tp_R and fav >= tp_R * R:
            px = ent + pos * tp_R * R
            if (o[i] - ent) * pos > tp_R * R:
                px = o[i]
            close(i, px, "tp")
    if pos != 0:
        close(n - 1, c[-1], "end")
    return pd.DataFrame(out)
