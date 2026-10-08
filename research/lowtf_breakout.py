"""第九部分：M1 / M3 / M5 上的開盤突破。

A. LONBRK 訊號不變（亞洲 01~10 區間，10~13 突破，19:00 出場），改變進場確認與停損：
   entry = touch（觸價）或 close（TF K 棒收盤在區間外，下一根開盤進）
   stop  = range（區間另一側）/ swing（TF 最近 N 根的反向極值）/ atr（k × TF ATR14）
B. ORB：開盤區間 = [open_h:open_m, + or_min 分鐘) 的高低；之後到 win_end 第一次突破，19:00（或指定）出場，
   停損 = 開盤區間另一側。倫敦 10:00、紐約 broker 15:30（= NY 08:30）、COMEX 15:20 等。
路徑一律用 M1 逐根模擬（同根同時觸停損/停利算停損；進場那根觸停損算停損）；成本 = 進場點差。
"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample

M1 = load("data/XAUUSD_M1_full.csv")
M1["sp"] = M1.spread * 0.01
DAY = (M1.index - pd.Timedelta(hours=1)).normalize()
DAYS = {k: g for k, g in M1.groupby(DAY) if len(g) >= 900}


def hm(g):
    return g.index.hour * 60 + g.index.minute


def run_path(g1, t_entry, s, ep, stop, exit_min):
    """g1 = 當日 M1；從 t_entry 這根 M1 開始，觸停損或到 exit_min 平倉。"""
    path = g1[(g1.index >= t_entry) & (hm(g1) < exit_min)]
    if len(path) == 0:
        return None
    for j, (t, b) in enumerate(path.iterrows()):
        if (b.low <= stop) if s == 1 else (b.high >= stop):
            xp = stop if j == 0 else (min(stop, b.open) if s == 1 else max(stop, b.open))
            return xp, t, "stop"
    return path.close.iloc[-1], path.index[-1], "time"


def lonbrk_tf(tf_min, entry="touch", stop_mode="range", k=3.0, swing_n=5, exit_h=19):
    out = []
    for day, g1 in DAYS.items():
        h = hm(g1)
        asia = g1[(h >= 60) & (h < 600)]
        if len(asia) < 300:
            continue
        hi, lo = asia.high.max(), asia.low.min()
        g = resample(g1, f"{tf_min}min") if tf_min > 1 else g1
        pc = g.close.shift()
        atr = pd.concat([g.high - g.low, (g.high - pc).abs(), (g.low - pc).abs()], axis=1).max(axis=1).rolling(14).mean()
        gh = hm(g)
        win = g[(gh >= 600) & (gh < 780)]
        for t, b in win.iterrows():
            if entry == "touch":
                s = 1 if b.high > hi else -1 if b.low < lo else 0
            else:
                s = 1 if b.close > hi else -1 if b.close < lo else 0
            if not s:
                continue
            i = g.index.get_loc(t)
            if entry == "touch":
                # 在 M1 上找到實際觸價的那一分鐘
                sub = g1[(g1.index >= t) & (g1.index < t + pd.Timedelta(minutes=tf_min))]
                hit = sub[(sub.high > hi) if s == 1 else (sub.low < lo)]
                t_e = hit.index[0]
                ep = max(hi, hit.open.iloc[0]) if s == 1 else min(lo, hit.open.iloc[0])
                ref = i - 1                                   # 停損參考只用進場前已收完的 TF K 棒
            else:
                t_e = t + pd.Timedelta(minutes=tf_min)
                if t_e not in g1.index:
                    break
                ep = g1.open.loc[t_e]
                ref = i
            if stop_mode == "range":
                stop = lo if s == 1 else hi
            elif stop_mode == "swing":
                seg = g.iloc[max(ref - swing_n + 1, 0):ref + 1]
                stop = seg.low.min() if s == 1 else seg.high.max()
            else:
                stop = ep - s * k * atr.iloc[ref]
            R = s * (ep - stop)
            if not R > 0:
                break
            r = run_path(g1, t_e, s, ep, stop, exit_h * 60)
            if r is None:
                break
            xp, xt, why = r
            pnl = s * (xp - ep) - g1.sp.loc[t_e]
            out.append(dict(day=day, side=s, R=R, pnl=pnl, pnl_R=pnl / R, exit=why))
            break
    return pd.DataFrame(out)


def orb(open_hm, or_min, tf_min=1, win_min=180, exit_h=19, entry="touch"):
    """開盤區間突破。open_hm = (時, 分) broker。"""
    o0 = open_hm[0] * 60 + open_hm[1]
    out = []
    for day, g1 in DAYS.items():
        h = hm(g1)
        rng = g1[(h >= o0) & (h < o0 + or_min)]
        if len(rng) < or_min * 0.8:
            continue
        hi, lo = rng.high.max(), rng.low.min()
        g = resample(g1, f"{tf_min}min") if tf_min > 1 else g1
        gh = hm(g)
        win = g[(gh >= o0 + or_min) & (gh < o0 + or_min + win_min) & (gh < exit_h * 60)]
        for t, b in win.iterrows():
            ref_hi, ref_lo = (b.high, b.low) if entry == "touch" else (b.close, b.close)
            s = 1 if ref_hi > hi else -1 if ref_lo < lo else 0
            if not s:
                continue
            if entry == "touch":
                sub = g1[(g1.index >= t) & (g1.index < t + pd.Timedelta(minutes=tf_min))]
                hit = sub[(sub.high > hi) if s == 1 else (sub.low < lo)]
                t_e = hit.index[0]
                ep = max(hi, hit.open.iloc[0]) if s == 1 else min(lo, hit.open.iloc[0])
            else:
                t_e = t + pd.Timedelta(minutes=tf_min)
                if t_e not in g1.index:
                    break
                ep = g1.open.loc[t_e]
            stop = lo if s == 1 else hi
            R = s * (ep - stop)
            if not R > 0:
                break
            r = run_path(g1, t_e, s, ep, stop, exit_h * 60)
            if r is None:
                break
            xp, xt, why = r
            pnl = s * (xp - ep) - g1.sp.loc[t_e]
            out.append(dict(day=day, side=s, R=R, pnl=pnl, pnl_R=pnl / R, exit=why))
            break
    return pd.DataFrame(out)


def summ(T):
    if len(T) == 0:
        return "無交易"
    r = T.pnl_R
    h = T.day < pd.Timestamp("2026-08-15")
    eq = r.cumsum()
    return (f"{len(T):3d} 筆 期望 {r.mean():+.3f}R t={r.mean() / r.std() * np.sqrt(len(r)):+.2f} 勝率 {(r > 0).mean():.2f} "
            f"中位R {T.R.median():5.2f}點 前 {r[h].mean():+.2f} 後 {r[~h].mean():+.2f} 累計 {r.sum():+5.1f}R 回撤 {(eq.cummax() - eq).max():.1f}R")
