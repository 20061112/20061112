"""趨勢段後的晨星/夜星/吞噬反轉單 — 回測框架。

規則（多單；空單鏡像）：
  型態（在 K 棒 t 收盤時判定）
    star   : t-2 為陰線且實體 >= 0.5 ATR；t-1 實體 <= t-2 實體的一半；t 為陽線且收盤 > t-2 實體中點
    engulf : t-1 陰線、t 陽線，t 收盤 >= t-1 開盤，t 實體 >= t-1 實體
  趨勢背景
    型態最低點 = 過去 swing 根最低點
    trend_len 根內最高點到型態最低點 >= trend_atr × ATR14
  進場
    close : t 收盤進場
    break : 之後 wait 根內突破 t 高點才進場（stop 單）；若先跌破型態低點則取消
  出場：停損 = 型態低點 − buf×ATR；停利 = tp × R；超過 hold 根以收盤平倉
  成本：每筆扣當下點差（MT5 K 棒為 bid 價，spread 單位 0.01 美元）
  同一時間只持有一筆；同一根 K 棒同時碰到停損與停利時，一律算停損（保守）。
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


def resample(m1, rule):
    if rule == "1min":
        return m1
    return m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "max"}).dropna()


@dataclass(frozen=True)
class Cfg:
    pattern: str = "both"      # star / engulf / both
    swing: int = 20
    trend_len: int = 30
    trend_atr: float = 6.0
    entry: str = "break"       # close / break
    wait: int = 3
    buf: float = 0.2
    tp: float = 2.0
    hold: int = 60


def _arrays(df):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    return o, h, l, c, atr, df.spread.to_numpy(float) * 0.01


def find_setups(df, cfg: Cfg):
    """回傳所有型態成立的 K 棒（尚未考慮進場與持倉重疊）。"""
    o, h, l, c, atr, _ = _arrays(df)
    n = len(df)
    body = c - o
    rows = []
    start = max(cfg.swing, cfg.trend_len) + 15
    for t in range(start, n - 1):
        A = atr[t]
        if not A > 0:
            continue
        for s in (1, -1):
            b2, b1, b0 = s * body[t - 2], s * body[t - 1], s * body[t]   # s=1：反轉向上，前段為陰線（負值）
            star = (-b2 >= 0.5 * A and abs(b1) <= 0.5 * -b2 and b0 > 0 and
                    s * (c[t] - (o[t - 2] + c[t - 2]) / 2) > 0)
            engulf = (b1 < 0 and b0 > 0 and s * (c[t] - o[t - 1]) >= 0 and b0 >= -b1)
            kind = "star" if star else "engulf" if engulf else None
            if kind is None or (cfg.pattern != "both" and cfg.pattern != kind):
                continue
            k = 3 if kind == "star" else 2
            if s == 1:
                ext = l[t - k + 1:t + 1].min()
                if ext > l[t - cfg.swing + 1:t + 1].min():
                    continue
                trend = h[t - cfg.trend_len:t + 1].max() - ext
            else:
                ext = h[t - k + 1:t + 1].max()
                if ext < h[t - cfg.swing + 1:t + 1].max():
                    continue
                trend = ext - l[t - cfg.trend_len:t + 1].min()
            if trend < cfg.trend_atr * A:
                continue
            rows.append((t, s, kind, ext, A, trend / A))
    return rows


def backtest(df, cfg: Cfg, setups=None):
    """setups 可傳入預先算好的 find_setups 結果（以較寬鬆 trend_atr 算出，這裡再過濾），加速參數掃描。"""
    o, h, l, c, atr, spr = _arrays(df)
    n = len(df)
    trades = []
    busy_until = -1
    if setups is None:
        setups = find_setups(df, cfg)
    for t, s, kind, ext, A, trend in setups:
        if trend < cfg.trend_atr or (cfg.pattern != "both" and kind != cfg.pattern):
            continue
        if t <= busy_until:
            continue
        stop = ext - s * cfg.buf * A
        # 進場
        if cfg.entry == "close":
            ei, ep = t, c[t]
        else:
            trig = h[t] if s == 1 else l[t]
            ei = None
            for j in range(t + 1, min(t + 1 + cfg.wait, n)):
                if (s == 1 and l[j] <= stop) or (s == -1 and h[j] >= stop):
                    break                                   # 先破極值 → 取消
                if (s == 1 and h[j] > trig) or (s == -1 and l[j] < trig):
                    ei, ep = j, max(trig, o[j]) if s == 1 else min(trig, o[j])
                    break
            if ei is None:
                continue
        R = s * (ep - stop)
        if R <= 0:
            continue
        tgt = ep + s * cfg.tp * R
        # 出場（進場那根若是 stop 單觸發，該根剩餘走勢未知，從下一根開始檢查）
        xi, xp, why = None, None, "time"
        for j in range(ei + 1, min(ei + 1 + cfg.hold, n)):
            hit_stop = (l[j] <= stop) if s == 1 else (h[j] >= stop)
            hit_tgt = (h[j] >= tgt) if s == 1 else (l[j] <= tgt)
            if hit_stop:
                xi, xp, why = j, (min(stop, o[j]) if s == 1 else max(stop, o[j])), "stop"; break
            if hit_tgt:
                xi, xp, why = j, tgt, "tp"; break
        if xi is None:
            xi = min(ei + cfg.hold, n - 1); xp = c[xi]
        pnl = s * (xp - ep) - spr[ei]
        trades.append(dict(setup=df.index[t], entry_time=df.index[ei], exit_time=df.index[xi], side=s,
                           kind=kind, trend_atr=trend, entry=ep, stop=stop, R_usd=R, exit=why,
                           pnl_usd=pnl, pnl_R=pnl / R))
        busy_until = xi
    return pd.DataFrame(trades)


def stats(tr):
    if len(tr) == 0:
        return dict(n=0, win=np.nan, expR=np.nan, t=np.nan, pf=np.nan, totR=0.0, maxddR=np.nan)
    r = tr.pnl_R.to_numpy()
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), win=(r > 0).mean(), expR=r.mean(), t=r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan,
                pf=gp / gl if gl > 0 else np.inf, totR=r.sum(), maxddR=dd)


def backtest_m1(df, m1, cfg: Cfg, tf_minutes, setups=None):
    """型態在 df（較大週期）上判定，進出場改用 M1 逐根模擬，消除 K 棒內順序的不確定。
    同一根 M1 同時觸及停損與停利 → 算停損；突破進場那根 M1 若也碰到停損 → 算停損。"""
    _, h, l, c, atr, _ = _arrays(df)
    mo, mh, ml, mc = (m1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    mspr = m1.spread.to_numpy(float) * 0.01
    mt = m1.index
    if setups is None:
        setups = find_setups(df, cfg)
    trades, busy = [], pd.Timestamp.min
    step = pd.Timedelta(minutes=tf_minutes)
    for t, s, kind, ext, A, trend in setups:
        if trend < cfg.trend_atr or (cfg.pattern != "both" and kind != cfg.pattern):
            continue
        t_close = df.index[t] + step                       # 型態 K 棒收盤時間
        if t_close <= busy:
            continue
        i0 = mt.searchsorted(t_close)
        stop = ext - s * cfg.buf * A
        wait_end = mt.searchsorted(t_close + cfg.wait * step)
        if cfg.entry == "close":
            ei, ep = i0, mo[i0] if i0 < len(mo) else np.nan
            if not np.isfinite(ep) or s * (ep - stop) <= 0:
                continue
        else:
            trig = h[t] if s == 1 else l[t]
            ei = None
            for j in range(i0, min(wait_end, len(mo))):
                up = mh[j] > trig if s == 1 else ml[j] < trig
                dn = ml[j] <= stop if s == 1 else mh[j] >= stop
                if up:
                    ei, ep = j, (max(trig, mo[j]) if s == 1 else min(trig, mo[j]))
                    break
                if dn:
                    break
            if ei is None:
                continue
        R = s * (ep - stop)
        tgt = ep + s * cfg.tp * R
        end = mt.searchsorted(mt[ei] + cfg.hold * step)
        xi, why = None, "time"
        for j in range(ei, min(end, len(mo))):
            hs = ml[j] <= stop if s == 1 else mh[j] >= stop
            ht = mh[j] >= tgt if s == 1 else ml[j] <= tgt
            if j == ei and cfg.entry == "break":
                ht = False                                  # 進場那根不給停利（保守）
            if hs:
                xi, xp, why = j, (min(stop, mo[j]) if s == 1 and j > ei else stop if j == ei else max(stop, mo[j]) if s == -1 else stop), "stop"; break
            if ht:
                xi, xp, why = j, tgt, "tp"; break
        if xi is None:
            xi = min(end, len(mo)) - 1; xp = mc[xi]
        pnl = s * (xp - ep) - mspr[ei]
        trades.append(dict(setup=df.index[t], entry_time=mt[ei], exit_time=mt[xi], side=s, kind=kind,
                           trend_atr=trend, entry=ep, stop=stop, R_usd=R, exit=why, pnl_usd=pnl, pnl_R=pnl / R))
        busy = mt[xi]
    return pd.DataFrame(trades)
