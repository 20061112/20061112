"""第六部分：均線密集 / sigma 密集後的突破（波動收縮 → 擴張，順突破方向做）。

訊號（在 K 棒 t 收盤時判定，下一根開盤進場）：
  1. 壓縮：過去 lookback 根內，曾有至少 min_bars 根處於「壓縮」狀態
       ma    : 三條均線最大 - 最小 的距離 / ATR，位於自身過去 rank_len 根的下 pct 分位
       sigma : 布林帶寬 (std20 / ATR) 位於下 pct 分位
       none  : 不要求壓縮（對照組：單純突破）
  2. 突破：收盤 > 過去 box 根最高點（做多）/ < 最低點（做空）
  3. 發散確認（可選）：三條均線依序排列（快 > 中 > 慢）且距離比 3 根前大
出場：
  stop : 箱體另一側（box）或箱體中點（mid）
  exit : 固定 kR，或 trail = 收盤跌破中均線出場（同時保留初始停損）；最長 hold 根
保守處理：同一根同時碰停損與停利 → 停損；每筆扣進場當下點差。一次只持有一筆。
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SqCfg:
    ma_type: str = "ema"        # ema / sma
    ma_lens: tuple = (10, 20, 50)
    comp: str = "ma"            # ma / sigma / none
    pct: float = 0.2
    rank_len: int = 500
    lookback: int = 10
    min_bars: int = 5
    box: int = 20
    fan: bool = True
    stop: str = "box"           # box / mid
    exit: str = "2R"            # 1.5R / 2R / 3R / trail
    hold: int = 120


def prep(d, cfg: SqCfg):
    c, h, l = d.close, d.high, d.low
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    mas = [c.ewm(span=n, adjust=False).mean() if cfg.ma_type == "ema" else c.rolling(n).mean() for n in cfg.ma_lens]
    M = pd.concat(mas, axis=1)
    spread = (M.max(axis=1) - M.min(axis=1)) / atr
    bbw = c.rolling(20).std() / atr
    metric = spread if cfg.comp == "ma" else bbw
    rank = metric.rolling(cfg.rank_len).rank(pct=True)
    comp_state = (rank <= cfg.pct).astype(int)
    comp_recent = comp_state.rolling(cfg.lookback).sum().shift(1) >= cfg.min_bars
    if cfg.comp == "none":
        comp_recent = pd.Series(True, index=d.index)
    hi_box = h.rolling(cfg.box).max().shift(1)
    lo_box = l.rolling(cfg.box).min().shift(1)
    f, m, s = mas
    fan_up = (f > m) & (m > s) & (spread > spread.shift(3))
    fan_dn = (f < m) & (m < s) & (spread > spread.shift(3))
    long_sig = comp_recent & (c > hi_box) & (fan_up if cfg.fan else True)
    short_sig = comp_recent & (c < lo_box) & (fan_dn if cfg.fan else True)
    return dict(o=d.open.to_numpy(float), h=h.to_numpy(float), l=l.to_numpy(float), c=c.to_numpy(float),
                mid_ma=m.to_numpy(float), hi_box=hi_box.to_numpy(float), lo_box=lo_box.to_numpy(float),
                spr=d.spread.to_numpy(float) * 0.01, long=long_sig.fillna(False).to_numpy(),
                short=short_sig.fillna(False).to_numpy(), idx=d.index)


def run(d, cfg: SqCfg, P=None):
    P = P or prep(d, cfg)
    o, h, l, c, mid_ma = P["o"], P["h"], P["l"], P["c"], P["mid_ma"]
    n = len(c)
    trades, t = [], cfg.rank_len
    while t < n - 2:
        s = 1 if P["long"][t] else -1 if P["short"][t] else 0
        if s == 0:
            t += 1
            continue
        hb, lb = P["hi_box"][t], P["lo_box"][t]
        stop = (lb if s == 1 else hb) if cfg.stop == "box" else (hb + lb) / 2
        ei = t + 1
        ep = o[ei]
        R = s * (ep - stop)
        if not R > 0:
            t += 1
            continue
        tgt = ep + s * float(cfg.exit[:-1]) * R if cfg.exit != "trail" else None
        xi, why = None, "time"
        for j in range(ei, min(ei + cfg.hold, n)):
            if (l[j] <= stop) if s == 1 else (h[j] >= stop):
                xp = (min(stop, o[j]) if s == 1 else max(stop, o[j])) if j > ei else stop
                xi, why = j, "stop"; break
            if tgt is not None and ((h[j] >= tgt) if s == 1 else (l[j] <= tgt)):
                xi, xp, why = j, tgt, "tp"; break
            if tgt is None and ((c[j] < mid_ma[j]) if s == 1 else (c[j] > mid_ma[j])):
                xi, xp, why = j, c[j], "trail"; break
        if xi is None:
            xi = min(ei + cfg.hold, n) - 1; xp = c[xi]
        pnl = s * (xp - ep) - P["spr"][ei]
        trades.append(dict(signal=P["idx"][t], entry_time=P["idx"][ei], exit_time=P["idx"][xi], side=s,
                           R_usd=R, exit=why, pnl_usd=pnl, pnl_R=pnl / R))
        t = xi + 1
    return pd.DataFrame(trades)
