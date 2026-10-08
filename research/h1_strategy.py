"""H1「衝刺竭盡反轉」策略（M15）。

  1. 一段走勢後出現晨星/夜星或吞噬（極值為 10 根最低/高，40 根內反向幅度 >= 2 ATR）
  2. 型態極值 K 棒「收盤」在布林帶 (bb_len, bb_k) 外軌之外
  3. 最後一段在加速：|ROC(roc_n)| 於極值 K 棒 > roc_n 根之前（衝刺而非鈍化）
  4. 型態後 3 根內突破型態 K 棒高/低點進場；停損極值外 0.1 ATR；停利 tp R；最長 16 根
"""
import numpy as np
import pandas as pd
from backtest import Cfg, find_setups, backtest_bars, _arrays

BASE = Cfg(pattern="both", swing=10, trend_len=40, trend_atr=2.0, entry="break", wait=3, tp=2.0, buf=0.1, hold=16)


def h1_setups(d, cfg=BASE, bb_len=20, bb_k=2.0, roc_n=3, require_accel=True, require_outside=True):
    _, h, l, c, atr, _ = _arrays(d)
    cs = d.close
    ma, sd = cs.rolling(bb_len).mean().to_numpy(), cs.rolling(bb_len).std().to_numpy()
    up, dn = ma + bb_k * sd, ma - bb_k * sd
    roc = np.abs((cs - cs.shift(roc_n)).to_numpy())
    out = []
    for st in find_setups(d, cfg):
        t, s, kind = st[0], st[1], st[2]
        k = 3 if kind == "star" else 2
        seg = range(t - k + 1, t + 1)
        e = min(seg, key=lambda i: l[i]) if s == 1 else max(seg, key=lambda i: h[i])
        outside = c[e] < dn[e] if s == 1 else c[e] > up[e]
        accel = roc[e] > roc[e - roc_n]
        if (outside or not require_outside) and (accel or not require_accel):
            out.append(st)
    return out


def run_h1(d, cfg=BASE, **kw):
    return backtest_bars(d, cfg, h1_setups(d, cfg, **kw))
