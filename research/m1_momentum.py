"""第五部分：M1 晨星/夜星/吞噬 + 多時框 ER + K 線速度/加速度 + 背離。

候選：M1 型態，極值為 swing 根內最高/低；型態後 3 根內突破型態 K 棒高/低點進場，
停損極值外 0.1 ATR，停利 2R，最長 60 分鐘（可重疊，用於特徵研究）。

所有較大週期指標都只用「已收完」的 K 棒（對 M1 時間 t，取 t 之前最後一根已收盤的 M5/M15/H1）。
特徵方向統一：多單看「前段下跌」，空單鏡像；數值意義寫在每個欄位旁。
"""
import numpy as np
import pandas as pd
from backtest import Cfg, find_setups, resample, _arrays
from mr_features import simulate

CAND = Cfg(pattern="both", swing=15, trend_len=20, trend_atr=0.0, entry="break", wait=3, tp=2.0, buf=0.1, hold=60)


def efficiency(c, n):
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


def htf_to_m1(m1_index, htf_series, minutes):
    """把較大週期指標對齊到 M1：M1 時間 t 只能看到收盤時間 <= t 的 HTF K 棒（K 棒開盤時間 + 週期 <= t）。"""
    s = htf_series.copy()
    s.index = s.index + pd.Timedelta(minutes=minutes)       # 改成收盤時間
    return s.reindex(m1_index, method="ffill")


def indicators(m1):
    c = m1.close
    pc = c.shift()
    tr = pd.concat([m1.high - m1.low, (m1.high - pc).abs(), (m1.low - pc).abs()], axis=1).max(axis=1)
    X = pd.DataFrame(index=m1.index)
    X["atr"] = tr.rolling(14).mean()
    X["atr60"] = tr.rolling(60).mean()
    # 多時框 ER
    for n in (5, 10, 20, 60):
        X[f"er_m1_{n}"] = efficiency(c, n)
    for rule, mins, n in (("5min", 5, 12), ("15min", 15, 16), ("60min", 60, 12)):
        h = resample(m1, rule)
        X[f"er_{rule}"] = htf_to_m1(m1.index, efficiency(h.close, n), mins)
        # 大週期方向（+1 上漲）
        X[f"dir_{rule}"] = htf_to_m1(m1.index, np.sign(h.close - h.close.shift(n)), mins)
    # 速度 / 加速度（以 ATR60 為單位，每根 K 棒的平均移動）
    for n in (3, 5):
        X[f"vel{n}"] = (c - c.shift(n)) / (n * X.atr60)
        X[f"acc{n}"] = X[f"vel{n}"] - X[f"vel{n}"].shift(n)
    # 動能指標（用於背離）
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    X["rsi"] = 100 - 100 / (1 + up / dn)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    X["macd_h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / X.atr60
    return X


def candidate_table(m1, cfg=CAND, div_len=40):
    X = indicators(m1)
    o, h, l, c, _, spr = _arrays(m1)
    A60 = X.atr60.to_numpy()
    cols = {k: X[k].to_numpy() for k in X.columns}
    rows = []
    for t, s, kind, ext, A, trend in find_setups(m1, cfg):
        if t < 300 or not np.isfinite(A60[t]):
            continue
        stop = ext - s * cfg.buf * A
        trig = h[t] if s == 1 else l[t]
        r = simulate(o, h, l, c, spr, t, s, stop, trig, lambda ep, R: ep + s * 2.0 * R, cfg.wait, cfg.hold)
        if r is None:
            continue
        ei, xi, ep, R, pnl = r
        k = 3 if kind == "star" else 2
        seg = range(t - k + 1, t + 1)
        e = min(seg, key=lambda i: l[i]) if s == 1 else max(seg, key=lambda i: h[i])
        f = dict(setup=m1.index[t], t=t, entry_time=m1.index[ei], exit_time=m1.index[xi], side=s, kind=kind,
                 R_usd=R, R_atr=R / A60[t], pnl_R=pnl, trend_atr=trend)
        # --- 多時框 ER（型態收盤當下可見）
        for key in ("er_m1_5", "er_m1_10", "er_m1_20", "er_m1_60", "er_5min", "er_15min", "er_60min"):
            f[key] = cols[key][t]
        f["er_ratio_m1_5_60"] = cols["er_m1_5"][e] / max(cols["er_m1_60"][e], 1e-6)
        f["er_ratio_m1_m15"] = cols["er_m1_20"][e] / max(cols["er_15min"][t], 1e-6)
        f["er_ratio_m5_h1"] = cols["er_5min"][t] / max(cols["er_60min"][t], 1e-6)
        for rule in ("5min", "15min", "60min"):
            f[f"align_{rule}"] = s * cols[f"dir_{rule}"][t]          # +1 = 反轉方向與大週期一致
        # --- 速度 / 加速度（統一方向：正 = 往原趨勢方向，例如多單時 = 下跌速度）
        for n in (3, 5):
            f[f"vel{n}_e"] = -s * cols[f"vel{n}"][e]                  # 極值時仍朝原方向的速度
            f[f"acc{n}_e"] = -s * cols[f"acc{n}"][e]                  # 正 = 仍在加速，負 = 已減速
            f[f"vel{n}_t"] = s * cols[f"vel{n}"][t]                   # 確認棒時反轉方向的速度
        # 速度背離：極值創新低，但最後一段的最大下跌速度 < 之前 div_len 根內的最大下跌速度
        w0, w1 = max(e - div_len, 0), max(e - 5, 1)
        v3 = -s * cols["vel3"]
        f["vel_div"] = np.nanmax(v3[w0:w1]) - np.nanmax(v3[e - 3:e + 1])   # 正 = 背離（最後一段較慢）
        rsi = cols["rsi"] if s == 1 else 100 - cols["rsi"]
        f["rsi_div"] = rsi[e] - np.nanmin(rsi[w0:w1])                       # 正 = RSI 沒有創新低
        mh = s * cols["macd_h"]
        f["macd_div"] = mh[e] - np.nanmin(mh[w0:w1])                         # 正 = MACD 柱沒有創新低
        f["rsi_e"] = rsi[e]
        rows.append(f)
    return pd.DataFrame(rows)
