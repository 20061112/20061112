"""M15 反轉候選 + 布林帶 / 低延遲指標 / 微觀特徵。所有特徵只用型態 K 棒收盤（含）之前的資料。"""
import numpy as np
import pandas as pd
from backtest import Cfg, find_setups, backtest_bars, _arrays

LOOSE = Cfg(pattern="both", swing=10, trend_len=40, trend_atr=2.0, entry="break", wait=3, tp=2.0, buf=0.1, hold=16)


def wma(x, n):
    w = np.arange(1, n + 1)
    return x.rolling(n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True)


def indicators(d):
    c, h, l, o, v = d.close, d.high, d.low, d.open, d.tickvol
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    X = pd.DataFrame(index=d.index)
    X["atr"] = tr.rolling(14).mean()
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    X["bb_mid"], X["bb_up"], X["bb_dn"] = ma, ma + 2 * sd, ma - 2 * sd
    X["bbw"] = 4 * sd / ma
    X["bbw_pct"] = X.bbw.rolling(200).rank(pct=True)              # 帶寬在過去 200 根的百分位（低 = 收縮）
    X["bbw_chg3"] = X.bbw / X.bbw.shift(3) - 1                     # 帶寬近 3 根變化（正 = 擴張中）
    X["bbw_chg10"] = X.bbw / X.bbw.shift(10) - 1
    kc_w = 1.5 * X.atr
    X["squeeze"] = ((ma + 2 * sd) < (ma + kc_w)).astype(int)       # BB 在 KC 內 = 擠壓
    X["squeeze_recent"] = X["squeeze"].rolling(20).max()
    X["pctb"] = (c - X.bb_dn) / (X.bb_up - X.bb_dn)
    d1 = c.diff()
    for n in (2, 14):
        up = d1.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d1.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        X[f"rsi{n}"] = 100 - 100 / (1 + up / dn)
    hma = wma(2 * wma(c, 8) - wma(c, 16), 4)                        # Hull MA(16)：低延遲
    X["hma_slope"] = hma.diff() / X.atr
    X["hma_slope_prev"] = X.hma_slope.shift(1)
    X["roc3"] = (c - c.shift(3)) / X.atr
    X["er20"] = (c - c.shift(20)).abs() / d1.abs().rolling(20).sum()
    X["tv_rel"] = v / v.rolling(96).median()
    X["atr_regime"] = X.atr / X.atr.rolling(96 * 10).mean()        # 當下波動 vs 近 10 天
    day = d.index.normalize()
    dh, dl = h.groupby(day).max(), l.groupby(day).min()
    X["pdh"] = pd.Series(day, index=d.index).map(dh.shift(1))
    X["pdl"] = pd.Series(day, index=d.index).map(dl.shift(1))
    return X


def candidate_table(d, cfg=LOOSE):
    X = indicators(d)
    st = find_setups(d, cfg)
    tr = backtest_bars(d, cfg, st, overlap=True)
    o, h, l, c, atr, _ = _arrays(d)
    v = d.tickvol.to_numpy(float)
    rows = []
    for r in tr.itertuples():
        t, s = r.t, r.side
        k = 3 if r.kind == "star" else 2
        seg = range(t - k + 1, t + 1)
        e = min(seg, key=lambda i: l[i]) if s == 1 else max(seg, key=lambda i: h[i])   # 極值 K 棒
        A = X.atr.iat[t]
        f = {}
        f["bbw_pct"] = X.bbw_pct.iat[t]
        f["bbw_chg3"] = X.bbw_chg3.iat[t]
        f["bbw_chg10"] = X.bbw_chg10.iat[t]
        f["squeeze_recent"] = X.squeeze_recent.iat[t]
        # 極值超出布林帶的程度（統一成 正 = 往趨勢方向超出外軌）
        f["band_pierce"] = ((X.bb_dn.iat[e] - l[e]) if s == 1 else (h[e] - X.bb_up.iat[e])) / A
        f["close_outside"] = int((c[e] < X.bb_dn.iat[e]) if s == 1 else (c[e] > X.bb_up.iat[e]))
        f["pctb_t"] = X.pctb.iat[t] if s == 1 else 1 - X.pctb.iat[t]   # 確認棒收盤回到帶內的位置
        f["rsi2_e"] = X.rsi2.iat[e] if s == 1 else 100 - X.rsi2.iat[e]  # 越小 = 越超賣/超買
        f["rsi14_e"] = X.rsi14.iat[e] if s == 1 else 100 - X.rsi14.iat[e]
        # RSI 背離：極值創新低，但 RSI14 高於趨勢段內的 RSI 最低點
        w = slice(t - 40, e)
        rs = X.rsi14.iloc[w] if s == 1 else 100 - X.rsi14.iloc[w]
        f["rsi_div"] = f["rsi14_e"] - rs.min()
        f["hma_turn"] = int(s * X.hma_slope.iat[t] > 0 and s * X.hma_slope_prev.iat[t] <= 0)
        f["hma_slope"] = s * X.hma_slope.iat[t]
        f["roc_decel"] = abs(X.roc3.iat[e]) - abs(X.roc3.iat[max(e - 3, 0)])  # 負 = 最後一段在減速
        f["er20"] = X.er20.iat[t]
        rng = max(h[e] - l[e], 1e-9)
        f["wick_e"] = ((min(o[e], c[e]) - l[e]) if s == 1 else (h[e] - max(o[e], c[e]))) / rng
        f["rng_e"] = rng / A
        f["conf_rng"] = (h[t] - l[t]) / A
        f["tv_e"] = X.tv_rel.iat[e]
        f["tv_t"] = X.tv_rel.iat[t]
        f["tv_push"] = v[e - 2:e + 1].mean() / max(v[e - 8:e - 2].mean(), 1)   # 最後一段量能 vs 之前
        f["atr_regime"] = X.atr_regime.iat[t]
        ref = X.pdl.iat[t] if s == 1 else X.pdh.iat[t]
        f["dist_pd"] = (s * ((l[e] if s == 1 else h[e]) - ref)) / A if np.isfinite(ref) else np.nan  # 0 附近 = 碰到前日高/低
        f["hour"] = d.index[t].hour
        rows.append(f)
    return pd.concat([tr.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
