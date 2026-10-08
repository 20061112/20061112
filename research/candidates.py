"""以寬鬆參數產生候選訊號，並為每個候選計算更多特徵，用來比較「被標記」與「未被標記」者。"""
import numpy as np
import pandas as pd
from detector import detect, Params


def enrich(f, sig):
    o, h, l, c, A = (f[k].to_numpy() for k in ("open", "high", "low", "close", "atr60"))
    tv = f.tickvol.to_numpy()
    rows = []
    for r in sig.itertuples():
        t = f.index.get_loc(r.time); e = f.index.get_loc(r.ext_time); s = r.side; a = r.atr
        d = {}
        if e < 240:
            rows.append(d); continue
        for L in (30, 60, 120, 240):
            d[f"x{L}"] = int((l[e] <= l[e - L:e].min()) if s == 1 else (h[e] >= h[e - L:e].max()))
        for N in (5, 8, 20, 30, 60):
            seg = slice(e - N, e + 1)
            d[f"run{N}"] = ((h[seg].max() - l[e]) if s == 1 else (h[e] - l[seg].min())) / a
        d["run60_usd"] = d["run60"] * a
        # 前 3 根的速度 / 最後一段是否加速
        d["last3"] = -s * (c[e] - c[e - 3]) / a
        rng = max(h[e] - l[e], 1e-9)
        d["wick"] = (((min(o[e], c[e]) - l[e]) if s == 1 else (h[e] - max(o[e], c[e]))) / rng)
        d["ext_rng"] = rng / a
        d["tv_e"] = tv[e] / np.median(tv[e - 60:e])
        d["tv_t"] = tv[t] / np.median(tv[t - 60:t])
        d["conf_body"] = s * (c[t] - o[t]) / a
        d["bars"] = t - e
        d["atr_ratio"] = a / np.nanmean(A[e - 1440:e]) if e > 1440 else np.nan  # 當下波動 vs 過去一天
        d["hour"] = f.index[t].hour
        rows.append(d)
    return pd.concat([sig.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


LOOSE = Params(swing_len=10, run_len=12, run_atr=2.0, er_min=0.0, max_wait=4, confirm_atr=0.5)


def add_context(f, X):
    """額外的「延伸程度」特徵：乖離、RSI、z-score（皆以極值 K 棒時點計算，無未來資料）。"""
    c = f.close
    ctx = pd.DataFrame(index=f.index)
    for n in (20, 50, 200):
        ctx[f"dev_ema{n}"] = (c - c.ewm(span=n, adjust=False).mean()) / f.atr60
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    ctx["rsi14"] = 100 - 100 / (1 + up / dn)
    for n in (60, 240):
        ctx[f"z{n}"] = (c - c.rolling(n).mean()) / c.rolling(n).std()
    # 當日 VWAP（以 tick volume 加權）乖離
    tp = (f.high + f.low + f.close) / 3
    day = f.index.normalize()
    vw = (tp * f.tickvol).groupby(day).cumsum() / f.tickvol.groupby(day).cumsum()
    ctx["dev_vwap"] = (c - vw) / f.atr60
    ctx["tv_max5"] = f.tickvol.rolling(5).max() / f.tickvol.rolling(120).median()
    e = f.index.get_indexer(X.ext_time)
    side = X.side.to_numpy()
    out = X.copy()
    for col in ctx:
        v = ctx[col].to_numpy()[e]
        if col == "rsi14":
            v = np.where(side == 1, 100 - v, v)  # 統一成「越大 = 越超買/超賣（越延伸）」
        elif col.startswith(("dev", "z")):
            v = -side * v
        out[col] = v
    return out
