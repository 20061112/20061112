"""插針（影線）KDE 支撐壓力價位。

1. 插針：在 pin TF（M15 / H1）上，影線 >= wick_frac × K 棒全長，且影線 >= wick_atr × ATR14。
   上影線 → 價位 = high（壓力型針），下影線 → 價位 = low（支撐型針）。
   權重 = 影線長度 / ATR（上限 3）。
2. KDE：以權重做高斯核密度，頻寬 h = bw × 視窗內 pin TF ATR 中位數。
3. 價位 = 密度的局部極大值，強度 = 峰值 / 價格網格上平均密度（只保留 >= min_strength），
   兩峰距離 < sep × h 只留較強者。
所有計算只用到視窗內（已收盤）的 K 棒。
"""
import numpy as np
import pandas as pd


def find_pins(df, A, wick_frac=0.5, wick_atr=0.8):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    rng = h - l
    up = h - np.maximum(o, c)
    dn = np.minimum(o, c) - l
    with np.errstate(invalid="ignore", divide="ignore"):
        is_up = (up >= wick_frac * rng) & (up >= wick_atr * A) & (rng > 0)
        is_dn = (dn >= wick_frac * rng) & (dn >= wick_atr * A) & (rng > 0)
    rows = []
    for m, price, wick, side in ((is_up, h, up, -1), (is_dn, l, dn, 1)):
        idx = np.flatnonzero(m)
        rows.append(pd.DataFrame({"time": df.index[idx], "price": price[idx],
                                  "w": np.minimum(wick[idx] / A[idx], 3.0), "side": side,
                                  "atr": A[idx]}))
    return pd.concat(rows).sort_values("time").set_index("time")


def kde_peaks(prices, weights, h, grid_step=None, min_strength=1.5, sep=2.0):
    """回傳 DataFrame(price, strength)，依強度由大到小。"""
    if len(prices) < 3:
        return pd.DataFrame(columns=["price", "strength"])
    step = grid_step or h / 5
    g = np.arange(prices.min() - 3 * h, prices.max() + 3 * h, step)
    z = (g[:, None] - prices[None, :]) / h
    dens = (np.exp(-0.5 * z * z) * weights[None, :]).sum(1)
    # 強度以「價格範圍內平均密度」為基準
    inside = (g >= prices.min()) & (g <= prices.max())
    base = dens[inside].mean() if inside.any() else dens.mean()
    pk = np.flatnonzero((dens[1:-1] > dens[:-2]) & (dens[1:-1] >= dens[2:])) + 1
    pk = pk[dens[pk] / base >= min_strength]
    pk = pk[np.argsort(-dens[pk])]
    keep = []
    for p in pk:
        if all(abs(g[p] - g[q]) >= sep * h for q in keep):
            keep.append(p)
    return pd.DataFrame({"price": g[keep], "strength": dens[keep] / base})


def levels_from_pins(pins, bw=0.5, side=None, **kw):
    p = pins if side is None else pins[pins.side == side]
    if len(p) < 3:
        return pd.DataFrame(columns=["price", "strength"])
    h = bw * np.nanmedian(pins.atr)
    out = kde_peaks(p.price.to_numpy(), p.w.to_numpy(), h, **kw)
    out["h"] = h
    return out
