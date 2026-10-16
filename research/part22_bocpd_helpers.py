"""part22 共用的檢定函式（事件研究、隨機方向）。"""
import math
import numpy as np
import pandas as pd


def event_study(b, sig, hors):
    """訊號 K 棒收盤 → 之後 h 根收盤的報酬 × 方向 / ATR。同時給出扣掉無條件平均漂移後的數字。"""
    c, atr = b.close.to_numpy(), b.atr.to_numpy()
    n = len(c)
    rows = []
    for h in hors:
        fwd = np.full(n, np.nan); fwd[: n - h] = (c[h:] - c[: n - h]) / atr[: n - h]
        base = np.nanmean(fwd)                                    # 整段的平均漂移（多頭）
        i = np.where((sig != 0) & np.isfinite(fwd))[0]
        d = sig[i]
        x = fwd[i] * d
        xa = (fwd[i] - base) * d
        se = x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else np.nan
        # 訊號彼此重疊時 t 會高估 → 另外給「不重疊」版本（每 h 根最多取一個）
        keep, last = [], -10 ** 9
        for j in i:
            if j - last >= h:
                keep.append(j); last = j
        keep = np.array(keep, int)
        xn = fwd[keep] * sig[keep] - base * sig[keep]
        rows.append(dict(h=h, n=len(x), raw=x.mean() if len(x) else np.nan, adj=xa.mean() if len(x) else np.nan,
                         t_adj=xa.mean() / se if len(x) > 1 else np.nan, hit=(x > 0).mean() if len(x) else np.nan,
                         n_nov=len(keep), adj_nov=xn.mean() if len(keep) else np.nan,
                         t_nov=xn.mean() / xn.std(ddof=1) * math.sqrt(len(xn)) if len(keep) > 2 else np.nan,
                         long_adj=xa[d > 0].mean() if (d > 0).any() else np.nan,
                         short_adj=xa[d < 0].mean() if (d < 0).any() else np.nan))
    return pd.DataFrame(rows)


def random_dir(T, b, iters=500, seed=0):
    """同樣進出場時間，方向隨機；看策略方向勝過幾 % 的隨機。（停損不重算：用原本出場價的 ± 損益近似）"""
    rng = np.random.default_rng(seed)
    g = (T.exit - T.entry).to_numpy() / T.R.to_numpy()
    cost = (T.cost / T.R).to_numpy()
    real = (g * T.side.to_numpy() - cost).mean()
    sims = np.array([(g * rng.choice([-1, 1], len(g)) - cost).mean() for _ in range(iters)])
    longonly = (g - cost).mean()
    return real, sims.mean(), (sims < real).mean(), longonly


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + list(X)); bb, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ bb; s2 = e @ e / (len(y) - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return bb, bb / se, 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
