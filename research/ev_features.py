"""第五十四部分的特徵庫：Entropy Engine 與 Volatility Engine。

所有特徵在 K 棒 t 收盤時只用到 <= t 的資料（滾動窗口、expanding 時段因子都往前錯開）。
回報 r_t = log(C_t / C_{t-1})；跨週末 / 缺資料的報酬照常計入（只在目標變數端排除跨缺口）。
"""
import numpy as np
import pandas as pd

LN3 = np.log(3)
EPS = 1e-12


# ---------------------------------------------------------------- 基本工具
def to_bars(m1, rule="5min"):
    """M1 → 指定週期，並用 M1 報酬算每根 K 棒的 realized variance 與 bipower variation。"""
    r1 = np.log(m1.close).diff()
    rv1 = (r1 ** 2).rename("rv")
    bp1 = (np.abs(r1) * np.abs(r1.shift(1))).rename("bp")
    big = pd.concat([m1[["open", "high", "low", "close", "spread"]], rv1, bp1], axis=1)
    bars = big.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last",
         "spread": "mean", "rv": "sum", "bp": "sum"}).dropna(subset=["close"])
    bars["r"] = np.log(bars.close).diff().fillna(0.0)
    return bars


def roll_counts(codes, k, w):
    """codes：整數 0..k-1（-1 = 無效），回傳 (n, k) 的滾動 w 期計數。"""
    n = len(codes)
    oh = np.zeros((n, k))
    ok = codes >= 0
    oh[np.flatnonzero(ok), codes[ok]] = 1.0
    cs = np.vstack([np.zeros(k), np.cumsum(oh, axis=0)])
    out = cs[w:] - cs[:-w]
    res = np.full((n, k), np.nan)
    res[w - 1:] = out
    return res


def ent(counts, axis=-1):
    """plug-in Shannon entropy（nats），counts 可為多維。"""
    tot = counts.sum(axis=axis, keepdims=True)
    p = counts / np.where(tot > 0, tot, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=axis)
    return h


# ---------------------------------------------------------------- Entropy Engine
def direction_states(r, sig, band=0.3):
    """s_t ∈ {0: 下跌, 1: 中性, 2: 上漲}；中性帶 = band × 事前波動。"""
    th = band * sig
    s = np.ones(len(r), dtype=int)
    s[r > th] = 2
    s[r < -th] = 0
    s[np.isnan(sig)] = -1
    return s


def entropy_engine(r, sig, W=144, W2=288, m=4, close=None):
    """回傳 DataFrame：
    H_D   方向熵（3 狀態，/log3）
    H_T   轉移熵 H(S_{t+1}|S_t)（/log3）
    I1    I(S_{t+1};S_t)，Miller–Madow 偏誤修正後（/log3）
    I2    I(S_{t+1};S_t,S_{t-1})，窗口 W2，修正後（/log3）
    TF    轉移預測：P(上|s_t) − P(下|s_t)（窗口內轉移矩陣，Laplace 平滑）→ 有方向
    PERS  非中性狀態的延續機率 − 0.5
    H_P   permutation entropy（m 階，/log m!）
    """
    s = direction_states(r, sig)
    n = len(s)
    out = {}
    cD = roll_counts(s, 3, W)
    out["H_D"] = ent(cD) / LN3

    prev = np.r_[-1, s[:-1]]
    pair = np.where((s >= 0) & (prev >= 0), prev * 3 + s, -1)
    cP = roll_counts(pair, 9, W).reshape(n, 3, 3)          # [prev, next]
    Nn = cP.sum(axis=(1, 2))
    h_joint = ent(cP.reshape(n, 9))
    h_prev = ent(cP.sum(axis=2))
    h_next = ent(cP.sum(axis=1))
    out["H_T"] = (h_joint - h_prev) / LN3
    mi1 = h_prev + h_next - h_joint - 4.0 / (2 * Nn)       # (3−1)(3−1)/(2N)
    out["I1"] = mi1 / LN3

    pv2 = np.r_[-1, -1, s[:-2]]
    trip = np.where((s >= 0) & (prev >= 0) & (pv2 >= 0), (pv2 * 3 + prev) * 3 + s, -1)
    cT = roll_counts(trip, 27, W2).reshape(n, 9, 3)
    N2 = cT.sum(axis=(1, 2))
    mi2 = ent(cT.sum(axis=2)) + ent(cT.sum(axis=1)) - ent(cT.reshape(n, 27)) - 16.0 / (2 * N2)
    out["I2"] = mi2 / LN3

    # 以「目前狀態」查窗口內轉移矩陣 → 下一步上漲 / 下跌機率
    cur = np.where(s >= 0, s, 1)
    row = cP[np.arange(n), cur, :] + 1.0                    # Laplace
    pr = row / row.sum(axis=1, keepdims=True)
    out["TF"] = pr[:, 2] - pr[:, 0]
    stay = (cP[:, 0, 0] + cP[:, 2, 2] + 1) / (cP[:, 0, :].sum(1) + cP[:, 2, :].sum(1) + 2)
    out["PERS"] = stay - 0.5

    if close is not None:
        out["H_P"] = perm_entropy(close, m, W)
    return pd.DataFrame(out)


def perm_entropy(x, m, W):
    """序數模式（相同值依出現順序排序 = stable argsort）的滾動 permutation entropy。"""
    x = np.asarray(x, float)
    n = len(x)
    import itertools
    perms = {p: i for i, p in enumerate(itertools.permutations(range(m)))}
    lag = np.column_stack([np.r_[np.full(m - 1 - j, np.nan), x[:n - (m - 1 - j)]] for j in range(m)])
    codes = np.full(n, -1)
    ok = ~np.isnan(lag).any(axis=1)
    order = np.argsort(lag[ok], axis=1, kind="stable")
    keys = [perms[tuple(o)] for o in order]
    codes[ok] = keys
    c = roll_counts(codes, len(perms), W)
    return ent(c) / np.log(len(perms))


# ---------------------------------------------------------------- Volatility Engine
def tod_factor(bars):
    """時段（一天中的 5 分鐘位置）波動因子：只用前幾天的同時段 rv（expanding mean，錯開一天）。"""
    key = bars.index.hour * 60 + bars.index.minute
    day = bars.index.normalize()
    df = pd.DataFrame({"rv": bars.rv.to_numpy(), "key": key, "day": day})
    df["f"] = df.groupby("key").rv.transform(lambda v: v.shift(1).expanding().mean())
    glob = df.rv.shift(1).expanding().mean()
    f = (df.f / glob).to_numpy().copy()
    f[~np.isfinite(f)] = 1.0
    return np.clip(f, 0.05, 20), key.to_numpy()


def volatility_engine(bars, S=12, M=48, L=288, J=48):
    rv = bars.rv.to_numpy()
    bp = bars.bp.to_numpy()
    r = bars.r.to_numpy()
    s = pd.Series(rv)
    out = {}
    out["sigS"] = np.sqrt(s.rolling(S).mean().to_numpy())
    out["sigM"] = np.sqrt(s.rolling(M).mean().to_numpy())
    out["sigL"] = np.sqrt(s.rolling(L).mean().to_numpy())
    eps = np.nanmedian(out["sigL"]) * 0.01
    out["V_SL"] = np.log((out["sigS"] + eps) / (out["sigL"] + eps))
    out["V_ML"] = np.log((out["sigM"] + eps) / (out["sigL"] + eps))
    f, key = tod_factor(bars)
    out["tod"] = f
    # 去季節化的短期波動（相對同時段常態）
    sdes = pd.Series(rv / f).rolling(S).mean()
    ldes = pd.Series(rv / f).rolling(L).mean()
    out["V_des"] = np.log((np.sqrt(sdes) + eps) / (np.sqrt(ldes) + eps)).to_numpy()
    # 波動聚集：滾動 corr(|r_t|, |r_{t-1}|)
    a = pd.Series(np.abs(r))
    out["rho1"] = a.rolling(L).corr(a.shift(1)).to_numpy()
    # 跳躍：1 − BV/RV（M1 bipower，J 根窗口）
    RVw = s.rolling(J).sum()
    BVw = (np.pi / 2) * pd.Series(bp).rolling(J).sum()
    out["JUMP"] = np.clip(1 - BVw / RVw, 0, 1).to_numpy()
    # 尾部：滾動峰態（M5 報酬 / 時段因子）
    out["KURT"] = pd.Series(r / np.sqrt(f)).rolling(L).kurt().to_numpy()
    return pd.DataFrame(out, index=bars.index), key


def ewma_sigma(r, lam=0.94):
    v = pd.Series(r ** 2).ewm(alpha=1 - lam, adjust=False).mean().to_numpy()
    return np.sqrt(v)


def future_tod_sum(f, key, h):
    """未來 h 根的時段因子總和（課表固定，事前可知）；用目前 expanding 估計值。"""
    # 每個時段 key 的最新因子值（只用到 t 為止的估計）
    df = pd.DataFrame({"key": key, "f": f})
    nkeys = 288
    latest = np.ones(nkeys)
    out = np.empty(len(f))
    step = int(np.median(np.diff(np.unique(key)))) if len(np.unique(key)) > 1 else 5
    for i, (k, v) in enumerate(zip(key, f)):
        latest[(k // step) % nkeys] = v
        ks = ((k // step + np.arange(1, h + 1)) % nkeys)
        out[i] = latest[ks].sum()
    return out


# ---------------------------------------------------------------- 方向（非熵）基礎特徵
def base_directional(bars, sig_bar, W=144):
    c = np.log(bars.close.to_numpy())
    n = len(c)
    out = {}
    for k in (12, 48):
        mom = c - np.r_[np.full(k, np.nan), c[:-k]]
        out[f"MOM{k}"] = mom / (sig_bar * np.sqrt(k))
    ema = pd.Series(c).ewm(span=48, adjust=False).mean().to_numpy()
    out["Z48"] = (c - ema) / (sig_bar * np.sqrt(48))
    return pd.DataFrame(out)
