"""第二十二部分：均值定義 × 回復力（restoring force）研究的共用函式。

時間：broker 時間，交易日 = 01:00 ~ 23:59。資料：data/XAUUSD_M1_2026.csv（1/2 ~ 10/8，4/3~4/14 缺）。
樣本內 IS = 1/2 ~ 5/31；樣本外 OOS = 6/1 ~ 10/8。所有選擇（均值、量測方向、權重、門檻）只用 IS。

所有特徵都只用「該根 K 棒收盤（含）之前」的資料；進場一律在下一根開盤。
"""
import numpy as np
import pandas as pd
from load import load

CUT = pd.Timestamp("2026-06-01")
TF_MIN = {"M1": 1, "M5": 5, "M15": 15}


def load_tf(tf="M5", path="data/XAUUSD_M1_2026.csv"):
    d = load(path)
    d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
    if tf != "M1":
        m = TF_MIN[tf]
        g = d.groupby([d.day, d.index.floor(f"{m}min")])
        d = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last(),
                          "tickvol": g.tickvol.sum(), "spread": g.spread.first()}).reset_index(level=0)
        d.index.name = "time"
    # 連續區段：超過 3 小時沒有 K 棒 = 週末或缺資料，前瞻報酬不能跨過去
    gap = d.index.to_series().diff() > pd.Timedelta(hours=3)
    d["seg"] = gap.cumsum().to_numpy()
    return d


# ---------------------------------------------------------------- 均值定義
def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def kama(c, n=10, fast=2, slow=30):
    er = (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
    sc = (er.fillna(0) * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.empty(len(c)); v = c.to_numpy(); s = sc.to_numpy()
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = out[i - 1] + s[i] * (v[i] - out[i - 1])
    return pd.Series(out, c.index)


def linreg_end(c, n=20):
    """最近 n 根收盤的線性回歸，在最後一根的配適值（趨勢修正後的中心）。"""
    x = np.arange(n) - (n - 1) / 2
    w_slope = x / (x ** 2).sum()
    mean = c.rolling(n).mean()
    slope = c.rolling(n).apply(lambda a: (a * w_slope).sum(), raw=True)
    return mean + slope * (n - 1) / 2


def means(d):
    c, tp, v = d.close, (d.high + d.low + d.close) / 3, d.tickvol.clip(lower=1)
    M = pd.DataFrame(index=d.index)
    M["ema20"], M["ema50"], M["ema100"] = ema(c, 20), ema(c, 50), ema(c, 100)
    M["sma20"], M["sma50"] = c.rolling(20).mean(), c.rolling(50).mean()
    M["kama"] = kama(c)
    M["linreg20"] = linreg_end(c, 20)
    pv = tp * v
    M["vwap_sess"] = pv.groupby(d.day).cumsum() / v.groupby(d.day).cumsum()
    M["rvwap20"] = pv.rolling(20).sum() / v.rolling(20).sum()
    M["rvwap50"] = pv.rolling(50).sum() / v.rolling(50).sum()
    M["mix_ema50_vwap"] = (M.ema50 + M.vwap_sess) / 2
    return M


def atr(d, n):
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def fwd_returns(d, A, H):
    """(C[t+H] − C[t]) / ATR[t]；跨區段 → NaN。"""
    c = d.close
    f = (c.shift(-H) - c) / A
    same = d.seg.shift(-H) == d.seg
    return f.where(same)


# ---------------------------------------------------------------- 回復力量測
def rolling_slope(y, x, n):
    mx, my = x.rolling(n).mean(), y.rolling(n).mean()
    cov = (x * y).rolling(n).mean() - mx * my
    var = (x * x).rolling(n).mean() - mx * mx
    return cov / var


def rolling_ols2(y, x1, x2, n):
    """y = a + b1 x1 + b2 x2 的滾動係數 (b1, b2)。"""
    m = lambda s: s.rolling(n).mean()
    c11 = m(x1 * x1) - m(x1) ** 2
    c22 = m(x2 * x2) - m(x2) ** 2
    c12 = m(x1 * x2) - m(x1) * m(x2)
    c1y = m(x1 * y) - m(x1) * m(y)
    c2y = m(x2 * y) - m(x2) * m(y)
    det = c11 * c22 - c12 ** 2
    return (c22 * c1y - c12 * c2y) / det, (c11 * c2y - c12 * c1y) / det


def er(c, n):
    return (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()


def restoring_features(d, mean, A):
    """回傳 DataFrame，每一欄是一種「回復力」或背景量測。

    慣例：s = sign(D)（價格在均值哪一側）；『順離開方向』的量都乘上 s，
    所以 vel > 0 = 還在往外跑，acc > 0 = 往外加速。
    """
    c, h, l, o, v = d.close, d.high, d.low, d.open, d.tickvol.astype(float)
    D = (c - mean) / A
    s = np.sign(D)
    r = np.log(c).diff()
    F = pd.DataFrame(index=d.index)
    F["D"] = D
    F["dist"] = D.abs()                                         # 1. 距離（ATR 標準化）
    # 2. sigma 擴張 / 收縮
    F["sig_ratio"] = r.rolling(10).std() / r.rolling(60).std()   # 短 σ / 長 σ
    sd20 = c.rolling(20).std()
    F["bbw_chg"] = sd20 / sd20.shift(5)                         # 布林帶寬 5 根變化
    F["bbw_pct"] = (sd20 / A).rolling(300).rank(pct=True)       # σ/ATR 的歷史分位（帶寬水位）
    # 3. ER 乾淨程度
    F["er10"], F["er20"] = er(c, 10), er(c, 20)
    F["er_ratio"] = er(c, 5) / er(c, 20)
    # 4. ATR 膨脹
    F["atr_ratio"] = atr(d, 5) / atr(d, 60)
    F["atr_slope"] = A / A.shift(10)
    # 5. 速度 / 加速度（距離變化，順離開方向）
    vel = (D - D.shift(3)) / 3
    F["vel"] = s * vel
    F["acc"] = s * (vel - vel.shift(3))
    # 6. 動態回復係數（只用過去 120 根）
    dD = D.diff()
    F["ou_theta"] = -rolling_slope(dD, D.shift(), 120)               # OU：ΔD = −θ·D + ε
    k, cdamp = rolling_ols2(dD.diff(), D.shift(), dD.shift(), 120)    # 阻尼振盪：Δ²D = −k·D − c·ΔD
    F["osc_k"], F["osc_c"] = -k, -cdamp
    # 7. 報酬自相關 / 變異數比
    F["ac1"] = r.rolling(60).corr(r.shift())
    r6 = np.log(c).diff(6)
    F["vr6"] = r6.rolling(120).var() / (6 * r.rolling(120).var())
    # 8. K 棒微觀：外側影線（被拒絕）、量能
    rng = (h - l).replace(0, np.nan)
    up_w, dn_w = h - np.maximum(o, c), np.minimum(o, c) - l
    F["wick_out"] = np.where(s > 0, up_w, dn_w) / rng
    F["vol_rel"] = v / v.rolling(60).median()
    # 9. 離開均值的這一段走了多久（連續同側根數）
    side_change = (s != s.shift()).cumsum()
    F["leg_bars"] = s.groupby(side_change).cumcount() + 1
    # 10. 這一段（從穿越均值到現在）的 ER：離開均值的那段走得乾不乾淨
    lb = F.leg_bars.clip(upper=60).to_numpy().astype(int)
    cv, ad = c.to_numpy(), np.concatenate([[0], np.cumsum(np.abs(np.diff(c.to_numpy())))])
    i = np.arange(len(cv)); j = np.maximum(i - np.maximum(lb, 3), 0)
    F["er_leg"] = np.abs(cv - cv[j]) / np.where(ad[i] - ad[j] > 0, ad[i] - ad[j], np.nan)
    # 11. 從這段極值已經拉回多少（ATR）：回復力是否已經啟動
    ext = np.where(s > 0, h.rolling(5).max(), l.rolling(5).min())
    F["pullback"] = s * (ext - c) / A
    # 12. 距離用 σ 標準化（z）相對於用 ATR 標準化：σ/ATR 高 = 偏離是由「單向位移」構成
    F["z_over_d"] = (D / ((c - mean).rolling(50).std() / A)).abs() / F.dist
    return F


def context_features(d, A, mean_main):
    """背景因子（非回復力本身，但可能和它配合）。"""
    c = d.close
    D = (c - mean_main) / A
    s = np.sign(D)
    F = pd.DataFrame(index=d.index)
    hr = d.index.hour
    F["sess"] = np.select([hr < 10, hr < 15, hr < 20], ["ASIA", "LDN", "NY"], "LATE")
    # 高週期趨勢：EMA(12×20) 斜率，順離開方向為正
    e = ema(c, 240)
    F["htf"] = s * (e - e.shift(12)) / A
    # 今天已用掉的波幅 / 前 10 日平均日波幅
    dh = d.high.groupby(d.day).cummax(); dl = d.low.groupby(d.day).cummin()
    dr = (d.high.groupby(d.day).max() - d.low.groupby(d.day).min())
    adr = dr.shift(1).rolling(10).mean()
    F["day_used"] = (dh - dl) / d.day.map(adr).to_numpy()
    # 今天到目前 ER（第二十一部分），順離開方向
    o0 = d.open.groupby(d.day).transform("first")
    path = c.diff().abs().where(d.day == d.day.shift(), 0).groupby(d.day).cumsum()
    F["day_der"] = s * (c - o0) / path.replace(0, np.nan)
    # 和 session VWAP 是否同側、距離
    tp, v = (d.high + d.low + c) / 3, d.tickvol.clip(lower=1)
    vw = (tp * v).groupby(d.day).cumsum() / v.groupby(d.day).cumsum()
    F["vwap_side"] = s * (c - vw) / A
    # 是否超過前一日高/低（往離開方向）
    ph = d.day.map(d.high.groupby(d.day).max().shift(1)).to_numpy()
    pl = d.day.map(d.low.groupby(d.day).min().shift(1)).to_numpy()
    F["beyond_pd"] = np.where(s > 0, (c - ph), (pl - c)) / A
    return F


# ---------------------------------------------------------------- 事件
def events(D, seg, d0=1.5, rearm=0.5):
    """|D| 第一次 ≥ d0（上次事件後 |D| 必須先回到 < rearm 才能再觸發）。"""
    a = D.abs().to_numpy(); sg = seg.to_numpy()
    out, armed = [], True
    for i in range(len(a)):
        if np.isnan(a[i]):
            continue
        if not armed and a[i] < rearm:
            armed = True
        if armed and a[i] >= d0:
            out.append(i); armed = False
    return np.array(out, int)


def race(d, idx, mean_at, A, sgn, ext=1.0, horizon=24):
    """以下一根開盤為起點：先碰到事件當下的均值（回歸，+1），還是先再往外走 ext×ATR（延續，−1）；都沒碰 0。
    同一根同時碰到算延續（保守）。"""
    h, l, o = d.high.to_numpy(), d.low.to_numpy(), d.open.to_numpy()
    sg = d.seg.to_numpy()
    res = np.zeros(len(idx))
    for k, i in enumerate(idx):
        if i + 1 >= len(o):
            continue
        s = sgn[k]; m = mean_at[k]; a = A[k]
        start = o[i + 1]
        far = start + s * ext * a
        for j in range(i + 1, min(i + 1 + horizon, len(o))):
            if sg[j] != sg[i]:
                break
            hit_far = h[j] >= far if s > 0 else l[j] <= far
            hit_m = l[j] <= m if s > 0 else h[j] >= m
            if hit_far:
                res[k] = -1; break
            if hit_m:
                res[k] = 1; break
    return res


def spearman(x, y):
    x, y = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    m = x.notna() & y.notna()
    if m.sum() < 20:
        return np.nan
    return x[m].rank().corr(y[m].rank())


def tstat_ic(ic, n):
    return ic * np.sqrt(max(n - 2, 1)) / np.sqrt(max(1 - ic ** 2, 1e-9))
