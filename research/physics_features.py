"""第二十二部分：幾何光學 / 電磁學類比指標（只用當下以前的資料計算）。
資料只有 XAUUSD M1 的 OHLC + tick volume + 點差（沒有訂單簿、只有單一商品），
所以每個物理量都要先換成「可以從 K 棒算出來」的代理變數，定義寫在各函式的 docstring。
"""
import numpy as np
import pandas as pd
from numba import njit

EPS = 1e-12


def base(df):
    """共用欄位：ATR60、相對量（tick volume / 過去 1440 根中位數，消除 2023→2026 量能 5 倍成長）、連續段落 id。"""
    d = df.copy()
    pc = d.close.shift()
    tr = np.maximum(d.high, pc) - np.minimum(d.low, pc)
    d["atr"] = tr.ewm(span=60, adjust=False).mean()
    d["rvol"] = d.tickvol / d.tickvol.rolling(1440, min_periods=200).median()
    gap = d.index.to_series().diff() > pd.Timedelta("10min")
    d["seg"] = gap.cumsum().values
    d["ret"] = d.close.diff()
    d.loc[gap.values, "ret"] = np.nan
    return d


def forward(d, hs=(5, 15, 60)):
    """前瞻報酬（ATR 單位）；跨越休市的視為缺值。"""
    out = {}
    seg = d.seg.values
    for h in hs:
        f = (d.close.shift(-h) - d.close) / d.atr
        same = pd.Series(seg).shift(-h).values == seg
        f[~same] = np.nan
        out[h] = f
        r = d.ret.abs().rolling(h).sum().shift(-h) / d.atr  # 未來 h 根路徑長（波動）
        r[~same] = np.nan
        out[f"path{h}"] = r
    return pd.DataFrame(out, index=d.index)


# ---------------------------------------------------------------- 1. 折射率 n
def refractive(d, w=15):
    """n = σ / √V：每單位（相對）成交量的價格波動。
    σ = w 根 1 分鐘報酬標準差 / ATR 的年度尺度（用 close 計價會被金價水準影響，所以除以 ATR1440 中位數）；
    V = w 根平均相對量。平方根衝擊模型下 成本 ∝ σ·√(q/V)，故 n 就是「單位下單量的光程」。"""
    sig = d.ret.rolling(w).std()
    scale = d.atr.rolling(1440, min_periods=200).median()
    V = d.rvol.rolling(w).mean()
    return (sig / scale) / np.sqrt(V + EPS)


# ---------------------------------------------------------------- 2. 司乃耳定律
@njit(cache=True)
def _profile_density(hi, lo, vol, atr, level, side, look, band):
    """過去 look 根的成交量剖面：在 [level, level+band·ATR]（side=+1）或 [level-band·ATR, level] 的
    每單位價格成交量密度。每根 K 棒的量平均攤在它的高低區間。"""
    n = len(level)
    out = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(level[t]) or t < look:
            continue
        b = band * atr[t]
        if side[t] > 0:
            a0, a1 = level[t], level[t] + b
        else:
            a0, a1 = level[t] - b, level[t]
        s = 0.0
        for k in range(t - look, t):
            r = hi[k] - lo[k]
            ov = min(hi[k], a1) - max(lo[k], a0)
            if ov > 0:
                s += vol[k] * ov / max(r, 1e-9)
            elif r <= 0 and a0 <= hi[k] <= a1:
                s += vol[k]
        out[t] = s / b
    return out


def snell_events(d, lev_look=120, fresh=10, prof_look=1440, band=1.0, horizon=30, k_out=1.5):
    """事件：價格第一次碰到「已存在至少 fresh 根」的 lev_look 根高點（壓力）或低點（支撐）。
    入射角：相空間 (時間, 價格/ATR) 中路徑與介面法線的夾角，v = 碰觸前 5 根朝介面的速度（ATR/根），
            θ1 = atan(1/v)；直衝 v 大 → θ1 小（正入射），緩慢磨上去 → θ1 → 90°（掠射）。
    折射率：n ∝ 1/成交量密度，n1 = 介面內側 1 ATR 帶，n2 = 介面外側 1 ATR 帶（過去 1 天剖面）。
    司乃耳比 R = n1·sinθ1 / n2；R > 1 → 光學上全反射（預測反彈），R < 1 → 折射穿透（預測突破）。
    結果：未來 horizon 根內先到介面外 k_out ATR（突破）或內側 k_out ATR（反彈）。"""
    hi, lo, cl = d.high.values, d.low.values, d.close.values
    atr = d.atr.values
    H = d.high.shift(fresh).rolling(lev_look - fresh).max().values
    L = d.low.shift(fresh).rolling(lev_look - fresh).min().values
    recent_hi = d.high.shift(1).rolling(fresh).max().values
    recent_lo = d.low.shift(1).rolling(fresh).min().values
    up = (hi >= H) & (recent_hi < H)
    dn = (lo <= L) & (recent_lo > L)
    idx = np.where(up | dn)[0]
    idx = idx[(idx > prof_look) & (idx < len(d) - horizon - 1)]
    side = np.where(up[idx], 1, -1)
    level = np.where(side > 0, H[idx], L[idx])
    lvl_full = np.full(len(d), np.nan)
    side_full = np.zeros(len(d))
    lvl_full[idx], side_full[idx] = level, side
    vol = d.rvol.fillna(1).values
    outside = _profile_density(hi, lo, vol, atr, lvl_full, side_full, prof_look, band)
    inside = _profile_density(hi, lo, vol, atr, lvl_full, -side_full, prof_look, band)
    v = side * (cl[idx - 1] - cl[idx - 6]) / 5 / atr[idx]
    vpos = np.maximum(v, 1e-3)
    sin1 = 1 / np.sqrt(1 + vpos ** 2)  # 與法線夾角 θ1 = atan(1/v)，sinθ1 = 1/√(1+v²)
    n1 = 1 / (inside[idx] + 1e-6)
    n2 = 1 / (outside[idx] + 1e-6)
    R = n1 * sin1 / n2
    # 結果
    seg = d.seg.values
    res, f30 = np.zeros(len(idx)), np.full(len(idx), np.nan)
    for j, (t, s, lv) in enumerate(zip(idx, side, level)):
        if seg[t + horizon] != seg[t]:
            res[j] = np.nan
            continue
        a = atr[t] * k_out
        out = 0
        for k in range(t + 1, t + horizon + 1):
            brk = hi[k] >= lv + a if s > 0 else lo[k] <= lv - a
            rev = lo[k] <= lv - a if s > 0 else hi[k] >= lv + a
            if rev:
                out = -1
                break
            if brk:
                out = 1
                break
        res[j] = out
        f30[j] = s * (cl[t + horizon] - lv) / atr[t]
    return pd.DataFrame({"time": d.index[idx], "side": side, "v": v, "sin1": sin1, "n1": n1, "n2": n2,
                         "dens_ratio": n2 / n1, "R": R, "outcome": res, "fwd30": f30})


# ---------------------------------------------------------------- 3. 焦散（停損光束匯聚）
@njit(cache=True)
def _pivots(hi, lo, k):
    n = len(hi)
    ph = np.zeros(n, dtype=np.bool_)
    pl = np.zeros(n, dtype=np.bool_)
    for i in range(k, n - k):
        h, l = hi[i], lo[i]
        okh, okl = True, True
        for j in range(i - k, i + k + 1):
            if j == i:
                continue
            if hi[j] >= h:
                okh = False
            if lo[j] <= l:
                okl = False
            if not okh and not okl:
                break
        ph[i], pl[i] = okh, okl
    return ph, pl


@njit(cache=True)
def _caustic(hi, lo, atr, piv_t, piv_c, piv_p, piv_s, life, delta):
    """piv_*: 依確認時間排序的轉折點（時間、確認時間、價格、+1 高點 / −1 低點）。
    每根 K 棒：找出本根第一次被突破的轉折點；F = 突破前一刻，同側尚未被破、且價格在
    [最近被破價位, +delta·ATR] 內的轉折點數（= 會同時被觸發的停損光束數）。"""
    n = len(hi)
    m = len(piv_p)
    act = np.empty(m, dtype=np.int64)  # 活著的轉折點索引
    na = 0
    ptr = 0
    ev_t = np.empty(2 * n, dtype=np.int64)
    ev_s = np.empty(2 * n, dtype=np.int64)
    ev_p = np.empty(2 * n)
    ev_f = np.empty(2 * n, dtype=np.int64)
    ne = 0
    for t in range(1, n):
        while ptr < m and piv_c[ptr] <= t - 1:
            act[na] = ptr
            na += 1
            ptr += 1
        # 移除過期
        k = 0
        for a in range(na):
            q = act[a]
            if t - piv_t[q] <= life:
                act[k] = q
                k += 1
        na = k
        for s in (1, -1):
            best = np.nan
            for a in range(na):
                q = act[a]
                if piv_s[q] != s:
                    continue
                p = piv_p[q]
                if (s > 0 and hi[t] > p) or (s < 0 and lo[t] < p):
                    if np.isnan(best) or (s > 0 and p < best) or (s < 0 and p > best):
                        best = p
            if np.isnan(best):
                continue
            dd = delta * atr[t - 1]
            f = 0
            for a in range(na):
                q = act[a]
                if piv_s[q] == s:
                    p = piv_p[q]
                    if (s > 0 and best <= p <= best + dd) or (s < 0 and best - dd <= p <= best):
                        f += 1
            ev_t[ne], ev_s[ne], ev_p[ne], ev_f[ne] = t, s, best, f
            ne += 1
            k = 0  # 本根被破的全部移除
            for a in range(na):
                q = act[a]
                if piv_s[q] == s and ((s > 0 and hi[t] > piv_p[q]) or (s < 0 and lo[t] < piv_p[q])):
                    continue
                act[k] = q
                k += 1
            na = k
    return ev_t[:ne], ev_s[:ne], ev_p[:ne], ev_f[:ne]


def caustic_events(d, scales=(5, 15, 60, 240), life=1440, delta=0.5, horizon=15):
    hi, lo, atr = d.high.values, d.low.values, d.atr.values
    rows = []
    for k in scales:
        ph, pl = _pivots(hi, lo, k)
        for arr, s in ((ph, 1), (pl, -1)):
            t = np.where(arr)[0]
            rows.append(pd.DataFrame({"t": t, "c": t + k, "p": np.where(s > 0, hi[t], lo[t]), "s": s}))
    P = pd.concat(rows).drop_duplicates(["t", "s"]).sort_values("c").reset_index(drop=True)
    et, es, ep, ef = _caustic(hi, lo, atr, P.t.values.astype(np.int64), P.c.values.astype(np.int64),
                              P.p.values.astype(np.float64), P.s.values.astype(np.int64), life, delta)
    E = pd.DataFrame({"i": et, "side": es, "level": ep, "F": ef})
    E = E[(E.i > 1440) & (E.i < len(d) - horizon - 1)]
    seg = d.seg.values
    ok = seg[E.i + horizon] == seg[E.i]
    E = E[ok].copy()
    i, s, a = E.i.values, E.side.values, atr[E.i.values - 1]
    hmax = pd.Series(hi).rolling(horizon).max().shift(-horizon).values  # t+1..t+h
    lmin = pd.Series(lo).rolling(horizon).min().shift(-horizon).values
    E["mfe"] = np.where(s > 0, hmax[i] - E.level, E.level - lmin[i]) / a  # 突破方向最大延伸
    E["mae"] = np.where(s > 0, E.level - lmin[i], hmax[i] - E.level) / a
    E["fwd"] = s * (d.close.values[i + horizon] - E.level.values) / a
    E["rng"] = (hmax[i] - lmin[i]) / a
    E["time"] = d.index[i]
    return E


# ---------------------------------------------------------------- 4. 阻抗與反射係數
def impedance(d, w1=30, w2=5):
    """Z = Σ|Δp| / Σ相對量（單位成交量推動的價格路徑，ATR 單位）；
    Z1 = 前 w1 根（t−w2 以前），Z2 = 最近 w2 根；Γ = (Z2−Z1)/(Z2+Z1)，T = 1+Γ。"""
    a = d.ret.abs() / d.atr
    v = d.rvol
    Z2 = a.rolling(w2).sum() / v.rolling(w2).sum()
    Z1 = (a.rolling(w1).sum() / v.rolling(w1).sum()).shift(w2)
    G = (Z2 - Z1) / (Z2 + Z1)
    # 對照組：只用價格的波動比（不含量），看 Γ 是否只是波動比換個名字
    vr = a.rolling(w2).mean() / a.rolling(w1).mean().shift(w2)
    Gp = (vr - 1) / (vr + 1)
    mv = (d.close - d.close.shift(w2)) / d.atr  # 最近 w2 根的波（方向）
    return pd.DataFrame({"Z1": Z1, "Z2": Z2, "Gamma": G, "Gamma_price": Gp, "move": mv})


# ---------------------------------------------------------------- 5. 坡印廷向量
def poynting(d, ema=60, w=30):
    """E = (價格 − EMA60)/ATR：價格勢能梯度（偏離）；
    H = Σ 量·(C−O)/(H−L) / Σ 量：K 棒內推估的淨主動委託流（−1~+1）；
    S = E·H：兩者同號 = 能量往偏離方向持續注入；異號 = 價格偏離但委託流反向（能流轉向）。"""
    E = (d.close - d.close.ewm(span=ema, adjust=False).mean()) / d.atr
    rng = (d.high - d.low).replace(0, np.nan)
    sv = (d.rvol * (d.close - d.open) / rng).fillna(0)
    H = sv.rolling(w).sum() / d.rvol.rolling(w).sum()
    return pd.DataFrame({"E": E, "H": H, "S": E * H})
