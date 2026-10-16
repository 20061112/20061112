"""第二十二部分的共用工具：K 棒重取樣、波動標準化、BOCPD（Adams & MacKay 2007）、Page CUSUM、回測。

所有訊號都只用到「訊號 K 棒收盤時已知」的資料；進場 = 下一根開盤。
"""
import math
import numpy as np
import pandas as pd
from load import load

M15_PATH = "data/XAUUSD_M15_2023_2026.csv"
_erf = np.vectorize(math.erf)


def Phi(x):
    return 0.5 * (1 + _erf(np.asarray(x, float) / math.sqrt(2)))


# ---------------------------------------------------------------- 資料
def bars(tf):
    """由 M15 重取樣：'D1'（broker 日）、'H4'、'H1'、'M15'。"""
    m = load(M15_PATH)
    m["sp"] = m.spread * 0.01
    if tf == "M15":
        b = m
    else:
        rule = {"D1": "1D", "H4": "4h", "H1": "1h"}[tf]
        b = m.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "sp": "median"}).dropna()
    b = b[["open", "high", "low", "close", "sp"]].copy()
    pc = b.close.shift()
    tr = np.maximum(b.high - b.low, np.maximum((b.high - pc).abs(), (b.low - pc).abs()))
    b["atr"] = tr.rolling(14).mean()
    b["r"] = np.log(b.close).diff()
    b["day"] = b.index.normalize()
    return b


def zscore(b, tf, span=20, clip=4.0):
    """報酬 / 事前波動。日線：EWMA 標準差（span 20 日）；日內：同一時段過去 20 日 |r| 的 EWMA（去掉日內季節性）。
    一律用到 t-1 為止的資訊，z 截在 ±clip。"""
    r = b.r
    if tf == "D1":
        sig = np.sqrt((r ** 2).ewm(span=span, min_periods=span).mean().shift())
    else:
        slot = b.index.hour * 60 + b.index.minute
        a = r.abs().groupby(slot).transform(lambda x: x.ewm(span=span, min_periods=10).mean().shift())
        sig = a * math.sqrt(math.pi / 2)
    z = (r / sig).clip(-clip, clip)
    return z.fillna(0.0)


# ---------------------------------------------------------------- BOCPD
def bocpd(z, lam, tau, w=5, rmax=None):
    """段內模型：z ~ N(mu, 1)，mu ~ N(0, tau²)；固定 hazard H = 1/lam。
    回傳每根 K 棒收盤後：
      cp   = P(目前這段的長度 ≤ w)            （「剛開始」的機率）
      conf = P(mu > 0 | 段長 1..w)            （新段漂移為正的信心）
      mapr = 最可能的段長
    """
    z = np.asarray(z, float)
    n = len(z)
    rmax = rmax or int(lam * 6)
    H = 1.0 / lam
    p = np.zeros(rmax); p[0] = 1.0
    S = np.zeros(rmax)                     # 各段長的累計和
    cnt = np.arange(rmax, dtype=float)     # 段長 r 含 r 個觀測
    cp = np.zeros(n); conf = np.full(n, 0.5); mapr = np.zeros(n, int)
    inv_t2 = 1.0 / tau ** 2
    for t in range(n):
        x = z[t]
        prec = inv_t2 + cnt
        m = S / prec
        v = 1.0 + 1.0 / prec
        pred = np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(v)
        g = p * pred
        new = np.empty(rmax)
        new[0] = g.sum() * H
        new[1:] = g[:-1] * (1 - H)
        new[-1] += g[-1] * (1 - H)         # 截斷：尾端合併
        new /= new.sum()
        S2 = np.empty(rmax); S2[0] = 0.0; S2[1:] = S[:-1] + x
        p, S = new, S2
        cp[t] = p[: w + 1].sum()
        mapr[t] = p.argmax()
        k = slice(1, w + 1)
        pr = inv_t2 + cnt[k]
        pw = p[k].sum()
        if pw > 1e-12:
            conf[t] = float((p[k] * Phi((S[k] / pr) * np.sqrt(pr))).sum() / pw)
    return cp, conf, mapr


def bocpd_ng(z, lam, kappa0=1.0, alpha0=2.0, beta0=2.0, w=5, rmax=None):
    """Normal-Gamma 版（未知平均與變異數，Adams & MacKay 原文設定）：預測分布為 Student-t，對單根大 K 棒較不敏感。
    先驗 mu|s² ~ N(0, s²/kappa0)，1/s² ~ Gamma(alpha0, beta0)（alpha0=beta0=2 → 先驗 E[s²]=2、厚尾）。
    回傳同 bocpd()：cp、conf（新段 1..w 根 P(mu>0)，以 t 分布近似）、mapr。"""
    z = np.asarray(z, float)
    n = len(z)
    rmax = rmax or int(lam * 6)
    H = 1.0 / lam
    p = np.zeros(rmax); p[0] = 1.0
    cnt = np.arange(rmax, dtype=float)
    S = np.zeros(rmax); Q = np.zeros(rmax)          # 和、平方和
    cp = np.zeros(n); conf = np.full(n, 0.5); mapr = np.zeros(n, int)
    def post(S, Q, cnt):
        k = kappa0 + cnt
        m = S / k
        a = alpha0 + cnt / 2
        xbar = np.where(cnt > 0, S / np.maximum(cnt, 1), 0.0)
        ss = Q - cnt * xbar ** 2
        bb = beta0 + 0.5 * ss + kappa0 * cnt * xbar ** 2 / (2 * k)
        return k, m, a, bb
    LH = lg_half(2 * alpha0 + cnt)                  # 自由度只跟段長有關 → 先算好
    for t in range(n):
        x = z[t]
        k, m, a, bb = post(S, Q, cnt)
        nu = 2 * a
        sc2 = bb * (k + 1) / (a * k)
        lp = (LH - 0.5 * np.log(nu * math.pi * sc2) - (nu + 1) / 2 * np.log1p((x - m) ** 2 / (nu * sc2)))
        pred = np.exp(lp - lp.max())
        g = p * pred
        new = np.empty(rmax)
        new[0] = g.sum() * H
        new[1:] = g[:-1] * (1 - H)
        new[-1] += g[-1] * (1 - H)
        new /= new.sum()
        S2 = np.empty(rmax); S2[0] = 0.0; S2[1:] = S[:-1] + x
        Q2 = np.empty(rmax); Q2[0] = 0.0; Q2[1:] = Q[:-1] + x * x
        p, S, Q = new, S2, Q2
        cp[t] = p[: w + 1].sum()
        mapr[t] = p.argmax()
        sl = slice(1, w + 1)
        pw = p[sl].sum()
        if pw > 1e-12:
            k, m, a, bb = post(S[sl], Q[sl], cnt[sl])
            tstat = m / np.sqrt(bb / (a * k))
            nu = 2 * a
            zz = tstat * (1 - 1 / (4 * nu)) / np.sqrt(1 + tstat ** 2 / (2 * nu))   # t → 常態近似
            conf[t] = float((p[sl] * Phi(zz)).sum() / pw)
    return cp, conf, mapr


def lg_half(nu):
    """log Γ((nu+1)/2) − log Γ(nu/2)（nu 隨段長只取有限個值，向量化）。"""
    return _lgv((nu + 1) / 2) - _lgv(nu / 2)


_lgv = np.vectorize(math.lgamma)


def bocpd_signals(cp, conf, theta, c):
    """事件：cp ≥ theta 的一段期間（episode）裡，第一根 conf ≥ c（做多）或 ≤ 1-c（做空）的 K 棒。
    回傳 sig（+1/-1/0，於該 K 棒收盤）與 ep_start（episode 開始的 K 棒 = 「下一個變點」）。"""
    n = len(cp)
    sig = np.zeros(n, int); ep_start = np.zeros(n, bool)
    active = False; fired = False
    for t in range(n):
        if cp[t] >= theta:
            if not active:
                active, fired = True, False
                ep_start[t] = True
            if not fired:
                if conf[t] >= c:
                    sig[t], fired = 1, True
                elif conf[t] <= 1 - c:
                    sig[t], fired = -1, True
        else:
            active = False
    return sig, ep_start


# ---------------------------------------------------------------- CUSUM
def cusum_signals(z, k, h):
    """Page 雙邊 CUSUM（標準化報酬）：S+ = max(0, S+ + z − k)，S− = max(0, S− − z − k)；超過 h 發出事件後兩邊歸零。"""
    z = np.asarray(z, float)
    sig = np.zeros(len(z), int)
    sp = sm = 0.0
    for t, x in enumerate(z):
        sp = max(0.0, sp + x - k); sm = max(0.0, sm - x - k)
        if sp > h:
            sig[t] = 1; sp = sm = 0.0
        elif sm > h:
            sig[t] = -1; sp = sm = 0.0
    return sig


# ---------------------------------------------------------------- 回測
def backtest(b, sig, stop_atr, exit_mode="flip", ep_start=None, fin=0.0002):
    """訊號於 K 棒收盤 → 下一根開盤進場。停損 = stop_atr × ATR14（進場後固定）。
    exit_mode:
      'flip' — 持有直到反向訊號（直接反手）或停損；同向訊號不動作
      'cp'   — 持有直到下一個變點 episode 開始（下一根開盤平倉），或停損
    成本：點差（進出各半，用進場 K 棒點差）+ 融資 fin × 價格 × 持有日曆天數（多空都扣，保守）。
    """
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr, sp = b.atr.to_numpy(float), b.sp.to_numpy(float)
    idx = b.index
    n = len(b)
    out = []
    pos = 0; ent = stop = R = 0.0; ei = -1
    def close(i, px, why):
        nonlocal pos
        days = max((idx[i].normalize() - idx[ei].normalize()).days, 0)
        gross = (px - ent) * pos
        cost = sp[ei] + fin * ent * days
        out.append(dict(entry_time=idx[ei], exit_time=idx[i], side=pos, entry=ent, exit=px, R=R, atr=R / stop_atr,
                        gross=gross, cost=cost, pnl=gross - cost, pnl_R=(gross - cost) / R, gross_R=gross / R,
                        bars=i - ei, days=days, why=why))
        pos = 0
    for i in range(1, n):
        s_prev = sig[i - 1]
        # 1) 上一根收盤的出場 / 反手指令，在本根開盤執行
        if pos != 0:
            if exit_mode == "flip" and s_prev == -pos:
                close(i, o[i], "flip")
            elif exit_mode == "cp" and ep_start[i - 1]:
                close(i, o[i], "cp")
        if pos == 0 and s_prev != 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
            pos, ent, ei = s_prev, o[i], i
            R = stop_atr * atr[i - 1]
            stop = ent - pos * R
        # 2) 本根盤中停損（跳空穿越 → 用開盤價）
        if pos == 1 and l[i] <= stop:
            close(i, min(o[i], stop), "stop")
        elif pos == -1 and h[i] >= stop:
            close(i, max(o[i], stop), "stop")
    if pos != 0:
        close(n - 1, c[-1], "end")
    return pd.DataFrame(out)


def daily_pnl_R(b, T, unit="R"):
    """逐日盯市損益（以每筆的 R 為單位），用來算與其他策略的相關性。"""
    if len(T) == 0:
        return pd.Series(dtype=float)
    days = b.day.unique()
    closes = b.groupby("day").close.last()
    s = pd.Series(0.0, index=closes.index)
    for tr in T.itertuples():
        d0, d1 = tr.entry_time.normalize(), tr.exit_time.normalize()
        seg = closes.loc[d0:d1].copy()
        if len(seg) == 0:
            continue
        seg.iloc[-1] = tr.exit
        prev = np.r_[tr.entry, seg.values[:-1]]
        inc = (seg.values - prev) * tr.side / tr.R
        inc[-1] -= tr.cost / tr.R
        s.loc[seg.index] += inc
    return s


def stats(T, col="pnl_R"):
    if len(T) < 2:
        return dict(n=len(T), mean=np.nan, t=np.nan, pf=np.nan, win=np.nan, tot=0.0, dd=np.nan)
    r = T[col].to_numpy()
    eq = np.cumsum(r)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(n=len(r), mean=r.mean(), t=r.mean() / r.std(ddof=1) * math.sqrt(len(r)),
                pf=gp / gl if gl > 0 else np.inf, win=(r > 0).mean(), tot=r.sum(),
                dd=(np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max())


def fmt(s):
    if s["n"] < 2:
        return f"n={s['n']:4d}"
    return (f"n={s['n']:4d} 每筆 {s['mean']:+.3f}R t={s['t']:+.2f} PF {s['pf']:.2f} 勝率 {s['win']:.2f} "
            f"累計 {s['tot']:+6.1f}R 回撤 {s['dd']:.1f}R")


def bar_pnl(b, T):
    """逐根損益（÷ 該筆的 R）：持有期間用開盤到下一根開盤；停損那根用 停損價 − 開盤；扣成本記在出場那根。"""
    o = b.open.to_numpy(float)
    n = len(o)
    pnl = np.zeros(n)
    pos = np.zeros(n)
    loc = {t: i for i, t in enumerate(b.index)}
    for tr in T.itertuples():
        ei, xi = loc[tr.entry_time], loc[tr.exit_time]
        if tr.why == "stop" or tr.why == "end":
            seg = np.arange(ei, xi)
            pnl[seg] += tr.side * (o[seg + 1] - o[seg]) / tr.R
            pnl[xi] += tr.side * (tr.exit - o[xi]) / tr.R
            pos[ei:xi + 1] = tr.side
        else:                                 # 在 xi 開盤出場
            seg = np.arange(ei, xi)
            pnl[seg] += tr.side * (o[seg + 1] - o[seg]) / tr.R
            pos[ei:xi] = tr.side
        pnl[xi] -= tr.cost / tr.R
    return pnl, pos


def detrend(b, T):
    """扣掉整段平均漂移（多頭 beta）後的每筆 R：pnl_R − 方向 × 每根平均漲幅 × 持有根數 ÷ R。"""
    if len(T) == 0:
        return pd.Series(dtype=float)
    drift = b.open.diff().mean()
    return T.pnl_R - T.side * drift * T.bars / T.R
