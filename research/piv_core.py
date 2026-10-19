"""NAS100 前高 / 前低突破引擎（第四十一部分起）—— 以 XAU `FINAL_STRATEGY_M15.md` 的核心為雛形。

XAU 核心：日線（前一日收盤 vs SMA50、SMA50 10 日斜率）決定今天只做多 / 只做空 / 不做；
          每根 M15（01:00 起算）若開盤在前一根 M15 高（低）內側，盤中第一次碰到就以停損單在該價成交；
          停損 2 × ATR15（前 20 根 M15 的高低差平均），停利 3 × 停損距離，22:00 後不開新單，當天最後一根平倉。
這裡把每個面向都變成參數：K 線週期 k、進場時段、方向規則、點位（前 1 / N 根）、停損倍數、RR、出場、B 條件。
資料與期間沿用 `nas_core.py`（研究期 2023/10~2025/09 四個半年，保留期 2025/10~2026/10）。
成交：觸價在點位成交（該分鐘開盤已越過點位則用開盤價）；進場那分鐘只檢查停損（保守）；同一分鐘停損停利都碰到算停損。
"""
import numpy as np
import pandas as pd
from numba import njit
from nas_core import O, H, L, C, S, DAYS, D, period, HALVES, RESEARCH, HOLDOUT, hm

ND, NM = C.shape
DAY0 = 60                      # 01:00
DAY_END = 1440                 # 交易日最後一分鐘 < 24:00
_BARS = {}


def bars(k):
    """k 分鐘 K 棒（從 01:00 起算）：每天 (nb) 根的 open / high / low / ATR20 / 前 n 根高低。跨日連續計算。"""
    if k in _BARS:
        return _BARS[k]
    nb = (DAY_END - DAY0) // k
    bo, bh, bl = (np.full((ND, nb), np.nan) for _ in range(3))
    for j in range(nb):
        a, b = DAY0 + j * k, DAY0 + (j + 1) * k
        hh, ll = H[:, a:b], L[:, a:b]
        ok = np.isfinite(hh).any(1)
        bh[ok, j] = np.nanmax(hh[ok], 1)
        bl[ok, j] = np.nanmin(ll[ok], 1)
        oo = O[:, a:b]
        first = np.argmax(np.isfinite(oo), 1)
        bo[ok, j] = oo[ok, first[ok]]
    flat_ok = np.isfinite(bh.ravel())
    fh, fl = pd.Series(bh.ravel()[flat_ok]), pd.Series(bl.ravel()[flat_ok])
    atr = (fh - fl).rolling(20).mean().shift(1)
    out = dict(nb=nb, o=bo, h=bh, l=bl)
    out["atr"] = np.full(ND * nb, np.nan)
    out["atr"][flat_ok] = atr.values
    out["atr"] = out["atr"].reshape(ND, nb)
    for n in (1, 2, 3, 4, 6, 8):
        ph, pl = np.full(ND * nb, np.nan), np.full(ND * nb, np.nan)
        ph[flat_ok] = fh.rolling(n).max().shift(1).values
        pl[flat_ok] = fl.rolling(n).min().shift(1).values
        out[f"ph{n}"], out[f"pl{n}"] = ph.reshape(ND, nb), pl.reshape(ND, nb)
    _BARS[k] = out
    return out


def direction(n=50, slope=10, rule="sma"):
    """每天的方向（只用前一日以前的日線）：+1 只做多、−1 只做空、0 不做。"""
    c = D.c
    if rule == "sma":
        sma = c.rolling(n).mean()
        a = np.sign(c.shift(1) - sma.shift(1))
        b = np.sign(sma.shift(1) - sma.shift(1 + slope))
        d = np.where((a == b) & (a != 0), a, 0)
    elif rule == "price":                     # 只看收盤在均線上下
        sma = c.rolling(n).mean()
        d = np.sign(c.shift(1) - sma.shift(1)).fillna(0).to_numpy()
    elif rule == "mom":                       # n 日報酬正負
        d = np.sign(c.shift(1) - c.shift(1 + n)).fillna(0).to_numpy()
    return np.nan_to_num(np.asarray(d, float))


ATRD = ((D.h - D.l).rolling(10).mean().shift(1)).to_numpy()
DOPEN = D.o.to_numpy()


@njit(cache=True)
def _sim(O, H, L, C, S, k, nb, bo, bh_lvl, bl_lvl, batr, dirs, mode, m0, m1, sm, rr, exit_min, bthr, atrd, dopen, ctrl):
    """mode 1 = 順日線方向；-1 = 逆日線方向；2 = 不分方向（兩邊都做）。ctrl=1：不等突破，每根開盤直接順方向進場（beta 對照）。"""
    nd, nm = C.shape
    out = np.full((nd * nb * 2, 9), np.nan)
    t = 0
    for d in range(nd):
        last = nm - 1
        while last > 0 and not np.isfinite(C[d, last]):
            last -= 1
        if exit_min > 0 and exit_min < last:
            last = exit_min
            while last > 0 and not np.isfinite(C[d, last]):
                last -= 1
        for j in range(nb):
            a = 60 + j * k
            if a < m0 or a >= m1:
                continue
            atr = batr[d, j]
            if not np.isfinite(atr) or not np.isfinite(bo[d, j]):
                continue
            for side in (1, -1):
                if mode == 2:
                    pass
                elif mode == 1:
                    if dirs[d] != side:
                        continue
                else:
                    if dirs[d] != -side:
                        continue
                if ctrl == 1:
                    lvl = bo[d, j]
                    ent = a
                    while ent < a + k and not np.isfinite(O[d, ent]):
                        ent += 1
                    if ent >= a + k:
                        continue
                else:
                    lvl = bh_lvl[d, j] if side == 1 else bl_lvl[d, j]
                    if not np.isfinite(lvl):
                        continue
                    if (bo[d, j] >= lvl) if side == 1 else (bo[d, j] <= lvl):
                        continue
                    ent = -1
                    for m in range(a, min(a + k, nm)):
                        if not np.isfinite(C[d, m]):
                            continue
                        if (H[d, m] >= lvl) if side == 1 else (L[d, m] <= lvl):
                            ent = m
                            if side == 1 and O[d, m] > lvl:
                                lvl = O[d, m]
                            if side == -1 and O[d, m] < lvl:
                                lvl = O[d, m]
                            break
                    if ent < 0:
                        continue
                if bthr > -1e8 and np.isfinite(atrd[d]):
                    if (lvl - dopen[d]) * side / atrd[d] <= -bthr:
                        continue
                if ent >= last:
                    continue
                risk = sm * atr
                st = lvl - side * risk
                tg = lvl + side * rr * risk if rr > 0 else np.nan
                px = C[d, last]
                xm = last
                why = 0
                for m in range(ent, last + 1):
                    if not np.isfinite(C[d, m]):
                        continue
                    if (L[d, m] <= st) if side == 1 else (H[d, m] >= st):
                        px = st if m == ent else (min(O[d, m], st) if side == 1 else max(O[d, m], st))
                        xm = m
                        why = 1
                        break
                    if m > ent and rr > 0 and ((H[d, m] >= tg) if side == 1 else (L[d, m] <= tg)):
                        px = tg
                        xm = m
                        why = 2
                        break
                cost = S[d, ent] if np.isfinite(S[d, ent]) else 0.0
                pnl = side * (px - lvl) - cost
                out[t, 0] = d
                out[t, 1] = j
                out[t, 2] = side
                out[t, 3] = ent
                out[t, 4] = xm
                out[t, 5] = pnl / risk
                out[t, 6] = pnl
                out[t, 7] = risk
                out[t, 8] = why
                t += 1
    return out[:t]


def run(k=15, mode=1, hours=(1, 22), sm=2.0, rr=3.0, exit_min=0, bthr=None, nprev=1, dirs=None, ctrl=False, slip=0.0):
    """回傳每筆交易 DataFrame。hours = 可進場的 K 棒開始時間（broker 小時，含頭不含尾）。"""
    B = bars(k)
    if dirs is None:
        dirs = direction()
    a = _sim(O, H, L, C, S, int(k), B["nb"], B["o"], B[f"ph{nprev}"], B[f"pl{nprev}"], B["atr"], dirs.astype(np.float64),
             int(mode), int(hours[0] * 60), int(hours[1] * 60), float(sm), float(rr), int(exit_min),
             float(-1e9 if bthr is None else bthr), ATRD, DOPEN, int(ctrl))
    X = pd.DataFrame(a, columns=["d", "j", "side", "ent", "xm", "R", "pnl", "risk", "why"])
    X["d"] = X.d.astype(int)
    X["day"] = DAYS[X.d.values]
    if slip:
        X["pnl"] -= slip
        X["R"] = X.pnl / X.risk
    return X


def summ(X, a=RESEARCH[0], b=RESEARCH[1]):
    """研究期（或任一期間）：每筆 R、逐日 R 加總的 t 值、四個半年（每天平均 R）、多空、PF。"""
    X = period(X, a, b)
    days = D.index[(D.index >= a) & (D.index < b)]
    if len(X) < 10:
        return dict(n=len(X))
    d = X.groupby("day").R.sum().reindex(days, fill_value=0)
    h = [d[(d.index >= p) & (d.index < q)].mean() for _, p, q in HALVES]
    return dict(n=len(X), per_day=len(X) / max(1, X.day.nunique()), mR=X.R.mean(), PF=X.R[X.R > 0].sum() / -X.R[X.R <= 0].sum(),
                win=(X.R > 0).mean(), dayR=d.mean(), dt=d.mean() / d.std() * np.sqrt(len(d)),
                hpos=int(np.nansum(np.array(h) > 0)), h1=h[0], h2=h[1], h3=h[2], h4=h[3],
                L=X.R[X.side == 1].mean(), S=X.R[X.side == -1].mean(), nL=int((X.side == 1).sum()), nS=int((X.side == -1).sum()))
