"""NAS100 z-score 反轉引擎（第四十五部分起）—— 以 XAU 第四部分「型態 + z-score / 布林帶」為雛形。

XAU 第四部分核心（規則 D）：
  型態（K 棒 t 收盤判定，多單；空單鏡像）
    star   : t-2 陰線且實體 ≥ 0.5 ATR14；t-1 實體 ≤ t-2 實體一半；t 陽線且收盤 > t-2 實體中點
    engulf : t-1 陰線、t 陽線、t 收盤 ≥ t-1 開盤、t 實體 ≥ t-1 實體
  型態最低點 = 過去 10 根最低（極值）；**極值那根 K 收盤在布林下軌外（z20 ≤ −2）**
  進場：之後 3 根內突破 t 高點（停損單；先破停損就取消）
  停損：極值 − 0.1 × ATR14；停利 2R；最多持有 16 根，以收盤平倉；每筆扣進場那根點差
這裡每個面向都是參數；也提供「純 z」訊號源（不要型態：z 由帶外回到帶內那根就是 t）。
K 棒由 M1 合成（每天 01:00 起算，不跨過每天 00:00~01:00 的休市）。期間、半年切法同 `nas_core.py`。
"""
import numpy as np
import pandas as pd
from numba import njit
from nas_core import O, H, L, C, S, DAYS, D, period, HALVES, RESEARCH, HOLDOUT, hm

_BARS = {}


def bars(k):
    """k 分鐘 K 棒的連續序列（跨日接起來算指標）。"""
    if k in _BARS:
        return _BARS[k]
    nb = (1440 - 60) // k
    rows = []
    for j in range(nb):
        a, b = 60 + j * k, 60 + (j + 1) * k
        hh = H[:, a:b]
        ok = np.isfinite(hh).any(1)
        oo, cc = O[:, a:b], C[:, a:b]
        fi = np.argmax(np.isfinite(oo), 1)
        li = b - a - 1 - np.argmax(np.isfinite(cc[:, ::-1]), 1)
        d = np.flatnonzero(ok)
        rows.append(pd.DataFrame(dict(d=d, j=j, o=oo[d, fi[d]], h=np.nanmax(hh[d], 1), l=np.nanmin(L[d, a:b], 1),
                                      c=cc[d, li[d]], sp=S[d, a + fi[d]])))
    B = pd.concat(rows).sort_values(["d", "j"]).reset_index(drop=True)
    B["min"] = 60 + B.j * k
    B["day"] = DAYS[B.d.values]
    c, h, l = B.c, B.h, B.l
    pc = c.shift()
    tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
    B["atr"] = tr.rolling(14).mean()
    for n in (10, 20, 30, 50, 100):
        B[f"z{n}"] = (c - c.rolling(n).mean()) / c.rolling(n).std()
    sd20 = c.rolling(20).std()
    B["bbw"] = 4 * sd20 / c.rolling(20).mean()
    B["bbw_pct"] = B.bbw.rolling(500).rank(pct=True)
    eff = lambda n: (c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()
    B["er5_20"] = eff(5) / eff(20)
    B["ma20"] = c.rolling(20).mean()
    _BARS[k] = B
    return B


def signals(k, zlen=20, zthr=2.0, src="pattern", swing=10, pattern="both"):
    """回傳訊號：t（型態 / 回到帶內那根）、side、ext（極值）、atr。
    src = 'pattern'：型態 + 極值 K 的 z 超過門檻；'pattern_noz'：只要型態（不看 z，對照）；'zcross'：純 z 回到帶內。"""
    B = bars(k)
    o, h, l, c, A = (B[x].to_numpy() for x in ("o", "h", "l", "c", "atr"))
    z = B[f"z{zlen}"].to_numpy()
    n = len(B)
    body = c - o
    out = []
    lmin = pd.Series(l).rolling(swing).min().to_numpy()
    hmax = pd.Series(h).rolling(swing).max().to_numpy()
    for s in (1, -1):
        if src == "zcross":
            zz = -s * z                                  # 多單：z ≤ −thr → −z ≥ thr
            t = np.flatnonzero((np.r_[np.nan, zz[:-1]] >= zthr) & (zz < zthr) & (A > 0))
            for i in t:
                if i < 3:
                    continue
                ext = l[i - 2:i + 1].min() if s == 1 else h[i - 2:i + 1].max()
                out.append((i, s, ext, A[i]))
            continue
        b2, b1, b0 = s * np.r_[np.nan, np.nan, body[:-2]], s * np.r_[np.nan, body[:-1]], s * body
        c2mid = np.r_[np.nan, np.nan, ((o + c) / 2)[:-2]]
        o1 = np.r_[np.nan, o[:-1]]
        star = (-b2 >= 0.5 * A) & (np.abs(b1) <= 0.5 * -b2) & (b0 > 0) & (s * (c - c2mid) > 0)
        eng = (b1 < 0) & (b0 > 0) & (s * (c - o1) >= 0) & (b0 >= -b1)
        if pattern == "star":
            eng[:] = False
        if pattern == "engulf":
            star[:] = False
        for i in np.flatnonzero(star | eng):
            if i < 30:
                continue
            kk = 3 if star[i] else 2
            seg = np.arange(i - kk + 1, i + 1)
            if s == 1:
                e = seg[np.argmin(l[seg])]
                ext = l[e]
                if ext > lmin[i]:
                    continue
            else:
                e = seg[np.argmax(h[seg])]
                ext = h[e]
                if ext < hmax[i]:
                    continue
            if src == "pattern" and not (-s * z[e] >= zthr):
                continue
            out.append((i, s, ext, A[i]))
    if not out:
        return pd.DataFrame(columns=["t", "side", "ext", "atr"])
    return pd.DataFrame(out, columns=["t", "side", "ext", "atr"]).sort_values("t").reset_index(drop=True)


@njit(cache=True)
def _sim(o, h, l, c, sp, dd, t_arr, s_arr, ext_arr, a_arr, buf, rr, wait, hold, mid, use_mid, entry_close):
    n = len(c)
    m = len(t_arr)
    out = np.full((m, 6), np.nan)
    for q in range(m):
        t, s, ext, A = t_arr[q], s_arr[q], ext_arr[q], a_arr[q]
        stop = ext - s * buf * A
        ei = -1
        ep = 0.0
        if entry_close:
            ei, ep = t, c[t]
        else:
            trig = h[t] if s == 1 else l[t]
            for j in range(t + 1, min(t + 1 + wait, n)):
                if (l[j] <= stop) if s == 1 else (h[j] >= stop):
                    break
                if (h[j] > trig) if s == 1 else (l[j] < trig):
                    ei = j
                    ep = max(trig, o[j]) if s == 1 else min(trig, o[j])
                    break
        if ei < 0:
            continue
        R = s * (ep - stop)
        if not R > 0:
            continue
        if use_mid:
            tg = mid[t]
            if s * (tg - ep) < 0.5 * R:
                tg = ep + s * 0.5 * R
        else:
            tg = ep + s * rr * R if rr > 0 else np.nan
        xi = min(ei + hold, n - 1)
        xp = c[xi]
        why = 0
        for j in range(ei + 1, min(ei + 1 + hold, n)):
            if (l[j] <= stop) if s == 1 else (h[j] >= stop):
                xi = j
                xp = min(stop, o[j]) if s == 1 else max(stop, o[j])
                why = 1
                break
            if (use_mid or rr > 0) and ((h[j] >= tg) if s == 1 else (l[j] <= tg)):
                xi = j
                xp = tg
                why = 2
                break
        pnl = s * (xp - ep) - sp[ei]
        out[q, 0] = ei
        out[q, 1] = xi
        out[q, 2] = pnl / R
        out[q, 3] = pnl
        out[q, 4] = R
        out[q, 5] = why
    return out


def run(k=15, zlen=20, zthr=2.0, src="pattern", swing=10, pattern="both", buf=0.1, rr=2.0, wait=3, hold=16,
        mid_exit=False, entry_close=False, hours=(1, 24), slip=0.0):
    """回傳每筆交易。hours = 型態 K 棒（t）開始的 broker 小時範圍（含頭不含尾）。"""
    B = bars(k)
    G = signals(k, zlen, zthr, src, swing, pattern)
    if len(G) == 0:
        return pd.DataFrame(columns=["day", "side", "R", "pnl", "risk", "hour"])
    hr = B["min"].to_numpy()[G.t.values] // 60
    G = G[(hr >= hours[0]) & (hr < hours[1])]
    a = _sim(B.o.to_numpy(), B.h.to_numpy(), B.l.to_numpy(), B.c.to_numpy(), B.sp.to_numpy(), B.d.to_numpy(),
             G.t.to_numpy(np.int64), G.side.to_numpy(np.int64), G.ext.to_numpy(float), G.atr.to_numpy(float),
             float(buf), float(rr), int(wait), int(hold), B.ma20.to_numpy(), bool(mid_exit), bool(entry_close))
    X = G.copy().reset_index(drop=True)
    X[["ei", "xi", "R", "pnl", "risk", "why"]] = a
    X = X[np.isfinite(X.R)].copy()
    X["ei"] = X.ei.astype(int)
    X["t"] = X.t.astype(int)
    X["day"] = B.day.to_numpy()[X.ei.values]
    X["hour"] = B["min"].to_numpy()[X.t.values] // 60
    if slip:
        X["pnl"] -= slip
        X["R"] = X.pnl / X.risk
    return X


def summ(X, a=RESEARCH[0], b=RESEARCH[1]):
    X = period(X, a, b)
    days = D.index[(D.index >= a) & (D.index < b)]
    if len(X) < 10:
        return dict(n=len(X))
    d = X.groupby("day").R.sum().reindex(days, fill_value=0)
    h = [d[(d.index >= p) & (d.index < q)].mean() for _, p, q in HALVES]
    return dict(n=len(X), per_day=len(X) / len(days), mR=X.R.mean(), PF=X.R[X.R > 0].sum() / -X.R[X.R <= 0].sum(),
                win=(X.R > 0).mean(), dt=d.mean() / d.std() * np.sqrt(len(d)), hpos=int(np.nansum(np.array(h) > 0)),
                h1=h[0], h2=h[1], h3=h[2], h4=h[3], L=X.R[X.side == 1].mean(), S=X.R[X.side == -1].mean())


def m1_check(X, k, wait=3, buf=0.1, rr=2.0, hold=16):
    """用 M1 逐分鐘重算同一批訊號（進場那分鐘就檢查停損；同一分鐘停損停利都碰到算停損）→ 檢查 K 棒層級回測有沒有偏樂觀。"""
    B = bars(k)
    bd, bm = B.d.to_numpy(), B["min"].to_numpy()
    out = []
    for r in X.itertuples():
        t, s = int(r.t), int(r.side)
        stop = r.ext - s * buf * r.atr
        trig = B.h.iat[t] if s == 1 else B.l.iat[t]
        mins = []
        for j in range(t + 1, min(t + 1 + wait + hold, len(B))):
            mins += [(bd[j], m, j) for m in range(bm[j], bm[j] + k)]
        ent = None
        last_entry_bar = t + wait
        res = np.nan
        for (d, m, j) in mins:
            if not np.isfinite(C[d, m]):
                continue
            if ent is None:
                if j > last_entry_bar:
                    break
                if (L[d, m] <= stop) if s == 1 else (H[d, m] >= stop):
                    break
                if (H[d, m] > trig) if s == 1 else (L[d, m] < trig):
                    ep = max(trig, O[d, m]) if s == 1 else min(trig, O[d, m])
                    R = s * (ep - stop)
                    if not R > 0:
                        break
                    tg = ep + s * rr * R
                    ent = (d, m, j, ep, R, tg, S[d, m])
                    endbar = j + hold
                    if (L[d, m] <= stop) if s == 1 else (H[d, m] >= stop):
                        res = (s * (stop - ep) - ent[6]) / R
                        break
                continue
            d0, m0, j0, ep, R, tg, sp = ent
            if j > endbar:
                break
            if (L[d, m] <= stop) if s == 1 else (H[d, m] >= stop):
                px = min(stop, O[d, m]) if s == 1 else max(stop, O[d, m])
                res = (s * (px - ep) - sp) / R
                break
            if (H[d, m] >= tg) if s == 1 else (L[d, m] <= tg):
                res = (s * (tg - ep) - sp) / R
                break
            lastpx = C[d, m]
        else:
            pass
        if ent is not None and not np.isfinite(res):
            d0, m0, j0, ep, R, tg, sp = ent
            # 時間到：最後一根 K 收盤
            jj = min(ent[2] + hold, len(B) - 1)
            res = (s * (B.c.iat[jj] - ep) - sp) / R
        out.append(res)
    Y = X.copy()
    Y["R_m1"] = out
    return Y
