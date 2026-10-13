"""費馬光程核心（第二十三、二十四部分共用）。見 part23_fermat.py 的說明。"""
import numpy as np
from numba import njit


@njit(cache=True)
def _fermat(hi, lo, cl, atr, ts, W, look, binw, eps):
    """對每個取樣點 t：剖面 = [t−W−look, t−W) 的 K 棒（每根 1 分鐘平均攤在高低區間），
    回傳 光程淨值、光程路徑長、目前價位的 n、視窗起點價位的 n。"""
    m = len(ts)
    onet = np.full(m, np.nan)
    opath = np.full(m, np.nan)
    n_now = np.full(m, np.nan)
    n_start = np.full(m, np.nan)
    for j in range(m):
        t = ts[j]
        s = t - W
        if s - look < 0:
            continue
        a = atr[s]
        bw = binw * a
        pmin, pmax = 1e18, -1e18
        for k in range(s - look, s + 1):
            pmin = min(pmin, lo[k])
            pmax = max(pmax, hi[k])
        for k in range(s, t + 1):
            pmin = min(pmin, cl[k])
            pmax = max(pmax, cl[k])
        pmin -= bw
        pmax += bw
        nb = int((pmax - pmin) / bw) + 2
        h = np.zeros(nb)
        for k in range(s - look, s):
            b0 = (lo[k] - pmin) / bw
            b1 = (hi[k] - pmin) / bw
            if b1 - b0 < 1e-9:
                h[int(b0)] += 1.0
                continue
            i0, i1 = int(b0), int(b1)
            for i in range(i0, i1 + 1):
                ov = min(b1, i + 1.0) - max(b0, float(i))
                if ov > 0:
                    h[i] += ov / (b1 - b0)
        # n：每 1 ATR 停留分鐘數 + eps；F 為累積光程（單位：分鐘）
        n = h / binw + eps
        F = np.zeros(nb + 1)
        for i in range(nb):
            F[i + 1] = F[i] + n[i] * binw  # 每格寬 binw ATR

        def Fx(x):
            u = (x - pmin) / bw
            i = int(u)
            if i < 0:
                return F[0]
            if i >= nb:
                return F[nb]
            return F[i] + (u - i) * (F[i + 1] - F[i])

        onet[j] = abs(Fx(cl[t]) - Fx(cl[s]))
        acc = 0.0
        prev = Fx(cl[s])
        for k in range(s + 1, t + 1):
            cur = Fx(cl[k])
            acc += abs(cur - prev)
            prev = cur
        opath[j] = acc
        n_now[j] = n[int((cl[t] - pmin) / bw)]
        n_start[j] = n[int((cl[s] - pmin) / bw)]
    return onet, opath, n_now, n_start
