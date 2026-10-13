"""第二十六部分：離散時間的作用量最小化（費馬 / 變分法框架）。

標準化價格 x = (P − P0)/s，s = 起點前 60 根 1 分鐘收盤變動的標準差（在 t0 凍結）。
離散路徑 x_{-3..H}，x_{-3..0} 由起點的狀態估計 (v0, a0, j0) 決定；x_H = 終點條件（或自由）。
作用量（每一步 k = 1..H）：
    A = Σ [ (v_k − μ)²/(2 q) + λa/2 a_k² + λj/2 j_k² + λr R(x_k) ]
    v_k = Δx, a_k = Δ²x, j_k = Δ³x（後向差分），q = 1（x 以單步標準差為單位，隨機漫步下 v ~ N(0,1)）
R(x) = log n(P0 + x s)：過去 1440 根的時間價格剖面密度（停留越久越貴 = 折射率高），高斯平滑。
二次部分有封閉解（固定矩陣，所有樣本共用）；有 R 時用 MM（majorize–minimize）迭代：
    (G + λr c I) y = rhs + λr (c y − R'(y))，c 為 R'' 的上界 → 作用量單調下降。
"""
import numpy as np
from numba import njit, prange

GRID_HALF = 40.0   # 剖面網格 ±40 s
GRID_STEP = 0.25
NB = int(2 * GRID_HALF / GRID_STEP) + 1


# ------------------------------------------------------------------ 狀態估計
def endpoint_derivs(close, s, K=15, deg=3):
    """因果局部多項式（最後 K 根收盤，三次）在端點的 v、a、j（每根，以 s 為單位）。"""
    tt = np.arange(-K + 1, 1, dtype=float)
    V = np.vander(tt, deg + 1, increasing=True)        # [1, t, t², t³]
    Pinv = np.linalg.pinv(V)                            # (deg+1, K)
    # 端點 t=0 的導數：c1, 2c2, 6c3
    w_v, w_a, w_j = Pinv[1], 2 * Pinv[2], 6 * Pinv[3]
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(close, K)                   # 第 i 列 = close[i..i+K-1]
    out = np.full((len(close), 3), np.nan)
    out[K - 1:, 0] = W @ w_v
    out[K - 1:, 1] = W @ w_a
    out[K - 1:, 2] = W @ w_j
    return out / s[:, None]


# ------------------------------------------------------------------ 介質（剖面）
@njit(parallel=True, cache=True)
def profiles(hi, lo, cl, s, ts, look, smooth_bins):
    """每個樣本：以 cl[t] 為中心、s[t] 為單位的網格，[t−look, t) 每根 K 棒 1 分鐘攤在高低區間。
    回傳 R = log(停留分鐘/每 s + 1)，已高斯平滑（float32, len(ts) × NB）。"""
    m = len(ts)
    R = np.empty((m, NB), dtype=np.float32)
    ker_half = 3 * smooth_bins
    ker = np.empty(2 * ker_half + 1)
    for i in range(-ker_half, ker_half + 1):
        ker[i + ker_half] = np.exp(-0.5 * (i / smooth_bins) ** 2)
    ker /= ker.sum()
    for j in prange(m):
        t = ts[j]
        p0, sc = cl[t], s[t]
        h = np.zeros(NB)
        for k in range(t - look, t):
            a = (lo[k] - p0) / sc / GRID_STEP + (NB - 1) / 2
            b = (hi[k] - p0) / sc / GRID_STEP + (NB - 1) / 2
            if b < 0 or a > NB - 1:
                continue
            if b - a < 1e-9:
                h[int(a + 0.5)] += 1.0
                continue
            i0, i1 = max(int(a), 0), min(int(b), NB - 1)
            for i in range(i0, i1 + 1):
                ov = min(b, i + 1.0) - max(a, float(i))
                if ov > 0:
                    h[i] += ov / (b - a)
        n = h / GRID_STEP  # 每 1 s 停留分鐘數
        for i in range(NB):
            acc = 0.0
            for q in range(-ker_half, ker_half + 1):
                ii = min(max(i + q, 0), NB - 1)
                acc += ker[q + ker_half] * n[ii]
            R[j, i] = np.log(acc + 1.0)
    return R


def interp_rows(R, x):
    """R: (B, NB)；x: (B, M) → 線性內插的 R(x)、R'(x)、R''(x)（對 x 的導數）。"""
    u = np.clip(x / GRID_STEP + (NB - 1) / 2, 0, NB - 1.000001)
    i = u.astype(np.int64)
    f = u - i
    r0 = np.take_along_axis(R, i, 1)
    r1 = np.take_along_axis(R, i + 1, 1)
    val = r0 + f * (r1 - r0)
    d1 = (r1 - r0) / GRID_STEP
    im = np.clip(i - 1, 0, NB - 1)
    ip = np.clip(i + 2, 0, NB - 1)
    rm = np.take_along_axis(R, im, 1)
    rp = np.take_along_axis(R, ip, 1)
    d2 = ((rp - r1) - (r0 - rm)) / (2 * GRID_STEP ** 2)  # 粗略二階導數（只用在 log det）
    return val, d1, d2


# ------------------------------------------------------------------ 作用量求解器
class Action:
    def __init__(self, H, lam_a, lam_j, lam_r):
        self.H, self.lam_r = H, lam_r
        n = H + 4                      # z = [x_-3, x_-2, x_-1, x_0, x_1 .. x_H]
        rows, w, cv = [], [], []
        for k in range(1, H + 1):
            p = k + 3
            r = np.zeros(n); r[p] = 1; r[p - 1] = -1
            rows.append(r); w.append(1.0); cv.append(1.0)
            r = np.zeros(n); r[p] = 1; r[p - 1] = -2; r[p - 2] = 1
            rows.append(r); w.append(lam_a); cv.append(0.0)
            r = np.zeros(n); r[p] = 1; r[p - 1] = -3; r[p - 2] = 3; r[p - 3] = -1
            rows.append(r); w.append(lam_j); cv.append(0.0)
        self.A, self.w, self.cv = np.array(rows), np.array(w), np.array(cv)
        self.K = np.arange(4)
        self.Fc = np.arange(4, n - 1)  # 固定終點時的自由變數 x_1..x_{H-1}
        self.Ff = np.arange(4, n)      # 自由終點時 x_1..x_H

    def _mats(self, F, T):
        A, W = self.A, self.w[:, None]
        AF = A[:, F]
        G = AF.T @ (W * AF)
        return dict(G=G, BW=AF.T * self.w, AK=A[:, self.K], AT=A[:, T] if T is not None else None, F=F)

    def solve(self, zK, mu, X=None, R=None, iters=25, c=None):
        """zK: (B,4) 起點狀態；mu: (B,)；X: (B,) 終點或 None（自由）；R: (B,NB) 介質或 None。
        回傳完整路徑 z (B, H+4) 與作用量 (B,)。"""
        H = self.H
        F = self.Ff if X is None else self.Fc
        M = self._mats(F, None if X is None else np.array([H + 3]))
        Bn = len(mu)
        base = mu[:, None] * self.cv[None, :] - zK @ M["AK"].T
        if X is not None:
            base = base - X[:, None] * M["AT"][:, 0][None, :]
        rhs0 = base @ M["BW"].T                                    # (B, |F|)
        use_r = R is not None and self.lam_r > 0
        if use_r:
            c = c if c is not None else 1.0
            Gm = M["G"] + self.lam_r * c * np.eye(len(F))
        else:
            Gm = M["G"]
        Ginv = np.linalg.inv(Gm)
        y = rhs0 @ Ginv.T
        if use_r:
            for _ in range(iters):
                _, d1, _ = interp_rows(R, y)
                y = (rhs0 + self.lam_r * (c * y - d1)) @ Ginv.T
        z = np.empty((Bn, H + 4))
        z[:, :4] = zK
        z[:, F] = y
        if X is not None:
            z[:, H + 3] = X
        return z, self.cost(z, mu, R)

    def cost(self, z, mu, R=None):
        r = z @ self.A.T - mu[:, None] * self.cv[None, :]
        A = 0.5 * (self.w[None, :] * r ** 2).sum(1)
        if R is not None and self.lam_r > 0:
            val, _, _ = interp_rows(R, z[:, 4:])
            A = A + self.lam_r * val.sum(1)
        return A

    def logdet(self, z, R, X_fixed=True):
        """作用量在最優路徑上的 Hessian log det（Laplace 近似：接近最優的路徑「體積」∝ det^{-1/2}）。"""
        F = self.Fc if X_fixed else self.Ff
        G = self._mats(F, None)["G"]
        if R is None or self.lam_r == 0:
            return np.full(len(z), np.linalg.slogdet(G)[1])
        _, _, d2 = interp_rows(R, z[:, F])
        Hs = G[None] + self.lam_r * np.einsum("bi,ij->bij", d2, np.eye(len(F)))
        sg, ld = np.linalg.slogdet(Hs)
        return np.where(sg > 0, ld, np.nan)


def initial_state(v0, a0, j0):
    """由端點 v、a、j（後向差分定義）還原 x_-3..x_0（x_0 = 0）。"""
    x0 = np.zeros_like(v0)
    xm1 = x0 - v0
    vm1 = v0 - a0
    xm2 = xm1 - vm1
    am1 = a0 - j0
    vm2 = vm1 - am1
    xm3 = xm2 - vm2
    return np.stack([xm3, xm2, xm1, x0], 1)
