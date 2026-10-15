"""第二十二部分：雙類型（上漲 / 下跌事件）指數核 Hawkes 過程的工具。

模型（對稱、β 固定）：
    λ↑(t) = μ + a_s·S↑(t) + a_c·S↓(t)
    λ↓(t) = μ + a_s·S↓(t) + a_c·S↑(t)
    S_x(t) = Σ_{t_i^x < t} exp(−β (t − t_i^x))        ← 遞迴：S ← S·exp(−β dt) + 1
a_s = 同向激發（上漲引發上漲）、a_c = 反向激發（上漲引發下跌）、ρ = a_c / a_s。

時間一律用「交易分鐘」：第 d 個交易日的 broker 01:00 = d·1380，分鐘數往後加，
所以夜間 / 週末的空檔被壓掉，S 在每天開頭歸零。

事件的三種做法：
  close  ：每根 M1 收盤價相對前一根的漲跌（0 不算事件），事件時間 = K 棒結束。
           幾乎每分鐘都有事件，時間間隔沒有資訊 → Hawkes 退化（只當示範）。
  bar k  ：只有 |Δ收盤| ≥ k × 尺度 的「大 K 棒」才算事件。
  step k ：價格每穿越一格 δ 就是一個事件（δ = k × 前一天 M1 平均高低差），
           K 棒內的路徑假設為 O→L→H→C（陽線）或 O→H→L→C（陰線），事件時間按路徑長度內插。
           在隨機漫步下，下一個事件同向或反向的機率剛好各 1/2，是乾淨的虛無假設。

ρ 的兩種算法：
  batch_em ：給定一段事件，用 EM（β 固定時 M-step 有封閉解）估 μ, a_s, a_c。
  online_em：每個事件用目前參數算責任（背景 / 同向 / 反向），累加器按時間指數遺忘，
             參數 = 累加器比值。ρ_online = Σp_反 / Σp_同，每個事件 O(1) 更新，不需重新估計。
"""
import numpy as np
import pandas as pd
from numba import njit

DAY_MIN = 1380.0          # broker 01:00 ~ 23:59


def trading_clock(m):
    """回傳每根 K 棒的交易日編號、日內分鐘（01:00 = 0）、交易分鐘起點。"""
    idx = m.index
    day = idx.normalize()
    days, day_id = np.unique(day.values, return_inverse=True)
    minute = (idx.hour.values * 60 + idx.minute.values - 60).astype(np.float64)
    t0 = day_id * DAY_MIN + minute
    return day_id.astype(np.int64), minute, t0, pd.DatetimeIndex(days)


def day_scale(m, day_id):
    """前一天的 M1 平均高低差（只用過去資料），當作 δ 與報酬的尺度。"""
    rng = (m.high - m.low).values
    nd = day_id.max() + 1
    s = np.bincount(day_id, rng, nd) / np.maximum(np.bincount(day_id, None, nd), 1)
    prev = np.empty(nd)
    prev[0] = s[0]            # 第一天沒有前一天，只用來暖機，之後不會拿來交易
    prev[1:] = s[:-1]
    return prev


def events_close(m, day_id, t0):
    c = m.close.values
    d = np.diff(c, prepend=np.nan)
    first = np.r_[True, day_id[1:] != day_id[:-1]]
    d[first] = 0.0
    sel = np.nan_to_num(d) != 0
    bar = np.nonzero(sel)[0]
    return t0[bar] + 1.0, np.sign(d[bar]).astype(np.int64), bar.astype(np.int64)


@njit(cache=True)
def _events_step(o, h, l, c, day_id, t0, delta):
    n = len(o)
    cap = 4 * n + 1000
    T = np.empty(cap); S = np.empty(cap, np.int64); B = np.empty(cap, np.int64)
    k = 0
    ref = o[0]
    prev_c = o[0]
    for i in range(n):
        if i == 0 or day_id[i] != day_id[i - 1]:
            ref = o[i]; prev_c = o[i]
        dl = delta[day_id[i]]
        if c[i] >= o[i]:
            p = (prev_c, o[i], l[i], h[i], c[i])
        else:
            p = (prev_c, o[i], h[i], l[i], c[i])
        tot = abs(p[2] - p[1]) + abs(p[3] - p[2]) + abs(p[4] - p[3])
        cum = 0.0
        for s in range(4):
            a = p[s]; b = p[s + 1]
            seg = abs(b - a) if s > 0 else 0.0
            if b > a:
                while b >= ref + dl:
                    ref += dl
                    frac = 0.0 if (s == 0 or tot == 0) else (cum + (ref - a)) / tot
                    if k >= cap:
                        break
                    T[k] = t0[i] + min(max(frac, 0.0), 1.0); S[k] = 1; B[k] = i; k += 1
            elif b < a:
                while b <= ref - dl:
                    ref -= dl
                    frac = 0.0 if (s == 0 or tot == 0) else (cum + (a - ref)) / tot
                    if k >= cap:
                        break
                    T[k] = t0[i] + min(max(frac, 0.0), 1.0); S[k] = -1; B[k] = i; k += 1
            cum += seg
        prev_c = c[i]
    return T[:k], S[:k], B[:k]


def events_step(m, day_id, t0, scale, k):
    return _events_step(m.open.values, m.high.values, m.low.values, m.close.values,
                        day_id, t0, scale * k)


def events_bigbar(m, day_id, t0, scale, k):
    """只把「大 K 棒」當事件：|收盤 − 前收盤| ≥ k × 尺度，時間 = K 棒結束。"""
    c = m.close.values
    d = np.diff(c, prepend=np.nan)
    first = np.r_[True, day_id[1:] != day_id[:-1]]
    d[first] = 0.0
    d = np.nan_to_num(d)
    sel = np.abs(d) >= k * scale[day_id]
    bar = np.nonzero(sel & (d != 0))[0]
    return t0[bar] + 1.0, np.sign(d[bar]).astype(np.int64), bar.astype(np.int64)


def make_events(m, day_id, t0, scale, kind):
    """kind：'close'、'step_k'（價格格子 δ = k × 尺度）、'bar_k'（大 K 棒）。"""
    if kind == "close":
        return events_close(m, day_id, t0)
    name, k = kind.split("_")
    if name == "bar":
        return events_bigbar(m, day_id, t0, scale, float(k))
    return events_step(m, day_id, t0, scale, float(k))


# ---------------------------------------------------------------- batch EM
@njit(cache=True)
def batch_em(t, s, T0, T1, beta, n_iter=60):
    """一段事件（t 已排序、皆在 [T0, T1)）的 EM 估計。回傳 μ, a_s, a_c, loglik。"""
    n = len(t)
    if n < 8:
        return np.nan, np.nan, np.nan, np.nan
    L = T1 - T0
    G = 0.0
    for i in range(n):
        G += (1.0 - np.exp(-beta * (T1 - t[i]))) / beta
    # 預先算每個事件發生前的 S_same、S_cross
    Ss = np.zeros(n); Sc = np.zeros(n)
    su = 0.0; sd = 0.0; last = T0
    for j in range(n):
        dec = np.exp(-beta * (t[j] - last))
        su *= dec; sd *= dec; last = t[j]
        if s[j] > 0:
            Ss[j] = su; Sc[j] = sd; su += 1.0
        else:
            Ss[j] = sd; Sc[j] = su; sd += 1.0
    mu = 0.5 * n / (2 * L); a_s = 0.25 * beta; a_c = 0.25 * beta
    for it in range(n_iter):
        pb = 0.0; ps = 0.0; pc = 0.0
        for j in range(n):
            lam = mu + a_s * Ss[j] + a_c * Sc[j]
            pb += mu / lam; ps += a_s * Ss[j] / lam; pc += a_c * Sc[j] / lam
        mu = pb / (2 * L); a_s = ps / G; a_c = pc / G
        a_s = max(a_s, 1e-9); a_c = max(a_c, 1e-9)
    ll = 0.0
    for j in range(n):
        ll += np.log(mu + a_s * Ss[j] + a_c * Sc[j])
    ll -= 2 * mu * L + (a_s + a_c) * G
    return mu, a_s, a_c, ll


@njit(cache=True)
def batch_em_days(t, s, ev_day, nd, lo, hi, beta, n_iter=60):
    """每個交易日日內分鐘 [lo, hi) 的事件分別估一次。"""
    out = np.full((nd, 4), np.nan)
    start = 0
    n = len(t)
    for d in range(nd):
        while start < n and ev_day[start] < d:
            start += 1
        end = start
        while end < n and ev_day[end] == d:
            end += 1
        T0 = d * 1380.0 + lo; T1 = d * 1380.0 + hi
        a = start
        while a < end and t[a] < T0:
            a += 1
        b = a
        while b < end and t[b] < T1:
            b += 1
        r = batch_em(t[a:b], s[a:b], T0, T1, beta, n_iter)
        out[d, 0] = r[0]; out[d, 1] = r[1]; out[d, 2] = r[2]; out[d, 3] = r[3]
        start = end
    return out


# --------------------------------------------------------------- online EM
@njit(cache=True)
def online_em(t, s, ev_day, beta, half_life, prior=20.0):
    """遞迴 Hawkes：每個事件更新一次，回傳每個事件之後的 ρ、a_s/β、a_c/β、μ。"""
    n = len(t)
    g = np.log(2.0) / half_life
    rho = np.empty(n); ns = np.empty(n); nc = np.empty(n); mus = np.empty(n)
    # 先驗：同向與反向各 0.25 分支比，背景佔一半
    rate0 = 0.5
    B = prior * 0.5; Cs = prior * 0.25; Cc = prior * 0.25; P = prior / beta
    Teff = B / (2 * rate0 * 0.5)
    mu = B / (2 * Teff); a_s = Cs / P; a_c = Cc / P
    su = 0.0; sd = 0.0; last = t[0] if n > 0 else 0.0
    for j in range(n):
        dt = t[j] - last
        if j > 0 and ev_day[j] != ev_day[j - 1]:
            su = 0.0; sd = 0.0
        else:
            dec = np.exp(-beta * dt); su *= dec; sd *= dec
        f = np.exp(-g * dt)
        B *= f; Cs *= f; Cc *= f; P *= f
        Teff = Teff * f + (1.0 - f) / g
        last = t[j]
        if s[j] > 0:
            sa = su; sb = sd
        else:
            sa = sd; sb = su
        lam = mu + a_s * sa + a_c * sb
        B += mu / lam; Cs += a_s * sa / lam; Cc += a_c * sb / lam
        P += 1.0 / beta
        if s[j] > 0:
            su += 1.0
        else:
            sd += 1.0
        mu = B / (2 * Teff); a_s = max(Cs / P, 1e-9); a_c = max(Cc / P, 1e-9)
        rho[j] = Cc / Cs; ns[j] = a_s / beta; nc[j] = a_c / beta; mus[j] = mu
    return rho, ns, nc, mus


@njit(cache=True)
def online_reversal(t, s, ev_day, half_life):
    """最簡單的對照：下一個事件與前一個反向的比例（時間指數遺忘）。隨機漫步下 = 0.5。"""
    n = len(t)
    g = np.log(2.0) / half_life
    out = np.empty(n)
    num = 5.0; den = 10.0; last = t[0] if n > 0 else 0.0
    for j in range(n):
        f = np.exp(-g * (t[j] - last)); num *= f; den *= f; last = t[j]
        if j > 0 and ev_day[j] == ev_day[j - 1]:
            den += 1.0
            if s[j] != s[j - 1]:
                num += 1.0
        out[j] = num / den
    return out


def to_bars(values, ev_bar, nbar, fill=np.nan):
    """事件序列的值 → 每根 K 棒收盤時可用的最新值（前向填補）。"""
    out = np.full(nbar, np.nan)
    out[ev_bar] = values            # 同一根多個事件時保留最後一個
    return pd.Series(out).ffill().fillna(fill).values


# ------------------------------------------------- 非對稱 online EM + 強度
@njit(cache=True)
def online_em_asym(t, s, ev_day, beta, half_life, prior=20.0):
    """四個激發係數各自估：a[接收方, 來源方]，0 = 上、1 = 下；μ 也分上下。
    回傳每個事件之後的狀態：S↑、S↓、μ↑、μ↓、a_uu、a_ud、a_du、a_dd（a_ud = 下跌引發上漲）。"""
    n = len(t)
    g = np.log(2.0) / half_life
    out = np.empty((n, 8))
    Bk = np.full(2, prior * 0.25)                   # 背景責任
    C = np.full((2, 2), prior * 0.125)              # C[接收, 來源]
    P = np.full(2, prior * 0.5 / beta)              # 每種來源事件的曝險
    Teff = prior * 0.5
    mu = Bk / Teff; a = C / P.reshape(1, 2)
    S = np.zeros(2); last = t[0] if n > 0 else 0.0
    for j in range(n):
        dt = t[j] - last
        if j > 0 and ev_day[j] != ev_day[j - 1]:
            S[:] = 0.0
        else:
            S *= np.exp(-beta * dt)
        f = np.exp(-g * dt)
        Bk *= f; C *= f; P *= f
        Teff = Teff * f + (1.0 - f) / g
        last = t[j]
        k = 0 if s[j] > 0 else 1
        lam = mu[k] + a[k, 0] * S[0] + a[k, 1] * S[1]
        Bk[k] += mu[k] / lam
        C[k, 0] += a[k, 0] * S[0] / lam
        C[k, 1] += a[k, 1] * S[1] / lam
        P[k] += 1.0 / beta
        S[k] += 1.0
        mu = Bk / Teff
        for r in range(2):
            for q in range(2):
                a[r, q] = max(C[r, q] / P[q], 1e-9)
        out[j, 0] = S[0]; out[j, 1] = S[1]; out[j, 2] = mu[0]; out[j, 3] = mu[1]
        out[j, 4] = a[0, 0]; out[j, 5] = a[0, 1]; out[j, 6] = a[1, 0]; out[j, 7] = a[1, 1]
    return out


def intensity_at_bars(t, state, ev_bar, t_bar_end, beta):
    """把事件後的狀態帶到每根 K 棒收盤：λ↑、λ↓（只用到收盤前的事件）。"""
    nbar = len(t_bar_end)
    st = np.full((nbar, 8), np.nan); te = np.full(nbar, np.nan)
    st[ev_bar] = state; te[ev_bar] = t
    st = pd.DataFrame(st).ffill().values; te = pd.Series(te).ffill().values
    dec = np.exp(-beta * np.clip(t_bar_end - te, 0, None))
    su = st[:, 0] * dec; sd = st[:, 1] * dec
    lu = st[:, 2] + st[:, 4] * su + st[:, 5] * sd
    ld = st[:, 3] + st[:, 6] * su + st[:, 7] * sd
    return lu, ld, st
