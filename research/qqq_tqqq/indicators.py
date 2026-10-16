"""把 XAUUSD 研究各分支的指標改寫成 QQQ 日線版。每個函式只用到「當天收盤以前」的資料（未 shift，
交給引擎統一 shift 1 天）。

來源對照：
  er            Kaufman 效率比                          features.py / FINAL_STRATEGY_ER（optimistic-meitner）
  perm_entropy  排列熵                                  entropy_vol.py（optimistic-meitner / ccr-2a50af23）
  dir_entropy   三狀態方向熵、轉移互資訊               ev_features.py entropy_engine（optimistic-meitner）
  jump_ratio    bipower 跳躍占比                        entropy_vol.py time_features J60
  har_vol       HAR 波動預測（walk-forward）            volfc.py / part55_vol_verify（ccr-2a50af23）
  cusum_state   Page 雙邊 CUSUM 狀態                    changepoint.py / FINAL_STRATEGY_CUSUM（ccr-ef7b4962）
  var_ratio     變異數比（ATR 時間縮放比）              part22_atr_scaling（ccr-d03ae866）
  hawkes        下跌衝擊的 Hawkes 強度                  hawkes.py（magical-goldberg）
  ou_z          滾動 AR(1)/OU 偏離                      part36_ou / FINAL_STRATEGY_OU（optimistic-meitner）
  td_setup      TD Sequential setup 9                   td_sequential.py（ccr-8d530ef9）
  tsmom         時間序列動能                            part26_direction（ccr-b41b8211）
"""
import math
import numpy as np
import pandas as pd


def rz(x, look=252):
    """滾動 z-score（與 ATR z 同一套做法）。"""
    return (x - x.rolling(look).mean()) / x.rolling(look).std()


def er(c, n):
    """帶方向 ER：(C_t − C_{t−n}) / Σ|ΔC|，−1~+1。"""
    return c.diff(n) / c.diff().abs().rolling(n).sum()


def perm_entropy(c, m=3, win=60):
    x = c.values
    W = np.lib.stride_tricks.sliding_window_view(x, m)
    ranks = np.argsort(np.argsort(W, axis=1, kind="stable"), axis=1, kind="stable")
    code = np.zeros(len(W), int)
    for j in range(m):
        code = code * m + ranks[:, j]
    K = m ** m
    oh = np.zeros((len(code), K)); oh[np.arange(len(code)), code] = 1
    cnt = pd.DataFrame(oh).rolling(win - m + 1).sum().values
    p = cnt / (win - m + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(p > 0, p * np.log(p), 0), axis=1) / np.log(math.factorial(m))
    H[np.isnan(cnt[:, 0])] = np.nan
    return pd.Series(np.r_[np.full(m - 1, np.nan), H], index=c.index)


def ewma_sigma(r, span=20):
    """事前波動：|r| 的 EWMA × √(π/2)，只用到前一天（CUSUM 文件的做法）。"""
    return (r.abs().ewm(span=span).mean() * np.sqrt(np.pi / 2)).shift(1)


def dir_entropy(r, W=60, band=0.3):
    """三狀態（下/中性/上）方向熵 H_D 與 Miller–Madow 修正後的轉移互資訊 I1，皆 /ln3。"""
    sig = ewma_sigma(r)
    s = pd.Series(1, index=r.index)
    s[r > band * sig] = 2; s[r < -band * sig] = 0
    oh = pd.get_dummies(s).reindex(columns=[0, 1, 2], fill_value=0).astype(float)
    cnt = oh.rolling(W).sum().values
    p = cnt / W
    with np.errstate(divide="ignore", invalid="ignore"):
        HD = -np.nansum(np.where(p > 0, p * np.log(p), 0), axis=1) / np.log(3)
    prev = s.shift(1)
    pair = (prev * 3 + s).where(prev.notna(), -1).astype(int)
    oh2 = pd.get_dummies(pair).reindex(columns=range(9), fill_value=0).astype(float)
    c2 = oh2.rolling(W).sum().values.reshape(-1, 3, 3)
    n = c2.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        pj = c2 / n[:, None, None]
        hj = -np.nansum(np.where(pj > 0, pj * np.log(pj), 0), axis=(1, 2))
        pr = pj.sum(2)
        hp = -np.nansum(np.where(pr > 0, pr * np.log(pr), 0), axis=1)
    HT = hj - hp
    I1 = (HD * np.log(3) + 2 / (2 * W)) - (HT + 6 / (2 * W))
    HD = pd.Series(HD, index=r.index); HD.iloc[:W] = np.nan
    I1 = pd.Series(I1 / np.log(3), index=r.index); I1.iloc[:W + 1] = np.nan
    return HD, I1


def jump_ratio(r, win=60):
    rv = (r ** 2).rolling(win).sum()
    bv = (np.pi / 2) * (win / (win - 1)) * (r.abs() * r.abs().shift(1)).rolling(win).sum()
    return ((rv - bv).clip(lower=0) / rv)


def parkinson_var(q):
    return np.log(q.high / q.low) ** 2 / (4 * np.log(2))


def har_vol(q, h=22, refit="YS"):
    """HAR-RV：log 未來 h 日平均變異數 ~ log RV 日 / 5 日 / 22 日。每年初只用過去資料重估（walk-forward）。
    RV 日 = Parkinson + 隔夜跳空平方。回傳年化預測波動。"""
    gap = np.log(q.open / q.close.shift(1)).fillna(0) ** 2
    rv = (parkinson_var(q) + gap).clip(lower=1e-10)
    ld = np.log(rv); lw = np.log(rv.rolling(5).mean()); lm = np.log(rv.rolling(22).mean())
    fut = np.log(rv[::-1].rolling(h).mean()[::-1].shift(-1))   # t+1..t+h
    X = pd.concat([ld, lw, lm], axis=1).values
    y = fut.values
    out = np.full(len(q), np.nan)
    years = q.index.year
    for yv in sorted(set(years))[3:]:
        te = years == yv
        # 訓練：目標期間必須在測試年之前結束
        last_tr = np.flatnonzero(years < yv)
        if len(last_tr) == 0:
            continue
        cut = last_tr[-1] - h
        tr = np.zeros(len(q), bool); tr[:max(cut, 0)] = True
        tr &= np.isfinite(X).all(1) & np.isfinite(y)
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        res = y[tr] - A @ b
        sm = np.mean(np.exp(res))
        fin = te & np.isfinite(X).all(1)
        out[fin] = np.exp(np.column_stack([np.ones(fin.sum()), X[fin]]) @ b) * sm
    return pd.Series(np.sqrt(out * 252), index=q.index)


def cusum_state(r, k=0.25, h=6.0, span=20):
    """Page 雙邊 CUSUM，z 截在 ±4；回傳最近一次訊號的方向（+1/−1，起始 0）。"""
    z = (r / ewma_sigma(r, span)).clip(-4, 4).fillna(0).values
    sp = sn = 0.0; st = 0; out = np.zeros(len(z))
    for i, zi in enumerate(z):
        sp = max(0.0, sp + zi - k); sn = max(0.0, sn - zi - k)
        if sp > h:
            st = 1; sp = sn = 0.0
        elif sn > h:
            st = -1; sp = sn = 0.0
        out[i] = st
    return pd.Series(out, index=r.index)


def var_ratio(r, qn=5, win=250):
    """VR = Var(q 日報酬) / (q · Var(1 日報酬))；>1 偏趨勢、<1 偏均值回歸。"""
    rq = r.rolling(qn).sum()
    return rq.rolling(win).var() / (qn * r.rolling(win).var())


def hawkes(r, thr=2.0, tau=10.0, span=20):
    """下跌衝擊（r < −thr·σ）的指數衰減強度。"""
    ev = (r < -thr * ewma_sigma(r, span)).astype(float).values
    a = np.exp(-1 / tau); lam = 0.0; out = np.zeros(len(ev))
    for i, e in enumerate(ev):
        lam = lam * a + e
        out[i] = lam
    return pd.Series(out, index=r.index)


def ou_z(c, L=250):
    """滾動 AR(1)：Δp = a + b·p_{t−1}；z = (p − μ*) / σ_eq。b ≥ 0（無回歸）時 NaN。"""
    p = np.log(c.values); n = len(p); out = np.full(n, np.nan)
    for i in range(L, n):
        x = p[i - L:i]; y = p[i - L + 1:i + 1] - x
        xm = x.mean(); b = ((x - xm) * (y - y.mean())).sum() / ((x - xm) ** 2).sum()
        if b >= 0:
            continue
        a = y.mean() - b * xm
        res = y - a - b * x
        phi = 1 + b
        if abs(phi) >= 1:
            continue
        s_eq = res.std() / math.sqrt(1 - phi ** 2)
        out[i] = (p[i] - (-a / b)) / s_eq
    return pd.Series(out, index=c.index)


def td_setup(c, n=9):
    """TD 賣出 setup：連續 n 天 C > C_{t−4}。回傳 setup 完成後的天數（未完成 = 大數）。"""
    up = (c > c.shift(4)).values
    cnt = 0; since = 10 ** 6; out = np.zeros(len(c))
    for i, u in enumerate(up):
        cnt = cnt + 1 if u else 0
        since = 0 if cnt == n else since + 1
        out[i] = since
    return pd.Series(out, index=c.index)


def tsmom(c, n=252):
    return c / c.shift(n) - 1
