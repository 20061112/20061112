"""第五十二部分：Entropy × Volatility 特徵庫（全部只用「當下以前」的資料）。

兩條資料流：
  1. 價格事件流（Range Bar / 磚塊）：負責「路徑結構」
     - 磚塊大小 = K_BRICK × ATR10（前 10 個交易日，當天開盤就固定，不用到當天資料）
     - 以 M1 收盤建構：收盤離參考價 ≥ 1 塊 → 產生同方向磚塊（可一次多塊），反轉也只要 1 塊（上下對稱）
     - 每塊方向 s ∈ {+1, −1}；磚塊完成時間 = 該 M1 收盤時間
     - 特徵（最近 N 塊）：
         H_dir    方向熵（上/下比例，bits，0~1）
         H_trans  轉移熵 H(S_t+1 | S_t)（bits，0~1）
         MI       互資訊 = H(S_t+1) − H(S_t+1 | S_t)，已做 Miller–Madow 小樣本偏誤修正
         p_cont   同方向延續機率（MI 沒有方向：高 MI 可能是強延續，也可能是規律交替；p_cont 補上方向）
         drift    最近 N 塊的淨方向（Σs / N）
  2. 固定時間流（M1）：負責「波動尺度」
     - sig_s   最近 30 分鐘 RV（每分鐘尺度）
     - V_roll  log(sig_s / 最近 1440 分鐘 RV)：短期相對 24 小時
     - V_seas  log(sig_s / 過去 20 天「同一時段 30 分鐘」RV 中位數)：扣掉日內季節性（亞洲盤天生就安靜）
     - J60     最近 60 分鐘跳躍占比 max(RV − BV, 0) / RV（Barndorff-Nielsen & Shephard bipower variation）
     - ER60    最近 60 分鐘 Kaufman 效率比（0 = 來回、1 = 單邊）
     - PE      最近 60 根 M1 收盤的排列熵（m = 3，正規化 0~1）
"""
import math
import numpy as np
import pandas as pd

K_BRICK = 0.05          # 磚塊 = 0.05 × ATR10（ATR10 ≈ 60~130 美元 → 約 3~6 美元一塊）
LN2 = np.log(2)


# ---------------------------------------------------------------- 磚塊（價格事件流）
def build_bricks(d, atr_by_day, k=K_BRICK):
    """d：M1（index = 時間，需有 close、day）。回傳 DataFrame：t（完成時間）、s（方向）、px（磚塊收盤參考價）、size、i（M1 位置）。"""
    C = d.close.values; T = d.index.values; days = d.day.values
    size_of_day = {dy: k * a for dy, a in atr_by_day.items() if np.isfinite(a)}
    out_t, out_s, out_p, out_z, out_i = [], [], [], [], []
    ref = None
    for n in range(len(C)):
        z = size_of_day.get(pd.Timestamp(days[n]))
        if z is None:
            continue
        if ref is None:
            ref = C[n]; continue
        diff = C[n] - ref
        m = int(np.floor(abs(diff) / z))
        if m >= 1:
            s = 1 if diff > 0 else -1
            for _ in range(m):
                ref += s * z
                out_t.append(T[n]); out_s.append(s); out_p.append(ref); out_z.append(z); out_i.append(n)
    return pd.DataFrame(dict(t=out_t, s=out_s, px=out_p, size=out_z, i=out_i))


def _h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def brick_features(B, N=60):
    """在每塊完成時，用最近 N 塊（含這塊）計算熵特徵。向量化的滾動計數。"""
    s = B.s.values
    up = (s == 1).astype(float)
    cont = np.r_[np.nan, (s[1:] == s[:-1]).astype(float)]          # 第 j 塊是否延續第 j−1 塊
    prev_up = np.r_[np.nan, up[:-1]]
    S = pd.DataFrame(dict(up=up, cont=cont,
                          uu=(prev_up == 1) & (up == 1), ud=(prev_up == 1) & (up == 0),
                          du=(prev_up == 0) & (up == 1), dd=(prev_up == 0) & (up == 0)))
    R = S.rolling(N, min_periods=N).sum()
    p_up = R.up / N
    H_dir = _h2(p_up)
    # 條件熵 H(S_t+1 | S_t)
    nu = R.uu + R.ud; nd = R.du + R.dd; n_tr = nu + nd                # 窗內的轉移數
    pu_u = R.uu / nu.replace(0, np.nan); pu_d = R.du / nd.replace(0, np.nan)
    H_trans = (nu / n_tr) * _h2(pu_u.fillna(0.5)) + (nd / n_tr) * _h2(pu_d.fillna(0.5))
    # Miller–Madow：H 的偏誤 ≈ −(K−1)/(2n ln2)；MI 的偏誤 ≈ +(格子數修正)，對 2×2 約 1/(2n ln2)
    mm_marg = 1 / (2 * N * LN2)
    mm_cond = 2 / (2 * N * LN2)
    MI = (H_dir + mm_marg) - (H_trans + mm_cond)
    p_cont = R.cont / n_tr
    drift = (2 * R.up - N) / N
    return pd.DataFrame(dict(t=B.t.values, i=B.i.values, s=s, size=B["size"].values, px=B.px.values,
                             H_dir=H_dir.values, H_trans=H_trans.values, MI=MI.values,
                             p_cont=p_cont.values, drift=drift.values))


# ---------------------------------------------------------------- 固定時間流
def perm_entropy(x, m=3, win=60):
    """排列熵（滾動），x：一維陣列。回傳與 x 等長，0~1。"""
    n = len(x)
    if n < m:
        return np.full(n, np.nan)
    W = np.lib.stride_tricks.sliding_window_view(x, m)
    code = np.zeros(len(W), int)
    ranks = np.argsort(np.argsort(W, axis=1, kind="stable"), axis=1, kind="stable")
    for j in range(m):
        code = code * m + ranks[:, j]
    K = m ** m
    oh = np.zeros((len(code), K)); oh[np.arange(len(code)), code] = 1
    cnt = pd.DataFrame(oh).rolling(win - m + 1, min_periods=win - m + 1).sum().values
    p = cnt / (win - m + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(p > 0, p * np.log(p), 0), axis=1) / np.log(math.factorial(m))
    H[np.isnan(cnt[:, 0])] = np.nan
    return np.r_[np.full(m - 1, np.nan), H]


def time_features(d):
    """每根 M1 收盤時可得的波動 / 路徑特徵（位置 n 的值只用到第 n 根以前，含第 n 根收盤）。"""
    c = d.close.values
    r = np.r_[0.0, np.diff(np.log(c))]
    r[np.r_[False, d.day.values[1:] != d.day.values[:-1]]] = 0.0     # 跨日缺口不算進日內波動
    r2 = pd.Series(r ** 2)
    sig_s = np.sqrt(r2.rolling(30, min_periods=25).mean())
    sig_l = np.sqrt(r2.rolling(1440, min_periods=1000).mean())
    absr = np.abs(r)
    bp = pd.Series(np.r_[0.0, absr[1:] * absr[:-1]])
    rv60 = r2.rolling(60, min_periods=50).sum()
    bv60 = (np.pi / 2) * (60 / 59) * bp.rolling(60, min_periods=50).sum()
    J60 = ((rv60 - bv60).clip(lower=0) / rv60.replace(0, np.nan))
    dc = pd.Series(np.abs(np.r_[0.0, np.diff(c)]))
    net60 = pd.Series(c).diff(60)
    ER60 = net60.abs() / dc.rolling(60).sum().replace(0, np.nan)
    sER60 = net60 / dc.rolling(60).sum().replace(0, np.nan)          # 帶方向
    PE = perm_entropy(c, 3, 60)

    F = pd.DataFrame(dict(sig_s=sig_s.values, V_roll=np.log((sig_s + 1e-9) / (sig_l + 1e-9)).values,
                          J60=J60.values, ER60=ER60.values, sER60=sER60.values, PE=PE), index=d.index)
    # 日內季節性：過去 20 天、同一個 30 分鐘時段（以 30 分鐘結束時計）的 sig_s 中位數
    slot = (d.index.hour * 60 + d.index.minute) // 30
    tmp = pd.DataFrame(dict(day=d.day.values, slot=slot, sig=sig_s.values), index=d.index)
    end_of_slot = tmp.groupby(["day", "slot"]).sig.last().unstack("slot")
    base = end_of_slot.rolling(20, min_periods=10).median().shift(1)    # shift(1)：只用前幾天
    b = base.stack().rename("base").reset_index()
    base_v = tmp[["day", "slot"]].merge(b, on=["day", "slot"], how="left").base.values
    F["V_seas"] = np.log((F.sig_s.values + 1e-9) / (base_v + 1e-9))
    return F
