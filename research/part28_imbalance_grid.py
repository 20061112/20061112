"""第二十八部分：紅綠不平衡的完整網格 —— 切片大小 N × 不平衡比例 × 持有根數 × 時框。
N（切片）     5 6 8 10 12 15 20 25 30 40 50 60 80 100
不平衡比例    多數色 ≥ 60% 65% 70% 75% 80% 85% 90% 100%（k = ceil(比例 × N)）
持有 h        1 3 6 12 24 48 根
時框          M1（對照）M5 M15 M30 H1
每格分多（陽多數）/ 空（陰多數）兩邊：
  順勢報酬 = 往多數色方向的 h 根報酬（ATR），扣掉同期間所有 K 棒的平均（去掉黃金多頭漂移）
  對稱值   = (多邊 + 空邊) / 2 → 漂移中性的「顏色效應」；> 0 = 延續、< 0 = 反轉
  可交易   = |對稱值| − 平均點差（ATR）
t 值以「日」為單位（同一天的訊號高度重疊）。發現 2023/6~2024/12、驗證 2025/1~2026/10。
"""
import numpy as np
import pandas as pd
from load import load

SPLIT = pd.Timestamp("2025-01-01")
TFS = [("M1", "1min", 1), ("M5", "5min", 5), ("M15", "15min", 15), ("M30", "30min", 30), ("H1", "1h", 60)]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "mean"}
NS = [5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]
FR = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
HS = [1, 3, 6, 12, 24, 48]
MIN_N = 50


def stats(v, day):
    """平均、以日為單位的 t、筆數。"""
    if len(v) < MIN_N:
        return np.nan, np.nan, len(v)
    dm = pd.Series(v).groupby(day).sum()               # 每日加總（同一天的訊號當成一筆）
    t = dm.mean() / dm.std() * np.sqrt(len(dm)) if len(dm) > 2 else np.nan
    return v.mean(), t, len(v)


def run_tf(m1, tf, rule, mins):
    df = m1 if rule == "1min" else m1.resample(rule, label="left", closed="left").agg(AGG).dropna()
    d = np.sign(df.close - df.open).to_numpy()
    atr = (df.high - df.low).rolling(20).mean().to_numpy()
    c = df.close.to_numpy()
    tm = df.index.values.astype("datetime64[m]").astype(np.int64)
    per = (df.index >= SPLIT).astype(int)
    day = (df.index - pd.Timedelta(hours=1)).normalize().values
    cost = df.spread.to_numpy() * 0.01 / atr
    n = len(df)
    gcum = np.r_[0, np.cumsum(d > 0)]
    rcum = np.r_[0, np.cumsum(d < 0)]
    fwd, base = {}, {}
    for h in HS:
        f = np.full(n, np.nan)
        span = np.full(n, np.inf)
        span[:-h] = tm[h:] - tm[:-h]
        f[:-h] = (c[h:] - c[:-h]) / atr[:-h]
        f[span > h * mins * 1.5 + 3 * 1440] = np.nan          # 容許跨日休市 / 週末，只排除資料缺口
        fwd[h] = f
        base[h] = np.array([np.nanmean(f[per == q]) for q in (0, 1)])[per]
    rows = []
    for N in NS:
        idx = np.arange(n)
        g = np.full(n, np.nan)
        r = np.full(n, np.nan)
        g[N - 1:] = gcum[N:] - gcum[:-N]
        r[N - 1:] = rcum[N:] - rcum[:-N]
        span = np.full(n, np.inf)
        span[N - 1:] = tm[N - 1:] - tm[:n - N + 1]
        bad = (span > (N - 1) * mins * 1.5 + 3 * 1440) | np.isnan(atr)   # 同上
        g[bad] = np.nan
        r[bad] = np.nan
        for fr in FR:
            k = int(np.ceil(fr * N - 1e-9))
            if k <= N / 2:
                continue
            L, S = g >= k, r >= k
            freq = (L | S).sum() / np.isfinite(g).sum()
            for h in HS:
                f = fwd[h]
                x = f - base[h]
                rec = dict(tf=tf, N=N, fr=fr, k=k, h=h, freq=freq)
                for q in (0, 1):
                    mL = L & (per == q) & np.isfinite(f)
                    mS = S & (per == q) & np.isfinite(f)
                    mL_, tL, nL = stats(x[mL], day[mL])
                    mS_, tS, nS = stats(-x[mS], day[mS])
                    both = np.r_[x[mL], -x[mS]]
                    dd = np.r_[day[mL], day[mS]]
                    m_, t_, n_ = stats(both, dd)
                    rec.update({f"L{q}": mL_, f"S{q}": mS_, f"nL{q}": nL, f"nS{q}": nS, f"all{q}": m_, f"t{q}": t_,
                                f"sym{q}": (mL_ + mS_) / 2, f"cost{q}": np.nanmean(np.r_[cost[mL], cost[mS]]) if n_ else np.nan,
                                # 實際損益（不扣漂移、扣點差）：多邊做多、空邊做空
                                f"rawL{q}": np.nanmean(f[mL] - cost[mL]) if nL >= MIN_N else np.nan,
                                f"rawS{q}": np.nanmean(-f[mS] - cost[mS]) if nS >= MIN_N else np.nan})
                rows.append(rec)
    print(f"  {tf} 完成", flush=True)
    return rows


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    rows = []
    for tf, rule, mins in TFS:
        rows += run_tf(m1, tf, rule, mins)
    pd.DataFrame(rows).to_csv("part28_grid.csv", index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
