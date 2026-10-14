"""對帳：第二十八部分逐年檢查（M30 延續兩邊都正）vs 第三十一部分單獨交易（做空虧）為什麼不同。
同一個組合（M30、N 根中 ≥ k 根同色、持有 48 根），一步一步從 A 改到 C：
  A 第二十八部分：所有符合條件的 K 棒（重疊）、扣掉「當年所有 K 棒持有 48 根的平均」（去漂移）、不扣點差
  B 同 A 但不去漂移（真實漲跌）
  C 第三十一部分：不重疊（進場後 48 根內不再進場）、不去漂移、扣點差
單位 ATR20(M30)，每格 = 做多那邊 / 做空那邊 的平均。
"""
import numpy as np
import pandas as pd
from load import load

m1 = load("data/XAUUSD_M1_2023_2026.csv")
m30 = m1.resample("30min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}).dropna()
d = np.sign(m30.close - m30.open)
atr = (m30.high - m30.low).rolling(20).mean()
f = (m30.close.shift(-48) - m30.close) / atr
cost = m30.spread * 0.01 / atr
yr = m30.index.year
drift = f.groupby(yr).transform("mean")
out = open("results_part31_reconcile.txt", "w")


def P(s=""):
    print(s)
    out.write(s + "\n")


for N, k in ((25, 15), (30, 18)):
    g, r = (d > 0).rolling(N).sum(), (d < 0).rolling(N).sum()
    L, S = g >= k, r >= k
    # 不重疊
    sig = np.where(L, 1, np.where(S, -1, 0))
    keep = np.zeros(len(sig), bool)
    i = N
    while i < len(sig) - 48:
        if sig[i] != 0 and np.isfinite(atr.iloc[i]):
            keep[i] = True
            i += 48
        else:
            i += 1
    P(f"\n## M30 N={N} ≥{k} 根、持有 48 根（做多 / 做空）")
    P(f"  {'年':4s} {'當年漂移(做多48根平均)':>20s} | {'A 第28部分(重疊、去漂移)':>22s} | {'B 重疊、真實漲跌':>18s} | {'C 第31部分(不重疊、扣點差)':>24s} | C 筆數")
    for y in sorted(set(yr)):
        m = yr == y
        A = ((f - drift)[L & m].mean(), (-(f - drift))[S & m].mean())
        B = (f[L & m].mean(), (-f)[S & m].mean())
        C = ((f - cost)[L & m & keep].mean(), (-f - cost)[S & m & keep].mean())
        P(f"  {y} {drift[m].iloc[0]:+20.2f} | {A[0]:+10.2f} / {A[1]:+9.2f} | {B[0]:+8.2f} / {B[1]:+7.2f} | {C[0]:+11.2f} / {C[1]:+10.2f} | {int((L & m & keep).sum())}/{int((S & m & keep).sum())}")
out.close()
