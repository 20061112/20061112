"""TD Sequential（狄馬克 9 轉 13 展）訊號。

Setup（9 轉）
  買進 setup：連續 9 根 close < 4 根前 close（中斷就歸零；第 1 根前一根不符合 = price flip）
  賣出 setup：鏡像（close > 4 根前 close）
  完美化（perfected）：買進為第 8 或第 9 根的 low <= 第 6、7 根的 low（兩者皆是）；賣出鏡像
  TDST：買進 setup 9 根中的最高 high（賣出為最低 low）

Countdown（13 展）
  買進 setup 完成那一根起算（含第 9 根），每根 close <= 2 根前 low 記一次（不必連續）
  第 13 次另需 low <= 第 8 次那根的 close，否則延後（之後符合的 K 棒才算 13）
  取消：出現反向 setup 9、或收盤突破 TDST（買進：close > setup 最高 high）
  已有進行中的同向 countdown 時，新的同向 setup 不重啟（DeMark 的 recycle 規則不實作）

所有訊號都只用到該 K 棒收盤（含）以前的資料。
"""
import numpy as np
import pandas as pd


def resample(m1, rule):
    if rule == "1min":
        return m1[["open", "high", "low", "close", "spread"]].copy()
    return m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "max"}).dropna()


def atr(df, n=14):
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().to_numpy()


def td_signals(df):
    """回傳 list of dict：每個 setup 9 與 countdown 13 一筆。

    dir = +1 代表買進訊號（下跌 setup → 預期反彈），-1 為賣出訊號。
    """
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(c)
    bs = ss = 0                        # 目前 setup 連續計數
    cd = {1: None, -1: None}           # 進行中的 countdown 狀態
    out = []
    for t in range(4, n):
        bs = bs + 1 if c[t] < c[t - 4] else 0
        ss = ss + 1 if c[t] > c[t - 4] else 0
        for d, cnt in ((1, bs), (-1, ss)):
            if cnt != 9:
                continue
            w = slice(t - 8, t + 1)
            if d == 1:
                perf = min(l[t], l[t - 1]) <= min(l[t - 2], l[t - 3])
                ext, tdst = l[w].min(), h[w].max()
            else:
                perf = max(h[t], h[t - 1]) >= max(h[t - 2], h[t - 3])
                ext, tdst = h[w].max(), l[w].min()
            out.append(dict(i=t, kind="S9", dir=d, perf=perf, ext=ext, tdst=tdst))
            cd[-d] = None              # 反向 setup 完成 → 取消反向 countdown
            if cd[d] is None:
                cd[d] = dict(start=t, n=0, c8=None, tdst=tdst, ext=ext)
        for d in (1, -1):
            s = cd[d]
            if s is None:
                continue
            if t > s["start"] and (c[t] > s["tdst"] if d == 1 else c[t] < s["tdst"]):
                cd[d] = None
                continue
            s["ext"] = min(s["ext"], l[t]) if d == 1 else max(s["ext"], h[t])
            if t < 2:
                continue
            hit = c[t] <= l[t - 2] if d == 1 else c[t] >= h[t - 2]
            if not hit:
                continue
            if s["n"] < 12:
                s["n"] += 1
                if s["n"] == 8:
                    s["c8"] = c[t]
                continue
            qual = l[t] <= s["c8"] if d == 1 else h[t] >= s["c8"]
            if qual:
                out.append(dict(i=t, kind="C13", dir=d, perf=True, ext=s["ext"], tdst=s["tdst"],
                                bars=t - s["start"]))
                cd[d] = None
    return pd.DataFrame(out)
