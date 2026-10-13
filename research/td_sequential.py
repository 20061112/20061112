"""TD Sequential（DeMark）標準規則，不調參，2023~2026 全期驗證。

Setup（買進；賣出鏡像）：連續 9 根 close < close[t-4]，第 1 根前一根不成立（price flip）。
  第 9 根收盤即訊號；計數超過 9 不再出訊號，條件中斷才歸零。
  完美（perfected）：第 8 或第 9 根的 low <= 第 6、第 7 根的 low。
Countdown：買進 setup 9 完成後（含第 9 根）開始，計數 close <= low[t-2] 的 K 棒（不需連續），
  第 13 根須 low <= 第 8 個計數棒的 close，否則延後到第一根同時符合的 K 棒。
  反向 setup 完成 → 取消；13 出現後歸零。
交易：反轉方向，下一根開盤進場；停損 = DeMark 風險價位：訊號期間（setup：9 根；countdown：setup 起點到 13）極值那根的極值 −/+ 該根真實波幅；
  停利 1R / 2R，最長 20 根收盤平倉；同一根同時碰停損與停利 → 停損；扣進場 K 棒點差。訊號彼此獨立計算。
另外報告不設停損的 N 根後報酬（ATR 單位，扣點差），看方向本身有沒有預測力。
"""
import numpy as np
import pandas as pd
from load import load


def resample(df, rule):
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "median"}).dropna()


def risk_level(h, l, c, a, t, s):
    """DeMark 風險價位：區間極值那根 K 棒的極值 −/+ 該根真實波幅。"""
    k = a + (np.argmin(l[a:t + 1]) if s == 1 else np.argmax(h[a:t + 1]))
    tr = max(h[k], c[k - 1]) - min(l[k], c[k - 1])
    return l[k] - tr if s == 1 else h[k] + tr


def signals(df):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(c)
    out = []                                    # (t, side, kind, stop)
    cnt = {1: 0, -1: 0}                         # 1 = 買進 setup（下跌計數），-1 = 賣出 setup
    start = {1: 0, -1: 0}
    cd = {1: None, -1: None}                    # countdown 狀態 dict(n, bar8_close, start)
    for t in range(4, n):
        for s in (1, -1):
            cond = c[t] < c[t - 4] if s == 1 else c[t] > c[t - 4]
            if cond:
                if cnt[s] == 0:
                    start[s] = t
                cnt[s] += 1
            else:
                cnt[s] = 0
            if cnt[s] == 9:
                a = start[s]
                stop = risk_level(h, l, c, a, t, s)
                perf = (min(l[t], l[t - 1]) <= min(l[t - 3], l[t - 2])) if s == 1 else (max(h[t], h[t - 1]) >= max(h[t - 3], h[t - 2]))
                out.append((t, s, "setup9", stop))
                if perf:
                    out.append((t, s, "setup9_perfect", stop))
                cd[-s] = None                   # 反向 countdown 取消
                if cd[s] is None:
                    cd[s] = dict(n=0, c8=np.nan, start=a)
        for s in (1, -1):
            st = cd[s]
            if st is None:
                continue
            if t >= 2 and ((c[t] <= l[t - 2]) if s == 1 else (c[t] >= h[t - 2])):
                if st["n"] < 12:
                    st["n"] += 1
                    if st["n"] == 8:
                        st["c8"] = c[t]
                elif (l[t] <= st["c8"]) if s == 1 else (h[t] >= st["c8"]):
                    a = st["start"]
                    stop = risk_level(h, l, c, a, t, s)
                    out.append((t, s, "countdown13", stop))
                    cd[s] = None
    return out


def evaluate(df, sigs, hold=20):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    spr = df.spread.to_numpy(float) * 0.01
    pc = np.r_[c[0], c[:-1]]
    atr = pd.Series(np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))).rolling(14).mean().to_numpy()
    n, rows = len(c), []
    for t, s, kind, stop in sigs:
        ei = t + 1
        if ei + hold >= n or not atr[t] > 0:
            continue
        ep = o[ei]
        R = s * (ep - stop)
        row = dict(time=df.index[t], side=s, kind=kind, R_atr=R / atr[t])
        for k in (1, 4, 8, 16):
            row[f"f{k}"] = (s * (c[ei + k - 1] - ep) - spr[ei]) / atr[t]
        row["gross8"] = s * (c[ei + 7] - ep) / atr[t]
        for tp in (1.0, 2.0):
            if not R > 0:
                row[f"r{tp:g}"] = np.nan
                continue
            tgt, xp = ep + s * tp * R, None
            for j in range(ei, ei + hold):
                if (l[j] <= stop) if s == 1 else (h[j] >= stop):
                    xp = (min(stop, o[j]) if s == 1 else max(stop, o[j])) if j > ei else stop; break
                if (h[j] >= tgt) if s == 1 else (l[j] <= tgt):
                    xp = tgt; break
            if xp is None:
                xp = c[ei + hold - 1]
            row[f"r{tp:g}"] = (s * (xp - ep) - spr[ei]) / R
        rows.append(row)
    return pd.DataFrame(rows)


def tstat(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 1 else np.nan


def report(name, E):
    print(f"\n### {name}")
    for kind in ("setup9", "setup9_perfect", "countdown13"):
        X = E[E.kind == kind]
        if X.empty:
            continue
        fw = "  ".join(f"{k}根 {X[f'f{k}'].mean():+.3f}(t{tstat(X[f'f{k}']):+.1f})" for k in (1, 4, 8, 16))
        r1, r2 = X.r1.dropna(), X.r2.dropna()
        pf = lambda r: r[r > 0].sum() / -r[r < 0].sum()
        yr = X.groupby(X.time.dt.year).r2.mean()
        print(f"  {kind:15s} n={len(X):5d} | 後續報酬(ATR) {fw} | 8根未扣點差 {X.gross8.mean():+.3f}(t{tstat(X.gross8):+.1f})")
        print(f"  {'':15s} 交易 1R: {r1.mean():+.3f}R t{tstat(r1):+.2f} PF {pf(r1):.2f} | 2R: {r2.mean():+.3f}R t{tstat(r2):+.2f} PF {pf(r2):.2f} "
              f"| 停損中位 {X.R_atr.median():.1f} ATR | 2R 逐年 " + " ".join(f"{y}:{v:+.2f}" for y, v in yr.items()))
        for s, nm in ((1, "多"), (-1, "空")):
            Y = X[X.side == s]
            print(f"  {'':15s}   {nm} n={len(Y)} 8根 {Y.f8.mean():+.3f} 2R {Y.r2.mean():+.3f}R")


if __name__ == "__main__":
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    sets = [("M5（2023/6~2026/10）", resample(m1, "5min")), ("M15（2023/1~2026/10）", m15),
            ("H1", resample(m15, "1h")), ("H4", resample(m15, "4h")), ("D1", resample(m15, "1D"))]
    allE = []
    for name, df in sets:
        E = evaluate(df, signals(df))
        report(name, E)
        allE.append(E.assign(tf=name.split("（")[0]))
    pd.concat(allE).to_csv("td_sequential_signals.csv", index=False)
