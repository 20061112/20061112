"""第二十八部分補充：網格中看起來穩健的區域，逐年檢查（漂移中性，ATR）。"""
import numpy as np
import pandas as pd
from load import load

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}
CASES = [  # 時框, 分鐘, N, k, h, 預期方向（+1 延續 / −1 反轉）
    ("M30", 30, 15, 12, 24, 1), ("M30", 30, 20, 14, 24, 1), ("M30", 30, 25, 15, 48, 1), ("M30", 30, 30, 18, 48, 1),
    ("H1", 60, 15, 11, 6, 1), ("H1", 60, 30, 18, 24, 1),
    ("M5", 5, 20, 17, 24, -1), ("M5", 5, 30, 23, 24, -1), ("M5", 5, 25, 20, 48, -1),
    ("M15", 15, 30, 18, 48, 1),
]
m1 = load("data/XAUUSD_M1_2023_2026.csv")
cache = {}
out = open("results_part28_robust.txt", "w")
out.write("穩健區域逐年檢查：每格 = 順預期方向的漂移中性報酬（ATR）多邊/空邊；最後 = 每年兩邊都 > 0 的年數\n")
for tf, mins, N, k, h, sgn in CASES:
    if tf not in cache:
        cache[tf] = m1.resample(f"{mins}min").agg(AGG).dropna()
    df = cache[tf]
    d = np.sign(df.close - df.open)
    atr = (df.high - df.low).rolling(20).mean()
    g, r = (d > 0).rolling(N).sum(), (d < 0).rolling(N).sum()
    f = (df.close.shift(-h) - df.close) / atr
    yr = df.index.year
    x = f - f.groupby(yr).transform("mean")               # 逐年去漂移
    cost = df.spread * 0.01 / atr
    cells, ok = [], 0
    for y in sorted(set(yr)):
        m = yr == y
        L = (sgn * x[(g >= k) & m]).mean()
        S = (-sgn * x[(r >= k) & m]).mean()
        c = cost[m].mean()
        good = L > c and S > c
        ok += good
        cells.append(f"{y}: {L:+.2f}/{S:+.2f}{'✓' if good else ' '}")
    line = f"{tf:3s} N={N:3d} ≥{k:2d}根({k / N:.0%}) 持有{h:2d} {'延續' if sgn > 0 else '反轉'} | " + "  ".join(cells) + f" | 兩邊都 > 點差的年數 {ok}/4"
    print(line)
    out.write(line + "\n")
out.close()
