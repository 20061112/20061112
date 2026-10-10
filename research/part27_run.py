"""第二十七部分：OW 的 M5 精確重跑、參數穩健性、停用開關。結果 → results_part27.txt
每格 = 每筆 R / PF（以 R 計）/ 勝率 / 每日筆數。"""
import numpy as np
import pandas as pd
from part27_ow import run, per_cells, pf, PER

M15, M5 = "data/XAUUSD_M15_2023_2026.csv", "data/XAUUSD_M5_2025_2026.csv"
out = open("results_part27.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

P("第二十七部分 A：OW 基準（IB 60 分每 30 分 02:00~10:30、收盤確認 +0.05ATR、IB 另一側停損、收盤出場、含週一）")
X15, d15 = run(M15, 15)
P(f"  M15 全部突破   {per_cells(X15, d15)}")
P(f"  M15 OW         {per_cells(X15[X15.ow], d15)}")
X5, d5 = run(M5, 5)
P5 = [p for p in PER if p[1] >= "2025-07-01"]
P(f"  M5  全部突破   {per_cells(X5, d5, P5)}")
P(f"  M5  OW         {per_cells(X5[X5.ow], d5, P5)}   ← 與 M1 原版確認方式相同")
X5[X5.ow].to_csv("part27_ow_M5.csv", index=False); X15[X15.ow].to_csv("part27_ow_M15.csv", index=False)

P("\n第二十七部分 B：參數穩健性（M15，只改一個參數）")
grid = [("基準", {}), ("「價值區外」改用 IB 中點", dict(mode="mid")), ("「價值區外」改用進場價", dict(mode="lvl")),
        ("價值區 60%", dict(va_pct=0.60)), ("價值區 80%", dict(va_pct=0.80)),
        ("IB 30 分", dict(ib_min=30)), ("IB 90 分", dict(ib_min=90)), ("起點每 60 分", dict(step=60)),
        ("緩衝 0", dict(buf=0.0)), ("緩衝 0.1 ATR", dict(buf=0.10)), ("時段 01~12 點", dict(hours=(1, 12))),
        ("時段 02~08 點", dict(hours=(2, 8))), ("突破窗 2 小時", dict(win=2)), ("突破窗 5 小時", dict(win=5)),
        ("不做週一", dict(monday=False)), ("停損 IB 中點", dict(stop_mode="mid"))]
for lab, kw in grid:
    X, d = run(M15, 15, **kw)
    P(f"  {lab:22s} {per_cells(X[X.ow], d)}")

P("\n第二十七部分 C：停用開關（M15 OW，影子交易：暫停期間照常記錄但不下單）")
T = X15[X15.ow].sort_values("t_in").reset_index(drop=True)


def kill(T, n, th=1.0, min_n=None):
    taken, closed, hist = [], [], []
    for r in T.itertuples():
        done = [c for c in closed if c[0] <= r.t_in]; closed = [c for c in closed if c[0] > r.t_in]
        hist += [c[1] for c in sorted(done)]
        last = np.array(hist[-n:])
        on = len(last) < (min_n or n) or pf(last) >= th
        closed.append((r.exit_time, r.R))
        if on: taken.append(r.Index)
    return T.loc[taken]


P(f"  不用開關       {per_cells(T, d15)} | 全 {T.R.mean():+.3f}R 總 {T.R.sum():+.0f}R")
for n in (50, 100, 150):
    for th in (1.0, 1.2):
        K = kill(T, n, th)
        P(f"  最近 {n:3d} 筆 PF < {th} 暫停 {per_cells(K, d15)} | 全 {K.R.mean():+.3f}R 總 {K.R.sum():+.0f}R 筆數 {len(K)}/{len(T)}")
out.close()
