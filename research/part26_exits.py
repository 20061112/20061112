"""第二十六部分：停利 / 保本 / 移動停損 / 部分停利 —— 能否提高勝率、又不犧牲期望值？

交易：part24_bo_tpo.csv（M15，2023/3~2026/10，IB 突破 M15 收盤確認、IB 另一側停損）。
  兩組：BO（不篩）、OW（只做 IB 在前一天價值區外、順交易方向）。
出場方式（R = 進場到初始停損的距離）：
  base      不設停利，當天收盤出場（現行）
  tp k      固定停利 k×R，沒到就收盤出場
  half k    一半在 k×R 停利，另一半抱到收盤；觸發後全部停損移到成本
  be k      浮盈達 k×R 後停損移到成本
  trail k   浮盈達 1R 後啟動移動停損，距離最高（低）點 k×R
同一根 M15 內同時碰到停損與停利 → 一律先算停損（保守）；保本 / 移動停損從下一根才生效。
"""
import numpy as np
import pandas as pd
from oos_engine import load_bars

out = open("results_part26.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

days, D, ARR = load_bars("data/XAUUSD_M15_2023_2026.csv", 15)
X = pd.read_csv("part24_bo_tpo.csv", parse_dates=["day", "t_in", "exit_time"])
PER = [("2023", "2023-03-01", "2023-12-31"), ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
       ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]


def exit_one(r, mode, k):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st0 = r.ibl if s == 1 else r.ibh
    R = s * (e - st0)
    st = st0; best = e
    units = [1.0] if mode != "half" else [0.5, 0.5]
    res = [None] * len(units)
    armed = False
    for kk in range(r.i, len(C)):
        hi, lo, op = H[kk], L[kk], O[kk]
        # 1. 停損（含已移動的）
        if (lo <= st) if s == 1 else (hi >= st):
            px = min(op, st) if s == 1 else max(op, st)
            for u in range(len(units)):
                if res[u] is None: res[u] = px
            break
        # 2. 停利
        if mode in ("tp", "half"):
            tgt = e + s * k * R
            if (hi >= tgt) if s == 1 else (lo <= tgt):
                px = max(op, tgt) if s == 1 else min(op, tgt)
                if mode == "tp":
                    res[0] = px; break
                if res[0] is None:
                    res[0] = px; st = e            # 剩下一半移到成本
        # 3. 下一根才生效的保本 / 移動停損
        best = max(best, hi) if s == 1 else min(best, lo)
        fav = s * (best - e)
        if mode == "be" and fav >= k * R:
            st = max(st, e) if s == 1 else min(st, e)
        if mode == "trail" and fav >= R:
            nst = best - s * k * R
            st = max(st, nst) if s == 1 else min(st, nst)
    for u in range(len(units)):
        if res[u] is None: res[u] = C[-1]
    pnl = sum(w * s * (px - e) for w, px in zip(units, res)) - r.spread
    return pnl / R


MODES = [("base", None, "現行：收盤出場"), ("tp", 1.0, "停利 1R"), ("tp", 1.5, "停利 1.5R"), ("tp", 2.0, "停利 2R"),
         ("tp", 3.0, "停利 3R"), ("half", 1.0, "1R 平一半 + 保本"), ("half", 2.0, "2R 平一半 + 保本"),
         ("be", 1.0, "1R 後保本"), ("be", 2.0, "2R 後保本"), ("trail", 1.0, "1R 後移動停損（距 1R）"), ("trail", 2.0, "1R 後移動停損（距 2R）")]

for grp, sel in [("BO 不篩", X.R == X.R), ("OW IB 在前日價值區外順勢", X.ib_va == "out_with")]:
    Y = X[sel].copy()
    P("\n" + "=" * 130 + f"\n[{grp}]  每格 = 勝率 / 每筆 R / PF（PF 以 R 計）")
    for mode, k, lab in MODES:
        Y["r"] = [exit_one(r, mode, k if k else 0) for r in Y.itertuples()]
        cells = []
        for n, a, b in PER:
            x = Y[(Y.day >= a) & (Y.day <= b)].r
            cells.append(f"{n} {(x > 0).mean() * 100:3.0f}%/{x.mean():+.2f}/{x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-9):.2f}")
        a = Y.r
        P(f"  {lab:22s} " + "  ".join(cells) + f" | 全 {(a > 0).mean() * 100:3.0f}%/{a.mean():+.3f}R")
out.close()
