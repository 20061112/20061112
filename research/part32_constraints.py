"""第三十二部分：OW 的持倉管理限制 —— 時間停損、最長持倉、移動止損、套保（部分平倉）。

進場不變（FINAL_STRATEGY_OW），只改進場後的管理。R = 進場到初始停損（IB 另一側）的距離。
  time_np (分, x)  進場 N 分鐘時，最大浮盈 < x R → 在那根收盤出場（「多久沒浮盈」）
  max_h  (小時)     最多持有 H 小時，到時收盤出場
  steps  [(觸發R, 新停損R)]  浮盈達觸發 R 後，停損移到 進場 + 新停損R（階梯式移動止損）
  trail  (啟動R, 距離R)     浮盈達啟動 R 後，停損 = 最高（低）點 − 距離R
  half   (k)              k R 時平一半（套保），剩下一半停損移到成本
同一根 K 棒同時碰到停損與其他條件 → 先算停損（保守）；移動停損下一根才生效；時間條件在 K 棒收盤判斷。
資料：M15 OW 2023/3~2026/10（part27_ow_M15.csv）、M1 OW 2026（part31_m1_trades.csv）。
"""
import numpy as np
import pandas as pd
from part27_ow import data, pf

out = open("results_part32.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()


def manage(r, ARR, bm, time_np=None, max_h=None, steps=None, trail=None, half=None):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st)
    if R <= 0:
        return np.nan, 0.0
    best = e; parts = [1.0] if half is None else [0.5, 0.5]; res = [None] * len(parts)
    k_end = len(C) - 1
    for k in range(r.i, len(C)):
        el = (k - r.i + 1) * bm                      # 這根收盤時已持有的分鐘
        hi, lo, op = H[k], L[k], O[k]
        if (lo <= st) if s == 1 else (hi >= st):
            px = min(op, st) if s == 1 else max(op, st)
            res = [x if x is not None else px for x in res]; k_end = k; break
        if half is not None and res[0] is None:
            tg = e + s * half * R
            if (hi >= tg) if s == 1 else (lo <= tg):
                res[0] = max(op, tg) if s == 1 else min(op, tg)
                st = max(st, e) if s == 1 else min(st, e)
        best = max(best, hi) if s == 1 else min(best, lo)
        fav = s * (best - e) / R
        if time_np is not None and el >= time_np[0] and el - bm < time_np[0] and fav < time_np[1]:
            res = [x if x is not None else C[k] for x in res]; k_end = k; break
        if max_h is not None and el >= max_h * 60:
            res = [x if x is not None else C[k] for x in res]; k_end = k; break
        if steps:
            for trig, new in steps:
                if fav >= trig:
                    ns = e + s * new * R
                    st = max(st, ns) if s == 1 else min(st, ns)
        if trail is not None and fav >= trail[0]:
            ns = best - s * trail[1] * R
            st = max(st, ns) if s == 1 else min(st, ns)
    res = [x if x is not None else C[-1] for x in res]
    pnl = sum(w * s * (px - e) for w, px in zip(parts, res)) - r.spread
    return pnl / R, (k_end - r.i + 1) * bm / 60


RULES = [("現行（收盤出場）", {}),
         ("30 分沒浮盈 0.5R 就出", dict(time_np=(30, 0.5))), ("60 分沒浮盈 0.5R 就出", dict(time_np=(60, 0.5))),
         ("120 分沒浮盈 0.5R 就出", dict(time_np=(120, 0.5))), ("120 分沒浮盈 1R 就出", dict(time_np=(120, 1.0))),
         ("240 分沒浮盈 1R 就出", dict(time_np=(240, 1.0))),
         ("最多持有 2 小時", dict(max_h=2)), ("最多持有 4 小時", dict(max_h=4)), ("最多持有 6 小時", dict(max_h=6)),
         ("最多持有 8 小時", dict(max_h=8)),
         ("階梯：1R→−0.5R", dict(steps=[(1, -0.5)])), ("階梯：1R→−0.5R、2R→0", dict(steps=[(1, -0.5), (2, 0)])),
         ("階梯：2R→0、3R→+1R", dict(steps=[(2, 0), (3, 1)])), ("階梯：2R→+0.5R、4R→+2R", dict(steps=[(2, 0.5), (4, 2)])),
         ("移動：2R 啟動、距 1.5R", dict(trail=(2, 1.5))), ("移動：2R 啟動、距 2R", dict(trail=(2, 2))),
         ("移動：3R 啟動、距 2R", dict(trail=(3, 2))),
         ("套保：2R 平一半 + 保本", dict(half=2)), ("套保：3R 平一半 + 保本", dict(half=3)),
         ("組合：120分沒0.5R出 + 2R→0", dict(time_np=(120, 0.5), steps=[(2, 0)])),
         ("組合：2R 平一半 + 移動 3R/2R", dict(half=2, trail=(3, 2)))]

PER = [("2023", "2023-03-01", "2023-12-31"), ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
       ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]

for lab, csv, src, per in [("M15 OW（2023~2026）", "part27_ow_M15.csv", ("data/XAUUSD_M15_2023_2026.csv", 15), PER),
                           ("M1 OW（2026）", "part31_m1_trades.csv", ("data/XAUUSD_M1_2026.csv", 1), [("2026", "2026-01-01", "2026-12-31")])]:
    X = pd.read_csv(csv, parse_dates=["day", "t_in", "exit_time"])
    X = X[X.day >= "2023-03-01"]
    days, D, ARR = data(*src)
    P("\n" + "=" * 140 + f"\n[{lab}]  每格 = 每筆 R / PF / 勝率；右側 = 全期 每筆R、勝率、平均持倉、最長連虧")
    base = None
    for name, kw in RULES:
        rh = [manage(r, ARR, src[1], **kw) for r in X.itertuples()]
        Y = X.assign(r=[a for a, b in rh], hold=[b for a, b in rh]).dropna(subset=["r"])
        cells = [f"{n} {x.r.mean():+.2f}/{pf(x.r):.2f}/{(x.r > 0).mean() * 100:.0f}%" for n, a, b in per for x in [Y[(Y.day >= a) & (Y.day <= b)]] if len(x)]
        lose = (Y.r <= 0).astype(int)
        P(f"  {name:28s} " + "  ".join(cells) +
          f" | 全 {Y.r.mean():+.3f}R 勝{(Y.r > 0).mean() * 100:.0f}% 持倉{Y.hold.mean():4.1f}h 連虧{lose.groupby((lose != lose.shift()).cumsum()).sum().max()}")
out.close()
