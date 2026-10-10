"""第三十五部分：鎖利 —— 讓「曾經大賺」的單不要變成虧損。

單筆規則（R = 初始停損距離；浮盈以最高 / 最低點計，停損下一根 K 棒才生效）：
  ladder  [(觸發R, 鎖住R)]   浮盈達觸發 R → 停損至少移到 進場 + 鎖住R
  pct     (觸發R, 比例)       浮盈達觸發 R 後，停損 = 進場 + 比例 × 最大浮盈（隨浮盈上升）
整籃規則（同一天所有持倉）：
  basket  (門檻R, 回吐比例)   當天持倉的「未實現 R 合計」曾達門檻，之後跌到 最高 ×(1−回吐比例) 以下 → 全部平倉（該根收盤）
統計：每筆 R、PF、勝率、總 R，以及「浮盈曾 ≥ 2R 卻以虧損收場」的筆數與合計。
資料：M1 OW 2026（part31_m1_trades.csv）、M15 OW 2023/3~2026/10（part27_ow_M15.csv）。
"""
import numpy as np
import pandas as pd
from part27_ow import data, pf

out = open("results_part35.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()


def path(r, ARR):
    """回傳這筆單逐根的 (時間, 收盤未實現R, 最高浮盈R, 是否碰停損價) 所需陣列。"""
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    R = s * (e - (r.ibl if s == 1 else r.ibh))
    return t, O, H, L, C, s, e, R


def run_trade(r, ARR, ladder=None, pct=None, cut_at=None):
    """單筆管理；cut_at = 整籃強制平倉的 K 棒索引（None = 無）。回傳 (R, 最大浮盈R, 出場索引)。"""
    t, O, H, L, C, s, e, R = path(r, ARR)
    st = r.ibl if s == 1 else r.ibh
    best = e; mfe = 0.0
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            px = min(O[k], st) if s == 1 else max(O[k], st)
            return (s * (px - e) - r.spread) / R, mfe, k
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        mfe = s * (best - e) / R
        if cut_at is not None and k >= cut_at:
            return (s * (C[k] - e) - r.spread) / R, mfe, k
        if ladder:
            for trig, lock in ladder:
                if mfe >= trig:
                    ns = e + s * lock * R
                    st = max(st, ns) if s == 1 else min(st, ns)
        if pct and mfe >= pct[0]:
            ns = e + s * pct[1] * mfe * R
            st = max(st, ns) if s == 1 else min(st, ns)
    return (s * (C[-1] - e) - r.spread) / R, mfe, len(C) - 1


def run_set(T, ARR, ladder=None, pct=None, basket=None):
    res = {}
    for day, g in T.groupby("day"):
        cut = {}
        if basket:
            # 先用單筆規則跑出每筆的出場點，再逐根算當天未實現 R 合計
            first = {r.Index: run_trade(r, ARR, ladder, pct) for r in g.itertuples()}
            t, O, H, L, C, S, mins = ARR[day]
            n = len(C); tot = np.zeros(n); active = np.zeros(n, bool)
            for r in g.itertuples():
                _, _, kx = first[r.Index]
                s, e = r.side, r.lvl; R = s * (e - (r.ibl if s == 1 else r.ibh))
                tot[r.i:kx] += s * (C[r.i:kx] - e) / R; active[r.i:kx] = True
            peak = -np.inf; fired = None
            for k in range(n):
                if not active[k]:
                    continue
                peak = max(peak, tot[k])
                if peak >= basket[0] and tot[k] <= peak * (1 - basket[1]):
                    fired = k; break
            if fired is not None:
                cut = {r.Index: fired for r in g.itertuples() if r.i <= fired}
        for r in g.itertuples():
            res[r.Index] = run_trade(r, ARR, ladder, pct, cut.get(r.Index))
    Y = T.copy()
    Y["r"] = [res[i][0] for i in Y.index]; Y["mfe"] = [res[i][1] for i in Y.index]
    return Y


RULES = [("現行", {}),
         ("階梯 2R→鎖 0.5R", dict(ladder=[(2, 0.5)])),
         ("階梯 2R→0.5R、3R→1R、4R→2R", dict(ladder=[(2, 0.5), (3, 1), (4, 2)])),
         ("階梯 1.5R→0.25R、2R→0.5R、3R→1.5R", dict(ladder=[(1.5, 0.25), (2, 0.5), (3, 1.5)])),
         ("鎖最大浮盈 30%（2R 起）", dict(pct=(2, 0.3))), ("鎖最大浮盈 50%（2R 起）", dict(pct=(2, 0.5))),
         ("鎖最大浮盈 50%（3R 起）", dict(pct=(3, 0.5))),
         ("組合：2R→0.5R + 3R 起鎖 50%", dict(ladder=[(2, 0.5)], pct=(3, 0.5))),
         ("整籃：曾達 6R、回吐 50% 全平", dict(basket=(6, 0.5))), ("整籃：曾達 10R、回吐 40% 全平", dict(basket=(10, 0.4))),
         ("階梯(2R→0.5R…) + 整籃 6R/50%", dict(ladder=[(2, 0.5), (3, 1), (4, 2)], basket=(6, 0.5)))]
PER = [("2023", "2023-03-01", "2023-12-31"), ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
       ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]

for lab, csv, src, per in [("M1 OW 2026", "part31_m1_trades.csv", ("data/XAUUSD_M1_2026.csv", 1), PER[-1:]),
                           ("M15 OW 2023~2026", "part27_ow_M15.csv", ("data/XAUUSD_M15_2023_2026.csv", 15), PER)]:
    T = pd.read_csv(csv, parse_dates=["day", "t_in", "exit_time"]).sort_values("t_in").reset_index(drop=True)
    T = T[T.day >= "2023-03-01"]
    days, D, ARR = data(*src)
    P("\n" + "=" * 150 + f"\n[{lab}]  各期 每筆R/PF；右側：全期 每筆R、PF、勝率、總R、「浮盈曾≥2R 卻虧損」筆數（合計R）、每筆 0.01 手總美元")
    for name, kw in RULES:
        Y = run_set(T, ARR, **kw)
        bad = Y[(Y.mfe >= 2) & (Y.r <= 0)]
        cells = [f"{n} {x.r.mean():+.2f}/{pf(x.r):.2f}" for n, a, b in per for x in [Y[(Y.day >= a) & (Y.day <= b)]] if len(x)]
        P(f"  {name:30s} " + "  ".join(cells) +
          f" | 全 {Y.r.mean():+.3f}R PF{pf(Y.r):.2f} 勝{(Y.r > 0).mean() * 100:.0f}% 總{Y.r.sum():+5.0f}R | 大賺變虧 {len(bad):3d}筆({bad.r.sum():+.0f}R) | 0.01手 {(Y.r * Y.risk).sum():+6.0f}美元")
out.close()
