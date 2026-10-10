"""第二十三部分：不靠價格 / ATR 標準化的「狀態」特徵 —— 什麼會影響勝率？

基準交易：M15 資料 2023/3~2026/10，IB 60 分每 30 分（02:00~10:30）、M15 收盤確認、IB 另一側停損、收盤出場，
不加任何篩選、含星期一（樣本最大），同一根同方向只算一筆。勝 = 收盤或停損後損益 > 0。
特徵全部是進場當下可知、且不需要價格水準或 ATR 的量：
  時間：wd 星期、ib_h IB 起點、delay IB 結束到突破的分鐘、k_today 今天第幾個訊號
  今天的進度：n_same / n_opp 今天之前已出現的同向 / 反向訊號數、n_stopped 進場時今天已被停損的單數
  位置（0~1 比例）：pos_day 進場價在今天區間的位置（1 = 在交易方向的極端）、ib_pos IB 中點在今天區間的位置
  字母結構：letters 今天已有幾個 15 分字母、otf 最近連續單向延伸的字母數、rf 輪動因子 / 字母數、er_day 今天的效率比
  昨天：aligned 突破方向 = 昨天方向、prev_eff 昨天效率（|收−開|/區間）、prev_cl 昨天收在區間的位置（順交易方向）、
        open_vs_prev 今天開盤在昨天區間外（順 / 逆交易方向）或內、streak 昨天以前連續同方向收盤的天數（順交易方向為正）
期間：2023（3~12）、2024、2025H1、2025H2、2026；只看各期方向一致的特徵。
"""
import numpy as np
import pandas as pd
from oos_engine import load_bars, signals, simulate

out = open("results_part23.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()

days, D, ARR = load_bars("data/XAUUSD_M15_2023_2026.csv", 15)
X = simulate(signals(days, D, ARR, 15, monday=True), ARR, 15)
X = X[X.day >= "2023-03-01"].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).reset_index(drop=True)
D["eff"] = (D.close - D.open).abs() / (D.high - D.low)
D["dir"] = np.sign(D.close - D.open)
D["cl"] = (D.close - D.low) / (D.high - D.low)
didx = {d: i for i, d in enumerate(days)}

feats = []
for day, g in X.groupby("day"):
    t, O, H, L, C, S, mins = ARR[day]
    di = didx[day]; prev = D.loc[days[di - 1]]
    hist = D.dir.iloc[:di]
    for r in g.itertuples():
        i = r.i - 1                                       # 確認那根
        side = r.side
        hi, lo = H[:i + 1].max(), L[:i + 1].min(); rg = hi - lo
        pos = (C[i] - lo) / rg if rg > 0 else 0.5
        ibm = ((r.ibh + r.ibl) / 2 - lo) / rg if rg > 0 else 0.5
        per = mins[:i + 1] // 15; ks = np.unique(per)
        ph = np.array([H[:i + 1][per == k].max() for k in ks]); pl = np.array([L[:i + 1][per == k].min() for k in ks])
        rf = (np.sign(np.diff(ph)).sum() + np.sign(np.diff(pl)).sum()) / max(len(ks) - 1, 1)
        otf = 0
        for k in range(len(ks) - 1, 0, -1):
            if (pl[k] > pl[k - 1]) if side == 1 else (ph[k] < ph[k - 1]): otf += 1
            else: break
        path = np.abs(np.diff(C[:i + 1])).sum()
        earlier = g[g.t_in < r.t_in]
        n_stop = int(((earlier.exit_reason == "停損") & (earlier.exit_time <= r.t_in)).sum())
        o_prev = 2 if O[0] > prev.high else (-2 if O[0] < prev.low else 0)
        st = 0
        for v in hist.values[::-1]:
            if v == side: st += 1
            else: break
        if st == 0:
            for v in hist.values[::-1]:
                if v == -side: st -= 1
                else: break
        feats.append(dict(idx=r.Index, wd=day.dayofweek, ib_h=int(r.h), delay=int((r.t_in - day).total_seconds() // 60 - r.h * 60 - 60),
                          k_today=len(earlier) + 1, n_same=int((earlier.side == side).sum()), n_opp=int((earlier.side != side).sum()),
                          n_stopped=n_stop, pos_day=pos if side == 1 else 1 - pos, ib_pos=ibm if side == 1 else 1 - ibm,
                          letters=len(ks), otf=otf, rf=side * rf, er_day=abs(C[i] - O[0]) / path if path > 0 else 0,
                          aligned=int(r.aligned), prev_eff=prev.eff, prev_cl=prev.cl if side == 1 else 1 - prev.cl,
                          open_vs_prev=side * o_prev // 2, streak=st))
F = pd.DataFrame(feats).set_index("idx")
X = X.join(F, rsuffix="_f")
X["win"] = (X.pnl > 0).astype(int)
X.to_csv("part23_trades_states.csv", index=False)

PER = [("2023", "2023-03-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"), ("25H1", "2025-01-01", "2025-06-30"),
       ("25H2", "2025-07-01", "2025-12-31"), ("2026", "2026-01-01", "2026-12-31")]
X["per"] = None
for n, a, b in PER:
    X.loc[(X.day >= a) & (X.day <= b), "per"] = n
P("第二十三部分：狀態特徵與勝率（M15，基準不篩選、含週一，2023/3~2026/10）")
P("  基準勝率 / 每筆R： " + "  ".join(f"{n} {X[X.per == n].win.mean() * 100:.0f}%/{X[X.per == n].R.mean():+.2f}（{(X.per == n).sum()}筆）" for n, *_ in PER))

BINS = {"wd": None, "ib_h": [1, 3, 5, 7, 9, 11], "delay": [-1, 15, 45, 90, 240], "k_today": [0, 1, 3, 6, 99],
        "n_same": [-1, 0, 2, 5, 99], "n_opp": [-1, 0, 1, 3, 99], "n_stopped": [-1, 0, 1, 3, 99],
        "pos_day": [-.01, .6, .8, .95, 1.01], "ib_pos": [-.01, .3, .5, .7, 1.01], "letters": [0, 12, 20, 28, 99],
        "otf": [-1, 0, 1, 3, 99], "rf": [-9, -0.2, 0.2, 0.5, 9], "er_day": [-.01, .15, .3, .45, 1.01],
        "aligned": None, "prev_eff": [-.01, .25, .5, .75, 1.01], "prev_cl": [-.01, .3, .6, .85, 1.01],
        "open_vs_prev": None, "streak": [-99, -2, -1, 0, 1, 2, 99]}
summary = []
for f, b in BINS.items():
    g = X[f] if b is None else pd.cut(X[f], b)
    P(f"\n  {f}")
    tab = X.groupby([g, "per"], observed=True).win.agg(["mean", "size"]).unstack("per")
    rr = X.groupby([g, "per"], observed=True).R.mean().unstack("per")
    for k in tab.index:
        cells = []
        for n, *_ in PER:
            m, s = tab.loc[k, ("mean", n)], tab.loc[k, ("size", n)]
            cells.append(f"{n} {m * 100:3.0f}%/{rr.loc[k, n]:+.2f}({int(s) if s == s else 0})" if s == s and s > 0 else f"{n}   -   ")
        P(f"    {str(k):14s} " + "  ".join(cells))
    # 一致性：每期「最好組 − 最差組」勝率差的方向是否相同（用全期排序決定最好 / 最差組）
    allw = X.groupby(g, observed=True).win.mean(); cnt = X.groupby(g, observed=True).size()
    allw = allw[cnt >= 80]
    if len(allw) >= 2:
        best, worst = allw.idxmax(), allw.idxmin()
        diffs = [tab.loc[best, ("mean", n)] - tab.loc[worst, ("mean", n)] for n, *_ in PER]
        summary.append((f, str(best), str(worst), diffs))
P("\n" + "=" * 100 + "\n一致性總表：全期最好組 vs 最差組，各期勝率差（百分點；五期都 > 0 才算穩定）")
for f, b, w, d in sorted(summary, key=lambda x: -min(x[3])):
    flag = "★ 五期一致" if min(d) > 0 else ""
    P(f"  {f:13s} 最好 {b:14s} 最差 {w:14s} " + "  ".join(f"{n} {v * 100:+5.1f}" for (n, *_), v in zip(PER, d)) + f"  {flag}")
out.close()
