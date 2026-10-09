"""第二十部分 B：針對弱點的事前規則測試（版本 A，滾動前推 3~10 月）。
 a. 紐約數據前（15:25 broker = 08:25 ET）處理持倉：浮動損益 < x R 的單先平倉
 b. 單日停損上限：當天已有 N 筆停損 → 當天不再開新單
 c. 前一天單邊（效率 > 門檻，前一天就知道）→ 當天不做
"""
import numpy as np, pandas as pd
from part14_deep import ARR, D, DAYS
from part17_balance import gen, score
from final_A import CFG, walk_forward
import part17_risk as RK

out = open("results_part20.txt", "a")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
T, _ = walk_forward(gen(**CFG)); T = T.sort_values("t_in").reset_index(drop=True)

def fmt(X):
    s = score(X); ret, dd = (lambda e: (e[0], e[1]))(sizing(X))
    return (f"{len(X):4d}筆 每筆 {s['avg']:+6.2f}（{X.R.mean():+.2f}R） PF {s['pf']:.2f} 總 {s['total']:+6.0f} DD {s['dd']:5.0f}"
            f" 總/DD {s['ret_dd']:4.1f} Sharpe {s['sharpe']:.2f} 去前10天 {s['ex10']:+5.0f} | 0.5%/筆 {ret:+4.0f}% DD {dd:5.1f}%")

def sizing(X, r=0.5):
    X = X.sort_values("exit_time"); eq = 1e4 + (1e4 * r / 100 / X.risk * X.pnl).cumsum()
    return (eq.iloc[-1] / 1e4 - 1) * 100, ((eq - eq.cummax()) / eq.cummax()).min() * 100

P("\n" + "=" * 110 + "\n第二十部分 B：針對弱點的事前規則")
P(f"  基準                         {fmt(T)}")

def pre_ny(T, x, hh=15, mm=25):
    Y = T.copy(); pn, et = [], []
    for r in Y.itertuples():
        t, O, H, L, C, S = ARR[r.day]; k = np.searchsorted(t, r.day + pd.Timedelta(hours=hh, minutes=mm))
        if r.t_in < t[min(k, len(t) - 1)] < r.exit_time and k < len(C):
            u = r.side * (C[k] - r.lvl)
            if u < x * r.risk:
                pn.append(u - r.spread); et.append(t[k]); continue
        pn.append(r.pnl); et.append(r.exit_time)
    Y["pnl"] = pn; Y["exit_time"] = et; Y["R"] = Y.pnl / Y.risk
    return Y

P("\n a) 15:25（美國數據 / 紐約開盤前）浮動損益 < x R 的單先平倉")
for x in (-0.5, 0.0, 0.5, 1.0):
    P(f"  x = {x:+.1f}R                     {fmt(pre_ny(T, x))}")

def daily_stop_cap(T, N):
    keep = []
    for d, g in T.groupby("day"):
        stops = []
        for r in g.sort_values("t_in").itertuples():
            n_stopped = sum(1 for (et, why) in stops if et <= r.t_in and why == "停損")
            if n_stopped >= N:
                continue
            keep.append(r.Index); stops.append((r.exit_time, r.exit_reason))
    return T.loc[keep]

P("\n b) 當天停損達 N 筆後不再開新單")
for N in (2, 3, 4, 5):
    P(f"  N = {N}                        {fmt(daily_stop_cap(T, N))}")

P("\n c) 前一天效率（|收−開|/波幅）> 門檻 → 當天不做（門檻 = 過去所有天的分位，逐月更新）")
D["eff"] = (D.close - D.open).abs() / (D.high - D.low)
D["prev_eff"] = D.eff.shift(1)
T["prev_eff"] = D.prev_eff.reindex(T.day).values
for q in (0.75, 0.9):
    parts = []
    for m, cur in T.groupby(T.day.dt.to_period("M")):
        thr = D.eff[D.index < m.start_time].quantile(q)
        parts.append(cur[cur.prev_eff <= thr])
    P(f"  去掉前一天效率最高 {int((1 - q) * 100)}%       {fmt(pd.concat(parts))}")
out.close()
