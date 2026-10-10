"""第四十五部分：把 18 個 IB 配合起來看。

（OW、窗 3h、2R 保本、收盤出場；依進場時間排序）
A. 共振：同一根 K 棒同方向有幾個 IB 一起突破（現行只開 1 單）
B. 當天第幾筆（同方向第 1、2、3… 筆）
C. 進場時，當天已開的單是否浮盈（加碼在賺錢的單上 vs 攤平在虧錢的單上）
D. 當天已有反方向的 OW 單 / 反方向的 OW IB（多空打架）
E. 依上面的結果做分倉 / 篩選：0.01 手最多 5 張的美元、回撤、報酬÷回撤、日 Sharpe
"""
import numpy as np
import pandas as pd
from oos_engine import signals, simulate
from part27_ow import data, pf, PER, prev_profiles, label_ow

out = open("results_part45.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
SRC = {"M1": ("data/XAUUSD_M1_2026.csv", 1), "M5": ("data/XAUUSD_M5_2025_2026.csv", 5), "M15": ("data/XAUUSD_M15_2023_2026.csv", 15)}


def manage(r, ARR, bm):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; be = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            return (min(O[k], st) if s == 1 else max(O[k], st)), t[k] + pd.Timedelta(minutes=bm)
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if not be and s * (best - e) >= 2 * R:
            be = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=bm)


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def grp(T, col, lab, multi, order=None):
    P(f"  [{lab}]")
    keys = order if order is not None else sorted(T[col].dropna().unique())
    for k in keys:
        g = T[T[col] == k]
        if not len(g):
            continue
        cells = ("  " + "  ".join(f"{n} {x.R.mean():+.2f}" for n, a, b in PER for x in [g[(g.day >= a) & (g.day <= b)]] if len(x))) if multi else ""
        P(f"   {str(k):14s} {len(g):5d}筆 勝{(g.R > 0.05).mean() * 100:3.0f}% 每筆{g.R.mean():+.3f}R PF{pf(g.R):.2f} 每筆÷ATR{g.u.mean() * 100:+5.1f}%" + cells)


def money(name, Y, w, alldays):
    Y = Y.assign(v=Y.pnl * w).sort_values("exit_time"); eq = Y.v.cumsum(); dd = (eq - eq.cummax()).min()
    dl = Y.groupby("day").v.sum().reindex(alldays, fill_value=0)
    P(f"   {name:34s} 平均手數{np.mean(w) * 0.01:.3f} 總{Y.v.sum():+7,.0f}美元 回撤{dd:+6,.0f} 報酬÷回撤{Y.v.sum() / -dd:5.1f} Sharpe{dl.mean() / dl.std() * np.sqrt(250):5.2f}")


for tf, (path, bm) in SRC.items():
    days, D, ARR = data(path, bm)
    S = signals(days, D, ARR, bm, monday=True)
    X = simulate(S, ARR, bm)
    X["ow"] = label_ow(X, prev_profiles(days, ARR))
    A = X[X.ow & (X.day >= "2023-03-01")].copy()                       # 去重前（每個 IB 一筆）
    conf = A.groupby(["t_in", "side"]).size().rename("conf")
    T = A.sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).join(conf, on=["t_in", "side"]).reset_index(drop=True)
    ex = [manage(r, ARR, bm) for r in T.itertuples()]
    T["exit_time"] = [b for a, b in ex]; T["pnl"] = T.side * (np.array([a for a, b in ex]) - T.lvl) - T.spread
    T["R"] = T.pnl / T.risk; T["u"] = T.pnl / T.atr
    T = T.sort_values("t_in").reset_index(drop=True)
    order, nopen, flt, opp_open, opp_seen = [], [], [], [], []
    owib = X[X.ow].groupby("day")                                     # 當天所有 OW 的 IB（含沒成交的反方向不算：只有突破的）
    for i, r in T.iterrows():
        prev = T[(T.day == r.day) & (T.t_in < r.t_in)]
        same = prev[prev.side == r.side]
        order.append(len(same) + 1)
        op = same[same.exit_time > r.t_in]
        nopen.append(len(op))
        flt.append((r.side * (r.lvl - op.lvl) / op.risk).sum() if len(op) else np.nan)
        opp_open.append(int(((prev.side != r.side) & (prev.exit_time > r.t_in)).any()))
        opp_seen.append(int((prev.side != r.side).any()))
    T["order"] = order; T["nopen"] = nopen; T["flt"] = flt; T["opp_open"] = opp_open; T["opp_seen"] = opp_seen
    T["conf_b"] = T.conf.clip(upper=4).map({1: "1 個", 2: "2 個", 3: "3 個", 4: "4 個以上"})
    T["order_b"] = pd.cut(T.order, [0, 1, 2, 3, 5, 99], labels=["第1筆", "第2筆", "第3筆", "第4~5筆", "第6筆以後"]).astype(str)
    T["flt_b"] = np.where(T.nopen == 0, "沒有持倉", np.where(T.flt > 1, "已開的單合計浮盈 >1R", np.where(T.flt > 0, "浮盈 0~1R", "浮虧")))
    T["opp_b"] = np.where(T.opp_open == 1, "有反向持倉", np.where(T.opp_seen == 1, "今天做過反向", "今天只有同向"))
    multi = tf == "M15"
    alldays = [d for d in days if d >= T.day.min()]
    P("\n" + "=" * 150 + f"\n[{tf}]  {len(T)} 筆（去重前 {len(A)} 個 IB 訊號）" + ("；右側各期每筆 R" if multi else ""))
    grp(T, "conf_b", "A. 同一根 K 棒幾個 IB 一起突破", multi, ["1 個", "2 個", "3 個", "4 個以上"])
    grp(T, "order_b", "B. 當天同方向第幾筆", multi, ["第1筆", "第2筆", "第3筆", "第4~5筆", "第6筆以後"])
    grp(T, "flt_b", "C. 進場時當天同向已開單的浮動", multi, ["沒有持倉", "已開的單合計浮盈 >1R", "浮盈 0~1R", "浮虧"])
    grp(T, "opp_b", "D. 多空打架", multi, ["今天只有同向", "今天做過反向", "有反向持倉"])
    C5 = cap(T)
    P(f"  E. 分倉 / 篩選（同一組最多 5 張 {len(C5)} 筆；篩選版重新套 5 張）")
    money("全部 0.01（現行）", C5, np.ones(len(C5)), alldays)
    money("共振 ≥2 個 IB → 0.02", C5, np.where(C5.conf >= 2, 2.0, 1.0), alldays)
    money("共振 ≥3 個 IB → 0.02", C5, np.where(C5.conf >= 3, 2.0, 1.0), alldays)
    money("當天第 1 筆 0.02、其餘 0.01", C5, np.where(C5.order == 1, 2.0, 1.0), alldays)
    money("同向已開單浮虧時不加", C5, np.where(C5.flt_b == "浮虧", 0.0, 1.0), alldays)
    money("同向已開單浮盈 >1R 時才加碼（其他第 2 筆起不做）", C5, np.where((C5.nopen == 0) | (C5.flt > 1), 1.0, 0.0), alldays)
    Y = cap(T[T.opp_open == 0])
    money("有反向持倉時不開", Y, np.ones(len(Y)), alldays)
out.close()
