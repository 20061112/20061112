"""第四十七部分：新的 M1 資料（2024/9/2 ~ 2026/2/11，與原 2026 檔合併到 2026/10/8）驗證最終規則與候選改良。

M1 先前只看過 2026/1~10；2024/9 ~ 2025/12 是 M1 第一次見到的資料（但 M15 已看過同期間）。
A. 最終版（OW + 2R 保本 + 依突破快慢 0.03/0.02/0.01 手 + 最多 5 筆）逐期；對照 M5 / M15 同期間
B. 依突破快慢分組（驗證分倉）
C. 候選改良 A~F 逐期
D. 保本：2R vs 不保本
"""
import numpy as np
import pandas as pd
from oos_engine import signals, simulate
from part27_ow import data, pf, prev_profiles

out = open("results_part47.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
M1 = ("data/XAUUSD_M1_2024_2026.csv", 1)
PER = [("24H2", "2024-09-01", "2024-12-31"), ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"),
       ("2026", "2026-01-01", "2026-12-31"), ("新資料 24/9~25/12", "2024-09-01", "2025-12-31"), ("全部", "2024-09-01", "2026-12-31")]


def manage(r, ARR, bm, be="2R"):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; done = False
    trig = {"2R": 2 * R, "atr": 0.5 * r.atr, "none": np.inf}[be]
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            return (min(O[k], st) if s == 1 else max(O[k], st)), t[k] + pd.Timedelta(minutes=bm), s * (best - e) / R
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if not done and s * (best - e) >= trig:
            done = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=bm), s * (best - e) / R


def finish(T, ARR, bm, be="2R"):
    T = T.copy()
    ex = [manage(r, ARR, bm, be) for r in T.itertuples()]
    T["exit_time"] = [b for a, b, c in ex]; T["mfe"] = [c for a, b, c in ex]
    T["pnl"] = T.side * (np.array([a for a, b, c in ex]) - T.lvl) - T.spread
    T["R"] = T.pnl / T.risk; T["u"] = T.pnl / T.atr
    T["delay"] = (T.t_in - T.day - pd.to_timedelta(T.h + 1, unit="h")).dt.total_seconds() / 3600
    T["lots"] = np.where(T.delay <= 0.5, 3.0, np.where(T.delay <= 1, 2.0, 1.0))
    return T.sort_values("t_in").reset_index(drop=True)


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


def line(name, T, alldays, sized=True, w=None):
    C = cap(T)
    lots = (C.lots if sized else pd.Series(1.0, C.index)) if w is None else w(C)
    C = C.assign(v=C.pnl * lots)
    cells = []
    for n, a, b in PER:
        x = C[(C.day >= a) & (C.day <= b)]
        if not len(x):
            continue
        xs = x.sort_values("exit_time"); eq = xs.v.cumsum(); dd = (eq - eq.cummax()).min()
        dl = x.groupby("day").v.sum().reindex([d for d in alldays if pd.Timestamp(a) <= d <= pd.Timestamp(b)], fill_value=0)
        cells.append(f"{n} {x.R.mean():+.2f}R/PF{pf(x.R):.2f}/{x.v.sum():+,.0f}$/DD{dd:,.0f}/S{dl.mean() / dl.std() * np.sqrt(250):.1f}")
    P(f"  {name:30s} " + "  ".join(cells))


days, D, ARR = data(*M1)
PV = prev_profiles(days, ARR)
alldays = [d for d in days if d >= pd.Timestamp("2024-09-03")]
S = signals(days, D, ARR, 1, monday=True)
X = simulate(S, ARR, 1)
q = np.array([(1 if r.ibl > PV[r.day][1] else (-1 if r.ibh < PV[r.day][2] else 0)) if r.day in PV else 0 for r in X.itertuples()])
X["qual"] = q
OW = X[X.side == X.qual].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])
OPP = X[(X.qual != 0) & (X.side == -X.qual)].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"])
base = finish(OW, ARR, 1)

P("第四十七部分：M1 2024/9 ~ 2026/10 驗證（每格 = 每筆R / PF / 美元 / 平倉回撤 / 日Sharpe；美元 = 該方案手數，最多 5 筆）")
P(f"  交易日 {len(alldays)} 天；OW 訊號 {len(base)} 筆")
P("\nA. 最終版 vs 參考")
line("最終版（分倉 3/2/1）", base, alldays)
line("全部 0.01 手", base, alldays, sized=False)
line("不保本（分倉）", finish(OW, ARR, 1, "none"), alldays)
line("分倉 ≤1h 0.02、其餘 0.01", base, alldays, w=lambda C: np.where(C.delay <= 1, 2.0, 1.0))
line("分倉 0.5~1h 0.03、≤0.5h 0.02、其餘 0.01", base, alldays, w=lambda C: np.where(C.delay <= 0.5, 2.0, np.where(C.delay <= 1, 3.0, 1.0)))

P("\nB. 依突破快慢分組（全部、不設上限；每筆 R / PF / 筆數）")
for lab, a, b in [("0~0.5h", -1, 0.5), ("0.5~1h", 0.5, 1), ("1~2h", 1, 2), ("2~3h", 2, 3.1)]:
    g = base[(base.delay > a) & (base.delay <= b)]
    P(f"  {lab:7s} " + "  ".join(f"{n} {x.R.mean():+.2f}/{pf(x.R):.2f}/{len(x)}" for n, a0, b0 in PER for x in [g[(g.day >= a0) & (g.day <= b0)]] if len(x)))

P("\nC. 候選改良（皆含分倉 3/2/1，最多 5 筆）")
line("A 保本改 0.5 ATR", finish(OW, ARR, 1, "atr"), alldays)
W = {}
for d in days:
    t, O, H, L, C, Sp, mins = ARR[d]
    row = {}
    for h in np.arange(2, 11, 0.5):
        m = (mins >= h * 60) & (mins < h * 60 + 60)
        if m.sum() >= 48:
            row[h] = H[m].max() - L[m].min()
    W[d] = row
REL = pd.DataFrame(W).T.sort_index().rolling(20, min_periods=10).median().shift(1)
base["w_atr"] = (base.ibh - base.ibl) / base.atr
base["w_rel"] = [(r.ibh - r.ibl) / REL.at[r.day, r.h] if r.day in REL.index and r.h in REL.columns and REL.at[r.day, r.h] > 0 else np.nan for r in base.itertuples()]
line("B IB 寬 ≤0.4ATR 且 相對寬 ≥0.6", base[(base.w_atr <= 0.4) & (base.w_rel >= 0.6)], alldays)
line("C 突破窗 1h（分倉 3/2）", base[base.delay <= 1], alldays)
both = finish(pd.concat([OW, OPP]).sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]), ARR, 1)
line("E 反向先突破也做", both, alldays)
line("E 只看反向單本身", finish(OPP, ARR, 1), alldays)
nopen, flt = [], []
for r in base.itertuples():
    op = base[(base.day == r.day) & (base.t_in < r.t_in) & (base.side == r.side) & (base.exit_time > r.t_in)]
    nopen.append(len(op)); flt.append((r.side * (r.lvl - op.lvl) / op.risk).sum() if len(op) else 0)
base["nopen"] = nopen; base["flt"] = flt
line("F 第2筆起需已開單浮盈>1R", base, alldays, w=lambda C: C.lots * np.where((C.nopen == 0) | (C.flt > 1), 1.0, 0.0))
line("A+B 合併", finish(OW, ARR, 1, "atr").merge(base[["t_in", "side", "w_atr", "w_rel"]], on=["t_in", "side"]).query("w_atr <= 0.4 and w_rel >= 0.6"), alldays)
base.to_csv("part47_m1_trades.csv", index=False)
out.close()
