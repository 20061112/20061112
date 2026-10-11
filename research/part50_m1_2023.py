"""第五十部分：M1 2023/6 ~ 2026/10 全期（新上傳 2023/6~2024/9，與既有 M1 合併；重疊 28,681 根完全相同）。

2023/6 ~ 2024/8 是 M1 第一次看到的期間（M15 顯示這段是區間盤、策略不賺）。
A. M1 定案版（OW + IB 寬度 + 2R 保本 + 分倉 3/2/1 + 最多 5 筆）逐期
B. 對照：無寬度篩選、全部 0.01 手、不保本
C. 候選 F（只加碼在賺錢部位）、G（前 20 天日內單邊比例中位 < 0.4 暫停）
D. 依突破快慢分組（驗證分倉）
"""
import numpy as np
import pandas as pd
from oos_engine import signals, simulate
from part27_ow import data, pf, prev_profiles

out = open("results_part50.txt", "w")
def P(s=""): print(s); out.write(s + "\n"); out.flush()
PER = [("23H2", "2023-06-01", "2023-12-31"), ("24H1", "2024-01-01", "2024-06-30"), ("24H2", "2024-07-01", "2024-12-31"),
       ("25H1", "2025-01-01", "2025-06-30"), ("25H2", "2025-07-01", "2025-12-31"), ("26H1", "2026-01-01", "2026-06-30"),
       ("26H2", "2026-07-01", "2026-12-31"), ("新 23/6~24/8", "2023-06-01", "2024-08-31"), ("全部", "2023-06-01", "2026-12-31")]
days, D, ARR = data("data/XAUUSD_M1_2023_2026.csv", 1)
PV = prev_profiles(days, ARR)


def manage(r, be=True):
    t, O, H, L, C, S, mins = ARR[r.day]
    s, e = r.side, r.lvl
    st = r.ibl if s == 1 else r.ibh
    R = s * (e - st); best = e; done = False
    for k in range(r.i, len(C)):
        if (L[k] <= st) if s == 1 else (H[k] >= st):
            return (min(O[k], st) if s == 1 else max(O[k], st)), t[k] + pd.Timedelta(minutes=1)
        best = max(best, H[k]) if s == 1 else min(best, L[k])
        if be and not done and s * (best - e) >= 2 * R:
            done = True; st = e
    return C[-1], t[-1] + pd.Timedelta(minutes=1)


def cap(T, n=5):
    keep, open_ = [], []
    for r in T.sort_values("t_in").itertuples():
        open_ = [x for x in open_ if x > r.t_in]
        if len(open_) < n:
            keep.append(r.Index); open_.append(r.exit_time)
    return T.loc[keep]


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
X = simulate(signals(days, D, ARR, 1, monday=True), ARR, 1)
X["ow"] = [(r.day in PV) and ((r.ibl > PV[r.day][1]) if r.side == 1 else (r.ibh < PV[r.day][2])) for r in X.itertuples()]
X["w_atr"] = (X.ibh - X.ibl) / X.atr
X["w_rel"] = [(r.ibh - r.ibl) / REL.at[r.day, r.h] if (r.day in REL.index and r.h in REL.columns and REL.at[r.day, r.h] > 0) else np.nan for r in X.itertuples()]
X["wok"] = (X.w_atr <= 0.4) & (X.w_rel >= 0.6)
TR = ((D.close - D.open).abs() / (D.high - D.low)).rolling(20).median().shift(1)


def build(mask, be=True):
    T = X[mask].sort_values(["t_in", "h"]).drop_duplicates(["t_in", "side"]).copy()
    ex = [manage(r, be) for r in T.itertuples()]
    T["exit_px"] = [a for a, b in ex]; T["exit_time"] = [b for a, b in ex]
    T["pnl"] = T.side * (T.exit_px - T.lvl) - T.spread; T["R"] = T.pnl / T.risk
    T["delay"] = (T.t_in - T.day - pd.to_timedelta(T.h + 1, unit="h")).dt.total_seconds() / 3600
    T["lots"] = np.where(T.delay <= 0.5, 3.0, np.where(T.delay <= 1, 2.0, 1.0))
    return T.sort_values("t_in").reset_index(drop=True)


alldays = [d for d in days if d >= pd.Timestamp("2023-06-02")]


def line(name, T, w=None):
    C = cap(T)
    lots = C.lots if w is None else w(C)
    C = C.assign(v=C.pnl * lots)
    cells = []
    for n, a, b in PER:
        x = C[(C.day >= a) & (C.day <= b)]
        if not len(x):
            continue
        xs = x.sort_values("exit_time"); eq = xs.v.cumsum(); dd = (eq - eq.cummax()).min()
        dl = x.groupby("day").v.sum().reindex([d for d in alldays if pd.Timestamp(a) <= d <= pd.Timestamp(b)], fill_value=0)
        cells.append(f"{n} {x.R.mean():+.2f}R/{x.v.sum():+,.0f}$/DD{dd:,.0f}/S{dl.mean() / dl.std() * np.sqrt(250):.1f}")
    P(f"  {name:26s} " + "  ".join(cells))
    return C


P("第五十部分：M1 2023/6 ~ 2026/10（每格 = 每筆R / 美元 / 該期平倉回撤 / 日Sharpe；最多 5 筆；美元以 0.01 手 = 1 盎司計 × 手數倍數）")
fin = build(X.ow & X.wok)
P("\nA/B. 定案與對照")
Cf = line("定案（寬度 + 分倉 3/2/1）", fin)
line("定案、全部 0.01 手", fin, w=lambda C: np.ones(len(C)))
line("無寬度篩選（分倉）", build(X.ow))
line("定案、不保本", build(X.ow & X.wok, be=False))
P("\nC. 候選")
nopen, flt = [], []
for r in fin.itertuples():
    op = fin[(fin.day == r.day) & (fin.t_in < r.t_in) & (fin.side == r.side) & (fin.exit_time > r.t_in)]
    nopen.append(len(op)); flt.append((r.side * (r.lvl - op.lvl) / op.risk).sum() if len(op) else 0)
fin["nopen"] = nopen; fin["flt"] = flt
line("F 第2筆起需已開單浮盈>1R", fin, w=lambda C: C.lots * np.where((C.nopen == 0) | (C.flt > 1), 1.0, 0.0))
line("G 前20天單邊中位≥0.4才做", fin[fin.day.map(TR).fillna(1) >= 0.4])
line("G 前20天單邊中位≥0.3才做", fin[fin.day.map(TR).fillna(1) >= 0.3])
P("\nD. 依突破快慢分組（全部、不設上限；每筆 R / 筆數）")
for lab, a, b in [("0~0.5h", -1, 0.5), ("0.5~1h", 0.5, 1), ("1~2h", 1, 2), ("2~3h", 2, 3.1)]:
    g = fin[(fin.delay > a) & (fin.delay <= b)]
    P(f"  {lab:7s} " + "  ".join(f"{n} {x.R.mean():+.2f}/{len(x)}" for n, a0, b0 in PER for x in [g[(g.day >= a0) & (g.day <= b0)]] if len(x)))
eq = Cf.sort_values("exit_time").assign(v=lambda d: d.pnl * d.lots).v.cumsum()
P(f"\n定案全期：{len(Cf)} 筆、每筆 {Cf.R.mean():+.3f}R、PF {pf(Cf.R):.2f}、{(Cf.pnl * Cf.lots).sum():+,.0f} 美元、最大回撤 {(eq - eq.cummax()).min():+,.0f}")
mon = Cf.assign(v=Cf.pnl * Cf.lots).groupby(Cf.day.dt.to_period("M")).v.sum()
P("逐月：" + "  ".join(f"{p.strftime('%y/%m')} {v:+,.0f}" for p, v in mon.items()))
Cf.to_csv("part50_m1_trades.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(13, 4.5))
C2 = Cf.sort_values("exit_time")
ax.plot(C2.exit_time, (C2.pnl * C2.lots).cumsum(), color="#2e8b57", lw=1.5, label="final (0.03/0.02/0.01)")
ax.plot(C2.exit_time, C2.pnl.cumsum(), color="#999", lw=1.1, label="final, all 0.01 lot")
ax.axvspan(pd.Timestamp("2023-06-01"), pd.Timestamp("2024-08-31"), color="#f3e3c3", alpha=.5, label="new M1 data")
ax.set_title("M1 OW final rules, 2023/6 - 2026/10 (USD, max 5 open)"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("part50_m1.png", dpi=110)
out.close()
